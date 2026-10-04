"""
Stage 3 - Enrich POIs with popularity, seasonality and prominence.

Rewritten to fix three faults in the first version:

  1. The sleep was inside the success branch, so a failed request was followed
     immediately by another. One 429 cascaded into hundreds. There is now a rate
     limiter plus exponential backoff that honours Retry-After.
  2. Only the OSM `wikipedia` tag was used, which is present on ~1% of POIs. The
     `wikidata` tag is far more common and resolves to a Wikipedia title via the
     Wikidata API, which is where most of the coverage gain comes from.
  3. Every POI was iterated even with no resolvable title. Only resolvable POIs
     are fetched now.

Results are cached to disk, so a re-run costs nothing for titles already fetched
and an interrupted run can be restarted safely.

Run:  python -m etl.enrich
"""
from __future__ import annotations
import json
import logging
import sys
import time
from datetime import date, timedelta
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from .config import DATA_PROC, DATA_RAW, PAGEVIEWS_API, USER_AGENT

log = logging.getLogger(__name__)

CACHE_PATH = DATA_RAW / "pageviews_cache.json"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

# Wikimedia throttles aggressively. 0.4 s between calls (~150/min) is polite and
# stays under the limit in practice; raise this if 429s still appear.
MIN_INTERVAL = 0.40
MAX_RETRIES = 4


class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        delta = time.monotonic() - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()


LIMITER = RateLimiter(MIN_INTERVAL)


def _get(url: str, params: dict | None = None):
    """GET with rate limiting and exponential backoff on 429/5xx."""
    for attempt in range(MAX_RETRIES):
        LIMITER.wait()
        try:
            r = SESSION.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            log.debug("network error: %s", e)
            time.sleep(2 ** attempt)
            continue

        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None                       # article genuinely absent
        if r.status_code in (429, 502, 503, 504):
            ra = r.headers.get("Retry-After")
            wait = float(ra) if ra and ra.isdigit() else min(2 ** attempt * 2, 30)
            log.debug("HTTP %s, backing off %.1fs", r.status_code, wait)
            time.sleep(wait)
            continue
        return None
    return None


# ---------------------------------------------------------------------------
# Title resolution
# ---------------------------------------------------------------------------
def title_from_wikipedia_tag(tag) -> str | None:
    """OSM stores `en:Sigiriya`. Non-English prefixes are skipped."""
    if not tag or not isinstance(tag, str):
        return None
    if ":" in tag:
        lang, title = tag.split(":", 1)
        if lang.strip().lower() != "en":
            return None                       # only en.wikipedia pageviews are used
        return title.strip().replace(" ", "_")
    return tag.strip().replace(" ", "_")


def titles_from_wikidata(qids: list[str]) -> dict[str, str]:
    """Resolve Wikidata Q-ids to English Wikipedia titles, 50 per request."""
    out: dict[str, str] = {}
    qids = [q for q in qids if isinstance(q, str) and q.startswith("Q")]
    if not qids:
        return out

    log.info("Resolving %d Wikidata ids to Wikipedia titles", len(qids))
    for i in range(0, len(qids), 50):
        batch = qids[i:i + 50]
        r = _get(WIKIDATA_API, {
            "action": "wbgetentities", "ids": "|".join(batch),
            "props": "sitelinks", "sitefilter": "enwiki", "format": "json",
        })
        if r is None:
            continue
        try:
            entities = r.json().get("entities", {})
        except ValueError:
            continue
        for qid, ent in entities.items():
            link = (ent.get("sitelinks") or {}).get("enwiki")
            if link and link.get("title"):
                out[qid] = link["title"].replace(" ", "_")
        if (i // 50) % 10 == 0:
            log.info("  %d/%d ids processed, %d titles found",
                     min(i + 50, len(qids)), len(qids), len(out))

    log.info("Wikidata resolved %d/%d titles", len(out), len(qids))
    return out


# ---------------------------------------------------------------------------
# Pageviews
# ---------------------------------------------------------------------------
def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except ValueError:
            return {}
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def fetch_pageviews(title: str, months_back: int = 36) -> dict | None:
    """Returns {'mean': float, 'seasonality': {month: multiplier}} or None."""
    end = date.today().replace(day=1)
    start = end - timedelta(days=31 * months_back)
    url = PAGEVIEWS_API.format(
        article=quote(title, safe=""),
        start=start.strftime("%Y%m01") + "00",
        end=end.strftime("%Y%m01") + "00",
    )
    r = _get(url)
    if r is None:
        return None
    try:
        items = r.json().get("items", [])
    except ValueError:
        return None
    if not items:
        return None

    s = pd.Series({pd.to_datetime(i["timestamp"][:6], format="%Y%m"): i["views"]
                   for i in items}).sort_index()
    if s.empty:
        return None
    by_month = s.groupby(s.index.month).mean()
    overall = by_month.mean() or 1.0
    return {
        "mean": float(s.mean()),
        "seasonality": {int(k): round(float(v), 3) for k, v in (by_month / overall).items()},
    }


# ---------------------------------------------------------------------------
# Prominence
# ---------------------------------------------------------------------------
def compute_prominence(df: pd.DataFrame) -> pd.Series:
    """
    A 0-1 signal of how likely a POI is to be a genuine visitor destination
    rather than a minor local feature.

    Deliberately NOT the same as popularity. Prominence answers "is this a real
    destination"; popularity answers "is this crowded". A quiet UNESCO site
    should score high prominence and low popularity - exactly the kind of place
    the sustainability objective wants to surface.

    The first version leaned on Wikidata/Wikipedia presence, which only ~1.5% of
    Sri Lankan OSM POIs carry, so almost everything scored near zero and the
    signal could not discriminate. This version is built mainly from signals
    that are actually present: tag richness, an explicit tourism tag, recorded
    opening hours or fees, protected-area status and rail access. External
    identifiers now act as a bonus rather than the foundation.
    """
    idx = df.index
    score = pd.Series(0.0, index=idx)
    tags = df.get("osm_tags_kept", pd.Series("{}", index=idx)).fillna("{}").astype(str)

    # --- signals available on most POIs ---
    # Tag richness: mappers add detail to places that matter. Saturates at 12.
    if "tag_count" in df.columns:
        score += (df["tag_count"].fillna(0).clip(0, 12) / 12.0) * 0.25

    # An explicit tourism tag is a mapper asserting visitor relevance.
    score += tags.str.contains('"tourism"', regex=False).astype(float) * 0.20
    score += tags.str.contains("website", case=False).astype(float) * 0.12
    score += (~df["hours_estimated"]).astype(float) * 0.12
    score += (~df["fee_estimated"]).astype(float) * 0.08
    score += tags.str.contains('"historic"', regex=False).astype(float) * 0.08

    # Protected-area status marks a managed, visitable natural site.
    score += tags.str.contains("protect", case=False).astype(float) * 0.08

    # Rail access: reachable places are more viable destinations.
    if "rail_access_score" in df.columns:
        score += df["rail_access_score"].fillna(0.0) * 0.10

    # --- bonuses from external identifiers (rare but strong) ---
    score += df["wikidata"].notna().astype(float) * 0.20
    score += df["wikipedia"].notna().astype(float) * 0.15
    if "unesco" in df.columns:
        score += df["unesco"].fillna(False).astype(float) * 0.25

    cat_weight = {"wildlife": 0.10, "heritage": 0.10, "beach": 0.08,
                  "nature": 0.05, "adventure": 0.06, "cultural": 0.05,
                  "religious": 0.02, "urban": 0.02}
    score += df["category"].map(cat_weight).fillna(0.0)

    return score.clip(0, 1).round(4)


# ---------------------------------------------------------------------------
def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cache = load_cache()

    # 1. Direct wikipedia tags
    titles: dict = {}
    for i, tag in df["wikipedia"].items():
        t = title_from_wikipedia_tag(tag)
        if t:
            titles[i] = t
    log.info("Titles from wikipedia tag: %d", len(titles))

    # 2. Wikidata -> wikipedia for everything still unresolved
    need = df.index.difference(pd.Index(list(titles.keys())))
    qids = df.loc[need, "wikidata"].dropna().astype(str)
    log.info("POIs with a wikidata tag and no wikipedia tag: %d", len(qids))
    qid_map = titles_from_wikidata(sorted(set(qids.tolist())))
    for i, q in qids.items():
        if q in qid_map:
            titles[i] = qid_map[q]

    log.info("Resolvable titles: %d of %d POIs (%.1f%%)",
             len(titles), len(df), 100 * len(titles) / max(len(df), 1))

    # 3. Fetch pageviews for resolvable titles only, using the cache
    unique_titles = sorted(set(titles.values()))
    todo = [t for t in unique_titles if t not in cache]
    log.info("Pageviews: %d unique titles, %d cached, %d to fetch (~%.1f min)",
             len(unique_titles), len(unique_titles) - len(todo), len(todo),
             len(todo) * MIN_INTERVAL / 60)

    for i, t in enumerate(todo, 1):
        cache[t] = fetch_pageviews(t) or {}
        if i % 50 == 0:
            save_cache(cache)
            log.info("  %d/%d fetched (%d with data)", i, len(todo),
                     sum(1 for v in cache.values() if v))
    save_cache(cache)

    # 4. Assemble columns
    means, seasons, sources = [], [], []
    for i in df.index:
        entry = cache.get(titles.get(i, ""), None)
        if entry:
            means.append(entry["mean"])
            seasons.append(json.dumps(entry["seasonality"]))
            sources.append("wikipedia")
        else:
            means.append(np.nan)
            seasons.append("{}")
            sources.append("none")

    raw = pd.Series(means, index=df.index)
    logged = np.log1p(raw)
    lo, hi = logged.min(), logged.max()
    df["popularity_index"] = (((logged - lo) / (hi - lo)).fillna(0.0).round(4)
                              if pd.notna(lo) and hi > lo else 0.0)
    df["seasonality_json"] = seasons
    df["popularity_source"] = sources
    df["prominence"] = compute_prominence(df)

    got = int((df.popularity_source == "wikipedia").sum())
    log.info("Popularity resolved for %d/%d POIs (%.1f%%)",
             got, len(df), 100 * got / max(len(df), 1))
    log.info("Prominence: mean %.3f, %d POIs above 0.4",
             df.prominence.mean(), int((df.prominence > 0.4).sum()))
    return df


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    src = DATA_PROC / "pois.parquet"
    if not src.exists():
        raise SystemExit("Run `python -m etl.clean` first.")

    df = build_signals(pd.read_parquet(src))
    df.to_parquet(src, index=False)
    log.info("Updated %s", src.name)
    print("\nNext: python -m etl.travel_matrix")
    return 0


if __name__ == "__main__":
    sys.exit(main())
