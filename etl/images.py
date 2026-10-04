"""
Stage 5b - Find a photograph for each POI and store it on the graph.

    python -m etl.images            # search (resumable), then load into Neo4j
    python -m etl.images --no-load  # search only
    python -m etl.images --load-only

Sources, in order of trust:
  1. wikipedia   - the lead image of the place's own Wikipedia article.
  2. nearby_300m - a Wikimedia Commons photo geotagged within 300 m.
  3. nearby_1km  - a Commons photo within 1 km. May show a neighbouring
                   building or street, so clients label it "nearby".
No match leaves the image empty and the client shows the category icon.

Only URLs and attribution are stored, never image bytes. Commons files are
freely licensed, which is why they are used instead of commercial photo APIs,
whose terms allow display but not retention.

Wikimedia rate-limits bursts, so requests are serial, about one per second,
with backoff on 429. Results are cached to data/raw/images_cache.json after
every lookup, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations
import argparse
import json
import logging
import math
import re
import sys
import time
from urllib.parse import quote

import pandas as pd
import requests

from .config import DATA_RAW, DATA_PROC, USER_AGENT, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

log = logging.getLogger(__name__)

CACHE = DATA_RAW / "images_cache.json"
OUT = DATA_PROC / "poi_images.parquet"
COMMONS = "https://commons.wikimedia.org/w/api.php"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
MIN_INTERVAL = 1.0
WIDTH = 640
PHOTO_EXT = (".jpg", ".jpeg", ".png", ".webp")
# Commons geotags maps, diagrams and scanned documents too; these are not photos
# of a place and would mislead on a card.
NOT_A_PHOTO = re.compile(r"\b(map|locator|diagram|plan|logo|flag|coat of arms|chart|"
                         r"svg|pdf|scan|document|seal)\b", re.I)

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_last = 0.0


def _get(url: str, params: dict | None = None) -> dict | None:
    global _last
    for attempt in range(6):
        wait = MIN_INTERVAL - (time.monotonic() - _last)
        if wait > 0:
            time.sleep(wait)
        _last = time.monotonic()
        try:
            r = _session.get(url, params=params, timeout=20)
        except requests.RequestException as e:
            log.warning("request failed (%s), retrying", str(e)[:80])
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                return None
        if r.status_code == 404:
            return None
        retry_after = r.headers.get("Retry-After")
        delay = int(retry_after) if retry_after and retry_after.isdigit() else 2 ** (attempt + 1)
        log.info("HTTP %s, backing off %ss", r.status_code, delay)
        time.sleep(min(delay, 60))
    return None


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    p = math.pi / 180
    h = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


def _strip_html(s: str | None) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def from_wikipedia(tag: str) -> dict | None:
    title = (tag.split(":", 1)[1] if ":" in tag else tag).strip().replace(" ", "_")
    if not title:
        return None
    data = _get(WIKI_SUMMARY.format(title=quote(title, safe="")))
    thumb = ((data or {}).get("thumbnail") or {}).get("source")
    if not thumb:
        return None
    return {
        "image_url": re.sub(r"/(\d+)px-", f"/{WIDTH}px-", thumb),
        "image_source": "wikipedia",
        "image_distance_m": 0.0,
        "image_credit": "Wikipedia / Wikimedia Commons",
        "image_page": (data.get("content_urls") or {}).get("desktop", {}).get("page"),
    }


def from_commons(lat: float, lon: float) -> dict | None:
    data = _get(COMMONS, {
        "action": "query", "format": "json", "generator": "geosearch",
        "ggscoord": f"{lat}|{lon}", "ggsradius": 1000, "ggsnamespace": 6, "ggslimit": 15,
        "prop": "imageinfo|coordinates", "iiprop": "url|extmetadata|mime",
        "iiurlwidth": WIDTH, "iiextmetadatafilter": "Artist|LicenseShortName",
    })
    pages = ((data or {}).get("query") or {}).get("pages") or {}
    best = None
    for p in pages.values():
        title = p.get("title", "")
        info = (p.get("imageinfo") or [{}])[0]
        if not title.lower().endswith(PHOTO_EXT) or NOT_A_PHOTO.search(title):
            continue
        if not str(info.get("mime", "")).startswith("image/") or not info.get("thumburl"):
            continue
        coords = (p.get("coordinates") or [{}])[0]
        if "lat" not in coords:
            continue
        d = _haversine_m(lat, lon, coords["lat"], coords["lon"])
        if best is None or d < best[0]:
            best = (d, title, info)
    if best is None:
        return None

    d, title, info = best
    meta = info.get("extmetadata") or {}
    artist = _strip_html((meta.get("Artist") or {}).get("value"))[:120]
    licence = _strip_html((meta.get("LicenseShortName") or {}).get("value"))[:40]
    credit = " · ".join(x for x in (artist, licence, "Wikimedia Commons") if x)
    return {
        "image_url": info["thumburl"],
        "image_source": "nearby_300m" if d <= 300 else "nearby_1km",
        "image_distance_m": round(d, 1),
        "image_credit": credit,
        "image_page": info.get("descriptionurl"),
    }


def search(pois: pd.DataFrame, limit: int = 0) -> dict:
    cache: dict = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [r for r in pois.itertuples() if r.poi_id not in cache]
    if limit:
        todo = todo[:limit]
    log.info("%d POIs cached, %d to search (~%d min)", len(cache), len(todo),
             round(len(todo) * MIN_INTERVAL * 1.1 / 60))

    for i, r in enumerate(todo, 1):
        found = None
        wiki = getattr(r, "wikipedia", None)
        if isinstance(wiki, str) and wiki.strip():
            found = from_wikipedia(wiki)
        if found is None:
            found = from_commons(float(r.lat), float(r.lon))
        cache[r.poi_id] = found
        if i % 25 == 0 or i == len(todo):
            CACHE.write_text(json.dumps(cache), encoding="utf-8")
            hit = sum(1 for v in cache.values() if v)
            log.info("  %d/%d searched, %d with a photo so far", i, len(todo), hit)
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def to_frame(pois: pd.DataFrame, cache: dict) -> pd.DataFrame:
    rows = []
    for pid in pois.poi_id:
        v = cache.get(pid) or {}
        rows.append({"poi_id": pid, "image_url": v.get("image_url"),
                     "image_source": v.get("image_source"),
                     "image_distance_m": v.get("image_distance_m"),
                     "image_credit": v.get("image_credit"),
                     "image_page": v.get("image_page")})
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    n = len(df)
    counts = df.image_source.fillna("none").value_counts()
    print(f"\nPhoto coverage over {n:,} POIs")
    for k in ("wikipedia", "nearby_300m", "nearby_1km", "none"):
        c = int(counts.get(k, 0))
        print(f"  {k:<12} {c:>5}  ({100 * c / max(n, 1):.1f}%)")


LOAD = """
UNWIND $rows AS row
MATCH (p:POI {poi_id: row.poi_id})
SET p.image_url = row.image_url,
    p.image_source = row.image_source,
    p.image_distance_m = row.image_distance_m,
    p.image_credit = row.image_credit,
    p.image_page = row.image_page
"""


def load(df: pd.DataFrame) -> None:
    if not NEO4J_URI or not NEO4J_PASSWORD:
        raise SystemExit("NEO4J_URI / NEO4J_PASSWORD not set")
    from neo4j import GraphDatabase
    rows = [{k: (None if pd.isna(v) else v) for k, v in r.items()}
            for r in df.to_dict("records")]
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as s:
            for i in range(0, len(rows), 500):
                s.run(LOAD, rows=rows[i:i + 500]).consume()
    finally:
        driver.close()
    log.info("Loaded image fields for %d POIs", len(rows))


def main() -> int:
    ap = argparse.ArgumentParser(description="Find and store a photo per POI")
    ap.add_argument("--no-load", action="store_true", help="search only")
    ap.add_argument("--load-only", action="store_true", help="load cached results only")
    ap.add_argument("--limit", type=int, default=0, help="search at most N uncached POIs")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    pois = pd.read_parquet(DATA_PROC / "pois.parquet")
    if args.load_only:
        cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    else:
        cache = search(pois, args.limit)

    df = to_frame(pois, cache)
    df.to_parquet(OUT, index=False)
    report(df)
    if not args.no_load:
        load(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
