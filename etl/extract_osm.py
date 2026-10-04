"""
Stage 1 - Extract raw POI features for Sri Lanka from OpenStreetMap via Overpass.

Writes an untouched, dated dump to data/raw/ so the pipeline is reproducible and
the provenance of every record is auditable.

Run:  python -m etl.extract_osm
"""
from __future__ import annotations
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

import requests

from .config import OVERPASS_URL, USER_AGENT, DATA_RAW

log = logging.getLogger(__name__)

# Overpass QL. `out center tags` gives ways a synthetic centroid so nodes and
# ways can be treated uniformly downstream.
OVERPASS_QUERY = """
[out:json][timeout:300];
area["ISO3166-1"="LK"][admin_level=2]->.lk;
(
  node["tourism"~"attraction|museum|viewpoint|zoo|theme_park|artwork|gallery|information"](area.lk);
  way ["tourism"~"attraction|museum|viewpoint|zoo|theme_park|gallery"](area.lk);

  node["historic"](area.lk);
  way ["historic"](area.lk);

  node["natural"~"beach|waterfall|peak|cave_entrance|spring|hot_spring|cliff"](area.lk);
  way ["natural"~"beach|waterfall|wood|water|wetland"](area.lk);

  node["waterway"="waterfall"](area.lk);

  node["leisure"~"park|nature_reserve|garden|beach_resort"](area.lk);
  way ["leisure"~"park|nature_reserve|garden|beach_resort"](area.lk);

  node["amenity"~"place_of_worship|monastery|marketplace|theatre|arts_centre"](area.lk);
  way ["amenity"~"place_of_worship|monastery"](area.lk);

  node["man_made"~"lighthouse|tower"](area.lk);

  way ["boundary"="protected_area"](area.lk);
  relation["boundary"="protected_area"](area.lk);
  way ["boundary"="national_park"](area.lk);
  relation["boundary"="national_park"](area.lk);
);
out center tags;
"""

# Accommodation is extracted separately: different tags, different downstream table.
OVERPASS_ACCOMMODATION = """
[out:json][timeout:300];
area["ISO3166-1"="LK"][admin_level=2]->.lk;
(
  node["tourism"~"hotel|guest_house|hostel|motel|apartment|chalet|resort"](area.lk);
  way ["tourism"~"hotel|guest_house|hostel|motel|apartment|chalet|resort"](area.lk);
);
out center tags;
"""


def fetch(query: str, attempts: int = 4, backoff: float = 20.0) -> dict:
    """
    POST a query to Overpass with polite retry.

    Overpass is a shared free service. 429 and 504 are normal under load and are
    retried with a long backoff rather than hammered; a permanent failure raises
    rather than returning partial data (fail-loud policy, plan section 7).
    """
    headers = {"User-Agent": USER_AGENT}
    last = None
    for i in range(1, attempts + 1):
        try:
            log.info("Overpass request attempt %d/%d", i, attempts)
            r = requests.post(OVERPASS_URL, data={"data": query},
                              headers=headers, timeout=360)
            if r.status_code == 200:
                payload = r.json()
                log.info("Received %d elements", len(payload.get("elements", [])))
                return payload
            if r.status_code in (429, 504, 503):
                wait = backoff * i
                log.warning("Overpass %s, backing off %.0fs", r.status_code, wait)
                time.sleep(wait)
                last = f"HTTP {r.status_code}"
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            last = str(e)
            log.warning("Request failed: %s", e)
            time.sleep(backoff * i)
    raise RuntimeError(f"Overpass extraction failed after {attempts} attempts: {last}")


def save_raw(payload: dict, name: str) -> Path:
    stamp = date.today().isoformat()
    path = DATA_RAW / f"{name}_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s (%d elements)", path.name, len(payload.get("elements", [])))
    return path


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log.info("Extracting POIs from Overpass. This typically takes 1-4 minutes.")
    save_raw(fetch(OVERPASS_QUERY), "osm_pois")

    log.info("Extracting accommodation from Overpass.")
    save_raw(fetch(OVERPASS_ACCOMMODATION), "osm_accommodation")

    log.info("Extraction complete. Next: python -m etl.clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
