"""
Run the complete ETL pipeline.

  python -m etl.run              # full pipeline
  python -m etl.run --skip-osm   # reuse the existing raw dump
  python -m etl.run --no-load    # stop before Neo4j

Each stage is idempotent, so a failed run can be resumed with --skip-osm.
"""
from __future__ import annotations
import argparse
import logging
import sys
import time

log = logging.getLogger("etl.run")


def main() -> int:
    ap = argparse.ArgumentParser(description="Travora ETL pipeline")
    ap.add_argument("--skip-osm", action="store_true", help="reuse the latest raw dump")
    ap.add_argument("--skip-enrich", action="store_true", help="skip Wikipedia pageviews")
    ap.add_argument("--no-osrm", action="store_true", help="use the distance model only")
    ap.add_argument("--no-load", action="store_true", help="do not load into Neo4j")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()

    if not args.skip_osm:
        from . import extract_osm
        log.info("=== Stage 1/6: extract from OpenStreetMap ===")
        extract_osm.main()
    else:
        log.info("=== Stage 1/6: skipped (reusing raw dump) ===")

    from . import clean
    log.info("=== Stage 2/6: clean, categorise, deduplicate ===")
    df = clean.run()

    from .config import DATA_PROC as _DP
    log.info("=== Stage 2b: district assignment ===")
    rc_b = 1
    try:
        from . import boundaries
        rc_b = boundaries.main()          # authoritative: point-in-polygon
    except Exception as e:                # noqa: BLE001
        log.warning("Boundary assignment unavailable: %s", e)
    if rc_b != 0:
        if (_DP / "accommodation.parquet").exists():
            log.warning("Falling back to SLTDA nearest-neighbour assignment")
            from . import districts
            districts.main()
        else:
            log.warning("No district refinement applied; centroid values retained")

    if not args.skip_enrich:
        from . import enrich
        log.info("=== Stage 3/6: enrich with popularity signals ===")
        enrich.main()
    else:
        log.info("=== Stage 3/6: skipped ===")

    from . import travel_matrix
    from .config import DATA_PROC
    import pandas as pd
    log.info("=== Stage 4/6: travel matrix ===")
    pois = pd.read_parquet(DATA_PROC / "pois.parquet")
    matrix = travel_matrix.build(pois, use_osrm=not args.no_osrm)
    matrix.to_parquet(DATA_PROC / "travel_matrix.parquet", index=False)

    if not args.no_load:
        from . import load_neo4j
        log.info("=== Stage 5/6: load into Neo4j ===")
        load_neo4j.main()
    else:
        log.info("=== Stage 5/6: skipped ===")

    from . import data_card
    log.info("=== Stage 6/6: data card and acceptance check ===")
    rc = data_card.main()

    log.info("Pipeline finished in %.1f minutes", (time.time() - t0) / 60)
    return rc


if __name__ == "__main__":
    sys.exit(main())
