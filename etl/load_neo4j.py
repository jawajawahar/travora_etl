"""
Stage 5 - Load the processed dataset into Neo4j.

Graph model
-----------
  (:POI {poi_id, name, lat, lon, category, open_min, close_min,
         entry_fee_lkr, typical_dwell_min, popularity_index,
         hours_estimated, fee_estimated, eco_certified, ...})
  (:District {name})
  (:Category {name})
  (:Accommodation {acc_id, name, lat, lon, type, price_band_*, sltda_grade})

  (:POI)-[:IN_DISTRICT]->(:District)
  (:POI)-[:HAS_CATEGORY]->(:Category)
  (:Accommodation)-[:IN_DISTRICT]->(:District)
  (:POI)-[:TRAVEL {distance_km, duration_min, method}]->(:POI)

TRAVEL is stored in one direction only; queries traverse it undirected. This
halves the relationship count, which matters on the AuraDB free tier.

Run:  python -m etl.load_neo4j
"""
from __future__ import annotations
import logging
import sys

import pandas as pd

from .config import (DATA_PROC, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
                     FORBIDDEN_PERSIST_FIELDS)

log = logging.getLogger(__name__)

CONSTRAINTS = [
    "CREATE CONSTRAINT poi_id IF NOT EXISTS FOR (p:POI) REQUIRE p.poi_id IS UNIQUE",
    "CREATE CONSTRAINT acc_id IF NOT EXISTS FOR (a:Accommodation) REQUIRE a.acc_id IS UNIQUE",
    "CREATE CONSTRAINT district_name IF NOT EXISTS FOR (d:District) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT acctype_name IF NOT EXISTS FOR (t:AccommodationType) REQUIRE t.name IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX poi_category IF NOT EXISTS FOR (p:POI) ON (p.category)",
    "CREATE INDEX poi_popularity IF NOT EXISTS FOR (p:POI) ON (p.popularity_index)",
    "CREATE INDEX poi_prominence IF NOT EXISTS FOR (p:POI) ON (p.prominence)",
    "CREATE POINT INDEX poi_location IF NOT EXISTS FOR (p:POI) ON (p.location)",
    "CREATE POINT INDEX acc_location IF NOT EXISTS FOR (a:Accommodation) ON (a.location)",
    "CREATE INDEX acc_sustainability IF NOT EXISTS FOR (a:Accommodation) ON (a.sustainability_score)",
    "CREATE INDEX acc_type IF NOT EXISTS FOR (a:Accommodation) ON (a.sltda_type)",
]

LOAD_POIS = """
UNWIND $rows AS row
MERGE (p:POI {poi_id: row.poi_id})
SET p.name              = row.name,
    p.lat               = row.lat,
    p.lon               = row.lon,
    p.location          = point({latitude: row.lat, longitude: row.lon}),
    p.category          = row.category,
    p.open_min          = toInteger(row.open_min),
    p.close_min         = toInteger(row.close_min),
    p.entry_fee_lkr     = toFloat(row.entry_fee_lkr),
    p.typical_dwell_min = toInteger(row.typical_dwell_min),
    p.popularity_index  = toFloat(row.popularity_index),
    p.prominence        = toFloat(row.prominence),
    p.wikipedia         = row.wikipedia,
    p.wikidata          = row.wikidata,
    p.popularity_source = row.popularity_source,
    p.rail_access_score = toFloat(row.rail_access_score),
    p.nearest_station_km = toFloat(row.nearest_station_km),
    p.hours_estimated   = row.hours_estimated,
    p.fee_estimated     = row.fee_estimated,
    p.dwell_estimated   = row.dwell_estimated,
    p.eco_certified     = row.eco_certified,
    p.community_run     = row.community_run,
    p.district          = row.district,
    p.district_method   = row.district_method,
    p.district_confidence = toFloat(row.district_confidence),
    p.unesco            = row.unesco,
    p.seasonality_json  = row.seasonality_json
MERGE (d:District {name: row.district})
MERGE (p)-[:IN_DISTRICT]->(d)
MERGE (c:Category {name: row.category})
MERGE (p)-[:HAS_CATEGORY]->(c)
"""

LOAD_TRAVEL = """
UNWIND $rows AS row
MATCH (a:POI {poi_id: row.from_id})
MATCH (b:POI {poi_id: row.to_id})
MERGE (a)-[t:TRAVEL]->(b)
SET t.distance_km = toFloat(row.distance_km),
    t.duration_min = toFloat(row.duration_min),
    t.method = row.method
"""

LOAD_ACCOMMODATION = """
UNWIND $rows AS row
MERGE (a:Accommodation {acc_id: row.acc_id})
SET a.name                 = row.name,
    a.lat                  = row.lat,
    a.lon                  = row.lon,
    a.location             = point({latitude: row.lat, longitude: row.lon}),
    a.sltda_type           = row.sltda_type,
    a.grade                = row.grade,
    a.rooms                = toInteger(row.rooms),
    a.district             = row.district,
    a.address              = row.address,
    a.category_score       = toFloat(row.category_score),
    a.size_score           = toFloat(row.size_score),
    a.sustainability_score = toFloat(row.sustainability_score),
    a.acc_impact           = toFloat(row.acc_impact),
    a.location_estimated   = row.location_estimated,
    a.location_method      = row.location_method,
    a.nearest_station_km   = toFloat(row.nearest_station_km),
    a.rail_access_score    = toFloat(row.rail_access_score)
MERGE (d:District {name: row.district})
MERGE (a)-[:IN_DISTRICT]->(d)
MERGE (t:AccommodationType {name: row.sltda_type})
MERGE (a)-[:OF_TYPE]->(t)
"""


def assert_no_tier_b(df: pd.DataFrame, label: str) -> None:
    """
    Plan section 4.0 / acceptance criterion A11.

    Commercial APIs licence access, not retention. TripAdvisor permits caching of
    the Location ID only; Google permits the place_id only. If a rating or review
    field ever reaches this point, the load is aborted rather than committed.
    """
    leaked = FORBIDDEN_PERSIST_FIELDS & set(df.columns)
    if leaked:
        raise RuntimeError(
            f"TIER B VIOLATION in {label}: {sorted(leaked)}. "
            "Commercial API content must never be persisted. Remove these columns."
        )


def batched(rows: list[dict], size: int = 500):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not NEO4J_URI or not NEO4J_PASSWORD:
        raise SystemExit(
            "Set NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD.\n"
            "  export NEO4J_URI='neo4j+s://xxxx.databases.neo4j.io'\n"
            "  export NEO4J_USER='neo4j'\n"
            "  export NEO4J_PASSWORD='...'"
        )

    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise SystemExit("pip install neo4j")

    pois = pd.read_parquet(DATA_PROC / "pois.parquet")
    assert_no_tier_b(pois, "pois.parquet")

    matrix_path = DATA_PROC / "travel_matrix.parquet"
    matrix = pd.read_parquet(matrix_path) if matrix_path.exists() else pd.DataFrame()

    acc_path = DATA_PROC / "accommodation.parquet"
    if acc_path.exists():
        acc = pd.read_parquet(acc_path)
        assert_no_tier_b(acc, "accommodation.parquet")
        for col in ("nearest_station_km", "rail_access_score"):
            if col not in acc.columns:
                acc[col] = 0.0
        acc = acc.where(pd.notna(acc), None)
    else:
        log.warning("accommodation.parquet not found - run `python -m etl.extract_sltda`")
        acc = pd.DataFrame()

    # Neo4j has no NaN; normalise before sending.
    # Properties the retrieval layer depends on. A missing column here is the
    # bug that produced a graph with 2,245 POIs and no `prominence`, against
    # which every candidate query returned zero rows.
    REQUIRED_POI_COLS = {
        "prominence": 0.0, "rail_access_score": 0.0, "nearest_station_km": 999.0,
        "popularity_index": 0.0, "popularity_source": "none",
        "district_method": "unknown", "district_confidence": 0.0,
        "seasonality_json": "{}", "eco_certified": "unknown",
        "wikipedia": None, "wikidata": None,
        "community_run": "unknown", "unesco": False,
    }
    for col, default in REQUIRED_POI_COLS.items():
        if col not in pois.columns:
            log.warning("pois.parquet has no '%s' column; defaulting to %r. "
                        "Run the missing pipeline stage (etl.enrich / etl.rail / "
                        "etl.boundaries) for real values.", col, default)
            pois[col] = default
        elif default is not None:
            # fillna(None) is invalid in pandas ("must specify a fill value or
            # method"), so columns whose real default IS null - wikipedia and
            # wikidata, where most places genuinely have no id - are left as
            # NaN here and normalised to None in the blanket pass just below.
            pois[col] = pois[col].fillna(default)

    pois = pois.where(pd.notna(pois), None)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as s:
        log.info("Applying constraints and indexes")
        for stmt in CONSTRAINTS + INDEXES:
            s.run(stmt)

        log.info("Loading %d POIs", len(pois))
        for chunk in batched(pois.to_dict("records")):
            s.run(LOAD_POIS, rows=chunk)

        if len(acc):
            log.info("Loading %d accommodation records", len(acc))
            for chunk in batched(acc.to_dict("records")):
                s.run(LOAD_ACCOMMODATION, rows=chunk)

        if len(matrix):
            log.info("Loading %d travel relationships", len(matrix))
            for chunk in batched(matrix.to_dict("records"), 1000):
                s.run(LOAD_TRAVEL, rows=chunk)

        c = s.run(
            "MATCH (p:POI) WITH count(p) AS pois "
            "OPTIONAL MATCH (a:Accommodation) WITH pois, count(a) AS acc "
            "OPTIONAL MATCH ()-[t:TRAVEL]->() WITH pois, acc, count(t) AS travel "
            "MATCH (d:District) RETURN pois, acc, travel, count(d) AS districts"
        ).single()
        log.info("Loaded: %d POIs, %d accommodation, %d travel edges, %d districts",
                 c["pois"], c["acc"], c["travel"], c["districts"])

        # Verify the properties the retrieval layer queries actually landed.
        # Writing a graph that silently lacks them is worse than failing here.
        probe = s.run(
            "MATCH (p:POI) RETURN "
            "count(p) AS total, "
            "count(p.prominence) AS prominence, "
            "count(p.popularity_index) AS popularity_index, "
            "count(p.rail_access_score) AS rail_access_score, "
            "count(p.entry_fee_lkr) AS entry_fee_lkr, "
            "count(p.location) AS location, "
            "count(p.wikipedia) AS wikipedia"
        ).single()
        total = probe["total"]
        gaps = {k: probe[k] for k in
                ("prominence", "popularity_index", "rail_access_score",
                 "entry_fee_lkr", "location") if probe[k] < total}
        # wikipedia is sparse by nature (~7% of places have an article), so it is
        # reported rather than treated as a gap. Zero would mean the loader is
        # not writing it at all, which is what broke the photographs.
        log.info("Wikipedia titles stored: %d of %d places (%.1f%%)",
                 probe["wikipedia"], total,
                 100.0 * probe["wikipedia"] / max(total, 1))
        if probe["wikipedia"] == 0:
            raise RuntimeError(
                "No Wikipedia titles were stored. Photographs and the crowding "
                "signal both depend on this field; check that pois.parquet has a "
                "'wikipedia' column and re-run `python -m etl.enrich`.")
        if gaps:
            raise RuntimeError(
                f"Schema verification FAILED. Of {total} POIs, these properties "
                f"are missing on some nodes: {gaps}. Candidate retrieval filters "
                f"on them and would return zero rows.")
        log.info("Schema verification passed: all %d POIs carry the properties "
                 "retrieval depends on", total)

        # Marker consumed by `python -m etl.status` so the tracker can report
        # this stage without needing live database credentials.
        import json as _json
        from datetime import datetime as _dt
        (DATA_PROC / ".load_neo4j.json").write_text(_json.dumps({
            "at": _dt.now().isoformat(timespec="seconds"),
            "pois": c["pois"], "acc": c["acc"],
            "travel": c["travel"], "districts": c["districts"],
        }), encoding="utf-8")
    driver.close()

    print("\nNext: python -m etl.data_card")
    return 0


if __name__ == "__main__":
    sys.exit(main())