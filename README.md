# Travora ETL — Phase 1

Builds the real Sri Lankan tourism dataset that replaces the hardcoded data, and
loads it into Neo4j.

This is the fix for the root cause your supervisor identified. The old system
repeated itineraries because there was nothing real to select *from*; the language
model was inventing places from its own priors. This pipeline produces the
candidate pool that makes data-driven selection possible.

---

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -v          # 9 test groups, all offline

cp .env.example .env                # add your AuraDB credentials
set -a && source .env && set +a

python -m etl.run                   # full pipeline, ~25-50 min
```

Stages can also be run individually:

```bash
python -m etl.extract_osm     # 1. Overpass -> data/raw/          (1-4 min)
python -m etl.clean           # 2. normalise, categorise, dedupe  (<1 min)
python -m etl.enrich          # 3. Wikipedia popularity           (15-40 min)
python -m etl.travel_matrix   # 4. OSRM durations                 (5-20 min)
python -m etl.load_neo4j      # 5. load into Neo4j                (2-5 min)
python -m etl.data_card       # 6. DATA_CARD.md + acceptance gate (<1 min)
```

Useful flags: `--skip-osm` (reuse the raw dump), `--no-osrm` (distance model only),
`--no-load` (stop before Neo4j).

---

## What each stage does

| Stage | Module | Output |
|---|---|---|
| 1 | `extract_osm` | Dated raw Overpass dump, never modified |
| 2 | `clean` | `pois.parquet` — categorised, deduplicated, imputation flagged |
| 3 | `enrich` | `popularity_index`, `seasonality_json` from Wikipedia pageviews |
| 4 | `travel_matrix` | `travel_matrix.parquet` — OSRM durations, haversine fallback |
| 5 | `load_neo4j` | Graph with constraints, indexes and a spatial point index |
| 6 | `data_card` | `data/DATA_CARD.md` + pass/fail on acceptance criteria |

---

## Three design decisions worth knowing

**Imputation is always flagged.** Where OSM has no `opening_hours`, a category
default is applied *and* `hours_estimated = true` is set. This turns a hidden
weakness into a reportable statistic: *"58% of opening hours were parsed from OSM,
42% imputed from category defaults."* Silent imputation makes a dataset look
complete while being partly fabricated.

**`eco_certified` is `unknown`, never `false`.** Absence of a certification record
is not evidence that a property lacks certification.

**No commercial API content is stored.** TripAdvisor licenses caching of the
Location ID only; Google licenses the `place_id` only. `load_neo4j.assert_no_tier_b()`
aborts the load if a rating or review field ever reaches the graph. Those sources
belong in the client as a live, attributed display layer.

---

## Acceptance gate

`python -m etl.data_card` exits non-zero if the dataset is not acceptable, so it
works as a CI gate:

| ID | Criterion |
|---|---|
| A1 | ≥ 400 usable POIs |
| A2 | All 25 districts have ≥ 5 POIs |
| A3 | All 8 categories have ≥ 30 POIs |
| A4 | 100% real coordinates |
| A5 | ≥ 50% opening hours parsed from source |
| A7 | Travel matrix built |
| A8 | Imputation counted per field |

**A2 is the anti-hardcoding gate at the data level.** If Jaffna, Mannar and
Monaragala hold no POIs, the system *cannot* recommend them, and popularity bias is
baked in before any algorithm runs.

---

## Graph model

```
(:POI {poi_id, name, lat, lon, location, category, open_min, close_min,
       entry_fee_lkr, typical_dwell_min, popularity_index, seasonality_json,
       hours_estimated, fee_estimated, dwell_estimated,
       eco_certified, community_run, district, unesco})

(:POI)-[:IN_DISTRICT]->(:District {name})
(:POI)-[:HAS_CATEGORY]->(:Category {name})
(:POI)-[:TRAVEL {distance_km, duration_min, method}]->(:POI)
```

`TRAVEL` is stored one-directional and traversed undirected, halving relationship
count for the AuraDB free tier (200k nodes / 400k relationships).

Candidate retrieval — the query the Retrieval Agent runs:

```cypher
MATCH (p:POI)-[:HAS_CATEGORY]->(c:Category)
WHERE c.name IN $interests
  AND p.entry_fee_lkr <= $max_fee
  AND point.distance(p.location, point({latitude:$lat, longitude:$lon})) <= $radius_m
RETURN p.poi_id AS poi_id, p.name AS name, p.category AS category,
       p.lat AS lat, p.lon AS lon, p.open_min AS open_min, p.close_min AS close_min,
       p.entry_fee_lkr AS fee, p.typical_dwell_min AS dwell,
       p.popularity_index AS popularity, p.district AS district
ORDER BY p.popularity_index ASC
LIMIT $limit
```

Note `ORDER BY popularity ASC` — under-visited places surface first, which is the
sustainability objective expressed directly in retrieval.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Overpass 429 / 504 | Shared free service under load | Built-in backoff retries; or try `OVERPASS_URL=https://overpass.kumi.systems/api/interpreter` |
| A1 fails (too few POIs) | Overpass returned partial data | Re-run extraction; check the raw dump element count |
| A2 fails (districts missing) | Genuine OSM gap in the north/east | Supplement with Wikidata or manual entry; record the gap in DATA_CARD |
| `travel_matrix` all haversine | OSRM demo unreachable/rate-limited | Acceptable for a first pass; self-host OSRM for the final run |
| `TIER B VIOLATION` on load | Commercial API field reached the graph | Working as designed — remove the column |

---

## Next after this phase

1. Run the pipeline and commit `DATA_CARD.md`.
2. Confirm A1–A8 pass, or agree relaxed thresholds with your supervisor.
3. Phase 2: candidate retrieval service over this graph.
4. Phase 3: the CSP solver (AC-3 + branch and bound), still with no LLM.

The language model is not involved until Phase 4, and only ever to *score* a list
it was given — never to choose what is on it.
