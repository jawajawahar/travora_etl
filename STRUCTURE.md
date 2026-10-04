# TRAVORA ETL — Structure and Run Guide

**Read this first if you lose track.** Everything you need is on this page.

Last updated: 17 August 2026

---

## 1. Where everything goes

Put this folder inside your project. On your machine that is:

```
C:\Users\ASUS TUF\Desktop\Final year project\Dev\Travoravs\travora_etl\
```

Full structure — **21 files**:

```
travora_etl\
│
├── README.md                       overview
├── STRUCTURE.md                    this file
├── Makefile                        shortcuts (optional on Windows)
├── requirements.txt                python packages
├── .env.example                    template for Neo4j credentials
│
├── config\
│   ├── categories.yaml             OSM tag -> 8 interest categories
│   └── sustainability.yaml         ALL sustainability weights (edit here)
│
├── etl\
│   ├── __init__.py                 EMPTY FILE — must exist
│   ├── config.py                   paths, thresholds, Tier-B guard list
│   ├── extract_osm.py       [1]    download POIs from OpenStreetMap
│   ├── clean.py             [2]    categorise, dedupe, impute
│   ├── extract_sltda.py     [3]    SLTDA hotels + sustainability scores
│   ├── enrich.py            [4]    Wikipedia popularity + prominence
│   ├── rail.py              [5]    railway stations + rail access score
│   ├── travel_matrix.py     [6]    kNN travel graph
│   ├── load_neo4j.py        [7]    load everything into Neo4j
│   ├── data_card.py         [8]    DATA_CARD.md + acceptance gate
│   ├── inspect.py                  dataset summary (run any time)
│   └── run.py                      runs 1,2,4,6,7,8 in sequence
│
├── tests\
│   ├── test_clean.py               9 tests — cleaning pipeline
│   ├── test_sltda.py               8 tests — accommodation + rail
│   └── fixtures\
│       └── sample_overpass.json    test data
│
└── data\                           created automatically
    ├── raw\                        downloaded source files
    ├── processed\                  pois.parquet, accommodation.parquet, etc.
    └── DATA_CARD.md                generated report
```

`[n]` = order to run. See section 3.

---

## 2. One-time setup

```powershell
cd "C:\Users\ASUS TUF\Desktop\Final year project\Dev\Travoravs\travora_etl"

pip install -r requirements.txt
python -m pytest tests\ -v
```

Expected: **17 passed** (9 from test_clean, 8 from test_sltda).

Then Neo4j credentials:

```powershell
copy .env.example .env
notepad .env
```

Fill in from your AuraDB console:

```
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

PowerShell does not read `.env` automatically. Load it each session:

```powershell
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*([^#][^=]*)=(.*)$') {
    [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
  }
}
echo $env:NEO4J_URI      # should print your URI
```

---

## 3. Run order

Run from inside `travora_etl\`. Each step writes a file the next step reads, so
order matters. Any step can be re-run safely.

| # | Command | Time | Produces |
|---|---|---|---|
| 1 | `python -m etl.extract_osm` | 1–4 min | `data\raw\osm_pois_*.json` |
| 2 | `python -m etl.clean` | <1 min | `data\processed\pois.parquet` |
| 3 | `python -m etl.extract_sltda --csv "C:\path\to\sltda_accommodations.csv"` | <1 min | `accommodation.parquet` |
| 4 | `python -m etl.enrich` | 10–25 min | adds popularity + prominence |
| 5 | `python -m etl.rail` | 2–5 min | adds rail access to both files |
| 6 | `python -m etl.travel_matrix` | 5–15 min | `travel_matrix.parquet` |
| 7 | `python -m etl.load_neo4j` | 3–8 min | loads Neo4j |
| 8 | `python -m etl.data_card` | <1 min | `data\DATA_CARD.md` |

Check progress any time:

```powershell
python -m etl.inspect
```

### Notes per step

**Step 1** — Overpass is a free shared service. 429/504 errors are retried
automatically. If all attempts fail, use the mirror:
```powershell
$env:OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"
```

**Step 3** — point `--csv` at wherever your SLTDA file is. Or copy it into
`data\raw\` and run `python -m etl.extract_sltda` with no arguments.

**Step 4** — slowest step, but **resumable**. Results cache to
`data\raw\pageviews_cache.json`. If interrupted, just run it again; already
fetched titles cost nothing.

**Step 8** — exits with an error code if the dataset fails the acceptance
criteria. That is intentional, not a bug.

---

## 4. What each module does

| Module | Purpose |
|---|---|
| `extract_osm.py` | Downloads every tourism feature in Sri Lanka from OpenStreetMap via the Overpass API. Saves a dated raw dump that is never modified, so results are traceable. |
| `clean.py` | Drops unnamed/uncoordinated/out-of-country records, maps OSM tags to 8 categories, merges duplicates (same place as both node and way), parses opening hours, assigns districts. **Every imputed value is flagged.** |
| `extract_sltda.py` | Loads the SLTDA register, recovers missing coordinates from AGA-division centroids, and computes an accommodation sustainability score from SLTDA category + room count + grade. |
| `enrich.py` | Wikipedia pageviews → `popularity_index` (crowding proxy) and `seasonality_json`. Also computes `prominence`, a separate signal for whether a POI is a real destination. |
| `rail.py` | Downloads railway stations, computes distance from every POI and hotel to the nearest one. Train is the lowest-emission mode in Sri Lanka. |
| `travel_matrix.py` | Builds a k-nearest-neighbour travel graph (k=25). All-pairs would be 66 million edges; this is ~158,000, which fits AuraDB Free. |
| `load_neo4j.py` | Creates constraints and indexes, loads POIs, accommodation and travel edges. **Aborts if any commercial API content is found in the data.** |
| `data_card.py` | Writes `DATA_CARD.md` for your dissertation appendix and checks the acceptance criteria. |
| `inspect.py` | Prints a dataset summary. Safe to run any time. |

---

## 5. Two ideas to keep straight

**`popularity_index` vs `prominence`** — these are different on purpose.

- **prominence** = "is this a real visitor destination?" (from Wikidata presence,
  UNESCO status, recorded opening hours, website tags, category)
- **popularity_index** = "is this crowded?" (from Wikipedia pageview volume)

A quiet UNESCO site scores **high prominence, low popularity** — exactly the kind
of place the sustainability objective should surface. Retrieval ranks by
prominence; the sustainability score penalises popularity.

**Tier A vs Tier B data** — Tier A (OSM, SLTDA, Wikipedia, OSRM) is stored in
Neo4j and drives the solver. Tier B (TripAdvisor, Google) may be displayed live
in the app but **never stored** — their terms licence access, not retention.
`load_neo4j.py` aborts the load if Tier B fields appear.

---

## 6. Acceptance criteria

`python -m etl.data_card` checks these and fails if unmet:

| ID | Criterion |
|---|---|
| A1 | ≥ 400 usable POIs |
| A2 | All 25 districts have ≥ 5 POIs |
| A3 | All 8 categories have ≥ 30 POIs |
| A4 | 100% real coordinates |
| A5 | ≥ 50% opening hours parsed from source |
| A7 | Travel matrix built |
| A8 | Imputation counted per field |

**A2 matters most.** If Jaffna and Mannar hold no POIs, the system *cannot*
recommend them, and popularity bias is baked in before any algorithm runs.

---

## 7. Results so far

From your real SLTDA file (2,130 records):

```
Category                        score   median rooms    n
Home Stay Units                  1.00        3.00      378
Bangalows                        0.75        4.00      348
Guest Houses                     0.65        9.00      893
Boutique Hotels                  0.40       16.00       28
Tourist Hotels                   0.30       28.00      229
Classified Hotels (1-5 Star)     0.20       60.00      141

Spearman rho = -0.785, p = 0.0042
Districts: 25/25    Records kept: 2,129 / 2,130
Coordinates: 1,367 published + 719 division centroid + 43 district centroid
```

The sustainability ordering is **validated against independent room-count data**
in the same register. Put this in Chapter 3.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named 'etl'` | You are in the wrong folder. `cd travora_etl` first. Also confirm `etl\__init__.py` exists (empty file). |
| `ERROR: file or directory not found: tests/` | Same cause — wrong folder. |
| `no tests ran` | Wrong folder, or the `tests\` folder was not copied. |
| Overpass 429/504 | Automatic retry; else use the kumi.systems mirror (section 3). |
| Wikipedia 429 storm | Fixed in the current `enrich.py`. If you still see it, raise `MIN_INTERVAL` from `0.40` to `0.80`. |
| `Run 'python -m etl.clean' first` | Steps run out of order. Follow the table in section 3. |
| `TIER B VIOLATION` on load | Working as designed — commercial API content reached the data. Remove that column. |
| A1/A2/A3 fail | Not enough data. Re-run step 1 and check the raw dump size. |

---

## 9. What comes next (not built yet)

Phase 1 (this ETL) is complete. Still to build:

1. **Candidate retrieval service** — Cypher query layer over the graph
2. **CSP solver** — AC-3 + branch and bound, **no LLM**
3. **Agents** — Preference and Sustainability scoring with ID validation
4. **API + orchestration** — FastAPI, CrewAI, typed blackboard
5. **Anti-regression suite** — catalogue coverage, Gini, perturbation tests

The full plan is in `TRAVORA_REBUILD_PLAN.md` (separate file).
