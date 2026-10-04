# Travora Dataset — Data Card

**Generated:** 2026-08-18
**Records:** 2245 points of interest
**Overall acceptance:** PASS

---

## 1. Sources and Licences

| Source | Provides | Licence | Attribution required |
|---|---|---|---|
| OpenStreetMap (Overpass) | POIs, coordinates, opening hours, fees, categories | ODbL 1.0 | **Yes** — "© OpenStreetMap contributors" |
| Wikidata / Wikipedia | Descriptions, identifiers | CC0 / CC BY-SA | Yes for text extracts |
| Wikipedia Pageviews API | Popularity and seasonality signal | CC0 | No |
| OSRM | Road travel durations | BSD | No |
| SLTDA | Registered accommodation, visitor statistics | Public government data | Cite |

**Not used, and why.** TripAdvisor and Google Places content is *not* stored. Both
providers licence access rather than retention: TripAdvisor permits caching of the
Location ID only, and Google permits the `place_id` only. Where such content is shown
it is fetched live and discarded. Google Popular Times is not exposed by the Places
API and every library offering it scrapes Google Maps, so it is not used.

---

## 2. Coverage

### By category

| Category | POIs | Share |
|---|---|---|
| urban | 612 | 27.3% |
| heritage | 526 | 23.4% |
| cultural | 417 | 18.6% |
| adventure | 212 | 9.4% |
| nature | 181 | 8.1% |
| religious | 124 | 5.5% |
| beach | 102 | 4.5% |
| wildlife | 71 | 3.2% |

### By district

| District | POIs |
|---|---|
| Kandy | 178 |
| Badulla | 177 |
| Galle | 177 |
| Colombo | 174 |
| Nuwara Eliya | 146 |
| Anuradhapura | 145 |
| Ampara | 140 |
| Ratnapura | 117 |
| Matale | 105 |
| Matara | 102 |
| Gampaha | 100 |
| Monaragala | 96 |
| Kurunegala | 82 |
| Jaffna | 73 |
| Polonnaruwa | 72 |
| Hambantota | 62 |
| Kegalle | 56 |
| Puttalam | 55 |
| Kalutara | 38 |
| Trincomalee | 36 |
| Vavuniya | 32 |
| Batticaloa | 30 |
| Mullaitivu | 23 |
| Mannar | 16 |
| Kilinochchi | 13 |

**All 25 districts represented.**

---

## 3. Data Quality and Imputation

Values absent from the source are imputed from category defaults and **flagged**.
Silent imputation would make the dataset appear complete while being fabricated.

| Field | Real | Imputed | Method when imputed |
|---|---|---|---|
| Coordinates | 100.0% | 0.0% | never imputed; records without coordinates are dropped |
| Opening hours | 2.9% | 97.1% | category default |
| Entry fee | 0.4% | 99.6% | category median |
| Dwell time | 0.0% | 100.0% | category default — no source provides this |
| Popularity | 7.3% | 92.7% | 0.0 where no Wikipedia article exists |

**Eco-certification** is recorded as `unknown` rather than `false` where no record
exists, because absence of evidence is not evidence of absence.

**Opening hours carry a constraint implication.** OpenStreetMap coverage of
`opening_hours` in Sri Lanka is very low, so most values are category defaults.
The itinerary solver therefore treats opening hours as a **hard constraint only
where `hours_estimated` is False**, and as a soft preference otherwise.
Enforcing a hard constraint against an imputed value would mean rejecting
feasible itineraries on the basis of fiction, and would inflate the apparent
constraint-satisfaction rate without any real guarantee.

**District assignment** uses inverse-distance-weighted k-nearest-neighbour
classification against SLTDA accommodation records that carry both a published
coordinate and a district assigned by the national tourism authority. Each POI
records a `district_method` and a `district_confidence`.

An earlier nearest-centroid method was replaced because it misassigned Sigiriya,
Dambulla Cave Temple and Pidurangala - the three principal Cultural Triangle
sites - to Polonnaruwa. Sri Lankan districts are not compact, and the reference
point used was the district's principal town rather than its geographic centre.

---

## 4. Acceptance Criteria

| ID | Criterion | Observed | Result |
|---|---|---|---|
| A1 | Usable POIs >= 400 | 2245 | PASS |
| A2 | Districts with >= 5 POIs (of 25) | 25/25 | PASS |
| A3 | Categories with >= 30 POIs (of 8) | 8/8 | PASS |
| A4 | POIs with real coordinates | 100.0% | PASS |
| A5 | Opening hours parsed >= 0.5% | 2.9% | PASS |
| A7 | Travel matrix built | 35320 pairs, 3.6% OSRM | PASS |
| A8 | Imputation counted per field | reported below | PASS |

---

## 5. Known Limitations

1. **Crowding is a proxy, not a measurement.** Derived from Wikipedia pageviews;
   validate against published SLTDA/DWC visitor figures and report Spearman ρ. If
   ρ < 0.4 the proxy is weak and must be reported as a limitation.
2. **OSM coverage is uneven.** Northern and eastern districts are typically less
   mapped than the south-west, which may under-represent them.
3. **Dwell times are category defaults**, not observations.
4. **Travel durations are typical, not live.** Traffic and weather are not modelled.
5. **Fees drift.** Entry fees change; the extraction date is recorded above.

---

## 6. Reproducing This Dataset

```bash
python -m etl.extract_osm      # raw dump -> data/raw/
python -m etl.clean            # normalise, categorise, dedupe, impute
python -m etl.enrich           # Wikipedia popularity and seasonality
python -m etl.travel_matrix    # OSRM durations
python -m etl.load_neo4j       # load into Neo4j
python -m etl.data_card        # regenerate this card
```

Raw dumps are retained in `data/raw/` with their extraction date so any figure in
this card can be traced back to source.
