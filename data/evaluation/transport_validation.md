# Transport model validation

## Rail

Reference times are APPROXIMATE (see config/reference_journeys.yaml) and must be checked against the official timetable before citing.

Fitted effective speeds: **lowland 47.6 km/h, hill 24.1 km/h, fast 64.0 km/h, slow 24.0 km/h**.

| Route | Track km | Hill km | Fast km | Reference (min) | Model (min) | Leave-one-out (min) |
|---|---|---|---|---|---|---|
| Colombo Fort -> Kandy | 121 | 38 | 0 | 165 | 199 | 204 |
| Colombo Fort -> Galle | 114 | 0 | 0 | 140 | 144 | 146 |
| Colombo Fort -> Matara | 156 | 0 | 0 | 210 | 197 | 182 |
| Colombo Fort -> Anuradhapura | 206 | 0 | 131 | 255 | 217 | 209 |
| Colombo Fort -> Jaffna | 397 | 0 | 323 | 380 | 396 | 500 |
| Colombo Fort -> Trincomalee | 297 | 0 | 64 | 465 | 441 | 425 |
| Colombo Fort -> Batticaloa | 350 | 0 | 64 | 510 | 529 | 560 |
| Colombo Fort -> Badulla | 293 | 193 | 0 | 600 | 607 | 620 |
| Kandy -> Ella | 163 | 147 | 0 | 405 | 386 | 374 |

Mean absolute error: **7.1%** in sample, **13.3%** leave-one-out.

## Road

40 random pairs of real places, 5-150 km apart, compared with OSRM road routes.

| Measure | Old estimate vs OSRM |
|---|---|
| Mean absolute distance error | 19.0% |
| Mean absolute time error | 30.1% |
| Median road / straight-line ratio | 1.62 (model assumes 1.35) |
| Share of legs where the estimate was too short | 22% |

_Generated 2026-09-26 16:58_
