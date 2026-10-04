# Table II - Technical evaluation

| Metric | rule_based | single_agent_llm | greedy_eco | travora |
|---|---|---|---|---|
| Answer rate (%) | 100.0 | 99.1 | 100.0 | 66.7 |
| Constraint satisfaction rate, returned plans (%) | 0.0 | 0.0 | 0.0 | 100.0 |
| Constraint satisfaction rate, all queries (%) | 0.0 | 0.0 | 0.0 | 66.7 |
| False feasibility claims (%) | 100.0 | 100.0 | 100.0 | 0.0 |
| Mean stops | 18.89 | 17.66 | 20.08 | 9.47 |
| Mean entry cost (LKR) | 17,065 | 27,859 | 9,280 | 10,641 |
| Mean preference score | n/a | n/a | n/a | 0.875 |
| Mean place sustainability score | 0.564 | 0.566 | 0.647 | 0.614 |
| Travel emissions (kg CO2e / traveller-day) | 15.61 | 9.66 | 18.16 | 2.56 |
| Trip sustainability (0.6 places + 0.4 journey) | 0.574 | 0.638 | 0.597 | 0.741 |
| Catalogue coverage (%) | 4.6 | 8.2 | 5.8 | 4.2 |
| Gini index | 0.971 | 0.956 | 0.963 | 0.973 |
| Distinct-plan rate (%) | 72.2 | 100.0 | 55.6 | 52.8 |
| AC-3 domain reduction (%) | n/a | n/a | n/a | 0.58 |
| Search nodes (mean) | n/a | n/a | n/a | 43,378 |
| Response time p50 (s) | 0.00 | 2.11 | 0.00 | 6.23 |
| Response time p95 (s) | 0.00 | 6.40 | 0.00 | 7.29 |

## Violation breakdown (% of runs)

| Violation | rule_based | single_agent_llm | greedy_eco | travora |
|---|---|---|---|---|
| unknown_poi | 0.0 | 3.7 | 0.0 | 0.0 |
| duplicate_poi | 0.0 | 5.6 | 0.0 | 0.0 |
| budget_exceeded | 0.0 | 53.7 | 0.0 | 0.0 |
| day_overflow | 100.0 | 98.1 | 100.0 | 0.0 |
| opening_hours | 72.2 | 50.0 | 75.0 | 0.0 |
| empty_plan | 0.0 | 0.9 | 0.0 | 33.3 |
| too_many_days | 0.0 | 0.0 | 0.0 | 0.0 |
| transfer_too_long | 100.0 | 94.4 | 100.0 | 0.0 |

## Travel emissions sensitivity (kg CO2e / traveller-day)

| Factor scenario | rule_based | single_agent_llm | greedy_eco | travora | Lowest-emission system |
|---|---|---|---|---|---|
| low | 13.28 | 8.16 | 15.43 | 1.94 | travora |
| central | 15.61 | 9.66 | 18.16 | 2.56 | travora |
| high | 21.62 | 13.91 | 25.33 | 5.76 | travora |

The ranking of systems by emissions is the same under all three scenarios.

_Generated 2026-09-27 00:00_
