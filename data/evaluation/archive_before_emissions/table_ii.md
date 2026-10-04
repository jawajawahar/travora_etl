# Table II - Technical evaluation

| Metric | rule_based | single_agent_llm | greedy_eco | travora |
|---|---|---|---|---|
| Answer rate (%) | 100.0 | 97.2 | 100.0 | 66.7 |
| Constraint satisfaction rate, returned plans (%) | 0.0 | 0.0 | 0.0 | 100.0 |
| Constraint satisfaction rate, all queries (%) | 0.0 | 0.0 | 0.0 | 66.7 |
| False feasibility claims (%) | 100.0 | 100.0 | 100.0 | 0.0 |
| Mean stops | 18.89 | 16.25 | 20.08 | 9.50 |
| Mean entry cost (LKR) | 17,065 | 27,405 | 13,308 | 10,113 |
| Mean preference score | n/a | n/a | n/a | 0.871 |
| Mean sustainability score | 0.626 | 0.613 | 0.766 | 0.726 |
| Catalogue coverage (%) | 4.6 | 8.1 | 6.0 | 4.1 |
| Gini index | 0.971 | 0.956 | 0.961 | 0.972 |
| Distinct-plan rate (%) | 72.2 | 97.2 | 63.9 | 55.6 |
| AC-3 domain reduction (%) | n/a | n/a | n/a | 0.58 |
| Search nodes (mean) | n/a | n/a | n/a | 31,705 |
| Response time p50 (s) | 0.00 | 2.04 | 0.00 | 6.32 |
| Response time p95 (s) | 0.00 | 6.44 | 0.00 | 7.12 |

## Violation breakdown (% of runs)

| Violation | rule_based | single_agent_llm | greedy_eco | travora |
|---|---|---|---|---|
| unknown_poi | 0.0 | 5.6 | 0.0 | 0.0 |
| duplicate_poi | 0.0 | 3.7 | 0.0 | 0.0 |
| budget_exceeded | 0.0 | 51.9 | 0.0 | 0.0 |
| day_overflow | 100.0 | 91.7 | 100.0 | 0.0 |
| opening_hours | 72.2 | 46.3 | 55.6 | 0.0 |
| empty_plan | 0.0 | 2.8 | 0.0 | 33.3 |
| too_many_days | 0.0 | 0.0 | 0.0 | 0.0 |
| transfer_too_long | 100.0 | 94.4 | 77.8 | 0.0 |

_Generated 2026-09-26 18:31_
