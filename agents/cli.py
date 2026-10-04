"""
Run the evaluation and the anti-regression gates.

  python -m evaluation.cli --gates              anti-regression only (Phase 6)
  python -m evaluation.cli --full               all systems, Table II (Phase 8)
  python -m evaluation.cli --full --repeats 1   quick pass

Writes CSV and Markdown to data/evaluation/ so results can be pasted straight
into the paper and the raw records can be re-analysed.
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from etl.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATA_PROC
from solver.models import SolverConfig
from agents.preference import groq_completion
from .querysets import build_query_set, Query
from .harness import Harness, run_gates, gate_t8_fake_id, SYSTEMS

log = logging.getLogger(__name__)
OUT = DATA_PROC.parent / "evaluation"


def district_map(session) -> dict[str, str]:
    try:
        return {r["id"]: r["d"] for r in session.run(
            "MATCH (p:POI) RETURN p.poi_id AS id, p.district AS d")}
    except Exception:                                   # noqa: BLE001
        return {}


def write_table_ii(summaries: dict, path: Path) -> str:
    rows = [
        ("Constraint satisfaction rate (%)", "csr_pct", "{:.1f}"),
        ("False feasibility claims (%)", "false_feasible_pct", "{:.1f}"),
        ("Mean stops", "mean_stops", "{:.2f}"),
        ("Mean entry cost (LKR)", "mean_cost", "{:,.0f}"),
        ("Mean preference score", "mean_pref", "{:.3f}"),
        ("Mean sustainability score", "mean_sust", "{:.3f}"),
        ("Catalogue coverage (%)", None, "{:.1f}"),
        ("Gini index", None, "{:.3f}"),
        ("Distinct-plan rate (%)", None, "{:.1f}"),
        ("AC-3 domain reduction (%)", "ac3_reduction_pct", "{:.2f}"),
        ("Search nodes (mean)", "mean_nodes", "{:,.0f}"),
        ("Response time p50 (s)", "p50_s", "{:.2f}"),
        ("Response time p95 (s)", "p95_s", "{:.2f}"),
    ]
    div_keys = {"Catalogue coverage (%)": "catalogue_coverage_pct",
                "Gini index": "gini",
                "Distinct-plan rate (%)": "distinct_plan_pct"}

    header = "| Metric | " + " | ".join(SYSTEMS) + " |"
    sep = "|---" * (len(SYSTEMS) + 1) + "|"
    lines = [header, sep]
    for label, key, fmt in rows:
        cells = []
        for s in SYSTEMS:
            summ = summaries.get(s)
            if summ is None or summ.runs == 0:
                cells.append("n/a"); continue
            if key is None:
                val = getattr(summ.diversity, div_keys[label], None)
            else:
                val = getattr(summ, key, None)
            if val is None or (key in ("ac3_reduction_pct", "mean_nodes")
                               and s != "travora"):
                cells.append("n/a")
            else:
                cells.append(fmt.format(val))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    md = "\n".join(lines)
    path.write_text("# Table II - Technical evaluation\n\n" + md +
                    f"\n\n_Generated {datetime.now():%Y-%m-%d %H:%M}_\n",
                    encoding="utf-8")
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description="Travora evaluation")
    ap.add_argument("--gates", action="store_true", help="anti-regression only")
    ap.add_argument("--full", action="store_true", help="all systems, Table II")
    ap.add_argument("--repeats", type=int, default=3,
                    help="repeats per stratum (5 gives the full 180 queries)")
    ap.add_argument("--limit", type=int, default=0, help="cap the query count")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--time-budget", type=float, default=6.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not args.gates and not args.full:
        args.gates = True

    if not NEO4J_URI or not NEO4J_PASSWORD:
        raise SystemExit("Load your .env first (NEO4J_URI / NEO4J_PASSWORD)")

    OUT.mkdir(parents=True, exist_ok=True)
    queries = build_query_set(args.repeats)
    if args.limit:
        queries = queries[:args.limit]
    completion = None if args.no_llm else groq_completion(
        model=os.getenv("GROQ_MODEL"))
    cfg = SolverConfig(max_slots_per_day=args.slots,
                       time_budget_s=args.time_budget)

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    bar = "=" * 78
    try:
        with driver.session() as session:
            dmap = district_map(session)
            h = Harness(session, completion, cfg)
            print(f"\n{bar}\nTRAVORA EVALUATION\n{bar}")
            print(f"Catalogue : {h.catalogue_size:,} POIs")
            print(f"Queries   : {len(queries)} "
                  f"({args.repeats} repeats per stratum)")
            print(f"Model     : {'disabled' if completion is None else 'enabled'}")

            summaries = {}
            if args.gates:
                print(f"\n-- Anti-regression gates (Phase 6) --")
                gates, summ = run_gates(h, queries, "travora", dmap)
                gates.append(gate_t8_fake_id())
                summaries["travora"] = summ

                print(f"\n{'ID':<5}{'Gate':<46}{'Observed':<16}{'Target':<12}Result")
                print("-" * 92)
                for g in gates:
                    print(f"{g.gate_id:<5}{g.name[:45]:<46}{g.observed[:15]:<16}"
                          f"{g.threshold:<12}{'PASS' if g.passed else 'FAIL'}")
                failed = [g for g in gates if not g.passed]
                print(f"\n{len(gates) - len(failed)}/{len(gates)} gates passed")

                with open(OUT / "gates.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["gate", "name", "observed", "threshold", "passed"])
                    for g in gates:
                        w.writerow([g.gate_id, g.name, g.observed,
                                    g.threshold, g.passed])
                print(f"\nDiversity: {summ.diversity.summary()}")
                if h.llm_calls or h.llm_cache_hits:
                    print(f"Model calls: {h.llm_calls} scored, "
                          f"{h.llm_cache_hits} served from cache")

            if args.full:
                print(f"\n-- Full evaluation (Phase 8) --")
                all_records = []
                for s in SYSTEMS:
                    if s == "single_agent_llm" and completion is None:
                        log.warning("Skipping single_agent_llm: no model configured")
                        continue
                    if s in summaries:
                        continue
                    log.info("Running %s over %d queries", s, len(queries))
                    recs = h.run_system(queries, s)
                    all_records.extend(recs)
                    summaries[s] = Harness.summarise(s, recs, h.catalogue_size, dmap)

                md = write_table_ii(summaries, OUT / "table_ii.md")
                print(f"\n{md}\n")

                with open(OUT / "runs.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["query_id", "system", "feasible", "claimed",
                                "stops", "cost", "s_pref", "s_sust", "seconds",
                                "violations"])
                    for r in all_records:
                        w.writerow([r.query_id, r.system, r.verdict.feasible,
                                    r.claimed_feasible, r.verdict.stops,
                                    r.verdict.total_cost, r.s_pref, r.s_sust,
                                    round(r.seconds, 3),
                                    ";".join(r.verdict.violated)])

                print("Violation rates (% of runs):")
                for s, summ in summaries.items():
                    if summ.runs:
                        print(f"  {s:<18}{summ.violations or 'none'}")

            print(f"\nWritten to {OUT}")
            print(bar)
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
