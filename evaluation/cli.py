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
from .querysets import build_query_set, weight_sweep_queries, Query
from .harness import Harness, run_gates, gate_t8_fake_id, SYSTEMS
from .metrics import VIOLATIONS, diversity

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
        ("Answer rate (%)", "answer_rate_pct", "{:.1f}"),
        ("Constraint satisfaction rate, returned plans (%)", "csr_pct", "{:.1f}"),
        ("Constraint satisfaction rate, all queries (%)", "csr_all_queries_pct", "{:.1f}"),
        ("False feasibility claims (%)", "false_feasible_pct", "{:.1f}"),
        ("Mean stops", "mean_stops", "{:.2f}"),
        ("Mean entry cost (LKR)", "mean_cost", "{:,.0f}"),
        ("Mean preference score", "mean_pref", "{:.3f}"),
        ("Mean place sustainability score", "mean_sust", "{:.3f}"),
        ("Travel emissions (kg CO2e / traveller-day)", "mean_kg_ptd", "{:.2f}"),
        ("Trip sustainability (0.6 places + 0.4 journey)", "mean_trip_sust", "{:.3f}"),
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
            elif isinstance(val, float) and val != val:      # NaN
                cells.append("n/a")
            else:
                cells.append(fmt.format(val))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    # Why plans fail, not only how often: each cell is the share of that
    # system's runs showing the violation, from the independent verifier.
    ran = [s for s in SYSTEMS if summaries.get(s) and summaries[s].runs]
    vlines = ["| Violation | " + " | ".join(ran) + " |", "|---" * (len(ran) + 1) + "|"]
    for v in VIOLATIONS:
        vlines.append(f"| {v} | " + " | ".join(
            f"{summaries[s].violations.get(v, 0.0):.1f}" for s in ran) + " |")

    # Sensitivity to the emission factors: the same plans under the low,
    # central and high factor of every mode (sources in SOURCES.md).
    slines = ["| Factor scenario | " + " | ".join(ran) + " | Lowest-emission system |",
              "|---" * (len(ran) + 2) + "|"]
    order_by = {}
    for sc in ("low", "central", "high"):
        vals = {s: summaries[s].kg_by_scenario.get(sc) for s in ran}
        known = {s: v for s, v in vals.items() if v is not None}
        best = min(known, key=known.get) if known else "n/a"
        order_by[sc] = tuple(sorted(known, key=known.get))
        slines.append(f"| {sc} | " + " | ".join(
            f"{vals[s]:.2f}" if vals[s] is not None else "n/a" for s in ran) + f" | {best} |")
    stable = len(set(order_by.values())) == 1
    smd = "\n".join(slines) + ("\n\nThe ranking of systems by emissions is the same under all "
                                "three scenarios." if stable else
                                "\n\nThe ranking of systems by emissions CHANGES between "
                                "scenarios; conclusions depend on the factors.")

    md = "\n".join(lines)
    vmd = "\n".join(vlines) if ran else "_No system completed a run._"
    path.write_text("# Table II - Technical evaluation\n\n" + md +
                    "\n\n## Violation breakdown (% of runs)\n\n" + vmd +
                    "\n\n## Travel emissions sensitivity (kg CO2e / traveller-day)\n\n" + smd +
                    f"\n\n_Generated {datetime.now():%Y-%m-%d %H:%M}_\n",
                    encoding="utf-8")
    return md + "\n\n" + vmd


def run_sweep(h: Harness, dmap: dict, out: Path) -> str:
    """Trade-off frontier: the same query cells at each interest weight."""
    queries = weight_sweep_queries()
    recs = h.run_system(queries, "travora")
    by_w: dict[float, list] = {}
    for q, r in zip(queries, recs):
        by_w.setdefault(q.w_pref, []).append(r)

    lines = ["| w_pref | w_sust | Answer rate (%) | Mean preference | Mean sustainability "
             "| Catalogue coverage (%) | Gini | Distinct-plan rate (%) | Districts |",
             "|---" * 9 + "|"]
    rows = []
    for w in sorted(by_w):
        s = Harness.summarise("travora", by_w[w], h.catalogue_size, dmap)
        d = s.diversity
        rows.append([w, round(1 - w, 2), s.answer_rate_pct, s.mean_pref, s.mean_sust,
                     d.catalogue_coverage_pct, d.gini, d.distinct_plan_pct,
                     d.districts_covered])
        lines.append(f"| {w:.2f} | {1 - w:.2f} | {s.answer_rate_pct:.1f} | "
                     f"{s.mean_pref:.3f} | {s.mean_sust:.3f} | "
                     f"{d.catalogue_coverage_pct:.1f} | {d.gini:.3f} | "
                     f"{d.distinct_plan_pct:.1f} | {d.districts_covered} |")

    md = "\n".join(lines)
    (out / "weight_sweep.md").write_text(
        "# Interest / sustainability trade-off\n\nEach row runs the same 36 query "
        "cells at one weight. w_pref = 1 optimises interest only; 0 optimises "
        "sustainability only.\n\n" + md +
        f"\n\n_Generated {datetime.now():%Y-%m-%d %H:%M}_\n", encoding="utf-8")
    with open(out / "weight_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["w_pref", "w_sust", "answer_rate_pct", "mean_pref", "mean_sust",
                    "coverage_pct", "gini", "distinct_plan_pct", "districts"])
        w.writerows(rows)
    return md


def run_alternatives(h: Harness, queries: list[Query], k: int, dmap: dict,
                     out: Path) -> str:
    """
    Diversity gained by offering k alternatives per query, and what it costs.

    Coverage and Gini are computed twice over the same catalogue: once over the
    best plans only, once over every plan offered.
    """
    recs = h.run_system(queries, "travora", alternatives=k)
    primary = [r.plan for r in recs if r.verdict.feasible]
    offered = primary + [p for r in recs for p in r.alt_plans]
    quality = [x for r in recs for x in r.alt_quality]
    with_alt = sum(1 for r in recs if r.alt_plans)

    d1 = diversity(primary, h.catalogue_size, dmap)
    d2 = diversity(offered, h.catalogue_size, dmap)
    mean_q = sum(quality) / len(quality) if quality else float("nan")

    md = "\n".join([
        "| Measure | Best plan only | Best + alternatives |",
        "|---|---|---|",
        f"| Plans | {len(primary)} | {len(offered)} |",
        f"| Distinct POIs used | {d1.distinct_pois} | {d2.distinct_pois} |",
        f"| Catalogue coverage (%) | {d1.catalogue_coverage_pct:.1f} | {d2.catalogue_coverage_pct:.1f} |",
        f"| Gini index | {d1.gini:.3f} | {d2.gini:.3f} |",
        f"| Districts reached | {d1.districts_covered} | {d2.districts_covered} |",
        f"| Top-5 share (%) | {d1.top5_share_pct:.1f} | {d2.top5_share_pct:.1f} |",
    ])
    note = (f"Queries with at least one alternative: {with_alt}/{len(recs)}. "
            + (f"Mean alternative utility: {mean_q:.1%} of the best plan "
               if quality else "No alternatives were produced. ")
            + "(Floor 90%; each shares at most half its stops with earlier plans.)")
    (out / "alternatives.md").write_text(
        f"# Diversity from alternatives (k={k})\n\n{md}\n\n{note}\n\n"
        f"_Generated {datetime.now():%Y-%m-%d %H:%M}_\n", encoding="utf-8")
    return md + "\n\n" + note


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
    ap.add_argument("--sweep", action="store_true",
                    help="interest/sustainability weight sweep (trade-off frontier)")
    ap.add_argument("--alternatives", type=int, default=0, metavar="K",
                    help="measure diversity gained by K alternative plans per query")
    ap.add_argument("--out", default=None,
                    help="output directory (default data/evaluation); use a scratch "
                         "directory for smoke runs so real results are not overwritten")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not (args.gates or args.full or args.sweep or args.alternatives):
        args.gates = True

    if not NEO4J_URI or not NEO4J_PASSWORD:
        raise SystemExit("Load your .env first (NEO4J_URI / NEO4J_PASSWORD)")

    out = Path(args.out) if args.out else OUT
    out.mkdir(parents=True, exist_ok=True)
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

                with open(out / "gates.csv", "w", newline="", encoding="utf-8") as f:
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

                md = write_table_ii(summaries, out / "table_ii.md")
                print(f"\n{md}\n")

                with open(out / "runs.csv", "w", newline="", encoding="utf-8") as f:
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

            if args.sweep:
                print(f"\n-- Weight sweep --")
                print(run_sweep(h, dmap, out))

            if args.alternatives:
                print(f"\n-- Alternatives (k={args.alternatives}) --")
                print(run_alternatives(h, queries, args.alternatives, dmap, out))

            print(f"\nWritten to {out}")
            print(bar)
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
