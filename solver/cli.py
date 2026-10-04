"""
End-to-end: retrieve candidates from Neo4j, then solve for an itinerary.

  python -m solver.cli --days 5 --budget 120000 --interests heritage,wildlife

Scoring here is a PLACEHOLDER. Preference uses the prominence already stored in
the graph and sustainability uses inverted crowding, so the solver can be
exercised before the agents exist. Phase 4 replaces both with the Preference
Agent (LLM, ID-validated) and the Sustainability Agent (pure arithmetic).

No language model is involved in this command.
"""
from __future__ import annotations
import argparse
import logging
import sys
from datetime import date, timedelta

from etl.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from retrieval.models import TripRequest, Category
from retrieval.service import CandidateRetriever
from .models import SolverConfig
from .search import solve

log = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description="Travora retrieval + solver")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--budget", type=float, default=120000)
    ap.add_argument("--party", type=int, default=2)
    ap.add_argument("--interests", default="heritage,wildlife")
    ap.add_argument("--lat", type=float, default=7.2906)
    ap.add_argument("--lon", type=float, default=80.6337)
    ap.add_argument("--radius", type=float, default=250.0)
    ap.add_argument("--w-pref", type=float, default=0.5,
                    help="1.0 = preference only, 0.0 = sustainability only")
    ap.add_argument("--slots", type=int, default=3, help="max stops per day")
    ap.add_argument("--time-budget", type=float, default=10.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not NEO4J_URI or not NEO4J_PASSWORD:
        raise SystemExit("Load your .env first (NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD)")

    try:
        cats = [Category(c.strip()) for c in args.interests.split(",") if c.strip()]
    except ValueError:
        raise SystemExit(f"Valid interests: {', '.join(c.value for c in Category)}")

    req = TripRequest(
        start_date=date.today() + timedelta(days=14), days=args.days,
        budget_lkr=args.budget, party_size=args.party, interests=cats,
        start_lat=args.lat, start_lon=args.lon, w_pref=args.w_pref,
        max_travel_radius_km=args.radius)

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            r = CandidateRetriever(session)
            cset = r.retrieve(req)
            edges = r.travel_times(cset)
    finally:
        driver.close()

    if not cset.candidates:
        raise SystemExit("No candidates retrieved; nothing to solve.")

    # Placeholder scores. Phase 4 replaces these with the agents.
    pref = {c.poi_id: c.prominence for c in cset.candidates}
    sust = {c.poi_id: round(1.0 - (c.popularity_index if c.popularity_known else 0.3), 4)
            for c in cset.candidates}

    cfg = SolverConfig(max_slots_per_day=args.slots,
                       time_budget_s=args.time_budget)
    res = solve(cset.request_id, cset.candidates, req.days, req.poi_budget_lkr,
                pref, sust, req.w_pref, edges, cfg)

    bar = "=" * 84
    print(f"\n{bar}\nITINERARY\n{bar}")
    print(f"Request : {req.days} days from {req.start_date}, "
          f"LKR {req.budget_lkr:,.0f}, party {req.party_size}")
    print(f"Weights : preference {req.w_pref}, sustainability {req.w_sust}")
    print(f"Pool    : {len(cset)} candidates across {cset.districts_covered} districts")

    d = res.diagnostics
    print(f"\nCSP     : {d.variables} variables, domain {d.domain_size_before} "
          f"-> {d.domain_size_after} "
          f"({d.domain_reduction_pct}% pruned by AC-3, {d.arc_revisions} revisions)")
    if d.domain_capped_to:
        print(f"          top-{d.domain_capped_to} cap then reduced it to "
              f"{d.domain_size_capped} (a search heuristic, reported separately "
              f"from AC-3)")
    print(f"Search  : {d.search_nodes:,} nodes in {d.time_ms:.0f} ms"
          f"{' (BOUNDED)' if d.hit_time_budget or d.hit_node_cap else ''}")

    if not res.feasible:
        print(f"\nNO FEASIBLE ITINERARY")
        print(f"  conflict   : {res.conflict.conflict.value}")
        print(f"  detail     : {res.conflict.detail}")
        if res.conflict.suggestion:
            print(f"  suggestion : {res.conflict.suggestion}")
        print(bar)
        return 2

    it = res.itinerary
    print(f"Result  : {len(it.stops)} stops, LKR {it.total_cost_lkr:,.0f} in fees, "
          f"U={it.utility:.3f} (pref {it.s_pref:.3f}, sust {it.s_sust:.3f})")
    print(f"Optimal : {it.optimal}")
    print(f"Travel  : {d.travel_edges_stored} legs from stored road data, "
          f"{d.travel_edges_estimated} estimated")
    if d.soft_violations:
        print(f"Soft    : {d.soft_violations}")

    for day in it.days:
        print(f"\n--- Day {day.day} "
              f"({len(day.stops)} stops, LKR {day.cost_lkr:,.0f}) ---")
        for s in day.stops:
            flags = []
            if not s.hours_enforced:
                flags.append("hours est.")
            if s.travel_estimated and s.slot > 0:
                flags.append("travel est.")
            note = f"  [{', '.join(flags)}]" if flags else ""
            leg = f"  (+{s.travel_from_prev_min:.0f} min travel)" if s.slot > 0 else ""
            print(f"  {s.arrive_hhmm}-{s.depart_hhmm}  {s.name[:38]:<40}"
                  f"{s.district[:12]:<13}LKR {s.cost_lkr:>6,.0f}{leg}{note}")

    print(f"\nDistricts visited: {sorted(it.districts)}")
    print(bar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
