"""
Run candidate retrieval against the live Neo4j graph.

  python -m retrieval.cli --days 7 --budget 120000 --interests heritage,wildlife

Use this to sanity-check the graph before the solver exists: it shows what the
system would have to choose from, how the candidates are distributed, and
whether under-visited destinations are actually surfacing.
"""
from __future__ import annotations
import argparse
import logging
import sys
from datetime import date, timedelta

from etl.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from .models import TripRequest, Category
from .service import CandidateRetriever

log = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(description="Travora candidate retrieval")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--budget", type=float, default=120000)
    ap.add_argument("--party", type=int, default=2)
    ap.add_argument("--interests", default="heritage,wildlife")
    ap.add_argument("--lat", type=float, default=7.2906, help="start latitude (Kandy)")
    ap.add_argument("--lon", type=float, default=80.6337, help="start longitude")
    ap.add_argument("--radius", type=float, default=250.0)
    ap.add_argument("--show", type=int, default=20)
    ap.add_argument("--travel", action="store_true",
                    help="also report stored travel-edge coverage")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not NEO4J_URI or not NEO4J_PASSWORD:
        raise SystemExit("Load your .env first: NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD")

    try:
        cats = [Category(c.strip()) for c in args.interests.split(",") if c.strip()]
    except ValueError:
        raise SystemExit(f"Valid interests: {', '.join(c.value for c in Category)}")

    req = TripRequest(
        start_date=date.today() + timedelta(days=14),
        days=args.days, budget_lkr=args.budget, party_size=args.party,
        interests=cats, start_lat=args.lat, start_lon=args.lon,
        max_travel_radius_km=args.radius,
    )

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            r = CandidateRetriever(session)
            cset = r.retrieve(req)
            travel = r.travel_times(cset) if args.travel else {}
    finally:
        driver.close()

    bar = "=" * 74
    print(f"\n{bar}\nCANDIDATE RETRIEVAL\n{bar}")
    print(f"Request  : {req.days} days, LKR {req.budget_lkr:,.0f}, "
          f"party {req.party_size}, interests {[c.value for c in req.interests]}")
    print(f"Weights  : preference {req.w_pref}, sustainability {req.w_sust}")
    print(f"Result   : {len(cset)} candidates across {cset.districts_covered} "
          f"districts in {cset.retrieval_ms:.0f} ms")
    if cset.query_widened:
        print(f"WIDENED  : {cset.widening_reason}")

    d = r.diagnostics
    print(f"\nBy category : {d.category_counts}")
    print(f"By district : {dict(list(d.district_counts.items())[:8])}")
    print(f"Mean prominence {d.mean_prominence:.3f} | "
          f"popularity known for {d.popularity_known_pct:.1f}% of candidates "
          f"(mean {d.mean_popularity:.3f} among those)")

    if travel:
        print(f"Travel edges: {len(travel)} stored pairs")

    print(f"\nTop {args.show} candidates (prominence, less known crowding). "
          f"pop '-' = not measured:")
    print(f"{'name':<36}{'category':<10}{'district':<13}"
          f"{'prom':>6}{'pop':>7}{'score':>7}{'fee':>8}")
    print("-" * 87)
    for c in cset.candidates[:args.show]:
        pop = f"{c.popularity_index:.2f}" if c.popularity_known else "  -"
        print(f"{c.name[:35]:<36}{c.category.value:<10}{c.district[:12]:<13}"
              f"{c.prominence:>6.2f}{pop:>7}{c.retrieval_score:>7.2f}"
              f"{c.entry_fee_lkr:>8,.0f}")
    print(bar)

    if len(cset) < 15:
        print("\nWARNING: fewer than 15 candidates. The solver will have little "
              "room to optimise. Widen the radius or the interest list.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
