"""
Tests for district assignment.

These cover the two regressions found on the real dataset:

  1. SLTDA spells the district "Moneragala"; this pipeline used "Monaragala".
     The mismatch produced one district holding 49 POIs and another holding
     zero, and district coverage silently dropped from 25/25 to 24/25.

  2. No SLTDA accommodation in Mullaitivu carries a published coordinate, so
     the nearest-neighbour classifier could never vote for that district and
     its POIs were absorbed by neighbouring districts.

Run:  python -m pytest tests/test_districts.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from etl.districts import (canonicalise_district, assign, CANONICAL,
                           DISTRICT_ANCHORS, ANCHOR_WEIGHT, MIN_REFERENCE_POINTS)


def test_canonicalisation():
    cases = [
        ("Moneragala", "Monaragala"),      # the SLTDA spelling
        ("Monaragala", "Monaragala"),
        ("Mullaithivu", "Mullaitivu"),
        ("MULLAITIVU", "Mullaitivu"),
        ("NuwaraEliya", "Nuwara Eliya"),
        ("Nuwara-Eliya", "Nuwara Eliya"),
        ("Amparai", "Ampara"),
        ("Kaluthara", "Kalutara"),
        ("Kegalla", "Kegalle"),
        ("Killinochchi", "Kilinochchi"),
        ("  Kandy  ", "Kandy"),
        ("Atlantis", None),
        (None, None),
        ("", None),
    ]
    for raw, expected in cases:
        got = canonicalise_district(raw)
        assert got == expected, f"{raw!r} -> {got!r}, expected {expected!r}"
    print(f"  canonicalisation: {len(cases)}/{len(cases)}")


def test_all_canonical_names_have_anchors():
    missing = set(CANONICAL) - set(DISTRICT_ANCHORS)
    assert not missing, f"districts without an anchor: {sorted(missing)}"
    assert len(CANONICAL) == 25, f"expected 25 districts, got {len(CANONICAL)}"
    print(f"  all {len(CANONICAL)} districts have anchors")


def test_anchors_are_self_consistent():
    """Every anchor must classify to its own district."""
    ref = pd.DataFrame([
        {"district": d, "lat": la, "lon": lo, "weight": ANCHOR_WEIGHT}
        for d, (la, lo) in DISTRICT_ANCHORS.items()
    ])
    pois = pd.DataFrame([
        {"name": d, "lat": la, "lon": lo, "district": "Unknown"}
        for d, (la, lo) in DISTRICT_ANCHORS.items()
    ])
    out = assign(pois, ref)
    wrong = out[out.name != out.district]
    assert wrong.empty, f"anchors misclassified: {wrong[['name','district']].to_dict('records')}"
    print(f"  {len(out)}/{len(out)} anchors classify to themselves")


def test_uncovered_district_still_reachable():
    """
    Mullaitivu has no SLTDA published coordinate. Its anchor must let a POI
    there be assigned correctly instead of being absorbed by Kilinochchi.
    """
    rng = np.random.default_rng(7)
    rows = []
    for d in ("Kilinochchi", "Jaffna", "Vavuniya"):
        la, lo = DISTRICT_ANCHORS[d]
        for _ in range(12):
            rows.append({"district": d,
                         "lat": la + rng.uniform(-.05, .05),
                         "lon": lo + rng.uniform(-.05, .05), "weight": 1.0})
    ref = pd.DataFrame(rows)
    per = ref.district.value_counts()
    anchors = [{"district": d, "lat": la, "lon": lo, "weight": ANCHOR_WEIGHT}
               for d, (la, lo) in DISTRICT_ANCHORS.items()
               if per.get(d, 0) < MIN_REFERENCE_POINTS]
    ref = pd.concat([ref, pd.DataFrame(anchors)], ignore_index=True)

    la, lo = DISTRICT_ANCHORS["Mullaitivu"]
    pois = pd.DataFrame({"name": ["Mullaitivu site"], "lat": [la], "lon": [lo],
                         "district": ["Kilinochchi"]})
    out = assign(pois, ref)
    assert out.iloc[0]["district"] == "Mullaitivu", \
        f"uncovered district not reachable, got {out.iloc[0]['district']}"
    print("  uncovered district reachable via anchor")


def test_real_sltda_points_outvote_anchors():
    """
    An anchor must never override genuine government-labelled data. A POI beside
    a cluster of real SLTDA points takes their district, not the anchor's.
    """
    ref = pd.DataFrame(
        [{"district": "Matale", "lat": 7.957 + i * 0.001, "lon": 80.760,
          "weight": 1.0} for i in range(10)]
        + [{"district": "Polonnaruwa", "lat": 7.9403, "lon": 81.0188,
            "weight": ANCHOR_WEIGHT}]
    )
    pois = pd.DataFrame({"name": ["Sigiriya"], "lat": [7.9570], "lon": [80.7603],
                         "district": ["Polonnaruwa"]})
    out = assign(pois, ref)
    assert out.iloc[0]["district"] == "Matale"
    assert out.iloc[0]["district_confidence"] > 0.9
    print(f"  real points outvote anchors "
          f"(confidence {out.iloc[0]['district_confidence']:.2f})")


# ---------------------------------------------------------------------------
# Boundary-based assignment (preferred method)
# ---------------------------------------------------------------------------
def test_boundary_polygonize_and_assign():
    """
    OSM boundary relations store their outline as UNORDERED way fragments, so
    the assembler must polygonize rather than assume ordering. This also checks
    that boundary names are canonicalised ("Moneragala" -> "Monaragala") and
    that offshore points snap to the nearest district at reduced confidence.
    """
    import json, tempfile
    from pathlib import Path
    try:
        import shapely  # noqa: F401
    except ImportError:
        print("  boundary tests skipped (shapely not installed)")
        return

    from etl.boundaries import build_polygons, assign as bassign

    def rel(rid, name, x0, y0, x1, y1):
        c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        segs = [c[2:4], c[0:2], c[3:5], c[1:3]]        # deliberately shuffled
        return {"type": "relation", "id": rid,
                "tags": {"name": name, "admin_level": "6",
                         "boundary": "administrative"},
                "members": [{"type": "way", "role": "outer",
                             "geometry": [{"lon": a, "lat": b} for a, b in s]}
                            for s in segs]}

    payload = {"elements": [
        rel(1, "Matale", 80.5, 7.6, 81.0, 8.1),
        rel(2, "Polonnaruwa", 81.0, 7.6, 81.5, 8.1),
        rel(3, "Moneragala", 81.0, 6.6, 81.6, 7.2),     # SLTDA spelling
    ]}
    p = Path(tempfile.mkdtemp()) / "osm_districts_test.json"
    p.write_text(json.dumps(payload), encoding="utf-8")

    polys = build_polygons(p)
    assert set(polys) == {"Matale", "Polonnaruwa", "Monaragala"}, \
        f"polygon assembly failed: {sorted(polys)}"

    pois = pd.DataFrame({
        "name": ["Sigiriya", "Vatadage", "Monaragala site", "Offshore islet"],
        "lat": [7.957, 7.94, 6.87, 8.30],
        "lon": [80.760, 81.02, 81.35, 80.75],
        "district": ["Polonnaruwa", "Polonnaruwa", "Ampara", "Matale"]})
    out = bassign(pois, polys)

    expected = {"Sigiriya": "Matale", "Vatadage": "Polonnaruwa",
                "Monaragala site": "Monaragala", "Offshore islet": "Matale"}
    for _, r in out.iterrows():
        assert r["district"] == expected[r["name"]], \
            f"{r['name']} -> {r['district']}, expected {expected[r['name']]}"

    islet = out[out.name == "Offshore islet"].iloc[0]
    assert islet["district_method"] == "boundary_nearest"
    assert islet["district_confidence"] < 1.0, \
        "snapped points must carry reduced confidence"
    print("  boundary assignment: 4/4, unordered ways polygonized, "
          "offshore snapped at reduced confidence")


if __name__ == "__main__":
    print("Travora - district assignment tests\n" + "-" * 52)
    for fn in [test_canonicalisation, test_all_canonical_names_have_anchors,
               test_anchors_are_self_consistent,
               test_uncovered_district_still_reachable,
               test_real_sltda_points_outvote_anchors,
               test_boundary_polygonize_and_assign]:
        fn()
    print("-" * 52 + "\nAll tests passed.")