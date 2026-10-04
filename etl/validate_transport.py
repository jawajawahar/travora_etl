"""
Calibrate and validate the transport model.

    python -m etl.validate_transport            # rail + road
    python -m etl.validate_transport --no-road  # rail only (no OSRM calls)

Rail: fits the effective train speeds (lowland, hill, fast, slow) to the reference
journeys in config/reference_journeys.yaml by least squares on 1/speed, and
reports both in-sample error and leave-one-out error (each journey predicted by
a model fitted without it), the honest out-of-sample figure.

Road: for random pairs of real places, compares the solver's fallback estimate
(straight line x 1.35 at 35 km/h) with the OSRM route, which is why finished
itineraries are re-measured with OSRM.

Writes data/evaluation/transport_validation.md.
"""
from __future__ import annotations
import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import DATA_PROC, OSRM_URL
from solver.routing import RailNetwork, road_legs, estimate_road, haversine_km

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "config" / "reference_journeys.yaml"
OUT = DATA_PROC.parent / "evaluation" / "transport_validation.md"


CLASSES = ("lowland", "hill", "fast", "slow")


def fit(X, T):
    """Least squares for T = X @ w, w = minutes per km for each track class."""
    w, *_ = np.linalg.lstsq(X, T, rcond=None)
    return w


def rail_validation():
    refs = yaml.safe_load(REF.read_text(encoding="utf-8"))["rail"]
    net = RailNetwork()
    rows = []
    for r in refs:
        sa, sb = net.station_named(r["from"]), net.station_named(r["to"])
        if not sa or not sb:
            print(f"  skipped {r['from']} -> {r['to']}: station not in graph")
            continue
        rows.append({"route": f"{r['from']} -> {r['to']}", "a": sa["node"], "b": sb["node"],
                     "ref": float(r["minutes"])})

    # Path choice depends on the speeds, so fit, re-route, and fit again.
    speeds = dict(net.speeds)
    for _ in range(4):
        net.speeds = speeds
        net.path.cache_clear()
        for r in rows:
            p = net.path(r["a"], r["b"])
            r["km"], r["split"] = (p[1], p[2]) if p else (None, None)
        ok = [r for r in rows if r["km"] is not None]
        X = np.array([[r["split"].get(c, 0.0) for c in CLASSES] for r in ok])
        T = np.array([r["ref"] for r in ok])
        w = fit(X, T)
        speeds = {c: round(60.0 / m, 1) if m > 0 else speeds[c] for c, m in zip(CLASSES, w)}

    for i, r in enumerate(ok):
        r["hill"] = r["split"].get("hill", 0.0)
        r["fast"] = r["split"].get("fast", 0.0)
        r["pred"] = sum(r["split"].get(c, 0.0) * 60.0 / speeds[c] for c in CLASSES)
        mask = np.arange(len(ok)) != i
        r["loo"] = float(X[i] @ fit(X[mask], T[mask]))

    mape = np.mean([abs(r["pred"] - r["ref"]) / r["ref"] for r in ok]) * 100
    loo = np.mean([abs(r["loo"] - r["ref"]) / r["ref"] for r in ok]) * 100
    return ok, speeds, mape, loo


def road_validation(n=40, seed=7):
    pois = pd.read_parquet(DATA_PROC / "pois.parquet")[["lat", "lon"]].to_numpy()
    rng = random.Random(seed)
    pairs = []
    while len(pairs) < n:
        a, b = pois[rng.randrange(len(pois))], pois[rng.randrange(len(pois))]
        d = haversine_km(a[0], a[1], b[0], b[1])
        if 5 <= d <= 150:
            pairs.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
    out = []
    for a, b in pairs:
        measured = road_legs([a, b], OSRM_URL)[0]
        if measured:
            ek, em = estimate_road(a, b)
            out.append({"est_km": ek, "est_min": em, "km": measured[0], "min": measured[1]})
        time.sleep(1.0)                      # OSRM demo server: one request a second
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-road", action="store_true")
    args = ap.parse_args()

    ok, speeds, mape, loo = rail_validation()
    lines = ["# Transport model validation", "",
             "## Rail", "",
             "Reference times are APPROXIMATE (see config/reference_journeys.yaml) "
             "and must be checked against the official timetable before citing.", "",
             f"Fitted effective speeds: **lowland {speeds['lowland']} km/h, "
             f"hill {speeds['hill']} km/h, fast {speeds['fast']} km/h, slow {speeds['slow']} km/h**.", "",
             "| Route | Track km | Hill km | Fast km | Reference (min) | Model (min) | Leave-one-out (min) |",
             "|---|---|---|---|---|---|---|"]
    for r in ok:
        lines.append(f"| {r['route']} | {r['km']:.0f} | {r['hill']:.0f} | {r['fast']:.0f} | "
                     f"{r['ref']:.0f} | {r['pred']:.0f} | {r['loo']:.0f} |")
    lines += ["", f"Mean absolute error: **{mape:.1f}%** in sample, "
                  f"**{loo:.1f}%** leave-one-out.", ""]
    print("\n".join(lines))

    if not args.no_road:
        road = road_validation()
        if road:
            df = pd.DataFrame(road)
            km_err = (abs(df.est_km - df.km) / df.km).mean() * 100
            min_err = (abs(df.est_min - df["min"]) / df["min"]).mean() * 100
            ratio = (df.km / df.est_km * 1.35).median()
            under = (df.est_min < df["min"]).mean() * 100
            road_lines = ["## Road", "",
                          f"{len(df)} random pairs of real places, 5-150 km apart, "
                          "compared with OSRM road routes.", "",
                          "| Measure | Old estimate vs OSRM |", "|---|---|",
                          f"| Mean absolute distance error | {km_err:.1f}% |",
                          f"| Mean absolute time error | {min_err:.1f}% |",
                          f"| Median road / straight-line ratio | {ratio:.2f} (model assumes 1.35) |",
                          f"| Share of legs where the estimate was too short | {under:.0f}% |", ""]
            print("\n".join(road_lines))
            lines += road_lines

    lines.append(f"_Generated {datetime.now():%Y-%m-%d %H:%M}_")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
