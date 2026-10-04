"""Rail network, leg construction and re-timing with measured travel times."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solver.routing import rail
from solver.logistics import build_legs, retime, Leg, START_ID

NET = rail()
needs_rail = pytest.mark.skipif(NET is None, reason="run python -m etl.rail_network first")


@needs_rail
def test_station_names_match_whole_words():
    """A substring match once turned "Ella" into "Avissawella"."""
    assert NET.station_named("Ella")["name"].lower().startswith("ella")
    assert NET.station_named("Kandy")["name"] == "Kandy"


@needs_rail
def test_known_lines_are_connected_with_plausible_track_length():
    for frm, to, lo, hi in [("Colombo Fort", "Kandy", 110, 130),
                            ("Colombo Fort", "Jaffna", 380, 410),
                            ("Colombo Fort", "Batticaloa", 330, 370)]:
        a, b = NET.station_named(frm), NET.station_named(to)
        p = NET.path(a["node"], b["node"])
        assert p is not None, f"{frm} -> {to} should be connected"
        assert lo <= p[1] <= hi, f"{frm} -> {to}: {p[1]:.0f} km of track"


class Stop:
    def __init__(self, pid, name, dwell=60, from_anchor=False):
        self.poi_id, self.name, self.dwell_min, self.from_anchor = pid, name, dwell, from_anchor


class Place:
    def __init__(self, lat, lon):
        self.lat, self.lon, self.hours_are_hard = lat, lon, False


def test_legs_cover_start_between_and_overnight():
    by_id = {"a": Place(7.30, 80.64), "b": Place(7.31, 80.65),
             "c": Place(7.32, 80.66), "d": Place(7.33, 80.67)}
    days = [[Stop("a", "A", from_anchor=True), Stop("b", "B")],
            [Stop("c", "C", from_anchor=True), Stop("d", "D")]]
    legs = build_legs(days, by_id, start=(7.2906, 80.6337), start_name="Kandy")
    assert [(l.from_id, l.to_id, l.kind) for l in legs] == [
        (START_ID, "a", "start"), ("a", "b", "between"),
        ("b", "c", "overnight"), ("c", "d", "between")]
    assert legs[0].from_name == "Kandy"
    assert all(l.basis in ("estimate", "track") for l in legs), "no road routing offline"


def test_retime_reports_a_day_that_no_longer_fits():
    by_id = {"a": Place(7.30, 80.64), "b": Place(7.31, 80.65)}
    day = [Stop("a", "A", dwell=240, from_anchor=True), Stop("b", "B", dwell=240)]
    legs = [Leg(START_ID, "a", 150.0, 200.0, "bus", 0.0, "", kind="start"),
            Leg("a", "b", 5.0, 20.0, "tuktuk", 0.0, "")]
    times, overflow, _ = retime([day], legs, by_id)
    assert times["a"][0] == 480 + 200 + 15, "the start transfer must use morning time"
    assert overflow[1] > 0, "a day past 18:00 must be reported, not hidden"
