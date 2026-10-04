"""
Pipeline status tracker and interactive menu.

Shows which stages have run, what they produced, and what to run next. Status is
derived from the artifacts actually on disk rather than from a log file, so it
cannot go stale or disagree with reality: if a parquet is deleted the stage
reports as pending, and if a column is missing the stage that produces it
reports as incomplete.

  python -m etl.status          status table only
  python -m etl.status --menu   interactive menu to run stages
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .config import DATA_RAW, DATA_PROC

ROOT = DATA_PROC.parent.parent
MARKER = DATA_PROC / ".load_neo4j.json"

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, YELLOW, RED, CYAN = "\033[32m", "\033[33m", "\033[31m", "\033[36m"


def _colour_ok() -> bool:
    if sys.platform == "win32":
        try:                                   # enable ANSI on Windows 10+
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
            return True
        except Exception:                      # noqa: BLE001
            return False
    return sys.stdout.isatty()


COLOUR = _colour_ok()


def c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if COLOUR else text


# ---------------------------------------------------------------------------
def _newest(pattern: str) -> Path | None:
    found = sorted(DATA_RAW.glob(pattern))
    return found[-1] if found else None


def _parquet_info(name: str, required_cols: tuple[str, ...] = ()) -> dict:
    """Rows, modification time, and whether the required columns are present."""
    path = DATA_PROC / name
    if not path.exists():
        return {"exists": False}
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        missing = [col for col in required_cols if col not in df.columns]
        return {
            "exists": True, "rows": len(df), "missing": missing,
            "mtime": datetime.fromtimestamp(path.stat().st_mtime),
            "df": df,
        }
    except Exception as e:                     # noqa: BLE001
        return {"exists": True, "error": str(e)}


def collect() -> list[dict]:
    """Build the stage table. Each entry: name, command, state, detail."""
    stages: list[dict] = []

    def add(n, cmd, state, detail, when=None):
        stages.append({"n": n, "cmd": cmd, "state": state,
                       "detail": detail, "when": when})

    # 1 extract_osm
    raw = _newest("osm_pois_*.json")
    if raw:
        try:
            n = len(json.loads(raw.read_text(encoding="utf-8")).get("elements", []))
            add("1  extract_osm", "python -m etl.extract_osm", "DONE",
                f"{n:,} raw elements",
                datetime.fromtimestamp(raw.stat().st_mtime))
        except Exception:                      # noqa: BLE001
            add("1  extract_osm", "python -m etl.extract_osm", "ERROR",
                "raw dump unreadable")
    else:
        add("1  extract_osm", "python -m etl.extract_osm", "PENDING",
            "no raw OSM dump")

    # 2 clean
    p = _parquet_info("pois.parquet")
    if p.get("exists") and "rows" in p:
        add("2  clean", "python -m etl.clean", "DONE",
            f"{p['rows']:,} POIs", p["mtime"])
    else:
        add("2  clean", "python -m etl.clean", "PENDING", "no pois.parquet")

    # 3 extract_sltda
    a = _parquet_info("accommodation.parquet")
    if a.get("exists") and "rows" in a:
        add("3  extract_sltda", "python -m etl.extract_sltda --csv <path>", "DONE",
            f"{a['rows']:,} properties", a["mtime"])
    else:
        add("3  extract_sltda", "python -m etl.extract_sltda --csv <path>",
            "PENDING", "no accommodation.parquet")

    df = p.get("df")

    # 4 boundaries / districts
    if df is not None and "district_method" in df.columns:
        methods = set(df.district_method.dropna().unique())
        if any(str(m).startswith("boundary") for m in methods):
            covered = int((df.district.value_counts() >= 5).sum())
            add("4  boundaries", "python -m etl.boundaries", "DONE",
                f"point-in-polygon, {covered}/25 districts >=5", p.get("mtime"))
        elif "sltda_knn" in methods:
            add("4  boundaries", "python -m etl.boundaries", "PARTIAL",
                "kNN fallback in use; boundaries preferred", p.get("mtime"))
        else:
            add("4  boundaries", "python -m etl.boundaries", "PARTIAL",
                "centroid values only", p.get("mtime"))
    else:
        add("4  boundaries", "python -m etl.boundaries", "PENDING",
            "districts not refined")

    # 5 enrich
    if df is not None and "prominence" in df.columns:
        pop = (df.popularity_source == "wikipedia").mean() * 100 \
            if "popularity_source" in df.columns else 0.0
        add("5  enrich", "python -m etl.enrich", "DONE",
            f"prominence set, popularity {pop:.1f}%", p.get("mtime"))
    else:
        add("5  enrich", "python -m etl.enrich", "PENDING", "no prominence column")

    # 6 rail
    if df is not None and "rail_access_score" in df.columns:
        walk = (df.nearest_station_km <= 2).mean() * 100
        add("6  rail", "python -m etl.rail", "DONE",
            f"{walk:.1f}% within 2 km of a station", p.get("mtime"))
    else:
        add("6  rail", "python -m etl.rail", "PENDING", "no rail_access_score")

    # 7 travel_matrix
    m = _parquet_info("travel_matrix.parquet")
    if m.get("exists") and "rows" in m:
        osrm = 0.0
        if m.get("df") is not None and "method" in m["df"].columns:
            osrm = (m["df"].method == "osrm").mean() * 100
        warn = "  OVER AURA FREE LIMIT" if m["rows"] > 400_000 else ""
        add("7  travel_matrix", "python -m etl.travel_matrix", "DONE",
            f"{m['rows']:,} edges, {osrm:.1f}% OSRM{warn}", m["mtime"])
    else:
        add("7  travel_matrix", "python -m etl.travel_matrix", "PENDING",
            "no travel_matrix.parquet")

    # 8 load_neo4j
    if MARKER.exists():
        try:
            info = json.loads(MARKER.read_text(encoding="utf-8"))
            add("8  load_neo4j", "python -m etl.load_neo4j", "DONE",
                f"{info.get('pois', 0):,} POIs, {info.get('acc', 0):,} acc, "
                f"{info.get('travel', 0):,} edges",
                datetime.fromisoformat(info["at"]))
        except Exception:                      # noqa: BLE001
            add("8  load_neo4j", "python -m etl.load_neo4j", "DONE", "loaded")
    else:
        add("8  load_neo4j", "python -m etl.load_neo4j", "PENDING",
            "not loaded into Neo4j")

    # 9 data_card
    card = DATA_PROC.parent / "DATA_CARD.md"
    if card.exists():
        text = card.read_text(encoding="utf-8", errors="ignore")
        failed = text.count("| FAIL |")
        state = "DONE" if failed == 0 else "FAILED"
        detail = "all criteria passed" if failed == 0 \
            else f"{failed} criteria failed"
        add("9  data_card", "python -m etl.data_card", state, detail,
            datetime.fromtimestamp(card.stat().st_mtime))
    else:
        add("9  data_card", "python -m etl.data_card", "PENDING",
            "no DATA_CARD.md")

    return stages


STATE_COLOUR = {"DONE": GREEN, "PENDING": DIM, "PARTIAL": YELLOW,
                "FAILED": RED, "ERROR": RED}


def render(stages: list[dict]) -> None:
    bar = "=" * 78
    print(bar)
    print(c(f"{BOLD}TRAVORA PIPELINE STATUS{RESET}" if COLOUR
            else "TRAVORA PIPELINE STATUS", ""))
    print(bar)
    print(f"{'Stage':<22}{'Status':<10}{'Detail':<32}{'Updated':<14}")
    print("-" * 78)
    for s in stages:
        when = s["when"].strftime("%d %b %H:%M") if s.get("when") else ""
        print(f"{s['n']:<22}"
              f"{c(s['state'], STATE_COLOUR.get(s['state'], '')):<{10 + (len(STATE_COLOUR.get(s['state'],'')) + 4 if COLOUR else 0)}}"
              f"{s['detail'][:31]:<32}{when:<14}")
    print("-" * 78)

    done = sum(1 for s in stages if s["state"] == "DONE")
    print(f"{done}/{len(stages)} stages complete")

    nxt = next((s for s in stages if s["state"] in ("PENDING", "PARTIAL", "FAILED")), None)
    if nxt:
        print(f"\nNext:  {c(nxt['cmd'], CYAN)}")
    else:
        print(f"\n{c('Pipeline complete.', GREEN)} "
              "Next phase: candidate retrieval service.")
    print(bar)


def menu(stages: list[dict]) -> int:
    while True:
        render(stages)
        print("\nSelect a stage to run (number), [a] all pending, [r] refresh, [q] quit")
        try:
            choice = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if choice in ("q", "quit", "exit"):
            return 0
        if choice in ("r", ""):
            stages = collect()
            continue

        targets: list[dict] = []
        if choice == "a":
            targets = [s for s in stages if s["state"] in ("PENDING", "PARTIAL", "FAILED")]
        elif choice.isdigit():
            hit = [s for s in stages if s["n"].strip().startswith(choice)]
            if not hit:
                print("No such stage.")
                continue
            targets = hit[:1]
        else:
            print("Unrecognised option.")
            continue

        for t in targets:
            if "<path>" in t["cmd"]:
                print(f"\n{t['cmd']}")
                print("This stage needs the SLTDA CSV path. Run it manually.")
                continue
            print(f"\n>>> {t['cmd']}\n")
            subprocess.run(t["cmd"].split(), cwd=str(ROOT))
        stages = collect()


def main() -> int:
    ap = argparse.ArgumentParser(description="Travora pipeline status")
    ap.add_argument("--menu", action="store_true",
                    help="interactive menu to run stages")
    args = ap.parse_args()

    stages = collect()
    if args.menu:
        return menu(stages)
    render(stages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
