#!/usr/bin/env python
"""Merges the per-agent-count CBS pieces into one CSV per map.

    python scripts/merge_cbs.py --map room-64-64-8
    python scripts/merge_cbs.py --all

CBSH2-RTC is single-threaded and `run_cbs.sh` appends to one file per map,
so running a map's agent counts as separate jobs writes one piece each under
`results/cbs_parts/<map>_n<count>/<map>.csv`. This puts them back together
under `results/cbs/<map>.csv`, where `load.load_all` looks.
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys

PARTS_DIR = pathlib.Path("results/cbs_parts")
OUT_DIR = pathlib.Path("results/cbs")


def merge_map(map_name: str) -> int:
    """Writes one map's merged CSV and returns the row count.

    args:
        map_name: MovingAI map name.

    returns:
        Number of data rows written.

    raises:
        SystemExit: if pieces disagree on the header, or two rows describe
            the same (scenario, agent count, time limit).
    """
    pieces = sorted(PARTS_DIR.glob(f"{map_name}_n*/{map_name}.csv"))
    if not pieces:
        sys.exit(f"no pieces under {PARTS_DIR}/{map_name}_n*/")

    header = None
    rows: list[str] = []
    seen: dict[tuple[str, str, str], str] = {}
    for piece in pieces:
        lines = piece.read_text().splitlines()
        if not lines:
            continue
        if header is None:
            header = lines[0]
        elif lines[0] != header:
            sys.exit(f"{piece} has a different header from {pieces[0]}")
        for row in lines[1:]:
            if not row.strip():
                continue
            f = row.split(",")
            key = (f[1], f[2], f[3])          # scenario, agents, timeout
            if key in seen:
                sys.exit(f"duplicate row for scenario={f[1]} agents={f[2]} "
                         f"timeout={f[3]}: in {seen[key]} and {piece}")
            seen[key] = str(piece)
            rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{map_name}.csv"
    out.write_text("\n".join([header, *rows]) + "\n")
    counts = collections.Counter(r.split(",")[2] for r in rows)
    print(f"{out}: {len(rows)} rows from {len(pieces)} piece(s)")
    for agents in sorted(counts, key=int):
        print(f"  n={agents}: {counts[agents]}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--map", dest="map_name")
    group.add_argument("--all", action="store_true",
                       help="merge every map with pieces on disk")
    args = parser.parse_args()

    if args.map_name:
        merge_map(args.map_name)
        return
    maps = sorted({p.name.rsplit("_n", 1)[0] for p in PARTS_DIR.glob("*_n*")
                   if p.is_dir()})
    if not maps:
        sys.exit(f"no pieces under {PARTS_DIR}")
    for map_name in maps:
        merge_map(map_name)


if __name__ == "__main__":
    main()
