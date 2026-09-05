#!/usr/bin/env python
"""Ports MovingAI scenarios onto a finer grid by scaling every coordinate.

    python scripts/port_scenarios.py --from-map empty-16-16 \
        --to-map empty-32-32 --scale 2

Each start/goal pair (x, y) becomes (scale*x, scale*y), putting the same
physical instance on a grid `scale` times finer, so one instance can be
solved at two discretizations. An agent of radius sqrt(2)/8 on the coarse
grid is the same physical agent as one of sqrt(2)/4 on a grid twice as fine.

Costs are NOT adjusted: a finer path covers the same distance in `scale`
times as many cell-lengths, and the records keep that literal value. Halving
belongs in the plotting, where it can be stated.

Output scenarios are named after the SOURCE map with a scale marker, as in
`empty-16-16-random-1-x2.scen`, which is what reaches `scen_file` and keeps a
ported instance from colliding with the target map's own scenario of the same
number.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

BENCHMARKS = pathlib.Path("benchmarks")


def read_map(map_path: pathlib.Path) -> tuple[int, int, list[str]]:
    """Returns a MovingAI map's width, height and its rows of terrain.

    args:
        map_path: path to the .map file.

    returns:
        (width, height, rows), rows being the terrain lines in order.
    """
    lines = map_path.read_text().splitlines()
    header = {}
    for index, line in enumerate(lines):
        if line.strip() == "map":
            return int(header["width"]), int(header["height"]), lines[index + 1:]
        key, _, value = line.partition(" ")
        header[key] = value
    raise ValueError(f"{map_path} has no 'map' line")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-map", required=True, dest="source_map")
    parser.add_argument("--to-map", required=True, dest="target_map")
    parser.add_argument("--scale", type=int, default=2,
                        help="resolution multiplier (default 2)")
    parser.add_argument("--out-dir", default=None,
                        help="where to write, default "
                             "benchmarks/<to-map>/scenarios_x<scale>")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_dir = BENCHMARKS / args.source_map / "scenarios"
    target_map_path = BENCHMARKS / args.target_map / f"{args.target_map}.map"
    width, height, rows = read_map(target_map_path)
    out_dir = pathlib.Path(args.out_dir or
                           BENCHMARKS / args.target_map /
                           f"scenarios_x{args.scale}")

    scens = sorted(source_dir.glob("*.scen"))
    if not scens:
        sys.exit(f"no .scen files under {source_dir}")
    print(f"{len(scens)} scenarios: {source_dir} -> {out_dir}")
    print(f"target {args.target_map} is {width}x{height}, scale {args.scale}")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for scen in scens:
        out_lines = ["version 1"]
        for line in scen.read_text().splitlines()[1:]:
            if not line.strip():
                continue
            f = line.split("\t")
            bucket, _, _, _, sx, sy, gx, gy, length = f[:9]
            coords = [int(v) * args.scale for v in (sx, sy, gx, gy)]
            for x, y in ((coords[0], coords[1]), (coords[2], coords[3])):
                if not (0 <= x < width and 0 <= y < height):
                    sys.exit(f"{scen.name}: ({x},{y}) is outside "
                             f"{width}x{height}")
                if rows[y][x] not in ".G":
                    sys.exit(f"{scen.name}: ({x},{y}) is not passable "
                             f"on {args.target_map} (terrain {rows[y][x]!r})")
            out_lines.append("\t".join([
                bucket, f"{args.target_map}.map", str(width), str(height),
                *(str(v) for v in coords),
                f"{float(length) * args.scale:.8f}"]))
            total += 1
        out_name = f"{scen.stem}-x{args.scale}.scen"
        if not args.dry_run:
            (out_dir / out_name).write_text("\n".join(out_lines) + "\n")

    print(f"{total} pairs scaled, {len(scens)} files "
          f"{'checked' if args.dry_run else 'written'}")


if __name__ == "__main__":
    main()
