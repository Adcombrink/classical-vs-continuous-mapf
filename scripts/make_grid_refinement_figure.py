#!/usr/bin/env python
"""Builds the grid-refinement comparison figure.

empty-32-32 runs empty-16-16's scenarios with every coordinate doubled, so
both maps solve the same physical instances at two resolutions. The question:
for an agent half the largest size a 4-connected grid represents soundly, is
refining the grid as good as moving to continuous time?

Usage:
    python scripts/make_grid_refinement_figure.py [results_dir] [figures_dir]
"""

from __future__ import annotations

import pathlib
import sys

from ctvsdt import load
from ctvsdt import plots


def main() -> None:
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    figures_dir = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "figures")

    records = load.load_all(results_dir)
    empty = [r for r in records
             if r.map_name in (plots.COARSE_MAP, plots.FINE_MAP)]
    print(f"loaded {len(records)} records, {len(empty)} on the two empty maps")

    out_path = figures_dir / "F_grid_refinement"
    sizes = plots.plot_grid_refinement(empty, out_path)
    print("instances averaged over, per agent count:")
    for n, count in sizes.items():
        print(f"  n={n}: {count}")
    print(f"\nfigure written to {out_path}")


if __name__ == "__main__":
    main()
