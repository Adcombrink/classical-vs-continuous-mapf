#!/usr/bin/env python
"""Builds the cost-change figure, one row per map and one column per k.

Checks the CT lower bounds against the DT costs first and raises on any
violation, then prints the common instance count behind each drawn point.

Usage:
    python scripts/make_figures.py [results_dir] [figures_dir]
"""

from __future__ import annotations

import pathlib
import sys

from ctvsdt import load
from ctvsdt import plots
from ctvsdt import validate


def main() -> None:
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    figures_dir = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "figures")

    records = load.load_all(results_dir)
    print(f"loaded {len(records)} records")

    paired = load.pair_by_instance(records)
    # Restrict to conditions actually present in this batch: the experiment
    # table also lists conditions with no extracted records, and checking
    # those would report every D0 instance as an "orphan" rather than the
    # real, small coverage gaps AOC-CBS leaves behind.
    extracted = tuple(sorted({load.condition_id_of(r) for r in records
                               if r.solver == "aoccbs"}))
    violations, orphans = validate.check_ct_lower_bound_sound(
        paired, condition_ids=extracted)
    validate.raise_if_unsound(violations)  # stops the pipeline on any hit
    # A run outside the reported batches -- a different time limit or solver
    # commit -- covers one agent count by design, so it is an orphan on every
    # other instance. Listing those buries the coverage gaps that are worth
    # looking at.
    orphans = [o for o in orphans if load.is_batch_run(o.condition_id)]
    print(f"soundness check: 0 violations, {len(orphans)} orphan "
          "instance/condition pairs (expected coverage gaps -- see below)")
    for orphan in orphans:
        print(f"  {orphan.instance_key} present in {orphan.present_in}, "
              f"missing for {orphan.condition_id}")
    exact = validate.exact_lower_bound_counts(paired, condition_ids=extracted)
    print("instances where ct_lb == D0's exact cost (gap proven exactly "
          f"zero): {exact}")

    maps = plots.map_names(records)
    print("maps:", ", ".join(maps))

    # One figure covering every map, with a row per map.
    counts = plots.plot_normalised_cost(
        records, figures_dir / "F_normalised_cost")

    for map_name in maps:
        agent_counts = sorted(counts[map_name])
        print(f"\n{map_name}")
        print("  normalised cost: common instances per agent count (every "
              "curve of this row is averaged over these)")
        excluded = plots.EXCLUDED_AGENT_COUNTS.get(map_name, ())
        for n in agent_counts:
            note = " (excluded, see EXCLUDED_AGENT_COUNTS)" if n in excluded \
                else ""
            print(f"    n={n}: {counts[map_name][n]}{note}")

    print("\nfigure written to", figures_dir / "F_normalised_cost")


if __name__ == "__main__":
    main()
