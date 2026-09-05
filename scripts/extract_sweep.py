#!/usr/bin/env python
"""Extracts the six batches of one k/radius sweep.

A sweep run by `scripts/run_aoccbs_sweep.py` leaves six run directories, one
per (k, radius) configuration, all sharing the batch name suffix that run was
given. This extracts all six into
`results/aoccbs/<map>_k<k>_rsqrt2over<d><out_suffix>.jsonl`, checks each
against the condition and agent counts it should hold, and prints the run
directories to delete by hand. It deletes nothing itself: that is the one
irreversible step in the pipeline.

    python scripts/extract_sweep.py --map empty-32-32 \
        --suffix sweep70to100 --agents 70,80,90,100 --out-suffix _sweep70to100

One condition's records may be split across files; `load.load_all` reads
every jsonl in the directory and pairs by instance.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from ctvsdt import conditions
from ctvsdt.adapters import aoccbs_extract

RUNS_DIR = pathlib.Path("external/AOC-CBS/runs")
OUT_DIR = pathlib.Path("results/aoccbs")

# Agent model id -> the output file's radius label. The model library is keyed
# by name; `run_aoccbs_sweep.py` checks each name against the radius the
# experiment table calls for before running anything with it.
RADII = (("Circular_sqrt2over4", "rsqrt2over4"),
         ("Circular_sqrt2over8", "rsqrt2over8"),
         ("Circular_sqrt2over16", "rsqrt2over16"))
CONNECTEDNESS = (2, 3)


def find_run_dir(batch_name: str) -> pathlib.Path:
    """Returns the newest run directory for a batch name.

    args:
        batch_name: Batch name as passed to `run_batch`, without the
            timestamp `run_batch` prefixes it with.

    returns:
        The run directory.

    raises:
        FileNotFoundError: If no run directory matches.
    """
    matches = sorted(p for p in RUNS_DIR.glob(f"*_{batch_name}") if p.is_dir())
    if not matches:
        raise FileNotFoundError(f"no run directory under {RUNS_DIR} ending in "
                                f"{batch_name!r}")
    return matches[-1]


def parse_args() -> argparse.Namespace:
    """Returns the map, batch suffix, agent counts and output suffix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, dest="map_name")
    parser.add_argument("--suffix", required=True,
                        help="batch name suffix the sweep was run with")
    parser.add_argument("--agents", required=True,
                        help="comma-separated agent counts the sweep covered, "
                             "which coverage is checked against")
    parser.add_argument("--out-suffix", default="",
                        help="appended to each output file's stem, to keep "
                             "one condition's batches in separate files")
    parser.add_argument("--k",
                        default=",".join(str(k) for k in CONNECTEDNESS),
                        help="comma-separated move-set parameters the sweep "
                             "covered (default: 2,3). Match what the run was "
                             "launched with, or find_run_dir looks for a "
                             "batch that was never run")
    parser.add_argument("--radii", default=",".join(m for m, _ in RADII),
                        help="comma-separated agent model ids the sweep "
                             "covered (default: all three)")
    args = parser.parse_args()
    args.agent_counts = [int(n) for n in args.agents.split(",")]
    args.connectedness = [int(k) for k in args.k.split(",")]
    by_id = dict(RADII)
    unknown = [m for m in args.radii.split(",") if m not in by_id]
    if unknown:
        parser.error(f"unknown agent model(s) {unknown}, expected any of "
                     f"{sorted(by_id)}")
    args.radii = [(m, by_id[m]) for m in args.radii.split(",")]
    return args


def main() -> None:
    args = parse_args()
    run_dirs = []
    ok = True
    for k in args.connectedness:
        for agent_model_id, radius_label in args.radii:
            batch_name = (f"{args.map_name}_k{k}_{agent_model_id}_"
                          f"{args.suffix}")
            run_dir = find_run_dir(batch_name)
            out_file = (OUT_DIR / f"{args.map_name}_k{k}_{radius_label}"
                                  f"{args.out_suffix}.jsonl")
            extraction = aoccbs_extract.extract_run(run_dir, out_file)
            print(f"\n=== {run_dir.name} -> {out_file} ===")
            print(extraction.report())

            expected = conditions.for_run(
                "aoccbs", k, _radius_of(agent_model_id)).condition_id
            gaps = aoccbs_extract.check_coverage(
                extraction.records, expected_conditions=[expected],
                expected_agent_counts=args.agent_counts)
            for gap in gaps or ["coverage complete"]:
                print(f"  {gap}")
            if set(extraction.by_condition) != {expected}:
                print(f"  WARNING: expected only {expected}, got "
                      f"{sorted(extraction.by_condition)}")
                ok = False
            if extraction.errored_runs:
                ok = False
            run_dirs.append(run_dir)

    print("\nrun directories, deletable once the reports above look right:")
    for run_dir in run_dirs:
        print(f"  rm -rf {run_dir}")
    if not ok:
        print("\nsomething above needs looking at before deleting anything")
        sys.exit(1)


def _radius_of(agent_model_id: str) -> float:
    """Returns the radius of an agent model, from AOC-CBS's model library."""
    radii = aoccbs_extract.load_agent_radii(
        pathlib.Path("external/AOC-CBS") / aoccbs_extract.MODEL_LIBRARY)
    return radii[agent_model_id]


if __name__ == "__main__":
    main()
