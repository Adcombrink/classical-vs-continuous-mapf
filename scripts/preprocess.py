#!/usr/bin/env python
"""Builds the preprocessing every solve needs, so no solve has to build it.

Three products per configuration, each derived from the one before it:

    state graph          from the map file, one per (map, k)
    distance table       all-pairs shortest durations, one per (map, k)
    intersection intervals   the collision geometry, one per (map, k, radius)

AOC-CBS builds all three lazily, which is fine sequentially and wrong in
parallel: solves starting at once each find the file missing, repeat the work
and race to write the same path. Run this once before submitting any solving.

    python scripts/preprocess.py --maps empty-16-16 --k 2,3,4 --radii 4,8,16
    python scripts/preprocess.py --check --maps empty-16-16   # verify only

`--check` builds nothing and exits non-zero if anything is missing.

Cost follows map size, not k or radius: seconds for a 16x16 grid, about 48
minutes for one radius on warehouse-10-20-10-2-2. The cache root honours
AOCCBS_CACHE_DIR, so point it at shared storage to build once for every
later job.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

AOC_ROOT = Path(__file__).resolve().parent.parent / "external" / "AOC-CBS"
sys.path.insert(0, str(AOC_ROOT / "src"))

from aoccbs import paths  # noqa: E402
from aoccbs.benchmarking import movingai  # noqa: E402
from aoccbs.models import intersection_intervals  # noqa: E402
from aoccbs.models import state_graph_distances  # noqa: E402

# Radius denominator -> (agent model id, the radius the model must have).
# The model library is keyed by name and a name is not evidence of a radius:
# the library's own Circular_sqrt2over32 in fact holds sqrt2/16. Each model is
# checked against the value the experiment table calls for before anything is
# built with it, exactly as scripts/run_aoccbs_sweep.py does.
RADII = {
    4: ("Circular_sqrt2over4", math.sqrt(2) / 4),
    8: ("Circular_sqrt2over8", math.sqrt(2) / 8),
    16: ("Circular_sqrt2over16", math.sqrt(2) / 16),
}
CONNECTEDNESS = (2, 3, 4)


def check_radius(agent_model_id: str, expected: float) -> None:
    """Fails if the named agent model does not have the expected radius.

    args:
        agent_model_id: id of a model in AOC-CBS's model library.
        expected: the radius the experiment table calls for.

    raises:
        FileNotFoundError: if the model is not in the library, which means
            scripts/setup.sh has not been run.
        ValueError: if the model's radius differs from `expected`.
    """
    import json
    try:
        path = paths.agent_model_file(agent_model_id)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"agent model {agent_model_id!r} is not in the model library "
            f"({paths.MODEL_LIBRARY_DIR}). Run scripts/setup.sh first.")
    radius = json.loads(path.read_text())["radius"]
    if abs(radius - expected) > 1e-9:
        raise ValueError(
            f"{path.name} has radius {radius!r}, expected {expected!r}")


def state_graph_present(state_graph_id: str) -> bool:
    """Whether the model library holds this state graph."""
    try:
        paths.state_graph_model_file(state_graph_id)
        return True
    except FileNotFoundError:
        return False


def distances_present(state_graph_id: str) -> bool:
    """Whether the cache holds this state graph's distance table."""
    try:
        paths.state_graph_distance_file(state_graph_id)
        return True
    except FileNotFoundError:
        return False


def intervals_present(state_graph_id: str, agent_model_id: str) -> bool:
    """Whether the cache holds the self-pair intersection intervals.

    Matched by glob rather than by rebuilding the filename, which is composed
    inside AOC-CBS's own writer from both halves of the agent-type pair. Every
    pair this paper uses has an agent type against itself, so the prefix is
    enough to identify it.

    args:
        state_graph_id: as returned by movingai.grid_state_graph_id.
        agent_model_id: id of a model in the library.

    returns:
        True if a matching file exists.
    """
    prefix = f"IntersectionIntervals_{state_graph_id}_{agent_model_id}_"
    return any(paths.INTERSECTION_INTERVAL_PP_DIR.glob(f"{prefix}*"))


def ensure_state_graph(map_name: str, k: int) -> str:
    """Builds the map's state graph if the model library does not hold it.

    The distance table and the intersection intervals are both computed from
    this graph, so a map that has never been run has to have it built first.
    It depends only on the map and k, so it is built once and reused.

    args:
        map_name: MovingAI map name, as in "warehouse-10-20-10-2-2".
        k: connectedness, 2 for 4-connected, 3 for 8-, 4 for 16-.

    returns:
        The state graph id.
    """
    state_graph_id = movingai.grid_state_graph_id(map_name, k)
    if state_graph_present(state_graph_id):
        print(f"  state graph cached: {state_graph_id}")
        return state_graph_id
    map_path = paths.MOVINGAI_DIR / map_name / f"{map_name}.map"
    if not map_path.is_file():
        raise FileNotFoundError(f"no map file at {map_path}")
    print(f"  building state graph {state_graph_id} from {map_path.name}")
    return movingai.map_to_grid_state_graph(map_path, k=k)


def parse_args() -> argparse.Namespace:
    """Returns the maps, connectednesses and radii to prepare."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maps", required=True,
                        help="comma-separated MovingAI map names")
    parser.add_argument("--k", default=",".join(str(k) for k in CONNECTEDNESS),
                        help="comma-separated connectednesses (default: 2,3,4)")
    parser.add_argument("--radii", default=",".join(str(d) for d in RADII),
                        help="comma-separated radius denominators, as in "
                             "4,8,16 for sqrt2/4, sqrt2/8, sqrt2/16 "
                             "(default: all three)")
    parser.add_argument("--workers", type=int, default=None,
                        help="worker processes for the distance table and "
                             "the narrowphase (default: AOC-CBS's own)")
    parser.add_argument("--check", action="store_true",
                        help="report what is missing and exit non-zero, "
                             "building nothing")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even what is already cached")
    args = parser.parse_args()
    args.map_names = [m for m in args.maps.split(",") if m]
    args.connectedness = [int(k) for k in args.k.split(",")]
    unknown = [d for d in args.radii.split(",") if int(d) not in RADII]
    if unknown:
        parser.error(f"unknown radius denominator(s) {unknown}, expected any "
                     f"of {sorted(RADII)}")
    args.radii = [RADII[int(d)] for d in args.radii.split(",")]
    return args


def report_missing(args) -> int:
    """Prints what is absent for the requested grid, building nothing.

    args:
        args: parsed command line.

    returns:
        The number of missing products.
    """
    missing = 0
    for map_name in args.map_names:
        for k in args.connectedness:
            state_graph_id = movingai.grid_state_graph_id(map_name, k)
            if not state_graph_present(state_graph_id):
                print(f"  MISSING state graph: {state_graph_id}")
                missing += 1
            if not distances_present(state_graph_id):
                print(f"  MISSING distance table: {state_graph_id}")
                missing += 1
            for agent_model_id, _ in args.radii:
                if not intervals_present(state_graph_id, agent_model_id):
                    print(f"  MISSING intersection intervals: "
                          f"{state_graph_id} / {agent_model_id}")
                    missing += 1
    return missing


def main() -> None:
    args = parse_args()
    print(f"maps: {args.map_names}")
    print(f"k: {args.connectedness}")
    print(f"radii: {[m for m, _ in args.radii]}")
    print(f"cache root: {paths.CACHE_ROOT}")
    print(f"model library: {paths.MODEL_LIBRARY_DIR}")

    for agent_model_id, expected_radius in args.radii:
        check_radius(agent_model_id, expected_radius)

    if args.check:
        missing = report_missing(args)
        if missing:
            print(f"\n{missing} product(s) missing; run without --check to "
                  "build them")
            sys.exit(1)
        print("\neverything the requested grid needs is present")
        return

    for map_name in args.map_names:
        for k in args.connectedness:
            print(f"\n=== {map_name} k={k} ===")
            state_graph_id = ensure_state_graph(map_name, k)
            state_graph_distances.ensure_state_graph_distances(
                state_graph_id, force=args.force, workers=args.workers)
            for agent_model_id, _ in args.radii:
                print(f"  --- {agent_model_id} ---")
                intersection_intervals.ensure_intersection_intervals(
                    state_graph_id, agent_model_id,
                    state_graph_id, agent_model_id,
                    force=args.force, workers=args.workers)

    remaining = report_missing(args)
    if remaining:
        print(f"\n{remaining} product(s) still missing after building; "
              "something above failed")
        sys.exit(1)
    print("\npreprocessing complete for the requested grid")


if __name__ == "__main__":
    main()
