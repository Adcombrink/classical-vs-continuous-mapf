"""Runs a sweep over k and agent radius on one map, at AOC-CBS a39a4b9.

One configuration per (k, radius) pair over the 25 `even` scenarios of the
map, at the agent counts given. Each configuration gets its own problem set
and its own batch, so extraction produces one jsonl per configuration.

    python scripts/run_aoccbs_sweep.py --map empty-32-32 \
        --agents 70,80,90,100 --suffix sweep70to100

The solver settings are the single `config/solver/solver.yaml`, which
`scripts/setup.sh` copies from `configs/aoccbs/solver.yaml`. The per-instance
time limit is whatever that file says; `--timeout` asserts it rather than
selecting a file (see `solver_settings`).

The TPR improvement sweeps are on, three sweeps over all agents. They are
environment variables rather than YAML keys, since AOC-CBS's config loader
does ``SolverConfig(**data)`` and would raise on an unknown key, and they are
read at call time inside worker processes -- so they are set before any
AOC-CBS import.

PYTHONHASHSEED is pinned for the same reason: importing ``run_batch`` does
not fix the seed, and the seed changes which collision the search expands
first.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("AOCCBS_TPR_SWEEPS", "3")
os.environ.setdefault("AOCCBS_TPR_SWEEP_SET", "all")

if "PYTHONHASHSEED" not in os.environ:
    # Python fixes hash randomization at interpreter startup, so this only
    # takes effect after a re-exec. Doing it before any AOC-CBS import keeps
    # the restart free of side effects.
    os.environ["PYTHONHASHSEED"] = "0"
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

AOC_ROOT = Path(__file__).resolve().parent.parent / "external" / "AOC-CBS"
sys.path.insert(0, str(AOC_ROOT / "src"))

from aoccbs import paths  # noqa: E402
from aoccbs.benchmarking import batch_run, movingai, problem_sets  # noqa: E402
from aoccbs.models import intersection_intervals  # noqa: E402
from aoccbs.models import state_graph_distances  # noqa: E402

SOLVER_SETTINGS_STEM = "solver"

# Worker processes for the preprocessing this script still builds lazily. It is
# only a default: a caller that knows how many cores it has should say so with
# --workers, since nothing here can find that out. os.cpu_count() would be the
# wrong answer on a cluster, where it reports the whole node rather than the
# cores the job was given.
DEFAULT_WORKERS = 12


def solver_settings(expected_timeout_s: int | None = None,
                    stem: str = SOLVER_SETTINGS_STEM) -> Path:
    """Returns the chosen solver settings file, checking its time limit.

    `scripts/setup.sh` copies every `configs/aoccbs/solver*.yaml` into the
    checkout, and `stem` picks one of them. The reported batches all use
    `solver`; a run needing anything else -- a different time limit, a shorter
    portfolio -- gets its own file, since a record carries neither and two such
    runs would otherwise be indistinguishable.

    The time limit is whatever the chosen file says. `--timeout` does not
    select a file: it states what the caller believes the limit to be, and a
    mismatch is an error rather than a silent run at the wrong budget.

    args:
        expected_timeout_s: the limit the caller expects, in whole seconds,
            or None to accept whatever the file says.
        stem: file stem under the checkout's config/solver directory.

    returns:
        Path to the settings file.

    raises:
        FileNotFoundError: if setup has not been run.
        ValueError: if the file's limit is not `expected_timeout_s`.
    """
    import yaml
    path = paths.SOLVER_CONFIG_DIR / f"{stem}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Run scripts/setup.sh, which copies "
            f"every configs/aoccbs/solver*.yaml into the checkout.")
    actual = float(yaml.safe_load(path.read_text())["timelimit"])
    if expected_timeout_s is not None and actual != float(expected_timeout_s):
        raise ValueError(
            f"{path.name} has timelimit {actual:g}s, but --timeout says "
            f"{expected_timeout_s}s. Edit configs/aoccbs/{path.stem}.yaml and "
            f"run scripts/setup.sh, or pass --timeout {actual:g}.")
    return path

# (agent model id, the radius it must have) -- the model library is keyed by
# name, and a name is not evidence of a radius: AgentModel_Circular_sqrt2over32
# in fact holds sqrt2/16. Each model is checked against the radius the
# experiment table's row calls for before anything is run with it.
RADII = (
    ("Circular_sqrt2over4", math.sqrt(2) / 4),
    ("Circular_sqrt2over8", math.sqrt(2) / 8),
    ("Circular_sqrt2over16", math.sqrt(2) / 16),
)
# Defaults for --k and --radii: the full experiment table. A run that covers
# only part of it, such as a repeat of one row at a longer time limit, names
# the part it covers on the command line.
CONNECTEDNESS = (2, 3)


def check_radius(agent_model_id: str, expected: float) -> None:
    """Fails if the named agent model does not have the expected radius.

    args:
        agent_model_id: id of a model in AOC-CBS's model library.
        expected: the radius the experiment table calls for.

    raises:
        ValueError: if the model's radius differs from `expected`.
    """
    import json
    path = paths.MODEL_LIBRARY_DIR / f"AgentModel_{agent_model_id}.json"
    radius = json.loads(path.read_text())["radius"]
    if abs(radius - expected) > 1e-9:
        raise ValueError(
            f"{path.name} has radius {radius!r}, expected {expected!r}")


def ensure_state_graph(map_name: str, k: int) -> str:
    """Builds the map's state graph if the model library does not hold it.

    The distance table and the intersection intervals are both computed from
    this graph, so a map that has never been run has to have it built first.
    It depends only on the map and k, so it is built once and reused.

    args:
        map_name: MovingAI map name, as in "warehouse-10-20-10-2-2".
        k: connectedness, 2 for 4-connected and 3 for 8-connected.

    returns:
        The state graph id.
    """
    state_graph_id = movingai.grid_state_graph_id(map_name, k)
    path = paths.MODEL_LIBRARY_DIR / f"StateGraph_{state_graph_id}.json"
    if path.is_file():
        print(f"state graph cached: {path.name}")
        return state_graph_id
    map_path = paths.MOVINGAI_DIR / map_name / f"{map_name}.map"
    print(f"building state graph {state_graph_id} from {map_path.name}")
    return movingai.map_to_grid_state_graph(map_path, k=k)


def parse_args() -> argparse.Namespace:
    """Returns the map, agent counts and batch name suffix to run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, dest="map_name",
                        help="MovingAI map name, e.g. empty-32-32")
    parser.add_argument("--agents", required=True,
                        help="comma-separated agent counts, e.g. 70,80,90,100")
    parser.add_argument("--suffix", required=True,
                        help="batch name suffix, which is what extraction "
                             "matches on; give a run its own so it does not "
                             "collide with an earlier one on the same map")
    parser.add_argument("--k",
                        default=",".join(str(k) for k in CONNECTEDNESS),
                        help="comma-separated move-set parameters to run "
                             "(default: 2,3)")
    parser.add_argument("--radii", default=",".join(m for m, _ in RADII),
                        help="comma-separated agent model ids to run "
                             "(default: all three of the experiment table)")
    parser.add_argument("--solver-settings", default=SOLVER_SETTINGS_STEM,
                        dest="solver_stem",
                        help="stem of the solver settings file to run, from "
                             "configs/aoccbs/ (default: "
                             f"{SOLVER_SETTINGS_STEM})")
    parser.add_argument("--scenarios-dir", default=None,
                        help="folder of .scen files to run, instead of the "
                             "map's own scenarios under AOC-CBS's movingai "
                             "tree. Used for the grid-refinement comparison, "
                             "where empty-32-32 runs empty-16-16's scenarios "
                             "with every coordinate doubled (see "
                             "scripts/port_scenarios.py). Those live under "
                             "benchmarks/, which syncs to the cluster, rather "
                             "than in the AOC-CBS checkout, which does not")
    parser.add_argument("--max-scenarios", type=int, default=None,
                        help="run only the first N of the map's scenarios "
                             "instead of all 25. For probing cost or memory, "
                             "where the answer does not need the full set; a "
                             "reported batch always runs all of them")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="worker processes for any preprocessing this run "
                             "has to build, which is none when "
                             "scripts/preprocess.py has already run "
                             f"(default: {DEFAULT_WORKERS})")
    parser.add_argument("--timeout", type=int, default=None,
                        help="the per-instance time limit you expect the "
                             "solver settings to carry, in seconds. Checked "
                             "against configs/aoccbs/solver.yaml rather than "
                             "used to pick a file; omit to accept whatever "
                             "that file says")
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
    scenarios_dir = (Path(args.scenarios_dir) if args.scenarios_dir
                     else paths.MOVINGAI_DIR / args.map_name / "scenarios")
    scenarios = movingai.load_scenarios(scenarios_dir)
    print(f"{len(scenarios)} scenarios from {scenarios_dir}")
    if args.max_scenarios is not None:
        # load_scenarios returns a dict keyed by file stem, in file-name
        # order, not a list -- so take the first N items rather than slicing.
        scenarios = dict(list(scenarios.items())[:args.max_scenarios])
        print(f"limited to the first {len(scenarios)} scenario(s): "
              f"{', '.join(scenarios)}")
    print(f"agent counts: {args.agent_counts}")

    for agent_model_id, expected_radius in args.radii:
        check_radius(agent_model_id, expected_radius)

    settings_path = solver_settings(args.timeout, args.solver_stem)
    print(f"solver settings: {settings_path.name}")

    batch_dirs = []
    for k in args.connectedness:
        state_graph_id = ensure_state_graph(args.map_name, k)
        state_graph_distances.ensure_state_graph_distances(
            state_graph_id, workers=args.workers)
        for agent_model_id, _ in args.radii:
            print(f"\n=== {state_graph_id} / {agent_model_id} ===")
            intersection_intervals.ensure_intersection_intervals(
                state_graph_id, agent_model_id,
                state_graph_id, agent_model_id,
                workers=args.workers)
            problems = problem_sets.create_problem_set(
                state_graph_id,
                agent_model_id,
                scenarios=scenarios,
                agent_range=args.agent_counts,
                name=f"{state_graph_id}_{agent_model_id}_Problems_"
                     f"{args.suffix}")
            batch_dir = batch_run.run_batch(
                solver_settings_path=settings_path,
                problems_path=problems,
                batch_name=f"{state_graph_id}_{agent_model_id}_{args.suffix}")
            print(f"batch directory: {batch_dir}")
            batch_dirs.append(batch_dir)

    print("\nall batches:")
    for batch_dir in batch_dirs:
        print(f"  {batch_dir}")


if __name__ == "__main__":
    main()
