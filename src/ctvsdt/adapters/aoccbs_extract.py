"""Extracts AOC-CBS results into the form this paper needs.

AOC-CBS is run in its own repository, where a batch produces a large
directory of per-run folders under `runs/`. This reads one such directory and
writes its records to a single JSON Lines file under `results/aoccbs/`, after
which the run directory can be deleted. AOC-CBS's own `aggregate_runs` step
is not involved; the raw run output is read directly.

A batch laid down by `benchmarking.batch_run.run_batch` looks like

    <run_dir>/index.json
    <run_dir>/<config>/<map path...>/<scenario>/problem_agents<NNN>/
        problem.yaml       the instance: agents, their model and state graph
        solver.yaml        the solver settings the run used
        summary.json       the outcome
        best_solution.json, iterations.jsonl, progress.html   (not read)

`<map path...>` is one component in practice but may be empty or deeper; the
depth is not assumed. Taken from each run: the agent count, agent model and
state graph from `problem.yaml`, the time limit from `solver.yaml`, and
`obj_lb`, `obj_ub`, `found_optimal` and `runtime` from `summary.json`.
`index.json` gives the run list and the entries AOC-CBS skipped.

Nothing is declared by hand. A record says what the run physically was --
solver, k, radius -- and that triple is the condition, so no label is stored:
k comes off the state graph id and the radius from AOC-CBS's model library,
and the pair is looked up in conditions.py while extracting, so a run the
experiment table does not call for is an error rather than a record nothing
can explain. `condition_of` puts a name back when a figure wants one.

Not retained: per-iteration traces, portfolio and branch diagnostics, and the
plans. Those are the bulk of a run directory and nothing here reads them.
Deleting a run directory is the one irreversible step in the pipeline, so
check an extraction first. The `__main__` block at the bottom is a worked
example.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import re
import subprocess
from typing import Any, Optional

import yaml

from ctvsdt import conditions
from ctvsdt.record import Record, write_jsonl

# AOC-CBS instances take the first `agent_count` agents of the scenario file,
# so the index of the first agent used is always zero. The DT side must follow
# the same convention; `check_instances_match` is what verifies it did.
FIRST_AGENT_INDEX = 0

# Where AOC-CBS keeps its agent models, relative to its repository root. The
# radius lives here and nowhere in the run output, so it is read from here
# while the run is being extracted.
MODEL_LIBRARY = pathlib.Path("scratch") / "models"

# `problem_agents010` -> 10.
_AGENT_COUNT_RE = re.compile(r"agents0*(\d+)")

# A state graph id carries the map it was built on and the move-set parameter,
# as in `empty-32-32_k2`.
_STATE_GRAPH_RE = re.compile(r"^(?P<map>.+)_k(?P<k>\d+)$")


class _TupleTolerantLoader(yaml.SafeLoader):
    """yaml.SafeLoader plus a constructor for the `!!python/tuple` tag.

    AOC-CBS's solver.yaml tags the search-portfolio heuristics with
    `!!python/tuple`, an artifact of dumping dataclasses that hold tuples.
    Plain safe_load rejects that tag and unsafe_load would accept arbitrary
    object construction; this accepts that one tag, as a list.
    """


_TupleTolerantLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: loader.construct_sequence(node))


@dataclasses.dataclass(frozen=True)
class RunLeaf:
    """One executed run: the directory holding a single instance's output.

    attributes:
        config: Solver-config directory name.
        scenario: Scenario name, which is the scenario file without extension.
        agent_folder: Agent-count folder name, such as `problem_agents010`.
        path: The directory itself.
        label: The run's path relative to the run directory, which is how
            index.json names it.
    """

    config: str
    scenario: str
    agent_folder: str
    path: pathlib.Path
    label: str


@dataclasses.dataclass(frozen=True)
class ExtractionSummary:
    """What an extraction produced.

    attributes:
        records: The extracted records.
        by_condition: Number of records per condition id.
        by_status: Number of records per status.
        skipped_runs: Labels of entries AOC-CBS planned but never ran, which
            it does once an agent count in a scenario finds no solution.
        errored_runs: Labels of runs that crashed, mapped to the reason.
    """

    records: list[Record]
    by_condition: dict[str, int]
    by_status: dict[str, int]
    skipped_runs: list[str]
    errored_runs: dict[str, str]

    def report(self) -> str:
        """Renders the summary as the text to read before deleting the run."""
        lines = [f"{len(self.records)} record(s)",
                 f"  by condition: {self.by_condition}",
                 f"  by status:    {self.by_status}"]
        if self.skipped_runs:
            lines.append(f"  {len(self.skipped_runs)} entr(ies) skipped by "
                         f"AOC-CBS, first: {self.skipped_runs[0]}")
        if self.errored_runs:
            lines.append(f"  {len(self.errored_runs)} run(s) errored, first: "
                         f"{sorted(self.errored_runs)[0]}")
        return "\n".join(lines)


def _load_yaml(path: pathlib.Path) -> Any:
    return yaml.load(path.read_text(encoding="utf-8"),
                     Loader=_TupleTolerantLoader)


def aoccbs_root(run_dir: pathlib.Path) -> pathlib.Path:
    """Returns the AOC-CBS repository a run directory sits in.

    args:
        run_dir: A run directory, that is `<repo>/runs/<run name>`.

    returns:
        The repository root.

    raises:
        ValueError: If the run directory is not under a `runs` directory.
    """
    for parent in run_dir.resolve().parents:
        if parent.name == "runs":
            return parent.parent
    raise ValueError(
        f"{run_dir} is not inside an AOC-CBS `runs` directory, so the model "
        f"library it belongs to cannot be found; pass model_library instead")


def solver_commit(repo_root: pathlib.Path) -> str:
    """Returns the commit of AOC-CBS that produced a run.

    A record has to say which solver produced it, and a run directory does not
    say. The working commit is the best available answer, so an extraction is
    worth doing before moving the submodule on.

    args:
        repo_root: The AOC-CBS repository root.

    returns:
        The commit hash, or "unknown" if it cannot be read.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def load_agent_radii(library_dir: pathlib.Path) -> dict[str, float]:
    """Reads the radius of every circular agent model in AOC-CBS's library.

    The library sits outside the run directories and survives their deletion,
    but the radius is what tells the conditions of a radius sweep apart and it
    appears nowhere in a run's own output -- only as a model name, which
    nothing forces to be honest.

    args:
        library_dir: AOC-CBS's model library directory.

    returns:
        Agent model id to radius, for the models that state one.

    raises:
        FileNotFoundError: If the library is not there.
    """
    if not library_dir.is_dir():
        raise FileNotFoundError(
            f"{library_dir} is not there; without it an agent model's radius "
            f"is unknown and the condition a run realises cannot be resolved")
    radii = {}
    for path in sorted(library_dir.glob("AgentModel_*.json")):
        model = json.loads(path.read_text(encoding="utf-8"))
        if "radius" in model:
            radii[model.get("id", path.stem[len("AgentModel_"):])] = float(
                model["radius"])
    return radii


def condition_of(record: Record) -> conditions.Condition:
    """Returns the condition a record realises.

    Records store (solver, k, radius), which *is* the row of the experiment
    table; the row's name is not stored, because renumbering the table would
    silently falsify every extracted file long after the runs behind them were
    deleted. This is the lookup that puts the name back when one is wanted.

    args:
        record: Any record.

    returns:
        The matching condition.

    raises:
        ValueError: If the table has no such row.
    """
    return conditions.for_run(record.solver, record.k, record.radius)


def find_run_leaves(run_dir: pathlib.Path) -> tuple[list[RunLeaf], list[str]]:
    """Finds every executed run in a run directory.

    Uses index.json when it is there, since that is also the only record of
    the entries AOC-CBS planned and then skipped. Falls back to walking for
    problem.yaml, which every executed run has.

    args:
        run_dir: One run directory, that is `runs/<run name>`.

    returns:
        The executed runs, and the labels of the entries that were skipped.

    raises:
        ValueError: If an entry's path is too shallow to be a run.
    """
    index_path = run_dir / "index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        skipped = [e["label"] for e in index if e.get("status") == "skipped"]
        # The label is resolved against the run directory actually passed in.
        # index.json's own `run_dir` is an absolute path baked in at run time
        # and goes stale as soon as the run folder moves.
        leaf_paths = [run_dir / e["label"] for e in index
                      if e.get("status") != "skipped"]
    else:
        skipped = []
        leaf_paths = [p.parent for p in sorted(run_dir.rglob("problem.yaml"))]

    leaves = []
    for path in leaf_paths:
        parts = path.relative_to(run_dir).parts
        if len(parts) < 3:
            raise ValueError(
                f"{'/'.join(parts)} is not deep enough to be a run; expected "
                f"<config>/.../<scenario>/<agent folder>")
        leaves.append(RunLeaf(config=parts[0], scenario=parts[-2],
                              agent_folder=parts[-1], path=path,
                              label="/".join(parts)))
    return leaves, skipped


def _bound(value: Any) -> Optional[float]:
    """Returns a finite float, or None for a missing or infinite bound."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _instance_facts(problem_path: pathlib.Path) -> tuple[int, list[str],
                                                         list[str]]:
    """Reads the agent count, agent models and state graphs of an instance."""
    agents = (_load_yaml(problem_path) or {}).get("agents", {})
    return (len(agents),
            sorted({a["agent_model"] for a in agents.values()}),
            sorted({a["state_graph"] for a in agents.values()}))


def record_from_run(leaf: RunLeaf, agent_radii: dict[str, float],
                    timeout_s: float, commit: str, run_id: str) -> Record:
    """Converts one executed run into a Record.

    args:
        leaf: The run directory of one instance.
        agent_radii: Agent model id to radius, from the model library.
        timeout_s: Time budget the run was given.
        commit: Commit of AOC-CBS that produced it.
        run_id: Identifier of the batch, which is the run directory's name.

    returns:
        The record.

    raises:
        ValueError: If the run mixes agent models or state graphs, names a
            model the library does not define, or realises no condition.
    """
    num_agents, models, state_graphs = _instance_facts(
        leaf.path / "problem.yaml")

    if len(models) != 1:
        raise ValueError(f"{leaf.label}: expected one agent model, got "
                         f"{models}")
    if models[0] not in agent_radii:
        raise ValueError(
            f"{leaf.label}: the model library defines no radius for agent "
            f"model {models[0]!r}, so its condition cannot be resolved")

    if len(state_graphs) != 1:
        raise ValueError(f"{leaf.label}: expected one state graph, got "
                         f"{state_graphs}")
    match = _STATE_GRAPH_RE.match(state_graphs[0])
    if match is None:
        raise ValueError(f"{leaf.label}: state graph {state_graphs[0]!r} does "
                         f"not name a move-set parameter, as in "
                         f"'empty-32-32_k2'")

    # Not stored on the record -- (solver, k, radius) is the condition, and
    # the record carries all three. Resolved here so that a run the paper's
    # experiment table does not call for is an error rather than a record.
    condition = conditions.for_run(
        "aoccbs", int(match["k"]), agent_radii[models[0]])

    folder_count = _AGENT_COUNT_RE.search(leaf.agent_folder)
    if folder_count is not None and int(folder_count.group(1)) != num_agents:
        raise ValueError(
            f"{leaf.label}: folder says {folder_count.group(1)} agents, "
            f"problem.yaml has {num_agents}")

    summary_path = leaf.path / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        # The run crashed: batch_run writes error.txt instead of summary.json.
        summary = {}

    soc_lb = _bound(summary.get("obj_lb"))
    soc_ub = _bound(summary.get("obj_ub"))
    found_optimal = bool(summary.get("found_optimal", False))

    if found_optimal and soc_ub is not None:
        status = "optimal"
    elif soc_ub is not None:
        status = "timeout"
    else:
        status = "failed"

    return Record(
        map_name=match["map"],
        scen_file=leaf.scenario,
        scen_index=FIRST_AGENT_INDEX,
        num_agents=num_agents,
        solver="aoccbs",
        k=condition.k,
        radius=condition.radius,
        timeout_s=timeout_s,
        soc_lb=soc_lb,
        soc_ub=soc_ub,
        makespan_lb=None,
        makespan_ub=None,
        closed=status == "optimal",
        runtime_s=float(summary.get("runtime", 0.0)),
        status=status,
        solver_commit=commit,
        run_id=run_id,
    )


def extract_run(run_dir: pathlib.Path, out_file: pathlib.Path,
                model_library: Optional[pathlib.Path] = None
                ) -> ExtractionSummary:
    """Extracts one AOC-CBS run directory into one JSON Lines file.

    After this returns and the summary has been checked, the run directory can
    be deleted. Nothing downstream reads it again.

    args:
        run_dir: One run directory, that is `runs/<run name>`.
        out_file: The .jsonl file to write, replacing it if it is there. Its
            directory is created if it is not.
        model_library: AOC-CBS's model library, defaulting to
            `scratch/models` in the repository the run directory sits in.

    returns:
        The extraction summary.

    raises:
        FileNotFoundError: If the run directory is not there, which is what
            asking for an already-deleted run looks like.
    """
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"{run_dir} is not there; it has not been produced, or has "
            f"already been deleted")

    repo_root = aoccbs_root(run_dir) if model_library is None else None
    agent_radii = load_agent_radii(
        model_library if model_library is not None
        else repo_root / MODEL_LIBRARY)
    commit = solver_commit(repo_root) if repo_root is not None else "unknown"

    leaves, skipped = find_run_leaves(run_dir)

    records: list[Record] = []
    errored_runs: dict[str, str] = {}
    timeouts: dict[str, float] = {}
    for leaf in leaves:
        if leaf.config not in timeouts:
            # solver.yaml is copied into every run under a config and is
            # identical across them, so the first one read stands for all.
            timeouts[leaf.config] = float(
                _load_yaml(leaf.path / "solver.yaml")["timelimit"])

        if not (leaf.path / "summary.json").is_file():
            error_path = leaf.path / "error.txt"
            errored_runs[leaf.label] = (
                error_path.read_text(encoding="utf-8").strip()
                if error_path.is_file() else "no summary.json, no error.txt")

        records.append(record_from_run(leaf, agent_radii,
                                       timeouts[leaf.config], commit,
                                       run_dir.name))

    by_condition: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for record in records:
        label = condition_of(record).condition_id
        by_condition[label] = by_condition.get(label, 0) + 1
        by_status[record.status] = by_status.get(record.status, 0) + 1

    out_file.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, str(out_file), append=False)

    return ExtractionSummary(records, by_condition, by_status, sorted(skipped),
                             errored_runs)


def check_coverage(records: list[Record], expected_conditions: list[str],
                   expected_agent_counts: list[int]) -> list[str]:
    """Reports conditions and agent counts the records do not cover.

    A missing condition is not an error here; it means the corresponding runs
    have not been done yet. The point is that it is stated rather than left as
    a hole in a figure.

    args:
        records: Extracted AOC-CBS records.
        expected_conditions: Condition ids this paper needs.
        expected_agent_counts: Agent counts this paper needs.

    returns:
        One message per gap found, empty if the records are complete.
    """
    present = {(condition_of(r).condition_id, r.num_agents)
               for r in records}
    problems = []
    for condition_id in expected_conditions:
        counts = {n for c, n in present if c == condition_id}
        if not counts:
            problems.append(f"condition {condition_id} has no runs")
            continue
        missing = sorted(set(expected_agent_counts) - counts)
        if missing:
            problems.append(
                f"condition {condition_id} is missing agent counts {missing}")
    return problems


def check_instances_match(ct_records: list[Record],
                          dt_records: list[Record]) -> list[str]:
    """Reports instances solved by one side and not the other.

    This is the check that protects the comparison. Costs from two different
    instance sets still produce plausible numbers and readable figures, so a
    mismatch has to be caught here rather than noticed later.

    args:
        ct_records: Extracted AOC-CBS records.
        dt_records: CBS records produced in this repository.

    returns:
        One message per instance missing from either side, empty if the two
        sides cover exactly the same instances.
    """
    ct_keys = {r.instance_key for r in ct_records}
    dt_keys = {r.instance_key for r in dt_records}
    problems = []
    for key in sorted(ct_keys - dt_keys):
        problems.append(f"no CBS run for instance {key}")
    for key in sorted(dt_keys - ct_keys):
        problems.append(f"no AOC-CBS run for instance {key}")
    return problems


if __name__ == "__main__":

    ### EXAMPLE USAGE ###
    # Extract one AOC-CBS run directory into one file under results/aoccbs.
    # Run from the repository root:
    #
    #     python -m ctvsdt.adapters.aoccbs_extract
    #
    # Read the report before deleting anything: the record counts are what the
    # run directory is being traded for. The run can then go, with `rm -rf`
    # over RUN_DIR.

    RUN_DIR = pathlib.Path(
        "external/AOC-CBS/runs/"
        "2026-08-13_141314_Empty32_k2_CircularSqrt2over8_30agents")
    OUT_FILE = pathlib.Path(
        "results/aoccbs/empty-32-32_k2_rsqrt2over8_new_AOCCBS.jsonl")

    extraction = extract_run(RUN_DIR, OUT_FILE)
    print(extraction.report())
    print(f"wrote {OUT_FILE}")

    gaps = check_coverage(extraction.records,
                          expected_conditions=["C2-a"],
                          expected_agent_counts=list(range(10, 101, 10)))
    print("\ncoverage:")
    for gap in gaps or ["complete"]:
        print(f"  {gap}")
