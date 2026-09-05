"""Loads Records from disk and pairs them by instance.

Coverage differs across conditions: AOC-CBS stops a scenario once an agent
count finds no feasible solution and skips the higher counts. Any comparison
across conditions therefore goes through `common_instances` rather than
plotting mismatched instance sets side by side.
"""

from __future__ import annotations

import pathlib
from typing import Iterable, Optional

from ctvsdt import conditions
from ctvsdt.adapters import cbs
from ctvsdt.record import Record, read_jsonl

InstanceKey = tuple
Paired = dict[InstanceKey, dict[str, Record]]

# The time limit a map's reported batch ran under, where that is not the
# standard one. Runs at any other limit answer the same experiment-table row,
# so `condition_id_of` suffixes them to keep a calibration run beside the
# reported batch rather than on top of it. Empty because every reported map
# ran at 300 s; a map with no entry is reported at STANDARD_TIMEOUT_S.
STANDARD_TIMEOUT_S = 300.0
REPORTED_TIMEOUTS_S: dict[str, float] = {}


def reported_timeout_s(map_name: str) -> float:
    """Returns the time limit the reported batch of a map was run under.

    args:
        map_name: MovingAI map name, as carried on a record.

    returns:
        The limit in seconds, STANDARD_TIMEOUT_S for a map that is not named
        in REPORTED_TIMEOUTS_S.
    """
    return REPORTED_TIMEOUTS_S.get(map_name, STANDARD_TIMEOUT_S)


# The AOC-CBS commit each map's reported batch ran under. A run at another
# commit can report different bounds for the same table row, so it is
# suffixed for the same reason a non-standard time limit is. Per map because
# a map rerun later moves on its own; a map with no entry has no reported
# batch, so nothing of it is suffixed.
REPORTED_COMMITS = {
    "empty-32-32": "a39a4b96fb9634568bcbea00d72391e31f86519a",
    "empty-16-16": "a39a4b96fb9634568bcbea00d72391e31f86519a",
    "maze-32-32-2": "a39a4b96fb9634568bcbea00d72391e31f86519a",
    "warehouse-10-20-10-2-2": "a39a4b96fb9634568bcbea00d72391e31f86519a",
    "room-64-64-8": "a39a4b96fb9634568bcbea00d72391e31f86519a",
}


def reported_commit(map_name: str) -> Optional[str]:
    """Returns the AOC-CBS commit a map's reported batch was run under.

    args:
        map_name: MovingAI map name, as carried on a record.

    returns:
        The full commit hash, or None for a map with no reported batch.
    """
    return REPORTED_COMMITS.get(map_name)


def load_all(results_dir: str = "results") -> list[Record]:
    """Reads every AOC-CBS jsonl and CBS csv result file under results_dir.

    args:
        results_dir: Root of the results tree, holding `aoccbs/*.jsonl` and
            `cbs/*.csv`.

    returns:
        All records found, in no particular order.
    """
    root = pathlib.Path(results_dir)
    records: list[Record] = []
    for path in sorted(root.glob("aoccbs/*.jsonl")):
        records.extend(read_jsonl(str(path)))
    for path in sorted(root.glob("cbs/*.csv")):
        records.extend(cbs.read_csv(str(path)))
    return records


def condition_id_of(record: Record) -> str:
    """Returns the condition_id a record realises, from (solver, k, radius).

    Records do not store condition_id: the name is looked up from (solver, k,
    radius) in conditions.py, so renumbering the table cannot falsify old
    records.

    A run at a time limit or commit other than its map's reported one gets
    that appended -- "C2-b@120s", "C2-b~a0c8c59" -- so it sits beside the
    reported batch instead of replacing it when records are grouped by
    instance and condition.
    """
    condition_id = conditions.for_run(record.solver, record.k,
                                       record.radius).condition_id
    if (record.solver == "aoccbs"
            and record.timeout_s != reported_timeout_s(record.map_name)):
        condition_id = f"{condition_id}@{record.timeout_s:g}s"
    expected_commit = reported_commit(record.map_name)
    if (record.solver == "aoccbs" and expected_commit is not None
            and record.solver_commit != expected_commit):
        condition_id = f"{condition_id}~{record.solver_commit[:7]}"
    return condition_id


def split_condition_id(condition_id: str) -> tuple[str, Optional[float]]:
    """Splits a condition_id into its table-row name and time limit.

    The commit suffix is dropped too; see commit_suffix_of for that.

    args:
        condition_id: A value returned by condition_id_of.

    returns:
        (base condition_id, time limit in seconds), where the time limit is
        None for a run at the limit its map is reported at.
    """
    base = condition_id.partition("~")[0]
    if "@" not in base:
        return base, None
    base, _, suffix = base.partition("@")
    return base, float(suffix.rstrip("s"))


def commit_suffix_of(condition_id: str) -> Optional[str]:
    """Returns the short commit hash a condition_id carries, if any.

    args:
        condition_id: A value returned by condition_id_of.

    returns:
        The short commit hash, or None if the run was at the reported
        commit for its map.
    """
    if "~" not in condition_id:
        return None
    return condition_id.partition("~")[2]


def is_batch_run(condition_id: str) -> bool:
    """True if the condition_id names a run of a reported batch.

    False for a suffixed run, which covers a single agent count by design and
    so cannot be drawn as a curve over the grid.

    args:
        condition_id: A value returned by condition_id_of.
    """
    return condition_id == split_condition_id(condition_id)[0]


def pair_by_instance(records: Iterable[Record]) -> Paired:
    """Groups records by instance, then by the condition each one realises.

    args:
        records: Any collection of records.

    returns:
        instance_key -> {condition_id: Record}.
    """
    paired: Paired = {}
    for record in records:
        by_condition = paired.setdefault(record.instance_key, {})
        by_condition[condition_id_of(record)] = record
    return paired


def common_instances(paired: Paired,
                      condition_ids: Iterable[str]) -> set[InstanceKey]:
    """Returns the instance keys present for every one of condition_ids.

    args:
        paired: The output of pair_by_instance.
        condition_ids: The conditions a comparison needs.

    returns:
        The instance keys for which all of condition_ids have a record.
    """
    wanted = set(condition_ids)
    return {key for key, by_condition in paired.items()
            if wanted <= by_condition.keys()}
