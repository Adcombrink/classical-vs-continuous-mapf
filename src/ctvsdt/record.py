"""Canonical result record shared by both solvers.

Every solver run is reduced to one Record, and every metric and figure is a
function of a collection of them. Costs are stored as bounds, with lb == ub
where the run proved optimality, so both sides share one schema.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Iterator, Optional


@dataclasses.dataclass(frozen=True)
class Record:
    """The outcome of one solver run on one instance.

    attributes:
        map_name: MovingAI map name, without extension.
        scen_file: Scenario file name the instance was drawn from.
        scen_index: Index of the first agent taken from the scenario file.
        num_agents: Number of agents in the instance.
        solver: "cbs", "aoccbs" or "sic".
        k: Move-set parameter of the graph. 2 is the 4-connected grid.
        radius: Agent radius, or None for the DT side where cost does not
            depend on it.
        timeout_s: Time budget given to the run.
        soc_lb: Lower bound on the optimal sum of costs.
        soc_ub: Upper bound on the optimal sum of costs, that is the cost of
            the best feasible solution found. None if no solution was found.
        makespan_lb: Lower bound on the optimal makespan, if computed.
        makespan_ub: Makespan of the best feasible solution found, if computed.
        closed: True if the run proved optimality.
        runtime_s: Wall-clock time of the run.
        status: "optimal", "timeout" or "failed".
        solver_commit: Commit hash of the solver used.
        run_id: Identifier of the batch this run belongs to.
        restarts: How many independent runs this record summarises. 1
            throughout the reported campaign.
    """

    map_name: str
    scen_file: str
    scen_index: int
    num_agents: int
    solver: str
    k: int
    radius: Optional[float]
    timeout_s: float
    soc_lb: Optional[float]
    soc_ub: Optional[float]
    makespan_lb: Optional[float]
    makespan_ub: Optional[float]
    closed: bool
    runtime_s: float
    status: str
    solver_commit: str
    run_id: str
    restarts: int = 1

    def __post_init__(self):
        if self.soc_lb is not None and self.soc_ub is not None:
            if self.soc_lb > self.soc_ub + 1e-9:
                raise ValueError(
                    f"soc_lb {self.soc_lb} exceeds soc_ub {self.soc_ub}")
        if self.closed and self.soc_ub is None:
            raise ValueError("a closed run must have a solution cost")

    @property
    def instance_key(self) -> tuple:
        """Identifies the instance, independent of which condition was run."""
        return (self.map_name, self.scen_file, self.scen_index,
                self.num_agents)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Record":
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - fields
        if unknown:
            raise ValueError(f"unknown fields in record: {sorted(unknown)}")
        return cls(**data)


def write_jsonl(records: list[Record], path: str, append: bool = True) -> None:
    """Appends records to a JSON Lines file, one record per line."""
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict()) + "\n")


def read_jsonl(path: str) -> Iterator[Record]:
    """Yields the records stored in a JSON Lines file."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield Record.from_dict(json.loads(line))
