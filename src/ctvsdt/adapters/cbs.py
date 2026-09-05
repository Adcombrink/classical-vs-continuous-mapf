"""Adapter for CBSH2-RTC's raw CSV output, the D0 and SIC2 conditions.

CBSH2-RTC is always run at k=2 with no agent radius, so every row is the D0
baseline. Its CSV states `solution cost` for the best solution found and `min
f value`, the lower bound from the high-level open list; a run that did not
close optimality reports `solution cost` as -1, but `min f value` still holds
a usable bound.

Every row also carries `root g value`, the CBS root node's cost: the sum of
each agent's individually optimal path, ignoring every other agent. On a
4-connected unit-weight grid that is exactly SIC at k=2 (see conditions.py's
SIC2 row), so each row yields a second record for it at no extra cost -- no
search beyond what CBSH2-RTC already does to build its root node.
"""

from __future__ import annotations

import csv
import pathlib
from typing import Optional

from ctvsdt.record import Record


def read_csv(path: str, solver_commit: Optional[str] = None) -> list[Record]:
    """Reads a CBSH2-RTC output CSV into Records.

    args:
        path: Path to the CSV file written by CBSH2-RTC.
        solver_commit: Commit hash of the CBS binary used, or None if
            unrecorded.

    returns:
        Two records per row: solver="cbs", k=2, radius=None (D0), and
        solver="sic", k=2, radius=0.0 (SIC2), same instance key.
    """
    run_id = pathlib.Path(path).stem
    records = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            soc_ub: Optional[float] = float(row["solution cost"])
            if soc_ub <= 0:
                soc_ub = None
            soc_lb = float(row["min f value"])
            closed = soc_ub is not None and soc_ub == soc_lb
            map_name = row["map"]
            scen_file = row["scen_file"]
            num_agents = int(row["num_agents"])
            timeout_s = float(row["timeout_s"])

            records.append(Record(
                map_name=map_name,
                scen_file=scen_file,
                scen_index=0,
                num_agents=num_agents,
                solver="cbs",
                k=2,
                radius=None,
                timeout_s=timeout_s,
                soc_lb=soc_lb,
                soc_ub=soc_ub,
                makespan_lb=None,
                makespan_ub=None,
                closed=closed,
                runtime_s=float(row["runtime"]),
                status="optimal" if closed else "timeout",
                solver_commit=solver_commit or "",
                run_id=run_id,
            ))

            sic = float(row["root g value"])
            records.append(Record(
                map_name=map_name,
                scen_file=scen_file,
                scen_index=0,
                num_agents=num_agents,
                solver="sic",
                k=2,
                radius=0.0,
                timeout_s=timeout_s,
                soc_lb=sic,
                soc_ub=sic,
                makespan_lb=None,
                makespan_ub=None,
                closed=True,
                runtime_s=0.0,
                status="optimal",
                solver_commit=solver_commit or "",
                run_id=run_id,
            ))
    return records
