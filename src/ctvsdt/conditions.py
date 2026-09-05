"""The experiment table.

D0 is the discrete-time baseline. The C* conditions vary the move-set
parameter k and the agent radius under continuous time. The SIC conditions
are sums of individual shortest-path costs, the r -> 0 limit of a radius
sweep, computed without search.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

SQRT2 = math.sqrt(2.0)

# The largest radius for which the combinatorial conflict test of classical
# MAPF is sound on a unit 4-connected grid. Used as the DT-equivalent radius.
MAX_DT_SOUND_RADIUS = SQRT2 / 4.0


@dataclasses.dataclass(frozen=True)
class Condition:
    """One row of the experiment table.

    attributes:
        condition_id: Short identifier used in records, tables and figures.
        solver: "cbs", "aoccbs" or "sic".
        k: Move-set parameter of the graph. 2 is the 4-connected grid.
        radius: Agent radius, or None where cost does not depend on it.
        radius_label: Radius as written in the paper, for axis labels.
        purpose: What this condition isolates.
    """

    condition_id: str
    solver: str
    k: int
    radius: Optional[float]
    radius_label: str
    purpose: str


CONDITIONS: tuple[Condition, ...] = (
    Condition("D0", "cbs", 2, None, "n/a",
              "Baseline: the classical discrete-time formulation."),
    Condition("C2-a", "aoccbs", 2, SQRT2 / 4, "sqrt2/4",
              "Pure timing effect: same graph and radius as D0."),
    Condition("C2-b", "aoccbs", 2, SQRT2 / 8, "sqrt2/8",
              "Radius sweep at k=2."),
    Condition("C2-c", "aoccbs", 2, SQRT2 / 16, "sqrt2/16",
              "Radius sweep at k=2."),
    Condition("C2-d", "aoccbs", 2, SQRT2 / 64, "sqrt2/64",
              "Near-degenerate radius at k=2."),
    Condition("C3-a", "aoccbs", 3, SQRT2 / 4, "sqrt2/4",
              "Connectivity effect at the DT-equivalent radius."),
    Condition("C3-b", "aoccbs", 3, SQRT2 / 8, "sqrt2/8",
              "Connectivity and radius combined."),
    Condition("C3-c", "aoccbs", 3, SQRT2 / 16, "sqrt2/16",
              "Connectivity and radius combined."),
    Condition("C3-d", "aoccbs", 3, SQRT2 / 64, "sqrt2/64",
              "Connectivity and radius combined, near-degenerate."),
    Condition("C4-a", "aoccbs", 4, SQRT2 / 4, "sqrt2/4",
              "16-connected grid at the DT-equivalent radius."),
    Condition("C4-b", "aoccbs", 4, SQRT2 / 8, "sqrt2/8",
              "16-connected grid, radius sweep."),
    Condition("C4-c", "aoccbs", 4, SQRT2 / 16, "sqrt2/16",
              "16-connected grid, radius sweep."),
    Condition("C4-d", "aoccbs", 4, SQRT2 / 64, "sqrt2/64",
              "16-connected grid, near-degenerate radius."),
    Condition("SIC2", "sic", 2, 0.0, "-> 0",
              "Asymptote of the k=2 sweep and denominator of RF."),
    Condition("SIC3", "sic", 3, 0.0, "-> 0",
              "Asymptote of the k=3 sweep and denominator of RF."),
    Condition("SIC4", "sic", 4, 0.0, "-> 0",
              "Asymptote of the k=4 sweep and denominator of RF."),
)

BY_ID: dict[str, Condition] = {c.condition_id: c for c in CONDITIONS}

# Radii are written as expressions of sqrt(2) wherever they appear, so two
# statements of the same radius agree to floating-point noise or not at all.
RADIUS_TOL = 1e-9


def by_solver(solver: str) -> tuple[Condition, ...]:
    """Returns the conditions run with the given solver."""
    return tuple(c for c in CONDITIONS if c.solver == solver)


def for_run(solver: str, k: int, radius: Optional[float]) -> Condition:
    """Returns the condition a run with these parameters realises.

    A condition is a name for a row of the experiment table, and the row is
    exactly (solver, k, radius). Records therefore store those three and not
    the name: they are what the run physically was, whereas the name is an
    editorial choice that renumbering the table would invalidate, long after
    the runs that produced the records have been deleted. Call this to put a
    name to a record when labelling a table or a figure.

    args:
        solver: "cbs", "aoccbs" or "sic".
        k: Move-set parameter of the graph.
        radius: Agent radius, or None where cost does not depend on it.

    returns:
        The matching condition.

    raises:
        ValueError: If the table has no such row, which means the run is not
            one the paper calls for.
    """
    for condition in by_solver(solver):
        if condition.k != k:
            continue
        if condition.radius is None or radius is None:
            if condition.radius is None and radius is None:
                return condition
            continue
        if abs(condition.radius - radius) <= RADIUS_TOL:
            return condition
    rows = [f"{c.condition_id} (k={c.k}, r={c.radius_label})"
            for c in by_solver(solver)]
    raise ValueError(
        f"the experiment table has no {solver} row with k={k} and radius "
        f"{radius!r}; it has {rows}")
