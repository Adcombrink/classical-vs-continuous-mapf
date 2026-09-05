"""Validates the soundness condition the DT-vs-CT comparison rests on.

Under the sqrt(2)/4 embedding condition every DT solution is a feasible CT
schedule on the same move set, so CT_opt <= DT_opt wherever D0 closed. A CT
lower bound above that exact DT cost is never a result: it is a broken agent
model, a mismatched instance set, or a solver bug. SIC bounds are checked the
same way, being admissible by construction.

`scripts/make_figures.py` runs this before plotting and raises rather than
warns, since a violation makes every number downstream wrong.
"""

from __future__ import annotations

import dataclasses

from ctvsdt.conditions import CONDITIONS
from ctvsdt.load import InstanceKey, Paired

TOLERANCE = 1e-9

CT_CONDITION_IDS: tuple[str, ...] = tuple(
    c.condition_id for c in CONDITIONS if c.solver == "aoccbs")
SIC_CONDITION_IDS: tuple[str, ...] = tuple(
    c.condition_id for c in CONDITIONS if c.solver == "sic")

# The default for check_ct_lower_bound_sound: every condition producing a
# lower bound that cannot exceed D0's exact cost.
LOWER_BOUND_CONDITION_IDS: tuple[str, ...] = (
    CT_CONDITION_IDS + SIC_CONDITION_IDS)


@dataclasses.dataclass(frozen=True)
class Violation:
    """A lower bound that exceeds D0's exact cost on the same instance.

    attributes:
        instance_key: The instance the violation occurred on.
        condition_id: The CT or SIC condition whose lower bound is unsound.
        ct_lb: The offending lower bound (that condition's soc_lb).
        dt_cost: D0's exact cost (soc_ub) on this instance.
        excess: ct_lb - dt_cost, always greater than the tolerance.
    """
    instance_key: InstanceKey
    condition_id: str
    ct_lb: float
    dt_cost: float
    excess: float


@dataclasses.dataclass(frozen=True)
class Orphan:
    """An instance with a record on one side of a D0/condition pair and not the other.

    attributes:
        instance_key: The instance.
        condition_id: The CT or SIC condition being checked against D0.
        present_in: Which side has the record: "D0" or condition_id.
    """
    instance_key: InstanceKey
    condition_id: str
    present_in: str


def check_ct_lower_bound_sound(
        paired: Paired,
        condition_ids: tuple[str, ...] = LOWER_BOUND_CONDITION_IDS,
        tolerance: float = TOLERANCE,
) -> tuple[list[Violation], list[Orphan]]:
    """Checks every closed D0 instance against each condition's lower bound.

    args:
        paired: The output of load.pair_by_instance.
        condition_ids: The CT and/or SIC conditions to check. Defaults to
            every solver "aoccbs" or "sic" condition in the experiment
            table.
        tolerance: Float tolerance for the comparison.

    returns:
        (violations, orphans). violations is empty exactly when the
        soundness condition holds everywhere it was checked. orphans lists
        instances present for D0 or a condition but not both -- an expected
        coverage gap (AOC-CBS stopping a scenario early) shows up here too,
        so this is informational rather than proof of a bug, but it is the
        same class of comparison-integrity problem as a violation, so it is
        reported rather than dropped.
    """
    violations: list[Violation] = []
    orphans: list[Orphan] = []
    for instance_key, by_condition in paired.items():
        dt = by_condition.get("D0")
        for condition_id in condition_ids:
            ct = by_condition.get(condition_id)
            if dt is None and ct is None:
                continue
            if dt is None:
                orphans.append(Orphan(instance_key, condition_id,
                                       condition_id))
                continue
            if ct is None:
                orphans.append(Orphan(instance_key, condition_id, "D0"))
                continue
            if not dt.closed or dt.soc_ub is None or ct.soc_lb is None:
                continue
            excess = ct.soc_lb - dt.soc_ub
            if excess > tolerance:
                violations.append(Violation(instance_key, condition_id,
                                             ct.soc_lb, dt.soc_ub, excess))
    return violations, orphans


def raise_if_unsound(violations: list[Violation]) -> None:
    """Raises if any violation was found, listing every one.

    args:
        violations: The first return value of check_ct_lower_bound_sound.

    raises:
        ValueError: If violations is non-empty.
    """
    if not violations:
        return
    lines = [
        f"  {v.instance_key} {v.condition_id}: ct_lb={v.ct_lb} > "
        f"dt_cost={v.dt_cost} (excess={v.excess:.6g})"
        for v in violations
    ]
    raise ValueError(
        f"CT lower bound soundness violated on {len(violations)} "
        "instance(s); every downstream number is unsound until this is "
        "fixed:\n" + "\n".join(lines))


def exact_lower_bound_counts(
        paired: Paired,
        condition_ids: tuple[str, ...] = CT_CONDITION_IDS,
        tolerance: float = TOLERANCE,
) -> dict[str, int]:
    """Counts, per CT condition, instances where ct_lb exactly matches D0's cost.

    CT_opt <= DT_opt always (the embedding condition) and ct_lb <= CT_opt
    always (it is a lower bound), so ct_lb == dt_ub pins CT_opt = DT_opt: the
    gap is proven exactly zero on that instance, not merely bounded.

    args:
        paired: The output of load.pair_by_instance.
        condition_ids: The CT conditions to count over.
        tolerance: Float tolerance for the comparison.

    returns:
        condition_id -> count of instances where ct_lb == D0's exact cost.
    """
    counts = {condition_id: 0 for condition_id in condition_ids}
    for by_condition in paired.values():
        dt = by_condition.get("D0")
        if dt is None or not dt.closed or dt.soc_ub is None:
            continue
        for condition_id in condition_ids:
            ct = by_condition.get(condition_id)
            if ct is None or ct.soc_lb is None:
                continue
            if abs(ct.soc_lb - dt.soc_ub) <= tolerance:
                counts[condition_id] += 1
    return counts
