"""Comparisons between a discrete-time and a continuous-time cost.

Both sides are bounds rather than exact costs, since either solver can time
out, so every metric here takes the four bounds soc_dt_lb, soc_dt_ub,
soc_ct_lb, soc_ct_ub and is None whenever one it needs is None.

Under the sqrt(2)/4 embedding condition, any DT solution is also a feasible
CT schedule on the same move set, so CT_opt <= DT_opt always. That is what
lets `tighten_ct_upper_bound` replace a CT upper bound above the DT cost, and
what makes the true gap one-sided.

Costs are sums of costs; the same functions apply to makespan.
"""

from __future__ import annotations

from typing import Optional

Interval = tuple[Optional[float], Optional[float]]


def tighten_ct_lower_bound(ct_lb: float, sic: Optional[float]) -> float:
    """Raises a CT lower bound to the sum of individual costs.

    Each agent costs at least its individual shortest path, so the SIC is a
    valid lower bound and needs no search.

    args:
        ct_lb: Lower bound reported by AOC-CBS.
        sic: Sum of individual shortest-path costs, or None.

    returns:
        The tighter bound, or ct_lb if sic is None.
    """
    if sic is None:
        return ct_lb
    return max(ct_lb, sic)


def guaranteed_gap(dt_lb: Optional[float],
                    ct_ub: Optional[float]) -> Optional[float]:
    """The gap already proved, without either solver closing optimality.

    Clamped at zero: CT_opt <= DT_opt, so a negative dt_lb - ct_ub means
    ct_ub is loose, not that the gap is negative.

    args:
        dt_lb: Lower bound on the optimal discrete-time cost.
        ct_ub: Cost of the best continuous-time solution found.

    returns:
        The proved gap, at least zero, or None if a bound is missing.
    """
    if dt_lb is None or ct_ub is None:
        return None
    return max(0.0, dt_lb - ct_ub)


def gap_upper_bound(dt_ub: Optional[float],
                     ct_lb: Optional[float]) -> Optional[float]:
    """The largest the true gap can be.

    Claiming the gap is *small* rests on this end: `guaranteed_gap` can read
    zero purely because ct_ub is loose.

    args:
        dt_ub: Cost of the best discrete-time solution found.
        ct_lb: Lower bound on the optimal continuous-time cost.

    returns:
        The gap upper bound, or None if a bound is missing.
    """
    if dt_ub is None or ct_lb is None:
        return None
    return dt_ub - ct_lb


def tighten_ct_upper_bound(ct_ub: Optional[float], soc_dt: float) -> float:
    """Lowers a CT upper bound to at most the DT cost.

    The DT cost is itself a valid CT upper bound, so a ct_ub above it is
    dominated. With no CT solution at all the DT cost still bounds it.

    args:
        ct_ub: Cost of the best continuous-time solution found, or None.
        soc_dt: The DT cost on the same instance.

    returns:
        min(ct_ub, soc_dt), or soc_dt if ct_ub is None.
    """
    if ct_ub is None:
        return soc_dt
    return min(ct_ub, soc_dt)


def gap_percent(soc_dt: float, ct_cost: float) -> float:
    """How much more the DT solution costs than the CT one, in percent.

    The CT cost is the denominator, so the CT bounds swap roles: the CT upper
    bound gives the gap lower bound.

    args:
        soc_dt: The DT cost on the instance.
        ct_cost: A continuous-time cost or bound on the same instance.

    returns:
        100 * (soc_dt / ct_cost - 1).
    """
    return 100.0 * (soc_dt / ct_cost - 1.0)


def reduction_percent(ct_cost: float, soc_dt: float) -> float:
    """What the CT solution costs relative to the DT one, in percent.

    Negative is a saving. The DT cost is the denominator and does not move,
    so each CT bound stays the bound of the same name. This is the direction
    the normalised-cost figure plots.

    args:
        ct_cost: A continuous-time cost or bound on the instance.
        soc_dt: The DT cost on the same instance.

    returns:
        100 * (ct_cost / soc_dt - 1).
    """
    return 100.0 * (ct_cost / soc_dt - 1.0)


def gap_interval(dt_lb: Optional[float], dt_ub: Optional[float],
                  ct_lb: Optional[float], ct_ub: Optional[float]) -> Interval:
    """Bounds the amount by which the DT optimum exceeds the CT optimum.

    args:
        dt_lb: Lower bound on the optimal discrete-time cost.
        dt_ub: Cost of the best discrete-time solution found.
        ct_lb: Lower bound on the optimal continuous-time cost.
        ct_ub: Cost of the best continuous-time solution found.

    returns:
        (guaranteed_gap(dt_lb, ct_ub), gap_upper_bound(dt_ub, ct_lb)).
    """
    return (guaranteed_gap(dt_lb, ct_ub), gap_upper_bound(dt_ub, ct_lb))


def recovery_fraction_interval(
        dt_lb: Optional[float], dt_ub: Optional[float],
        ct_lb: Optional[float], ct_ub: Optional[float],
        sic: Optional[float]) -> Optional[Interval]:
    """Bounds the share of the achievable headroom that CT recovers.

    The headroom is dt_lb - sic, the distance from the DT lower bound to the
    r -> 0 limit.

    args:
        dt_lb: Lower bound on the optimal discrete-time cost.
        dt_ub: Cost of the best discrete-time solution found.
        ct_lb: Lower bound on the optimal continuous-time cost.
        ct_ub: Cost of the best continuous-time solution found.
        sic: Sum of individual shortest-path costs, or None.

    returns:
        The interval, or None when sic is None or the headroom is not
        positive.
    """
    if sic is None or dt_lb is None:
        return None
    headroom = dt_lb - sic
    if headroom <= 0.0:
        return None
    if ct_lb is not None:
        ct_lb = tighten_ct_lower_bound(ct_lb, sic)
    low, high = gap_interval(dt_lb, dt_ub, ct_lb, ct_ub)
    low = None if low is None else low / headroom
    high = None if high is None else high / headroom
    return (low, high)


def interval_width(interval: Interval) -> Optional[float]:
    """Returns the width of an interval, or None if either end is None."""
    low, high = interval
    if low is None or high is None:
        return None
    return high - low
