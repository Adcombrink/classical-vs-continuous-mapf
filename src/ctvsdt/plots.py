"""The two figures of the paper.

`plot_normalised_cost` draws the cost change from solving MAPF_R instead of
classical MAPF: one row per map, one column per connectedness, each cost
panel over a success-rate strip. `plot_grid_refinement` compares moving to
continuous time against refining the grid, on the empty-16-16 / empty-32-32
pair. Both take records and an output path without a suffix, and write a PDF
and a PNG.

Colour means radius everywhere: sqrt2/4, sqrt2/8 and sqrt2/16 each get one
colour shared by every k that uses it, and a panel draws a single k, so
colour is unambiguous within a panel. Conditions are labelled by what they
are, `k=2, r=sqrt2/4`, not by their table row name `C2-a`.

Maps run different agent-count grids, so x is shared within a row only.
"""

from __future__ import annotations

import pathlib
import statistics
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import legend_handler
from matplotlib import ticker

from ctvsdt import conditions
from ctvsdt import load
from ctvsdt import metrics
from ctvsdt.record import Record

CT_CONDITIONS = ("C2-a", "C2-b", "C2-c",
                 "C3-a", "C3-b", "C3-c",
                 "C4-a", "C4-b", "C4-c")

# One column per connectedness, left to right. A k with no records on a map
# keeps its column so the grid lines up across rows.
PANEL_KS = (2, 3, 4)

# Agent counts the campaign does not support, per map: too few CBS closures
# for a meaningful average, or bounds too wide to say anything. Only
# room-64-64-8 at 50 agents qualifies, where CBS closed 1 instance of 25.
# A list rather than a threshold, since the second half of that rule is a
# judgement: empty-16-16 at 60 and maze-32-32-2 at 30 both sit at 11 of 25
# and are kept deliberately.
EXCLUDED_AGENT_COUNTS: dict[str, tuple[int, ...]] = {
    "room-64-64-8": (50,),
}

# One fixed colour per radius, shared by every k at that radius.
COLORS = {
    "D0": "#2a78d6",
    "C2-a": "#eb6834",
    "C2-b": "#1baf7a",
    "C2-c": "#8250c4",
    "C3-a": "#eb6834",
    "C3-b": "#1baf7a",
    "C3-c": "#8250c4",
    "C4-a": "#eb6834",
    "C4-b": "#1baf7a",
    "C4-c": "#8250c4",
}

# Column headings: records carry k, readers recognise the grid.
K_TITLES = {2: "4-Connected", 3: "8-Connected", 4: "16-Connected"}

# Both bounds of a condition are solid and in its colour, with a dot at each
# agent count sampled: they are two edges of one band, and which is which is
# not in question.
BOUND_LINESTYLE = "solid"
BOUND_MARKER = "o"
BOUND_MARKERSIZE = 1.2
# The refinement figure is one panel, so its markers can be larger.
REFINEMENT_MARKERSIZE = 1.8
# A panel holds three conditions, so only the mean band is shaded; shading
# the min/max envelope too compounds into one grey mass and loses which
# envelope belongs to which condition. The envelope is two thin lines
# instead, told apart from the bound curves by weight and by having no
# markers.
MEAN_BAND_ALPHA = 0.15
MINMAX_LINESTYLE = "solid"
MINMAX_LINEWIDTH = 0.5
# Bound curve width, in points, so it does not scale with the figure.
CURVE_LINEWIDTH = 0.8
# The success-rate strips are a quarter of a panel high, so their markers
# have to carry on their own.
STRIP_LINESTYLE = "solid"
STRIP_MARKERSIZE = 2.6

SURFACE = "#fcfcfb"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
MUTED = "#898781"
# Row labels and other text that should read as strongly as a panel heading.
INK = "#2e2d2a"


def _percent_tick_label(value: float, _pos) -> str:
    """Formats a symlog tick as a plain percentage rather than 1e-02.

    args:
        value: the tick value, a percentage.
        _pos: tick index, required by FuncFormatter and unused.

    returns:
        The value with only as many decimals as it needs.
    """
    if value == 0:
        return "0"
    if abs(value) >= 1:
        return f"{value:.0f}"
    return f"{value:g}"


def _relative_percent(value: float, _pos=None) -> str:
    """Formats a ratio to the reference as a signed percentage difference.

    1.0 is the reference itself and prints as "0%", 0.8 as "-20%". The scale
    decides the decimals: the detail panel spans a few percent, where whole
    numbers would collapse several ticks onto one label.
    """
    percent = (value - 1.0) * 100.0
    if abs(percent) < 5e-3:
        return "0%"
    digits = 0 if abs(percent) >= 5.0 else 1
    return f"\N{MINUS SIGN}{-percent:.{digits}f}%" if percent < 0 \
        else f"+{percent:.{digits}f}%"


def _agent_counts(records: list[Record]) -> list[int]:
    return sorted(set(r.num_agents for r in records))


# Row order: least to most structured environment, maze last as the one map
# where CBS rather than AOC-CBS sets the limit. A map not named here sorts
# after those that are, so new results are never silently dropped.
MAP_ORDER = ("empty-16-16", "empty-32-32", "room-64-64-8",
             "warehouse-10-20-10-2-2", "maze-32-32-2")

# empty-32-32 runs empty-16-16's scenarios with coordinates doubled, so the
# two are one problem set at two resolutions; showing both would present it
# as two benchmarks. empty-16-16 is the one dropped, since empty-32-32
# reaches a higher agent count on the same area. The refinement figure
# selects its own records and ignores this.
EXCLUDED_MAPS: tuple[str, ...] = ("empty-16-16",)

def map_names(records: list[Record]) -> list[str]:
    """Returns the map names to draw as rows, in MAP_ORDER.

    A map that MAP_ORDER does not name comes after every map it does, in
    alphabetical order. Maps in EXCLUDED_MAPS are left out even when records
    for them are present.

    args:
        records: Records to take map names from.

    returns:
        The map names to draw, ordered for display.
    """
    present = set(r.map_name for r in records) - set(EXCLUDED_MAPS)
    ordered = [m for m in MAP_ORDER if m in present]
    return ordered + sorted(present - set(MAP_ORDER))


def for_map(records: list[Record], map_name: str) -> list[Record]:
    """Returns the records for one map."""
    return [r for r in records if r.map_name == map_name]


def ct_conditions_present(records: list[Record]) -> tuple[str, ...]:
    """CT_CONDITIONS with a reported batch run in records, in table order.

    A figure function draws only the conditions this returns for the map it
    is drawing, instead of assuming all of CT_CONDITIONS, so that a map
    missing a condition draws the rest rather than nothing.

    args:
        records: Records already filtered to one map, e.g. by `for_map`.

    returns:
        The subset of CT_CONDITIONS that some record realises as a batch
        run (see `load.is_batch_run`), in CT_CONDITIONS order.
    """
    present = {load.condition_id_of(r) for r in records}
    return tuple(cid for cid in CT_CONDITIONS if cid in present)


def _ct_conditions_by_k(records: list[Record]) -> dict[int, tuple[str, ...]]:
    """`ct_conditions_present`, grouped by k.

    args:
        records: Records already filtered to one map.

    returns:
        One entry per k in PANEL_KS, holding that k's present condition ids.
    """
    by_k: dict[int, list[str]] = {k: [] for k in PANEL_KS}
    for cid in ct_conditions_present(records):
        by_k[conditions.BY_ID[cid].k].append(cid)
    return {k: tuple(cids) for k, cids in by_k.items()}


def radius_mathtext(radius_label: str) -> str:
    """Turns a plain radius label into matplotlib mathtext.

    `conditions.Condition.radius_label` is plain text ("sqrt2/4") because
    `conditions.describe` prints it to a terminal. Figures set it as mathtext
    instead, so the square root is drawn as a radical rather than spelled out.

    args:
        radius_label: A radius label as written in conditions.py.

    returns:
        The label as a mathtext fragment without its enclosing "$", e.g.
        "\\sqrt{2}/4". A label that is not of the form "sqrt2/<n>" is
        returned unchanged, so "n/a" and anything added later still render.
    """
    if not radius_label.startswith("sqrt2/"):
        return radius_label
    denominator = radius_label[len("sqrt2/"):]
    return rf"\sqrt{{2}}/{denominator}"


def condition_label(condition_id: str, with_k: bool = True) -> str:
    """Returns the figure label for a condition.

    The table row names of Section 6.2 (C2-a and so on) say nothing to a
    reader of a figure, so figures state the two parameters the condition
    actually sets.

    args:
        condition_id: the condition to label.
        with_k: whether to state k. False for a figure whose panel already
            fixes k, such as `plot_normalised_cost`, where repeating it in
            every legend entry is noise.

    returns:
        The label. A non-standard time limit is stated; the AOC-CBS commit
        is not, even though `load.condition_id_of` puts one
        in the condition id. The suffix is there to keep records of different
        solver versions from overwriting each other when they are paired by
        instance, which it still does -- it is not something a reader of the
        figure has to see. Two runs of one table row at different commits
        therefore label identically, so state which is which in the caption if
        a figure ever shows both.
    """
    base, timeout_s = load.split_condition_id(condition_id)
    condition = conditions.BY_ID[base]
    if condition.solver == "sic":
        return (f"Optimistic Bound, k={condition.k}" if with_k
                else "Optimistic Bound")
    if condition.radius is None:
        return "discrete time"
    radius = radius_mathtext(condition.radius_label)
    label = (rf"$k={condition.k},\ r={radius}$" if with_k
             else rf"$r={radius}$")
    if timeout_s is not None:
        label += f", {timeout_s:g}s limit"
    return label


def _style_ax(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# Drawn narrower than it is printed, so the document enlarges it and its
# text with it. 7 inches is the full text width of a two-column article, so
# drawing at 6.76 scales by 1.035 and a 7 pt tick label lands at about
# 7.2 pt on the page. Widening this flattens the panels rather than making
# the figure taller, and shrinks the text on the page; at 7.0 the
# enlargement is gone.
PAPER_WIDTH_INCHES = 6.76
PAPER_RC = {
    "font.size": 7.0,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 5.5,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.25,
    "legend.borderaxespad": 0.25,
    "legend.columnspacing": 1.2,
}
# The rotated map name beside each row reads as strongly as a heading.
ROW_LABEL_FONTSIZE = 7.5

# Room above the panels for the column headings and the figure legend. In
# inches, so adding a row does not squash it.
HEADER_INCHES = 0.86
# Height of one row's cost panel, set to keep it about twice as wide as it
# is tall and the whole figure inside a page's text height.
ROW_INCHES = 1.24
# Height of a row's survival strip relative to its cost panel.
SURVIVAL_HEIGHT_RATIO = 0.25
# Gap between a cost panel and its strip, as a fraction of the strip's
# height. Smaller than ROW_HSPACE, so a cell holds together as one row.
SURVIVAL_HSPACE = 0.22
# Gap between rows; has to clear the strip's x tick labels.
ROW_HSPACE = 0.30
# Gap between columns; has to clear the next column's y tick labels.
COLUMN_WSPACE = 0.22
# The strip is a quarter of a panel's height, so its label has to be
# smaller than the panel's to fit beside it.
SURVIVAL_LABEL_FONTSIZE = 5.5
# Below the default: the label is long enough that at print size it would
# run past its panel and collide with the strip's label, which
# fig.align_ylabels puts at the same x.
COST_LABEL_FONTSIZE = 6.0

# Columns sharing one y-axis within a row, by k. 8- and 16-connected share,
# since how much the extra moves add only reads on one scale. 4-connected
# scales alone: its values are two orders of magnitude smaller and would
# flatten the other two.
SHARED_Y_KS = (3, 4)

# y-scale per column. Every k in one SHARED_Y_KS group must match.
#
# The 4-connected values span two or three orders of magnitude down to
# exactly 0, which a linear axis collapses onto the zero line; symlog
# resolves them and still gives 0 a place. The other two sit in a band a few
# percent wide away from 0, where linear reads directly.
Y_SCALES = {2: "symlog", 3: "linear", 4: "linear"}
# Where symlog hands over from linear to logarithmic, in percent. Just below
# the smallest value worth resolving (the warehouse's k=2 bound, ~0.015%), so
# the linear strip holds only what is indistinguishable from zero.
SYMLOG_LINTHRESH = 0.01
# Width of the linear strip in decades; below 1 so the near-zero band does
# not dominate a short panel.
SYMLOG_LINSCALE = 0.4
# Room at the left for the rotated map names, clearing the leftmost
# column's y-labels.
ROW_LABEL_INCHES = 0.28

# Room to the left of that again for the label the cost panels share. Three
# columns of text stand at the left edge: this, the map names, then the
# strips' own small label.
FIGURE_LABEL_INCHES = 0.26
FIGURE_LABEL = (r"Cost Change from using MAPF$_R$ instead of "
                r"Classical MAPF (%)")
SURVIVAL_LABEL = "Success Rate"


def _new_grid_axes(n_rows: int, n_cols: int,
                   width: float = PAPER_WIDTH_INCHES):
    """A grid of cost panels, each with a short survival strip beneath it.

    Rows are maps and columns are k. Each cell is a cost panel over a strip
    a quarter of its height, sharing its x-axis, showing what fraction of the
    instance set CBS closed -- how much of the suite the curves above rest
    on, since the common instance set requires a D0 closure.

    Maps run different agent-count grids, so x is shared within a row, never
    down a column.

    returns:
        (fig, axes, survival_axes, header_fraction), where axes[row][col] is
        a cost panel, survival_axes[row][col] is the strip below it, and
        header_fraction is the share of the figure height reserved above the
        panels for the column headings and the figure-level legend.
    """
    row_inches = ROW_INCHES * (1.0 + SURVIVAL_HEIGHT_RATIO)
    height = HEADER_INCHES + row_inches * n_rows
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
    # One outer cell per (map, k), each split into the cost panel and its
    # strip. Nesting is what lets the strip sit tight under its own panel
    # while the rows stay far enough apart to read as separate maps; a flat
    # grid of 2 * n_rows rows can only space both the same.
    outer = fig.add_gridspec(n_rows, n_cols)

    axes, survival_axes = [], []
    for row in range(n_rows):
        cost_row, survival_row = [], []
        for col in range(n_cols):
            cell = outer[row, col].subgridspec(
                2, 1, height_ratios=[1.0, SURVIVAL_HEIGHT_RATIO],
                hspace=SURVIVAL_HSPACE)
            # The columns named in SHARED_Y_KS share one y-axis within a
            # row, so their magnitudes read directly against each other; any
            # other column scales to its own range. The first column of the
            # group owns the axis and the rest are tied to it.
            share_with = None
            if col < len(PANEL_KS) and PANEL_KS[col] in SHARED_Y_KS:
                for earlier, earlier_k in enumerate(PANEL_KS[:col]):
                    if earlier_k in SHARED_Y_KS:
                        share_with = cost_row[earlier]
                        break
            cost_ax = fig.add_subplot(cell[0], sharey=share_with)
            # sharex ties the strip to the panel it belongs under, not to
            # the row above: the maps run different agent counts.
            survival_ax = fig.add_subplot(cell[1], sharex=cost_ax)
            _style_ax(cost_ax)
            _style_ax(survival_ax)
            # The strip carries the x-axis for the cell, so the panel above
            # shows no tick labels of its own.
            cost_ax.tick_params(labelbottom=False)
            cost_row.append(cost_ax)
            survival_row.append(survival_ax)
        axes.append(cost_row)
        survival_axes.append(survival_row)
    return fig, axes, survival_axes, HEADER_INCHES / height


def _save(fig, out_path) -> None:
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # transparent=True drops the figure and axes background fills, so the
    # figure takes the background colour of whatever it is placed on -- the
    # paper's page, or a slide. The SURFACE fill set on the axes is what it
    # overrides; everything drawn on top keeps its own colour.
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight",
                transparent=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight",
                transparent=True)
    plt.close(fig)


def normalised_common_keys(paired: load.Paired,
                            ct_condition_ids: tuple[str, ...]) -> set:
    """The instances every curve of the normalised-cost figure can use.

    An instance qualifies when D0 closed on it and every CT condition reports
    a lower bound. This is what makes the curves comparable: averaged over
    each condition's own instances instead, a condition that answers only the
    harder instances of an agent count looks different for that reason
    alone.

    args:
        paired: The output of load.pair_by_instance, for a single map.
        ct_condition_ids: The CT conditions drawn in the figure.

    returns:
        The qualifying instance keys.
    """
    keys = set()
    for key, by_condition in paired.items():
        dt = by_condition.get("D0")
        if dt is None or not dt.closed or dt.soc_ub is None:
            continue
        cts = [by_condition.get(cid) for cid in ct_condition_ids]
        if any(ct is None or ct.soc_lb is None for ct in cts):
            continue
        keys.add(key)
    return keys


def reduction_series(paired: load.Paired, keys, condition_id: str,
                      agent_counts) -> tuple:
    """Per-agent-count mean and extreme cost bounds of one condition.

    Every value is `metrics.reduction_percent`: the continuous-time cost as a
    percentage above or below the discrete-time one, so 0 is the
    discrete-time solution and negative is cheaper than it.

    args:
        paired: pair_by_instance output for one map.
        keys: The instances to average over, normally the row's common set.
        condition_id: The CT condition to measure.
        agent_counts: The agent counts to produce a point for, in order.

    returns:
        (xs, mean upper bound, mean lower bound, largest upper bound,
        smallest lower bound), all the same length. An agent count no
        instance answers is left out, so a condition's curve stops where its
        coverage does.
    """
    xs, ub_mean, lb_mean, ub_max, lb_min = [], [], [], [], []
    for n in agent_counts:
        row_ub, row_lb = [], []
        for key in keys:
            by_condition = paired[key]
            dt = by_condition["D0"]
            if dt.num_agents != n:
                continue
            ct = by_condition.get(condition_id)
            if ct is None or ct.soc_lb is None:
                continue
            ct_ub = metrics.tighten_ct_upper_bound(ct.soc_ub, dt.soc_ub)
            # The DT cost is the denominator and does not move, so each CT
            # bound stays the bound of the same name here.
            row_ub.append(metrics.reduction_percent(ct_ub, dt.soc_ub))
            row_lb.append(metrics.reduction_percent(ct.soc_lb, dt.soc_ub))
        if not row_ub:
            continue
        xs.append(n)
        ub_mean.append(statistics.mean(row_ub))
        lb_mean.append(statistics.mean(row_lb))
        # From the same per-instance quantities as the means, so the
        # envelope always contains the band.
        ub_max.append(max(row_ub))
        lb_min.append(min(row_lb))
    return xs, ub_mean, lb_mean, ub_max, lb_min


def _draw_reduction_series(ax, series: tuple, color: str, label: str) -> None:
    """Draws one condition: shaded mean band, min/max lines, bound curves.

    args:
        ax: The panel to draw on.
        series: The output of `reduction_series`.
        color: The radius colour of the condition.
        label: Legend label, or "" to leave the condition out of the legend.
    """
    xs, ub_mean, lb_mean, ub_max, lb_min = series
    if not xs:
        return
    ax.fill_between(xs, lb_mean, ub_mean, color=color,
                     alpha=MEAN_BAND_ALPHA, linewidth=0, zorder=2)
    for ys in (ub_max, lb_min):
        ax.plot(xs, ys, color=color, linewidth=MINMAX_LINEWIDTH,
                 linestyle=MINMAX_LINESTYLE, zorder=3)
    # Only one of the two bounds is labelled: they share a colour, so one
    # legend entry covers the pair.
    ax.plot(xs, lb_mean, color=color, linewidth=CURVE_LINEWIDTH,
             linestyle=BOUND_LINESTYLE, marker=BOUND_MARKER,
             markersize=BOUND_MARKERSIZE, label=label or None, zorder=4)
    ax.plot(xs, ub_mean, color=color, linewidth=CURVE_LINEWIDTH,
             linestyle=BOUND_LINESTYLE, marker=BOUND_MARKER,
             markersize=BOUND_MARKERSIZE, zorder=4)


def _label_rows(fig, axes, survival_axes, maps: list[str]) -> None:
    """Writes the shared cost label and each row's map name at the left edge.

    A map name belongs to the whole row and the cost label to every row, so
    each is written once, the label farther out. Both are placed after the
    layout is fixed, since each is centred on what it names.

    args:
        fig: the figure.
        axes: cost panels, indexed axes[row][col].
        survival_axes: survival strips, indexed the same way.
        maps: the map name of each row, in order.
    """
    name_x = FIGURE_LABEL_INCHES / fig.get_figwidth() + 0.012
    for row, map_name in enumerate(maps):
        top = axes[row][0].get_position().y1
        bottom = survival_axes[row][0].get_position().y0
        fig.text(name_x, (top + bottom) / 2.0, _display_map_name(map_name),
                 rotation=90, va="center", ha="left",
                 fontsize=ROW_LABEL_FONTSIZE, color=INK)
    # Centred on the rows, not on the figure, so the legend above them does
    # not pull it off centre.
    top = axes[0][0].get_position().y1
    bottom = survival_axes[-1][0].get_position().y0
    fig.text(0.012, (top + bottom) / 2.0, FIGURE_LABEL,
             rotation=90, va="center", ha="left",
             fontsize=ROW_LABEL_FONTSIZE, color=INK)


def _display_map_name(map_name: str) -> str:
    """Capitalises a map name for display.

    Only the first letter changes: the rest identifies the benchmark file.
    """
    return map_name[:1].upper() + map_name[1:]


def survival_rates(paired: load.Paired, agent_counts: list[int],
                   condition_id: str) -> dict[int, float]:
    """Percentage of a map's instances a condition survived, per agent count.

    Surviving means proving optimality for CBS, which returns a plan only at
    the end, and returning any feasible solution for AOC-CBS, which is
    already a usable plan and an upper bound. So the two rates compare "CBS
    finished" against "AOC-CBS produced something": what a user of each would
    get within the limit.

    The denominator is every instance of the map at that agent count, not
    the common set the cost curves are averaged over: the common set already
    requires D0 to have closed, which would make the discrete-time rate 100%
    by construction and hide exactly what this is meant to show. An instance
    the condition has no record for -- one AOC-CBS skipped after finding no
    feasible solution at a lower agent count -- counts as not surviving.

    args:
        paired: instances of one map, grouped by instance key.
        agent_counts: the agent counts to report, in order.
        condition_id: the condition to count.

    returns:
        {agent_count: percentage survived}, omitting agent counts with no
        instances at all.
    """
    base, _ = load.split_condition_id(condition_id)
    wants_optimal = conditions.BY_ID[base].solver == "cbs"
    rates = {}
    for n in agent_counts:
        keys = [key for key, by_condition in paired.items()
                if key[3] == n and "D0" in by_condition]
        if not keys:
            continue
        survived = 0
        for key in keys:
            record = paired[key].get(condition_id)
            if record is None:
                continue
            if record.closed if wants_optimal else record.soc_ub is not None:
                survived += 1
        rates[n] = 100.0 * survived / len(keys)
    return rates


def plot_normalised_cost(
        records: list[Record],
        out_path) -> dict[str, dict[str, dict[int, int]]]:
    """The cost change from solving MAPF-R instead of classical MAPF.

    Per instance the quantity is `metrics.reduction_percent(ct, dt_ub)` =
    100 * (ct / dt_ub - 1), with dt_ub CBS's solution cost. Negative is
    cheaper than discrete time, so the panels read downwards from 0.

    Each condition draws two curves against that fixed denominator, an upper
    and a lower bound on the reduction, with the band between them shaded.
    A thin line above and below carries the extent across scenarios: the max
    and min of the same per-instance quantities, so it always encloses the
    band. It says whether a mean describes the instances or averages over
    two unlike halves of them.

    The upper bound is clipped at 0% by `metrics.tighten_ct_upper_bound`,
    since the DT cost is itself a valid CT upper bound. That also makes an
    instance with no CT solution well-defined, so the instance filter needs
    only a D0 closure and the condition's ct_lb. Zero is therefore the
    ceiling of the quantity, but no reference line is drawn there: it would
    force every panel's range up to it and cost the 8- and 16-connected
    panels most of their height.

    Every curve of a row is averaged over one common instance set, from
    `normalised_common_keys`.

    args:
        records: Every record to draw, all maps.
        out_path: Output path without a suffix; a PDF and a PNG are written.

    returns:
        counts[map_name][agent_count] = instances in the common set there.
    """
    # Drawn under PAPER_RC so that the point sizes written into the figure
    # are the ones it is read at on the page. See PAPER_WIDTH_INCHES.
    with matplotlib.rc_context(PAPER_RC):
        return _draw_normalised_cost(records, out_path)


def _draw_normalised_cost(
        records: list[Record],
        out_path) -> dict[str, dict[str, dict[int, int]]]:
    """Draws the figure of `plot_normalised_cost`, which see."""
    maps = map_names(records)
    fig, axes, survival_axes, header = _new_grid_axes(len(maps),
                                                      len(PANEL_KS))
    counts: dict[str, dict[int, int]] = {}
    # One figure-level radius key rather than a legend per panel, gathered
    # over all maps since a map may be missing a radius the others have.
    drawn_conditions: list[str] = []

    for row, map_name in enumerate(maps):
        map_records = for_map(records, map_name)
        paired = load.pair_by_instance(map_records)
        agent_counts = _agent_counts(map_records)
        k_groups = _ct_conditions_by_k(map_records)
        ct_condition_ids = tuple(cid for k in PANEL_KS
                                 for cid in k_groups[k])
        for condition_id in ct_condition_ids:
            if condition_id not in drawn_conditions:
                drawn_conditions.append(condition_id)
        common = normalised_common_keys(paired, ct_condition_ids)
        counts[map_name] = {
            n: sum(1 for key in common if paired[key]["D0"].num_agents == n)
            for n in agent_counts}
        drawn = [n for n in agent_counts if counts[map_name][n]
                 and n not in EXCLUDED_AGENT_COUNTS.get(map_name, ())]

        for col, k in enumerate(PANEL_KS):
            ax = axes[row][col]
            for condition_id in k_groups[k]:
                _draw_reduction_series(
                    ax, reduction_series(paired, common, condition_id, drawn),
                    COLORS[condition_id],
                    condition_label(condition_id, with_k=False))

            if Y_SCALES[k] == "symlog":
                ax.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH,
                               linscale=SYMLOG_LINSCALE)
                ax.yaxis.set_major_locator(ticker.SymmetricalLogLocator(
                    base=10.0, linthresh=SYMLOG_LINTHRESH))
                ax.yaxis.set_major_formatter(
                    ticker.FuncFormatter(_percent_tick_label))
            else:
                # The warehouse sits within 0.1% of 0, which matplotlib
                # renders as an offset ("+1e2" in the corner). Force plain
                # ticks.
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                # A short panel gets too few ticks by default, leaving the
                # curves with no labelled line near them.
                ax.yaxis.set_major_locator(
                    ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
            # Agent counts are whole numbers, so the default locator's
            # half-steps name counts neither solver was run at. nbins and
            # steps matter as much as integer=True, or the labels of the
            # wider rows run together ("90100"). The strip shares this axis.
            ax.xaxis.set_major_locator(
                ticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10],
                                   integer=True))
            # k is stated once at the top of its column.
            if row == 0:
                ax.set_title(K_TITLES[k])
            # The radius key is drawn once for the figure, below.

            strip = survival_axes[row][col]
            for condition_id in ("D0",):
                rates = survival_rates(paired, drawn, condition_id)
                if not rates:
                    continue
                xs = sorted(rates)
                strip.plot(xs, [rates[n] for n in xs],
                           color=COLORS[condition_id],
                           linewidth=CURVE_LINEWIDTH,
                           linestyle=STRIP_LINESTYLE, marker="o",
                           markersize=STRIP_MARKERSIZE, zorder=3)
            strip.set_ylim(0.0, 100.0)
            strip.set_yticks((0, 50, 100))
            if row == len(maps) - 1:
                strip.set_xlabel("Number of Agents")

        # Nothing here is positive, so hold a symlog panel's top at 0 or the
        # symmetric locator offers decades above zero that no curve reaches.
        for col, k in enumerate(PANEL_KS):
            if Y_SCALES[k] == "symlog":
                axes[row][col].set_ylim(top=0.0)

        # The cost panels share one figure-level label, written in
        # `_label_rows`; only the strip is labelled per row.
        survival_axes[row][0].set_ylabel(SURVIVAL_LABEL,
                                          fontsize=SURVIVAL_LABEL_FONTSIZE)

    # Each row places its own y-label just clear of its own tick labels, so a
    # row with wider ticks -- the warehouse reads 100.00 where the others read
    # 100 -- puts its label further left than the rest. Aligning the column
    # keeps the strip labels in one line rather than a ragged edge.
    fig.align_ylabels([survival_axes[row][0] for row in range(len(maps))])

    # hspace here separates the map rows only; the gap between a cost panel
    # and its own strip is set by the nested grid in _new_grid_axes. It has
    # to clear one row's x-axis label before the next row's panel starts.
    # hspace and wspace are fractions of the average panel size, so they
    # have to grow when the panels shrink.
    fig.subplots_adjust(
        top=1.0 - header, wspace=COLUMN_WSPACE, hspace=ROW_HSPACE,
        left=(FIGURE_LABEL_INCHES + ROW_LABEL_INCHES) / fig.get_figwidth()
             + 0.075)
    _label_rows(fig, axes, survival_axes, maps)

    # One entry for the pair of bounds: they are drawn identically, and the
    # band between them is the point. The lower one is clipped at 0.
    bounds_proxy = plt.Line2D([], [], color=MUTED, linewidth=CURVE_LINEWIDTH,
                               linestyle=BOUND_LINESTYLE, marker=BOUND_MARKER,
                               markersize=BOUND_MARKERSIZE,
                               label=r"mean $\Delta_{LB}$ and $\Delta_{UB}$")
    # The line rather than the fill: at this alpha a patch swatch is close
    # to invisible at legend size.
    minmax_proxy = plt.Line2D([], [], color=MUTED,
                               linewidth=MINMAX_LINEWIDTH,
                               linestyle=MINMAX_LINESTYLE,
                               label=r"min $\Delta_{LB}$ and max $\Delta_{UB}$")
    # One entry per radius, de-duplicated by label: every k at one radius
    # shares a colour and would otherwise repeat "r=sqrt2/4".
    radius_proxies, seen = [], set()
    for condition_id in CT_CONDITIONS:
        if condition_id not in drawn_conditions:
            continue
        label = condition_label(condition_id, with_k=False)
        if label in seen:
            continue
        seen.add(label)
        radius_proxies.append(plt.Line2D([], [], color=COLORS[condition_id],
                                          linewidth=CURVE_LINEWIDTH,
                                          label=label))
    # Two rows: the radii above, how a curve is drawn below. matplotlib
    # fills a multi-column legend column by column, so the handles are
    # padded and transposed to land row by row.
    blank = plt.Line2D([], [], linestyle="none", label="")
    legend_rows = [list(radius_proxies),
                   [bounds_proxy, minmax_proxy]]
    columns = max(len(row) for row in legend_rows)
    legend_rows = [row + [blank] * (columns - len(row)) for row in legend_rows]
    handles = [row[column] for column in range(columns) for row in legend_rows]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0 - 0.22 * header), ncol=columns,
               frameon=False, alignment="left")

    _save(fig, out_path)
    return counts


# --- The grid-refinement comparison -----------------------------------------
#
# empty-32-32 runs empty-16-16's scenarios with every coordinate doubled, so
# the two maps solve the same physical instances at two resolutions. The
# question that makes answerable: for an agent half the largest size a
# 4-connected grid represents soundly, is refining the grid -- where that
# agent is again the largest sound size -- as good as moving to continuous
# time?
#
# Costs are compared in coarse units, so every fine cost is halved. That is a
# unit conversion, one coarse cell spanning two fine ones; any departure from
# a ratio of exactly 2 is the effect being measured.
#
# The physical agent is the same in every series. It is r = sqrt2/8 in coarse
# cells, which is r = sqrt2/4 in fine cells, which is why the two grids are
# read at different radius labels.
COARSE_MAP = "empty-16-16"
FINE_MAP = "empty-32-32"
FINE_SCALE = 2.0
COARSE_RADIUS = 0.1767766952966369    # sqrt2/8 in coarse cells
FINE_RADIUS = 0.3535533905932738      # sqrt2/4 in fine cells, the same agent

# Colour is connectedness here, not radius: this figure fixes the agent size
# and varies the move set. Each move set gets a pair of hues, darker for the
# coarse grid and lighter for the fine one, and line style repeats the same
# split so the pairing survives greyscale.
#
# Discrete time takes one colour for both grids, told apart by line style
# alone: it is the baseline the figure measures against, not a fourth move
# set, and one colour says the two curves are the same thing on two grids.
REFINEMENT_COLORS = {
    "DT": {"coarse": "#4a5560", "fine": "#4a5560"},
    2: {"coarse": "#c0392b", "fine": "#f0913f"},
    3: {"coarse": "#1f5fa8", "fine": "#35b6c4"},
    4: {"coarse": "#1e7a3c", "fine": "#8cc63f"},
}
COARSE_LINESTYLE = "solid"
FINE_LINESTYLE = (0, (3.2, 1.4))


def _refinement_key(record: Record) -> tuple:
    """Returns the physical instance a record answers, across both grids.

    The ported scenarios carry the source map's name plus an "-x2" marker, so
    dropping the marker is what makes a coarse and a fine record name the same
    physical instance.
    """
    stem = record.scen_file.split("/")[-1].replace(".scen:", "")
    return stem.replace("-x2", ""), record.scen_index, record.num_agents


def refinement_series(records: list[Record]) -> tuple[list[int], dict]:
    """Returns the agent counts and per-series mean costs of the comparison.

    Every cost is divided by the coarse grid's exact discrete-time cost for
    the same instance, so 1 is that solution and 0.8 is 20% cheaper. The mean
    is over instances where both grids' CBS runs closed, which is where the
    reference is exact.

    Unclosed continuous-time runs are kept: their upper bound is a solution
    that exists, and the lower bound says how much further the cost could
    fall. Upper bounds are clipped to the discrete-time cost on the same
    grid; see `upper_bound` below.

    args:
        records: Every record of the two empty maps. empty-32-32 records on
            its own scenarios are ignored, having no coarse counterpart.

    returns:
        The agent counts in order, and a mapping from series name to a dict
        with "lb", "ub" and "closed" lists parallel to those counts, plus
        "label", "color" and "linestyle".
    """
    wanted = {}
    for record in records:
        if record.map_name not in (COARSE_MAP, FINE_MAP):
            continue
        fine = record.map_name == FINE_MAP
        if fine and "-x2" not in record.scen_file:
            continue
        if record.solver == "cbs":
            name = "DT-fine" if fine else "DT-coarse"
        elif record.solver == "aoccbs":
            radius = FINE_RADIUS if fine else COARSE_RADIUS
            if abs(record.radius - radius) > 1e-9:
                continue
            name = f"CT-{'fine' if fine else 'coarse'}-k{record.k}"
        else:
            continue
        scale = FINE_SCALE if fine else 1.0
        wanted.setdefault(_refinement_key(record), {})[name] = (
            record.soc_lb / scale if record.soc_lb is not None else None,
            record.soc_ub / scale if record.soc_ub is not None else None,
            record.closed)

    series_names = ["DT-fine"] + [f"CT-{grid}-k{k}"
                                  for grid in ("coarse", "fine")
                                  for k in PANEL_KS]
    required = ["DT-coarse"] + series_names
    usable = [key for key, got in wanted.items()
              if all(name in got for name in required)
              and got["DT-coarse"][2] and got["DT-fine"][2]]
    agent_counts = sorted({key[2] for key in usable})

    # A discrete-time plan is also a continuous-time plan: on a 4-connected
    # grid two agents obeying the vertex and edge constraints stay at least
    # sqrt2/2 apart while moving, so a disc of radius up to sqrt2/4 is sound,
    # and each grid's agent is at or below its own limit. The CT optimum
    # therefore cannot exceed the DT cost on the same grid, and an upper
    # bound above it is the budget running out, not the problem.
    def upper_bound(key: tuple, name: str, grid: str, k) -> float:
        """Returns the cost of the best plan known to exist for one instance."""
        found = wanted[key][name][1]
        if k is None:
            return found
        discrete = "DT-coarse" if grid == "coarse" else "DT-fine"
        return min(found, wanted[key][discrete][1])

    out = {}
    for name in series_names:
        grid = "fine" if name.endswith("fine") or "-fine-" in name else "coarse"
        k = None if name.startswith("DT") else int(name[-1])
        lbs, ubs, closed = [], [], []
        for n in agent_counts:
            keys = [key for key in usable if key[2] == n]
            reference = [wanted[key]["DT-coarse"][1] for key in keys]
            lbs.append(statistics.mean(wanted[key][name][0] / ref
                                       for key, ref in zip(keys, reference)))
            ubs.append(statistics.mean(upper_bound(key, name, grid, k) / ref
                                       for key, ref in zip(keys, reference)))
            closed.append(sum(wanted[key][name][2] for key in keys))
        out[name] = {
            "lb": lbs, "ub": ubs, "closed": closed,
            "color": REFINEMENT_COLORS["DT" if k is None else k][grid],
            "grid": grid, "k": k,
            "linestyle": COARSE_LINESTYLE if grid == "coarse"
                         else FINE_LINESTYLE,
            "label": ("discrete time" if k is None
                      else f"continuous time, {K_TITLES[k].lower()}"),
        }
    sizes = {n: sum(1 for key in usable if key[2] == n) for n in agent_counts}
    return agent_counts, out, sizes


# One panel, one column of a two-column paper. Drawn at 3.4/1.25 so PAPER_RC's
# point sizes land on the page at the same size as in the other figure.
REFINEMENT_WIDTH_INCHES = 2.72
REFINEMENT_FIGSIZE = (REFINEMENT_WIDTH_INCHES, 2.05)

# The band spans the lower bound to a solution that exists. Both edges get a
# thin line in the series' colour: a fill alone has no definite boundary
# where two of them overlap.
REFINEMENT_BAND_ALPHA = 0.18
BAND_EDGE_LINEWIDTH = 0.45


def plot_grid_refinement(records: list[Record], out_path) -> dict:
    """Draws the grid-refinement comparison and returns its instance counts.

    args:
        records: Every record of the two empty maps.
        out_path: Path without a suffix; a .png and a .pdf are written.

    returns:
        Instances averaged over, per agent count.
    """
    agent_counts, series, sizes = refinement_series(records)
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=REFINEMENT_FIGSIZE, facecolor=SURFACE)
        _style_ax(ax)
        ax.axhline(1.0, color=REFINEMENT_COLORS["DT"]["coarse"],
                   linewidth=CURVE_LINEWIDTH, linestyle=COARSE_LINESTYLE,
                   zorder=2)
        for spec in series.values():
            # The fine grid's discrete-time curve goes on top: it is the
            # figure's second reference, what refining the grid buys on its
            # own, and the CT series run close enough to hide it.
            lift = 4 if spec["k"] is None else 0
            ax.fill_between(agent_counts, spec["lb"], spec["ub"],
                            color=spec["color"], alpha=REFINEMENT_BAND_ALPHA,
                            linewidth=0, zorder=3 + lift)
            ax.plot(agent_counts, spec["lb"], color=spec["color"],
                    linestyle=spec["linestyle"],
                    linewidth=BAND_EDGE_LINEWIDTH, zorder=4 + lift)
            ax.plot(agent_counts, spec["ub"], color=spec["color"],
                    linestyle=spec["linestyle"], linewidth=CURVE_LINEWIDTH,
                    marker=BOUND_MARKER, markersize=REFINEMENT_MARKERSIZE,
                    zorder=5 + lift)
        ax.set_xticks(agent_counts)
        ax.set_xlabel("Number of Agents")
        ax.set_ylabel(f"Cost vs. {DT_LEGEND_LABEL} on the {COARSE_GRID_LABEL}")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_relative_percent))
        _refinement_legend(fig, ax, series)
        _save(fig, out_path)
    return sizes


# One legend row per move set, both its colours in one handle, so a reader
# sees that the dark solid line and the light dashed one are the same move
# set on two grids. Eight separate rows would not fit beside a column-wide
# figure.
#
# The grids are named by resolution rather than by their MovingAI names, and
# two neutral sample lines on the first row say which style is which grid.
# The legend sits above the panel, anchored in axes coordinates, so nothing
# in it can collide with a curve.
DT_LEGEND_LABEL = "Classical MAPF"
CT_LEGEND_PREFIX = r"MAPF$_R$"

COARSE_GRID_LABEL = "Coarse Map"
FINE_GRID_LABEL = "2x Finer Map"

LEGEND_BBOX = (0.5, 1.03)
LEGEND_NCOL = 2
LEGEND_HANDLELENGTH = 2.4


def _refinement_legend(fig, ax, series: dict) -> None:
    """Draws the grid-refinement figure's legend above the panel."""
    style_rows = [
        (COARSE_LINESTYLE, COARSE_GRID_LABEL),
        (FINE_LINESTYLE, FINE_GRID_LABEL),
    ]
    series_rows = [("DT", DT_LEGEND_LABEL)]
    series_rows += [(k, f"{CT_LEGEND_PREFIX}, {K_TITLES[k].lower()}")
                    for k in PANEL_KS]

    def style_handle(linestyle):
        return plt.Line2D([], [], color=MUTED, linewidth=CURVE_LINEWIDTH,
                          linestyle=linestyle)

    def series_handle(key):
        return tuple(
            plt.Line2D([], [], color=REFINEMENT_COLORS[key][grid],
                       linewidth=CURVE_LINEWIDTH,
                       linestyle=(COARSE_LINESTYLE if grid == "coarse"
                                  else FINE_LINESTYLE))
            for grid in ("coarse", "fine"))

    # A legend fills its columns top to bottom, so the entries are listed one
    # column at a time to put the two grid styles side by side on the first
    # row and the move sets in the two rows under them.
    columns = [
        [(style_handle(style_rows[0][0]), style_rows[0][1])]
        + [(series_handle(key), label) for key, label in series_rows[:2]],
        [(style_handle(style_rows[1][0]), style_rows[1][1])]
        + [(series_handle(key), label) for key, label in series_rows[2:]],
    ]
    entries = [entry for column in columns for entry in column]
    handles = [handle for handle, _ in entries]
    labels = [label for _, label in entries]
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=LEGEND_BBOX,
        bbox_transform=ax.transAxes, ncol=LEGEND_NCOL,
        frameon=False, handlelength=LEGEND_HANDLELENGTH, alignment="left",
        handler_map={tuple: legend_handler.HandlerTuple(ndivide=None, pad=0.3)})
