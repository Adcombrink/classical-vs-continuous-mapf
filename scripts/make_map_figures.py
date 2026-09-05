#!/usr/bin/env python
"""Draws each benchmark map as its own PDF.

Blocked cells filled, passable cells left as the page. An edge with any
passable cell along it gets one extra row or column of blocked cells, so what
stops an agent at the boundary is drawn rather than implied; an edge that is
already solid gets nothing.

Usage:
    python scripts/make_map_figures.py [figures_dir]
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from ctvsdt import plots  # noqa: E402

BENCHMARKS = pathlib.Path("benchmarks")
MAPS = ("empty-32-32", "room-64-64-8", "warehouse-10-20-10-2-2",
        "maze-32-32-2")

# MovingAI terrain: everything that is not one of these stops an agent.
PASSABLE = ".G"

# Width of a drawn map on the page, in inches. The height follows from the
# map's own aspect ratio, so a wide map is drawn short rather than stretched.
MAP_WIDTH_INCHES = 3.2


def read_map(map_path: pathlib.Path) -> tuple[int, int, list[str]]:
    """Returns a MovingAI map's width, height and its rows of terrain.

    args:
        map_path: path to the .map file.

    returns:
        (width, height, rows), rows being the terrain lines in order.
    """
    lines = map_path.read_text().splitlines()
    header = {}
    for index, line in enumerate(lines):
        if line.strip() == "map":
            return int(header["width"]), int(header["height"]), lines[index + 1:]
        key, _, value = line.partition(" ")
        header[key] = value
    raise ValueError(f"{map_path} has no 'map' line")


def blocked_grid(rows: list[str], width: int, height: int) -> np.ndarray:
    """Returns which cells are blocked, indexed [row][column].

    args:
        rows: the map's terrain lines.
        width: the map's width in cells.
        height: the map's height in cells.

    returns:
        A height by width boolean array, True where a cell is blocked.
    """
    grid = np.ones((height, width), dtype=bool)
    for y in range(height):
        row = rows[y]
        for x in range(width):
            grid[y, x] = row[x] not in PASSABLE
    return grid


def with_border(grid: np.ndarray) -> tuple[np.ndarray, dict[str, bool]]:
    """Adds a blocked row or column to every edge that is not already solid.

    args:
        grid: which cells are blocked, indexed [row][column].

    returns:
        The padded grid, and which of "top", "bottom", "left" and "right"
        gained a row or column.
    """
    added = {
        "top": not grid[0, :].all(),
        "bottom": not grid[-1, :].all(),
        "left": not grid[:, 0].all(),
        "right": not grid[:, -1].all(),
    }
    padded = np.pad(
        grid,
        ((int(added["top"]), int(added["bottom"])),
         (int(added["left"]), int(added["right"]))),
        constant_values=True)
    return padded, added


def draw_map(grid: np.ndarray, out_path: pathlib.Path) -> None:
    """Writes one map to a PDF, blocked cells filled and nothing else drawn.

    args:
        grid: which cells are blocked, indexed [row][column].
        out_path: path to write, including the .pdf suffix.
    """
    height, width = grid.shape
    figsize = (MAP_WIDTH_INCHES, MAP_WIDTH_INCHES * height / width)
    fig, ax = plt.subplots(figsize=figsize)
    # imshow's own extent puts cell centres on integers; the half-cell offset
    # in the extent is what makes the drawn area end exactly at the outermost
    # cells' edges, so no half cell of margin is left around the map.
    ax.imshow(grid, cmap=matplotlib.colors.ListedColormap(
                  [plots.SURFACE, plots.INK]),
              vmin=0, vmax=1, interpolation="nearest", origin="upper",
              extent=(0, width, height, 0))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0,
                transparent=True)
    plt.close(fig)


def main() -> None:
    figures_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
    for map_name in MAPS:
        width, height, rows = read_map(
            BENCHMARKS / map_name / f"{map_name}.map")
        grid = blocked_grid(rows, width, height)
        padded, added = with_border(grid)
        out_path = figures_dir / f"F_map_{map_name}.pdf"
        draw_map(padded, out_path)
        edges = ", ".join(name for name, gained in added.items() if gained)
        print(f"{map_name}: {width}x{height} -> "
              f"{padded.shape[1]}x{padded.shape[0]}, "
              f"border added on {edges or 'no edge'}, "
              f"{int(grid.sum())} blocked cell(s) of {width * height}")
        print(f"  written to {out_path}")


if __name__ == "__main__":
    main()
