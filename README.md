# What Classical Multi-Agent Path Finding Forfeits to the Continuous-time Formulation

Experiments, results and figures for a paper measuring how much solution
quality the classical discrete-time MAPF formulation forfeits relative to the
continuous-time formulation MAPF_R.

## What is compared

- **Discrete time**: CBS, using
  [CBSH2-RTC](https://github.com/Jiaoyang-Li/CBSH2-RTC), unmodified and at its
  default settings. Run from this repository.
- **Continuous time**: MAPF_R, using
  [AOC-CBS](https://github.com/Adcombrink/AOC-CBS). Run in its own repository,
  where its benchmarking machinery lives; its output is extracted into
  `results/aoccbs/`.

Both are git submodules pinned to the commits that produced the reported
results, and every result record stores the commit hash of the solver that
produced it.

The comparison is between the optimal costs of two formulations, not between
two implementations. Implementation quality determines whether the optimum is
reached within the time budget, not what the optimum is.

## Experimental settings

Both solvers get a 300 s budget per instance and minimise sum of costs, and
both take the first *N* rows of a scenario file as its *N* agents, so the two
sides answer identical questions.

`configs/aoccbs/solver.yaml` is the AOC-CBS settings file every reported run
used; `scripts/setup.sh` copies it into the AOC-CBS checkout, which resolves
solver settings from one fixed path. The three agent models beside it are the
discs of the radius sweep, r = sqrt2/4, sqrt2/8 and sqrt2/16.

CBS is run at CBSH2-RTC's defaults — WDG heuristics, prioritizing conflicts,
bypass, generalised rectangle and corridor reasoning, target reasoning, no
disjoint splitting, no mutex reasoning, no SIPP. `scripts/run_cbs.sh` passes
only the map, scenario, agent count, time limit and output path, so the
baseline is the solver as its authors ship it.

## Layout

    benchmarks/     MovingAI maps and scenarios, plus scenarios_x2 (see below).
    configs/aoccbs/ The AOC-CBS solver settings and agent models.
    src/ctvsdt/     The harness: records, loading, metrics and plotting.
    scripts/        Setup, preprocessing, the two solver runners, extraction
                    and figure generation.
    results/        The reported campaign, as committed data.
    figures/        The figures the paper prints, generated from results/.

`results/aoccbs/` holds one JSON Lines file per (map, k, radius, agent count),
one record per instance; see its README for what a record carries and how it
is extracted. `results/cbs/` holds one CSV per map, CBSH2-RTC's own output
rows concatenated across agent counts. AOC-CBS run directories are not kept:
they run to gigabytes, and what this paper needs is extracted from them first.

A record is one solver run on one instance, storing cost bounds `soc_lb` and
`soc_ub` rather than a single cost, since AOC-CBS often exhausts its budget
with bounds on the optima rather than proved. Everything downstream is a
function of a collection of records, which is the only coupling between the
two solvers.

`benchmarks/empty-32-32/scenarios_x2/` is empty-16-16's scenarios with every
coordinate doubled. Running empty-32-32 on those makes the two maps two
discretizations of the same physical instances, which is what the grid
refinement figure compares.

## Reproducing the figures

    pip install -e .
    python scripts/make_figures.py                  # figures/F_normalised_cost
    python scripts/make_grid_refinement_figure.py   # figures/F_grid_refinement
    python scripts/make_map_figures.py              # figures/F_map_*.pdf

The first two read only `results/`, the third only `benchmarks/`. Rerunning
the solvers is not needed to rebuild any figure.

To rerun the experiments instead, in order: `scripts/setup.sh`,
`scripts/preprocess.py`, then `scripts/run_aoccbs_sweep.py` and
`scripts/run_cbs.sh`, then `scripts/extract_sweep.py` to reduce the AOC-CBS
run directories to records. Preprocessing must finish before anything runs in
parallel: several solves share one state graph and distance table, and
building them concurrently is both wasted work and a race.

## Setup

    git clone --recurse-submodules https://github.com/Adcombrink/CTvsDT_MAPF.git
    cd CTvsDT_MAPF
    pip install -e .

`requirements.txt` covers AOC-CBS's dependencies as well, which its own
metadata does not declare.

Build CBSH2-RTC per the instructions in `external/CBSH2-RTC/README.md`. It
requires CMake and Boost.

## Licences and attribution

This repository is under the MIT licence; see `LICENSE`. That covers the
harness, the scripts and the result records, and nothing else.

The benchmark maps and scenarios under `benchmarks/` are from the MovingAI
2D pathfinding benchmark set:

> N. R. Sturtevant. Benchmarks for Grid-Based Pathfinding. *IEEE Transactions
> on Computational Intelligence and AI in Games*, 4(2):144–148, 2012.
> https://movingai.com/benchmarks/

CBSH2-RTC is distributed under a USC non-commercial research licence, which
applies to that submodule and not to this repository.
