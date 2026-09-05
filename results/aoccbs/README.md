# Extracted AOC-CBS results

AOC-CBS is run in its own repository, where the solver and its benchmarking
machinery live. This directory holds what this paper keeps from those runs:
one JSON Lines file per AOC-CBS run directory, one record per solved instance.

    room-64-64-8_k2_rsqrt2over4_main_n30.jsonl

The AOC-CBS run directories themselves are not kept. They are large -- four
runs of 2026-08-10 came to 4.9 GB of run output, against 516 KB extracted --
and nothing in this paper reads them after extraction.

## The reported campaign

Every file here belongs to one campaign, run 2026-09-01 to 2026-09-03 on
Arrhenius. There is nothing else left: the earlier 30 s, 100 s and 120 s
batches, the k=4 staging runs, the portfolio probes and the repeated-run
groups were all removed once the figures were final, because none of them
contributes to a figure the paper prints.

Every record is therefore at AOC-CBS commit `a39a4b9` and a 300 s
per-instance limit, with the settings in `configs/aoccbs/solver.yaml`, over
the 25 `random` scenarios of its map. That uniformity is what
`load.REPORTED_COMMITS` and `load.STANDARD_TIMEOUT_S` record, and it is why no
condition id here carries a suffix.

One file per (map, k, radius, agent count), named `<map>_k<k>_r<radius>_main_n<agents>.jsonl`:

    map                      conditions  agent counts                     records  closed
    empty-16-16                       9  10..60 by 10                        1350     528
    empty-32-32                       9  10..100 by 10                       2250     790
    room-64-64-8                      9  10, 15, 20, 25, 30, 35, 40, 50      1800     325
    warehouse-10-20-10-2-2            9  10, 25, 50, 75, 100, 125            1350     558
    maze-32-32-2                      9  5..30 by 5                          1350     440

The nine conditions are every pair of k in {2, 3, 4} and radius in
{sqrt2/4, sqrt2/8, sqrt2/16}. `closed` counts the instances where AOC-CBS
proved optimality; the rest report a lower and an upper bound, which is what
the figure draws as a band.

`empty-32-32` was run on `benchmarks/empty-32-32/scenarios_x2`, the
empty-16-16 scenarios with every coordinate doubled, so the two maps solve the
same physical instances at two resolutions. That pair is what
`scripts/make_grid_refinement_figure.py` compares.

## Procedure

1. Run the problems in the AOC-CBS repository at this paper's agent counts.
   AOC-CBS's own `aggregate_runs` step is not needed; extraction reads the raw
   `runs/` output.
2. Extract, from the repository root:

       python -m ctvsdt.adapters.aoccbs_extract

   The `__main__` block of `src/ctvsdt/adapters/aoccbs_extract.py` names the
   run directory it extracts and where to put it; point it at the new one. Or
   call it directly:

       from pathlib import Path
       from ctvsdt.adapters import aoccbs_extract

       summary = aoccbs_extract.extract_run(
           Path("external/AOC-CBS/runs/<run name>"),
           Path("results/aoccbs/<name>.jsonl"))
       print(summary.report())

3. Check the report: record counts per condition and per status, the entries
   AOC-CBS skipped, and any runs that errored.
4. Delete the run directory.
5. Run CBS here on the same instances.

Step 4 is the only irreversible step in the pipeline. Do not do it until step
3 looks right. Extract before moving the AOC-CBS submodule on, too: the commit
a record attributes itself to is the submodule's working commit at extraction
time, since the run output does not state it.

## Nothing is declared by hand

A run directory says everything the records need. Two of the fields are read
indirectly:

- the state graph id gives the move-set parameter, `empty-32-32_k2` -> k=2,
  and the map name;
- the agent model gives the radius, read from AOC-CBS's model library at
  `scratch/models` -- `Circular_sqrt2over4` -> 0.3535...

The radius matters because it is what separates the conditions of a radius
sweep, and it exists in the run output only as a model *name*, which nothing
forces to be honest.

The pair is looked up in the experiment table of `conditions.py` while
extracting -- k=2 with sqrt2/4 is the row named C2-a -- but the *name* is not
written to the records, because (solver, k, radius) already is the condition
and the records carry all three. `aoccbs_extract.condition_of` puts the name
back when a table or a figure wants one. What the lookup is for at extraction
time is the error: a run the experiment table does not call for, or an agent
model the library gives no radius for, stops the extraction rather than
becoming a record that nothing can later explain.

The time budget comes from each run's own `solver.yaml` `timelimit`, and the
agent count in `problem.yaml` is checked against the one in the folder name.

## What is kept and what is not

Kept: map, scenario, agent count, k, radius, cost bounds, whether optimality
was closed, runtime, status and the AOC-CBS commit. Together those say which
condition a run realises without naming it.

Not kept: per-iteration bound and runtime traces, portfolio and branch
diagnostics, the solver configuration, preprocessing time, and the plans. If a
later question needs any of these, the run has to be repeated.

## Two things to check before plotting

**Coverage.** `check_coverage` reports conditions and agent counts the records
do not contain. The point is that a gap is stated rather than left as a hole
in a figure.

**Instance identity.** `check_instances_match` reports instances present on
one side and not the other. Both solvers must answer exactly the same
questions. Costs taken from two different instance sets still look plausible
and still plot, so nothing else in the pipeline will catch this.

Note also that AOC-CBS stops a scenario once some agent count finds no
feasible solution, and skips the higher counts. Absent records are therefore
not random, and coverage gaps at high agent counts usually mean the solver
found nothing rather than that the run was forgotten. The extraction report
names those skipped entries; after step 4 they are not recoverable.
