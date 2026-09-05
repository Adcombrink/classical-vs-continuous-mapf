#!/usr/bin/env bash
# Prepares a fresh clone so the experiments can be run.
#
#   scripts/setup.sh [--force-build]
#
# Does the three cheap steps: puts this repository's AOC-CBS inputs where
# AOC-CBS looks for them, and builds the CBS binary. Safe on a login node --
# nothing here takes more than a couple of minutes.
#
# The expensive step, building each map's state graph, distance table and
# intersection intervals, is scripts/preprocess.py and belongs in a submitted
# job. Run this first, then that.
#
# Both submodules must be checked out before this runs:
#
#   git submodule update --init --recursive
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
AOC=$ROOT/external/AOC-CBS
CBS=$ROOT/external/CBSH2-RTC

force_build=0
[ "${1:-}" = "--force-build" ] && force_build=1

fail() { echo "setup: $*" >&2; exit 1; }

[ -f "$AOC/src/aoccbs/paths.py" ] \
    || fail "external/AOC-CBS is empty; run: git submodule update --init --recursive"
[ -f "$CBS/CMakeLists.txt" ] \
    || fail "external/CBSH2-RTC is empty; run: git submodule update --init --recursive"

# ---------------------------------------------------------------------------
# 1. Solver settings
# ---------------------------------------------------------------------------
# paths.SOLVER_CONFIG_DIR is PROJECT_ROOT/config/solver with no environment
# override, so these have to be copied rather than pointed at. Whatever they
# say at the moment this runs is what a run uses -- edit one, then run this
# again.
#
# Every configs/aoccbs/solver*.yaml is copied, not just solver.yaml. A run
# that needs settings the reported batches do not use -- a different time
# limit, a shorter portfolio -- gets its own file rather than an edit to the
# shared one, because a record carries neither, so two such runs are only
# distinguishable by the file they used. run_aoccbs_sweep.py --solver-settings
# picks one by stem.
echo "==> solver settings -> $AOC/config/solver/"
mkdir -p "$AOC/config/solver"
for settings in "$ROOT"/configs/aoccbs/solver*.yaml; do
    cp -v "$settings" "$AOC/config/solver/$(basename "$settings")"
    echo "    $(grep -m1 '^timelimit:' "$settings")"
done

# ---------------------------------------------------------------------------
# 2. Agent models
# ---------------------------------------------------------------------------
# A model id resolves only against paths.MODEL_LIBRARY_DIR, which is
# models/ under the scratch root. That root honours AOCCBS_SCRATCH_DIR, so
# mirror the same rule here rather than assuming the in-checkout default.
MODEL_LIBRARY=${AOCCBS_SCRATCH_DIR:-$AOC/scratch}/models
echo "==> agent models -> $MODEL_LIBRARY/"
mkdir -p "$MODEL_LIBRARY"
cp -v "$ROOT"/configs/aoccbs/agent_models/*.json "$MODEL_LIBRARY/"

# ---------------------------------------------------------------------------
# 3. The CBS binary
# ---------------------------------------------------------------------------
# CBSH2-RTC is C++ and ships no binary, so a fresh clone has to build it.
# Needs cmake and Boost (program_options, system, filesystem); on a cluster
# those usually come from `module load`, which is why this only reports the
# failure rather than trying to install anything.
if [ -x "$CBS/build/cbs" ] && [ $force_build -eq 0 ]; then
    echo "==> cbs binary already built ($CBS/build/cbs); --force-build to rebuild"
else
    echo "==> building cbs"
    command -v cmake >/dev/null || fail "cmake not found (module load cmake?)"
    mkdir -p "$CBS/build"
    (cd "$CBS/build" && cmake .. -DCMAKE_BUILD_TYPE=RELEASE && make -j"$(getconf _NPROCESSORS_ONLN)")
    [ -x "$CBS/build/cbs" ] || fail "build finished but $CBS/build/cbs is missing"
fi

# ---------------------------------------------------------------------------
# 4. Check
# ---------------------------------------------------------------------------
# Named individually rather than counted, so a partial copy is reported as
# the specific missing file instead of a wrong total.
echo "==> checking"
status=0
for f in "$ROOT"/configs/aoccbs/solver*.yaml; do
    [ -f "$AOC/config/solver/$(basename "$f")" ] \
        || { echo "  MISSING solver settings: $(basename "$f")" >&2; status=1; }
done
for f in "$ROOT"/configs/aoccbs/agent_models/*.json; do
    [ -f "$MODEL_LIBRARY/$(basename "$f")" ] \
        || { echo "  MISSING agent model: $(basename "$f")" >&2; status=1; }
done
[ -x "$CBS/build/cbs" ] || { echo "  MISSING cbs binary" >&2; status=1; }
[ $status -eq 0 ] || fail "setup incomplete, see above"

echo "  solver settings, agent models and cbs binary all present"
echo
echo "next: preprocessing, which is the slow part and wants to be a job."
echo "  python scripts/preprocess.py --maps <map>[,<map>...] --k 2,3,4 --radii 4,8,16"
echo "  python scripts/preprocess.py --check --maps ...   # verify without building"
