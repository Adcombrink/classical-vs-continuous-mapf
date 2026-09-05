#!/usr/bin/env bash
# Run CBSH2-RTC over every scenario of one map, at each of the given agent
# counts, and aggregate every run into one CSV under results/cbs/.
#
#   usage: scripts/run_cbs.sh [-t SECONDS] [-o DIR] MAP SCEN_DIR AGENTS...
#
#     MAP       the .map file
#     SCEN_DIR  folder of .scen files for that map
#     AGENTS    one or more agent counts
#
# CBS takes the first N rows of a .scen file as its N agents, which is how the
# AOC-CBS problems were built, so `-k N` on the same file reproduces them.
#
# The solver writes one CSV row per run and records the scenario but not the
# agent count, so it cannot tell two agent counts apart in one file. The
# aggregate here prepends the map, scenario, agent count and time budget, which
# makes one file per map sufficient and each row self-describing.
#
# A row already in the aggregate is not re-run, so an interrupted sweep resumes
# by being started again.

set -euo pipefail

usage() {
    echo "usage: $0 [-t SECONDS] [-o DIR] MAP SCEN_DIR AGENTS..." >&2
    exit 2
}

# The budget the AOC-CBS batches were run under, so both sides of an instance
# get the same time.
timeout_s=30
out_dir=results/cbs
cbs=${CBS:-external/CBSH2-RTC/build/cbs}

while getopts ':t:o:' opt; do
    case $opt in
        t) timeout_s=$OPTARG ;;
        o) out_dir=$OPTARG ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

[ $# -ge 3 ] || usage
map_file=$1
scen_dir=$2
shift 2
agent_counts=("$@")

[ -f "$map_file" ] || { echo "no map file at $map_file" >&2; exit 1; }
[ -d "$scen_dir" ] || { echo "no scenario folder at $scen_dir" >&2; exit 1; }
[ -x "$cbs" ] || { echo "no cbs binary at $cbs; build it first" >&2; exit 1; }

scens=("$scen_dir"/*.scen)
[ -e "${scens[0]}" ] || { echo "no .scen file in $scen_dir" >&2; exit 1; }

map_name=$(basename "$map_file" .map)
mkdir -p "$out_dir"
aggregate=$out_dir/$map_name.csv

# The solver appends to its -o file and writes a header only when the file is
# absent, so each run gets a fresh one of its own and is read back immediately.
row_file=$(mktemp)
trap 'rm -f "$row_file"' EXIT

for k in "${agent_counts[@]}"; do
    for scen in "${scens[@]}"; do
        scen_name=$(basename "$scen" .scen)
        key="$map_name,$scen_name,$k,"
        if [ -f "$aggregate" ] && grep -qF "$key" "$aggregate"; then
            echo "have $scen_name k=$k" >&2
            continue
        fi

        rm -f "$row_file"
        "$cbs" -m "$map_file" -a "$scen" -k "$k" \
               -t "$timeout_s" -o "$row_file" >/dev/null
        [ -s "$row_file" ] || {
            echo "cbs wrote nothing for $scen_name k=$k" >&2
            exit 1
        }

        if [ ! -f "$aggregate" ]; then
            echo "map,scen_file,num_agents,timeout_s,$(head -1 "$row_file")" \
                > "$aggregate"
        fi
        row=$(tail -n +2 "$row_file")
        echo "$map_name,$scen_name,$k,$timeout_s,$row" >> "$aggregate"

        # Columns 1 and 6 of the solver's row: runtime and solution cost.
        echo "$scen_name k=$k -> $(echo "$row" | cut -d, -f6)" \
             "in $(echo "$row" | cut -d, -f1)s" >&2
    done
done

echo "aggregate: $aggregate ($(($(wc -l < "$aggregate") - 1)) rows)" >&2
