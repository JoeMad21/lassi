#!/usr/bin/env bash
# Task P17.70's spike (plans/spikes/ttsim-cores.md): ways to count the cores a TT program's kernels run on under
# ttsim, and ttsim's own grid. It reruns built tt-par-v0 references (tracked host programs and upstream kernels, no
# model-generated code, so outside the sandbox; Agent Rule 6) from the check run ttpar-check-dev-1117 on ttsim three
# ways: plain, with tt-metal's device profiler (the settings lassi.executors.tt_profiler.profiler_environment sets on
# the cards), and with the Watcher (TT_METAL_WATCHER=1). For each it prints the exit status, whether the outputs equal
# the check run's, the profiler's files and the core columns of its device log and report, and the Watcher's last
# complete dump: its core lines, and those that hold a nonzero kernel id. Every count is a count of cores, never a
# time: ttsim is a simulator, and nothing it reports is performance (Agent Rule 2). Each copy and every file a run
# writes lies under $LASSI_SCRATCH/spike-ttsim-cores/<stamp> (Agent Rule 7). For alpha01, from the repository root:
#   uv run tools/rx.py run --slot spike-cores -- 'bash plans/spikes/ttsim-cores/run.sh [item ...]'
prlimit --pid $$ --core=1:1
set -u -o pipefail

repo=$PWD
runs=${LASSI_RUNS_ROOT:?LASSI_RUNS_ROOT is not set; run through the remote runner}
scratch=${LASSI_SCRATCH:?LASSI_SCRATCH is not set; run through the remote runner}
toolchains=${LASSI_TOOLCHAINS:?LASSI_TOOLCHAINS is not set; run through the remote runner}
tree="$toolchains/$(source "$repo/toolchains/tt-metal.pin" && echo "$PREFIX_NAME")"
library="$toolchains/$(source "$repo/toolchains/ttsim.pin" && echo "$PREFIX_NAME/$LIBRARY")"
source_root="$runs/runs/ttpar-check-dev-1117/dev-cpp2tt/mock-reference/tt-par-v0/cpp-tt"
out="$scratch/spike-ttsim-cores/$(date +%Y%m%d-%H%M%S)"
items=("$@")
[ ${#items[@]} -eq 0 ] && items=(vecadd_multi_core matmul_multicore_reuse pad_multi_core)
echo "commit $(git -C "$repo" rev-parse HEAD) ($(git -C "$repo" status --porcelain | wc -l) changed paths)"
echo "tree $tree; library $library; output $out"
echo "core limit (soft hard): $(prlimit --pid $$ --core --output SOFT,HARD --noheadings | tr -s ' ')"
mkdir -p "$out"

for item in "${items[@]}"; do
  # An item is a bare name, so no argument reaches another arm's build (the check run's reference arm only).
  [[ "$item" =~ ^[A-Za-z0-9_]+$ ]] || { echo "== $item: not an item name; skipped"; continue; }
  build="$source_root/$item/run01/attempt00/build"
  [ -x "$build/main" ] || { echo "== $item: no built reference at $build"; continue; }
  for mode in plain profiler watcher; do
    work="$out/$item-$mode"
    mkdir -p "$work/@ttsim/cache" "$work/@ttsim/logs" "$work/@ttsim/profiler" "$work/tmp"
    cp "$build/main" "$work/main"
    # Every directory of the build but the earlier run's own @ttsim: the inputs and the kernels at their paths.
    for directory in "$build"/*/ "$build"/@inputs/; do
      name=$(basename "$directory")
      [ "$name" = "@ttsim" ] || [ -e "$work/$name" ] || cp -r "$build/$name" "$work/$name"
    done
    mapfile -t inputs < <(cd "$work" && ls @inputs/*.lassiio | sort)
    mapfile -t outputs < <(cd "$build" && ls ./*.lassiio | sed 's|^\./||' | sort)
    environment=(PATH=/usr/sbin:/usr/bin:/sbin:/bin TMPDIR="$work/tmp" LANG=C LC_ALL=C
                 TT_METAL_SIMULATOR="$library" TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DISABLE_SFPLOADMACRO=1
                 TT_METAL_RUNTIME_ROOT="$tree" TT_METAL_CACHE="$work/@ttsim/cache"
                 TT_METAL_LOGS_PATH="$work/@ttsim/logs" TT_METAL_INSPECTOR_RPC=0 TT_METAL_THREADCOUNT=8)
    case "$mode" in
      profiler) environment+=(TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
                              TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1
                              TT_METAL_PROFILER_DIR="$work/@ttsim/profiler") ;;
      watcher) environment+=(TT_METAL_WATCHER=1) ;;
    esac
    (cd "$work" && env -i "${environment[@]}" timeout 600 ./main "${inputs[@]}" "${outputs[@]}") \
      > "$work/stdout.txt" 2> "$work/stderr.txt"
    status=$?
    same=yes
    [ ${#outputs[@]} -eq 0 ] && same="n/a (no output named)"
    for name in "${outputs[@]}"; do cmp -s "$work/$name" "$build/$name" || same=no; done
    echo "== $item $mode: exit $status; inputs ${inputs[*]}; outputs ${outputs[*]} equal to the check run's: $same"
    if [ "$mode" = profiler ]; then
      find "$work/@ttsim/profiler" -type f | sed "s|$work/||" | head -10
      log="$work/@ttsim/profiler/.logs/profile_log_device.csv"
      if [ -f "$log" ]; then
        echo "device log header: $(head -1 "$log")"
        echo "device log columns: $(sed -n 2p "$log")"
        distinct=$(tail -n +3 "$log" | awk -F, '{print $2","$3}' | sort -u | wc -l)
        echo "distinct (core_x, core_y) in the device log: $distinct"
      fi
      report="$work/@ttsim/profiler/.logs/cpp_device_perf_report.csv"
      if [ -f "$report" ]; then
        column=$(head -1 "$report" | tr ',' '\n' | grep -n -x ' *CORE COUNT *' | cut -d: -f1)
        counts=$(tail -n +2 "$report" | cut -d, -f"${column:-1}" | tr '\n' ' ')
        if [ -n "$column" ]; then
          echo "report rows: $(($(wc -l < "$report") - 1)); CORE COUNT column $column: $counts"
        else
          echo "report rows: $(($(wc -l < "$report") - 1)); no CORE COUNT column"
        fi
      fi
      grep -i -m3 "profil" "$work/stderr.txt" "$work/stdout.txt"
    fi
    if [ "$mode" = watcher ]; then
      wlog="$work/@ttsim/logs/generated/watcher/watcher.log"
      if [ -f "$wlog" ]; then
        dumps=$(grep -c "^Dump #" "$wlog")
        last=$(grep "^Dump #.*completed" "$wlog" | tail -1 | sed 's/^Dump #\([0-9]*\).*/\1/')
        if [ -z "$last" ]; then
          echo "watcher: $dumps dump line(s); no complete dump"
        else
          body=$(awk -v n="$last" '$0 ~ "^Dump #"n" at" {on=1; next} $0 ~ "^Dump #"n" completed" {on=0} on' "$wlog")
          # Core lines by kind ("worker", "idleth", ...), then each worker line's k_ids field alone: a worker whose
          # k_ids hold a nonzero id ran a kernel; a worker line without k_ids counts as none.
          printf '%s\n' "$body" | grep "core(" | awk '{print $3}' | sort | uniq -c | sed 's/^/watcher core lines: /'
          ids=$(printf '%s\n' "$body" | grep " worker core(" | grep -o 'k_ids:.*' | sed 's/^k_ids://' | tr -d ' ')
          busy=$(printf '%s\n' "$ids" | grep -c '[1-9]')
          echo "watcher: $dumps dump line(s); last complete dump #$last: $busy worker core(s) with a nonzero kernel id"
          printf '%s\n' "$ids" | sort | uniq -c | sed 's/^/watcher worker k_ids: /'
        fi
      else
        echo "watcher: no watcher.log"
      fi
    fi
    tail -3 "$work/stderr.txt" | cut -c1-300
  done
done
echo "done; output under $out"
