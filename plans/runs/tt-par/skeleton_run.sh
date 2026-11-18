#!/usr/bin/env bash
# Task P17.71's run of the v1 skeletons on ttsim (plans/p17-portable.md, P17.71). tt-kh-skeleton-v1 is a whole
# placeholder program: its host.cpp, which tt-host-skeleton-v1 holds byte for byte, splits the tiles of one input over
# the compute grid with split_work_to_cores, and its reader, compute, and writer kernels copy each core's tiles to the
# output. This script compiles the host program with plans/runs/night/skeleton_build.sh, puts the three kernels at the
# paths its CreateKernel calls name, writes a bf16 input of N tiles for each N given (default 1 and 200: one tile, and
# more tiles than the grid has cores, so some cores take one tile more than others), runs the program on ttsim v1.3.4
# with the ttsim executor's environment (TMPDIR and TT_METAL_THREADCOUNT as plans/spikes/ttsim-cores/run.sh sets
# them), plain and with the Watcher, and prints whether the output equals the input and, from the Watcher's last
# complete dump, how many worker cores ran a kernel (plans/spikes/ttsim-cores.md). The skeleton is tracked code, not a
# model's, so it runs outside the sandbox (Agent Rule 6); every file it writes lies under $LASSI_SCRATCH, and uv's cache
# under the gate's cache root (Agent Rule 7).
# ttsim is a simulator: nothing here is a time (Agent Rule 2). For alpha01, from the repository root:
#   uv run tools/rx.py run -- 'bash plans/runs/tt-par/skeleton_run.sh [tiles ...]'
prlimit --pid $$ --core=1:1
set -u -o pipefail
limit=$(prlimit --pid $$ --core --output SOFT,HARD --noheadings | tr -s ' ' | sed 's/^ //')
[ "$limit" = "1 1" ] || { echo "skeleton_run: the core limit reads '$limit', not '1 1'; stopping" >&2; exit 2; }

repo=$PWD
toolchains=${LASSI_TOOLCHAINS:?LASSI_TOOLCHAINS is not set; run through the remote runner}
tree="$toolchains/$(source "$repo/toolchains/tt-metal.pin" && echo "$PREFIX_NAME")"
library="$toolchains/$(source "$repo/toolchains/ttsim.pin" && echo "$PREFIX_NAME/$LIBRARY")"
skeleton="$repo/assets/skeletons/tt-kh-skeleton-v1"
counts=("$@")
[ ${#counts[@]} -eq 0 ] && counts=(1 200)

build_log=$(bash "$repo/plans/runs/night/skeleton_build.sh" assets/skeletons/tt-kh-skeleton-v1)
status=$?
printf '%s\n' "$build_log" | grep -v "^command:"
[ "$status" -eq 0 ] || { echo "skeleton_run: the compile failed; nothing ran"; exit 1; }
build=$(printf '%s\n' "$build_log" | sed -n 's/^build directory: //p')
[ -n "$build" ] && [ -d "$build" ] || { echo "skeleton_run: no build directory; nothing ran"; exit 1; }

for tiles in "${counts[@]}"; do
  [[ "$tiles" =~ ^[1-9][0-9]*$ ]] || { echo "== $tiles: not a tile count; skipped"; continue; }
  for mode in plain watcher; do
    work="$build/run-$tiles-$mode"
    [ -e "$work" ] && { echo "== $tiles tile(s), $mode: run folder exists (a repeated count); skipped"; continue; }
    mkdir -p "$work/@ttsim/cache" "$work/@ttsim/logs" "$work/tmp"
    cp "$build/main" "$work/main"
    cp -r "$skeleton/kernels" "$work/kernels"
    # The input: tiles * 1024 bf16 values, each a small whole number (exact in bf16), in one dimension.
    uv run --project "$repo" python -c "
import struct, sys
from lassi.harness.lassi_io import write_array
n = int(sys.argv[2]) * 1024
bits = b''.join(struct.pack('<f', float(i % 251 - 125))[2:] for i in range(n))
write_array(sys.argv[1], 'in0', 'bf16', [n], bits)
" "$work/in0.lassiio" "$tiles"
    environment=(PATH=/usr/sbin:/usr/bin:/sbin:/bin TMPDIR="$work/tmp" LANG=C LC_ALL=C
                 TT_METAL_SIMULATOR="$library" TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_DISABLE_SFPLOADMACRO=1
                 TT_METAL_RUNTIME_ROOT="$tree" TT_METAL_CACHE="$work/@ttsim/cache"
                 TT_METAL_LOGS_PATH="$work/@ttsim/logs" TT_METAL_INSPECTOR_RPC=0 TT_METAL_THREADCOUNT=8)
    [ "$mode" = watcher ] && environment+=(TT_METAL_WATCHER=1)
    (cd "$work" && env -i "${environment[@]}" timeout 600 ./main in0.lassiio out0.lassiio) \
      > "$work/stdout.txt" 2> "$work/stderr.txt"
    status=$?
    same=$(uv run --project "$repo" python -c "
import sys
from lassi.harness.lassi_io import read_array
a, b = read_array(sys.argv[1]), read_array(sys.argv[2])
print('yes' if (a.dtype, a.shape, a.data) == (b.dtype, b.shape, b.data) else 'no')
" "$work/in0.lassiio" "$work/out0.lassiio" 2>&1 | tail -1)
    echo "== $tiles tile(s), $mode: exit $status; output equals input: $same"
    if [ "$mode" = watcher ]; then
      wlog="$work/@ttsim/logs/generated/watcher/watcher.log"
      last=$( [ -f "$wlog" ] && grep "^Dump #.*completed" "$wlog" | tail -1 | sed 's/^Dump #\([0-9]*\).*/\1/')
      if [ -z "$last" ]; then
        echo "watcher: no complete dump"
      else
        body=$(awk -v n="$last" '$0 ~ "^Dump #"n" at" {on=1; next} $0 ~ "^Dump #"n" completed" {on=0} on' "$wlog")
        ids=$(printf '%s\n' "$body" | grep " worker core(" | grep -o 'k_ids:.*' | sed 's/^k_ids://' | tr -d ' ')
        echo "watcher: dump #$last: $(printf '%s\n' "$ids" | grep -c '[1-9]') worker core(s) with a nonzero kernel id" \
          "of $(printf '%s\n' "$body" | grep -c ' worker core(')"
      fi
    fi
    [ "$status" -eq 0 ] || tail -5 "$work/stderr.txt" | cut -c1-300
  done
done
echo "done; output under $build"
