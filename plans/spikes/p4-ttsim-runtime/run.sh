#!/usr/bin/env bash
# P4.9 spike batch: ttsim runtime facts, unpack_to_dest, and Watcher (plans/p4-ttsim.md, task P4.9; the report
# is plans/spikes/p4-ttsim-runtime.md). One invocation, started as one rx job from a clean commit, runs every
# remote step of the spike in order on the build host. It replaces the plan's "several rx run calls" with one
# big job (justified in the report). Nothing here opens a device: every tt-metal program run names ttsim through
# TT_METAL_SIMULATOR, and preflight refuses a host with a PCI device of vendor 1e52 (Tenstorrent; Agent Rule 9),
# which alpha01 does not have (bible, Environment State).
#
# Layout, under $LASSI_RUNS_ROOT/p49-ttsim-runtime/$LASSI_RX_RUN_ID:
#   report/    small text evidence, ASCII only, no upstream source and no other account's paths; pulled into
#              results/p4-ttsim-runtime;
#   fixtures/  the seeded and Watcher captures, ASCII only, mechanically sanitized of upstream source lines;
#              pulled into .rx/pulls/ and copied by hand;
#   local/     capped logs, the JIT-generated descriptor lines, source excerpts, and every path list the checks
#              produce; quotes upstream text, so pulled only into the untracked .rx/pulls/;
#   raw/       JIT caches, working directories, seeded kernel copies, the strace log; never pulled.
#
# Steps, in order. The acceptance items run first (clean runs, the probe, the seeds, Watcher, the sandbox), then
# the extras (sfplm, one unbuffered control), so a tight budget drops the extras, never an acceptance item.
# Each step has its own wall limit: spike.py measure stops the step's process group at the limit, and an inner
# `timeout -k 10` 30 s later stops it even if the helper has died. A step is skipped when the 9000 s budget
# cannot hold its limit, when a stop file says so (see below), when the run directory passes its 6 GiB cap (du
# before each step), or when the scratch root plus the run directory would pass the P4 plan's 115 GiB stop line;
# the checks and the report still run.
#   1. clean runs of the gate's example and the five Tier A examples with the bible's ttsim settings, each with
#      its own JIT cache, logs directory, working directory, and HOME in the run, under /usr/bin/time -v;
#   2. the file-access and environment probe: the gate's example under strace, the getenv shim, and no HOME (a
#      one-shot rerun with HOME if it fails fast, so a HOME dependency does not lose the probe's answer);
#   3. seeded failures: copies of the gate example's kernel, one seeded line each, found through the step's
#      working directory (tt_metal/impl/kernels/kernel.cpp:64-68 at the pin checks the cwd first); UB, gap,
#      JIT-error, and hang candidates, run line-buffered so a printed finding is not lost if the step is killed;
#      plus one unbuffered control in the extras, to answer the buffering question;
#   4. Watcher: the gate's example with TT_METAL_WATCHER=1, then the seeded hang with it (its limit sized from
#      the measured Watcher run, not the non-Watcher hang);
#   5. the sandbox: the gate's example through lassi.executors.sandbox with the pinned tree read-only, then
#      eltwise_binary (the one compute example checked in the sandbox); a one-shot HOME retry on a fast failure;
#   6. extras: eltwise_sfpu without TT_METAL_DISABLE_SFPLOADMACRO, and the unbuffered control seed.
# Every step has its own JIT cache, so no step can reuse another's kernel binary.
#
# The measuring runs (steps 1, 2, 3, 4, 6) run outside the sandbox. The seeded lines are hand-written test
# inputs, not model output, run only on ttsim's simulated cores under an unmodified upstream host program, so
# they stay within the plan's outside-the-sandbox allowance and Agent Rule 6. Each measuring run is confined by:
# `unshare -rmnipf --kill-child --mount-proc`: its own user, mount, network, IPC, and pid namespaces, so it
# reaches no host network or localhost port (TurboQuant's tt-metal may listen there, e.g. Inspector's
# localhost:50051), shares no SysV IPC or message queue with the account's other runs, and its whole process
# tree dies with the namespace when the step ends. Loopback stays down, as in the P0.16 sandbox, except in the
# one-shot fallback of step 1, where it comes up inside the run's own network namespace. Private tmpfs on /tmp,
# /var/tmp, /dev/shm, and /run (listed before they go away). Read-only recursive binds of $LASSI_TOOLCHAINS,
# the TurboQuant checkout, and the shared default JIT cache root, made read-only with mount_setattr (the call
# lassi/executors/sandbox.py makes on / in its user namespace, measured on alpha01). env -i for the whole
# environment, with LANG=C and LC_ALL=C (GCC diagnostics carry UTF-8 quotes under the gate's en_US.UTF-8 locale,
# bible Host Facts; the kernel compiler is a GCC), so jit-stage diagnostics stay ASCII, as compiles already do.
# prlimit --core=1 --fsize --as on the whole chain: no core dump (a crash stores nothing; the OQ-014 incident),
# a per-file size cap, and a generous per-process address-space backstop. Its own JIT cache, so the JIT's cache
# clearing sees only the step's own root.
#
# Checks: every path under /mnt/nvme10/joseph_ufl (Agent Rule 7); no PCI device of vendor 1e52; RLIMIT_CORE is
# exactly (1, 1) byte, so a crash stores no core with alpha01's systemd-coredump on the root filesystem (Agent
# Rule 7); the pinned installs as recorded; the scratch root under the 115 GiB stop line with room for the run;
# after the steps, whether this account changed anything under /tmp, /var/tmp, /dev/shm (counts in report, the
# path list in local), any path in the pinned trees (one find), entries in the default cache roots; du of the
# scratch root before and after. Nothing is deleted. PROJECTED, not measured: 30 to 90 minutes and under 2 GiB;
# the budget and the post-step checks keep it under 3 hours even when every step runs to its limit.
set -euo pipefail
# A core limit of exactly 1 byte, soft and hard. bash's `ulimit -c 1` sets 1024 bytes (1-KiB blocks), which the
# kernel does not treat as "skip the core", so a piped systemd-coredump would still store a root-owned core on
# the root filesystem (Agent Rule 7; OQ-014). prlimit takes bytes, so `--core=1:1` is one byte, verified below.
prlimit --pid $$ --core=1:1 || { echo "p49: cannot set a 1-byte core limit with prlimit" >&2; exit 2; }
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
: "${LASSI_TOOLCHAINS:?LASSI_TOOLCHAINS must be set; the gate sets it}"
: "${LASSI_SCRATCH:?LASSI_SCRATCH must be set; the gate sets it}"
: "${LASSI_RUNS_ROOT:?LASSI_RUNS_ROOT must be set; the gate sets it}"
: "${LASSI_RX_RUN_ID:?LASSI_RX_RUN_ID must be set; the gate sets it for rx runs and jobs}"
for root in "$LASSI_TOOLCHAINS" "$LASSI_SCRATCH" "$LASSI_RUNS_ROOT"; do
  case "$(realpath -m "$root")/" in
    /mnt/nvme10/joseph_ufl/*) ;;
    *) echo "p49: refusing to work outside /mnt/nvme10/joseph_ufl (Agent Rule 7): $root" >&2; exit 2 ;;
  esac
done
export TMPDIR="$LASSI_SCRATCH/tmp"
mkdir -p "$TMPDIR"
for name in $(compgen -e); do
  case "$name" in
    TT_METAL_* | TT_MLIR_* | TT_LOGGER_* | TT_VISIBLE_DEVICES | ARCH_NAME) unset "$name" ;;
  esac
done

T="$LASSI_TOOLCHAINS/tt-metal@5280a9cf"
S="$LASSI_TOOLCHAINS/ttsim@v1.3.4"
EX="$T/build_Release/programming_examples"
K_ADD2=tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp
FOREIGN=/mnt/nvme10/joseph_ufl/tt-metal
DEFAULT_CACHE="$HOME/.cache/tt-metal-cache"
PIN_COMMIT=5280a9cfb00998fd49667a29523d03aee905c129
EXAMPLES="add_2_integers_in_riscv loopback eltwise_binary eltwise_sfpu matmul_single_core matmul_multi_core"
SEEDS_ORDER="ub-unaligned ub-fence ub-fencei gap-zicsr gap-decode gap-ecall gap-wfi jit-error"
CLEAN_S=900
BUDGET_S=9000
PLANNED_KIB=$((2 * 1024 * 1024))
RUN_CAP_KIB=$((6 * 1024 * 1024))
STOP_KIB=$((115 * 1024 * 1024))
STEP_MAX_BYTES=$((2 * 1024 * 1024 * 1024))
FSIZE_BYTES=$((2 * 1024 * 1024 * 1024))
AS_BYTES=$((1024 * 1024 * 1024 * 1024))
run="$LASSI_RUNS_ROOT/p49-ttsim-runtime/$LASSI_RX_RUN_ID"
raw="$run/raw"
report="$run/report"
local_dir="$run/local"
marker="$run/start.marker"
# The stop files. RUN_STOP is this batch's own, inside the run directory, so it never blocks a later batch and
# an agent never has to delete it. SHARED_STOP is honored only when it is fresh (created after this batch
# started), so one left behind does not block later batches. GATE_STOP is the owner's halt (gate.py), which
# also makes the gate refuse the rx exec that would touch RUN_STOP, so it is honored too.
RUN_STOP="$run/STOP"
SHARED_STOP="$LASSI_RUNS_ROOT/p49-ttsim-runtime/STOP"
GATE_STOP="$LASSI_SCRATCH/lassi-gate/STOP"
HONOR_SHARED=0
commit="$(git -C "$repo" rev-parse HEAD)"
changed="$(git -C "$repo" status --short | wc -l)"
dirty=$([ "$changed" -eq 0 ] && echo false || echo true)
base=(PATH=/usr/sbin:/usr/bin:/sbin:/bin LANG=C LC_ALL=C TMPDIR=/tmp "TT_METAL_RUNTIME_ROOT=$T"
  "TT_METAL_SIMULATOR=$S/libttsim_wh.so" TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_INSPECTOR_RPC=0)
prefix=()
# down: the measuring runs' private network namespace keeps loopback down; up only after the fallback in step 1.
LOOPBACK=down
# Why the remaining steps are skipped (a stop file, the run cap, the stop line), once one of them holds.
HALT=""
# The trees every measuring run sees read-only: the toolchains root, the TurboQuant checkout, and the shared
# default JIT cache root (its entries are what the JIT would prune without TT_METAL_CACHE set; PHASE-NOTES P4).
ro_paths="$(realpath "$LASSI_TOOLCHAINS")"
for extra_ro in "$FOREIGN" "$DEFAULT_CACHE"; do
  if [ -d "$extra_ro" ]; then ro_paths="$ro_paths $(realpath "$extra_ro")"; fi
done
# Makes the mount at argv[1] read-only: mount_setattr(AT_FDCWD, path, AT_RECURSIVE, {attr_set=MOUNT_ATTR_RDONLY}),
# syscall 442, as READONLY_PROGRAM in lassi/executors/sandbox.py does for /; it changes no other mount flag.
RO_PROGRAM='import ctypes, sys
libc = ctypes.CDLL(None, use_errno=True)
attr = (ctypes.c_uint64 * 4)(1, 0, 0, 0)
if libc.syscall(442, -100, sys.argv[1].encode(), 0x8000, attr, 32) != 0:
    sys.exit("mount_setattr %s: errno %d" % (sys.argv[1], ctypes.get_errno()))'
# The measuring runs' wrapper, run as `... unshare -rmnipf --kill-child --mount-proc -- sh -c "$WRAP" sh
# <listing file> <loopback: up|down> <command...>`. Any failed mount stops it before the command runs. It traps
# SIGTERM with a no-op handler (not an ignore, so the program still dies of the default action) so that when the
# step is stopped at its limit the wrapper survives to list the private directories before the SIGKILL 10 s
# later. A .ready marker beside the listing says the wrapper reached the program (so a mount failure is not read
# as the program's exit status). Uses --rbind, the form the sandbox measured on alpha01, for trees with locked
# submounts.
# shellcheck disable=SC2016
WRAP='set -e
PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH
list=$1
loopback=$2
shift 2
for dir in /tmp /var/tmp /dev/shm /run; do
  if [ -d "$dir" ]; then
    mount -t tmpfs -o size=1g,mode=1777 p49-private "$dir"
  fi
done
for path in $P49_RO_PATHS; do
  [ -d "$path" ] || continue
  mount --rbind "$path" "$path"
  python3 -I -S -c "$P49_RO_PROGRAM" "$path"
done
if [ "$loopback" = up ]; then
  ip link set lo up
fi
trap ":" TERM
: > "$list.ready"
set +e
"$@"
status=$?
find /tmp /var/tmp /dev/shm /run -mindepth 1 -printf "%y %s %p\n" > "$list" 2>&1
exit "$status"'

# preflight: refuse unless the host has no Tenstorrent PCI device, the core limit is one byte, the tools, the
# pinned installs, and the examples are there, no owner or per-run halt is already set, and the run is new.
preflight() {
  local tool name want got vendor
  if command -v lspci >/dev/null && [ -n "$(lspci -d 1e52: 2>/dev/null || true)" ]; then
    echo "p49: lspci lists a PCI device of vendor 1e52 (Tenstorrent); this batch runs only without silicon"
    return 1
  fi
  for vendor in /sys/bus/pci/devices/*/vendor; do
    if [ "$(cat "$vendor" 2>/dev/null || true)" = 0x1e52 ]; then
      echo "p49: $vendor names vendor 1e52 (Tenstorrent); this batch runs only without silicon"
      return 1
    fi
  done
  if ! python3 -c 'import resource,sys; sys.exit(0 if resource.getrlimit(resource.RLIMIT_CORE)==(1,1) else 1)'; then
    echo "p49: RLIMIT_CORE is not exactly (1, 1) byte; a crash could store a core on the root filesystem"
    return 1
  fi
  for tool in strace /usr/bin/time unshare prlimit ip gcc python3 uv timeout find du sha256sum gzip stdbuf; do
    command -v "$tool" >/dev/null || { echo "p49: $tool is not on PATH"; return 1; }
  done
  if [ -e "$GATE_STOP" ]; then echo "p49: the owner's gate STOP is set ($GATE_STOP); not starting"; return 1; fi
  got="$(cut -d' ' -f2 "$T/lassi-install.txt" 2>/dev/null | head -n 1 || true)"
  [ "$got" = "$PIN_COMMIT" ] || { echo "p49: $T is not the recorded install of $PIN_COMMIT"; return 1; }
  want="$(sed -n 's/^SHA256=\([0-9a-f]*\).*/\1/p' "$repo/toolchains/ttsim.pin")"
  got="$(sha256sum "$S/libttsim_wh.so" | cut -d' ' -f1)"
  [ -n "$want" ] && [ "$want" = "$got" ] || { echo "p49: libttsim_wh.so has sha256 $got, not $want"; return 1; }
  [ -f "$S/soc_descriptor.yaml" ] || { echo "p49: $S/soc_descriptor.yaml is missing"; return 1; }
  for name in $EXAMPLES; do
    [ -x "$EX/metal_example_$name" ] || { echo "p49: $EX/metal_example_$name is missing"; return 1; }
  done
  [ -f "$T/$K_ADD2" ] || { echo "p49: $T/$K_ADD2 is missing"; return 1; }
  [ ! -e "$run" ] || { echo "p49: $run exists already"; return 1; }
  case "$(git -C "$repo" log -1 --format=%s)" in
    'rx snapshot of '*) echo "p49: dirty launch (rx snapshot); relaunch from a clean commit"; return 1 ;;
  esac
}

# stop_files: print the stop-file paths that are in force, one per line, for spike.py measure --stop-file.
stop_files() {
  echo "$RUN_STOP"
  echo "$GATE_STOP"
  if [ "$HONOR_SHARED" = 1 ]; then echo "$SHARED_STOP"; fi
}

# scratch_kib: print du -sk of the scratch root (unreadable entries skipped).
scratch_kib() {
  (du -sk "$LASSI_SCRATCH" 2>/dev/null || true) | cut -f1
}

# fits <limit_s>: succeed when the batch's budget still holds a step of <limit_s> seconds.
fits() {
  [ $((SECONDS + $1)) -le "$BUDGET_S" ]
}

# a_stop_is_set: succeed when a stop file in force exists (RUN_STOP, GATE_STOP, or a fresh SHARED_STOP).
a_stop_is_set() {
  [ -e "$RUN_STOP" ] || [ -e "$GATE_STOP" ] || { [ "$HONOR_SHARED" = 1 ] && [ -e "$SHARED_STOP" ]; }
}

# halted: succeed when the remaining steps are to be skipped: a stop file in force, the run directory over
# RUN_CAP_KIB, or the scratch root plus the run directory over the stop line. The run directory is measured
# before every step; the whole scratch root only when the last reading (SCRATCH_KIB, taken at SCRATCH_AT) is
# older than 15 minutes, to keep du load on the shared disk low. The reason is printed once and kept in HALT.
halted() {
  local run_kib used_now
  if [ -z "$HALT" ]; then
    run_kib="$( (du -sk "$run" 2>/dev/null || true) | cut -f1)"
    run_kib="${run_kib:-0}"
    if [ $((SECONDS - SCRATCH_AT)) -ge 900 ]; then
      used_now="$(scratch_kib)"
      case "$used_now" in '' | *[!0-9]*) used_now="$SCRATCH_KIB" ;; esac
      SCRATCH_KIB="$used_now"
      SCRATCH_AT=$SECONDS
    fi
    used_now="$SCRATCH_KIB"
    if a_stop_is_set; then
      HALT="a stop file is set"
    elif [ "$run_kib" -gt "$RUN_CAP_KIB" ]; then
      HALT="the run directory holds $run_kib KiB, past its cap of $RUN_CAP_KIB KiB"
    elif [ $((used_now + run_kib)) -gt "$STOP_KIB" ]; then
      HALT="the scratch root ($used_now KiB now) plus the run ($run_kib KiB) would pass 115 GiB"
    fi
    if [ -n "$HALT" ]; then echo "p49: skipping the remaining steps: $HALT" | tee -a "$report/batch.txt"; fi
  fi
  [ -n "$HALT" ]
}

# clamp <value> <min> <max>: print the value bounded to [min, max].
clamp() {
  local value="$1"
  [ "$value" -ge "$2" ] || value="$2"
  [ "$value" -le "$3" ] || value="$3"
  echo "$value"
}

# wall_of <step>: print the step's wall time in whole seconds, rounded up, or 0 when it has none.
wall_of() {
  python3 -c 'import json, math, sys
try:
    print(math.ceil(json.load(open(sys.argv[1]))["wall_s"] or 0))
except Exception:
    print(0)' "$raw/$1/result.json"
}

# rss_mib_of <step>: print the step's peak resident set (/usr/bin/time -v) in MiB, rounded up, or 0.
rss_mib_of() {
  python3 -c 'import json, math, sys
try:
    print(math.ceil((json.load(open(sys.argv[1])).get("maxrss_kib") or 0) / 1024))
except Exception:
    print(0)' "$raw/$1/result.json"
}

# step_ok <step>: succeed when the step exited 0 without a hang.
step_ok() {
  python3 -c 'import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
sys.exit(0 if r.get("rc") == 0 and not r.get("timed_out") else 1)' "$raw/$1/result.json"
}

# register <name> <kind> <example> <cache> <logs> <seed>: add the step to raw/steps.tsv with the LOOPBACK value
# as a seventh column, for the summary.
register() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" "$LOOPBACK" >>"$raw/steps.tsv"
}

# run_step <name> <kind> <example> <limit_s> <cwd> [NAME=value...] -- <program>: one measuring run, with its own
# JIT cache (<step>/cache). Knobs, set for the call: STEP_HOME (none: no HOME; default <step>/home),
# STEP_SFPLOADMACRO (on: TT_METAL_DISABLE_SFPLOADMACRO unset; default off, which sets it to 1), STEP_STDBUF (on:
# the program runs under stdbuf -oL -eL), STEP_KERNEL_PATH (on: TT_METAL_KERNEL_PATH=<cwd>), and STEP_SEED; the
# global prefix array, a command put between /usr/bin/time and env (the probe's strace); and LOOPBACK.
run_step() {
  local name="$1" kind="$2" example="$3" limit="$4" cwd="$5"
  shift 5
  local dir="$raw/$name" extra=() envs=() linebuf=() stop_args=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
    extra+=("$1")
    shift
  done
  shift
  mkdir -p "$dir/cache" "$dir/logs" "$cwd"
  envs=("${base[@]}" "TT_METAL_CACHE=$dir/cache" "TT_METAL_LOGS_PATH=$dir/logs")
  if [ "${STEP_SFPLOADMACRO:-off}" = off ]; then envs+=(TT_METAL_DISABLE_SFPLOADMACRO=1); fi
  if [ "${STEP_KERNEL_PATH:-off}" = on ]; then envs+=("TT_METAL_KERNEL_PATH=$cwd"); fi
  if [ "${STEP_HOME:-}" != none ]; then
    mkdir -p "$dir/home"
    envs+=("HOME=$dir/home")
  fi
  if [ "${STEP_STDBUF:-off}" = on ]; then linebuf=(stdbuf -oL -eL); fi
  register "$name" "$kind" "$example" "$dir/cache" "$dir/logs" "${STEP_SEED:--}"
  if halted; then return 0; fi
  if ! fits "$limit"; then
    echo "$name: skipped, the batch budget cannot hold its ${limit} s limit"
    return 0
  fi
  local stop_file
  while IFS= read -r stop_file; do stop_args+=(--stop-file "$stop_file"); done < <(stop_files)
  python3 "$here/spike.py" measure --name "$name" --out "$dir" --timeout "$limit" --cwd "$cwd" \
    --max-bytes "$STEP_MAX_BYTES" "${stop_args[@]}" -- \
    prlimit "--core=1" "--fsize=$FSIZE_BYTES" "--as=$AS_BYTES" -- \
    timeout -k 10 $((limit + 30)) \
    env LC_ALL=C P49_RO_PATHS="$ro_paths" P49_RO_PROGRAM="$RO_PROGRAM" \
    unshare -rmnipf --kill-child --mount-proc -- sh -c "$WRAP" sh "$dir/private-tmp.txt" "$LOOPBACK" \
    /usr/bin/time -v -o "$dir/time.txt" "${prefix[@]}" env -i "${envs[@]}" "${extra[@]}" "${linebuf[@]}" "$@" \
    || echo "$name: the measuring helper failed (status $?)"
}

# sandbox_step <name> <example> <wall_s> <threads> <memory_mb> [home]: one run of the example inside the P0.16
# sandbox; a sixth argument "home" gives the program HOME=<workdir> (the one-shot HOME retry).
sandbox_step() {
  local name="$1" example="$2" wall="$3" threads="$4" memory="$5" home="${6:-}" dir="$raw/$1" home_arg=()
  mkdir -p "$dir/work/cache" "$dir/work/logs"
  register "$name" sandbox "$example" "$dir/work/cache" "$dir/work/logs" -
  if halted; then return 0; fi
  if ! fits $((wall + 120)); then
    echo "$name: skipped, the batch budget cannot hold its $((wall + 120)) s limit"
    return 0
  fi
  if [ "$home" = home ]; then home_arg=(--home); fi
  (cd "$repo" && PYTHONPATH="$repo" PYTHONDONTWRITEBYTECODE=1 LC_ALL=C timeout -k 30 $((wall + 120)) \
    uv run --frozen --offline python "$here/sandbox_attempt.py" --name "$name" \
    --example "$EX/metal_example_$example" --workdir "$dir/work" --out "$dir" --wall "$wall" \
    --memory-mb "$memory" --tt-metal "$T" --ttsim "$S" --threads "$threads" "${home_arg[@]}") \
    >"$dir/helper.txt" 2>&1 \
    || echo "$name: the sandbox helper failed (status $?); raw/$name/helper.txt ends:"
  tail -n 1 "$dir/helper.txt"
}

# sandbox_memory <step>: print the sandbox memory limit in MiB for an example: twice its clean run's peak
# resident set plus 4 GiB, between 8 and 64 GiB.
sandbox_memory() {
  clamp $((2 * $(rss_mib_of "$1") + 4096)) 8192 65536
}

# seed_step <seed> <limit_s> <name> [NAME=value...]: a measuring run of the gate's example with one seeded kernel
# copy at <cwd>/<K_ADD2>. The JIT finds it because kernel.cpp:64-68 checks the cwd first; STEP_KERNEL_PATH=on
# also names it through TT_METAL_KERNEL_PATH (checked next, still ahead of the runtime root), and the compile
# commands are logged, so step_record can confirm the copy was used from the cache. STEP_KIND (default seed)
# names the step's kind; STEP_STDBUF passes through to run_step.
seed_step() {
  local seed="$1" limit="$2" name="$3" cwd
  shift 3
  cwd="$raw/$name/cwd"
  if halted; then
    register "$name" "${STEP_KIND:-seed}" add_2_integers_in_riscv - - "$seed"
    return 0
  fi
  if ! python3 "$here/spike.py" seed --seed "$seed" --src "$T/$K_ADD2" --dst "$cwd/$K_ADD2" \
    --protect "$T" --protect "$S" --protect "$FOREIGN"; then
    echo "$name: not seeded"
    return 0
  fi
  STEP_SEED="$seed" STEP_KERNEL_PATH=on run_step "$name" "${STEP_KIND:-seed}" add_2_integers_in_riscv \
    "$limit" "$cwd" TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 "$@" -- "$EX/metal_example_add_2_integers_in_riscv"
}

# excerpts: record the source lines the report cites, in local/ only (upstream text never enters report/).
excerpts() {
  local api f
  api="$(find "$T/tt_metal" -path '*/api/compute/tile_move_copy.h' -not -path '*/build*' -print -quit 2>/dev/null \
    || true)"
  echo "== logs_dir_ in tt_metal/llrt/rtoptions.cpp (the default logs directory)"
  grep -n 'logs_dir_' "$T/tt_metal/llrt/rtoptions.cpp" || true
  echo "== tt_metal/common/executor.hpp (TT_METAL_THREADCOUNT)"
  grep -n -A3 'EXECUTOR_NTHREADS' "$T/tt_metal/common/executor.hpp" || true
  echo "== kernel.cpp search order (the cwd-first check)"
  grep -n -E 'current_path|get_kernel_dir|get_system_kernel_dir|get_root_dir' \
    "$T/tt_metal/impl/kernels/kernel.cpp" || true
  echo "== files naming DISABLE_SFPLOADMACRO"
  grep -rl 'DISABLE_SFPLOADMACRO' "$T/tt_metal" 2>/dev/null | sed "s|^$T/||" | head -n 40 || true
  echo "== unpack_to_dest in the compute API ($api)"
  if [ -n "$api" ]; then
    grep -rn -E 'unpack_to_dest|UnpackToDest' "$(dirname "$api")" | sed "s|^$T/||" | head -n 30 || true
  fi
  echo "== UnpackToDestMode and the ComputeConfig defaults"
  grep -rn -E -A3 'enum class UnpackToDestMode|unpack_to_dest_mode|fp32_dest_acc_en' "$T/tt_metal/api/tt-metalium" \
    | sed "s|^$T/||" | head -n 40 || true
  echo "== the gate kernel's runtime arguments (arg 3 is an L1 address)"
  grep -n -E 'get_arg_val' "$T/$K_ADD2" || true
  for f in add_2_integers_in_riscv/add_2_integers_in_riscv.cpp loopback/loopback.cpp \
    eltwise_binary/eltwise_binary.cpp eltwise_sfpu/eltwise_sfpu.cpp; do
    echo "== the example's checks and prints: $f"
    grep -n -E 'fmt::print|log_info|std::cout|printf|TT_FATAL|TT_THROW|abs\(|isclose|tolerance|pass = false' \
      "$T/tt_metal/programming_examples/$f" | head -n 40 || true
  done
  echo "== simulator mentions in llrt.cpp, the JIT, and the debug servers"
  grep -rn -i 'simulator' "$T/tt_metal/llrt/llrt.cpp" "$T/tt_metal/jit_build" "$T/tt_metal/impl/debug" 2>/dev/null \
    | sed "s|^$T/||" | head -n 30 || true
}

# root_fs_check: print a verdict and counts for what this account changed under /tmp, /var/tmp, /dev/shm since
# the start; the path list goes to local/root-fs-changed.txt, never to report/ (it filters by account, which
# TurboQuant shares, so a path can name another project's file). A "No such file" race counts like a denial.
root_fs_check() {
  local out line status=0 denied=0 gone=0 leaked=0 failed=0
  : >"$local_dir/root-fs-changed.txt"
  out="$(LC_ALL=C find /tmp /var/tmp /dev/shm -xdev -user "$(id -un)" -newer "$marker" -printf 'changed %p\n' 2>&1)" \
    || status=$?
  while IFS= read -r line; do
    case "$line" in
      '') ;;
      'changed '*) leaked=$((leaked + 1)); printf '%s\n' "${line#changed }" >>"$local_dir/root-fs-changed.txt" ;;
      'find: '*': Permission denied') denied=$((denied + 1)) ;;
      'find: '*': No such file or directory') gone=$((gone + 1)) ;;
      *) failed=$((failed + 1)); printf '%s\n' "$line" >>"$local_dir/root-fs-changed.txt" ;;
    esac
  done <<<"$out"
  if [ "$failed" -ne 0 ] || { [ "$status" -ne 0 ] && [ "$denied" -eq 0 ] && [ "$gone" -eq 0 ]; }; then
    echo "root-fs check: did not run cleanly (find rc=$status, $failed unexpected errors; see local/)"
  elif [ "$leaked" -ne 0 ]; then
    echo "root-fs check: this account changed $leaked path(s) under /tmp, /var/tmp, /dev/shm since the start" \
      "(this batch or another of its projects; paths in local/root-fs-changed.txt); $denied unreadable, $gone raced"
  else
    echo "root-fs check: passed; this account changed nothing there since the start" \
      "($denied unreadable, $gone raced)"
  fi
}

# post_checks <du before>: write report/checks.txt: the root-fs check (counts only), the pinned trees (one find
# into local/), the default cache roots, and the scratch du. No raw /tmp path reaches report/.
post_checks() {
  local d count
  root_fs_check
  find "$T" "$S" -newer "$marker" 2>/dev/null >"$local_dir/pinned-changed.txt" || true
  count="$(wc -l <"$local_dir/pinned-changed.txt")"
  echo "pinned trees: $count path(s) changed since the start in tt-metal@5280a9cf and ttsim@v1.3.4 (list in local/)"
  echo "default cache roots (top entries changed since the start by anyone; the first is the default root under"
  echo "the shared HOME, which the JIT would prune without TT_METAL_CACHE set, PHASE-NOTES P4):"
  for d in "$DEFAULT_CACHE" /tmp/tt-metal-cache /tmp/tt_umd_listeners; do
    if [ -e "$d" ]; then
      echo "  $(basename "$d"): $(find "$d" -maxdepth 1 -newer "$marker" 2>/dev/null | wc -l) top entries changed"
    else
      echo "  $(basename "$d"): absent"
    fi
  done
  echo "scratch du -sk: before $1, after $(scratch_kib) KiB; run directory $(du -sk "$run" | cut -f1) KiB;" \
    "run cap ${RUN_CAP_KIB} KiB"
}

# A stopped runner drops the read end of the job's stdout pipe; without this trap the next echo would take
# SIGPIPE and end the batch before the checks and the summary. The stop file, not rx job kill, is the way to stop.
trap '' PIPE

preflight || exit 2
used_before="$(scratch_kib)"
case "$used_before" in '' | *[!0-9]*) echo "p49: cannot measure the scratch root"; exit 2 ;; esac
SCRATCH_KIB="$used_before"
SCRATCH_AT=$SECONDS
if [ $((used_before + PLANNED_KIB)) -gt "$STOP_KIB" ]; then
  echo "p49: $used_before KiB used; $PLANNED_KIB KiB more would pass the 115 GiB stop line; queue a backup item"
  exit 2
fi
mkdir -p "$raw" "$report" "$local_dir" "$run/fixtures"
: >"$marker"
# A SHARED_STOP is honored only if it is fresh (newer than the start marker); a stale one from a past batch is
# ignored, so it does not block this run.
if [ -e "$SHARED_STOP" ] && [ "$SHARED_STOP" -nt "$marker" ]; then HONOR_SHARED=1; fi
echo "p49: commit $commit, changed paths $changed; run $run"
echo "p49: $(hostname) $(date -Iseconds); scratch du -sk $used_before KiB; PROJECTED 30 to 90 min"
{
  echo "commit $commit (dirty=$dirty, changed paths $changed); rx id $LASSI_RX_RUN_ID"
  echo "start $(date -Iseconds); host $(hostname)"
  echo "device: ttsim v1.3.4 (libttsim_wh.so, a virtual Wormhole) on the host CPU, with tt-metal 5280a9cf;"
  echo "  every wall time and ttsim rate line is simulator wall time: exploratory, sizing only, never performance"
  echo "tt-metal: $(head -n 1 "$T/lassi-install.txt")"
  echo "ttsim: $S/libttsim_wh.so sha256 $(sha256sum "$S/libttsim_wh.so" | cut -d' ' -f1)"
  echo "tools: $(strace -V | head -n 1); $(/usr/bin/time --version 2>&1 | head -n 1); kernel $(uname -r)"
  echo "core limit (RLIMIT_CORE): 1 byte, set with prlimit --core=1:1 and verified by getrlimit; no core is stored"
  echo "base environment of every measuring run (env -i): ${base[*]}"
  echo "  plus TT_METAL_DISABLE_SFPLOADMACRO=1 (not in sfplm-on), TT_METAL_CACHE=<step>/cache,"
  echo "  TT_METAL_LOGS_PATH=<step>/logs, and HOME=<step>/home (the probe: no HOME); LC_ALL=C is the locale"
  echo "  P4.11 must use too (ASCII compiler diagnostics)"
  echo "isolation: unshare -rmnipf --kill-child --mount-proc (user, mount, net, ipc, pid namespaces); private"
  echo "  tmpfs on /tmp /var/tmp /dev/shm /run; read-only rbinds of: $ro_paths"
  echo "  prlimit per chain: --core=1 (one byte), --fsize=$FSIZE_BYTES, --as=$AS_BYTES;"
  echo "  per-step output cap ${STEP_MAX_BYTES} bytes"
  echo "limits: clean ${CLEAN_S}s each; batch budget ${BUDGET_S}s; run directory cap ${RUN_CAP_KIB} KiB"
  echo "stop files: per-run $RUN_STOP; shared $SHARED_STOP (only if fresh); owner $GATE_STOP; rx job kill is not used"
} >"$report/batch.txt"
excerpts >"$local_dir/source-excerpts.txt" 2>&1
gcc -shared -fPIC -O2 -Wall -o "$raw/getenv_log.so" "$here/getenv_log.c" -ldl >"$raw/getenv_log.build.txt" 2>&1 \
  || echo "p49: the getenv shim did not build; see raw/getenv_log.build.txt"
# The sandbox steps run under uv; a fresh slot builds its virtual environment (in the slot, with the uv cache on
# scratch) here, outside their wall limits and offline, so a missing locked package fails here and never fetches.
(cd "$repo" && PYTHONPATH="$repo" PYTHONDONTWRITEBYTECODE=1 timeout -k 30 900 uv run --frozen --offline python \
  -c 'import lassi.executors.sandbox') >"$raw/uv-warmup.txt" 2>&1 \
  || echo "p49: uv could not import lassi.executors.sandbox offline; see raw/uv-warmup.txt"

# 1. Clean runs; the gate's example first, and once more with loopback up in its namespace when it fails fast.
for name in $EXAMPLES; do
  run_step "clean-$name" clean "$name" "$CLEAN_S" "$raw/clean-$name/cwd" -- "$EX/metal_example_$name"
  if [ "$name" = add_2_integers_in_riscv ] && ! step_ok "clean-$name" \
    && python3 "$here/spike.py" quiet --dir "$raw/clean-$name" >/dev/null 2>&1; then
    LOOPBACK=up
    run_step clean-add2-lo-up clean-lo-up "$name" "$CLEAN_S" "$raw/clean-add2-lo-up/cwd" -- \
      "$EX/metal_example_$name"
    if step_ok clean-add2-lo-up; then
      echo "p49: the gate's example passes only with loopback up (inside its own namespace); later runs bring it up" \
        | tee -a "$report/batch.txt"
    else
      LOOPBACK=down
    fi
  fi
done
add2_step=clean-add_2_integers_in_riscv
if [ "$LOOPBACK" = up ]; then add2_step=clean-add2-lo-up; fi
w_add2="$(wall_of "$add2_step")"
[ "$w_add2" -gt 0 ] || w_add2=120
seed_s="$(clamp $((3 * w_add2 + 60)) 120 600)"
hang_s="$(clamp $((2 * w_add2 + 60)) 120 600)"
probe_s="$(clamp $((8 * w_add2 + 120)) 300 1800)"
watch_s="$(clamp $((6 * w_add2 + 60)) 180 1200)"
echo "limits: seed ${seed_s}s, hang ${hang_s}s, probe ${probe_s}s, watcher ${watch_s}s" \
  "(from the gate example's ${w_add2}s of simulator wall time)" | tee -a "$report/batch.txt"

# 2. File-access and environment probe (strace by absolute path: the wrapper's PATH is the system directories).
# One HOME retry if it fails fast, so a HOME dependency does not lose the probe's answer; it has its own trace.
# strace's first child is a shell that always exits normally: when strace's first child dies of a signal,
# strace sets its own RLIMIT_CORE to 0 and re-raises it (strace.c terminate()), and a limit of 0 does not stop a
# piped core on alpha01 (P0 probe G1), so an aborting probe would otherwise store a core of strace.
# probe_prefix <step>: set the prefix array for a probe step, tracing into raw/<step>/strace.txt.gz.
probe_prefix() {
  # shellcheck disable=SC2016
  prefix=("$(command -v strace)" -f -qq -s 512 -e trace=%file,%network,execve -e signal=none
    -o "|gzip -1 >$raw/$1/strace.txt.gz" -- /bin/sh -c '"$@"; exit "$?"' sh)
}
probe_prefix probe-add2
STEP_HOME=none run_step probe-add2 probe add_2_integers_in_riscv "$probe_s" "$raw/probe-add2/cwd" \
  "LD_PRELOAD=$raw/getenv_log.so" "LASSI_GETENV_LOG=$raw/probe-add2/getenv.log" -- \
  "$EX/metal_example_add_2_integers_in_riscv"
if ! step_ok probe-add2 && python3 "$here/spike.py" quiet --dir "$raw/probe-add2" >/dev/null 2>&1; then
  probe_prefix probe-add2-home
  run_step probe-add2-home probe add_2_integers_in_riscv "$probe_s" "$raw/probe-add2-home/cwd" \
    "LD_PRELOAD=$raw/getenv_log.so" "LASSI_GETENV_LOG=$raw/probe-add2-home/getenv.log" -- \
    "$EX/metal_example_add_2_integers_in_riscv"
fi
prefix=()
for probe in probe-add2 probe-add2-home; do
  suffix="${probe#probe-add2}"
  if [ -e "$raw/$probe/result.json" ]; then
    python3 "$here/spike.py" strace --log "$raw/$probe/strace.txt.gz" --out "$report/strace-summary$suffix.txt" \
      --root "run=$run" --root "tt-metal=$T" --root "ttsim=$S" --root "scratch-other=$LASSI_SCRATCH" \
      || echo "p49: the strace summary of $probe failed"
    python3 "$here/spike.py" getenv --log "$raw/$probe/getenv.log" --out "$report/getenv$suffix.txt" \
      || echo "p49: the getenv summary of $probe failed"
  fi
done

# 3. Seeded failures (acceptance: the UB, gap, JIT-error, and hang captures), line-buffered so a printed finding
# is not lost if the step is killed. The unbuffered control seed runs in the extras.
for seed in $SEEDS_ORDER; do
  STEP_STDBUF=on seed_step "$seed" "$seed_s" "seed-$seed"
done
STEP_STDBUF=on seed_step hang "$hang_s" seed-hang

# 4. Watcher (acceptance: bible question 4). The seeded hang's Watcher limit comes from the measured Watcher run.
run_step watcher-add2 watcher add_2_integers_in_riscv "$watch_s" "$raw/watcher-add2/cwd" TT_METAL_WATCHER=1 -- \
  "$EX/metal_example_add_2_integers_in_riscv"
wh_s="$(clamp $(( $(wall_of watcher-add2) + w_add2 + 60 )) "$hang_s" 1200)"
STEP_KIND=watcher STEP_STDBUF=on seed_step hang "$wh_s" watcher-hang TT_METAL_WATCHER=1

# 5. Sandbox attempts (acceptance), walls capped at 900 s so they cannot starve the extras; a one-shot HOME
# retry on a fast failure, as the probe has.
threads=0
add2_memory="$(sandbox_memory "$add2_step")"
sandbox_step sandbox-add2 add_2_integers_in_riscv "$(clamp $((10 * w_add2)) 60 900)" 0 "$add2_memory"
if ! step_ok sandbox-add2 && python3 "$here/spike.py" quiet --dir "$raw/sandbox-add2" >/dev/null 2>&1; then
  sandbox_step sandbox-add2-home add_2_integers_in_riscv "$(clamp $((10 * w_add2)) 60 900)" 0 "$add2_memory" home
fi
if ! step_ok sandbox-add2; then
  threads=16
  sandbox_step sandbox-add2-threads16 add_2_integers_in_riscv "$(clamp $((10 * w_add2)) 60 900)" 16 "$add2_memory"
fi
w_eb="$(wall_of clean-eltwise_binary)"
sandbox_step sandbox-eltwise_binary eltwise_binary "$(clamp $((10 * w_eb)) 60 900)" "$threads" \
  "$(sandbox_memory clean-eltwise_binary)"

# 6. Extras (not acceptance): SFPLOADMACRO left on, and one unbuffered control seed for the buffering question.
w_sfpu="$(wall_of clean-eltwise_sfpu)"
STEP_SFPLOADMACRO=on run_step sfplm-on-eltwise_sfpu sfplm eltwise_sfpu "$(clamp $((4 * w_sfpu + 60)) 180 1200)" \
  "$raw/sfplm-on-eltwise_sfpu/cwd" -- "$EX/metal_example_eltwise_sfpu"
seed_step ub-unaligned "$seed_s" seed-ub-unaligned-unbuffered

# The summary runs before the checks (A57): if the gate's timeout fired during the checks, the fixtures and the
# tables would still exist. asciify_tree runs again inside summarize after this, so checks.txt is covered too.
python3 "$here/spike.py" summarize --run "$run" --commit "$commit" --rx-id "$LASSI_RX_RUN_ID" --dirty "$dirty" \
  || echo "p49: the summary failed"
post_checks "$used_before" >"$report/checks.txt" 2>&1
# checks.txt was written after summarize, so bring it (and anything else new) back to ASCII.
python3 "$here/spike.py" asciify --run "$run" || echo "p49: the ascii pass failed"
echo "== checks (report/checks.txt)"
head -n 14 "$report/checks.txt"
echo "p49: done in $SECONDS s; pull: rx pull --path lassi-runs/p49-ttsim-runtime/$LASSI_RX_RUN_ID/report"
