#!/usr/bin/env bash
# P5.3 batch: the TurboQuant codec's programs at the joint pin (plans/p5-ir.md, task P5.3; report
# plans/spikes/p5-tier-b.md). One invocation, started as one rx job from a clean commit after the owner's approval
# (Agent Rule 17; OQ-050 and the report's Owner queue), does every remote step of the spike, in order:
#   1. preflight: refuses a dirty launch (an rx snapshot commit), any root outside /mnt/nvme10/joseph_ufl, a set
#      owner STOP, an existing run directory, a missing tool or pin value, a core limit that is not exactly one
#      byte, and a scratch root that the run's planned 1 GiB would push past the 115 GiB stop line;
#   2. fetch: the codec repository at CODEC_COMMIT, by its public URL, into the run directory only, with no global
#      or system git configuration (never the shared tt-metal checkout at /mnt/nvme10/joseph_ufl/tt-metal, which
#      belongs to TurboQuant's own work and which nothing here reads or writes; PHASE-NOTES P4, the collision rule);
#   3. tier_b.py prepare: the source manifest (sha256 per file), license facts, and the program table;
#   4. tier_b.py copy, only with --alpha01-copy: its own bounded step, whose failure never ends the run;
#   5. the uv warmup, then tier_b.py batch (under the slot's uv environment): a probe that the run sandbox shows no
#      device, then every host program built with the Toolchain ttmetal-host in the compile sandbox against the
#      pinned tree, and run through the ttsim executor (lassi.executors.ttsim) on N = 32 vectors, the six multi_tile
#      stages first and the baseline last; every program run is sandboxed, with TT_METAL_SIMULATOR, its own JIT
#      cache, and the pinned tree read-only (bible, Execution Backends, ttsim row);
#   6. the NumPy probes, then tier_b.py compare under a Python that has NumPy, against reference/turboquant.py;
#   7. tier_b.py summarize (stdout), then read-only host checks and the scratch du.
# Stop: touch $run/STOP (never rx job kill); the owner's gate STOP is honored too. run.sh checks both before the
# fetch, prepare, the copy comparison, the uv warmup, the batch, the NumPy probes, compare, and the scratch du; the
# first one it sees skips that step and every later one ("stop file"), while summarize and the other host checks,
# read only and bounded, still run. Inside the batch, tier_b.py checks both, and the run directory's 1 GiB cap,
# before the probe and before every build and run. The exit status is the batch's, and 5 after any stop or the cap.
# Bounds: `timeout -k K T` ends its step by T + K seconds.
#   - before the budget: the scratch du, 610;
#   - from the budget's start (SECONDS=0) to the batch step's end, 3930: the fetch (310), prepare (610), the copy
#     comparison (610, only with --alpha01-copy), and the uv warmup (630) spend the 3000 s budget; the batch step
#     starts only while 300 s of it are left, which always holds (those four take at most 2160 s), and is bounded at
#     what is left plus 900 s for one Watcher rerun, plus a 30 s kill grace: 3000 + 900 + 30;
#   - after it: the NumPy probes 270 (70 + 70 + 130), compare 610, summarize 130, and the host checks 1070 (two
#     finds of 130, two JIT cache finds of 35, the scratch du 610, the run directory du 130);
#   - an allowance of 60 for the unbounded small commands (git rev-parse, status, init, and checkout of small trees,
#     mkdir, date, the pins, a sha256sum of the 162 KB simulator library, the stop checks).
#   Worst case 610 + 3930 + 270 + 610 + 130 + 1070 + 60 = 6680 s, under the rx --timeout of 7200 s (the gate's
#   maximum for rx run; rx job's own cap is higher). Inside the batch, each build is bounded by the toolchain's 900 s
#   timeout and each run by its wall limit (900 s for the baseline, 300 s per multi_tile stage, 120 s per "asis"
#   run) plus one Watcher rerun after a hang; a step the budget cannot hold is skipped. If an outer timeout fires,
#   the sandboxed program then running outlives it by at most its wall limit plus 7 s, since the sandbox starts it
#   in its own session (lassi/toolchains/_base.py), and its own timeout ends it.
# Nothing here deletes a file it did not create (lassi removes only its own: the compile's private TMPDIR, the
# --version check's directory, a build directory's stale output, and the ttsim executor's @ttsim/watcher copy).
# PROJECTED, not measured: 10 to 35 minutes, under 0.5 GiB in the run directory, one clang++ compile or one ttsim
# run at a time (each ttsim run sets TT_METAL_THREADCOUNT=16 and runs under 16 CPUs' limits).
#
# Layout under $LASSI_RUNS_ROOT/p53-tier-b/$LASSI_RX_RUN_ID:
#   report/   ASCII text evidence (counts, hashes, identifiers, verdicts, simulator messages); pulled for review.
#   outputs/  the programs' input and output dumps (binary numbers, a few KiB each); pull only to recompare.
#   local/    capped program streams and compiler stderr, which may quote source; pulled only into .rx/pulls/.
#   raw/      build directories with their JIT caches; never pulled.
#   src/      the codec checkout at CODEC_COMMIT.
#
# Local dry run only (plans/spikes/p5-tier-b.md, Dry run): P53_BUDGET_S may lower the budget and P53_WALL_SCALE may
# shorten every wall limit; neither can raise them. Nothing on alpha01 sets them.
set -euo pipefail
# A core limit of exactly 1 byte, soft and hard (bash's `ulimit -c 1` is 1024 bytes; plans/LESSONS.md, alpha01).
prlimit --pid $$ --core=1:1 || { echo "p53: cannot set a 1-byte core limit with prlimit" >&2; exit 2; }
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
: "${LASSI_TOOLCHAINS:?LASSI_TOOLCHAINS must be set; the gate sets it}"
: "${LASSI_SCRATCH:?LASSI_SCRATCH must be set; the gate sets it}"
: "${LASSI_RUNS_ROOT:?LASSI_RUNS_ROOT must be set; the gate sets it}"
: "${LASSI_RX_RUN_ID:?LASSI_RX_RUN_ID must be set; the gate sets it for rx runs and jobs}"
for root in "$LASSI_TOOLCHAINS" "$LASSI_SCRATCH" "$LASSI_RUNS_ROOT"; do
  case "$(realpath -m "$root")/" in
    /mnt/nvme10/joseph_ufl/*) ;;
    *) echo "p53: refusing to work outside /mnt/nvme10/joseph_ufl (Agent Rule 7): $root" >&2; exit 2 ;;
  esac
done
export TMPDIR="$LASSI_SCRATCH/tmp"
export LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1 GIT_TERMINAL_PROMPT=0 GIT_CONFIG_NOSYSTEM=1
for name in $(compgen -e); do
  case "$name" in
    TT_METAL_* | TT_MLIR_* | TT_LOGGER_* | TT_VISIBLE_DEVICES | ARCH_NAME) unset "$name" ;;
  esac
done

# Optional, only with the owner's approval of that read (Agent Rule 17): --alpha01-copy DIR also compares a copy
# of the codec found on alpha01 with the pinned checkout, read only (git state with --no-optional-locks, sha256 per
# file; never a remote URL). Off by default.
copy=""
if [ "${1:-}" = --alpha01-copy ]; then
  copy="$(realpath -e "${2:-/nonexistent}" 2>/dev/null || true)"
  case "$copy/" in
    /mnt/nvme10/joseph_ufl/lassi-* | /mnt/nvme10/joseph_ufl/toolchains/* | /mnt/nvme10/joseph_ufl/tt-metal/* | \
      /mnt/nvme10/joseph_ufl/.cache/*) copy="" ;;
    /mnt/nvme10/joseph_ufl/?* | /home/joseph_ufl/?*) [ -d "$copy" ] || copy="" ;;
    *) copy="" ;;
  esac
  [ -n "$copy" ] || { echo "p53: --alpha01-copy needs an existing directory under the scratch root or home"; exit 2; }
elif [ "$#" -gt 0 ]; then
  echo "p53: usage: run.sh [--alpha01-copy DIR]"
  exit 2
fi
CODEC_URL=https://github.com/JoeMad21/TurboQuant-Tenstorrent.git
CODEC_COMMIT=81817eb09076d17b29491c6646b54c6927cac0bd
BUDGET_S=3000
if [ -n "${P53_BUDGET_S:-}" ] && [ "$P53_BUDGET_S" -lt "$BUDGET_S" ] 2>/dev/null; then BUDGET_S="$P53_BUDGET_S"; fi
WALL_SCALE="${P53_WALL_SCALE:-1}"
PLANNED_KIB=$((1024 * 1024))
STOP_KIB=$((115 * 1024 * 1024))
run="$LASSI_RUNS_ROOT/p53-tier-b/$LASSI_RX_RUN_ID"
RUN_STOP="$run/STOP"
GATE_STOP="$LASSI_SCRATCH/lassi-gate/STOP"
commit="$(git -C "$repo" rev-parse HEAD)"
changed="$(git -C "$repo" status --short | wc -l)"
dirty=$([ "$changed" -eq 0 ] && echo false || echo true)
helper=(python3 -I -B "$here/tier_b.py")
# The lassi side of the helper runs from the slot in its locked environment, offline (a missing package fails,
# never fetches), as P4.9's sandbox step did; PYTHONPATH names the slot so lassi is imported from this commit.
lassi_python=(env PYTHONPATH="$repo" uv run --frozen --offline python -B)

# pin_value PIN KEY: the value of KEY in toolchains/PIN.pin, quotes dropped; empty when absent.
pin_value() {
  { sed -n "s/^$2=//p" "$repo/toolchains/$1.pin" 2>/dev/null || true; } | head -n 1 | tr -d '"'
}
TM_VERSION="$(pin_value tt-metal VERSION)"
TM_PREFIX="$(pin_value tt-metal PREFIX_NAME)"
TS_VERSION="$(pin_value ttsim VERSION)"
TS_PREFIX="$(pin_value ttsim PREFIX_NAME)"
TS_LIBRARY="$(pin_value ttsim LIBRARY)"

preflight() {
  local tool value
  if ! python3 -c 'import resource,sys; sys.exit(0 if resource.getrlimit(resource.RLIMIT_CORE)==(1,1) else 1)'; then
    echo "p53: RLIMIT_CORE is not exactly (1, 1) byte; a crash could store a core on the root filesystem"
    return 1
  fi
  for tool in git python3 uv timeout du find sha256sum prlimit systemd-run realpath; do
    command -v "$tool" >/dev/null || { echo "p53: $tool is not on PATH"; return 1; }
  done
  for value in "$TM_VERSION" "$TM_PREFIX" "$TS_VERSION" "$TS_PREFIX" "$TS_LIBRARY"; do
    [ -n "$value" ] || { echo "p53: a value is missing from toolchains/tt-metal.pin or ttsim.pin"; return 1; }
  done
  if [ -e "$GATE_STOP" ]; then echo "p53: the owner's gate STOP is set; not starting"; return 1; fi
  [ -d "$TMPDIR" ] || { echo "p53: TMPDIR $TMPDIR is not a directory; the gate makes it"; return 1; }
  [ ! -e "$run" ] || { echo "p53: $run exists already"; return 1; }
  case "$(git -C "$repo" log -1 --format=%s)" in
    'rx snapshot of '*) echo "p53: dirty launch (rx snapshot); relaunch from a clean commit"; return 1 ;;
  esac
  [ "$dirty" = false ] || { echo "p53: the slot has $changed changed paths; relaunch from a clean commit"; return 1; }
  case "$WALL_SCALE" in 1 | 0.[0-9]*) ;; *) echo "p53: P53_WALL_SCALE must be 1 or 0.<digits>"; return 1 ;; esac
}

# stop_check STEP: status 0 when a stop file is set. The first time, it says so (stdout and report/batch.txt):
# STEP and every later step that checks are skipped.
stopped=""
stop_check() {
  local f
  [ -z "$stopped" ] || return 0
  for f in "$RUN_STOP" "$GATE_STOP"; do
    if [ -e "$f" ]; then
      stopped="$f"
      echo "p53: stop file $f; $1 and every later step skipped" | tee -a "$run/report/batch.txt"
      return 0
    fi
  done
  return 1
}

# scratch_kib: du -sk of the scratch root (unreadable entries skipped), bounded at 600 s.
scratch_kib() {
  (timeout -k 10 600 du -sk "$LASSI_SCRATCH" 2>/dev/null || true) | cut -f1
}

# fetch: the codec at CODEC_COMMIT into $run/src: no global or system git configuration, no hooks, no fsmonitor,
# no credential helper, depth 1, then a check of HEAD.
fetch() {
  local -x GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
  local g=(git -c core.hooksPath=/dev/null -c core.fsmonitor=false -c credential.helper= -c protocol.file.allow=never
    -c advice.detachedHead=false)
  "${g[@]}" init -q "$run/src"
  timeout -k 10 300 "${g[@]}" -C "$run/src" fetch -q --depth 1 --no-tags "$CODEC_URL" "$CODEC_COMMIT"
  "${g[@]}" -C "$run/src" checkout -q --detach FETCH_HEAD
  [ "$("${g[@]}" -C "$run/src" rev-parse HEAD)" = "$CODEC_COMMIT" ]
}

# numpy_python: print the command (one word per line) of a Python that imports NumPy: the host's python3 isolated,
# then with the user site allowed, then a uv environment from the cache only (no download); nothing if none.
# Bounded at 70, 70, and 130 s.
numpy_python() {
  if timeout -k 10 60 python3 -I -c 'import numpy' 2>/dev/null; then printf '%s\n' python3 -I -B
  elif timeout -k 10 60 python3 -E -c 'import numpy' 2>/dev/null; then printf '%s\n' python3 -E -B
  elif timeout -k 10 120 uv run --offline --no-project --with numpy==2.2.6 -- python -I -c 'import numpy' \
    >/dev/null 2>&1
  then printf '%s\n' uv run --offline --no-project --with numpy==2.2.6 -- python -I -B
  fi
}

# root_fs_check: counts of what this account changed under /tmp, /var/tmp, /dev/shm since the start (paths go to
# local/ only, since the account is shared with TurboQuant's own work).
root_fs_check() {
  local list="$run/local/root-fs-changed.txt"
  timeout -k 10 120 find /tmp /var/tmp /dev/shm -xdev -user "$(id -un)" -newer "$run/start.marker" -print \
    >"$list" 2>/dev/null || true
  echo "root-fs: $(wc -l <"$list") path(s) this account changed under /tmp, /var/tmp, /dev/shm since the start" \
    "(this batch or another of the account's projects; list in local/)"
}

# post_checks USED_BEFORE: read-only host checks, each bounded; the scratch du honors the stop files.
post_checks() {
  local d
  root_fs_check
  echo "pinned trees: $(timeout -k 10 120 find "$LASSI_TOOLCHAINS/$TM_PREFIX" "$LASSI_TOOLCHAINS/$TS_PREFIX" \
    -newer "$run/start.marker" 2>/dev/null | wc -l) path(s) changed since the start" \
    "(a find stopped at 120 s counts fewer)"
  for d in "${HOME:-/nonexistent}/.cache/tt-metal-cache" /tmp/tt-metal-cache; do
    if [ -e "$d" ]; then
      echo "default JIT cache root $(basename "$(dirname "$d")")/$(basename "$d"):" \
        "$(timeout -k 5 30 find "$d" -maxdepth 1 -newer "$run/start.marker" 2>/dev/null | wc -l) top entries" \
        "changed since the start"
    else
      echo "default JIT cache root $d: absent"
    fi
  done
  if stop_check "the scratch du"; then
    echo "scratch du -sk: before $1, after not measured (stop file)"
  else
    echo "scratch du -sk: before $1, after $(scratch_kib) KiB"
  fi
  echo "run directory du -sk: $( (timeout -k 10 120 du -sk "$run" 2>/dev/null || true) | cut -f1) KiB" \
    "(cap 1048576 KiB)"
}

trap '' PIPE
preflight || exit 2
used_before="$(scratch_kib)"
case "$used_before" in '' | *[!0-9]*) echo "p53: cannot measure the scratch root"; exit 2 ;; esac
if [ $((used_before + PLANNED_KIB)) -gt "$STOP_KIB" ]; then
  echo "p53: $used_before KiB used; $PLANNED_KIB KiB more would pass the 115 GiB stop line; not starting"
  exit 2
fi
mkdir -p "$run/report" "$run/local" "$run/raw" "$run/outputs"
: >"$run/start.marker"
SECONDS=0
echo "p53: commit $commit (dirty=$dirty); run $run; scratch du -sk $used_before KiB; budget ${BUDGET_S}s"
{
  echo "commit $commit (dirty=$dirty, changed paths $changed); rx id $LASSI_RX_RUN_ID"
  echo "start $(date -Iseconds); host $(hostname); codec $CODEC_URL at $CODEC_COMMIT"
  echo "device: ttsim $TS_VERSION ($TS_LIBRARY, a virtual Wormhole) on the host CPU, with tt-metal $TM_VERSION"
  echo "  (both from toolchains/ttsim.pin and tt-metal.pin); a simulator, not silicon; every wall time is"
  echo "  simulator wall time, sizing only, never performance"
  echo "tt-metal install:" \
    "$(head -n 1 "$LASSI_TOOLCHAINS/$TM_PREFIX/lassi-install.txt" 2>/dev/null || echo missing)"
  echo "ttsim: sha256 $(sha256sum "$LASSI_TOOLCHAINS/$TS_PREFIX/$TS_LIBRARY" 2>/dev/null | cut -d' ' -f1)"
  echo "core limit (RLIMIT_CORE): 1 byte, set with prlimit --core=1:1 and verified by getrlimit"
  echo "limits: budget ${BUDGET_S}s; wall scale $WALL_SCALE; run directory cap 1 GiB;" \
    "stop files $RUN_STOP and the owner's gate STOP"
  echo "alpha01 copy compared (read only): ${copy:-no}"
  echo "tools: $(git --version); $(uv --version 2>/dev/null);" \
    "python3 $(python3 -c 'import sys; print(sys.version.split()[0])')"
} >"$run/report/batch.txt"

if ! stop_check "the fetch"; then
  if ! fetch >"$run/local/fetch.txt" 2>&1; then
    echo "p53: the codec fetch failed (local/fetch.txt); nothing was built"
    exit 1
  fi
fi
if ! stop_check "prepare"; then
  timeout -k 10 600 "${helper[@]}" prepare --src "$run/src" --report "$run/report" \
    || { echo "p53: prepare failed"; exit 1; }
fi
if [ -n "$copy" ] && ! stop_check "the alpha01-copy comparison"; then
  # Its own bounded step: a failure or a timeout is reported, and the run goes on.
  timeout -k 10 600 "${helper[@]}" copy --src "$run/src" --report "$run/report" --copy "$copy" \
    2>"$run/local/copy.stderr.txt" \
    || echo "p53: the alpha01-copy comparison ended with status $? (local/copy.stderr.txt); the run goes on"
fi
if ! stop_check "the uv warmup"; then
  # The slot's environment, built here outside the budget's steps, offline: a missing locked package fails here.
  (cd "$repo" && timeout -k 30 600 "${lassi_python[@]}" -c 'import lassi.core.runner, lassi.executors.ttsim') \
    >"$run/local/uv-warmup.txt" 2>&1 \
    || { echo "p53: the slot's uv environment did not import lassi offline (local/uv-warmup.txt)"; exit 1; }
fi
remaining=$((BUDGET_S - SECONDS))
batch_rc=0
if stop_check "the batch"; then
  batch_rc=5
elif [ "$remaining" -lt 300 ]; then
  echo "p53: ${remaining}s of the budget left after the fetch; the batch step does not start"
  batch_rc=4
else
  # The step's own bound allows one Watcher rerun (a wall limit, at most 900 s) past the budget it is given.
  (cd "$repo" && timeout -k 30 $((remaining + 900)) "${lassi_python[@]}" "$here/tier_b.py" batch --run "$run" \
    --toolchains "$LASSI_TOOLCHAINS" --runs-root "$LASSI_RUNS_ROOT" --budget-s "$((remaining - 60))" \
    --wall-scale "$WALL_SCALE" --stop-file "$RUN_STOP" --stop-file "$GATE_STOP") \
    2>"$run/local/batch.stderr.txt" || batch_rc=$?
fi
case "$batch_rc" in
  0) ;;
  5) echo "p53: the batch ended early (status 5): a stop file or the run directory cap" ;;
  *) echo "p53: the batch step ended with status $batch_rc (local/batch.stderr.txt)" ;;
esac
if ! stop_check "the NumPy probes"; then
  mapfile -t np_python < <(numpy_python)
  if [ "${#np_python[@]}" -eq 0 ]; then
    echo "compare python: none with NumPy (host python3, user site, uv cache); compare runs after the pull" \
      | tee -a "$run/report/batch.txt"
  elif ! stop_check "compare"; then
    echo "compare python: ${np_python[*]}" >>"$run/report/batch.txt"
    timeout -k 10 600 "${np_python[@]}" "$here/tier_b.py" compare --run "$run" 2>"$run/local/compare.stderr.txt" \
      || echo "p53: compare failed (local/compare.stderr.txt)"
  fi
fi
timeout -k 10 120 "${helper[@]}" summarize --run "$run" || echo "p53: summarize failed"
post_checks "$used_before" >"$run/report/checks.txt" 2>&1 || true
echo "== checks (report/checks.txt)"
head -n 8 "$run/report/checks.txt"
# A stop that run.sh saw after a finished batch still marks the run: status 5.
if [ -n "$stopped" ] && [ "$batch_rc" -eq 0 ]; then batch_rc=5; fi
echo "p53: done in ${SECONDS}s, status $batch_rc; pull: rx pull --path lassi-runs/p53-tier-b/$LASSI_RX_RUN_ID/report"
exit "$batch_rc"
