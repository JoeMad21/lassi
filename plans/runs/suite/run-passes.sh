#!/usr/bin/env bash
# Passes of plans/runs/suite/run-suite.sh in one rx job, for a window that opens after the launch (task P17.61; the
# owner, 2026-11-02: "Consider updating the background program in case you run out of usage."). It waits on the
# build host until --start, runs run-suite.sh with --deadline and the options after "--", and runs it again after
# each pass ends while at least --min-left seconds remain to the deadline (run-suite.sh refuses under 1800). Once
# started it needs no session and no workstation: a pass that ends early is followed by the next on the host.
# Pass 1 writes $LASSI_RUNS_ROOT/suite/<rx job id> (tools/suite.py pull fetches it); pass k > 1 writes
# $LASSI_RUNS_ROOT/suite/<rx job id>-p<k>. Between passes it waits for this user's furiosa-llm servers and run.sh
# to exit (Agent Rule 8: no relaunch before a full exit), then --gap seconds, so run.sh's pass names, which are by
# the minute, differ. It stops at the STOP file ($LASSI_RUNS_ROOT/rngd-sweep/STOP, tools/suite.py stop), when a
# pass ends in under --min-pass seconds (it failed fast, so it is not repeated), after --max-passes, or when too
# little time is left. Every TT program runs on ttsim, a simulator, as run-suite.sh runs it; nothing here opens a
# Wormhole card.
set -u -o pipefail

usage() {
  cat <<'EOF'
Usage: bash plans/runs/suite/run-passes.sh --start EPOCH --deadline EPOCH [options] -- [run-suite.sh options]

Waits until --start, then runs plans/runs/suite/run-suite.sh --deadline EPOCH with the options after "--", again
and again while time is left, as one rx job on the build host.

Options:
  --start EPOCH     when the first pass starts, in epoch seconds (required; a past time starts at once)
  --deadline EPOCH  the deadline every pass gets, in epoch seconds (required)
  --min-left S      start a pass only with at least S seconds to the deadline (default 1860)
  --min-pass S      a pass that ends in under S seconds is not repeated (default 600)
  --max-passes N    run at most N passes (default 8)
  --gap S           seconds between a pass's end and the next pass, after the servers exit (default 60)
  --help            print this text and exit

Environment: LASSI_RUNS_ROOT (the gate sets it); LASSI_RX_RUN_ID (the rx job id); RUN_SUITE (the script each pass
runs; default plans/runs/suite/run-suite.sh); PASSES_POLL_S (seconds between checks while waiting; default 30).
EOF
}

start=""
deadline=""
min_left=1860
min_pass=600
max_passes=8
gap=60
while [ $# -gt 0 ]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --start) start=${2:-}; shift 2 || { echo "run-passes: --start needs a value" >&2; exit 2; } ;;
    --deadline) deadline=${2:-}; shift 2 || { echo "run-passes: --deadline needs a value" >&2; exit 2; } ;;
    --min-left) min_left=${2:-}; shift 2 || { echo "run-passes: --min-left needs a value" >&2; exit 2; } ;;
    --min-pass) min_pass=${2:-}; shift 2 || { echo "run-passes: --min-pass needs a value" >&2; exit 2; } ;;
    --max-passes) max_passes=${2:-}; shift 2 || { echo "run-passes: --max-passes needs a value" >&2; exit 2; } ;;
    --gap) gap=${2:-}; shift 2 || { echo "run-passes: --gap needs a value" >&2; exit 2; } ;;
    --) shift; break ;;
    *) echo "run-passes: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
done
for name in start deadline min_left min_pass max_passes gap; do
  [[ "${!name}" =~ ^[0-9]+$ ]] || { echo "run-passes: --${name//_/-} must be a whole number (see --help)" >&2; exit 2; }
done

cd "$(dirname "$0")/../../.." || exit 2
if ! prlimit --pid $$ --core=1:1 \
  || ! python3 -I -c 'import resource, sys; sys.exit(resource.getrlimit(resource.RLIMIT_CORE) != (1, 1))'; then
  echo "run-passes: cannot hold RLIMIT_CORE at exactly 1 byte, soft and hard (prlimit, read back as (1, 1))" >&2
  exit 2
fi
runs=${LASSI_RUNS_ROOT:?LASSI_RUNS_ROOT must be set; the gate sets it}
suite=${RUN_SUITE:-plans/runs/suite/run-suite.sh}
poll=${PASSES_POLL_S:-30}
stopfile="$runs/rngd-sweep/STOP"
base=${LASSI_RX_RUN_ID:-manual-$(date +%Y%m%d-%H%M%S)}
user=$(whoami)
ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }
left() { echo $((deadline - $(date +%s))); }
busy() { pgrep -u "$user" -f '[f]uriosa-llm serve|rngd-sweep/[r]un.sh' > /dev/null; }

echo "$(ts) run-passes $base: start $(date -d "@$start" '+%Y-%m-%d %H:%M:%S %z'), deadline" \
  "$(date -d "@$deadline" '+%Y-%m-%d %H:%M:%S %z'), at most $max_passes pass(es) of $suite $*"
while [ "$(date +%s)" -lt "$start" ]; do
  [ -e "$stopfile" ] && { echo "$(ts) STOP file: no pass started"; exit 0; }
  sleep "$poll"
done
count=0
while [ "$count" -lt "$max_passes" ]; do
  [ -e "$stopfile" ] && { echo "$(ts) STOP file: no more passes"; break; }
  [ "$(left)" -ge "$min_left" ] || { echo "$(ts) $(left) s to the deadline, under $min_left: no more passes"; break; }
  count=$((count + 1))
  pass_id=$base
  [ "$count" -gt 1 ] && pass_id="$base-p$count"
  began=$(date +%s)
  echo "$(ts) pass $count as $pass_id, $(left) s to the deadline"
  LASSI_RX_RUN_ID="$pass_id" bash "$suite" --deadline "$deadline" "$@"
  rc=$?
  took=$(( $(date +%s) - began ))
  echo "$(ts) pass $count ended rc=$rc after $took s; its output: $runs/suite/$pass_id"
  [ "$took" -ge "$min_pass" ] || { echo "$(ts) pass $count ended in under $min_pass s: not repeated"; break; }
  for _ in $(seq 1 20); do busy || break; sleep 15; done
  busy && { echo "$(ts) a furiosa-llm server or run.sh still runs 5 minutes after pass $count: no more passes"; break; }
  sleep "$gap"
done
echo "$(ts) run-passes ended after $count pass(es)"
