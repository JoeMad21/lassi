#!/usr/bin/env bash
# Training on a Wormhole card as one big job on the build host, which needs no session once started: the remote side
# of tools/train_tt.py (task P17.38). The owner starts it with
#   uv run tools/train_tt.py launch [options]
# which runs, from a clean pushed commit,
#   uv run tools/rx.py job start --big --slot train-tt --name train-tt --timeout <deadline + 900 s> -- \
#     'LASSI_ALLOWED_ROOT=/mnt/nvme10/joseph_ufl LASSI_STOP_GIB=345 bash plans/runs/train-tt/run.sh \
#      --deadline <epoch> --spacing <s> --recipe <path>:<timeout_s> ...'
# OQ-100 (a) with (s2): training runs of tracked code on the cards, started by the owner; one start runs a sequence of
# runs, each its own process, spaced, the sequence stopping at the first failure or hang. Steps, in order:
# 1. Checks: RLIMIT_CORE held at exactly 1 byte (prlimit, read back as (1, 1)); the gate's LASSI_* variables; the STOP
#    file absent (else it refuses: an earlier stop is still in force); the memory, the scratch disk, the commit.
# 2. The venv: toolchains/tt-xla.sh, CPU only, which keeps a checked install or installs it (downloads only, no device).
#    --install-only stops here. A failed install stops the job: nothing can train.
# 3. CPU work before any card: tt-train-v0's records built from tt-metal's examples at the pin (tools/build_tt_train.py,
#    the manifest compared with the tracked one), and each recipe's prepare step (tools/prepare_train.py, the base into
#    $HF_HOME/hub); a failure is recorded and stops the sequence when that recipe's turn comes.
# 4. The sequence, in the order given: before each recipe the STOP file and the deadline (it starts only when the time
#    left covers its trainer.timeout_s and a margin; else it is skipped), and SPACING seconds after the previous run;
#    then `uv run lassi --graphics off train <recipe> --silicon` once, never retried: the trainer holds the cards' lock
#    for the run's length, polls the cards, and leaves the quarantine marker after a hang, a kill, or a crash
#    (lassi.train.silicon). The first failure or hang stops the sequence.
# Nothing here runs model output: the recipes train tracked code on upstream text (Agent Rule 6), sft only, so no
# reward runs on silicon (Agent Rule 9). Output under $LASSI_RUNS_ROOT/train-tt/<rx job id>/ (summary.tsv, one log per
# step, and each run's records under runs/<recipe>/; tools/train_tt.py pull fetches it); the train trees are
# $LASSI_RUNS_ROOT/train/tt-<stamp>-<recipe>, their checkpoints left on the host.
set -u -o pipefail

usage() {
  cat <<'EOF'
Usage: bash plans/runs/train-tt/run.sh --deadline EPOCH --recipe PATH:SECONDS [--recipe ...] [options]
       bash plans/runs/train-tt/run.sh --install-only

Trains the given recipes on a Wormhole card, one process each, in order, from the repository root of a checkout:
the tt-xla venv (toolchains/tt-xla.sh) first, then the records and the bases (CPU only), then each recipe with
`lassi --graphics off train <recipe> --silicon`, the sequence stopping at the first failure or hang (OQ-100 (s2)).
tools/train_tt.py launch starts it.

Options:
  --deadline EPOCH     when the sequence must end, in epoch seconds (required unless --install-only)
  --recipe PATH:SECS   a train recipe and its trainer.timeout_s (repeatable; run in the order given; required unless
                       --install-only); a recipe starts only while the time left covers SECS and a margin
  --spacing SECONDS    the pause between two runs (default 120)
  --install-only       install or check the tt-xla venv and stop (CPU only, no card)
  --help               print this text and exit

Environment: LASSI_SCRATCH, LASSI_RUNS_ROOT, and LASSI_TOOLCHAINS (the gate sets them); LASSI_ALLOWED_ROOT and
LASSI_STOP_GIB (the rx command line gives them, for toolchains/tt-xla.sh); LASSI_RX_RUN_ID (the rx job id, which
names the output directory $LASSI_RUNS_ROOT/train-tt/<id>).

Stop early: create $LASSI_RUNS_ROOT/train-tt/STOP (tools/train_tt.py stop). The sequence stops before its next
recipe; a run already going ends at its trainer.timeout_s or when it finishes. While the file exists this script
refuses to start; tools/train_tt.py stop --clear removes it.
EOF
}

deadline=""
spacing=120
install_only=0
recipes=()
while [ $# -gt 0 ]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --deadline) deadline=${2:-}; shift 2 || { echo "train-tt: --deadline needs a value" >&2; exit 2; } ;;
    --spacing) spacing=${2:-}; shift 2 || { echo "train-tt: --spacing needs a value" >&2; exit 2; } ;;
    --recipe) recipes+=("${2:-}"); shift 2 || { echo "train-tt: --recipe needs a value" >&2; exit 2; } ;;
    --install-only) install_only=1; shift ;;
    *) echo "train-tt: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
done
if [ "$install_only" = 0 ]; then
  [[ "$deadline" =~ ^[0-9]+$ ]] || { echo "train-tt: --deadline must be epoch seconds (see --help)" >&2; exit 2; }
  [ "${#recipes[@]}" -gt 0 ] || { echo "train-tt: at least one --recipe PATH:SECONDS is required" >&2; exit 2; }
  for entry in "${recipes[@]}"; do
    [[ "$entry" =~ ^[A-Za-z0-9._/-]+\.yaml:[0-9]+$ ]] || { echo "train-tt: bad --recipe '$entry'" >&2; exit 2; }
  done
fi
[[ "$spacing" =~ ^[0-9]+$ ]] || { echo "train-tt: --spacing must be whole seconds" >&2; exit 2; }

cd "$(dirname "$0")/../../.." || exit 2
# A core limit of exactly 1 byte, soft and hard (plans/LESSONS.md, alpha01): prlimit takes bytes, Python's getrlimit
# reads it back, and every child inherits it, the trainer's worker included.
if ! prlimit --pid $$ --core=1:1 \
  || ! python3 -I -c 'import resource, sys; sys.exit(resource.getrlimit(resource.RLIMIT_CORE) != (1, 1))'; then
  echo "train-tt: cannot hold RLIMIT_CORE at exactly 1 byte, soft and hard (prlimit, read back as (1, 1))" >&2
  exit 2
fi
scratch=${LASSI_SCRATCH:?LASSI_SCRATCH must be set; the gate sets it}
runs=${LASSI_RUNS_ROOT:?LASSI_RUNS_ROOT must be set; the gate sets it}
: "${LASSI_TOOLCHAINS:?LASSI_TOOLCHAINS must be set; the gate sets it}"
export HF_HOME="$scratch/.cache/huggingface"
id=${LASSI_RX_RUN_ID:-manual-$(date +%Y%m%d-%H%M%S)}
[[ "$id" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "train-tt: unusable job id '$id'" >&2; exit 2; }
out="$runs/train-tt/$id"
stopfile="$runs/train-tt/STOP"
stamp=$(date +%Y%m%d-%H%M)
margin=600  # seconds a run needs beyond its trainer.timeout_s: the poll, the base's load and hash, the records
mkdir -p "$out/runs"
ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }
left() { echo $((deadline - $(date +%s))); }

echo "$(ts) train-tt $id: ${#recipes[@]} recipe(s), spacing $spacing s, output $out"
[ "$install_only" = 1 ] || echo "$(ts) deadline $(date -d "@$deadline" '+%Y-%m-%d %H:%M:%S %z'), $(left) s left"
echo "$(ts) commit $(git rev-parse HEAD 2>/dev/null || echo unknown), $(git status --porcelain 2>/dev/null | wc -l) changed paths"
echo "$(ts) memory (free -g):"; free -g
echo "$(ts) scratch disk: $(df -h "$scratch" | tail -1); scratch root $(du -sh "$scratch" 2>/dev/null | cut -f1)"
if [ -e "$stopfile" ]; then
  echo "$(ts) REFUSED: $stopfile exists, so an earlier stop is still in force. Nothing was started. Clear it with"
  echo "$(ts) uv run tools/train_tt.py stop --clear (owner), then launch again."
  exit 2
fi

# 2. The venv, CPU only: kept when it checks out, installed when missing.
echo "$(ts) the tt-xla venv: toolchains/tt-xla.sh (log install.log)"
if ! bash toolchains/tt-xla.sh > "$out/install.log" 2>&1; then
  echo "$(ts) FAIL the tt-xla venv; nothing can train. Last lines:"; tail -8 "$out/install.log"; exit 1
fi
tail -2 "$out/install.log"
[ "$install_only" = 0 ] || { echo "$(ts) --install-only: done"; exit 0; }

# 3. CPU work before any card: the records (when a recipe reads the corpus) and each recipe's base.
printf 'recipe\tstep\trc\ttrain_id\tstatus\tsteps\tstarted\tended\n' > "$out/summary.tsv"
records_rc=-
if grep -qE '^ +corpus: tt-train-v0' "${recipes[@]%%:*}" 2>/dev/null; then
  tree="$scratch/train-data/tt-metal-5280a9cf"
  if timeout --kill-after=60 1800 uv run tools/build_tt_train.py fetch --dest "$tree" > "$out/records-fetch.log" 2>&1 \
    && timeout --kill-after=60 1800 uv run tools/build_tt_train.py build --tree "$tree" --manifest "$out/manifest.json" \
      > "$out/records-build.log" 2>&1 \
    && cmp -s "$out/manifest.json" assets/train/tt-train-v0/manifest.json; then
    records_rc=0
    echo "$(ts) tt-train-v0's records built; the manifest equals the tracked one"
  else
    records_rc=1
    echo "$(ts) FAIL tt-train-v0's records (records-*.log); the corpus recipes will be refused at their turn"
  fi
fi
declare -A prepared
for entry in "${recipes[@]}"; do
  recipe=${entry%%:*}
  name=$(basename "$recipe" .yaml)
  timeout --kill-after=60 3600 uv run --extra cpu tools/prepare_train.py "$recipe" > "$out/$name.prepare.log" 2>&1
  prepared[$name]=$?
  echo "$(ts) prepare $name rc=${prepared[$name]}"
  printf '%s\tprepare\t%s\t-\t-\t-\t-\t%s\n' "$name" "${prepared[$name]}" "$(ts)" >> "$out/summary.tsv"
done

# 4. The sequence: one process per recipe, spaced, stopping at the first failure or hang.
first=1
for entry in "${recipes[@]}"; do
  recipe=${entry%%:*}
  seconds=${entry##*:}
  name=$(basename "$recipe" .yaml)
  if [ -e "$stopfile" ]; then echo "$(ts) STOP file: the sequence ends before $name"; break; fi
  if [ "$(left)" -lt $((seconds + margin + (first == 1 ? 0 : spacing))) ]; then
    echo "$(ts) SKIP $name: $(left) s left, it needs $seconds s (trainer.timeout_s) + $margin s"
    printf '%s\ttrain\tskipped\t-\t-\t-\t-\t%s\n' "$name" "$(ts)" >> "$out/summary.tsv"
    continue
  fi
  if [ "${prepared[$name]}" != 0 ]; then
    echo "$(ts) FAIL $name: its prepare step failed (rc ${prepared[$name]}); the sequence stops (OQ-100 (s2))"
    break
  fi
  [ "$first" = 1 ] || { echo "$(ts) spacing: $spacing s before $name"; sleep "$spacing"; }
  first=0
  train_id="tt-$stamp-$name"
  started=$(ts)
  echo "$(ts) START $name (train id $train_id, trainer.timeout_s $seconds)"
  timeout --signal=TERM --kill-after=120 $((seconds + margin)) \
    uv run lassi --graphics off train "$recipe" --silicon --train-id "$train_id" > "$out/$name.log" 2>&1
  rc=$?
  tree_dir="$runs/train/$train_id"
  mkdir -p "$out/runs/$name"
  for file in provenance.json recipe.resolved.yaml; do
    cp "$tree_dir/$file" "$out/runs/$name/" 2>/dev/null || true
  done
  for file in "$tree_dir"/output/checkpoint-*/checkpoint.json "$tree_dir"/output/checkpoint-*/steps.jsonl \
              "$tree_dir"/output/worker/result.json "$tree_dir"/output/worker/stage; do
    [ -f "$file" ] && cp "$file" "$out/runs/$name/"
  done
  tail -60 "$tree_dir/output/worker/worker.log" > "$out/runs/$name/worker.log.tail" 2>/dev/null || true
  status=$(sed -n 's/^  "status": "\([a-z]*\)",$/\1/p' "$tree_dir/provenance.json" 2>/dev/null | head -1)
  steps=$(sed -n 's/^  "steps": \([0-9]*\),$/\1/p' "$tree_dir/provenance.json" 2>/dev/null | head -1)
  printf '%s\ttrain\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$rc" "$train_id" "${status:--}" "${steps:--}" "$started" \
    "$(ts)" >> "$out/summary.tsv"
  echo "$(ts) END $name rc=$rc status ${status:-none}; last lines:"; tail -4 "$out/$name.log"
  if [ "$rc" != 0 ]; then
    echo "$(ts) the sequence stops at its first failure or hang (OQ-100 (s2)); $name ended rc=$rc"
    ls "$scratch/locks" 2>/dev/null | grep -i quarantine && echo "$(ts) a quarantine marker is in $scratch/locks"
    break
  fi
done
echo "$(ts) records rc $records_rc; scratch root $(du -sh "$scratch" 2>/dev/null | cut -f1) at the end"
echo "$(ts) train-tt ended"
