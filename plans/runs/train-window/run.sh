#!/usr/bin/env bash
# Train the four tt-train-v0 recipes of projects/train on alpha01's CPU in an owner's window (owner, 2026-10-23:
# "Just train if possible during our testing window"; OQ-076: CPU training on alpha01 in the windows only, never the
# RTX 3080). Started from a commit on the build host, as a big job, as
#   PYTHONIOENCODING=utf-8 uv run tools/rx.py job start --big --name train-window --timeout 14400 -- \
#     'bash plans/runs/train-window/run.sh <deadline as epoch seconds>'
# Steps: hold the core limit at exactly 1 byte (read back as (1, 1)); log the memory (`free -g`; the recipes' memory
# arithmetic rests on the gate's host record of 2026-10-23, mem_avail_gb 2945.3), the scratch disk's free space, and
# the scratch root's size; fetch tt-metal's examples at the pin and build the records into $LASSI_SCRATCH/train-data (the
# manifest goes to the output directory and must equal the tracked one, so the slot stays as committed), each under
# the time left; then, one recipe at a time, in the order below, run the prepare step (downloads the pinned Hub base
# into $HF_HOME) and `lassi train`, each under the time left. A recipe starts only when the time left covers its
# prepare and training estimates, and its training starts only when the time left after the prepare step still
# covers the training estimate (each the PROJECTED upper estimate of the recipe's comments, plus a margin), so the
# window ends with no training running. Torch's threads are capped (TRAIN_THREADS, 64 when unset) so a sweep beside it
# keeps its CPUs. No model inference and no device: every recipe trains on the CPU. Output under
# $LASSI_RUNS_ROOT/train-window/<stamp>/ (one log per step); the train trees are
# $LASSI_RUNS_ROOT/train/train-<stamp>-<name> (lassi train's <runs root>/train/<train id>).
set -u -o pipefail
deadline=${1:?usage: run.sh <deadline as epoch seconds>}
cd "$(dirname "$0")/../../.." || exit 2
# A core limit of exactly 1 byte, soft and hard (plans/LESSONS.md, alpha01), as plans/runs/p17-gate-a/run.sh sets it:
# bash's `ulimit -c 1` would set 1024 bytes, which a piped systemd-coredump still stores. prlimit takes bytes, and
# Python's getrlimit reads it back; every child inherits it.
if ! prlimit --pid $$ --core=1:1 \
  || ! python3 -I -c 'import resource, sys; sys.exit(resource.getrlimit(resource.RLIMIT_CORE) != (1, 1))'; then
  echo "train-window: cannot hold RLIMIT_CORE at exactly 1 byte, soft and hard (prlimit, read back as (1, 1)); a" \
    "crash could store a core on the root filesystem (Agent Rule 7)" >&2
  exit 2
fi
scratch=${LASSI_SCRATCH:?LASSI_SCRATCH must be set; the gate sets it}
runs=${LASSI_RUNS_ROOT:?LASSI_RUNS_ROOT must be set; the gate sets it}
threads=${TRAIN_THREADS:-64}
export HF_HOME="$scratch/.cache/huggingface" OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
stamp=$(date +%Y%m%d-%H%M)
out="$runs/train-window/$stamp"
mkdir -p "$out"
ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }
left() { echo $((deadline - $(date +%s))); }
# Stop, naming the step, when the deadline has passed before it.
before() { if [ "$(left)" -le 0 ]; then echo "$(ts) STOP before $1: the deadline has passed"; exit 1; fi; }

echo "$(ts) train-window $stamp: deadline $(date -d "@$deadline" '+%H:%M:%S'), $(left) s left, $threads threads"
echo "$(ts) memory (free -g):"; free -g
echo "$(ts) scratch disk: $(df -h "$scratch" | tail -1)"
echo "$(ts) scratch root: $(du -sh "$scratch" 2>/dev/null | cut -f1) in $scratch"
tree="$scratch/train-data/tt-metal-5280a9cf"
before fetch
if ! timeout --kill-after=60 "$(left)" uv run tools/build_tt_train.py fetch --dest "$tree" > "$out/fetch.log" 2>&1; then
  echo "$(ts) FAIL fetch; last lines:"; tail -5 "$out/fetch.log"; exit 1
fi
before build
if ! timeout --kill-after=60 "$(left)" uv run tools/build_tt_train.py build --tree "$tree" \
    --manifest "$out/manifest.json" > "$out/build.log" 2>&1; then
  echo "$(ts) FAIL build; last lines:"; tail -5 "$out/build.log"; exit 1
fi
if ! cmp -s "$out/manifest.json" assets/train/tt-train-v0/manifest.json; then
  echo "$(ts) FAIL the built manifest differs from the tracked one"; exit 1
fi
echo "$(ts) records built; the manifest equals the tracked one"

# name:prepare seconds:train seconds, PROJECTED. Prepare: the base's download (1.52 GB for Qwen3-0.6B, 4.08 GB for
# Qwen3-1.7B, about 11 MB of tokenizer for the scratch model) with a margin. Train: the recipe's upper estimate at the
# lower assumed rate (15, 40, 90, and 81 minutes) plus 5 minutes to load and save.
for entry in tt-full-qwen3-0.6b:600:1200 tt-lora-qwen3-1.7b:900:2700 tt-scratch-qwen3-77m:120:5700 \
             tt-prompt-tuning-qwen3-1.7b:900:5160; do
  name=${entry%%:*}
  rest=${entry#*:}
  prepare_need=${rest%%:*}
  train_need=${rest##*:}
  if [ "$(left)" -lt $((prepare_need + train_need)) ]; then
    echo "$(ts) SKIP $name: $(left) s left, needs $prepare_need + $train_need"; continue
  fi
  recipe="projects/train/$name.yaml"
  if ! timeout --kill-after=60 "$(left)" uv run --extra cpu tools/prepare_train.py "$recipe" \
      > "$out/$name.prepare.log" 2>&1; then
    echo "$(ts) FAIL prepare $name; last lines:"; tail -5 "$out/$name.prepare.log"; continue
  fi
  if [ "$(left)" -lt "$train_need" ]; then
    echo "$(ts) SKIP $name after its prepare step: $(left) s left, training needs $train_need"; continue
  fi
  echo "$(ts) START $name"
  timeout --kill-after=60 "$(left)" uv run --extra cpu lassi --graphics off train "$recipe" \
    --train-id "train-$stamp-$name" > "$out/$name.log" 2>&1
  rc=$?
  echo "$(ts) END $name rc=$rc; last lines:"; tail -3 "$out/$name.log"
done
echo "$(ts) scratch root: $(du -sh "$scratch" 2>/dev/null | cut -f1) at the end"
echo "$(ts) train-window ended"
