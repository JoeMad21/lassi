#!/usr/bin/env bash
# One window's work as one big job, so it needs no session once started (owner, 2026-10-25: "launch as much as you
# can. Try bigger models like GPT-OSS especially with the modifications"): the CPU training
# (plans/runs/train-window/run.sh), the GPT-OSS downloads, and the feature matrix (plans/runs/rngd-sweep/matrix.sh over
# the models of a table, models7.tsv when MATRIX_MODELS is unset), side by side under one deadline. The gate runs one
# big job at a time, so the three share this one. Started from a clean commit on the build host as
#   PYTHONIOENCODING=utf-8 uv run tools/rx.py job start --big --slot window --name window --timeout <seconds to the
#     window's end> -- 'bash plans/runs/window/run-all.sh <deadline as epoch seconds, 15 minutes before the end>'
# Downloads: each GPT-OSS model of the table at the revision its third column names, into $HF_HOME/hub, the RNGD
# build only (the repository's original/ and metal/ folders, other formats of the same weights, are left out). Sizes
# at v2026.4, the sums of the tag's root files in https://huggingface.co/api/models/furiosa-ai/<model>/tree/v2026.4,
# read 2026-10-25: gpt-oss-20b 14.59 GB, gpt-oss-120b 68.12 GB. Each is fetched only while the scratch disk keeps the gate's 60 GB floor for big jobs plus a margin on the shared
# disk after it (80 GB free for the 20b, 140 GB for the 120b, in GB of 1e9 bytes as the gate counts); otherwise it is
# left out, and run.sh skips a model whose snapshot is absent, logging why. The 20b is fetched before the sweep starts;
# the 120b in the background, while the models before it in the table run. Every TT program runs on ttsim, a
# simulator; no Wormhole card is used. Output under $LASSI_RUNS_ROOT/window/<stamp>/ and the scripts' own. The STOP
# file of run.sh stops the matrix before its next model; the training and a download run on to the deadline or their
# own time limit.
set -u -o pipefail
deadline=${1:?usage: run-all.sh <deadline as epoch seconds>}
cd "$(dirname "$0")/../../.." || exit 2
if ! prlimit --pid $$ --core=1:1 \
  || ! python3 -I -c 'import resource, sys; sys.exit(resource.getrlimit(resource.RLIMIT_CORE) != (1, 1))'; then
  echo "run-all: cannot hold RLIMIT_CORE at exactly 1 byte, soft and hard (prlimit, read back as (1, 1))" >&2
  exit 2
fi
scratch=${LASSI_SCRATCH:?LASSI_SCRATCH must be set; the gate sets it}
runs=${LASSI_RUNS_ROOT:?LASSI_RUNS_ROOT must be set; the gate sets it}
table=${MATRIX_MODELS:-plans/runs/rngd-sweep/models7.tsv}
export HF_HOME="$scratch/.cache/huggingface"
out="$runs/window/$(date +%Y%m%d-%H%M)"
mkdir -p "$out"
ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }
free_gb() { df -B1000000000 --output=avail "$scratch" | tail -1 | tr -dc '0-9'; }  # GB of 1e9 bytes, as the gate
python="$scratch/furiosa-venv-2026.4/bin/python"
echo "$(ts) run-all: deadline $(date -d "@$deadline" '+%H:%M:%S'), table $table, output $out"
echo "$(ts) scratch disk: $(free_gb) GB free; $(df -h "$scratch" | tail -1); scratch root $(du -sh "$scratch" 2>/dev/null | cut -f1)"

echo "$(ts) training starts (plans/runs/train-window/run.sh)"
bash plans/runs/train-window/run.sh "$deadline" > "$out/train-window.log" 2>&1 &
train_pid=$!

# Download one model at a revision, the RNGD build only, with the hub online for this one process; a marker file
# beside the log records a download that finished.
fetch() {
  HF_HUB_OFFLINE=0 timeout "$3" "$python" -I -c 'import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2], ignore_patterns=["original/*", "metal/*"]))' "$1" "$2" \
    > "$out/fetch-${1##*/}.log" 2>&1
  rc=$?
  [ "$rc" = 0 ] && touch "$out/fetch-${1##*/}.done"
  return "$rc"
}
revision() { awk -F'\t' -v m="$1" '$2 == m {print $3}' "$table"; }
fetch_pid=""
if [ -n "$(revision furiosa-ai/gpt-oss-20b)" ]; then
  if [ "$(free_gb)" -ge 80 ]; then
    fetch furiosa-ai/gpt-oss-20b "$(revision furiosa-ai/gpt-oss-20b)" 2400
    rc=$?
    echo "$(ts) gpt-oss-20b download ended rc=$rc: $(tail -1 "$out/fetch-gpt-oss-20b.log")"
  else
    echo "$(ts) SKIP the gpt-oss-20b download: $(free_gb) GB free, under 80 (14.59 GB and the gate's 60 GB floor)"
  fi
fi
if [ -n "$(revision furiosa-ai/gpt-oss-120b)" ]; then
  if [ "$(free_gb)" -ge 140 ]; then
    echo "$(ts) $(free_gb) GB free: downloading gpt-oss-120b in the background"
    fetch furiosa-ai/gpt-oss-120b "$(revision furiosa-ai/gpt-oss-120b)" 7200 &
    fetch_pid=$!
  else
    echo "$(ts) SKIP the gpt-oss-120b download: $(free_gb) GB free, under 140 (68.12 GB and the gate's 60 GB floor)"
  fi
fi

MATRIX_MODELS="$table" bash plans/runs/rngd-sweep/matrix.sh "$deadline" > "$out/matrix.log" 2>&1
rc=$?
echo "$(ts) matrix ended rc=$rc"
if [ -n "$fetch_pid" ]; then
  wait "$fetch_pid"
  rc=$?
  echo "$(ts) gpt-oss-120b download ended rc=$rc"
fi
wait "$train_pid"
rc=$?
echo "$(ts) training ended rc=$rc"
echo "$(ts) scratch disk at the end: $(free_gb) GB free; $(df -h "$scratch" | tail -1)"
