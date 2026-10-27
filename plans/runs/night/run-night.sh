#!/usr/bin/env bash
# The owner's window of 2026-10-25, 20:00 to 02:00 ("We must significantly improve results"), as one big job that
# needs no session once started: the models' downloads, the generated assets, the sweep over
# projects/lassi-demo/sweep7 (cpp -> tt, 3 samples per item, cap 8; OQ-086), and the night's tables and counts.
# Started from a clean commit on the build host as
#   PYTHONIOENCODING=utf-8 uv run tools/rx.py job start --big --slot night --name night --timeout <seconds to 02:00> \
#     -- 'bash plans/runs/night/run-night.sh <deadline as epoch seconds, 01:45 EDT>'
# Models (OQ-087), the rows of plans/runs/rngd-sweep/models10.tsv: Llama-3.3-70B-Instruct-FP8-dynamic first, fetched
# before the sweep, then gpt-oss-120b, fetched in the background while Llama's recipes run. The sweep runs run.sh
# once per model, Llama until the night's midpoint and gpt-oss-120b, once its download has ended, until the
# deadline, so a quick failure on Llama's turn never reaches a snapshot still being written. Every file of each revision is
# fetched (the server checks the cached snapshot offline), and each only while the scratch disk keeps the gate's
# 60 GB floor plus a 5 GB margin after the bytes still missing. Sizes at v2026.4, the sums of every file in
# https://huggingface.co/api/models/furiosa-ai/<model>/tree/v2026.4?recursive=1, read 2026-10-25: Llama
# 73775947887 bytes, gpt-oss-120b 198607992266.
# Assets, each built on the host from the tt-metal install at the pin and checked against its tracked manifest;
# a recipe whose asset did not build as tracked is left out (SWEEP_SKIP): the full API pack
# (tools/build_tt_context.py --name tt-metalium-5280a9cf-full --limit-bytes 131072; combinations lk and all), the
# worked examples (tools/build_tt_examples.py; ex and all), and the docs index (tools/build_tt_rag.py; all).
# Every TT program runs on ttsim, a simulator; no Wormhole card is used. Output under $LASSI_RUNS_ROOT/night/<stamp>/
# and run.sh's own; tables under $LASSI_RUNS_ROOT/rngd-sweep/tables7. Stop early: the STOP file of run.sh.
set -u -o pipefail
deadline=${1:?usage: run-night.sh <deadline as epoch seconds>}
cd "$(dirname "$0")/../../.." || exit 2
if ! prlimit --pid $$ --core=1:1 \
  || ! python3 -I -c 'import resource, sys; sys.exit(resource.getrlimit(resource.RLIMIT_CORE) != (1, 1))'; then
  echo "run-night: cannot hold RLIMIT_CORE at exactly 1 byte, soft and hard (prlimit, read back as (1, 1))" >&2
  exit 2
fi
scratch=${LASSI_SCRATCH:?LASSI_SCRATCH must be set; the gate sets it}
runs=${LASSI_RUNS_ROOT:?LASSI_RUNS_ROOT must be set; the gate sets it}
toolchains=${LASSI_TOOLCHAINS:?LASSI_TOOLCHAINS must be set; the gate sets it}
# The pinned tt-metal install, toolchains/tt-metal.pin's PREFIX_NAME read in a subshell (since task P17.42 the Tracy
# build of 5280a9cf; the asset builders read only its sources, which are the same).
tree="$toolchains/$(source toolchains/tt-metal.pin && echo "$PREFIX_NAME")"
table=plans/runs/rngd-sweep/models10.tsv
export HF_HOME="$scratch/.cache/huggingface"
out="$runs/night/$(date +%Y%m%d-%H%M)"
mkdir -p "$out"
ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }
free_gb() { df -B1000000000 --output=avail "$scratch" | tail -1 | tr -dc '0-9'; }  # GB of 1e9 bytes, as the gate
python="$scratch/furiosa-venv-2026.4/bin/python"
echo "$(ts) run-night: deadline $(date -d "@$deadline" '+%H:%M:%S'), table $table, output $out"
echo "$(ts) scratch disk: $(free_gb) GB free; scratch root $(du -sh "$scratch" 2>/dev/null | cut -f1)"
[ -e "$runs/rngd-sweep/STOP" ] && echo "$(ts) WARNING: $runs/rngd-sweep/STOP exists, so run.sh will stop at once"

# Download one model at a revision, every file, with the hub online for this one process.
fetch() {
  HF_HUB_OFFLINE=0 timeout "$3" "$python" -I -c 'import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2]))' "$1" "$2" > "$out/fetch-${1##*/}.log" 2>&1
  rc=$?
  [ "$rc" = 0 ] && touch "$out/fetch-${1##*/}.done"
  return "$rc"
}
# GB a download still needs, rounded up: the revision's size in bytes ($2) less what the model's snapshots hold,
# their links followed (large files may be links into a shared blob store under $HF_HOME/hub/blobs).
missing_gb() {
  local have
  have=$(du -sbL "$HF_HOME/hub/models--${1/\//--}/snapshots" 2>/dev/null | cut -f1)
  have=${have:-0}
  [ "$have" -ge "$2" ] && { echo 0; return; }
  echo $(( ($2 - have + 999999999) / 1000000000 ))
}
need_gb() { echo $(( 60 + 5 + $(missing_gb "$1" "$2") )); }

llama=furiosa-ai/Llama-3.3-70B-Instruct-FP8-dynamic
oss=furiosa-ai/gpt-oss-120b
need=$(need_gb "$llama" 73775947887)
if [ "$(free_gb)" -ge "$need" ]; then
  fetch "$llama" v2026.4 3000
  rc=$?
  echo "$(ts) Llama download ended rc=$rc: $(tail -1 "$out/fetch-${llama##*/}.log")"
else
  echo "$(ts) SKIP the Llama download: $(free_gb) GB free, under $need (the bytes missing, the 60 GB floor, a 5 GB margin)"
fi
fetch_pid=""
need=$(need_gb "$oss" 198607992266)
if [ "$(free_gb)" -ge "$need" ]; then
  echo "$(ts) $(free_gb) GB free: downloading gpt-oss-120b in the background"
  fetch "$oss" v2026.4 7200 &
  fetch_pid=$!
else
  echo "$(ts) SKIP the gpt-oss-120b download: $(free_gb) GB free, under $need (the bytes missing, the 60 GB floor, a 5 GB margin)"
fi

skip=()
if [ -d "$tree" ] && uv run tools/build_tt_context.py --tree "$tree" --name tt-metalium-5280a9cf-full \
     --limit-bytes 131072 > "$out/context-full.log" 2>&1 && git diff --quiet HEAD -- assets/context; then
  echo "$(ts) the full API pack built; its manifest equals the tracked one"
else
  echo "$(ts) the full API pack did not build as tracked; lk and all are left out:"
  tail -5 "$out/context-full.log" 2>/dev/null; git diff --stat HEAD -- assets/context; git checkout -- assets/context
  skip+=("lk" "all")
fi
if [ -d "$tree" ] && uv run tools/build_tt_examples.py build --tree "$tree" > "$out/examples.log" 2>&1 \
   && git diff --quiet HEAD -- assets/examples && uv run tools/build_tt_examples.py check >> "$out/examples.log" 2>&1; then
  echo "$(ts) the worked examples built and checked; their manifest equals the tracked one"
else
  echo "$(ts) the worked examples did not build as tracked; ex and all are left out:"
  tail -5 "$out/examples.log" 2>/dev/null; git diff --stat HEAD -- assets/examples; git checkout -- assets/examples
  skip+=("ex" "all")
fi
if [ -d "$tree" ] && uv run tools/build_tt_rag.py build --tree "$tree" > "$out/rag.log" 2>&1 \
   && git diff --quiet HEAD -- assets/rag && uv run tools/build_tt_rag.py check --tree "$tree" >> "$out/rag.log" 2>&1; then
  echo "$(ts) the docs index built and checked; its manifest equals the tracked one"
else
  echo "$(ts) the docs index did not build as tracked; all is left out:"
  tail -5 "$out/rag.log" 2>/dev/null; git checkout -- assets/rag
  skip+=("all")
fi
pattern=""
[ "${#skip[@]}" -gt 0 ] && skip+=("lowall")  # every failed asset leaves out all, and lowall is all at low effort
[ "${#skip[@]}" -gt 0 ] && pattern="^($(IFS='|'; echo "${skip[*]}"))-"
[ -n "$pattern" ] && echo "$(ts) SWEEP_SKIP $pattern"

# One run.sh per model, each with one row of the table, so each model's share is its own part of the night.
sweep() {  # sweep <slug> <deadline>: run.sh over the table's row for <slug>, logged to $out/sweep-<slug>.log
  grep -P "^$1\t" "$table" > "$out/models-$1.tsv"
  SWEEP_SKIP="$pattern" SWEEP_MAX_SHARE=${SWEEP_MAX_SHARE:-14400} bash plans/runs/rngd-sweep/run.sh "$2" \
    projects/lassi-demo/sweep7 "$out/models-$1.tsv" > "$out/sweep-$1.log" 2>&1
  local code=$?
  echo "$(ts) sweep $1 ended rc=$code; its log: $out/sweep-$1.log"
  grep -E "START|READY|FAIL|SKIP|ended" "$out/sweep-$1.log" | tail -40
}
now=$(date +%s)
midpoint=$(( now + (deadline - now) / 2 ))
echo "$(ts) Llama until $(date -d "@$midpoint" '+%H:%M:%S'), then gpt-oss-120b until $(date -d "@$deadline" '+%H:%M:%S')"
sweep llama33-70b "$midpoint"
if [ -n "$fetch_pid" ]; then
  wait "$fetch_pid"
  rc=$?
  echo "$(ts) gpt-oss-120b download ended rc=$rc"
fi
sweep gpt-oss-120b "$deadline"

python3 plans/runs/rngd-sweep/tabulate.py "$runs" "$runs/rngd-sweep/tables7"
for stamp in $(grep -ohE "sweep [0-9]{8}-[0-9]{4}" "$out"/sweep-*.log | cut -d' ' -f2 | sort -u); do
  python3 plans/runs/window/count_features.py "$runs" "$stamp" "$out/features-$stamp.tsv"
done
python3 tools/classify_build_errors.py "$runs" > "$out/build-errors.txt" 2> "$out/build-errors.stderr"
rc=$?
echo "$(ts) classify rc=$rc"
cp "$runs/rngd-sweep/tables7/trials.tsv" "$runs/rngd-sweep/tables7/passes.tsv" "$out/" 2>/dev/null
echo "$(ts) scratch disk at the end: $(free_gb) GB free"
echo "$(ts) run-night ended"
