#!/usr/bin/env bash
# The owner's RNGD model sweep (2026-10-23, 00:00 to 06:00 EDT): Tier A, one trial per item and direction, cap 5,
# for every model in models.tsv beside this file, one model at a time on port 8123 (the owner's choice; Agent Rule
# 8). Started from a clean commit on the build host as
#   PYTHONIOENCODING=utf-8 uv run tools/rx.py job start --big --name rngd-sweep --timeout 21600 -- \
#     'bash plans/runs/rngd-sweep/run.sh <deadline as epoch seconds>'
# For each model, in order: skip it when its artifact is not in $HF_HOME or less than 30 minutes are left; poll
# `furiosa-smi status` every 60 s (each poll logged in card-polls.txt) until enough cards are free, then pick
# them (npu0 to npu7, each at 0.00 GiB; Agent Rule 8 as the owner changed it on 2026-10-22, OQ-071); serve it
# on 127.0.0.1:8123 and wait for /v1/models to list its id; run its two recipes (projects/lassi-demo/sweep/
# <slug>-maxabs.yaml and -pcc.yaml) side by side under one time limit, the time left shared among the models still
# to run (at most 50 minutes); stop the server and wait for its exit, the port to close, and its cards to read
# 0.00 GiB again (Agent Rule 8: no relaunch before a full exit). A run cut by its limit keeps every trial it
# finished. Stop: create $LASSI_RUNS_ROOT/rngd-sweep/STOP (one rx exec); the sweep stops before the next model and
# at its next check, and always stops its server on exit. Every TT program runs on ttsim, a simulator.
# Output under $LASSI_RUNS_ROOT/rngd-sweep/<stamp>/: per model the serve log, the two run logs, and summary.tsv
# (one row per finished trial); the run trees are $LASSI_RUNS_ROOT/runs/sweep-<stamp>-<slug>-<half>.
set -u
deadline=${1:?usage: run.sh <deadline as epoch seconds>}
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/../../.." || exit 2
prlimit --pid $$ --core=1:1 2>/dev/null || true
scratch=${LASSI_SCRATCH:?LASSI_SCRATCH must be set; the gate sets it}
runs=${LASSI_RUNS_ROOT:?LASSI_RUNS_ROOT must be set; the gate sets it}
export HF_HOME="$scratch/.cache/huggingface" HF_HUB_OFFLINE=1
stamp=$(date +%Y%m%d-%H%M)
out="$runs/rngd-sweep/$stamp"
stopfile="$runs/rngd-sweep/STOP"
mkdir -p "$out"
server=""

ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }
left() { echo $((deadline - $(date +%s))); }

free_cards() {
  # One npuN per line: each of npu0 to npu7 whose memory reads 0.00 GiB.
  # A row reads "| RNGD | npuN | <used>/<total> GiB (<pct>) | <utilization> |"; memory is the field holding GiB.
  furiosa-smi status 2>/dev/null | grep -a -E '\| npu[0-7] ' |
    awk -F'|' '{card=$3; gsub(/ /, "", card); used="";
                for (i = 4; i <= NF; i++) if ($i ~ /GiB/) { split($i, mem, "/"); used=mem[1]; gsub(/ /, "", used); break }
                if (used == "0.00") print card}'
}

stop_server() {
  [ -n "$server" ] || return 0
  if kill -0 "$server" 2>/dev/null; then
    kill -TERM "$server" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$server" 2>/dev/null || break; sleep 1; done
    kill -0 "$server" 2>/dev/null && { echo "$(ts) server $server still up after 60 s; KILL"; kill -KILL "$server"; }
  fi
  wait "$server" 2>/dev/null
  echo "$(ts) server $server exited"
  server=""
  for _ in $(seq 1 36); do
    ss -ltn | grep -q ':8123 ' || break
    sleep 5
  done
}
trap 'stop_server; echo "$(ts) sweep ended"' EXIT

summarize() {
  # Append one row per finished trial of run $1 to summary.tsv and print them.
  python3 - "$runs/runs/$1" "$2" <<'PY' | tee -a "$out/summary.tsv"
import glob, json, os, sys
root, slug = sys.argv[1], sys.argv[2]
for path in sorted(glob.glob(os.path.join(root, "**", "trial.json"), recursive=True)):
    with open(path) as handle:
        trial = json.load(handle)
    final = trial.get("final") or {}
    end = (final.get("end_reason") or {}).get("code", "")
    tid = "/".join(trial.get("trial_id", "").split("/")[3:])
    print("\t".join(str(v) for v in (slug, tid, final.get("stage_reached"), final.get("alignment"),
                                       final.get("corrections"), end or "-", len(trial.get("attempts") or []))))
PY
}

echo "$(ts) sweep $stamp: deadline $(date -d "@$deadline" '+%H:%M:%S'), $(left) s left, output $out"
printf 'slug\ttrial\tstage\talignment\tcorrections\tend\tattempts\n' > "$out/summary.tsv"
mapfile -t rows < <(grep -v '^#' "$here/models.tsv" | grep -v '^$')
total=${#rows[@]}
for index in "${!rows[@]}"; do
  IFS=$'\t' read -r slug model rev cards parser venv <<< "${rows[$index]}"
  remaining=$((total - index))
  if [ -e "$stopfile" ]; then echo "$(ts) STOP file: stopping before $slug"; break; fi
  if [ "$(left)" -lt 1800 ]; then echo "$(ts) under 30 min left: stopping before $slug"; break; fi
  snap=$(ls -d "$HF_HOME/hub/models--${model/\//--}/snapshots/"* 2>/dev/null | head -1)
  if [ -z "$snap" ]; then echo "$(ts) SKIP $slug: $model is not in $HF_HOME"; continue; fi
  # Poll the cards until enough of them read 0.00 GiB (other tenants done), every 60 s, logging each poll.
  while :; do
    furiosa-smi status 2>/dev/null | grep -a -E '\| npu' | sed "s/^/$(ts) /" >> "$out/card-polls.txt"
    mapfile -t free < <(free_cards)
    [ "${#free[@]}" -ge "$cards" ] && break
    if [ -e "$stopfile" ] || [ "$(left)" -lt 1800 ]; then break; fi
    echo "$(ts) WAIT $slug: ${#free[@]} card(s) at 0.00 GiB, needs $cards; polling again in 60 s"
    sleep 60
  done
  if [ "${#free[@]}" -lt "$cards" ]; then echo "$(ts) SKIP $slug: ${#free[@]} free card(s), needs $cards"; continue; fi
  if ss -ltn | grep -q ':8123 '; then echo "$(ts) ABORT: port 8123 is in use"; break; fi
  devices=$(printf '%s\n' "${free[@]:0:$cards}" | sed 's/^npu/npu:/' | paste -sd, -)
  args=(serve "$model" --revision "$rev" --devices "$devices" --host 127.0.0.1 --port 8123 --no-enable-prefix-caching)
  [ "$parser" != "-" ] && args+=(--reasoning-parser "$parser")
  echo "$(ts) START $slug: $model@$rev on $devices${parser:+ (parser $parser)}"
  furiosa-smi status 2>/dev/null | grep -a -E '\| npu' > "$out/$slug.cards-before.txt"
  "$scratch/$venv/bin/furiosa-llm" "${args[@]}" > "$out/$slug.serve.log" 2>&1 &
  server=$!
  ready=0
  for _ in $(seq 1 240); do
    if ! kill -0 "$server" 2>/dev/null; then break; fi
    if curl -s -m 5 http://127.0.0.1:8123/v1/models | grep -q -E "\"id\": ?\"$model\""; then ready=1; break; fi
    [ -e "$stopfile" ] && break
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    echo "$(ts) FAIL $slug: the server never listed $model; last log lines:"; tail -5 "$out/$slug.serve.log"
    stop_server
    continue
  fi
  curl -s -m 5 http://127.0.0.1:8123/v1/models > "$out/$slug.models.json"
  share=$(( ($(left) - 600) / remaining ))
  limit=$(( share < 3000 ? share : 3000 ))
  [ "$limit" -lt 300 ] && limit=300
  echo "$(ts) READY $slug: running both recipes for at most $limit s"
  pids=()
  for half in maxabs pcc; do
    timeout --kill-after=60 "$limit" uv run lassi --graphics off run "projects/lassi-demo/sweep/$slug-$half.yaml" \
      --run-id "sweep-$stamp-$slug-$half" > "$out/$slug-$half.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; rc=$?; echo "$(ts) $slug run $pid ended rc=$rc"; done
  for half in maxabs pcc; do summarize "sweep-$stamp-$slug-$half" "$slug"; done
  stop_server
  free_after=$(free_cards | wc -l)
  echo "$(ts) DONE $slug; free cards now $free_after; $(left) s left"
done
echo "$(ts) summary:"; cat "$out/summary.tsv"
