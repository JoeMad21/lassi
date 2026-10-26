#!/usr/bin/env bash
# LASSI's full RNGD suite as one big job on the build host, which needs no session once started: the remote side
# of tools/suite.py, a generalization of plans/runs/night/run-night.sh (the night of 2026-10-25) to any rows of a
# model table, any recipe directories, and an optional filter on recipe names. The owner starts it with
#   uv run tools/suite.py launch [options]
# which runs, from a clean pushed commit,
#   uv run tools/rx.py job start --big --slot <slot> --name <name> --timeout <deadline + 900 s> -- \
#     'bash plans/runs/suite/run-suite.sh --deadline <epoch> --table <tsv> --models <a,b> --sweep <dir> ...'
# Steps, in order (--help prints the options):
# 1. Checks: RLIMIT_CORE held at exactly 1 byte (prlimit, read back as (1, 1)); the gate's LASSI_* variables;
#    the sweep's STOP file absent (else it refuses: an earlier stop is still in force); every model in the table.
# 2. The recipes: plans/runs/suite/recipes.py loads every <sweep>/<slug>/<name>.yaml with the repository's recipe
#    loader and writes recipes.tsv (each recipe's hash, model id, and needed assets) and the selected recipes'
#    resolved text under recipes/ (the provenance of what runs); a recipe that does not load stops the job.
# 3. Downloads, unless --skip-download: each model's revision, every file (the server checks the cached snapshot
#    offline), one at a time in the background in table order, each only while the scratch disk keeps the gate's
#    60 GB big-job floor plus a 5 GB margin after the bytes still missing (run-night.sh's rule; GB of 1e9 bytes).
# 4. Assets the selected recipes need, built from the tt-metal install at the pin and checked against their
#    tracked manifests, as run-night.sh builds them; a recipe whose asset did not build as tracked is left out.
# 5. The sweep: plans/runs/rngd-sweep/run.sh once per model and recipe directory, model by model, so one model is
#    served at a time on port 8123; run.sh claims only cards that read 0.00 GiB in `furiosa-smi status` (polling
#    `furiosa-smi ps` and status), checks the port is free and /v1/models lists the model, and waits for the
#    server's full exit and the port to close (Agent Rule 8). Each call gets an equal share of the time left to
#    the deadline. Before each call: the STOP file, the deadline, and the free-space floor are checked.
# 6. Tables: tabulate.py over the runs root, count_features.py per pass, the build-error classifier.
# Every TT program runs on ttsim, a simulator; no Wormhole card is used, and nothing here opens one. Output under
# $LASSI_RUNS_ROOT/suite/<rx job id>/ (tools/suite.py pull fetches it) and run.sh's own under
# $LASSI_RUNS_ROOT/rngd-sweep/<stamp>/ and $LASSI_RUNS_ROOT/runs/. Stop early: tools/suite.py stop, which creates
# $LASSI_RUNS_ROOT/rngd-sweep/STOP; run.sh stops before its next model and this script before its next call, and
# a running download is ended. The scratch root's 150G cap is lifted while OQ-074's lift holds; its size is logged.
set -u -o pipefail

usage() {
  cat <<'EOF'
Usage: bash plans/runs/suite/run-suite.sh --deadline EPOCH --models SLUG[,SLUG...] [options]

Runs LASSI's RNGD suite on the build host as one rx job, from the repository root of a checkout: downloads, the
generated TT assets, plans/runs/rngd-sweep/run.sh once per model and recipe directory (one model served at a time
on RNGD cards; every TT program on ttsim, a simulator), then the tables. tools/suite.py launch starts it.

Options:
  --deadline EPOCH  when the sweep must end, in epoch seconds (required); the rx job's timeout should be later
  --models LIST     comma-separated model slugs from the table, served one at a time in this order (required)
  --sweep DIR       a recipe directory holding <slug>/<combo>-<half>.yaml (repeatable; run in the order given;
                    default projects/lassi-demo/sweep8)
  --only GLOB       run only the recipes whose name (<combo>-<half>, for example v1ex-maxabs) matches GLOB
                    (repeatable; a recipe runs when it matches any; default every recipe)
  --table TSV       the model table (default plans/runs/suite/models.tsv): slug, model, revision, cards,
                    reasoning parser, venv under the scratch root, and the revision's size in bytes (- unknown)
  --skip-download   download nothing; run.sh then skips a model whose snapshot is not in the cache
  --help            print this text and exit

Environment: LASSI_SCRATCH, LASSI_RUNS_ROOT, and LASSI_TOOLCHAINS (the gate sets them); LASSI_RX_RUN_ID (the rx
job id, which names the output directory $LASSI_RUNS_ROOT/suite/<id>); SWEEP_MAX_SHARE (the most seconds one
run.sh call's recipes may run; default 14400, run-night.sh's); SUITE_FETCH_TIMEOUT (the most seconds one download
may take; default 7200, run-night.sh's longest).

Stop early: create $LASSI_RUNS_ROOT/rngd-sweep/STOP (tools/suite.py stop). While it exists this script refuses to
start; tools/suite.py stop --clear removes it.
EOF
}

deadline=""
models=""
sweeps=()
only=()
table=plans/runs/suite/models.tsv
skip_download=0
while [ $# -gt 0 ]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --deadline) deadline=${2:-}; shift 2 || { echo "run-suite: --deadline needs a value" >&2; exit 2; } ;;
    --models) models=${2:-}; shift 2 || { echo "run-suite: --models needs a value" >&2; exit 2; } ;;
    --sweep) sweeps+=("${2:-}"); shift 2 || { echo "run-suite: --sweep needs a value" >&2; exit 2; } ;;
    --only) only+=("${2:-}"); shift 2 || { echo "run-suite: --only needs a value" >&2; exit 2; } ;;
    --table) table=${2:-}; shift 2 || { echo "run-suite: --table needs a value" >&2; exit 2; } ;;
    --skip-download) skip_download=1; shift ;;
    *) echo "run-suite: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
done
[[ "$deadline" =~ ^[0-9]+$ ]] || { echo "run-suite: --deadline must be epoch seconds (see --help)" >&2; exit 2; }
[ -n "$models" ] || { echo "run-suite: --models is required (see --help)" >&2; exit 2; }
[ "${#sweeps[@]}" -gt 0 ] || sweeps=(projects/lassi-demo/sweep8)
IFS=',' read -r -a slugs <<< "$models"

cd "$(dirname "$0")/../../.." || exit 2
if ! prlimit --pid $$ --core=1:1 \
  || ! python3 -I -c 'import resource, sys; sys.exit(resource.getrlimit(resource.RLIMIT_CORE) != (1, 1))'; then
  echo "run-suite: cannot hold RLIMIT_CORE at exactly 1 byte, soft and hard (prlimit, read back as (1, 1))" >&2
  exit 2
fi
scratch=${LASSI_SCRATCH:?LASSI_SCRATCH must be set; the gate sets it}
runs=${LASSI_RUNS_ROOT:?LASSI_RUNS_ROOT must be set; the gate sets it}
toolchains=${LASSI_TOOLCHAINS:?LASSI_TOOLCHAINS must be set; the gate sets it}
tree="$toolchains/tt-metal@5280a9cf"
export HF_HOME="$scratch/.cache/huggingface"
id=${LASSI_RX_RUN_ID:-manual-$(date +%Y%m%d-%H%M%S)}
[[ "$id" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "run-suite: unusable job id '$id'" >&2; exit 2; }
out="$runs/suite/$id"
stopfile="$runs/rngd-sweep/STOP"
floor_gb=60   # the gate's free-space floor for big jobs (tools/server/config.default.json, min_free_gb_big)
margin_gb=5   # run-night.sh's margin on the shared disk
max_share=${SWEEP_MAX_SHARE:-14400}
fetch_max=${SUITE_FETCH_TIMEOUT:-7200}
mkdir -p "$out"
ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }
left() { echo $((deadline - $(date +%s))); }
free_gb() { df -B1000000000 --output=avail "$scratch" | tail -1 | tr -dc '0-9'; }  # GB of 1e9 bytes, as the gate
row_of() { awk -F'\t' -v s="$1" '$0 !~ /^#/ && $1 == s' "$table" | head -1; }

echo "$(ts) run-suite $id: deadline $(date -d "@$deadline" '+%Y-%m-%d %H:%M:%S %z'), $(left) s left"
echo "$(ts) models ${slugs[*]}; sweeps ${sweeps[*]}; only ${only[*]:-(every recipe)}; table $table; output $out"
echo "$(ts) commit $(git rev-parse HEAD 2>/dev/null || echo unknown), $(git status --porcelain 2>/dev/null | wc -l) changed paths"
echo "$(ts) scratch disk: $(free_gb) GB free; scratch root $(du -sh "$scratch" 2>/dev/null | cut -f1)"
if [ -e "$stopfile" ]; then
  echo "$(ts) REFUSED: $stopfile exists, so an earlier stop is still in force and run.sh would stop at once."
  echo "$(ts) Nothing was started. Clear it with: uv run tools/suite.py stop --clear (owner), then launch again."
  exit 2
fi
[ "$(left)" -ge 1800 ] || { echo "$(ts) REFUSED: under 30 minutes to the deadline"; exit 2; }
[ -f "$table" ] || { echo "$(ts) REFUSED: no model table $table"; exit 2; }
for slug in "${slugs[@]}"; do
  [ -n "$(row_of "$slug")" ] || { echo "$(ts) REFUSED: $slug is not a row of $table"; exit 2; }
  row_of "$slug" | cut -f1-6 > "$out/models-$slug.tsv"
done
for sweep in "${sweeps[@]}"; do
  [ -d "$sweep" ] || { echo "$(ts) REFUSED: no recipe directory $sweep"; exit 2; }
done

# 2. The recipes, loaded, listed, and recorded.
margs=(--out "$out/recipes.tsv" --resolved "$out/recipes")
for sweep in "${sweeps[@]}"; do margs+=(--sweep "$sweep"); done
for slug in "${slugs[@]}"; do margs+=(--model "$slug"); done
for glob in "${only[@]}"; do margs+=(--only "$glob"); done
if ! uv run python plans/runs/suite/recipes.py manifest "${margs[@]}" > "$out/recipes.log" 2>&1; then
  echo "$(ts) REFUSED: a recipe does not load:"; tail -5 "$out/recipes.log"; exit 2
fi
tail -1 "$out/recipes.log"
mapfile -t needs < <(uv run python plans/runs/suite/recipes.py needs --manifest "$out/recipes.tsv" | tr -d '\r')

# 3. Downloads, one at a time in the background.
missing_gb() {  # GB a download still needs, rounded up: the revision's size ($2) less what the snapshots hold
  local have
  have=$(du -sbL "$HF_HOME/hub/models--${1/\//--}/snapshots" 2>/dev/null | cut -f1)
  have=${have:-0}
  [ "$have" -ge "$2" ] && { echo 0; return; }
  echo $(( ($2 - have + 999999999) / 1000000000 ))
}
fetch() {  # fetch <slug>: download the row's revision, every file, with the hub online for this one process
  local slug=$1 model rev venv size need limit pid rc
  IFS=$'\t' read -r _ model rev _ _ venv size <<< "$(row_of "$slug")"
  if [ -e "$stopfile" ]; then echo "$(ts) STOP file: no download of $slug"; return 1; fi
  if [ -e "$out/fetch.end" ]; then echo "$(ts) the sweep has ended: no download of $slug"; return 1; fi
  if [ "$(left)" -lt 1800 ]; then echo "$(ts) under 30 min left: no download of $slug"; return 1; fi
  if ! [[ "${size:-}" =~ ^[0-9]+$ ]]; then
    echo "$(ts) SKIP the $slug download: its size is not in $table; run.sh serves it only from the cache"; return 1
  fi
  need=$(( floor_gb + margin_gb + $(missing_gb "$model" "$size") ))
  if [ "$(free_gb)" -lt "$need" ]; then
    echo "$(ts) SKIP the $slug download: $(free_gb) GB free, under $need (the bytes missing, the 60 GB floor, a 5 GB margin)"
    return 1
  fi
  limit=$(left); [ "$limit" -gt "$fetch_max" ] && limit=$fetch_max
  echo "$(ts) $(free_gb) GB free: downloading $model@$rev for at most $limit s"
  HF_HUB_OFFLINE=0 timeout "$limit" "$scratch/$venv/bin/python" -I -c 'import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2]))' "$model" "$rev" > "$out/fetch-$slug.log" 2>&1 &
  pid=$!
  echo "$pid" > "$out/fetch.pid"
  wait "$pid"
  rc=$?
  echo "-" > "$out/fetch.pid"
  [ "$rc" = 0 ] && touch "$out/fetch-$slug.done"
  echo "$(ts) $slug download ended rc=$rc"
  return "$rc"
}
end_fetch() {  # start no more downloads, and end a running one (timeout passes the TERM on to the download)
  local pid
  touch "$out/fetch.end"
  pid=$(cat "$out/fetch.pid" 2>/dev/null)
  [[ "${pid:-}" =~ ^[0-9]+$ ]] && kill -TERM "$pid" 2>/dev/null && echo "$(ts) ended the running download ($pid)"
}
chain=""
if [ "$skip_download" = 0 ]; then
  ( for slug in "${slugs[@]}"; do fetch "$slug"; touch "$out/fetch-$slug.ended"; done ) &
  chain=$!
fi
trap 'end_fetch; [ -n "$chain" ] && wait "$chain" 2>/dev/null; echo "$(ts) run-suite ended"' EXIT

# 4. The assets the selected recipes need, each built and checked against its tracked manifest.
build_asset() {  # build_asset <need>: 0 when the asset built and its manifest equals the tracked one
  [ -d "$tree" ] || { echo "no tt-metal install at $tree"; return 1; }
  case "$1" in
    context:tt-metalium-5280a9cf-full)
      uv run tools/build_tt_context.py --tree "$tree" --name tt-metalium-5280a9cf-full --limit-bytes 131072 \
        > "$out/context-full.log" 2>&1 && git diff --quiet HEAD -- assets/context ;;
    context:tt-metalium-5280a9cf)
      uv run tools/build_tt_context.py --tree "$tree" > "$out/context.log" 2>&1 \
        && git diff --quiet HEAD -- assets/context ;;
    examples:tt-examples-v0)
      uv run tools/build_tt_examples.py build --tree "$tree" > "$out/examples.log" 2>&1 \
        && git diff --quiet HEAD -- assets/examples && uv run tools/build_tt_examples.py check >> "$out/examples.log" 2>&1 ;;
    retrieval:tt-metal-docs)
      uv run tools/build_tt_rag.py build --tree "$tree" > "$out/rag.log" 2>&1 \
        && git diff --quiet HEAD -- assets/rag && uv run tools/build_tt_rag.py check --tree "$tree" >> "$out/rag.log" 2>&1 ;;
    *) echo "run-suite has no build step for $1"; return 1 ;;
  esac
}
failed=()
for need in "${needs[@]}"; do
  if build_asset "$need"; then
    echo "$(ts) $need built; its manifest equals the tracked one"
  else
    echo "$(ts) $need did not build as tracked; the recipes that need it are left out"
    git diff --stat HEAD -- assets/context assets/examples assets/rag
    git checkout -- assets/context assets/examples assets/rag
    failed+=(--failed "$need")
  fi
done

# 5. The sweep: run.sh once per model and recipe directory, one model served at a time.
calls=()
for slug in "${slugs[@]}"; do
  for sweep in "${sweeps[@]}"; do
    read -r count pattern < <(uv run python plans/runs/suite/recipes.py select --manifest "$out/recipes.tsv" \
      --sweep "$sweep" --model "$slug" "${failed[@]}" | tr -d '\r')
    if [ "${count:-0}" -gt 0 ]; then calls+=("$slug|$sweep|$count|${pattern:-}"); else
      echo "$(ts) SKIP $slug over $sweep: no recipe to run"; fi
  done
done
await_fetch() {  # await_fetch <slug>: 0 when the model may be served, 1 to skip it
  local slug=$1 model size
  [ "$skip_download" = 1 ] && return 0
  while [ ! -e "$out/fetch-$slug.ended" ]; do
    if [ -e "$stopfile" ] || [ "$(left)" -lt 1800 ]; then return 1; fi
    sleep 30
  done
  [ -e "$out/fetch-$slug.done" ] && return 0
  IFS=$'\t' read -r _ model _ _ _ _ size <<< "$(row_of "$slug")"
  [[ "${size:-}" =~ ^[0-9]+$ ]] || return 0  # never downloaded: run.sh decides from the cache
  [ "$(missing_gb "$model" "$size")" = 0 ] && return 0
  echo "$(ts) SKIP $slug: its download did not finish, and a partial snapshot does not serve"
  return 1
}
printf 'call\tslug\tsweep\tstamp\trc\trecipes\tskip\n' > "$out/calls.tsv"
total=${#calls[@]}
last=""
echo "$(ts) $total run.sh call(s): ${calls[*]%%|*}"
for index in "${!calls[@]}"; do
  IFS='|' read -r slug sweep count pattern <<< "${calls[$index]}"
  n=$((index + 1))
  if [ -e "$stopfile" ]; then echo "$(ts) STOP file: no more calls"; break; fi
  if [ "$(free_gb)" -lt $((floor_gb + margin_gb)) ]; then
    echo "$(ts) STOP LINE: $(free_gb) GB free, under $((floor_gb + margin_gb)); no more calls"; break
  fi
  await_fetch "$slug" || { echo "$(ts) SKIP call $n ($slug over $sweep)"; continue; }
  if [ "$(left)" -lt 1800 ]; then echo "$(ts) under 30 min left: no more calls"; break; fi
  while [ "$(date +%Y%m%d-%H%M)" = "$last" ]; do sleep 5; done  # run.sh names its pass by the minute
  last=$(date +%Y%m%d-%H%M)
  now=$(date +%s)
  ends=$(( now + (deadline - now) / (total - index) ))  # an equal share of the time left among the calls left
  echo "$(ts) call $n of $total: $slug over $sweep, $count recipe(s), until $(date -d "@$ends" '+%H:%M:%S')"
  SWEEP_SKIP="$pattern" SWEEP_MAX_SHARE="$max_share" bash plans/runs/rngd-sweep/run.sh "$ends" "$sweep" \
    "$out/models-$slug.tsv" > "$out/sweep-$n-$slug.log" 2>&1
  rc=$?
  stamp=$(grep -m1 -oE "sweep [0-9]{8}-[0-9]{4}:" "$out/sweep-$n-$slug.log" | grep -oE "[0-9]{8}-[0-9]{4}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$slug" "$sweep" "${stamp:--}" "$rc" "$count" "${pattern:--}" \
    >> "$out/calls.tsv"
  echo "$(ts) call $n ended rc=$rc, pass ${stamp:--}; its log: $out/sweep-$n-$slug.log"
  grep -E "START|READY|FAIL|SKIP|ABORT|ended" "$out/sweep-$n-$slug.log" | tail -40
  if [ -n "$stamp" ]; then
    mkdir -p "$out/passes/$stamp"
    cp "$runs/rngd-sweep/$stamp/summary.tsv" "$runs/rngd-sweep/$stamp/"*.models.json "$out/passes/$stamp/" 2>/dev/null
  fi
done
if [ -n "$chain" ] && kill -0 "$chain" 2>/dev/null; then end_fetch; fi

# 6. The tables.
python3 plans/runs/rngd-sweep/tabulate.py "$runs" "$out/tables" > "$out/tabulate.log" 2>&1
echo "$(ts) tabulate rc=$?: $(tail -2 "$out/tabulate.log" | head -1)"
for stamp in $(cut -f4 "$out/calls.tsv" | tail -n +2 | grep -E '^[0-9]{8}-[0-9]{4}$' | sort -u); do
  python3 plans/runs/window/count_features.py "$runs" "$stamp" "$out/features-$stamp.tsv"
  echo "$(ts) count_features $stamp rc=$?"
done
python3 tools/classify_build_errors.py "$runs" > "$out/build-errors.txt" 2> "$out/build-errors.stderr"
echo "$(ts) classify rc=$?"
echo "$(ts) scratch disk at the end: $(free_gb) GB free; scratch root $(du -sh "$scratch" 2>/dev/null | cut -f1)"
