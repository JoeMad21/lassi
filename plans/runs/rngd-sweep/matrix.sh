#!/usr/bin/env bash
# The feature matrix in an owner's window (owner, 2026-10-23: "do separate tests for all combinations of features
# added to base LASSI"): projects/lassi-demo/sweep6, cpp -> tt, the API reference pack (P17.21) on or off, retrieval
# over the tt-metal docs (P17.22) on or off, and the coarse mismatch report on or off, plus mmapiragrl, all three
# with restarts (P17.25) and declaration lookup (P17.26), with the fuzzer adversary in every recipe, for the models
# of models6.tsv. Self-contained, so it needs no session once started: it builds the
# two generated assets on the build host, checks that each tracked manifest is unchanged, runs run.sh, and tabulates.
# Started from a clean commit on the build host (plans/runs/rngd-sweep/start-matrix.sh does this) as
#   PYTHONIOENCODING=utf-8 uv run tools/rx.py job start --big --name rngd-matrix --timeout <seconds> -- \
#     'bash plans/runs/rngd-sweep/matrix.sh <deadline as epoch seconds>'
# The pack is written into the slot's assets/context (its fragments are gitignored) and the index under
# $LASSI_SCRATCH/rag, the default tools/build_tt_rag.py and the runner read. An asset that cannot be built leaves out
# only the recipes that need it (SWEEP_SKIP), so the others still run. Every TT program runs on ttsim, a simulator.
# Output under $LASSI_RUNS_ROOT/rngd-matrix/<stamp>/ (the asset logs) and run.sh's own; the tables go to
# $LASSI_RUNS_ROOT/rngd-sweep/tables6. Stop early: the STOP file of run.sh.
set -u -o pipefail
deadline=${1:?usage: matrix.sh <deadline as epoch seconds>}
cd "$(dirname "$0")/../../.." || exit 2
if ! prlimit --pid $$ --core=1:1 \
  || ! python3 -I -c 'import resource, sys; sys.exit(resource.getrlimit(resource.RLIMIT_CORE) != (1, 1))'; then
  echo "matrix: cannot hold RLIMIT_CORE at exactly 1 byte, soft and hard (prlimit, read back as (1, 1))" >&2
  exit 2
fi
scratch=${LASSI_SCRATCH:?LASSI_SCRATCH must be set; the gate sets it}
runs=${LASSI_RUNS_ROOT:?LASSI_RUNS_ROOT must be set; the gate sets it}
toolchains=${LASSI_TOOLCHAINS:?LASSI_TOOLCHAINS must be set; the gate sets it}
tree="$toolchains/tt-metal@5280a9cf"
out="$runs/rngd-matrix/$(date +%Y%m%d-%H%M)"
mkdir -p "$out"
ts() { date '+%Y-%m-%dT%H:%M:%S%z'; }
echo "$(ts) matrix: deadline $(date -d "@$deadline" '+%H:%M:%S'), output $out"
echo "$(ts) scratch disk: $(df -h "$scratch" | tail -1)"
echo "$(ts) scratch root: $(du -sh "$scratch" 2>/dev/null | cut -f1) in $scratch"
skip=()

# The API reference pack, from the install at the pin; its manifest must be the tracked one.
if [ -d "$tree" ] && uv run tools/build_tt_context.py --tree "$tree" > "$out/context.log" 2>&1 \
   && git diff --quiet HEAD -- assets/context; then
  echo "$(ts) context pack built; its manifest equals the tracked one"
else
  echo "$(ts) the context pack did not build as tracked; the recipes with the API reference are left out:"
  tail -5 "$out/context.log" 2>/dev/null; git diff --stat HEAD -- assets/context
  git checkout -- assets/context 2>/dev/null
  skip+=("api")
fi

# The docs index, from the install, else from the corpus fetched at the pin; its manifest must be the tracked one.
rag_ok=0
if [ -d "$tree" ] && uv run tools/build_tt_rag.py build --tree "$tree" > "$out/rag.log" 2>&1 \
   && git diff --quiet HEAD -- assets/rag && uv run tools/build_tt_rag.py check --tree "$tree" >> "$out/rag.log" 2>&1; then
  rag_ok=1
else
  echo "$(ts) the docs index did not reproduce the tracked manifest from the install; fetching the corpus at the pin"
  git checkout -- assets/rag 2>/dev/null
  fetched="$scratch/rag-tree/tt-metal-5280a9cf"
  if ! uv run tools/build_tt_rag.py fetch --dest "$fetched" > "$out/rag-fetch.log" 2>&1; then
    echo "$(ts) the fetch failed; last lines:"; tail -5 "$out/rag-fetch.log"
  elif uv run tools/build_tt_rag.py build --tree "$fetched" > "$out/rag.log" 2>&1 \
       && git diff --quiet HEAD -- assets/rag \
       && uv run tools/build_tt_rag.py check --tree "$fetched" >> "$out/rag.log" 2>&1; then
    rag_ok=1
  else
    echo "$(ts) the build from the fetched corpus failed; last lines:"; tail -5 "$out/rag.log"
    git diff --stat HEAD -- assets/rag; git checkout -- assets/rag 2>/dev/null
  fi
fi
if [ "$rag_ok" = 1 ]; then
  echo "$(ts) docs index built and checked; its manifest equals the tracked one"
else
  echo "$(ts) the recipes with retrieval are left out"
  skip+=("rag")
fi

pattern=""
[ "${#skip[@]}" -gt 0 ] && pattern=$(IFS='|'; echo "${skip[*]}")
# The share caps each model but the last, which takes the time left (two models: the first gets about half).
SWEEP_SKIP="$pattern" SWEEP_MAX_SHARE=${SWEEP_MAX_SHARE:-14400} bash plans/runs/rngd-sweep/run.sh "$deadline" \
  projects/lassi-demo/sweep6 "${MATRIX_MODELS:-plans/runs/rngd-sweep/models6.tsv}"  # MATRIX_MODELS: another table
python3 plans/runs/rngd-sweep/tabulate.py "$runs" "$runs/rngd-sweep/tables6"
echo "$(ts) scratch root: $(du -sh "$scratch" 2>/dev/null | cut -f1) at the end"
echo "$(ts) matrix ended"
