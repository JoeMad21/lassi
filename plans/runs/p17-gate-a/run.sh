#!/usr/bin/env bash
# Gate part (a) of P17 on alpha01's CPU (task P17.11; bible Build Roadmap, P17 row, Gate column): hf_local on
# PyTorch CPU runs one HeCBench item in both directions, `lassi train` runs sft, dpo, and grpo briefly on the CPU,
# and each recipe that names a GPU is refused. Started from the detached clean worktree at a clean commit as
#   PYTHONIOENCODING=utf-8 uv run tools/rx.py job start --big --name p17-gate-a --timeout 21600 -- \
#     'bash plans/runs/p17-gate-a/run.sh'
# The launch waits for the owner's yes in the working session (Agent Rule 17: speed and hf-run load the model in
# process, and the train-* steps train), asked with what runs, for how long, and what it touches.
# It takes no argument; `--stand-in` is for the local dry run only (see Stand-ins below). gate.py beside it holds
# the checks and records. Each step with a number in steps.tsv's limit column runs under that limit (they sum to
# 17,040 s). preflight, header, and du record "-": each du in them has its own 600 s limit, and du-new, verdict,
# and ascii have none, so their worst case is bounded only by the job's 21,600 s, which would leave no verdict.
#
# Stop: `rx job kill` ends only the gate's runner (PHASE-NOTES P3), and every step runs in its own process group
# (timeout), so neither it nor the job's timeout ends a running step. The owner's gate STOP is read before every
# step. To stop at once, one `rx exec` of `pkill -TERM -f` on this file's path and on the venv's path (it holds
# the rx id), each written with a bracket (ru[n].sh, bi[n]/) so the pattern cannot match the exec's own shell.
#
# Steps, in order (each step's status, seconds, and limit go to report/steps.tsv; its output to raw/<step>.out and
# raw/<step>.err; a step whose input failed is recorded as skipped):
#   preflight  a core limit of exactly 1 byte; the gate's variables, HOME, and the repository under
#              /mnt/nvme10/joseph_ufl (Agent Rule 7); no owner STOP; not an rx snapshot; a new run directory and a
#              new venv; the tools and the two pinned toolchain trees present; the scratch root under the 115 GiB
#              stop line with room for the planned 4 GiB (plan Constraints). A refusal prints one line, exits 2.
#   header     date, host, kernel, commit and tree, versions, and du -sk of the scratch root, HF_HOME, and the uv
#              cache before anything is installed; a stamp file and the scratch root's first-level entries.
#   env        `uv sync --frozen --no-dev --extra cpu` into build/p17-gate-a-venv-<rx id> (gitignored): the lock's
#              explicit pytorch-cpu index for torch, PyPI for the rest; never uv's --torch-backend (it queries the
#              GPUs; PHASE-NOTES P17), and every index or backend variable unset first. Then the freeze, and
#              env-pins checks the five framework pins in it.
#   fetch-model  gate.py fetch-model: the model at its pinned revision under HF_HOME, fetched online once only
#              when absent, checked against the bible's pin. Every later step runs with the hub offline.
#   fetch-bench, fetch-upstream, extract-assets: the suite's sources, upstream LASSI at its pin, and the slot's
#              prompt and context files, before any lassi run (plans/LESSONS.md, alpha01).
#   speed      gate.py speed: hf_local's load, decode, and prefill times on the CPU (before hf-run, so a load
#              failure shows in minutes).
#   hf-run     `lassi run tests/fixtures/recipes/p17-cpu-hf.yaml`, then gate.py trials.
#   train-sft, train-dpo, train-grpo: `lassi train` of each smoke recipe, then gate.py checkpoint.
#   refuse-*   `lassi run` of each p17-gpu-*.yaml under strace (open, access, and stat calls only), with a bash
#              between strace and lassi that exits normally, so a crash never reaches strace's re-raise with
#              its own core limit of 0 (plans/LESSONS.md, alpha01); then gate.py refusal.
#   du         du -sk of HF_HOME, the uv cache, the venv, the run and train trees, the batch's directory, and the
#              scratch root; gate.py du-new.
#   close      the slot tree still clean, gate.py verdict, gate.py ascii (report/files.txt).
#
# Layout, under $LASSI_RUNS_ROOT/p17-gate-a/$LASSI_RX_RUN_ID:
#   p17-gate-a-report/  small ASCII text and JSON: the evidence; pulled with
#            PYTHONIOENCODING=utf-8 uv run tools/rx.py pull --path \
#              lassi-runs/p17-gate-a/<rx id>/p17-gate-a-report --into results/p17-gate-a
#   raw/     every step's output and the strace logs; not pulled.
# lassi's own trees stay where lassi puts them: $LASSI_RUNS_ROOT/runs/p17-gate-a-<rx id> and
# $LASSI_RUNS_ROOT/train/p17-gate-a-<method>-<rx id>; only their records are copied into the report.
#
# Writes on the host: the run directory, the run and train trees, the venv in the slot's build/, the uv cache,
# HF_HOME (the model, about 1.0 GB), the bench sources under $LASSI_SCRATCH/bench, upstream and the generated
# prompt files in the slot (gitignored), and the torch, Triton, and inductor caches under
# $LASSI_SCRATCH/.cache/lassi-p17-* (names no other checkout under the same HOME uses).
# Nothing is deleted, no sudo, no server, no listening port; the network is used only by env, fetch-model,
# fetch-bench, and fetch-upstream, never by a sandboxed run. No GPU wheel is installed and no device node is
# opened (OQ-002); no RNGD card is used. torch, tokenizers, and BLAS use 16 threads (OMP_NUM_THREADS); sandboxed
# programs get the native executor's own settings.
#
# Stdout: every line goes through `say`, which prints at most SAY_MAX (58) lines, then one cap notice; the EXIT
# trap prints one final line, so the hard bound is 60 lines. Every said line is also in report/stdout.txt, and the
# batch's own stderr goes to report/stderr.txt.
#
# Stand-ins (local dry run only; the real run never passes --stand-in): with --stand-in, the scratch prefix, the
# Python request, the hf_local recipe of fetch-model, speed, and hf-run, and the model pin come from
# P17_STANDIN_PREFIX, P17_STANDIN_PYTHON, P17_STANDIN_HF_RECIPE, and P17_STANDIN_MODEL_EXPECT (a JSON file in
# gate.py MODEL_PIN's shape), and report/batch.txt and the first stdout line say STAND-IN. Every other line of the
# batch is the same.
set -euo pipefail
# A core limit of exactly 1 byte, soft and hard (plans/LESSONS.md, alpha01): bash's `ulimit -c 1` would set
# 1024 bytes, which a piped systemd-coredump still stores. prlimit takes bytes; verified in the preflight.
prlimit --pid $$ --core=1:1 || { echo "p17-gate-a: cannot set a 1-byte core limit with prlimit"; exit 2; }
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
STAND_IN=0
case "${1:-}" in
  "") ;;
  --stand-in) STAND_IN=1 ;;
  *) echo "p17-gate-a: unknown argument '$1'; the real run takes none"; exit 2 ;;
esac
: "${LASSI_SCRATCH:?LASSI_SCRATCH must be set; the gate sets it}"
: "${LASSI_RUNS_ROOT:?LASSI_RUNS_ROOT must be set; the gate sets it}"
: "${LASSI_TOOLCHAINS:?LASSI_TOOLCHAINS must be set; the gate sets it}"
: "${LASSI_RX_RUN_ID:?LASSI_RX_RUN_ID must be set; the gate sets it for rx jobs}"
: "${UV_CACHE_DIR:?UV_CACHE_DIR must be set; the gate sets it}"
: "${HF_HOME:?HF_HOME must be set; the gate sets it}"
: "${XDG_CACHE_HOME:?XDG_CACHE_HOME must be set; the gate sets it}"

PREFIX=/mnt/nvme10/joseph_ufl
PY_REQUEST=3.10
HF_RECIPE=tests/fixtures/recipes/p17-cpu-hf.yaml
EXPECT_ARGS=()
if [ "$STAND_IN" = 1 ]; then
  : "${P17_STANDIN_PREFIX:?}" "${P17_STANDIN_PYTHON:?}" "${P17_STANDIN_HF_RECIPE:?}" "${P17_STANDIN_MODEL_EXPECT:?}"
  PREFIX="$P17_STANDIN_PREFIX"
  PY_REQUEST="$P17_STANDIN_PYTHON"
  HF_RECIPE="$P17_STANDIN_HF_RECIPE"
  EXPECT_ARGS=(--expect "$P17_STANDIN_MODEL_EXPECT")
fi
for root in "$LASSI_SCRATCH" "$LASSI_RUNS_ROOT" "$LASSI_TOOLCHAINS" "$UV_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" \
  "${TMPDIR:-/tmp}" "${HOME:-/}" "$repo"; do
  case "$(realpath -m "$root")/" in
    "$PREFIX"/*) ;;
    *) echo "p17-gate-a: refusing to work outside $PREFIX (Agent Rule 7): $root"; exit 2 ;;
  esac
done

ID="$LASSI_RX_RUN_ID"
PLANNED_KIB=$((4 * 1024 * 1024))
STOP_KIB=$((115 * 1024 * 1024))
SAY_MAX=58
FREEZE_PINS=(torch==2.14.1+cpu transformers==5.18.0 trl==1.14.1 peft==0.21.2 accelerate==1.15.0)
REFUSALS=(p17-gpu-hf-cuda p17-gpu-hf-rocm p17-gpu-exec-cuda p17-gpu-exec-rocm)
TRACED=open,openat,openat2,creat,access,faccessat,faccessat2,stat,lstat,newfstatat,statx
TOOLCHAIN_TREES=(nvhpc@24.11 cuda@12.6.3)
run="$LASSI_RUNS_ROOT/p17-gate-a/$ID"
raw="$run/raw"
report="$run/p17-gate-a-report"
VENV="$repo/build/p17-gate-a-venv-$ID"
PY="$VENV/bin/python"
LASSI="$VENV/bin/lassi"
GATE_STOP="$LASSI_SCRATCH/lassi-gate/STOP"
said=0

# Index, link-source, backend, interpreter, and environment settings that could change where uv downloads from,
# what it installs, or which environment it syncs; unset before any uv call and named in batch.txt. The hub
# tokens are unset too: the gate model is public, and no credential is ever needed (Agent Rule 12).
unset_names=""
for name in $(compgen -e); do
  case "$name" in
    UV_INDEX* | UV_EXTRA_INDEX_URL | UV_DEFAULT_INDEX | UV_FIND_LINKS | UV_NO_INDEX | UV_TORCH_BACKEND | \
      UV_PYTHON | UV_PYTHON_PREFERENCE | UV_OFFLINE | UV_PROJECT_ENVIRONMENT | UV_NO_SYNC | UV_FROZEN | \
      UV_CONFIG_FILE | PIP_INDEX_URL | PIP_EXTRA_INDEX_URL | PIP_FIND_LINKS | HF_TOKEN | HUGGING_FACE_HUB_TOKEN | \
      HF_ENDPOINT)
      unset "$name"
      unset_names="$unset_names $name"
      ;;
  esac
done
export UV_PROJECT_ENVIRONMENT="$VENV"
export LASSI_GRAPHICS=off NO_COLOR=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
# Plain HTTP into $HF_HOME/hub: no xet chunk cache and no shared blob store (huggingface_hub 1.33.0
# constants.py:342-343, utils/_runtime.py:157, file_download.py:2025, utils/_shared_blobs.py:383).
export HF_HUB_DISABLE_XET=1
# LASSI-owned cache names: HOME is the scratch root, which other checkouts share (PHASE-NOTES P4).
export TORCH_HOME="$LASSI_SCRATCH/.cache/lassi-p17-torch" TRITON_CACHE_DIR="$LASSI_SCRATCH/.cache/lassi-p17-triton"
export TORCHINDUCTOR_CACHE_DIR="$LASSI_SCRATCH/.cache/lassi-p17-torchinductor"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 RAYON_NUM_THREADS=16

# say <line>...: print each line while fewer than SAY_MAX have been printed, then one cap notice; keep all of
# them in report/stdout.txt.
say() {
  local arg line
  for arg in "$@"; do
    while IFS= read -r line; do
      printf '%s\n' "$line" >>"$report/stdout.txt"
      if [ "$said" -lt "$SAY_MAX" ]; then
        printf '%s\n' "${line:0:200}"
      elif [ "$said" -eq "$SAY_MAX" ]; then
        echo "p17-gate-a: stdout cap of $SAY_MAX lines reached; the rest is in report/stdout.txt"
      fi
      said=$((said + 1))
    done <<<"$arg"
  done
}

# say_file <file> <count>: say the first <count> lines of a file.
say_file() {
  local line
  while IFS= read -r line; do say "$line"; done < <(head -n "$2" "$1" 2>/dev/null || true)
}

# kib <path>...: print du -sk of the paths together (0 when none exists, unreadable entries skipped, or du
# passes its 600 s limit).
kib() {
  local total
  total="$( (timeout -k 10 600 du -sk -c "$@" 2>/dev/null || true) | tail -n 1 | cut -f1)"
  case "$total" in '' | *[!0-9]*) echo 0 ;; *) echo "$total" ;; esac
}

# step <name> <limit_s> <command> <args...>: run the command from the repository root under timeout, with its
# output in raw/<name>.out and raw/<name>.err; append name, status, seconds, and limit to report/steps.tsv; set
# STEP_RC. When the status is not EXPECT_RC (0 unless the caller sets it), say one line ending with the last line
# of its stderr (bytes outside printable ASCII shown as ?), unless the step is a gate.py check (CHECKING=1) whose
# stderr is empty: that check says what failed. While the owner's gate STOP is set, the step is skipped (STEP_RC 1).
STEP_RC=0
EXPECT_RC=0
CHECKING=0
step() {
  local name="$1" limit="$2" start=$SECONDS rc=0 last
  shift 2
  if [ -e "$GATE_STOP" ]; then skip "$name" "the owner's gate STOP is set"; STEP_RC=1; return 0; fi
  (cd "$repo" && timeout -k 30 "$limit" "$@") >"$raw/$name.out" 2>"$raw/$name.err" </dev/null || rc=$?
  printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$((SECONDS - start))" "$limit" >>"$report/steps.tsv"
  STEP_RC=$rc
  if [ "$rc" -ne "$EXPECT_RC" ] && { [ "$CHECKING" = 0 ] || [ -s "$raw/$name.err" ]; }; then
    last="$(tail -n 1 "$raw/$name.err" | LC_ALL=C tr -c '\n -~' '?' | cut -c1-120)"
    say "$name: status $rc after $((SECONDS - start)) s; raw/$name.err ends: $last"
  fi
  return 0
}

# skip <name> <reason>: record a step that did not run, and say why.
skip() {
  printf '%s\tskipped\t0\t-\n' "$1" >>"$report/steps.tsv"
  say "$1: skipped ($2)"
}

# check <name> <limit_s> <lines> <gate.py args...>: run gate.py as step <name> with the gate venv's Python and
# say the first <lines> lines it printed.
check() {
  local name="$1" limit="$2" lines="$3"
  shift 3
  CHECKING=1
  step "$name" "$limit" "$PY" "$here/gate.py" "$@"
  CHECKING=0
  say_file "$raw/$name.out" "$lines"
}

preflight() {
  local tool tree
  if ! python3 -c 'import resource,sys; sys.exit(0 if resource.getrlimit(resource.RLIMIT_CORE)==(1,1) else 1)'; then
    echo "p17-gate-a: RLIMIT_CORE is not exactly (1, 1) byte; a crash could store a core on the root filesystem"
    return 1
  fi
  for tool in uv python3 git du find sort timeout prlimit strace systemd-run unshare realpath; do
    command -v "$tool" >/dev/null || { echo "p17-gate-a: $tool is not on PATH"; return 1; }
  done
  for tree in "${TOOLCHAIN_TREES[@]}"; do
    [ -d "$LASSI_TOOLCHAINS/$tree" ] || { echo "p17-gate-a: $LASSI_TOOLCHAINS/$tree is not installed"; return 1; }
  done
  if [ -e "$GATE_STOP" ]; then echo "p17-gate-a: the owner's gate STOP is set; not starting"; return 1; fi
  [ ! -e "$run" ] || { echo "p17-gate-a: $run exists already"; return 1; }
  [ ! -e "$VENV" ] || { echo "p17-gate-a: $VENV exists already"; return 1; }
  case "$(git -C "$repo" log -1 --format=%s)" in
    'rx snapshot of '*) echo "p17-gate-a: dirty launch (rx snapshot); relaunch from a clean commit"; return 1 ;;
  esac
}

preflight || exit 2
commit="$(git -C "$repo" rev-parse HEAD)"
changed="$(git -C "$repo" status --porcelain | wc -l)"
scratch_before="$(kib "$LASSI_SCRATCH")"
if [ "$scratch_before" -eq 0 ]; then
  echo "p17-gate-a: du of the scratch root failed or timed out; not starting"
  exit 2
fi
if [ $((scratch_before + PLANNED_KIB)) -gt "$STOP_KIB" ]; then
  echo "p17-gate-a: the scratch root holds $scratch_before KiB; $PLANNED_KIB KiB more would pass the 115 GiB stop line"
  exit 2
fi
mkdir -p "$raw" "$report"
exec 2>>"$report/stderr.txt"
trap '' PIPE
finish() {
  local rc=$?
  echo "p17-gate-a: ended with status $rc after $SECONDS s; pull: PYTHONIOENCODING=utf-8 uv run tools/rx.py pull" \
    "--path lassi-runs/p17-gate-a/$ID/p17-gate-a-report --into results/p17-gate-a"
}
trap finish EXIT
printf 'step\trc\tseconds\tlimit_s\n' >"$report/steps.tsv"
printf 'preflight\t0\t%s\t-\n' "$SECONDS" >>"$report/steps.tsv"

# Header.
header_start=$SECONDS
hf_before="$(kib "$HF_HOME")"
cache_before="$(kib "$UV_CACHE_DIR")"
touch "$raw/stamp"
python3 "$here/gate.py" scratch-entries --scratch "$LASSI_SCRATCH" >"$raw/scratch-entries-before.txt" || true
# uv's user and system configuration files, which `uv sync` reads when present; recorded, not said.
uv_configs=""
for file in /etc/uv/uv.toml /etc/xdg/uv/uv.toml "${XDG_CONFIG_HOME:-$HOME/.config}/uv/uv.toml"; do
  [ ! -e "$file" ] || uv_configs="$uv_configs $file"
done
mode=real
[ "$STAND_IN" = 0 ] || mode="STAND-IN (local dry run; not evidence)"
{
  echo "mode: $mode"
  echo "rx id $ID; run $run"
  echo "date (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ); host $(hostname); kernel $(uname -r); nproc $(nproc)"
  echo "load: $(cut -d' ' -f1-3 /proc/loadavg); commit $commit; tree $([ "$changed" -eq 0 ] && echo clean \
    || echo "dirty ($changed changed paths)")"
  echo "python3: $(python3 --version 2>&1); uv: $(uv --version 2>&1); strace: $(strace -V 2>&1 | head -n 1)"
  echo "du before (KiB): scratch root $scratch_before (stop line $STOP_KIB); HF_HOME $hf_before; uv cache" \
    "$cache_before"
  echo "core limit 1 byte; unset:${unset_names:- none}; HF_HUB_DISABLE_XET=1; OMP_NUM_THREADS=$OMP_NUM_THREADS"
  echo "venv $VENV; HF_HOME $HF_HOME; TMPDIR ${TMPDIR:-unset}; HOME $HOME"
  echo "hf recipe $HF_RECIPE"
  echo "uv config files (UV_CONFIG_FILE unset):${uv_configs:- none}"
} >"$report/batch.txt"
say_file "$report/batch.txt" 9
printf 'header\t0\t%s\t-\n' "$((SECONDS - header_start))" >>"$report/steps.tsv"

# env: the gate's own environment from the lock, with the cpu extra (never --torch-backend).
step env 1200 uv sync --frozen --no-dev --extra cpu --python "$PY_REQUEST" --no-managed-python --no-python-downloads
env_rc=$STEP_RC
if [ "$env_rc" -eq 0 ]; then
  uv pip freeze --python "$PY" >"$report/freeze.txt" 2>>"$raw/env.err" || true
  missing=""
  for pin in "${FREEZE_PINS[@]}"; do grep -qxF "$pin" "$report/freeze.txt" || missing="$missing $pin"; done
  [ -z "$missing" ] || env_rc=1
  printf 'env-pins\t%s\t0\t-\n' "$env_rc" >>"$report/steps.tsv"
  packages="$(grep -c '==' "$report/freeze.txt" || true)"
  say "env: $packages packages; venv $(kib "$VENV") KiB; uv cache $cache_before -> $(kib "$UV_CACHE_DIR") KiB;\
 pins ${missing:+MISSING}${missing:-match}"
fi

# fetch-model, then the hub goes offline for every later step.
if [ "$env_rc" -eq 0 ]; then
  check fetch-model 1800 1 fetch-model --recipe "$HF_RECIPE" "${EXPECT_ARGS[@]}" --report "$report"
  model_rc=$STEP_RC
else
  skip fetch-model "no environment"
  model_rc=1
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# The suite's sources, upstream LASSI, and the slot's prompt and context files, before any lassi run.
inputs_rc=1
if [ "$env_rc" -eq 0 ]; then
  step fetch-bench 600 "$PY" tools/fetch_bench.py assets/bench/lassi-hecbench-10.yaml
  bench_rc=$STEP_RC
  step fetch-upstream 300 "$PY" tools/fetch_upstream.py
  upstream_rc=$STEP_RC
  extract_rc=skipped
  if [ "$upstream_rc" -eq 0 ]; then
    step extract-assets 300 "$PY" tools/extract_lassi_assets.py
    extract_rc=$STEP_RC
    inputs_rc=$((bench_rc + extract_rc))
  else
    skip extract-assets "no upstream checkout"
  fi
  prompts="$(find "$repo/assets/prompts/lassi-2024" -name '*.txt' 2>/dev/null | wc -l)"
  say "inputs: fetch-bench $bench_rc, fetch-upstream $upstream_rc, extract-assets $extract_rc; prompt files $prompts"
else
  for name in fetch-bench fetch-upstream extract-assets; do skip "$name" "no environment"; done
fi

# speed and hf-run: the model on the CPU.
if [ "$model_rc" -eq 0 ]; then
  check speed 600 2 speed --recipe "$HF_RECIPE" --report "$report"
  speed_rc=$STEP_RC
else
  skip speed "no model"
  speed_rc=1
fi
run_dir="$LASSI_RUNS_ROOT/runs/p17-gate-a-$ID"
if [ "$speed_rc" -eq 0 ] && [ "$inputs_rc" -eq 0 ]; then
  step hf-run 7200 "$LASSI" --graphics off run "$HF_RECIPE" --run-id "p17-gate-a-$ID"
  check hf-trials 300 2 trials --recipe "$HF_RECIPE" --run-dir "$run_dir" --stderr "$raw/hf-run.err" --report "$report"
else
  skip hf-run "no model, no inputs, or a failed load"
fi

# The three train smoke runs.
for method in sft dpo grpo; do
  if [ "$env_rc" -eq 0 ]; then
    step "train-$method" 900 "$LASSI" --graphics off train "tests/fixtures/recipes/p17-train-$method.yaml" \
      --train-id "p17-gate-a-$method-$ID"
    check "check-train-$method" 120 1 checkpoint --train-dir "$LASSI_RUNS_ROOT/train/p17-gate-a-$method-$ID" \
      --method "$method" --report "$report/train-$method"
  else
    skip "train-$method" "no environment"
  fi
done

# The four refusals, under strace, with a bash between strace and lassi that exits normally.
for name in "${REFUSALS[@]}"; do
  short="refuse-${name#p17-gpu-}"
  if [ "$env_rc" -eq 0 ] && [ "$inputs_rc" -eq 0 ]; then
    EXPECT_RC=2
    step "$short" 300 strace -f -qq -o "$raw/$name.strace" -e trace="$TRACED" -- \
      bash -c '"$@"; exit $?' refuse "$LASSI" --graphics off run "tests/fixtures/recipes/$name.yaml" \
      --run-id "p17-gate-a-$name-$ID"
    EXPECT_RC=0
    check "check-$short" 120 1 refusal --name "$name" --rc "$STEP_RC" --stderr "$raw/$short.err" \
      --strace "$raw/$name.strace" --runs-root "$LASSI_RUNS_ROOT" --run-id "p17-gate-a-$name-$ID" --report "$report"
  else
    skip "$short" "no environment or no inputs"
  fi
done

# du: what the batch added.
du_start=$SECONDS
{
  echo "hf_home_kib $(kib "$HF_HOME") (before $hf_before)"
  echo "uv_cache_kib $(kib "$UV_CACHE_DIR") (before $cache_before)"
  echo "venv_kib $(kib "$VENV")"
  echo "run_tree_kib $(kib "$run_dir")"
  for method in sft dpo grpo; do echo "train_${method}_kib $(kib "$LASSI_RUNS_ROOT/train/p17-gate-a-$method-$ID")"; done
  echo "batch_dir_kib $(kib "$run")"
  echo "torch_caches_kib $(kib "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR")"
  echo "scratch_root_kib $(kib "$LASSI_SCRATCH") (before $scratch_before)"
} >"$report/sizes.txt"
printf 'du\t0\t%s\t-\n' "$((SECONDS - du_start))" >>"$report/steps.tsv"
say "du: $(grep -E '^(hf_home|venv|scratch_root)_kib' "$report/sizes.txt" | paste -sd ';' - | sed 's/;/; /g')"
python3 "$here/gate.py" du-new --stamp "$raw/stamp" --before "$raw/scratch-entries-before.txt" --hf-home "$HF_HOME" \
  --scratch "$LASSI_SCRATCH" --report "$report" >"$raw/du-new.out" 2>>"$raw/du-new.err" || true
say_file "$raw/du-new.out" 1

# close: the slot tree, the verdict, and the ASCII pass with the report's file list.
changed_after="$(git -C "$repo" status --porcelain | wc -l)"
tree_after=clean
[ "$changed_after" -eq 0 ] || tree_after="dirty"
say "close: slot tree $tree_after after the batch ($changed_after paths); report $(kib "$report") KiB"
verdict_rc=0
python3 "$here/gate.py" verdict --tree "$tree_after" --report "$report" >"$raw/verdict.out" 2>>"$raw/verdict.err" \
  || verdict_rc=$?
say_file "$raw/verdict.out" 7
python3 "$here/gate.py" ascii --dir "$report" >"$raw/ascii.out" 2>>"$raw/ascii.err" || true
say_file "$raw/ascii.out" 1
exit "$verdict_rc"
