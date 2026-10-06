#!/usr/bin/env bash
# P17.1 spike batch: the CPU framework set in a scratch venv, and hipcc's gfx942 compile without a GPU
# (plans/p17-portable.md, task P17.1; the report is plans/spikes/p17-frameworks.md). This is the one `rx run` the
# plan allows ("one rx run that installs the CPU set in a scratch venv and measures it"); it also answers whether
# alpha01's hipcc builds gfx942 code without a GPU. Started from a clean commit as
#   PYTHONIOENCODING=utf-8 uv run tools/rx.py run --timeout 7200 -- 'bash plans/spikes/p17-frameworks/run.sh'
# It takes no argument; `--stand-in` is for the local dry run only (see Stand-ins below). Every step has its own
# limit, and the limits sum to 5680 s, under the 7200 s run limit; the expected time is minutes (the download of
# the CPU set dominates).
#
# Steps, in order:
#   0. preflight and header: a core limit of exactly 1 byte; the gate's variables under /mnt/nvme10/joseph_ufl
#      (Agent Rule 7); no owner STOP; a new run directory and a new venv path; the scratch root under the
#      115 GiB stop line with room for the planned 4 GiB; then the UTC date, the slot's commit and whether its
#      tree is clean, the python3 and uv versions, and du -sk of $UV_CACHE_DIR before anything is installed.
#   1. the venv: `uv venv` with the system Python 3.10 (--no-managed-python --no-python-downloads) at
#      build/p17-venv-cpu-<rx id> inside the slot worktree. build/ is ignored by git, so the slot's tree stays
#      clean and the gate's `git clean -fd` on a later sync keeps it; `rx slot-rm` frees it (agents never delete
#      files on alpha01). The rx id in the name makes every attempt a fresh venv.
#   2. the resolution record: `uv pip compile --emit-index-annotation` with the install's own index flags, so the
#      report names the index each package came from (metadata only).
#   3. the install: the pins researcher's command, verbatim except --no-config (no uv.toml or pyproject setting
#      can add an index): the PyTorch CPU index first, PyPI as the extra index, unsafe-best-match. Downloads
#      come only from https://download.pytorch.org/whl/cpu and https://pypi.org/simple (and PyPI's file host).
#      Any UV_* or PIP_* variable that names an index, a link source, a torch backend, or a Python is unset
#      first. --torch-backend is never used (it queries the GPUs).
#   4. the measures: `uv pip freeze` checked against the expected pins, `uv pip check`, du -sk of the venv
#      (and its apparent size), of the uv cache after, and of both together (the disk the install added), the
#      count of hard-linked files (uv links from its cache), and the largest site-packages entries.
#   5. the imports, in the sandbox (frameworks.py imports): each package's version, torch.version.cuda and
#      torch.version.hip, the CPU capability, and one 2x3 matmul. The import program makes no torch.cuda call and no
#      device query (a library may at import); the
#      sandbox's private /dev holds no GPU node, so even a library that tried could open none.
#   6. hipcc, in the sandbox (frameworks.py hip): --version; compile-only for gfx942 with an explicit
#      --offload-arch, -c to an object in the workdir, and env -i (HIP_PATH and ROCM_PATH unset); the ROCm root
#      read-only; the object inspected with the ROCm LLVM tools (bundle targets, the gfx942 code object's ELF
#      header and notes); the same compile under strace -f, when the host has strace, listing the programs
#      hipcc started and every /dev and /sys path it touched.
#   7. closing checks: the slot's tree still clean, du of the run directory, the ASCII pass over report/.
#
# Layout, under $LASSI_RUNS_ROOT/p17-frameworks/$LASSI_RX_RUN_ID:
#   report/  small ASCII text and JSON: the evidence; pulled with
#            PYTHONIOENCODING=utf-8 uv run tools/rx.py pull --path lassi-runs/p17-frameworks/<rx id>/report \
#              --into results/p17-frameworks
#   raw/     full logs, the sandbox workdirs (objects, the strace log); not pulled.
# Writes on the host: that run directory, the venv in the slot, the uv cache under $UV_CACHE_DIR, uv's own
# temporary files under $TMPDIR (the gate's scratch tmp), and the slot's .venv if a fresh slot needs one. Nothing
# is deleted, no sudo, no server, no listening port; the sandboxed steps have no network at all.
#
# Stdout: every line goes through `say`, which prints at most SAY_MAX (58) lines, then one cap notice; the EXIT
# trap prints one final line, so the hard bound is 60 lines. Counted worst case: 55 (header 9; venv, resolve,
# install, and warm-up 1 each; freeze 4; pip check 1; sizes 3; imports 8; hip 23; close 2; the final line 1). A
# failed preflight prints one line and exits. Every said line is also in report/stdout.txt, and the batch's own
# stderr goes to report/stderr.txt.
#
# Stand-ins (local dry run only; the real run never passes --stand-in): with --stand-in, the ROCm root, the
# Python request, the package list, the import list, and the expected pins come from P17_STANDIN_ROCM,
# P17_STANDIN_PYTHON, P17_STANDIN_PACKAGES, P17_STANDIN_MODULES, and P17_STANDIN_EXPECT, and report/batch.txt
# and the first stdout line say STAND-IN. Every other line of the batch is the same.
set -euo pipefail
# A core limit of exactly 1 byte, soft and hard (plans/LESSONS.md, alpha01): bash's `ulimit -c 1` would set
# 1024 bytes, which a piped systemd-coredump still stores. prlimit takes bytes; verified in the preflight.
prlimit --pid $$ --core=1:1 || { echo "p17: cannot set a 1-byte core limit with prlimit"; exit 2; }
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
STAND_IN=0
case "${1:-}" in
  "") ;;
  --stand-in) STAND_IN=1 ;;
  *) echo "p17: unknown argument '$1'; the real run takes none"; exit 2 ;;
esac
: "${LASSI_SCRATCH:?LASSI_SCRATCH must be set; the gate sets it}"
: "${LASSI_RUNS_ROOT:?LASSI_RUNS_ROOT must be set; the gate sets it}"
: "${LASSI_RX_RUN_ID:?LASSI_RX_RUN_ID must be set; the gate sets it for rx runs}"
: "${UV_CACHE_DIR:?UV_CACHE_DIR must be set; the gate sets it}"
for root in "$LASSI_SCRATCH" "$LASSI_RUNS_ROOT" "$UV_CACHE_DIR" "${TMPDIR:-/tmp}" "$repo"; do
  case "$(realpath -m "$root")/" in
    /mnt/nvme10/joseph_ufl/*) ;;
    *) echo "p17: refusing to work outside /mnt/nvme10/joseph_ufl (Agent Rule 7): $root"; exit 2 ;;
  esac
done

ROCM=/opt/rocm/core-7.12
PY_REQUEST=3.10
TORCH_INDEX=https://download.pytorch.org/whl/cpu
PYPI_INDEX=https://pypi.org/simple
PACKAGES=("torch==2.14.1+cpu" "transformers==5.18.0" "trl==1.14.1" "peft==0.21.2" "accelerate==1.15.0")
MODULES=(torch transformers trl peft accelerate datasets huggingface_hub tokenizers safetensors numpy pandas)
EXPECT=(torch=2.14.1+cpu transformers=5.18.0 trl=1.14.1 peft=0.21.2 accelerate=1.15.0 datasets=5.1.0
  huggingface-hub=1.33.0 tokenizers=0.23.2 safetensors=0.8.0 numpy=2.2.6 pandas=2.3.3)
if [ "$STAND_IN" = 1 ]; then
  : "${P17_STANDIN_ROCM:?}" "${P17_STANDIN_PYTHON:?}" "${P17_STANDIN_PACKAGES:?}" "${P17_STANDIN_MODULES:?}"
  : "${P17_STANDIN_EXPECT:?}"
  ROCM="$P17_STANDIN_ROCM"
  PY_REQUEST="$P17_STANDIN_PYTHON"
  read -r -a PACKAGES <<<"$P17_STANDIN_PACKAGES"
  read -r -a MODULES <<<"$P17_STANDIN_MODULES"
  read -r -a EXPECT <<<"$P17_STANDIN_EXPECT"
fi
PLANNED_KIB=$((4 * 1024 * 1024))
STOP_KIB=$((115 * 1024 * 1024))
INSTALL_S=1200
SAY_MAX=58
run="$LASSI_RUNS_ROOT/p17-frameworks/$LASSI_RX_RUN_ID"
raw="$run/raw"
report="$run/report"
VENV="$repo/build/p17-venv-cpu-$LASSI_RX_RUN_ID"
GATE_STOP="$LASSI_SCRATCH/lassi-gate/STOP"
said=0

# Index, link-source, backend, and interpreter settings that could change where uv downloads from or what it
# installs; unset before any uv call, and named in the header.
unset_names=""
for name in $(compgen -e); do
  case "$name" in
    UV_INDEX* | UV_EXTRA_INDEX_URL | UV_DEFAULT_INDEX | UV_FIND_LINKS | UV_NO_INDEX | UV_TORCH_BACKEND | \
      UV_PYTHON | UV_PYTHON_PREFERENCE | UV_OFFLINE | PIP_INDEX_URL | PIP_EXTRA_INDEX_URL | PIP_FIND_LINKS)
      unset "$name"
      unset_names="$unset_names $name"
      ;;
  esac
done
export NO_COLOR=1

# say <line>...: print each line while fewer than SAY_MAX have been printed, then one cap notice; keep all of
# them in report/stdout.txt.
say() {
  local arg line
  for arg in "$@"; do
    # An argument holding newlines counts as one line per newline-separated piece.
    while IFS= read -r line; do
      printf '%s\n' "$line" >>"$report/stdout.txt"
      if [ "$said" -lt "$SAY_MAX" ]; then
        printf '%s\n' "${line:0:200}"
      elif [ "$said" -eq "$SAY_MAX" ]; then
        echo "p17: stdout cap of $SAY_MAX lines reached; the rest is in report/stdout.txt"
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

# helper <timeout_s> <log> <lines> <command> <args...>: run `frameworks.py <command> --lines <lines> <args...>`
# under the repository's locked environment, with its output in <log>. On success, say the log's first <lines>
# lines; on a failure, say one line plus the log's last two lines instead (3 lines at most).
UV_RUN=(uv run --frozen --offline)
helper() {
  local limit="$1" log="$2" lines="$3" command="$4" rc=0 line
  shift 4
  (cd "$repo" && PYTHONPATH="$repo" PYTHONDONTWRITEBYTECODE=1 timeout -k 30 "$limit" \
    "${UV_RUN[@]}" python "$here/frameworks.py" "$command" --lines "$lines" "$@") >"$log" 2>"${log%.txt}.stderr.txt" \
    || rc=$?
  if [ "$rc" -eq 0 ]; then
    say_file "$log" "$lines"
  else
    local tail_of="$log"
    [ ! -s "${log%.txt}.stderr.txt" ] || tail_of="${log%.txt}.stderr.txt"
    say "p17: frameworks.py $command failed (status $rc); raw/$(basename "$tail_of") ends:"
    while IFS= read -r line; do say "  $line"; done < <(tail -n 2 "$tail_of")
  fi
  return 0
}

# warm_up: make sure the repository's locked environment imports the sandbox, for frameworks.py. Offline first,
# so a slot fetches nothing it does not have; once more online (the repository's own lock, from PyPI) only if the
# offline attempt fails, and then the helpers run online too.
warm_up() {
  if ! (cd "$repo" && PYTHONDONTWRITEBYTECODE=1 timeout -k 30 600 uv run --frozen --offline python -c \
    'import lassi.executors.sandbox') >"$raw/uv-warmup.txt" 2>&1; then
    UV_RUN=(uv run --frozen)
    (cd "$repo" && PYTHONDONTWRITEBYTECODE=1 timeout -k 30 600 uv run --frozen python -c \
      'import lassi.executors.sandbox') >>"$raw/uv-warmup.txt" 2>&1 || true
    say "p17: the repository environment needed an online uv sync (raw/uv-warmup.txt)"
  fi
}

preflight() {
  local tool
  if ! python3 -c 'import resource,sys; sys.exit(0 if resource.getrlimit(resource.RLIMIT_CORE)==(1,1) else 1)'; then
    echo "p17: RLIMIT_CORE is not exactly (1, 1) byte; a crash could store a core on the root filesystem"
    return 1
  fi
  for tool in uv python3 git du find sort timeout prlimit systemd-run unshare realpath; do
    command -v "$tool" >/dev/null || { echo "p17: $tool is not on PATH"; return 1; }
  done
  if [ -e "$GATE_STOP" ]; then echo "p17: the owner's gate STOP is set; not starting"; return 1; fi
  [ ! -e "$run" ] || { echo "p17: $run exists already"; return 1; }
  [ ! -e "$VENV" ] || { echo "p17: $VENV exists already"; return 1; }
  case "$(git -C "$repo" log -1 --format=%s)" in
    'rx snapshot of '*) echo "p17: dirty launch (rx snapshot); relaunch from a clean commit"; return 1 ;;
  esac
}

preflight || exit 2
commit="$(git -C "$repo" rev-parse HEAD)"
changed="$(git -C "$repo" status --porcelain | wc -l)"
scratch_before="$(kib "$LASSI_SCRATCH")"
if [ "$scratch_before" -eq 0 ]; then echo "p17: du of the scratch root failed or timed out; not starting"; exit 2; fi
if [ $((scratch_before + PLANNED_KIB)) -gt "$STOP_KIB" ]; then
  echo "p17: the scratch root holds $scratch_before KiB; $PLANNED_KIB KiB more would pass the 115 GiB stop line"
  exit 2
fi
cache_before="$(kib "$UV_CACHE_DIR")"
mkdir -p "$raw" "$report"
exec 2>>"$report/stderr.txt"
trap '' PIPE
finish() {
  local rc=$?
  local note=""
  [ "$rc" -eq 0 ] || note="; see report/stderr.txt"
  echo "p17: ended with status $rc after $SECONDS s$note; pull: PYTHONIOENCODING=utf-8 uv run tools/rx.py pull --path" \
    "lassi-runs/p17-frameworks/$LASSI_RX_RUN_ID/report --into results/p17-frameworks"
}
trap finish EXIT

# 0. Header.
mode=real
[ "$STAND_IN" = 0 ] || mode="STAND-IN (local dry run; not evidence)"
{
  echo "mode: $mode"
  echo "rx id $LASSI_RX_RUN_ID; run $run"
  echo "date (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ); host $(hostname); kernel $(uname -r); nproc $(nproc)"
  echo "commit $commit; tree $([ "$changed" -eq 0 ] && echo clean || echo "dirty ($changed changed paths)")"
  echo "python3: $(command -v python3) $(python3 --version 2>&1)"
  echo "uv: $(command -v uv) $(uv --version 2>&1)"
  echo "uv cache before: $cache_before KiB at $UV_CACHE_DIR; scratch root $scratch_before KiB" \
    "(stop line $STOP_KIB KiB)"
  echo "core limit: 1 byte (prlimit --core=1:1, checked with getrlimit); variables unset:${unset_names:- none}"
  echo "UV_LINK_MODE=${UV_LINK_MODE:-unset (uv default)}; TMPDIR=${TMPDIR:-unset}"
} >"$report/batch.txt"
say_file "$report/batch.txt" 9

# 1. The venv, with the system Python (no managed Python, no download).
venv_rc=0
timeout -k 30 300 uv venv --no-config --python "$PY_REQUEST" --no-managed-python --no-python-downloads "$VENV" \
  >"$raw/venv.log" 2>&1 || venv_rc=$?
venv_python="$("$VENV/bin/python" -c 'import sys; print(sys.version.split()[0], sys.base_prefix)' 2>/dev/null \
  || echo none)"
say "venv: rc=$venv_rc at $VENV; python $venv_python"

if [ "$venv_rc" -eq 0 ]; then
  # 2. The resolution record, with the install's own index flags.
  printf '%s\n' "${PACKAGES[@]}" >"$report/requirements.in"
  compile_rc=0
  timeout -k 30 300 uv pip compile --no-config --python "$VENV/bin/python" --index-url "$TORCH_INDEX" \
    --extra-index-url "$PYPI_INDEX" --index-strategy unsafe-best-match --no-build --emit-index-annotation \
    -o "$report/resolved.txt" "$report/requirements.in" >"$raw/compile.log" 2>&1 || compile_rc=$?
  pins="$(grep -c '==' "$report/resolved.txt" 2>/dev/null || true)"
  say "resolve (uv pip compile, index annotations): rc=$compile_rc; ${pins:-0} pins"

  # 3. The install: the pins researcher's command, plus --no-config and --no-build (wheels only).
  install_rc=0
  install_start=$SECONDS
  timeout -k 30 "$INSTALL_S" uv pip install --no-config --python "$VENV/bin/python" --index-url "$TORCH_INDEX" \
    --extra-index-url "$PYPI_INDEX" --index-strategy unsafe-best-match --no-build "${PACKAGES[@]}" \
    >"$raw/install.log" 2>&1 \
    || install_rc=$?
  install_s=$((SECONDS - install_start))
  tail -n 80 "$raw/install.log" >"$report/install-tail.txt"
  say "install: rc=$install_rc in $install_s s ($INSTALL_S s limit); $(grep -E '^(Resolved|Prepared|Installed)' \
    "$raw/install.log" | tr '\n' ';' | cut -c1-150)"

  # 4. The measures, all taken before the repository's own environment is warmed up (that may add to the cache).
  timeout -k 10 120 uv pip freeze --no-config --python "$VENV/bin/python" >"$report/freeze.txt" \
    2>>"$raw/freeze.log" || true
  check_rc=0
  timeout -k 10 120 uv pip check --no-config --python "$VENV/bin/python" >"$report/pip-check.txt" 2>&1 || check_rc=$?
  venv_kib="$(kib "$VENV")"
  venv_apparent="$( (du -sk --apparent-size "$VENV" 2>/dev/null || echo 0) | cut -f1)"
  cache_after="$(kib "$UV_CACHE_DIR")"
  together="$(kib "$UV_CACHE_DIR" "$VENV")"
  files="$(find "$VENV" -type f | wc -l)"
  linked="$(find "$VENV" -type f -links +1 | wc -l)"
  du -sk "$VENV"/lib/python*/site-packages/* 2>/dev/null | sort -rn | head -n 12 >"$report/site-packages-top.txt" \
    || true
  {
    echo "venv_kib $venv_kib"
    echo "venv_apparent_kib $venv_apparent"
    echo "uv_cache_before_kib $cache_before"
    echo "uv_cache_after_kib $cache_after"
    echo "uv_cache_growth_kib $((cache_after - cache_before))"
    echo "cache_and_venv_kib $together"
    echo "disk_added_kib $((together - cache_before))"
    echo "venv_files $files"
    echo "venv_files_hard_linked $linked"
    echo "install_s $install_s"
  } >"$report/sizes.txt"
  warm_up
  expect_args=()
  for pin in "${EXPECT[@]}"; do expect_args+=(--expect "$pin"); done
  helper 120 "$raw/helper-freeze.txt" 4 freeze --freeze "$report/freeze.txt" --compiled "$report/resolved.txt" \
    "${expect_args[@]}" --report "$report"
  say "pip check: rc=$check_rc; $(tail -n 1 "$report/pip-check.txt" | cut -c1-150)"
  say "size: venv $venv_kib KiB (apparent $venv_apparent KiB; $linked of $files files hard-linked to the uv cache)"
  added=$((together - cache_before))
  say "  uv cache $cache_before -> $cache_after KiB; disk added (cache growth + venv-only blocks) $added KiB"
  say "  largest site-packages entries (KiB): $(head -n 5 "$report/site-packages-top.txt" \
    | sed 's|\t.*/|=|' | tr '\n' ' ')"

  # 5. The imports, in the sandbox.
  helper 700 "$raw/helper-imports.txt" 8 imports --venv "$VENV" --raw "$raw" --report "$report" "${MODULES[@]}"
else
  say "p17: no venv, so no install, no measures, no imports (raw/venv.log ends: $(tail -n 1 "$raw/venv.log"))"
  warm_up
fi

# 6. hipcc, in the sandbox. Nothing here runs a ROCm program outside it.
if [ -x "$ROCM/bin/hipcc" ]; then
  helper 1500 "$raw/helper-hip.txt" 23 hip --rocm "$ROCM" --source "$here/probe.hip" \
    --strace "$(command -v strace || true)" --raw "$raw" --report "$report"
else
  say "hip: $ROCM/bin/hipcc is missing or not executable; no compile probe"
fi

# 7. Closing checks.
changed_after="$(git -C "$repo" status --porcelain | wc -l)"
tree_after=clean
[ "$changed_after" -eq 0 ] || tree_after="dirty ($changed_after paths)"
say "close: slot tree $tree_after after the batch; run directory $(kib "$run") KiB; report $(kib "$report") KiB"
(cd "$repo" && PYTHONDONTWRITEBYTECODE=1 timeout -k 10 120 "${UV_RUN[@]}" python "$here/frameworks.py" ascii \
  --dir "$report") \
  >"$raw/ascii.txt" 2>&1 || true
say "$(tail -n 1 "$raw/ascii.txt")"
