#!/usr/bin/env bash
# Remote probe of the C++ -> MLIR lowering step (task P17.41), for the owner's window: on alpha01, from the repository
# root of a clean checkout (an rx run slot), with the gate's environment (LASSI_TOOLCHAINS, LASSI_RUNS_ROOT,
# LASSI_SCRATCH, TMPDIR). Run it as: bash plans/runs/mlir/lowering-probe.sh
#
# It reads the five tt-pairs-v0-ttkernel compute units (each item's `compute`: its computation as one function with C
# linkage and no I/O), and for each:
#   1. runs the pinned cgeist by hand, as cgeist-affine's command line has it, with and without
#      --raise-scf-to-affine (the latter for comparison only), and reads the module back through the pinned
#      polygeist-opt (parse and verify, output discarded);
#   2. asks the pinned polygeist-opt for --raise-affine-to-linalg, which the spike found missing at this pin
#      (plans/spikes/mlir-issues.md, Half 2, finding 3), to record that linalg is not available;
#   3. runs the frontend step itself, lassi.toolchains.cgeist CgeistAffine built by lassi.core.runner build_toolchain
#      (the --version check against toolchains/polygeist.pin, then each lowering in the compile sandbox), exactly as
#      a source_mlir run lowers a unit.
# It prints "step <item> <name>: rc=<n> ..." lines with each module's ops by dialect, writes only under
# $LASSI_RUNS_ROOT/lowering-probe/<stamp>/, starts no model, and opens no device (CPU only). The outputs are
# exploratory, never [MEASURED] unless run from a clean commit and recorded with provenance. Running it on alpha01
# needs the owner's yes in the working session (Agent Rule 17).
set -u
for name in LASSI_TOOLCHAINS LASSI_RUNS_ROOT LASSI_SCRATCH TMPDIR; do
  if [ -z "${!name:-}" ]; then echo "probe: $name is not set (the gate sets it on alpha01)" >&2; exit 2; fi
done
# The 1-byte core limit every build and run sets (AGENTS.md, Local Execution), read back.
if ! prlimit --pid $$ --core=1:1 \
  || ! read -r core_soft core_hard < <(prlimit --pid $$ --core --raw --noheadings --output SOFT,HARD) \
  || [ "$core_soft $core_hard" != "1 1" ]; then
  echo "probe: cannot hold RLIMIT_CORE at exactly 1 byte, soft and hard" >&2
  exit 2
fi
pin() { (source toolchains/polygeist.pin && eval "echo \"\$$1\""); }
home="$LASSI_TOOLCHAINS/$(pin PREFIX_NAME)"
cgeist="$home/$(pin CGEIST)"
popt="$home/$(pin POLYGEIST_OPT)"
out="$LASSI_RUNS_ROOT/lowering-probe/$(date -u +%Y%m%d-%H%M%S)-$$"
mkdir -p "$out/hand"
echo "probe: commit $(git rev-parse HEAD 2>/dev/null) dirty=$([ -z "$(git status --porcelain 2>/dev/null)" ] && echo no || echo yes)"
echo "probe: host $(hostname) date $(date -u +%Y-%m-%dT%H:%M:%SZ) out=$out"
echo "probe: cgeist $("$cgeist" --version 2>&1 | grep -m1 'LLVM version')"
ops() { grep -oE '(affine|scf|memref|arith|math|func|llvm|linalg|cf|polygeist)\.[a-z_]+' "$1" | cut -d. -f1 | sort \
  | uniq -c | tr -s ' ' | tr '\n' ';'; }
# The units, one "<item> <tracked path> <function>" line each, read from the suite manifest by the registry.
units=$(uv run python -c '
from pathlib import Path
from lassi.bench import load_suite
suite = load_suite(Path("assets/bench/tt-pairs-v0-ttkernel.yaml"))
for name, item in sorted(suite.items.items()):
    print(name, item.compute.path, item.compute.function)
') || { echo "probe: cannot read the compute units" >&2; exit 2; }
while read -r item path function; do
  for mode in affine scf; do
    flag=""; [ "$mode" = affine ] && flag="--raise-scf-to-affine"
    module="$out/hand/$item.$mode.mlir"
    timeout 300 "$cgeist" "$path" -S $flag --function="$function" -o "$module" > "$module.err" 2>&1; rc=$?
    echo "step $item cgeist-$mode: rc=$rc ops: $([ -s "$module" ] && ops "$module")"
    [ -s "$module.err" ] && echo "   err: $(head -c 400 "$module.err" | tr '\n' ' ')"
    if [ -s "$module" ]; then
      timeout 300 "$popt" "$module" -o /dev/null > "$module.verify.err" 2>&1
      echo "step $item polygeist-opt-verify-$mode: rc=$?"
    fi
  done
done <<< "$units"
first=$(ls "$out"/hand/*.affine.mlir 2>/dev/null | head -n 1)
if [ -n "$first" ]; then
  timeout 300 "$popt" "$first" --raise-affine-to-linalg -o "$out/hand/linalg.mlir" > "$out/hand/linalg.err" 2>&1
  echo "step linalg polygeist-opt --raise-affine-to-linalg: rc=$? $(head -c 200 "$out/hand/linalg.err" | tr '\n' ' ')"
fi
# 3. The frontend step as a run takes it: cgeist-affine through build_toolchain and the compile sandbox.
uv run python - "$out" <<'PY'
import os
import sys
from pathlib import Path

from lassi.bench import load_suite
from lassi.core.runner import build_toolchain

out = Path(sys.argv[1]) / "frontend"
out.mkdir()
suite = load_suite(Path("assets/bench/tt-pairs-v0-ttkernel.yaml"))
built = build_toolchain("cgeist-affine", Path(os.environ["LASSI_TOOLCHAINS"]), build_root=out)
print(f"step frontend build_toolchain: executable {built.executable}; version {' | '.join(built.version)}")
for name in sorted(suite.items):
    unit = suite.compute_source(name, purpose="eval")
    work = out / name / "build"
    work.mkdir(parents=True)
    lowering = built.toolchain.lower(unit.file, unit.text, unit.function, work)
    size = "none" if lowering.module is None else f"{len(lowering.module)} characters"
    errors = [f"{item.code}: {item.message}" for item in lowering.diagnostics if item.severity == "error"]
    print(f"step {name} frontend: module {size}; level {lowering.level}; errors {errors or 'none'}")
PY
echo "step frontend: rc=$?"
echo "out=$out"
