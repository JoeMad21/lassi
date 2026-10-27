#!/usr/bin/env bash
# Remote check of the ttkernel host programs of task P17.37 (built under OQ-103 (a) by task P17.41), for the owner's
# window: on alpha01, from the repository root of a clean checkout (an rx run slot), with the gate's environment
# (LASSI_TOOLCHAINS, LASSI_RUNS_ROOT, LASSI_SCRATCH, TMPDIR), after
#   uv run tools/fetch_bench.py assets/bench/tt-pairs-v0-ttkernel.yaml
# Run it as: bash plans/runs/mlir/ttkernel-hosts.sh
#
# For eltwise_binary, eltwise_sfpu, matmul_single_core, and matmul_multi_core it builds two programs with the
# item's upstream kernels (the tt language's fetched support files) and the ttmetal-host toolchain as a run builds
# one (lassi.core.runner build_toolchain: the pinned clang++-20 against the pinned tt-metal tree, in the compile
# sandbox): the item's tt host program, and its ttkernel host program, which appends each kernel's circular buffer
# indices to its compile-time arguments. It runs both through the ttsim executor (a simulator, never silicon; its wall
# time is never performance, Agent Rule 2) on the item's held-out inputs (lassi.bench stage_inputs), and compares
# their output files byte for byte. The upstream kernels read only the compile-time arguments before the appended
# ones (inferred from the host programs, which pass them nothing else), so the two must agree: "step <item>: PASS"
# when they do and both runs exited 0 with no hang, gap, or undefined behavior, else FAIL with the reason. It writes
# only under $LASSI_RUNS_ROOT/ttkernel-hosts/<stamp>/ and starts no model. Exit status 0 when every item passed, 1
# otherwise, 2 on a missing variable or core limit. Running it on alpha01 needs the owner's yes in the working
# session (Agent Rule 17).
set -u
for name in LASSI_TOOLCHAINS LASSI_RUNS_ROOT LASSI_SCRATCH TMPDIR; do
  if [ -z "${!name:-}" ]; then echo "hosts: $name is not set (the gate sets it on alpha01)" >&2; exit 2; fi
done
if ! prlimit --pid $$ --core=1:1 \
  || ! read -r core_soft core_hard < <(prlimit --pid $$ --core --raw --noheadings --output SOFT,HARD) \
  || [ "$core_soft $core_hard" != "1 1" ]; then
  echo "hosts: cannot hold RLIMIT_CORE at exactly 1 byte, soft and hard" >&2
  exit 2
fi
out="$LASSI_RUNS_ROOT/ttkernel-hosts/$(date -u +%Y%m%d-%H%M%S)-$$"
mkdir -p "$out"
echo "hosts: commit $(git rev-parse HEAD 2>/dev/null) dirty=$([ -z "$(git status --porcelain 2>/dev/null)" ] && echo no || echo yes)"
echo "hosts: host $(hostname) date $(date -u +%Y-%m-%dT%H:%M:%SZ) out=$out"
uv run python - "$out" <<'PY'
import os
import sys
from pathlib import Path

from lassi.bench import Direction, load_suite, sources_dir, stage_inputs
from lassi.core.interfaces import Limits
from lassi.core.runner import build_toolchain
from lassi.executors.ttsim import TtsimExecutor

out = Path(sys.argv[1])
suite = load_suite(Path("assets/bench/tt-pairs-v0-ttkernel.yaml"))
sources = sources_dir(Path(os.environ["LASSI_SCRATCH"]), suite)
built = build_toolchain("ttmetal-host", Path(os.environ["LASSI_TOOLCHAINS"]), build_root=out)
executor = TtsimExecutor()
executor.install()
limits = Limits(wall_s=1800.0, memory_mb=16384, cpus=16)
print(f"hosts: device {executor.device()}; compiler {built.executable}")
failed = 0
for name in ("eltwise_binary", "eltwise_sfpu", "matmul_single_core", "matmul_multi_core"):
    item = suite.item(name, purpose="eval")
    host = item.languages["tt"].files[0]
    kernels = suite.support_files(name, sources, purpose="eval", language="tt")
    programs = {
        "tt": suite.reference_target(name, Direction("cpp", "tt"), sources, purpose="eval"),
        "ttkernel-host": {host: suite.support_files(name, sources, purpose="eval", language="ttkernel")[host]},
    }
    runs, reasons = {}, []
    for label, files in programs.items():
        work = out / name / label / "build"
        work.mkdir(parents=True)
        result = built.toolchain.build(files, work, harness=kernels)
        if result.artifact is None:
            reasons.append(f"{label} did not build ({len(result.diagnostics)} diagnostics; {work})")
            continue
        run = executor.run(result.artifact, stage_inputs(item, work), limits)
        runs[label] = run
        if run.exit_code != 0 or run.hang or run.sim_gap is not None or run.sim_ub:
            reasons.append(f"{label} run: exit {run.exit_code}, hang {run.hang}, gap {run.sim_gap}, ub {run.sim_ub}")
    if len(runs) == 2:
        tt, kk = runs["tt"].output_files, runs["ttkernel-host"].output_files
        if not tt or not kk:
            reasons.append(f"a run wrote no output file ({len(tt)} and {len(kk)})")
        elif set(tt) != set(kk):
            reasons.append(f"output files differ: {sorted(tt)} vs {sorted(kk)}")
        for file in sorted(set(tt) & set(kk)):
            if Path(tt[file]).read_bytes() != Path(kk[file]).read_bytes():
                reasons.append(f"{file} differs byte for byte")
    failed += bool(reasons)
    print(f"step {name}: {'FAIL ' + '; '.join(reasons) if reasons else 'PASS'}")
sys.exit(1 if failed else 0)
PY
status=$?
echo "out=$out"
exit $status
