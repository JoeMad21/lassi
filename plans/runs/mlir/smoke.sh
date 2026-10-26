#!/usr/bin/env bash
# Remote smoke of the C++ -> TT MLIR path (task P17.36), for the owner's window: on alpha01, from the repository
# root of a clean checkout (an rx run slot), with the gate's environment (LASSI_TOOLCHAINS, LASSI_RUNS_ROOT,
# LASSI_SCRATCH, TMPDIR). Run it as: bash plans/runs/mlir/smoke.sh
#
# It checks that the pinned tt-mlir install exists, then lowers one hand-written ttkernel kernel module, written
# here (a one-page DRAM-to-DRAM copy through an L1 scratch page, a SYNTHETIC kernel that is no tt-pairs-v0 item),
# through the ttkernel-ttmetal toolchain's exact commands, builds it with a hand-written host program, and runs the
# program on ttsim, a simulator (never silicon; its wall time is never performance, Agent Rule 2). Each step prints
# "step <n> <name>: PASS|FAIL|SKIP <detail>". The smoke's files go under $LASSI_RUNS_ROOT/mlir-smoke/<stamp>/; the
# sandboxed --version checks use short-lived directories under $TMPDIR, and uv its own cache, as every run does. No
# model is called. Exit status: 0 when every step passed, 1 when one failed, 2 on a missing variable or core limit,
# 3 when tt-mlir is not installed. Running it on alpha01 needs the owner's yes in the working session (Agent Rule 17).
set -u
fail=0
report() { echo "step $1: $2 ${3:-}"; [ "$2" = FAIL ] && fail=1; return 0; }
for name in LASSI_TOOLCHAINS LASSI_RUNS_ROOT LASSI_SCRATCH TMPDIR; do
  if [ -z "${!name:-}" ]; then echo "smoke: $name is not set (the gate sets it on alpha01)" >&2; exit 2; fi
done
# The 1-byte core limit every build and run sets (AGENTS.md, Local Execution; toolchains/tt-mlir.sh), read back.
if ! prlimit --pid $$ --core=1:1 \
  || ! read -r core_soft core_hard < <(prlimit --pid $$ --core --raw --noheadings --output SOFT,HARD) \
  || [ "$core_soft $core_hard" != "1 1" ]; then
  echo "smoke: cannot hold RLIMIT_CORE at exactly 1 byte, soft and hard (prlimit, read back as 1 and 1)" >&2
  exit 2
fi
echo "smoke: core limit read back as ($core_soft, $core_hard)"
# The pins, read as their install scripts read them.
mlir_pin() { (source toolchains/tt-mlir.pin && eval "echo \"\$$1\""); }
metal_pin() { (source toolchains/tt-metal.pin && eval "echo \"\$$1\""); }
mlir_home="$LASSI_TOOLCHAINS/$(mlir_pin PREFIX_NAME)"
mlir_commit="$(mlir_pin COMMIT)"
opt="$mlir_home/$(mlir_pin TTMLIR_OPT)"
translate="$mlir_home/$(mlir_pin TTMLIR_TRANSLATE)"
expect="$(mlir_pin EXPECT_VERSION)"
metal_home="$LASSI_TOOLCHAINS/$(metal_pin PREFIX_NAME)"
echo "smoke: commit $(git rev-parse HEAD 2>/dev/null) dirty=$([ -z "$(git status --porcelain 2>/dev/null)" ] && echo no || echo yes)"
echo "smoke: host $(hostname) date $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 1. The tt-mlir install: the prefix, its install record naming the pinned commit, and the two tools.
first="$(head -n 1 "$mlir_home/lassi-install.txt" 2>/dev/null | cut -d' ' -f1-2)"
if [ "$first" != "tt-mlir $mlir_commit" ]; then
  detail="$mlir_home/lassi-install.txt reads '${first:-nothing}', not 'tt-mlir $mlir_commit'"
  [ -e "$mlir_home/lassi-install.unfinished" ] && detail="$detail; the tree is unfinished (lassi-install.unfinished)"
  [ -d "$mlir_home" ] || detail="$mlir_home does not exist; install it with toolchains/tt-mlir.sh (task P5.4)"
  report "1 tt-mlir install" FAIL "$detail"
  exit 3
fi
report "1 tt-mlir install" PASS "$mlir_home ($first)"
for tool in "$opt" "$translate"; do
  banner="$("$tool" --version 2>&1)"
  if printf '%s\n' "$banner" | grep -qF "$expect"; then
    report "2 $(basename "$tool") --version" PASS "prints '$expect'"
  else
    report "2 $(basename "$tool") --version" FAIL "does not print '$expect': $(printf '%s' "$banner" | head -n 3 | tr '\n' ' ')"
  fi
done

# 3. The tt-metal tree the host build and the kernel JIT use.
first="$(head -n 1 "$metal_home/lassi-install.txt" 2>/dev/null | cut -d' ' -f1-2)"
if [ "$first" = "tt-metal $(metal_pin COMMIT)" ]; then report "3 tt-metal install" PASS "$metal_home"
else report "3 tt-metal install" FAIL "$metal_home/lassi-install.txt reads '${first:-nothing}'"; fi

work="$LASSI_RUNS_ROOT/mlir-smoke/$(date -u +%Y%m%d-%H%M%S)-$$"
mkdir -p "$work/src" "$work/manual/kernels" "$work/manual/_ttkernel_emitc/kernels"
echo "smoke: work directory $work"

cat >"$work/src/copy.mlir" <<'MLIR'
// SYNTHETIC ttkernel module written for the P17.36 smoke; no tt-pairs-v0 item. One data-movement kernel copies one
// page from page 0 of one DRAM buffer to page 0 of another through an L1 scratch page. Runtime arguments, in order:
// the source DRAM address, the destination DRAM address, the L1 scratch address, and the byte count.
func.func @copy() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %src = ttkernel.get_arg_val(%c0) : (i32) -> i32
  %dst = ttkernel.get_arg_val(%c1) : (i32) -> i32
  %scratch = ttkernel.get_arg_val(%c2) : (i32) -> i32
  %bytes = ttkernel.get_arg_val(%c3) : (i32) -> i32
  %src_noc = ttkernel.get_noc_addr_from_bank_id(%c0, %src) : (i32, i32) -> !ttkernel.noc_addr
  ttkernel.noc_async_read(%src_noc, %scratch, %bytes) : (!ttkernel.noc_addr, i32, i32) -> ()
  ttkernel.noc_async_read_barrier() : () -> ()
  %dst_noc = ttkernel.get_noc_addr_from_bank_id(%c0, %dst) : (i32, i32) -> !ttkernel.noc_addr
  ttkernel.noc_async_write(%scratch, %dst_noc, %bytes) : (i32, !ttkernel.noc_addr, i32) -> ()
  ttkernel.noc_async_write_barrier() : () -> ()
  func.return
}
MLIR

cat >"$work/src/smoke_copy.cpp" <<'CPP'
// SYNTHETIC host program written for the P17.36 smoke; no tt-pairs-v0 item. It writes one 2048-byte page to a DRAM
// buffer, runs the copy kernel on core (0, 0), reads the other DRAM buffer back, and checks the copy.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

using namespace tt::tt_metal;

int main() {
    constexpr uint32_t kBytes = 2048;
    std::vector<uint32_t> input(kBytes / sizeof(uint32_t));
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<uint32_t>(i) * 2654435761u + 12345u;
    }
    std::shared_ptr<distributed::MeshDevice> device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = device->mesh_command_queue();
    distributed::DeviceLocalBufferConfig dram_page{.page_size = kBytes, .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig l1_page{.page_size = kBytes, .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig one_page{.size = kBytes};
    auto src = distributed::MeshBuffer::create(one_page, dram_page, device.get());
    auto dst = distributed::MeshBuffer::create(one_page, dram_page, device.get());
    auto scratch = distributed::MeshBuffer::create(one_page, l1_page, device.get());
    distributed::EnqueueWriteMeshBuffer(cq, src, input, false);

    Program program = CreateProgram();
    const CoreCoord core = {0, 0};
    KernelHandle copier = CreateKernel(
        program,
        "kernels/copy.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    const std::vector<uint32_t> run_args = {
        static_cast<uint32_t>(src->address()),
        static_cast<uint32_t>(dst->address()),
        static_cast<uint32_t>(scratch->address()),
        kBytes};
    SetRuntimeArgs(program, copier, core, run_args);
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> output;
    distributed::EnqueueReadMeshBuffer(cq, output, dst, true);
    const bool closed = device->close();
    const bool matched = output == input;
    std::printf("smoke: copy %s (%zu words; device closed: %s)\n", matched ? "matched" : "DIFFERS", output.size(),
                closed ? "yes" : "no");
    return matched && closed ? 0 : 1;
}
CPP

# 4. The toolchain's three commands, run by hand on the module (outside the sandbox), each with its status.
cp "$work/src/copy.mlir" "$work/manual/kernels/copy.mlir"
(
  cd "$work/manual" || exit 1
  steps=(
    "verify|$opt kernels/copy.mlir -o _ttkernel_emitc/kernels/copy.verified.mlir"
    "lower|$opt --convert-ttkernel-to-emitc --mlir-print-debuginfo -o _ttkernel_emitc/kernels/copy.emitc.mlir kernels/copy.mlir"
    "translate|$translate --ttkernel-to-cpp -o kernels/copy.cpp _ttkernel_emitc/kernels/copy.emitc.mlir"
  )
  for step in "${steps[@]}"; do
    name="${step%%|*}"; command="${step#*|}"
    echo "smoke: \$ $command"
    # shellcheck disable=SC2086
    if $command 2>"$name.stderr"; then echo "step 4 $name: PASS"; else echo "step 4 $name: FAIL (see $work/manual/$name.stderr)"; cat "$name.stderr"; exit 1; fi
  done
  entries="$(grep -c 'void kernel_main' kernels/copy.cpp)"
  echo "smoke: kernels/copy.cpp holds $entries kernel_main; its first 30 lines:"
  head -n 30 kernels/copy.cpp
  [ "$entries" = 1 ]
) && report "4 manual lowering" PASS "$work/manual/kernels/copy.cpp" || report "4 manual lowering" FAIL

# 5 to 8. The same through the toolchain as a run builds it: build_toolchain (the sandboxed --version checks of
# clang++-20, ttmlir-opt, and ttmlir-translate), build() in the compile sandbox, a broken module's parsed
# diagnostics, and the program on ttsim through the ttsim executor.
cat >"$work/driver.py" <<'PY'
"""Drive the P17.36 smoke's toolchain build and ttsim run; SYNTHETIC inputs only (see smoke.sh)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

from lassi.core.interfaces import BuildResult, Limits  # noqa: E402
from lassi.core.runner import RunError, build_toolchain  # noqa: E402
from lassi.executors.sandbox import SandboxUnavailableError  # noqa: E402
from lassi.executors.ttsim import TtsimExecutor  # noqa: E402

KERNEL = "kernels/copy.mlir"
HOST = "smoke_copy.cpp"


def report(step: str, ok: bool, detail: str = "") -> bool:
    """Print one step's status line and return whether it passed."""
    print(f"step {step}: {'PASS' if ok else 'FAIL'} {detail}", flush=True)
    return ok


def show(result: BuildResult) -> None:
    """Print every diagnostic of a build."""
    for item in result.diagnostics:
        print(f"  diagnostic: stage={item.stage} severity={item.severity} file={item.file} line={item.line} "
              f"column={item.column} code={item.code} message={item.message}", flush=True)


def main() -> int:
    """Run steps 5 to 8; return 0 when all passed."""
    work = Path(sys.argv[1])
    runs_root, root = Path(os.environ["LASSI_RUNS_ROOT"]), Path(os.environ["LASSI_TOOLCHAINS"])
    try:
        built = build_toolchain("ttkernel-ttmetal", root, build_root=runs_root)
    except (RunError, SandboxUnavailableError) as error:
        report("5 build_toolchain", False, str(error))
        return 1
    toolchain = built.toolchain
    report("5 build_toolchain", True, f"pins {sorted(built.pins)}; {toolchain.opt}; {toolchain.translate}")
    kernel = (work / "src" / "copy.mlir").read_text(encoding="utf-8")
    harness = {HOST: (work / "src" / HOST).read_text(encoding="utf-8")}
    build = work / "build"
    build.mkdir()
    result = toolchain.build({KERNEL: kernel}, build, harness=harness)
    show(result)
    ok = report("6 toolchain build", result.artifact is not None, f"{build}; attachment {build / 'compile.stderr'}")
    emitted = build / "kernels" / "copy.cpp"
    if emitted.is_file():
        print("".join(emitted.read_text(encoding="utf-8").splitlines(keepends=True)[:30]), flush=True)
    bad = work / "bad"
    bad.mkdir()
    broken = kernel.replace("ttkernel.noc_async_read_barrier()", "ttkernel.no_such_op()")
    failed = toolchain.build({KERNEL: broken}, bad, harness=harness)
    show(failed)
    placed = [item for item in failed.diagnostics if item.file == KERNEL and item.line and item.severity == "error"]
    ok &= report("7 parsed diagnostics", failed.artifact is None and bool(placed),
                 "an error with file and line" if placed else "no error with a file and line was parsed")
    if result.artifact is None:
        report("8 run on ttsim", False, "skipped: no program")
        return 1
    try:
        run = TtsimExecutor().run(Path(result.artifact), [], Limits(wall_s=120.0, memory_mb=8192, cpus=16))
    except (SandboxUnavailableError, ValueError) as error:
        report("8 run on ttsim", False, f"the executor refused the run: {error}")
        return 1
    for item in run.diagnostics:
        print(f"  run diagnostic: {item.stage} {item.severity} {item.file}:{item.line} {item.message}", flush=True)
    print(f"  exit={run.exit_code} hang={run.hang} sim_ub={run.sim_ub} sim_gap={run.sim_gap} "
          f"simulator wall time {run.wall_s:.2f} s (ttsim, exploratory, never performance)", flush=True)
    print("  stdout tail: " + " | ".join(run.stdout.splitlines()[-5:]), flush=True)
    print("  stderr tail: " + " | ".join(run.stderr.splitlines()[-8:]), flush=True)
    jit = [item for item in run.diagnostics if item.stage == "jit" and item.severity == "error"]
    clean = run.exit_code == 0 and not run.hang and not run.sim_ub and run.sim_gap is None and not jit
    ok &= report("8 run on ttsim", clean and "smoke: copy matched" in run.stdout,
                 "the emitted kernel JIT-compiled and the copy matched" if clean else "see the lines above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
PY
uv run python "$work/driver.py" "$work" || fail=1
echo "smoke: done; work directory $work; $([ "$fail" = 0 ] && echo 'every step passed' || echo 'a step FAILED')"
exit "$fail"
