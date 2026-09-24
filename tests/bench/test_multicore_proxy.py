"""Remote check of the OpenMP references under the nvc++ multicore proxy, built and run on the host CPU (DEMO.1).

Bible: Execution Backends (Harness Contract, the bullet "CUDA -> OMP proxy
without a GPU: build a second binary with `nvc++ -mp=multicore` so target
regions run on the host. It checks outputs, never runtime."; Sandbox),
Risks And Questions (OQ-003: the compile-only tier plus the -mp=multicore
proxy), Benchmark Suites (lassi-hecbench-10), Agent Rules 5, 6, and 7.

What the test does, on the build host only:

- tools/fetch_bench.py materializes the pinned lassi-hecbench-10 sources
  under $LASSI_SCRATCH/bench (it is idempotent and checks every sha256),
  and the bench registry reads each item's OpenMP reference and support
  files from there, as tests/bench/test_hecbench10.py does.
- Each of the 10 OpenMP reference mains compiles with Toolchain
  "nvcpp-multicore", built by build_toolchain exactly as a run builds it:
  the pinned nvc++ in the compile sandbox. 10/10 must compile.
- Each built main runs once with the manifest's run_args through the
  registered "native" executor, under LIMITS (180 s wall). Its sandbox's
  runner is wrapped only to record each command: every run must go through
  the sandbox command (prlimit, systemd-run, unshare, then the artifact and
  its arguments), so no run escapes the sandbox (Agent Rule 6), and must
  carry OMP_NUM_THREADS=<LIMITS.cpus> in the program's environment, before
  the artifact (DEMO.2). A SandboxUnavailableError is never caught.
- One line per app is printed: compile ok, exit code, wall seconds, the
  timeout (hang) flag, stdout bytes, and, for the apps whose OpenMP program
  prints PASS or FAIL (the manifest's passfail), whether PASS appeared; a
  run with a nonzero or missing exit code adds the tail of its stderr. Run
  outcomes are reported, never asserted.

The limits are test inputs. The run's environment is the native executor's
(DEMO.2, tests/executors/test_native_threads.py): the sandbox default with
OMP_NUM_THREADS set to LIMITS.cpus, so the OpenMP runtime starts that many
threads rather than one per host CPU (on the build host one per CPU passed
the sandbox's TasksMax, plans/spikes/demo-multicore-proxy.md), and the
CPU-time budget is wall x cpus (Sandbox, OQ-011): a run that spends it ends
by a signal, and its exit code shows that. Wall seconds are
exploratory: the proxy checks outputs, never runtime, and no value this
test prints is a performance number. Output from a dirty tree is
exploratory; only a clean-commit rx run is evidence.

The references are pinned benchmark sources, not model output, and they
still run only in the sandbox. Building and running a reference is harness
work, never training or tuning on the eval items (Agent Rule 5). The build
directories live in a temporary directory under $LASSI_RUNS_ROOT, which the
test removes afterwards (Agent Rule 7). Without the build host the test
skips; with LASSI_REQUIRE_SANDBOX=1 it fails instead. Run it with
`uv run tools/rx.py run -- 'LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -s -m remote tests/bench/test_multicore_proxy.py'`.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lassi.bench import Direction, load_suite, sources_dir
from lassi.core.interfaces import Limits, RunResult
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.core.runner import build_toolchain
from lassi.executors import Sandbox
from lassi.toolchains import CommandResult, capped_runner

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "assets" / "bench" / "lassi-hecbench-10.yaml"
FETCH_BENCH = REPO / "tools" / "fetch_bench.py"

SUITE = "lassi-hecbench-10"
TOOLCHAIN = "nvcpp-multicore"
EXECUTOR = "native"
LANGUAGE = "omp"
DIRECTION = Direction(source="cuda", target=LANGUAGE)
APPS = (
    "atomicCost",
    "bsearch",
    "colorwheel",
    "dense-embedding",
    "entropy",
    "jacobi",
    "layout",
    "matrix-rotate",
    "pathfinder",
    "randomAccess",
)
# The task's wall limit; cpus sets the CPU-time budget. Memory: atomicCost's OpenMP main allocates two arrays of
# 922521600 doubles (about 14.8 GB, read from its source), so the limit is 32768 MB rather than the compile
# sandbox's 8192 MB (COMPILE_MEMORY_MB); the build host has far more.
LIMITS = Limits(wall_s=180.0, memory_mb=32768, cpus=16)
# The program-environment element that bounds each run's OpenMP threads by LIMITS.cpus (the native executor, DEMO.2).
THREAD_BOUND = f"OMP_NUM_THREADS={LIMITS.cpus}"
# The head of every sandboxed command (lassi.executors.sandbox.sandbox_command) and the tools it must hold.
SANDBOX_HEAD = ["prlimit", "--core=1", "--"]
SANDBOX_TOOLS = ("systemd-run", "unshare")
REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"
# How much of a failed run's stderr the report keeps (its last characters, blanks folded), so a failure says why.
STDERR_TAIL_CHARS = 240


def host_problem() -> str:
    """Return why this host cannot fetch, compile, and run the suite, or "" when it can."""
    if not sys.platform.startswith("linux"):
        return f"needs the build host (Linux), not {sys.platform}; run it through `uv run tools/rx.py run`"
    for name in ("LASSI_SCRATCH", "LASSI_RUNS_ROOT", "LASSI_TOOLCHAINS", "TMPDIR"):
        if not os.environ.get(name):
            return f"{name} is not set; run it through `uv run tools/rx.py run`"
    return ""


def fetch_tool() -> ModuleType:
    """Import tools/fetch_bench.py by path and return the module."""
    spec = importlib.util.spec_from_file_location("fetch_bench", FETCH_BENCH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class RecordingRunner:
    """The sandbox's default runner (capped_runner), recording each command it gets and its cwd."""

    calls: list[tuple[list[str], Path]] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Record the command, then run it exactly as capped_runner does."""
        self.calls.append((list(argv), Path(cwd)))
        return capped_runner(argv, cwd, timeout_s)


def run_line(app: str, result: RunResult, passfail: bool) -> str:
    """Return the report line for one app whose main compiled and ran; a failed run adds its stderr's tail."""
    stdout_bytes = len(result.stdout.encode("utf-8"))  # the runner decodes with errors="replace"
    passed = ("yes" if "PASS" in result.stdout else "no") if passfail else "n/a"
    line = (
        f"{app:<16} compile=ok exit={result.exit_code} wall_s={result.wall_s:.2f} (exploratory) "
        f"timeout={result.hang} stdout_bytes={stdout_bytes} pass={passed}"
    )
    if result.exit_code != 0:
        tail = " ".join(result.stderr.split())[-STDERR_TAIL_CHARS:]
        line += f" stderr_tail={ascii(tail)}"
    return line


def check_sandboxed(app: str, calls: list[tuple[list[str], Path]], artifact: Path, args: Sequence[str]) -> None:
    """Assert that the one command the sandbox ran for `app` is the sandboxed run of `artifact` with `args`."""
    assert len(calls) == 1, f"{app}: expected one sandboxed command, got {len(calls)}"
    argv, cwd = calls[0]
    assert argv[: len(SANDBOX_HEAD)] == SANDBOX_HEAD, f"{app}: the command does not start the sandbox: {argv[:6]}"
    missing = [tool for tool in SANDBOX_TOOLS if tool not in argv]
    assert not missing, f"{app}: the command lacks {missing}"
    assert argv[-(len(args) + 1) :] == [str(artifact), *args], f"{app}: the command's tail is {argv[-4:]}"
    assert THREAD_BOUND in argv[: -(len(args) + 1)], f"{app}: the program's environment lacks {THREAD_BOUND}"
    assert cwd == artifact.parent.resolve(), f"{app}: the sandbox ran in {cwd}, not the build dir"


def compile_and_run(suite: Any, root: Path, base: Path) -> tuple[list[str], list[str]]:
    """Compile each OpenMP reference with the proxy and run each built main in the sandbox; return lines, failures."""
    lines: list[str] = []
    failures: list[str] = []
    built = build_toolchain(TOOLCHAIN, Path(os.environ["LASSI_TOOLCHAINS"]))
    for app in APPS:
        workdir = base / f"{app}-{LANGUAGE}" / "build"
        workdir.mkdir(parents=True)
        files = suite.reference_target(app, DIRECTION, root, purpose="eval")
        harness = suite.support_files(app, root, purpose="eval")
        result = built.toolchain.build(files, workdir, harness=harness)
        if result.artifact is None or not result.artifact.is_file():
            errors = [f"{d.file}:{d.line} {d.message}"[:200] for d in result.diagnostics if d.severity == "error"]
            line = f"{app:<16} compile=FAIL: {'; '.join(errors[:3]) or 'no error parsed'}"
            lines.append(line)
            failures.append(line)
            continue
        item = suite.items[app]
        recorder = RecordingRunner()
        executor = DEFAULT_REGISTRY.get("Executor", EXECUTOR).factory(sandbox=Sandbox(runner=recorder))
        run = executor.run(result.artifact, list(item.run_args), LIMITS)
        check_sandboxed(app, recorder.calls, result.artifact, item.run_args)
        lines.append(run_line(app, run, LANGUAGE in item.passfail))
    return lines, failures


@pytest.mark.remote
@pytest.mark.slow
def test_the_ten_openmp_references_compile_with_the_multicore_proxy_and_run_in_the_sandbox() -> None:
    problem = host_problem()
    if problem:
        if REQUIRE:
            pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {problem}")
        pytest.skip(problem)
    entry = DEFAULT_REGISTRY.get("Executor", EXECUTOR)
    assert {"runs_code", "sandboxed"} <= entry.capabilities, entry.capabilities
    assert fetch_tool().main([str(MANIFEST)]) == 0, "tools/fetch_bench.py failed"
    suite = load_suite(MANIFEST)
    assert tuple(sorted(suite.items)) == tuple(sorted(APPS)), sorted(suite.items)
    root = sources_dir(Path(os.environ["LASSI_SCRATCH"]), suite)
    base = Path(tempfile.mkdtemp(prefix="lassi-multicore-proxy.", dir=os.environ["LASSI_RUNS_ROOT"]))
    try:
        lines, failures = compile_and_run(suite, root, base)
    finally:
        shutil.rmtree(base)
    compiled = len(APPS) - len(failures)
    exited_zero = sum(" exit=0 " in line for line in lines)
    lines.append(
        f"compiled {compiled}/{len(APPS)} OpenMP references of {SUITE} at HeCBench {suite.commit} with {TOOLCHAIN}; "
        f"ran {compiled} through the {EXECUTOR} executor in the sandbox (wall {LIMITS.wall_s:.0f} s, "
        f"memory {LIMITS.memory_mb} MB, cpus {LIMITS.cpus}); exit 0: {exited_zero}/{compiled}"
    )
    lines.append("wall_s values are exploratory and never performance numbers: the proxy checks outputs, never runtime")
    report = "\n".join(lines)
    print(report)
    assert not failures, report
