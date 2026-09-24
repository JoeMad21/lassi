"""Remote test of the native executor's OpenMP thread bound on the build host (DEMO.2).

Bible: Execution Backends (executor native, Sandbox), Agent Rules 6, 7, and
12. tests/executors/test_native_threads.py pins, without a sandbox, the
program environment the native executor hands the sandbox: PATH=SANDBOX_PATH,
LANG=C.UTF-8, TMPDIR=/tmp, and OMP_NUM_THREADS=<Limits.cpus>, and no HOME
(SandboxSpec.environment never holds HOME). This test checks that a program
run through NativeExecutor.run in a real sandbox sees exactly that
environment, and no variable of the caller's, a secret and a caller's
OMP_NUM_THREADS included.

The program is a copy of the system's `env` (found on SANDBOX_PATH), placed
as the artifact in a fresh build directory under $LASSI_RUNS_ROOT that the
test removes afterwards (Agent Rule 7). It is a system tool, not generated
code, and it still runs only in the sandbox (Agent Rule 6). The test is
marked `remote` and skips unless the host can run a sandbox (Linux, the
sandbox tools on PATH, a reachable user systemd manager, $LASSI_SCRATCH, and
$LASSI_RUNS_ROOT); with LASSI_REQUIRE_SANDBOX=1 it fails instead, so a silent
skip never passes for evidence. Run it through `uv run tools/rx.py run`
with `LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote <this file>`.
No value here is a measurement; the CPU count is a test input.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from lassi.core.interfaces import Limits
from lassi.executors.sandbox import SANDBOX_PATH

TOOLS = ("unshare", "systemd-run", "nice", "setpriv", "prlimit", "timeout", "python3", "awk")
REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"
# Three CPUs: a count no OpenMP runtime or host default would pick by itself, so the bound is told apart.
LIMITS = Limits(wall_s=10.0, memory_mb=64, cpus=3)


def host_problem() -> str:
    """Return why this host cannot run the native executor's sandbox, or "" when it can."""
    if not sys.platform.startswith("linux"):
        return f"the sandbox needs Linux, not {sys.platform}; run it through `uv run tools/rx.py run`"
    missing = [tool for tool in TOOLS if shutil.which(tool) is None]
    if missing:
        return f"not on PATH: {', '.join(missing)}"
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    if not runtime or not (Path(runtime) / "bus").exists():
        return "no user systemd manager: $XDG_RUNTIME_DIR/bus does not exist"
    for name in ("LASSI_SCRATCH", "LASSI_RUNS_ROOT"):
        if not os.environ.get(name):
            return f"{name} is not set"
    if shutil.which("env", path=SANDBOX_PATH) is None:
        return f"no env in the sandbox's PATH {SANDBOX_PATH}"
    return ""


PROBLEM = host_problem()
pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(bool(PROBLEM) and not REQUIRE, reason=PROBLEM or "the host can run the sandbox"),
]


@pytest.fixture(autouse=True)
def host_ready() -> None:
    """Fail, rather than skip, when LASSI_REQUIRE_SANDBOX=1 asks for a real sandbox and the host cannot run one."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")


@pytest.fixture
def workdir() -> Iterator[Path]:
    """Create a fresh build directory under $LASSI_RUNS_ROOT; remove it afterwards."""
    base = Path(tempfile.mkdtemp(prefix="lassi-native-threads.", dir=os.environ["LASSI_RUNS_ROOT"]))
    try:
        build = base / "attempt00" / "build"
        build.mkdir(parents=True)
        yield build
    finally:
        shutil.rmtree(base)


def test_a_native_run_sees_exactly_the_default_environment_plus_the_thread_bound(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lassi.executors.native import NativeExecutor

    secret = f"lassi-native-threads-test-secret-{uuid.uuid4().hex}"
    monkeypatch.setenv("LASSI_TEST_API_KEY", secret)
    monkeypatch.setenv("OMP_NUM_THREADS", "999")
    program = shutil.which("env", path=SANDBOX_PATH)
    assert program is not None
    artifact = workdir / "main"
    shutil.copyfile(program, artifact)
    artifact.chmod(0o755)
    result = NativeExecutor().run(artifact, [], LIMITS)
    assert result.exit_code == 0, result
    expected = {"PATH": SANDBOX_PATH, "LANG": "C.UTF-8", "TMPDIR": "/tmp", "OMP_NUM_THREADS": str(LIMITS.cpus)}
    assert sorted(result.stdout.splitlines()) == sorted(f"{name}={value}" for name, value in expected.items()), result
    assert secret not in result.stdout + result.stderr, result
