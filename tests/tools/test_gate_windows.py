"""Windows branches of tools/server/gate.py, plus a guard that pins its POSIX paths.

On Windows os.kill(pid, 0) terminates the process, so the key test here checks that the gate's
liveness probe leaves a live process running. Windows also reuses pids, so job kill must leave a
live process alone when its creation time does not match the recorded runner. The Windows tests
are skipped elsewhere; the POSIX guard is skipped on Windows.
"""

import importlib.util
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import pytest

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "tools" / "server" / "gate.py"
# The base interpreter, not a venv launcher, so each Popen pid is the process that runs the code.
PY = getattr(sys, "_base_executable", None) or sys.executable
SLEEP = "import time; time.sleep(30)"
PARENT = ("import subprocess, sys, time\n"
          "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
          "print(g.pid, flush=True)\n"
          "time.sleep(60)\n")
JOB_ID = "20260923-000000-t-0000"

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows branches of the gate")
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX path of the gate")


def load_gate() -> ModuleType:
    """Import gate.py by path; it is a standalone script, not a package module."""
    spec = importlib.util.spec_from_file_location("lassi_gate_under_test", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load_gate()


@contextmanager
def owned(argv: list[str], **kw: Any) -> Iterator[subprocess.Popen]:
    """Start argv and always kill and reap it through its own handle afterwards."""
    child = subprocess.Popen(argv, **kw)
    try:
        yield child
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=30)
        if child.stdout is not None:
            child.stdout.close()


@contextmanager
def held(pid: int) -> Iterator[None]:
    """Hold a query handle on pid so Windows cannot reuse it while the test checks it."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = k32.OpenProcess(0x1000, False, pid)
    assert handle, "could not open pid %d" % pid
    try:
        yield
    finally:
        k32.CloseHandle(handle)


def job_meta(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **fields: Any) -> None:
    """Point the gate's run and busy dirs at tmp_path and write one running job's meta.json."""
    monkeypatch.setattr(gate, "RUNS", str(tmp_path / "runs"))
    monkeypatch.setattr(gate, "BUSY", str(tmp_path / "busy"))
    run_dir = tmp_path / "runs" / JOB_ID
    run_dir.mkdir(parents=True)
    meta = dict({"id": JOB_ID, "kind": "job", "state": "running", "slot": "t"}, **fields)
    (run_dir / "meta.json").write_text(json.dumps(meta))


@windows_only
def test_pid_alive_leaves_live_child_running() -> None:
    with owned([PY, "-c", SLEEP]) as child:
        assert gate.pid_alive(child.pid) is True
        assert gate.pid_alive(child.pid) is True
        assert child.poll() is None


@windows_only
def test_pid_alive_false_for_exited_child() -> None:
    with owned([PY, "-c", "pass"]) as child:
        child.wait(timeout=30)
        assert gate.pid_alive(child.pid) is False


def test_pid_alive_false_for_invalid_input() -> None:
    assert gate.pid_alive("x") is False
    assert gate.pid_alive(None) is False


@windows_only
def test_kill_group_kills_child_and_grandchild() -> None:
    with owned([PY, "-c", PARENT], stdout=subprocess.PIPE, **gate.new_group()) as child:
        grandchild = int(child.stdout.readline())
        with held(grandchild):
            assert gate.pid_alive(grandchild) is True
            gate.kill_group(child.pid, grace=10)
            child.wait(timeout=15)
            end = time.time() + 10
            while gate.pid_alive(grandchild) and time.time() < end:
                time.sleep(0.2)
            assert gate.pid_alive(grandchild) is False


@windows_only
def test_runner_identity_records_creation_time() -> None:
    with owned([PY, "-c", SLEEP]) as child:
        created = gate.runner_identity(child.pid)["runner_created"]
        assert isinstance(created, int) and created > 0
        assert gate.runner_identity(child.pid) == {"runner_created": created}
        assert child.poll() is None
    assert gate.runner_identity("x") == {"runner_created": None}


@windows_only
@pytest.mark.parametrize("recorded", ["other-creation-time", "no-creation-time"])
def test_job_kill_leaves_unmatched_pid_running(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                               recorded: str) -> None:
    with owned([PY, "-c", SLEEP], **gate.new_group()) as child:
        created = gate.runner_identity(child.pid)["runner_created"]
        stale = {"runner_created": created + 1} if recorded == "other-creation-time" else {}
        job_meta(monkeypatch, tmp_path, runner_pid=child.pid, **stale)
        assert gate.verb_job_kill({"id": JOB_ID}, {}) == {"id": JOB_ID, "state": "killed"}
        assert child.poll() is None
        assert gate.pid_alive(child.pid) is True


@windows_only
def test_job_kill_ends_recorded_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with owned([PY, "-c", SLEEP], **gate.new_group()) as child:
        job_meta(monkeypatch, tmp_path, runner_pid=child.pid, **gate.runner_identity(child.pid))
        assert gate.verb_job_kill({"id": JOB_ID}, {}) == {"id": JOB_ID, "state": "killed"}
        child.wait(timeout=15)
        meta = json.loads((tmp_path / "runs" / JOB_ID / "meta.json").read_text())
        assert meta["state"] == "killed"


@windows_only
def test_new_group_detaches_job_runner_console() -> None:
    no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    assert gate.new_group() == {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    assert gate.new_group(detach=True) == {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | no_window}


@windows_only
def test_stream_process_logs_echoes_and_passes_status(tmp_path: Path,
                                                       capfdbinary: pytest.CaptureFixture[bytes]) -> None:
    log = tmp_path / "output.log"
    rc = gate.stream_process([PY, "-c", "import sys; print('hello-gate'); sys.exit(7)"], str(tmp_path), None,
                             str(log), 30, echo=True)
    assert rc == 7
    assert b"hello-gate" in log.read_bytes()
    assert b"hello-gate" in capfdbinary.readouterr().out
    assert gate.stream_process([PY, "-c", "pass"], str(tmp_path), None, str(log), 30, echo=False) == 0


@windows_only
def test_stream_process_timeout_kills_and_returns_124(tmp_path: Path) -> None:
    log = tmp_path / "output.log"
    start = time.time()
    rc = gate.stream_process([PY, "-c", "import time; print('started', flush=True); time.sleep(60)"],
                             str(tmp_path), None, str(log), 1, echo=False)
    assert rc == 124
    assert time.time() - start < 30
    text = log.read_bytes()
    assert b"started" in text and b"[gate] timeout after 1s; process group killed" in text


@windows_only
def test_stream_process_timeout_falls_back_when_taskkill_fails(monkeypatch: pytest.MonkeyPatch,
                                                                tmp_path: Path) -> None:
    monkeypatch.setattr(gate, "win_kill_tree", lambda pid, grace: None)
    start = time.time()
    rc = gate.stream_process([PY, "-c", "import time; time.sleep(60)"], str(tmp_path), None,
                             str(tmp_path / "output.log"), 1, echo=False)
    assert rc == 124
    assert time.time() - start < 30


@windows_only
def test_free_space_and_host_info(tmp_path: Path) -> None:
    free = gate.free_gb(str(tmp_path))
    assert free > 0 and round(free, 1) == free
    assert gate.free_gb(str(tmp_path / "missing" / "dir")) == -1.0
    info = gate.host_info()
    assert "mem_gb" not in info and "load" not in info and info["nproc"]


@posix_only
def test_pid_alive_posix_probe_pinned() -> None:
    assert gate.pid_alive(os.getpid()) is True
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=30)
    assert gate.pid_alive(child.pid) is False


@posix_only
def test_posix_group_and_meta_pinned() -> None:
    assert gate.new_group() == {"start_new_session": True}
    assert gate.new_group(detach=True) == {"start_new_session": True}
    assert gate.runner_identity(os.getpid()) == {}
