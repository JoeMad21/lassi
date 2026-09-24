"""Tests for the output caps of the command runners in lassi.toolchains._base (P0.16, R4).

The sandbox runs every program through lassi.toolchains capped_runner (its
default runner, lassi.executors.sandbox), so that runner is where stdout and
stderr are capped. It drains both pipes while the command runs, so the
command never blocks on a full pipe, and for each stream it keeps the first
OUTPUT_CAP_BYTES - OUTPUT_TAIL_BYTES bytes and the last OUTPUT_TAIL_BYTES
bytes, so it keeps at most OUTPUT_CAP_BYTES of either stream (after a timeout
the runner's own "timed out" line comes on top), and once it has its result
it drops whatever a pipe still delivers. It sets
CommandResult.stdout_truncated or stderr_truncated for each stream it cut.
The tail keeps the sandbox setup's final line when a program floods stderr.
The design follows plans/spikes/p0-sandbox-hardening.md (probe K and the
recommended design, item 5). The compilers' runners, subprocess_runner and
EnvRunner, drain the same way but keep everything, as before P0.16, since
compile capture is out of that task's scope (review finding).

The commands here are the current Python interpreter writing generated
bytes; no compiler and no generated code runs. Byte counts are test inputs
and design constants, not measurements. The memory test runs only on Linux,
where the resource module reports a child's peak resident set size; the
bound on what the capped pipe holds is also checked directly, on every
platform.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from lassi.toolchains import CommandResult, EnvRunner, _base, capped_runner, subprocess_runner

REPO = Path(__file__).resolve().parents[2]
# The memory test's allowance, in MiB, past two streams of OUTPUT_CAP_BYTES: decode copies and allocator slack.
SLACK_MIB = 24
# The generator both this module and each child command use, so the expected bytes are known exactly:
# numbered 8-byte lines ("<tag><6 digits>\n"), cut to `size` bytes.
PATTERN = '''
def pattern(size: int, tag: bytes) -> bytes:
    count = size // 8 + 1
    return b"".join(tag + b"%06d\\n" % (index % 1000000) for index in range(count))[:size]
'''
# Writes pattern(out, b"o") to stdout and pattern(err, b"e") to stderr, then exits with the given status.
WRITER = PATTERN + '''
import sys
out, err, status = (int(value) for value in sys.argv[1:4])
sys.stdout.buffer.write(pattern(out, b"o"))
sys.stdout.flush()
sys.stderr.buffer.write(pattern(err, b"e"))
sys.stderr.flush()
sys.exit(status)
'''
# Alternates 64 KiB writes to stdout and stderr, so a runner that reads one pipe to its end would block.
INTERLEAVED = PATTERN + '''
import sys
total = int(sys.argv[1])
out, err = pattern(total, b"o"), pattern(total, b"e")
for start in range(0, total, 65536):
    sys.stdout.buffer.write(out[start : start + 65536])
    sys.stdout.flush()
    sys.stderr.buffer.write(err[start : start + 65536])
    sys.stderr.flush()
sys.exit(3)
'''
# Writes pattern(size, b"o") to stdout, then sleeps past any timeout the test uses.
FLOOD_THEN_HANG = PATTERN + '''
import sys, time
sys.stdout.buffer.write(pattern(int(sys.argv[1]), b"o"))
sys.stdout.flush()
time.sleep(60)
'''
# Runs capped_runner on a command that writes argv[1] MiB to stdout and half as much to stderr, and prints the
# runner's peak resident set size after a baseline run with a few bytes of output and after the flood, with what
# it kept. The baseline makes the threads, imports, and decoding warm before the flood is measured.
MEMORY_PROBE = '''
import json, resource, sys
from pathlib import Path
from lassi.toolchains import capped_runner
mib = int(sys.argv[1])
flood = (
    "import sys\\n"
    f"block = bytes(1 << 20)\\n"
    f"for _ in range({mib}): sys.stdout.buffer.write(block)\\n"
    f"for _ in range({mib // 2}): sys.stderr.buffer.write(block)\\n"
)
capped_runner([sys.executable, "-c", "print(1)"], Path.cwd(), 300.0)
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
result = capped_runner([sys.executable, "-c", flood], Path.cwd(), 300.0)
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({
    "before_kib": before, "after_kib": after, "returncode": result.returncode,
    "stdout_bytes": len(result.stdout.encode("utf-8")), "stderr_bytes": len(result.stderr.encode("utf-8")),
    "stdout_truncated": result.stdout_truncated, "stderr_truncated": result.stderr_truncated,
}))
'''


def pattern(size: int, tag: bytes) -> bytes:
    """Return the bytes the children write for `size` and `tag` (the PATTERN generator, run here)."""
    scope: dict[str, Callable[[int, bytes], bytes]] = {}
    exec(PATTERN, scope)
    return scope["pattern"](size, tag)


def kept(data: bytes) -> str:
    """Return what the runner keeps of `data`: all of it within the cap, else the head and the tail, decoded."""
    cap, tail = _base.OUTPUT_CAP_BYTES, _base.OUTPUT_TAIL_BYTES
    if len(data) <= cap:
        return data.decode("utf-8", errors="replace")
    return (data[: cap - tail] + data[-tail:]).decode("utf-8", errors="replace")


def run_writer(tmp_path: Path, out: int, err: int, status: int = 0) -> CommandResult:
    """Run WRITER through capped_runner with a generous timeout."""
    return capped_runner([sys.executable, "-c", WRITER, str(out), str(err), str(status)], tmp_path, 120.0)


def test_the_documented_caps() -> None:
    assert _base.OUTPUT_CAP_BYTES == 1 << 20
    assert 0 < _base.OUTPUT_TAIL_BYTES < _base.OUTPUT_CAP_BYTES
    # The tail must hold the sandbox setup's done line whole, with its two clock readings and its incomplete mark.
    assert _base.OUTPUT_TAIL_BYTES >= len("lassi-sandbox-done 99999999999.99 99999999999.99 incomplete\n")


def test_command_result_carries_truncation_flags() -> None:
    names = [f.name for f in dataclasses.fields(CommandResult)]
    assert names == ["returncode", "stdout", "stderr", "stdout_truncated", "stderr_truncated"]
    result = CommandResult(returncode=0, stdout="o", stderr="e")
    assert (result.stdout_truncated, result.stderr_truncated) == (False, False)


@pytest.mark.parametrize("size", [0, 1, 1 << 20], ids=["empty", "one-byte", "exactly-the-cap"])
def test_output_within_the_cap_is_kept_whole_and_not_flagged(tmp_path: Path, size: int) -> None:
    result = run_writer(tmp_path, size, size, status=4)
    assert result.returncode == 4
    assert result.stdout == pattern(size, b"o").decode("ascii")
    assert result.stderr == pattern(size, b"e").decode("ascii")
    assert (result.stdout_truncated, result.stderr_truncated) == (False, False)


@pytest.mark.parametrize(
    ("out", "err"),
    [((1 << 20) + 1, 100), (3 * (1 << 20) + 12345, 100), (100, 2 * (1 << 20) + 7), (5 << 20, 5 << 20)],
    ids=["stdout-one-past", "stdout-far-past", "stderr-past", "both-past"],
)
def test_output_past_the_cap_keeps_the_head_and_the_tail_and_is_flagged(tmp_path: Path, out: int, err: int) -> None:
    result = run_writer(tmp_path, out, err, status=5)
    assert result.returncode == 5
    assert result.stdout == kept(pattern(out, b"o"))
    assert result.stderr == kept(pattern(err, b"e"))
    assert (result.stdout_truncated, result.stderr_truncated) == (out > 1 << 20, err > 1 << 20)
    assert len(result.stdout.encode("utf-8")) <= _base.OUTPUT_CAP_BYTES
    assert len(result.stderr.encode("utf-8")) <= _base.OUTPUT_CAP_BYTES


def test_both_streams_are_drained_together_so_the_command_never_blocks(tmp_path: Path) -> None:
    total = 8 << 20
    began = time.monotonic()
    result = capped_runner([sys.executable, "-c", INTERLEAVED, str(total)], tmp_path, 120.0)
    assert time.monotonic() - began < 60
    assert result.returncode == 3
    assert result.stdout == kept(pattern(total, b"o"))
    assert result.stderr == kept(pattern(total, b"e"))
    assert (result.stdout_truncated, result.stderr_truncated) == (True, True)


def test_a_timeout_keeps_the_capped_output_and_adds_the_timeout_line(tmp_path: Path) -> None:
    size = 3 << 20
    began = time.monotonic()
    result = capped_runner([sys.executable, "-c", FLOOD_THEN_HANG, str(size)], tmp_path, 3.0)
    assert time.monotonic() - began < 30
    assert result.returncode == -1
    assert result.stdout == kept(pattern(size, b"o"))
    assert result.stdout_truncated is True
    assert re.fullmatch(r"timed out after 3(\.0)? s", result.stderr.rstrip("\n").split("\n")[-1]), result.stderr


@pytest.mark.parametrize("runner", ["subprocess_runner", "EnvRunner"])
def test_the_compile_runners_keep_the_whole_output(tmp_path: Path, runner: str) -> None:
    # The raw stderr attachment and the diagnostics parsers see everything a compiler printed (review finding).
    size = 2 << 20
    argv = [sys.executable, "-c", WRITER, str(size), str(size + 3), "0"]
    run = subprocess_runner if runner == "subprocess_runner" else EnvRunner(dict(os.environ))
    result = run(argv, tmp_path, 120.0)
    assert result.returncode == 0
    assert result.stdout == pattern(size, b"o").decode("ascii")
    assert result.stderr == pattern(size + 3, b"e").decode("ascii")
    assert (result.stdout_truncated, result.stderr_truncated) == (False, False)


@pytest.mark.parametrize(
    "sizes",
    [
        [1] * 5000,
        [65536] * 40,
        [3 << 20],
        [1, (1 << 20) - 4096, 1, 4095, 4096, 4097, 7, 2 << 20, 3],
        [_base.OUTPUT_CAP_BYTES - _base.OUTPUT_TAIL_BYTES - 1, 2, 5000, 1],
    ],
    ids=["single-bytes", "chunks", "one-large-chunk", "mixed", "straddling-the-head"],
)
def test_the_capped_pipe_never_holds_more_than_the_cap(sizes: list[int]) -> None:
    # R4 on every platform: after each chunk the pipe holds at most OUTPUT_CAP_BYTES, and what it keeps is the
    # stream's head and its tail.
    pipe = _base._CappedPipe(None, _base.OUTPUT_CAP_BYTES)
    data = pattern(sum(sizes), b"c")
    offset = 0
    for size in sizes:
        pipe._keep(data[offset : offset + size])
        offset += size
        assert len(pipe._head) + len(pipe._tail) <= _base.OUTPUT_CAP_BYTES, (offset, len(pipe._head), len(pipe._tail))
    assert pipe.text() == (kept(data), len(data) > _base.OUTPUT_CAP_BYTES)


def test_the_uncapped_pipe_keeps_everything() -> None:
    pipe = _base._CappedPipe(None, None)
    data = pattern(3 << 20, b"u")
    for start in range(0, len(data), 65536):
        pipe._keep(data[start : start + 65536])
    assert pipe.text() == (data.decode("ascii"), False)


@pytest.mark.parametrize("cap", [None, _base.OUTPUT_CAP_BYTES], ids=["uncapped", "capped"])
def test_an_abandoned_pipe_keeps_nothing_more(cap: int | None) -> None:
    # Review finding 4: after a timeout, a writer that escaped the kill may keep a pipe open, and its drain
    # thread keeps reading; once the runner has its result, what arrives is read and dropped, never kept.
    pipe = _base._CappedPipe(None, cap)
    pipe._keep(b"kept\n")
    pipe.abandon()
    for _ in range(8):
        pipe._keep(b"x" * (1 << 20))
    assert pipe.text() == ("kept\n", False)
    assert len(pipe._head) + len(pipe._tail) == len(b"kept\n")


@pytest.mark.parametrize("timeout_s", [120.0, 3.0], ids=["finished", "timed-out"])
def test_the_runner_abandons_both_pipes_once_it_has_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeout_s: float
) -> None:
    abandoned: list[object] = []
    original = _base._CappedPipe.abandon

    def record(self: _base._CappedPipe) -> None:
        """Note the pipe, then abandon it as the runner asked."""
        abandoned.append(self)
        original(self)

    monkeypatch.setattr(_base._CappedPipe, "abandon", record)
    command = [sys.executable, "-c", FLOOD_THEN_HANG, "10"] if timeout_s < 10 else [sys.executable, "-c", "pass"]
    result = capped_runner(command, tmp_path, timeout_s)
    assert result.returncode == (-1 if timeout_s < 10 else 0), result
    assert len(abandoned) == 2 and len(set(map(id, abandoned))) == 2


@pytest.mark.slow
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="ru_maxrss in KiB is read on Linux only")
def test_the_runner_never_holds_more_than_the_cap_of_a_flood(tmp_path: Path) -> None:
    # 512 MiB on stdout and 256 MiB on stderr: a runner that kept it all would grow by hundreds of MiB. One that
    # keeps the cap holds two streams of OUTPUT_CAP_BYTES plus the copies it makes to decode them; the bound
    # allows that plus SLACK_MIB for the allocator. The bound is a test input.
    mib = 512
    argv = [sys.executable, "-c", MEMORY_PROBE, str(mib)]
    done = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, timeout=600, check=False)
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)
    assert report["returncode"] == 0, report
    assert (report["stdout_truncated"], report["stderr_truncated"]) == (True, True), report
    assert report["stdout_bytes"] <= _base.OUTPUT_CAP_BYTES and report["stderr_bytes"] <= _base.OUTPUT_CAP_BYTES
    assert report["after_kib"] - report["before_kib"] < (2 * _base.OUTPUT_CAP_BYTES >> 10) + SLACK_MIB * 1024, report


# ---------------------------------------------------------------------------
# CappedRunner: a runner with a cap of its own (P0.20, the compile sandbox's runner)


def test_a_capped_runner_keeps_the_head_and_the_tail_at_its_own_cap(tmp_path: Path) -> None:
    # The compile sandbox caps a compiler's output far above OUTPUT_CAP_BYTES, with a runner that keeps its own cap.
    cap = 3 * _base.OUTPUT_CAP_BYTES
    out, err = cap + 12345, cap - 1
    runner = _base.CappedRunner(cap)
    assert runner.cap_bytes == cap
    result = runner([sys.executable, "-c", WRITER, str(out), str(err), "6"], tmp_path, 120.0)
    assert result.returncode == 6
    data = pattern(out, b"o")
    tail = _base.OUTPUT_TAIL_BYTES
    assert result.stdout == (data[: cap - tail] + data[-tail:]).decode("ascii")
    assert result.stderr == pattern(err, b"e").decode("ascii"), "within the cap, a stream is kept whole"
    assert (result.stdout_truncated, result.stderr_truncated) == (True, False)


def test_a_capped_runner_is_exported_with_the_other_runners() -> None:
    from lassi import toolchains

    assert toolchains.CappedRunner is _base.CappedRunner
    assert "CappedRunner" in toolchains.__all__


@pytest.mark.parametrize("cap", [0, -1, _base.OUTPUT_TAIL_BYTES, True, 2.5, None, "1048576"])
def test_a_capped_runner_refuses_a_cap_that_cannot_hold_its_tail(cap: object) -> None:
    with pytest.raises(ValueError):
        _base.CappedRunner(cap)  # type: ignore[arg-type]
