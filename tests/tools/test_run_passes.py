"""Tests for plans/runs/suite/run-passes.sh: passes of run-suite.sh in one rx job (task P17.61).

Bible: Agent Rules 2, 7, 8, and 10; plans/runs/suite/README.md.

What they fix:

- The script parses, its help lists every option, and a bad option or a value that is not a whole number exits 2.
- It waits until --start, then runs the suite script (RUN_SUITE, a stand-in here) with --deadline and the options
  after "--", pass 1 under the rx job id and pass k under <id>-p<k>, until --max-passes.
- A pass that ends in under --min-pass seconds is not repeated; under --min-left seconds to the deadline, or with
  the STOP file present, no pass starts.

The loop tests need prlimit (util-linux) and a Python with the resource module, as the build host has; elsewhere
they are skipped.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = "plans/runs/suite/run-passes.sh"
BASH = shutil.which("bash")
LINUX = BASH is not None and shutil.which("prlimit") is not None and sys.platform.startswith("linux")


def run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run the script from the repository root with `args`."""
    assert BASH is not None
    return subprocess.run([BASH, SCRIPT, *args], cwd=REPO, capture_output=True, text=True, timeout=60,
                          env={**os.environ, **(env or {})})


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_the_script_parses_and_its_help_lists_every_option() -> None:
    assert BASH is not None
    parsed = subprocess.run([BASH, "-n", SCRIPT], cwd=REPO, capture_output=True, text=True)
    assert parsed.returncode == 0, parsed.stderr
    shown = run(["--help"])
    assert shown.returncode == 0, shown.stderr
    for option in ("--start", "--deadline", "--min-left", "--min-pass", "--max-passes", "--gap", "--help"):
        assert option in shown.stdout


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
@pytest.mark.parametrize("args", [["--no-such-option"], ["--start", "soon", "--deadline", "1"],
                                  ["--start", "1"], ["--start", "1", "--deadline", "2", "--gap", "-1"]])
def test_a_bad_option_exits_2(args: list[str]) -> None:
    assert run(args).returncode == 2


def stand_in(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """Return the environment for a loop test and the log its stand-in suite script writes, one line per pass."""
    log = tmp_path / "passes.log"
    fake = tmp_path / "fake-suite.sh"
    fake.write_text(f'#!/usr/bin/env bash\necho "$LASSI_RX_RUN_ID $*" >> "{log}"\n', encoding="ascii")
    runs = tmp_path / "runs"
    (runs / "rngd-sweep").mkdir(parents=True)
    env = {"LASSI_RUNS_ROOT": str(runs), "LASSI_RX_RUN_ID": "job-1", "RUN_SUITE": str(fake), "PASSES_POLL_S": "1"}
    return env, log


def window(start: int, deadline: int, *extra: str) -> list[str]:
    """Return the script's arguments for a window, with quick limits, and the suite options after "--"."""
    return ["--start", str(start), "--deadline", str(deadline), "--min-pass", "0", "--gap", "0", *extra,
            "--", "--models", "llama33-70b", "--sweep", "projects/lassi-demo/sweep18", "--skip-download"]


@pytest.mark.skipif(not LINUX, reason="needs bash and prlimit on Linux, as the build host")
def test_it_waits_for_the_start_then_runs_each_pass_under_its_own_id(tmp_path: Path) -> None:
    env, log = stand_in(tmp_path)
    now = int(time.time())
    result = run(window(now + 2, now + 7200, "--min-left", "0", "--max-passes", "3"), env)
    assert result.returncode == 0, result.stderr
    assert time.time() >= now + 2
    rest = f"--deadline {now + 7200} --models llama33-70b --sweep projects/lassi-demo/sweep18 --skip-download"
    assert log.read_text(encoding="ascii").splitlines() == [f"job-1 {rest}", f"job-1-p2 {rest}", f"job-1-p3 {rest}"]
    assert "ended after 3 pass(es)" in result.stdout


@pytest.mark.skipif(not LINUX, reason="needs bash and prlimit on Linux, as the build host")
def test_a_pass_that_ends_fast_is_not_repeated(tmp_path: Path) -> None:
    env, log = stand_in(tmp_path)
    now = int(time.time())
    args = window(now, now + 7200, "--min-left", "0")
    args[args.index("--min-pass") + 1] = "600"
    result = run(args, env)
    assert result.returncode == 0, result.stderr
    assert len(log.read_text(encoding="ascii").splitlines()) == 1 and "not repeated" in result.stdout


@pytest.mark.skipif(not LINUX, reason="needs bash and prlimit on Linux, as the build host")
def test_too_little_time_or_the_stop_file_starts_no_pass(tmp_path: Path) -> None:
    env, log = stand_in(tmp_path)
    now = int(time.time())
    late = run(window(now, now + 600), env)
    assert late.returncode == 0 and "no more passes" in late.stdout and not log.exists()
    (Path(env["LASSI_RUNS_ROOT"]) / "rngd-sweep" / "STOP").write_text("", encoding="ascii")
    stopped = run(window(now + 3600, now + 7200), env)
    assert stopped.returncode == 0 and "STOP file: no pass started" in stopped.stdout and not log.exists()
