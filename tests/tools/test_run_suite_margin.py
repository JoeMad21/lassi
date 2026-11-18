"""Tests for run-suite.sh's free-space margin, SUITE_MARGIN_GB (the owner's window of 2026-11-17).

run-suite.sh keeps the gate's 60 GB big-job floor plus a margin free before each
download and each run.sh call. The margin was a fixed 5 GB; SUITE_MARGIN_GB sets
it in whole GB, 5 when unset, and anything else stops the script. The floor
itself is not configurable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "plans" / "runs" / "suite" / "run-suite.sh"
BASH = shutil.which("bash")


def margin_lines() -> str:
    """Return the script's floor, margin, and margin check lines, which set and check margin_gb."""
    lines = SCRIPT.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("floor_gb="))
    return "\n".join(lines[start:start + 3])


def run(value: str | None) -> subprocess.CompletedProcess[str]:
    """Run the margin lines in bash with SUITE_MARGIN_GB set to value (unset for None), then print the margin."""
    env = {key: val for key, val in os.environ.items() if key != "SUITE_MARGIN_GB"}
    if value is not None:
        env["SUITE_MARGIN_GB"] = value
    code = margin_lines() + '\necho "floor=$floor_gb margin=$margin_gb"\n'
    return subprocess.run([BASH, "-c", code], capture_output=True, text=True, env=env, check=False)


def test_the_floor_stays_60_and_the_margin_reads_the_variable() -> None:
    """The floor is the gate's 60 GB; the margin is SUITE_MARGIN_GB with run-night.sh's 5 GB as its default."""
    text = margin_lines()
    assert "floor_gb=60 " in text
    assert "margin_gb=${SUITE_MARGIN_GB:-5}" in text


@pytest.mark.skipif(BASH is None, reason="needs bash")
@pytest.mark.parametrize(("value", "margin"), [(None, "5"), ("1", "1"), ("0", "0"), ("12", "12")])
def test_a_whole_number_sets_the_margin(value: str | None, margin: str) -> None:
    """Unset gives 5; a whole number of GB is taken as it is."""
    done = run(value)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == f"floor=60 margin={margin}"


@pytest.mark.skipif(BASH is None, reason="needs bash")
@pytest.mark.parametrize("value", ["", "-1", "1.5", "five", "5 GB"])
def test_anything_else_stops_the_script(value: str) -> None:
    """A negative, fractional, or worded value exits 2 naming SUITE_MARGIN_GB; an empty one counts as unset."""
    done = run(value)
    if value == "":
        # An empty variable is unset for ${VAR:-5}, so the default holds.
        assert done.returncode == 0 and done.stdout.strip() == "floor=60 margin=5"
        return
    assert done.returncode == 2
    assert "SUITE_MARGIN_GB" in done.stderr
