"""Tests for plans/runs/tt-par/count_cores.py, the core count of each tt-par translation (tasks P17.74, P17.70).

What they fix:
- A trial's target is its last attempt that reached S5 with a run record;
  a trial with none is left out.
- The build is copied without the earlier run's @ttsim directory and its
  recorded output files, so the rerun's outputs are its own.
- A row carries the cores and workers of the run's cores note (code
  lassi.core.capabilities CORES_CODE), "-" without one, and whether every
  recorded output's sha256 equals the rerun's.
Nothing here runs a program: the run is a fake. No value is a measurement.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from lassi.core.capabilities import CORES_CODE
from lassi.core.record import Diagnostic

SCRIPT = Path(__file__).resolve().parents[2] / "plans" / "runs" / "tt-par" / "count_cores.py"


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    """Import the script as a module."""
    spec = importlib.util.spec_from_file_location("count_cores", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["count_cores"] = module  # a dataclass reads its module from sys.modules
    spec.loader.exec_module(module)
    return module


def digest(data: bytes) -> str:
    """Return the hex sha256 of `data`."""
    return hashlib.sha256(data).hexdigest()


def write_trial(
    root: Path, item: str, stages: list[str], alignment: float, output: bytes = b"OUT", split: str = "dev"
) -> Path:
    """Write a SYNTHETIC trial.json and an attempt build per attempt under `root`; return the trial's folder."""
    folder = root / "proj" / "model" / "tt-par-v0" / "cpp-tt" / item / "run01"
    attempts = []
    for index, stage in enumerate(stages):
        build = folder / f"attempt{index:02d}" / "build"
        (build / "@ttsim" / "logs").mkdir(parents=True)
        (build / "main").write_bytes(b"PLACEHOLDER main")
        (build / "c.lassiio").write_bytes(output)
        run = {"exit_code": 0, "outputs": {"c.lassiio": digest(output)}} if stage == "S5" else None
        attempts.append({"index": index, "stage_reached": stage, "run": run})
    trial = {
        "bench_item": {"suite": "tt-par-v0", "item": item, "split": split, "direction": "cpp-tt"},
        "model": {"id": "SYNTHETIC/model"},
        "final": {"stage_reached": stages[-1], "alignment": alignment},
        "attempts": attempts,
    }
    (folder / "trial.json").write_text(json.dumps(trial), encoding="utf-8")
    return folder


def test_the_target_is_the_last_clean_attempt_and_unclean_trials_are_left_out(tool: ModuleType, tmp_path: Path) -> None:
    root = tmp_path / "sweep-run"
    write_trial(root, "matmul_multicore_reuse", ["S4", "S5", "S5", "S4"], 0.0)
    write_trial(root, "pad_multi_core", ["S1", "S1"], 0.0)
    found = tool.targets([root])
    assert [(t.item, t.attempt, t.alignment) for t in found] == [("matmul_multicore_reuse", 2, 0.0)]
    assert found[0].build.name == "build" and found[0].build.parent.name == "attempt02"
    assert found[0].run == "sweep-run" and found[0].model == "SYNTHETIC/model" and found[0].split == "dev"


def test_held_out_trials_are_counted_only_when_asked(tool: ModuleType, tmp_path: Path) -> None:
    root = tmp_path / "sweep-run"
    write_trial(root, "row_sum", ["S5"], 1.0, split="eval")
    write_trial(root, "vecadd_multi_core", ["S5"], 1.0)
    assert [t.item for t in tool.targets([root])] == ["vecadd_multi_core"]
    assert sorted(t.split for t in tool.targets([root], include_eval=True)) == ["dev", "eval"]


def test_the_copy_leaves_out_the_earlier_runs_files(tool: ModuleType, tmp_path: Path) -> None:
    folder = write_trial(tmp_path / "r", "vecadd_multi_core", ["S5"], 1.0)
    work = tmp_path / "work"
    tool.fresh_copy(folder / "attempt00" / "build", ["c.lassiio"], work)
    assert sorted(path.name for path in work.iterdir()) == ["main"]


@pytest.mark.parametrize(("rerun_output", "equal"), [(b"OUT", "yes"), (b"OTHER", "no")])
def test_a_row_carries_the_cores_note_and_the_output_check(
    tool: ModuleType, tmp_path: Path, rerun_output: bytes, equal: str
) -> None:
    write_trial(tmp_path / "r", "vecadd_multi_core", ["S5"], 1.0)
    (target,) = tool.targets([tmp_path / "r"])
    note = Diagnostic(stage="run", severity="note", code=CORES_CODE,
                      message="cores used: 72 of 80 worker cores ran a kernel (Watcher dump #1); a count, not a time")

    def run(artifact: Path, inputs: list[str]) -> Any:
        """Write the rerun's output and return a result with the cores note."""
        (artifact.parent / "c.lassiio").write_bytes(rerun_output)
        assert inputs == ["SYNTHETIC-arg"]
        return SimpleNamespace(exit_code=0, diagnostics=[note])

    row = tool.count_one(target, tmp_path / "work", run, lambda t, w: ["SYNTHETIC-arg"])
    assert (row["cores"], row["workers"], row["outputs_equal"], row["exit"]) == ("72", "80", equal, "0")
    assert row["alignment"] == "1.0" and row["attempt"] == "0"


def test_a_run_without_a_cores_note_gives_dashes(tool: ModuleType, tmp_path: Path) -> None:
    write_trial(tmp_path / "r", "vecadd_multi_core", ["S5"], 0.0)
    (target,) = tool.targets([tmp_path / "r"])
    row = tool.count_one(target, tmp_path / "work", lambda a, i: SimpleNamespace(exit_code=1, diagnostics=[]),
                         lambda t, w: [])
    assert (row["cores"], row["workers"], row["note"], row["outputs_equal"]) == ("-", "-", "no cores note", "no")


def test_a_rerun_that_raises_becomes_that_trials_row(tool: ModuleType, tmp_path: Path) -> None:
    write_trial(tmp_path / "r", "vecadd_multi_core", ["S5"], 0.0)
    (target,) = tool.targets([tmp_path / "r"])

    def run(artifact: Path, inputs: list[str]) -> Any:
        """Fail as a sandbox that is unavailable would."""
        raise RuntimeError("SYNTHETIC sandbox\nunavailable")

    row = tool.count_one(target, tmp_path / "work", run, lambda t, w: [])
    assert row["note"] == "error: RuntimeError: SYNTHETIC sandbox unavailable"
    assert (row["exit"], row["cores"], row["item"]) == ("-", "-", "vecadd_multi_core")


def test_the_tsv_names_the_device_then_the_columns_in_order(tool: ModuleType, tmp_path: Path) -> None:
    row = {name: name.upper() for name in tool.COLUMNS}
    with (tmp_path / "t.tsv").open("w", encoding="ascii") as out:
        tool.write_header(out, "ttsim SYNTHETIC: a simulator, not silicon")
        tool.write_row(out, row)
    lines = (tmp_path / "t.tsv").read_text(encoding="ascii").splitlines()
    assert lines[0] == "# device: ttsim SYNTHETIC: a simulator, not silicon"
    assert lines[1].split("\t") == list(tool.COLUMNS) and lines[2].split("\t")[0] == "RUN"
