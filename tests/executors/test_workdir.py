"""Tests for per-trial build directories under $LASSI_RUNS_ROOT (P0.10).

Every attempt builds in its own fresh directory under the run tree (bible
Sandbox: per-trial build directory under the runs root), never inside the
repository. No value here is a measurement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lassi.executors import workdir

REPO = Path(__file__).resolve().parents[2]
TRIAL = "lassi-repro/mock-fixture/lassi-hecbench-10/omp-cuda/layout/run01"


def test_build_dir_nests_trial_id_and_attempt(tmp_path: Path) -> None:
    path = workdir.build_dir(tmp_path, TRIAL, 3)
    assert path == tmp_path.joinpath(*TRIAL.split("/"), "attempt03", "build")


def test_build_dir_rejects_bad_ids_and_attempts(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        workdir.build_dir(tmp_path, "not/a/trial", 0)
    with pytest.raises(ValueError, match="attempt"):
        workdir.build_dir(tmp_path, TRIAL, -1)
    with pytest.raises(ValueError, match="attempt"):
        workdir.build_dir(tmp_path, TRIAL, True)


def test_build_dir_refuses_the_repository(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repository"):
        workdir.build_dir(REPO / "runs", TRIAL, 0)


def test_runs_root_comes_from_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LASSI_RUNS_ROOT", str(tmp_path))
    assert workdir.runs_root() == tmp_path
    monkeypatch.delenv("LASSI_RUNS_ROOT")
    with pytest.raises(ValueError, match="LASSI_RUNS_ROOT"):
        workdir.runs_root()
    monkeypatch.setenv("LASSI_RUNS_ROOT", "relative/runs")
    with pytest.raises(ValueError, match="absolute"):
        workdir.runs_root()


def test_fresh_build_dir_is_created_empty_and_refuses_reuse(tmp_path: Path) -> None:
    path = workdir.fresh_build_dir(tmp_path, TRIAL, 0)
    assert path.is_dir() and not any(path.iterdir())
    with pytest.raises(FileExistsError):
        workdir.fresh_build_dir(tmp_path, TRIAL, 0)
