"""Tests for a run whose recipe binds the replay backend with model.recording (task P16.13), and the runner docstring.

Bible: Project Recipes (the model section; Notes, server settings and
in-process model settings), Component Interfaces (LLMBackend), Result Record,
Design Principle 5, Agent Rule 10; plans/p16-adversarial.md, task P16.13;
PHASE-NOTES P16 (Hand-offs from P16.6).

The contract these tests fix:

- A recipe that binds `model: {backend: replay, id: ..., recording: <path>}`,
  the path relative to the repository root under tests/fixtures/, runs
  through lassi.core.runner from any working directory (here tmp_path) with
  the recording read by the replay backend:
  the trial's reply is the recording's first completion, word for word, and
  its model record names the replay backend and the model id.
- The run's recipe.resolved.yaml holds the recording's path as written and the
  sha256 of its bytes; the trial's recipe hash is the loaded recipe's, and the
  saved resolved recipe reloads to that hash.
- The loader hashes the recording and the backend reads it again when the
  runner builds it, so the runner compares the sha256 of the bytes the
  backend read (its recording_sha256) with model.recording_sha256: a
  recording whose bytes changed in between, or a backend that reports no
  sha256, is a RunError naming model.recording_sha256, and one removed in
  between is a RunError naming model.recording, never a raw OSError; each
  comes before the run directory exists.
- lassi.core.runner's docstring names the adversary metric tables where it
  states the metrics step and the parquet/ mirror: the tables
  lassi.analysis.adversary builds and parquet/adversary_metrics, which
  write_adversary_parquet writes (task P16.6).

The run uses the real replay backend, generate and compile_loop stages, and
none executor, with a fake toolchain registered as "nvcc-sm80" in a test
Registry; nothing compiles, runs a program, or reaches the network. The bench
sources are small SYNTHETIC files standing in for the layout item, and the
recording is the committed SYNTHETIC fixture
tests/fixtures/replay/recordings/first-try-success.json. The tests of the
gap between load and build copy it into a stand-in repository under tmp_path
(lassi.core.recipe.REPO, the root the loader reads model.recording against),
whose copy they change or remove. No value in this module is a measurement.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.core import recipe as recipe_module
from lassi.core import runner as runner_module
from lassi.core.interfaces import BuildResult
from lassi.core.recipe import load_recipe
from lassi.core.record import ModelInfo, Trial
from lassi.core.registry import Registry
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.stages import CompileLoopStage, GenerateStage
from lassi.core.store import TextStore, read_trial
from lassi.executors import NoneExecutor
from lassi.llm import ReplayBackend

REPO = Path(__file__).resolve().parents[2]
RECORDING = "tests/fixtures/replay/recordings/first-try-success.json"
MODEL_ID = "replay-fixture"
PROJECT = "replay-run"

# SYNTHETIC bench sources standing in for the layout item of lassi-hecbench-10.
FAKE_OMP_SOURCE = '#include <cstdio>\nint main() {\n#pragma omp target\n  { }\n  std::printf("done\\n");\n}\n'
FAKE_CUDA_SOURCE = "#include <cstdio>\n__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n"

# The recipe: one item, one direction, one trial, no correction, so the trial asks the model exactly once.
RECIPE: dict[str, Any] = {
    "extends": "base",
    "model": {"backend": "replay", "id": MODEL_ID, "recording": RECORDING},
    "llm": {"sampling": {"max_tokens": 4096}},
    "loop": {"max_corrections": 0},
    "bench": {"suite": "lassi-hecbench-10", "split": "eval", "items": ["layout"]},
    "directions": [{"source": "omp", "target": "cuda"}],
    "prompts": "p0-smoke",
    "toolchain": {"cuda": "nvcc-sm80"},
    "stages": ["generate", "compile_loop"],
    "executor": {"kind": "none"},
    "trials": {"n": 1},
}


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate variables a test could inherit, point TMPDIR at a test directory, and leave the repository.

    The working directory is tmp_path, so a recording path read against the
    working directory instead of the repository root finds no file.
    """
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.chdir(tmp_path)


class FakeToolchain:
    """A Toolchain without PIN, built as factory(); it writes the files, compiles nothing, and returns an artifact."""

    name = "nvcc-sm80"
    capabilities = frozenset({"diagnostics"})

    def build(self, files: Mapping[str, str], workdir: Path) -> BuildResult:
        """Write the files under `workdir` and return a placeholder artifact."""
        for path, text in files.items():
            target = Path(workdir) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode("utf-8"))
        artifact = Path(workdir) / "main"
        artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
        return BuildResult(artifact=artifact, diagnostics=[])


def make_registry(backend: type = ReplayBackend) -> Registry:
    """Return a test Registry: `backend` as "replay", the real stages and none executor, and a fake toolchain."""
    registry = Registry()
    registry.register("LLMBackend", "replay", backend)
    registry.register("Stage", "generate", GenerateStage)
    registry.register("Stage", "compile_loop", CompileLoopStage)
    registry.register("Executor", "none", NoneExecutor)
    registry.register("Toolchain", "nvcc-sm80", FakeToolchain)
    return registry


def write_bench(root: Path) -> Path:
    """Write the SYNTHETIC layout sources under `root` as the manifest lays them out, and return `root`."""
    for relative, text in (("src/layout-omp/main.cpp", FAKE_OMP_SOURCE), ("src/layout-cuda/main.cu", FAKE_CUDA_SOURCE)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def write_recipe(directory: Path, data: Mapping[str, Any]) -> Path:
    """Write `data` as the recipe <directory>/<PROJECT>.yaml and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{PROJECT}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


def only_trial(run_dir: Path) -> Trial:
    """Read back the run's one trial through its text store."""
    (found,) = sorted(run_dir.rglob("trial.json"))
    return read_trial(found.parent, TextStore(run_dir))


def recorded_texts() -> list[str]:
    """Return the reply texts of the committed recording, in order."""
    data = json.loads((REPO / RECORDING).read_bytes().decode("ascii"))
    assert data["synthetic"] is True
    return [entry["text"] for entry in data["completions"]]


def replay_run(
    tmp_path: Path, data: Mapping[str, Any] = RECIPE, backend: type = ReplayBackend
) -> tuple[Path, Path, Registry]:
    """Run the recipe `data` once with `backend` as "replay"; return its run directory, recipe path, and registry."""
    registry = make_registry(backend)
    recipe_path = write_recipe(tmp_path / "recipes", data)
    options = RunOptions(
        runs_root=tmp_path / "runs-root", run_id="replay", bench_root=write_bench(tmp_path / "bench"), registry=registry
    )
    return run_recipe(recipe_path, options), recipe_path, registry


# The stand-in repository's copy of the recording, relative to its root.
STAND_IN_RECORDING = "tests/fixtures/replay/stand-in.json"
# A SYNTHETIC recording of one completion, other bytes than the committed fixture's.
OTHER_BYTES = b'{"synthetic": true, "completions": [{"text": "SYNTHETIC reply written after load"}]}\n'


@pytest.fixture
def stand_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return the stand-in repository's copy of the recording, with lassi.core.recipe.REPO pointed at its root."""
    root = tmp_path / "stand-in-repo"
    path = root / STAND_IN_RECORDING
    path.parent.mkdir(parents=True)
    path.write_bytes((REPO / RECORDING).read_bytes())
    monkeypatch.setattr(recipe_module, "REPO", root)
    return path


def stand_in_recipe() -> dict[str, Any]:
    """Return RECIPE with model.recording naming the stand-in repository's copy."""
    data = copy.deepcopy(RECIPE)
    data["model"]["recording"] = STAND_IN_RECORDING
    return data


def replay_after(change: Callable[[Path], None], *, reports_sha256: bool = True) -> type:
    """Return a replay backend class that applies `change` to its recording file before reading it.

    It stands in for whatever touches the file between the loader's read and
    the backend's. With reports_sha256 false, it reports no recording_sha256.
    """

    class ChangedReplay(ReplayBackend):
        """ReplayBackend, with its recording changed between load and build."""

        def __init__(self, model_id: str, *, recording: str | os.PathLike[str] | None = None) -> None:
            """Apply the change to the file at `recording`, then build the replay backend from it."""
            assert recording is not None, "the runner passes the checked file"
            change(Path(recording))
            super().__init__(model_id, recording=recording)
            if not reports_sha256:
                self.recording_sha256 = None

    return ChangedReplay


def refused_run(tmp_path: Path, backend: type) -> str:
    """Return the RunError message of a stand-in run with `backend`; no run directory may exist after it."""
    with pytest.raises(RunError) as info:
        replay_run(tmp_path, stand_in_recipe(), backend)
    assert not (tmp_path / "runs-root" / runner_module.RUNS_DIR / "replay").exists(), "refused before any mkdir"
    return str(info.value)


# ---------------------------------------------------------------------------
# A run through the runner


def test_the_trial_reply_is_the_recording_one(tmp_path: Path) -> None:
    run_dir, _, _ = replay_run(tmp_path)
    trial = only_trial(run_dir)
    assert trial.model == ModelInfo(backend="replay", id=MODEL_ID, sampling=trial.model.sampling)
    (attempt,) = trial.attempts
    assert attempt.response_text == recorded_texts()[0], "the reply is the recording's first completion"


def test_the_run_records_the_recording_path_and_sha256(tmp_path: Path) -> None:
    run_dir, recipe_path, registry = replay_run(tmp_path)
    saved = (run_dir / "recipe.resolved.yaml").read_bytes().decode("ascii")
    digest = hashlib.sha256((REPO / RECORDING).read_bytes()).hexdigest()
    assert RECORDING in saved and digest in saved, saved
    loaded = load_recipe(recipe_path, registry=registry)
    assert only_trial(run_dir).recipe_hash == loaded.recipe_hash
    assert load_recipe(run_dir / "recipe.resolved.yaml", registry=registry).recipe_hash == loaded.recipe_hash


# ---------------------------------------------------------------------------
# The gap between the loader's read and the backend's


def test_an_unchanged_stand_in_recording_runs(tmp_path: Path, stand_in: Path) -> None:
    # The control: the stand-in seam and the wrapping class alone refuse nothing.
    run_dir, _, _ = replay_run(tmp_path, stand_in_recipe(), replay_after(lambda path: None))
    (attempt,) = only_trial(run_dir).attempts
    assert attempt.response_text == recorded_texts()[0]


def test_a_recording_changed_after_load_is_refused(tmp_path: Path, stand_in: Path) -> None:
    loaded = hashlib.sha256(stand_in.read_bytes()).hexdigest()
    message = refused_run(tmp_path, replay_after(lambda path: path.write_bytes(OTHER_BYTES)))
    assert "model.recording_sha256" in message and "LLMBackend 'replay'" in message, message
    assert loaded in message and hashlib.sha256(OTHER_BYTES).hexdigest() in message, message


def test_a_recording_removed_after_load_is_a_run_error(tmp_path: Path, stand_in: Path) -> None:
    message = refused_run(tmp_path, replay_after(lambda path: path.unlink()))
    assert "model.recording" in message and "LLMBackend 'replay'" in message, message


def test_a_backend_that_reports_no_sha256_is_refused(tmp_path: Path, stand_in: Path) -> None:
    message = refused_run(tmp_path, replay_after(lambda path: None, reports_sha256=False))
    assert "model.recording_sha256" in message and "reports no recording_sha256" in message, message


# ---------------------------------------------------------------------------
# The runner docstring names the adversary metric tables


def docstring_part(opening: str, closing: str) -> str:
    """Return lassi.core.runner's docstring from `opening` to the next `closing`, whitespace runs collapsed."""
    text = runner_module.__doc__ or ""
    start = text.index(opening)
    return " ".join(text[start : text.index(closing, start + len(opening))].split())


def test_the_metrics_step_names_the_adversary_tables() -> None:
    step = docstring_part("\n6. ", "\n7. ")
    assert "lassi.analysis.adversary" in step, f"step 6 does not name the adversary tables: {step}"


def test_the_parquet_mirror_names_adversary_metrics() -> None:
    bullet = docstring_part("\n- texts/", "\n\n")
    assert "adversary_metrics" in bullet, f"the parquet/ bullet does not name adversary_metrics: {bullet}"
