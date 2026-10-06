"""Tests of hf_local in `lassi run` on the CPU with a tiny model (task P17.4): provenance, hash, context refusal.

Bible: Model Serving (Serving Rules), Result Record (provenance, Storage),
Project Recipes (Notes), Agent Rules 7 and 10; plans/p17-portable.md, task
P17.4 ("Provenance records the framework versions and the device record");
the P17.4 design (sections 3, 8, and 9).

These tests need a framework extra and skip without one, with a reason that
names it: run them as `uv run --extra cpu pytest tests/llm/test_hf_local_runs.py`.

The contract these tests fix, end to end through run_recipe with the real
hf_local class and a tiny model in a fake HF_HOME (tests/tiny_hf.py):

- provenance.json's device record for model.device names the framework
  torch and its version (task P17.2, through hf_local's framework()), and
  `serving` holds hf_local's record: model, revision, seed, device, the
  context, and the installed framework versions. run.md's Server row shows
  the in-process line. Each trial's provenance carries the device record.
- recipe.resolved.yaml holds model.revision and model.seed as written, and a
  different seed gives a different recipe hash.
- A request past the model's context ends that trial with the end reason
  context-exceeded and no attempt; the run goes on to the next trial and
  ends with status complete.
- A snapshot the hub cache lacks, and an unset HF_HOME, are each a RunError
  before any directory exists.

The model's weights are random, its replies hold no usable file, and every
other component is a fake (tests/llm/hf_runs.py). No value in this module is
a measurement.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import pytest
import yaml
from hf_runs import (
    CPU_NAME,
    SEED,
    FakeCpuProbe,
    clean_environment,
    make_registry,
    model_section,
    read_ascii,
    recipe_data,
    run,
    trial_id,
)
from tiny_hf import NEEDS_EXTRA, OTHER_REVISION, REPO_ID, REVISION, build_snapshot

torch = pytest.importorskip("torch", reason=NEEDS_EXTRA)
pytest.importorskip("transformers", reason=NEEDS_EXTRA)

from lassi.core.parquet import read_run_parquet  # noqa: E402
from lassi.core.registry import DEFAULT_REGISTRY  # noqa: E402
from lassi.core.runner import RunError  # noqa: E402
from lassi.core.store import TextStore, read_trial, trial_dir  # noqa: E402

CONTEXT_EXCEEDED = "context-exceeded"
# Room for the p0-smoke generation prompt of the layout item and a few new tokens.
WIDE_CONTEXT = 4096


def hf_local_class() -> type:
    """Return the class lassi.llm registers as LLMBackend hf_local, failing clearly while there is none."""
    if "hf_local" not in DEFAULT_REGISTRY.names("LLMBackend"):
        pytest.fail("importing lassi.llm registers no LLMBackend hf_local yet (task P17.4)")
    return DEFAULT_REGISTRY.get("LLMBackend", "hf_local").factory


def manifest(run_dir: Path) -> dict:
    """Return the run's provenance.json."""
    return json.loads(read_ascii(run_dir / "provenance.json"))


@pytest.fixture
def hf_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return the test's HF_HOME, after the clean run environment (tests/llm/hf_runs.py) is set up."""
    return clean_environment(tmp_path, monkeypatch)


def test_provenance_records_the_device_record_and_framework_versions(tmp_path: Path, hf_home: Path) -> None:
    build_snapshot(hf_home, context=WIDE_CONTEXT)
    run_dir = run(tmp_path, make_registry(hf_local_class()), recipe_data(model_section()), "hf-cpu", FakeCpuProbe())
    written = manifest(run_dir)
    record = {
        "key": "model.device", "kind": "cpu", "indices": [], "name": CPU_NAME, "count": 8, "memory_bytes": 4096,
        "driver": None, "runtime": None, "framework": "torch", "framework_version": torch.__version__,
    }
    assert written["device_records"] == [record]
    serving = written["serving"]
    assert (serving["model"], serving["revision"], serving["seed"]) == (REPO_ID, REVISION, SEED)
    assert (serving["device"], serving["max_position_embeddings"]) == ("cpu", WIDE_CONTEXT)
    versions = serving["versions"]
    assert versions["torch"] == torch.__version__
    assert versions["transformers"] == importlib.metadata.version("transformers")
    expected_line = (
        f"transformers {versions['transformers']}; torch {versions['torch']}; revision {REVISION[:12]}; "
        f"max_position_embeddings {WIDE_CONTEXT}; device cpu; seed {SEED}"
    )
    assert f"| Server | {expected_line} |" in read_ascii(run_dir / "run.md")
    trial = read_trial(trial_dir(run_dir, trial_id("hf-cpu")), TextStore(run_dir))
    assert trial.model.backend == "hf_local" and trial.model.id == REPO_ID
    assert [item.framework_version for item in trial.provenance.device_records] == [torch.__version__]
    assert trial.requests and trial.requests[0].stage == "generate", "the model was asked"
    assert written["status"] == "complete"


def test_resolved_recipe_and_hash_carry_revision_and_seed(tmp_path: Path, hf_home: Path) -> None:
    build_snapshot(hf_home, context=WIDE_CONTEXT)
    registry = make_registry(hf_local_class())
    first = run(tmp_path, registry, recipe_data(model_section()), "hf-hash", FakeCpuProbe(), run_id="seed-a")
    other = recipe_data(model_section(seed=SEED + 1))
    second = run(tmp_path, registry, other, "hf-hash", FakeCpuProbe(), run_id="seed-b")
    resolved = yaml.safe_load(read_ascii(first / "recipe.resolved.yaml"))
    assert (resolved["model"]["revision"], resolved["model"]["seed"]) == (REVISION, SEED)
    hashes = [manifest(path)["recipe_hash"] for path in (first, second)]
    assert hashes[0] != hashes[1], "the seed enters the recipe hash"
    rows = read_run_parquet(first / "parquet")["trials"]
    assert [row["recipe_hash"] for row in rows] == [hashes[0]]


def test_a_context_refusal_ends_the_trial_and_the_run_goes_on(tmp_path: Path, hf_home: Path) -> None:
    # The generation prompt alone passes 64 - 48 = 16 tokens, so the first request of each trial is refused.
    build_snapshot(hf_home, context=64)
    data = recipe_data(model_section(), max_tokens=48, trials=2)
    run_dir = run(tmp_path, make_registry(hf_local_class()), data, "hf-context", FakeCpuProbe())
    assert manifest(run_dir)["status"] == "complete", "a refused request fails the trial, not the run"
    assert (run_dir / "run.md").is_file()
    for number in (1, 2):
        trial = read_trial(trial_dir(run_dir, trial_id("hf-context", number)), TextStore(run_dir))
        reason = trial.final.end_reason
        assert reason is not None and reason.code == CONTEXT_EXCEEDED, reason
        assert "generate" in reason.message and "64" in reason.message and "48" in reason.message, reason.message
        assert trial.attempts == [] and trial.requests == [], "nothing was answered, so nothing is recorded"
        assert CONTEXT_EXCEEDED in read_ascii(trial_dir(run_dir, trial_id("hf-context", number)) / "trial.md")
    rows = read_run_parquet(run_dir / "parquet")["trials"]
    assert [row["final_end_reason_code"] for row in rows] == [CONTEXT_EXCEEDED] * 2


def test_a_missing_snapshot_is_refused_before_any_directory(tmp_path: Path, hf_home: Path) -> None:
    build_snapshot(hf_home, revision=REVISION)
    probe = FakeCpuProbe()
    data = recipe_data(model_section(revision=OTHER_REVISION))
    with pytest.raises(RunError) as info:
        run(tmp_path, make_registry(hf_local_class()), data, "hf-missing", probe)
    message = str(info.value)
    assert REPO_ID in message and OTHER_REVISION in message and "model.id" in message, message
    assert "base_url" not in message, message
    assert not (tmp_path / "runs-root").exists(), "the model loads at the check, before any directory exists"


def test_an_unset_hf_home_is_refused_before_any_directory(
    tmp_path: Path, hf_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_snapshot(hf_home)
    monkeypatch.delenv("HF_HOME")
    with pytest.raises(RunError) as info:
        run(tmp_path, make_registry(hf_local_class()), recipe_data(model_section()), "hf-no-home", FakeCpuProbe())
    assert "HF_HOME" in str(info.value), str(info.value)
    assert not (tmp_path / "runs-root").exists()
