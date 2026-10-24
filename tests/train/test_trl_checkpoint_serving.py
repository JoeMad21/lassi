"""Slow: train a tiny Hub-style base with each weight mode, then serve the checkpoint through hf_local (2026-10-23).

Bible: Training Module (Weight Modes), Model Serving (Serving Rules), Agent
Rules 7 and 10; OQ-076 (3a): a trained model is served to the pipeline in
process on the CPU (hf_local).

The contract these tests fix, end to end on the CPU with the cpu extra:

- A Hub base `<owner>/<name>@<revision>` loads from $HF_HOME/hub at the
  revision with nothing downloaded: here tests/tiny_hf.py build_snapshot's
  tiny chat model, whose snapshot the test writes (the prepare step's
  stand-in). sft with chat_format true trains it in the chat template for
  two steps with full, lora, and prompt_tuning weights; checkpoint.json
  records the base's kind, repo, revision, and every snapshot file's
  sha256, and provenance.json records the checkpoint's content digest.
- hf_local, given model.checkpoint and that digest with model.id and
  model.revision naming the base, loads the checkpoint (the adapter on the
  base for lora and prompt_tuning) and completes one request; its serving
  record names the checkpoint. A changed checkpoint file is refused.

These tests need the cpu extra and skip without it (tests/tiny_hf.py
NEEDS_EXTRA); CI runs them. Every weight is random and every text
SYNTHETIC; no value is a measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tiny_hf import NEEDS_EXTRA, REPO_ID, REVISION, build_snapshot
from train_fakes import write_recipe
from trl_smoke import smoke_recipe, trl_trainer

pytestmark = pytest.mark.slow

pytest.importorskip("torch", reason=NEEDS_EXTRA)
pytest.importorskip("transformers", reason=NEEDS_EXTRA)
pytest.importorskip("trl", reason=NEEDS_EXTRA)
pytest.importorskip("peft", reason=NEEDS_EXTRA)

import lassi.cli  # noqa: E402,F401  (registers the host's device probes and every trainer)
from lassi.core.interfaces import Message, Sampling  # noqa: E402
from lassi.llm import ServingError  # noqa: E402
from lassi.llm.hf_local import HFLocalBackend  # noqa: E402
from lassi.train import run as train_run  # noqa: E402

BASE = f"{REPO_ID}@{REVISION}"


def train(tmp_path: Path, weights: str) -> tuple[Path, str]:
    """Train the sft smoke on the tiny Hub-style base with `weights`; return the checkpoint and its digest."""
    trl_trainer()
    recipe = smoke_recipe("sft", weights, base_model=BASE)
    recipe["trainer"].update({"chat_format": True, "max_length": 192, "gradient_checkpointing": False})
    path = write_recipe(tmp_path / "recipes", recipe, name=f"sft-{weights}")
    options = train_run.TrainOptions(runs_root=tmp_path / "runs-root", train_id=weights, roots=[tmp_path / "recipes"])
    train_dir = train_run.run_training(path, options)
    provenance = json.loads((train_dir / "provenance.json").read_bytes())
    (name,) = provenance["checkpoints"]
    return train_dir / "output" / name, provenance["checkpoint_sha256"][name]


@pytest.fixture
def hf_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build the tiny chat model's snapshot in a fresh HF_HOME and point HF_HOME at it."""
    home = tmp_path / "hf"
    build_snapshot(home, context=256)
    monkeypatch.setenv("HF_HOME", str(home))
    return home


@pytest.mark.parametrize("weights", ["full", "lora", "prompt_tuning"])
def test_a_trained_checkpoint_is_served_by_hf_local(weights: str, hf_home: Path, tmp_path: Path) -> None:
    checkpoint, digest = train(tmp_path, weights)
    records = json.loads((checkpoint / "checkpoint.json").read_bytes())["trainer_records"]
    base = records["base_model"]
    assert (base["kind"], base["repo"], base["revision"]) == ("hub", REPO_ID, REVISION)
    assert "config.json" in base["files"] and records["threads"] >= 1
    backend = HFLocalBackend(REPO_ID, device={"kind": "cpu"}, revision=REVISION, seed=7,
                             checkpoint=str(checkpoint), checkpoint_sha256=digest)
    assert backend.check()["id"] == REPO_ID
    reply = backend.complete([Message(role="user", content="SYNTHETIC request")], Sampling(0.0, 1.0, 4))
    assert reply.completion_tokens >= 1
    serving = backend.serving()
    assert (serving["checkpoint"], serving["checkpoint_sha256"]) == (str(checkpoint), digest)


def test_a_changed_checkpoint_is_refused(hf_home: Path, tmp_path: Path) -> None:
    checkpoint, digest = train(tmp_path, "lora")
    (checkpoint / "steps.jsonl").write_bytes((checkpoint / "steps.jsonl").read_bytes() + b"\n")
    backend = HFLocalBackend(REPO_ID, device={"kind": "cpu"}, revision=REVISION, seed=7,
                             checkpoint=str(checkpoint), checkpoint_sha256=digest)
    with pytest.raises(ServingError, match="content digest"):
        backend.check()
