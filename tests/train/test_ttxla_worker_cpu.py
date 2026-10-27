"""The ttxla worker's training loop on the CPU, with the fixture model (task P17.38; needs the cpu extra).

Run with `uv run --extra cpu pytest tests/train/test_ttxla_worker_cpu.py`.

Bible: Training Module (Weight Modes, Compute, Safeguards), Agent Rule 1;
plans/spikes/tt-training.md, Finding 3 and its test list: only the device
differs on the card, so the loop runs here on the CPU with LASSI's own
torch, transformers, and peft (the cpu extra's versions, not the tt-xla
venv's), a few steps of each weight mode on fixture:tiny-causal-lm.

The worker is run as on the card, a script under an interpreter with
`-I`, given the job file the ttxla trainer builds (worker_job), with its
backend set to cpu, which only this test writes. These tests fix: each of
full, lora, and prompt_tuning exits 0 and leaves its checkpoint (the full
weights or the adapter, and the tokenizer), one steps line per step with a
finite loss, and result.json (status complete, global_step, the token
counts, the parameter counts, the framework versions); a record longer
than trainer.max_length exits 3 with status refused naming the record
before any step; chat_format on a base whose tokenizer has no chat
template is refused the same way. Losses here are SYNTHETIC: random
weights, four records; no value is a measurement.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import ttxla_fakes as fakes
from train_fakes import REPO

pytestmark = pytest.mark.slow

NEEDS_EXTRA = "needs the cpu extra: uv run --extra cpu pytest tests/train/test_ttxla_worker_cpu.py"
pytest.importorskip("torch", reason=NEEDS_EXTRA)
pytest.importorskip("transformers", reason=NEEDS_EXTRA)
pytest.importorskip("peft", reason=NEEDS_EXTRA)

from lassi.core.interfaces import TrainData  # noqa: E402
from lassi.train import ttxla_trainer  # noqa: E402

WORKER = REPO / "lassi" / "train" / "ttxla_worker.py"
FIXTURE = REPO / "tests" / "fixtures" / "train" / "sft-smoke.jsonl"
CONFIG: dict[str, Any] = {
    "device": {"kind": "tenstorrent", "indices": [0]}, "steps": 3, "seed": 7, "batch_size": 2,
    "learning_rate": 0.01, "max_length": 48, "timeout_s": 600, "network": "none",
}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set HF_HOME, LASSI_SCRATCH, and LASSI_TOOLCHAINS (a SYNTHETIC venv) for building the trainer."""
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("LASSI_SCRATCH", str(tmp_path))
    monkeypatch.setenv("LASSI_TOOLCHAINS", str(tmp_path / "toolchains"))
    fakes.write_venv(tmp_path / "toolchains")
    return tmp_path


def records() -> tuple[dict[str, str], ...]:
    """Return the four SYNTHETIC sft records of the committed fixture."""
    lines = FIXTURE.read_text(encoding="ascii").splitlines()
    return tuple(json.loads(line) for line in lines)


def run_worker(tmp_path: Path, weights: str, *, items: tuple[dict[str, str], ...] | None = None,
               **settings: Any) -> tuple[subprocess.CompletedProcess[str], Path, dict[str, Any]]:
    """Build the trainer's job for `weights` with backend cpu, run the worker on it, and return what it left."""
    config = {**CONFIG, **settings}
    trainer = ttxla_trainer.TtxlaTrainer(**config)
    recipe: dict[str, Any] = {"base_model": "fixture:tiny-causal-lm", "method": "sft", "weights": weights,
                              "trainer": {"kind": "ttxla", **config}}
    if weights == "lora":
        recipe["lora"] = {"r": 4, "targets": "all-linear"}
    data = TrainData(source="synthetic", split_hash="0" * 64, records=items or records(), suite=None, items=())
    out = tmp_path / weights / "output"
    work = out / "worker"
    work.mkdir(parents=True)
    job = {**trainer.worker_job(recipe, data, work, out / f"checkpoint-{config['steps']}"), "backend": "cpu"}
    path = work / "job.json"
    path.write_bytes(json.dumps(job).encode("ascii"))
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    done = subprocess.run([sys.executable, "-I", str(WORKER), str(path)], capture_output=True, text=True,
                          timeout=900, check=False, env=environment)
    result_file = work / "result.json"
    result = json.loads(result_file.read_text(encoding="ascii")) if result_file.exists() else {}
    return done, out, result


@pytest.mark.parametrize(
    ("weights", "settings", "weights_file"),
    [
        pytest.param("full", {}, "model.safetensors", id="full"),
        pytest.param("lora", {"gradient_checkpointing": True}, "adapter_model.safetensors", id="lora-recompute"),
        pytest.param("prompt_tuning", {"virtual_tokens": 4}, "adapter_model.safetensors", id="prompt-tuning"),
    ],
)
def test_each_weight_mode_trains_and_saves_on_the_cpu(
    env: Path, tmp_path: Path, weights: str, settings: dict[str, Any], weights_file: str
) -> None:
    done, out, result = run_worker(tmp_path, weights, **settings)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert result["status"] == "complete" and result["global_step"] == 3
    checkpoint = out / "checkpoint-3"
    assert (checkpoint / weights_file).is_file(), sorted(path.name for path in checkpoint.iterdir())
    assert (checkpoint / "tokenizer.json").is_file()
    lines = [json.loads(line) for line in (out / "worker" / "steps.jsonl").read_text(encoding="ascii").splitlines()]
    assert [line["step"] for line in lines] == [1, 2, 3]
    assert all(isinstance(line["loss"], float) and math.isfinite(line["loss"]) for line in lines), lines
    assert result["tokens"]["max_record"] <= CONFIG["max_length"]
    assert 0 < result["parameters"]["trainable"] <= result["parameters"]["total"]
    if weights != "full":
        assert result["parameters"]["trainable"] < result["parameters"]["total"], "the base stays frozen"
    assert result["versions"]["torch"] and result["versions"]["transformers"]


def test_a_record_longer_than_max_length_is_refused_before_any_step(env: Path, tmp_path: Path) -> None:
    long = ({"prompt": "SYNTHETIC " * 40, "completion": " abc"}, *records())
    done, out, result = run_worker(tmp_path, "full", items=long)
    assert done.returncode == 3, done.stderr[-3000:]
    assert result["status"] == "refused" and "record 1" in result["message"], result
    assert "trainer.max_length" in result["message"]
    assert not (out / "worker" / "steps.jsonl").exists() or not (out / "worker" / "steps.jsonl").read_text()
    assert not (out / "checkpoint-3").exists()


def test_chat_format_without_a_chat_template_is_refused(env: Path, tmp_path: Path) -> None:
    done, _, result = run_worker(tmp_path, "full", chat_format=True)
    assert done.returncode == 3, done.stderr[-3000:]
    assert "chat template" in result["message"]
