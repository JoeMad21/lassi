"""The committed train recipes for tt-train-v0 (2026-10-23; OQ-076 and the owner's direction of 2026-10-23).

Bible: Training Module (Weight Modes, Compute), Project Recipes (train
recipes), Agent Rules 5, 7, 10, and 17.

The contract these tests fix, for each projects/train/*.yaml:

- It loads with lassi's registry: method sft on data.corpus tt-train-v0,
  trainer trl on the CPU (alpha01's CPU; no GPU trains), batch size 1,
  max_length 4096, gradient checkpointing off, the chat format on, and a
  base pinned by a full commit id or a scratch config whose tokenizer is
  pinned. Together they cover the owner's techniques: full weights on
  Qwen3-0.6B, lora and prompt_tuning on Qwen3-1.7B, and a model from
  scratch, each base a Qwen3ForCausalLM (an architecture furiosa-llm
  compiles; the coordinator's note of 2026-10-23).
- Each carries its PROJECTED time arithmetic in its comments.
- Without the framework (as on Windows here), lassi train refuses it at
  the device probe, the framework build of trainer.device, naming the cpu
  extra, before any directory exists, once the records load.
- With a SYNTHETIC framework build, a Hub cache holding the pinned
  snapshots, and synthetic corpus records, every refusal passes and
  train() is reached: the trainer's check() carries the recipe out.

The corpus records and the cache are SYNTHETIC; no value is a measurement.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from train_fakes import RECORDS, REPO, Log, fake_probes, jsonl, train_run
from trl_smoke import FRAMEWORK_MODULES, SYNTHETIC_BUILD, trl_trainer

import lassi.train  # noqa: F401  (registers the trl trainer)
from lassi.bench import load_suite
from lassi.bench.registry import HELD_SPLITS
from lassi.core.devices import FrameworkBuild
from lassi.core.interfaces import TrainResult
from lassi.core.recipe import load_train_recipe
from lassi.core.runner import RunError
from lassi.train import bases

RECIPES = REPO / "projects" / "train"
NAMES = ("tt-full-qwen3-0.6b", "tt-lora-qwen3-1.7b", "tt-prompt-tuning-qwen3-1.7b", "tt-scratch-qwen3-77m")
WEIGHTS = {"tt-full-qwen3-0.6b": "full", "tt-lora-qwen3-1.7b": "lora", "tt-prompt-tuning-qwen3-1.7b": "prompt_tuning",
           "tt-scratch-qwen3-77m": "full"}
SUITE = REPO / "assets" / "bench" / "tt-pairs-v0.yaml"


def recipe(name: str) -> Path:
    """Return the path of a committed train recipe."""
    return RECIPES / f"{name}.yaml"


def test_the_recipes_are_the_four_techniques() -> None:
    assert sorted(path.stem for path in RECIPES.glob("*.yaml")) == sorted(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_each_recipe_loads_on_the_cpu_from_the_corpus(name: str) -> None:
    data = load_train_recipe(recipe(name)).data
    assert (data["method"], data["weights"], data["data"]) == ("sft", WEIGHTS[name], {"corpus": "tt-train-v0"})
    trainer = data["trainer"]
    assert trainer["kind"] == "trl" and trainer["device"] == {"kind": "cpu"}
    assert (trainer["batch_size"], trainer["max_length"]) == (1, 4096)
    assert trainer["gradient_checkpointing"] is False and trainer["chat_format"] is True
    base = bases.parse_base(data["base_model"])
    assert base.kind in ("hub", "scratch")
    if base.kind == "hub":
        assert base.repo in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B")
    assert "runs_root" not in data and "bench" not in data


@pytest.mark.parametrize("name", NAMES)
def test_each_recipe_carries_its_projected_arithmetic(name: str) -> None:
    text = recipe(name).read_bytes().decode("ascii")
    assert "PROJECTED" in text and "TFLOP/s" in text and "tools/prepare_train.py" in text
    assert "hf_local" in text


def test_the_scratch_config_is_a_qwen3_model_of_50_to_100m_parameters() -> None:
    directory = bases.SCRATCH_DIR / "qwen3-77m"
    config = json.loads((directory / "config.json").read_bytes())
    assert config["architectures"] == ["Qwen3ForCausalLM"] and config["model_type"] == "qwen3"
    hidden, layers, vocab = config["hidden_size"], config["num_hidden_layers"], config["vocab_size"]
    heads, groups, head_dim = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
    layer = hidden * heads * head_dim * 2 + hidden * groups * head_dim * 2 + 3 * hidden * config["intermediate_size"]
    total = vocab * hidden + layers * layer
    assert 50_000_000 <= total <= 100_000_000, total
    assert bases.read_tokenizer_pin(directory / "tokenizer-pin.json") == (
        "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca")


def synthetic_corpus(root: Path) -> tuple[Path, Path]:
    """Write a SYNTHETIC tt-train-v0 manifest and records under `root`; return (manifests, records)."""
    content = jsonl(RECORDS)
    held = sorted(name for name, item in load_suite(SUITE).items.items() if item.split in HELD_SPLITS)
    manifest = {"corpus": "tt-train-v0", "records": {"file": "tt-train-v0.jsonl",
                                                     "sha256": hashlib.sha256(content).hexdigest(), "count": 2},
                "held_out": {"suite": "tt-pairs-v0", "items": held}}
    (root / "manifests" / "tt-train-v0").mkdir(parents=True)
    (root / "manifests" / "tt-train-v0" / "manifest.json").write_bytes(json.dumps(manifest).encode("ascii"))
    (root / "records").mkdir()
    (root / "records" / "tt-train-v0.jsonl").write_bytes(content)
    return root / "manifests", root / "records"


def options(tmp_path: Path, **changes: Any) -> Any:
    """Return TrainOptions with the SYNTHETIC corpus, a runs root under tmp_path, and the given changes."""
    manifests, records = synthetic_corpus(tmp_path / "corpus")
    values: dict[str, Any] = {"runs_root": tmp_path / "runs-root", "train_id": "t1", "corpus_root": records,
                              "corpus_manifests": manifests}
    values.update(changes)
    return train_run().TrainOptions(**values)


@pytest.mark.parametrize("name", NAMES)
def test_without_the_framework_lassi_train_stops_at_the_device_probe(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trl_trainer()
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RunError) as refused:
        train_run().run_training(recipe(name), options(tmp_path, probes=fake_probes(Log())))
    message = str(refused.value)
    assert "trainer.device" in message and "--extra cpu" in message, message
    assert not (tmp_path / "runs-root").exists()


def prepared_cache(hf_home: Path) -> None:
    """Write SYNTHETIC snapshots of every pinned Hub base and tokenizer the recipes name."""
    for name in NAMES:
        base = bases.parse_base(load_train_recipe(recipe(name)).data["base_model"])
        if base.kind == "scratch":
            assert base.path is not None
            repo, revision = bases.read_tokenizer_pin(base.path / bases.TOKENIZER_PIN)
        else:
            assert base.repo is not None and base.revision is not None
            repo, revision = base.repo, base.revision
        snapshot = bases.snapshot_dir(str(hf_home), repo, revision)
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_bytes(b"{}\n")
        (snapshot / "tokenizer.json").write_bytes(b"{}\n")


@pytest.mark.parametrize("name", NAMES)
def test_with_a_framework_and_a_prepared_cache_every_refusal_passes(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cls = trl_trainer().TRLTrainer
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    prepared_cache(tmp_path / "hf")
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda module, package=None: (
        importlib.machinery.ModuleSpec(module, None) if module.split(".")[0] in FRAMEWORK_MODULES
        else real(module, package)))
    monkeypatch.setattr(cls, "framework", staticmethod(lambda: FrameworkBuild(*SYNTHETIC_BUILD)))
    seen: list[Any] = []
    monkeypatch.setattr(cls, "train", lambda self, job: seen.append(job) or TrainResult(steps=0, checkpoints=()))
    train_dir = train_run().run_training(recipe(name), options(tmp_path, probes=fake_probes(Log())))
    assert len(seen) == 1 and seen[0].data.source == "corpus"
    provenance = json.loads((train_dir / "provenance.json").read_bytes())
    assert provenance["data"]["name"] == "tt-train-v0" and provenance["status"] == "complete"
