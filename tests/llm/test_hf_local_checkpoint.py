"""Tests of hf_local serving a trained checkpoint, without the framework (2026-10-23; OQ-076 (3a)).

Bible: Model Serving (Serving Rules), Training Module (Weight Modes), Agent
Rules 7 and 10.

The contract these tests fix (lassi.llm.hf_local):

- Two optional model-section keys, model.checkpoint (an absolute path to a
  checkpoint directory lassi train wrote) and model.checkpoint_sha256 (its
  content digest, lassi.core.digest tree_sha256, which lassi train records
  in provenance.json as checkpoint_sha256), are set together or not at all.
  model.id and model.revision still pin the Hub snapshot the checkpoint was
  trained from: its base, or a scratch base's tokenizer. Construction
  checks their form and reads no file. A recipe without them builds and
  loads exactly as before.
- check() then refuses, with a ServingError and before any framework call,
  a checkpoint whose digest is not model.checkpoint_sha256, one whose
  checkpoint.json names a fixture base, a base other than model.id at
  model.revision, weights it cannot serve, or a prompt_tuning checkpoint
  whose peft record is missing, not a mapping, or names no soft-prompt
  length of at least 1. It loads full weights (a Hub
  or scratch base fine-tuned whole) from the checkpoint directory alone,
  and lora and prompt_tuning weights as the base from $HF_HOME/hub at the
  revision with the adapter applied by PEFT's PeftModel.from_pretrained
  (is_trainable False), every load local_files_only with no remote code.
  The tokenizer is the checkpoint's own (lassi train saves it).
- A prompt-tuned model's soft prompt takes num_virtual_tokens positions of
  the context, so the context rule counts them (prompt_positions).
- serving() adds checkpoint and checkpoint_sha256 only when a checkpoint is
  served.

The framework here is fake modules; every value is SYNTHETIC and none is a
measurement.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hf_runs import model_section, recipe_data, write_recipe
from tiny_hf import REPO_ID, REVISION

from lassi.core.digest import tree_sha256
from lassi.core.recipe import load_recipe
from lassi.llm import ServingError

CPU = {"kind": "cpu"}
SEED = 7
OTHER_REPO = "SYNTHETIC-org/other-model"


def hf_local() -> Any:
    """Import lassi.llm.hf_local."""
    return importlib.import_module("lassi.llm.hf_local")


def stand_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let construction find torch and transformers without importing either."""
    real = importlib.util.find_spec

    def find_spec(name: str, package: str | None = None) -> Any:
        if name in ("torch", "transformers"):
            return importlib.machinery.ModuleSpec(name, None)
        return real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def write_checkpoint(root: Path, weights: str, base: dict[str, Any], peft: Any = None) -> Path:
    """Write a SYNTHETIC checkpoint directory with checkpoint.json naming `weights` and `base`; return it."""
    root.mkdir(parents=True)
    (root / "adapter_model.safetensors" if weights != "full" else root / "model.safetensors").write_bytes(b"SYN")
    (root / "tokenizer.json").write_bytes(b"{}\n")
    records = {"trainer_records": {"weights": weights, "base_model": base, "peft": peft}}
    (root / "checkpoint.json").write_bytes((json.dumps(records) + "\n").encode("ascii"))
    return root


HUB_BASE = {"name": f"{REPO_ID}@{REVISION}", "kind": "hub", "repo": REPO_ID, "revision": REVISION}
SCRATCH_BASE = {"name": "scratch:tiny", "kind": "scratch", "tokenizer": {"repo": REPO_ID, "revision": REVISION}}
PROMPT_PEFT = {"type": "prompt_tuning", "num_virtual_tokens": 6}


def build(monkeypatch: pytest.MonkeyPatch, checkpoint: Path | None, sha: str | None) -> Any:
    """Build hf_local on REPO_ID at REVISION with the checkpoint keys given (None leaves one out)."""
    stand_in(monkeypatch)
    settings: dict[str, Any] = {"device": dict(CPU), "revision": REVISION, "seed": SEED}
    if checkpoint is not None:
        settings["checkpoint"] = str(checkpoint)
    if sha is not None:
        settings["checkpoint_sha256"] = sha
    return hf_local().HFLocalBackend(REPO_ID, **settings)


# ---------------------------------------------------------------------------
# Declarations and construction


def test_hf_local_declares_the_checkpoint_keys() -> None:
    assert hf_local().HFLocalBackend.config_keys == frozenset({"revision", "seed", "checkpoint", "checkpoint_sha256"})
    assert hf_local().CHECKPOINT_RECORDS == importlib.import_module("lassi.train.checkpoint").RECORDS_JSON


@pytest.mark.parametrize(
    ("where", "sha", "key"),
    [
        ("absolute", None, "model.checkpoint_sha256"),
        (None, "0" * 64, "model.checkpoint"),
        ("relative", "0" * 64, "model.checkpoint"),
        ("absolute", "0" * 63, "model.checkpoint_sha256"),
        ("absolute", "Z" * 64, "model.checkpoint_sha256"),
    ],
)
def test_the_checkpoint_keys_are_checked_at_construction(
    where: str | None, sha: str | None, key: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = {None: None, "absolute": tmp_path / "ckpt", "relative": Path("relative") / "ckpt"}[where]
    with pytest.raises(ValueError, match=key):
        build(monkeypatch, path, sha)


def test_construction_reads_no_checkpoint_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    backend = build(monkeypatch, tmp_path / "absent", "0" * 64)
    assert backend.checkpoint == tmp_path / "absent" and backend.checkpoint_sha256 == "0" * 64


def test_a_recipe_without_a_checkpoint_loads_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []
    loaded = hf_local()._Loaded(tokenizer=None, model=None, context=64, stops=(1,), pad=0, device="cpu", dtype="f32")
    monkeypatch.setattr(hf_local(), "_load", lambda *args: calls.append(args) or loaded)
    monkeypatch.setattr(hf_local(), "_load_checkpoint", lambda *args: pytest.fail("a checkpoint load ran"))
    backend = build(monkeypatch, None, None)
    assert backend.check() == {"id": REPO_ID, "revision": REVISION, "max_position_embeddings": 64}
    assert calls == [(REPO_ID, REVISION, "cpu")]
    assert loaded.virtual_tokens == 0 and hf_local().prompt_positions(10, loaded) == 10


# ---------------------------------------------------------------------------
# Refusals before any framework call


@pytest.fixture
def no_framework(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the checkpoint path imports any framework module."""
    monkeypatch.setattr(hf_local(), "_import", lambda name: pytest.fail(f"{name} was imported"))


def test_a_digest_mismatch_is_refused(no_framework: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path / "ckpt", "full", HUB_BASE)
    with pytest.raises(ServingError) as refused:
        build(monkeypatch, checkpoint, "0" * 64).check()
    assert tree_sha256(checkpoint) in str(refused.value) and "0" * 64 in str(refused.value)


@pytest.mark.parametrize(
    ("weights", "base", "needle"),
    [
        ("full", {"name": "fixture:tiny", "files": {}, "seed": 1}, "fixture"),
        ("full", {**HUB_BASE, "repo": OTHER_REPO}, OTHER_REPO),
        ("lora", {**HUB_BASE, "revision": "f" * 40}, "f" * 40),
        ("lora", SCRATCH_BASE, "scratch"),
        ("dora", HUB_BASE, "dora"),
    ],
    ids=["fixture-base", "other-repo", "other-revision", "adapter-on-scratch", "unknown-weights"],
)
def test_a_checkpoint_hf_local_cannot_serve_is_refused(
    weights: str, base: dict[str, Any], needle: str, no_framework: None, monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = write_checkpoint(tmp_path / "ckpt", weights, base)
    with pytest.raises(ServingError) as refused:
        build(monkeypatch, checkpoint, tree_sha256(checkpoint)).check()
    assert needle in str(refused.value), str(refused.value)


@pytest.mark.parametrize(
    "peft",
    [None, [6], "prompt_tuning", {"type": "prompt_tuning"}, {"type": "prompt_tuning", "num_virtual_tokens": 0},
     {"type": "prompt_tuning", "num_virtual_tokens": True}],
    ids=["missing", "a-list", "a-string", "no-length", "zero-length", "bool-length"],
)
def test_a_prompt_tuning_checkpoint_without_a_soft_prompt_record_is_refused(
    peft: Any, no_framework: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A prompt-tuned checkpoint must say how many positions its soft prompt takes, or the context rule would count
    # none (the audit of 2026-10-24).
    checkpoint = write_checkpoint(tmp_path / "ckpt", "prompt_tuning", HUB_BASE, peft)
    with pytest.raises(ServingError) as refused:
        build(monkeypatch, checkpoint, tree_sha256(checkpoint)).check()
    assert "peft" in str(refused.value) and "num_virtual_tokens" in str(refused.value), str(refused.value)


# ---------------------------------------------------------------------------
# Loads, with fake modules


class Recorder:
    """A fake Auto class or PeftModel: records from_pretrained calls and returns what `make` builds."""

    def __init__(self, calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]], label: str, make: Any) -> None:
        """Keep the call list, the label, and the factory of results."""
        self.calls, self.label, self.make = calls, label, make

    def from_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        """Record the call and return a fresh result."""
        self.calls.append((self.label, args, kwargs))
        return self.make()


def fake_model() -> Any:
    """Return a stand-in model with a generation config, a dtype, and to/eval."""
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2), dtype="torch.float32")
    model.to = lambda device: model
    model.eval = lambda: model
    return model


def fake_modules(calls: list[Any]) -> dict[str, Any]:
    """Return fake torch, transformers, and peft modules whose loads record into `calls`."""
    tokenizer = SimpleNamespace(chat_template="SYNTHETIC", eos_token_id=2, pad_token_id=0)
    transformers = SimpleNamespace(
        AutoConfig=Recorder(calls, "AutoConfig", lambda: SimpleNamespace(max_position_embeddings=64)),
        AutoTokenizer=Recorder(calls, "AutoTokenizer", lambda: tokenizer),
        AutoModelForCausalLM=Recorder(calls, "AutoModelForCausalLM", fake_model),
        GenerationConfig=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    peft = SimpleNamespace(PeftModel=Recorder(calls, "PeftModel", fake_model))
    return {"torch": SimpleNamespace(), "transformers": transformers, "peft": peft}


@pytest.fixture
def framework(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[Any]:
    """Install fake framework modules and HF_HOME; return the list of recorded loads."""
    calls: list[Any] = []
    modules = fake_modules(calls)
    monkeypatch.setattr(hf_local(), "_import", lambda name: modules[name])
    monkeypatch.setattr(hf_local(), "hub_offline", lambda: calls.append(("hub_offline", (), {})))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    return calls


@pytest.mark.parametrize("base", [HUB_BASE, SCRATCH_BASE], ids=["hub", "scratch"])
def test_full_weights_load_from_the_checkpoint_alone(
    base: dict[str, Any], framework: list[Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = write_checkpoint(tmp_path / "ckpt", "full", base)
    backend = build(monkeypatch, checkpoint, tree_sha256(checkpoint))
    assert backend.check()["id"] == REPO_ID
    loads = [call for call in framework if call[0] != "hub_offline"]
    assert framework[0][0] == "hub_offline", "the hub library is offline before any load"
    assert {call[0] for call in loads} == {"AutoTokenizer", "AutoConfig", "AutoModelForCausalLM"}
    for _, args, kwargs in loads:
        assert args == (str(checkpoint),)
        assert kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False
        assert "revision" not in kwargs and "cache_dir" not in kwargs


@pytest.mark.parametrize("weights", ["lora", "prompt_tuning"])
def test_adapter_weights_load_the_pinned_base_then_the_adapter(
    weights: str, framework: list[Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    peft = PROMPT_PEFT if weights == "prompt_tuning" else {"r": 8, "target_modules": "all-linear"}
    checkpoint = write_checkpoint(tmp_path / "ckpt", weights, HUB_BASE, peft)
    backend = build(monkeypatch, checkpoint, tree_sha256(checkpoint))
    backend.check()
    by_label = {call[0]: call for call in framework}
    base_call = by_label["AutoModelForCausalLM"]
    assert base_call[1] == (REPO_ID,)
    assert base_call[2]["revision"] == REVISION and base_call[2]["cache_dir"] == str(tmp_path / "hf" / "hub")
    assert base_call[2]["local_files_only"] is True and base_call[2]["trust_remote_code"] is False
    adapter = by_label["PeftModel"]
    assert adapter[1][1] == str(checkpoint) and adapter[2]["is_trainable"] is False
    assert by_label["AutoTokenizer"][1] == (str(checkpoint),), "the tokenizer is the checkpoint's own"
    loaded = backend._model()
    expected = PROMPT_PEFT["num_virtual_tokens"] if weights == "prompt_tuning" else 0
    assert loaded.virtual_tokens == expected
    assert hf_local().prompt_positions(10, loaded) == 10 + expected


def test_serving_names_the_checkpoint(framework: list[Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path / "ckpt", "full", HUB_BASE)
    backend = build(monkeypatch, checkpoint, tree_sha256(checkpoint))
    torch = SimpleNamespace(get_num_threads=lambda: 4)
    real_import = hf_local()._import
    monkeypatch.setattr(hf_local(), "_import", lambda name: torch if name == "torch" else real_import(name))
    record = backend.serving()
    assert (record["checkpoint"], record["checkpoint_sha256"]) == (str(checkpoint), tree_sha256(checkpoint))
    assert record["model"] == REPO_ID and record["revision"] == REVISION


# ---------------------------------------------------------------------------
# A run recipe binds a checkpoint


def test_a_run_recipe_with_a_checkpoint_loads_and_passes_both_keys(tmp_path: Path) -> None:
    checkpoint = str(tmp_path / "ckpt")
    data = recipe_data(model_section(checkpoint=checkpoint, checkpoint_sha256="a" * 64))
    loaded = load_recipe(write_recipe(tmp_path, "ckpt", data))
    model = next(binding for binding in loaded.bindings if binding.interface == "LLMBackend")
    assert (model.config["checkpoint"], model.config["checkpoint_sha256"]) == (checkpoint, "a" * 64)
