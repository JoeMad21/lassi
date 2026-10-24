"""Tests for the trl Trainer's base kinds, prompt tuning, and sizing keys, without the framework (2026-10-23).

Bible: Training Module (Weight Modes, Compute), Project Recipes (train.yaml),
Agent Rules 7 and 10; OQ-076 (the owner's techniques: full training, frozen
weights, LoRA, and a model of our own).

The contract these tests fix (lassi.train.bases, lassi.train.trl_trainer):

- base_model takes three kinds. `fixture:<name>` is unchanged (task P17.9).
  `scratch:<name>` is the from-scratch baseline: a directory under
  assets/train/scratch/ (bases.SCRATCH_DIR) holding config.json and
  tokenizer-pin.json ({repo, revision}, a Hub tokenizer at a 40-hex
  commit); the model is built from the config with random weights under the
  seed. `<owner>/<name>@<revision>` is a Hub base at a full commit id. A
  Hub id without a revision is refused naming Agent Rule 10.
- A Hub base, and a scratch base's tokenizer, are read from $HF_HOME/hub
  only (models--<owner>--<name>/snapshots/<revision>/), never downloaded by
  check() or train(): check() refuses a snapshot the cache lacks, naming
  the snapshot and the separate prepare step, tools/prepare_train.py.
- weights prompt_tuning (the frozen-weights mode: every base weight frozen,
  a PEFT prompt-tuning soft prompt trained) needs trainer.virtual_tokens and
  method sft; trainer.virtual_tokens is refused for other weights. A scratch
  base trains full weights only: lora and prompt_tuning on random weights
  are refused.
- trainer.max_length (sft only) and trainer.gradient_checkpointing are
  optional settings, passed to TRL when set; building refuses a bad value
  naming the key.
- data.corpus is a declared data source; record refusals name the source.
- Without the framework, the helpers are fixed with fake modules: the Hub
  and scratch loads pass the revision, $HF_HOME/hub, local_files_only, and
  trust_remote_code False; prompt tuning builds PEFT's PromptTuningConfig
  with the virtual tokens, CAUSAL_LM, and RANDOM init; the SFT arguments
  carry max_length and gradient_checkpointing when set; and check_lengths
  refuses a record whose tokens plus TRL's end token pass max_length,
  naming the record, since TRL would cut it (nothing is truncated).

Every text and file here is SYNTHETIC; no value is a measurement.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from train_fakes import MISSING
from trl_smoke import (
    FRAMEWORK_MODULES,
    SYNTHETIC_DIR,
    smoke_recipe,
    train_module,
    trainer_section,
    trl_trainer,
    with_trainer,
)

from lassi.core.interfaces import TrainData
from lassi.train import data as train_data

HUB_REPO = "SYNTHETIC-org/tiny-model"
REVISION = "0123456789abcdef0123456789abcdef01234567"
HUB_BASE = f"{HUB_REPO}@{REVISION}"
TOKENIZER_REPO = "SYNTHETIC-org/tiny-tokenizer"


def bases() -> Any:
    """Import lassi.train.bases."""
    return train_module("bases")


def snapshot(hf_home: Path, repo: str, revision: str = REVISION) -> Path:
    """Write a SYNTHETIC snapshot of `repo` at `revision` into the hub cache under `hf_home`; return its directory."""
    owner, name = repo.split("/")
    directory = hf_home / "hub" / f"models--{owner}--{name}" / "snapshots" / revision
    directory.mkdir(parents=True)
    (directory / "config.json").write_bytes(b'{"model_type": "SYNTHETIC"}\n')
    (directory / "tokenizer.json").write_bytes(b"{}\n")
    return directory


def scratch_base(root: Path, name: str = "tiny-scratch", pin: Any = None) -> Path:
    """Write a SYNTHETIC scratch base directory under `root`; return it."""
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "config.json").write_bytes((SYNTHETIC_DIR / "tiny-causal-lm" / "config.json").read_bytes())
    document = {"repo": TOKENIZER_REPO, "revision": REVISION} if pin is None else pin
    (directory / "tokenizer-pin.json").write_bytes((json.dumps(document) + "\n").encode("ascii"))
    return directory


@pytest.fixture
def buildable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> type:
    """Return TRLTrainer with HF_HOME set, find_spec finding the framework, and every framework import blocked."""
    cls = trl_trainer().TRLTrainer
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    real = importlib.util.find_spec

    def find_spec(name: str, package: str | None = None) -> Any:
        if name.split(".")[0] in FRAMEWORK_MODULES:
            return importlib.machinery.ModuleSpec(name, None)
        return real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    for name in (*FRAMEWORK_MODULES, "huggingface_hub", "tokenizers", "safetensors"):
        monkeypatch.setitem(sys.modules, name, None)
    return cls


@pytest.fixture
def ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, buildable: type) -> Path:
    """Return HF_HOME (set by buildable) with the scratch directory moved under tmp_path."""
    monkeypatch.setattr(bases(), "SCRATCH_DIR", tmp_path / "scratch")
    return tmp_path / "hf"


def check(recipe: dict[str, Any], data: TrainData | None = None) -> None:
    """Build the trl trainer from the recipe's trainer section and run check() on the recipe."""
    trainer = trl_trainer().TRLTrainer(**trainer_section(recipe))
    records = train_data.load_synthetic(recipe["data"]["synthetic"], SYNTHETIC_DIR) if data is None else data
    trainer.check(recipe, records)


# ---------------------------------------------------------------------------
# Declarations


def test_the_trainer_declares_prompt_tuning_corpus_data_and_the_sizing_keys() -> None:
    cls = trl_trainer().TRLTrainer
    assert tuple(cls.weight_modes) == ("full", "lora", "prompt_tuning")
    assert tuple(cls.data_sources) == ("synthetic", "corpus")
    assert {"max_length", "gradient_checkpointing", "virtual_tokens"} <= set(cls.config_keys)


# ---------------------------------------------------------------------------
# Parsing base_model


@pytest.mark.parametrize(
    ("value", "kind"),
    [("fixture:tiny-causal-lm", "fixture"), ("scratch:tiny-scratch", "scratch"), (HUB_BASE, "hub")],
)
def test_parse_base_reads_each_kind(value: str, kind: str) -> None:
    found = bases().parse_base(value)
    assert found.kind == kind and found.name == value
    if kind == "hub":
        assert (found.repo, found.revision) == (HUB_REPO, REVISION)


@pytest.mark.parametrize(
    ("value", "needles"),
    [
        (HUB_REPO, ("base_model", "Agent Rule 10")),
        (f"{HUB_REPO}@main", ("base_model", "40")),
        (f"{HUB_REPO}@{REVISION[:12]}", ("base_model", "40")),
        ("scratch:", ("base_model",)),
        ("scratch:../escape", ("base_model",)),
        ("scratch:a/b", ("base_model",)),
        ("./tiny", ("base_model",)),
        ("C:/models/tiny", ("base_model",)),
        (7, ("base_model",)),
    ],
)
def test_parse_base_refuses_every_other_value(value: Any, needles: tuple[str, ...]) -> None:
    with pytest.raises(ValueError) as refused:
        bases().parse_base(value)
    assert all(needle in str(refused.value) for needle in needles), str(refused.value)


def test_snapshot_dir_is_the_hub_cache_layout(tmp_path: Path) -> None:
    found = bases().snapshot_dir(str(tmp_path), HUB_REPO, REVISION)
    assert found == tmp_path / "hub" / "models--SYNTHETIC-org--tiny-model" / "snapshots" / REVISION


# ---------------------------------------------------------------------------
# check(): the cache, the scratch directory, and the weight modes


def test_a_hub_base_the_cache_lacks_is_refused_naming_the_prepare_step(ready: Path) -> None:
    with pytest.raises(ValueError) as refused:
        check(smoke_recipe("sft", "full", base_model=HUB_BASE))
    message = str(refused.value)
    assert "base_model" in message and "tools/prepare_train.py" in message and REVISION in message, message


def test_a_prepared_hub_base_passes(ready: Path) -> None:
    snapshot(ready, HUB_REPO)
    for weights in ("full", "lora", "prompt_tuning"):
        recipe = smoke_recipe("sft", weights, base_model=HUB_BASE)
        assert check(recipe) is None


def test_a_scratch_base_passes_with_its_tokenizer_prepared(ready: Path, tmp_path: Path) -> None:
    scratch_base(tmp_path / "scratch")
    snapshot(ready, TOKENIZER_REPO)
    assert check(smoke_recipe("sft", "full", base_model="scratch:tiny-scratch")) is None


def test_a_scratch_base_without_its_tokenizer_is_refused(ready: Path, tmp_path: Path) -> None:
    scratch_base(tmp_path / "scratch")
    with pytest.raises(ValueError) as refused:
        check(smoke_recipe("sft", "full", base_model="scratch:tiny-scratch"))
    assert TOKENIZER_REPO in str(refused.value) and "tools/prepare_train.py" in str(refused.value)


@pytest.mark.parametrize(
    "pin", [{"repo": TOKENIZER_REPO}, {"repo": TOKENIZER_REPO, "revision": "main"}, {"repo": "x", "revision": REVISION},
            {"repo": TOKENIZER_REPO, "revision": REVISION, "extra": 1}, []],
    ids=["no-revision", "branch", "bad-repo", "extra-key", "not-an-object"],
)
def test_a_bad_tokenizer_pin_is_refused(pin: Any, ready: Path, tmp_path: Path) -> None:
    scratch_base(tmp_path / "scratch", pin=pin)
    snapshot(ready, TOKENIZER_REPO)
    with pytest.raises(ValueError, match="tokenizer-pin.json"):
        check(smoke_recipe("sft", "full", base_model="scratch:tiny-scratch"))


def test_a_missing_scratch_directory_is_refused(ready: Path) -> None:
    with pytest.raises(ValueError, match="base_model"):
        check(smoke_recipe("sft", "full", base_model="scratch:absent"))


@pytest.mark.parametrize("weights", ["lora", "prompt_tuning"])
def test_a_scratch_base_trains_full_weights_only(weights: str, ready: Path, tmp_path: Path) -> None:
    scratch_base(tmp_path / "scratch")
    snapshot(ready, TOKENIZER_REPO)
    with pytest.raises(ValueError) as refused:
        check(smoke_recipe("sft", weights, base_model="scratch:tiny-scratch"))
    assert "scratch" in str(refused.value) and weights in str(refused.value)


REFUSED = {
    "prompt-tuning-without-virtual-tokens": (
        lambda: with_trainer(smoke_recipe("sft", "prompt_tuning"), virtual_tokens=MISSING), "trainer.virtual_tokens"),
    "virtual-tokens-with-full-weights": (
        lambda: with_trainer(smoke_recipe("sft", "full"), virtual_tokens=4), "trainer.virtual_tokens"),
    "virtual-tokens-with-lora-weights": (
        lambda: with_trainer(smoke_recipe("sft", "lora"), virtual_tokens=4), "trainer.virtual_tokens"),
    "prompt-tuning-with-dpo": (lambda: smoke_recipe("dpo", "prompt_tuning"), "prompt_tuning"),
    "prompt-tuning-with-lora-section": (
        lambda: smoke_recipe("sft", "prompt_tuning", lora={"r": 8, "targets": "all-linear"}), "lora"),
    "max-length-with-dpo": (lambda: with_trainer(smoke_recipe("dpo", "full"), max_length=128), "trainer.max_length"),
    "max-length-with-grpo": (lambda: with_trainer(smoke_recipe("grpo", "lora"), max_length=128), "trainer.max_length"),
}


@pytest.mark.parametrize("name", list(REFUSED))
def test_check_refuses_each_mode_value_it_does_not_carry_out(name: str, buildable: type) -> None:
    recipe, needle = REFUSED[name]
    built = recipe()
    trainer = trl_trainer().TRLTrainer(**trainer_section(built))
    data = train_data.load_synthetic(built["data"]["synthetic"], SYNTHETIC_DIR)
    with pytest.raises(ValueError) as refused:
        trainer.check(built, data)
    assert needle in str(refused.value), str(refused.value)


@pytest.mark.parametrize(
    ("key", "value"),
    [("max_length", 0), ("max_length", True), ("max_length", "4096"), ("virtual_tokens", 0),
     ("gradient_checkpointing", 1), ("gradient_checkpointing", "false")],
)
def test_building_refuses_a_bad_sizing_value_naming_it(key: str, value: Any, buildable: type) -> None:
    with pytest.raises(ValueError, match=f"trainer.{key}"):
        trl_trainer().TRLTrainer(**trainer_section(with_trainer(smoke_recipe("sft", "full"), **{key: value})))


def test_the_sizing_keys_pass_check_for_sft(buildable: type) -> None:
    recipe = with_trainer(smoke_recipe("sft", "full"), max_length=256, gradient_checkpointing=False)
    assert check(recipe) is None


def test_corpus_records_are_checked_and_refusals_name_the_corpus(buildable: type) -> None:
    good = TrainData(source="corpus", split_hash="0" * 64, records=({"prompt": "p", "completion": "c"},),
                     suite=None, items=())
    recipe = smoke_recipe("sft", "full", data={"corpus": "synthetic-corpus"})
    trainer = trl_trainer().TRLTrainer(**trainer_section(recipe))
    assert trainer.check(recipe, good) is None
    bad = TrainData(source="corpus", split_hash="0" * 64, records=({"prompt": "p"},), suite=None, items=())
    with pytest.raises(ValueError, match="data.corpus record 1"):
        trainer.check(recipe, bad)


# ---------------------------------------------------------------------------
# The framework seams, with fake modules


class FakeAuto:
    """A fake transformers Auto class: each call is recorded and returns `result`."""

    def __init__(self, calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]], label: str, result: Any) -> None:
        """Keep the shared call list, this class's label, and what every call returns."""
        self.calls, self.label, self.result = calls, label, result

    def from_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        """Record a from_pretrained call and return the result."""
        self.calls.append((f"{self.label}.from_pretrained", args, kwargs))
        return self.result

    def from_config(self, *args: Any, **kwargs: Any) -> Any:
        """Record a from_config call and return the result."""
        self.calls.append((f"{self.label}.from_config", args, kwargs))
        return self.result


class FakeTokenizer:
    """A stand-in tokenizer of 258 entries."""

    def __len__(self) -> int:
        """Return the stand-in vocabulary size."""
        return 258


def fake_transformers(calls: list[Any]) -> Any:
    """Return a fake transformers module whose Auto classes and set_seed record their calls."""
    return SimpleNamespace(
        AutoConfig=FakeAuto(calls, "AutoConfig", SimpleNamespace(vocab_size=258)),
        AutoModelForCausalLM=FakeAuto(calls, "AutoModelForCausalLM", SimpleNamespace(name="model")),
        AutoTokenizer=FakeAuto(calls, "AutoTokenizer", FakeTokenizer()),
        set_seed=lambda seed: calls.append(("set_seed", (seed,), {})),
    )


def test_a_hub_base_loads_offline_from_the_cache_at_its_revision(tmp_path: Path) -> None:
    calls: list[Any] = []
    torch = SimpleNamespace(float32="float32")
    bases().load_base(fake_transformers(calls), torch, bases().parse_base(HUB_BASE), 7, str(tmp_path))
    loads = [call for call in calls if call[0].endswith("from_pretrained")]
    assert {call[0] for call in loads} == {"AutoTokenizer.from_pretrained", "AutoModelForCausalLM.from_pretrained"}
    for _, args, kwargs in loads:
        assert args == (HUB_REPO,)
        assert kwargs["revision"] == REVISION and kwargs["cache_dir"] == str(tmp_path / "hub")
        assert kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False
    model_call = next(call for call in loads if call[0].startswith("AutoModel"))
    assert model_call[2]["dtype"] == "float32", "a Hub base trains from float32 weights under bf16 mixed precision"
    assert calls.index(("set_seed", (7,), {})) < calls.index(model_call)


def test_a_scratch_base_builds_from_its_config_with_the_pinned_tokenizer(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bases(), "SCRATCH_DIR", tmp_path / "scratch")
    scratch_base(tmp_path / "scratch")
    snapshot(tmp_path, TOKENIZER_REPO)
    calls: list[Any] = []
    base = bases().check_base(bases().parse_base("scratch:tiny-scratch"), str(tmp_path))
    bases().load_base(fake_transformers(calls), SimpleNamespace(float32="float32"), base, 9, str(tmp_path))
    names = [call[0] for call in calls]
    assert "AutoModelForCausalLM.from_config" in names and "AutoModelForCausalLM.from_pretrained" not in names
    assert names.index("set_seed") < names.index("AutoModelForCausalLM.from_config")
    tokenizer = next(call for call in calls if call[0] == "AutoTokenizer.from_pretrained")
    assert tokenizer[1] == (TOKENIZER_REPO,) and tokenizer[2]["revision"] == REVISION
    assert tokenizer[2]["local_files_only"] is True and tokenizer[2]["cache_dir"] == str(tmp_path / "hub")


def test_prompt_tuning_builds_a_prompt_tuning_config() -> None:
    made: list[dict[str, Any]] = []
    peft = SimpleNamespace(PromptTuningConfig=lambda **kwargs: made.append(kwargs) or kwargs,
                           LoraConfig=lambda **kwargs: pytest.fail("prompt tuning built a LoraConfig"))
    recipe = smoke_recipe("sft", "prompt_tuning")
    found = trl_trainer()._peft_config(peft, recipe, recipe["trainer"]["virtual_tokens"])
    assert made == [{"num_virtual_tokens": recipe["trainer"]["virtual_tokens"], "task_type": "CAUSAL_LM",
                     "prompt_tuning_init": "RANDOM"}]
    assert found == made[0]


def test_the_sft_arguments_carry_the_sizing_keys(buildable: type, tmp_path: Path) -> None:
    trl = SimpleNamespace(SFTConfig=lambda **kwargs: kwargs)
    recipe = with_trainer(smoke_recipe("sft", "full"), max_length=512, gradient_checkpointing=False)
    trainer = trl_trainer().TRLTrainer(**trainer_section(recipe))
    args = trl_trainer()._arguments(trl, trainer, recipe, tmp_path)
    assert (args["max_length"], args["gradient_checkpointing"]) == (512, False)
    plain = trl_trainer().TRLTrainer(**trainer_section(smoke_recipe("sft", "full")))
    args = trl_trainer()._arguments(trl, plain, smoke_recipe("sft", "full"), tmp_path)
    assert "max_length" not in args and "gradient_checkpointing" not in args, "unset keys stay at TRL's defaults"


class CountingTokenizer:
    """A stand-in tokenizer: one id per character."""

    def __call__(self, text: str) -> dict[str, list[int]]:
        """Return one id per character of `text`."""
        return {"input_ids": list(range(len(text)))}


def test_check_lengths_refuses_a_record_trl_would_cut() -> None:
    records = ({"prompt": "ab", "completion": "cd"}, {"prompt": "abc", "completion": "defgh"})
    check_lengths = trl_trainer().check_lengths
    assert check_lengths(CountingTokenizer(), records, 9, "data.synthetic") is None
    with pytest.raises(ValueError) as refused:
        check_lengths(CountingTokenizer(), records, 8, "data.synthetic")
    message = str(refused.value)
    assert "record 2" in message and "9" in message and "8" in message and "max_length" in message, message
    assert check_lengths(CountingTokenizer(), records, None, "data.synthetic") is None


# ---------------------------------------------------------------------------
# trainer.chat_format: train in the format hf_local serves


def test_chat_format_is_an_sft_only_setting(buildable: type) -> None:
    assert "chat_format" in trl_trainer().TRLTrainer.config_keys
    assert check(with_trainer(smoke_recipe("sft", "full"), chat_format=True)) is None
    recipe = with_trainer(smoke_recipe("dpo", "full"), chat_format=True)
    trainer = trl_trainer().TRLTrainer(**trainer_section(recipe))
    with pytest.raises(ValueError, match="trainer.chat_format"):
        trainer.check(recipe, train_data.load_synthetic(recipe["data"]["synthetic"], SYNTHETIC_DIR))
    with pytest.raises(ValueError, match="trainer.chat_format"):
        trl_trainer().TRLTrainer(**trainer_section(with_trainer(smoke_recipe("sft", "full"), chat_format="yes")))


def test_chat_records_are_one_user_turn_and_one_assistant_turn() -> None:
    record = trl_trainer().chat_record({"prompt": "SYNTHETIC ask", "completion": "SYNTHETIC answer"})
    assert record == {"prompt": [{"role": "user", "content": "SYNTHETIC ask"}],
                      "completion": [{"role": "assistant", "content": "SYNTHETIC answer"}]}


class ChatTokenizer(CountingTokenizer):
    """A stand-in tokenizer with a chat template: one id per character of the rendered turns."""

    chat_template = "SYNTHETIC"

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, list[int]]:
        """Return one id per character of `<role>:<content>;` for each message."""
        assert kwargs == {"tokenize": True, "return_dict": True}, kwargs
        text = "".join(f"{message['role']}:{message['content']};" for message in messages)
        return {"input_ids": list(range(len(text)))}


def test_check_lengths_counts_the_rendered_chat_turns() -> None:
    record = trl_trainer().chat_record({"prompt": "ab", "completion": "cd"})
    # "user:ab;" is 8 characters and "assistant:cd;" 13, plus the end token: 22.
    assert trl_trainer().check_lengths(ChatTokenizer(), [record], 22, "data.corpus") is None
    with pytest.raises(ValueError, match="record 1 is 22 tokens"):
        trl_trainer().check_lengths(ChatTokenizer(), [record], 21, "data.corpus")


def test_a_hub_snapshots_links_are_followed_when_recorded(tmp_path: Path) -> None:
    directory = snapshot(tmp_path, HUB_REPO)
    blob = tmp_path / "hub" / "models--SYNTHETIC-org--tiny-model" / "blobs" / "000"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"SYNTHETIC weights\n")
    try:
        (directory / "model.safetensors").symlink_to(blob)
    except OSError:
        pytest.skip("this host does not let the test create a symbolic link")
    found = bases().snapshot_files(directory)
    assert found["model.safetensors"] == hashlib.sha256(b"SYNTHETIC weights\n").hexdigest()
    assert sorted(found) == ["config.json", "model.safetensors", "tokenizer.json"]
    record = bases().base_record(bases().parse_base(HUB_BASE), str(tmp_path), 7)
    assert record["files"] == found
    assert (record["repo"], record["revision"], record["kind"]) == (HUB_REPO, REVISION, "hub")
