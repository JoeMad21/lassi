"""Tests for the trl Trainer's declarations, settings, and refusals, without the framework (task P17.9).

Bible: Training Module (Algorithms, Weight Modes, Compute, Safeguards),
Project Recipes (train.yaml and the Notes, Train recipes), Component
Interfaces (Trainer), Model Serving (Serving Rules: the hub library is set
offline before any load), Agent Rules 1, 5, 6, 7, and 10; plans/
p17-portable.md, task P17.9 (B3: "Every method, weight mode, and train
recipe key or value that the Training Module and Project Recipes
(train.yaml) name beyond sft, dpo, and grpo with full and lora weights is
refused as not carried out, naming it (P7, P8, P16)"); the P17.9 design,
sections 2 and 3.

The contract these tests fix (lassi.train.trl_trainer):

- TRLTrainer is registered as Trainer "trl" when lassi.train is imported,
  with no framework imported. It declares takes_device, methods (sft, dpo,
  grpo), weight_modes (full, lora), data_sources (synthetic), the packages
  it records as pins (torch, transformers, trl, peft, accelerate, datasets,
  tokenizers, huggingface-hub, safetensors), config_keys (steps, seed,
  batch_size, learning_rate, max_completion_length), a staticmethod
  framework() that reads torch's build only, check(recipe, data), and
  train(job).
- Built as factory(**config) from its trainer section, it raises
  ValueError, in this order, for: a missing or malformed setting, naming
  trainer.<key> (steps and batch_size integers of at least 1, seed an
  integer from 0 to 2**32 - 1, learning_rate a finite number above 0,
  max_completion_length, when set, an integer of at least 1; a bool is no
  number; the code picks no default); a missing device or one of kind cuda
  or rocm, naming trainer.device and P17.13 (CPU only, OQ-040); an unset or
  empty HF_HOME, naming it (Agent Rule 7); and a framework module that
  importlib.util.find_spec cannot find, naming it and `--extra cpu`.
  Building imports no framework and reads no file. framework() without
  torch raises an ImportError naming trl and the extra.
- check(recipe, data) raises ValueError naming the key, the value, or the
  record (by its 1-based number) for each value it does not carry out:
  a base_model that is not `fixture:<one plain name>` naming a directory
  under tests/fixtures/train/ with the three base files (a Hub id needs a
  revision pin, Agent Rule 10; P7); lora missing for lora weights or set
  for full weights; lora.r missing; lora.targets other than all-linear;
  rollout missing for grpo or set for sft and dpo; rollout.engine other
  than transformers (vllm is P8's); rollout.group_size missing or below 2,
  or a trainer.batch_size that is not a whole number of groups;
  trainer.max_completion_length missing for grpo or set for sft and dpo;
  and a record whose keys are not exactly the method's fields or whose
  values are not strings (grpo's target may be null). It imports no
  framework. The layer turns its ValueError into a RunError naming the
  trainer, before any directory exists.
- The loader refuses, naming the key and the value: methods rft (P7), gspo,
  ppo (P8), and adversarial (P16); weights qlora and dora (P7); and bench
  data (P7, P8). The layer refuses episode and reward (P7, P8), adversary
  (P16), and export (P7) for the trl trainer as for any.
- hf_local's hub_offline is public and is the one helper that sets the hub
  library offline; the trl trainer calls it.

These tests run without the extra: find_spec answers for the framework
modules, every framework import is blocked where the test says so, and the
layer tests replace framework() with a SYNTHETIC build and train() with a
stub. The slow runs are in tests/train/test_trl_runs.py. Every text is
SYNTHETIC; no value in this module is a measurement.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import importlib.machinery
import importlib.util
import io
import math
import pathlib
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from train_fakes import (
    LAYER_KEYS,
    MISSING,
    REPO,
    Log,
    bible_train_block,
    fake_probes,
    jsonl,
    on_bench,
    train_run,
    write_fixture,
    write_recipe,
)
from trl_smoke import (
    CONFIG_KEYS,
    DATA_SOURCES,
    EXTRA_HINT,
    FIXTURES,
    FRAMEWORK_MODULES,
    LORA,
    METHODS,
    PACKAGES,
    ROLLOUT,
    SYNTHETIC_BUILD,
    SYNTHETIC_DIR,
    TRAINER,
    WEIGHT_MODES,
    settings,
    smoke_recipe,
    trainer_section,
    trl_trainer,
    with_trainer,
)

import lassi.train  # noqa: F401  (registers every trainer)
from lassi.core.devices import FrameworkBuild
from lassi.core.interfaces import TrainResult
from lassi.core.recipe import RecipeError, load_train_recipe
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.core.runner import RunError
from lassi.train import data as train_data

TAKES_DEVICE = "takes_device"
# Every module a check or a build must not import: the framework and the libraries beneath it.
BLOCKED = (*FRAMEWORK_MODULES, "huggingface_hub", "tokenizers", "safetensors")
BAD_FIXTURE = "bad-records.jsonl"
GPU_DEVICES = {"cuda": {"kind": "cuda", "indices": [0]}, "rocm": {"kind": "rocm", "indices": [0]}}


def trl_class() -> type:
    """Return lassi.train.trl_trainer.TRLTrainer."""
    return trl_trainer().TRLTrainer


def answer_find_spec(monkeypatch: pytest.MonkeyPatch, missing: tuple[str, ...] = ()) -> None:
    """Make importlib.util.find_spec find each framework module but `missing`, as with the cpu extra installed."""
    real = importlib.util.find_spec

    def find_spec(name: str, package: str | None = None) -> Any:
        root = name.split(".")[0]
        if root in missing:
            return None
        if root in FRAMEWORK_MODULES:
            return importlib.machinery.ModuleSpec(name, None)
        return real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    module = trl_trainer()
    if hasattr(module, "find_spec"):
        monkeypatch.setattr(module, "find_spec", find_spec)


def block_frameworks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every import of a framework module raise ImportError for the rest of the test."""
    for name in BLOCKED:
        monkeypatch.setitem(sys.modules, name, None)


@pytest.fixture
def buildable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> type:
    """Return TRLTrainer with HF_HOME set, find_spec answering, and every framework import blocked."""
    cls = trl_class()
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    answer_find_spec(monkeypatch)
    block_frameworks(monkeypatch)
    return cls


def build(data: Mapping[str, Any]) -> Any:
    """Build TRLTrainer from a recipe's trainer section, as the layer does (factory(**config))."""
    return trl_class()(**trainer_section(data))


@pytest.fixture
def layer_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Log:
    """Make the trl trainer buildable without the extra, with a SYNTHETIC framework() and a train() stub.

    The stub records "train" in the returned Log and reports no steps and no
    checkpoint; the refusal tests expect it never to run.
    """
    cls = trl_class()
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    answer_find_spec(monkeypatch)
    name, version, cuda, hip = SYNTHETIC_BUILD
    monkeypatch.setattr(cls, "framework", staticmethod(lambda: FrameworkBuild(name, version, cuda, hip)))
    log = Log()

    def steps(self: Any, job: Any) -> TrainResult:
        log.events.append("train")
        return TrainResult(steps=0, checkpoints=())

    monkeypatch.setattr(cls, "train", steps)
    return log


def run_trl(tmp_path: Path, data: Mapping[str, Any], synthetic_dir: Path | None = None) -> Path:
    """Write `data` as a train recipe and run it with lassi's registry and SYNTHETIC device probes."""
    path = write_recipe(tmp_path / "recipes", data)
    options = train_run().TrainOptions(
        runs_root=tmp_path / "runs-root", train_id="t1", roots=[tmp_path / "recipes"], probes=fake_probes(Log()),
        synthetic_dir=synthetic_dir,
    )
    return train_run().run_training(path, options)


# ---------------------------------------------------------------------------
# Registration and declarations


def test_trl_is_registered_with_its_declarations() -> None:
    cls = trl_class()
    entry = DEFAULT_REGISTRY.get("Trainer", TRAINER)
    assert entry.factory is cls
    assert cls.name == TRAINER
    assert set(cls.capabilities) == {TAKES_DEVICE}
    assert tuple(cls.methods) == METHODS
    assert tuple(cls.weight_modes) == WEIGHT_MODES
    assert tuple(cls.data_sources) == DATA_SOURCES
    assert tuple(cls.packages) == PACKAGES
    assert frozenset(cls.config_keys) == CONFIG_KEYS
    assert isinstance(vars(cls)["framework"], staticmethod)
    assert callable(getattr(cls, "check", None)) and callable(getattr(cls, "train", None))
    assert DEFAULT_REGISTRY.names("Trainer") == [TRAINER]


def test_framework_without_torch_names_trl_and_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError) as refused:
        trl_class().framework()
    message = str(refused.value)
    assert "trl" in message and EXTRA_HINT in message, message
    assert "hf_local" not in message, "the trl trainer names itself, not hf_local"


def test_the_layer_refuses_trl_without_the_extra_before_any_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trl_class()
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RunError) as refused:
        run_trl(tmp_path, smoke_recipe("sft", "full"))
    message = str(refused.value)
    assert "trainer.device" in message and EXTRA_HINT in message, message
    assert not (tmp_path / "runs-root").exists()


# ---------------------------------------------------------------------------
# Building: settings, device, HF_HOME, then the framework


def test_the_smoke_settings_build(buildable: type) -> None:
    for method in METHODS:
        built = build(smoke_recipe(method, "full"))
        assert isinstance(built, buildable)


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_the_seed_range_ends_are_accepted(seed: int, buildable: type) -> None:
    assert isinstance(build(with_trainer(smoke_recipe("sft", "full"), seed=seed)), buildable)


def test_an_integer_learning_rate_is_accepted(buildable: type) -> None:
    assert isinstance(build(with_trainer(smoke_recipe("sft", "full"), learning_rate=1)), buildable)


BAD_SETTINGS = [
    ("steps", 0), ("steps", -1), ("steps", True), ("steps", "2"), ("steps", 2.0),
    ("seed", -1), ("seed", 2**32), ("seed", True), ("seed", "7"), ("seed", 7.0),
    ("batch_size", 0), ("batch_size", True), ("batch_size", "2"), ("batch_size", 2.0),
    ("learning_rate", 0), ("learning_rate", -0.001), ("learning_rate", math.nan), ("learning_rate", math.inf),
    ("learning_rate", True), ("learning_rate", "0.001"),
    ("max_completion_length", 0), ("max_completion_length", True), ("max_completion_length", "8"),
    ("steps", MISSING), ("seed", MISSING), ("batch_size", MISSING), ("learning_rate", MISSING),
]


@pytest.mark.parametrize(
    ("key", "value"), BAD_SETTINGS, ids=[f"{key}-{'missing' if v is MISSING else v!r}" for key, v in BAD_SETTINGS]
)
def test_building_refuses_a_bad_setting_naming_it(key: str, value: Any, buildable: type) -> None:
    with pytest.raises(ValueError) as refused:
        build(with_trainer(smoke_recipe("grpo", "full"), **{key: value}))
    assert f"trainer.{key}" in str(refused.value), str(refused.value)


@pytest.mark.parametrize("kind", list(GPU_DEVICES))
def test_building_refuses_a_gpu_device_naming_trainer_device(kind: str, buildable: type) -> None:
    with pytest.raises(ValueError) as refused:
        build(with_trainer(smoke_recipe("sft", "full"), device=GPU_DEVICES[kind]))
    message = str(refused.value)
    assert "trainer.device" in message and "P17.13" in message, message


def test_building_refuses_a_missing_device(buildable: type) -> None:
    with pytest.raises(ValueError, match="trainer.device"):
        build(with_trainer(smoke_recipe("sft", "full"), device=MISSING))


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_building_refuses_an_unset_hf_home(value: str | None, buildable: type, monkeypatch: pytest.MonkeyPatch) -> None:
    if value is None:
        monkeypatch.delenv("HF_HOME", raising=False)
    else:
        monkeypatch.setenv("HF_HOME", value)
    with pytest.raises(ValueError, match="HF_HOME"):
        build(smoke_recipe("sft", "full"))


@pytest.mark.parametrize("missing", FRAMEWORK_MODULES)
def test_building_without_the_extra_names_it(missing: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trl_class()
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    answer_find_spec(monkeypatch, missing=(missing,))
    with pytest.raises(ValueError) as refused:
        build(smoke_recipe("sft", "full"))
    message = str(refused.value)
    assert missing in message and EXTRA_HINT in message, message


def test_settings_and_the_device_are_checked_before_hf_home_and_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trl_class()
    monkeypatch.delenv("HF_HOME", raising=False)
    answer_find_spec(monkeypatch, missing=FRAMEWORK_MODULES)
    with pytest.raises(ValueError, match="trainer.steps"):
        build(with_trainer(smoke_recipe("sft", "full"), steps=0))
    with pytest.raises(ValueError, match="trainer.device"):
        build(with_trainer(smoke_recipe("sft", "full"), device=GPU_DEVICES["rocm"]))
    with pytest.raises(ValueError, match="HF_HOME"):
        build(smoke_recipe("sft", "full"))


def test_building_imports_nothing_and_reads_no_file(buildable: type, monkeypatch: pytest.MonkeyPatch) -> None:
    # buildable blocks every framework import, so an import while building fails here too.
    reads: list[str] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        reads.append(repr(args))
        raise AssertionError(f"building the trl trainer read a file: {args!r}")

    with monkeypatch.context() as patch:
        for owner, name in ((builtins, "open"), (io, "open"), (pathlib.Path, "open"), (pathlib.Path, "read_bytes"),
                            (pathlib.Path, "read_text")):
            patch.setattr(owner, name, refuse)
        build(smoke_recipe("grpo", "lora"))
    assert reads == []


def test_the_layer_refuses_a_bad_setting_before_any_directory(tmp_path: Path, layer_ready: Log) -> None:
    with pytest.raises(RunError) as refused:
        run_trl(tmp_path, with_trainer(smoke_recipe("sft", "full"), learning_rate=0))
    message = str(refused.value)
    assert "trainer.learning_rate" in message and TRAINER in message, message
    assert not (tmp_path / "runs-root").exists()
    assert layer_ready.events == []


# ---------------------------------------------------------------------------
# check(recipe, data): what the trl trainer does not carry out (B3)


@dataclass(frozen=True)
class NotCarried:
    """One recipe value or record check() refuses: the recipe, the records, what the message names, and the owner.

    `owner` is the phase that carries the value out later ("-" when none
    does); `line` is the 1-based record number the message must name.
    """

    recipe: Callable[[], dict[str, Any]]
    needles: tuple[str, ...]
    owner: str = "-"
    records: tuple[dict[str, Any], ...] | None = None
    line: int | None = None


def _bad(method: str) -> dict[str, Any]:
    """Return the method's full-weights smoke recipe reading BAD_FIXTURE."""
    return smoke_recipe(method, "full", data={"synthetic": BAD_FIXTURE})


SFT_OK = {"prompt": "SYNTHETIC prompt", "completion": " abc"}
DPO_OK = {"prompt": "SYNTHETIC prompt", "chosen": " abc", "rejected": " xyz"}
GRPO_OK = {"prompt": "SYNTHETIC prompt", "target": "abc"}

NOT_CARRIED: dict[str, NotCarried] = {
    # The base model: a fixture built from its config; a Hub base needs a revision pin first (P7).
    "base-model-hub-id": NotCarried(lambda: smoke_recipe("sft", "full", base_model="SYNTHETIC-org/tiny-model"),
                                    ("base_model", "Agent Rule 10"), owner="P7"),
    "base-model-unknown-fixture": NotCarried(lambda: smoke_recipe("sft", "full", base_model="fixture:no-such-model"),
                                             ("base_model", "no-such-model")),
    "base-model-leaves-the-fixtures": NotCarried(
        lambda: smoke_recipe("sft", "full", base_model="fixture:../train"), ("base_model",)),
    "base-model-nested-name": NotCarried(
        lambda: smoke_recipe("sft", "full", base_model="fixture:tiny-causal-lm/sub"), ("base_model",)),
    "base-model-empty-name": NotCarried(lambda: smoke_recipe("sft", "full", base_model="fixture:"), ("base_model",)),
    "base-model-a-fixture-file": NotCarried(
        lambda: smoke_recipe("sft", "full", base_model="fixture:sft-smoke.jsonl"), ("base_model",)),
    "base-model-a-path": NotCarried(lambda: smoke_recipe("sft", "full", base_model="./tiny-causal-lm"),
                                    ("base_model",)),
    # lora: PEFT LoraConfig(r, all-linear) for lora weights only.
    "lora-missing": NotCarried(lambda: smoke_recipe("sft", "lora", lora=MISSING), ("lora",)),
    "lora-with-full-weights": NotCarried(lambda: smoke_recipe("dpo", "full", lora=LORA), ("lora",)),
    "lora-r-missing": NotCarried(lambda: smoke_recipe("sft", "lora", lora={"targets": "all-linear"}), ("lora.r",)),
    "lora-targets-other": NotCarried(lambda: smoke_recipe("sft", "lora", lora={"r": 8, "targets": "q_proj"}),
                                     ("lora.targets", "q_proj"), owner="P7"),
    "lora-targets-missing": NotCarried(lambda: smoke_recipe("dpo", "lora", lora={"r": 8}), ("lora.targets",)),
    # rollout: grpo's in-process generation through TRL; vLLM rollouts are P8's.
    "rollout-missing-for-grpo": NotCarried(lambda: smoke_recipe("grpo", "full", rollout=MISSING), ("rollout",)),
    "rollout-set-for-sft": NotCarried(lambda: smoke_recipe("sft", "full", rollout=ROLLOUT), ("rollout",)),
    "rollout-set-for-dpo": NotCarried(lambda: smoke_recipe("dpo", "lora", rollout=ROLLOUT), ("rollout",)),
    "rollout-engine-vllm": NotCarried(lambda: smoke_recipe("grpo", "full", rollout={**ROLLOUT, "engine": "vllm"}),
                                      ("rollout.engine", "vllm", "P8"), owner="P8"),
    "rollout-engine-missing": NotCarried(lambda: smoke_recipe("grpo", "full", rollout={"group_size": 2}),
                                         ("rollout.engine",)),
    "rollout-group-size-one": NotCarried(lambda: smoke_recipe("grpo", "full", rollout={**ROLLOUT, "group_size": 1}),
                                         ("rollout.group_size",)),
    "rollout-group-size-missing": NotCarried(
        lambda: smoke_recipe("grpo", "full", rollout={"engine": "transformers"}), ("rollout.group_size",)),
    "batch-not-whole-groups": NotCarried(
        lambda: smoke_recipe("grpo", "full", rollout={**ROLLOUT, "group_size": 3}),
        ("rollout.group_size", "trainer.batch_size")),
    # trainer.max_completion_length: grpo's generation length, ignored by sft and dpo.
    "length-missing-for-grpo": NotCarried(
        lambda: with_trainer(smoke_recipe("grpo", "lora"), max_completion_length=MISSING),
        ("trainer.max_completion_length",)),
    "length-set-for-sft": NotCarried(lambda: with_trainer(smoke_recipe("sft", "full"), max_completion_length=8),
                                     ("trainer.max_completion_length",)),
    "length-set-for-dpo": NotCarried(lambda: with_trainer(smoke_recipe("dpo", "full"), max_completion_length=8),
                                     ("trainer.max_completion_length",)),
    # Records: exactly the method's fields, string values (grpo's target may be null).
    "sft-record-missing-completion": NotCarried(lambda: _bad("sft"), ("data.synthetic",),
                                                records=(SFT_OK, {"prompt": "SYNTHETIC prompt"}), line=2),
    "sft-record-extra-key": NotCarried(lambda: _bad("sft"), ("data.synthetic",),
                                       records=(SFT_OK, {**SFT_OK, "label": "SYNTHETIC"}), line=2),
    "sft-record-null-completion": NotCarried(lambda: _bad("sft"), ("data.synthetic",),
                                             records=(SFT_OK, SFT_OK, {**SFT_OK, "completion": None}), line=3),
    "dpo-record-missing-rejected": NotCarried(lambda: _bad("dpo"), ("data.synthetic",),
                                              records=(DPO_OK, {"prompt": "SYNTHETIC", "chosen": " abc"}), line=2),
    "dpo-on-sft-records": NotCarried(lambda: _bad("dpo"), ("data.synthetic",), records=(SFT_OK,), line=1),
    "grpo-record-number-prompt": NotCarried(lambda: _bad("grpo"), ("data.synthetic",),
                                            records=(GRPO_OK, {"prompt": 3, "target": "abc"}), line=2),
    "grpo-record-null-prompt": NotCarried(lambda: _bad("grpo"), ("data.synthetic",),
                                          records=(GRPO_OK, {"prompt": None, "target": "abc"}), line=2),
    "grpo-record-number-target": NotCarried(lambda: _bad("grpo"), ("data.synthetic",),
                                            records=(GRPO_OK, {"prompt": "SYNTHETIC", "target": 5}), line=2),
}
CASE_IDS = [f"{name}-{case.owner}" if case.owner != "-" else name for name, case in NOT_CARRIED.items()]


def case_data(case: NotCarried, tmp_path: Path) -> tuple[dict[str, Any], Any, Path]:
    """Return the case's recipe, its TrainData, and the fixture directory the data was read from."""
    recipe = case.recipe()
    if case.records is None:
        return recipe, train_data.load_synthetic(recipe["data"]["synthetic"], SYNTHETIC_DIR), SYNTHETIC_DIR
    directory = tmp_path / "fixtures"
    write_fixture(directory, jsonl(case.records), name=BAD_FIXTURE)
    return recipe, train_data.load_synthetic(BAD_FIXTURE, directory), directory


def assert_names(case: NotCarried, message: str) -> None:
    """Fail unless `message` names every needle of the case and, for a record, its number."""
    missing = [needle for needle in case.needles if needle not in message]
    assert not missing, f"the message does not name {missing}: {message}"
    if case.line is not None:
        assert re.search(rf"\b(record|line) {case.line}\b", message), f"no record number {case.line}: {message}"


@pytest.mark.parametrize("case", list(NOT_CARRIED.values()), ids=CASE_IDS)
def test_check_refuses_each_value_not_carried_out(case: NotCarried, tmp_path: Path, buildable: type) -> None:
    recipe, data, _ = case_data(case, tmp_path)
    trainer = build(recipe)
    with pytest.raises(ValueError) as refused:
        trainer.check(recipe, data)
    assert_names(case, str(refused.value))


@pytest.mark.parametrize("case", list(NOT_CARRIED.values()), ids=CASE_IDS)
def test_the_layer_refuses_each_value_before_any_directory(
    case: NotCarried, tmp_path: Path, layer_ready: Log
) -> None:
    recipe, _, directory = case_data(case, tmp_path)
    with pytest.raises(RunError) as refused:
        run_trl(tmp_path, recipe, synthetic_dir=directory)
    message = str(refused.value)
    assert_names(case, message)
    assert TRAINER in message, message
    assert not (tmp_path / "runs-root").exists(), "check() runs before any directory exists"
    assert layer_ready.events == [], "no step ran"


SMOKE = [(method, weights) for method in METHODS for weights in WEIGHT_MODES]


@pytest.mark.parametrize(("method", "weights"), SMOKE, ids=[f"{m}-{w}" for m, w in SMOKE])
def test_check_accepts_the_smoke_recipes(method: str, weights: str, buildable: type) -> None:
    recipe = smoke_recipe(method, weights)
    data = train_data.load_synthetic(FIXTURES[method], SYNTHETIC_DIR)
    assert build(recipe).check(recipe, data) is None


def test_check_accepts_the_layer_fixture_for_sft(buildable: type) -> None:
    recipe = smoke_recipe("sft", "full", data={"synthetic": "layer-smoke.jsonl"})
    data = train_data.load_synthetic("layer-smoke.jsonl", SYNTHETIC_DIR)
    assert build(recipe).check(recipe, data) is None


def test_the_layer_runs_a_checked_smoke_recipe_to_the_steps(tmp_path: Path, layer_ready: Log) -> None:
    train_dir = run_trl(tmp_path, smoke_recipe("grpo", "lora"))
    assert layer_ready.events == ["train"], "check() passed and train() ran once"
    assert (train_dir / "provenance.json").is_file()


# ---------------------------------------------------------------------------
# Refused at load or by the layer: the methods, weight modes, sources, and keys of later phases


AT_LOAD = {
    "method-rft-P7": (lambda: {**smoke_recipe("sft", "full"), "method": "rft"}, ("method", "rft")),
    "method-gspo-P8": (lambda: {**smoke_recipe("grpo", "full"), "method": "gspo"}, ("method", "gspo")),
    "method-ppo-P8": (lambda: {**smoke_recipe("grpo", "full"), "method": "ppo"}, ("method", "ppo")),
    "method-adversarial-P16": (lambda: {**smoke_recipe("sft", "full"), "method": "adversarial"},
                               ("method", "adversarial")),
    "weights-qlora-P7": (lambda: {**smoke_recipe("sft", "lora"), "weights": "qlora"}, ("weights", "qlora")),
    "weights-dora-P7": (lambda: {**smoke_recipe("sft", "lora"), "weights": "dora"}, ("weights", "dora")),
    "bench-P7-P8": (lambda: {**on_bench(), "base_model": "fixture:tiny-causal-lm",
                             "trainer": smoke_recipe("sft", "full")["trainer"]}, ("bench",)),
}


@pytest.mark.parametrize("name", list(AT_LOAD))
def test_load_refuses_each_method_weight_mode_and_source_beyond_the_trainer(name: str, tmp_path: Path) -> None:
    trl_class()
    recipe, needles = AT_LOAD[name]
    path = write_recipe(tmp_path / "recipes", recipe())
    with pytest.raises(RecipeError) as refused:
        load_train_recipe(path, roots=[tmp_path / "recipes"])
    message = str(refused.value)
    assert all(needle in message for needle in needles) and TRAINER in message, message


LATER = {**LAYER_KEYS, "episode-multi_turn": "multi_turn"}
LATER_OWNERS = {
    "episode": "P7-P8", "reward": "P7-P8", "adversary": "P16", "export": "P7", "episode-multi_turn": "P7-P8",
}


@pytest.mark.parametrize("name", list(LATER), ids=[f"{name}-{LATER_OWNERS[name]}" for name in LATER])
def test_the_layer_refuses_each_key_of_a_later_phase(name: str, tmp_path: Path) -> None:
    trl_class()
    key = name.split("-")[0]
    with pytest.raises(RunError) as refused:
        run_trl(tmp_path, smoke_recipe("grpo", "lora", **{key: LATER[name]}))
    assert key in str(refused.value), str(refused.value)
    assert not (tmp_path / "runs-root").exists()


def test_the_bible_train_block_is_refused_with_the_trl_trainer(tmp_path: Path) -> None:
    trl_class()
    path = write_recipe(tmp_path / "recipes", bible_train_block())
    with pytest.raises(RecipeError) as refused:
        load_train_recipe(path, roots=[tmp_path / "recipes"])
    message = str(refused.value)
    assert "method" in message and "gspo" in message and TRAINER in message, message


# ---------------------------------------------------------------------------
# One hub-offline helper (Serving Rules; PHASE-NOTES P17)


def test_hf_local_shares_one_public_hub_offline_helper() -> None:
    hf_local = importlib.import_module("lassi.llm.hf_local")
    helper = getattr(hf_local, "hub_offline", None)
    assert callable(helper), "lassi.llm.hf_local has no public hub_offline yet (task P17.9)"
    assert (helper.__doc__ or "").strip()
    assert not hasattr(hf_local, "_hub_offline"), "one helper: the private name is gone"


def offline_setters(path: Path) -> list[str]:
    """Return the functions of a module that set the hub library's offline flag (assignment, setattr, or env)."""
    tree = ast.parse(path.read_bytes().decode("ascii"))
    found: list[str] = []

    def names_flag(node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute):
            return node.attr == "HF_HUB_OFFLINE"
        if isinstance(node, ast.Subscript):
            return isinstance(node.slice, ast.Constant) and node.slice.value == "HF_HUB_OFFLINE"
        return False

    def visit(node: ast.AST, where: str) -> None:
        for child in ast.iter_child_nodes(node):
            here = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else where
            targets = child.targets if isinstance(child, ast.Assign) else []
            if any(names_flag(target) for target in targets):
                found.append(here)
            if isinstance(child, ast.Call) and any(
                isinstance(arg, ast.Constant) and arg.value == "HF_HUB_OFFLINE" for arg in child.args
            ):
                found.append(here)
            visit(child, here)

    visit(tree, "<module>")
    return found


def test_only_hub_offline_sets_the_hub_offline_flag() -> None:
    found = {
        path.relative_to(REPO).as_posix(): setters
        for path in sorted((REPO / "lassi").rglob("*.py"))
        if (setters := offline_setters(path))
    }
    assert found == {"lassi/llm/hf_local.py": ["hub_offline"]}, found


def test_the_trl_trainer_calls_the_shared_helper() -> None:
    origin = trl_trainer().__file__
    tree = ast.parse(Path(origin).read_bytes().decode("ascii"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "id", None) == "hub_offline" or getattr(node.func, "attr", None) == "hub_offline")
    ]
    assert calls, "lassi.train.trl_trainer never calls hf_local's hub_offline before a load"


def test_the_smoke_settings_cover_every_config_key() -> None:
    keys: set[str] = set()
    for method in METHODS:
        keys.update(settings(method))
    assert keys == set(CONFIG_KEYS)
