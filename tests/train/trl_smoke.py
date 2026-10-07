"""The smoke recipes, fixtures, and pins the trl trainer tests share (task P17.9).

The trl trainer (lassi.train.trl_trainer, registered as Trainer "trl") runs
sft and dpo with full and lora weights, and grpo with in-process generation
and the fixture reward, on the CPU, from a tiny model built from the config
under tests/fixtures/train/tiny-causal-lm/ (base_model
`fixture:tiny-causal-lm`) on the SYNTHETIC smoke fixtures beside it. This
module holds what the fast tests (no framework) and the slow runs (the cpu
extra) both name: the recipes, the record fields per method, the trainer's
settings and declarations, and the pinned versions. It imports no
framework.

Every text is SYNTHETIC and no value here is a measurement.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from train_fakes import MISSING, REPO

SYNTHETIC_DIR = REPO / "tests" / "fixtures" / "train"
TRAINER = "trl"
BASE_NAME = "tiny-causal-lm"
BASE_MODEL = f"fixture:{BASE_NAME}"
BASE_DIR = SYNTHETIC_DIR / BASE_NAME
BASE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")

METHODS = ("sft", "dpo", "grpo")
WEIGHT_MODES = ("full", "lora")
DATA_SOURCES = ("synthetic",)
# The distributions the trainer records as its framework pins, in its declared order.
PACKAGES = (
    "torch", "transformers", "trl", "peft", "accelerate", "datasets", "tokenizers", "huggingface-hub", "safetensors",
)
CONFIG_KEYS = frozenset({"steps", "seed", "batch_size", "learning_rate", "max_completion_length"})
# The framework modules the trainer needs; building it without one is refused, naming it and the extra.
FRAMEWORK_MODULES = ("torch", "transformers", "trl", "peft", "accelerate", "datasets")
EXTRA_HINT = "--extra cpu"
# The pinned versions the cpu, cuda, and rocm extras hold (bible Toolchain Pins, framework pins).
PINS = {"torch": "2.14.1", "transformers": "5.18.0", "trl": "1.14.1", "peft": "0.21.2", "accelerate": "1.15.0"}

# The smoke fixtures and each method's record fields (TRL's standard, non-conversational formats).
FIXTURES = {"sft": "sft-smoke.jsonl", "dpo": "dpo-smoke.jsonl", "grpo": "grpo-smoke.jsonl"}
FIELDS = {"sft": ("prompt", "completion"), "dpo": ("prompt", "chosen", "rejected"), "grpo": ("prompt", "target")}
TARGET = "abcdefghijklmnopqrstuvwxyz"

# The smoke settings: two steps, a fixed seed, small batches; grpo takes 4 completions per step in groups of 2.
STEPS = 2
SEED = 7
LEARNING_RATE = 0.001
BATCH_SIZE = {"sft": 2, "dpo": 2, "grpo": 4}
GROUP_SIZE = 2
MAX_COMPLETION_LENGTH = 8
LORA = {"r": 8, "targets": "all-linear"}
ROLLOUT = {"engine": "transformers", "group_size": GROUP_SIZE}

# SYNTHETIC: a framework build in the shape torch's reading gives, for the tests that run without the extra.
SYNTHETIC_BUILD = ("torch", "2.99.0+synthetic-cpu", None, None)


def train_module(name: str) -> ModuleType:
    """Import lassi.train.<name>, failing the test clearly while it does not exist (task P17.9)."""
    try:
        return importlib.import_module(f"lassi.train.{name}")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.train.{name} does not exist yet (task P17.9): {error}")


def trl_trainer() -> ModuleType:
    """Import lassi.train.trl_trainer."""
    return train_module("trl_trainer")


def settings(method: str) -> dict[str, Any]:
    """Return the smoke settings for `method`: steps, seed, batch_size, learning_rate, and grpo's length."""
    found: dict[str, Any] = {
        "steps": STEPS, "seed": SEED, "batch_size": BATCH_SIZE[method], "learning_rate": LEARNING_RATE,
    }
    if method == "grpo":
        found["max_completion_length"] = MAX_COMPLETION_LENGTH
    return found


def smoke_recipe(method: str, weights: str, **changes: Any) -> dict[str, Any]:
    """Return the smoke train recipe for `method` and `weights`; `changes` replace top-level keys, MISSING drops one.

    lora is set exactly for lora weights and rollout exactly for grpo, the
    device is the CPU, and the data is the method's committed smoke fixture.
    """
    data: dict[str, Any] = {
        "base_model": BASE_MODEL,
        "method": method,
        "weights": weights,
        "trainer": {"kind": TRAINER, "device": {"kind": "cpu"}, **settings(method)},
        "data": {"synthetic": FIXTURES[method]},
    }
    if weights == "lora":
        data["lora"] = dict(LORA)
    if method == "grpo":
        data["rollout"] = dict(ROLLOUT)
    for key, value in changes.items():
        if value is MISSING:
            data.pop(key, None)
        else:
            data[key] = copy.deepcopy(value)
    return data


def with_trainer(data: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    """Return a copy of a train recipe with trainer keys replaced by `changes`; MISSING drops a key."""
    found = copy.deepcopy(dict(data))
    for key, value in changes.items():
        if value is MISSING:
            found["trainer"].pop(key, None)
        else:
            found["trainer"][key] = copy.deepcopy(value)
    return found


def trainer_section(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recipe's trainer section without kind: what the layer builds the Trainer with."""
    return {key: copy.deepcopy(value) for key, value in data["trainer"].items() if key != "kind"}


def file_sha256(path: Path) -> str:
    """Return the sha256 hex digest of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_hashes() -> dict[str, str]:
    """Return the sha256 of each committed base-model file, by file name."""
    return {name: file_sha256(BASE_DIR / name) for name in BASE_FILES}
