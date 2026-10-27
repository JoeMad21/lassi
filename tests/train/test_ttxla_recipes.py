"""Tests for the training recipes of the Wormhole window (task P17.38): projects/lassi-demo/train-tt/.

Bible: Training Module (Weight Modes, Training Data, Compute), Agent Rules 1,
5, 9, and 10; plans/spikes/tt-training.md, Finding 3, step 4; OQ-076 (the
owner's techniques: full training, frozen weights, LoRA, a model of our
own); OQ-100 (s2); OQ-102 (3a): the smoke on the synthetic fixture first,
then tt-train-v0, any tt-train-v0 failure reported as found.

These tests fix the six recipes, in the window's order: the smoke on the
synthetic fixture (fixture:tiny-causal-lm, data.synthetic sft-smoke.jsonl),
then on tt-train-v0 (data.corpus): LoRA on Qwen2.5-0.5B, Qwen3-0.6B, and
Qwen3-1.7B at their pinned revisions, full training of the 77M scratch
model, and prompt tuning on Qwen3-1.7B. Each loads with the train loader,
binds the ttxla trainer on Tenstorrent card 0 with method sft, sets
max_length (a multiple of 32, at least tt-train-v0's longest record in the
chat format for the corpus recipes) and timeout_s, and carries PROJECTED
time and memory in its comments. No value here is a measurement.
"""

from __future__ import annotations

import pytest
from train_fakes import REPO

import lassi.train  # noqa: F401  (importing the package registers the trainers)
from lassi.core.recipe import load_train_recipe

RECIPE_DIR = REPO / "projects" / "lassi-demo" / "train-tt"
ORDER = (
    "ttxla-smoke-fixture",
    "ttxla-lora-qwen2.5-0.5b",
    "ttxla-lora-qwen3-0.6b",
    "ttxla-lora-qwen3-1.7b",
    "ttxla-full-scratch-77m",
    "ttxla-prompt-tuning-qwen3-1.7b",
)
# The bases at the revisions plans/spikes/tt-training.md, Command 7, read, and the bible's Training host bullet pins.
BASES = {
    "ttxla-smoke-fixture": ("fixture:tiny-causal-lm", "full"),
    "ttxla-lora-qwen2.5-0.5b": ("Qwen/Qwen2.5-0.5B@060db6499f32faf8b98477b0a26969ef7d8b9987", "lora"),
    "ttxla-lora-qwen3-0.6b": ("Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca", "lora"),
    "ttxla-lora-qwen3-1.7b": ("Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "lora"),
    "ttxla-full-scratch-77m": ("scratch:qwen3-77m", "full"),
    "ttxla-prompt-tuning-qwen3-1.7b": ("Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "prompt_tuning"),
}
# tt-train-v0's longest record in the Qwen3 chat format (bible, Training Data; a workstation count, PROJECTED).
LONGEST_RECORD = 3058


def test_the_directory_holds_the_six_recipes_and_nothing_else() -> None:
    assert sorted(path.stem for path in RECIPE_DIR.glob("*.yaml")) == sorted(ORDER)


@pytest.mark.parametrize("name", ORDER)
def test_each_recipe_binds_ttxla_on_card_0_with_sft_and_its_base(name: str) -> None:
    recipe = load_train_recipe(RECIPE_DIR / f"{name}.yaml")
    data = recipe.data
    base, weights = BASES[name]
    assert (data["base_model"], data["weights"], data["method"]) == (base, weights, "sft")
    trainer = data["trainer"]
    assert trainer["kind"] == "ttxla"
    assert trainer["device"] == {"kind": "tenstorrent", "indices": [0]}
    assert trainer["network"] == "none", "the worker runs in its own empty network namespace (plans/spikes/tt-perf.md)"
    assert trainer["max_length"] % 32 == 0
    assert isinstance(trainer["timeout_s"], int) and 600 <= trainer["timeout_s"] <= 7200
    assert trainer["batch_size"] == 1
    if name == "ttxla-smoke-fixture":
        assert data["data"] == {"synthetic": "sft-smoke.jsonl"}
        assert trainer.get("chat_format") in (None, False), "the fixture tokenizer has no chat template"
    else:
        assert data["data"] == {"corpus": "tt-train-v0"}
        assert trainer["chat_format"] is True
        assert trainer["max_length"] >= LONGEST_RECORD + 1
    if weights == "lora":
        assert data["lora"]["targets"] == "all-linear"
    if weights == "prompt_tuning":
        assert trainer["virtual_tokens"] > 0


@pytest.mark.parametrize("name", ORDER)
def test_each_recipe_carries_projected_time_and_memory(name: str) -> None:
    text = (RECIPE_DIR / f"{name}.yaml").read_text(encoding="ascii")
    comments = " ".join(line for line in text.splitlines() if line.lstrip().startswith("#"))
    assert "PROJECTED" in comments
    for word in ("minutes", "memory", "timeout_s"):
        assert word in comments.lower(), f"{name}: no {word!r} in its comments"
    assert "GB" in comments or "MB" in comments, f"{name}: no memory figure in its comments"
    assert "\r" not in text


def test_the_qwen2_5_recipe_leaves_room_for_its_default_system_turn() -> None:
    trainer = load_train_recipe(RECIPE_DIR / "ttxla-lora-qwen2.5-0.5b.yaml").data["trainer"]
    assert trainer["max_length"] >= 3104, "Qwen2.5's template adds a default system turn (the spike, Finding 2)"
