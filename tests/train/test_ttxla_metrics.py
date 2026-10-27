"""Tests for the training metrics of the ttxla worker and trainer (task P17.42; plans/spikes/tt-perf.md, (F)).

Bible: Training Module (Compute, Safeguards), Evaluation Protocol (Reporting
Rules), Agent Rules 1, 2, and 9; the owner's answer of 2026-10-26 to the
spike's OQ-106, option (a): PaLM model FLOPs for training, rates over
a measured time that they name.

The worker (lassi/train/ttxla_worker.py; framework-free parts, tested here
without torch) gains:

- palm_flop_per_token(frozen, trained, layers, heads, head_dim, sequence):
  PaLM's model FLOPs per token (arXiv:2204.02311, Appendix B), 6N + 12LHQT
  with N the matmul weights a token passes through (the output projection
  in, the embedding lookup out): each trained weight 6 (forward, input
  gradient, weight gradient), each frozen weight 4 (no weight gradient),
  and the attention matmuls 12LHQT whatever is frozen, since they hold no
  weights and their backward computes both operands' gradients; rematerial-
  ization (gradient checkpointing) is not counted.
- matmul_weights(model, linear): (frozen, trained) weight counts over every
  module of class `linear` (torch.nn.Linear on the card), by requires_grad.
- attention_shape(config): (layers, heads, head_dim) from a transformers
  config: num_hidden_layers, num_attention_heads, and head_dim, else
  hidden_size // num_attention_heads.
- model_flops(model, config, job, linear): the convention record result.json
  carries: its name palm-model-flops, the formula, the counts, the sequence
  T (max_length plus any virtual tokens, every sequence padded to it), and
  flop_per_token.
- step_line(...): one steps.jsonl line: step, loss, learning_rate,
  label_tokens, elapsed_s as before, then step_time_s (host monotonic time
  from the step's start to after its second device sync), tokens (the
  batch's non-pad tokens), tokens_computed (batch_size x T, the padded
  tokens run), model_flop (flop_per_token x tokens_computed), compiles (the
  graph compiles the step added, from the backend's compile counter; None
  when it has none), and compile_step (compiles > 0, or, without a counter,
  the first COMPILE_RULE_STEPS steps by rule).
- The worker never sets a profiler variable: training runs have no network
  and no device profiler (tests/train/test_ttxla_trainer.py checks the
  environment).

The trainer writes checkpoint.json's trainer_records.performance:
lassi.analysis.tt_rates training_summary over the steps and the worker's
model_flops, naming the card (tests/train/test_ttxla_trainer.py).

Every number here is SYNTHETIC arithmetic; none is a measurement.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


@pytest.fixture(scope="module")
def module() -> ModuleType:
    """Import lassi.train.ttxla_worker."""
    return importlib.import_module("lassi.train.ttxla_worker")


@dataclass
class Weight:
    """A stand-in for a parameter: its element count and whether it trains."""

    size: int
    requires_grad: bool

    def numel(self) -> int:
        """Return the element count."""
        return self.size


@dataclass
class Linear:
    """A stand-in for torch.nn.Linear: one weight."""

    weight: Weight


@dataclass
class Embedding:
    """A stand-in for torch.nn.Embedding, which no matmul passes through."""

    weight: Weight


class Model:
    """A stand-in model whose modules() lists its parts."""

    def __init__(self, *parts: Any) -> None:
        """Keep the parts."""
        self.parts = parts

    def modules(self) -> list[Any]:
        """Return the parts, as torch's modules() walks a model."""
        return list(self.parts)


def test_palm_flop_per_token_for_full_and_frozen_weights(module: ModuleType) -> None:
    # Full weights: 6N + 12LHQT (PaLM, Appendix B).
    assert module.palm_flop_per_token(frozen=0, trained=1000, layers=2, heads=4, head_dim=8, sequence=16) == (
        6 * 1000 + 12 * 2 * 4 * 8 * 16
    )
    # A frozen base with an adapter: 4 per frozen weight, 6 per trained one; attention unchanged.
    assert module.palm_flop_per_token(frozen=1000, trained=10, layers=2, heads=4, head_dim=8, sequence=16) == (
        4 * 1000 + 6 * 10 + 12 * 2 * 4 * 8 * 16
    )


def test_matmul_weights_counts_linear_weights_by_requires_grad(module: ModuleType) -> None:
    model = Model(Embedding(Weight(5000, True)), Linear(Weight(300, False)), Linear(Weight(200, False)),
                  Linear(Weight(7, True)), SimpleNamespace())
    assert module.matmul_weights(model, Linear) == (500, 7), "the embedding lookup is not a matmul"


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (SimpleNamespace(num_hidden_layers=28, num_attention_heads=16, head_dim=128, hidden_size=1024), (28, 16, 128)),
        (SimpleNamespace(num_hidden_layers=24, num_attention_heads=14, hidden_size=896), (24, 14, 64)),
        (SimpleNamespace(num_hidden_layers=2, num_attention_heads=4, head_dim=None, hidden_size=64), (2, 4, 16)),
    ],
)
def test_attention_shape_reads_the_config(module: ModuleType, config: Any, expected: tuple[int, int, int]) -> None:
    assert module.attention_shape(config) == expected


def test_model_flops_records_the_convention_and_its_parts(module: ModuleType) -> None:
    model = Model(Linear(Weight(1000, False)), Linear(Weight(24, True)))
    config = SimpleNamespace(num_hidden_layers=2, num_attention_heads=4, head_dim=8, hidden_size=32)
    job = {"max_length": 16, "virtual_tokens": 4}
    found = module.model_flops(model, config, job, Linear)
    per_token = 4 * 1000 + 6 * 24 + 12 * 2 * 4 * 8 * 20
    assert found == {
        "convention": "palm-model-flops",
        "formula": module.PALM_FORMULA,
        "frozen_weights": 1000,
        "trained_weights": 24,
        "layers": 2,
        "heads": 4,
        "head_dim": 8,
        "sequence": 20,
        "flop_per_token": per_token,
    }
    for words in ("6", "4", "12", "PaLM", "recompute"):
        assert words in module.PALM_FORMULA


def line(module: ModuleType, step: int, compiles: int | None, **changes: Any) -> dict[str, Any]:
    """Return step_line's SYNTHETIC output for `step` with `compiles`."""
    values: dict[str, Any] = {"step": step, "loss": 0.5, "learning_rate": 0.001, "label_tokens": 30,
                              "elapsed_s": 3.25, "step_time_s": 0.75, "tokens": 40, "tokens_computed": 64,
                              "flop_per_token": 1000, "compiles": compiles}
    values.update(changes)
    return module.step_line(**values)


def test_a_step_line_carries_the_time_the_tokens_and_the_model_flops(module: ModuleType) -> None:
    found = line(module, 3, 0)
    assert found == {
        "step": 3, "loss": 0.5, "learning_rate": 0.001, "label_tokens": 30, "elapsed_s": 3.25,
        "step_time_s": 0.75, "tokens": 40, "tokens_computed": 64, "model_flop": 64_000, "compiles": 0,
        "compile_step": False,
    }


def test_a_step_that_compiled_is_flagged(module: ModuleType) -> None:
    assert line(module, 5, 2)["compile_step"] is True


def test_without_a_compile_counter_the_first_steps_are_flagged_by_rule(module: ModuleType) -> None:
    assert module.COMPILE_RULE_STEPS == 2
    flags = [line(module, step, None)["compile_step"] for step in (1, 2, 3, 4)]
    assert flags == [True, True, False, False]
    assert line(module, 3, None)["compiles"] is None


def test_the_cpu_backend_counts_no_compile(module: ModuleType) -> None:
    counter = module.compile_counter({"backend": "cpu"})
    assert counter() == 0


def test_the_worker_names_no_profiler_variable(module: ModuleType) -> None:
    source = Path(module.__file__).read_text(encoding="ascii")
    assert "TT_METAL_DEVICE_PROFILER" not in source and "TT_METAL_PROFILER" not in source
    assert sys.modules["lassi.train.ttxla_worker"] is module
