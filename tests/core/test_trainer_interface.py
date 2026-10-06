"""Tests for the Trainer interface and its job types (task P17.8).

Bible: Component Interfaces (the table's last row, Trainer, and its contract
rule), Training Module (Compute: a replaceable trainer behind lassi/train;
Safeguards: eval splits are refused by the trainer), Agent Rules 3, 5, and
10; plans/p17-portable.md, the planning decision "Trainer (P17.8)",
confirmed by the owner (OQ-043, entry 14).

The contract these tests fix:

- `Trainer` is a Protocol in lassi/core/interfaces.py that extends
  Component, and the registry's INTERFACES holds it last, after Stage, as
  the thirteenth name (lassi/core/registry.py), so a Trainer registers like
  any component.
- It has one instance method, train(job: TrainJob) -> TrainResult, and a
  staticmethod framework() -> FrameworkBuild that the layer calls on the
  class before anything is built (as hf_local's framework() is).
- TrainData, TrainJob, and TrainResult are frozen dataclasses beside it.
  TrainData holds the data source ("synthetic" or "bench"), the split hash,
  the synthetic records, and for bench the suite and the item names; its
  one read path to a bench item is bench_item(name), which goes through
  Suite.item(name, purpose="train"), so an eval or unassigned item raises
  EvalSplitError (Agent Rule 5), and a train item outside the selected
  items and a synthetic handle are refused with a ValueError. TrainJob
  carries the resolved recipe (mapping and YAML text), the data, the probed
  DeviceRecord, the first provenance, and the output directory; TrainResult
  the steps run and the checkpoint paths.
- A Trainer class registers in a Registry under "Trainer" and the registry
  reads the device key from it only when it declares takes_device.

The suite here is a SYNTHETIC manifest written by the test. No value in this
module is a measurement.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lassi.bench import EvalSplitError, Suite, load_suite
from lassi.core.capabilities import Component
from lassi.core.registry import Registry

INTERFACES_MODULE = "lassi.core.interfaces"
# A SYNTHETIC suite manifest: two train items, one eval item, and one unassigned item.
SUITE_NAME = "synthetic-train"
SUITE_TEXT = """\
suite: synthetic-train
repo: https://example.invalid/synthetic-train
commit: 0123456789abcdef0123456789abcdef01234567
items:
  alpha:
    split: train
    languages:
      omp: {dir: src/alpha-omp, files: [main.cpp]}
  beta:
    split: train
    languages:
      omp: {dir: src/beta-omp, files: [main.cpp]}
  held:
    split: eval
    languages:
      omp: {dir: src/held-omp, files: [main.cpp]}
  open:
    split: unassigned
    languages:
      omp: {dir: src/open-omp, files: [main.cpp]}
"""


def interfaces() -> ModuleType:
    """Import lassi.core.interfaces."""
    return importlib.import_module(INTERFACES_MODULE)


def named(name: str) -> Any:
    """Return lassi.core.interfaces.<name>, failing the test clearly while it does not exist (task P17.8)."""
    module = interfaces()
    if not hasattr(module, name):
        pytest.fail(f"lassi.core.interfaces has no {name} yet (task P17.8)")
    return getattr(module, name)


def synthetic_suite(tmp_path: Path) -> Suite:
    """Write and load the SYNTHETIC suite manifest."""
    path = tmp_path / f"{SUITE_NAME}.yaml"
    path.write_bytes(SUITE_TEXT.encode("ascii"))
    return load_suite(path)


def field_names(cls: type) -> set[str]:
    """Return the field names of a dataclass."""
    assert dataclasses.is_dataclass(cls), f"{cls.__name__} is not a dataclass"
    return {item.name for item in dataclasses.fields(cls)}


def train_data(**changes: Any) -> Any:
    """Return a synthetic TrainData with every field given by keyword; `changes` replaces fields."""
    values: dict[str, Any] = {
        "source": "synthetic",
        "split_hash": "0" * 64,
        "records": ({"prompt": "SYNTHETIC prompt", "completion": "SYNTHETIC completion"},),
        "suite": None,
        "items": (),
    }
    values.update(changes)
    return named("TrainData")(**values)


# ---------------------------------------------------------------------------
# The thirteenth interface


def test_trainer_is_the_thirteenth_interface_last_in_the_registry() -> None:
    registry = importlib.import_module("lassi.core.registry")
    assert len(registry.INTERFACES) == 13, registry.INTERFACES
    assert registry.INTERFACES[-2:] == ("Stage", "Trainer"), registry.INTERFACES


def test_trainer_is_a_documented_protocol_that_extends_component() -> None:
    trainer = named("Trainer")
    assert inspect.isclass(trainer)
    assert trainer.__module__ == INTERFACES_MODULE
    assert getattr(trainer, "_is_protocol", False), "Trainer is not a typing.Protocol"
    assert Component in trainer.__mro__
    assert (trainer.__dict__.get("__doc__") or "").strip()


def test_trainer_train_takes_a_train_job_and_returns_a_train_result() -> None:
    trainer = named("Trainer")
    method = vars(trainer).get("train")
    assert inspect.isfunction(method), "Trainer defines no train method"
    assert (method.__doc__ or "").strip()
    parameters = list(inspect.signature(method).parameters)
    assert parameters == ["self", "job"], parameters
    assert method.__annotations__["job"] in ("TrainJob", named("TrainJob"))
    assert method.__annotations__["return"] in ("TrainResult", named("TrainResult"))


def test_trainer_framework_is_a_staticmethod_returning_a_framework_build() -> None:
    trainer = named("Trainer")
    declared = vars(trainer).get("framework")
    assert isinstance(declared, staticmethod), "Trainer.framework must be a staticmethod, called on the class"
    function = declared.__func__
    assert (function.__doc__ or "").strip()
    assert list(inspect.signature(function).parameters) == []
    assert function.__annotations__["return"] in ("FrameworkBuild", importlib.import_module(
        "lassi.core.devices").FrameworkBuild)


def test_the_job_types_are_frozen_dataclasses_with_the_designed_fields() -> None:
    assert field_names(named("TrainData")) == {"source", "split_hash", "records", "suite", "items"}
    assert field_names(named("TrainJob")) == {"recipe", "recipe_yaml", "data", "device", "provenance", "out_dir"}
    assert field_names(named("TrainResult")) == {"steps", "checkpoints"}
    for name in ("TrainData", "TrainJob", "TrainResult"):
        cls = named(name)
        assert cls.__dataclass_params__.frozen, f"{name} is not frozen"
        assert (cls.__doc__ or "").strip(), f"{name} has no docstring"
    result = named("TrainResult")(steps=2, checkpoints=("checkpoint-1",))
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.steps = 3  # type: ignore[misc]


def test_the_job_types_are_not_protocols() -> None:
    for name in ("TrainData", "TrainJob", "TrainResult"):
        assert not getattr(named(name), "_is_protocol", False), name


# ---------------------------------------------------------------------------
# TrainData.bench_item: the only bench read path a trainer gets (Agent Rule 5)


def test_bench_item_reads_through_the_registry_for_purpose_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = synthetic_suite(tmp_path)
    calls: list[tuple[str, str]] = []
    original = Suite.item

    def spy(self: Suite, name: str, *, purpose: str) -> Any:
        calls.append((name, purpose))
        return original(self, name, purpose=purpose)

    monkeypatch.setattr(Suite, "item", spy)
    data = train_data(source="bench", records=(), suite=suite, items=("alpha", "beta"))
    found = data.bench_item("alpha")
    assert found.name == "alpha" and found.split == "train"
    assert calls == [("alpha", "train")]


@pytest.mark.parametrize("item", ["held", "open"], ids=["eval", "unassigned"])
def test_bench_item_refuses_an_eval_or_unassigned_item(item: str, tmp_path: Path) -> None:
    suite = synthetic_suite(tmp_path)
    data = train_data(source="bench", records=(), suite=suite, items=("alpha", "beta"))
    with pytest.raises(EvalSplitError) as refused:
        data.bench_item(item)
    assert "Agent Rule 5" in str(refused.value)


def test_bench_item_refuses_a_train_item_that_was_not_selected(tmp_path: Path) -> None:
    data = train_data(source="bench", records=(), suite=synthetic_suite(tmp_path), items=("alpha",))
    with pytest.raises(ValueError, match="not among the selected items") as refused:
        data.bench_item("beta")
    assert not isinstance(refused.value, EvalSplitError)


def test_a_synthetic_handle_has_no_bench_item() -> None:
    data = train_data()
    with pytest.raises(ValueError):
        data.bench_item("alpha")


# ---------------------------------------------------------------------------
# Registration


def test_a_trainer_class_registers_under_the_trainer_interface() -> None:
    class Declared:
        """A SYNTHETIC trainer declaration; never built."""

        name = "declared"
        capabilities = frozenset({"takes_device"})
        config_keys = frozenset({"steps"})
        methods = frozenset({"sft"})
        weight_modes = frozenset({"full"})
        data_sources = frozenset({"synthetic"})
        packages = ("pytest",)

    registry = Registry()
    entry = registry.register("Trainer", "declared", Declared)
    assert (entry.interface, entry.name, entry.factory) == ("Trainer", "declared", Declared)
    assert registry.names("Trainer") == ["declared"]
