"""Tests for the adversary section's binding: its Agent must declare proposes_inputs (task P16.2).

Bible: Component Interfaces (capabilities: recipes are validated against
them at load time), Agents And Judges (the Adversary role), Project Recipes
(the agent and judge bindings block: `adversary: {kind: fuzzer, budget:
{inputs: 64}, seed: 1}`; Notes: a mismatch fails before any model call), Agent Rule
3. Plan: plans/p16-adversarial.md, P16.2.

The contract these tests fix: the adversary section binds an Agent at
adversary.kind (lassi.core.recipe _KIND_SECTIONS), and that binding requires
the capability proposes_inputs (lassi.core.capabilities PROPOSES_INPUTS).
lassi.core.registry check_bindings refuses, with a RegistryError, an
adversary.kind whose Agent does not declare it, naming adversary.kind, the
capability, and the Agent; load_recipe reports it as a RecipeError caused
by that RegistryError, before anything is built. The requirement belongs to
the adversary section, not to the Agent interface: an Agent bound under
agents.<role> needs no such capability.

Components here are SYNTHETIC fakes whose constructors fail the test, so a
load that builds anything fails. No value in this module is a measurement.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.core.recipe import Recipe, RecipeError, load_recipe
from lassi.core.registry import Registry, RegistryError, check_bindings

CAPABILITY = "proposes_inputs"
WHERE = "adversary.kind"
CAPABLE = "synthetic-adversary"
INCAPABLE = "synthetic-fuzzes"  # declares a capability, but not proposes_inputs
BARE = "synthetic-bare"  # declares no capability at all
GENERATOR = "synthetic-generator"

CONSTRUCTED: list[str] = []


class NeverBuilt:
    """Base of every fake component: constructing one records its name and fails the test."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Record the construction and fail."""
        name = getattr(type(self), "name", type(self).__name__)
        CONSTRUCTED.append(name)
        raise AssertionError(f"component {name!r} was constructed while loading a recipe")


def fake(name: str, capabilities: Iterable[str] = (), *, config_keys: Iterable[str] = ()) -> type:
    """Return a fake component class with the class attributes the registry reads."""
    namespace: dict[str, Any] = {
        "__doc__": f"SYNTHETIC component {name} for the adversary binding tests; never constructed.",
        "name": name,
        "capabilities": frozenset(capabilities),
        "config_keys": frozenset(config_keys),
    }
    return type(f"Fake_{name.replace('-', '_')}", (NeverBuilt,), namespace)


def registry() -> Registry:
    """Return a Registry of fakes: an executor, a stage, three adversary candidates, and a generator agent."""
    found = Registry()
    found.register("Executor", "none", fake("none", {"compile_only"}))
    found.register("Stage", "generate", fake("generate"))
    found.register("Agent", CAPABLE, fake(CAPABLE, {CAPABILITY}, config_keys={"budget"}))
    found.register("Agent", INCAPABLE, fake(INCAPABLE, {"fuzzes_inputs"}, config_keys={"budget"}))
    found.register("Agent", BARE, fake(BARE, config_keys={"budget"}))
    found.register("Agent", GENERATOR, fake(GENERATOR, {"generates"}))
    return found


# A complete recipe that extends nothing (every REQUIRED choice stated), with SYNTHETIC values.
STANDALONE: dict[str, Any] = {
    "llm": {"sampling": {"temperature": 0.2, "top_p": 0.9}},
    "loop": {"max_corrections": 10},
    "trials": {"n": 1},
    "runs_root": "synthetic-runs",
    "sandbox": {"network": False, "wall_s": 60, "mem_gb": 4},
    "report": {"trial_md": True, "parquet": False},
    "bench": {"suite": "synthetic-suite", "split": "eval"},
    "directions": [{"source": "cpp", "target": "cpp"}],
    "stages": ["generate"],
    "executor": {"kind": "none"},
}


@pytest.fixture(autouse=True)
def clear_constructed() -> Iterator[None]:
    """Start and end every test with no recorded construction."""
    CONSTRUCTED.clear()
    yield
    CONSTRUCTED.clear()


def write(directory: Path, data: Mapping[str, Any], name: str = "recipe-under-test.yaml") -> Path:
    """Write `data` as a YAML recipe under `directory` and return its path."""
    path = directory / name
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


def with_adversary(kind: str, **more: Any) -> dict[str, Any]:
    """Return STANDALONE with an adversary section binding `kind`, plus any other top-level keys."""
    return {**STANDALONE, "adversary": {"kind": kind, "budget": {"inputs": 4}}, **more}


def load(tmp_path: Path, data: Mapping[str, Any]) -> Recipe:
    """Load `data` from a file under `tmp_path` with the fake registry and no extends roots."""
    return load_recipe(write(tmp_path, data), roots=[tmp_path], registry=registry())


def test_an_adversary_that_declares_the_capability_binds_at_adversary_kind(tmp_path: Path) -> None:
    recipe = load(tmp_path, with_adversary(CAPABLE))
    bound = [(b.interface, b.name, b.where, dict(b.config)) for b in recipe.bindings if b.where == WHERE]
    assert bound == [("Agent", CAPABLE, WHERE, {"budget": {"inputs": 4}})]
    assert CONSTRUCTED == []


@pytest.mark.parametrize("kind", [INCAPABLE, BARE], ids=["another-capability", "no-capability"])
def test_load_refuses_an_adversary_whose_agent_lacks_the_capability(tmp_path: Path, kind: str) -> None:
    with pytest.raises(RecipeError) as info:
        load(tmp_path, with_adversary(kind))
    message = str(info.value)
    for needle in (WHERE, CAPABILITY, kind):
        assert needle in message, f"the refusal does not name {needle!r}: {message}"
    assert isinstance(info.value.__cause__, RegistryError), "the refusal does not come from check_bindings"
    assert CONSTRUCTED == []


@pytest.mark.parametrize("kind", [INCAPABLE, BARE], ids=["another-capability", "no-capability"])
def test_check_bindings_refuses_an_adversary_kind_whose_agent_lacks_the_capability(tmp_path: Path, kind: str) -> None:
    # The bindings are the loader's own (whatever a Binding carries for the section), with only the name changed.
    bindings = list(load(tmp_path, with_adversary(CAPABLE)).bindings)
    check_bindings(bindings, registry())
    index = next(index for index, binding in enumerate(bindings) if binding.where == WHERE)
    bindings[index] = dataclasses.replace(bindings[index], name=kind)
    with pytest.raises(RegistryError) as info:
        check_bindings(bindings, registry())
    message = str(info.value)
    for needle in (WHERE, CAPABILITY, kind):
        assert needle in message, f"the refusal does not name {needle!r}: {message}"
    assert CONSTRUCTED == []


def test_the_requirement_belongs_to_the_adversary_section_not_to_every_agent(tmp_path: Path) -> None:
    data = {**STANDALONE, "agents": {GENERATOR: {"model": "synthetic-model"}}}
    recipe = load(tmp_path, data)
    assert [(b.interface, b.where) for b in recipe.bindings if b.interface == "Agent"] == [
        ("Agent", f"agents.{GENERATOR}")
    ]
    assert CONSTRUCTED == []


def test_an_unregistered_adversary_is_still_reported_as_unregistered(tmp_path: Path) -> None:
    with pytest.raises(RecipeError) as info:
        load(tmp_path, with_adversary("synthetic-missing"))
    message = str(info.value)
    assert WHERE in message and "no Agent component is registered as 'synthetic-missing'" in message, message
