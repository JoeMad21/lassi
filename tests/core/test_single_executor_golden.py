"""Golden pins: the single-executor form keeps its resolved recipes and hashes (task P4.5).

Bible: Project Recipes (the recipe blocks; Notes: the recipe hashes of
lassi-repro's mock child and of p1-dry-run), Design Principle 5 (a run
reruns from its resolved recipe), Agent Rule 10.

P4.5 adds a second `executor` form, one executor per language
(test_executors_per_language.py). Every recipe that keeps the single form,
`executor: {kind: ...}`, must resolve and hash exactly as before:

- tests/fixtures/recipes/child.yaml resolves to the golden file
  tests/fixtures/recipes/expected/child.resolved.yaml with the pinned hash
  that tests/core/test_recipe.py pins too (CHILD_HASH), loaded with fake
  components as that module loads it;
- the committed recipes loaded with the default registry keep the hashes
  below, each one Executor binding at executor.kind. The lassi-repro mock
  child and p1-dry-run begin with the prefixes the bible's Project Recipes
  Notes record (0871552c4846 and 43dab1d14715).

Each hash below is the sha256 of a canonical YAML, computed by
lassi.core.recipe at commit f9bb560 (lassi/core/recipe.py and the recipe
files unchanged in the working tree) on 2026-09-25; a hash is a function of
its input text, not a measurement. The fake components are never
constructed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.recipe import load_recipe, resolved_yaml
from lassi.core.registry import Registry

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "recipes"
GOLDEN = FIXTURES / "expected" / "child.resolved.yaml"
CHILD_HASH = "5f6b570b8c15eb771fe27dbd4f525f061cd7665e69318bee8512f44ae1f006c0"

# Committed recipes that use the single form, by path from the repository root, with their recipe hashes.
COMMITTED_HASHES = {
    "tests/fixtures/recipes/p0-smoke.yaml": "b0b5c6e26ece9fc5308b6f736bd2504b7bcc5642c68ed849c77c3685728c6920",
    "tests/fixtures/recipes/p1-dry-run.yaml": "43dab1d147154042349b791f00c0b1ec9c6c5cd44076fdc1d0b34af2089bcef7",
    "tests/fixtures/recipes/p2-proxy-mock.yaml": "6ca943f7597258d98b81bddf440948c31d467796d9ee54448f1692b29104e424",
}
# lassi-repro binds no model, so it loads through a child that names only the mock (bible, Project Recipes Notes).
REPRO_CHILD = b"extends: lassi-repro\nmodel: {backend: mock, id: mock-reference}\n"
REPRO_CHILD_HASH = "0871552c4846877b5c66c42ff0b5a11ad3370ba874f886ed1ffeb3f0642e80bb"


class NeverBuilt:
    """Base for every fake component: constructing one fails the test, since loading constructs nothing."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError(f"component {type(self).__name__} was constructed while loading a recipe")


def fake(
    name: str,
    capabilities: Iterable[str] = (),
    *,
    requires: Mapping[str, Iterable[str]] | None = None,
    config_keys: Iterable[str] = (),
) -> type:
    """Return a fake component class with the class attributes the registry reads."""
    namespace: dict[str, Any] = {
        "__doc__": f"Fake component {name} for the golden recipe pins; never constructed.",
        "name": name,
        "capabilities": frozenset(capabilities),
        "requires": {interface: frozenset(caps) for interface, caps in (requires or {}).items()},
        "config_keys": frozenset(config_keys),
    }
    return type(f"Fake_{name.replace('-', '_')}", (NeverBuilt,), namespace)


def fixture_registry() -> Registry:
    """Return the fake components tests/fixtures/recipes/child.yaml binds, declared as test_recipe.py declares them."""
    registry = Registry()
    for interface, component in (
        ("LLMBackend", fake("mock_llm", {"chat"})),
        ("Toolchain", fake("fake_cc", {"emits_warnings"})),
        ("Toolchain", fake("fake_cxx", {"emits_warnings", "openmp"})),
        ("Executor", fake("native", {"runs_code"}, config_keys={"host"})),
        ("Oracle", fake("stdout_mask", {"stdout_diff"}, config_keys={"passfail"})),
        ("Profiler", fake("timer", {"timing"}, config_keys={"interval_ms"})),
        ("Agent", fake("fuzzer", {"fuzzes_inputs"}, config_keys={"budget"})),
        ("Agent", fake("generator", {"generates"}, requires={"LLMBackend": {"chat"}})),
        ("Agent", fake("fixer", {"fixes_code"})),
        ("Judge", fake("equivalence", {"verdict"})),
        ("Judge", fake("efficiency", {"verdict"})),
        ("ScoreProfile", fake("eval_metrics", {"alignment"})),
        ("Stage", fake("generate", requires={"LLMBackend": {"chat"}})),
        ("Stage", fake("compile_loop", requires={"Toolchain": {"emits_warnings"}})),
        ("Stage", fake("run_loop", requires={"Executor": {"runs_code"}})),
        ("Stage", fake("report")),
    ):
        registry.register(interface, component.name, component)
    return registry


def assert_single_form(recipe: Any) -> None:
    """Assert the recipe binds exactly one Executor, at executor.kind, from a kind section."""
    assert "kind" in recipe.data["executor"]
    bindings = [binding for binding in recipe.bindings if binding.interface == "Executor"]
    assert [binding.where for binding in bindings] == ["executor.kind"]


def test_the_fixture_child_resolves_to_its_golden_file_and_hash() -> None:
    recipe = load_recipe(FIXTURES / "child.yaml", roots=(FIXTURES,), registry=fixture_registry())
    assert recipe.recipe_hash == CHILD_HASH
    assert resolved_yaml(recipe) == GOLDEN.read_bytes().decode("ascii")
    assert_single_form(recipe)


@pytest.mark.parametrize("relative", sorted(COMMITTED_HASHES))
def test_a_committed_single_executor_recipe_keeps_its_hash(relative: str) -> None:
    recipe = load_recipe(REPO / relative)
    assert recipe.recipe_hash == COMMITTED_HASHES[relative]
    assert_single_form(recipe)


def test_the_lassi_repro_mock_child_keeps_the_hash_the_bible_records(tmp_path: Path) -> None:
    child = tmp_path / "model-only.yaml"
    child.write_bytes(REPRO_CHILD)
    recipe = load_recipe(child)
    assert recipe.recipe_hash == REPRO_CHILD_HASH
    assert recipe.recipe_hash.startswith("0871552c4846")
    assert_single_form(recipe)


def test_the_p1_dry_run_hash_begins_with_the_prefix_the_bible_records() -> None:
    assert COMMITTED_HASHES["tests/fixtures/recipes/p1-dry-run.yaml"].startswith("43dab1d14715")
