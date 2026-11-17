"""Every committed recipe of tt-pairs-v0 and its siblings reads the dev split (task P17.31; OQ-083, option (b)).

Bible: Benchmark Suites (split rules: a recipe selects the items of its split
by the split's exact name, and the runner refuses a split with no items;
tt-pairs-v0 is a development set, 2026-10-25), Project Recipes, Agent Rule 5.
Plan: plans/p17-portable.md, P17.31 ("every recipe that reads them with
`split: unassigned` moved to it").

The contract these tests fix:

- Every run recipe under projects/ whose resolved bench.suite is tt-pairs-v0,
  tt-pairs-v0-kh, tt-pairs-v0-ttkernel, tt-pairs-v0-ttir, or tt-pairs-v0-d2m
  names bench.split dev, never unassigned, and still loads (as
  tests/core/test_faithful_bindings.py loads every committed recipe).
- A recipe of any other suite keeps its split: none names dev.

The recipes are the committed ones, resolved through their extends chains;
loading builds no component. No value here is a measurement.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import pytest
from test_faithful_bindings import PROJECTS, committed_recipes, load_committed

from lassi.core.recipe import resolved_data

DEV_SUITES = frozenset({"tt-pairs-v0", "tt-pairs-v0-kh", "tt-pairs-v0-ttkernel", "tt-pairs-v0-ttir", "tt-pairs-v0-d2m"})


@cache
def benches() -> dict[Path, dict[str, Any]]:
    """Return {recipe path: resolved bench section} for every committed run recipe that has one."""
    found: dict[Path, dict[str, Any]] = {}
    for path in committed_recipes():
        bench = resolved_data(path).get("bench")
        if isinstance(bench, dict):
            found[path] = bench
    return found


def development_recipes() -> list[Path]:
    """Return the committed recipes whose resolved bench.suite is one of the development suites."""
    return sorted(path for path, bench in benches().items() if bench.get("suite") in DEV_SUITES)


def label(path: Path) -> str:
    """Return a recipe's path under projects/, as a test id."""
    return path.relative_to(PROJECTS).as_posix()


def test_the_development_suites_have_committed_recipes() -> None:
    """The walk finds recipes of the development suites, so the per-recipe tests below are not vacuous."""
    assert development_recipes(), "no committed recipe names a tt-pairs-v0 suite"
    assert {benches()[path]["suite"] for path in development_recipes()} == DEV_SUITES


def test_no_recipe_of_a_development_suite_reads_split_unassigned() -> None:
    """No committed recipe of the five development suites still names bench.split unassigned."""
    still = [label(path) for path in development_recipes() if benches()[path].get("split") == "unassigned"]
    assert not still, f"{len(still)} recipes of a development suite still read split unassigned, e.g. {still[:5]}"


@pytest.mark.parametrize("path", development_recipes(), ids=label)
def test_a_recipe_of_a_development_suite_reads_split_dev_and_loads(tmp_path: Path, path: Path) -> None:
    """Each recipe of a development suite resolves to bench.split dev and loads through the recipe loader."""
    assert benches()[path].get("split") == "dev", f"{label(path)}: bench.split is {benches()[path].get('split')!r}"
    loaded = load_committed(path, tmp_path)
    assert loaded.data["bench"]["split"] == "dev"


def test_no_recipe_of_another_suite_reads_split_dev() -> None:
    """Recipes of every other suite keep their split: none names dev."""
    moved = [label(path) for path, bench in benches().items()
             if bench.get("suite") not in DEV_SUITES and bench.get("split") == "dev"]
    assert not moved, f"recipes of suites outside the development set read split dev: {moved[:5]}"
