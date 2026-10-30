"""Tests for sweep15's TTIR feature ladder and the bare template set (task P17.61).

Bible: Project Recipes (stages, template sets, the context, retrieval, and
source_mlir sections, the loop opt-ins), Agent Rules 2 and 5 (nothing runs;
tt-pairs-v0 is a development set, OQ-083 (b); no item named in a template).

What they fix for projects/lassi-demo/sweep15/gpt-oss-120b/:

- Six recipes, ttir-l0 to ttir-l5, each labelled compile only and
  exploratory in its header; each rung adds exactly one feature to the rung
  below: l1 the API reference (with verbatim_context), l2 retrieval, l3 the
  cgeist affine source, l4 tt-ttir-v0's guidance, l5 restarts.
- l0 is the bare LASSI loop: generate and compile_loop alone, the bare
  template set, a cap of 8 corrections, no restarts; l5 is sweep13's
  ttir-t10 but for its project name.
- tt-ttir-bare-v0 holds the templates the stages render, plain ASCII, each
  using only its fields and naming no item or item shape; its generate and
  correct carry no guidance beyond the task, the declared arrays, and the
  errors; its other templates are tt-ttir-v0's, unchanged.
"""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any

import pytest

from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.recipe import load_recipe
from lassi.core.stages import (
    CORRECT_FIELDS,
    GENERATE_FIELDS,
    KNOWLEDGE_FIELDS,
    PASSAGE_FIELDS,
    RETRIEVED_FIELDS,
    SOURCE_MLIR_FIELDS,
    SUMMARY_FIELDS,
)
from lassi.prompts import render

REPO = Path(__file__).resolve().parents[2]
RECIPES = REPO / "projects" / "lassi-demo" / "sweep15" / "gpt-oss-120b"
SWEEP13 = REPO / "projects" / "lassi-demo" / "sweep13" / "gpt-oss-120b" / "ttir-t10.yaml"
BARE = REPO / "assets" / "prompts" / "tt-ttir-bare-v0"
GUIDED = REPO / "assets" / "prompts" / "tt-ttir-v0"
NAMES = [f"ttir-l{rung}" for rung in range(6)]
# The keys each rung changes from the rung below, besides its project name.
ADDED = {1: {"context", "stages"}, 2: {"retrieval"}, 3: {"source_mlir"}, 4: {"prompts"}, 5: {"loop"}}
FIELDS = {"generate": GENERATE_FIELDS, "correct": CORRECT_FIELDS, "knowledge": KNOWLEDGE_FIELDS,
          "summarize_context": SUMMARY_FIELDS, "retrieved": RETRIEVED_FIELDS, "passage": PASSAGE_FIELDS,
          "source_mlir": SOURCE_MLIR_FIELDS}
ITEMS = ("eltwise_binary", "eltwise_sfpu", "loopback", "matmul_multi_core", "matmul_single_core")
SHAPES = ("65536", "51200", "640", "256x256", "160x320")


def load(name: str) -> dict[str, Any]:
    """Return the resolved data of one sweep15 recipe."""
    return load_recipe(RECIPES / f"{name}.yaml").data


def changed(upper: dict[str, Any], lower: dict[str, Any]) -> set[str]:
    """Return the top-level keys whose values differ between two resolved recipes, the project name aside."""
    return {key for key in set(upper) | set(lower) if key != "project" and upper.get(key) != lower.get(key)}


def placeholders(text: str) -> set[str]:
    """Return the `$name` placeholders of a template; fail on a `$` that is no placeholder."""
    names = set()
    for match in Template.pattern.finditer(text):
        assert match.group("invalid") is None
        name = match.group("named") or match.group("braced")
        if name:
            names.add(name)
    return names


def test_the_directory_holds_exactly_the_six_rungs() -> None:
    assert sorted(path.stem for path in RECIPES.glob("*.yaml")) == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_each_rung_is_compile_only_and_says_so(name: str) -> None:
    data = load(name)
    assert data["executor"] == {"kind": "none"} and "oracle" not in data
    assert data["stages"][-2:] == ["generate", "compile_loop"]
    assert data["toolchain"] == {"ttir": "ttir-ttmetal-compile", "tt": "ttmetal-host"}
    assert data["llm"]["sampling"]["temperature"] == 1.0 and data["project"] == name
    text = (RECIPES / f"{name}.yaml").read_text(encoding="ascii")
    header = " ".join(line.lstrip("# ") for line in text.split("\nextends:")[0].split("\n"))
    assert "COMPILE ONLY" in header and "no output is checked" in header.lower()
    assert "exploratory" in header.lower()


@pytest.mark.parametrize("rung", sorted(ADDED))
def test_each_rung_adds_exactly_one_feature(rung: int) -> None:
    assert changed(load(f"ttir-l{rung}"), load(f"ttir-l{rung - 1}")) == ADDED[rung]


def test_the_bottom_rung_is_the_bare_loop() -> None:
    data = load("ttir-l0")
    assert data["stages"] == ["generate", "compile_loop"] and data["prompts"] == "tt-ttir-bare-v0"
    for key in ("context", "retrieval", "source_mlir", "dialect_lookup", "examples"):
        assert key not in data, key
    assert data["loop"] == {"max_corrections": 8}


def test_the_top_rung_is_sweep13s_ttir_t10() -> None:
    assert changed(load("ttir-l5"), load_recipe(SWEEP13).data) == set()


def test_the_bare_set_holds_the_templates_the_stages_render() -> None:
    assert {path.stem for path in BARE.glob("*.txt")} == set(FIELDS)
    for name in ("knowledge", "passage", "retrieved", "source_mlir", "summarize_context"):
        assert (BARE / f"{name}.txt").read_bytes() == (GUIDED / f"{name}.txt").read_bytes(), name


@pytest.mark.parametrize("template", sorted(FIELDS))
def test_each_bare_template_is_ascii_uses_its_fields_and_names_no_item(template: str) -> None:
    data = (BARE / f"{template}.txt").read_bytes()
    assert data.isascii() and b"\r" not in data
    text = data.decode("ascii")
    assert placeholders(text) <= set(FIELDS[template])
    assert render("tt-ttir-bare-v0", template, dict.fromkeys(FIELDS[template], ""))
    for word in ITEMS + SHAPES:
        assert word not in text, f"tt-ttir-bare-v0/{template}.txt names {word}"


@pytest.mark.parametrize("template", ["generate", "correct"])
def test_the_bare_prompts_carry_no_guidance(template: str) -> None:
    text = (BARE / f"{template}.txt").read_text(encoding="ascii")
    guided = (GUIDED / f"{template}.txt").read_text(encoding="ascii")
    assert len(text) < len(guided) / 2
    for hint in ("multiply, maximum", "Element types", "32x32", "No loops", "the reference gives", "Remember"):
        assert hint not in text, hint
    if template == "generate":
        assert {"input_shapes", "source_files", "target_files", "source_mlir"} <= placeholders(text)
        assert "--ttir-to-ttmetal-pipeline" in text and "5f396fd6" in text
    else:
        assert {"files", "diagnostics", "target_files"} <= placeholders(text)
