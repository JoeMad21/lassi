"""Tests for sweep17's TTIR and D2M arms that compile and run on ttsim with their outputs checked (task P17.48).

Bible: Project Recipes (stages, template sets, the context, retrieval, and
source_mlir sections, the loop opt-ins, the oracle), Execution Backends (the
ttsim row), Agent Rules 2 and 5 (ttsim is a simulator; tt-pairs-v0 is a
development set, OQ-083 (b); no item named in a template).

What they fix for projects/lassi-demo/sweep17/:

- gpt-oss-120b/ holds three recipes, one per arm: ttir-min (sweep16's
  ttir-min, rung l1's features, plus the run), ttir-full (ttir-min plus rung
  l2's retrieval and rung l3's cgeist source), and d2m-min (rung l1's
  features for D2M: the bare loop and the D2M API reference). Each judges
  every item by its own declared tolerance, so one recipe holds the max_abs
  and the pcc items (OQ-115).
- The run, the same in every arm: the target compiled by its ttsim preset
  (ttir-ttmetal-ttsim, d2m-ttmetal-ttsim) and run by the ttmlir-ttsim
  executor on ttsim, both references built and run (baseline_both; tt on
  ttsim, cpp natively), binary_io against the tt program within the item's
  declared tolerance (metric and threshold from_item) with the element-count
  shape rule, and run_loop with correct_on_mismatch within rung l1's cap of 8.
- Each header says it compiles and runs on ttsim, a simulator, that outputs
  are checked against the item's tolerance, and that it is exploratory.
- tt-ttir-bare-v1 is tt-ttir-bare-v0 with correct.txt worded for run failures
  too and a bare correct_mismatch.txt; tt-d2m-bare-v1 is the same for D2M,
  with tt-d2m-v0's reference, retrieval, and cgeist templates. Each template
  is plain ASCII, uses only its fields, renders, and names no item or shape.
- check/ holds the end-to-end check recipes: the replay backend on
  SYNTHETIC recordings under tests/fixtures/ttmlir-ttsim/, and the same
  toolchains, executors, and oracle as the arms.
"""

from __future__ import annotations

import json
from pathlib import Path
from string import Template
from typing import Any

import pytest

from lassi.bench.registry import load_suite
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.recipe import load_recipe
from lassi.core.stages import (
    CORRECT_FIELDS,
    GENERATE_FIELDS,
    KNOWLEDGE_FIELDS,
    MISMATCH_FIELDS,
    PASSAGE_FIELDS,
    RETRIEVED_FIELDS,
    SOURCE_MLIR_FIELDS,
    SUMMARY_FIELDS,
)
from lassi.prompts import render

REPO = Path(__file__).resolve().parents[2]
SWEEP = REPO / "projects" / "lassi-demo" / "sweep17"
RECIPES = SWEEP / "gpt-oss-120b"
CHECKS = SWEEP / "check"
SWEEP15 = REPO / "projects" / "lassi-demo" / "sweep15" / "gpt-oss-120b"
SWEEP16 = REPO / "projects" / "lassi-demo" / "sweep16" / "gpt-oss-120b" / "ttir-min.yaml"
PROMPTS = REPO / "assets" / "prompts"
ARMS = {"ttir-min": "ttir", "ttir-full": "ttir", "d2m-min": "d2m"}
NAMES = sorted(ARMS)
CHECK_NAMES = ["d2m-sfpu-replay", "ttir-eltwise-replay", "ttir-matmul-replay", "ttir-sfpu-replay"]
ARM_ITEMS = {
    "ttir": ["eltwise_binary", "eltwise_sfpu", "matmul_multi_core"],
    "d2m": ["eltwise_binary", "eltwise_sfpu", "loopback", "matmul_multi_core"],
}
# The tolerances the suite manifests declare for those items (assets/bench/tt-pairs-v0-ttir.yaml, -d2m.yaml).
TOLERANCES = {"eltwise_binary": ("max_abs", 0.01), "eltwise_sfpu": ("max_abs", 0.05), "loopback": ("max_abs", 0),
              "matmul_multi_core": ("pcc", 0.97)}
ITEM_ORACLE = {"kind": "binary_io", "metric": "from_item", "threshold": "from_item", "shape": "elements"}
# The keys ttir-min changes from sweep16's ttir-min, besides its project name: the run (its items are sweep16's).
# Every fix is on by default, so the recipe's explicit fixes.baseline_both (as tier-a-rngd-maxabs.yaml states it)
# changes nothing.
RUN_KEYS = {"prompts", "toolchain", "stages", "executor", "oracle", "loop"}
FIELDS = {"generate": GENERATE_FIELDS, "correct": CORRECT_FIELDS, "correct_mismatch": MISMATCH_FIELDS,
          "knowledge": KNOWLEDGE_FIELDS, "summarize_context": SUMMARY_FIELDS, "retrieved": RETRIEVED_FIELDS,
          "passage": PASSAGE_FIELDS, "source_mlir": SOURCE_MLIR_FIELDS}
ITEMS = ("eltwise_binary", "eltwise_sfpu", "loopback", "matmul_multi_core", "matmul_single_core")
SHAPES = ("65536", "51200", "640", "256x256", "160x320")


def load(path: Path) -> dict[str, Any]:
    """Return the resolved data of one recipe."""
    return load_recipe(path).data


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


def header(path: Path) -> str:
    """Return a recipe's leading comment lines joined into one line."""
    text = path.read_text(encoding="ascii")
    return " ".join(line.lstrip("# ") for line in text.split("\nextends:")[0].split("\n"))


# ---------------------------------------------------------------------------
# The arms


def test_the_directory_holds_exactly_the_three_recipes() -> None:
    assert sorted(path.stem for path in RECIPES.glob("*.yaml")) == NAMES


@pytest.mark.parametrize("name", NAMES)
def test_each_recipe_compiles_and_runs_on_ttsim_with_its_outputs_checked(name: str) -> None:
    data = load(RECIPES / f"{name}.yaml")
    target = ARMS[name]
    assert data["project"] == name and data["directions"] == [{"source": "cpp", "target": target}]
    assert data["toolchain"] == {target: f"{target}-ttmetal-ttsim", "tt": "ttmetal-host", "cpp": "gcc-native"}
    assert data["executor"] == {
        "cpp": {"kind": "native"},
        "tt": {"kind": "ttsim", "arch": "wormhole_b0", "dispatch": "slow"},
        target: {"kind": "ttmlir-ttsim", "arch": "wormhole_b0", "dispatch": "slow"},
    }
    assert data["stages"] == ["baseline", "verbatim_context", "generate", "compile_loop", "run_loop", "oracle"]
    assert data["fixes"]["baseline_both"] is True
    assert data["oracle"] == ITEM_ORACLE
    assert data["loop"] == {"max_corrections": 8, "correct_on_mismatch": True}
    assert data["llm"]["sampling"]["temperature"] == 1.0 and data["trials"] == {"n": 2}
    assert data["model"]["id"] == "furiosa-ai/gpt-oss-120b"
    assert data["bench"]["items"] == ARM_ITEMS[target] and data["bench"]["suite"] == f"tt-pairs-v0-{target}"


@pytest.mark.parametrize("target", sorted(ARM_ITEMS))
def test_each_arm_item_declares_the_tolerance_the_oracle_binds(target: str) -> None:
    suite = load_suite(REPO / "assets" / "bench" / f"tt-pairs-v0-{target}.yaml")
    for item in ARM_ITEMS[target]:
        tolerance = suite.items[item].tolerance
        assert tolerance is not None, item
        assert (tolerance.metric, tolerance.threshold) == TOLERANCES[item], item


@pytest.mark.parametrize("name", NAMES)
def test_each_header_says_ttsim_a_simulator_the_item_tolerance_and_exploratory(name: str) -> None:
    text = header(RECIPES / f"{name}.yaml")
    assert "COMPILE AND RUN ON TTSIM" in text and "COMPILE ONLY" not in text
    assert "simulator" in text and "exploratory" in text.lower()
    assert "item's declared tolerance" in text and "from_baseline" not in text


def test_ttir_min_is_sweep16s_ttir_min_plus_the_run() -> None:
    data, sweep16 = load(RECIPES / "ttir-min.yaml"), load(SWEEP16)
    assert changed(data, sweep16) == RUN_KEYS
    assert data["prompts"] == "tt-ttir-bare-v1" and data["context"] == ["ttir-ref-5f396fd6"]
    for key in ("retrieval", "source_mlir", "dialect_lookup", "examples"):
        assert key not in data, key


def test_ttir_full_adds_rung_l2s_retrieval_and_rung_l3s_source() -> None:
    full, minimal = load(RECIPES / "ttir-full.yaml"), load(RECIPES / "ttir-min.yaml")
    assert changed(full, minimal) == {"retrieval", "source_mlir"}
    rung3 = load(SWEEP15 / "ttir-l3.yaml")
    assert full["retrieval"] == rung3["retrieval"] and full["source_mlir"] == rung3["source_mlir"]


def test_d2m_min_is_ttir_min_for_d2m() -> None:
    d2m, ttir = load(RECIPES / "d2m-min.yaml"), load(RECIPES / "ttir-min.yaml")
    assert changed(d2m, ttir) == {"bench", "directions", "prompts", "context", "toolchain", "executor"}
    assert d2m["prompts"] == "tt-d2m-bare-v1" and d2m["context"] == ["d2m-ref-5f396fd6"]
    for key in ("retrieval", "source_mlir"):
        assert key not in d2m, key


# ---------------------------------------------------------------------------
# The template sets


def bare_v1_sets() -> dict[str, Path]:
    """Return the two run-aware bare sets by target."""
    return {"ttir": PROMPTS / "tt-ttir-bare-v1", "d2m": PROMPTS / "tt-d2m-bare-v1"}


@pytest.mark.parametrize("target", ["ttir", "d2m"])
def test_each_set_holds_the_templates_the_stages_render(target: str) -> None:
    assert {path.stem for path in bare_v1_sets()[target].glob("*.txt")} == set(FIELDS)


def test_tt_ttir_bare_v1_is_bare_v0_with_the_run_corrections() -> None:
    v1, v0 = bare_v1_sets()["ttir"], PROMPTS / "tt-ttir-bare-v0"
    for name in ("generate", "knowledge", "passage", "retrieved", "source_mlir", "summarize_context"):
        assert (v1 / f"{name}.txt").read_bytes() == (v0 / f"{name}.txt").read_bytes(), name
    correct = (v1 / "correct.txt").read_text(encoding="ascii")
    assert "failed when it ran" in correct and "standard error" in correct
    assert {"files", "diagnostics", "target_files"} <= placeholders(correct)
    mismatch = (v1 / "correct_mismatch.txt").read_text(encoding="ascii")
    assert {"files", "outputs", "target_files"} <= placeholders(mismatch)
    for hint in ("multiply, maximum", "Element types", "32x32", "No loops", "the reference gives", "Remember"):
        assert hint not in correct and hint not in mismatch, hint


def test_tt_d2m_bare_v1_is_the_ttir_set_for_d2m() -> None:
    d2m, ttir, guided = bare_v1_sets()["d2m"], bare_v1_sets()["ttir"], PROMPTS / "tt-d2m-v0"
    for name in ("correct", "correct_mismatch"):
        assert (d2m / f"{name}.txt").read_bytes() == (ttir / f"{name}.txt").read_bytes(), name
    for name in ("knowledge", "passage", "retrieved", "source_mlir", "summarize_context"):
        assert (d2m / f"{name}.txt").read_bytes() == (guided / f"{name}.txt").read_bytes(), name
    generate = (d2m / "generate.txt").read_text(encoding="ascii")
    bare = (PROMPTS / "tt-ttir-bare-v0" / "generate.txt").read_text(encoding="ascii")
    assert generate == bare.replace("one TTIR MLIR module", "one D2M MLIR module") and generate != bare


@pytest.mark.parametrize("target", ["ttir", "d2m"])
@pytest.mark.parametrize("template", sorted(FIELDS))
def test_each_template_is_ascii_uses_its_fields_and_names_no_item(target: str, template: str) -> None:
    folder = bare_v1_sets()[target]
    data = (folder / f"{template}.txt").read_bytes()
    assert data.isascii() and b"\r" not in data
    text = data.decode("ascii")
    assert placeholders(text) <= set(FIELDS[template])
    assert render(folder.name, template, dict.fromkeys(FIELDS[template], ""))
    for word in ITEMS + SHAPES:
        assert word not in text, f"{folder.name}/{template}.txt names {word}"


# ---------------------------------------------------------------------------
# The end-to-end checks


def test_the_check_directory_holds_exactly_the_four_checks() -> None:
    assert sorted(path.stem for path in CHECKS.glob("*.yaml")) == CHECK_NAMES


@pytest.mark.parametrize("name", CHECK_NAMES)
def test_each_check_replays_a_synthetic_recording_through_the_arms_components(name: str) -> None:
    data = load(CHECKS / f"{name}.yaml")
    target = data["directions"][0]["target"]
    assert data["model"]["backend"] == "replay" and data["trials"] == {"n": 1}
    recording = json.loads((REPO / data["model"]["recording"]).read_text(encoding="ascii"))
    assert recording["synthetic"] is True
    assert len(recording["completions"]) == data["loop"]["max_corrections"] + 1
    arm = load(RECIPES / f"{target}-min.yaml")
    for key in ("toolchain", "executor", "oracle", "fixes", "prompts"):
        assert data[key] == arm[key], key
    assert data["stages"] == [stage for stage in arm["stages"] if stage != "verbatim_context"]
    assert "simulator" in header(CHECKS / f"{name}.yaml")
