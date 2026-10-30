"""Tests for sweep13's compile-only cpp -> ttir and cpp -> d2m recipes and their template sets (tasks P17.48, P17.49).

Bible: Project Recipes (stages, executor kind none, template sets, the
context and retrieval sections), IR Strategy (TTIR and D2M above ttkernel),
Agent Rules 2 and 5 (nothing runs; tt-pairs-v0 is a development set,
OQ-083 (b); no item named in a template). plans/spikes/ttir-d2m.md (task
P17.46) holds the route and which items each level can express.

What they fix for projects/lassi-demo/sweep13/gpt-oss-120b/:

- Eight recipes, ttir-t02, -t05, -t07, -t10 and d2m-t02, -t05, -t07,
  -t10, each loading, each labelled compile only in its header: the stages
  verbatim_context, generate, and compile_loop (no baseline, run_loop, or
  oracle), executor kind none, the language's compile-only toolchain, and
  toolchain.tt bound for the suite's tt reference, which nothing builds.
- Every recipe carries the API reference pack, retrieval over the level's
  tt-mlir tests at generate and correct, and source_mlir on cgeist-affine
  (the owner's direction of 2026-10-29: every feature that applies).
- ttir covers eltwise_binary, eltwise_sfpu, and matmul_multi_core; d2m
  adds loopback; matmul_single_core is in neither.
- gpt-oss-120b as sweep12's arms set it (the model of sweep10's mlir
  recipes, reasoning effort high, 65536 tokens, top_p 1.0, 2 samples, a
  cap of 8, restart_after 3, restart_on_repeat); the four arms of a
  language differ in the temperature alone (0.2, 0.5, 0.7, 1.0).
- tt-ttir-v0 and tt-d2m-v0 hold the templates the stages render, plain
  ASCII, each using only its fields and naming no item; generate shows the
  item's declared arrays ($input_shapes) and the lowered compute unit
  ($source_mlir).
- The suite job builds every asset the recipes need and fetches their
  suites; with stand-in fetched kernels, the runner's checks before any
  directory pass (skipped when a gitignored reference pack is not built).
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from string import Template
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from lassi.bench import load_suite
from lassi.core import runner as runner_module
from lassi.core.recipe import load_recipe
from lassi.core.stages import (
    CORRECT_FIELDS,
    GENERATE_FIELDS,
    KNOWLEDGE_FIELDS,
    PASSAGE_FIELDS,
    RETRIEVED_FIELDS,
    SOURCE_MLIR_FIELDS,
    SUMMARY_FIELDS,
    input_shapes,
)
from lassi.prompts import render

REPO = Path(__file__).resolve().parents[2]
RECIPES = REPO / "projects" / "lassi-demo" / "sweep13" / "gpt-oss-120b"
SWEEP12 = REPO / "projects" / "lassi-demo" / "sweep12-t02" / "gpt-oss-120b" / "mlir-rag-t02-maxabs.yaml"
TEMPERATURES = {"t02": 0.2, "t05": 0.5, "t07": 0.7, "t10": 1.0}
NAMES = [f"{language}-{arm}" for language in ("ttir", "d2m") for arm in TEMPERATURES]
ITEMS = {
    "ttir": ["eltwise_binary", "eltwise_sfpu", "matmul_multi_core"],
    "d2m": ["eltwise_binary", "eltwise_sfpu", "loopback", "matmul_multi_core"],
}
TOOLCHAINS = {"ttir": "ttir-ttmetal-compile", "d2m": "d2m-ttmetal-compile"}
SETS = {"ttir": "tt-ttir-v0", "d2m": "tt-d2m-v0"}
PACKS = {"ttir": "ttir-ref-5f396fd6", "d2m": "d2m-ref-5f396fd6"}
INDEXES = {"ttir": "tt-mlir-ttir-tests", "d2m": "tt-mlir-d2m-tests"}
FIELDS = {"generate": GENERATE_FIELDS, "correct": CORRECT_FIELDS, "knowledge": KNOWLEDGE_FIELDS,
          "summarize_context": SUMMARY_FIELDS, "retrieved": RETRIEVED_FIELDS, "passage": PASSAGE_FIELDS,
          "source_mlir": SOURCE_MLIR_FIELDS}
ALL_ITEMS = ("eltwise_binary", "eltwise_sfpu", "loopback", "matmul_multi_core", "matmul_single_core")


def load(name: str) -> dict[str, Any]:
    """Return the resolved data of one sweep13 recipe."""
    return load_recipe(RECIPES / f"{name}.yaml").data


def language_of(name: str) -> str:
    """Return the target language a recipe name starts with."""
    return name.split("-")[0]


def placeholders(text: str) -> set[str]:
    """Return the `$name` placeholders of a template; fail on a `$` that is no placeholder."""
    names = set()
    for match in Template.pattern.finditer(text):
        assert match.group("invalid") is None
        name = match.group("named") or match.group("braced")
        if name:
            names.add(name)
    return names


def test_the_directory_holds_exactly_the_eight_recipes() -> None:
    assert sorted(path.stem for path in RECIPES.glob("*.yaml")) == sorted(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_each_recipe_is_compile_only_and_says_so(name: str) -> None:
    language, data = language_of(name), load(name)
    assert data["stages"] == ["verbatim_context", "generate", "compile_loop"]
    assert data["executor"] == {"kind": "none"} and "oracle" not in data
    assert data["toolchain"] == {language: TOOLCHAINS[language], "tt": "ttmetal-host"}
    assert data["directions"] == [{"source": "cpp", "target": language}]
    assert data["bench"] == {"suite": f"tt-pairs-v0-{language}", "split": "unassigned", "items": ITEMS[language]}
    assert data["prompts"] == SETS[language] and data["context"] == [PACKS[language]]
    assert data["retrieval"] == {"index": INDEXES[language], "k": 4, "max_chars": 8000,
                                 "prompts": ["generate", "correct"]}
    assert data["source_mlir"] == {"toolchain": "cgeist-affine", "max_chars": 24000}
    assert "dialect_lookup" not in data and "examples" not in data, "no TTIR or D2M tree or pack exists"
    assert data["faithful"] is False
    lines = (RECIPES / f"{name}.yaml").read_text(encoding="ascii").split("\nextends:")[0].split("\n")
    header = " ".join(line.lstrip("# ") for line in lines)
    assert "COMPILE ONLY" in header and "no output is checked" in header.lower()
    assert "exploratory" in header.lower()


@pytest.mark.parametrize("name", NAMES)
def test_each_recipe_has_sweep12s_model_settings_and_restarts(name: str) -> None:
    data, sweep12 = load(name), load_recipe(SWEEP12).data
    for key in ("model", "extract", "trials"):
        assert data[key] == sweep12[key], key
    for key in ("max_corrections", "restart_after", "restart_on_repeat"):
        assert data["loop"][key] == sweep12["loop"][key], key
    sampling = {key: value for key, value in data["llm"]["sampling"].items() if key != "temperature"}
    assert sampling == {key: value for key, value in sweep12["llm"]["sampling"].items() if key != "temperature"}
    assert sampling == {"top_p": 1.0, "reasoning_effort": "high", "max_tokens": 65536}
    assert data["llm"]["sampling"]["temperature"] == TEMPERATURES[name.split("-")[1]]
    assert data["model"]["id"] == "furiosa-ai/gpt-oss-120b" and data["trials"] == {"n": 2}
    assert data["loop"] == {"max_corrections": 8, "restart_after": 3, "restart_on_repeat": True}


@pytest.mark.parametrize("language", sorted(ITEMS))
def test_the_arms_differ_in_the_temperature_alone(language: str) -> None:
    def without_temperature(data: dict[str, Any]) -> dict[str, Any]:
        sampling = {key: value for key, value in data["llm"]["sampling"].items() if key != "temperature"}
        rest = {key: value for key, value in data.items() if key not in ("llm", "project")}
        return {**rest, "llm": {**data["llm"], "sampling": sampling}}

    arms = {arm: load(f"{language}-{arm}") for arm in TEMPERATURES}
    first = without_temperature(arms["t02"])
    for arm, data in arms.items():
        assert without_temperature(data) == first, arm
        assert data["project"] == f"{language}-{arm}", "each arm has its own trial ids"
    assert sorted(data["llm"]["sampling"]["temperature"] for data in arms.values()) == [0.2, 0.5, 0.7, 1.0]


def test_neither_level_runs_matmul_single_core_and_only_d2m_runs_loopback() -> None:
    assert "matmul_single_core" not in ITEMS["ttir"] + ITEMS["d2m"]
    assert "loopback" in ITEMS["d2m"] and "loopback" not in ITEMS["ttir"]


@pytest.mark.parametrize("language", sorted(SETS))
def test_each_set_holds_the_templates_the_stages_render(language: str) -> None:
    directory = REPO / "assets" / "prompts" / SETS[language]
    assert {path.stem for path in directory.glob("*.txt")} == set(FIELDS)


@pytest.mark.parametrize("language", sorted(SETS))
@pytest.mark.parametrize("template", sorted(FIELDS))
def test_each_template_is_ascii_uses_its_fields_and_names_no_item(language: str, template: str) -> None:
    data = (REPO / "assets" / "prompts" / SETS[language] / f"{template}.txt").read_bytes()
    assert data.isascii() and b"\r" not in data
    text = data.decode("ascii")
    assert placeholders(text) <= set(FIELDS[template])
    assert render(SETS[language], template, dict.fromkeys(FIELDS[template], ""))
    for item in ALL_ITEMS:
        assert item not in text, f"{SETS[language]}/{template}.txt names the item {item}"
    for shape in ("65536", "51200", "640", "256x256", "160x320"):
        assert shape not in text, f"{SETS[language]}/{template}.txt names an item's shape {shape}"


@pytest.mark.parametrize("language", sorted(SETS))
def test_generate_says_what_the_module_is_and_shows_the_declared_arrays(language: str) -> None:
    text = (REPO / "assets" / "prompts" / SETS[language] / "generate.txt").read_text(encoding="ascii")
    assert {"input_shapes", "source_files", "target_files", "source_mlir"} <= placeholders(text)
    assert "5f396fd6" in text and "--ttir-to-ttmetal-pipeline" in text and "--ttmetal-to-flatbuffer" in text
    assert "the compiler writes the device kernels" in text
    level = {"ttir": "TTIR", "d2m": "D2M"}[language]
    assert f"one {level} MLIR module" in text
    knowledge = (REPO / "assets" / "prompts" / SETS[language] / "knowledge.txt").read_text(encoding="ascii")
    assert f"of the {level} MLIR dialect at the pin" in knowledge


def test_input_shapes_lists_each_declared_input_and_output_without_values() -> None:
    suite = load_suite(REPO / "assets" / "bench" / "tt-pairs-v0-ttir.yaml")
    context: Any = SimpleNamespace(suite=suite, item="matmul_multi_core")
    assert input_shapes(context) == (
        "- input a: bf16, shape [640, 640]\n- input b: bf16, shape [640, 640]\n- output c\n")
    assert "input_shapes" in GENERATE_FIELDS


def suite_helper() -> ModuleType:
    """Import plans/runs/suite/recipes.py, the suite job's recipe helper, as a module."""
    path = REPO / "plans" / "runs" / "suite" / "recipes.py"
    spec = importlib.util.spec_from_file_location("suite_recipes_sweep13", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_suite_job_builds_every_asset_and_fetches_every_suite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    helper = suite_helper()
    manifest = tmp_path / "recipes.tsv"
    with open(manifest, "w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(helper.HEADER)
        for name in NAMES:
            recipe = load_recipe(RECIPES / f"{name}.yaml")
            writer.writerow(["sweep13", "gpt-oss-120b", name, "yes", recipe.recipe_hash, "furiosa-ai/gpt-oss-120b",
                             ",".join(helper.needs_of(recipe.data)), f"{RECIPES / name}.yaml"])
    assert helper.main(["needs", "--manifest", str(manifest)]) == 0
    needs = capsys.readouterr().out.split()
    assert needs == sorted([*(f"context:{pack}" for pack in PACKS.values()),
                            *(f"retrieval:{index}" for index in INDEXES.values())])
    script = (REPO / "plans" / "runs" / "suite" / "run-suite.sh").read_text(encoding="utf-8")
    assert "    context:ttir-ref-5f396fd6|context:d2m-ref-5f396fd6)\n" in script
    assert "    retrieval:tt-mlir-ttir-tests|retrieval:tt-mlir-d2m-tests)\n" in script
    assert "for suite in tt-pairs-v0-ttir tt-pairs-v0-d2m; do" in script
    assert 'uv run tools/fetch_bench.py "assets/bench/$suite.yaml"' in script


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a sources root with a SYNTHETIC stand-in at every upstream support path of both suites."""
    root = tmp_path_factory.mktemp("ttir-d2m-sources")
    for language in ITEMS:
        suite = load_suite(REPO / "assets" / "bench" / f"tt-pairs-v0-{language}.yaml")
        for found in suite.items.values():
            for spec in found.languages.values():
                for entry in spec.support.values():
                    if not entry.tracked:
                        (root / entry.path).parent.mkdir(parents=True, exist_ok=True)
                        (root / entry.path).write_bytes(f"// SYNTHETIC stand-in for {entry.path}\n".encode("ascii"))
    return root


@pytest.mark.parametrize("name", NAMES)
def test_the_runners_checks_before_any_directory_pass(name: str, sources: Path) -> None:
    language = language_of(name)
    built = REPO / "assets" / "context" / PACKS[language] / f"{language}_ref.{language}.txt"
    if not built.is_file():
        pytest.skip(f"the reference pack is not built here ({built}); run tools/build_ttkernel_reference.py "
                    f"build --dialect {language}")
    options = runner_module.RunOptions(bench_root=sources)
    registry = runner_module.DEFAULT_REGISTRY
    recipe = runner_module._load(RECIPES / f"{name}.yaml", options, registry)
    settings = runner_module._settings(recipe)
    assert settings.packs[language].startswith(f"// The {language} dialect of tt-mlir")
    bench = runner_module._bench(recipe, settings, options)
    assert list(bench.items) == ITEMS[language]
    runner_module._check_plan(recipe, registry, settings, bench)
    runner_module._check_oracle(recipe, registry, bench)
