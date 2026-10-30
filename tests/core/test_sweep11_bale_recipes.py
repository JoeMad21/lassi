"""Tests for sweep11, bale-v0-tt with the feature combinations, and the bale-tt-v1 template set (task P17.45).

Bible: Project Recipes (template sets; the context, lookup, retrieval, and
loop sections), Benchmark Suites (bale-v0-tt), Agent Rule 5 (every
bale-v0-tt item unassigned). The owner's standing direction of 2026-10-23:
test every combination of the features added to base LASSI.

What they fix:

- assets/prompts/bale-tt-v1 is bale-tt-v0's five files byte for byte (its
  wording unchanged: rewording it after sweep10's results would be prompt
  tuning on unassigned items), plus the frames the kernels pack, the
  lookup, and retrieval need, each byte for byte tt-kh-v0's: knowledge.txt
  and summarize_context.txt (the runner renders both when a recipe names a
  context pack), core_api.txt, declaration.txt, looked_up.txt, and
  undeclared.txt (the lookup), and retrieved.txt and passage.txt
  (retrieval). Every file is plain ASCII with LF ends, uses only its
  fields, and names neither bale nor an item.
- projects/lassi-demo/sweep11/gpt-oss-120b/ holds bale-{api,rag,all}-
  {cpp,omp}-{a,b} and nothing else. Each resolves to sweep10's recipe of
  its direction and half but for the template set (bale-tt-v1 for
  bale-tt-v0), the loop's restarts (restart_after 3 and restart_on_repeat
  true; task P17.50), and its features: api the kernels pack
  tt-metalium-5280a9cf-kh with the declaration lookup, rag retrieval over
  the tt-metal docs, all both, each section as sweep9's kh-allsk sets it.
  No recipe lists a stage that fills the knowledge block, so no stage
  sends the whole pack and summarize_context.txt and knowledge.txt never
  reach the model.
- With stand-in fetched sources, the runner's checks before any directory
  pass for each (skipped when a generated pack, gitignored, is not built
  here, as sweep9's test skips).
- plans/runs/suite/recipes.py lists for each recipe exactly the assets of
  its features, each one run-suite.sh has a build step for, and each
  recipe's resolved text names bale-v0-tt, so run-suite.sh fetches its
  sources.

The stand-in sources are SYNTHETIC. No value here is a measurement.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from string import Template
from types import ModuleType
from typing import Any

import pytest

from lassi.bench import load_suite
from lassi.core import runner as runner_module
from lassi.core.recipe import load_recipe, resolved_yaml
from lassi.core.stages import (
    CONTEXT_PROMPTS,
    CORE_TEMPLATES,
    CORRECT_FIELDS,
    EXAMPLE_TEMPLATES,
    GENERATE_FIELDS,
    LOOKUP_TEMPLATES,
    MISMATCH_FIELDS,
    MISMATCH_PROMPT,
    NEAREST_TEMPLATES,
    RETRIEVAL_PROMPTS,
)
from lassi.prompts import render

REPO = Path(__file__).resolve().parents[2]
SET, V0, KH = "bale-tt-v1", "bale-tt-v0", "tt-kh-v0"
PROMPTS = REPO / "assets" / "prompts"
SET_DIR = PROMPTS / SET
DEMO = REPO / "projects" / "lassi-demo"
RECIPES = DEMO / "sweep11" / "gpt-oss-120b"
SWEEP10 = DEMO / "sweep10" / "gpt-oss-120b"
KH_ALLSK = DEMO / "sweep9" / "gpt-oss-120b" / "kh-allsk-maxabs.yaml"
SCRIPT = REPO / "plans" / "runs" / "suite" / "run-suite.sh"
SUITE = load_suite(REPO / "assets" / "bench" / "bale-v0-tt.yaml")
ITEMS = ("histo", "ig", "permute_matrix", "randperm", "sssp", "toposort", "transpose_matrix", "triangle")
FEATURES = ("context", "lookup", "retrieval")
COMBOS = {"api": ("context", "lookup"), "rag": ("retrieval",), "all": ("context", "lookup", "retrieval")}
SOURCES = ("cpp", "omp")
HALVES = ("a", "b")
NAMES = [f"bale-{combo}-{source}-{half}" for combo in COMBOS for source in SOURCES for half in HALVES]
KH_PACK = "tt-metalium-5280a9cf-kh"
# Each feature's asset as recipes.py needs prints it; every recipe also carries sweep10's worked examples.
ASSETS = {"context": f"context:{KH_PACK}", "examples": "examples:tt-kh-examples-v0",
          "retrieval": "retrieval:tt-metal-docs"}
# The frames copied from tt-kh-v0, with their fields: the context pack's two, the lookup's four, retrieval's two.
FRAMES = {**CONTEXT_PROMPTS, **LOOKUP_TEMPLATES, **NEAREST_TEMPLATES, **CORE_TEMPLATES, **RETRIEVAL_PROMPTS}
# bale-tt-v0's own files, with their fields.
OWN = {"generate": GENERATE_FIELDS, "correct": CORRECT_FIELDS, MISMATCH_PROMPT: MISMATCH_FIELDS, **EXAMPLE_TEMPLATES}
# The stages that fill the knowledge block, whose request (summarize_context) or block (knowledge) a frame renders.
KNOWLEDGE_STAGES = {"summarize_context", "verbatim_context"}
# Files of the generated packs (gitignored; tools/build_tt_examples.py and build_tt_context.py --kernels write them).
GENERATED = (
    REPO / "assets" / "examples" / "tt-kh-examples-v0" / "hello_world_compute_kernel" / "kernels" / "compute"
    / "void_compute_kernel.cpp",
    REPO / "assets" / "context" / KH_PACK / "kernel_api.ttkh.txt",
)


def load(name: str) -> dict[str, Any]:
    """Return the resolved data of one sweep11 recipe."""
    return load_recipe(RECIPES / f"{name}.yaml").data


def sweep10(name: str) -> dict[str, Any]:
    """Return the resolved data of the sweep10 recipe of a sweep11 recipe's direction and half."""
    _, _, source, half = name.split("-")
    return load_recipe(SWEEP10 / f"bale-{source}-{half}.yaml").data


def without(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Return `data` less the named top-level keys."""
    return {key: value for key, value in data.items() if key not in keys}


def placeholders(text: str) -> set[str]:
    """Return the `$name` placeholders of a template; fail on a `$` that is no placeholder."""
    names = set()
    for match in Template.pattern.finditer(text):
        assert match.group("invalid") is None
        name = match.group("named") or match.group("braced")
        if name:
            names.add(name)
    return names


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a sources root with a SYNTHETIC stand-in at every upstream support path of bale-v0-tt."""
    root = tmp_path_factory.mktemp("bale-tt-v1-sources")
    for found in SUITE.items.values():
        for spec in found.languages.values():
            for entry in spec.support.values():
                if not entry.tracked:
                    (root / entry.path).parent.mkdir(parents=True, exist_ok=True)
                    text = f"/* SYNTHETIC stand-in for the fetched upstream file {entry.path} */\n"
                    (root / entry.path).write_bytes(text.encode("ascii"))
    return root


# ---------------------------------------------------------------------------
# The template set


def test_the_set_is_bale_tt_v0_and_the_feature_frames() -> None:
    v0 = {path.name for path in (PROMPTS / V0).iterdir()}
    assert v0 == {f"{name}.txt" for name in OWN}
    assert sorted(path.name for path in SET_DIR.iterdir()) == sorted(v0 | {f"{name}.txt" for name in FRAMES})
    assert len(FRAMES) == 8


@pytest.mark.parametrize("name", sorted(OWN))
def test_bale_tt_v0s_files_are_unchanged_byte_for_byte(name: str) -> None:
    assert (SET_DIR / f"{name}.txt").read_bytes() == (PROMPTS / V0 / f"{name}.txt").read_bytes(), (
        f"{name}.txt differs from bale-tt-v0's: rewording it after sweep10's results is prompt tuning (Agent Rule 5)")


@pytest.mark.parametrize("name", sorted(FRAMES))
def test_each_frame_is_tt_kh_v0s_byte_for_byte(name: str) -> None:
    assert (SET_DIR / f"{name}.txt").read_bytes() == (PROMPTS / KH / f"{name}.txt").read_bytes()


@pytest.mark.parametrize("name", sorted({**OWN, **FRAMES}))
def test_each_template_is_ascii_uses_only_its_fields_and_names_no_item(name: str) -> None:
    data = (SET_DIR / f"{name}.txt").read_bytes()
    assert data.isascii() and b"\r" not in data
    text = data.decode("ascii")
    fields = {**OWN, **FRAMES}[name]
    assert placeholders(text) <= set(fields), name
    assert render(SET, name, {field: f"<SYNTHETIC {field}>" for field in fields})
    lowered = text.lower()
    assert "bale" not in lowered and not any(re.search(rf"\b{item}\b", lowered) for item in ITEMS)


# ---------------------------------------------------------------------------
# The recipes


def test_sweep11_holds_the_twelve_recipes() -> None:
    assert sorted(path.stem for path in RECIPES.glob("*.yaml")) == sorted(NAMES)
    assert sorted(path.name for path in RECIPES.parent.iterdir()) == ["gpt-oss-120b"]


@pytest.mark.parametrize("name", NAMES)
def test_each_recipe_is_sweep10s_but_for_the_set_the_restarts_and_its_features(name: str) -> None:
    data, base, kh = load(name), sweep10(name), load_recipe(KH_ALLSK).data
    combo = name.split("-")[1]
    assert (data["prompts"], base["prompts"]) == (SET, V0)
    assert data["loop"] == {**base["loop"], "restart_after": 3, "restart_on_repeat": True}
    assert data["loop"]["restart_after"] == kh["loop"]["restart_after"]
    for feature in FEATURES:
        if feature in COMBOS[combo]:
            assert data[feature] == kh[feature], feature
        else:
            assert feature not in data, feature
    # Everything else is sweep10's: the model settings, the suite and split, the half's items, the direction, the
    # toolchains and executors, the examples, baseline_both, the stages, the oracle, the samples, and the cap.
    assert without(data, "project", "prompts", "loop", *FEATURES) == without(base, "project", "prompts", "loop")
    assert data["project"] == name


@pytest.mark.parametrize("name", NAMES)
def test_no_recipe_sends_the_whole_pack(name: str) -> None:
    # summarize_context.txt (copied from tt-kh-v0) describes tiles and three kernels; no stage here renders it.
    assert not KNOWLEDGE_STAGES & set(load(name)["stages"])


@pytest.mark.parametrize("half", HALVES)
@pytest.mark.parametrize("source", SOURCES)
def test_all_is_api_plus_rags_retrieval(source: str, half: str) -> None:
    api, rag, every = (load(f"bale-{combo}-{source}-{half}") for combo in COMBOS)
    assert without(every, "project") == {**without(api, "project"), "retrieval": rag["retrieval"]}
    assert without(rag, "project", "retrieval") == without(api, "project", "context", "lookup")


@pytest.mark.parametrize("combo", sorted(COMBOS))
@pytest.mark.parametrize("source", SOURCES)
def test_the_two_halves_of_a_direction_cover_every_item_once(combo: str, source: str) -> None:
    items = [item for half in HALVES for item in load(f"bale-{combo}-{source}-{half}")["bench"]["items"]]
    assert sorted(items) == list(ITEMS)


@pytest.mark.parametrize("name", NAMES)
def test_the_runners_checks_before_any_directory_pass(name: str, sources: Path) -> None:
    missing = [str(path) for path in GENERATED if not path.is_file()]
    if missing:
        pytest.skip(f"a generated pack is not built here ({missing[0]}); run tools/build_tt_examples.py and "
                    "tools/build_tt_context.py --kernels")
    options = runner_module.RunOptions(bench_root=sources)
    registry = runner_module.DEFAULT_REGISTRY
    recipe = runner_module._load(RECIPES / f"{name}.yaml", options, registry)
    settings = runner_module._settings(recipe)
    bench = runner_module._bench(recipe, settings, options)
    runner_module._check_plan(recipe, registry, settings, bench)
    assert runner_module._examples_pack(recipe)
    runner_module._check_oracle(recipe, registry, bench)


def test_the_retrieval_corpus_is_tracked() -> None:
    # The runner reads this manifest first; the index itself lies under $LASSI_SCRATCH, built by run-suite.sh.
    index = load("bale-rag-cpp-a")["retrieval"]["index"]
    assert (REPO / "assets" / "rag" / index / "MANIFEST.yaml").is_file()


# ---------------------------------------------------------------------------
# The suite job: the assets run-suite.sh builds and the sources it fetches


def suite_helper() -> ModuleType:
    """Return plans/runs/suite/recipes.py, the suite job's recipe helper."""
    name = "suite_recipes_sweep11"
    spec = importlib.util.spec_from_file_location(name, REPO / "plans" / "runs" / "suite" / "recipes.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_steps() -> set[str]:
    """Return every need run-suite.sh's build_asset has a case for."""
    script = SCRIPT.read_text(encoding="utf-8")
    start = script.index("build_asset() {")
    body = script[start:script.index("\n}\n", start)]
    return {need for label in re.findall(r"^    ([^\s()*]+)\)$", body, flags=re.M) for need in label.split("|")}


def expected_needs(name: str) -> list[str]:
    """Return the needs recipes.py lists for a sweep11 recipe, in its order: context, examples, retrieval."""
    combo = name.split("-")[1]
    return [ASSETS[key] for key in ("context", "examples", "retrieval") if key == "examples" or key in COMBOS[combo]]


@pytest.mark.parametrize("name", NAMES)
def test_recipes_py_needs_the_assets_of_its_features_and_run_suite_builds_each(name: str) -> None:
    found = suite_helper().needs_of(load(name))
    assert found == expected_needs(name)
    assert set(found) <= build_steps(), "run-suite.sh has a build step for each"


def test_the_manifest_lists_every_recipe_with_its_needs_and_bale_v0_tt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    module = suite_helper()
    monkeypatch.chdir(REPO)
    tsv, resolved = tmp_path / "recipes.tsv", tmp_path / "recipes"
    sweep = "projects/lassi-demo/sweep11"
    assert module.main(["manifest", "--out", str(tsv), "--resolved", str(resolved), "--sweep", sweep,
                        "--model", "gpt-oss-120b"]) == 0
    rows = {row["recipe"]: row for row in module.read_manifest(str(tsv))}
    assert sorted(rows) == sorted(NAMES)
    for name, row in rows.items():
        assert row["selected"] == "yes" and row["model_id"] == "furiosa-ai/gpt-oss-120b"
        assert row["needs"].split(",") == expected_needs(name)
        # run-suite.sh fetches bale-v0-tt's sources when a selected recipe's resolved text names the suite.
        text = (resolved / sweep.replace("/", "_") / "gpt-oss-120b" / f"{name}.yaml").read_text(encoding="ascii")
        assert "suite: bale-v0-tt" in text and text == resolved_yaml(load_recipe(RECIPES / f"{name}.yaml"))
    capsys.readouterr()
    assert module.main(["needs", "--manifest", str(tsv)]) == 0
    assert capsys.readouterr().out.split() == sorted(ASSETS.values())
    assert "grep -rqs 'suite: bale-v0-tt' \"$out/recipes\"" in SCRIPT.read_text(encoding="utf-8")
