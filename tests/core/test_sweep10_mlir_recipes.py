"""Tests for sweep10's cpp -> ttkernel recipes, the API reference and retrieval combos (task P17.44).

Bible: Project Recipes (the context, retrieval, examples, and loop
sections; template sets), Agent Rules 2 and 5 (ttsim is a simulator;
tt-pairs-v0 is a development set, OQ-083 (b)). In the window of 2026-10-27
(rx 20261027-005611-window-7998) no cpp -> ttkernel attempt built, and
the records of 2026-10-26 trace the wrong syntax to the reference card the
model wrote in summarize_context (plans/spikes/mlir-issues.md).

What they fix for projects/lassi-demo/sweep10/gpt-oss-120b/mlir-*.yaml (the
directory's bale-* recipes are another suite's and are tested elsewhere):

- Four recipes, mlir-ref and mlir-rag in the maxabs and pcc halves, each
  loading with the repository's recipe loader; each half covers
  tt-pairs-v0-ttkernel's items once by their declared metric.
- ref is mlir1's ttkernel-affine recipe of its half with the reference in
  place of the card: the prompt set tt-mlir-v1, the context pack
  ttkernel-ref-5f396fd6 (tools/build_ttkernel_reference.py), and the stage
  verbatim_context in place of summarize_context; every other setting is
  mlir1's (source_mlir on cgeist-affine, dialect_lookup, the loop opt-ins,
  a cap of 8), with 2 samples where mlir1 has 3 (the window of
  2026-10-29). rag is ref plus retrieval over tt-mlir-tests.
- tt-mlir-v1 is tt-mlir-v0 but for knowledge.txt (worded for a reference)
  and the two retrieval templates it adds; every template is plain ASCII
  and uses only its fields.
- The verbatim_context stage puts the pack, as it is, in the knowledge
  block before generate.txt and the corrections, asks no model, and
  records no request.
- The suite job builds every asset the recipes need (plans/runs/suite
  recipes.py needs and run-suite.sh's build steps), and a failed asset
  leaves out exactly the recipes that need it (recipes.py select).
- With stand-in fetched kernels, the runner's checks before any directory
  pass (skipped when a generated pack, gitignored, is not built here).
"""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import sys
from pathlib import Path
from string import Template
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from lassi.bench import Direction, load_suite
from lassi.core import record
from lassi.core import runner as runner_module
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import Sampling
from lassi.core.recipe import load_recipe
from lassi.core.record import BenchItem, ModelInfo, Provenance, Trial
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.core.stages import (
    CORRECT_FIELDS,
    GENERATE_FIELDS,
    MISMATCH_FIELDS,
    PURPOSE,
    VerbatimContextStage,
    generate_query,
    host_files,
    kernel_files,
    with_context_blocks,
)
from lassi.prompts import render

REPO = Path(__file__).resolve().parents[2]
RECIPES = REPO / "projects" / "lassi-demo" / "sweep10" / "gpt-oss-120b"
MLIR1 = REPO / "projects" / "lassi-demo" / "mlir1" / "gpt-oss-120b"
SUITE = load_suite(REPO / "assets" / "bench" / "tt-pairs-v0-ttkernel.yaml")
SET, BASE_SET = "tt-mlir-v1", "tt-mlir-v0"
SET_DIR, BASE_DIR = REPO / "assets" / "prompts" / SET, REPO / "assets" / "prompts" / BASE_SET
HALVES = {"maxabs": "max_abs", "pcc": "pcc"}
NAMES = [f"mlir-{combo}-{half}" for combo in ("ref", "rag") for half in HALVES]
REF_PACK = "ttkernel-ref-5f396fd6"
INDEX = "tt-mlir-tests"
STAGES = ["baseline", "verbatim_context", "generate", "compile_loop", "run_loop", "oracle"]
NEW_TEMPLATES = {"knowledge": ("knowledge_summary",), "retrieved": ("passages",),
                 "passage": ("path", "heading", "start", "end", "text")}
GENERATED = (
    REPO / "assets" / "context" / REF_PACK / "ttkernel_ref.ttkernel.txt",
    REPO / "assets" / "context" / "ttkernel-defs-5f396fd6" / "ttkernel_defs.ttkernel.txt",
    REPO / "assets" / "examples" / "ttkernel-examples-5f396fd6" / "ttkernel_translate" / "ttkernel_noc.mlir",
)


def load(name: str) -> dict[str, Any]:
    """Return the resolved data of one sweep10 mlir recipe."""
    return load_recipe(RECIPES / f"{name}.yaml").data


def suite_helper() -> ModuleType:
    """Import plans/runs/suite/recipes.py, the suite job's recipe helper, as a module."""
    path = REPO / "plans" / "runs" / "suite" / "recipes.py"
    spec = importlib.util.spec_from_file_location("suite_recipes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def placeholders(text: str) -> set[str]:
    """Return the `$name` placeholders of a template; fail on a `$` that is no placeholder."""
    names = set()
    for match in Template.pattern.finditer(text):
        assert match.group("invalid") is None
        name = match.group("named") or match.group("braced")
        if name:
            names.add(name)
    return names


@pytest.mark.parametrize("name", NAMES)
def test_each_recipe_loads_and_binds_cpp_to_ttkernel_on_ttsim(name: str) -> None:
    data = load(name)
    assert data["model"]["id"] == "furiosa-ai/gpt-oss-120b"
    assert data["directions"] == [{"source": "cpp", "target": "ttkernel"}]
    assert (data["bench"]["suite"], data["bench"]["split"]) == (SUITE.name, "unassigned")
    assert data["executor"]["ttkernel"]["kind"] == "ttsim" and data["executor"]["tt"]["kind"] == "ttsim"
    assert data["prompts"] == SET and data["context"] == [REF_PACK] and data["stages"] == STAGES
    assert "summarize_context" not in data["stages"], "the reference replaces the model-written card"
    assert data["source_mlir"] == {"toolchain": "cgeist-affine", "max_chars": 24000}
    assert data["dialect_lookup"] == {"defs": "ttkernel-defs-5f396fd6", "max_chars": 12000, "nearest": 4}
    assert data["loop"] == {"max_corrections": 8, "correct_on_mismatch": True, "correct_on_sim_gap": True,
                            "coord_gap_as_error": True}
    assert data["trials"] == {"n": 2}
    assert data["examples"]["pack"] == "ttkernel-examples-5f396fd6"


@pytest.mark.parametrize("half", sorted(HALVES))
def test_ref_is_mlir1s_affine_recipe_with_the_reference_in_place_of_the_card(half: str) -> None:
    ref, affine = load(f"mlir-ref-{half}"), load_recipe(MLIR1 / f"ttkernel-affine-{half}.yaml").data
    changed = {"prompts", "context", "stages", "project", "trials"}  # 2 samples where mlir1 has 3
    assert {key: value for key, value in ref.items() if key not in changed} == {
        key: value for key, value in affine.items() if key not in changed}
    assert affine["stages"] == [stage if stage != "verbatim_context" else "summarize_context" for stage in STAGES]
    assert "retrieval" not in ref


@pytest.mark.parametrize("half", sorted(HALVES))
def test_rag_is_ref_plus_retrieval_over_the_tt_mlir_tests(half: str) -> None:
    ref, rag = load(f"mlir-ref-{half}"), load(f"mlir-rag-{half}")
    assert rag["retrieval"] == {"index": INDEX, "k": 4, "max_chars": 8000, "prompts": ["generate", "correct"]}
    assert {key: value for key, value in rag.items() if key not in ("retrieval", "project")} == {
        key: value for key, value in ref.items() if key != "project"}


def test_the_mlir_recipes_cover_every_item_once_per_combo_by_its_metric() -> None:
    assert sorted(path.stem for path in RECIPES.glob("mlir-*.yaml")) == sorted(NAMES)
    for combo in ("ref", "rag"):
        items = []
        for half, metric in HALVES.items():
            data = load(f"mlir-{combo}-{half}")
            assert data["oracle"] == {"kind": "binary_io", "metric": metric, "threshold": "from_baseline"}
            assert all(SUITE.items[item].tolerance.metric == metric for item in data["bench"]["items"])
            items += data["bench"]["items"]
        assert sorted(items) == sorted(SUITE.items), combo


def test_tt_mlir_v1_is_v0_but_the_knowledge_and_retrieval_templates() -> None:
    v0, v1 = {p.name for p in BASE_DIR.glob("*.txt")}, {p.name for p in SET_DIR.glob("*.txt")}
    assert v1 == v0 | {"retrieved.txt", "passage.txt"}
    for name in v0 - {"knowledge.txt"}:
        assert (SET_DIR / name).read_bytes() == (BASE_DIR / name).read_bytes(), name
    knowledge = (SET_DIR / "knowledge.txt").read_text(encoding="ascii")
    assert "reference card" not in knowledge and "exactly in the form it gives" in knowledge
    assert "5f396fd6" in knowledge and "a name it does not list is not in the dialect" in knowledge


@pytest.mark.parametrize("name", sorted(NEW_TEMPLATES))
def test_each_new_template_is_ascii_and_uses_only_its_fields(name: str) -> None:
    data = (SET_DIR / f"{name}.txt").read_bytes()
    assert data.isascii() and b"\r" not in data
    assert placeholders(data.decode("ascii")) <= set(NEW_TEMPLATES[name])
    assert render(SET, name, dict.fromkeys(NEW_TEMPLATES[name], ""))


def trial() -> Trial:
    """Return a SYNTHETIC trial with an empty context, before any stage."""
    trial_id = record.make_trial_id("sweep10", "scripted-fixture", SUITE.name, "cpp-ttkernel", "loopback", 1)
    return Trial(
        trial_id=trial_id,
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(commit=None, dirty=None, device=None, sdk=None, date="2026-10-29T00:00:00+00:00"),
        bench_item=BenchItem(suite=SUITE.name, item="loopback", split="unassigned", direction="cpp-ttkernel"),
        model=ModelInfo(backend="scripted", id="scripted-fixture", sampling=Sampling(0.2, 0.9, 8)),
    )


def stage_context(**changes: Any) -> Any:
    """Return a stand-in RunContext for cpp -> ttkernel under tt-mlir-v1 with a SYNTHETIC reference pack."""
    fields = {"fragments": {}, "prompts": SET, "direction": Direction("cpp", "ttkernel"),
              "packs": {"ttkernel": "// SYNTHETIC reference\nttkernel.zebra(%a) : (i32) -> ()\n"}}
    return SimpleNamespace(**{**fields, **changes})


def test_verbatim_context_puts_the_pack_itself_in_the_knowledge_block_and_asks_no_model() -> None:
    context = stage_context()
    before = trial()
    after = VerbatimContextStage(context=context)(before)
    assert after.context.knowledge_summary == context.packs["ttkernel"]
    assert after.requests == before.requests and after.attempts == before.attempts, "no model is asked"
    assert dataclasses.replace(after, context=before.context) == before, "nothing else changes"
    prompt, notes = with_context_blocks(context, after, "SYNTHETIC generation prompt")
    knowledge = render(SET, "knowledge", {"knowledge_summary": context.packs["ttkernel"]}).rstrip("\n")
    assert prompt == f"{knowledge}\n\nSYNTHETIC generation prompt" and notes == []
    assert "verbatim_context" in VerbatimContextStage(context=context).describe()


def test_verbatim_context_declares_what_the_runner_checks() -> None:
    entry = DEFAULT_REGISTRY.get("Stage", "verbatim_context")
    factory = getattr(entry, "factory", entry)
    assert factory is VerbatimContextStage
    assert VerbatimContextStage.needs_context is True
    assert VerbatimContextStage.fills_context == ("knowledge_summary",)
    assert VerbatimContextStage.prompt_fields == {"knowledge": ("knowledge_summary",)}
    assert getattr(VerbatimContextStage, "fragment_keys", None) is None, "a fragment set refuses it"
    assert not VerbatimContextStage.requires, "it needs no LLM backend"
    for name, fields in (("generate", GENERATE_FIELDS), ("correct", CORRECT_FIELDS),
                         ("correct_mismatch", MISMATCH_FIELDS)):
        assert render(SET, name, dict.fromkeys(fields, ""))


def manifest_rows(tmp_path: Path) -> Path:
    """Return a suite manifest TSV (recipes.py's HEADER) of the four recipes, every one selected."""
    helper = suite_helper()
    path = tmp_path / "recipes.tsv"
    with open(path, "w", encoding="ascii", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(helper.HEADER)
        for name in NAMES:
            recipe = load_recipe(RECIPES / f"{name}.yaml")
            writer.writerow(["sweep10", "gpt-oss-120b", name, "yes", recipe.recipe_hash, "furiosa-ai/gpt-oss-120b",
                             ",".join(helper.needs_of(recipe.data)), f"{RECIPES / name}.yaml"])
    return path


def test_the_suite_job_builds_every_asset_the_recipes_need(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    helper = suite_helper()
    assert helper.main(["needs", "--manifest", str(manifest_rows(tmp_path))]) == 0
    needs = capsys.readouterr().out.split()
    assert needs == [f"context:{REF_PACK}", "examples:ttkernel-examples-5f396fd6", f"retrieval:{INDEX}"]
    script = (REPO / "plans" / "runs" / "suite" / "run-suite.sh").read_text(encoding="utf-8")
    assert f"    context:{REF_PACK})\n" in script and f"    retrieval:{INDEX})\n" in script
    assert "|examples:ttkernel-examples-5f396fd6)" in script, "the examples case also builds the definitions"
    assert '> "$out/ttkernel-ref.log"' in script and '> "$out/ttkernel-rag.log"' in script, "each has its own log"


@pytest.mark.parametrize(("failed", "left_out"), [
    (f"retrieval:{INDEX}", ["mlir-rag-maxabs", "mlir-rag-pcc"]),
    (f"context:{REF_PACK}", NAMES),
    ("examples:ttkernel-examples-5f396fd6", NAMES),
])
def test_a_failed_asset_leaves_out_exactly_the_recipes_that_need_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], failed: str, left_out: list[str]
) -> None:
    helper = suite_helper()
    args = ["select", "--manifest", str(manifest_rows(tmp_path)), "--sweep", "sweep10", "--model", "gpt-oss-120b"]
    assert helper.main([*args, "--failed", failed]) == 0
    count, pattern = capsys.readouterr().out.rstrip("\n").split("\t")
    assert int(count) == len(NAMES) - len(left_out)
    assert pattern == "^(" + "|".join(sorted(left_out)) + ")$"


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a sources root with a SYNTHETIC stand-in at every upstream support path of the suite."""
    root = tmp_path_factory.mktemp("ttkernel-sources")
    for found in SUITE.items.values():
        for spec in found.languages.values():
            for entry in spec.support.values():
                if not entry.tracked:
                    (root / entry.path).parent.mkdir(parents=True, exist_ok=True)
                    (root / entry.path).write_bytes(f"// SYNTHETIC stand-in for {entry.path}\n".encode("ascii"))
    return root


@pytest.mark.parametrize("item", sorted(SUITE.items))
def test_the_generate_query_holds_the_host_program_between_the_kernels_and_the_sources(
    item: str, sources: Path
) -> None:
    # The rag recipes retrieve by what the item needs; for cpp -> ttkernel the trusted host program names it.
    direction = Direction("cpp", "ttkernel")
    context: Any = SimpleNamespace(suite=SUITE, item=item, direction=direction, sources_root=sources)
    host = host_files(context)
    shown = render_file_blocks(SUITE.source_files(item, direction, sources, purpose=PURPOSE))
    assert host and shown, "every ttkernel item has a trusted host program and a source"
    query = generate_query(context)
    assert query == "\n".join(part for part in (kernel_files(context), host, shown) if part)
    assert query.index(host) < query.index(shown)


@pytest.mark.parametrize("name", NAMES)
def test_the_runners_checks_before_any_directory_pass(name: str, sources: Path) -> None:
    missing = [str(path) for path in GENERATED if not path.is_file()]
    if missing:
        pytest.skip(f"the generated packs are not built here ({missing[0]}); run tools/build_ttkernel_reference.py "
                    "and tools/build_ttkernel_context.py")
    options = runner_module.RunOptions(bench_root=sources)
    registry = runner_module.DEFAULT_REGISTRY
    recipe = runner_module._load(RECIPES / f"{name}.yaml", options, registry)
    settings = runner_module._settings(recipe)
    assert settings.packs["ttkernel"].startswith("// The ttkernel dialect of tt-mlir")
    bench = runner_module._bench(recipe, settings, options)
    runner_module._check_plan(recipe, registry, settings, bench)
    assert runner_module._examples_pack(recipe)
    runner_module._check_oracle(recipe, registry, bench)
