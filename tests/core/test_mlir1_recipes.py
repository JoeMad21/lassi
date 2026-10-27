"""Tests for the tt-mlir-v0 template set and the cpp -> ttkernel recipes (tasks P17.36, P17.37, and P17.41).

Bible: Project Recipes (template sets; the context, examples, and loop
sections), Level Rules (TT host programs are never model output), Agent
Rule 5 (tt-pairs-v0 is a development set, OQ-083 (b); no upstream kernel
text in a prompt).

What they fix:

- assets/prompts/tt-mlir-v0 is a template set (generate.txt, correct.txt,
  correct_mismatch.txt, and the context and examples templates); each uses
  only its stage's fields. For every tt-pairs-v0-ttkernel item, generate
  shows the C++ source and the trusted host program (host_files, read-only),
  names the .mlir kernel files to write, and shows no kernel: the upstream
  kernel C++ would be the answer. source_mlir is empty unless the recipe
  sets source_mlir (tests/core/test_source_mlir.py). The corrections show
  the host program too: the ttkernel host program of task P17.37 for the
  four items with circular buffers, the tt one for loopback. The prompts say
  DRAM is addressed through its bank, never by a NOC coordinate, and name the
  mistakes of plans/spikes/mlir-issues.md (task P17.41).
- projects/lassi-demo/mlir1/gpt-oss-120b/ttkernel-{maxabs,pcc}.yaml load,
  bind cpp -> ttkernel with ttkernel-ttmetal on ttsim, carry sweep7's
  gpt-oss-120b model settings, 3 samples, a cap of 8, the mismatch,
  simulator-gap, and coord_to_tile loop opt-ins, and the dialect_lookup
  section, and cover each item once by its declared metric (task P17.37 under
  OQ-103 (a): every item again). ttkernel-affine-{maxabs,pcc}.yaml are the
  same plus the source_mlir section on cgeist-affine (task P17.41). With
  stand-in fetched kernels, the runner's checks before any directory pass for
  all four (skipped when the generated dialect pack, definitions, or worked
  example, gitignored, is not built in this checkout).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from string import Template
from types import SimpleNamespace
from typing import Any

import pytest

from lassi.bench import Direction, load_suite
from lassi.core import runner as runner_module
from lassi.core.recipe import load_recipe
from lassi.core.stages import (
    CORRECT_FIELDS,
    GENERATE_FIELDS,
    MISMATCH_FIELDS,
    MISMATCH_PROMPT,
    PURPOSE,
    GenerateStage,
    _correction_messages,
)
from lassi.prompts import render

REPO = Path(__file__).resolve().parents[2]
SET = "tt-mlir-v0"
SET_DIR = REPO / "assets" / "prompts" / SET
SUITE = load_suite(REPO / "assets" / "bench" / "tt-pairs-v0-ttkernel.yaml")
DIRECTION = Direction("cpp", "ttkernel")
RECIPES = REPO / "projects" / "lassi-demo" / "mlir1" / "gpt-oss-120b"
SWEEP7 = REPO / "projects" / "lassi-demo" / "sweep7" / "gpt-oss-120b"
FIELDS = {"generate": GENERATE_FIELDS, "correct": CORRECT_FIELDS, MISMATCH_PROMPT: MISMATCH_FIELDS}
CONTEXT_FIELDS = {
    "summarize_context": ("target_language", "context_pack"), "knowledge": ("knowledge_summary",),
    "examples": ("examples",), "example": ("name", "files"), "dialect_lookup": ("definitions",),
    "dialect_definition": ("name", "text"), "dialect_unknown": ("name", "candidates"),
    "source_mlir": ("path", "function", "level", "mlir"),
}
# The recipes, each with its sweep7 counterpart and the metric of its items.
NAMES = {
    "ttkernel-maxabs": ("base-maxabs", "max_abs"), "ttkernel-pcc": ("base-pcc", "pcc"),
    "ttkernel-affine-maxabs": ("base-maxabs", "max_abs"), "ttkernel-affine-pcc": ("base-pcc", "pcc"),
}
GENERATED = (
    REPO / "assets" / "context" / "ttkernel-5f396fd6" / "ttkernel_ops.ttkernel.txt",
    REPO / "assets" / "context" / "ttkernel-defs-5f396fd6" / "ttkernel_defs.ttkernel.txt",
    REPO / "assets" / "examples" / "ttkernel-examples-5f396fd6" / "ttkernel_translate" / "ttkernel_noc.mlir",
)


def host_text(item: str) -> tuple[str, str]:
    """Return the item's ttkernel host program as the prompt shows it: its build name and its text."""
    spec = SUITE.items[item].languages["ttkernel"]
    name = SUITE.items[item].languages["tt"].files[0]
    return name, (REPO / spec.support[name].path).read_text(encoding="utf-8")


def stand_in(path: str) -> str:
    """Return the SYNTHETIC text written in place of a fetched upstream file."""
    return f"// SYNTHETIC stand-in for the fetched kernel {path}\n"


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a sources root with a SYNTHETIC stand-in at every upstream support path of the suite."""
    root = tmp_path_factory.mktemp("ttkernel-sources")
    for found in SUITE.items.values():
        for spec in found.languages.values():
            for entry in spec.support.values():
                if not entry.tracked:
                    (root / entry.path).parent.mkdir(parents=True, exist_ok=True)
                    (root / entry.path).write_bytes(stand_in(entry.path).encode("ascii"))
    return root


def placeholders(text: str) -> set[str]:
    """Return the `$name` placeholders of a template; fail on a `$` that is no placeholder."""
    names = set()
    for match in Template.pattern.finditer(text):
        assert match.group("invalid") is None
        name = match.group("named") or match.group("braced")
        if name:
            names.add(name)
    return names


def context(item: str, sources: Path) -> Any:
    """Return a stand-in RunContext for one item of cpp -> ttkernel under tt-mlir-v0."""
    return SimpleNamespace(fragments={}, suite=SUITE, item=item, direction=DIRECTION, sources_root=sources,
                           prompts=SET)


@pytest.mark.parametrize("name", sorted({**FIELDS, **CONTEXT_FIELDS}))
def test_each_template_is_ascii_and_uses_only_its_fields(name: str) -> None:
    data = (SET_DIR / f"{name}.txt").read_bytes()
    assert data.isascii() and b"\r" not in data
    allowed = {**FIELDS, **CONTEXT_FIELDS}[name]
    assert placeholders(data.decode("ascii")) <= set(allowed), name
    assert render(SET, name, {field: "" for field in allowed})


@pytest.mark.parametrize("item", sorted(SUITE.items))
def test_generate_shows_the_source_and_the_host_program_and_no_kernel(item: str, sources: Path) -> None:
    expected = list(SUITE.items[item].languages["ttkernel"].files)
    stage: Any = SimpleNamespace(context=context(item, sources))
    system, prompt = GenerateStage._prompt(stage, None, expected)
    assert system is None
    name, host = host_text(item)
    source = SUITE.source_files(item, DIRECTION, sources, purpose=PURPOSE)
    assert host in prompt and f"// FILE: {name}" in prompt, "the trusted host program, read-only"
    assert all(text in prompt for text in source.values())
    assert ", ".join(expected) in prompt and all(path.endswith(".mlir") for path in expected)
    assert "SYNTHETIC stand-in for the fetched kernel" not in prompt, "never an upstream kernel"
    assert "lassi_io_read(" not in prompt.replace(host, "").replace("".join(source.values()), ""), (
        "lassi_io.h is never shown")
    assert "$" not in prompt.replace(host, "").replace("".join(source.values()), "")


@pytest.mark.parametrize("item", sorted(SUITE.items))
def test_the_corrections_show_the_host_program_and_no_kernel(item: str, sources: Path) -> None:
    expected = list(SUITE.items[item].languages["ttkernel"].files)
    previous: Any = SimpleNamespace(files={expected[0]: "// SYNTHETIC previous module\n"})
    _, host = host_text(item)
    for mismatch in (False, True):
        _, prompt = _correction_messages(context(item, sources), previous, expected,
                                         "SYNTHETIC error text", run_error=False, mismatch=mismatch)
        assert host in prompt and "SYNTHETIC error text" in prompt
        assert "SYNTHETIC stand-in for the fetched kernel" not in prompt


def test_the_generate_template_explains_ttkernel_and_the_lowering() -> None:
    text = (SET_DIR / "generate.txt").read_text(encoding="ascii")
    for words in ("#ttkernel.thread<noc>", "#ttkernel.thread<compute>", "--convert-ttkernel-to-emitc",
                  "--ttkernel-to-cpp", "void kernel_main()", "0.9.0.dev20260221", "5280a9cf", "read-only",
                  "with .mlir in place of .cpp", "$source_mlir"):
        assert words in text, words


def test_the_prompts_say_dram_is_addressed_by_bank_and_name_the_spikes_mistakes() -> None:
    # plans/spikes/mlir-issues.md, Half 1: the types and ops the dialect lacks, the custom form of TensorAccessorArgs
    # (TTKernelOps.cpp:265-269 at 5f396fd6: no trailing type), the literal index of get_compile_time_arg_val, and DRAM
    # read at the NOC coordinate (0, 0).
    generate = (SET_DIR / "generate.txt").read_text(encoding="ascii")
    for words in ("DRAM is addressed through its bank, never by a NOC coordinate", "get_noc_addr_from_bank_id",
                  "noc_async_read_tile", "tensor_accessor.get_noc_addr", "noc_async_read_barrier",
                  "!ttkernel.tensor_accessor_args", "ttkernel.InterleavedAddrGenFast", "no trailing type",
                  "ttkernel.TensorAccessorArgs(prev = %args)", "#0", "arith.index_cast %i : index to i32", "(0, 0)"):
        assert words in generate, words
    correct = (SET_DIR / "correct.txt").read_text(encoding="ascii")
    for words in ("through its bank, never by a NOC coordinate", "no trailing type", "get_noc_addr_from_bank_id"):
        assert words in correct, words
    assert "through its bank" in (SET_DIR / "examples.txt").read_text(encoding="ascii")


def load(name: str) -> dict[str, Any]:
    """Return the resolved data of one mlir1 recipe."""
    return load_recipe(RECIPES / f"{name}.yaml").data


@pytest.mark.parametrize("name", sorted(NAMES))
def test_the_recipes_bind_cpp_to_ttkernel_with_sweep7s_model_settings(name: str) -> None:
    data = load(name)
    base, metric = NAMES[name]
    sweep7 = load_recipe(SWEEP7 / f"{base}.yaml").data
    assert data["model"] == sweep7["model"]
    assert data["llm"] == sweep7["llm"]
    assert data["extract"] == sweep7["extract"]
    assert data["directions"] == [{"source": "cpp", "target": "ttkernel"}]
    assert (data["bench"]["suite"], data["bench"]["split"], data["prompts"]) == (SUITE.name, "unassigned", SET)
    assert data["toolchain"]["ttkernel"] == "ttkernel-ttmetal" and data["toolchain"]["tt"] == "ttmetal-host"
    assert data["executor"]["ttkernel"]["kind"] == "ttsim" and data["executor"]["tt"]["kind"] == "ttsim"
    assert data["trials"] == {"n": 3}
    assert data["loop"]["max_corrections"] == 8
    assert data["loop"]["correct_on_mismatch"] is True and data["loop"]["correct_on_sim_gap"] is True
    assert data["loop"]["coord_gap_as_error"] is True, "task P17.41: the coord_to_tile gap as an addressing error"
    assert data["dialect_lookup"] == {"defs": "ttkernel-defs-5f396fd6", "max_chars": 12000, "nearest": 4}
    lowering = {"toolchain": "cgeist-affine", "max_chars": 24000} if "affine" in name else None
    assert data.get("source_mlir") == lowering
    assert data["context"] == ["ttkernel-5f396fd6"]
    assert data["examples"]["pack"] == "ttkernel-examples-5f396fd6"
    assert data["oracle"] == {"kind": "binary_io", "metric": metric, "threshold": "from_baseline"}
    assert "summarize_context" in data["stages"] and "adversary" not in data["stages"]
    for item in data["bench"]["items"]:
        assert SUITE.items[item].tolerance.metric == metric, item


def test_the_recipes_cover_every_item_once_by_its_metric() -> None:
    # Task P17.37 under OQ-103 (a): with the ttkernel host programs passing the CB indices, the max_abs recipe holds
    # every item of its half again and a pcc recipe for the matmuls joins it.
    assert sorted(path.stem for path in RECIPES.glob("*.yaml")) == sorted(NAMES)
    for pair in (("ttkernel-maxabs", "ttkernel-pcc"), ("ttkernel-affine-maxabs", "ttkernel-affine-pcc")):
        items = [item for name in pair for item in load(name)["bench"]["items"]]
        assert sorted(items) == sorted(SUITE.items), pair


@pytest.mark.parametrize("half", ["maxabs", "pcc"])
def test_an_affine_recipe_is_its_base_plus_the_lowering(half: str) -> None:
    base, affine = load(f"ttkernel-{half}"), load(f"ttkernel-affine-{half}")
    assert {key: value for key, value in affine.items() if key not in ("source_mlir", "project")} == {
        key: value for key, value in base.items() if key != "project"}


@pytest.mark.parametrize("name", sorted(NAMES))
def test_the_runners_checks_before_any_directory_pass(name: str, sources: Path) -> None:
    missing = [str(path) for path in GENERATED if not path.is_file()]
    if missing:
        pytest.skip(f"the generated packs are not built here ({missing[0]}); run tools/build_ttkernel_context.py")
    options = runner_module.RunOptions(bench_root=sources)
    registry = runner_module.DEFAULT_REGISTRY
    recipe = runner_module._load(RECIPES / f"{name}.yaml", options, registry)
    settings = runner_module._settings(recipe)
    bench = runner_module._bench(recipe, settings, options)
    runner_module._check_plan(recipe, registry, settings, bench)
    assert runner_module._examples_pack(recipe) is not None
    runner_module._check_oracle(recipe, registry, bench)


def test_unfetched_reference_kernels_are_refused_before_any_directory(tmp_path: Path) -> None:
    empty = tmp_path / "unfetched"
    empty.mkdir()
    options = runner_module.RunOptions(bench_root=empty)
    recipe = runner_module._load(RECIPES / "ttkernel-maxabs.yaml", options, runner_module.DEFAULT_REGISTRY)
    settings = runner_module._settings(recipe)
    bench = runner_module._bench(recipe, settings, options)
    with pytest.raises(runner_module.RunError, match="fetch"):
        runner_module._check_trials(recipe, settings, bench)


def test_a_reference_that_is_the_directions_source_is_refused(
    tmp_path: Path, sources: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = REPO / "assets" / "bench" / "tt-pairs-v0-ttkernel.yaml"
    bench_dir = tmp_path / "bench"
    bench_dir.mkdir()
    text = manifest.read_text(encoding="utf-8").replace("        reference: tt\n", "        reference: cpp\n")
    (bench_dir / manifest.name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(runner_module, "BENCH_DIR", bench_dir)
    options = runner_module.RunOptions(bench_root=sources)
    recipe = runner_module._load(RECIPES / "ttkernel-maxabs.yaml", options, runner_module.DEFAULT_REGISTRY)
    settings = runner_module._settings(recipe)
    bench = runner_module._bench(recipe, settings, options)
    with pytest.raises(runner_module.RunError, match="own source"):
        runner_module._check_trials(recipe, settings, bench)


@pytest.mark.parametrize(("section", "key"), [("toolchain", "toolchain.tt"), ("executor", "executor.tt")])
def test_a_reference_language_without_a_toolchain_or_executor_is_refused(
    sources: Path, section: str, key: str
) -> None:
    options = runner_module.RunOptions(bench_root=sources)
    recipe = runner_module._load(RECIPES / "ttkernel-maxabs.yaml", options, runner_module.DEFAULT_REGISTRY)
    recipe.data[section].pop("tt")
    settings = runner_module._settings(recipe)
    bench = runner_module._bench(recipe, settings, options)
    with pytest.raises(runner_module.RunError, match=key.replace(".", r"\.")):
        runner_module._check_trials(recipe, settings, bench)
    assert PurePosixPath(SUITE.items["loopback"].languages["ttkernel"].files[0]).suffix == ".mlir"
