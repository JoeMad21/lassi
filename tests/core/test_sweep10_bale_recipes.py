"""Tests for bale-v0-tt, the bale-tt-v0 template set, and sweep10's bale recipes (the window of 2026-10-29).

Bible: Project Recipes (template sets; the loop's opt-ins), Benchmark Suites
(bale-v0, a reference language), Oracles (binary_io), Agent Rules 4 and 5
(every bale-v0-tt item unassigned).

What they fix (the owner, 2026-10-29: "I want at least the majority of the
BALE benchmark and I want C++/openMP to Tenstorrent"; all eight of bale's
classic apps):

- assets/bench/bale-v0-tt.yaml holds the eight apps, each unassigned with a
  seed of its own, inputs of at most 16384 elements, an exact tolerance,
  and the cpp, omp, and ttkh languages; ttkh is the host program and one
  kernel under kernels/, with the serial C++ program as its reference and
  lassi_io.h alone as support. The items bale-v0 also holds use bale-v0's
  programs and upstream files.
- assets/prompts/bale-tt-v0 is generate.txt, correct.txt,
  correct_mismatch.txt, example.txt, and examples.txt, plain ASCII, each
  using only its stage's fields; the three stage templates say what cpp,
  omp, and ttkh mean, and no template names bale or an item. Generate shows
  the source program and names the target files, and no prompt shows
  bale's upstream C file.
- projects/lassi-demo/sweep10/gpt-oss-120b/ holds bale-cpp-a, bale-cpp-b,
  bale-omp-a, and bale-omp-b, which load, extend bale-tt-base.yaml (sweep8's
  gpt-oss-120b settings, bale-v0-tt's unassigned split, bale-tt-v0, the
  reference built by gcc-native on the native executor, ttkh on ttsim, the
  tt-kh-examples-v0 examples, both references built first, binary_io
  max_abs 0, 2 samples, a correction cap of 8, the mismatch and
  simulator-gap feedback on), and between the two halves of each direction
  cover every item once.
- With stand-in fetched sources, the runner's checks before any directory
  pass for each (skipped when the generated examples pack is not built).

The stand-in sources are SYNTHETIC. No value here is a measurement.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
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
from lassi.toolchains.openmp_tag import NO_OPENMP_CODE, read_openmp_tags

REPO = Path(__file__).resolve().parents[2]
SET = "bale-tt-v0"
SET_DIR = REPO / "assets" / "prompts" / SET
SUITE = load_suite(REPO / "assets" / "bench" / "bale-v0-tt.yaml")
BALE_V0 = load_suite(REPO / "assets" / "bench" / "bale-v0.yaml")
RECIPES = REPO / "projects" / "lassi-demo" / "sweep10" / "gpt-oss-120b"
SWEEP8 = REPO / "projects" / "lassi-demo" / "sweep8" / "gpt-oss-120b" / "tc0-maxabs.yaml"
ITEMS = ("histo", "ig", "permute_matrix", "randperm", "sssp", "toposort", "transpose_matrix", "triangle")
NAMES = ("bale-cpp-a", "bale-cpp-b", "bale-omp-a", "bale-omp-b")
FIELDS = {"generate": GENERATE_FIELDS, "correct": CORRECT_FIELDS, MISMATCH_PROMPT: MISMATCH_FIELDS}
DIRECTIONS = {"cpp": Direction("cpp", "ttkh"), "omp": Direction("omp", "ttkh")}
# A file of the generated examples pack (gitignored; tools/build_tt_examples.py writes it), as sweep9's test marks it.
GENERATED = (REPO / "assets" / "examples" / "tt-kh-examples-v0" / "hello_world_compute_kernel" / "kernels" / "compute"
             / "void_compute_kernel.cpp")


def stand_in(path: str) -> str:
    """Return the SYNTHETIC text written in place of a fetched upstream file."""
    return f"/* SYNTHETIC stand-in for the fetched upstream file {path} */\n"


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a sources root with a SYNTHETIC stand-in at every upstream support path of bale-v0-tt."""
    root = tmp_path_factory.mktemp("bale-tt-sources")
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


# ---------------------------------------------------------------------------
# The suite


def test_the_suite_holds_the_eight_apps_at_bale_v0s_commit() -> None:
    assert (SUITE.name, SUITE.repo, SUITE.commit) == ("bale-v0-tt", BALE_V0.repo, BALE_V0.commit)
    assert tuple(sorted(SUITE.items)) == ITEMS


@pytest.mark.parametrize("item", ITEMS)
def test_each_item_is_unassigned_small_and_exact_with_three_languages(item: str) -> None:
    found = SUITE.item(item, purpose="eval")
    assert found.split == "unassigned"
    assert (found.tolerance.metric, found.tolerance.threshold) == ("max_abs", 0)
    assert all(math.prod(entry["shape"]) <= 16384 for entry in found.inputs)
    assert all(entry["dtype"] == "i64" for entry in found.inputs)
    assert set(found.languages) == {"cpp", "omp", "ttkh"}


def test_the_seeds_are_new() -> None:
    seeds = [SUITE.items[item].seed for item in ITEMS]
    assert len(set(seeds)) == len(ITEMS)
    assert not set(seeds) & {found.seed for found in BALE_V0.items.values()}


@pytest.mark.parametrize("item", ITEMS)
def test_ttkh_is_the_host_program_and_one_kernel_with_the_serial_program_as_reference(item: str) -> None:
    spec = SUITE.items[item].languages["ttkh"]
    assert list(spec.files) == [f"{item}.cpp", f"kernels/{item}_kernel.cpp"]
    assert spec.reference == "cpp"
    assert SUITE.reference_language(item, "ttkh") == "cpp"
    assert list(spec.support) == ["lassi_io.h"]


@pytest.mark.parametrize("item", ITEMS)
@pytest.mark.parametrize("language", ["cpp", "omp"])
def test_each_program_is_tracked_ascii_and_names_its_upstream_file(item: str, language: str) -> None:
    spec = SUITE.items[item].languages[language]
    assert spec.dir == f"assets/bench/bale-v0/{item}/{language}" and list(spec.files) == [f"{item}.cpp"]
    data = (REPO / spec.dir / f"{item}.cpp").read_bytes()
    assert data.isascii() and b"\r" not in data
    upstream = [entry for entry in spec.support.values() if not entry.tracked]
    assert len(upstream) == 1 and upstream[0].path.startswith("src/other_serial/C/")
    assert upstream[0].path.rsplit("/", 1)[1].encode("ascii") in data


def program(item: str, language: str) -> str:
    """Return the text of an item's program in cpp or omp."""
    return (REPO / "assets" / "bench" / "bale-v0" / item / language / f"{item}.cpp").read_text(encoding="ascii")


@pytest.mark.parametrize("item", ITEMS)
def test_the_serial_program_is_serial_and_the_omp_one_uses_openmp_with_the_same_arguments(item: str) -> None:
    serial, parallel = program(item, "cpp"), program(item, "omp")
    assert [tag.code for tag in read_openmp_tags({f"{item}.cpp": serial})] == [NO_OPENMP_CODE]
    assert "pragma" not in serial and not re.search(r"<thread>|\bstd::thread\b|\bpthread_|<omp\.h>|\bomp_", serial)
    assert read_openmp_tags({f"{item}.cpp": parallel}) == ()
    usage = [re.search(r'"usage: [^"]*"', text) for text in (serial, parallel)]
    assert usage[0] is not None and usage[1] is not None and usage[0].group() == usage[1].group()
    for text in (serial, parallel):
        joined = re.sub(r"\n// ", " ", text)
        assert "Copyright (c) 2020, Institute for Defense Analyses" in joined and "no bale text is copied" in joined


@pytest.mark.parametrize("item", sorted(BALE_V0.items))
def test_an_item_bale_v0_holds_uses_its_programs_and_upstream_files(item: str) -> None:
    for language in ("cpp", "omp"):
        ours, theirs = SUITE.items[item].languages[language], BALE_V0.items[item].languages[language]
        assert (ours.dir, list(ours.files)) == (theirs.dir, list(theirs.files))
        assert {key: entry.sha256 for key, entry in ours.support.items()} == {
            key: entry.sha256 for key, entry in theirs.support.items()}
    assert SUITE.items[item].outputs == BALE_V0.items[item].outputs


# ---------------------------------------------------------------------------
# The template set


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_each_template_is_ascii_uses_only_its_fields_and_explains_the_ids(name: str) -> None:
    data = (SET_DIR / f"{name}.txt").read_bytes()
    assert data.isascii() and b"\r" not in data
    text = data.decode("ascii")
    assert placeholders(text) <= set(FIELDS[name]), name
    assert render(SET, name, {field: "" for field in FIELDS[name]})
    for words in ("- cpp: ", "- omp: ", "- ttkh: ", "lassi_io.h", "never return it"):
        assert words in text, (name, words)
    assert ("kernel_main" in text) == (name != MISMATCH_PROMPT)
    lowered = text.lower()
    assert "bale" not in lowered and not any(re.search(rf"\b{item}\b", lowered) for item in ITEMS)


def test_the_set_holds_the_three_stage_templates_and_the_examples_frame() -> None:
    assert sorted(path.name for path in SET_DIR.iterdir()) == [
        "correct.txt", "correct_mismatch.txt", "example.txt", "examples.txt", "generate.txt"]
    for name in ("example.txt", "examples.txt"):
        assert (SET_DIR / name).read_bytes() == (REPO / "assets" / "prompts" / "tt-kh-v0" / name).read_bytes()


def context(item: str, direction: Direction, sources: Path) -> Any:
    """Return a stand-in RunContext for one item and direction under bale-tt-v0."""
    return SimpleNamespace(fragments={}, suite=SUITE, item=item, direction=direction, sources_root=sources,
                           prompts=SET)


@pytest.mark.parametrize("item", ITEMS)
@pytest.mark.parametrize("source", sorted(DIRECTIONS))
def test_generate_shows_the_source_and_never_the_upstream_file(item: str, source: str, sources: Path) -> None:
    direction = DIRECTIONS[source]
    expected = list(SUITE.items[item].languages["ttkh"].files)
    stage: Any = SimpleNamespace(context=context(item, direction, sources))
    system, prompt = GenerateStage._prompt(stage, None, expected)
    assert system is None
    shown = SUITE.source_files(item, direction, sources, purpose=PURPOSE)
    assert all(text in prompt for text in shown.values())
    assert f"Write the complete ttkh program as these files: {', '.join(expected)}." in prompt
    assert "SYNTHETIC stand-in" not in prompt, "bale's C file is support, never shown"


@pytest.mark.parametrize("item", ITEMS)
def test_the_corrections_never_show_the_upstream_file(item: str, sources: Path) -> None:
    expected = list(SUITE.items[item].languages["ttkh"].files)
    previous: Any = SimpleNamespace(files={name: "// SYNTHETIC previous program\n" for name in expected})
    for mismatch in (False, True):
        _, prompt = _correction_messages(context(item, DIRECTIONS["omp"], sources), previous, expected,
                                         "SYNTHETIC error text", run_error=False, mismatch=mismatch)
        assert "SYNTHETIC error text" in prompt and "SYNTHETIC previous program" in prompt
        assert "SYNTHETIC stand-in" not in prompt


# ---------------------------------------------------------------------------
# The recipes


def load(name: str) -> dict[str, Any]:
    """Return the resolved data of one sweep10 bale recipe."""
    return load_recipe(RECIPES / f"{name}.yaml").data


@pytest.mark.parametrize("name", NAMES)
def test_each_recipe_binds_one_direction_with_the_shared_settings(name: str) -> None:
    data = load(name)
    sweep8 = load_recipe(SWEEP8).data
    assert data["model"] == sweep8["model"] and data["llm"] == sweep8["llm"] and data["extract"] == sweep8["extract"]
    source = name.split("-")[1]
    assert data["directions"] == [{"source": source, "target": "ttkh"}]
    assert (data["bench"]["suite"], data["bench"]["split"]) == ("bale-v0-tt", "unassigned")
    assert data["prompts"] == SET and data["faithful"] is False
    assert data["toolchain"] == {"cpp": "gcc-native", "omp": "gcc-openmp", "ttkh": "ttmetal-host"}
    assert data["executor"]["cpp"] == {"kind": "native"} and data["executor"]["ttkh"]["kind"] == "ttsim"
    assert data["examples"] == {"pack": "tt-kh-examples-v0", "max_chars": 32000, "prompts": ["generate", "correct"]}
    # cpp is the ttkh reference, so a cpp recipe builds it once; an omp one also checks OpenMP against serial
    assert data["fixes"]["baseline_both"] is (source == "omp")
    assert data["oracle"] == {"kind": "binary_io", "metric": "max_abs", "threshold": 0}
    assert data["stages"] == ["baseline", "generate", "compile_loop", "run_loop", "oracle"]
    assert data["trials"] == {"n": 2} and data["sandbox"]["wall_s"] == 300
    loop = data["loop"]
    assert (loop["max_corrections"], loop["correct_on_mismatch"], loop["correct_on_sim_gap"]) == (8, True, True)


@pytest.mark.parametrize("source", sorted(DIRECTIONS))
def test_the_two_halves_of_a_direction_cover_every_item_once(source: str) -> None:
    items = load(f"bale-{source}-a")["bench"]["items"] + load(f"bale-{source}-b")["bench"]["items"]
    assert sorted(items) == list(ITEMS)


def checked(path: Path, sources: Path, *, both: bool | None = None, source: str | None = None) -> None:
    """Run the runner's checks before any directory on one recipe.

    `both` sets fixes.baseline_both and `source` the one direction's source
    language, when given.
    """
    options = runner_module.RunOptions(bench_root=sources)
    registry = runner_module.DEFAULT_REGISTRY
    recipe = runner_module._load(path, options, registry)
    if both is not None:
        recipe.data["fixes"]["baseline_both"] = both
    if source is not None:
        recipe.data["directions"] = [{"source": source, "target": "ttkh"}]
    settings = runner_module._settings(recipe)
    bench = runner_module._bench(recipe, settings, options)
    runner_module._check_plan(recipe, registry, settings, bench)
    runner_module._check_oracle(recipe, registry, bench)
    runner_module._check_trials(recipe, settings, bench)


@pytest.mark.parametrize("name", NAMES)
def test_the_runners_checks_before_any_directory_pass(name: str, sources: Path) -> None:
    if not GENERATED.is_file():
        pytest.skip("the tt-kh-examples-v0 pack is not built here; run tools/build_tt_examples.py")
    checked(RECIPES / f"{name}.yaml", sources)


def test_a_cpp_recipe_with_both_references_is_refused_as_its_own_reference(sources: Path) -> None:
    if not GENERATED.is_file():
        pytest.skip("the tt-kh-examples-v0 pack is not built here; run tools/build_tt_examples.py")
    with pytest.raises(runner_module.RunError, match="own source"):
        checked(RECIPES / "bale-cpp-a.yaml", sources, both=True)


def test_a_reference_that_is_the_source_passes_only_with_baseline_both_off(sources: Path) -> None:
    # The check recipe needs no generated pack, so this runs everywhere.
    path = REPO / "projects" / "lassi-demo" / "bale-tt-check.yaml"
    checked(path, sources, both=False, source="cpp")
    with pytest.raises(runner_module.RunError, match="own source"):
        checked(path, sources, both=True, source="cpp")


def test_the_check_recipe_runs_only_the_baseline_over_every_item(sources: Path) -> None:
    path = REPO / "projects" / "lassi-demo" / "bale-tt-check.yaml"
    data = load_recipe(path).data
    assert data["stages"] == ["baseline"] and data["directions"] == [{"source": "omp", "target": "ttkh"}]
    assert sorted(data["bench"]["items"]) == list(ITEMS) and data["fixes"]["baseline_both"] is True
    checked(path, sources)
