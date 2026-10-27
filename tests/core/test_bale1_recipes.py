"""Tests for the cpu-omp-v0 template set and the bale1 recipes (task P17.40).

Bible: Project Recipes (template sets; the loop's opt-ins), Benchmark Suites
(bale-v0), Harness Contract (the OpenMP-construct tag), Oracles (binary_io),
Agent Rules 4 and 5 (every bale-v0 item unassigned).
Plan: plans/p17-portable.md, P17.40.

What they fix:

- assets/prompts/cpu-omp-v0 is a template set of generate.txt, correct.txt,
  and correct_mismatch.txt, plain ASCII, each using only its stage's
  fields; each says what the language ids cpp and omp mean and that the
  harness provides lassi_io.h. Generate shows the source program and names
  the target files, and no prompt shows bale's upstream C file (a .c
  support file is neither a kernel nor a host program) or names a bench
  item or bale.
- projects/lassi-demo/bale1/gpt-oss-120b/ holds exactly cpp-to-omp.yaml and
  omp-to-cpp.yaml. Each loads, carries sweep8's gpt-oss-120b model, llm, and
  extract settings, runs bale-v0's unassigned split in its one direction
  with cpu-omp-v0, binds cpp to gcc-native and omp to gcc-openmp on the
  native executor, builds both references first (fixes.baseline_both),
  judges with binary_io max_abs 0, takes 3 samples at a correction cap of 8
  with loop.correct_on_mismatch, and has no adversary stage. Their comments
  say that every bale-v0 item is unassigned, that the times are PROJECTED, and
  that the fetch comes first.
- With stand-in fetched sources, the runner's checks before any directory
  pass for both, and they are refused when the sources are not fetched.
- tools/suite.py plan accepts the directory for gpt-oss-120b.

The stand-in sources are SYNTHETIC. No value here is a measurement.
"""

from __future__ import annotations

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

REPO = Path(__file__).resolve().parents[2]
SET = "cpu-omp-v0"
SET_DIR = REPO / "assets" / "prompts" / SET
SUITE = load_suite(REPO / "assets" / "bench" / "bale-v0.yaml")
RECIPES = REPO / "projects" / "lassi-demo" / "bale1" / "gpt-oss-120b"
SWEEP8 = REPO / "projects" / "lassi-demo" / "sweep8" / "gpt-oss-120b" / "tc0-maxabs.yaml"
FIELDS = {"generate": GENERATE_FIELDS, "correct": CORRECT_FIELDS, MISMATCH_PROMPT: MISMATCH_FIELDS}
DIRECTIONS = {"cpp-to-omp": Direction("cpp", "omp"), "omp-to-cpp": Direction("omp", "cpp")}


def stand_in(path: str) -> str:
    """Return the SYNTHETIC text written in place of a fetched upstream file."""
    return f"/* SYNTHETIC stand-in for the fetched upstream file {path} */\n"


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a sources root with a SYNTHETIC stand-in at every upstream support path of bale-v0."""
    root = tmp_path_factory.mktemp("bale-sources")
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


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_each_template_is_ascii_uses_only_its_fields_and_explains_the_ids(name: str) -> None:
    data = (SET_DIR / f"{name}.txt").read_bytes()
    assert data.isascii() and b"\r" not in data
    text = data.decode("ascii")
    assert placeholders(text) <= set(FIELDS[name]), name
    assert render(SET, name, {field: "" for field in FIELDS[name]})
    for words in ("- cpp: ", "- omp: ", "-O3 -fopenmp", "OpenMP", "lassi_io.h", "never return it"):
        assert words in text, (name, words)
    lowered = text.lower()
    assert "bale" not in lowered and not any(item in lowered for item in ("histo", "triangle"))


def test_the_set_holds_only_the_three_templates() -> None:
    assert sorted(path.name for path in SET_DIR.iterdir()) == ["correct.txt", "correct_mismatch.txt", "generate.txt"]


def context(item: str, direction: Direction, sources: Path) -> Any:
    """Return a stand-in RunContext for one item and direction under cpu-omp-v0."""
    return SimpleNamespace(fragments={}, suite=SUITE, item=item, direction=direction, sources_root=sources,
                           prompts=SET)


@pytest.mark.parametrize("item", sorted(SUITE.items))
@pytest.mark.parametrize("name", sorted(DIRECTIONS))
def test_generate_shows_the_source_and_never_the_upstream_file(item: str, name: str, sources: Path) -> None:
    direction = DIRECTIONS[name]
    expected = list(SUITE.items[item].languages[direction.target].files)
    stage: Any = SimpleNamespace(context=context(item, direction, sources))
    system, prompt = GenerateStage._prompt(stage, None, expected)
    assert system is None
    source = SUITE.source_files(item, direction, sources, purpose=PURPOSE)
    assert all(text in prompt for text in source.values())
    assert f"Write the complete {direction.target} program as these files: {', '.join(expected)}." in prompt
    assert "SYNTHETIC stand-in" not in prompt, "bale's C file is support, never shown"
    rest = prompt.replace("".join(source.values()), "")
    assert "$" not in rest and "lassi_io_read(path" in rest


@pytest.mark.parametrize("item", sorted(SUITE.items))
def test_the_corrections_never_show_the_upstream_file(item: str, sources: Path) -> None:
    direction = DIRECTIONS["cpp-to-omp"]
    previous: Any = SimpleNamespace(files={f"{item}.cpp": "// SYNTHETIC previous program\n"})
    for mismatch in (False, True):
        _, prompt = _correction_messages(context(item, direction, sources), previous, [f"{item}.cpp"],
                                         "SYNTHETIC error text", run_error=False, mismatch=mismatch)
        assert "SYNTHETIC error text" in prompt and "SYNTHETIC previous program" in prompt
        assert "SYNTHETIC stand-in" not in prompt


def load(name: str) -> dict[str, Any]:
    """Return the resolved data of one bale1 recipe."""
    return load_recipe(RECIPES / f"{name}.yaml").data


def test_the_directory_holds_the_two_recipes() -> None:
    assert sorted(path.name for path in RECIPES.parent.rglob("*.yaml")) == ["cpp-to-omp.yaml", "omp-to-cpp.yaml"]


@pytest.mark.parametrize("name", sorted(DIRECTIONS))
def test_the_recipes_bind_one_direction_with_sweep8s_model_settings(name: str) -> None:
    data = load(name)
    sweep8 = load_recipe(SWEEP8).data
    assert data["model"] == sweep8["model"] and data["model"]["id"] == "furiosa-ai/gpt-oss-120b"
    assert data["llm"] == sweep8["llm"]
    assert data["extract"] == sweep8["extract"]
    direction = DIRECTIONS[name]
    assert data["directions"] == [{"source": direction.source, "target": direction.target}]
    assert data["bench"] == {"suite": "bale-v0", "split": "unassigned"}
    assert data["prompts"] == SET and data["faithful"] is False
    assert data["toolchain"] == {"cpp": "gcc-native", "omp": "gcc-openmp"}
    assert data["executor"] == {"kind": "native"}
    assert data["fixes"]["baseline_both"] is True
    assert data["oracle"] == {"kind": "binary_io", "metric": "max_abs", "threshold": 0}
    assert data["stages"] == ["baseline", "generate", "compile_loop", "run_loop", "oracle"]
    assert data["trials"] == {"n": 3}
    assert data["loop"]["max_corrections"] == 8 and data["loop"]["correct_on_mismatch"] is True


@pytest.mark.parametrize("name", sorted(DIRECTIONS))
def test_the_recipe_comments_say_unassigned_projected_and_fetch_first(name: str) -> None:
    text = (RECIPES / f"{name}.yaml").read_bytes().decode("ascii")
    assert "unassigned" in text and "PROJECTED" in text
    assert "uv run tools/fetch_bench.py assets/bench/bale-v0.yaml" in text


@pytest.mark.parametrize("name", sorted(DIRECTIONS))
def test_the_runners_checks_before_any_directory_pass(name: str, sources: Path) -> None:
    options = runner_module.RunOptions(bench_root=sources)
    registry = runner_module.DEFAULT_REGISTRY
    recipe = runner_module._load(RECIPES / f"{name}.yaml", options, registry)
    settings = runner_module._settings(recipe)
    bench = runner_module._bench(recipe, settings, options)
    runner_module._check_plan(recipe, registry, settings, bench)
    runner_module._check_oracle(recipe, registry, bench)
    runner_module._check_trials(recipe, settings, bench)


@pytest.mark.parametrize("name", sorted(DIRECTIONS))
def test_unfetched_sources_are_refused_before_any_directory(name: str, tmp_path: Path) -> None:
    empty = tmp_path / "unfetched"
    empty.mkdir()
    options = runner_module.RunOptions(bench_root=empty)
    recipe = runner_module._load(RECIPES / f"{name}.yaml", options, runner_module.DEFAULT_REGISTRY)
    settings = runner_module._settings(recipe)
    bench = runner_module._bench(recipe, settings, options)
    with pytest.raises(runner_module.RunError, match="fetch"):
        runner_module._check_trials(recipe, settings, bench)


def test_suite_plan_accepts_the_directory(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    sys.path.insert(0, str(REPO / "tools"))
    import suite

    monkeypatch.setattr(suite, "read_tree", lambda: suite.Tree(head="0" * 40, branch="HEAD", dirty=False, pushed=True))
    status = suite.main(["plan", "--sweep", "projects/lassi-demo/bale1", "--models", "gpt-oss-120b", "--hours", "4"])
    out = capsys.readouterr().out
    assert status == 0
    assert "gpt-oss-120b over projects/lassi-demo/bale1: 2 (cpp-to-omp, omp-to-cpp)" in out
    assert "--sweep projects/lassi-demo/bale1" in out
