"""Tests for the tt-host-v0 prompt set, Tier A's C++ <-> TT template set.

tt-host-v0 is a template set like p0-smoke (no MANIFEST.yaml): generate.txt
and correct.txt under assets/prompts/tt-host-v0/, filled with string.Template
by lassi.prompts.render. One set serves both directions of a recipe, so the
templates carry fixed text for both language ids, cpp and tt. The render
tests fill the templates for every tt-pairs-v0 item in both directions
through the stages' own prompt builders, given stand-in contexts:
GenerateStage._prompt, with the item's tracked source files and the target
language's file names from assets/bench/tt-pairs-v0.yaml, and
_correction_messages, with the previous files and a compile error or a run
error (CompileLoopStage, RunLoopStage). The other tests read the template
text.

Both templates show the item's kernels (the owner's decision of
2026-10-06): the field kernel_files holds each support file of the
direction's two languages whose build path has a directory named kernels,
as FILE blocks in path order, under a short lead that says the harness
provides them at those paths, read-only, that a tt host program launches
them by those paths, and that a cpp program reproduces their math. A prompt
never shows lassi_io.h as a file. The kernels are fetched files
(tools/fetch_bench.py) that this machine does not hold, so the sources root
here holds a SYNTHETIC stand-in at each kernel's upstream path and no
tt-metal text. The diagnostics and the run's standard error are SYNTHETIC
too, and no value in this module is a measurement. The owner's two Tier A
run recipes, which use this set, are loaded (not run) to check that
together they judge every item once, by the metric it declares.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from string import Template
from types import SimpleNamespace
from typing import Any

import pytest

from lassi.bench import Direction
from lassi.bench.registry import load_suite
from lassi.core import runner as _runner  # noqa: F401  (registers every component a recipe may bind)
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import Limits, RunResult
from lassi.core.recipe import load_recipe
from lassi.core.record import Diagnostic
from lassi.core.stages import (
    CORRECT_FIELDS,
    GENERATE_FIELDS,
    PURPOSE,
    GenerateStage,
    _correction_messages,
    diagnostics_text,
    run_error_text,
)
from lassi.prompts import render

REPO = Path(__file__).resolve().parents[2]
SET = "tt-host-v0"
SET_DIR = REPO / "assets" / "prompts" / SET
SUITE = load_suite(REPO / "assets" / "bench" / "tt-pairs-v0.yaml")
DIRECTIONS = (Direction("cpp", "tt"), Direction("tt", "cpp"))
CASES = [(item, direction) for item in sorted(SUITE.items) for direction in DIRECTIONS]
IDS = [f"{item}-{direction.source}-to-{direction.target}" for item, direction in CASES]
FIELDS = {"generate": GENERATE_FIELDS, "correct": CORRECT_FIELDS}
# A short template is one a small model reads whole.
MAX_LINES = 60
# The harness header both languages build with; a prompt names it but never shows it as a file.
HEADER = "lassi_io.h"
HEADER_TEXT = (REPO / "assets" / "harness" / "c" / HEADER).read_bytes().decode("utf-8")
# A support file is a kernel when its build path has a directory of this name (the decision's rule).
KERNEL_DIR = "kernels"
FIELD = "kernel_files"
# The words the lead above $kernel_files holds (whitespace collapsed, lower case), and its most lines.
LEAD_WORDS = ("harness provides", "read-only", "path", "launch", "reproduce")
MAX_LEAD_LINES = 6
# The FILE-block answer rules both templates end with; the kernel field leaves them as they are.
ANSWER_RULES = (
    "Answer with one fenced code block per file. The first line inside each block must be\n"
    "// FILE: <path>\n"
    "with the file's relative path, and the rest of the block is the whole file.\n"
    "Return every file in full, not just the lines that change.\n"
)
# The owner's Tier A run recipes (projects/lassi-demo/), which use this set, and the metric each judges by.
RUN_RECIPES = {"tier-a-rngd-maxabs": "max_abs", "tier-a-rngd-pcc": "pcc"}


def stand_in_text(path: str) -> str:
    """Return the SYNTHETIC text written in place of the fetched file at upstream path `path`."""
    return f"// SYNTHETIC stand-in for the fetched file {path}\n"


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a sources root with a SYNTHETIC stand-in at every upstream support path of tt-pairs-v0."""
    root = tmp_path_factory.mktemp("tt-pairs-sources")
    for found in SUITE.items.values():
        paths = list(found.support.values())
        for spec in found.languages.values():
            paths += [entry.path for entry in spec.support.values() if not entry.tracked]
        for path in paths:
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_bytes(stand_in_text(path).encode("ascii"))
    return root


def template_text(name: str) -> str:
    """Return assets/prompts/tt-host-v0/<name>.txt after checking it is plain ASCII with LF newlines."""
    data = (SET_DIR / f"{name}.txt").read_bytes()
    assert data.isascii(), f"{name}.txt is not ASCII"
    assert b"\r" not in data, f"{name}.txt has a CR"
    return data.decode("ascii")


def placeholders(text: str) -> set[str]:
    """Return the names of the `$name` and `${name}` placeholders in a template; fail on a bad `$`."""
    names: set[str] = set()
    for match in Template.pattern.finditer(text):
        assert match.group("invalid") is None, f"a '$' that is no placeholder at offset {match.start()}"
        name = match.group("named") or match.group("braced")
        if name:
            names.add(name)
    return names


def kernel_lead(text: str) -> str:
    """Return the paragraph above the line `$kernel_files` of a template, whitespace collapsed, in lower case."""
    lines = text.splitlines()
    assert f"${FIELD}" in lines, f"the template shows no ${FIELD} on a line of its own"
    end = lines.index(f"${FIELD}")
    while end > 0 and not lines[end - 1].strip():
        end -= 1
    start = end
    while start > 0 and lines[start - 1].strip():
        start -= 1
    lead = lines[start:end]
    assert 0 < len(lead) <= MAX_LEAD_LINES, f"the lead above ${FIELD} is {len(lead)} lines: {lead}"
    return " ".join(" ".join(lead).split()).lower()


def is_kernel(path: str) -> bool:
    """Return True for a build path with a directory named kernels in it."""
    return KERNEL_DIR in PurePosixPath(path).parts[:-1]


def kernels(item: str, direction: Direction, sources: Path) -> dict[str, str]:
    """Return the kernel support files of both languages of `direction` (build path -> text)."""
    found: dict[str, str] = {}
    for language in (direction.source, direction.target):
        files = SUITE.support_files(item, sources, purpose=PURPOSE, language=language)
        found.update({path: text for path, text in files.items() if is_kernel(path)})
    return found


def expected_files(item: str, direction: Direction) -> list[str]:
    """Return the target language's file names of `item`, as lassi.core.stages.target_files reads them."""
    return list(SUITE.item(item, purpose=PURPOSE).languages[direction.target].files)


def generate_prompt(item: str, direction: Direction, sources: Path) -> str:
    """Return the prompt GenerateStage._prompt gives for `item` in `direction`, from a stand-in context."""
    context = SimpleNamespace(
        fragments={}, suite=SUITE, item=item, direction=direction, sources_root=sources, prompts=SET
    )
    stage: Any = SimpleNamespace(context=context)
    system, prompt = GenerateStage._prompt(stage, None, expected_files(item, direction))
    assert system is None
    return prompt


def compile_errors(item: str) -> str:
    """Return a compile error and a kernel JIT error as compile_loop gives them (diagnostics_text)."""
    return diagnostics_text(
        [
            Diagnostic(
                stage="compile",
                severity="error",
                code="SYNTHETIC-code",
                file=f"{item}.cpp",
                line=3,
                column=7,
                message="SYNTHETIC compile error",
            ),
            Diagnostic(stage="jit", severity="error", file="kernels/SYNTHETIC.cpp", message="SYNTHETIC jit error"),
        ]
    )


def run_errors(direction: Direction) -> str:
    """Return a failed run's error text as run_loop gives it for a template set: tt runs on ttsim, cpp natively."""
    run = RunResult(exit_code=134, hang=False, stdout="", stderr="SYNTHETIC standard error\n")
    limits = Limits(wall_s=30.0, memory_mb=1024, cpus=1)
    return run_error_text(run, limits, {}, simulator=direction.target == "tt")


def previous_files(item: str, direction: Direction, sources: Path) -> dict[str, str]:
    """Return the target reference's files, standing in for the last answer."""
    return SUITE.source_files(item, Direction(direction.target, direction.source), sources, purpose=PURPOSE)


def correct_prompt(item: str, direction: Direction, sources: Path, errors: str, *, run_error: bool) -> str:
    """Return the prompt _correction_messages gives when the last answer was the target reference."""
    context: Any = SimpleNamespace(
        fragments={}, suite=SUITE, item=item, direction=direction, sources_root=sources, prompts=SET
    )
    previous: Any = SimpleNamespace(files=previous_files(item, direction, sources))
    expected = expected_files(item, direction)
    system, prompt = _correction_messages(context, previous, expected, errors, run_error=run_error)
    assert system is None
    return prompt


def check_rendered(rendered: str, name: str, item: str, direction: Direction) -> None:
    """Check a filled template: plain ASCII, no placeholder left, and the target file names asked for."""
    assert rendered.isascii()
    for field in FIELDS[name]:
        assert f"${field}" not in rendered and f"${{{field}}}" not in rendered, field
    names = ", ".join(expected_files(item, direction))
    assert f"{direction.target} program as these files: {names}." in rendered
    assert f"// FILE: {item}.cpp" in rendered


def check_kernels(rendered: str, item: str, direction: Direction, sources: Path) -> None:
    """Check that `rendered` shows each kernel of the item once, as FILE blocks in path order, and never the header."""
    shown = kernels(item, direction, sources)
    assert shown, f"{item} declares no kernel, so this check would bite on nothing"
    for language in (direction.source, direction.target):
        assert HEADER in SUITE.item(item, purpose=PURPOSE).languages[language].support, language
    assert render_file_blocks(shown) in rendered, f"the prompt does not show {sorted(shown)} as FILE blocks"
    for path, text in shown.items():
        assert rendered.count(f"// FILE: {path}\n") == 1, path
        assert rendered.count(text) == 1, path
    assert f"// FILE: {HEADER}" not in rendered
    assert HEADER_TEXT not in rendered


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_template_uses_only_its_stage_fields(name: str) -> None:
    assert placeholders(template_text(name)) <= set(FIELDS[name])


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_template_is_short_and_asks_for_file_blocks(name: str) -> None:
    text = template_text(name)
    assert len(text.splitlines()) <= MAX_LINES
    assert "// FILE: <path>" in text
    assert text.endswith("\n")


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_template_keeps_the_file_block_answer_rules(name: str) -> None:
    assert template_text(name).endswith(ANSWER_RULES)


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_template_renders_with_every_field_empty(name: str) -> None:
    # The runner renders each declared template this way before any model call.
    assert render(SET, name, dict.fromkeys(FIELDS[name], "")).isascii()


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_template_shows_the_kernel_files(name: str) -> None:
    assert FIELD in placeholders(template_text(name)), f"{name}.txt does not show ${FIELD}"


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_kernel_lead_says_the_harness_provides_them_read_only_for_both_directions(name: str) -> None:
    # One template serves both directions: cpp -> tt launches the kernels by their paths, tt -> cpp reproduces them.
    lead = kernel_lead(template_text(name))
    for words in LEAD_WORDS:
        assert words in lead, f"{name}.txt: the lead above ${FIELD} lacks {words!r}: {lead!r}"


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_template_drops_the_line_that_could_not_name_the_kernels(name: str) -> None:
    # The kernel field replaces the line saying the harness provides the item's kernel files at fixed paths.
    assert "kernel files at fixed paths" not in " ".join(template_text(name).split())


def test_set_is_a_template_set() -> None:
    assert not (SET_DIR / "MANIFEST.yaml").exists()


def test_both_language_ids_are_explained() -> None:
    for name in FIELDS:
        text = template_text(name)
        assert "- cpp: " in text and "- tt: " in text, name
        assert "never by the host" in text, name
        assert "The harness provides lassi_io.h; never return it." in text, name


@pytest.mark.parametrize(("item", "direction"), CASES, ids=IDS)
def test_generate_renders_for_every_item_and_direction(item: str, direction: Direction, sources: Path) -> None:
    rendered = generate_prompt(item, direction, sources)
    check_rendered(rendered, "generate", item, direction)
    assert f"from {direction.source} to {direction.target}." in rendered
    source_files = SUITE.source_files(item, direction, sources, purpose=PURPOSE)
    assert render_file_blocks(source_files) in rendered


@pytest.mark.parametrize("errors", ["compile", "run"])
@pytest.mark.parametrize(("item", "direction"), CASES, ids=IDS)
def test_correct_renders_for_every_item_and_direction(
    item: str, direction: Direction, errors: str, sources: Path
) -> None:
    text = compile_errors(item) if errors == "compile" else run_errors(direction)
    rendered = correct_prompt(item, direction, sources, text, run_error=errors == "run")
    check_rendered(rendered, "correct", item, direction)
    assert text in rendered and render_file_blocks(previous_files(item, direction, sources)) in rendered


@pytest.mark.parametrize(("item", "direction"), CASES, ids=IDS)
def test_generate_shows_every_kernel_of_the_item_once(item: str, direction: Direction, sources: Path) -> None:
    check_kernels(generate_prompt(item, direction, sources), item, direction, sources)


@pytest.mark.parametrize("errors", ["compile", "run"])
@pytest.mark.parametrize(("item", "direction"), CASES, ids=IDS)
def test_correct_shows_every_kernel_of_the_item_once(
    item: str, direction: Direction, errors: str, sources: Path
) -> None:
    # A kernel JIT error names a kernel file, so a correction shows the kernels too.
    text = compile_errors(item) if errors == "compile" else run_errors(direction)
    rendered = correct_prompt(item, direction, sources, text, run_error=errors == "run")
    check_kernels(rendered, item, direction, sources)


def test_correct_does_not_say_the_build_failed_for_a_run_error() -> None:
    # run_loop sends a run error through the same correct.txt, so it never claims a build or compile failure.
    text = template_text("correct").lower()
    for claim in ("did not build", "failed to build", "build failed", "did not compile", "failed to compile"):
        assert claim not in text, claim


def run_recipe(name: str) -> dict[str, Any]:
    """Return the loaded data of the Tier A run recipe projects/lassi-demo/<name>.yaml."""
    return load_recipe(REPO / "projects" / "lassi-demo" / f"{name}.yaml").data


@pytest.mark.parametrize(("name", "metric"), sorted(RUN_RECIPES.items()))
def test_a_tier_a_run_recipe_uses_this_set_and_its_items_declared_metric(name: str, metric: str) -> None:
    data = run_recipe(name)
    assert (data["prompts"], data["bench"]["suite"], data["bench"]["split"]) == (SET, SUITE.name, "unassigned")
    assert data["oracle"] == {"kind": "binary_io", "metric": metric, "threshold": "from_baseline"}
    for item in data["bench"]["items"]:
        tolerance = SUITE.items[item].tolerance
        assert tolerance is not None and tolerance.metric == metric, item


def test_the_tier_a_run_recipes_cover_every_item_once() -> None:
    names = [item for name in RUN_RECIPES for item in run_recipe(name)["bench"]["items"]]
    assert sorted(names) == sorted(SUITE.items)
