"""Tests for the opt-in leniency extract.unfenced_file_lines: FILE lines read without a fence.

Bible: Execution Backends, Harness Contract (the FILE-block format: one fenced
block per file whose first line is `// FILE: <relative path>`; a missing file
is a build error fed back to the model); Project Recipes (the key); Training
Module, Reward Function (the stage ladder, and df-v0's W, which counts only
compile- and jit-stage warnings); Design Principle 4 and Agent Rule 4 (a fix or
leniency is a named toggle, never upstream behavior under faithful: true).

The contract these tests fix:

- The recipe key extract.unfenced_file_lines is a bool. Absent, it means
  false, and the loader materializes nothing, so a recipe without it keeps its
  resolved recipe and recipe hash (the golden-hash tests in test_recipe.py,
  test_recipe_model_pin.py, test_recipe_serving.py, and
  test_single_executor_golden.py hold the pinned hashes). Set true, it is
  refused at load, naming the key, under faithful: true (it is not upstream
  behavior) and with fixes.fence_tag off (it reads this project's FILE-block
  format, which a reply has only with that fix on).
- lassi.core.files.parse_file_blocks(text, expected, unfenced_file_lines=True)
  reads a text that holds no fenced FILE block at all from its unfenced FILE
  lines: each line outside every fenced block that starts, after optional
  spaces or tabs, with `// FILE:` opens a file that runs to the line before
  the next such line or to the end of the text; prose before the first is
  ignored; paths are normalized and checked as for fenced blocks (bad-path,
  duplicate-file, missing-file). Each file read this way gets one parse-stage
  warning, code unfenced-file, at its FILE line. A text with a fenced FILE
  block, closed or unclosed, reads exactly as with the leniency off.
- With the key on, generate and every correction read replies that way
  (lassi.core.stages _reply_attempt), so an attempt whose only diagnostics are
  unfenced-file warnings is S1, is built, and reaches S4 on a good build.
  With the key off or absent, every reply reads as before.

The suite, replies, sources, and file names are SYNTHETIC; the fake toolchain
writes files and a PLACEHOLDER artifact and compiles nothing; the scripted
backend answers from a script. No value in this module is a measurement.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.bench import Direction, LanguageSources, Suite, SuiteItem
from lassi.core import files as file_blocks
from lassi.core import recipe as recipe_module
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.files import ParsedFiles, parse_file_blocks, render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Message, Sampling
from lassi.core.recipe import RecipeError, load_recipe
from lassi.core.record import Attempt, Diagnostic, ModelInfo, Provenance, Trial, make_trial_id
from lassi.core.registry import Registry
from lassi.core.stages import CompileLoopStage, GenerateStage, RunContext, diagnostic_line
from lassi.core.store import TextStore
from lassi.executors import NoneExecutor
from lassi.scoring.df_v0 import warning_count

KEY = "extract.unfenced_file_lines"
UNFENCED = "unfenced-file"
MODEL_ID = "scripted-fixture"
TEMPLATE_SET = "p0-smoke"
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
DIRECTION = Direction("omp", "cuda")
# A commit id for synthetic provenance and the synthetic suite's pin; not a commit of any repository.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"

# The SYNTHETIC suite: one eval item whose target (cuda) version has two files and whose source has one.
SUITE_NAME = "synthetic-unfenced"
ITEM = "pair"
SOURCE_DIR = "src/pair-omp"
SOURCE_FILE = "main.cpp"
MAIN, KERNEL = "main.cu", "kernel.cuh"
TARGET_FILES = (MAIN, KERNEL)
SUITE = Suite(
    name=SUITE_NAME,
    repo="https://example.invalid/synthetic-unfenced",
    commit=FAKE_COMMIT,
    items={
        ITEM: SuiteItem(
            name=ITEM,
            split="eval",
            languages={
                "omp": LanguageSources(dir=SOURCE_DIR, files=(SOURCE_FILE,)),
                "cuda": LanguageSources(dir="src/pair-cuda", files=TARGET_FILES),
            },
            run_args=("1",),
        )
    },
)

# SYNTHETIC texts; the fake toolchain refuses any file holding an `#error` line.
OMP_SOURCE = "int main() {\n#pragma omp target\n  { }\n  return 0;\n}\n"
MAIN_CODE = '#include "kernel.cuh"\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n'
KERNEL_CODE = "__global__ void k() { }\n"
LOOPBACK = '#include "lassi_io.h"\nint main(int argc, char** argv) {\n  return 0;\n}\n'

# SYNTHETIC replies in the form the sweep's replies took: the FILE line and the whole file, with no fence.
ONE_UNFENCED = f"// FILE: loopback.cpp\n{LOOPBACK}"
PROSE_FIRST = f"Here is the translated program.\n\nIt keeps the output unchanged.\n// FILE: loopback.cpp\n{LOOPBACK}"
TWO_UNFENCED = f"// FILE: {MAIN}\n{MAIN_CODE}\n// FILE: {KERNEL}\n{KERNEL_CODE}"
PROSE_ONLY = "SYNTHETIC reply with prose and no FILE line.\n"
BOTH = {MAIN: MAIN_CODE, KERNEL: KERNEL_CODE}
BOTH_FENCED = render_file_blocks(BOTH)

PARSED, NO_OUTPUT, COMPILED = "S1", "S0", "S4"

# (stage, severity, code, file, line, column)
Where = tuple[str, str, str | None, str | None, int | None, int | None]


def where(diagnostic: Diagnostic) -> Where:
    """Return every Diagnostic field except the message."""
    d = diagnostic
    return (d.stage, d.severity, d.code, d.file, d.line, d.column)


def problem(code: str, file: str | None, line: int | None) -> Where:
    """Return what where() gives for a FILE-block problem: a compile-stage error with no column."""
    return ("compile", "error", code, file, line, None)


def unfenced(file: str | None, line: int) -> Where:
    """Return what where() gives for a file read from an unfenced FILE line: a parse-stage warning."""
    return ("parse", "warning", UNFENCED, file, line, None)


def lenient(text: str, expected: Sequence[str] = ()) -> ParsedFiles:
    """Parse `text` with the leniency on."""
    return parse_file_blocks(text, expected, unfenced_file_lines=True)


def outcome(parsed: ParsedFiles) -> tuple[dict[str, str], list[Where]]:
    """Return the files and the where() of each diagnostic, checking that every message says something."""
    for diagnostic in parsed.diagnostics:
        assert isinstance(diagnostic, Diagnostic)
        assert diagnostic.message.strip() and diagnostic.message.isascii(), diagnostic
    return parsed.files, [where(diagnostic) for diagnostic in parsed.diagnostics]


# ---------------------------------------------------------------------------
# The parser: off is today's reading


@pytest.mark.parametrize("text", [ONE_UNFENCED, PROSE_FIRST, TWO_UNFENCED], ids=["one", "prose-first", "two"])
def test_off_by_default_an_unfenced_reply_gives_todays_result(text: str) -> None:
    expected = ["loopback.cpp"]
    default = parse_file_blocks(text, expected)
    assert parse_file_blocks(text, expected, unfenced_file_lines=False) == default
    assert outcome(default) == ({}, [problem("missing-file", "loopback.cpp", None)]), (
        "with the leniency off an unfenced FILE line opens no file, as before"
    )


# ---------------------------------------------------------------------------
# The parser: on reads unfenced FILE lines when no fenced FILE block exists


def test_on_reads_an_unfenced_single_file_reply() -> None:
    assert outcome(lenient(ONE_UNFENCED, ["loopback.cpp"])) == (
        {"loopback.cpp": LOOPBACK},
        [unfenced("loopback.cpp", 1)],
    )


def test_on_reads_a_two_file_unfenced_reply_in_text_order() -> None:
    # Each file runs to the line before the next FILE line, so the blank line before it stays in the first file.
    assert outcome(lenient(TWO_UNFENCED, TARGET_FILES)) == (
        {MAIN: MAIN_CODE + "\n", KERNEL: KERNEL_CODE},
        [unfenced(MAIN, 1), unfenced(KERNEL, 8)],
    )


def test_on_ignores_prose_before_the_first_file_line() -> None:
    assert outcome(lenient(PROSE_FIRST, ["loopback.cpp"])) == (
        {"loopback.cpp": LOOPBACK},
        [unfenced("loopback.cpp", 4)],
    )


@pytest.mark.parametrize("indent", [" ", "\t", "  \t "], ids=["space", "tab", "mixed"])
def test_on_a_file_line_may_follow_spaces_or_tabs(indent: str) -> None:
    assert lenient(f"{indent}// FILE: a.cpp\nint a;\n").files == {"a.cpp": "int a;\n"}


def test_on_the_last_file_ends_with_one_line_feed_whatever_the_text_ends_with() -> None:
    assert lenient("// FILE: a.cpp\nint a;").files == {"a.cpp": "int a;\n"}
    assert lenient("// FILE: a.cpp\nint a;\n").files == {"a.cpp": "int a;\n"}
    assert lenient("// FILE: a.cpp\nint a;\n\n").files == {"a.cpp": "int a;\n\n"}
    assert lenient("// FILE: a.cpp\n").files == {"a.cpp": "\n"}
    assert lenient("// FILE: a.cpp").files == {"a.cpp": "\n"}


def test_on_a_file_line_inside_a_fenced_block_opens_no_file() -> None:
    # The fenced block's first line is no FILE line, so the text holds no fenced FILE block; the FILE line inside it
    # is not outside every fenced block, so only b.cpp is read, and the fence before it is prose.
    text = "Example:\n```text\nsee below\n// FILE: a.cpp\n```\n// FILE: b.cpp\nint b;\n"
    assert outcome(lenient(text)) == ({"b.cpp": "int b;\n"}, [unfenced("b.cpp", 6)])


def test_on_a_fenced_block_inside_a_file_is_part_of_its_text() -> None:
    # The fence's first line is no FILE line, so it is no FILE block, and the FILE line inside it opens no file.
    text = "// FILE: a.cpp\nint a;\n```\nnote\n// FILE: b.cpp\n```\nint c;\n"
    assert outcome(lenient(text)) == (
        {"a.cpp": "int a;\n```\nnote\n// FILE: b.cpp\n```\nint c;\n"},
        [unfenced("a.cpp", 1)],
    )


def test_on_a_fence_whose_first_line_is_a_file_line_is_a_file_block_so_it_reads_as_off() -> None:
    text = "// FILE: a.cpp\nint a;\n```\n// FILE: b.cpp\n```\nint c;\n"
    assert lenient(text) == parse_file_blocks(text) and lenient(text).files == {"b.cpp": "\n"}


def test_on_a_text_with_no_file_line_reads_as_off() -> None:
    for text in (PROSE_ONLY, "", f"```cuda\n{MAIN_CODE}```\n"):
        assert lenient(text, TARGET_FILES) == parse_file_blocks(text, TARGET_FILES)


# Texts that hold a fenced FILE block, closed or unclosed, so the leniency reads each exactly as off.
FENCED_TEXTS = {
    "two-blocks": BOTH_FENCED,
    "one-block-and-an-unfenced-file-line": render_file_blocks({MAIN: MAIN_CODE}) + f"// FILE: {KERNEL}\n{KERNEL_CODE}",
    "unfenced-file-line-then-a-block": f"// FILE: {KERNEL}\n{KERNEL_CODE}\n" + render_file_blocks({MAIN: MAIN_CODE}),
    "an-unclosed-block": f"```cuda\n// FILE: {MAIN}\n{MAIN_CODE}",
    "a-bad-path-block": f"```cuda\n// FILE: /abs/{MAIN}\n{MAIN_CODE}```\n// FILE: {KERNEL}\n{KERNEL_CODE}",
    "a-duplicate-block": BOTH_FENCED + "\n" + render_file_blocks({MAIN: MAIN_CODE}),
}


@pytest.mark.parametrize("text", list(FENCED_TEXTS.values()), ids=list(FENCED_TEXTS))
def test_on_a_text_with_a_fenced_file_block_reads_exactly_as_off(text: str) -> None:
    on = lenient(text, TARGET_FILES)
    assert on == parse_file_blocks(text, TARGET_FILES)
    assert all(item.code != UNFENCED for item in on.diagnostics)


# ---------------------------------------------------------------------------
# The parser: the same path normalization and checks


def test_on_a_path_is_normalized_and_the_warning_names_it_as_written() -> None:
    assert outcome(lenient("// FILE: ./src//a.cpp\nint a;\n", ["src/a.cpp"])) == (
        {"src/a.cpp": "int a;\n"},
        [unfenced("./src//a.cpp", 1)],
    )


@pytest.mark.parametrize("path", ["/etc/passwd", "../main.cu", "src\\main.cu", "-o.cu", "@rsp.cpp", "C:/main.cu"])
def test_on_a_refused_path_drops_its_file(path: str) -> None:
    text = f"// FILE: {path}\nint x;\n// FILE: ok.cu\nint ok;\n"
    assert outcome(lenient(text)) == ({"ok.cu": "int ok;\n"}, [problem("bad-path", path, 1), unfenced("ok.cu", 3)])


def test_on_an_empty_path_is_refused_without_a_file() -> None:
    assert outcome(lenient("// FILE:  \nint x;\n")) == ({}, [problem("bad-path", None, 1)])


def test_on_a_bad_path_message_names_the_line_the_path_and_the_fix() -> None:
    (diagnostic,) = lenient("// FILE: ../main.cu\nint x;\n").diagnostics
    assert "line 1" in diagnostic.message and "'../main.cu'" in diagnostic.message, diagnostic.message
    assert "relative path" in diagnostic.message, diagnostic.message


def test_on_a_second_file_line_for_the_same_path_is_a_duplicate_and_the_first_is_kept() -> None:
    text = "// FILE: main.cu\nint first;\n// FILE: ./main.cu\nint second;\n"
    assert outcome(lenient(text, ["main.cu"])) == (
        {"main.cu": "int first;\n"},
        [unfenced("main.cu", 1), problem("duplicate-file", "./main.cu", 3)],
    )
    duplicate = lenient(text).diagnostics[1].message
    assert "line 3" in duplicate and "line 1" in duplicate and "exactly one block" in duplicate, duplicate


def test_on_an_expected_file_with_no_file_line_is_missing() -> None:
    assert outcome(lenient(f"// FILE: {MAIN}\n{MAIN_CODE}", TARGET_FILES)) == (
        {MAIN: MAIN_CODE},
        [unfenced(MAIN, 1), problem("missing-file", KERNEL, None)],
    )


def test_on_an_expected_file_whose_path_is_refused_is_missing() -> None:
    assert outcome(lenient("// FILE: /main.cu\nint x;\n", ["main.cu"])) == (
        {},
        [problem("bad-path", "/main.cu", 1), problem("missing-file", "main.cu", None)],
    )


def test_the_warning_says_the_file_was_read_without_a_fence_and_what_the_format_asks() -> None:
    (warning,) = lenient(ONE_UNFENCED).diagnostics
    assert file_blocks.UNFENCED_FILE == UNFENCED
    message = warning.message
    assert "'loopback.cpp'" in message and "without a fence" in message, message
    assert "one fenced code block per file" in message and "// FILE: loopback.cpp" in message, message


# ---------------------------------------------------------------------------
# The stages: generate and every correction read replies with the leniency when it is on


@dataclass
class Log:
    """The replies the scripted backend gives, in order, what it was asked, and the build directories used."""

    replies: list[str] = field(default_factory=list)
    requests: list[list[Message]] = field(default_factory=list)
    builds: list[Path] = field(default_factory=list)


def scripted_backend(log: Log) -> type:
    """Return an LLMBackend class that answers from `log.replies` in order and records each request."""

    class ScriptedBackend:
        """Answers each request with the next scripted reply; a request past the script fails the test."""

        name = "scripted"
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id, as every backend does."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Record the request and return the next scripted reply."""
            log.requests.append(list(messages))
            assert log.replies, "the backend was asked for more replies than the script holds"
            return Completion(text=log.replies.pop(0), prompt_tokens=0, completion_tokens=0)

    return ScriptedBackend


def fake_toolchain(log: Log) -> type:
    """Return a Toolchain class: a file holding `#error` fails, anything else builds a PLACEHOLDER artifact."""

    class FakeToolchain:
        """Writes the files, records the build directory, and reports a build; compiles nothing."""

        name = "nvcc-sm80"
        capabilities = frozenset({"diagnostics"})

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Record the build and return a failed build or a PLACEHOLDER artifact."""
            workdir = Path(workdir)
            log.builds.append(workdir)
            for path, text in files.items():
                target = workdir / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            if any("#error" in text for text in files.values()):
                error = Diagnostic(stage="compile", severity="error", code="synthetic", message="SYNTHETIC error")
                return BuildResult(artifact=None, diagnostics=[error])
            artifact = workdir / "main"
            artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=[])

    return FakeToolchain


def registry(log: Log) -> Registry:
    """Return a Registry of the scripted backend, the fake toolchain, the none executor, and the two stages."""
    found = Registry()
    found.register("LLMBackend", "scripted", scripted_backend(log))
    found.register("Toolchain", "nvcc-sm80", fake_toolchain(log))
    found.register("Executor", "none", NoneExecutor)
    found.register("Stage", "generate", GenerateStage)
    found.register("Stage", "compile_loop", CompileLoopStage)
    return found


def recipe_data(**changes: Any) -> dict[str, Any]:
    """Return a template-set recipe with fixes.fence_tag on, with top-level keys replaced by `changes`."""
    data: dict[str, Any] = {
        "extends": "base",
        "faithful": False,
        "fixes": {"fence_tag": True},
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": "lassi-hecbench-10", "split": "eval", "items": ["layout"]},
        "directions": [{"source": DIRECTION.source, "target": DIRECTION.target}],
        "prompts": TEMPLATE_SET,
        "toolchain": {"cuda": "nvcc-sm80"},
        "stages": ["generate", "compile_loop"],
        "executor": {"kind": "none"},
        "trials": {"n": 1},
    }
    data.update(copy.deepcopy(changes))
    return data


def load(tmp_path: Path, data: Mapping[str, Any], name: str = "unfenced") -> Any:
    """Write `data` as <tmp_path>/<name>.yaml and load it with the test registry (base.yaml from projects/)."""
    path = tmp_path / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return load_recipe(path, registry=registry(Log()))


def load_error(tmp_path: Path, data: Mapping[str, Any]) -> str:
    """Return the message of the RecipeError `data` raises at load; it names the recipe file."""
    with pytest.raises(RecipeError) as info:
        load(tmp_path, data)
    message = str(info.value)
    assert "unfenced.yaml" in message, message
    return message


def context_for(tmp_path: Path, log: Log, extract: Mapping[str, Any] | None = None) -> RunContext:
    """Return a RunContext for the SYNTHETIC pair item, omp to cuda, with `extract` as the recipe's section."""
    data = recipe_data() if extract is None else recipe_data(extract=dict(extract))
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "stage-unfenced.yaml"
    path.write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    recipe = load_recipe(path, registry=registry(log))
    source = tmp_path / "bench" / SOURCE_DIR / SOURCE_FILE
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(OMP_SOURCE.encode("ascii"))
    return RunContext(
        recipe=recipe,
        backend=scripted_backend(log)(MODEL_ID),
        sampling=SAMPLING,
        toolchains={"cuda": fake_toolchain(log)()},
        executor=NoneExecutor(),
        store=TextStore(tmp_path / "store"),
        suite=SUITE,
        sources_root=tmp_path / "bench",
        item=ITEM,
        direction=DIRECTION,
        build_root=tmp_path / "builds",
        prompts=TEMPLATE_SET,
        max_corrections=10,
    )


def fresh_trial(context: RunContext) -> Trial:
    """Return a SYNTHETIC trial of the pair item with no attempts."""
    return Trial(
        trial_id=make_trial_id("stage-unfenced", MODEL_ID, SUITE_NAME, DIRECTION.name, ITEM, 1),
        recipe_hash=context.recipe.recipe_hash,
        provenance=Provenance(commit=FAKE_COMMIT, dirty=False, device="none (compile only)", sdk=None,
                              date="2026-10-23T00:00:00+00:00"),
        bench_item=SUITE.bench_item(ITEM, DIRECTION),
        model=ModelInfo(backend="scripted", id=MODEL_ID, sampling=SAMPLING),
        requests=[],
    )


ON = {"unfenced_file_lines": True}
OFF = {"unfenced_file_lines": False}


def generated(tmp_path: Path, reply: str, extract: Mapping[str, Any] | None) -> Attempt:
    """Return attempt 0 as the generate stage reads `reply` under the recipe's `extract` section (None: absent)."""
    context = context_for(tmp_path, Log(replies=[reply]), extract)
    (attempt,) = GenerateStage(context=context)(fresh_trial(context)).attempts
    return attempt


def compiled(tmp_path: Path, replies: Sequence[str], extract: Mapping[str, Any] | None) -> tuple[Trial, Log]:
    """Return the trial after generate and compile_loop with the scripted `replies`, and the log."""
    log = Log(replies=list(replies))
    context = context_for(tmp_path, log, extract)
    return CompileLoopStage(context=context)(GenerateStage(context=context)(fresh_trial(context))), log


@pytest.mark.parametrize("extract", [None, OFF], ids=["absent", "false"])
def test_off_generate_reads_an_unfenced_reply_as_today(tmp_path: Path, extract: Mapping[str, Any] | None) -> None:
    attempt = generated(tmp_path, TWO_UNFENCED, extract)
    parsed = parse_file_blocks(TWO_UNFENCED, TARGET_FILES)
    assert (attempt.files, attempt.diagnostics) == (parsed.files, parsed.diagnostics) == ({}, parsed.diagnostics)
    assert [item.code for item in attempt.diagnostics] == ["missing-file", "missing-file"]
    assert attempt.stage_reached == NO_OUTPUT


def test_on_generate_reads_an_unfenced_reply(tmp_path: Path) -> None:
    attempt = generated(tmp_path, TWO_UNFENCED, ON)
    parsed = lenient(TWO_UNFENCED, TARGET_FILES)
    assert (attempt.files, attempt.diagnostics) == (parsed.files, parsed.diagnostics)
    assert [where(item) for item in attempt.diagnostics] == [unfenced(MAIN, 1), unfenced(KERNEL, 8)]
    assert attempt.stage_reached == PARSED, "unfenced-file is a warning, never a FILE-block error: S1"


def test_on_an_attempt_with_only_unfenced_warnings_is_built_and_compiles(tmp_path: Path) -> None:
    trial, log = compiled(tmp_path, [TWO_UNFENCED], ON)
    (attempt,) = trial.attempts
    assert attempt.stage_reached == COMPILED, "the warnings ask for no correction and stop no build"
    assert len(log.builds) == 1 and len(log.requests) == 1 and trial.final.end_reason is None
    assert [item.code for item in attempt.diagnostics] == [UNFENCED, UNFENCED]
    assert warning_count(attempt) == 0, "a parse-stage note is no code warning, so df-v0's W does not count it"


def test_on_a_missing_file_beside_an_unfenced_file_stays_s1_and_is_fed_back(tmp_path: Path) -> None:
    reply = f"// FILE: {MAIN}\n{MAIN_CODE}"
    log = Log(replies=[reply, TWO_UNFENCED])
    context = context_for(tmp_path, log, ON)
    trial = CompileLoopStage(context=context)(GenerateStage(context=context)(fresh_trial(context)))
    first, second = trial.attempts
    assert first.stage_reached == PARSED and [item.code for item in first.diagnostics] == [UNFENCED, "missing-file"]
    assert (second.stage_reached, second.files) == (COMPILED, {MAIN: MAIN_CODE + "\n", KERNEL: KERNEL_CODE})
    assert second.prompt_ref is not None
    prompt = context.store.get(second.prompt_ref).split("\n")
    for item in first.diagnostics:
        assert diagnostic_line(item) in prompt, f"the correction prompt carries {item.code}"


def test_on_a_correction_reply_is_read_unfenced_too(tmp_path: Path) -> None:
    trial, log = compiled(tmp_path, [PROSE_ONLY, TWO_UNFENCED], ON)
    first, second = trial.attempts
    assert first.stage_reached == NO_OUTPUT, "prose with no FILE line still gives no file"
    assert [where(item) for item in second.diagnostics] == [unfenced(MAIN, 1), unfenced(KERNEL, 8)]
    assert second.stage_reached == COMPILED
    assert [request.stage for request in trial.requests] == ["generate", "compile_loop"]
    assert len(log.builds) == 1


@pytest.mark.parametrize(
    "reply",
    [BOTH_FENCED, FENCED_TEXTS["one-block-and-an-unfenced-file-line"], PROSE_ONLY],
    ids=["two-blocks", "one-block-and-an-unfenced-file-line", "prose-only"],
)
def test_on_a_reply_without_unfenced_reading_gives_exactly_the_off_attempt(tmp_path: Path, reply: str) -> None:
    off = generated(tmp_path / "off", reply, None)
    on = generated(tmp_path / "on", reply, ON)
    assert (on.files, on.diagnostics, on.stage_reached, on.response_text) == (
        off.files, off.diagnostics, off.stage_reached, off.response_text
    )


def test_generate_describes_the_leniency_only_when_it_is_on(tmp_path: Path) -> None:
    on = GenerateStage(context=context_for(tmp_path / "on", Log(), ON)).describe()
    off = GenerateStage(context=context_for(tmp_path / "off", Log(), None)).describe()
    assert on.endswith("parsed from FILE blocks, or unfenced FILE lines when the reply holds none"), on
    assert off.endswith("parsed from FILE blocks"), off


# ---------------------------------------------------------------------------
# The recipe key


def test_the_key_is_declared_with_a_description() -> None:
    assert set(recipe_module.EXTRACT) == {"unfenced_file_lines"}
    assert isinstance(recipe_module.EXTRACT["unfenced_file_lines"], str)
    assert recipe_module.EXTRACT["unfenced_file_lines"].strip()


def test_absent_the_key_adds_nothing_to_the_resolved_recipe(tmp_path: Path) -> None:
    recipe = load(tmp_path, recipe_data())
    assert "extract" not in recipe.data and "extract" not in recipe.canonical_yaml, (
        "an absent key is false and is never materialized, so every recipe keeps its resolved form and hash"
    )


def test_set_the_key_enters_the_resolved_recipe_and_the_hash(tmp_path: Path) -> None:
    # Each recipe is unfenced.yaml in its own directory, so each resolves with the same project.
    loaded = {}
    for label, data in (("plain", recipe_data()), ("on", recipe_data(extract=ON)), ("off", recipe_data(extract=OFF))):
        (tmp_path / label).mkdir()
        loaded[label] = load(tmp_path / label, data)
    assert loaded["on"].data["extract"] == ON and loaded["off"].data["extract"] == OFF
    assert len({recipe.recipe_hash for recipe in loaded.values()}) == 3, "a key that is set enters the hash"
    assert "unfenced_file_lines: true" in loaded["on"].canonical_yaml


def test_false_loads_under_faithful_and_with_the_fence_tag_fix_off(tmp_path: Path) -> None:
    assert load(tmp_path, recipe_data(faithful=True, extract=OFF)).data["extract"] == OFF
    assert load(tmp_path, recipe_data(fixes={"fence_tag": False}, extract=OFF)).data["extract"] == OFF


def test_true_is_refused_under_faithful(tmp_path: Path) -> None:
    message = load_error(tmp_path, recipe_data(faithful=True, extract=ON))
    assert KEY in message and "faithful: true" in message, message


def test_true_is_refused_with_the_fence_tag_fix_off(tmp_path: Path) -> None:
    message = load_error(tmp_path, recipe_data(fixes={"fence_tag": False}, extract=ON))
    assert KEY in message and "fixes.fence_tag" in message, message


def test_true_inherited_is_refused_when_the_child_turns_faithful_on(tmp_path: Path) -> None:
    load(tmp_path, recipe_data(extract=ON), name="lenient-parent")
    child = {"extends": "lenient-parent.yaml", "faithful": True}
    path = tmp_path / "unfenced.yaml"
    path.write_bytes(yaml.safe_dump(child, sort_keys=False).encode("ascii"))
    with pytest.raises(RecipeError, match=r"extract\.unfenced_file_lines"):
        load_recipe(path, registry=registry(Log()))


@pytest.mark.parametrize(
    ("section", "named"),
    [
        pytest.param({"unfenced_file_lines": 1}, KEY, id="integer"),
        pytest.param({"unfenced_file_lines": "yes"}, KEY, id="string"),
        pytest.param({"unfenced_file_lines": None}, KEY, id="null"),
        pytest.param({"other": True}, "extract.other", id="unknown-key"),
        pytest.param([True], "extract", id="list"),
    ],
)
def test_a_bad_section_is_refused_naming_the_key(tmp_path: Path, section: Any, named: str) -> None:
    message = load_error(tmp_path, recipe_data(extract=section))
    assert named in message, message
