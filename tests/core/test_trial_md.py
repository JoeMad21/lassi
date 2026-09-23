"""Tests for trial.md rendering (P0.2).

render_trial_md in lassi/core/trial_md.py must reproduce the committed golden
file tests/core/golden/trial.md for one fixed two-attempt Trial. The golden
file must itself contain each prompt, each attempt's code, the unified diff,
the parsed diagnostics, and the score breakdown (bible Readability Standards,
Trial row), with PLACEHOLDER for every unmeasured value. Output is plain ASCII
with LF newlines and ends with one newline. The fixture values are synthetic
renderer inputs; no value in the fixture or the golden file is a measurement.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from lassi.core import interfaces, record, store, trial_md

GOLDEN = Path(__file__).resolve().parent / "golden" / "trial.md"
GOLDEN_ID = "lassi-repro/mock-fixture/lassi-hecbench-10/omp-cuda/entropy/run01"
RECIPE_HASH = "0123456789abcdef" * 4

BACKSLASH = "\\"
E_ACUTE = "\N{LATIN SMALL LETTER E WITH ACUTE}"
I_DIAERESIS = "\N{LATIN SMALL LETTER I WITH DIAERESIS}"
APOSTROPHE = "\N{RIGHT SINGLE QUOTATION MARK}"
EMOJI = "\N{GRINNING FACE}"

KNOWLEDGE = (
    "Synthetic fixture for the trial.md golden test; no value in this trial is a measurement.\n"
    "OpenMP target teams distribute maps to a CUDA grid of thread blocks.\n"
)
PROMPT_0 = "Translate the OpenMP program entropy.cpp to CUDA.\nReturn every file in a // FILE: <relative path> block.\n"
PROMPT_1 = (
    f"The code did not compile: the compiler didn{APOSTROPHE}t accept blockDimx.\n"
    "Fix the errors below and return every file again.\n"
    'main.cu(3): error: identifier "blockDimx" is undefined'
)
CODE_0 = (
    "#include <cstdio>\n"
    "__global__ void entropy(const float* in, float* out, int n) {\n"
    "  int i = blockIdx.x * blockDimx + threadIdx.x;\n"
    "  if (i < n) out[i] = in[i] * 0.5f;\n"
    "}\n"
)
CODE_1 = (
    "#include <cstdio>\n"
    '#include "kernels/entropy.cuh"\n'
    "__global__ void entropy(const float* in, float* out, int n) {\n"
    "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "  if (i < n) out[i] = scale(in[i]);\n"
    "}\n"
)
HEADER_1 = (
    "// Usage:\n"
    "// ```\n"
    "// out[i] = scale(in[i]);\n"
    "// ```\n"
    "#pragma once\n"
    "__device__ inline float scale(float x) { return x * 0.5f; }"
)
DIFF_1 = (
    "--- /dev/null\n"
    "+++ b/kernels/entropy.cuh\n"
    "@@ -0,0 +1,6 @@\n"
    "+// Usage:\n"
    "+// ```\n"
    "+// out[i] = scale(in[i]);\n"
    "+// ```\n"
    "+#pragma once\n"
    "+__device__ inline float scale(float x) { return x * 0.5f; }\n"
    "\\ No newline at end of file\n"
    "--- a/main.cu\n"
    "+++ b/main.cu\n"
    "@@ -1,5 +1,6 @@\n"
    " #include <cstdio>\n"
    '+#include "kernels/entropy.cuh"\n'
    " __global__ void entropy(const float* in, float* out, int n) {\n"
    "-  int i = blockIdx.x * blockDimx + threadIdx.x;\n"
    "-  if (i < n) out[i] = in[i] * 0.5f;\n"
    "+  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
    "+  if (i < n) out[i] = scale(in[i]);\n"
    " }\n"
)
STDOUT_1 = "PASS\n"
RESPONSE_0 = "// FILE: main.cu\n" + CODE_0
RESPONSE_1 = "// FILE: main.cu\n" + CODE_1 + "// FILE: kernels/entropy.cuh\n" + HEADER_1 + "\n"
UNDEFINED_MESSAGE = 'identifier "blockDimx" is undefined'
PIPE_MESSAGE = '1 error detected in the compilation of "main.cu" | build stopped'


# ---------------------------------------------------------------------------
# Helpers and fixtures


def read_golden() -> str:
    """Return the golden trial.md text with newlines normalized."""
    return GOLDEN.read_text(encoding="utf-8")


def ascii_form(text: str) -> str:
    """Return text with every non-ASCII character written as a Python backslash escape."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


def fence_block(text: str, lang: str, width: int = 3) -> str:
    """Return the fenced block the contract specifies for text, with an explicit fence width."""
    fence = "`" * width
    body = text if text.endswith("\n") else text + "\n"
    return f"{fence}{lang}\n{ascii_form(body)}{fence}\n"


def stored_line(text: str) -> str:
    """Return the 'Stored as' line for a prompt, with the sha256 computed by hashlib."""
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"Stored as `texts/{sha[:2]}/{sha}.txt` (sha256 `{sha}`)."


def bench_item() -> record.BenchItem:
    """Return the bench item that matches GOLDEN_ID."""
    return record.BenchItem(suite="lassi-hecbench-10", item="entropy", split="eval", direction="omp-cuda")


def model_info() -> record.ModelInfo:
    """Return the fixed mock model used by every fixture here."""
    sampling = interfaces.Sampling(temperature=0.2, top_p=0.95, max_tokens=4096)
    return record.ModelInfo(backend="mock", id="mock-fixture", sampling=sampling)


def golden_attempts(text_store: store.TextStore) -> list[record.Attempt]:
    """Return the two golden attempts, putting their prompts and stdout in the store."""
    first = record.Attempt(
        index=0,
        prompt_ref=text_store.put(PROMPT_0),
        response_text=RESPONSE_0,
        files={"main.cu": CODE_0},
        stage_reached="S1",
        diagnostics=[
            record.Diagnostic(
                stage="compile",
                severity="error",
                code="20",
                file="main.cu",
                line=3,
                column=24,
                message=UNDEFINED_MESSAGE,
            ),
            record.Diagnostic(stage="compile", severity="note", message=PIPE_MESSAGE),
        ],
    )
    second = record.Attempt(
        index=1,
        prompt_ref=text_store.put(PROMPT_1),
        response_text=RESPONSE_1,
        files={"main.cu": CODE_1, "kernels/entropy.cuh": HEADER_1},
        diff_from_previous=DIFF_1,
        stage_reached="S5",
        run=record.RunInfo(exit_code=0, hang=False, stdout_ref=text_store.put(STDOUT_1)),
        alignment=record.Alignment(per_input=[1.0, 0.5, 0.75, 1.0], mean=0.8125),
        guards=record.Guards(host_compute=False, harness_tamper=False),
        score=record.ScoreBreakdown(components={"stage": 0.2, "warnings": 0.0, "alignment": 0.8125, "energy": None}),
    )
    return [first, second]


def golden_trial(text_store: store.TextStore) -> record.Trial:
    """Return the fixed Trial that tests/core/golden/trial.md renders."""
    return record.Trial(
        trial_id=GOLDEN_ID,
        recipe_hash=RECIPE_HASH,
        toolchain_pins=record.ToolchainPins(cuda="fixture-cuda", nvhpc="fixture-nvhpc"),
        bench_item=bench_item(),
        model=model_info(),
        context=record.Context(knowledge_summary=KNOWLEDGE),
        attempts=golden_attempts(text_store),
        final=record.Final(stage_reached="S5", alignment=0.8125, corrections=1),
    )


def base_trial(attempts: list[record.Attempt], **changes: Any) -> record.Trial:
    """Return a Trial for GOLDEN_ID with the given attempts and other fields changed."""
    fields: dict[str, Any] = {
        "trial_id": GOLDEN_ID,
        "recipe_hash": RECIPE_HASH,
        "bench_item": bench_item(),
        "model": model_info(),
        "attempts": attempts,
    }
    fields.update(changes)
    return record.Trial(**fields)


def render_attempt(text_store: store.TextStore, **attempt_fields: Any) -> str:
    """Render a Trial with one attempt (index 0, stage S1 unless given) built from the fields."""
    attempt_fields.setdefault("stage_reached", "S1")
    attempt = record.Attempt(index=0, **attempt_fields)
    return trial_md.render_trial_md(base_trial([attempt]), text_store)


def assert_layout(md: str) -> None:
    """Assert document rules: ASCII, LF, one final newline, single blank lines, no trailing spaces."""
    assert md.isascii()
    assert "\r" not in md
    assert md.endswith("\n") and not md.endswith("\n\n")
    assert "\n\n\n" not in md
    trailing = [line for line in md.split("\n") if line != line.rstrip()]
    assert not trailing, trailing


@pytest.fixture
def text_store(tmp_path: Path) -> store.TextStore:
    """Return a text store rooted in a fresh temporary directory."""
    root = tmp_path / "store"
    root.mkdir()
    return store.TextStore(root)


# ---------------------------------------------------------------------------
# Golden comparison


def test_placeholder_constant() -> None:
    assert trial_md.PLACEHOLDER == "PLACEHOLDER"


def test_render_matches_golden(text_store: store.TextStore) -> None:
    assert trial_md.render_trial_md(golden_trial(text_store), text_store) == read_golden()


def test_render_is_deterministic(tmp_path: Path) -> None:
    outputs = []
    for name in ("one", "two"):
        root = tmp_path / name
        root.mkdir()
        text_store = store.TextStore(root)
        trial = golden_trial(text_store)
        outputs += [trial_md.render_trial_md(trial, text_store), trial_md.render_trial_md(trial, text_store)]
    assert len(set(outputs)) == 1


def test_fixture_diff_is_the_unified_diff_of_its_files() -> None:
    previous = {"main.cu": CODE_0}
    current = {"main.cu": CODE_1, "kernels/entropy.cuh": HEADER_1}
    assert record.unified_diff(previous, current) == DIFF_1


# ---------------------------------------------------------------------------
# Golden content: the file itself must carry every required part


def test_golden_contains_each_prompt() -> None:
    golden = read_golden()
    for prompt in (PROMPT_0, PROMPT_1):
        assert stored_line(prompt) + "\n\n" + fence_block(prompt, "text") in golden
    assert "didn" + BACKSLASH + "u2019t accept" in golden


def test_golden_contains_each_attempts_code() -> None:
    golden = read_golden()
    assert "#### `main.cu`\n\n" + fence_block(CODE_0, "cuda") in golden
    assert "#### `main.cu`\n\n" + fence_block(CODE_1, "cuda") in golden
    assert "#### `kernels/entropy.cuh`\n\n" + fence_block(HEADER_1, "cuda", width=4) in golden
    assert golden.index("#### `kernels/entropy.cuh`") < golden.rindex("#### `main.cu`")


def test_golden_contains_the_unified_diff() -> None:
    golden = read_golden()
    assert "### Diff from previous attempt\n\nNone (initial attempt).\n" in golden
    assert "### Diff from previous attempt\n\n" + fence_block(DIFF_1, "diff", width=4) in golden


def test_golden_contains_parsed_diagnostics() -> None:
    golden = read_golden()
    table = (
        "### Diagnostics\n\n"
        "| Stage | Severity | Code | Location | Message |\n"
        "| --- | --- | --- | --- | --- |\n"
        f"| compile | error | 20 | main.cu:3:24 | {UNDEFINED_MESSAGE} |\n"
        "| compile | note | - | - | " + PIPE_MESSAGE.replace("|", BACKSLASH + "|") + " |\n"
    )
    assert table in golden
    assert "### Diagnostics\n\nNone.\n" in golden


def test_golden_contains_score_breakdown() -> None:
    golden = read_golden()
    partial_score = (
        "### Score breakdown\n\n"
        "| Component | Value |\n"
        "| --- | --- |\n"
        "| alignment | 0.8125 |\n"
        "| energy | PLACEHOLDER |\n"
        "| stage | 0.2 |\n"
        "| warnings | 0.0 |\n"
        "| scalar | PLACEHOLDER |\n"
    )
    empty = "### Score breakdown\n\n| Component | Value |\n| --- | --- |\n| scalar | PLACEHOLDER |\n"
    assert partial_score in golden
    assert empty in golden
    assert golden.endswith(partial_score)


def test_golden_marks_unmeasured_values_placeholder() -> None:
    golden = read_golden()
    assert "| Score | PLACEHOLDER |\n" in golden
    assert "| Wall time (s) | PLACEHOLDER |\n" in golden
    assert golden.count("| scalar | PLACEHOLDER |\n") == 2
    assert golden.count("| energy | PLACEHOLDER |\n") == 1
    for row in ("runtime_s", "avg_power_w", "energy_j"):
        assert golden.count(f"| {row} | PLACEHOLDER |\n") == 2, row
    assert golden.count("| per_input | PLACEHOLDER |\n") == 1
    assert "| per_input | 1.0, 0.5, 0.75, 1.0 |\n" in golden
    assert golden.count("| outputs_ref | PLACEHOLDER |\n") == 2
    assert golden.count("| oracle_access | PLACEHOLDER |\n") == 2


def test_golden_is_plain_ascii_lf() -> None:
    raw = GOLDEN.read_bytes()
    assert raw.isascii()
    assert b"\r" not in raw
    assert_layout(raw.decode("ascii"))


# ---------------------------------------------------------------------------
# Rendering rules


def test_output_is_ascii_for_non_ascii_input(text_store: store.TextStore) -> None:
    attempt = record.Attempt(
        index=0,
        prompt_ref=text_store.put(f"don{APOSTROPHE}t {EMOJI}\n"),
        files={f"k{E_ACUTE}.cu": f"// caf{E_ACUTE}\n"},
        diff_from_previous=f"+// caf{E_ACUTE}\n",
        stage_reached="S1",
        diagnostics=[record.Diagnostic(stage="compile", severity="error", message=f"na{I_DIAERESIS}ve")],
    )
    trial = base_trial(
        [attempt],
        toolchain_pins=record.ToolchainPins(llvm=f"19{APOSTROPHE}"),
        bench_item=record.BenchItem(
            suite="lassi-hecbench-10", item="entropy", split=f"{E_ACUTE}val", direction="omp-cuda"
        ),
        context=record.Context(knowledge_summary=f"caf{E_ACUTE}\n", source_description=f"{EMOJI}\n"),
    )
    md = trial_md.render_trial_md(trial, text_store)
    assert_layout(md)
    assert "don" + BACKSLASH + "u2019t " + BACKSLASH + "U0001f600\n" in md
    assert "#### `k\\xe9.cu`\n\n```cuda\n// caf\\xe9\n```\n" in md
    assert "| compile | error | - | - | na\\xefve |\n" in md
    assert "| llvm | 19" + BACKSLASH + "u2019 |\n" in md
    assert "| Split | \\xe9val |\n" in md
    assert "### Knowledge summary\n\n```text\ncaf\\xe9\n```\n" in md
    assert "### Source description\n\n```text\n" + BACKSLASH + "U0001f600\n```\n" in md


def test_trial_without_attempts_ends_after_context(text_store: store.TextStore) -> None:
    trial = base_trial([], model=record.ModelInfo(backend="mock", id="m", sampling=interfaces.Sampling(0.0, 1.0, 16)))
    pins = "".join(f"| {name} | not used |\n" for name in record.TOOLCHAIN_PIN_NAMES)
    expected = (
        f"# Trial {GOLDEN_ID}\n\n"
        "| Field | Value |\n"
        "| --- | --- |\n"
        f"| Recipe hash | `{RECIPE_HASH}` |\n"
        "| Suite | lassi-hecbench-10 |\n"
        "| Item | entropy |\n"
        "| Direction | omp-cuda |\n"
        "| Split | eval |\n"
        "| Model | mock `m` |\n"
        "| Sampling | temperature 0.0, top_p 1.0, max_tokens 16 |\n"
        "| Stage reached | PLACEHOLDER |\n"
        "| Alignment | PLACEHOLDER |\n"
        "| Score | PLACEHOLDER |\n"
        "| Corrections | 0 |\n"
        "| Wall time (s) | PLACEHOLDER |\n\n"
        "## Toolchain pins\n\n"
        "| Toolchain | Pin |\n"
        "| --- | --- |\n"
        f"{pins}\n"
        "## Context\n\n"
        "### Knowledge summary\n\n"
        "None.\n\n"
        "### Source description\n\n"
        "None.\n"
    )
    assert trial_md.render_trial_md(trial, text_store) == expected


def test_empty_attempt_sections_render_none(text_store: store.TextStore) -> None:
    attempts = [record.Attempt(index=0, stage_reached="S0"), record.Attempt(index=1, stage_reached="S0")]
    md = trial_md.render_trial_md(base_trial(attempts), text_store)
    assert_layout(md)
    for index, diff_line in ((0, "None (initial attempt)."), (1, "No changes.")):
        section = (
            f"## Attempt {index}\n\n"
            "Stage reached: S0\n\n"
            "### Prompt\n\nNone.\n\n"
            "### Code\n\nNone.\n\n"
            f"### Diff from previous attempt\n\n{diff_line}\n\n"
            "### Diagnostics\n\nNone.\n\n"
            "### Run\n\n"
        )
        assert section in md
    assert md.endswith("### Score breakdown\n\n| Component | Value |\n| --- | --- |\n| scalar | PLACEHOLDER |\n")


def test_fence_grows_past_backtick_runs(text_store: store.TextStore) -> None:
    files = {
        "a.txt": "one ` tick\n",
        "b.txt": "five ````` ticks\n",
        "c.txt": "none\n",
        "d.txt": "three ``` and two ``\n",
    }
    md = render_attempt(text_store, files=files)
    assert "#### `a.txt`\n\n```text\none ` tick\n```\n" in md
    assert "#### `b.txt`\n\n``````text\nfive ````` ticks\n``````\n" in md
    assert "#### `c.txt`\n\n```text\nnone\n```\n" in md
    assert "#### `d.txt`\n\n````text\nthree ``` and two ``\n````\n" in md


def test_fenced_text_is_verbatim(text_store: store.TextStore) -> None:
    md = render_attempt(text_store, files={"k.c": "int x;  \n\n\nint y;"})
    assert "#### `k.c`\n\n```c\nint x;  \n\n\nint y;\n```\n" in md


def test_fenced_text_gets_lf_line_breaks(text_store: store.TextStore) -> None:
    attempt = record.Attempt(
        index=0,
        prompt_ref=text_store.put("fix it\r\nnow\r\n"),
        files={"main.cu": "int x;\r\nint y;\r\n", "log.txt": "10%\r20%\rdone"},
        diff_from_previous="--- a/main.cu\r\n+++ b/main.cu\r\n@@ -1 +1 @@\r\n-int x;\r\n+int y;\r\n",
        stage_reached="S1",
    )
    context = record.Context(knowledge_summary="a\r\nb\r\n", source_description="c\rd\r")
    md = trial_md.render_trial_md(base_trial([attempt], context=context), text_store)
    assert "\r" not in md
    assert "### Knowledge summary\n\n```text\na\nb\n```\n" in md
    assert "### Source description\n\n```text\nc\nd\n```\n" in md
    assert "(sha256 `" + hashlib.sha256(b"fix it\r\nnow\r\n").hexdigest() + "`).\n\n```text\nfix it\nnow\n```\n" in md
    assert "#### `main.cu`\n\n```cuda\nint x;\nint y;\n```\n" in md
    assert "#### `log.txt`\n\n```text\n10%\n20%\ndone\n```\n" in md
    assert "```diff\n--- a/main.cu\n+++ b/main.cu\n@@ -1 +1 @@\n-int x;\n+int y;\n```\n" in md


@pytest.mark.parametrize(
    ("path", "lang"),
    [
        ("k.cu", "cuda"),
        ("k.cuh", "cuda"),
        ("K.CU", "cuda"),
        ("k.c", "c"),
        ("K.C", "c"),
        ("k.h", "cpp"),
        ("k.hpp", "cpp"),
        ("k.cpp", "cpp"),
        ("k.cc", "cpp"),
        ("k.cxx", "cpp"),
        ("k.f", "fortran"),
        ("k.F90", "fortran"),
        ("k.f95", "fortran"),
        ("k.py", "python"),
        ("k.rs", "rust"),
        ("k.cs", "csharp"),
        ("k.mlir", "mlir"),
        ("src/k.cu", "cuda"),
        ("Makefile", "text"),
        ("k.txt", "text"),
        ("k.cu.bak", "text"),
    ],
)
def test_language_from_suffix(text_store: store.TextStore, path: str, lang: str) -> None:
    md = render_attempt(text_store, files={path: "x\n"})
    assert f"#### `{path}`\n\n```{lang}\nx\n```\n" in md


@pytest.mark.parametrize(
    ("code", "file", "line", "column", "cells"),
    [
        ("E1", "a.cu", 7, 3, "E1 | a.cu:7:3"),
        (None, "a.cu", 7, None, "- | a.cu:7"),
        (None, "a.cu", None, None, "- | a.cu"),
        (None, "a.cu", None, 5, "- | a.cu"),
        ("E2", None, 7, 3, "E2 | -"),
        (None, None, None, None, "- | -"),
    ],
)
def test_diagnostic_code_and_location_cells(
    text_store: store.TextStore, code: str | None, file: str | None, line: int | None, column: int | None, cells: str
) -> None:
    diagnostic = record.Diagnostic(
        stage="run", severity="warning", code=code, file=file, line=line, column=column, message="m"
    )
    md = render_attempt(text_store, diagnostics=[diagnostic])
    header = "| Stage | Severity | Code | Location | Message |\n| --- | --- | --- | --- | --- |\n"
    assert f"### Diagnostics\n\n{header}| run | warning | {cells} | m |\n\n### Run\n" in md


def test_table_cells_escape_pipes_and_newlines(text_store: store.TextStore) -> None:
    diagnostic = record.Diagnostic(stage="compile", severity="error", message="left | right\r\nnext\nlast\rend")
    attempt = record.Attempt(index=0, stage_reached="S1", diagnostics=[diagnostic])
    trial = base_trial(
        [attempt],
        toolchain_pins=record.ToolchainPins(llvm="a|b"),
        context=record.Context(knowledge_summary="x | y\n"),
    )
    md = trial_md.render_trial_md(trial, text_store)
    assert "| compile | error | - | - | left \\| right next last end |\n" in md
    assert "| llvm | a\\|b |\n" in md
    assert "### Knowledge summary\n\n```text\nx | y\n```\n" in md


def test_value_formatting_in_tables(text_store: store.TextStore) -> None:
    ref = text_store.put("out\n")
    md = render_attempt(
        text_store,
        run=record.RunInfo(exit_code=3, hang=True, sim_ub=False, wall_s=0.5, stdout_ref=ref),
        alignment=record.Alignment(per_input=[0.25, 1.0]),
        profile=record.Profile(runtime_s=0.25, energy_j=37.5),
        guards=record.Guards(host_compute=True, oracle_access=False),
        score=record.ScoreBreakdown(components={"zeta": 0.5, "alpha": None}, scalar=-1.0),
    )
    run = (
        "### Run\n\n| Field | Value |\n| --- | --- |\n"
        "| exit_code | 3 |\n| hang | true |\n| sim_ub | false |\n| wall_s | 0.5 |\n"
        f"| stdout_ref | `{ref.path}` |\n| outputs_ref | PLACEHOLDER |\n"
    )
    alignment = "### Alignment\n\n| Field | Value |\n| --- | --- |\n| per_input | 0.25, 1.0 |\n| mean | PLACEHOLDER |\n"
    profile = (
        "### Profile\n\n| Field | Value |\n| --- | --- |\n"
        "| runtime_s | 0.25 |\n| avg_power_w | PLACEHOLDER |\n| energy_j | 37.5 |\n"
    )
    guards = (
        "### Guards\n\n| Field | Value |\n| --- | --- |\n"
        "| host_compute | true |\n| harness_tamper | PLACEHOLDER |\n| oracle_access | false |\n"
    )
    score = (
        "### Score breakdown\n\n| Component | Value |\n| --- | --- |\n"
        "| alpha | PLACEHOLDER |\n| zeta | 0.5 |\n| scalar | -1.0 |\n"
    )
    assert run + "\n" + alignment + "\n" + profile + "\n" + guards + "\n" + score == md[md.index("### Run") :]


def test_header_rows_format_values(text_store: store.TextStore) -> None:
    final = record.Final(stage_reached="S4", alignment=1.0, score=-0.4, corrections=2, wall_s=12.5)
    md = trial_md.render_trial_md(base_trial([], final=final), text_store)
    rows = (
        "| Model | mock `mock-fixture` |\n"
        "| Sampling | temperature 0.2, top_p 0.95, max_tokens 4096 |\n"
        "| Stage reached | S4 |\n"
        "| Alignment | 1.0 |\n"
        "| Score | -0.4 |\n"
        "| Corrections | 2 |\n"
        "| Wall time (s) | 12.5 |\n"
    )
    assert rows in md


def test_int_values_render_like_floats(text_store: store.TextStore) -> None:
    sampling = interfaces.Sampling(temperature=0, top_p=1, max_tokens=16)
    as_ints = base_trial(
        [record.Attempt(index=0, stage_reached="S5", alignment=record.Alignment(per_input=[1], mean=1))],
        model=record.ModelInfo(backend="mock", id="m", sampling=sampling),
        final=record.Final(alignment=1, wall_s=2),
    )
    md = trial_md.render_trial_md(as_ints, text_store)
    assert "| Sampling | temperature 0.0, top_p 1.0, max_tokens 16 |\n" in md
    assert "| Alignment | 1.0 |\n" in md
    assert "| Wall time (s) | 2.0 |\n" in md
    assert "| per_input | 1.0 |\n| mean | 1.0 |\n" in md
    back = record.from_json(record.Trial, record.to_json(as_ints))
    assert trial_md.render_trial_md(back, text_store) == md


def test_added_empty_file_shows_in_the_diff(text_store: store.TextStore) -> None:
    before = {"a.cu": "int x;\n"}
    after = {"a.cu": "int x;\n", "b/__init__.py": ""}
    attempts = [
        record.Attempt(index=0, stage_reached="S1", files=before),
        record.Attempt(index=1, stage_reached="S1", files=after, diff_from_previous=record.unified_diff(before, after)),
    ]
    md = trial_md.render_trial_md(base_trial(attempts), text_store)
    assert "### Diff from previous attempt\n\n```diff\n--- /dev/null\n+++ b/b/__init__.py\n```\n" in md
    assert "No changes." not in md


def test_prompt_is_resolved_through_the_store(text_store: store.TextStore) -> None:
    sha = hashlib.sha256(b"never stored").hexdigest()
    missing = record.TextRef(sha256=sha, path=f"texts/{sha[:2]}/{sha}.txt")
    attempt = record.Attempt(index=0, stage_reached="S0", prompt_ref=missing)
    with pytest.raises(store.TextNotFoundError):
        trial_md.render_trial_md(base_trial([attempt]), text_store)
