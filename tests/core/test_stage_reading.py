"""Tests for the stage ladder reading of FILE-block replies (task P2.3).

Bible: Training Module, Reward Function (the stage table: S0 no extractable
output, S1 parses); Harness Contract (a missing file is a build error fed
back to the model); Result Record (Attempt.stage_reached, Diagnostic).

The contract these tests fix, from the P2.3 acceptance criteria
(plans/p2-scoring.md) and the P2 phase note "Stage ladder reading":

- With fixes.fence_tag on, a reply is read as FILE blocks
  (lassi.core.files.parse_file_blocks). When the reply yields at least one
  file and its only FILE-block problems are expected files with no block
  (code `missing-file`), the attempt is S1, and each missing file is a
  compile-stage error Diagnostic with code `missing-file`, exactly as
  parse_file_blocks gives it.
- That error is the build error of the Harness Contract: the attempt is not
  built with its files incomplete, stays S1, and the correction prompt
  carries the missing-file diagnostic line (lassi.core.stages
  diagnostic_line), so the model is told which file to return. A
  correction reply is read the same way as attempt 0.
- A reply with no usable block (no FILE block at all, or only blocks that
  were dropped) stays S0, and so does a reply with any other FILE-block
  error (bad-path, duplicate-file, unclosed-block), with or without a
  missing file beside it.

The stages are called directly on a SYNTHETIC one-item suite built here,
whose target language has two files, so that one expected file can be
missing while the other is present. The reply texts, sources, and file names
are SYNTHETIC; the fake toolchain writes files and a PLACEHOLDER artifact and
compiles nothing; the scripted backend answers from a script. The p0-smoke
template prompt set is used, so no upstream text is needed. No value in this
module is a measurement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.bench import Direction, LanguageSources, Suite, SuiteItem
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.files import parse_file_blocks, render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Message, Sampling
from lassi.core.recipe import load_recipe
from lassi.core.record import Attempt, Diagnostic, ModelInfo, Provenance, Trial, make_trial_id
from lassi.core.registry import Registry
from lassi.core.stages import CompileLoopStage, GenerateStage, RunContext, diagnostic_line
from lassi.core.store import TextStore
from lassi.executors import NoneExecutor

MODEL_ID = "scripted-fixture"
TEMPLATE_SET = "p0-smoke"
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
DIRECTION = Direction("omp", "cuda")
# A commit id for synthetic provenance and the synthetic suite's pin; not a commit of any repository.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"

# The SYNTHETIC suite: one eval item whose target (cuda) version has two files and whose source has one.
SUITE_NAME = "synthetic-pair"
ITEM = "pair"
SOURCE_DIR = "src/pair-omp"
SOURCE_FILE = "main.cpp"
MAIN, KERNEL = "main.cu", "kernel.cuh"
TARGET_FILES = (MAIN, KERNEL)
SUITE = Suite(
    name=SUITE_NAME,
    repo="https://example.invalid/synthetic-pair",
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

# SYNTHETIC source and target texts; the fake toolchain refuses any file holding an `#error` line.
OMP_SOURCE = "int main() {\n#pragma omp target\n  { }\n  return 0;\n}\n"
MAIN_CODE = '#include "kernel.cuh"\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n'
KERNEL_CODE = "__global__ void k() { }\n"
NOTES = "// SYNTHETIC notes that are not one of the expected files\n"

MISSING_FILE = "missing-file"
PARSED, NO_OUTPUT, COMPILED = "S1", "S0", "S4"


# ---------------------------------------------------------------------------
# Fakes, the context, and trials


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
    """Return a Toolchain class without PIN: a file holding `#error` fails, anything else builds a PLACEHOLDER."""

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


def recipe_data() -> dict[str, Any]:
    """Return a template-set recipe with fixes.fence_tag on, so replies are read as FILE blocks."""
    return {
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


def context_for(tmp_path: Path, log: Log, max_corrections: int | None = 10) -> RunContext:
    """Return a RunContext for the SYNTHETIC pair item, omp to cuda, with the scripted backend and fake toolchain."""
    registry = Registry()
    registry.register("LLMBackend", "scripted", scripted_backend(log))
    registry.register("Toolchain", "nvcc-sm80", fake_toolchain(log))
    registry.register("Executor", "none", NoneExecutor)
    registry.register("Stage", "generate", GenerateStage)
    registry.register("Stage", "compile_loop", CompileLoopStage)
    path = tmp_path / "stage-reading.yaml"
    path.write_bytes(yaml.safe_dump(recipe_data(), sort_keys=False).encode("ascii"))
    recipe = load_recipe(path, registry=registry)
    assert recipe.data["fixes"]["fence_tag"] is True, "these tests read replies with fixes.fence_tag on"
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
        max_corrections=max_corrections,
    )


def fresh_trial(context: RunContext) -> Trial:
    """Return a SYNTHETIC trial of the pair item with no attempts."""
    return Trial(
        trial_id=make_trial_id("stage-reading", MODEL_ID, SUITE_NAME, DIRECTION.name, ITEM, 1),
        recipe_hash=context.recipe.recipe_hash,
        provenance=Provenance(commit=FAKE_COMMIT, dirty=False, device="none (compile only)", sdk=None,
                              date="2026-09-24T00:00:00+00:00"),
        bench_item=SUITE.bench_item(ITEM, DIRECTION),
        model=ModelInfo(backend="scripted", id=MODEL_ID, sampling=SAMPLING),
        requests=[],
    )


def generated(tmp_path: Path, reply: str) -> Attempt:
    """Return attempt 0 as the generate stage reads `reply`."""
    log = Log(replies=[reply])
    context = context_for(tmp_path, log)
    (attempt,) = GenerateStage(context=context)(fresh_trial(context)).attempts
    return attempt


def missing_file_errors(attempt: Attempt) -> list[Diagnostic]:
    """Return the attempt's missing-file diagnostics."""
    return [item for item in attempt.diagnostics if item.code == MISSING_FILE]


def prompt_lines(context: RunContext, attempt: Attempt) -> list[str]:
    """Return the lines of the prompt the attempt was asked with, from the text store."""
    assert attempt.prompt_ref is not None, f"attempt {attempt.index} kept no prompt"
    return context.store.get(attempt.prompt_ref).split("\n")


# SYNTHETIC replies. Each is a FILE-block reply (the reply form with fixes.fence_tag on) unless named otherwise.
BOTH = {MAIN: MAIN_CODE, KERNEL: KERNEL_CODE}
MAIN_ONLY = render_file_blocks({MAIN: MAIN_CODE})
KERNEL_ONLY = render_file_blocks({KERNEL: KERNEL_CODE})
BOTH_FILES = render_file_blocks(BOTH)
BAD_PATH_BLOCK = f"```cuda\n// FILE: /abs/{KERNEL}\n{KERNEL_CODE}```\n"
UNCLOSED_KERNEL = f"```cuda\n// FILE: {KERNEL}\n{KERNEL_CODE}"
DUPLICATE_MAIN = f"```cuda\n// FILE: {MAIN}\n{MAIN_CODE}```\n"


# ---------------------------------------------------------------------------
# S1: the only FILE-block problem is a missing expected file


def test_a_reply_whose_only_problem_is_a_missing_expected_file_is_s1_with_a_missing_file_error(
    tmp_path: Path,
) -> None:
    attempt = generated(tmp_path, MAIN_ONLY)
    parsed = parse_file_blocks(MAIN_ONLY, TARGET_FILES)
    assert attempt.files == parsed.files == {MAIN: MAIN_CODE}, "the file the reply gave is kept"
    assert attempt.diagnostics == parsed.diagnostics, "the parse diagnostics, exactly as parse_file_blocks gives them"
    (missing,) = attempt.diagnostics
    assert (missing.stage, missing.severity, missing.code, missing.file) == ("compile", "error", MISSING_FILE, KERNEL)
    assert attempt.stage_reached == PARSED, (
        "a reply whose only FILE-block problem is a missing expected file parses: S1, not S0 (Reward Function "
        "stage table; the missing file is a build error, Harness Contract)"
    )


def test_a_reply_that_gives_only_files_the_target_does_not_list_is_s1_with_a_missing_file_error_per_expected_file(
    tmp_path: Path,
) -> None:
    reply = render_file_blocks({"notes.h": NOTES})
    attempt = generated(tmp_path, reply)
    parsed = parse_file_blocks(reply, TARGET_FILES)
    assert attempt.files == parsed.files == {"notes.h": NOTES}
    assert attempt.diagnostics == parsed.diagnostics
    assert [(item.stage, item.severity, item.file) for item in missing_file_errors(attempt)] == [
        ("compile", "error", MAIN),
        ("compile", "error", KERNEL),
    ]
    assert attempt.stage_reached == PARSED, "a block was extracted and the only problems are missing files: S1"


def test_the_missing_file_error_is_fed_back_and_the_incomplete_attempt_is_never_built(tmp_path: Path) -> None:
    log = Log(replies=[MAIN_ONLY, BOTH_FILES])
    context = context_for(tmp_path, log)
    trial = CompileLoopStage(context=context)(GenerateStage(context=context)(fresh_trial(context)))
    first, second = trial.attempts
    assert first.stage_reached == PARSED, "the attempt missing a file stays S1: it never compiled"
    assert first.diagnostics == parse_file_blocks(MAIN_ONLY, TARGET_FILES).diagnostics, (
        "no build diagnostic joins the missing-file error: the incomplete file set is not built"
    )
    assert len(log.builds) == 1, "only the complete attempt is built; a missing file is the build error itself"
    (missing,) = missing_file_errors(first)
    assert diagnostic_line(missing) in prompt_lines(context, second), (
        "the correction prompt carries the missing-file error, so the model is told which file to return"
    )
    assert (second.stage_reached, second.files) == (COMPILED, BOTH)
    assert trial.final.end_reason is None
    assert len(log.requests) == 2 and log.replies == []


def test_a_correction_reply_is_read_the_same_way(tmp_path: Path) -> None:
    log = Log(replies=[MAIN_ONLY, KERNEL_ONLY, BOTH_FILES])
    context = context_for(tmp_path, log)
    trial = CompileLoopStage(context=context)(GenerateStage(context=context)(fresh_trial(context)))
    assert [attempt.stage_reached for attempt in trial.attempts] == [PARSED, PARSED, COMPILED]
    assert [item.file for item in missing_file_errors(trial.attempts[1])] == [MAIN]
    for previous, correction in zip(trial.attempts[:-1], trial.attempts[1:], strict=True):
        for missing in missing_file_errors(previous):
            assert diagnostic_line(missing) in prompt_lines(context, correction), (
                f"the prompt of attempt {correction.index} names the file attempt {previous.index} left out"
            )
    assert len(log.builds) == 1, "only the complete attempt is built"


def test_at_the_cap_the_attempt_missing_a_file_ends_the_trial_at_s1(tmp_path: Path) -> None:
    log = Log(replies=[MAIN_ONLY])
    context = context_for(tmp_path, log, max_corrections=0)
    trial = CompileLoopStage(context=context)(GenerateStage(context=context)(fresh_trial(context)))
    (attempt,) = trial.attempts
    assert attempt.stage_reached == PARSED
    reason = trial.final.end_reason
    assert reason is not None and reason.code == "correction-cap", "the missing-file error remains at the cap"
    assert log.builds == []


# ---------------------------------------------------------------------------
# S0: no usable block, or any other FILE-block error

S0_REPLIES = {
    "no-block": "SYNTHETIC reply with prose and no code block.\n",
    "fence-without-file-line": f"```cuda\n{MAIN_CODE}```\n",
    "only-a-bad-path": BAD_PATH_BLOCK,
    "only-an-unclosed-block": UNCLOSED_KERNEL,
    "complete-plus-a-bad-path": BOTH_FILES + "\n" + BAD_PATH_BLOCK,
    "complete-plus-a-duplicate": BOTH_FILES + "\n" + DUPLICATE_MAIN,
    "missing-plus-a-bad-path": MAIN_ONLY + "\n" + BAD_PATH_BLOCK,
    "missing-plus-a-duplicate": MAIN_ONLY + "\n" + DUPLICATE_MAIN,
    "one-closed-one-unclosed": MAIN_ONLY + "\n" + UNCLOSED_KERNEL,
}


@pytest.mark.parametrize("reply", list(S0_REPLIES.values()), ids=list(S0_REPLIES))
def test_a_reply_with_no_usable_block_or_another_block_error_stays_s0(tmp_path: Path, reply: str) -> None:
    attempt = generated(tmp_path, reply)
    parsed = parse_file_blocks(reply, TARGET_FILES)
    assert (attempt.files, attempt.diagnostics) == (parsed.files, parsed.diagnostics)
    assert any(item.severity == "error" for item in attempt.diagnostics), "each case has a FILE-block error"
    assert attempt.stage_reached == NO_OUTPUT


@pytest.mark.parametrize(
    "reply",
    [S0_REPLIES["no-block"], S0_REPLIES["missing-plus-a-bad-path"]],
    ids=["no-block", "missing-plus-a-bad-path"],
)
def test_an_s0_attempt_is_fed_back_and_never_built(tmp_path: Path, reply: str) -> None:
    log = Log(replies=[reply, BOTH_FILES])
    context = context_for(tmp_path, log)
    trial = CompileLoopStage(context=context)(GenerateStage(context=context)(fresh_trial(context)))
    first, second = trial.attempts
    assert (first.stage_reached, second.stage_reached) == (NO_OUTPUT, COMPILED)
    assert len(log.builds) == 1, "an S0 attempt is not built"
    lines = prompt_lines(context, second)
    for error in (item for item in first.diagnostics if item.severity == "error"):
        assert diagnostic_line(error) in lines, f"the correction prompt carries {error.code}"
