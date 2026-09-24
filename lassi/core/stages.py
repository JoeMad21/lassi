"""The stages of the compile-only path: generate and compile_loop (bible Component Interfaces, Stage row).

Stages are pure over the trial record (Stage contract rules): a stage reads
the Trial's fields, appends an attempt or annotates the last one, and
returns a new Trial; the Trial it was given is never changed. Side effects
go through components: the LLM backend, the toolchain, and the text store
that keeps each prompt. Each stage is built as `factory(context=<RunContext>)`
once per trial (lassi.core.registry states the construction convention), so
a stage object never carries state from one trial to the next.

Prompts are template files, never string literals here (Design Principle
3): generate renders `generate.txt` and compile_loop renders `correct.txt`
from the recipe's prompt set (lassi.prompts). Each stage class declares the
prompts it renders and their fields in `prompt_fields`, so the runner can
check a prompt set before any model is asked. The model answers with FILE
blocks, which lassi.core.files parses.

Stage reached (Result Record, Attempt.stage_reached) is the bible's stage
ladder (Training Module, Reward Function), the one scale every record, metric,
and reward reads (Design Principle 2):

- S0, no extractable output: the reply held no FILE block, or its FILE
  blocks had an error (a bad path, a repeated or unclosed block, or a
  missing expected file), so it did not yield the expected files.
- S1, parses: the FILE blocks gave the files with no FILE-block error.
- S2, verifies, and S3, lowers: the MLIR verifier and the lowering passes.
  Source-level translation has neither step, so these stages never record
  S2 or S3.
- S4, compiles: the toolchain built the files into an artifact.
- S5, runs clean: the artifact ran with no crash, undefined behavior, or
  hang; the run loop records it.

Output agreement with the oracle is never a stage: it is Attempt.alignment
and Final.alignment, which the reward weighs separately. These stages only
compile, so they reach at most S4. Nothing runs, so every attempt keeps
RunInfo at its defaults (None): a compile-only executor's placeholder values
are never copied into the record.
"""

from __future__ import annotations

import dataclasses
import errno
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from lassi.bench import Direction, Suite
from lassi.core.files import parse_file_blocks, render_file_blocks
from lassi.core.interfaces import BuildResult, Executor, LLMBackend, Message, Sampling, Toolchain
from lassi.core.recipe import Recipe
from lassi.core.record import Attempt, Diagnostic, TextRef, Trial, unified_diff
from lassi.core.registry import register
from lassi.core.store import TextStore
from lassi.executors.workdir import fresh_build_dir
from lassi.prompts import render

# The purpose every bench read here serves: these stages evaluate, they never train (Agent Rule 5).
PURPOSE = "eval"

# The rungs of the stage ladder these stages record (see the module docstring).
NO_OUTPUT = "S0"
PARSED = "S1"
COMPILED = "S4"

# The fields each prompt template gets; a template may use any of them and no other.
GENERATE_FIELDS = ("source_language", "target_language", "source_files", "target_files")
CORRECT_FIELDS = ("target_language", "files", "diagnostics", "target_files")

# A lone surrogate code point: JSON can decode one from a reply ("\ud83d"), but it is not Unicode text.
_SURROGATE = re.compile("[\ud800-\udfff]")
# What replaces each lone surrogate: U+FFFD, the Unicode replacement character.
_REPLACEMENT = "\ufffd"
# The errors a file name the model chose can cause while it is written: too long, or refused by the filesystem.
_NAME_ERRNOS = frozenset({errno.ENAMETOOLONG, errno.EINVAL})


@dataclass(frozen=True)
class RunContext:
    """What every stage of one trial reads: the recipe, the components, the bench item, and the run's settings.

    `toolchains` maps a language to the toolchain that builds it.
    `build_root` is the root that lassi.executors.workdir.build_dir places each
    attempt's build directory under; the runner passes the run directory, so
    one run's builds never meet another's. `prompts` is the prompt set, and
    `max_corrections` the correction cap (None means uncapped).
    """

    recipe: Recipe
    backend: LLMBackend
    sampling: Sampling
    toolchains: Mapping[str, Toolchain]
    executor: Executor
    store: TextStore
    suite: Suite
    sources_root: Path
    item: str
    direction: Direction
    build_root: Path
    prompts: str
    max_corrections: int | None


def target_files(context: RunContext) -> list[str]:
    """Return the file names the target language version of the item has, from the suite manifest, in order."""
    item = context.suite.item(context.item, purpose=PURPOSE)
    sources = item.languages.get(context.direction.target)
    if sources is None:
        known = ", ".join(item.languages)
        raise ValueError(
            f"{context.suite.name}/{context.item} has no {context.direction.target!r} files; languages: {known}"
        )
    return list(sources.files)


def diagnostic_line(diagnostic: Diagnostic) -> str:
    """Return one Diagnostic as a correction prompt shows it: `<severity> <file>:<line>:<column> [<code>] <message>`.

    Absent parts are left out: the location is file, file:line, or
    file:line:column as far as set, and nothing without a file. Line breaks
    inside the message become spaces, so each Diagnostic is exactly one line.
    """
    parts = [diagnostic.severity]
    if diagnostic.file is not None:
        location = diagnostic.file
        if diagnostic.line is not None:
            location += f":{diagnostic.line}"
            if diagnostic.column is not None:
                location += f":{diagnostic.column}"
        parts.append(location)
    if diagnostic.code is not None:
        parts.append(f"[{diagnostic.code}]")
    parts.append(" ".join(diagnostic.message.splitlines()))
    return " ".join(parts)


def _has_error(diagnostics: Sequence[Diagnostic]) -> bool:
    """Return True when any diagnostic is an error."""
    return any(diagnostic.severity == "error" for diagnostic in diagnostics)


def _compile_error(code: str, message: str) -> Diagnostic:
    """Return a compile-stage error Diagnostic with no location."""
    return Diagnostic(stage="compile", severity="error", code=code, message=message)


def _storable(reply: str) -> tuple[str, list[Diagnostic]]:
    """Return the reply with each lone surrogate replaced by U+FFFD, and a warning saying so when there was one.

    A lone surrogate cannot be stored as UTF-8 or written into a source
    file, so it would stop the run; the warning keeps the change visible in
    the record and the next correction prompt.
    """
    count = len(_SURROGATE.findall(reply))
    if not count:
        return reply, []
    message = (
        f"the reply held {count} lone surrogate code point(s), which are not Unicode text; "
        "each was replaced with U+FFFD before the FILE blocks were read"
    )
    return _SURROGATE.sub(_REPLACEMENT, reply), [
        Diagnostic(stage="parse", severity="warning", code="invalid-text", message=message)
    ]


def _parsed_attempt(index: int, prompt_ref: TextRef, reply: str, expected: Sequence[str]) -> Attempt:
    """Return the attempt for one model reply: its FILE blocks parsed, S1 when usable and S0 otherwise."""
    text, warnings = _storable(reply)
    parsed = parse_file_blocks(text, expected)
    usable = bool(parsed.files) and not _has_error(parsed.diagnostics)
    return Attempt(
        index=index,
        prompt_ref=prompt_ref,
        response_text=text,
        files=parsed.files,
        diff_from_previous="",
        stage_reached=PARSED if usable else NO_OUTPUT,
        diagnostics=[*warnings, *parsed.diagnostics],
    )


def _ask(context: RunContext, prompt: str) -> tuple[TextRef, str]:
    """Store `prompt`, send it to the backend as one user message, and return its reference and the reply text."""
    ref = context.store.put(prompt)
    completion = context.backend.complete([Message("user", prompt)], context.sampling)
    return ref, completion.text


@register("Stage", "generate")
class GenerateStage:
    """Appends attempt 0: the model's first translation of the item's source files, parsed from FILE blocks."""

    name = "generate"
    capabilities = frozenset({"generates"})
    requires = {"LLMBackend": {"chat"}}
    prompt_fields = {"generate": GENERATE_FIELDS}

    def __init__(self, *, context: RunContext) -> None:
        """Keep the trial's run context."""
        self.context = context

    def __call__(self, trial: Trial) -> Trial:
        """Return `trial` with attempt 0 appended: S1 when the reply holds the expected files, else S0.

        The prompt is generate.txt from the recipe's prompt set, filled with
        the direction's languages, the item's source files in FILE blocks, and
        the expected target file names; it is kept in the text store. The
        backend gets it as the only message.
        """
        context = self.context
        sources = context.suite.source_files(context.item, context.direction, context.sources_root, purpose=PURPOSE)
        expected = target_files(context)
        fields = {
            "source_language": context.direction.source,
            "target_language": context.direction.target,
            "source_files": render_file_blocks(sources),
            "target_files": ", ".join(expected),
        }
        ref, reply = _ask(context, render(context.prompts, "generate", fields))
        return trial.with_attempt(_parsed_attempt(0, ref, reply, expected))

    def describe(self) -> str:
        """Return a one-line description of the stage."""
        direction = self.context.direction
        return (
            f"generate: one {direction.source} to {direction.target} translation from "
            f"{self.context.prompts}/generate.txt, parsed from FILE blocks"
        )


@register("Stage", "compile_loop")
class CompileLoopStage:
    """Builds the last attempt and asks for corrections while an error remains, up to the correction cap."""

    name = "compile_loop"
    capabilities = frozenset({"compiles"})
    requires = {"Toolchain": {"diagnostics"}}
    prompt_fields = {"correct": CORRECT_FIELDS}

    def __init__(self, *, context: RunContext) -> None:
        """Keep the trial's run context."""
        self.context = context

    def __call__(self, trial: Trial) -> Trial:
        """Return `trial` with its last attempt built and any corrections appended.

        The last attempt, when it has files and no FILE-block error, is built
        with the target language's toolchain in a fresh build directory; its
        copy gets the compile diagnostics after the parse diagnostics, and S4
        when an artifact was built. A build with no artifact always carries an
        error: "no-artifact" when the toolchain reported none, and
        "unwritable" when a file name could not be written. While the latest
        attempt has an error, is not S4, and fewer corrections than the cap
        have been made (attempts after the first), the stage renders
        correct.txt from that attempt's files and diagnostics, sends it to the
        backend as the only message, appends the parsed reply with its diff
        from the previous files, and builds that. A cap of None never stops
        the loop.
        """
        if not trial.attempts:
            raise ValueError(f"{trial.trial_id}: compile_loop needs an attempt; run the generate stage first")
        cap = self.context.max_corrections
        expected = target_files(self.context)
        trial = self._build_last(trial)
        while self._needs_correction(trial.attempts[-1]):
            if cap is not None and len(trial.attempts) - 1 >= cap:
                break
            trial = self._build_last(trial.with_attempt(self._correction(trial.attempts[-1], expected)))
        return trial

    def describe(self) -> str:
        """Return a one-line description of the stage."""
        cap = self.context.max_corrections
        limit = "no correction cap" if cap is None else f"at most {cap} corrections"
        toolchain = self.context.toolchains.get(self.context.direction.target)
        builder = "no bound toolchain" if toolchain is None else getattr(toolchain, "name", type(toolchain).__name__)
        return (
            f"compile_loop: build {self.context.direction.target} with {builder}, then correct from "
            f"{self.context.prompts}/correct.txt while an error remains ({limit})"
        )

    def _toolchain(self) -> Toolchain:
        """Return the toolchain bound to the direction's target language."""
        language = self.context.direction.target
        toolchain = self.context.toolchains.get(language)
        if toolchain is None:
            bound = ", ".join(sorted(self.context.toolchains)) or "none"
            raise ValueError(f"no toolchain is bound for the target language {language!r}; bound: {bound}")
        return toolchain

    @staticmethod
    def _needs_correction(attempt: Attempt) -> bool:
        """Return True when the attempt did not build and has an error to feed back."""
        return attempt.stage_reached != COMPILED and _has_error(attempt.diagnostics)

    def _build_last(self, trial: Trial) -> Trial:
        """Return `trial` with its last attempt built, or unchanged unless that attempt is S1 with no error.

        An S0 attempt has no usable files, and an attempt that is S4 or holds
        a compile error was built already.
        """
        attempt = trial.attempts[-1]
        if attempt.stage_reached != PARSED or _has_error(attempt.diagnostics):
            return trial
        workdir = fresh_build_dir(self.context.build_root, trial.trial_id, attempt.index)
        result = self._build(attempt.files, workdir)
        diagnostics = [*attempt.diagnostics, *result.diagnostics]
        if result.artifact is None and not _has_error(result.diagnostics):
            message = "the build produced no program and the toolchain reported no error; return the complete files"
            diagnostics.append(_compile_error("no-artifact", message))
        built = dataclasses.replace(
            attempt,
            diagnostics=diagnostics,
            stage_reached=COMPILED if result.artifact is not None else attempt.stage_reached,
        )
        return dataclasses.replace(trial, attempts=[*trial.attempts[:-1], built])

    def _build(self, files: Mapping[str, str], workdir: Path) -> BuildResult:
        """Build `files` in `workdir`; a file name the filesystem refuses becomes an "unwritable" error.

        Only a name error (too long, or refused as invalid) is the model's to
        fix; any other OSError, such as a full disk, propagates and stops the
        run.
        """
        try:
            return self._toolchain().build(files, workdir)
        except UnicodeError as error:
            reason = f"a file is not valid Unicode text ({error.reason})"
        except OSError as error:
            if error.errno not in _NAME_ERRNOS:
                raise
            reason = f"the filesystem refused a file name ({error.strerror})"
        message = (
            f"the files could not be written to the build directory: {reason}. Name each file with a short "
            "relative path of letters, digits, '.', '_', '-', and '/', and return the files again"
        )
        return BuildResult(artifact=None, diagnostics=[_compile_error("unwritable", message)])

    def _correction(self, previous: Attempt, expected: Sequence[str]) -> Attempt:
        """Return the next attempt: correct.txt filled from `previous`, the model's reply parsed, and the diff."""
        fields = {
            "target_language": self.context.direction.target,
            "files": render_file_blocks(previous.files) if previous.files else "",
            "diagnostics": "\n".join(diagnostic_line(diagnostic) for diagnostic in previous.diagnostics),
            "target_files": ", ".join(expected),
        }
        ref, reply = _ask(self.context, render(self.context.prompts, "correct", fields))
        attempt = _parsed_attempt(previous.index + 1, ref, reply, expected)
        return dataclasses.replace(attempt, diff_from_previous=unified_diff(previous.files, attempt.files))
