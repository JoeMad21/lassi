"""The stages of the compile-only path: summarize_context, describe_source, generate, and compile_loop.

Bible Component Interfaces, Stage row. Stages are pure over the trial record
(Stage contract rules): a stage reads the Trial's fields, appends an attempt,
annotates the last one, or fills Trial.context, and returns a new Trial; the
Trial it was given is never changed. Side effects go through components: the
LLM backend, the toolchain, and the text store that keeps each prompt. Each
stage is built as `factory(context=<RunContext>)` once per trial
(lassi.core.registry states the construction convention), so a stage object
never carries state from one trial to the next.

Prompts are data files, never string literals here (Design Principle 3). A
template set (lassi.prompts.render) holds `generate.txt` and `correct.txt`;
each stage class declares the templates it renders and their fields in
`prompt_fields`. A fragment set (lassi.prompts.load_recipe_assets, which
the runner calls) holds checked fragments that lassi.core.fragments joins;
each stage class declares the fragment keys it reads in `fragment_keys`,
and `needs_context` when it needs a context pack for the target language.
The runner checks either kind before any model is asked.

- summarize_context sends [system, user]: the general system prompt, then
  the summary request followed by the target language's context pack. The
  reply fills Trial.context.knowledge_summary.
- describe_source sends the general system prompt and the description
  request followed by the source. The reply fills
  Trial.context.source_description.
- generate appends attempt 0. With a template set it sends generate.txt,
  filled with the source files in FILE blocks, as the only message; with a
  fragment set it sends the direction's system prompt and the generation
  prompt (lassi.core.fragments.generation_prompt), the source read as text
  mode reads it.

A context stage names the Trial.context field it fills in `fills_context`,
and generate names the fields its fragment prompt joins, when a pack serves
the target, in `joins_context`; the runner checks that an earlier stage
fills each joined field. A context reply is kept as returned, except that
each lone surrogate becomes U+FFFD, as in an attempt's reply (it is not
Unicode text and cannot be stored). Trial.context carries no diagnostics,
so generate adds to attempt 0 one `invalid-text` warning per context field
that holds U+FFFD.

Two upstream quirks are reproduced when their fixes (lassi.core.recipe.FIXES)
are off, and each stage class names the fixes it reproduces in `reproduces`:

- `prompt_spaces` off: generate cuts every run of spaces in its prompt to
  one space before sending it.
- `fence_tag` off: generate reads the reply's first fenced block with
  upstream's tag stripping as the target's one file (S1; S0 with an empty
  file and a `no-fence` warning when there is no block). A stripping that
  leaves text on the block's first line adds a parse-stage warning
  Diagnostic with code `fence-quirk`. With the fix on, the model answers
  with FILE blocks, which lassi.core.files parses.

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
from dataclasses import dataclass, field
from pathlib import Path

from lassi.bench import Direction, Suite
from lassi.core import fragments as fragment_text
from lassi.core.files import parse_file_blocks, render_file_blocks
from lassi.core.interfaces import BuildResult, Executor, LLMBackend, Message, Sampling, Toolchain
from lassi.core.recipe import Recipe
from lassi.core.record import Attempt, Context, Diagnostic, TextRef, Trial, unified_diff
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
    `max_corrections` the correction cap (None means uncapped). `fragments`
    holds the fragments of a fragment prompt set (empty for a template set),
    and `packs` the recipe's context packs, keyed by the language each
    serves.
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
    fragments: Mapping[str, str] = field(default_factory=dict)
    packs: Mapping[str, str] = field(default_factory=dict)


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


def fix_on(context: RunContext, name: str) -> bool:
    """Return True unless the recipe turns the named fix off (faithful: true turns every fix off)."""
    return context.recipe.data.get("fixes", {}).get(name, True) is not False


def source_as_read(context: RunContext) -> str:
    """Return the item's one source file in the direction's source language, as text mode reads it.

    A fragment prompt set joins one source text into its prompts, read as a
    file opened in text mode reads it (CRLF and CR become LF). Raises
    ValueError when the item has more or fewer than one source file.
    """
    sources = context.suite.source_files(context.item, context.direction, context.sources_root, purpose=PURPOSE)
    if len(sources) != 1:
        raise ValueError(
            f"{context.suite.name}/{context.item}: a fragment prompt set takes one source file, "
            f"the {context.direction.source!r} version has {len(sources)}"
        )
    return fragment_text.as_text_mode(next(iter(sources.values())))


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


def _parse_warning(code: str, message: str) -> Diagnostic:
    """Return a parse-stage warning Diagnostic with no location."""
    return Diagnostic(stage="parse", severity="warning", code=code, message=message)


def _context_text(reply: str) -> str:
    """Return a context reply with each lone surrogate replaced by U+FFFD, so it can be stored and sent."""
    return _SURROGATE.sub(_REPLACEMENT, reply)


def _context_warnings(context: Context) -> list[Diagnostic]:
    """Return one `invalid-text` warning per Trial.context field that holds U+FFFD, for attempt 0 to carry.

    The context stages replace each lone surrogate in a reply with U+FFFD
    (_context_text), and Trial.context has no diagnostics of its own, so
    attempt 0 keeps the note. A reply may also hold U+FFFD itself; the
    message states only what the field holds.
    """
    warnings = []
    for name in ("knowledge_summary", "source_description"):
        count = getattr(context, name).count(_REPLACEMENT)
        if count:
            message = (
                f"Trial.context.{name} holds {count} U+FFFD replacement character(s); a context stage writes one "
                "for each lone surrogate in its reply, which is not Unicode text"
            )
            warnings.append(_parse_warning("invalid-text", message))
    return warnings


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


def _fenced_attempt(index: int, prompt_ref: TextRef, reply: str, expected: Sequence[str]) -> Attempt:
    """Return the attempt for one reply read as upstream reads it: the first fenced block is the one target file.

    S1 when the reply holds a fenced block, else S0 with an empty file (upstream
    writes and builds one) and a `no-fence` warning. A tag stripping that
    leaves text on the block's first line adds a `fence-quirk` warning.
    """
    if len(expected) != 1:
        raise ValueError(f"the fence_tag quirk reads one fenced block, but the target has {len(expected)} files")
    text, diagnostics = _storable(reply)
    fence = fragment_text.first_fence(text)
    if fence.quirk:
        diagnostics.append(_parse_warning(fragment_text.FENCE_QUIRK, fragment_text.fence_quirk_message(fence)))
    if not fence.found:
        message = "the reply held no fenced block, so the target file is empty, as upstream writes it"
        diagnostics.append(_parse_warning(fragment_text.NO_FENCE, message))
    return Attempt(
        index=index,
        prompt_ref=prompt_ref,
        response_text=text,
        files={expected[0]: fence.text},
        diff_from_previous="",
        stage_reached=PARSED if fence.found else NO_OUTPUT,
        diagnostics=diagnostics,
    )


def _reply(context: RunContext, prompt: str, system: str | None = None) -> str:
    """Send `prompt` as the user message, after `system` as the system message when given; return the reply text."""
    messages = [Message("user", prompt)] if system is None else [Message("system", system), Message("user", prompt)]
    return context.backend.complete(messages, context.sampling).text


def _ask(context: RunContext, prompt: str, system: str | None = None) -> tuple[TextRef, str]:
    """Store `prompt`, send it as _reply does, and return its reference and the reply text."""
    ref = context.store.put(prompt)
    return ref, _reply(context, prompt, system)


@register("Stage", "generate")
class GenerateStage:
    """Appends attempt 0: the model's first translation of the item's source files, parsed from FILE blocks."""

    name = "generate"
    capabilities = frozenset({"generates"})
    requires = {"LLMBackend": {"chat"}}
    prompt_fields = {"generate": GENERATE_FIELDS}
    fragment_keys = fragment_text.GENERATE_KEYS
    reproduces = frozenset({"fence_tag", "prompt_spaces"})
    joins_context = ("knowledge_summary", "source_description")

    def __init__(self, *, context: RunContext) -> None:
        """Keep the trial's run context."""
        self.context = context

    def __call__(self, trial: Trial) -> Trial:
        """Return `trial` with attempt 0 appended: S1 when the reply yields the target files, else S0.

        The prompt comes from _prompt; with the prompt_spaces fix off, each
        run of spaces in it is cut to one. It is kept in the text store and
        sent as the user message, after the system prompt when there is one.
        With the fence_tag fix on the reply's FILE blocks are parsed, and with
        it off its first fenced block is read as upstream reads it. The
        attempt's diagnostics start with _context_warnings.
        """
        context = self.context
        expected = target_files(context)
        system, prompt = self._prompt(trial, expected)
        if not fix_on(context, "prompt_spaces"):
            prompt = fragment_text.collapse_spaces(prompt)
        ref, reply = _ask(context, prompt, system)
        if fix_on(context, "fence_tag"):
            attempt = _parsed_attempt(0, ref, reply, expected)
        else:
            attempt = _fenced_attempt(0, ref, reply, expected)
        notes = _context_warnings(trial.context)
        if notes:
            attempt = dataclasses.replace(attempt, diagnostics=[*notes, *attempt.diagnostics])
        return trial.with_attempt(attempt)

    def _prompt(self, trial: Trial, expected: Sequence[str]) -> tuple[str | None, str]:
        """Return the system prompt (None for a template set) and the user prompt.

        A template set gives generate.txt, filled with the direction's
        languages, the item's source files in FILE blocks, and the expected
        target file names. A fragment set gives the direction's system prompt
        and the generation prompt from the source (as text mode reads it),
        the target language's context pack when the recipe has one, and the
        trial's summary and description.
        """
        context = self.context
        if not context.fragments:
            sources = context.suite.source_files(
                context.item, context.direction, context.sources_root, purpose=PURPOSE
            )
            fields = {
                "source_language": context.direction.source,
                "target_language": context.direction.target,
                "source_files": render_file_blocks(sources),
                "target_files": ", ".join(expected),
            }
            return None, render(context.prompts, "generate", fields)
        direction = context.direction
        prompt = fragment_text.generation_prompt(
            context.fragments,
            direction,
            source_as_read(context),
            context.packs.get(direction.target),
            trial.context.knowledge_summary,
            trial.context.source_description,
        )
        return context.fragments[fragment_text.fragment_key(fragment_text.DIRECTION_SYSTEM, direction)], prompt

    def describe(self) -> str:
        """Return a one-line description of the stage."""
        direction = self.context.direction
        prompts = self.context.prompts
        source = f"the {prompts} fragments" if self.context.fragments else f"{prompts}/generate.txt"
        reply = "FILE blocks" if fix_on(self.context, "fence_tag") else "its first fenced block"
        return f"generate: one {direction.source} to {direction.target} translation from {source}, parsed from {reply}"


@register("Stage", "summarize_context")
class SummarizeContextStage:
    """Asks the model to summarize the target language's context pack and keeps the reply in Trial.context."""

    name = "summarize_context"
    capabilities: frozenset[str] = frozenset()
    requires = {"LLMBackend": {"chat"}}
    prompt_fields: dict[str, tuple[str, ...]] = {}
    fragment_keys = fragment_text.SUMMARY_KEYS
    needs_context = True
    fills_context = ("knowledge_summary",)

    def __init__(self, *, context: RunContext) -> None:
        """Keep the trial's run context."""
        self.context = context

    def __call__(self, trial: Trial) -> Trial:
        """Return `trial` with context.knowledge_summary set to the model's reply; no attempt.

        The general system prompt and the summary request, followed by the
        target language's pack, go to the backend unchanged. The reply is
        kept as returned, except that each lone surrogate becomes U+FFFD
        (_context_text).
        """
        context = self.context
        pack = context.packs[context.direction.target]
        prompt = fragment_text.summary_request(context.fragments, context.direction, pack)
        reply = _context_text(_reply(context, prompt, context.fragments[fragment_text.GENERAL_SYSTEM]))
        return dataclasses.replace(trial, context=dataclasses.replace(trial.context, knowledge_summary=reply))

    def describe(self) -> str:
        """Return a one-line description of the stage."""
        return f"summarize_context: summarize the {self.context.direction.target} context pack ({self.context.prompts})"


@register("Stage", "describe_source")
class DescribeSourceStage:
    """Asks the model to describe the item's source and keeps the reply in Trial.context."""

    name = "describe_source"
    capabilities: frozenset[str] = frozenset()
    requires = {"LLMBackend": {"chat"}}
    prompt_fields: dict[str, tuple[str, ...]] = {}
    fragment_keys = fragment_text.DESCRIPTION_KEYS
    fills_context = ("source_description",)

    def __init__(self, *, context: RunContext) -> None:
        """Keep the trial's run context."""
        self.context = context

    def __call__(self, trial: Trial) -> Trial:
        """Return `trial` with context.source_description set to the model's reply; no attempt.

        The general system prompt and the description request, followed by
        the source as text mode reads it, go to the backend unchanged. The
        reply is kept as returned, except that each lone surrogate becomes
        U+FFFD (_context_text).
        """
        context = self.context
        prompt = fragment_text.description_request(context.fragments, source_as_read(context))
        reply = _context_text(_reply(context, prompt, context.fragments[fragment_text.GENERAL_SYSTEM]))
        return dataclasses.replace(trial, context=dataclasses.replace(trial.context, source_description=reply))

    def describe(self) -> str:
        """Return a one-line description of the stage."""
        return f"describe_source: describe the {self.context.direction.source} source ({self.context.prompts})"


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

        The item's support files from the suite manifest go into the build
        directory as harness files, passed as the toolchain's `harness`
        argument only when the item has some. Only a name error (too long, or
        refused as invalid) is the model's to fix; any other OSError, such as
        a full disk, propagates and stops the run.
        """
        context = self.context
        harness = context.suite.support_files(context.item, context.sources_root, purpose=PURPOSE)
        try:
            if harness:
                return self._toolchain().build(files, workdir, harness=harness)
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
