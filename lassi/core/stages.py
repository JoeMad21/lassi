"""The LASSI stages: baseline, summarize_context, describe_source, generate, compile_loop, and run_loop.

Bible Component Interfaces, Stage row. Stages are pure over the trial record
(Stage contract rules): a stage reads the Trial's fields, appends an attempt,
annotates the last one, fills Trial.context, or records a model request, and
returns a new Trial; the Trial it was given is never changed. Side effects go
through components: the LLM backend, the toolchain, and the text store that
keeps each message and reply. Each stage is built as
`factory(context=<RunContext>)` once per trial (lassi.core.registry states
the construction convention), so a stage object never carries state from one
trial to the next.

Prompts are data files, never string literals here (Design Principle 3). A
template set (lassi.prompts.render) holds `generate.txt` and `correct.txt`;
each stage class declares the templates it renders and their fields in
`prompt_fields`. A fragment set (lassi.prompts.load_recipe_assets, which
the runner calls) holds checked fragments that lassi.core.fragments joins;
each stage class declares the fragment keys it reads in `fragment_keys`,
and `needs_context` when it needs a context pack for the target language.
The runner checks either kind before any model is asked.

- baseline builds the item's target reference program in a fresh
  directory under the trial's directory (<trial>/baseline-<language>/build,
  never an attempt's), with the target language's toolchain and the item's
  support files as harness files, and runs it when the executor runs
  programs (capability `runs_code`), with the item's run arguments and
  reference_limits. The target's run is kept in Trial.reference_run (exit
  code, hang flag, wall time, stdout in the text store, and the RunResult
  flags as bools); the source reference's run is never recorded. A cut
  output does not end the trial. A reference that does not build ends the
  trial with final.end_reason `baseline-compile`, and a run that exits
  nonzero or hangs with `baseline-run`; the runner then runs no later stage,
  so no model is asked. When the recipe's Oracle compares output files
  (capability aligns_output_files, output_file_oracle), the target's output
  files are also kept in the binary store (RunInfo.outputs), a reference
  run whose output files that Oracle cannot compare against (an unreadable
  file, two files holding one array name, or none) ends the trial with
  `baseline-run` (unless its workdir came back incomplete), and, when both
  references ran and the item declares a tolerance, the source reference's
  agreement with the target reference is recorded in
  Trial.reference_agreement; an output outside that tolerance ends the trial
  with `baseline-disagree`. When the agreement would be measured but either
  reference run's workdir came back incomplete, no agreement is recorded and
  Trial.baseline_diagnostics gains a `reference-workdir-incomplete` warning.
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
- compile_loop builds the last attempt and, while an error remains, asks
  for a correction and builds that, up to loop.max_corrections (none when
  uncapped). A cap hit with an error remaining ends the trial with
  final.end_reason `correction-cap`. With a template set the correction
  prompt is correct.txt, sent as the only message; with a fragment set it
  is upstream's (lassi.core.fragments.correction_prompt), sent after the
  direction's system prompt. Its error text is the parsed diagnostics,
  capped at DIAGNOSTIC_COUNT_CAP diagnostics and DIAGNOSTIC_BYTES_CAP bytes
  of whole lines, with a line saying how many were left out when any were
  (diagnostics_text).
- run_loop continues compile_loop's loop (it runs compile_loop first, which
  changes nothing when compile_loop already ran) and, when the executor runs
  programs (capability `runs_code`), runs each compiling attempt from its
  build directory with the item's run arguments and attempt_limits. The run
  is kept in Attempt.run (exit code, hang flag, wall time, stdout in the text
  store, the RunResult flags as bools, and, when the recipe's Oracle compares
  output files, every output file in the binary store by hash; a failed
  run's included); a clean
  run (exit status 0, no hang) is S5, and a failed one stays S4 with a
  run-stage `run-error` Diagnostic. A failed run is fed back
  with the execute-error prompt, whose error text (run_error_text) is, under
  a fragment set, upstream's report of the run (its execute_code
  return_result, joined from the execute.* fragments), and under a template
  set this project's wording, which correct.txt carries. The reply is built
  and corrected as compile_loop does. Model-generated code runs only in the
  sandbox (Agent Rule 6): run_loop refuses an executor that runs programs
  but does not declare `sandboxed`, and so does the runner, before any
  directory exists (`runs_model_code`). Run and compile corrections share
  one count and the cap: a run error left when the cap is reached ends the
  trial with `correction-cap`. Each RunResult flag becomes a run-stage warning
  (RUN_FLAGS). With a compile-only executor run_loop runs nothing, so the
  trial ends at its first compiling attempt. A backend that declares
  `unload_before_run` (lassi.core.capabilities) is asked to unload right
  before each run of an attempt; the runner asks it once more at trial
  start. compile_loop hands each built program to run_loop through the
  trial's RunContext (RunContext.artifacts), in memory and not in the
  record, so an attempt cannot be run again from the record alone.

A context stage names the Trial.context field it fills in `fills_context`,
and generate names the fields its fragment prompt joins, when a pack serves
the target, in `joins_context`; the runner checks that an earlier stage
fills each joined field. A context reply is kept as returned, except that
each lone surrogate becomes U+FFFD, as in an attempt's reply (it is not
Unicode text and cannot be stored).

Every model call is recorded in Trial.requests (lassi.core.record Request):
the stage that sends it keeps each message it sends in the text store
before sending, system messages included, and appends a Request with those
references, the attempt its reply became (None for a context request), and
the reply as kept. A correction is recorded under the loop stage that asked
for it: compile_loop for a compile error, run_loop for a failed run. An
attempt's own `invalid-text` warning stays on the attempt; Trial.context
carries no diagnostics, so a context reply that held a lone surrogate gives
its request one parse-stage `invalid-text` warning.

Upstream quirks are reproduced when their fixes (lassi.core.recipe.FIXES)
are off, and each stage class names the fixes it reproduces in
`reproduces`. Every quirk is keyed on its fix (or on loop.max_corrections),
never on the recipe's `faithful` flag:

- `baseline_both` off: baseline builds and runs only the target reference,
  as upstream does; on, it then builds and runs the source reference too,
  with the source language's toolchain (the runner refuses the recipe when
  none is bound).
- `prompt_spaces` off: generate cuts every run of spaces in its prompt to
  one space before sending it.
- `fence_tag` off: generate, and compile_loop for each correction, reads
  the reply's first fenced block with upstream's tag stripping as the
  target's one file (S1; S0 with an empty file and a `no-fence` warning
  when there is no block). A stripping that leaves text on the block's
  first line adds a parse-stage warning Diagnostic with code `fence-quirk`.
  compile_loop builds such an S0 attempt's empty file, as upstream compiles
  the empty file it writes. With the fix on, the model answers with FILE
  blocks, which lassi.core.files parses, and an attempt with a FILE-block
  error (S0, or S1 missing an expected file) is not built.
- `parsed_diagnostics` off: a correction prompt's error text is the whole
  raw stderr attachment of the build (BuildResult.stderr_ref), read as a
  file opened in text mode reads it (CRLF and CR become LF), uncapped, as
  upstream sends the compiler's stderr. An attempt whose build kept no
  attachment (or that was not built) sends its parsed diagnostics.
- `prompt_newlines` off: every line feed is removed from a correction
  prompt before it is stored and sent (a carriage return stays), as
  upstream removes them.
- `execution_gate` off: run_loop runs a compiling attempt only while its
  correction count (its index) is at most EXECUTION_GATE_CORRECTIONS (7),
  as upstream's loop does. A compiling attempt with a higher count ends the
  trial unexecuted. When an earlier attempt ran, the last such attempt's
  stdout stands as the trial's output, and the last attempt carries a
  run-stage `stale-output` warning naming it; when none ran, the trial ends
  with `upstream-crash`, where upstream's notebook raises. With the fix on,
  every compiling attempt runs, within the correction cap.

Stage reached (Result Record, Attempt.stage_reached) is the bible's stage
ladder (Training Module, Reward Function), the one scale every record, metric,
and reward reads (Design Principle 2):

- S0, no extractable output: the FILE blocks gave no file (the reply held
  no FILE block, or each was dropped), or they had an error other than a
  missing expected file (a bad path, or a repeated or unclosed block).
- S1, parses: the FILE blocks gave at least one file, and every FILE-block
  error is `missing-file`, an expected file with no block. That error is
  the build error of the Harness Contract: compile_loop feeds it back to
  the model in the correction prompt without building the incomplete
  files, so the attempt stays S1.
- S2, verifies, and S3, lowers: the MLIR verifier and the lowering passes.
  Source-level translation has neither step, so these stages never record
  S2 or S3.
- S4, compiles: the toolchain built the files into an artifact.
- S5, runs clean: the artifact ran with no crash, undefined behavior, or
  hang; the run loop records it.

Output agreement with the oracle is never a stage: it is Attempt.alignment
and Final.alignment, which the reward weighs separately. Only run_loop runs
an attempt, and only on an executor that runs programs; any other attempt
keeps RunInfo at its defaults (None) and reaches at most S4, and a
compile-only executor's placeholder values are never copied into the record.
"""

from __future__ import annotations

import dataclasses
import errno
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from lassi.bench import Direction, Suite
from lassi.core import fragments as fragment_text
from lassi.core.capabilities import ALIGNS_OUTPUT_FILES, OutputFileOracle, declares, unload_before_run
from lassi.core.files import parse_file_blocks, render_file_blocks
from lassi.core.interfaces import BuildResult, Executor, Limits, LLMBackend, Message, RunResult, Sampling, Toolchain
from lassi.core.recipe import Recipe
from lassi.core.record import (
    Attempt,
    Diagnostic,
    EndReason,
    Request,
    RequestMessage,
    RunInfo,
    TextRef,
    Trial,
    standing_attempt,
    unified_diff,
)
from lassi.core.registry import DEFAULT_REGISTRY, register
from lassi.core.store import BlobStore, TextStore
from lassi.executors.workdir import build_dir, fresh_build_dir
from lassi.prompts import render

# The purpose every bench read here serves: these stages evaluate, they never train (Agent Rule 5).
PURPOSE = "eval"

# The rungs of the stage ladder these stages record (see the module docstring).
NO_OUTPUT = "S0"
PARSED = "S1"
COMPILED = "S4"
RAN_CLEAN = "S5"
# The FILE-block error code (lassi.core.files parse_file_blocks) of an expected file with no block: the one block
# error an S1 attempt may hold (see the module docstring).
MISSING_FILE = "missing-file"

# The end codes these stages set in final.end_reason (lassi.core.record END_REASONS).
BASELINE_COMPILE = "baseline-compile"
BASELINE_RUN = "baseline-run"
BASELINE_DISAGREE = "baseline-disagree"
CORRECTION_CAP = "correction-cap"
UPSTREAM_CRASH = "upstream-crash"

# The capability of an executor that runs programs; baseline runs a reference only on such an executor.
RUNS_CODE = "runs_code"
# The capability of an executor that runs programs only inside the sandbox (Execution Backends, Sandbox). A stage
# that runs model-generated code (`runs_model_code`) needs it whenever the executor runs programs (Agent Rule 6).
SANDBOXED = "sandboxed"

# The caps on the parsed diagnostics one correction prompt carries [DESIGN]: at most DIAGNOSTIC_COUNT_CAP
# diagnostics, whose lines take at most DIAGNOSTIC_BYTES_CAP bytes of UTF-8 (one line feed after each line counted).
# Only whole lines are kept, the first ones in order, and the prompt says how many it left out. A compile keeps up
# to 64 MiB of stderr (lassi.executors.sandbox COMPILE_OUTPUT_CAP_BYTES), so an uncapped prompt could be that large.
DIAGNOSTIC_COUNT_CAP = 50
DIAGNOSTIC_BYTES_CAP = 16384

# The wall limit and CPU count of a baseline reference run [DESIGN]; its memory limit is the recipe's
# sandbox.mem_gb. Upstream runs the reference with no limit at all, and the recipe's sandbox.wall_s
# (baseline_x10) is derived from this run's wall time, so it cannot bound the run itself.
REFERENCE_WALL_S = 600.0
REFERENCE_CPUS = 16

# The highest correction count at which upstream's loop runs a compiling attempt: the execution gate that
# fixes.execution_gate turns off. A compiling attempt with a higher count ends the loop unexecuted.
EXECUTION_GATE_CORRECTIONS = 7

# The sandbox.wall_s value (projects/base.yaml) that makes an attempt's run wall limit BASELINE_FACTOR times the
# wall time of the trial's reference run (Trial.reference_run.wall_s); a number there is the limit in seconds.
BASELINE_X10 = "baseline_x10"
BASELINE_FACTOR = 10
# The least wall limit of an attempt run under baseline_x10, in seconds [DESIGN]. Ten times a reference run of a few
# milliseconds would stop a correct program on start-up noise, and a trial whose reference did not run (a
# compile-only baseline, or no baseline stage) has no wall time to multiply, so its limit is this floor.
RUN_WALL_FLOOR_S = 30.0
# The CPU count of an attempt run [DESIGN]: the reference run's, so a translation runs with the threads its
# reference had (an executor sets OMP_NUM_THREADS to Limits.cpus).
RUN_CPUS = REFERENCE_CPUS

# The run-stage warning code of a reference run whose workdir came back incomplete (RunResult.workdir_incomplete)
# under an Oracle that compares output files: the baseline notes it in Trial.baseline_diagnostics when it skips
# the references' agreement, and the oracle stage puts it on each attempt it does not align against a cut target.
REFERENCE_WORKDIR_INCOMPLETE = "reference-workdir-incomplete"

# The run-stage Diagnostic codes run_loop sets: a failed run (error) and stale output (warning).
RUN_ERROR = "run-error"
STALE_OUTPUT = "stale-output"
# Each RunResult flag run_loop records, with the code and the message of the run-stage warning it becomes.
RUN_FLAGS = {
    "stdout_truncated": (
        "stdout-truncated",
        "the run's stdout passed the executor's output cap, so only the part kept is recorded and aligned",
    ),
    "stderr_truncated": (
        "stderr-truncated",
        "the run's stderr passed the executor's output cap, so only the part kept is fed back",
    ),
    "workdir_incomplete": (
        "workdir-incomplete",
        "the executor returned only part of what the run wrote in its workdir, because the run passed a cap or limit",
    ),
}

# The fragment keys of upstream's execute-error prompt: the lead before and after its compiler and flag text.
CORRECT_RUN_HEAD = "correct.run_error_head"
CORRECT_RUN_TAIL = "correct.run_error_tail"
# The fragment keys of upstream's report of a failed run, the error text of its execute-error prompt (the notebook's
# execute_code return_result, tools/extract_lassi_assets.py): the lead before the return code, the note added for
# POPEN_SEGFAULT, and the lead before the stderr. The report's other literals (its success value and its exception
# lead) have no counterpart here: a run that ends reports no exception.
RUN_REPORT_EXIT = "execute.exit_lead"
RUN_REPORT_SEGFAULT = "execute.segfault"
RUN_REPORT_STDERR = "execute.stderr_lead"
# The return code Popen gives a death by SIGSEGV, for which upstream's report adds its segfault note.
POPEN_SEGFAULT = -11
# An executor that runs programs reports a death by signal N as SHELL_SIGNAL_BASE + N, the shell's form
# (lassi.executors.sandbox SandboxResult.returncode); Popen, which upstream reads, gives -N. Signal numbers run to
# MAX_SIGNAL (Linux, real-time signals included).
SHELL_SIGNAL_BASE = 128
MAX_SIGNAL = 64
# The fragment keys run_loop reads: compile_loop's, for the compile errors it corrects, the execute-error lead, and
# the run report.
RUN_LOOP_KEYS = (
    *fragment_text.CORRECT_KEYS,
    CORRECT_RUN_HEAD,
    CORRECT_RUN_TAIL,
    RUN_REPORT_EXIT,
    RUN_REPORT_SEGFAULT,
    RUN_REPORT_STDERR,
)

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
    serves. `artifacts` maps the index of each attempt a build turned into a
    program to that program (BuildResult.artifact): compile_loop fills it and
    run_loop runs from it. The runner builds one RunContext per trial, so it
    never carries an artifact from one trial to the next.
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
    artifacts: dict[int, Path] = field(default_factory=dict)


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


def diagnostics_text(diagnostics: Sequence[Diagnostic]) -> str:
    """Return the error text a correction prompt carries with fixes.parsed_diagnostics on.

    One diagnostic_line per diagnostic, in order, joined by line feeds: the
    first ones up to DIAGNOSTIC_COUNT_CAP of them and DIAGNOSTIC_BYTES_CAP
    bytes (each line's UTF-8 bytes plus one line feed), whole lines only.
    When any is left out, a last line says how many; when none is, the text
    is exactly the joined lines.
    """
    lines: list[str] = []
    used = 0
    for diagnostic in diagnostics:
        line = diagnostic_line(diagnostic)
        size = len(line.encode("utf-8", "surrogatepass")) + 1
        if len(lines) == DIAGNOSTIC_COUNT_CAP or used + size > DIAGNOSTIC_BYTES_CAP:
            break
        lines.append(line)
        used += size
    left_out = len(diagnostics) - len(lines)
    if left_out:
        lines.append(f"[{left_out} more diagnostic(s) not shown]")
    return "\n".join(lines)


def reference_limits(context: RunContext) -> Limits:
    """Return the limits of a baseline reference run: REFERENCE_WALL_S, the recipe's sandbox.mem_gb, REFERENCE_CPUS."""
    memory_mb = round(float(context.recipe.data["sandbox"]["mem_gb"]) * 1024)
    return Limits(wall_s=REFERENCE_WALL_S, memory_mb=memory_mb, cpus=REFERENCE_CPUS)


def _run_info(context: RunContext, run: RunResult, keep_outputs: bool = False) -> RunInfo:
    """Return the RunInfo of a run that happened: exit code, hang flag, wall time, stdout in the store, run flags.

    Each RunResult flag (RUN_FLAGS) is copied as a bool, so a run whose
    output was kept whole records False, never None. With `keep_outputs`,
    every output file's bytes go into the binary store beside the text store
    and RunInfo.outputs maps each file to its sha256; otherwise outputs
    stays None (not recorded).
    """
    flags = {flag: bool(getattr(run, flag)) for flag in RUN_FLAGS}
    stdout_ref = context.store.put(run.stdout)
    outputs = _kept_outputs(context, run) if keep_outputs else None
    return RunInfo(
        exit_code=run.exit_code, hang=run.hang, wall_s=run.wall_s, stdout_ref=stdout_ref, outputs=outputs, **flags
    )


def _kept_outputs(context: RunContext, run: RunResult) -> dict[str, str]:
    """Store every output file of `run` in BlobStore(<the text store's root>) and return file -> sha256, sorted."""
    blobs = BlobStore(context.store.root)
    return {file: blobs.put(Path(run.output_files[file]).read_bytes()) for file in sorted(run.output_files)}


def output_file_oracle(context: RunContext) -> OutputFileOracle | None:
    """Return the recipe's Oracle, built from its section, when it declares ALIGNS_OUTPUT_FILES; else None.

    The class is looked up by the binding's name in the default registry,
    as the oracle stage looks it up, since a RunContext carries no registry;
    the runner checks the binding against its own registry when the recipe
    loads. baseline and run_loop keep run output files only when this
    returns an oracle.
    """
    for binding in context.recipe.bindings:
        if binding.interface == "Oracle":
            entry = DEFAULT_REGISTRY.get("Oracle", binding.name)
            if ALIGNS_OUTPUT_FILES in entry.capabilities:
                return cast(OutputFileOracle, entry.factory(**binding.config))
    return None


def _ended(trial: Trial, code: str, message: str) -> Trial:
    """Return `trial` with final.end_reason set to `code` and `message`; the runner then runs no later stage."""
    reason = EndReason(code=code, message=message)
    return dataclasses.replace(trial, final=dataclasses.replace(trial.final, end_reason=reason))


def _toolchain(context: RunContext, language: str) -> Toolchain:
    """Return the toolchain bound to `language`; ValueError naming the bound languages when there is none."""
    toolchain = context.toolchains.get(language)
    if toolchain is None:
        bound = ", ".join(sorted(context.toolchains)) or "none"
        raise ValueError(f"no toolchain is bound for the language {language!r}; bound: {bound}")
    return toolchain


def _toolchain_name(toolchain: Toolchain) -> str:
    """Return a toolchain's registry name, or its class name when it has none."""
    return str(getattr(toolchain, "name", type(toolchain).__name__))


def _harness_build(context: RunContext, toolchain: Toolchain, files: Mapping[str, str], workdir: Path) -> BuildResult:
    """Build `files` in `workdir`, with the item's support files as the `harness` argument when it has any."""
    harness = context.suite.support_files(context.item, context.sources_root, purpose=PURPOSE)
    if harness:
        return toolchain.build(files, workdir, harness=harness)
    return toolchain.build(files, workdir)


def _attachment_text(workdir: Path, ref: str) -> str | None:
    """Return a build's raw stderr attachment as a file opened in text mode reads it; None when it kept none."""
    if not ref:
        return None
    path = workdir / ref
    if not path.is_file():
        return None
    return fragment_text.as_text_mode(path.read_bytes().decode("utf-8", "replace"))


def _baseline_dir(root: Path, trial_id: str, language: str) -> Path:
    """Create and return the fresh build directory of one reference program: <trial>/baseline-<language>/build."""
    path = build_dir(root, trial_id, 0).parent.parent / f"baseline-{language}" / "build"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    return path


def _has_error(diagnostics: Sequence[Diagnostic]) -> bool:
    """Return True when any diagnostic is an error."""
    return any(diagnostic.severity == "error" for diagnostic in diagnostics)


def _compile_error(code: str, message: str) -> Diagnostic:
    """Return a compile-stage error Diagnostic with no location."""
    return Diagnostic(stage="compile", severity="error", code=code, message=message)


def _parse_warning(code: str, message: str) -> Diagnostic:
    """Return a parse-stage warning Diagnostic with no location."""
    return Diagnostic(stage="parse", severity="warning", code=code, message=message)


def _storable(reply: str, before: str = "the FILE blocks were read") -> tuple[str, list[Diagnostic]]:
    """Return the reply with each lone surrogate replaced by U+FFFD, and a warning saying so when there was one.

    A lone surrogate cannot be stored as UTF-8 or written into a source
    file, so it would stop the run; the warning keeps the change visible in
    the record and the next correction prompt. `before` ends its message:
    what the reply went on to be used for.
    """
    count = len(_SURROGATE.findall(reply))
    if not count:
        return reply, []
    message = (
        f"the reply held {count} lone surrogate code point(s), which are not Unicode text; "
        f"each was replaced with U+FFFD before {before}"
    )
    return _SURROGATE.sub(_REPLACEMENT, reply), [_parse_warning("invalid-text", message)]


def _parsed_attempt(index: int, prompt_ref: TextRef, reply: str, expected: Sequence[str]) -> Attempt:
    """Return the attempt for one model reply: its FILE blocks parsed, S1 when usable and S0 otherwise.

    The reply is usable when its blocks gave at least one file and every
    block error is MISSING_FILE. The diagnostics are parse_file_blocks'
    own, after any `invalid-text` warning.
    """
    text, warnings = _storable(reply)
    parsed = parse_file_blocks(text, expected)
    errors = [diagnostic for diagnostic in parsed.diagnostics if diagnostic.severity == "error"]
    usable = bool(parsed.files) and all(error.code == MISSING_FILE for error in errors)
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


def _reply_attempt(
    context: RunContext, index: int, prompt_ref: TextRef, reply: str, expected: Sequence[str]
) -> Attempt:
    """Return the attempt for one reply: FILE blocks with fixes.fence_tag on, the first fenced block with it off."""
    if fix_on(context, "fence_tag"):
        return _parsed_attempt(index, prompt_ref, reply, expected)
    return _fenced_attempt(index, prompt_ref, reply, expected)


def _send(context: RunContext, prompt: str, system: str | None = None) -> tuple[list[RequestMessage], str]:
    """Send `prompt` as the user message, after `system` as the system message when given.

    Each message is kept in the text store before anything is sent. Returns
    the messages as a Request records them (role and text reference, in the
    order sent; the user message is last) and the reply text as returned.
    """
    messages = [Message("user", prompt)] if system is None else [Message("system", system), Message("user", prompt)]
    recorded = [RequestMessage(role=message.role, ref=context.store.put(message.content)) for message in messages]
    return recorded, context.backend.complete(messages, context.sampling).text


def _recorded(
    context: RunContext,
    trial: Trial,
    stage: str,
    attempt_index: int | None,
    messages: Sequence[RequestMessage],
    reply: str,
    diagnostics: Sequence[Diagnostic] = (),
) -> Trial:
    """Return `trial` with the request `stage` sent appended to Trial.requests; `reply` is kept in the text store.

    `reply` is the reply as kept (no lone surrogate), `attempt_index` the
    attempt it became (None for a context request), and `diagnostics` what
    was noted about it that no attempt carries.
    """
    request = Request(
        index=len(trial.requests or []),
        stage=stage,
        attempt_index=attempt_index,
        messages=list(messages),
        reply_ref=context.store.put(reply),
        diagnostics=list(diagnostics),
    )
    return trial.with_request(request)


def _context_reply(
    context: RunContext, trial: Trial, stage: str, name: str, messages: Sequence[RequestMessage], reply: str
) -> Trial:
    """Return `trial` with Trial.context field `name` set to the reply as kept, and the request recorded.

    Each lone surrogate in the reply becomes U+FFFD; when there was one, the
    request carries one parse-stage `invalid-text` warning, since
    Trial.context has no diagnostics of its own.
    """
    text, warnings = _storable(reply, f"the reply was kept in Trial.context.{name}")
    trial = dataclasses.replace(trial, context=dataclasses.replace(trial.context, **{name: text}))
    return _recorded(context, trial, stage, None, messages, text, warnings)


@register("Stage", "baseline")
class BaselineStage:
    """Builds the item's reference programs and runs them, before any model call, as upstream's pipeline does first.

    `source_build_fix` names the fix under which the stage builds the source
    reference too, so the runner can refuse a recipe that binds no toolchain
    for the source language while that fix is on. The capability
    `builds_references` makes the runner refuse the stage when a stage that
    asks the model is listed before it.

    When the recipe's Oracle compares output files (output_file_oracle), each
    reference run's output files must be ones it can compare against
    (reference_problem) unless the run's workdir came back incomplete, the
    target's are kept in the binary store (Trial.reference_run.outputs), and,
    when both references ran and the item declares a tolerance (suite
    manifest), the source reference's agreement with the target reference is
    recorded in Trial.reference_agreement, judged against that tolerance.
    When the agreement would be measured (those same conditions) but either
    reference run's workdir came back incomplete, no agreement is recorded
    and Trial.baseline_diagnostics gains one run-stage warning, code
    REFERENCE_WORKDIR_INCOMPLETE, naming the run.
    """

    name = "baseline"
    capabilities = frozenset({"builds_references"})
    requires = {"Toolchain": {"diagnostics"}}
    prompt_fields: dict[str, tuple[str, ...]] = {}
    reproduces = frozenset({"baseline_both"})
    source_build_fix = "baseline_both"

    def __init__(self, *, context: RunContext) -> None:
        """Keep the trial's run context and the recipe's Oracle when it compares output files."""
        self.context = context
        self.files_oracle = output_file_oracle(context)

    def __call__(self, trial: Trial) -> Trial:
        """Return `trial` with the target reference's run kept, or ended with a baseline end reason.

        The target reference is built, and run when the executor runs
        programs; with the baseline_both fix on, the source reference follows
        (unless the direction's two languages are one). The first failure
        sets final.end_reason and stops the stage. Then the references'
        agreement is measured when it applies (_agreement).
        """
        direction = self.context.direction
        languages = [direction.target]
        if fix_on(self.context, "baseline_both") and direction.source != direction.target:
            languages.append(direction.source)
        runs: dict[str, RunResult] = {}
        for language in languages:
            trial, run = self._reference(trial, language)
            if trial.final.end_reason is not None:
                return trial
            if run is not None:
                runs[language] = run
        return self._agreement(trial, runs)

    def describe(self) -> str:
        """Return a one-line description of the stage."""
        direction = self.context.direction
        both = fix_on(self.context, "baseline_both")
        which = f"the {direction.target} and {direction.source}" if both else f"only the {direction.target}"
        runs = RUNS_CODE in getattr(self.context.executor, "capabilities", ())
        action = "build and run" if runs else "build"
        text = f"baseline: {action} {which} reference program(s) before any model call"
        if runs and self.files_oracle is not None:
            text += ", keeping their output files for the oracle"
        return text

    def _reference(self, trial: Trial, language: str) -> tuple[Trial, RunResult | None]:
        """Build the item's reference in `language` and run it when the executor runs programs.

        Returns the trial and the RunResult (None when nothing ran). A build
        with no artifact ends the trial with `baseline-compile`, and a run
        that exits nonzero, ends with no exit status, or hangs ends it with
        `baseline-run`, as do output files the recipe's output-file Oracle
        cannot compare against (reference_problem names the file); that check
        is skipped for a run whose workdir came back incomplete, since a cut
        output alone ends nothing. The target's run is kept in
        Trial.reference_run (_run_info, with its output files in the binary
        store when that Oracle is bound), a failed one included; the source
        reference's files are read but not stored.
        """
        context = self.context
        files = self._files(language)
        toolchain = _toolchain(context, language)
        workdir = _baseline_dir(context.build_root, trial.trial_id, language)
        result = _harness_build(context, toolchain, files, workdir)
        if result.artifact is None:
            return _ended(trial, BASELINE_COMPILE, self._build_message(language, files, toolchain, result)), None
        if RUNS_CODE not in getattr(context.executor, "capabilities", ()):
            return trial, None
        item = context.suite.item(context.item, purpose=PURPOSE)
        run = context.executor.run(result.artifact, list(item.run_args), reference_limits(context))
        target = language == context.direction.target
        info = _run_info(context, run, keep_outputs=target and self.files_oracle is not None)
        if target:
            trial = dataclasses.replace(trial, reference_run=info)
        if run.hang:
            message = f"the {language} reference run hung past its wall limit of {REFERENCE_WALL_S} s"
            return _ended(trial, BASELINE_RUN, message), run
        if run.exit_code != 0:
            status = "no exit status" if run.exit_code is None else f"exit status {run.exit_code}"
            return _ended(trial, BASELINE_RUN, f"the {language} reference run ended with {status}"), run
        if self.files_oracle is not None and not run.workdir_incomplete:
            problem = self.files_oracle.reference_problem(run.output_files, side=f"{language} reference")
            if problem is not None:
                return _ended(trial, BASELINE_RUN, problem), run
        return trial, run

    def _agreement(self, trial: Trial, runs: Mapping[str, RunResult]) -> Trial:
        """Record the source reference's agreement with the target reference, and end the trial when it misses.

        It applies when the recipe's Oracle compares output files, both
        references ran, and the item declares a tolerance; otherwise the
        trial comes back unchanged. When either run's workdir came back
        incomplete, its output files may be cut, so no agreement is recorded
        and Trial.baseline_diagnostics gains one REFERENCE_WORKDIR_INCOMPLETE
        warning naming the run(s). Otherwise Trial.reference_agreement holds
        one OutputStats per output, judged against the item's tolerance under
        its own metric, and an output that fails ends the trial with
        `baseline-disagree`, before any model call.
        """
        context, oracle = self.context, self.files_oracle
        direction = context.direction
        tolerance = context.suite.item(context.item, purpose=PURPOSE).tolerance
        if oracle is None or tolerance is None or direction.source not in runs or direction.target not in runs:
            return trial
        cut = [language for language in (direction.target, direction.source) if runs[language].workdir_incomplete]
        if cut:
            names = " and ".join(f"the {language} reference" for language in cut)
            message = (
                f"{names} run's workdir came back incomplete, so its output files may be cut and the references' "
                "agreement was not measured"
            )
            note = Diagnostic(stage="run", severity="warning", code=REFERENCE_WORKDIR_INCOMPLETE, message=message)
            return dataclasses.replace(trial, baseline_diagnostics=[*trial.baseline_diagnostics, note])
        sides = (f"{direction.target} reference", f"{direction.source} reference")
        target, source = runs[direction.target].output_files, runs[direction.source].output_files
        stats = oracle.with_tolerance(tolerance).compare(target, source, sides=sides)
        trial = dataclasses.replace(trial, reference_agreement=stats)
        missed = [entry.name for entry in stats if not entry.passed]
        if not missed:
            return trial
        bound = "at least" if tolerance.metric == "pcc" else "at most"
        message = (
            f"the {direction.target} and {direction.source} references disagree past the item's tolerance, "
            f"{tolerance.metric} {bound} {tolerance.threshold!r}, on output(s) {', '.join(missed)}"
        )
        return _ended(trial, BASELINE_DISAGREE, message)

    def _files(self, language: str) -> dict[str, str]:
        """Return the item's reference program in `language` (file name -> text), as the manifest pins it."""
        context = self.context
        direction = context.direction
        if language == direction.target:
            return context.suite.reference_target(context.item, direction, context.sources_root, purpose=PURPOSE)
        return context.suite.source_files(context.item, direction, context.sources_root, purpose=PURPOSE)

    @staticmethod
    def _build_message(language: str, files: Mapping[str, str], toolchain: Toolchain, result: BuildResult) -> str:
        """Return the end reason message of a reference that did not build: its files, toolchain, and errors."""
        errors = [diagnostic for diagnostic in result.diagnostics if diagnostic.severity == "error"]
        text = diagnostics_text(errors).replace("\n", "; ") or "no error was reported"
        names = ", ".join(sorted(files))
        return f"the {language} reference ({names}) did not build with {_toolchain_name(toolchain)}: {text}"


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
        """Return `trial` with attempt 0 appended: S1 when the reply yields usable files, else S0.

        The prompt comes from _prompt; with the prompt_spaces fix off, each
        run of spaces in it is cut to one. It is kept in the text store and
        sent as the user message, after the system prompt when there is one.
        With the fence_tag fix on the reply's FILE blocks are parsed, and with
        it off its first fenced block is read as upstream reads it. The
        request is recorded with attempt 0 as the attempt its reply became.
        """
        context = self.context
        expected = target_files(context)
        system, prompt = self._prompt(trial, expected)
        if not fix_on(context, "prompt_spaces"):
            prompt = fragment_text.collapse_spaces(prompt)
        messages, reply = _send(context, prompt, system)
        attempt = _reply_attempt(context, 0, messages[-1].ref, reply, expected)
        return _recorded(context, trial.with_attempt(attempt), self.name, 0, messages, attempt.response_text)

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
        kept as returned, except that each lone surrogate becomes U+FFFD, and
        the request is recorded (_context_reply).
        """
        context = self.context
        pack = context.packs[context.direction.target]
        prompt = fragment_text.summary_request(context.fragments, context.direction, pack)
        messages, reply = _send(context, prompt, context.fragments[fragment_text.GENERAL_SYSTEM])
        return _context_reply(context, trial, self.name, "knowledge_summary", messages, reply)

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
        U+FFFD, and the request is recorded (_context_reply).
        """
        context = self.context
        prompt = fragment_text.description_request(context.fragments, source_as_read(context))
        messages, reply = _send(context, prompt, context.fragments[fragment_text.GENERAL_SYSTEM])
        return _context_reply(context, trial, self.name, "source_description", messages, reply)

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
    fragment_keys = fragment_text.CORRECT_KEYS
    reproduces = frozenset({"fence_tag", "prompt_newlines", "parsed_diagnostics"})

    def __init__(self, *, context: RunContext) -> None:
        """Keep the trial's run context."""
        self.context = context

    def __call__(self, trial: Trial) -> Trial:
        """Return `trial` with its last attempt built and any corrections appended.

        The last attempt is built (_build_last) with the target language's
        toolchain in a fresh build directory; its copy gets the compile
        diagnostics after the parse diagnostics, and S4 when an artifact was
        built. A build with no artifact always carries an error: "no-artifact"
        when the toolchain reported none, and "unwritable" when a file name
        could not be written. While the latest attempt has an error and is
        not S4, the stage asks for a correction (_correction), appends the
        reply with its diff from the previous files and records the request,
        and builds that. When the cap (corrections are the attempts after the
        first) stops the loop with an error remaining, final.end_reason is set
        to `correction-cap`. A cap of None never stops the loop.
        """
        if not trial.attempts:
            raise ValueError(f"{trial.trial_id}: compile_loop needs an attempt; run the generate stage first")
        cap = self.context.max_corrections
        expected = target_files(self.context)
        trial, stderr = self._build_last(trial)
        while self._needs_correction(trial.attempts[-1]):
            if cap is not None and len(trial.attempts) - 1 >= cap:
                message = f"an error remained after {cap} correction(s), the cap loop.max_corrections sets"
                return _ended(trial, CORRECTION_CAP, message)
            trial, stderr = self._build_last(self._correction(trial, expected, stderr))
        return trial

    def describe(self) -> str:
        """Return a one-line description of the stage."""
        context = self.context
        cap = context.max_corrections
        limit = "no correction cap" if cap is None else f"at most {cap} corrections"
        toolchain = context.toolchains.get(context.direction.target)
        builder = "no bound toolchain" if toolchain is None else _toolchain_name(toolchain)
        source = f"the {context.prompts} fragments" if context.fragments else f"{context.prompts}/correct.txt"
        errors = "parsed diagnostics" if fix_on(context, "parsed_diagnostics") else "the raw compiler stderr"
        return (
            f"compile_loop: build {context.direction.target} with {builder}, then correct from {source} with "
            f"{errors} while an error remains ({limit})"
        )

    @staticmethod
    def _needs_correction(attempt: Attempt) -> bool:
        """Return True when the attempt did not build and has an error to feed back."""
        return attempt.stage_reached != COMPILED and _has_error(attempt.diagnostics)

    def _buildable(self, attempt: Attempt) -> bool:
        """Return True for an attempt with no error that is S1, or S0 with fixes.fence_tag off.

        Under the fence_tag quirk an S0 attempt holds the empty target file
        upstream writes and compiles. An attempt that is S4 or holds a
        compile error was built already, or, as an S1 attempt with a
        MISSING_FILE error, is not built with its files incomplete.
        """
        if _has_error(attempt.diagnostics):
            return False
        if attempt.stage_reached == PARSED:
            return True
        return attempt.stage_reached == NO_OUTPUT and not fix_on(self.context, "fence_tag")

    def _build_last(self, trial: Trial) -> tuple[Trial, str | None]:
        """Return `trial` with its last attempt built, and the build's raw stderr as text mode reads it.

        The trial comes back unchanged, with None, when the last attempt is
        not buildable. A built program is kept in RunContext.artifacts under
        the attempt's index. The stderr is read only with
        fixes.parsed_diagnostics off, the one case a prompt carries it, and
        is None otherwise or when the build kept no attachment
        (BuildResult.stderr_ref).
        """
        attempt = trial.attempts[-1]
        if not self._buildable(attempt):
            return trial, None
        workdir = fresh_build_dir(self.context.build_root, trial.trial_id, attempt.index)
        result = self._build(attempt.files, workdir)
        if result.artifact is not None:
            self.context.artifacts[attempt.index] = Path(result.artifact)
        diagnostics = [*attempt.diagnostics, *result.diagnostics]
        if result.artifact is None and not _has_error(result.diagnostics):
            message = "the build produced no program and the toolchain reported no error; return the complete files"
            diagnostics.append(_compile_error("no-artifact", message))
        built = dataclasses.replace(
            attempt,
            diagnostics=diagnostics,
            stage_reached=COMPILED if result.artifact is not None else attempt.stage_reached,
        )
        trial = dataclasses.replace(trial, attempts=[*trial.attempts[:-1], built])
        if fix_on(self.context, "parsed_diagnostics"):
            return trial, None
        return trial, _attachment_text(workdir, result.stderr_ref)

    def _build(self, files: Mapping[str, str], workdir: Path) -> BuildResult:
        """Build `files` in `workdir`; a file name the filesystem refuses becomes an "unwritable" error.

        The item's support files from the suite manifest go into the build
        directory as harness files, passed as the toolchain's `harness`
        argument only when the item has some. Only a name error (too long, or
        refused as invalid) is the model's to fix; any other OSError, such as
        a full disk, propagates and stops the run.
        """
        context = self.context
        try:
            return _harness_build(context, _toolchain(context, context.direction.target), files, workdir)
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

    def _correction(self, trial: Trial, expected: Sequence[str], stderr: str | None) -> Trial:
        """Return `trial` with the correction of its last attempt's compile error appended (_corrected).

        The error text is the raw stderr (`stderr`, which _build_last reads
        only with fixes.parsed_diagnostics off) when it is not empty, else
        diagnostics_text of the attempt's diagnostics: upstream reads an empty
        stderr as a success, so a failed build that left none (a sandbox
        timeout, say) has no upstream counterpart.
        """
        errors = stderr if stderr else diagnostics_text(trial.attempts[-1].diagnostics)
        return _corrected(self.context, trial, self.name, expected, errors, run_error=False)


@register("Stage", "run_loop")
class RunLoopStage:
    """Runs each compiling attempt and asks for a correction while its run fails, sharing compile_loop's count and cap.

    It declares `compiles` because it builds and corrects the attempts its
    execute-error prompts yield, as compile_loop does. It requires no
    capability of the executor: with a compile-only executor it runs nothing.
    It runs model-generated code (`runs_model_code`), so an executor that
    runs programs must declare SANDBOXED; the runner refuses one that does
    not before any directory exists, and the stage refuses it too.
    """

    name = "run_loop"
    capabilities = frozenset({"compiles"})
    requires = {"Toolchain": {"diagnostics"}}
    prompt_fields = {"correct": CORRECT_FIELDS}
    fragment_keys = RUN_LOOP_KEYS
    reproduces = frozenset({"execution_gate", "fence_tag", "prompt_newlines", "parsed_diagnostics"})
    runs_model_code = True

    def __init__(self, *, context: RunContext) -> None:
        """Keep the trial's run context, and whether the recipe's Oracle compares output files."""
        self.context = context
        self.keeps_outputs = output_file_oracle(context) is not None

    def __call__(self, trial: Trial) -> Trial:
        """Return `trial` with its compiling attempts run and any corrections appended.

        compile_loop runs first (a no-op when it already ran). With a
        compile-only executor that is all; an executor that runs programs
        but does not declare SANDBOXED raises ValueError before anything
        runs (Agent Rule 6). Otherwise, while the last attempt
        is S4 and the trial has not ended: past the execution gate (fix off)
        the trial ends unexecuted (_past_gate); else the attempt runs
        (_run_last). A clean run ends the loop. A failed run with the cap
        reached ends the trial with `correction-cap`; below it, the
        execute-error correction is appended and compile_loop builds and
        corrects it.
        """
        context = self.context
        compile_loop = CompileLoopStage(context=context)
        trial = compile_loop(trial)
        if not declares(context.executor, RUNS_CODE):
            return trial
        if not declares(context.executor, SANDBOXED):
            raise ValueError(
                f"{trial.trial_id}: the executor runs programs but does not declare {SANDBOXED!r}; run_loop runs "
                "model-generated code only in the sandbox (Agent Rule 6)"
            )
        cap = context.max_corrections
        while trial.final.end_reason is None and trial.attempts[-1].stage_reached == COMPILED:
            index = trial.attempts[-1].index
            if not fix_on(context, "execution_gate") and index > EXECUTION_GATE_CORRECTIONS:
                return _past_gate(trial)
            trial, run, limits = self._run_last(trial)
            if trial.attempts[-1].stage_reached == RAN_CLEAN:
                break
            if cap is not None and index >= cap:
                message = f"a run error remained after {cap} correction(s), the cap loop.max_corrections sets"
                return _ended(trial, CORRECTION_CAP, message)
            errors = run_error_text(run, limits, context.fragments)
            trial = compile_loop(_corrected(context, trial, self.name, target_files(context), errors, run_error=True))
        return trial

    def describe(self) -> str:
        """Return a one-line description of the stage."""
        context = self.context
        if not declares(context.executor, RUNS_CODE):
            return "run_loop: the executor runs no programs, so no attempt runs"
        gate = (
            "every compiling attempt"
            if fix_on(context, "execution_gate")
            else f"a compiling attempt while its correction count is at most {EXECUTION_GATE_CORRECTIONS}"
        )
        return f"run_loop: run {gate} and correct each failed run, sharing compile_loop's correction count"

    def _run_last(self, trial: Trial) -> tuple[Trial, RunResult, Limits]:
        """Run the last attempt's program; return the trial with its run recorded, the RunResult, and the limits.

        A backend that declares `unload_before_run` is asked to unload first.
        The attempt keeps the run in Attempt.run (_run_info: stdout in the
        text store, each RunResult flag as a bool, and, when the recipe's
        Oracle compares output files, every output file in the binary store
        by hash), gains one warning per
        RunResult flag that is set (RUN_FLAGS), and is S5 after a
        clean run; after a failed one it stays S4 with a `run-error`.
        """
        context = self.context
        attempt = trial.attempts[-1]
        artifact = context.artifacts.get(attempt.index)
        if artifact is None:
            raise ValueError(
                f"{trial.trial_id}: attempt {attempt.index} compiled, but this trial kept no program for it; "
                "list compile_loop or run_loop, which build the attempts, before any stage that appends one"
            )
        limits = attempt_limits(context, trial)
        run_args = list(context.suite.item(context.item, purpose=PURPOSE).run_args)
        unload_before_run(context.backend)
        run = context.executor.run(artifact, run_args, limits)
        info = _run_info(context, run, keep_outputs=self.keeps_outputs)
        diagnostics = [*attempt.diagnostics, *_run_flag_warnings(run)]
        clean = run.exit_code == 0 and not run.hang
        if not clean:
            status = _run_status(run, limits)
            diagnostics.append(Diagnostic(stage="run", severity="error", code=RUN_ERROR, message=status))
        ran = dataclasses.replace(
            attempt, run=info, diagnostics=diagnostics, stage_reached=RAN_CLEAN if clean else COMPILED
        )
        return dataclasses.replace(trial, attempts=[*trial.attempts[:-1], ran]), run, limits


def attempt_limits(context: RunContext, trial: Trial) -> Limits:
    """Return the limits of an attempt run: the recipe's sandbox wall rule, its sandbox.mem_gb, and RUN_CPUS.

    With sandbox.wall_s BASELINE_X10 the wall limit is BASELINE_FACTOR times
    Trial.reference_run.wall_s, never less than RUN_WALL_FLOOR_S, which is
    also the limit when no reference ran; a number there is the limit
    itself. Any other value raises ValueError (the runner refuses it first).
    """
    sandbox = context.recipe.data["sandbox"]
    rule = sandbox["wall_s"]
    if rule == BASELINE_X10:
        reference = trial.reference_run.wall_s
        wall_s = RUN_WALL_FLOOR_S if reference is None else max(BASELINE_FACTOR * reference, RUN_WALL_FLOOR_S)
    elif isinstance(rule, (int, float)) and not isinstance(rule, bool) and rule > 0:
        wall_s = float(rule)
    else:
        raise ValueError(f"sandbox.wall_s must be a number of seconds above 0 or {BASELINE_X10!r}, not {rule!r}")
    return Limits(wall_s=wall_s, memory_mb=round(float(sandbox["mem_gb"]) * 1024), cpus=RUN_CPUS)


def run_error_text(run: RunResult, limits: Limits, fragments: Mapping[str, str]) -> str:
    """Return the error text of the execute-error prompt for the failed run `run`.

    With a fragment set (`fragments` not empty) it is upstream's report of
    the run, the notebook's execute_code return_result: RUN_REPORT_EXIT, the
    return code as Popen gives it (popen_return_code), one space,
    RUN_REPORT_SEGFAULT when that code is POPEN_SEGFAULT, then, when the run
    wrote any stderr, RUN_REPORT_STDERR and the stderr. A hang has no
    upstream counterpart (upstream waits for the program without a limit);
    its report is built the same way from the status the executor gives.
    With a template set it is this project's wording [DESIGN]: how the run
    ended, then the stderr when there is any. Either way the stderr is whole
    (the executor caps it) and read as text mode reads it, as upstream reads
    the program's output.
    """
    stderr = fragment_text.as_text_mode(run.stderr)
    if not fragments:
        status = _run_status(run, limits)
        return f"{status}; its standard error follows:\n{stderr}" if stderr else status
    code = popen_return_code(run.exit_code)
    report = fragments[RUN_REPORT_EXIT] + str(code) + " "
    if code == POPEN_SEGFAULT:
        report += fragments[RUN_REPORT_SEGFAULT]
    if stderr:
        report += fragments[RUN_REPORT_STDERR] + stderr
    return report


def popen_return_code(exit_code: int | None) -> int | None:
    """Return the return code Popen would give for an executor's exit status: -N for a death by signal N.

    An executor that runs programs reports such a death in the shell's form,
    SHELL_SIGNAL_BASE + N (N from 1 to MAX_SIGNAL); any other status, and
    None, comes back unchanged. A program that exits with such a status
    itself reads the same, since the status cannot tell the two apart.
    """
    if exit_code is not None and SHELL_SIGNAL_BASE < exit_code <= SHELL_SIGNAL_BASE + MAX_SIGNAL:
        return SHELL_SIGNAL_BASE - exit_code
    return exit_code


def _run_status(run: RunResult, limits: Limits) -> str:
    """Return one sentence saying how a failed run ended: stopped at its wall limit, no exit status, or its status."""
    if run.hang:
        return f"the program was stopped at its wall limit of {limits.wall_s:g} s"
    if run.exit_code is None:
        return "the program ended with no exit status"
    return f"the program exited with status {run.exit_code}"


def _run_flag_warnings(run: RunResult) -> list[Diagnostic]:
    """Return one run-stage warning per RunResult flag that is set, in RUN_FLAGS order."""
    return [
        Diagnostic(stage="run", severity="warning", code=code, message=message)
        for flag, (code, message) in RUN_FLAGS.items()
        if getattr(run, flag)
    ]


def _past_gate(trial: Trial) -> Trial:
    """Return `trial` ended at a compiling attempt past the execution gate, as upstream's loop ends there.

    The last attempt compiled but was not run, so the attempt whose output
    stands (lassi.core.record standing_attempt) is an earlier one. When one
    ran, the last attempt gains a `stale-output` warning naming it. When
    none ran, the trial ends with `upstream-crash`.
    """
    last = trial.attempts[-1]
    head = (
        f"attempt {last.index} compiled after {last.index} correction(s), past upstream's execution gate (a "
        f"compiling attempt runs only while its correction count is at most {EXECUTION_GATE_CORRECTIONS}), so it "
        "was not run"
    )
    standing = standing_attempt(trial)
    if standing is None:
        message = f"{head}; no earlier attempt ran, so upstream's notebook has no run output to read and raises there"
        return _ended(trial, UPSTREAM_CRASH, message)
    message = f"{head}; the stdout of attempt {standing.index}, the last attempt that ran, stands as the trial's output"
    warning = Diagnostic(stage="run", severity="warning", code=STALE_OUTPUT, message=message)
    stale = dataclasses.replace(last, diagnostics=[*last.diagnostics, warning])
    return dataclasses.replace(trial, attempts=[*trial.attempts[:-1], stale])


def _corrected(
    context: RunContext, trial: Trial, stage: str, expected: Sequence[str], errors: str, *, run_error: bool
) -> Trial:
    """Return `trial` with the next attempt appended and its request recorded under `stage`.

    The correction prompt for the last attempt with `errors` comes from
    _correction_messages. With fixes.prompt_newlines off every line feed is
    removed from it before it is stored and sent. The reply is read as
    generate reads attempt 0 (_reply_attempt), and the attempt keeps its diff
    from the last attempt's files.
    """
    previous = trial.attempts[-1]
    system, prompt = _correction_messages(context, previous, expected, errors, run_error=run_error)
    if not fix_on(context, "prompt_newlines"):
        prompt = prompt.replace("\n", "")
    messages, reply = _send(context, prompt, system)
    attempt = _reply_attempt(context, previous.index + 1, messages[-1].ref, reply, expected)
    attempt = dataclasses.replace(attempt, diff_from_previous=unified_diff(previous.files, attempt.files))
    return _recorded(context, trial.with_attempt(attempt), stage, attempt.index, messages, attempt.response_text)


def _correction_messages(
    context: RunContext, previous: Attempt, expected: Sequence[str], errors: str, *, run_error: bool
) -> tuple[str | None, str]:
    """Return the system prompt (None for a template set) and the correction prompt for `previous` with `errors`.

    A template set gives correct.txt, with `errors` as its diagnostics. A
    fragment set gives the direction's system prompt and upstream's
    correction prompt, whose code is the previous target file (FILE blocks
    when the target has more than one file): the compile-error form, or
    with `run_error` the execute-error form, which joins the execute-error
    lead (CORRECT_RUN_HEAD, CORRECT_RUN_TAIL) around the same compiler and
    flag text.
    """
    if not context.fragments:
        fields = {
            "target_language": context.direction.target,
            "files": render_file_blocks(previous.files) if previous.files else "",
            "diagnostics": errors,
            "target_files": ", ".join(expected),
        }
        return None, render(context.prompts, "correct", fields)
    direction, fragments = context.direction, context.fragments
    code = previous.files.get(expected[0], "") if len(expected) == 1 else render_file_blocks(previous.files)
    if run_error:
        setup = (
            fragments[fragment_text.fragment_key(fragment_text.SETUP_COMPILER, direction)]
            + " "
            + fragments[fragment_text.fragment_key(fragment_text.SETUP_FLAGS, direction)]
        )
        prompt = (
            code + fragments[CORRECT_RUN_HEAD] + setup + fragments[CORRECT_RUN_TAIL] + errors
            + fragments[fragment_text.CORRECT_OUTRO]
        )
    else:
        prompt = fragment_text.correction_prompt(fragments, direction, code, errors)
    return fragments[fragment_text.fragment_key(fragment_text.DIRECTION_SYSTEM, direction)], prompt
