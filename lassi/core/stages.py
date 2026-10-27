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
  support files for that language as harness files (Suite.support_files
  with the language: the item-level ones and the language's own, task
  P4.13), and runs it when that language's executor (executor_for) runs
  programs (capability `runs_code`); a language whose manifest entry names
  a `reference` language (task P17.36) has that language's program as its
  reference, with that language's toolchain, support files, and executor;
  it runs with the item's program arguments
  and reference_limits. The arguments come from
  lassi.bench.registry.stage_inputs: an item that declares held-out inputs
  has them written under the build directory's @inputs first, drawn with
  its seed, and takes the input files, then the output files, as its
  arguments; any other item takes its run_args. The target's run is kept in
  Trial.reference_run (exit code, hang flag, wall time, stdout in the text
  store, and the RunResult flags as bools); the source reference's run is
  never recorded. A cut
  output does not end the trial. A reference that does not build ends the
  trial with final.end_reason `baseline-compile`. A run is read in this
  order (task P4.6; _reference_run_end): a simulator gap
  (RunResult.sim_gap) ends it with `sim-gap`, the message naming the gap's
  class, whatever the exit status; then a hang, undefined behavior
  (RunResult.sim_ub, even after exit status 0), or an exit status other
  than 0 ends it with `baseline-run`. The baseline does not read a
  reference run's diagnostics, so a kernel JIT error there counts only
  through the exit status, and a reference run that hangs gets no hang
  diagnostic and no Watcher dump, even when its executor added one (task
  P4.11). The runner then runs no later stage, so no model is asked. When the recipe's Oracle compares output files
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
  reply fills Trial.context.knowledge_summary. Under a template set (task
  P17.21) it sends one user message and no system prompt:
  SUMMARY_PROMPT rendered with the target language and that pack.
- describe_source sends the general system prompt and the description
  request followed by the source. The reply fills
  Trial.context.source_description.
- generate appends attempt 0. With a template set it sends generate.txt,
  filled with the source files and the item's kernels (kernel_files) in
  FILE blocks, the target's trusted host program (host_files, task P17.36;
  correct.txt and the mismatch template get it too), and the source
  lowered to MLIR (source_mlir, "" until a frontend is bound), as the only
  message; with a fragment set it sends the
  direction's system prompt and the generation prompt
  (lassi.core.fragments.generation_prompt), the source read as text mode
  reads it.
- compile_loop builds the last attempt and, while an error remains, asks
  for a correction and builds that, up to loop.max_corrections (none when
  uncapped). A cap hit with an error remaining ends the trial with
  final.end_reason `correction-cap`. With a template set the correction
  prompt is correct.txt, with the item's kernels (kernel_files), sent as
  the only message; with a fragment set it is upstream's
  (lassi.core.fragments.correction_prompt), sent after the direction's
  system prompt. Its error text is the parsed diagnostics, capped at
  DIAGNOSTIC_COUNT_CAP diagnostics and DIAGNOSTIC_BYTES_CAP bytes of whole
  lines, with a line saying how many were left out when any were
  (diagnostics_text). After every build that gave a program, the target
  language's toolchain, when it declares host_compute_guard
  (lassi.core.capabilities, task P4.12; found by capability, never by
  name), reads the attempt's files with the item's support files for the
  target language: the reading sets Attempt.guards.host_compute and adds its parse-stage
  warnings or notes after the build's diagnostics. It never changes a
  stage or asks for a correction, and every later copy of the attempt
  keeps it. A guard's diagnostics (codes starting with "guard-") never
  enter a correction prompt (prompt_diagnostics). Attempt.guards.host_compute_not_checked
  (task P17.14) says why host_compute is null: generate and each
  correction set `not-built` when they append an attempt, or `no-guard`
  when the target toolchain declares no guard (it wins over the rest);
  a build with no program sets `no-program` (or keeps no-guard); and a
  build with one takes the reading's: `guard-not-checked` for a null
  reading, None for a true or false one. No prompt reads it. Then the
  target toolchain, when it declares static_tags (task P17.40; found by
  capability), reads the same files and its tags (the OpenMP-construct
  tag of gcc-openmp) follow the reading's diagnostics: parse-stage
  warnings or notes with "guard-" codes, so they never enter a prompt,
  change no stage or guard, and ask for no correction.
- run_loop continues compile_loop's loop (it runs compile_loop first, which
  changes nothing when compile_loop already ran) and, when the executor runs
  programs (capability `runs_code`), runs each compiling attempt from its
  build directory with the item's program arguments (stage_inputs, as the
  baseline gets them) and attempt_limits. The run
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

  Oracle mismatches (loop.correct_on_mismatch; corrects_on_mismatch). An
  opt-in departure from upstream, whose loop corrects only compile errors
  and failed runs (the owner's decision of 2026-10-23); the loader refuses
  it under faithful: true, and with it off or absent nothing here changes.
  With it true, each attempt whose run is clean (S5) is judged at once by
  the recipe's Oracle that compares output files, against the target
  reference's output files on the held-out inputs, bound as the oracle
  stage binds it (agreement_bound): the attempt takes the Alignment, and
  when an output fails (outside its threshold, missing, extra, repeated,
  or unreadable) it gains a run-stage error, code ORACLE_MISMATCH, naming
  each failed output in the coarse words below, and stays S5, since output
  agreement is never a rung. While the cap leaves room, the correction is
  asked with the template MISMATCH_PROMPT, whose field `outputs` is
  mismatch_text, a coarse report (OQ-077, option (a)): one line per output
  with its name, passed or failed, its pcc rounded to PCC_PLACES decimals
  when it was computed, and output_reading's fixed words, which say how far
  the output is from its tolerance as an order of magnitude only
  (tolerance_band) or name the kind of a failure the metric's value does
  not explain (a missing, extra, repeated, or unreadable output among
  them). It never shows the exact max_abs, max_ulp, or threshold, nor the
  Oracle's note text, which quotes element values and an unreadable file's
  first bytes: exact statistics give reference output values away (an
  all-zero candidate makes max_abs the reference's largest magnitude), and
  only an order of magnitude can leak. Nor does it show a held-out input
  or the seed (Harness Contract). The request is recorded under run_loop,
  and the reply is built, run, and judged again under the same count and
  cap; a mismatch left at the cap ends the trial with
  `correction-cap`. An attempt the oracle stage would not align either (no
  output files recorded, or a target reference whose workdir came back
  incomplete) is not judged here and ends the loop as before. The runner
  refuses the key, before any directory exists, without a listed stage that
  declares `mismatch_prompt_fields` (this one), without an Oracle that
  compares output files, with a fragment set (upstream's prompts hold no
  such correction), and with a template set whose MISMATCH_PROMPT does not
  render.

  Restarts (loop.restart_after; task P17.25). An opt-in departure from
  upstream, whose loop never asks a fresh generation; the loader refuses
  it under faithful: true and past loop.max_corrections, the runner with a
  fragment set, and with it absent nothing here changes. A correction is
  a request after attempt 0's that is not a restart, and it failed when
  its attempt did not reach a clean run (S5; failed_corrections, read from
  the record). When the last loop.restart_after requests since attempt 0
  or the last restart were corrections that all failed, and compile_loop
  or run_loop would ask for another, it asks for a restart instead
  (_restarted): attempt 0's prompt sent again byte for byte, recorded
  under its own stage name, RESTART, whose reply becomes the next attempt,
  built, run, and corrected like any other. A restart is not a
  correction, so the count starts again after it, and after a correction
  that reached a clean run. Restarts and corrections share the one count
  and cap, checked before a restart, so none comes past the cap.

  Profiles (task P17.7). When the recipe binds a profiler
  (RunContext.profiler), every attempt run is profiled, whatever its
  outcome (profiled_run): start() right after the backend's unload, then
  the run, then stop() in a finally block, so stop() runs when the
  executor raises too. Attempt.profile is stop()'s Profile, which
  checked_profile holds to the contract; an attempt that never ran keeps
  Profile() (all null), and the baseline's reference runs are not
  profiled.

  Simulator readings (task P4.6; RunResult's findings, never the
  executor's name). The attempt also gains the executor's diagnostics of
  the run (RunResult.diagnostics) and records RunResult.sim_ub in
  Attempt.run, then reads the run in this order (run_stage and
  RunLoopStage.__call__): a jit-stage error, then a gap, then undefined
  behavior, a hang, or the exit status. A jit-stage error among those
  diagnostics is a kernel JIT failure, whatever else the run reported: the
  attempt, run recorded, drops to S1 with no run-error, and
  compile_loop asks for its correction as for a compile error (Request
  stage compile_loop, the attempt's parsed diagnostics in the prompt,
  under the same count and cap). Undefined behavior (sim_ub True) is a
  failed run, S4 even after exit status 0, fed back like any other. A
  simulator gap (sim_gap) in a run with no jit-stage error leaves the
  attempt at S4 with no run-error and ends the trial with `sim-gap`, the
  message naming the gap's class, with no correction, whatever the exit
  status and even when the run also reported undefined behavior, unless
  loop.correct_on_sim_gap is true (Simulator gaps, below).
  The run-error message and the error text name the undefined
  behavior (UB_FINDING), and for a run that hung on an executor that
  declares `simulator` (lassi.core.capabilities SIMULATOR) they carry the
  Harness Contract's hang diagnostic (HANG_DIAGNOSTIC), a likely
  circular-buffer or semaphore deadlock, followed by the message of each
  Watcher dump the executor added to that run, in order: its run-stage
  notes coded WATCHER_CODE (task P4.11; run_findings). Those notes also
  stay among the attempt's diagnostics. The hang text is the ttsim hint
  (circular buffers and semaphores are tt-metal constructs); any executor
  that declares `simulator` gets it.

  Simulator gaps (loop.correct_on_sim_gap; corrects_on_sim_gap; OQ-098,
  option (a), the owner's answer of 2026-10-25). An opt-in departure from
  upstream, whose loop runs on no simulator; the loader refuses it under
  faithful: true, the runner with a fragment set (as run_loop does), and
  with it off or absent nothing here changes. With it true, an attempt
  whose run stopped at a gap with no jit-stage error keeps its record as
  above (S4, Attempt.run as the run gave it, no run-error) and gains one
  run-stage note, code SIM_GAP_NOTE, naming the class (gap_note): the
  record's mark of an attempt built but not verifiable on the simulator,
  never a failed run (lassi.core.record gap_noted reads it). While the cap
  leaves room, the gap is fed back as a failed run is, through correct.txt
  with the gap text (sim_gap_text) as its diagnostics: the gap's class;
  that the simulator cannot run a construct the program uses, which is a
  limit of the simulator and not necessarily an error in the program; that
  an equivalent construct the simulator runs should be used if one exists;
  then the run's findings of that class (gap_findings) and its stderr. The
  request is recorded under run_loop, and the reply is built, run, and
  corrected under the same count and cap; a gap left at the cap ends the
  trial with `sim-gap`, not `correction-cap` (_cap_end): a gap is no error
  that remained. The profiles read a gap attempt, the last attempt of a
  trial that ended at sim-gap or any attempt with the note, as holding no
  evidence either way, wherever it stands (lassi.scoring). It gets a
  run-error correction's context blocks (the knowledge block and
  retrieval's passages for the gap text; no examples, lookup, or
  skeleton), and a gap attempt counts, for loop.restart_after, as a
  correction that did not reach a clean run.

Context blocks under a template set (with_context_blocks). Each model
prompt after summarize_context (generate.txt, every correct.txt, and
MISMATCH_PROMPT) gets, before it, the knowledge block, KNOWLEDGE_PROMPT
rendered with Trial.context.knowledge_summary, when that summary is not
empty (task P17.21). With the recipe's retrieval section (task P17.22;
lassi.core.recipe, an opt-in the loader refuses under faithful: true and
the runner refuses with a fragment set), the prompts it lists (generate,
correct) get the retrieved block after it: the passages
(retrieved_passages) that Index.search (lassi.prompts.retrieval, on the
one index the runner loads per run, RunContext.retrieval) ranks for the
prompt's query, each skipped whole when it would take their text past
retrieval.max_chars, rendered with PASSAGE_PROMPT and joined into
RETRIEVED_PROMPT. Queries are built only from what the prompt shows:
generate's is the item's kernels and then its source files
(generate_query); a compile correction's is query_from_diagnostic of each
error its prompt lists, and a run-error correction's is its run error
text (correction_query). An oracle mismatch's correction gets no passages,
since its report names no API, and an empty query gives none. With the
recipe's lookup section (task P17.26; lassi.core.recipe, an opt-in the
loader refuses under faithful: true and the runner refuses with a fragment
set or without a context pack for each target language), a compile
correction (lookup.prompts lists correct) gets the looked-up block after
those: for each identifier its listed errors name (error_identifiers, read
as query_from_diagnostic reads them), the target language's pack's
declarations of it (pack_declarations, from the pack's header parts;
names_declaration), each skipped whole when it would take their text past
lookup.max_chars (looked_up_declarations), rendered with
DECLARATION_PROMPT and joined into LOOKED_UP_PROMPT; with lookup.nearest,
each identifier that names no declaration of the pack adds, after them,
one UNDECLARED_PROMPT line naming up to that many declared names nearest
to it (undeclared_names). With lookup.prompts listing generate, the
generation prompt gets a core block, CORE_API_PROMPT in words that name no
error, holding every declaration of the header parts lookup.core names, in
pack order, each skipped whole past lookup.max_chars (core_declarations). A run-error correction and an oracle
mismatch's correction get none, and with nothing carried there is no
block. Each block goes
before the prompt without its trailing line feeds, one blank line apart
(with_blocks), and a request that carried passages gets one
RETRIEVED_CODE note on its Request (retrieved_note), one that carried
declarations or nearest names one LOOKED_UP_CODE note (looked_up_note, or
core_note for the generation prompt), never on an
attempt, so neither enters a prompt. With none of these features a prompt
is byte for byte what it was before.

Worked examples (OQ-083 (b), the owner's answer of 2026-10-25). With the
recipe's examples section (lassi.core.recipe, an opt-in the loader refuses
under faithful: true and the runner refuses with a fragment set, a
template set that cannot render EXAMPLE_TEMPLATES, or a pack that does not
match its manifest), the generation prompt, and each compile correction
when examples.prompts lists correct, get the examples block after the
knowledge block and before the retrieved and looked-up blocks: the run's
pack (RunContext.examples, lassi.prompts.examples, loaded once per run)
carried whole in pack order, an example whose text would take their total
past examples.max_chars skipped whole, never cut (carried_examples), each
rendered with EXAMPLE_PROMPT and joined into EXAMPLES_PROMPT. A run-error
correction and an oracle mismatch's correction get none, and with no
example carried there is no block. A request that carried examples gets
one EXAMPLES_CODE note on its Request (examples_note); without the section
a prompt is byte for byte what it was before.

Host program skeleton (OQ-096, option (a), the owner's answer of
2026-10-25). With the recipe's skeleton section (lassi.core.recipe, an
opt-in the loader refuses under faithful: true, for a skeleton that does
not match its manifest, and for a direction to another language, and the
runner refuses with a fragment set), the generation prompt, and each
compile correction when skeleton.prompts lists correct, get the skeleton
block last, after every other block and so right before the prompt that
holds the source program: the run's skeleton (RunContext.skeleton,
lassi.prompts.skeletons, loaded once per run) rendered with its own
block.txt (block_text). A run-error correction and an oracle mismatch's
correction get none. A request that carried it gets one SKELETON_CODE note
on its Request (skeleton_note); without the section a prompt is byte for
byte what it was before.

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
for it: compile_loop for a compile error (a kernel JIT failure included),
run_loop for a failed run or, under loop.correct_on_mismatch, an oracle
mismatch, or, under loop.correct_on_sim_gap, a simulator gap; a restart
under RESTART, whichever loop asked for it. An
attempt's own `invalid-text` warning stays on the attempt; Trial.context
carries no diagnostics, so a context reply that held a lone surrogate gives
its request one parse-stage `invalid-text` warning.

A request the backend refuses as past the model's context
(lassi.core.interfaces ContextExceeded, task P17.4) ends the trial with
`context-exceeded` (_context_end), the message naming the stage: generate,
summarize_context, describe_source, the correction of compile_loop and
run_loop, and a restart (RESTART) each catch it around their one model
call. Every earlier attempt and request stays, no attempt or request is
added for the refused call, and
nothing is built or run again; the runner then runs no later stage and goes
on to the next trial.

Progress events (lassi.core.progress, task P4.8) go to RunContext.observer
when there is one, and nothing a stage does depends on them: before each
model call, once its messages are stored, a request-sent event with the
trial before the request is recorded (_send); an attempt event after
generate, a correction, or a restart appends an attempt (with its request
recorded; each event names the stage the request is recorded under),
after compile_loop builds the last attempt (_build_last; the event carries
the host-compute reading), and after run_loop runs it (_run_last). Later
changes (run_loop's `stale-output` warning, its oracle-mismatch reading,
an end reason, the oracle stage's alignment) reach the observer with the
next request-sent event or the runner's trial-end event.

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
  error (S0, or S1 missing an expected file) is not built. With the fix on
  and the opt-in leniency extract.unfenced_file_lines true (no quirk: the
  loader refuses it under faithful: true or with the fix off), generate and
  every correction read a reply that holds no FILE block from its unfenced
  FILE lines (extract_on; lassi.core.files), each file with a parse-stage
  `unfenced-file` warning, which no rung of the ladder below reads.
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
  Under extract.unfenced_file_lines, the unfenced FILE lines of a reply
  with no FILE block are read the same way here and in S1.
- S1, parses: the FILE blocks gave at least one file, and every FILE-block
  error is `missing-file`, an expected file with no block. That error is
  the build error of the Harness Contract: compile_loop feeds it back to
  the model in the correction prompt without building the incomplete
  files, so the attempt stays S1. A built attempt whose run reported a
  kernel JIT error (a jit-stage error in RunResult.diagnostics) is S1 too.
- S2, verifies, and S3, lowers: the MLIR verifier and the lowering passes.
  Source-level translation has neither step, so these stages never record
  S2 or S3.
- S4, compiles: the toolchain built the files into an artifact, and no
  run of it reported a kernel JIT error.
- S5, runs clean: the artifact ran with no crash, undefined behavior, or
  hang (exit status 0, no hang, sim_ub not True, no simulator gap;
  run_stage); the run loop records it.

Output agreement with the oracle is never a stage: it is Attempt.alignment
and Final.alignment, which the reward weighs separately. Only run_loop runs
an attempt, and only on an executor that runs programs; any other attempt
keeps RunInfo at its defaults (None) and reaches at most S4, and a
compile-only executor's placeholder values are never copied into the record.
"""

from __future__ import annotations

import dataclasses
import difflib
import errno
import functools
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import cast

from lassi.bench import Direction, Suite, stage_inputs
from lassi.core import fragments as fragment_text
from lassi.core.capabilities import (
    ALIGNS_OUTPUT_FILES,
    GUARD_CODE_PREFIX,
    HOST_COMPUTE_GUARD,
    SIMULATOR,
    STATIC_TAGS,
    SUPPORTS_POWER,
    WATCHER_CODE,
    OutputFileOracle,
    declares,
    host_compute_reading,
    static_tags,
    unload_before_run,
)
from lassi.core.files import parse_file_blocks, render_file_blocks
from lassi.core.interfaces import (
    BuildResult,
    ContextExceeded,
    Executor,
    Limits,
    LLMBackend,
    Message,
    Profiler,
    RunResult,
    Sampling,
    Toolchain,
)
from lassi.core.progress import ATTEMPT, REQUEST_SENT, Observer, notify
from lassi.core.recipe import (
    EXAMPLES,
    EXAMPLES_CORRECT,
    EXAMPLES_GENERATE,
    LOOKUP,
    LOOKUP_CORE,
    LOOKUP_CORRECT,
    LOOKUP_GENERATE,
    LOOKUP_NEAREST,
    RESTART_AFTER,
    RETRIEVAL,
    RETRIEVAL_CORRECT,
    RETRIEVAL_GENERATE,
    SKELETON,
    SKELETON_CORRECT,
    SKELETON_GENERATE,
    Recipe,
)
from lassi.core.record import (
    DEVICE_TIMING_NOTE,
    GUARD_NOT_CHECKED,
    NO_GUARD,
    NO_PROGRAM,
    NOT_BUILT,
    SIM_GAP_NOTE,
    Attempt,
    Diagnostic,
    EndReason,
    Guards,
    OutputStats,
    Profile,
    Request,
    RequestMessage,
    RunInfo,
    TextRef,
    Trial,
    standing_attempt,
    unified_diff,
)
from lassi.core.registry import DEFAULT_REGISTRY, Registry, register
from lassi.core.store import BlobStore, TextStore
from lassi.core.tolerance import STAT_FIELDS, within
from lassi.executors.workdir import build_dir, fresh_build_dir
from lassi.prompts import render
from lassi.prompts.examples import Example
from lassi.prompts.retrieval import SCORE_DIGITS, Index, Passage, query_from_diagnostic
from lassi.prompts.retrieval import _identifiers as message_identifiers
from lassi.prompts.skeletons import Skeleton, block_text

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
# The extract leniency (lassi.core.recipe EXTRACT) that reads a reply with no FILE block from its unfenced FILE lines.
UNFENCED_FILE_LINES = "unfenced_file_lines"

# The end codes these stages set in final.end_reason (lassi.core.record END_REASONS).
BASELINE_COMPILE = "baseline-compile"
BASELINE_RUN = "baseline-run"
BASELINE_DISAGREE = "baseline-disagree"
CORRECTION_CAP = "correction-cap"
UPSTREAM_CRASH = "upstream-crash"
SIM_GAP = "sim-gap"
CONTEXT_EXCEEDED = "context-exceeded"

# The Diagnostic stage of kernel JIT messages, which an executor parses from a run (RunResult.diagnostics). A jit-stage
# error means the program's kernels did not compile, so the attempt reached no more than S1 (see the module docstring).
JIT = "jit"
# What the stages say about a run's simulator findings (task P4.6), in its run-stage error and its correction prompt:
# undefined behavior (RunResult.sim_ub), and the Harness Contract's hang diagnostic for an attempt run that hung on an
# executor that declares SIMULATOR (lassi.core.capabilities). HANG_DIAGNOSTIC is the ttsim hint: circular buffers and
# semaphores are tt-metal constructs. The stages give it to any executor that declares SIMULATOR, so a simulator of
# another device gets the same text; a reference run's hang message does not carry it (_reference_run_end). After
# it come the Watcher dumps the executor added to the hung run, its run-stage notes coded WATCHER_CODE (task P4.11;
# lassi.core.capabilities defines the code, so an executor names it without importing the stages).
UB_FINDING = "the simulator reported undefined behavior"
HANG_DIAGNOSTIC = (
    "a hang on a simulator most likely means a deadlock: a circular buffer or a semaphore that a core waits on and "
    "that is never filled or signalled"
)

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

# The run-stage Diagnostic codes run_loop sets: a failed run (error), stale output (warning), and, under
# loop.correct_on_mismatch, a clean run whose output files failed the oracle (error).
RUN_ERROR = "run-error"
STALE_OUTPUT = "stale-output"
ORACLE_MISMATCH = "oracle-mismatch"
# The opt-in key of the loop section (lassi.core.recipe LOOP_OPT_INS) that feeds an oracle mismatch back.
CORRECT_ON_MISMATCH = "correct_on_mismatch"
# The opt-in key of the loop section that feeds a simulator gap back (OQ-098). The code of the run-stage note it puts
# on each attempt whose run stopped at a gap is lassi.core.record SIM_GAP_NOTE, imported above: the record's mark of
# an attempt built but not verifiable on the simulator, which the score profiles read as holding no evidence either
# way (lassi.core.record gap_noted) and a report counts apart from failures (plans/runs/rngd-sweep/tabulate.py).
CORRECT_ON_SIM_GAP = "correct_on_sim_gap"
# What a simulator gap's correction says around the gap's class (sim_gap_text) [DESIGN]: plainly, that the simulator
# cannot run the construct, that this is a limit of the simulator and not necessarily of the program, and that an
# equivalent construct the simulator runs should be used if one exists. GAP_FINDINGS leads the run's findings.
GAP_HEAD = "the program built, but its run stopped at a simulator gap of class "
GAP_LIMIT = (
    ": the simulator cannot run a construct this program uses. This is a limit of the simulator, not necessarily an "
    "error in the program. If an equivalent construct exists that the simulator runs, use it in place of the one it "
    "cannot run."
)
GAP_FINDINGS = "the simulator reported:"
# The stage name of a restart's Request and progress events (loop.restart_after, task P17.25): its own, never the
# name of the loop that asked for it.
RESTART = "restart"
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
GENERATE_FIELDS = (
    "source_language", "target_language", "source_files", "target_files", "kernel_files", "host_files", "source_mlir"
)
CORRECT_FIELDS = ("target_language", "files", "diagnostics", "target_files", "kernel_files", "host_files")
# The template of an oracle mismatch's correction (loop.correct_on_mismatch) and its fields: `outputs` is the
# Oracle's coarse per-output report (mismatch_text), in place of correct.txt's diagnostics.
MISMATCH_PROMPT = "correct_mismatch"
MISMATCH_FIELDS = ("target_language", "files", "outputs", "target_files", "kernel_files", "host_files")
# The templates a template set needs to serve context packs (task P17.21), with their fields: the summary request
# summarize_context sends, and the knowledge block that goes before each later prompt once the summary is filled.
# The runner renders both, before any directory exists, when the recipe names packs or lists summarize_context.
SUMMARY_PROMPT = "summarize_context"
SUMMARY_FIELDS = ("target_language", "context_pack")
KNOWLEDGE_PROMPT = "knowledge"
KNOWLEDGE_FIELDS = ("knowledge_summary",)
CONTEXT_PROMPTS = {SUMMARY_PROMPT: SUMMARY_FIELDS, KNOWLEDGE_PROMPT: KNOWLEDGE_FIELDS}
# The templates of the retrieved block (task P17.22), with their fields: the block, whose `passages` are the
# passages one blank line apart, and one passage. The runner renders both when the recipe sets retrieval.
RETRIEVED_PROMPT = "retrieved"
RETRIEVED_FIELDS = ("passages",)
PASSAGE_PROMPT = "passage"
PASSAGE_FIELDS = ("path", "heading", "start", "end", "text")
RETRIEVAL_PROMPTS = {RETRIEVED_PROMPT: RETRIEVED_FIELDS, PASSAGE_PROMPT: PASSAGE_FIELDS}
# The code of the note on a request that carried passages, and what joins one block (or passage) to the next.
RETRIEVED_CODE = "retrieved"
BLOCK_JOINER = "\n\n"
# The templates of the looked-up block (task P17.26), with their fields: the block, whose `declarations` are the
# entries one blank line apart, and one declaration. The runner renders both when the recipe sets lookup.
# LOOKED_UP_CODE is the code of the note on a request that carried declarations.
LOOKED_UP_PROMPT = "looked_up"
LOOKED_UP_FIELDS = ("declarations",)
DECLARATION_PROMPT = "declaration"
DECLARATION_FIELDS = ("name", "header", "scope", "text")
LOOKUP_TEMPLATES = {LOOKED_UP_PROMPT: LOOKED_UP_FIELDS, DECLARATION_PROMPT: DECLARATION_FIELDS}
LOOKED_UP_CODE = "looked-up"
# The line lookup.nearest adds to the looked-up block for an identifier the pack declares nothing of (2026-10-25),
# with its fields: the identifier, and its nearest declared names joined by NEAREST_JOINER. The runner renders it
# when the recipe sets lookup.nearest. NEAREST_MIN_RATIO is the least difflib ratio a name is listed at
# (nearest_declared).
UNDECLARED_PROMPT = "undeclared"
UNDECLARED_FIELDS = ("name", "candidates")
NEAREST_TEMPLATES = {UNDECLARED_PROMPT: UNDECLARED_FIELDS}
# The template of the generation prompt's core block (lookup.core, 2026-10-25): the same `declarations` field as the
# looked-up block, in words that name no error, since a generation prompt shows none. The runner renders it when
# lookup.prompts lists generate.
CORE_API_PROMPT = "core_api"
CORE_TEMPLATES = {CORE_API_PROMPT: LOOKED_UP_FIELDS}
NEAREST_JOINER = ", "
NEAREST_MIN_RATIO = 0.75
# The templates of the examples block (OQ-083 (b)), with their fields: the block, whose `examples` are the entries one
# blank line apart, and one example, its name and its files as FILE blocks. The runner renders both when the recipe
# sets examples. EXAMPLES_CODE is the code of the note on a request that carried examples.
EXAMPLES_PROMPT = "examples"
EXAMPLES_FIELDS = ("examples",)
EXAMPLE_PROMPT = "example"
EXAMPLE_FIELDS = ("name", "files")
EXAMPLE_TEMPLATES = {EXAMPLES_PROMPT: EXAMPLES_FIELDS, EXAMPLE_PROMPT: EXAMPLE_FIELDS}
EXAMPLES_CODE = "examples"
# The code of the note on a request that carried the host program skeleton (OQ-096 (a)); the block's template is the
# skeleton's own block.txt (lassi.prompts.skeletons), not a template of the prompt set.
SKELETON_CODE = "skeleton"
# The line that heads each header part of a context pack's API reference: tools/build_tt_context.py writes it before
# the declarations of header H (the include line of assets/context/tt-metalium-5280a9cf.head.txt).
_INCLUDE_LINE = re.compile(r"^// ==== #include <([^<>\n]+)> ====$", re.MULTILINE)
# What a declaration scan reads at namespace scope (pack_declarations): a namespace's head before its `{`; a class,
# struct, or enum head, whose name is the first after the keywords; and a function's head before its first `(`
# outside brackets: a return type, then the name.
_NAMESPACE_HEAD = re.compile(r"(?:inline\s+)?namespace(?:\s+([A-Za-z_][A-Za-z0-9_:]*))?")
_TYPE_HEAD = re.compile(r"(?:class|struct|enum(?:\s+(?:class|struct))?)\s+([A-Za-z_][A-Za-z0-9_]*)")
_FUNCTION_HEAD = re.compile(r"\S.*[\s*&>]([A-Za-z_][A-Za-z0-9_]*)\s*", re.DOTALL)
_TEMPLATE_HEAD = re.compile(r"template\s*<")
_FIRST_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# A unit that opens with one of these words declares no function, and a function head that ends in one names none.
_NOT_FUNCTION_WORDS = frozenset({"using", "typedef", "static_assert", "friend"})
_NOT_FUNCTION_NAMES = frozenset({"operator"})
# The words of that report (OQ-077, option (a)): exact statistics give reference values away, so each output shows
# its verdict, its pcc to PCC_PLACES decimals, and one of these fixed phrases, never a value, a threshold, or a note.
PCC_PLACES = 3
WITHIN_TOLERANCE = "within tolerance"
PCC_BELOW = "pattern match below the required level"
NO_ERROR_ALLOWED = "not an exact match, where the tolerance allows no error"
# How many times its threshold a max_abs or ulp value is, as an order of magnitude: the phrase of the first bound
# the ratio does not pass, else ERROR_OVER.
ERROR_BANDS = (
    (10, "up to 10 times the allowed error"),
    (100, "about 10 to 100 times the allowed error"),
    (1000, "about 100 to 1000 times the allowed error"),
)
ERROR_OVER = "over 1000 times the allowed error"
# A compared output that failed for a reason other than its metric's value (bible Oracles, binary_io rules).
UNMATCHED_SPECIAL = "unmatched NaN or infinity"
INTEGER_DIFFERS = "not an exact match, which an integer output needs"
TOO_LARGE = "a difference too large for a double"
NO_THRESHOLD = "no threshold recorded for this output"
# An output whose statistics were not computed, named by how lassi.oracles.binary_io's note starts (run_loop
# compares with the default sides, so a candidate note says "the candidate"); change the two together. NOT_COMPARED
# names one whose note starts otherwise. UNMATCHED_NOTE starts the note of an unmatched NaN or infinity.
NOT_COMPARED_KINDS = (
    ("missing output:", "missing output"),
    ("extra output:", "extra output"),
    ("the candidate wrote the array ", "repeated output"),
    ("the candidate's output file ", "unreadable output"),
    ("dtype mismatch:", "dtype mismatch"),
    ("shape mismatch:", "shape mismatch"),
)
NOT_COMPARED = "not compared"
UNMATCHED_NOTE = "unmatched "
# A support file is a kernel when its build path has a directory of this name, where the kernel JIT finds it.
# The TT host toolchain's is_kernel_source applies the same rule to place a kernel; change the two together.
KERNEL_DIR = "kernels"
# A target support file outside KERNEL_DIR with one of these suffixes is a host program the prompt shows read-only
# (host_files; task P17.36). lassi.toolchains.ttkernel_build HOST_SUFFIXES compiles the same files; change the two
# together.
HOST_SUFFIXES = (".cpp", ".cc", ".cxx")

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
    `executor` runs the trial's attempts: the target language's executor.
    `executors` maps each language to its executor when the recipe binds
    executors per language (task P4.5), and is empty for the single form,
    whose one executor is `executor`; executor_for reads both.
    `build_root` is the root that lassi.executors.workdir.build_dir places each
    attempt's build directory under; the runner passes the run directory, so
    one run's builds never meet another's. `prompts` is the prompt set, and
    `max_corrections` the correction cap (None means uncapped). `fragments`
    holds the fragments of a fragment prompt set (empty for a template set),
    and `packs` the recipe's context packs, keyed by the language each
    serves. `artifacts` maps the index of each attempt a build turned into a
    program to that program (BuildResult.artifact): compile_loop fills it and
    run_loop runs from it. The runner builds one RunContext per trial, so it
    never carries an artifact from one trial to the next. `observer` gets the
    stages' progress events (lassi.core.progress: request-sent and attempt);
    the runner passes its GuardedObserver, or None when the run has no
    observer, and no stage reads anything back from it. `profiler` is the
    run's Profiler (task P17.7), built once per run, or None when the recipe
    binds none; run_loop brackets each attempt run with it (profiled_run).
    `registry` is the run's registry (task P16.5), from which a stage that
    builds a component per trial builds it (the adversary stage, its
    Agent); None means DEFAULT_REGISTRY. `retrieval` is the run's
    documentation index (task P17.22), loaded once per run when the recipe
    sets retrieval and shared read-only by every trial, else None.
    `examples` is the run's worked examples pack (OQ-083 (b)), in pack
    order, loaded once per run when the recipe sets examples, else empty.
    `skeleton` is the run's host program skeleton (OQ-096 (a)), loaded
    once per run when the recipe sets skeleton, else None.
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
    executors: Mapping[str, Executor] = field(default_factory=dict)
    observer: Observer | None = None
    profiler: Profiler | None = None
    registry: Registry | None = None
    retrieval: Index | None = None
    examples: tuple[Example, ...] = ()
    skeleton: Skeleton | None = None


def executor_for(context: RunContext, language: str) -> Executor:
    """Return the executor that runs programs in `language`: its own with executors per language, else the one.

    With executors per language, a language without one raises ValueError
    rather than run on another language's executor; the runner refuses such
    a recipe before any directory exists.
    """
    if not context.executors:
        return context.executor
    executor = context.executors.get(language)
    if executor is None:
        bound = ", ".join(sorted(context.executors))
        raise ValueError(f"no executor is bound for {language!r}; executors are bound for: {bound}")
    return executor


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


def corrects_on_mismatch(context: RunContext) -> bool:
    """Return True only when the recipe sets loop.correct_on_mismatch true (lassi.core.recipe LOOP_OPT_INS).

    The key is off when absent; the loader refuses it set true under
    faithful: true.
    """
    return context.recipe.data.get("loop", {}).get(CORRECT_ON_MISMATCH, False) is True


def corrects_on_sim_gap(context: RunContext) -> bool:
    """Return True only when the recipe sets loop.correct_on_sim_gap true (lassi.core.recipe LOOP_OPT_INS; OQ-098).

    The key is off when absent; the loader refuses it set true under
    faithful: true, and the runner with a fragment set.
    """
    return context.recipe.data.get("loop", {}).get(CORRECT_ON_SIM_GAP, False) is True


def restart_after(context: RunContext) -> int | None:
    """Return loop.restart_after (lassi.core.recipe RESTART_AFTER), or None when the recipe does not set it.

    The key is off when absent. The loader refuses it under faithful: true
    and past loop.max_corrections, and the runner with a fragment set.
    """
    value = context.recipe.data.get("loop", {}).get(RESTART_AFTER)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def failed_corrections(trial: Trial) -> int:
    """Return how many corrections in a row, back from the last request, failed to reach a clean run (task P17.25).

    A correction is a request after attempt 0's that is not a restart; it
    failed when the attempt its reply became did not reach RAN_CLEAN (S5;
    a clean run that failed the oracle under loop.correct_on_mismatch did
    reach it). The count stops at attempt 0's request, a restart, or a
    correction that reached a clean run. It is read from the record, since
    each stage object is built anew (run_loop builds its own compile_loop).
    """
    count = 0
    for request in reversed(trial.requests or []):
        index = request.attempt_index
        if not index or request.stage == RESTART or trial.attempts[index].stage_reached == RAN_CLEAN:
            break
        count += 1
    return count


def restart_due(context: RunContext, trial: Trial) -> bool:
    """Return True when loop.restart_after is set and that many corrections in a row failed (failed_corrections)."""
    after = restart_after(context)
    return after is not None and failed_corrections(trial) >= after


def restarts_described(context: RunContext) -> str:
    """Return what compile_loop's and run_loop's describe() add for loop.restart_after: "" when it is not set."""
    after = restart_after(context)
    return "" if after is None else f", a restart from attempt 0's prompt after {after} failed correction(s) in a row"


def extract_on(context: RunContext, name: str) -> bool:
    """Return True only when the recipe sets the named extract leniency true (lassi.core.recipe EXTRACT).

    A leniency is off when absent, unlike a fix; the loader refuses one set
    true under faithful: true or with fixes.fence_tag off.
    """
    return context.recipe.data.get("extract", {}).get(name, False) is True


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
    lines = [diagnostic_line(diagnostic) for diagnostic in listed_diagnostics(diagnostics)]
    left_out = len(diagnostics) - len(lines)
    if left_out:
        lines.append(f"[{left_out} more diagnostic(s) not shown]")
    return "\n".join(lines)


def listed_diagnostics(diagnostics: Sequence[Diagnostic]) -> list[Diagnostic]:
    """Return the diagnostics diagnostics_text shows, in order: the first ones within both caps, whole lines only.

    At most DIAGNOSTIC_COUNT_CAP of them, whose lines (diagnostic_line, each
    line's UTF-8 bytes plus one line feed) take at most DIAGNOSTIC_BYTES_CAP
    bytes.
    """
    listed: list[Diagnostic] = []
    used = 0
    for diagnostic in diagnostics:
        size = len(diagnostic_line(diagnostic).encode("utf-8", "surrogatepass")) + 1
        if len(listed) == DIAGNOSTIC_COUNT_CAP or used + size > DIAGNOSTIC_BYTES_CAP:
            break
        listed.append(diagnostic)
        used += size
    return listed


def reference_limits(context: RunContext) -> Limits:
    """Return the limits of a baseline reference run: REFERENCE_WALL_S, the recipe's sandbox.mem_gb, REFERENCE_CPUS."""
    memory_mb = round(float(context.recipe.data["sandbox"]["mem_gb"]) * 1024)
    return Limits(wall_s=REFERENCE_WALL_S, memory_mb=memory_mb, cpus=REFERENCE_CPUS)


def _run_info(context: RunContext, run: RunResult, keep_outputs: bool = False, reference: bool = False) -> RunInfo:
    """Return the RunInfo of a run that happened: exit code, hang flag, UB, wall time, stdout in the store, run flags.

    Each RunResult flag (RUN_FLAGS) is copied as a bool, so a run whose
    output was kept whole records False, never None. sim_ub is copied as
    the executor reported it: True, False, or None when it did not check.
    With `keep_outputs`, every output file's bytes go into the binary store
    beside the text store and RunInfo.outputs maps each file to its sha256;
    otherwise outputs stays None (not recorded). The executor's device
    timing (task P17.42) is kept as read, with self_reported False for a
    `reference` run, the item's tracked program, and True otherwise: an
    attempt's program is model-written, and its process wrote the files.
    """
    flags = {flag: bool(getattr(run, flag)) for flag in RUN_FLAGS}
    stdout_ref = context.store.put(run.stdout)
    outputs = _kept_outputs(context, run) if keep_outputs else None
    timing = run.device_timing
    if timing is not None:
        timing = dataclasses.replace(timing, self_reported=not reference)
    return RunInfo(
        exit_code=run.exit_code, hang=run.hang, sim_ub=run.sim_ub, wall_s=run.wall_s, stdout_ref=stdout_ref,
        outputs=outputs, **flags, device_timing=timing,
    )


def _timing_notes(run: RunResult) -> list[Diagnostic]:
    """Return the run's DEVICE_TIMING_NOTE notes: why a silicon run's device time was asked for but not recorded."""
    return [item for item in run.diagnostics if item.stage == "run" and item.code == DEVICE_TIMING_NOTE]


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


def agreement_bound(oracle: OutputFileOracle, trial: Trial) -> OutputFileOracle:
    """Return `oracle` bound to Trial.reference_agreement when it reads one (agreement_setting), else `oracle`.

    The oracle stage and run_loop (loop.correct_on_mismatch) both bind it
    so. Raises ValueError starting with the trial id when the agreement is
    needed but was not recorded; the message adds any
    REFERENCE_WORKDIR_INCOMPLETE note the baseline made.
    """
    if oracle.agreement_setting is None:
        return oracle
    try:
        return oracle.with_baseline(trial.reference_agreement)
    except ValueError as error:
        notes = [note.message for note in trial.baseline_diagnostics if note.code == REFERENCE_WORKDIR_INCOMPLETE]
        why = f" (the baseline noted: {'; '.join(notes)})" if notes else ""
        raise ValueError(f"{trial.trial_id}: {error}{why}") from error


def stored_files(blobs: BlobStore, outputs: Mapping[str, str]) -> dict[str, Path]:
    """Return a run's recorded output files as file -> its file in the binary store, each checked against its hash."""
    return {file: blobs.path(digest) for file, digest in outputs.items()}


def tolerance_band(metric: str, value: float, threshold: float) -> str:
    """Return how far a metric's `value` is from `threshold` in fixed words, an order of magnitude at most.

    WITHIN_TOLERANCE when the value meets it (lassi.core.tolerance within).
    A pcc below it is PCC_BELOW, with no figure. A max_abs or ulp value past
    it is the ERROR_BANDS phrase of value / threshold (ERROR_OVER past the
    last bound), or NO_ERROR_ALLOWED when the threshold is 0. Neither number
    appears in the phrase.
    """
    if within(metric, value, threshold):
        return WITHIN_TOLERANCE
    if metric == "pcc":
        return PCC_BELOW
    if threshold == 0:
        return NO_ERROR_ALLOWED
    return next((words for bound, words in ERROR_BANDS if value <= bound * threshold), ERROR_OVER)


def output_reading(oracle: OutputFileOracle, entry: OutputStats) -> str:
    """Return how output `entry` compared with the reference, in fixed words that hold no number of either run.

    An output whose statistics were not computed (pcc None) is named by its
    kind (NOT_COMPARED_KINDS: a missing, extra, repeated, or unreadable
    output, or a dtype or shape mismatch). A compared output that passed is
    WITHIN_TOLERANCE. One that failed for a reason other than its metric's
    value names that reason (an unmatched NaN or infinity, an integer
    output that differs, a difference too large for a double, or no
    threshold), and any other is placed by tolerance_band against the
    oracle's threshold for it. The note's text is read only to name a kind
    and is never returned.
    """
    note = entry.note or ""
    if entry.pcc is None:
        return next((kind for start, kind in NOT_COMPARED_KINDS if note.startswith(start)), NOT_COMPARED)
    if entry.passed:
        return WITHIN_TOLERANCE
    if note.startswith(UNMATCHED_NOTE):
        return UNMATCHED_SPECIAL
    if entry.max_ulp is None:  # binary_io records no max_ulp for an integer output, which must match exactly
        return INTEGER_DIFFERS
    if entry.max_abs is None:
        return TOO_LARGE
    threshold = oracle.threshold_for(entry.name)
    if threshold is None:
        return NO_THRESHOLD
    return tolerance_band(oracle.metric, getattr(entry, STAT_FIELDS[oracle.metric]), threshold)


def _rounded_pcc(pcc: float) -> str:
    """Return `pcc` to PCC_PLACES decimals, a negative zero written as zero."""
    text = f"{pcc:.{PCC_PLACES}f}"
    return text.lstrip("-") if float(text) == 0 else text


def output_line(oracle: OutputFileOracle, entry: OutputStats) -> str:
    """Return one output as an oracle mismatch's correction prompt shows it, on one line (OQ-077, option (a)).

    `<name>: <passed|failed>; pcc <pcc>; <reading>`, with pcc to PCC_PLACES
    decimals and the reading of output_reading, or `<name>: failed;
    <kind>` for an output whose statistics were not computed. The exact
    max_abs, max_ulp, and threshold and the oracle's note never appear.
    pcc is kept for every output size: it is free of scale and shift, so no
    report gives an element's value, though across corrections a candidate
    built to probe it (a one-hot array) reads one element's standardized
    deviation per correction (OQ-077, option a). A one-element output's pcc is 1 (equal) or 0,
    and a two-element output's is 1 or -1 (the order of its two elements
    agrees or is reversed) or 0 (a side is constant), so even rounded it
    recovers no element of either. Nothing here reads an input or an output
    array.
    """
    verdict = "passed" if entry.passed else "failed"
    reading = output_reading(oracle, entry)
    if entry.pcc is None:
        return f"{entry.name}: {verdict}; {reading}"
    return f"{entry.name}: {verdict}; pcc {_rounded_pcc(entry.pcc)}; {reading}"


def mismatch_text(oracle: OutputFileOracle, stats: Sequence[OutputStats]) -> str:
    """Return the field `outputs` of MISMATCH_PROMPT: output_line for every output, in the oracle's order."""
    return "\n".join(output_line(oracle, entry) for entry in stats)


def _ended(trial: Trial, code: str, message: str) -> Trial:
    """Return `trial` with final.end_reason set to `code` and `message`; the runner then runs no later stage."""
    reason = EndReason(code=code, message=message)
    return dataclasses.replace(trial, final=dataclasses.replace(trial.final, end_reason=reason))


def _context_end(trial: Trial, stage: str, error: ContextExceeded) -> Trial:
    """Return `trial` ended at CONTEXT_EXCEEDED: the backend refused `stage`'s request as past its context.

    Every earlier attempt and request stays; the refused request adds none,
    since no reply exists (task P17.4).
    """
    return _ended(trial, CONTEXT_EXCEEDED, f"{stage}: the request was refused: {error}")


def _reference_run_end(run: RunResult, language: str) -> tuple[str, str] | None:
    """Return the end code and message of a reference run that ends the trial at the baseline, or None.

    Checked in this order (task P4.6): a simulator gap (RunResult.sim_gap)
    ends it at SIM_GAP, the message naming the gap's class, whatever the
    exit status; a hang ends it at BASELINE_RUN; so does undefined behavior
    (RunResult.sim_ub True), even after exit status 0, the message naming
    it; and so does an exit status other than 0, or none. A reference's
    diagnostics (RunResult.diagnostics, kernel JIT errors included) are not
    read here, so a JIT failure counts only through the exit status, and a
    hang gets the message it got before task P4.6, with no hang diagnostic
    and no Watcher dump (task P4.11).
    """
    if run.sim_gap is not None:
        message = (
            f"the {language} reference run stopped at a simulator gap, {run.sim_gap}: the simulator cannot run it, "
            "which is not a model error, so no model is asked"
        )
        return SIM_GAP, message
    if run.hang:
        return BASELINE_RUN, f"the {language} reference run hung past its wall limit of {REFERENCE_WALL_S} s"
    status = "no exit status" if run.exit_code is None else f"exit status {run.exit_code}"
    if run.sim_ub is True:
        return BASELINE_RUN, f"{UB_FINDING} in the {language} reference run, which ended with {status}"
    if run.exit_code != 0:
        return BASELINE_RUN, f"the {language} reference run ended with {status}"
    return None


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


def _support_files(context: RunContext, language: str) -> dict[str, str]:
    """Return the harness files of a build in `language` (build-directory name -> text).

    They are the item's support files for that language: the item-level
    ones and the language's own (Suite.support_files with `language`, task
    P4.13), so a TT build gets its kernels and a C++ build does not.
    """
    return context.suite.support_files(context.item, context.sources_root, purpose=PURPOSE, language=language)


def kernel_files(context: RunContext) -> str:
    """Return the template field kernel_files: the item's kernels as FILE blocks, or "" when it has none.

    A kernel is a support file of the direction's source or target language
    (_support_files, so item-level ones count) whose build path has a
    directory named KERNEL_DIR, tracked or upstream alike: the rule the TT
    host toolchain's is_kernel_source applies to place a kernel for the
    kernel JIT (Design Principle 8 keeps core from importing it). Each is
    shown once, in path order (render_file_blocks); a path both languages
    declare shows the target language's text. A prompt shows a kernel
    read-only (the owner's decision of 2026-10-06): a model file at its
    path is refused at any build whose harness files hold it (_unwritable
    in lassi.toolchains._base), which every TT build does for its own
    language's kernels and a C++ build only for item-level ones, so the
    tt-host-v0 prompts ask never to return one.
    """
    kernels: dict[str, str] = {}
    for language in (context.direction.source, context.direction.target):
        files = _support_files(context, language)
        kernels.update({path: text for path, text in files.items() if KERNEL_DIR in PurePosixPath(path).parts[:-1]})
    return render_file_blocks(kernels) if kernels else ""


def host_files(context: RunContext) -> str:
    """Return the template field host_files: the target's trusted host program as FILE blocks, or "" (P17.36).

    A host program is a support file of the direction's target language
    (_support_files) whose suffix is in HOST_SUFFIXES and whose build path
    has no directory named KERNEL_DIR: the files a ttkernel build compiles
    as the host program (lassi.toolchains.ttkernel_build host_sources,
    the same rule; Design Principle 8 keeps core from importing it). A
    prompt shows it read-only: a model file at its path is refused at the
    build. tt-pairs-v0's languages hold none, so their prompts get "".
    """
    files = _support_files(context, context.direction.target)
    hosts = {
        path: text for path, text in files.items()
        if PurePosixPath(path).suffix in HOST_SUFFIXES and KERNEL_DIR not in PurePosixPath(path).parts[:-1]
    }
    return render_file_blocks(hosts) if hosts else ""


def source_mlir(context: RunContext) -> str:
    """Return the template field source_mlir: the source lowered to MLIR by a frontend, or "" (task P17.36).

    No frontend is bound to a run yet (the C and C++ frontend, cgeist, is
    task P5.7), so the field is always "" for now; a template places it on
    a line of its own, which then stays blank.
    """
    del context
    return ""


def prompt_diagnostics(diagnostics: Sequence[Diagnostic]) -> list[Diagnostic]:
    """Return `diagnostics` without a guard's (code starting with GUARD_CODE_PREFIX), which never enter a prompt.

    The order is kept and the sequence given is not changed (task P4.12;
    the guard's outcome is never fed back to the model).
    """
    return [item for item in diagnostics if not (item.code or "").startswith(GUARD_CODE_PREFIX)]


def retrieval_on(context: RunContext, kind: str) -> bool:
    """Return True when the recipe's retrieval section lists the prompt kind `kind` (generate or correct).

    The section is off when absent (lassi.core.recipe RETRIEVAL); the loader
    refuses it under faithful: true, and the runner with a fragment set.
    """
    section = context.recipe.data.get(RETRIEVAL)
    return isinstance(section, Mapping) and kind in section["prompts"]


def generate_query(context: RunContext) -> str:
    """Return generate's retrieval query: the item's kernels (kernel_files), then its source files, one line feed apart.

    Both are FILE blocks exactly as generate.txt shows them ($kernel_files
    and $source_files); an empty part is left out.
    """
    sources = context.suite.source_files(context.item, context.direction, context.sources_root, purpose=PURPOSE)
    return "\n".join(part for part in (kernel_files(context), render_file_blocks(sources)) if part)


def correction_query(previous: Attempt, errors: str, *, run_error: bool) -> str:
    """Return a correction's retrieval query, built from the errors its prompt reports.

    A run-error correction's is `errors`, the run error text its prompt
    shows. A compile correction's is query_from_diagnostic
    (lassi.prompts.retrieval) of each error-severity diagnostic of
    `previous` that its prompt lists (listed_diagnostics of
    prompt_diagnostics), one per line.
    """
    if run_error:
        return errors
    listed = listed_diagnostics(prompt_diagnostics(previous.diagnostics))
    return "\n".join(query_from_diagnostic(item) for item in listed if item.severity == "error")


def retrieved_passages(context: RunContext, query: str) -> list[Passage]:
    """Return the passages one prompt carries for `query`: Index.search(query, retrieval.k), in rank order.

    A passage whose text would take the passages' total text past
    retrieval.max_chars is skipped whole, never cut, and a later smaller
    one may still fit. A blank query gives none and searches nothing.
    Raises ValueError when the RunContext carries no index or the prompt
    set is a fragment set; the runner refuses both first.
    """
    if not query.strip():
        return []
    if context.retrieval is None or context.fragments:
        raise ValueError(
            f"{context.recipe.path}: {RETRIEVAL} needs a template prompt set and the run's index, which the runner "
            "loads once per run; run the recipe with lassi run"
        )
    section = context.recipe.data[RETRIEVAL]
    carried: list[Passage] = []
    total = 0
    for passage in context.retrieval.search(query, section["k"]):
        if total + len(passage.text) > section["max_chars"]:
            continue
        carried.append(passage)
        total += len(passage.text)
    return carried


def with_blocks(blocks: Sequence[str], prompt: str) -> str:
    """Return `prompt` with `blocks` before it, in order, each without its trailing line feeds, one blank line apart."""
    return BLOCK_JOINER.join([*(block.rstrip("\n") for block in blocks), prompt])


def with_context_blocks(
    context: RunContext,
    trial: Trial,
    prompt: str,
    passages: Sequence[Passage] = (),
    looked_up: Sequence[tuple[str, Declaration]] = (),
    undeclared: Sequence[tuple[str, tuple[str, ...]]] = (),
    core: Sequence[Declaration] = (),
    examples: Sequence[Example] = (),
    skeleton: Skeleton | None = None,
) -> tuple[str, list[Diagnostic]]:
    """Return a template set's `prompt` with its context blocks before it, and the notes its request gets.

    The blocks (with_blocks) are KNOWLEDGE_PROMPT with
    Trial.context.knowledge_summary when that is not empty, then, when
    `examples` holds any (carried_examples), the examples block
    (examples_block), for which the request gets one note (examples_note),
    then, when `passages` holds any, the retrieved block (retrieved_block),
    for which the request gets one note (retrieved_note), then, when `looked_up`
    (identifier, declaration pairs; looked_up_declarations) or `undeclared`
    (identifier, nearest names pairs; undeclared_names) holds any, the
    looked-up block (looked_up_block), for which it gets one note
    (looked_up_note), or, for the generation prompt, when `core`
    (core_declarations) holds any, the core block of them (looked_up_block
    with CORE_API_PROMPT), with one core_note, and last, when `skeleton` is
    set (carried_skeleton), the skeleton's block (lassi.prompts.skeletons
    block_text), with one skeleton_note.
    With none, the prompt comes back unchanged with no note. A fragment
    set's prompt is never changed: its own generation prompt joins the
    summary.
    """
    if context.fragments:
        return prompt, []
    blocks: list[str] = []
    notes: list[Diagnostic] = []
    summary = trial.context.knowledge_summary
    if summary:
        blocks.append(render(context.prompts, KNOWLEDGE_PROMPT, {"knowledge_summary": summary}))
    if examples:
        blocks.append(examples_block(context.prompts, examples))
        notes.append(examples_note(context, examples))
    if passages:
        blocks.append(retrieved_block(context.prompts, passages))
        notes.append(retrieved_note(passages))
    if looked_up or undeclared:
        blocks.append(looked_up_block(context.prompts, [declaration for _, declaration in looked_up], undeclared))
        notes.append(looked_up_note(looked_up, undeclared))
    if core:
        blocks.append(looked_up_block(context.prompts, core, template=CORE_API_PROMPT))
        notes.append(core_note(core))
    if skeleton is not None:
        blocks.append(block_text(skeleton))
        notes.append(skeleton_note(skeleton))
    return with_blocks(blocks, prompt), notes


def retrieved_block(prompts: str, passages: Sequence[Passage]) -> str:
    """Return RETRIEVED_PROMPT of set `prompts` with `passages`, each PASSAGE_PROMPT rendered, one blank line apart."""
    texts = [render(prompts, PASSAGE_PROMPT, _passage_fields(passage)).rstrip("\n") for passage in passages]
    return render(prompts, RETRIEVED_PROMPT, {"passages": BLOCK_JOINER.join(texts)})


def _passage_fields(passage: Passage) -> dict[str, str]:
    """Return the PASSAGE_FIELDS of one passage: its file, heading path, first and last line, and text."""
    return {
        "path": passage.path,
        "heading": passage.heading,
        "start": str(passage.start_line),
        "end": str(passage.end_line),
        "text": passage.text,
    }


def retrieved_note(passages: Sequence[Passage]) -> Diagnostic:
    """Return the note of a request that carried `passages`: code RETRIEVED_CODE, severity note.

    Its message lists each passage as <path>:<start>-<end> with its score
    (to SCORE_DIGITS decimals, as the search rounds it), in rank order. Its
    stage is parse, the stage of a request's other notes, since the record
    has no stage for a prompt. It rides on the Request, never on an
    attempt, whose diagnostics a correction lists, so it never enters a
    prompt.
    """
    listing = ", ".join(
        f"{passage.path}:{passage.start_line}-{passage.end_line} (score {passage.score:.{SCORE_DIGITS}f})"
        for passage in passages
    )
    message = f"the prompt carried {len(passages)} documentation passage(s), in rank order: {listing}"
    return Diagnostic(stage="parse", severity="note", code=RETRIEVED_CODE, message=message)


def passages_described(context: RunContext, kind: str) -> str:
    """Return what describe() adds when retrieval is on for the prompt kind `kind`, else ""."""
    if not retrieval_on(context, kind):
        return ""
    section = context.recipe.data[RETRIEVAL]
    where = "the prompt" if kind == RETRIEVAL_GENERATE else "each compile or run correction prompt"
    return f", with up to {section['k']} {section['index']} documentation passages before {where}"


@dataclass(frozen=True)
class Declaration:
    """One declaration of a context pack's API reference: its name, header part, namespace scope, and exact text.

    `scope` is its enclosing namespaces joined by `::` ("" at global
    scope), and `text` the pack's text from its first character through its
    closing `;` (pack_declarations).
    """

    name: str
    header: str
    scope: str
    text: str


def lookup_on(context: RunContext, kind: str) -> bool:
    """Return True when the recipe's lookup section lists the prompt kind `kind` (generate or correct; P17.26).

    The section is off when absent (lassi.core.recipe LOOKUP); the loader
    refuses it under faithful: true, and the runner with a fragment set or
    without a context pack for the target language.
    """
    section = context.recipe.data.get(LOOKUP)
    return isinstance(section, Mapping) and kind in section["prompts"]


@functools.lru_cache(maxsize=4)
def pack_declarations(pack: str) -> tuple[Declaration, ...]:
    """Return the declarations of a context pack's text (its entries joined as the loader joins them), in pack order.

    The pack's header parts each run from a line `// ==== #include <H> ====`
    (_INCLUDE_LINE) to the next such line or the pack's end, H being the
    part's header; text outside them (a guide, a preamble) holds none. A
    declaration is a namespace-scope function declaration or class, struct,
    or enum definition of a part (_declared_name), its text verbatim, a
    class's members and inner blank lines included; a member is part of its
    class's declaration, never one of its own. The text is the cache key, so
    a run parses its pack once.
    """
    heads = list(_INCLUDE_LINE.finditer(pack))
    found: list[Declaration] = []
    for number, head in enumerate(heads):
        end = heads[number + 1].start() if number + 1 < len(heads) else len(pack)
        scopes: list[str] = []
        for kind, value in _namespace_units(pack[head.end():end]):
            if kind == "open":
                scopes.append(value)
            elif kind == "close":
                scopes = scopes[:-1]
            elif (name := _declared_name(value)) is not None:
                found.append(Declaration(name, head.group(1), "::".join(part for part in scopes if part), value))
    return tuple(found)


def _namespace_units(text: str) -> list[tuple[str, str]]:
    """Split header text into its units at namespace scope, in order: ("open", name), ("close", ""), ("unit", text).

    A unit starts at its first character that is no blank, comment, or
    preprocessor line, and runs through its `;` outside every bracket, or,
    when its first such `{` follows no class, struct, or enum head (a
    function body, say), through that brace's `}`, and is then dropped. A
    `{` outside every bracket after a namespace head opens the namespace
    instead ("" for an anonymous one), and a `}` before any unit starts
    closes it. Comments and string and character literals hold no bracket.
    """
    units: list[tuple[str, str]] = []
    start: int | None = None
    depth, body, index = 0, False, 0
    while index < len(text):
        skipped = _skipped(text, index)
        if skipped != index:
            index = skipped
            continue
        char = text[index]
        if start is None:
            if char == "}":
                units.append(("close", ""))
            if char.isspace() or char in "#}":
                index = _line_end(text, index) if char == "#" else index + 1
                continue
            start = index
        if char == "{" and depth == 0:
            head = text[start:index].strip()
            namespace = _NAMESPACE_HEAD.fullmatch(head)
            if namespace is not None:
                units.append(("open", namespace.group(1) or ""))
                start, index = None, index + 1
                continue
            body = _TYPE_HEAD.match(_without_template_head(head)) is None
        depth = _bracket_depth(depth, char)
        if depth == 0 and (char == ";" or (char == "}" and body)):
            if char == ";":
                units.append(("unit", text[start:index + 1]))
            start, body = None, False
        index += 1
    return units


def _bracket_depth(depth: int, char: str) -> int:
    """Return the depth after `char`: one more after an opening bracket, one less (never below 0) after a closing."""
    if char in ("(", "[", "{"):
        return depth + 1
    if char in (")", "]", "}"):
        return max(depth - 1, 0)
    return depth


def _skipped(text: str, index: int) -> int:
    """Return the index after the comment, string, or character literal at `index`; `index` when none starts there.

    A `'` right after a letter or digit is a digit separator, not a literal.
    """
    if text.startswith("//", index):
        return _line_end(text, index)
    if text.startswith("/*", index):
        end = text.find("*/", index + 2)
        return len(text) if end < 0 else end + 2
    quote = text[index]
    if quote == '"' or (quote == "'" and not (index and text[index - 1].isalnum())):
        position = index + 1
        while position < len(text) and text[position] != quote:
            position += 2 if text[position] == "\\" else 1
        return min(position + 1, len(text))
    return index


def _line_end(text: str, index: int) -> int:
    """Return the index of the line feed that ends the line holding `index`, or the text's end."""
    end = text.find("\n", index)
    return len(text) if end < 0 else end


def _without_template_head(text: str) -> str:
    """Return a unit's text without its leading `template <...>` parameter list, when it has one."""
    match = _TEMPLATE_HEAD.match(text)
    if match is None:
        return text
    depth = 0
    for index in range(match.end() - 1, len(text)):
        depth += {"<": 1, ">": -1}.get(text[index], 0)
        if depth == 0:
            return text[index + 1:].lstrip()
    return text


def _declared_name(unit: str) -> str | None:
    """Return the name a namespace-scope unit declares, or None when it is no function or type definition.

    A class, struct, or enum definition's name is the first after its
    keywords, when its first `{` outside brackets comes before any `(`
    (a forward declaration has none). Otherwise a function's name is the
    one right before its first `(` outside brackets, after a return type
    (_FUNCTION_HEAD), with no `=` before it (a variable's initializer); a
    unit that opens with a _NOT_FUNCTION_WORDS word, or whose name is an
    operator, declares none. A template's parameter list is skipped first.
    """
    head = _without_template_head(unit)
    brace, paren = _outside_brackets(head, "{"), _outside_brackets(head, "(")
    type_head = _TYPE_HEAD.match(head)
    if type_head is not None and brace is not None and (paren is None or brace < paren):
        return type_head.group(1)
    first = _FIRST_WORD.match(head)
    if paren is None or (first is not None and first.group() in _NOT_FUNCTION_WORDS):
        return None
    if _outside_brackets(head[:paren], "=") is not None:
        return None
    function = _FUNCTION_HEAD.fullmatch(head[:paren])
    if function is None or function.group(1) in _NOT_FUNCTION_NAMES:
        return None
    return function.group(1)


def _outside_brackets(text: str, wanted: str) -> int | None:
    """Return the index of the first `wanted` character of `text` outside brackets, comments, and literals, or None."""
    depth, index = 0, 0
    while index < len(text):
        skipped = _skipped(text, index)
        if skipped != index:
            index = skipped
            continue
        char = text[index]
        if char == wanted and depth == 0:
            return index
        depth = _bracket_depth(depth, char)
        index += 1
    return None


def error_identifiers(previous: Attempt) -> list[str]:
    """Return the identifiers a compile correction's listed errors name, each once, in order of first mention.

    They are, for each error-severity diagnostic of `previous` that its
    prompt lists (listed_diagnostics of prompt_diagnostics), the names its
    message gives as query_from_diagnostic (lassi.prompts.retrieval) reads
    them: the quoted names, then the other names holding an underscore, a
    scope (::), or an inner capital. Warnings and notes give none.
    """
    listed = listed_diagnostics(prompt_diagnostics(previous.diagnostics))
    names = [name for item in listed if item.severity == "error" for name in message_identifiers(item.message)]
    return list(dict.fromkeys(names))


def names_declaration(identifier: str, declaration: Declaration) -> bool:
    """Return True when `identifier` names `declaration`: its name, or its name qualified by the end of its scope.

    `Make`, `inner::Make`, and `outer::inner::Make` each name a Make
    declared in outer::inner; `other::Make` and `outer::Make` do not.
    """
    *qualifier, name = identifier.split("::")
    if name != declaration.name:
        return False
    scope = declaration.scope.split("::") if declaration.scope else []
    return len(qualifier) <= len(scope) and scope[len(scope) - len(qualifier):] == qualifier


def looked_up_declarations(context: RunContext, previous: Attempt) -> list[tuple[str, Declaration]]:
    """Return the declarations a compile correction of `previous` carries, each with the identifier that named it.

    For each identifier (error_identifiers) in order, every declaration of
    the target language's pack (pack_declarations of RunContext.packs) it
    names (names_declaration), in pack order, overloads included, each
    declaration once. One whose text would take the carried declarations'
    total text past lookup.max_chars is skipped whole, never cut, and a
    later smaller one may still fit. An identifier the pack declares no
    function or type of gives none. Raises ValueError when the prompt set
    is a fragment set or no pack serves the target; the runner refuses
    both first.
    """
    declarations = pack_declarations(_lookup_pack(context))
    budget = context.recipe.data[LOOKUP]["max_chars"]
    carried: list[tuple[str, Declaration]] = []
    total = 0
    for identifier in error_identifiers(previous):
        for declaration in declarations:
            named = names_declaration(identifier, declaration)
            if not named or any(item is declaration for _, item in carried):
                continue
            if total + len(declaration.text) > budget:
                continue
            carried.append((identifier, declaration))
            total += len(declaration.text)
    return carried


def _lookup_pack(context: RunContext) -> str:
    """Return the target language's pack text; raise ValueError for a fragment set or no pack (the runner's checks)."""
    pack = context.packs.get(context.direction.target)
    if pack is None or context.fragments:
        raise ValueError(
            f"{context.recipe.path}: {LOOKUP} needs a template prompt set and a context pack for the target language "
            f"{context.direction.target!r}, which the runner refuses the recipe without; run it with lassi run"
        )
    return pack


def pack_parts(pack: str) -> tuple[str, ...]:
    """Return the header of each header part of a context pack's text, in pack order (_INCLUDE_LINE's H)."""
    return tuple(head.group(1) for head in _INCLUDE_LINE.finditer(pack))


def core_declarations(context: RunContext) -> list[Declaration]:
    """Return the declarations the generation prompt carries when lookup.prompts lists generate (2026-10-25).

    They are every declaration of the target language's pack
    (pack_declarations) whose header part lookup.core names, in pack order,
    whatever the order of lookup.core; one whose text would take their
    total text past lookup.max_chars is skipped whole, never cut, and a
    later smaller one may still fit. The runner refuses a lookup.core
    header the pack has no part for. Raises ValueError as
    looked_up_declarations does.
    """
    section = context.recipe.data[LOOKUP]
    wanted, budget = set(section[LOOKUP_CORE]), section["max_chars"]
    carried: list[Declaration] = []
    total = 0
    for declaration in pack_declarations(_lookup_pack(context)):
        if declaration.header in wanted and total + len(declaration.text) <= budget:
            carried.append(declaration)
            total += len(declaration.text)
    return carried


def qualified_name(declaration: Declaration) -> str:
    """Return a declaration's name qualified by its scope (`scope::name`), or its name at global scope."""
    return f"{declaration.scope}::{declaration.name}" if declaration.scope else declaration.name


def nearest_declared(identifier: str, declarations: Sequence[Declaration], count: int) -> tuple[str, ...]:
    """Return up to `count` qualified names of `declarations` nearest `identifier`, nearest first.

    A declaration scores the higher of difflib.SequenceMatcher(None, form,
    identifier).ratio() over two forms, its name and its qualified name
    (qualified_name), as difflib.get_close_matches scores a word, so both
    `CreateKernal` and `distributed::EnqueueWriteMeshBufer` find their
    function. Each qualified name is listed once (overloads share one), only
    at a score of at least NEAREST_MIN_RATIO (0.75: a one-letter slip in a
    ten-letter name scores 0.9, and unrelated names are expected, not
    measured, to score lower), and ties go to the earlier in pack order.
    """
    matcher = difflib.SequenceMatcher()
    matcher.set_seq2(identifier)
    scored: dict[str, tuple[float, int]] = {}
    for order, declaration in enumerate(declarations):
        name = qualified_name(declaration)
        if name in scored:
            continue
        best = 0.0
        for form in (declaration.name, name):
            matcher.set_seq1(form)
            if matcher.real_quick_ratio() >= NEAREST_MIN_RATIO and matcher.quick_ratio() >= NEAREST_MIN_RATIO:
                best = max(best, matcher.ratio())
        if best >= NEAREST_MIN_RATIO:
            scored[name] = (best, order)
    return tuple(sorted(scored, key=lambda name: (-scored[name][0], scored[name][1]))[:count])


def undeclared_names(context: RunContext, previous: Attempt) -> list[tuple[str, tuple[str, ...]]]:
    """Return (identifier, nearest names) for each identifier of a correction's errors the pack declares nothing of.

    The identifiers are error_identifiers(previous), in order; one that
    names no declaration of the target language's pack (names_declaration)
    gets up to lookup.nearest names (nearest_declared), and one with no
    name at NEAREST_MIN_RATIO or above is left out. Off (empty) when the
    recipe does not set lookup.nearest. Raises ValueError as
    looked_up_declarations does.
    """
    count = context.recipe.data[LOOKUP].get(LOOKUP_NEAREST)
    if count is None:
        return []
    declarations = pack_declarations(_lookup_pack(context))
    found: list[tuple[str, tuple[str, ...]]] = []
    for identifier in error_identifiers(previous):
        if any(names_declaration(identifier, declaration) for declaration in declarations):
            continue
        names = nearest_declared(identifier, declarations, count)
        if names:
            found.append((identifier, names))
    return found


def looked_up_block(
    prompts: str, declarations: Sequence[Declaration], undeclared: Sequence[tuple[str, tuple[str, ...]]] = (),
    template: str = LOOKED_UP_PROMPT,
) -> str:
    """Return `template` (default LOOKED_UP_PROMPT) of `prompts` with each of `declarations` as DECLARATION_PROMPT.

    After them comes one UNDECLARED_PROMPT line per `undeclared` pair (an
    identifier and its nearest names, joined by NEAREST_JOINER). The entries
    go one blank line apart, as the retrieved block's passages do. The
    generation prompt's core block passes CORE_API_PROMPT.
    """
    texts = [render(prompts, DECLARATION_PROMPT, dataclasses.asdict(item)).rstrip("\n") for item in declarations]
    texts += [
        render(prompts, UNDECLARED_PROMPT, {"name": name, "candidates": NEAREST_JOINER.join(names)}).rstrip("\n")
        for name, names in undeclared
    ]
    return render(prompts, template, {"declarations": BLOCK_JOINER.join(texts)})


def core_note(declarations: Sequence[Declaration]) -> Diagnostic:
    """Return the note of a generation request that carried lookup.core's declarations: code LOOKED_UP_CODE.

    Its message gives the count and total text, then each header part in
    pack order with how many declarations came from it. Like looked_up_note
    it is a parse-stage note on the Request, so it never enters a prompt.
    """
    counts: dict[str, int] = {}
    for declaration in declarations:
        counts[declaration.header] = counts.get(declaration.header, 0) + 1
    listing = ", ".join(f"{header} ({count})" for header, count in counts.items())
    total = sum(len(declaration.text) for declaration in declarations)
    message = (
        f"the prompt carried {len(declarations)} declaration(s) from the context pack, {total} characters, from the "
        f"header parts lookup.core names, in pack order: {listing}"
    )
    return Diagnostic(stage="parse", severity="note", code=LOOKED_UP_CODE, message=message)


def looked_up_note(
    carried: Sequence[tuple[str, Declaration]], undeclared: Sequence[tuple[str, tuple[str, ...]]] = (),
) -> Diagnostic:
    """Return the note of a request that carried declarations or nearest names: code LOOKED_UP_CODE, severity note.

    Its message gives the count and total text, then each carried
    identifier in order with the header part of each declaration it carried
    and how many came from that part; with `undeclared` (lookup.nearest),
    it then names each such identifier with its nearest names, and with
    none it is what it was before lookup.nearest. Its stage is parse, as
    for the retrieved note, and it rides on the Request, never on an
    attempt, so it never enters a prompt.
    """
    sentences = [_carried_sentence(carried)] if carried else []
    if undeclared:
        listing = "; ".join(f"{name} -> {NEAREST_JOINER.join(names)}" for name, names in undeclared)
        sentences.append(
            f"the prompt named the nearest declared names for {len(undeclared)} identifier(s) its errors name that "
            f"the context pack declares nothing of, in order: {listing}"
        )
    return Diagnostic(stage="parse", severity="note", code=LOOKED_UP_CODE, message="; ".join(sentences))


def _carried_sentence(carried: Sequence[tuple[str, Declaration]]) -> str:
    """Return looked_up_note's account of the carried declarations: count, total text, identifiers, header parts."""
    parts: dict[str, dict[str, int]] = {}
    for identifier, declaration in carried:
        counts = parts.setdefault(identifier, {})
        counts[declaration.header] = counts.get(declaration.header, 0) + 1
    listing = "; ".join(
        f"{identifier} in " + ", ".join(f"{header} ({count})" for header, count in counts.items())
        for identifier, counts in parts.items()
    )
    total = sum(len(declaration.text) for _, declaration in carried)
    return (
        f"the prompt carried {len(carried)} declaration(s) from the context pack, {total} characters, for the "
        f"identifiers its errors name, in order: {listing}"
    )


def declarations_described(context: RunContext) -> str:
    """Return what compile_loop's describe() adds when lookup lists correct (task P17.26), else ""."""
    if not lookup_on(context, LOOKUP_CORRECT):
        return ""
    section = context.recipe.data[LOOKUP]
    nearest = section.get(LOOKUP_NEAREST)
    near = f", and up to {nearest} declared names nearest each name it declares nothing of" if nearest else ""
    return (
        f", with the context pack's declarations of the names its errors give (up to {section['max_chars']} "
        f"characters){near}"
    )


def core_described(context: RunContext) -> str:
    """Return what generate's describe() adds when lookup lists generate (2026-10-25), else ""."""
    if not lookup_on(context, LOOKUP_GENERATE):
        return ""
    section = context.recipe.data[LOOKUP]
    return (
        f", with the context pack's declarations of {len(section[LOOKUP_CORE])} header part(s) "
        f"(up to {section['max_chars']} characters)"
    )


def examples_on(context: RunContext, kind: str) -> bool:
    """Return True when the recipe's examples section lists the prompt kind `kind` (generate or correct; OQ-083 (b)).

    The section is off when absent (lassi.core.recipe EXAMPLES); the loader
    refuses it under faithful: true, and the runner with a fragment set or
    a pack that does not match its manifest.
    """
    section = context.recipe.data.get(EXAMPLES)
    return isinstance(section, Mapping) and kind in section["prompts"]


def carried_examples(context: RunContext) -> list[Example]:
    """Return the worked examples one prompt carries: RunContext.examples whole, in pack order, within max_chars.

    An example whose text (its files as FILE blocks, Example.text) would
    take the carried examples' total text past examples.max_chars is
    skipped whole, never cut, and a later smaller one may still fit. Raises
    ValueError when the RunContext carries no pack or the prompt set is a
    fragment set; the runner refuses both first.
    """
    if not context.examples or context.fragments:
        raise ValueError(
            f"{context.recipe.path}: {EXAMPLES} needs a template prompt set and the run's examples pack, which the "
            "runner loads once per run; run the recipe with lassi run"
        )
    budget = context.recipe.data[EXAMPLES]["max_chars"]
    carried: list[Example] = []
    total = 0
    for example in context.examples:
        size = len(example.text)
        if total + size > budget:
            continue
        carried.append(example)
        total += size
    return carried


def examples_block(prompts: str, examples: Sequence[Example]) -> str:
    """Return EXAMPLES_PROMPT of set `prompts` with `examples`, each EXAMPLE_PROMPT rendered, one blank line apart."""
    texts = [render(prompts, EXAMPLE_PROMPT, {"name": item.name, "files": item.text}).rstrip("\n") for item in examples]
    return render(prompts, EXAMPLES_PROMPT, {"examples": BLOCK_JOINER.join(texts)})


def examples_note(context: RunContext, examples: Sequence[Example]) -> Diagnostic:
    """Return the note of a request that carried `examples`: code EXAMPLES_CODE, severity note.

    Its message names the pack, the count and total text, and each carried
    example in order with its file count. Its stage is parse, as for the
    retrieved note, and it rides on the Request, never on an attempt, so it
    never enters a prompt.
    """
    pack = context.recipe.data[EXAMPLES]["pack"]
    total = sum(len(item.text) for item in examples)
    listing = ", ".join(f"{item.name} ({len(item.files)} files)" for item in examples)
    message = (
        f"the prompt carried {len(examples)} worked example(s) from the pack {pack}, {total} characters, in pack "
        f"order: {listing}"
    )
    return Diagnostic(stage="parse", severity="note", code=EXAMPLES_CODE, message=message)


def examples_described(context: RunContext, kind: str) -> str:
    """Return what describe() adds when the examples section lists the prompt kind `kind`, else ""."""
    if not examples_on(context, kind):
        return ""
    section = context.recipe.data[EXAMPLES]
    where = "the prompt" if kind == EXAMPLES_GENERATE else "each compile correction prompt"
    return f", with the {section['pack']} worked examples (up to {section['max_chars']} characters) before {where}"


def skeleton_on(context: RunContext, kind: str) -> bool:
    """Return True when the recipe's skeleton section lists the prompt kind `kind` (generate or correct; OQ-096 (a)).

    The section is off when absent (lassi.core.recipe SKELETON); the loader
    refuses it under faithful: true, and the runner with a fragment set.
    """
    section = context.recipe.data.get(SKELETON)
    return isinstance(section, Mapping) and kind in section["prompts"]


def carried_skeleton(context: RunContext, kind: str) -> Skeleton | None:
    """Return the run's skeleton when the skeleton section lists the prompt kind `kind`, else None.

    Raises ValueError when the section lists it but the RunContext carries
    no skeleton or the prompt set is a fragment set; the runner refuses both
    first.
    """
    if not skeleton_on(context, kind):
        return None
    if context.skeleton is None or context.fragments:
        raise ValueError(
            f"{context.recipe.path}: {SKELETON} needs a template prompt set and the run's skeleton, which the "
            "runner loads once per run; run the recipe with lassi run"
        )
    return context.skeleton


def skeleton_note(skeleton: Skeleton) -> Diagnostic:
    """Return the note of a request that carried `skeleton`: code SKELETON_CODE, severity note, stage parse.

    Its message names the skeleton, its sha256, its host program's size,
    and, for a skeleton with kernel skeletons (task P17.39), how many; it
    rides on the Request, never on an attempt, so it never enters a prompt.
    """
    kernels = f", and {len(skeleton.kernels)} kernel skeletons" if skeleton.kernels else ""
    message = (
        f"the prompt carried the host program skeleton {skeleton.name} (sha256 {skeleton.sha256}, "
        f"{len(skeleton.host)} characters of {skeleton.language} host program{kernels})"
    )
    return Diagnostic(stage="parse", severity="note", code=SKELETON_CODE, message=message)


def skeleton_described(context: RunContext, kind: str) -> str:
    """Return what describe() adds when the skeleton section lists the prompt kind `kind`, else ""."""
    if not skeleton_on(context, kind):
        return ""
    where = "the prompt" if kind == SKELETON_GENERATE else "each compile correction prompt"
    return f", with the {context.recipe.data[SKELETON]['name']} host program skeleton right before {where}"


def _harness_build(
    context: RunContext, toolchain: Toolchain, files: Mapping[str, str], workdir: Path, language: str
) -> BuildResult:
    """Build `files` in `workdir`, with the item's support files for `language` as `harness` when there are any."""
    harness = _support_files(context, language)
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


def _parsed_attempt(
    index: int, prompt_ref: TextRef, reply: str, expected: Sequence[str], *, unfenced_file_lines: bool = False
) -> Attempt:
    """Return the attempt for one model reply: its FILE blocks parsed, S1 when usable and S0 otherwise.

    The reply is usable when its blocks gave at least one file and every
    block error is MISSING_FILE. The diagnostics are parse_file_blocks'
    own, after any `invalid-text` warning. `unfenced_file_lines` is
    parse_file_blocks' leniency: a reply with no FILE block is read from its
    unfenced FILE lines, each file with an `unfenced-file` warning, which no
    rung of the ladder reads.
    """
    text, warnings = _storable(reply)
    parsed = parse_file_blocks(text, expected, unfenced_file_lines=unfenced_file_lines)
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
    """Return the attempt for one reply: FILE blocks with fixes.fence_tag on, the first fenced block with it off.

    With the fix on, extract.unfenced_file_lines (extract_on) lets a reply
    with no FILE block be read from its unfenced FILE lines. The attempt is
    not built yet, so its guards say why host_compute is null
    (_unbuilt_guards); compile_loop changes that when it builds it.
    """
    if fix_on(context, "fence_tag"):
        unfenced = extract_on(context, UNFENCED_FILE_LINES)
        attempt = _parsed_attempt(index, prompt_ref, reply, expected, unfenced_file_lines=unfenced)
    else:
        attempt = _fenced_attempt(index, prompt_ref, reply, expected)
    return dataclasses.replace(attempt, guards=_unbuilt_guards(context))


def _target_guarded(context: RunContext) -> bool:
    """Return True when the target language's toolchain is bound and declares host_compute_guard (by capability)."""
    toolchain = context.toolchains.get(context.direction.target)
    return toolchain is not None and declares(toolchain, HOST_COMPUTE_GUARD)


def _unbuilt_guards(context: RunContext) -> Guards:
    """Return the guards of an attempt not built yet: not-built, or no-guard for a target without a guard (P17.14).

    no-guard wins because no build of that target is ever read
    (_target_guarded; found by capability, never by name).
    """
    return Guards(host_compute_not_checked=NOT_BUILT if _target_guarded(context) else NO_GUARD)


def _with_reason(attempt: Attempt, reason: str | None) -> Attempt:
    """Return `attempt` with Attempt.guards.host_compute_not_checked set to `reason`."""
    return dataclasses.replace(attempt, guards=dataclasses.replace(attempt.guards, host_compute_not_checked=reason))


def _send(
    context: RunContext, trial: Trial, stage: str, prompt: str, system: str | None = None
) -> tuple[list[RequestMessage], str]:
    """Send `prompt` as the user message, after `system` as the system message when given.

    Each message is kept in the text store before anything is sent, and
    then the observer gets a request-sent event with `trial`, the trial
    before this request is recorded, and `stage`, the stage it is recorded
    under (lassi.core.progress). Returns
    the messages as a Request records them (role and text reference, in the
    order sent; the user message is last) and the reply text as returned.
    """
    messages = [Message("user", prompt)] if system is None else [Message("system", system), Message("user", prompt)]
    recorded = [RequestMessage(role=message.role, ref=context.store.put(message.content)) for message in messages]
    notify(context.observer, REQUEST_SENT, trial, stage)
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

        The target reference is built, and run when its language's executor
        runs programs; with the baseline_both fix on, the source reference
        follows (unless the direction's two languages are one), on the source
        language's executor (executor_for). The first failure
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
        """Return a one-line description of the stage: each reference it builds, and runs on its own executor.

        A reference runs only when its language's executor runs programs
        (executor_for), so with a compile-only source executor the source
        reference is described as built, not run.
        """
        direction = self.context.direction
        both = fix_on(self.context, "baseline_both") and direction.source != direction.target
        languages = [direction.target, direction.source] if both else [direction.target]
        running = [
            RUNS_CODE in getattr(executor_for(self.context, self._program(lang)), "capabilities", ())
            for lang in languages
        ]
        only = "" if fix_on(self.context, "baseline_both") else "only "
        parts = [
            f"{'build and run' if runs else 'build'} {only}the {language} reference program{self._as(language)}"
            for language, runs in zip(languages, running, strict=True)
        ]
        text = f"baseline: {' and '.join(parts)} before any model call"
        if any(running) and self.files_oracle is not None:
            text += ", keeping the target reference's output files for the oracle"
        return text

    def _reference(self, trial: Trial, language: str) -> tuple[Trial, RunResult | None]:
        """Build the item's reference in `language` and run it when that language's executor runs programs.

        Returns the trial and the RunResult (None when nothing ran). A build
        with no artifact ends the trial with `baseline-compile`; a run the
        reading of _reference_run_end ends (a simulator gap at `sim-gap`; a
        hang, undefined behavior, or an exit status other than 0 at
        `baseline-run`) ends it there, as do output files the recipe's
        output-file Oracle cannot compare against (reference_problem names
        the file, at `baseline-run`); that check is skipped for a run whose
        workdir came back incomplete, since a cut output alone ends nothing.
        The target's run is kept in Trial.reference_run (_run_info, with its
        output files in the binary store when that Oracle is bound), a failed
        one included; the source reference's files are read but not stored.
        A note that the target's device time was not recorded (a silicon
        run's DEVICE_TIMING_NOTE, task P17.42) goes to
        Trial.baseline_diagnostics, which no attempt carries.
        A language whose manifest entry names a `reference` language (task
        P17.36; lassi.bench Suite.reference_language) has that language's
        program as its reference, built with that language's toolchain and
        support files in <trial>/baseline-<that language>/build and run on
        its executor.
        """
        context = self.context
        files = self._files(language)
        program = self._program(language)
        toolchain = _toolchain(context, program)
        workdir = _baseline_dir(context.build_root, trial.trial_id, program)
        result = _harness_build(context, toolchain, files, workdir, program)
        if result.artifact is None:
            return _ended(trial, BASELINE_COMPILE, self._build_message(program, files, toolchain, result)), None
        executor = executor_for(context, program)
        if RUNS_CODE not in getattr(executor, "capabilities", ()):
            return trial, None
        item = context.suite.item(context.item, purpose=PURPOSE)
        args = stage_inputs(item, result.artifact.parent)
        run = executor.run(result.artifact, args, reference_limits(context))
        target = language == context.direction.target
        info = _run_info(context, run, keep_outputs=target and self.files_oracle is not None, reference=True)
        if target:
            notes = [*trial.baseline_diagnostics, *_timing_notes(run)]
            trial = dataclasses.replace(trial, reference_run=info, baseline_diagnostics=notes)
        end = _reference_run_end(run, language)
        if end is not None:
            return _ended(trial, *end), run
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

    def _program(self, language: str) -> str:
        """Return the language whose program is the item's reference in `language` (Suite.reference_language)."""
        return self.context.suite.reference_language(self.context.item, language)

    def _as(self, language: str) -> str:
        """Return how describe() names a reference that is another language's program, or "" for its own."""
        program = self._program(language)
        return "" if program == language else f" (the item's {program} program)"

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

        The prompt comes from _prompt, after the context blocks of a
        template set (with_context_blocks: the knowledge block, then the
        worked examples of carried_examples when the examples section lists
        generate, then the passages for generate_query when retrieval lists
        generate, then the declarations of core_declarations when lookup lists
        generate, then the skeleton of carried_skeleton when the skeleton
        section lists generate); with
        the prompt_spaces fix off, each run of spaces in it is cut to one.
        It is kept in the text store and sent as the user message, after the
        system prompt when there is one. With the fence_tag fix on the
        reply's FILE blocks are parsed, and with it off its first fenced
        block is read as upstream reads it. The request is recorded with
        attempt 0 as the attempt its reply became, and the blocks' note.
        """
        context = self.context
        expected = target_files(context)
        system, prompt = self._prompt(trial, expected)
        on = retrieval_on(context, RETRIEVAL_GENERATE)
        passages = retrieved_passages(context, generate_query(context)) if on else []
        core = core_declarations(context) if lookup_on(context, LOOKUP_GENERATE) else []
        examples = carried_examples(context) if examples_on(context, EXAMPLES_GENERATE) else []
        skeleton = carried_skeleton(context, SKELETON_GENERATE)
        prompt, notes = with_context_blocks(
            context, trial, prompt, passages, core=core, examples=examples, skeleton=skeleton)
        if not fix_on(context, "prompt_spaces"):
            prompt = fragment_text.collapse_spaces(prompt)
        try:
            messages, reply = _send(context, trial, self.name, prompt, system)
        except ContextExceeded as error:
            return _context_end(trial, self.name, error)
        attempt = _reply_attempt(context, 0, messages[-1].ref, reply, expected)
        trial = _recorded(context, trial.with_attempt(attempt), self.name, 0, messages, attempt.response_text, notes)
        notify(context.observer, ATTEMPT, trial, self.name)
        return trial

    def _prompt(self, trial: Trial, expected: Sequence[str]) -> tuple[str | None, str]:
        """Return the system prompt (None for a template set) and the user prompt.

        A template set gives generate.txt, filled with the direction's
        languages, the item's source files in FILE blocks, the expected
        target file names, and the item's kernels (kernel_files). A fragment
        set gives the direction's system prompt and the generation prompt
        from the source (as text mode reads it), the target language's
        context pack when the recipe has one, and the trial's summary and
        description.
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
                "kernel_files": kernel_files(context),
                "host_files": host_files(context),
                "source_mlir": source_mlir(context),
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
        if fix_on(self.context, "fence_tag") and extract_on(self.context, UNFENCED_FILE_LINES):
            reply += ", or unfenced FILE lines when the reply holds none"
        passages = passages_described(self.context, RETRIEVAL_GENERATE)
        examples = examples_described(self.context, EXAMPLES_GENERATE)
        skeleton = skeleton_described(self.context, SKELETON_GENERATE)
        return (
            f"generate: one {direction.source} to {direction.target} translation from {source}, parsed from "
            f"{reply}{examples}{passages}{core_described(self.context)}{skeleton}"
        )


@register("Stage", "summarize_context")
class SummarizeContextStage:
    """Asks the model to summarize the target language's context pack and keeps the reply in Trial.context."""

    name = "summarize_context"
    capabilities: frozenset[str] = frozenset()
    requires = {"LLMBackend": {"chat"}}
    prompt_fields = {SUMMARY_PROMPT: SUMMARY_FIELDS}
    fragment_keys = fragment_text.SUMMARY_KEYS
    needs_context = True
    fills_context = ("knowledge_summary",)

    def __init__(self, *, context: RunContext) -> None:
        """Keep the trial's run context."""
        self.context = context

    def __call__(self, trial: Trial) -> Trial:
        """Return `trial` with context.knowledge_summary set to the model's reply; no attempt.

        The request (_prompt) goes to the backend unchanged. The reply is
        kept as returned, except that each lone surrogate becomes U+FFFD, and
        the request is recorded (_context_reply).
        """
        context = self.context
        system, prompt = self._prompt(context.packs[context.direction.target])
        try:
            messages, reply = _send(context, trial, self.name, prompt, system)
        except ContextExceeded as error:
            return _context_end(trial, self.name, error)
        return _context_reply(context, trial, self.name, "knowledge_summary", messages, reply)

    def _prompt(self, pack: str) -> tuple[str | None, str]:
        """Return the system prompt (None for a template set) and the summary request for the target's `pack`.

        A template set gives SUMMARY_PROMPT filled with the target language
        and the pack (task P17.21). A fragment set gives the general system
        prompt and the summary request followed by the pack.
        """
        context = self.context
        if not context.fragments:
            fields = {"target_language": context.direction.target, "context_pack": pack}
            return None, render(context.prompts, SUMMARY_PROMPT, fields)
        prompt = fragment_text.summary_request(context.fragments, context.direction, pack)
        return context.fragments[fragment_text.GENERAL_SYSTEM], prompt

    def describe(self) -> str:
        """Return a one-line description of the stage, naming the template under a template set."""
        context = self.context
        source = context.prompts if context.fragments else f"{context.prompts}/{SUMMARY_PROMPT}.txt"
        return f"summarize_context: summarize the {context.direction.target} context pack ({source})"


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
        try:
            messages, reply = _send(context, trial, self.name, prompt, context.fragments[fragment_text.GENERAL_SYSTEM])
        except ContextExceeded as error:
            return _context_end(trial, self.name, error)
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
        built, then the host-compute reading when the toolchain declares
        host_compute_guard (_guarded). A build with no artifact always
        carries an error: "no-artifact" when the toolchain reported none,
        and "unwritable" when a file name could not be written. While the
        latest attempt has an error and is not S4, the stage asks for a
        correction (_correction; a restart when loop.restart_after makes one
        due, _corrected), appends the reply with its diff from the previous
        files and records the request, and builds that. When the
        cap (corrections are the attempts after the first) stops the loop
        with an error remaining, final.end_reason is set to
        `correction-cap`. A cap of None never stops the loop.
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
            corrected = self._correction(trial, expected, stderr)
            if corrected.final.end_reason is not None:
                return corrected
            trial, stderr = self._build_last(corrected)
        return trial

    def describe(self) -> str:
        """Return a one-line description of the stage."""
        context = self.context
        cap = context.max_corrections
        limit = "no correction cap" if cap is None else f"at most {cap} corrections"
        toolchain = context.toolchains.get(context.direction.target)
        builder = "no bound toolchain" if toolchain is None else _toolchain_name(toolchain)
        if toolchain is not None and declares(toolchain, HOST_COMPUTE_GUARD):
            builder += f" ({HOST_COMPUTE_GUARD} after each build)"
        if toolchain is not None and declares(toolchain, STATIC_TAGS):
            builder += f" ({STATIC_TAGS} after each build)"
        source = f"the {context.prompts} fragments" if context.fragments else f"{context.prompts}/correct.txt"
        errors = "parsed diagnostics" if fix_on(context, "parsed_diagnostics") else "the raw compiler stderr"
        return (
            f"compile_loop: build {context.direction.target} with {builder}, then correct from {source} with "
            f"{errors} while an error remains ({limit}{restarts_described(context)})"
            f"{examples_described(context, EXAMPLES_CORRECT)}"
            f"{passages_described(context, RETRIEVAL_CORRECT)}{declarations_described(context)}"
            f"{skeleton_described(context, SKELETON_CORRECT)}"
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
        the attempt's index, and the attempt gains the target toolchain's
        host-compute reading (_guarded) before the attempt event; with no
        program its host_compute_not_checked becomes no-program, or no-guard
        for a target without a guard (task P17.14). The stderr
        is read only with fixes.parsed_diagnostics off, the one case a
        prompt carries it, and is None otherwise or when the build kept no
        attachment (BuildResult.stderr_ref).
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
        if result.artifact is not None:
            built = self._tagged(self._guarded(built))
        else:
            built = _with_reason(built, NO_PROGRAM if _target_guarded(self.context) else NO_GUARD)
        trial = dataclasses.replace(trial, attempts=[*trial.attempts[:-1], built])
        notify(self.context.observer, ATTEMPT, trial, self.name)
        if fix_on(self.context, "parsed_diagnostics"):
            return trial, None
        return trial, _attachment_text(workdir, result.stderr_ref)

    def _guarded(self, attempt: Attempt) -> Attempt:
        """Return `attempt` with the target toolchain's host-compute reading (task P4.12); no-guard without one.

        The toolchain is found by capability (host_compute_reading), never by
        name, and is given the attempt's files and the item's support files
        for the target language.
        The reading sets Attempt.guards.host_compute, and its diagnostics
        follow the parse and build diagnostics. Attempt.guards.host_compute_not_checked
        (task P17.14) becomes no-guard without a reading, guard-not-checked
        for a null reading, and None for a true or false one. A reading
        outside the guard contract raises ValueError, which stops the run
        before the attempt is recorded.
        """
        context = self.context
        toolchain = _toolchain(context, context.direction.target)
        reading = host_compute_reading(toolchain, attempt.files, _support_files(context, context.direction.target))
        if reading is None:
            return _with_reason(attempt, NO_GUARD)
        reason = GUARD_NOT_CHECKED if reading.host_compute is None else None
        guards = dataclasses.replace(attempt.guards, host_compute=reading.host_compute, host_compute_not_checked=reason)
        return dataclasses.replace(attempt, guards=guards, diagnostics=[*attempt.diagnostics, *reading.diagnostics])

    def _tagged(self, attempt: Attempt) -> Attempt:
        """Return `attempt` with the target toolchain's static tags after its diagnostics (task P17.40).

        The toolchain is found by capability (static_tags), never by name, and
        is given the attempt's files and the item's support files for the
        target language. Tags change no stage and no guard, and a tag outside
        the guard contract raises ValueError, which stops the run before the
        attempt is recorded.
        """
        context = self.context
        toolchain = _toolchain(context, context.direction.target)
        if not declares(toolchain, STATIC_TAGS):
            return attempt
        tags = static_tags(toolchain, attempt.files, _support_files(context, context.direction.target))
        return dataclasses.replace(attempt, diagnostics=[*attempt.diagnostics, *tags]) if tags else attempt

    def _build(self, files: Mapping[str, str], workdir: Path) -> BuildResult:
        """Build `files` in `workdir`; a file name the filesystem refuses becomes an "unwritable" error.

        The item's support files for the target language from the suite
        manifest go into the build directory as harness files, passed as the
        toolchain's `harness` argument only when there are some. Only a name
        error (too long, or refused as invalid) is the model's to fix; any
        other OSError, such as a full disk, propagates and stops the run.
        """
        context = self.context
        try:
            target = context.direction.target
            return _harness_build(context, _toolchain(context, target), files, workdir, target)
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
        diagnostics_text of the attempt's diagnostics without a guard's
        (prompt_diagnostics): upstream reads an empty stderr as a success, so
        a failed build that left none (a sandbox timeout, say) has no
        upstream counterpart.
        """
        errors = stderr if stderr else diagnostics_text(prompt_diagnostics(trial.attempts[-1].diagnostics))
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
    `mismatch_prompt_fields` names the template an oracle mismatch's
    correction renders and its fields: a stage that declares it honors
    loop.correct_on_mismatch, and the runner renders it, before any
    directory exists, only when that key is true. It honors
    loop.correct_on_sim_gap too (module docstring, Simulator gaps).
    """

    name = "run_loop"
    capabilities = frozenset({"compiles"})
    requires = {"Toolchain": {"diagnostics"}}
    prompt_fields = {"correct": CORRECT_FIELDS}
    mismatch_prompt_fields = {MISMATCH_PROMPT: MISMATCH_FIELDS}
    fragment_keys = RUN_LOOP_KEYS
    reproduces = frozenset({"execution_gate", "fence_tag", "prompt_newlines", "parsed_diagnostics"})
    runs_model_code = True

    def __init__(self, *, context: RunContext) -> None:
        """Keep the trial's run context, the recipe's Oracle when it compares output files, and the loop opt-ins.

        Raises ValueError when loop.correct_on_mismatch is true but the
        recipe binds no Oracle that compares output files, and when
        loop.correct_on_sim_gap is true with a fragment prompt set; the
        runner refuses such a recipe first.
        """
        self.context = context
        self.files_oracle = output_file_oracle(context)
        self.keeps_outputs = self.files_oracle is not None
        self.on_mismatch = corrects_on_mismatch(context)
        self.on_gap = corrects_on_sim_gap(context)
        if self.on_mismatch and self.files_oracle is None:
            raise ValueError(
                f"{context.recipe.path}: loop.{CORRECT_ON_MISMATCH} is true, but the recipe binds no Oracle that "
                f"declares {ALIGNS_OUTPUT_FILES!r}, whose per-output statistics its correction prompt's report reads"
            )
        if self.on_gap and context.fragments:
            raise ValueError(
                f"{context.recipe.path}: loop.{CORRECT_ON_SIM_GAP} is true, but {context.prompts!r} is a fragment "
                "prompt set, upstream's prompts, which hold no simulator-gap correction"
            )

    def __call__(self, trial: Trial) -> Trial:
        """Return `trial` with its compiling attempts run and any corrections appended.

        compile_loop runs first (a no-op when it already ran). With a
        compile-only executor that is all; an executor that runs programs
        but does not declare SANDBOXED raises ValueError before anything
        runs (Agent Rule 6). Otherwise, while the last attempt
        is S4 and the trial has not ended: past the execution gate (fix off)
        the trial ends unexecuted (_past_gate); else the attempt runs
        (_run_last). A clean run ends the loop. A kernel JIT failure leaves
        the attempt at S1, and compile_loop corrects it as a compile error,
        under the same cap. A simulator gap ends the trial with `sim-gap`
        (_gap_message), with no correction, unless loop.correct_on_sim_gap
        is true: then it is read as a failed run is, with the gap text
        (sim_gap_text), and a gap left at the cap ends the trial with
        `sim-gap` (_cap_end; module docstring, Simulator gaps). A failed run
        with the cap reached ends the trial with `correction-cap`; below it,
        the execute-error correction is appended and compile_loop builds and
        corrects it. Under loop.correct_on_mismatch a clean run is judged
        (_judged), and an oracle mismatch is read as a failed run is, with
        the mismatch prompt (module docstring, Oracle mismatches).
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
            reached = trial.attempts[-1].stage_reached
            if reached == PARSED:
                trial = compile_loop(trial)
                continue
            outputs = None
            if reached == RAN_CLEAN:
                trial, outputs = self._judged(trial)
                if outputs is None:
                    break
            elif run.sim_gap is not None and not self.on_gap:
                return _ended(trial, SIM_GAP, _gap_message(index, run.sim_gap))
            if cap is not None and index >= cap:
                return _cap_end(trial, cap, outputs, run.sim_gap)
            corrected = self._correction(trial, run, limits, outputs)
            if corrected.final.end_reason is not None:
                return corrected
            trial = compile_loop(corrected)
        return trial

    def _correction(self, trial: Trial, run: RunResult, limits: Limits, outputs: str | None) -> Trial:
        """Return `trial` with the correction of its last attempt appended (_corrected), recorded under run_loop.

        With `outputs` None the last run failed: the execute-error prompt,
        whose error text is run_error_text's, or, for a run that stopped at
        a simulator gap (read so only under loop.correct_on_sim_gap),
        sim_gap_text's. Otherwise `outputs` is the oracle mismatch's
        mismatch_text, sent with MISMATCH_PROMPT.
        """
        context = self.context
        expected = target_files(context)
        if outputs is not None:
            return _corrected(context, trial, self.name, expected, outputs, run_error=True, mismatch=True)
        if run.sim_gap is not None:
            return _corrected(context, trial, self.name, expected, sim_gap_text(run), run_error=True)
        errors = run_error_text(run, limits, context.fragments, simulator=declares(context.executor, SIMULATOR))
        return _corrected(context, trial, self.name, expected, errors, run_error=True)

    def _judged(self, trial: Trial) -> tuple[Trial, str | None]:
        """Return `trial` with its clean last attempt judged by the output-file Oracle, and the mismatch text or None.

        Nothing is judged, and the trial comes back unchanged with None,
        unless loop.correct_on_mismatch is true, the attempt's run recorded
        output files, and the target reference recorded complete ones: the
        oracle stage aligns no other attempt either. Otherwise the attempt
        takes the Oracle's Alignment, bound as the oracle stage binds it
        (agreement_bound); when every output passes the text is None, and
        when one fails the attempt also gains an ORACLE_MISMATCH run-stage
        error naming each failed output, and the text is mismatch_text.
        """
        attempt, reference = trial.attempts[-1], trial.reference_run
        if self.files_oracle is None or not self.on_mismatch or attempt.run.outputs is None:
            return trial, None
        if reference.outputs is None or reference.workdir_incomplete is True:
            return trial, None
        oracle = agreement_bound(self.files_oracle, trial)
        blobs = BlobStore(self.context.store.root)
        aligned = oracle.alignment(stored_files(blobs, reference.outputs), stored_files(blobs, attempt.run.outputs))
        stats = aligned.outputs or []
        failed = [entry for entry in stats if not entry.passed]
        judged = dataclasses.replace(attempt, alignment=aligned)
        if failed:
            error = _mismatch_error(oracle, failed, len(stats))
            judged = dataclasses.replace(judged, diagnostics=[*attempt.diagnostics, error])
        trial = dataclasses.replace(trial, attempts=[*trial.attempts[:-1], judged])
        return trial, (mismatch_text(oracle, stats) if failed else None)

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
        mismatch = ""
        if self.on_mismatch:
            mismatch = f" and each clean run whose output files fail the oracle (loop.{CORRECT_ON_MISMATCH})"
        if self.on_gap:
            mismatch += f" and each run that stopped at a simulator gap (loop.{CORRECT_ON_SIM_GAP})"
        passages = passages_described(context, RETRIEVAL_CORRECT)
        return (
            f"run_loop: run {gate} and correct each failed run{mismatch}, sharing compile_loop's correction count"
            f"{restarts_described(context)}{passages}"
        )

    def _run_last(self, trial: Trial) -> tuple[Trial, RunResult, Limits]:
        """Run the last attempt's program; return the trial with its run recorded, the RunResult, and the limits.

        A backend that declares `unload_before_run` is asked to unload first.
        The attempt keeps the run in Attempt.run (_run_info: stdout in the
        text store, sim_ub as reported, each RunResult flag as a bool, and,
        when the recipe's Oracle compares output files, every output file in
        the binary store by hash), gains the executor's diagnostics of the
        run (RunResult.diagnostics) and one warning per RunResult flag that
        is set (RUN_FLAGS), and reaches the stage run_stage gives. A failed
        run that is neither a JIT failure (S1) nor a simulator gap stays S4
        with a `run-error` whose message is _run_status's. A simulator gap
        (S4) gets no run-error; under loop.correct_on_sim_gap it gets the
        SIM_GAP_NOTE note instead (gap_note).
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
        run_args = stage_inputs(context.suite.item(context.item, purpose=PURPOSE), artifact.parent)
        unload_before_run(context.backend)
        run, profile = profiled_run(context, artifact, run_args, limits)
        info = _run_info(context, run, keep_outputs=self.keeps_outputs)
        diagnostics = [*attempt.diagnostics, *run.diagnostics, *_run_flag_warnings(run)]
        reached = run_stage(run)
        if reached == COMPILED and run.sim_gap is None:
            status = _run_status(run, limits, simulator=declares(context.executor, SIMULATOR))
            diagnostics.append(Diagnostic(stage="run", severity="error", code=RUN_ERROR, message=status))
        elif reached == COMPILED and run.sim_gap is not None and self.on_gap:
            diagnostics.append(gap_note(run.sim_gap))
        ran = dataclasses.replace(attempt, run=info, profile=profile, diagnostics=diagnostics, stage_reached=reached)
        trial = dataclasses.replace(trial, attempts=[*trial.attempts[:-1], ran])
        notify(context.observer, ATTEMPT, trial, self.name)
        return trial, run, limits


def profiled_run(
    context: RunContext, artifact: Path, inputs: Sequence[str], limits: Limits
) -> tuple[RunResult, Profile]:
    """Run `artifact` on context.executor, profiled by context.profiler when there is one; return the run and Profile.

    Without a profiler the executor runs alone and the Profile is Profile()
    (nothing measured). With one, start() comes right before the run and
    stop() in a finally block right after it, so stop() runs, and a power
    profiler's sampling thread ends, when the executor raises too; the
    executor's error then propagates and stop()'s result is dropped. Only
    context.profiler and context.executor are read.
    """
    profiler = context.profiler
    if profiler is None:
        return context.executor.run(artifact, inputs, limits), Profile()
    profiler.start()
    try:
        run = context.executor.run(artifact, inputs, limits)
    finally:
        stopped = profiler.stop()
    return run, checked_profile(profiler, stopped)


def checked_profile(profiler: Profiler, value: object) -> Profile:
    """Return `value`, a profiler's stop() result, when it is a Profile inside the Profiler contract.

    Raises ValueError, naming the profiler, for a result that is not a
    lassi.core.record.Profile, and for a Profile with avg_power_w or
    energy_j set from a profiler that does not declare supports_power (only
    a profiler that reads power telemetry may report power).
    """
    name = getattr(profiler, "name", type(profiler).__name__)
    if not isinstance(value, Profile):
        raise ValueError(f"Profiler {name!r} returned {type(value).__name__} from stop(), not a Profile")
    if not declares(profiler, SUPPORTS_POWER) and (value.avg_power_w is not None or value.energy_j is not None):
        raise ValueError(
            f"Profiler {name!r} reported power, but it does not declare {SUPPORTS_POWER}: "
            f"avg_power_w {value.avg_power_w!r}, energy_j {value.energy_j!r}"
        )
    return value


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


def run_stage(run: RunResult) -> str:
    """Return the stage an attempt reaches with the run `run` (task P4.6).

    S1 (PARSED) when the run's diagnostics hold a jit-stage error: the
    kernels did not compile, and S4 needs host and kernel JIT. Otherwise S5
    (RAN_CLEAN) after a clean run, exit status 0 with no hang, no undefined
    behavior (sim_ub True), and no simulator gap, and S4 (COMPILED) after
    any other run.
    """
    if any(item.stage == JIT and item.severity == "error" for item in run.diagnostics):
        return PARSED
    clean = run.exit_code == 0 and not run.hang and run.sim_ub is not True and run.sim_gap is None
    return RAN_CLEAN if clean else COMPILED


def run_findings(run: RunResult, simulator: bool) -> list[str]:
    """Return what the stages say about a failed run's simulator findings, in order; [] when there is none.

    UB_FINDING when the run reported undefined behavior (sim_ub True), and,
    when it hung on an executor that declares SIMULATOR (`simulator`), the
    Harness Contract's hang diagnostic, HANG_DIAGNOSTIC, followed by the
    message of each Watcher dump the executor added to the run, in order:
    each of the run's diagnostics with stage run, severity note, and code
    WATCHER_CODE (task P4.11). A run that did not hang, or ran on an
    executor that does not declare SIMULATOR, gets neither.
    """
    findings = [UB_FINDING] if run.sim_ub is True else []
    if run.hang and simulator:
        findings.append(HANG_DIAGNOSTIC)
        findings += [item.message for item in run.diagnostics if _is_watcher_note(item)]
    return findings


def _is_watcher_note(item: Diagnostic) -> bool:
    """Return True for a Watcher dump an executor added to a hung run: a run-stage note coded WATCHER_CODE."""
    return item.stage == "run" and item.severity == "note" and item.code == WATCHER_CODE


def run_error_text(run: RunResult, limits: Limits, fragments: Mapping[str, str], *, simulator: bool = False) -> str:
    """Return the error text of the execute-error prompt for the failed run `run`.

    With a fragment set (`fragments` not empty) it is upstream's report of
    the run, the notebook's execute_code return_result: RUN_REPORT_EXIT, the
    return code as Popen gives it (popen_return_code), one space,
    RUN_REPORT_SEGFAULT when that code is POPEN_SEGFAULT, then, when the run
    wrote any stderr, RUN_REPORT_STDERR and the stderr. A hang has no
    upstream counterpart (upstream waits for the program without a limit);
    its report is built the same way from the status the executor gives.
    Upstream has no simulator either, so the run's findings (run_findings,
    with `simulator` saying whether the executor declares SIMULATOR)
    follow the report after one line feed, joined by "; ", when there
    are any; a run with none gets upstream's report alone. With
    fixes.prompt_newlines off, _corrected removes that line feed as it
    removes every other, so the findings then follow the report directly.
    With a template set it is this project's wording [DESIGN]: how the run
    ended and its findings (_run_status), then the stderr when there is any.
    Either way the stderr is whole (the executor caps it) and read as text
    mode reads it, as upstream reads the program's output.
    """
    stderr = fragment_text.as_text_mode(run.stderr)
    if not fragments:
        status = _run_status(run, limits, simulator=simulator)
        return f"{status}; its standard error follows:\n{stderr}" if stderr else status
    code = popen_return_code(run.exit_code)
    report = fragments[RUN_REPORT_EXIT] + str(code) + " "
    if code == POPEN_SEGFAULT:
        report += fragments[RUN_REPORT_SEGFAULT]
    if stderr:
        report += fragments[RUN_REPORT_STDERR] + stderr
    findings = run_findings(run, simulator)
    return f"{report}\n{'; '.join(findings)}" if findings else report


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


def _run_status(run: RunResult, limits: Limits, *, simulator: bool = False) -> str:
    """Return one sentence saying how a failed run ended, then its findings (run_findings), each after "; ".

    How it ended: stopped at its wall limit, no exit status, or its exit
    status; a run with no finding gets that clause alone, as before task
    P4.6.
    """
    if run.hang:
        status = f"the program was stopped at its wall limit of {limits.wall_s:g} s"
    elif run.exit_code is None:
        status = "the program ended with no exit status"
    else:
        status = f"the program exited with status {run.exit_code}"
    return "; ".join([status, *run_findings(run, simulator)])


def _gap_message(index: int, gap: str) -> str:
    """Return the end reason message of a trial whose attempt `index` stopped at the simulator gap `gap`."""
    return (
        f"the run of attempt {index} stopped at a simulator gap, {gap}: the simulator cannot run the program, which "
        "is not a model error, so the trial stops with no correction"
    )


def _gap_cap_message(index: int, gap: str, cap: int) -> str:
    """Return the end reason message of a trial whose attempt `index` stopped at the gap `gap` with the cap reached.

    Under loop.correct_on_sim_gap only. It shares _gap_message's opening
    words, which name the attempt and the gap's class, so a report reads
    the class from either (plans/runs/rngd-sweep/tabulate.py).
    """
    return (
        f"the run of attempt {index} stopped at a simulator gap, {gap}, after {cap} correction(s), the cap "
        "loop.max_corrections sets: the program built, but the simulator cannot run it, which is not a model error, "
        "so the attempt stands as built but not verifiable on the simulator"
    )


def _cap_end(trial: Trial, cap: int, outputs: str | None, gap: str | None) -> Trial:
    """Return `trial` ended at the cap with its last attempt's run read: a gap, an oracle mismatch, or a run error.

    A run that stopped at the simulator gap `gap` (read so only under
    loop.correct_on_sim_gap) ends it at SIM_GAP, not CORRECTION_CAP: a gap
    is not an error that remained, so the lassi profile's cap_hit reads 0,
    and the profiles read a gap attempt, by that end reason or by its note
    (lassi.core.record gap_noted), as holding no evidence either way
    (lassi.scoring). Otherwise it ends at CORRECTION_CAP, naming an oracle
    mismatch when `outputs` is its report, else a run error.
    """
    index = trial.attempts[-1].index
    if gap is not None:
        return _ended(trial, SIM_GAP, _gap_cap_message(index, gap, cap))
    left = "a run error" if outputs is None else "an oracle mismatch"
    message = f"{left} remained after {cap} correction(s), the cap loop.max_corrections sets"
    return _ended(trial, CORRECTION_CAP, message)


def gap_note(gap: str) -> Diagnostic:
    """Return the SIM_GAP_NOTE run-stage note of an attempt whose run stopped at the simulator gap `gap` (OQ-098).

    run_loop adds it under loop.correct_on_sim_gap, so the record marks the
    attempt as built but not verifiable on the simulator, never a failed
    run, and the score profiles read it (lassi.core.record gap_noted); its
    message names the class after the words _gap_message opens with, which
    a report reads.
    """
    message = (
        f"the run stopped at a simulator gap, {gap}: built, but not verifiable on the simulator, which cannot run the "
        "program; a gap is not a model failure"
    )
    return Diagnostic(stage="run", severity="note", code=SIM_GAP_NOTE, message=message)


def sim_gap_text(run: RunResult) -> str:
    """Return the error text of a simulator gap's correction (loop.correct_on_sim_gap; OQ-098) [DESIGN].

    GAP_HEAD, the gap's class (RunResult.sim_gap), and GAP_LIMIT: the
    simulator cannot run a construct the program uses, which is a limit of
    the simulator and not necessarily an error in the program, so an
    equivalent construct the simulator runs should take its place if one
    exists. Then, after a line feed, GAP_FINDINGS and the run's findings of
    that class (gap_findings), one per line, when there are any; then, after
    a line feed, the stderr as run_error_text gives it under a template set,
    when there is any. It goes into correct.txt as a failed run's text does;
    the runner refuses the key with a fragment set, whose upstream prompts
    hold no such correction.
    """
    parts = [f"{GAP_HEAD}{run.sim_gap}{GAP_LIMIT}"]
    findings = gap_findings(run)
    if findings:
        parts.append("\n".join([GAP_FINDINGS, *findings]))
    stderr = fragment_text.as_text_mode(run.stderr)
    if stderr:
        parts.append(f"its standard error follows:\n{stderr}")
    return "\n".join(parts)


def gap_findings(run: RunResult) -> list[str]:
    """Return the messages of the run's findings of its gap's class, each once, in order, within the caps.

    A finding is a run-stage Diagnostic of RunResult.diagnostics whose code
    is RunResult.sim_gap, as an executor that declares SIMULATOR reports one
    (lassi.executors.ttsim read_findings); none is read by the executor's
    name. A repeated message counts once. At most the diagnostics a
    correction prompt lists (listed_diagnostics) are kept, each on one line,
    and a last line says how many were left out when any were.
    """
    unique: dict[str, Diagnostic] = {}
    for item in run.diagnostics:
        if item.stage == "run" and item.code == run.sim_gap:
            unique.setdefault(item.message, item)
    listed = listed_diagnostics(list(unique.values()))
    lines = [" ".join(item.message.splitlines()) for item in listed]
    if len(listed) < len(unique):
        lines.append(f"[{len(unique) - len(listed)} more finding(s) not shown]")
    return lines


def _mismatch_error(oracle: OutputFileOracle, failed: Sequence[OutputStats], count: int) -> Diagnostic:
    """Return the ORACLE_MISMATCH run-stage error of a clean run whose `failed` outputs, of `count`, failed the oracle.

    Its message names each failed output with its output_reading, the
    correction prompt's coarse words; the attempt's Alignment keeps the
    exact statistics in the record.
    """
    named = "; ".join(f"{entry.name} ({output_reading(oracle, entry)})" for entry in failed)
    message = f"the run was clean, but {len(failed)} of {count} output(s) failed the oracle on the held-out inputs: "
    return Diagnostic(stage="run", severity="error", code=ORACLE_MISMATCH, message=message + named)


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
    context: RunContext,
    trial: Trial,
    stage: str,
    expected: Sequence[str],
    errors: str,
    *,
    run_error: bool,
    mismatch: bool = False,
) -> Trial:
    """Return `trial` with the next attempt appended and its request recorded under `stage`.

    The correction prompt for the last attempt with `errors` comes from
    _correction_messages (`mismatch`: an oracle mismatch's, with `errors`
    its mismatch_text), after the context blocks of a template set
    (with_context_blocks: the knowledge block, then, when the examples
    section lists correct and this is a compile correction, no run error or
    mismatch, the worked examples of carried_examples, then, when retrieval
    lists correct and this is no mismatch, the passages for
    correction_query, then, when lookup lists correct and this is a compile
    correction, the declarations of looked_up_declarations and, with
    lookup.nearest, the nearest names of undeclared_names, then, when the
    skeleton section lists correct and this is a compile correction, the
    skeleton of carried_skeleton).
    With fixes.prompt_newlines off every line feed is removed from it
    before it is stored and sent. The reply is read as generate reads
    attempt 0 (_reply_attempt), and the attempt keeps its diff from the last
    attempt's files; the request keeps the blocks' note. A request the
    backend refuses as past its context (ContextExceeded) ends the trial at
    CONTEXT_EXCEEDED instead, with no attempt appended (_context_end). When
    a restart is due (restart_due, loop.restart_after), the request is a
    restart in place of the correction (_restarted).
    """
    if restart_due(context, trial):
        return _restarted(context, trial, stage, expected)
    previous = trial.attempts[-1]
    system, prompt = _correction_messages(context, previous, expected, errors, run_error=run_error, mismatch=mismatch)
    passages: list[Passage] = []
    if not mismatch and retrieval_on(context, RETRIEVAL_CORRECT):
        passages = retrieved_passages(context, correction_query(previous, errors, run_error=run_error))
    looked_up: list[tuple[str, Declaration]] = []
    undeclared: list[tuple[str, tuple[str, ...]]] = []
    if not (mismatch or run_error) and lookup_on(context, LOOKUP_CORRECT):
        looked_up = looked_up_declarations(context, previous)
        undeclared = undeclared_names(context, previous)
    examples: list[Example] = []
    skeleton: Skeleton | None = None
    if not (mismatch or run_error):
        examples = carried_examples(context) if examples_on(context, EXAMPLES_CORRECT) else []
        skeleton = carried_skeleton(context, SKELETON_CORRECT)
    prompt, notes = with_context_blocks(
        context, trial, prompt, passages, looked_up, undeclared, examples=examples, skeleton=skeleton)
    if not fix_on(context, "prompt_newlines"):
        prompt = prompt.replace("\n", "")
    try:
        messages, reply = _send(context, trial, stage, prompt, system)
    except ContextExceeded as error:
        return _context_end(trial, stage, error)
    attempt = _reply_attempt(context, previous.index + 1, messages[-1].ref, reply, expected)
    attempt = dataclasses.replace(attempt, diff_from_previous=unified_diff(previous.files, attempt.files))
    trial = _recorded(
        context, trial.with_attempt(attempt), stage, attempt.index, messages, attempt.response_text, notes
    )
    notify(context.observer, ATTEMPT, trial, stage)
    return trial


def _restarted(context: RunContext, trial: Trial, asking: str, expected: Sequence[str]) -> Trial:
    """Return `trial` with a restart appended: attempt 0's prompt sent again, recorded under RESTART (task P17.25).

    The one user message is attempt 0's prompt byte for byte, read back from
    the text store: generate.txt as attempt 0 had it, its context blocks
    included, with no previous files, diagnostics, or run report. The
    request keeps the notes attempt 0's request kept (its passages' note).
    The reply is read as generate reads attempt 0 (_reply_attempt), the
    attempt keeps its diff from the last attempt's files, and the
    request-sent and attempt events name RESTART. A fragment set raises
    ValueError naming `asking`, the loop that asked (the runner refuses the
    key first). A request past the model's context ends the trial at
    CONTEXT_EXCEEDED (_context_end), with no attempt appended.
    """
    first, previous = trial.attempts[0], trial.attempts[-1]
    if context.fragments or first.prompt_ref is None:
        raise ValueError(
            f"{trial.trial_id}: {asking} is due a restart (loop.{RESTART_AFTER}), which sends attempt 0's template "
            "prompt again, but the prompt set is a fragment set or attempt 0 kept no prompt"
        )
    notes = next((request.diagnostics for request in trial.requests or [] if request.attempt_index == 0), [])
    try:
        messages, reply = _send(context, trial, RESTART, context.store.get(first.prompt_ref))
    except ContextExceeded as error:
        return _context_end(trial, RESTART, error)
    attempt = _reply_attempt(context, previous.index + 1, messages[-1].ref, reply, expected)
    attempt = dataclasses.replace(attempt, diff_from_previous=unified_diff(previous.files, attempt.files))
    trial = _recorded(
        context, trial.with_attempt(attempt), RESTART, attempt.index, messages, attempt.response_text, notes
    )
    notify(context.observer, ATTEMPT, trial, RESTART)
    return trial


def _correction_messages(
    context: RunContext,
    previous: Attempt,
    expected: Sequence[str],
    errors: str,
    *,
    run_error: bool,
    mismatch: bool = False,
) -> tuple[str | None, str]:
    """Return the system prompt (None for a template set) and the correction prompt for `previous` with `errors`.

    A template set gives correct.txt, with `errors` as its diagnostics and
    the item's kernels (kernel_files), or, with `mismatch`, MISMATCH_PROMPT
    with `errors` as its outputs (loop.correct_on_mismatch). A fragment set
    gives the direction's system prompt and upstream's correction prompt,
    whose code is the previous target file (FILE blocks when the target has
    more than one file): the compile-error form, or with `run_error` the
    execute-error form, which joins the execute-error lead
    (CORRECT_RUN_HEAD, CORRECT_RUN_TAIL) around the same compiler and flag
    text. Upstream has no mismatch prompt, so `mismatch` with a fragment set
    raises ValueError (the runner refuses the key first).
    """
    if not context.fragments:
        name, field_name = (MISMATCH_PROMPT, "outputs") if mismatch else ("correct", "diagnostics")
        fields = {
            "target_language": context.direction.target,
            "files": render_file_blocks(previous.files) if previous.files else "",
            field_name: errors,
            "target_files": ", ".join(expected),
            "kernel_files": kernel_files(context),
            "host_files": host_files(context),
        }
        return None, render(context.prompts, name, fields)
    if mismatch:
        raise ValueError(
            f"loop.{CORRECT_ON_MISMATCH}: the fragment prompt set {context.prompts!r} holds upstream's prompts, "
            "which have no oracle-mismatch correction"
        )
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
