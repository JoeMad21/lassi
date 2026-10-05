"""The oracle stage: align each run's stdout or output files with the reference's (bible Oracles; Stage row).

The stage is built as `factory(context=<RunContext>)` once per trial, like
every stage (lassi.core.stages). The recipe's Oracle must declare one of
ORACLE_CAPABILITIES; the runner refuses one that declares neither before
any directory exists, and so does the stage.

Output files (task P4.4): an Oracle that declares `aligns_output_files`
(lassi.core.capabilities ALIGNS_OUTPUT_FILES; lassi.oracles.binary_io) is
built from its section. Called on a trial, the stage aligns every attempt
whose run recorded its output files (run.outputs set) against the target
reference's (Trial.reference_run.outputs), both read from the binary store
beside the context's text store (lassi.core.store BlobStore): the attempt
gets the oracle's Alignment, per_input [v] and mean v with v 1.0 when every
output passes, else 0.0, and its statistics per output in
Alignment.outputs. An attempt whose run recorded none keeps its alignment
unset, and nothing else in the trial changes; a trial whose attempts
recorded no output files comes back unchanged. A trial whose attempts
recorded output files while its reference recorded none raises ValueError.
The stage never aligns output files against a target reference run whose
workdir came back incomplete (Trial.reference_run.workdir_incomplete true),
as it never aligns stdout against a truncated reference stdout: every
alignment stays unset, each attempt whose run recorded output files gets
one run-stage warning, code reference-workdir-incomplete
(unaligned_output_files), and nothing raises, whatever the threshold. A
flag that is false or not recorded aligns as before. Only then is an
oracle whose thresholds come from the references' agreement
(agreement_setting, as for threshold from_baseline) bound to
Trial.reference_agreement; a trial that recorded none raises ValueError
naming the setting (and any reference-workdir-incomplete note in
Trial.baseline_diagnostics, as when only the source reference's workdir
came back incomplete).

Stdout: an Oracle that declares `masks_stdout` (lassi.oracles.stdout_mask)
is bound to the trial's item: the item's masks from
`assets/harness/masks/<suite>.yaml`, and whether the manifest lists the
direction's target language under the item's `passfail`. The recipe's
`oracle.passfail` stays the oracle's own choice. The rest of this docstring
is about stdout.

`align_runs(trial, reference_stdout)` returns a new trial in which every
attempt whose run holds stdout (`run.stdout_ref`, read through the text
store) carries Alignment(per_input=[v], mean=v), v being the oracle's value
for that stdout against the reference; that includes an attempt whose
stdout later stands as the trial's stale output. An attempt that never ran
keeps its alignment unset, and nothing else in the trial changes.

Called as a stage, it returns a trial with no run stdout unchanged (every
compile-only trial). Otherwise it aligns the runs against the stdout of the
target reference's run, which the baseline stage keeps in
Trial.reference_run (stdout_ref, read through the text store), exactly as
align_runs does; a trial holding run stdout but no reference stdout raises
rather than go unaligned. A reference stdout the executor cut at its output
cap (reference_run.stdout_truncated True) is never aligned against: every
alignment stays unset, and each attempt whose run holds stdout gains one
run-stage warning, code REFERENCE_TRUNCATED (unaligned_runs). A flag that
is False or not recorded (None) aligns as before, and a cut stderr or an
incomplete workdir does not bear on a stdout oracle.

The oracle class is looked up in the default registry, since a RunContext
carries no registry; the runner checks the binding against its own registry
when the recipe loads, and builds it once from its section before any
directory is created.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Protocol, cast

from lassi.core.capabilities import ALIGNS_OUTPUT_FILES, OutputFileOracle
from lassi.core.interfaces import RunResult
from lassi.core.record import Attempt, Diagnostic, RunInfo, Trial
from lassi.core.registry import DEFAULT_REGISTRY, Binding, register
from lassi.core.stages import PURPOSE, REFERENCE_WORKDIR_INCOMPLETE, RunContext
from lassi.core.store import BlobStore
from lassi.oracles import alignment, load_masks

# The capability the stage needs from its Oracle: `for_item(masks, prints_passfail=...)` returns an ItemOracle
# (lassi.oracles.stdout_mask).
MASKS_STDOUT = "masks_stdout"
# The capabilities the stage accepts from its Oracle, one of which it must declare: stdout masks or output files.
ORACLE_CAPABILITIES = (MASKS_STDOUT, ALIGNS_OUTPUT_FILES)

# The run-stage warning an attempt gets when its run is not aligned because the reference stdout was cut.
REFERENCE_TRUNCATED = "reference-stdout-truncated"
REFERENCE_TRUNCATED_MESSAGE = (
    "the reference run's stdout was truncated at the executor's output cap, so this run's stdout was not aligned "
    "against it and its alignment stays unset"
)
# The message of the run-stage warning (code REFERENCE_WORKDIR_INCOMPLETE, lassi.core.stages) an attempt gets when its
# output files are not aligned because the target reference run's workdir came back incomplete.
REFERENCE_WORKDIR_INCOMPLETE_MESSAGE = (
    "the target reference run's workdir came back incomplete, so its output files may have been cut; this run's "
    "output files were not aligned against them and its alignment stays unset"
)


class ItemOracle(Protocol):
    """An Oracle bound to one item by `for_item`: what the stage reads from it."""

    name: str

    @property
    def reads_passfail(self) -> bool:
        """Return True when PASS/FAIL is read for the item's target language."""
        ...

    def align(self, reference: RunResult, candidate: RunResult) -> float:
        """Return the candidate's value in [0, 1] against the reference."""
        ...


@register("Stage", "oracle")
class OracleStage:
    """Fills Attempt.alignment for every attempt that ran: by output files or by stdout, as its Oracle declares.

    `oracle_capabilities` names the capabilities of which the recipe's
    Oracle must declare one; the runner reads it before any directory
    exists.
    """

    name = "oracle"
    capabilities = frozenset({"aligns"})
    requires: dict[str, frozenset[str]] = {"Oracle": frozenset()}
    oracle_capabilities = ORACLE_CAPABILITIES

    def __init__(self, *, context: RunContext) -> None:
        """Keep the context and build the recipe's Oracle: bound to the item for stdout, as is for output files."""
        self.context = context
        binding = _oracle_binding(context)
        entry = DEFAULT_REGISTRY.get("Oracle", binding.name)
        self.files_oracle: OutputFileOracle | None = None
        self.oracle: ItemOracle | None = None
        if ALIGNS_OUTPUT_FILES in entry.capabilities:
            self.files_oracle = cast(OutputFileOracle, entry.factory(**binding.config))
        elif MASKS_STDOUT in entry.capabilities:
            self.oracle = _item_oracle(context, binding)
        else:
            wanted = " or ".join(repr(name) for name in ORACLE_CAPABILITIES)
            raise ValueError(f"{binding.where}: Oracle {binding.name!r} declares neither {wanted}")

    def __call__(self, trial: Trial) -> Trial:
        """Return `trial` with its runs aligned; unchanged when none ran.

        With an output-file Oracle, see align_output_files. Otherwise runs
        are aligned against Trial.reference_run's stdout; a reference stdout
        marked truncated is not aligned against (unaligned_runs). Raises
        ValueError when an attempt holds run stdout but the trial records no
        reference stdout (list the baseline stage, with an executor that
        runs programs, before this one).
        """
        if self.files_oracle is not None:
            return self.align_output_files(trial, self.files_oracle)
        ran = [attempt.index for attempt in trial.attempts if attempt.run.stdout_ref is not None]
        if not ran:
            return trial
        reference = trial.reference_run.stdout_ref
        if reference is None:
            raise ValueError(
                f"{trial.trial_id}: attempts {ran} hold run stdout, but the trial records no reference stdout to "
                "align them with; list the baseline stage, which runs the target reference, before the oracle stage"
            )
        if trial.reference_run.stdout_truncated is True:
            return self.unaligned_runs(trial)
        return self.align_runs(trial, self.context.store.get(reference))

    def unaligned_runs(self, trial: Trial) -> Trial:
        """Return `trial` with one REFERENCE_TRUNCATED warning on each attempt whose run holds stdout.

        Every alignment stays unset, the attempt's other diagnostics stay in
        order, and nothing else in the trial changes.
        """
        return dataclasses.replace(trial, attempts=[_warned_unaligned(attempt) for attempt in trial.attempts])

    def align_output_files(self, trial: Trial, oracle: OutputFileOracle) -> Trial:
        """Return `trial` with every attempt whose run recorded output files aligned by `oracle` (module docstring).

        A target reference run whose workdir came back incomplete is never
        aligned against (unaligned_output_files), and then nothing raises.
        Raises ValueError when attempts recorded output files but the
        reference recorded none, or when the target reference is complete,
        the oracle needs the references' agreement, and the trial recorded
        none; the message then adds any reference-workdir-incomplete note the
        baseline made.
        """
        ran = [attempt.index for attempt in trial.attempts if attempt.run.outputs is not None]
        if not ran:
            return trial
        reference = trial.reference_run.outputs
        if reference is None:
            raise ValueError(
                f"{trial.trial_id}: attempts {ran} recorded output files, but the target reference recorded none to "
                "align them with; list the baseline stage, with an executor that runs programs, before this one"
            )
        if trial.reference_run.workdir_incomplete is True:
            return self.unaligned_output_files(trial)
        if oracle.agreement_setting is not None:
            try:
                oracle = oracle.with_baseline(trial.reference_agreement)
            except ValueError as error:
                cut = REFERENCE_WORKDIR_INCOMPLETE
                notes = [note.message for note in trial.baseline_diagnostics if note.code == cut]
                why = f" (the baseline noted: {'; '.join(notes)})" if notes else ""
                raise ValueError(f"{trial.trial_id}: {error}{why}") from error
        blobs = BlobStore(self.context.store.root)
        reference_files = _stored_files(blobs, reference)
        attempts = []
        for attempt in trial.attempts:
            outputs = attempt.run.outputs
            if outputs is not None:
                aligned = oracle.alignment(reference_files, _stored_files(blobs, outputs))
                attempt = dataclasses.replace(attempt, alignment=aligned)
            attempts.append(attempt)
        return dataclasses.replace(trial, attempts=attempts)

    def unaligned_output_files(self, trial: Trial) -> Trial:
        """Return `trial` with one REFERENCE_WORKDIR_INCOMPLETE warning on each attempt whose run recorded output files.

        Every alignment stays unset, the attempt's other diagnostics stay in
        order, and nothing else in the trial changes.
        """
        return dataclasses.replace(trial, attempts=[_warned_cut_reference(attempt) for attempt in trial.attempts])

    def align_runs(self, trial: Trial, reference_stdout: str) -> Trial:
        """Return `trial` with every attempt that ran aligned against `reference_stdout`; nothing else changes."""
        reference = RunResult(exit_code=None, hang=False, stdout=reference_stdout, stderr="")
        return dataclasses.replace(trial, attempts=[self._aligned(attempt, reference) for attempt in trial.attempts])

    def describe(self) -> str:
        """Return a one-line description of the stage."""
        context = self.context
        if self.files_oracle is not None:
            return f"oracle: align each run's output files with the target reference's ({self.files_oracle.describe()})"
        if self.oracle is None:
            return "oracle: no Oracle is bound"
        reading = "reads PASS/FAIL" if self.oracle.reads_passfail else "does not read PASS/FAIL"
        return (
            f"oracle: align each run's stdout with the reference under the {context.suite.name}/{context.item} "
            f"timing masks with {self.oracle.name}, which {reading} for {context.direction.target}"
        )

    def _aligned(self, attempt: Attempt, reference: RunResult) -> Attempt:
        """Return the attempt with its alignment set when its run holds stdout, else the attempt itself."""
        run = attempt.run
        if run.stdout_ref is None or self.oracle is None:
            return attempt
        value = self.oracle.align(reference, _candidate(run, self.context.store.get(run.stdout_ref)))
        return dataclasses.replace(attempt, alignment=alignment([value]))


def _warned_unaligned(attempt: Attempt) -> Attempt:
    """Return the attempt with a REFERENCE_TRUNCATED run-stage warning appended when its run holds stdout."""
    if attempt.run.stdout_ref is None:
        return attempt
    warning = Diagnostic(stage="run", severity="warning", code=REFERENCE_TRUNCATED, message=REFERENCE_TRUNCATED_MESSAGE)
    return dataclasses.replace(attempt, diagnostics=[*attempt.diagnostics, warning])


def _candidate(run: RunInfo, stdout: str) -> RunResult:
    """Return the RunResult an oracle reads for a recorded run: its exit code, hang flag, and stdout.

    The record keeps no stderr text, so stderr is empty; stdout oracles never read it.
    """
    return RunResult(exit_code=run.exit_code, hang=bool(run.hang), stdout=stdout, stderr="")


def _warned_cut_reference(attempt: Attempt) -> Attempt:
    """Return the attempt with a REFERENCE_WORKDIR_INCOMPLETE run-stage warning appended when it recorded outputs."""
    if attempt.run.outputs is None:
        return attempt
    warning = Diagnostic(
        stage="run", severity="warning", code=REFERENCE_WORKDIR_INCOMPLETE, message=REFERENCE_WORKDIR_INCOMPLETE_MESSAGE
    )
    return dataclasses.replace(attempt, diagnostics=[*attempt.diagnostics, warning])


def _stored_files(blobs: BlobStore, outputs: dict[str, str]) -> dict[str, Path]:
    """Return a run's recorded output files as file -> its file in the binary store, each checked against its hash."""
    return {file: blobs.path(digest) for file, digest in outputs.items()}


def _oracle_binding(context: RunContext) -> Binding:
    """Return the recipe's one Oracle binding; ValueError when the recipe binds none."""
    bindings = [binding for binding in context.recipe.bindings if binding.interface == "Oracle"]
    if not bindings:
        raise ValueError(
            f"{context.recipe.path}: the oracle stage needs an Oracle, but the recipe binds none; "
            "set oracle: {kind: <name>, ...}"
        )
    (binding,) = bindings
    return binding


def _item_oracle(context: RunContext, binding: Binding) -> ItemOracle:
    """Return the recipe's stdout Oracle bound to the context's item; ValueError when its suite has no masks."""
    entry = DEFAULT_REGISTRY.get("Oracle", binding.name)
    masks = load_masks(context.suite.name)
    if context.item not in masks:
        raise ValueError(f"the {context.suite.name} mask file lists no masks for item {context.item!r}")
    item = context.suite.item(context.item, purpose=PURPOSE)
    oracle = entry.factory(**binding.config)
    return oracle.for_item(masks[context.item], prints_passfail=context.direction.target in item.passfail)
