"""The oracle stage: align each run's stdout with the reference's (bible Oracles; Component Interfaces, Stage row).

The stage is built as `factory(context=<RunContext>)` once per trial, like
every stage (lassi.core.stages). It binds the recipe's Oracle, which must
declare the capability `masks_stdout` (lassi.oracles.stdout_mask), to the
trial's item: the item's masks from `assets/harness/masks/<suite>.yaml`, and
whether the manifest lists the direction's target language under the item's
`passfail`. The recipe's `oracle.passfail` stays the oracle's own choice.

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
from typing import Protocol

from lassi.core.interfaces import RunResult
from lassi.core.record import Attempt, Diagnostic, RunInfo, Trial
from lassi.core.registry import DEFAULT_REGISTRY, register
from lassi.core.stages import PURPOSE, RunContext
from lassi.oracles import alignment, load_masks

# The capability the stage needs from its Oracle: `for_item(masks, prints_passfail=...)` returns an ItemOracle
# (lassi.oracles.stdout_mask).
MASKS_STDOUT = "masks_stdout"

# The run-stage warning an attempt gets when its run is not aligned because the reference stdout was cut.
REFERENCE_TRUNCATED = "reference-stdout-truncated"
REFERENCE_TRUNCATED_MESSAGE = (
    "the reference run's stdout was truncated at the executor's output cap, so this run's stdout was not aligned "
    "against it and its alignment stays unset"
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
    """Fills Attempt.alignment for every attempt whose run holds stdout, unless the reference stdout was cut."""

    name = "oracle"
    capabilities = frozenset({"aligns"})
    requires = {"Oracle": {MASKS_STDOUT}}

    def __init__(self, *, context: RunContext) -> None:
        """Keep the context and bind the recipe's Oracle to the trial's item; fail loudly when none is bound."""
        self.context = context
        self.oracle = _item_oracle(context)

    def __call__(self, trial: Trial) -> Trial:
        """Return `trial` with its runs aligned against Trial.reference_run's stdout; unchanged when none ran.

        A reference stdout marked truncated is not aligned against
        (unaligned_runs). Raises ValueError when an attempt holds run stdout
        but the trial records no reference stdout (list the baseline stage,
        with an executor that runs programs, before this one).
        """
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

    def align_runs(self, trial: Trial, reference_stdout: str) -> Trial:
        """Return `trial` with every attempt that ran aligned against `reference_stdout`; nothing else changes."""
        reference = RunResult(exit_code=None, hang=False, stdout=reference_stdout, stderr="")
        return dataclasses.replace(trial, attempts=[self._aligned(attempt, reference) for attempt in trial.attempts])

    def describe(self) -> str:
        """Return a one-line description of the stage."""
        context = self.context
        reading = "reads PASS/FAIL" if self.oracle.reads_passfail else "does not read PASS/FAIL"
        return (
            f"oracle: align each run's stdout with the reference under the {context.suite.name}/{context.item} "
            f"timing masks with {self.oracle.name}, which {reading} for {context.direction.target}"
        )

    def _aligned(self, attempt: Attempt, reference: RunResult) -> Attempt:
        """Return the attempt with its alignment set when its run holds stdout, else the attempt itself."""
        run = attempt.run
        if run.stdout_ref is None:
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


def _item_oracle(context: RunContext) -> ItemOracle:
    """Return the recipe's Oracle bound to the context's item; ValueError when the recipe binds none or no masks."""
    bindings = [binding for binding in context.recipe.bindings if binding.interface == "Oracle"]
    if not bindings:
        raise ValueError(
            f"{context.recipe.path}: the oracle stage needs an Oracle, but the recipe binds none; "
            "set oracle: {kind: <name>, ...}"
        )
    (binding,) = bindings
    entry = DEFAULT_REGISTRY.get("Oracle", binding.name)
    if MASKS_STDOUT not in entry.capabilities:
        raise ValueError(f"{binding.where}: Oracle {binding.name!r} does not declare {MASKS_STDOUT!r}")
    masks = load_masks(context.suite.name)
    if context.item not in masks:
        raise ValueError(f"the {context.suite.name} mask file lists no masks for item {context.item!r}")
    item = context.suite.item(context.item, purpose=PURPOSE)
    oracle = entry.factory(**binding.config)
    return oracle.for_item(masks[context.item], prints_passfail=context.direction.target in item.passfail)
