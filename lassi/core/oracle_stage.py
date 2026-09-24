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
compile-only trial). The trial record does not carry the reference run's
stdout yet, so a trial holding run stdout raises rather than go unaligned.

The oracle class is looked up in the default registry, since a RunContext
carries no registry; the runner checks the binding against its own registry
when the recipe loads, and builds it once from its section before any
directory is created.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

from lassi.core.interfaces import RunResult
from lassi.core.record import Attempt, RunInfo, Trial
from lassi.core.registry import DEFAULT_REGISTRY, register
from lassi.core.stages import PURPOSE, RunContext
from lassi.oracles import alignment, load_masks

# The capability the stage needs from its Oracle: `for_item(masks, prints_passfail=...)` returns an ItemOracle
# (lassi.oracles.stdout_mask).
MASKS_STDOUT = "masks_stdout"


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
    """Fills Attempt.alignment for every attempt whose run holds stdout; leaves compile-only attempts unset."""

    name = "oracle"
    capabilities = frozenset({"aligns"})
    requires = {"Oracle": {MASKS_STDOUT}}

    def __init__(self, *, context: RunContext) -> None:
        """Keep the context and bind the recipe's Oracle to the trial's item; fail loudly when none is bound."""
        self.context = context
        self.oracle = _item_oracle(context)

    def __call__(self, trial: Trial) -> Trial:
        """Return `trial` unchanged when no attempt holds run stdout; raise when one does (no reference yet)."""
        ran = [attempt.index for attempt in trial.attempts if attempt.run.stdout_ref is not None]
        if ran:
            raise ValueError(
                f"{trial.trial_id}: attempts {ran} hold run stdout, but the trial records no reference stdout to "
                "align them with; no stage records the reference run yet"
            )
        return trial

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
