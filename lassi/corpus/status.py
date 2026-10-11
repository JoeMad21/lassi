"""The gate status of a corpus entry: the furthest gate step it passed and why it stopped (task P5.11).

This is v0's exit-code taxonomy moved into lassi/corpus/ (bible Review Of v0,
the Keep row; plans/spikes/p5-v0-corpus.md, Results 5, the migration map's
exit-code row). v0 ended each run of its driver with one exit code per
outcome (its scripts/lower_one.sh at 3ccd280, codes 0 to 5), and its batch
driver added timeout and killed (scripts/lower_all.sh:122-133, exit 124 and
137). Here the steps are those of the gate reading (plans/p5-ir.md, the
planning decision "Gate reading (P5.12)"): GATE_STEPS are raise, verify,
normalize, lower, build, run, and compare, in that order. Task P5.12 may add
a step for a C++ program's "reached structured cpu-mlir" and update these
names and their tests then.

Each step that can fail has one code, failure_code(step), spelled
`<step>-failed`; FAILURE_CODES holds those seven codes in step order, then
timeout and killed, which stop a step without its own failure. A GateStatus
records the furthest step an entry passed (None when it passed none) and the
code that stopped it (None exactly when it passed every step, a gated
entry). Task P5.12's per-kernel gate rate reads the same field.
"""

from __future__ import annotations

from dataclasses import dataclass

GATE_STEPS = ("raise", "verify", "normalize", "lower", "build", "run", "compare")
# Codes that stop a step from outside it: a wall-time limit, or a signal that killed the step's process.
STOP_CODES = ("timeout", "killed")
FAILURE_CODES = (*(f"{step}-failed" for step in GATE_STEPS), *STOP_CODES)


def failure_code(step: str) -> str:
    """Return the one code of a failure at gate step `step`; a name outside GATE_STEPS raises ValueError naming it."""
    if step not in GATE_STEPS:
        raise ValueError(f"unknown gate step {step!r}; the gate steps are {', '.join(GATE_STEPS)}")
    return f"{step}-failed"


@dataclass(frozen=True, kw_only=True)
class GateStatus:
    """The furthest gate step an entry passed, and the code that stopped it.

    `passed` is a GATE_STEPS name, or None when the entry passed no step.
    `code` is None exactly when `passed` is the last step (a gated entry);
    otherwise it is the failure code of the step right after `passed` (the
    first step when `passed` is None), or timeout or killed, which stop that
    step. Any other pair raises ValueError naming the value.
    """

    passed: str | None
    code: str | None

    def __post_init__(self) -> None:
        """Check the step, the code, and that the code fits the step after the one passed."""
        if self.passed is not None and self.passed not in GATE_STEPS:
            steps = ", ".join(GATE_STEPS)
            raise ValueError(f"GateStatus.passed must be a gate step ({steps}) or None, got {self.passed!r}")
        if self.code is not None and self.code not in FAILURE_CODES:
            raise ValueError(f"GateStatus.code must be one of {', '.join(FAILURE_CODES)} or None, got {self.code!r}")
        if self.passed == GATE_STEPS[-1]:
            if self.code is not None:
                raise ValueError(f"GateStatus.code must be None for an entry that passed every step, got {self.code!r}")
            return
        failing = GATE_STEPS[0] if self.passed is None else GATE_STEPS[GATE_STEPS.index(self.passed) + 1]
        if self.code not in (failure_code(failing), *STOP_CODES):
            raise ValueError(
                f"GateStatus.code must be {failure_code(failing)}, timeout, or killed after passed={self.passed!r}, "
                f"got {self.code!r}"
            )

    @property
    def gated(self) -> bool:
        """Return True when the entry passed every gate step."""
        return self.passed == GATE_STEPS[-1]
