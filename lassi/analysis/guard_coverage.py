"""Guard coverage: how much of a run the host-compute guard checked (task P17.14).

Bible: Evaluation Protocol (Run Metrics, Guard coverage), Result Record
(Guards), Readability Standards (Run row), Agent Rule 1; OQ-028 (the
owner's condition of 2026-10-05: the reason is logged whenever the guard
does not run, and the not-checked rate is reported per run and per target).

The unit is the attempt, not the trial: a group's attempts are every Attempt
of its trials, whatever the attempt's stage or the trial's end reason, and a
trial with no attempt adds nothing. Checked counts the attempts whose
Attempt.guards.host_compute is true or false, and not checked those whose
host_compute is null. Each not-checked attempt counts under its
host_compute_not_checked code, or under NOT_RECORDED when the code is null
(an attempt recorded before task P17.14), never under a code inferred from
anything else. The rate is not checked over attempts, None for a group with
no attempt, and has no interval: it describes the record's coverage rather
than estimating a rate.

Rows: one for the run (every arm and direction), then one per target
language (the part of the direction's name after its only hyphen), sorted,
then one per direction whose name holds more than one hyphen, since the
record cannot say which part is its target. The Markdown holds names and
counts only (OQ-018) and is plain ASCII.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from lassi.core.record import HOST_COMPUTE_NOT_CHECKED, Attempt, Trial, parse_trial_id

# The count of not-checked attempts whose code is null: recorded before task P17.14. Never a stored code.
NOT_RECORDED = "not recorded"
REASON_KEYS = (*HOST_COMPUTE_NOT_CHECKED, NOT_RECORDED)
RUN_SCOPE = "run"
COLUMNS = ("Scope", "Attempts", "Checked", "Not checked", "Not-checked rate", *REASON_KEYS)
COVERAGE_LEGEND = (
    "How much of the run the host-compute guard checked. The unit is the attempt: every attempt of every trial, "
    "whatever its stage or the trial's end reason; a trial with no attempt adds nothing. Checked: "
    "Attempt.guards.host_compute is true or false. Not checked: it is null, counted under the attempt's "
    "host_compute_not_checked code, or under not recorded when the attempt records none (recorded before task "
    "P17.14). The not-checked rate is not checked over attempts, - with no attempt, and has no interval. One row "
    "for the run, every arm and direction together, then one per target language, the part of the direction's "
    "name after its only hyphen; a direction whose name holds more than one hyphen gets its own row."
)


@dataclass(frozen=True)
class GuardCoverage:
    """The guard coverage of a group of attempts.

    `reasons` maps every REASON_KEYS key, in order, to its count of
    not-checked attempts, 0 included.
    """

    attempts: int
    checked: int
    reasons: Mapping[str, int]

    @property
    def not_checked(self) -> int:
        """Return the count of attempts whose host_compute is null: the sum of the reason counts."""
        return sum(self.reasons.values())

    @property
    def rate(self) -> float | None:
        """Return not checked over attempts, or None when there is no attempt."""
        if self.attempts == 0:
            return None
        return self.not_checked / self.attempts


def _coverage(attempts: int, checked: int, reasons: Mapping[str, int]) -> GuardCoverage:
    """Return a GuardCoverage whose reasons hold every REASON_KEYS key in order."""
    ordered = {key: reasons.get(key, 0) for key in REASON_KEYS}
    return GuardCoverage(attempts=attempts, checked=checked, reasons=MappingProxyType(ordered))


def count_attempts(attempts: Iterable[Attempt]) -> GuardCoverage:
    """Return the coverage of `attempts`: checked, and each not-checked attempt under its code or NOT_RECORDED."""
    total, checked = 0, 0
    reasons: Counter[str] = Counter()
    for attempt in attempts:
        total += 1
        guards = attempt.guards
        if guards.host_compute is not None:
            checked += 1
        else:
            reasons[guards.host_compute_not_checked or NOT_RECORDED] += 1
    return _coverage(total, checked, reasons)


def combined(coverages: Iterable[GuardCoverage]) -> GuardCoverage:
    """Return the sum of `coverages`; no coverage gives all zeros."""
    attempts, checked = 0, 0
    reasons: Counter[str] = Counter()
    for coverage in coverages:
        attempts += coverage.attempts
        checked += coverage.checked
        reasons.update(coverage.reasons)
    return _coverage(attempts, checked, reasons)


def target_language(direction: str) -> str | None:
    """Return the part of a direction's name after its only hyphen, or None when it does not hold exactly one.

    A direction's name is `<source>-<target>` (lassi.bench.registry), and a
    language name may hold a hyphen, so with more than one the record cannot
    say which part is the target.
    """
    if direction.count("-") != 1:
        return None
    return direction.split("-", 1)[1]


def summary_rows(groups: Sequence[tuple[str, GuardCoverage]]) -> list[tuple[str, GuardCoverage]]:
    """Return ("run", every group summed), ("target <language>", ...) per target sorted, then unreadable directions.

    `groups` holds (direction, coverage) pairs; a direction may appear more
    than once (one per arm). A direction whose target cannot be read
    (target_language) gets a row "direction <name>" of its groups summed.
    """
    by_target: dict[str, list[GuardCoverage]] = {}
    by_direction: dict[str, list[GuardCoverage]] = {}
    for direction, coverage in groups:
        target = target_language(direction)
        if target is None:
            by_direction.setdefault(direction, []).append(coverage)
        else:
            by_target.setdefault(target, []).append(coverage)
    rows = [(RUN_SCOPE, combined(coverage for _, coverage in groups))]
    rows += [(f"target {target}", combined(found)) for target, found in sorted(by_target.items())]
    rows += [(f"direction {name}", combined(found)) for name, found in sorted(by_direction.items())]
    return rows


def guard_coverage(trials: Sequence[Trial]) -> list[tuple[str, GuardCoverage]]:
    """Return the summary rows of `trials`, grouped by the trial_id's direction; independent of the trials' order."""
    groups: dict[str, list[Attempt]] = {}
    for trial in trials:
        groups.setdefault(parse_trial_id(trial.trial_id).direction, []).extend(trial.attempts)
    return summary_rows([(direction, count_attempts(groups[direction])) for direction in sorted(groups)])


def coverage_cells(coverage: GuardCoverage) -> list[str]:
    """Return attempts, checked, not checked, the rate with three decimals or "-", and each REASON_KEYS count."""
    rate = "-" if coverage.rate is None else f"{coverage.rate:.3f}"
    counts = [str(coverage.reasons.get(key, 0)) for key in REASON_KEYS]
    return [str(coverage.attempts), str(coverage.checked), str(coverage.not_checked), rate, *counts]


def table_lines(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    """Return a Markdown table as lines: the header, the rule, and one line per row."""
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def coverage_markdown(rows: Sequence[tuple[str, GuardCoverage]]) -> str:
    """Return the "## Guard coverage" section: the heading, the legend, and one table with one line per row."""
    table = table_lines(COLUMNS, ([scope, *coverage_cells(coverage)] for scope, coverage in rows))
    return "\n".join(["## Guard coverage", "", COVERAGE_LEGEND, "", *table]) + "\n"
