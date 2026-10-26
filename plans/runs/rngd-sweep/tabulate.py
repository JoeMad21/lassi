"""Tabulate every trial of the RNGD sweep's passes of 2026-10-23 into one TSV (standard library only).

Usage, on the build host from a clean commit (the run trees are read, never changed):

    python3 plans/runs/rngd-sweep/tabulate.py <runs root> <output dir>

It reads <runs root>/runs/sweep-<stamp>-<slug>-<half>/**/trial.json, where stamp is the pass's start (run.sh) and half
is maxabs or pcc, and the fifth pass's sweep-<stamp>-<slug>--<combo>-<half> (one recipe per feature combination), and
writes <output dir>/trials.tsv with one row per trial: the pass stamp, model slug, half, the
trial's direction, item, and run, the stage reached, alignment, corrections, end reason, the number of attempts, the
number of unfenced-file warnings over its attempts, and, when the trial has an adversary record, its proposals,
refusals, runs, and divergences, then the combination (base for a run id without one), and last the number of its
attempts whose run stopped at a simulator gap. It also writes
<output dir>/passes.tsv with each pass's counts per model and combination: trials, trials at S5, trials whose
alignment is 1.0, and trials with a divergence, and last the attempts whose run stopped at a simulator gap. Every
value is copied from a trial record; none is a measurement of speed.

Simulator gaps (OQ-098, the owner's answer of 2026-10-25). An attempt whose run stopped at a simulator gap built, but
the simulator could not run it, so it is not verifiable on ttsim: it is never counted as a failure, and the
sim_gap_attempts columns count it apart. The label "not verifiable on ttsim" assumes the simulator is ttsim, as every
RNGD sweep run is; the records name no simulator, and the script does not check one. An attempt hit a gap when it
carries a run-stage note coded sim-gap (run_loop
adds one under loop.correct_on_sim_gap), or when it is the last attempt of a trial that ended at sim-gap; the gap's
class is read from the words "stopped at a simulator gap, <class>" that both messages hold (gap_attempts). A trial
that ended at sim-gap with no attempt stopped at its reference run and lists none. The script writes
<output dir>/not_verifiable.tsv, one row per such attempt (pass, slug, half, combination, trial id, attempt index, gap
class, stage reached), and <output dir>/hardware_check.tsv, the same attempts with the path of each trial record under
<runs root>/runs: the list for a later hardware check, which a human starts. This script runs nothing, on silicon or
anywhere (Agent Rule 9), and a simulator result is never a silicon or performance result (Agent Rule 2).
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HEADER = ("pass", "slug", "half", "direction", "item", "run", "stage", "alignment", "corrections", "end", "attempts",
          "unfenced_warnings", "adv_proposals", "adv_refused", "adv_runs", "adv_divergences", "combo",
          "sim_gap_attempts")
PASS_HEADER = ("pass", "slug", "trials", "s5", "alignment_1", "with_divergence", "combo", "sim_gap_attempts")
GAP_HEADER = ("pass", "slug", "half", "combo", "trial_id", "attempt", "gap_class", "stage")
CHECK_HEADER = ("trial_id", "attempt", "gap_class", "record")
# The end code and the run-stage note code of a simulator gap (lassi.core.record END_REASONS and SIM_GAP_NOTE), and
# the words both messages name the gap's class after (lassi.core.stages _gap_message, gap_note).
SIM_GAP = "sim-gap"
GAP_CLASS = re.compile(r"stopped at a simulator gap, ([^\s,:]+)")


def gap_class(message: str | None) -> str:
    """Return the gap's class a sim-gap message names, or "-" when it names none."""
    found = GAP_CLASS.search(message or "")
    return found.group(1) if found else "-"


def gap_attempts(trial: dict) -> list[tuple[int, str, str]]:
    """Return (attempt index, gap class, stage reached) for each attempt of `trial` whose run hit a simulator gap.

    An attempt hit one when it carries a run-stage diagnostic coded SIM_GAP
    (the note of loop.correct_on_sim_gap), or when it is the last attempt of
    a trial whose end reason is SIM_GAP; each attempt is listed once, in
    attempt order, its class from the note when it has one.
    """
    attempts = trial.get("attempts") or []
    found: dict[int, str] = {}
    for attempt in attempts:
        for diag in attempt.get("diagnostics") or []:
            if diag.get("stage") == "run" and diag.get("code") == SIM_GAP:
                found.setdefault(attempt["index"], gap_class(diag.get("message")))
    end = (trial.get("final") or {}).get("end_reason") or {}
    if attempts and end.get("code") == SIM_GAP:
        found.setdefault(attempts[-1]["index"], gap_class(end.get("message")))
    stages = {attempt["index"]: str(attempt.get("stage_reached")) for attempt in attempts}
    return [(index, found[index], stages[index]) for index in sorted(found)]


def trial_row(stamp: str, slug: str, half: str, path: Path, combo: str = "base") -> list[str]:
    """Return one TSV row for the trial record at `path`."""
    trial = json.loads(path.read_text(encoding="utf-8"))
    final = trial.get("final") or {}
    parts = trial["trial_id"].split("/")
    attempts = trial.get("attempts") or []
    unfenced = sum(1 for attempt in attempts for diag in attempt.get("diagnostics") or []
                   if diag.get("code") == "unfenced-file")
    adversary = trial.get("adversary")
    adv = ["-", "-", "-", "-"]
    if adversary:
        adv = [str(adversary.get("proposals")), str(sum((adversary.get("refusals") or {}).values())),
               str(adversary.get("runs")), str(len(adversary.get("divergences") or []))]
    end = (final.get("end_reason") or {}).get("code") or "-"
    stage, alignment = str(final.get("stage_reached")), str(final.get("alignment"))
    return [stamp, slug, half, parts[3], parts[4], parts[5], stage, alignment,
            str(final.get("corrections")), end, str(len(attempts)), str(unfenced), *adv, combo,
            str(len(gap_attempts(trial)))]


def gap_rows(row: list[str], path: Path, runs: Path) -> tuple[list[list[str]], list[list[str]]]:
    """Return the not_verifiable.tsv and hardware_check.tsv rows of the trial record at `path` (trials row `row`)."""
    trial = json.loads(path.read_text(encoding="utf-8"))
    record = path.relative_to(runs).as_posix()
    listed, checks = [], []
    for index, cls, stage in gap_attempts(trial):
        listed.append([row[0], row[1], row[2], row[16], trial["trial_id"], str(index), cls, stage])
        checks.append([trial["trial_id"], str(index), cls, record])
    return listed, checks


def write_tsv(path: Path, header: tuple[str, ...], rows: list[list[str]]) -> None:
    """Write `header` and `rows` to `path` as tab-separated ASCII lines."""
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("\t".join(header) + "\n")
        handle.writelines("\t".join(row) + "\n" for row in rows)


def main(argv: list[str]) -> int:
    """Write trials.tsv, passes.tsv, not_verifiable.tsv, and hardware_check.tsv; return 0, or 2 for a usage error."""
    if len(argv) != 2:
        print(__doc__.split("\n")[2].strip(), file=sys.stderr)
        return 2
    runs, out = Path(argv[0]) / "runs", Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows, listed, checks = [], [], []
    for tree in sorted(runs.glob("sweep-*")):
        _, date, clock, *rest = tree.name.split("-")
        slug, _, combo = "-".join(rest[:-1]).partition("--")
        half = rest[-1]
        for path in sorted(tree.rglob("trial.json")):
            rows.append(trial_row(f"{date}-{clock}", slug, half, path, combo or "base"))
            gaps, hardware = gap_rows(rows[-1], path, runs)
            listed += gaps
            checks += hardware
    write_tsv(out / "trials.tsv", HEADER, rows)
    counts: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    for row in rows:
        entry = counts[(row[0], row[1], row[16])]
        entry[0] += 1
        entry[1] += row[6] == "S5"
        entry[2] += row[7] == "1.0"
        entry[3] += row[15] not in ("-", "0")
        entry[4] += int(row[17])
    passes = [[stamp, slug, str(n), str(s5), str(exact), str(diverged), combo, str(gaps)]
              for (stamp, slug, combo), (n, s5, exact, diverged, gaps) in sorted(counts.items())]
    write_tsv(out / "passes.tsv", PASS_HEADER, passes)
    write_tsv(out / "not_verifiable.tsv", GAP_HEADER, listed)
    write_tsv(out / "hardware_check.tsv", CHECK_HEADER, checks)
    print(f"{len(rows)} trials from {len({(r[0], r[1]) for r in rows})} pass-model pairs into {out}")
    print(f"{len(listed)} attempt(s) built but not verifiable on ttsim (simulator gaps; the label assumes the "
          "simulator is ttsim, as every RNGD sweep run is), listed for a human-started hardware check; nothing was run")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
