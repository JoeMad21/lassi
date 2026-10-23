"""Tabulate every trial of the RNGD sweep's passes of 2026-10-23 into one TSV (standard library only).

Usage, on the build host from a clean commit (the run trees are read, never changed):

    python3 plans/runs/rngd-sweep/tabulate.py <runs root> <output dir>

It reads <runs root>/runs/sweep-<stamp>-<slug>-<half>/**/trial.json, where stamp is the pass's start (run.sh) and half
is maxabs or pcc, and writes <output dir>/trials.tsv with one row per trial: the pass stamp, model slug, half, the
trial's direction, item, and run, the stage reached, alignment, corrections, end reason, the number of attempts, the
number of unfenced-file warnings over its attempts, and, when the trial has an adversary record, its proposals,
refusals, runs, and divergences. It also writes <output dir>/passes.tsv with each pass's counts per model: trials,
trials at S5, trials whose alignment is 1.0, and trials with a divergence. Every value is copied from a trial record;
none is a measurement of speed.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HEADER = ("pass", "slug", "half", "direction", "item", "run", "stage", "alignment", "corrections", "end", "attempts",
          "unfenced_warnings", "adv_proposals", "adv_refused", "adv_runs", "adv_divergences")


def trial_row(stamp: str, slug: str, half: str, path: Path) -> list[str]:
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
            str(final.get("corrections")), end, str(len(attempts)), str(unfenced), *adv]


def main(argv: list[str]) -> int:
    """Write trials.tsv and passes.tsv; return 0, or 2 for a usage error."""
    if len(argv) != 2:
        print(__doc__.split("\n")[2].strip(), file=sys.stderr)
        return 2
    runs, out = Path(argv[0]) / "runs", Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tree in sorted(runs.glob("sweep-*")):
        _, date, clock, *rest = tree.name.split("-")
        slug, half = "-".join(rest[:-1]), rest[-1]
        for path in sorted(tree.rglob("trial.json")):
            rows.append(trial_row(f"{date}-{clock}", slug, half, path))
    with (out / "trials.tsv").open("w", encoding="ascii", newline="\n") as handle:
        handle.write("\t".join(HEADER) + "\n")
        handle.writelines("\t".join(row) + "\n" for row in rows)
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for row in rows:
        entry = counts[(row[0], row[1])]
        entry[0] += 1
        entry[1] += row[6] == "S5"
        entry[2] += row[7] == "1.0"
        entry[3] += row[15] not in ("-", "0")
    with (out / "passes.tsv").open("w", encoding="ascii", newline="\n") as handle:
        handle.write("pass\tslug\ttrials\ts5\talignment_1\twith_divergence\n")
        for (stamp, slug), (n, s5, exact, diverged) in sorted(counts.items()):
            handle.write(f"{stamp}\t{slug}\t{n}\t{s5}\t{exact}\t{diverged}\n")
    print(f"{len(rows)} trials from {len({(r[0], r[1]) for r in rows})} pass-model pairs into {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
