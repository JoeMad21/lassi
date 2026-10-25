"""Count each trial's restarts, looked-up notes, and retrieved notes in a sweep's records (standard library only).

Usage, on the build host from a clean commit (the run trees are read, never changed):

    python3 plans/runs/window/count_features.py <runs root> <stamp> <output tsv>

It reads <runs root>/runs/sweep-<stamp>-*/**/trial.json and writes one TSV row per trial: the run tree's name, the
trial's direction, item, and run, its stage reached, alignment, corrections, and end reason, then the number of its
requests per stage name (generate, compile_loop, run_loop, restart), the number that carry a looked-up note, and
the number that carry a retrieved note (the codes lassi.core.stages writes, P17.26 and P17.22). Every value is
copied or counted from a trial record or its run tree's name; none is a measurement of speed, and nothing here
judges a request.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

STAGES = ("generate", "compile_loop", "run_loop", "restart")
NOTES = ("looked-up", "retrieved")


def row(run_id: str, path: Path) -> list[str]:
    """Return one TSV row for the trial record at `path`."""
    trial = json.loads(path.read_text(encoding="utf-8"))
    final = trial.get("final") or {}
    parts = trial["trial_id"].split("/")
    requests = trial.get("requests") or []
    by_stage = [sum(1 for r in requests if r.get("stage") == s) for s in STAGES]
    by_note = [sum(1 for r in requests if any(d.get("code") == c for d in r.get("diagnostics") or [])) for c in NOTES]
    end = (final.get("end_reason") or {}).get("code") or "-"
    head = [run_id, parts[3], parts[4], parts[5], str(final.get("stage_reached")), str(final.get("alignment")),
            str(final.get("corrections")), end]
    return head + [str(n) for n in by_stage + by_note]


def main(argv: list[str]) -> int:
    """Write the TSV and print how many trials it holds."""
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    root, stamp, out = Path(argv[0]), argv[1], Path(argv[2])
    rows = [row(run.name, path) for run in sorted((root / "runs").glob(f"sweep-{stamp}-*"))
            for path in sorted(run.rglob("trial.json"))]
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ["run_tree", "direction", "item", "run", "stage", "alignment", "corrections", "end", *STAGES,
              *(f"{n}_notes" for n in NOTES)]
    with out.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("\t".join(header) + "\n")
        handle.writelines("\t".join(r) + "\n" for r in rows)
    print(f"{len(rows)} trials into {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
