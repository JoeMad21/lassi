"""Print a short summary of each trial of the given LASSI run directories (plans/spikes/ttmlir-runtime.md, 2026-10-30).

    uv run python plans/spikes/ttmlir-runtime/summarize.py <run dir>...

It reads each trial.json under a run directory and prints the trial's pins,
its reference run, the references' agreement, each attempt's run,
diagnostics, and alignment, and the final record, with the name, dtype, and
shape of every output file kept in the run's blob store (the reference run's
and each attempt's). It reads files only; nothing runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from lassi.harness.lassi_io import read_array


def run_fields(run: dict[str, Any] | None) -> dict[str, Any]:
    """Return the run fields a summary shows."""
    run = run or {}
    return {key: run.get(key) for key in ("exit_code", "hang", "sim_ub", "sim_gap", "wall_s")}


def outputs(root: Path, run: dict[str, Any] | None) -> list[str]:
    """Return one line per kept output file of a run: file, array name, dtype, and shape."""
    lines = []
    for name, sha in sorted(((run or {}).get("outputs") or {}).items()):
        try:
            array = read_array(root / "blobs" / sha[:2] / sha)
            lines.append(f"{name}: {array.name} {array.dtype} {array.shape}")
        except (OSError, ValueError) as error:
            lines.append(f"{name}: unreadable ({error})")
    return lines


def summarize(root: Path) -> None:
    """Print each trial of the run directory `root`."""
    for trial_path in sorted(root.rglob("trial.json")):
        trial = json.loads(trial_path.read_text(encoding="utf-8"))
        print("trial", trial["trial_id"])
        print("  pins", {key: value for key, value in trial["toolchain_pins"].items() if value})
        print("  reference_run", run_fields(trial["reference_run"]), outputs(root, trial["reference_run"]))
        print("  agreement", trial["reference_agreement"])
        for attempt in trial["attempts"]:
            run = attempt.get("run")
            print("  attempt", attempt["index"], attempt["stage_reached"], run_fields(run), outputs(root, run))
            for item in attempt["diagnostics"]:
                print("    diag", item["stage"], item["severity"], item.get("code"), item["message"][:400])
            print("    alignment", attempt.get("alignment"))
        print("  final", trial["final"])


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        summarize(Path(argument))
