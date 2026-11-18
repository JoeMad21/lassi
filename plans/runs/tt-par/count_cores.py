"""Count the cores each tt-par translation ran on under ttsim (tasks P17.74 and P17.70).

For each trial record under the run trees given, the last attempt that ran
cleanly (stage_reached S5: it built, ran with exit status 0, and its outputs
reached the oracle, whether they matched or not) is run once more through the
sandboxed ttsim executor with count_cores true (lassi.executors.ttsim;
plans/spikes/ttsim-cores.md): its build directory is copied, without the
earlier run's @ttsim directory and output files, into a fresh workdir under
$LASSI_RUNS_ROOT/count-cores/<stamp>/, the item's inputs are written there
(lassi.bench.stage_inputs), and the program runs with the run stage's CPU
count (lassi.core.stages RUN_CPUS) and a wall limit of WALL_S. The model's
program runs only inside the sandbox (Agent Rule 6). Only development-split
trials are counted unless --include-eval is given, so held-out counts never
mix into a development table (Agent Rule 5). Each trial gives one row of a
TSV, written as it goes: the run tree, the split, the item, the direction,
the model, the trial's final alignment, the attempt, the exit status, whether
every output equals the attempt's recorded output (sha256), the cores used
and the worker cores of the Watcher's dump, and the note; a trial whose
rerun raised gets its error as the note. The TSV's first line names the
device. A count is a count of cores; ttsim is a simulator and nothing here is
a time (Agent Rule 2).

On the build host, from the repository root, through tools/rx.py:
    uv run python plans/runs/tt-par/count_cores.py --out <tsv> [--include-eval] <run tree> [<run tree> ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from lassi.core.capabilities import CORES_CODE
from lassi.toolchains._base import OUTPUT

REPO = Path(__file__).resolve().parents[3]
CLEAN = "S5"
DEV = "dev"
# The rerun's wall limit in seconds [DESIGN]: the tt-par references ran in about 30 s of pipeline time on ttsim
# (results/tt-par-v0-ttsim), so a translation that needs ten minutes gets no count rather than holding the window.
WALL_S = 600.0
COLUMNS = ("run", "split", "item", "direction", "model", "alignment", "attempt", "exit", "outputs_equal", "cores",
           "workers", "note")
NOTE = re.compile(r"cores used: (?P<cores>[0-9]+) of (?P<workers>[0-9]+) worker cores")
RESERVED = "@ttsim"


@dataclass(frozen=True)
class Target:
    """One trial's last clean attempt: where its build is and what its run recorded."""

    run: str
    split: str
    item: str
    suite: str
    direction: str
    model: str
    alignment: float | None
    attempt: int
    build: Path
    outputs: Mapping[str, str]


def last_clean_attempt(trial: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the trial's last attempt that reached S5 with a run record, or None."""
    clean = [a for a in trial.get("attempts") or [] if a.get("stage_reached") == CLEAN and a.get("run")]
    return clean[-1] if clean else None


def targets(roots: Iterable[Path], *, include_eval: bool = False) -> list[Target]:
    """Return a Target for each trial.json under `roots` with a clean attempt, in path order; dev only by default."""
    found: list[Target] = []
    for root in roots:
        for path in sorted(Path(root).rglob("trial.json")):
            trial = json.loads(path.read_text(encoding="utf-8"))
            item = trial["bench_item"]
            attempt = last_clean_attempt(trial)
            if attempt is None or (item.get("split") != DEV and not include_eval):
                continue
            found.append(Target(
                run=Path(root).name, split=str(item.get("split")), item=item["item"], suite=item["suite"],
                direction=item["direction"], model=trial["model"]["id"],
                alignment=(trial.get("final") or {}).get("alignment"), attempt=int(attempt["index"]),
                build=path.parent / f"attempt{int(attempt['index']):02d}" / "build",
                outputs=dict(attempt["run"].get("outputs") or {}),
            ))
    return found


def fresh_copy(build: Path, outputs: Iterable[str], workdir: Path) -> None:
    """Copy `build` into the new `workdir`, leaving out its @ttsim directory and its recorded output files."""
    left_out = {RESERVED, *outputs}
    shutil.copytree(build, workdir, symlinks=True, ignore=lambda folder, names: [
        name for name in names if Path(folder) == build and name in left_out
    ])


def sha256(path: Path) -> str:
    """Return the hex sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(target: Target, **values: str) -> dict[str, str]:
    """Return a TSV row for `target` with the run's values filled in."""
    row = {
        "run": target.run, "split": target.split, "item": target.item, "direction": target.direction,
        "model": target.model, "alignment": "-" if target.alignment is None else repr(target.alignment),
        "attempt": str(target.attempt), "exit": "-", "outputs_equal": "-", "cores": "-", "workers": "-", "note": "-",
    }
    row.update(values)
    return row


def count_one(target: Target, workdir: Path, run: Callable[[Path, list[str]], Any],
              inputs: Callable[[Target, Path], list[str]]) -> dict[str, str]:
    """Copy the target's build to `workdir`, run it with `run`, and return its TSV row; an error becomes the note."""
    try:
        fresh_copy(target.build, target.outputs, workdir)
        result = run(workdir / OUTPUT, inputs(target, workdir))
    except Exception as error:  # noqa: BLE001  (one trial's failure is that trial's row, never the whole table's)
        return _row(target, note=" ".join(f"error: {type(error).__name__}: {error}".split()))
    notes = [d.message for d in result.diagnostics if d.code == CORES_CODE]
    match = NOTE.search(notes[0]) if notes else None
    equal = bool(target.outputs) and all(
        (workdir / name).is_file() and sha256(workdir / name) == digest for name, digest in target.outputs.items()
    )
    return _row(
        target, exit=str(result.exit_code), outputs_equal="yes" if equal else "no",
        cores=match["cores"] if match else "-", workers=match["workers"] if match else "-",
        note=notes[0] if notes else "no cores note",
    )


def write_header(out: TextIO, device: str) -> None:
    """Write the device line and the column names."""
    out.write(f"# device: {device}\n" + "\t".join(COLUMNS) + "\n")


def write_row(out: TextIO, row: Mapping[str, str]) -> None:
    """Write one row under COLUMNS, any other character than ASCII escaped, and flush it, so a stopped run keeps it."""
    values = (row[name].encode("ascii", "backslashreplace").decode("ascii") for name in COLUMNS)
    out.write("\t".join(values) + "\n")
    out.flush()


def main(argv: Sequence[str] | None = None) -> int:
    """Count the cores of every clean last attempt under the run trees given; write the TSV as it goes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="the TSV to write")
    parser.add_argument("--include-eval", action="store_true", help="count held-out trials too (default: dev only)")
    parser.add_argument("roots", nargs="+", type=Path, help="run trees under $LASSI_RUNS_ROOT/runs")
    args = parser.parse_args(argv)

    from lassi.bench import registry
    from lassi.core.interfaces import Limits
    from lassi.core.stages import RUN_CPUS
    from lassi.executors.ttsim import TtsimExecutor

    executor = TtsimExecutor(count_cores=True)
    limits = Limits(wall_s=WALL_S, memory_mb=16384, cpus=RUN_CPUS)  # memory: the recipes' sandbox.mem_gb, 16
    suites: dict[str, Any] = {}

    def inputs(target: Target, workdir: Path) -> list[str]:
        """Write the item's inputs into the workdir as the run stage does, and return its arguments."""
        if target.suite not in suites:
            suites[target.suite] = registry.load_suite(REPO / "assets" / "bench" / f"{target.suite}.yaml")
        return registry.stage_inputs(suites[target.suite].item(target.item, purpose="eval"), workdir)

    base = Path(os.environ["LASSI_RUNS_ROOT"]) / "count-cores" / time.strftime("%Y%m%d-%H%M%S")
    found = targets(args.roots, include_eval=args.include_eval)
    print(f"device: {executor.device()}; {len(found)} trial(s); workdirs under {base}", flush=True)
    with args.out.open("w", encoding="ascii") as out:
        write_header(out, executor.device())
        for number, target in enumerate(found):
            row = count_one(target, base / f"{number:03d}", lambda a, i: executor.run(a, i, limits), inputs)
            write_row(out, row)
            print("\t".join(row[name] for name in COLUMNS), flush=True)
    print(f"{len(found)} trial(s) counted; table {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
