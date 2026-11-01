"""Write the ttsim system descriptor with LASSI's tt-mlir driver, in the sandbox, and install it (task P17.48).

The flatbuffers the ttmlir-ttsim executor runs must be compiled for the
system descriptor the runtime reads from ttsim (plans/spikes/ttir-d2m.md,
section 1), so the descriptor is written once, on the build host, by the
driver's describe mode inside the sandbox with the ttsim row's environment
(lassi.executors.ttmlir_ttsim TtmlirTtsimExecutor.write_system_desc): one
device open and one close on ttsim, never on a card (Agent Rule 9).

    uv run tools/ttmlir_system_desc.py [--install]

The run's directory is $LASSI_RUNS_ROOT/ttmlir-system-desc/<stamp>. The
script prints the run's exit status, its diagnostics, the tail of each
stream, and the descriptor's size and sha256. With --install it copies the
descriptor to <driver prefix>/<SYSTEM_DESC> (toolchains/tt-mlir-driver.pin)
when no file is there; a file already there is kept, and the script fails
unless it has the same bytes. It never removes or replaces a file. The
sha256 goes into the pin's SYSTEM_DESC_SHA256 by hand; the value pinned now
comes from exploratory runs (dirty trees) and stands until a clean commit's
run confirms it (AGENTS.md, Results).

Exit status: 0 when the descriptor was written (and installed, with
--install), 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

from lassi.core.interfaces import Limits
from lassi.executors.ttmlir_ttsim import TtmlirTtsimExecutor
from lassi.executors.workdir import runs_root

# The describe run's limits: ttsim opens the device and reads it in seconds (PROJECTED); the wall limit leaves room
# for tt-metal's start-up on a loaded host, and the memory for its libraries.
LIMITS = Limits(wall_s=300.0, memory_mb=16384, cpus=4)
# How many characters of each stream are printed.
TAIL_CHARS = 3000


def _sha256(path: Path) -> str:
    """Return the hex sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install(executor: TtmlirTtsimExecutor, written: Path) -> bool:
    """Copy the written descriptor into the driver prefix unless a file is there; return whether it is installed."""
    target = executor.recorded_system_desc()
    if target.exists() or target.is_symlink():
        same = target.is_file() and _sha256(target) == _sha256(written)
        print(f"{target} is already there and {'has' if same else 'does NOT have'} the same bytes; it is kept")
        return same
    shutil.copyfile(written, target)
    print(f"installed {target} (sha256 {_sha256(target)})")
    return True


def main(argv: list[str] | None = None) -> int:
    """Write the descriptor, print the run, and install it when asked; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--install", action="store_true", help="copy the descriptor into the driver prefix")
    args = parser.parse_args(argv)
    executor = TtmlirTtsimExecutor()
    workdir = runs_root() / "ttmlir-system-desc" / time.strftime("%Y%m%d-%H%M%S")
    workdir.mkdir(parents=True)
    run, written = executor.write_system_desc(workdir, LIMITS)
    print(f"device: {executor.device()}")
    print(f"workdir: {workdir}")
    print(f"exit status {run.exit_code}, hang {run.hang}, wall {run.wall_s:.1f} s (simulator wall time)")
    for item in run.diagnostics:
        print(f"diagnostic: {item.code}: {item.message}")
    print(f"--- stdout tail ---\n{run.stdout[-TAIL_CHARS:]}\n--- stderr tail ---\n{run.stderr[-TAIL_CHARS:]}")
    if run.exit_code != 0 or not written.is_file():
        print("no descriptor was written", file=sys.stderr)
        return 1
    print(f"descriptor: {written} ({written.stat().st_size} bytes, sha256 {_sha256(written)})")
    if args.install and not _install(executor, written):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
