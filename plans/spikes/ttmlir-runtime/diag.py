"""Exploratory diagnosis of a driver crash, in the sandbox (plans/spikes/ttmlir-runtime.md, 2026-10-30).

    uv run python plans/spikes/ttmlir-runtime/diag.py <step>...

Each step runs one argv in the sandbox with the ttmlir-ttsim executor's
install and the ttsim row's environment (TT_METAL_SIMULATOR on the pinned
library; no device node is visible), in a fresh directory under
$LASSI_RUNS_ROOT/ttmlir-runtime-spike/, and prints its exit status and the
tails of its streams. Steps: usage (the driver with no argument), refuse (run
mode on a missing flatbuffer), describe (describe mode with stdout and
stderr line-buffered by stdbuf), gdb (the driver with no argument under gdb
-batch, which prints a backtrace), ldd (the loader's trace of the driver's
libraries inside the sandbox), lddebug (the driver with no argument and the
loader's file and initializer trace, LD_DEBUG=files, on stderr), withhome
(the driver with no argument and HOME=/tmp, the sandbox's private tmpfs, to
test whether HOME is what the libraries need; a diagnosis, not a fix). `--tasks N` sets the sandbox's task cap
(SandboxSpec.tasks_max; 256 by default, as the executors leave it). `--home`
(after --tasks, if both) uses the ttmlir-ttsim executor's own environment,
the ttsim row's plus HOME=<workdir>/@ttsim/home (the owner's answer to
OQ-115), in place of the ttsim row's alone. Only the driver runs; it opens
at most one device, on ttsim.
"""

from __future__ import annotations

import shutil
import sys
import time

from lassi.core.interfaces import Limits
from lassi.executors.sandbox import SandboxSpec
from lassi.executors.ttmlir_ttsim import TtmlirTtsimExecutor
from lassi.executors.ttsim import RESERVED_DIR, _Mounts, program_environment
from lassi.executors.workdir import runs_root

LIMITS = Limits(wall_s=240.0, memory_mb=16384, cpus=4)


def argv_for(step: str, driver: str) -> list[str]:
    """Return the argv of one step."""
    if step == "usage":
        return [driver]
    if step == "refuse":
        return [driver, "run", "missing.ttm", "missing.ttsys"]
    if step == "describe":
        return ["/usr/bin/stdbuf", "-oL", "-eL", driver, "describe", "ttsim.ttsys"]
    if step == "gdb":
        gdb = shutil.which("gdb") or "/usr/bin/gdb"
        return [gdb, "-batch", "-nx", "-ex", "run", "-ex", "bt 40", "--args", driver]
    if step == "ldd":
        return ["/usr/bin/ldd", driver]
    if step == "withhome":
        return ["/usr/bin/env", "HOME=/tmp", driver]
    if step == "lddebug":
        return ["/usr/bin/env", "LD_DEBUG=files", driver]
    raise SystemExit(f"unknown step {step}")


def main() -> int:
    """Run each named step and print what it gave; `--tasks N` first sets the sandbox's task cap (default 256)."""
    args = sys.argv[1:]
    tasks = 256
    if args[:1] == ["--tasks"]:
        tasks, args = int(args[1]), args[2:]
    home = args[:1] == ["--home"]
    args = args[1:] if home else args
    executor = TtmlirTtsimExecutor()
    install = executor._install(need_desc=False)
    mounts = _Mounts(executor._hidden_roots(), None, install)
    for step in args:
        workdir = runs_root() / "ttmlir-runtime-spike" / f"{time.strftime('%Y%m%d-%H%M%S')}-diag-{step}-{tasks}"
        for name in executor.reserved_dirs:
            (workdir / RESERVED_DIR / name).mkdir(parents=True)
        environment = (executor._environment if home else program_environment)(install, workdir, LIMITS)
        spec = SandboxSpec(
            workdir=workdir,
            hidden_roots=mounts.hidden_roots,
            toolchains=install.root,
            tasks_max=tasks,
            environment=environment,
        )
        result = executor.sandbox.run(spec, argv_for(step, str(install.driver)), LIMITS)
        shown = f"HOME={environment['HOME']}" if "HOME" in environment else "no HOME"
        print(f"=== {step} ({shown}): rc {result.returncode} hang {result.hang} wall {result.wall_s:.1f} s; {workdir}")
        print(f"--- stdout tail ---\n{result.stdout[-6000:]}\n--- stderr tail ---\n{result.stderr[-4000:]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
