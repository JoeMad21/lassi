"""P4.9 spike: run one pinned tt-metal example on ttsim inside the P0.16 sandbox (lassi.executors.sandbox).

Run by plans/spikes/p4-ttsim-runtime/run.sh as `uv run --frozen python <this file> ...` from the repository root
on the build host (report: plans/spikes/p4-ttsim-runtime.md). The sandbox is the one generated code runs in:
$LASSI_SCRATCH, $HOME, and the runs root are hidden, the workdir is the one writable host directory, and
$LASSI_TOOLCHAINS, which holds the pinned tt-metal tree and ttsim, is mounted read-only. At this commit the
sandbox's environment allowlist (ENVIRONMENT_NAMES) holds no TT_METAL_* name, so the attempt keeps the P0.16
command (SandboxSpec.environment None) and gives the example its environment through its argv,
`env -i NAME=value... <example>`, with no HOME. It first records how SandboxSpec treats the same variables as
SandboxSpec.environment (P4.11 adds them to the allowlist). The JIT cache and the logs directory are inside the
workdir, as the P4 plan requires, and the locale is LANG=C with LC_ALL=C, as for compiles (bible, Sandbox), so
the kernel compiler's diagnostics stay ASCII. It writes <out>/result.json, stdout.txt, and stderr.txt and prints
one line; result.json names the device and says its times are simulator wall time (Agent Rule 2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from lassi.core.interfaces import Limits
from lassi.executors.sandbox import SANDBOX_PATH, Sandbox, SandboxSpec, SandboxUnavailableError

# The attempt's limits, with wide margin. The memory limit (--memory-mb; run.sh sizes it from the clean run's
# peak resident set) also bounds the workdir overlay, whose tmpfs pages count toward it, so the disk cap, which
# must hold the JIT cache, is at most half of it. CPUS is the reference run's count (REFERENCE_CPUS in
# lassi/core/stages.py); the CPU-time cap is wall_s x CPUS per process.
DISK_MB = 2048
CPUS = 16
DEVICE = ("ttsim v1.3.4 (libttsim_wh.so, a virtual Wormhole) on the host CPU, with tt-metal 5280a9cf; "
          "simulator, not silicon")
TIMING_NOTE = ("wall_s, program_s, and any sim_rate_line are simulator wall time and the simulator's own rate: "
               "exploratory, sizing only, never performance")


def program_environment(args: argparse.Namespace, workdir: Path) -> list[str]:
    """Return the example's NAME=value list: the bible's ttsim row, the cache and logs in the workdir.

    HOME is set to the workdir only when --home is given (the one-shot retry for a HOME dependency); otherwise no
    HOME, so the probe's answer about whether tt-metal needs it is not hidden.
    """
    env = [
        f"PATH={SANDBOX_PATH}",
        "LANG=C",
        "LC_ALL=C",
        "TMPDIR=/tmp",
        f"TT_METAL_RUNTIME_ROOT={args.tt_metal}",
        f"TT_METAL_SIMULATOR={args.ttsim}/libttsim_wh.so",
        "TT_METAL_SLOW_DISPATCH_MODE=1",
        "TT_METAL_DISABLE_SFPLOADMACRO=1",
        f"TT_METAL_CACHE={workdir}/cache",
        f"TT_METAL_LOGS_PATH={workdir}/logs",
        "TT_METAL_INSPECTOR_RPC=0",
    ]
    if args.home:
        env.append(f"HOME={workdir}")
    if args.threads:
        env.append(f"TT_METAL_THREADCOUNT={args.threads}")
    return env


def hidden_roots() -> tuple[Path, ...]:
    """Return $LASSI_SCRATCH, $HOME, and $LASSI_RUNS_ROOT, resolved, the ones that are set, without repeats."""
    roots: list[Path] = []
    for name in ("LASSI_SCRATCH", "HOME", "LASSI_RUNS_ROOT"):
        value = os.environ.get(name)
        if value and Path(os.path.realpath(value)) not in roots:
            roots.append(Path(os.path.realpath(value)))
    return tuple(roots)


def allowlist_check(workdir: Path, roots: tuple[Path, ...], toolchains: Path, env: list[str]) -> str:
    """Return how SandboxSpec treats the example's variables when given as SandboxSpec.environment."""
    try:
        SandboxSpec(workdir=workdir, hidden_roots=roots, toolchains=toolchains,
                    environment=dict(item.split("=", 1) for item in env))
    except ValueError as exc:
        return f"refused: {exc}"
    return "accepted"


def run(args: argparse.Namespace) -> dict:
    """Run the example in the sandbox and return the result record."""
    workdir = Path(os.path.realpath(args.workdir))
    toolchains = Path(os.path.realpath(os.environ["LASSI_TOOLCHAINS"]))
    roots = hidden_roots()
    env = program_environment(args, workdir)
    disk_mb = min(DISK_MB, args.memory_mb // 2)
    record: dict = {"name": args.name, "allowlist": allowlist_check(workdir, roots, toolchains, env),
                    "limits": {"wall_s": args.wall, "memory_mb": args.memory_mb, "cpus": CPUS, "disk_mb": disk_mb},
                    "threads": args.threads, "hidden_roots": [str(root) for root in roots],
                    "toolchains": str(toolchains), "workdir": str(workdir), "device": DEVICE,
                    "timing_note": TIMING_NOTE, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    spec = SandboxSpec(workdir=workdir, hidden_roots=roots, toolchains=toolchains, disk_mb=disk_mb)
    record["tasks_max"] = spec.tasks_max
    limits = Limits(wall_s=args.wall, memory_mb=args.memory_mb, cpus=CPUS)
    out = Path(args.out)
    try:
        result = Sandbox().run(spec, ["env", "-i", *env, args.example], limits)
    except SandboxUnavailableError as exc:
        # The message may carry the program's stderr tail (possibly upstream text). Keep the full text in the
        # workdir (raw/, copied to local only) and put a length and sha256, not the text, in the tracked record.
        text = str(exc)
        (out / "unavailable.txt").write_text(text)
        record.update(rc=None, timed_out=False, unavailable=None,
                      unavailable_len=len(text), unavailable_sha256=hashlib.sha256(text.encode()).hexdigest(),
                      ended_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        return record
    record["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out / "stdout.txt").write_text(result.stdout)
    (out / "stderr.txt").write_text(result.stderr)
    record.update(rc=result.returncode, timed_out=result.hang, killed=result.killed, wall_s=round(result.wall_s, 2),
                  program_s=result.program_s, stdout_truncated=result.stdout_truncated,
                  stderr_truncated=result.stderr_truncated, workdir_incomplete=result.workdir_incomplete,
                  unavailable=None)
    return record


def main(argv: list[str]) -> int:
    """Parse the arguments, run the attempt, write result.json, and print one line; 0 unless the record fails."""
    top = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    top.add_argument("--name", required=True)
    top.add_argument("--example", required=True, help="absolute path of the example binary")
    top.add_argument("--workdir", required=True, help="an existing directory under the runs root")
    top.add_argument("--out", required=True)
    top.add_argument("--wall", type=float, required=True, help="the wall limit in seconds")
    top.add_argument("--memory-mb", type=int, required=True, help="the memory limit in MiB (MemoryMax)")
    top.add_argument("--tt-metal", required=True, help="the pinned tree, TT_METAL_RUNTIME_ROOT")
    top.add_argument("--ttsim", required=True, help="the directory of libttsim_wh.so and soc_descriptor.yaml")
    top.add_argument("--threads", type=int, default=0, help="TT_METAL_THREADCOUNT; 0 leaves it unset")
    top.add_argument("--home", action="store_true", help="set HOME=<workdir> (the one-shot HOME retry)")
    args = top.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    record = run(args)
    Path(args.out, "result.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    flag = " HANG" if record.get("timed_out") else ""
    unavailable = f" unavailable ({record['unavailable_len']} B; see local/)" if record.get("unavailable_len") else ""
    print(f"{args.name:<30} rc={record.get('rc')}{flag} sim_wall_s={record.get('wall_s')} "
          f"sim_program_s={record.get('program_s')} memory_mb={args.memory_mb}{unavailable}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
