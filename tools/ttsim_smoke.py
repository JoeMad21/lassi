"""Build one tt-metal example with ttmetal-host and run it on ttsim through the ttsim executor (task P4.11).

Usage (on the build host, through tools/rx.py):

    uv run python tools/ttsim_smoke.py <example>

<example> is one of the six examples P4.9 ran on ttsim (EXAMPLES:
add_2_integers_in_riscv, the gate's example, loopback, eltwise_binary,
eltwise_sfpu, matmul_single_core, and matmul_multi_core), by its name or
by the pinned build's target name, metal_example_<name> (TARGET_PREFIX),
as P4.G names it; the summary keeps the argument as given (argv) and the
bare name (example). The tool:

1. refuses, with exit status 2 and before anything is created, an unknown
   example (argparse's usage error), $LASSI_RUNS_ROOT or $LASSI_TOOLCHAINS
   unset or not absolute, a ttsim install the executor refuses
   (lassi.executors.ttsim TtsimExecutor.install), a toolchain that
   lassi.core.runner build_toolchain("ttmetal-host", ...) refuses (the
   tt-metal tree, the pinned clang++-20 and its --version check in the
   compile sandbox), and example sources it cannot read from the pinned
   tree;
2. reads the example from the pinned tree at run time (example_sources):
   its host program, each kernel it names after OVERRIDE_KERNEL_PREFIX
   (relative to tt_metal/programming_examples; the example must fall back
   to an empty prefix, so the kernel lies at that path in the build
   directory, where the kernel JIT looks first), and, for the matmuls,
   Matmul::Common's bmm_op.hpp at the path the program includes it by, as
   a harness file. No tt-metal text enters a tracked file;
3. builds it with the toolchain's own build(), exactly as a run builds an
   attempt, in <runs root>/ttsim-smoke/<id>/build, a new directory each time
   (<id> is the UTC time, the example, and a unique suffix);
4. runs the artifact with no arguments through TtsimExecutor under LIMITS
   (16 CPUs; the sandbox's default workdir disk cap and task limit), which
   sets the ttsim row's environment and reads findings, kernel JIT errors,
   and, after a hang, the Watcher note;
5. writes <id>/summary.json (plain ASCII) and prints a short report and its
   path.

The smoke passes only when the build gave an artifact, the run exited 0
with no hang, sim_ub is False, sim_gap is None, no jit-stage error came
back, and stdout holds the example's own success line (SUCCESS: "Success:
Result is 21" for the gate's example, "Test Passed" for the others;
plans/spikes/p4-ttsim-runtime.md, "The examples' own checks"), and, for the
matmuls, "Metalium vs Golden -- PCC = <value>" with a value above
PCC_FLOOR. Exit status 0 on a pass, 1 on a fail, 2 on a usage error or a
refusal (a sandbox that cannot run the program counts as one, recorded
with verdict "refused").

summary.json records the provenance (Agent Rules 1, 2, and 10): the commit,
dirty flag, and snapshot_of (git, as tools/capture_toolchain_fixtures.py
reads them; a run from a dirty tree or an rx snapshot is exploratory),
rx_run_id ($LASSI_RX_RUN_ID or null), host, date (UTC, ISO 8601), argv, the
pins (ttsim with VERSION, LIBRARY, SHA256, and SOC_DESCRIPTOR_SHA256;
tt-metal with VERSION, COMMIT, EXECUTABLE, and EXPECT_VERSION), the host
compiler and its --version lines, the device (TtsimExecutor.device(): a
simulator, never silicon), the limits, the build's diagnostics count, the
exit status, hang, sim_ub, sim_gap, every diagnostic of the run (findings),
the success line found or null, the PCC when printed, wall_s with the label
"simulator wall time, exploratory, sizing only" (never performance), the
verdict, and the reasons for a fail.

The helpers that assemble an example (EXAMPLES, MATMUL_COMMON,
NAMED_KERNEL, EMPTY_PREFIX, MATMUL_INCLUDE, example_sources, and seeded,
which puts a hand-written line into a kernel as P4.9's batch did) live here
once; tests/executors/test_ttsim_remote.py imports them. `--graphics` does
not apply: this is a repository tool, not a lassi command.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from capture_toolchain_fixtures import git_state, snapshot_of  # noqa: E402

from lassi.core.interfaces import Limits, RunResult  # noqa: E402
from lassi.core.runner import BuiltToolchain, RunError, build_toolchain  # noqa: E402
from lassi.executors.sandbox import WORKDIR_DISK_MB, SandboxSpec, SandboxUnavailableError  # noqa: E402
from lassi.executors.ttsim import TT_METAL_PIN, TTSIM_PIN, TtsimExecutor, TtsimInstall  # noqa: E402

# The examples' directory in the tt-metal tree, and each example's directory under it (the pinned layout; the
# ttmetal-host remote test builds the same six).
EXAMPLES_DIR = "tt_metal/programming_examples"
EXAMPLES = {
    "add_2_integers_in_riscv": "add_2_integers_in_riscv",
    "loopback": "loopback",
    "eltwise_binary": "eltwise_binary",
    "eltwise_sfpu": "eltwise_sfpu",
    "matmul_single_core": "matmul/matmul_single_core",
    "matmul_multi_core": "matmul/matmul_multi_core",
}
# The header library Matmul::Common, whose bmm_op.hpp the matmul examples include.
MATMUL_COMMON = "matmul/matmul_common"
MATMUL_HEADER = "bmm_op.hpp"
# A kernel path an example names: the string literal after OVERRIDE_KERNEL_PREFIX.
NAMED_KERNEL = re.compile(r'OVERRIDE_KERNEL_PREFIX\s*"([^"]+)"')
# The example's own definition when the build leaves the define unset: an empty prefix.
EMPTY_PREFIX = re.compile(r'#\s*ifndef\s+OVERRIDE_KERNEL_PREFIX\s*\n\s*#\s*define\s+OVERRIDE_KERNEL_PREFIX\s+""\s*\n')
# An include of Matmul::Common's header, in either form; the group is the path the program names.
MATMUL_INCLUDE = re.compile(r'#\s*include\s*[<"]([^>"]*bmm_op\.hpp)[>"]')
# The line that opens a kernel's entry function; seeded puts a line right after the brace that opens it.
KERNEL_ENTRY = re.compile(r"\bvoid\s+kernel_main\s*\(\s*\)")
# Each example's own success line (plans/spikes/p4-ttsim-runtime.md), matched as whole words in stdout.
GATE = "add_2_integers_in_riscv"
SUCCESS = {example: "Success: Result is 21" if example == GATE else "Test Passed" for example in EXAMPLES}
# The matmul examples' own check: the PCC they print must exceed PCC_FLOOR.
MATMULS = ("matmul_single_core", "matmul_multi_core")
PCC_LINE = re.compile(r"Metalium vs Golden -- PCC = (?P<value>[^\s,;]+)")
PCC_FLOOR = 0.97
# The run's limits: 16 CPUs (REFERENCE_CPUS in lassi.core.stages, and the P0.16 sandbox runs of P4.9) and 8192 MiB,
# under which those runs passed. The wall limit is 120 s: P4.9's clean runs of these six took at most 3.73 s of
# simulator wall time (matmul_single_core; exploratory, sizing only), so a run past it has hung. The executor keeps
# the sandbox's default workdir disk cap (WORKDIR_DISK_MB) and task limit; P4.9's JIT caches were at most 6.1 MiB.
LIMITS = Limits(wall_s=120.0, memory_mb=8192, cpus=16)
# The label every wall time in the summary carries (Agent Rule 2).
WALL_LABEL = "simulator wall time, exploratory, sizing only"
# The toolchain that builds the examples, the directory under the runs root, and the summary's name.
TOOLCHAIN = "ttmetal-host"
SMOKE_DIR = "ttsim-smoke"
SUMMARY = "summary.json"
# The pin keys the summary records.
TTSIM_KEYS = ("NAME", "VERSION", "LIBRARY", "SHA256", "SOC_DESCRIPTOR_SHA256")
TT_METAL_KEYS = ("NAME", "VERSION", "COMMIT", "EXECUTABLE", "EXPECT_VERSION")
# The exit statuses.
PASSED, FAILED, REFUSED = 0, 1, 2
# The prefix of an example's target name in the pinned build (metal_example_<name>), which P4.G passes.
TARGET_PREFIX = "metal_example_"
# A summary's run fields when nothing ran (the build failed, or the executor refused the run).
NO_RUN: dict[str, Any] = {
    "exit_code": None,
    "hang": None,
    "sim_ub": None,
    "sim_gap": None,
    "findings": [],
    "success_line": None,
    "pcc": None,
    "wall_s": None,
}


class Refusal(RuntimeError):
    """The smoke cannot start: the message says what is missing or refused."""


@dataclass(frozen=True)
class ExampleSources:
    """An example as a build takes it: files and harness files (relative path -> text), and its kernels' paths."""

    files: dict[str, str]
    harness: dict[str, str]
    kernels: tuple[str, ...]


def _text(path: Path) -> str:
    """Return a file of the pinned tree as text, decoded as UTF-8; OSError or UnicodeDecodeError when it cannot be."""
    return path.read_bytes().decode("utf-8")


def seeded(kernel: str, seed: str) -> str:
    """Return the kernel with `seed` as its own line right after the brace that opens kernel_main.

    ValueError when the kernel has no kernel_main or no opening brace after
    it. P4.9's batch placed its hand-written seeds the same way, so a
    compiler message quotes only that line.
    """
    lines = kernel.splitlines(keepends=True)
    start = next((number for number, line in enumerate(lines) if KERNEL_ENTRY.search(line)), None)
    opening = None if start is None else next((n for n in range(start, len(lines)) if "{" in lines[n]), None)
    if opening is None:
        raise ValueError("the kernel has no kernel_main() with an opening brace to put the seeded line after")
    lines.insert(opening + 1, seed.rstrip("\n") + "\n")
    return "".join(lines)


def example_sources(tree: Path, example: str, seed: str | None = None) -> ExampleSources:
    """Return `example`'s sources read from the tt-metal tree `tree` (see the module docstring, step 2).

    With `seed`, the example must name exactly one kernel, and the seed is
    put into it (seeded). Raise ValueError when the example does not fall
    back to an empty OVERRIDE_KERNEL_PREFIX, names no kernel, or cannot be
    seeded, and OSError or UnicodeDecodeError when a file cannot be read.
    """
    examples = tree / EXAMPLES_DIR
    source = _text(examples / EXAMPLES[example] / f"{example}.cpp")
    if not EMPTY_PREFIX.search(source):
        raise ValueError(f"{example} does not fall back to an empty OVERRIDE_KERNEL_PREFIX, so its kernel paths "
                         "would not be relative to its working directory")
    kernels = tuple(dict.fromkeys(NAMED_KERNEL.findall(source)))
    if not kernels:
        raise ValueError(f"{example} names no kernel after OVERRIDE_KERNEL_PREFIX")
    files = {f"{example}.cpp": source, **{path: _text(examples / path) for path in kernels}}
    if seed is not None:
        if len(kernels) != 1:
            raise ValueError(f"{example} names {len(kernels)} kernels; a seed goes into an example with one")
        files[kernels[0]] = seeded(files[kernels[0]], seed)
    header = examples / MATMUL_COMMON / MATMUL_HEADER
    harness = {path: _text(header) for path in dict.fromkeys(MATMUL_INCLUDE.findall(source))}
    return ExampleSources(files=files, harness=harness, kernels=kernels)


def success_found(example: str, stdout: str) -> str | None:
    """Return the example's own success line when stdout holds it as whole words, else None."""
    line = SUCCESS[example]
    return line if re.search(rf"(?<!\w){re.escape(line)}(?!\w)", stdout) else None


def read_pcc(stdout: str) -> float | None:
    """Return the last PCC a matmul example printed as a finite number, or None when it printed none."""
    values = [match["value"] for match in PCC_LINE.finditer(stdout)]
    try:
        value = float(values[-1]) if values else None
    except ValueError:
        return None
    return value if value is not None and math.isfinite(value) else None


def failures(example: str, run: RunResult) -> list[str]:
    """Return why a run misses the smoke's pass rule (see the module docstring), in order; [] for a pass."""
    reasons: list[str] = []
    if run.exit_code != 0:
        reasons.append(f"the program exited with status {run.exit_code}")
    if run.hang:
        reasons.append("the program hung and was stopped at its wall limit")
    if run.sim_ub is not False:
        reasons.append(f"sim_ub is {run.sim_ub}, not False")
    if run.sim_gap is not None:
        reasons.append(f"the simulator reported a gap, {run.sim_gap}")
    if any(item.stage == "jit" and item.severity == "error" for item in run.diagnostics):
        reasons.append("the kernel JIT reported an error")
    if success_found(example, run.stdout) is None:
        reasons.append(f"stdout lacks the example's own success line {SUCCESS[example]!r}")
    if example in MATMULS:
        pcc = read_pcc(run.stdout)
        if pcc is None or not pcc > PCC_FLOOR:
            reasons.append(f"the PCC is {pcc}, not above {PCC_FLOOR}")
    return reasons


@dataclass(frozen=True)
class Setup:
    """What a smoke run starts from: the runs root, the executor and its checked install, the toolchain, the sources."""

    runs_root: Path
    executor: TtsimExecutor
    install: TtsimInstall
    built: BuiltToolchain
    sources: ExampleSources


def _absolute_variable(name: str) -> Path:
    """Return $name as an absolute Path; Refusal when it is unset, empty, or relative."""
    value = os.environ.get(name, "")
    if not value or not Path(value).is_absolute():
        raise Refusal(f"${name} must be set to an absolute path (the gate sets it on the build host), got {value!r}")
    return Path(value)


def prepare(example: str) -> Setup:
    """Check the environment, the ttsim install, and the toolchain, and read the example; Refusal says what failed.

    Nothing is created: the install check reads files, build_toolchain
    runs only its --version check in the compile sandbox, and the sources
    are read from the executor's checked tree.
    """
    runs_root = _absolute_variable("LASSI_RUNS_ROOT")
    toolchains = _absolute_variable("LASSI_TOOLCHAINS")
    executor = TtsimExecutor()
    try:
        install = executor.install()
        built = build_toolchain(TOOLCHAIN, toolchains, build_root=runs_root)
    except (RunError, SandboxUnavailableError) as error:
        raise Refusal(str(error)) from None
    try:
        sources = example_sources(install.tree, example)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise Refusal(f"cannot read {example} from the pinned tree {install.tree}: {error}") from None
    return Setup(runs_root=runs_root, executor=executor, install=install, built=built, sources=sources)


def _pins(executor: TtsimExecutor) -> dict[str, dict[str, str]]:
    """Return the pin pairs the summary records: ttsim's TTSIM_KEYS and tt-metal's TT_METAL_KEYS."""
    pins, chosen = executor.pins, {TTSIM_PIN: TTSIM_KEYS, TT_METAL_PIN: TT_METAL_KEYS}
    return {name: {key: pins[name][key] for key in keys if key in pins[name]} for name, keys in chosen.items()}


def provenance(setup: Setup, example: str, argv: Sequence[str]) -> dict[str, Any]:
    """Return the summary's provenance fields (see the module docstring); git runs for the commit and dirty flag."""
    commit, dirty = git_state()
    spec_fields = {field.name: field.default for field in dataclasses.fields(SandboxSpec)}
    return {
        "tool": "tools/ttsim_smoke.py",
        "example": example,
        "argv": list(argv),
        "commit": commit,
        "dirty": dirty,
        "snapshot_of": snapshot_of(),
        "rx_run_id": os.environ.get("LASSI_RX_RUN_ID") or None,
        "host": platform.node(),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": setup.executor.device(),
        "pins": _pins(setup.executor),
        "compiler": {"executable": setup.built.executable, "version": list(setup.built.version)},
        "limits": {**dataclasses.asdict(LIMITS), "workdir_disk_mb": WORKDIR_DISK_MB,
                   "tasks_max": spec_fields["tasks_max"]},
    }


def _run_fields(example: str, run: RunResult) -> dict[str, Any]:
    """Return the summary's fields for a run: its status, flags, findings, own check, PCC, wall time, and verdict."""
    reasons = failures(example, run)
    return {
        "exit_code": run.exit_code,
        "hang": run.hang,
        "sim_ub": run.sim_ub,
        "sim_gap": run.sim_gap,
        "findings": [dataclasses.asdict(item) for item in run.diagnostics],
        "success_line": success_found(example, run.stdout),
        "pcc": read_pcc(run.stdout),
        "wall_s": {"value": run.wall_s, "label": WALL_LABEL},
        "verdict": "pass" if not reasons else "fail",
        "reasons": reasons,
    }


def smoke(setup: Setup, example: str, argv: Sequence[str], run_dir: Path) -> dict[str, Any]:
    """Build the example in <run_dir>/build, run it through the executor, and return the summary."""
    summary = provenance(setup, example, argv)
    build = run_dir / "build"
    build.mkdir()
    sources = setup.sources
    result = setup.built.toolchain.build(sources.files, build, harness=sources.harness or None)
    summary["build_diagnostics"] = len(result.diagnostics)
    if result.artifact is None:
        reason = "the example did not build; see build/compile.stderr"
        return {**summary, **NO_RUN, "verdict": "fail", "reasons": [reason]}
    try:
        run = setup.executor.run(result.artifact, [], LIMITS)
    except (SandboxUnavailableError, ValueError) as error:
        return {**summary, **NO_RUN, "verdict": "refused", "reasons": [f"the executor refused the run: {error}"]}
    return {**summary, **_run_fields(example, run)}


def new_run_dir(runs_root: Path, example: str) -> Path:
    """Create and return a new directory <runs root>/ttsim-smoke/<UTC time>-<example>-<unique suffix>."""
    parent = runs_root / SMOKE_DIR
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path(tempfile.mkdtemp(prefix=f"{stamp}-{example}-", dir=parent))


def write_summary(run_dir: Path, summary: Mapping[str, Any]) -> Path:
    """Write <run_dir>/summary.json as plain ASCII JSON (sorted keys, one space indent) and return its path."""
    path = run_dir / SUMMARY
    path.write_bytes((json.dumps(summary, indent=1, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii"))
    return path


def _parser() -> argparse.ArgumentParser:
    """Return the argument parser: one example, read through bare_name, then one of EXAMPLES."""
    parser = argparse.ArgumentParser(prog="ttsim_smoke.py", description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "example", type=bare_name, choices=sorted(EXAMPLES), help="the example to build and run on ttsim"
    )
    return parser


def bare_name(argument: str) -> str:
    """Return `argument` without a leading TARGET_PREFIX, the pinned build's target name for an example."""
    return argument[len(TARGET_PREFIX):] if argument.startswith(TARGET_PREFIX) else argument


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke for the example `argv` names; return 0 on a pass, 1 on a fail, 2 on a refusal.

    A usage error exits 2 through argparse (SystemExit). A refusal prints
    "ttsim_smoke: <why>" on stderr before anything is created.
    """
    given = list(sys.argv[1:] if argv is None else argv)
    example = _parser().parse_args(given).example
    try:
        setup = prepare(example)
    except Refusal as refusal:
        print(f"ttsim_smoke: {refusal}", file=sys.stderr)
        return REFUSED
    run_dir = new_run_dir(setup.runs_root, example)
    summary = smoke(setup, example, given, run_dir)
    path = write_summary(run_dir, summary)
    reasons = "; ".join(summary["reasons"]) or "every check passed"
    print(f"ttsim_smoke: {example}: {summary['verdict']} ({reasons})")
    print(f"ttsim_smoke: device: {summary['device']}")
    print(f"ttsim_smoke: summary: {path}")
    return {"pass": PASSED, "fail": FAILED}.get(summary["verdict"], REFUSED)


if __name__ == "__main__":
    sys.exit(main())
