"""Device profiles of the saved TT translations on the Wormhole cards (task P17.65; plans/runs/tracy/README.md).

The owner starts this run in a window (OQ-127): every card is in an identity IOMMU domain, so no agent launches a
model-generated program on the cards (lassi.executors.tt_silicon, Known limits). From a clean checkout on alpha01,
through `uv run tools/rx.py job start`:

    uv run python plans/runs/tracy/profile.py --minutes 120      # the run; no step starts with < --budget s left
    uv run python plans/runs/tracy/profile.py --dry-run          # reads and checks the candidates' trial files only

Each saved translation in plans/runs/tracy/candidates.tsv (an exact match on ttsim, written by a served model) is
built again with its target's toolchain, at the pinned tt-metal built with Tracy, and run once on the cards through
the sandboxed tt_silicon executor with device_profiler kernel and visible_card VISIBLE_CARD (card 0 alone), which
reads tt-metal's device profiler: the device kernel time of every program the run launched, at the AICLK the cards
reported. Before an item's first translation,
its C++ program runs once on the host CPU in the sandbox (the outputs every TT run is judged against, and a host CPU
wall time), and its tracked TT reference, when the item has one, runs once on the cards the same way. The first card
run is the qualification, tt-pairs-v0 loopback's tracked reference: nothing model-written runs unless its C++ program
and that reference both ran and passed and the reference gave a device timing (qualification_problem).

Every step appends one JSON record to report/records.jsonl and one row to report/summary.tsv as it ends, so a run
cut short keeps what it measured. A translation's device time is self-reported: its own process writes the profiler
files (lassi.core.record DeviceTiming). Wall times include process start, device open, and the kernel JIT; they are
never the device time.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

COLUMNS = ("order", "suite", "item", "source", "target", "model", "sweep", "files_sha1", "trial")
TT_TARGETS = ("tt", "ttkh")
# The qualification: the tracked reference of tt-pairs-v0 loopback, the first program on the cards.
QUALIFY = ("tt-pairs-v0", "loopback", "tt")
DONE, DEADLINE, STOPPED, QUALIFY_FAILED = 0, 2, 3, 4
SUMMARY_COLUMNS = ("step", "kind", "suite", "item", "model", "status", "passed", "exit_code", "wall_s", "kernel_ns",
                   "programs", "aiclk_mhz", "self_reported", "note")
# The toolchain each language builds with, as the recipes bind them (projects/lassi-demo/bale-tt-30b-base.yaml).
TOOLCHAIN = {"cpp": "gcc-native", "tt": "ttmetal-host", "ttkh": "ttmetal-host"}
PURPOSE = "eval"  # an evaluation run (lassi.core.stages PURPOSE; Agent Rule 5)
# The one card every card run uses (TT_VISIBLE_DEVICES; task P17.66): with all eight cards visible, tt-metal's fabric
# control plane aborted the loopback reference (rx 20261116-192634-tracy-07f1); the smoke ran on card 0 alone.
VISIBLE_CARD = 0
REPO = Path(__file__).resolve().parents[3]
CANDIDATES = REPO / "plans" / "runs" / "tracy" / "candidates.tsv"
_SHA = re.compile(r"[0-9a-f]{10}")
_TAIL = 3000


@dataclass(frozen=True)
class Candidate:
    """One saved translation: a row of candidates.tsv."""

    order: int
    suite: str
    item: str
    source: str
    target: str
    model: str
    sweep: str
    files_sha1: str
    trial: str


class CandidateError(ValueError):
    """A saved translation that cannot be run: its trial is not the exact match its row says."""


class BuildError(Exception):
    """A build that produced no program; its text is the compiler's tail."""


@dataclass(frozen=True)
class Step:
    """One run: an item's C++ program ("cpp"), its tracked TT reference ("reference"), or a translation."""

    kind: str
    suite: str
    item: str
    target: str
    candidate: Candidate | None = None


def read_candidates(path: Path) -> list[Candidate]:
    """Return the rows of candidates.tsv sorted by order; ValueError naming the line for anything malformed."""
    rows: list[Candidate] = []
    lines: dict[int, int] = {}
    header_seen = False
    for number, line in enumerate(Path(path).read_text(encoding="ascii").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        if not header_seen:
            if line != "\t".join(COLUMNS):
                raise ValueError(f"{path}: line {number}: the header must be {chr(9).join(COLUMNS)!r}")
            header_seen = True
            continue
        row = _candidate(path, number, line.split("\t"))
        if row.order in lines:
            first = lines[row.order]
            raise ValueError(f"{path}: line {number}: order {row.order} appears again (first on line {first})")
        lines[row.order] = number
        rows.append(row)
    if not header_seen:
        raise ValueError(f"{path}: line 1: no header line {chr(9).join(COLUMNS)!r}")
    return sorted(rows, key=lambda row: row.order)


def _candidate(path: Path, number: int, cells: list[str]) -> Candidate:
    """Return one checked row; ValueError naming `number` when a cell breaks the file's rules."""
    where = f"{path}: line {number}"
    if len(cells) != len(COLUMNS):
        raise ValueError(f"{where}: {len(cells)} fields, expected {len(COLUMNS)}")
    order, suite, item, source, target, model, sweep, sha, trial = cells
    if not order.isdigit() or int(order) < 1:
        raise ValueError(f"{where}: order must be a positive integer, got {order!r}")
    if source != "cpp":
        raise ValueError(f"{where}: source must be cpp, got {source!r}")
    if target not in TT_TARGETS:
        raise ValueError(f"{where}: target must be one of {', '.join(TT_TARGETS)}, got {target!r}")
    if not _SHA.fullmatch(sha):
        raise ValueError(f"{where}: files_sha1 must be 10 lowercase hex digits, got {sha!r}")
    parts = PurePosixPath(trial).parts
    if trial.startswith("/") or "\\" in trial or ".." in parts or not trial.endswith("/trial.json"):
        raise ValueError(f"{where}: trial must be a relative path ending in /trial.json with no '..', got {trial!r}")
    if not all((suite, item, model, sweep)):
        raise ValueError(f"{where}: suite, item, model, and sweep must not be empty")
    return Candidate(int(order), suite, item, source, target, model, sweep, sha, trial)


def files_digest(files: Mapping[str, str]) -> str:
    """Return the first 10 hex digits of the sha1 of the files as sorted JSON (the inventory's digest)."""
    return hashlib.sha1(json.dumps(dict(files), sort_keys=True).encode()).hexdigest()[:10]


def candidate_files(trial: Mapping[str, Any], expected: str) -> dict[str, str]:
    """Return the final attempt's files of a saved trial; CandidateError unless it is the row's exact match."""
    attempts = trial.get("attempts") or []
    if not attempts:
        raise CandidateError("the trial holds no attempt")
    last = attempts[-1]
    if last.get("stage_reached") != "S5":
        raise CandidateError(f"its final attempt reached {last.get('stage_reached')!r}, not S5")
    alignment = last.get("alignment")
    mean = alignment.get("mean") if isinstance(alignment, Mapping) else alignment
    if mean != 1.0:
        raise CandidateError(f"its final attempt's alignment is {mean!r}, not 1.0")
    if (last.get("guards") or {}).get("host_compute") is True:
        raise CandidateError("its final attempt computes on the host (the host-compute guard is True)")
    files = dict(last.get("files") or {})
    if files_digest(files) != expected:
        raise CandidateError(f"its final files digest to {files_digest(files)}, not the row's {expected}")
    return files


def plan_steps(candidates: Sequence[Candidate], has_tt_reference: Callable[[str, str, str], bool]) -> list[Step]:
    """Return the run's steps: the qualification, then each item's C++ program and reference before its first row."""
    steps = [Step("cpp", *QUALIFY), Step("reference", *QUALIFY)]
    seen = {QUALIFY[:2]}
    for candidate in candidates:
        key = (candidate.suite, candidate.item)
        if key not in seen:
            seen.add(key)
            steps.append(Step("cpp", candidate.suite, candidate.item, candidate.target))
            if has_tt_reference(candidate.suite, candidate.item, candidate.target):
                steps.append(Step("reference", candidate.suite, candidate.item, candidate.target))
        steps.append(Step("candidate", candidate.suite, candidate.item, candidate.target, candidate))
    return steps


class World(Protocol):
    """What run_steps needs from the host: programs, builds, inputs, runs, the oracle, and the quarantine check."""

    def program(self, step: Step) -> tuple[str, dict[str, str], dict[str, str]]:
        """Return the step's language, files, and harness files; CandidateError for a bad translation."""
        ...

    def build(self, language: str, files: Mapping[str, str], harness: Mapping[str, str], workdir: Path) -> Path:
        """Build the program in `workdir` and return the artifact; BuildError when no program was built."""
        ...

    def inputs(self, step: Step, artifact: Path) -> list[str]:
        """Stage the item's inputs beside the artifact and return its arguments."""
        ...

    def run_cpu(self, artifact: Path, inputs: Sequence[str]) -> Any:
        """Run a C++ program once on the host CPU in the sandbox; return its RunResult."""
        ...

    def run_card(self, artifact: Path, inputs: Sequence[str]) -> Any:
        """Run a TT program once on the cards in the sandbox; return its RunResult (any error ends the run)."""
        ...

    def compare(self, step: Step, expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[Any]:
        """Return the oracle's per-output statistics of `actual` against `expected`."""
        ...

    def quarantined(self) -> list[str]:
        """Return the quarantine markers that exist now."""
        ...

    def keep_profile(self, artifact: Path, dest: Path) -> list[str]:
        """Copy the card run's profiler files to `dest`; return the copied paths."""
        ...


@dataclass
class _State:
    """What later steps read: each item's C++ and reference outputs, the programs already run, the card runs."""

    cpu: dict[tuple[str, str], Any] = field(default_factory=dict)
    reference: dict[tuple[str, str], Any] = field(default_factory=dict)
    outputs: dict[int, Any] = field(default_factory=dict)
    seen: dict[tuple[str, str, str, str], int] = field(default_factory=dict)
    card_runs: int = 0


class _Log:
    """Appends one JSON record to records.jsonl and one row to summary.tsv per step, and prints a short line."""

    DEFAULTS: dict[str, Any] = {
        "order": None, "model": None, "sweep": None, "trial": None, "language": None, "files_sha1": None,
        "status": None, "reason": None, "passed": None, "same_as": None, "exit_code": None, "hang": None,
        "wall_s": None, "outputs": [], "vs_reference": None, "device_timing": None, "notes": [], "profile": [],
    }

    def __init__(self, report: Path) -> None:
        """Create `report` and write the summary's header."""
        self.report = Path(report)
        self.report.mkdir(parents=True, exist_ok=True)
        (self.report / "summary.tsv").write_text("\t".join(SUMMARY_COLUMNS) + "\n", encoding="ascii")
        self.last: dict[str, Any] = {}

    def write(self, record: Mapping[str, Any]) -> None:
        """Append `record`, completed with DEFAULTS, to both files; keep it as `last`."""
        full = {**self.DEFAULTS, **record}
        self.last = full
        with open(self.report / "records.jsonl", "a", encoding="ascii") as handle:
            handle.write(json.dumps(full, sort_keys=True, default=str) + "\n")
        row = _summary_row(full)
        with open(self.report / "summary.tsv", "a", encoding="ascii") as handle:
            handle.write("\t".join(row) + "\n")
        print("  ".join(cell for cell in row if cell), flush=True)

    def stop(self, status: str, reason: str) -> None:
        """Append the stop record that ends the run."""
        self.write({"step": None, "kind": "stop", "suite": None, "item": None, "status": status, "reason": reason})


def _summary_row(record: Mapping[str, Any]) -> list[str]:
    """Return the summary.tsv cells of one record, as plain ASCII with no tab or line break."""
    timing = record.get("device_timing") or {}
    note = record.get("reason") or ""
    if record.get("same_as"):
        note = f"same program as step {record['same_as']}"
    values = [record.get("step"), record.get("kind"), record.get("suite"), record.get("item"), record.get("model"),
              record.get("status"), record.get("passed"), record.get("exit_code"), record.get("wall_s"),
              timing.get("kernel_ns_total"), len(timing.get("programs") or []) if timing else None,
              timing.get("aiclk_mhz"), timing.get("self_reported"), note]
    return [_cell(value) for value in values]


def _cell(value: Any) -> str:
    """Return one summary cell: "" for None, else the text with tabs and line breaks as spaces, ASCII only."""
    if value is None:
        return ""
    text = " ".join(str(value).split())[:300]
    return text.encode("ascii", "replace").decode("ascii")


def run_steps(
    steps: Sequence[Step],
    world: World,
    report: Path,
    work: Path,
    *,
    deadline: float,
    pause_s: float,
    budget_s: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> int:
    """Run `steps` in order through `world`, recording each; return DONE, DEADLINE, STOPPED, or QUALIFY_FAILED."""
    log, state = _Log(report), _State()
    for number, step in enumerate(steps, 1):
        if clock() + budget_s > deadline:
            log.stop("deadline", f"fewer than {budget_s:.0f} s were left before step {number} of {len(steps)}")
            return DEADLINE
        code = _step(number, step, world, Path(work), state, log, lambda: _pause(state, pause_s, sleep))
        if code is not None:
            return code
        problem = qualification_problem(step, log.last)
        if problem is not None:
            log.stop("qualification-failed", f"{problem}; nothing model-written runs")
            return QUALIFY_FAILED
    return DONE


def qualification_problem(step: Step, record: Mapping[str, Any]) -> str | None:
    """Return why a qualification step (QUALIFY) failed, or None: every other step, and one that ran and passed.

    The qualification's C++ program and its tracked reference must each have
    run and passed, and the reference must have given a device timing, before
    anything model-written runs on the cards: a build failure, a skip, an
    unclean run, a failed output, or a missing timing all fail it.
    """
    if step.kind not in ("cpp", "reference") or (step.suite, step.item, step.target) != QUALIFY:
        return None  # a translation of the same item is judged like any other, never as the qualification
    where = f"the qualification's {step.kind} step ({QUALIFY[0]}/{QUALIFY[1]})"
    if record.get("status") != "ran":
        return f"{where} ended {record.get('status')}: {record.get('reason') or 'no reason given'}"
    if record.get("passed") is not True:
        return f"{where} did not pass"
    if step.kind == "reference" and record.get("device_timing") is None:
        return f"{where} gave no device timing, so the device profiler is not working on the cards"
    return None


def _pause(state: _State, pause_s: float, sleep: Callable[[float], None]) -> None:
    """Sleep `pause_s` before every card run but the first, then count the run."""
    if state.card_runs:
        sleep(pause_s)
    state.card_runs += 1


def _base(number: int, step: Step) -> dict[str, Any]:
    """Return the fields every record of `step` holds."""
    record: dict[str, Any] = {"step": number, "kind": step.kind, "suite": step.suite, "item": step.item,
                              "target": step.target}
    if step.candidate is not None:
        found = step.candidate
        record.update(order=found.order, model=found.model, sweep=found.sweep, trial=found.trial,
                      files_sha1=found.files_sha1)
    return record


def _step(
    number: int, step: Step, world: World, work: Path, state: _State, log: _Log, pause: Callable[[], None]
) -> int | None:
    """Do one step: skip, reuse, or build and run it; return an exit code that ends the run, or None."""
    base, key = _base(number, step), (step.suite, step.item)
    if step.kind != "cpp" and state.cpu.get(key) is None:
        log.write({**base, "status": "skipped", "reason": "its C++ program did not run cleanly, so nothing checks it"})
        return None
    try:
        language, files, harness = world.program(step)
    except CandidateError as error:
        log.write({**base, "status": "skipped", "reason": str(error)})
        return None
    base.update(language=language, files_sha1=base.get("files_sha1") or files_digest(files))
    # The key leaves the suite out on purpose: tt-pairs-v0-kh keeps tt-pairs-v0's items, inputs, seeds, and tolerances
    # (assets/bench/tt-pairs-v0-kh.yaml, its header), so the same item's same files and harness are the same run.
    program = (step.item, language, files_digest(files), files_digest(harness))
    same = state.seen.get(program)
    if same is not None:
        _reuse(state, step, key, same)
        log.write({**base, "status": "skipped", "same_as": same, "reason": f"the same program as step {same}"})
        return None
    state.seen[program] = number
    try:
        artifact = world.build(language, files, harness, work / f"{number:03d}-{step.kind}-{step.item}")
    except BuildError as error:
        log.write({**base, "status": "build-failed", "reason": str(error)[-_TAIL:]})
        return None
    if step.kind == "cpp":
        _cpu_step(number, step, world, artifact, state, log, base)
        return None
    return _card_step(number, step, world, artifact, state, log, base, pause)


def _reuse(state: _State, step: Step, key: tuple[str, str], same: int) -> None:
    """Let a skipped duplicate stand for the earlier program: its outputs serve this item's later steps."""
    if step.kind == "cpp":
        state.cpu[key] = state.outputs.get(same)
    elif step.kind == "reference":
        state.reference[key] = state.outputs.get(same)


def _run_fields(run: Any) -> dict[str, Any]:
    """Return a run's status fields: exit code, hang, wall time, error notes, and the streams' tails."""
    notes = [f"{item.code}: {item.message}" for item in (run.diagnostics or []) if item.severity != "info"]
    return {"exit_code": run.exit_code, "hang": bool(run.hang), "wall_s": round(float(run.wall_s), 3),
            "notes": notes[:20], "stdout_tail": (run.stdout or "")[-_TAIL:], "stderr_tail": (run.stderr or "")[-_TAIL:]}


def _clean(run: Any) -> bool:
    """Return True when a run ended with status 0 and no hang."""
    return run.exit_code == 0 and not run.hang


def _cpu_step(number: int, step: Step, world: World, artifact: Path, state: _State, log: _Log,
              base: dict[str, Any]) -> None:
    """Run an item's C++ program on the host CPU; keep its outputs for the item's later steps."""
    run = world.run_cpu(artifact, world.inputs(step, artifact))
    outputs = dict(run.output_files) if _clean(run) else None
    state.cpu[(step.suite, step.item)] = state.outputs[number] = outputs
    log.write({**base, **_run_fields(run), "status": "ran", "passed": _clean(run)})


def _stats(stats: Sequence[Any]) -> list[dict[str, Any]]:
    """Return the oracle's per-output statistics as plain dicts."""
    return [{"name": entry.name, "passed": bool(entry.passed), "max_abs": entry.max_abs, "pcc": entry.pcc,
             "note": entry.note} for entry in stats]


def _judge(world: World, step: Step, expected: Mapping[str, Any], run: Any) -> tuple[list[dict[str, Any]], str]:
    """Return the stats of a clean run's outputs against `expected`, and a note when the oracle refused them."""
    if not _clean(run):
        return [], ""
    try:
        return _stats(world.compare(step, expected, run.output_files)), ""
    except ValueError as error:
        return [], f"the oracle could not compare: {error}"


def _card_step(number: int, step: Step, world: World, artifact: Path, state: _State, log: _Log,
               base: dict[str, Any], pause: Callable[[], None]) -> int | None:
    """Run a reference or a translation once on the cards; judge it, record its device time; maybe stop the run."""
    key = (step.suite, step.item)
    inputs = world.inputs(step, artifact)
    pause()
    try:
        run = world.run_card(artifact, inputs)
    except Exception as error:  # noqa: BLE001 - any failure to run on the cards ends the run
        log.write({**base, "status": "error", "reason": f"{type(error).__name__}: {error}"[-_TAIL:]})
        log.stop("card-error", f"step {number} could not run on the cards; nothing more runs")
        return STOPPED
    outputs, note = _judge(world, step, state.cpu[key], run)
    passed = _clean(run) and bool(outputs) and all(entry["passed"] for entry in outputs)
    versus = None
    if step.kind == "candidate" and state.reference.get(key) is not None and _clean(run):
        versus, _ = _judge(world, step, state.reference[key], run)
    if step.kind == "reference":
        state.reference[key] = state.outputs[number] = dict(run.output_files) if _clean(run) else None
    profile = world.keep_profile(artifact, log.report / "profiler" / f"{number:03d}-{step.kind}-{step.item}")
    timing = timing_record(run.device_timing, reference=step.kind == "reference")
    log.write({**base, **_run_fields(run), "status": "ran", "passed": passed, "outputs": outputs,
               "reason": note or None, "vs_reference": versus, "profile": profile, "device_timing": timing})
    markers = world.quarantined()
    if markers:
        log.stop("quarantine", f"a quarantine marker exists after step {number}: {', '.join(markers)}")
        return STOPPED
    return None


def timing_record(timing: Any, *, reference: bool) -> dict[str, Any] | None:
    """Return a DeviceTiming as a dict, its self_reported False for a tracked reference; None when not measured."""
    if timing is None:
        return None
    programs = [{"device_id": item.device_id, "op_name": item.op_name, "call_count": item.call_count,
                 "cores": item.cores, "kernel_cycles": item.kernel_cycles, "kernel_ns": item.kernel_ns,
                 "fw_ns": item.fw_ns} for item in timing.programs]
    return {"source": timing.source, "build": timing.build, "arch": timing.arch, "aiclk_mhz": timing.aiclk_mhz,
            "self_reported": False if reference else bool(timing.self_reported),
            "kernel_ns_total": sum(item["kernel_ns"] for item in programs), "programs": programs}


class CardWorld:
    """The host's World on alpha01: the suites, the pinned toolchains, the native and tt_silicon executors."""

    def __init__(self, runs: Path, scratch: Path, toolchains: Path, cpu: Any, card: Any) -> None:
        """Keep the saved runs' root ($LASSI_RUNS_ROOT/runs), the scratch and toolchains roots, and the executors."""
        self.runs, self.scratch, self.toolchains = Path(runs), Path(scratch), Path(toolchains)
        self.cpu, self.card = cpu, card
        # The directory every build dir lies under, which each compile hides apart from its own (build_toolchain's
        # build_root, as a run passes its runs root); main sets it to the run's work directory.
        self.build_root: Path | None = None
        self._suites: dict[str, tuple[Any, Path]] = {}
        self._built: dict[str, Any] = {}

    def suite(self, name: str) -> tuple[Any, Path]:
        """Return a suite loaded from assets/bench/<name>.yaml and the root of its pinned sources."""
        if name not in self._suites:
            from lassi import bench
            from lassi.bench import registry

            suite = bench.load_suite(REPO / "assets" / "bench" / f"{name}.yaml")
            self._suites[name] = (suite, registry.sources_dir(self.scratch, suite))
        return self._suites[name]

    def has_tt_reference(self, suite: str, item: str, target: str) -> bool:
        """Return True when the item's reference for `target` is a TT program, not its C++ source."""
        return self.suite(suite)[0].reference_language(item, target) != "cpp"

    def program(self, step: Step) -> tuple[str, dict[str, str], dict[str, str]]:
        """Return the language, files, and harness files of a step's program."""
        from lassi.bench.registry import Direction

        suite, root = self.suite(step.suite)
        direction = Direction("cpp", step.target)
        if step.kind == "cpp":
            language, files = "cpp", suite.source_files(step.item, direction, root, purpose=PURPOSE)
        elif step.kind == "reference":
            language = suite.reference_language(step.item, step.target)
            files = suite.reference_target(step.item, direction, root, purpose=PURPOSE)
        else:
            assert step.candidate is not None
            trial = json.loads((self.runs / step.candidate.trial).read_text(encoding="utf-8"))
            language, files = step.target, candidate_files(trial, step.candidate.files_sha1)
        return language, files, suite.support_files(step.item, root, purpose=PURPOSE, language=language)

    def build(self, language: str, files: Mapping[str, str], harness: Mapping[str, str], workdir: Path) -> Path:
        """Build with the language's pinned toolchain in a new `workdir`; BuildError with the compiler's tail."""
        from lassi.core.runner import build_toolchain

        name = TOOLCHAIN[language]
        if name not in self._built:
            self._built[name] = build_toolchain(name, self.toolchains, build_root=self.build_root)
        workdir.mkdir(parents=True)
        toolchain = self._built[name].toolchain
        result = toolchain.build(files, workdir, harness=harness) if harness else toolchain.build(files, workdir)
        errors = [item for item in result.diagnostics if item.severity == "error"]
        if result.artifact is None or errors:
            stderr = workdir / "compile.stderr"
            text = stderr.read_bytes().decode("utf-8", "replace") if stderr.is_file() else ""
            raise BuildError(text[-_TAIL:] or "; ".join(item.message for item in errors) or "no program was built")
        return Path(result.artifact)

    def inputs(self, step: Step, artifact: Path) -> list[str]:
        """Stage the item's held-out inputs beside the program and return its arguments."""
        from lassi.bench import registry

        return registry.stage_inputs(self.suite(step.suite)[0].item(step.item, purpose=PURPOSE), artifact.parent)

    def run_cpu(self, artifact: Path, inputs: Sequence[str]) -> Any:
        """Run a C++ program once on the host CPU in the sandbox."""
        return self.cpu.run(artifact, inputs, _limits(8192))

    def run_card(self, artifact: Path, inputs: Sequence[str]) -> Any:
        """Run a TT program once on the cards through the sandboxed tt_silicon executor."""
        return self.card.run(artifact, inputs, _limits(32768))

    def compare(self, step: Step, expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[Any]:
        """Compare output files with binary_io under the item's declared tolerance."""
        from lassi.oracles.binary_io import BinaryIOOracle

        spec = self.suite(step.suite)[0].item(step.item, purpose=PURPOSE)
        if spec.tolerance is None:
            raise ValueError(f"{step.suite}/{step.item} declares no tolerance")
        oracle = BinaryIOOracle(metric=spec.tolerance.metric, threshold=spec.tolerance.threshold)
        return oracle.compare(expected, actual, sides=("expected", "on the cards"))

    def quarantined(self) -> list[str]:
        """Return the quarantine markers under $LASSI_SCRATCH/locks."""
        from lassi.executors.tt_silicon import LOCKS_DIR, quarantine_markers

        return [str(path) for path in quarantine_markers(self.scratch / LOCKS_DIR)]

    def keep_profile(self, artifact: Path, dest: Path) -> list[str]:
        """Copy the run's profiler files (<workdir>/@tt-silicon/profiler) to `dest`; return their relative paths."""
        from lassi.executors.tt_silicon import PROFILER_DIR, RESERVED_DIR

        source = artifact.parent / RESERVED_DIR / PROFILER_DIR
        kept = []
        for path in sorted(source.rglob("*")) if source.is_dir() else []:
            if path.is_file() and not path.is_symlink():
                target = dest / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                kept.append(str(target.relative_to(dest.parent.parent)))
        return kept


def _limits(memory_mb: int) -> Any:
    """Return a run's limits: 300 s of wall time, `memory_mb` of memory, 16 CPUs (sweep10-si's card memory)."""
    from lassi.core.interfaces import Limits

    return Limits(wall_s=300.0, memory_mb=memory_mb, cpus=16)


def _git(*args: str) -> str:
    """Return a git command's output in the repository."""
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()


def provenance(card: Any, cpu: Any, candidates: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Return the run's provenance: commit and clean state, pins, both devices, host, date, and the inputs."""
    from lassi.toolchains import pins

    return {
        "task": "P17.65", "commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain")),
        "pins": {name: pins.read_pin(name) for name in ("tt-metal", "gcc")},
        "device": card.device(), "cpu_device": cpu.device(), "host": platform.node(),
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "candidates": str(candidates.relative_to(REPO)) if candidates.is_relative_to(REPO) else str(candidates),
        "candidates_sha256": hashlib.sha256(candidates.read_bytes()).hexdigest(),
        "minutes": args.minutes, "pause_s": args.pause, "budget_s": args.budget, "limit": args.limit,
    }


def dry_run(steps: Sequence[Step], runs: Path) -> int:
    """Print the plan and check every candidate's trial file; return 0 when all are the exact matches listed."""
    bad = 0
    for number, step in enumerate(steps, 1):
        line = f"{number:3d} {step.kind:9s} {step.suite}/{step.item} -> {step.target}"
        if step.candidate is not None:
            try:
                trial = json.loads((runs / step.candidate.trial).read_text(encoding="utf-8"))
                files = candidate_files(trial, step.candidate.files_sha1)
                line += f"  {step.candidate.model}  {len(files)} files  ok"
            except (OSError, ValueError) as error:
                bad += 1
                line += f"  {step.candidate.model}  REFUSED: {error}"
        print(line, flush=True)
    print(f"dry run: {len(steps)} steps, {bad} candidate(s) refused", flush=True)
    return 0 if bad == 0 else 1


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    parser.add_argument("--minutes", type=float, help="start no step once fewer than --budget seconds are left")
    parser.add_argument("--pause", type=float, default=20.0, help="seconds between card runs (default 20)")
    parser.add_argument("--budget", type=float, default=420.0,
                        help="seconds a step needs to start (default 420: a build, the pause, a 300 s run)")
    parser.add_argument("--limit", type=int, help="only the first N candidates")
    parser.add_argument("--dry-run", action="store_true", help="read and check the candidates; build and run nothing")
    args = parser.parse_args(argv)
    if not args.dry_run and (args.minutes is None or args.minutes <= 0):
        parser.error("--minutes is required (a positive number) unless --dry-run")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the profiles (or the dry run) on alpha01; return run_steps's exit code."""
    args = _arguments(argv)
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (1, 1))  # the 1-byte core limit every run holds (AGENTS.md)
    candidates = read_candidates(args.candidates)[: args.limit]
    runs_root, scratch = Path(os.environ["LASSI_RUNS_ROOT"]), Path(os.environ["LASSI_SCRATCH"])
    world = CardWorld(runs_root / "runs", scratch, Path(os.environ["LASSI_TOOLCHAINS"]), None, None)
    steps = plan_steps(candidates, world.has_tt_reference)
    if args.dry_run:
        return dry_run(steps, runs_root / "runs")
    from lassi.executors.native import NativeExecutor
    from lassi.executors.tt_silicon import TtSiliconExecutor

    world.cpu = NativeExecutor()
    try:
        world.card = TtSiliconExecutor(
            hugepages="private-per-run", reset_filter=True, device_profiler="kernel", visible_card=VISIBLE_CARD
        )
    except Exception as error:  # noqa: BLE001 - DeviceUnavailable or a refused setting: nothing runs
        print(f"tracy profile: the tt_silicon executor refused the cards: {type(error).__name__}: {error}", flush=True)
        return STOPPED
    base = runs_root / "tracy" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    (base / "report").mkdir(parents=True)
    (base / "work").mkdir()
    world.build_root = base / "work"
    record = provenance(world.card, world.cpu, args.candidates.resolve(), args)
    (base / "report" / "provenance.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"tracy profile: {len(steps)} steps; report {base / 'report'}; device {record['device']}", flush=True)
    start = time.monotonic()
    code = run_steps(steps, world, base / "report", base / "work", deadline=start + args.minutes * 60,
                     pause_s=args.pause, budget_s=args.budget, clock=time.monotonic, sleep=time.sleep)
    print(f"tracy profile: exit {code}; report {base / 'report'}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
