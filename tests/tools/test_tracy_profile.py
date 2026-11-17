"""Tests for task P17.65: plans/runs/tracy/profile.py, the device profiles of the saved TT translations.

Bible: Execution Backends (tt_silicon), Result Record (DeviceTiming),
Oracles (binary_io), Evaluation Protocol (Reporting Rules), Agent Rules 1,
2, 6, 9, and 17. plans/p17-portable.md, task P17.65.

The contract these tests fix:

1. read_candidates reads plans/runs/tracy/candidates.tsv: comment and blank
   lines skipped, the header exactly COLUMNS, source cpp, a target in
   TT_TARGETS, a 10-digit lowercase hex files_sha1, a relative trial path
   ending in /trial.json with no `..` segment, unique positive orders; the
   rows come back sorted by order, and anything else is a ValueError naming
   the line. The committed file holds 41 rows.
2. files_digest is the first 10 hex digits of the sha1 of the files' JSON
   with sorted keys; candidate_files returns a trial's final attempt files,
   or raises CandidateError unless that attempt reached S5 with alignment
   1.0, its host-compute guard is not True, and its digest is the row's.
3. plan_steps puts the qualification first (the C++ program and the tracked
   TT reference of tt-pairs-v0 loopback), then, in candidate order, before
   each item's first candidate its C++ program and, when the item has a TT
   reference, that reference, then the candidate.
4. run_steps runs each step through a World and appends one JSON record and
   one summary row per step as it ends. It starts no step once fewer than
   its budget of seconds are left before the deadline (DEADLINE), stops
   after any exception from a card run or any quarantine marker (STOPPED),
   and stops when a qualification step (its C++ program or its reference,
   never a translation of the same item) did not run and pass or the
   reference gave no device timing (QUALIFY_FAILED), and pauses between
   card runs. A reference's device timing is recorded as not
   self-reported; a candidate's stays self-reported. A program identical
   to one already run is recorded as
   such and not run again. A step whose C++ program did not run cleanly is
   skipped. Each card run is judged against the C++ outputs, and a
   candidate also against the TT reference's outputs when it ran.

Every program, file set, output, timing, marker, wall time, and clock value
here is SYNTHETIC: the fake world compiles nothing, starts no process, and
opens no device. No value here is a measurement, and nothing here is a
simulator or silicon result.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lassi.core.interfaces import RunResult
from lassi.core.record import DeviceProgram, DeviceTiming

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "plans" / "runs" / "tracy" / "profile.py"
CANDIDATES = REPO / "plans" / "runs" / "tracy" / "candidates.tsv"

COLUMNS = ("order", "suite", "item", "source", "target", "model", "sweep", "files_sha1", "trial")
TT_TARGETS = ("tt", "ttkh")
QUALIFY = ("tt-pairs-v0", "loopback", "tt")
DONE, DEADLINE, STOPPED, QUALIFY_FAILED = 0, 2, 3, 4
SUMMARY_COLUMNS = ("step", "kind", "suite", "item", "model", "status", "passed", "exit_code", "wall_s", "kernel_ns",
                   "programs", "aiclk_mhz", "self_reported", "note")
RECORD_KEYS = ("step", "kind", "suite", "item", "status", "reason", "passed", "same_as", "device_timing")
HEADER = "\t".join(COLUMNS)
# The fake world's names of the qualification's two steps (label below).
Q_CPP, Q_REF = f"cpp/{QUALIFY[0]}/{QUALIFY[1]}/0", f"reference/{QUALIFY[0]}/{QUALIFY[1]}/0"

# SYNTHETIC suites and items of the fake world (a BALE-like item with no TT reference, a pairs-like one with one).
BALE, TRANSPOSE, RANDPERM = "syn-bale", "syn_transpose", "syn_randperm"
PAIRS, ELTWISE = "syn-pairs", "syn_eltwise"
KH = "tt-pairs-v0-kh"
MODEL, SWEEP = "SYNTHETIC-model", "SYNTHETIC-sweep"
# SYNTHETIC program files, harness, and outputs.
FILES_A = {"host.cpp": "SYNTHETIC host program A", "kernels/compute.cpp": "SYNTHETIC kernel A"}
FILES_B = {"host.cpp": "SYNTHETIC host program B"}
HARNESS = {"harness.cpp": "SYNTHETIC harness"}
EXPECTED = b"SYNTHETIC expected output"
WRONG = b"SYNTHETIC wrong output"
REFERENCE_OUT = b"SYNTHETIC reference output"
OUT = "out.bin"
# SYNTHETIC pause, budget, and deadline values for the fake clock (seconds); no value is a measurement.
PAUSE, BUDGET, FAR = 7.5, 60.0, 1.0e9


def digest(files: Mapping[str, str]) -> str:
    """Return the contract's files digest: the first 10 hex digits of sha1 of the sorted-key JSON."""
    return hashlib.sha1(json.dumps(files, sort_keys=True).encode()).hexdigest()[:10]


def timing(*kernel_ns: int, self_reported: bool = True) -> DeviceTiming:
    """Return a SYNTHETIC DeviceTiming with one program per kernel_ns value, as the executor gives it."""
    programs = [DeviceProgram(device_id=0, op_name=f"SYNTHETIC-op-{index}", call_count=index + 1, cores=index + 1,
                              kernel_cycles=ns, kernel_ns=ns, fw_ns=ns + 100)
                for index, ns in enumerate(kernel_ns)]
    return DeviceTiming(source="tt-metal device profiler", build="SYNTHETIC-build", arch="SYNTHETIC-arch",
                        aiclk_mhz=1000, programs=programs, self_reported=self_reported)


def load_script() -> ModuleType:
    """Load plans/runs/tracy/profile.py as a module; fail clearly while it does not exist."""
    assert SCRIPT.is_file(), "plans/runs/tracy/profile.py is missing"
    spec = importlib.util.spec_from_file_location("tracy_profile_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tracy() -> ModuleType:
    """Return the script loaded as a module."""
    return load_script()


# ---------------------------------------------------------------------------
# Constants and types


def test_the_constants_are_the_contracts(tracy: ModuleType) -> None:
    assert tuple(tracy.COLUMNS) == COLUMNS
    assert tuple(tracy.TT_TARGETS) == TT_TARGETS
    assert tuple(tracy.QUALIFY) == QUALIFY
    assert (tracy.DONE, tracy.DEADLINE, tracy.STOPPED, tracy.QUALIFY_FAILED) == (DONE, DEADLINE, STOPPED,
                                                                                  QUALIFY_FAILED)
    assert tuple(tracy.SUMMARY_COLUMNS) == SUMMARY_COLUMNS


def test_every_card_run_uses_card_0_alone(tracy: ModuleType) -> None:
    """Task P17.66: the executor is built with visible_card 0, as the smoke ran (TT_VISIBLE_DEVICES=0)."""
    assert tracy.VISIBLE_CARD == 0
    assert "visible_card=VISIBLE_CARD" in Path(tracy.__file__).read_text(encoding="utf-8")


def test_the_types_are_the_contracts(tracy: ModuleType) -> None:
    row = tracy.Candidate(1, BALE, TRANSPOSE, "cpp", "ttkh", MODEL, SWEEP, "0123456789", "a/run01/trial.json")
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.order = 2
    step = tracy.Step("cpp", BALE, TRANSPOSE, "ttkh")
    assert step.candidate is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        step.kind = "candidate"
    assert issubclass(tracy.CandidateError, ValueError)
    assert issubclass(tracy.BuildError, Exception)
    assert str(tracy.BuildError("SYNTHETIC compiler tail")) == "SYNTHETIC compiler tail"


# ---------------------------------------------------------------------------
# read_candidates (contract point 1)


def row(**fields: str) -> str:
    """Return one SYNTHETIC tab-separated candidates row, valid unless `fields` overrides a column."""
    values = {"order": "1", "suite": BALE, "item": TRANSPOSE, "source": "cpp", "target": "ttkh", "model": MODEL,
              "sweep": SWEEP, "files_sha1": "0123456789",
              "trial": f"{SWEEP}/arm/{MODEL}/{BALE}/cpp-ttkh/{TRANSPOSE}/run01/trial.json"}
    values.update(fields)
    return "\t".join(values[name] for name in COLUMNS)


def write_lines(tmp_path: Path, lines: Sequence[str]) -> Path:
    """Write `lines` as a candidates file under tmp_path and return its path."""
    path = tmp_path / "candidates.tsv"
    path.write_text("".join(f"{line}\n" for line in lines), encoding="ascii")
    return path


def test_read_candidates_skips_comments_and_blank_lines_and_sorts_by_order(tracy: ModuleType,
                                                                           tmp_path: Path) -> None:
    path = write_lines(tmp_path, ["# SYNTHETIC candidates", "", HEADER,
                                  row(order="2", item=RANDPERM, trial="s/a/run02/trial.json"),
                                  "# SYNTHETIC comment between rows", "", row(order="1")])
    rows = tracy.read_candidates(path)
    assert [candidate.order for candidate in rows] == [1, 2]
    assert rows[0] == tracy.Candidate(1, BALE, TRANSPOSE, "cpp", "ttkh", MODEL, SWEEP, "0123456789",
                                      f"{SWEEP}/arm/{MODEL}/{BALE}/cpp-ttkh/{TRANSPOSE}/run01/trial.json")
    assert rows[1].item == RANDPERM
    assert rows[1].trial == "s/a/run02/trial.json"


@pytest.mark.parametrize("header", [
    "\t".join(reversed(COLUMNS)),
    "\t".join(COLUMNS[:-1]),
    "\t".join((*COLUMNS, "extra")),
    ",".join(COLUMNS),
    row(),
], ids=["reordered", "short", "extra-column", "commas", "a-row-first"])
def test_read_candidates_refuses_a_header_other_than_the_columns(tracy: ModuleType, tmp_path: Path,
                                                                 header: str) -> None:
    path = write_lines(tmp_path, ["# SYNTHETIC", "", header, row()])
    with pytest.raises(ValueError, match=r"\bline 3\b"):
        tracy.read_candidates(path)


def second(**fields: str) -> str:
    """Return a SYNTHETIC row of order 2 (so only the overridden column is wrong), with `fields` overriding."""
    return row(**{"order": "2", **fields})


BAD_ROWS = {
    "eight-fields": "\t".join(second().split("\t")[:-1]),
    "ten-fields": second() + "\textra",
    "order-zero": second(order="0"),
    "order-negative": second(order="-1"),
    "order-word": second(order="two"),
    "order-float": second(order="2.0"),
    "source-c": second(source="c"),
    "source-upper": second(source="CPP"),
    "target-omp": second(target="omp"),
    "target-cpp": second(target="cpp"),
    "target-upper": second(target="TT"),
    "sha-upper": second(files_sha1="44E24BF62C"),
    "sha-short": second(files_sha1="44e24bf62"),
    "sha-long": second(files_sha1="44e24bf62c0"),
    "sha-not-hex": second(files_sha1="44e24bf62g"),
    "trial-absolute": second(trial="/runs/a/run01/trial.json"),
    "trial-dotdot": second(trial="a/../b/run01/trial.json"),
    "trial-leading-dotdot": second(trial="../a/run01/trial.json"),
    "trial-other-name": second(trial="a/run01/result.json"),
    "trial-suffix-only": second(trial="a/run01/mytrial.json"),
    "trial-bare": second(trial="trial.json"),
}


@pytest.mark.parametrize("bad", list(BAD_ROWS.values()), ids=list(BAD_ROWS))
def test_read_candidates_refuses_a_bad_row_naming_its_line(tracy: ModuleType, tmp_path: Path, bad: str) -> None:
    path = write_lines(tmp_path, ["# SYNTHETIC", HEADER, row(order="1"), "", bad])
    with pytest.raises(ValueError, match=r"\bline 5\b"):
        tracy.read_candidates(path)


def test_read_candidates_refuses_a_repeated_order_naming_the_second_line(tracy: ModuleType, tmp_path: Path) -> None:
    path = write_lines(tmp_path, [HEADER, row(order="7"), row(order="7", item=RANDPERM)])
    with pytest.raises(ValueError, match=r"\bline 3\b"):
        tracy.read_candidates(path)


def test_the_committed_candidates_file_reads(tracy: ModuleType) -> None:
    rows = tracy.read_candidates(CANDIDATES)
    assert len(rows) == 41
    assert [candidate.order for candidate in rows] == list(range(1, 42))
    assert {candidate.target for candidate in rows} <= set(TT_TARGETS)
    assert {candidate.source for candidate in rows} == {"cpp"}
    assert all(candidate.model != "mock-reference" for candidate in rows)
    assert [(rows[0].suite, rows[0].item), (rows[1].suite, rows[1].item)] == [("bale-v0-tt", "transpose_matrix"),
                                                                              ("bale-v0-tt", "randperm")]


# ---------------------------------------------------------------------------
# files_digest and candidate_files (contract point 2)

MISSING = object()


def attempt(files: Mapping[str, str], *, stage: str = "S5", alignment: Any = 1.0,
            guards: Any = MISSING) -> dict[str, Any]:
    """Return a SYNTHETIC attempt as trial.json holds one, with only the fields candidate_files reads."""
    record: dict[str, Any] = {"stage_reached": stage, "alignment": alignment, "files": dict(files)}
    if guards is not MISSING:
        record["guards"] = guards
    return {"index": 0, **record}


def test_files_digest_is_the_sorted_json_sha1_prefix(tracy: ModuleType) -> None:
    assert tracy.files_digest(FILES_A) == digest(FILES_A)
    assert re.fullmatch(r"[0-9a-f]{10}", tracy.files_digest(FILES_A))
    assert tracy.files_digest(dict(reversed(list(FILES_A.items())))) == tracy.files_digest(FILES_A)
    assert tracy.files_digest(FILES_A) != tracy.files_digest(FILES_B)


@pytest.mark.parametrize("final", [
    attempt(FILES_A),
    attempt(FILES_A, alignment={"mean": 1.0, "per_input": [1.0, 1.0]}),
    attempt(FILES_A, guards=None),
    attempt(FILES_A, guards={"host_compute": False, "harness_tamper": None}),
    attempt(FILES_A, guards={"host_compute": None, "host_compute_not_checked": "not-built"}),
], ids=["number", "mean-dict", "guards-null", "host-compute-false", "host-compute-null"])
def test_candidate_files_returns_the_final_attempts_files(tracy: ModuleType, final: dict[str, Any]) -> None:
    trial = {"attempts": [attempt(FILES_B, stage="S2", alignment=None), final]}
    assert tracy.candidate_files(trial, digest(FILES_A)) == FILES_A


REFUSED_TRIALS = {
    "no-attempts": {"attempts": []},
    "stage-s4": {"attempts": [attempt(FILES_A, stage="S4")]},
    "alignment-half": {"attempts": [attempt(FILES_A, alignment=0.5)]},
    "alignment-mean-half": {"attempts": [attempt(FILES_A, alignment={"mean": 0.5, "per_input": [0.5]})]},
    "alignment-null": {"attempts": [attempt(FILES_A, alignment=None)]},
    "alignment-mean-null": {"attempts": [attempt(FILES_A, alignment={"mean": None, "per_input": []})]},
    "host-compute-true": {"attempts": [attempt(FILES_A, guards={"host_compute": True})]},
    "other-files": {"attempts": [attempt(FILES_B)]},
    "earlier-attempt-only": {"attempts": [attempt(FILES_A), attempt(FILES_A, stage="S4")]},
}


@pytest.mark.parametrize("trial", list(REFUSED_TRIALS.values()), ids=list(REFUSED_TRIALS))
def test_candidate_files_refuses_a_trial_that_is_not_a_saved_exact_match(tracy: ModuleType,
                                                                         trial: dict[str, Any]) -> None:
    with pytest.raises(tracy.CandidateError):
        tracy.candidate_files(trial, digest(FILES_A))


# ---------------------------------------------------------------------------
# plan_steps (contract point 3)


def candidate(tracy: ModuleType, order: int, suite: str = BALE, item: str = TRANSPOSE,
              target: str = "ttkh") -> Any:
    """Return a SYNTHETIC Candidate for `order` on (suite, item)."""
    return tracy.Candidate(order, suite, item, "cpp", target, MODEL, SWEEP, "0123456789",
                           f"{SWEEP}/{suite}/{item}/run{order:02d}/trial.json")


def qualification(tracy: ModuleType) -> list[Any]:
    """Return the two qualification steps the plan starts with."""
    return [tracy.Step("cpp", *QUALIFY), tracy.Step("reference", *QUALIFY)]


def test_plan_steps_with_no_candidates_is_the_qualification(tracy: ModuleType) -> None:
    assert tracy.plan_steps([], lambda suite, item, target: True) == qualification(tracy)


def test_plan_steps_puts_each_items_programs_before_its_first_candidate(tracy: ModuleType) -> None:
    rows = [candidate(tracy, 1), candidate(tracy, 2, item=RANDPERM), candidate(tracy, 3),
            candidate(tracy, 4, *QUALIFY[:2], target="tt"), candidate(tracy, 5, "tt-pairs-v0", ELTWISE, "tt")]
    asked: list[tuple[str, str, str]] = []

    def has_reference(suite: str, item: str, target: str) -> bool:
        """Answer True for the SYNTHETIC pairs suite only, recording each question."""
        asked.append((suite, item, target))
        return suite == "tt-pairs-v0"

    step = tracy.Step
    assert tracy.plan_steps(rows, has_reference) == [
        *qualification(tracy),
        step("cpp", BALE, TRANSPOSE, "ttkh"), step("candidate", BALE, TRANSPOSE, "ttkh", rows[0]),
        step("cpp", BALE, RANDPERM, "ttkh"), step("candidate", BALE, RANDPERM, "ttkh", rows[1]),
        step("candidate", BALE, TRANSPOSE, "ttkh", rows[2]),
        step("candidate", *QUALIFY, rows[3]),
        step("cpp", "tt-pairs-v0", ELTWISE, "tt"), step("reference", "tt-pairs-v0", ELTWISE, "tt"),
        step("candidate", "tt-pairs-v0", ELTWISE, "tt", rows[4]),
    ]
    assert ("tt-pairs-v0", ELTWISE, "tt") in asked


def test_plan_steps_keeps_the_given_order_and_tells_suites_apart(tracy: ModuleType) -> None:
    rows = [candidate(tracy, 9, KH, "loopback", "ttkh"), candidate(tracy, 2, item=RANDPERM)]
    step = tracy.Step
    assert tracy.plan_steps(rows, lambda suite, item, target: suite == KH) == [
        *qualification(tracy),
        step("cpp", KH, "loopback", "ttkh"), step("reference", KH, "loopback", "ttkh"),
        step("candidate", KH, "loopback", "ttkh", rows[0]),
        step("cpp", BALE, RANDPERM, "ttkh"), step("candidate", BALE, RANDPERM, "ttkh", rows[1]),
    ]


# ---------------------------------------------------------------------------
# The fake world (contract point 4)


@dataclass(frozen=True)
class Stat:
    """A SYNTHETIC oracle statistic for one output, with the attributes run_steps reads."""

    name: str
    passed: bool
    max_abs: float | None
    pcc: float | None
    note: str


@dataclass
class Script:
    """The SYNTHETIC behavior the fake world gives one step; files and language None mean the step's default."""

    files: dict[str, str] | None = None
    harness: dict[str, str] = field(default_factory=lambda: dict(HARNESS))
    language: str | None = None
    program_error: str | None = None
    build_error: str | None = None
    card_error: str | None = None
    exit_code: int | None = 0
    hang: bool = False
    output: bytes | None = EXPECTED
    wall_s: float = 1.0
    timing: DeviceTiming | None = field(default_factory=lambda: timing(1000))
    markers: tuple[str, ...] = ()


def label(step: Any) -> str:
    """Return the fake world's name for `step`: kind/suite/item/order (0 when the step has no candidate)."""
    order = step.candidate.order if step.candidate is not None else 0
    return f"{step.kind}/{step.suite}/{step.item}/{order}"


class FakeClock:
    """A SYNTHETIC clock that moves only when the fake world runs or the fake sleep sleeps."""

    def __init__(self) -> None:
        """Start the clock at 0."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current SYNTHETIC time."""
        return self.now


class FakeWorld:
    """A SYNTHETIC World: scripted programs, builds, runs, compares, markers, and profile copies, all recorded."""

    def __init__(self, tracy: ModuleType, root: Path, report: Path, clock: FakeClock) -> None:
        """Keep the module's error types, a directory for outputs, the report directory, and the clock."""
        self.tracy, self.root, self.report, self.clock = tracy, root, report, clock
        self.scripts: dict[str, Script] = {}
        self.events: list[tuple[str, str]] = []
        self.builds: list[tuple[str, str, Path]] = []
        self.compares: list[tuple[str, dict[str, bytes], dict[str, bytes]]] = []
        self.profiles: list[tuple[Path, Path, list[str]]] = []
        self.card_artifacts: list[Path] = []
        self.snapshots: list[tuple[int, int]] = []
        self.programs: list[tuple[str, tuple[str, dict[str, str], dict[str, str]]]] = []
        self.artifacts: dict[Path, str] = {}
        self.last_card = ""
        self.runs = 0

    def script(self, name: str) -> Script:
        """Return the script for the step named `name`, the default one when the test set none."""
        return self.scripts.setdefault(name, Script())

    def program(self, step: Any) -> tuple[str, dict[str, str], dict[str, str]]:
        """Return the step's SYNTHETIC language, files, and harness, or raise its scripted CandidateError."""
        name = label(step)
        self.events.append(("program", name))
        script = self.script(name)
        if script.program_error is not None:
            raise self.tracy.CandidateError(script.program_error)
        default_language = "cpp" if step.kind == "cpp" else step.target
        files = script.files if script.files is not None else {"main.cpp": f"SYNTHETIC {name}"}
        given = (script.language or default_language, dict(files), dict(script.harness))
        self.programs.append((name, given))
        return given[0], dict(given[1]), dict(given[2])

    def owner(self, language: str, files: Mapping[str, str], harness: Mapping[str, str]) -> str:
        """Return the step whose program() last gave this language, files, and harness."""
        for name, given in reversed(self.programs):
            if given == (language, dict(files), dict(harness)):
                return name
        raise AssertionError("fake world: build was given a program that program() never returned")

    def build(self, language: str, files: Mapping[str, str], harness: Mapping[str, str], workdir: Path) -> Path:
        """Record the build and return a SYNTHETIC artifact, or raise the scripted BuildError."""
        name = self.owner(language, files, harness)
        self.events.append(("build", name))
        self.builds.append((name, language, workdir))
        script = self.script(name)
        if script.build_error is not None:
            raise self.tracy.BuildError(script.build_error)
        workdir.mkdir(parents=True, exist_ok=True)
        artifact = workdir / f"artifact-{len(self.builds)}"
        artifact.write_text("SYNTHETIC artifact\n", encoding="ascii")
        self.artifacts[artifact] = name
        return artifact

    def inputs(self, step: Any, artifact: Path) -> list[str]:
        """Return SYNTHETIC input arguments for the step."""
        return [f"SYNTHETIC-input-{label(step)}"]

    def _run(self, artifact: Path, kind: str) -> RunResult:
        """Write the scripted output, move the clock, and return the scripted RunResult."""
        name = self.artifacts[artifact]
        script = self.script(name)
        self.events.append((kind, name))
        self.runs += 1
        self.clock.now += script.wall_s
        outputs: dict[str, Path] = {}
        if script.output is not None:
            directory = self.root / f"run-{self.runs}"
            directory.mkdir(parents=True)
            (directory / OUT).write_bytes(script.output)
            outputs[OUT] = directory / OUT
        return RunResult(exit_code=script.exit_code, hang=script.hang, stdout="SYNTHETIC stdout\n", stderr="",
                         output_files=outputs, wall_s=script.wall_s,
                         device_timing=script.timing if kind == "run_card" else None)

    def run_cpu(self, artifact: Path, inputs: Sequence[str]) -> RunResult:
        """Run the SYNTHETIC artifact on the fake host CPU."""
        return self._run(artifact, "run_cpu")

    def run_card(self, artifact: Path, inputs: Sequence[str]) -> RunResult:
        """Run the SYNTHETIC artifact on the fake card, raising the scripted card error, and snapshot the report."""
        self.snapshots.append((count_lines(self.report / "records.jsonl"), count_lines(self.report / "summary.tsv")))
        self.card_artifacts.append(artifact)
        self.last_card = self.artifacts[artifact]
        script = self.script(self.last_card)
        if script.card_error is not None:
            self.events.append(("run_card", self.last_card))
            raise RuntimeError(script.card_error)
        return self._run(artifact, "run_card")

    def compare(self, step: Any, expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[Stat]:
        """Compare output bytes per expected output, recording the call; no expected output gives no stat."""
        want = {name: as_bytes(value) for name, value in expected.items()}
        got = {name: as_bytes(value) for name, value in actual.items()}
        self.compares.append((label(step), want, got))
        stats = []
        for name, data in want.items():
            same = got.get(name) == data
            stats.append(Stat(name, same, 0.0 if same else 1.0, 1.0 if same else 0.0, "SYNTHETIC"))
        return stats

    def quarantined(self) -> list[str]:
        """Return the scripted markers of the last card run."""
        self.events.append(("quarantined", self.last_card))
        return list(self.script(self.last_card).markers) if self.last_card else []

    def keep_profile(self, artifact: Path, dest: Path) -> list[str]:
        """Record the copy and return SYNTHETIC profiler file names."""
        kept = [f"SYNTHETIC-profile-{len(self.profiles) + 1}.csv"]
        self.profiles.append((artifact, dest, kept))
        return kept


def as_bytes(value: Any) -> bytes:
    """Return the bytes of an output given as bytes or as a path."""
    return value if isinstance(value, bytes) else Path(value).read_bytes()


def count_lines(path: Path) -> int:
    """Return the number of lines in `path`, 0 when it does not exist."""
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


class FakeSleep:
    """A SYNTHETIC sleep that records each pause in the world's events and moves the clock."""

    def __init__(self, world: FakeWorld, clock: FakeClock) -> None:
        """Keep the world whose events record the pauses, and the clock to move."""
        self.world, self.clock = world, clock
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        """Record the pause and move the clock by it."""
        self.calls.append(seconds)
        self.world.events.append(("sleep", str(seconds)))
        self.clock.now += seconds


@dataclass
class Rig:
    """A fake world, clock, and sleep wired together for run_steps, with readers of what it wrote."""

    tracy: ModuleType
    root: Path
    world: FakeWorld
    clock: FakeClock
    sleep: FakeSleep

    @property
    def report(self) -> Path:
        """Return the report directory run_steps writes (missing until it runs)."""
        return self.root / "out" / "report"

    @property
    def work(self) -> Path:
        """Return the work directory run_steps builds under."""
        return self.root / "out" / "work"

    def run(self, steps: Sequence[Any], *, deadline: float = FAR, budget_s: float = BUDGET) -> int:
        """Run `steps` through run_steps and return its exit code."""
        return self.tracy.run_steps(steps, self.world, self.report, self.work, deadline=deadline, pause_s=PAUSE,
                                    budget_s=budget_s, clock=self.clock, sleep=self.sleep)

    def records(self) -> list[dict[str, Any]]:
        """Return the records of report/records.jsonl in file order."""
        text = (self.report / "records.jsonl").read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines()]

    def summary(self) -> list[list[str]]:
        """Return the rows of report/summary.tsv, its header first."""
        return [line.split("\t") for line in (self.report / "summary.tsv").read_text(encoding="utf-8").splitlines()]

    def ran(self, kind: str) -> list[str]:
        """Return the names of the steps for which the world saw an event of `kind`, in order."""
        return [name for event, name in self.world.events if event == kind]


@pytest.fixture
def rig(tracy: ModuleType, tmp_path: Path) -> Rig:
    """Return a fresh rig whose report directory does not exist yet."""
    clock = FakeClock()
    world = FakeWorld(tracy, tmp_path / "world", tmp_path / "out" / "report", clock)
    return Rig(tracy, tmp_path, world, clock, FakeSleep(world, clock))


def item_steps(rig: Rig, suite: str, item: str, target: str, *orders: int, reference: bool = False) -> list[Any]:
    """Return an item's C++ step, its reference step when asked, and one candidate step per order."""
    step = rig.tracy.Step
    steps = [step("cpp", suite, item, target)]
    if reference:
        steps.append(step("reference", suite, item, target))
    steps += [step("candidate", suite, item, target, candidate(rig.tracy, order, suite, item, target))
              for order in orders]
    return steps


def stops(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return the stop records."""
    return [record for record in records if record["kind"] == "stop"]


# ---------------------------------------------------------------------------
# run_steps: records, summary, and the host CPU


def test_a_clean_run_records_every_step_and_returns_done(rig: Rig) -> None:
    steps = [*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1)]
    assert not rig.report.exists()
    assert rig.run(steps) == DONE
    records = rig.records()
    assert [record["step"] for record in records] == [1, 2, 3, 4]
    assert [record["kind"] for record in records] == ["cpp", "reference", "cpp", "candidate"]
    assert [(record["suite"], record["item"]) for record in records] == [QUALIFY[:2], QUALIFY[:2],
                                                                         (BALE, TRANSPOSE), (BALE, TRANSPOSE)]
    for record in records:
        assert set(RECORD_KEYS) <= set(record), record
        assert (record["status"], record["passed"], record["same_as"]) == ("ran", True, None)
    assert stops(records) == []


def test_builds_happen_under_the_work_directory(rig: Rig) -> None:
    rig.run([*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1)])
    assert [language for _, language, _ in rig.world.builds] == ["cpp", "tt", "cpp", "ttkh"]
    for _, _, workdir in rig.world.builds:
        assert workdir.resolve().is_relative_to(rig.work.resolve()), workdir


def test_the_summary_has_the_header_and_one_row_per_record(rig: Rig) -> None:
    rig.world.scripts["candidate/syn-bale/syn_transpose/1"] = Script(timing=timing(1000, 2500))
    rig.run([*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1)])
    rows = rig.summary()
    assert rows[0] == list(SUMMARY_COLUMNS)
    assert len(rows) == len(rig.records()) + 1
    assert all(len(values) == len(SUMMARY_COLUMNS) for values in rows[1:])
    table = [dict(zip(SUMMARY_COLUMNS, values, strict=True)) for values in rows[1:]]
    assert [(entry["step"], entry["kind"], entry["status"]) for entry in table] == [
        ("1", "cpp", "ran"), ("2", "reference", "ran"), ("3", "cpp", "ran"), ("4", "candidate", "ran")]
    assert (table[3]["suite"], table[3]["item"], table[3]["model"]) == (BALE, TRANSPOSE, MODEL)
    assert table[3]["kernel_ns"] == "3500"


def test_records_and_summary_rows_are_appended_as_each_step_ends(rig: Rig) -> None:
    rig.run([*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1)])
    # At the card run of step 2, step 1 is written; at the card run of step 4, steps 1 to 3 are.
    assert rig.world.snapshots == [(1, 2), (3, 4)]


def test_cpp_steps_run_on_the_host_cpu_and_never_on_the_cards(rig: Rig) -> None:
    rig.run([*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1)])
    assert rig.ran("run_cpu") == [Q_CPP, "cpp/syn-bale/syn_transpose/0"]
    assert rig.ran("run_card") == [Q_REF, "candidate/syn-bale/syn_transpose/1"]
    cpp_records = [record for record in rig.records() if record["kind"] == "cpp"]
    assert [record["device_timing"] for record in cpp_records] == [None, None]


@pytest.mark.parametrize(("exit_code", "hang"), [(1, False), (None, True), (0, True)],
                         ids=["exit-1", "hang", "exit-0-hang"])
def test_a_cpp_program_that_did_not_run_cleanly_does_not_pass(rig: Rig, exit_code: int | None, hang: bool) -> None:
    rig.world.scripts["cpp/syn-bale/syn_transpose/0"] = Script(exit_code=exit_code, hang=hang)
    assert rig.run([*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1)]) == DONE
    record = rig.records()[2]
    assert (record["kind"], record["status"], record["passed"]) == ("cpp", "ran", False)


def test_steps_after_a_failed_cpp_program_are_skipped(rig: Rig) -> None:
    rig.world.scripts[f"cpp/{PAIRS}/{ELTWISE}/0"] = Script(exit_code=1)
    steps = [*qualification(rig.tracy), *item_steps(rig, PAIRS, ELTWISE, "tt", 3, reference=True)]
    assert rig.run(steps) == DONE
    records = rig.records()
    for record in records[3:]:
        assert record["status"] == "skipped", record
        assert "C++" in record["reason"], record
    skipped = {f"reference/{PAIRS}/{ELTWISE}/0", f"candidate/{PAIRS}/{ELTWISE}/3"}
    assert not skipped & set(rig.ran("build") + rig.ran("run_card") + rig.ran("run_cpu"))


# ---------------------------------------------------------------------------
# run_steps: judging card runs


CARD_FAILURES = {
    "exit-1": Script(exit_code=1),
    "hang": Script(exit_code=None, hang=True),
    "wrong-output": Script(output=WRONG),
    "no-output": Script(output=None),
}


@pytest.mark.parametrize("script", list(CARD_FAILURES.values()), ids=list(CARD_FAILURES))
def test_a_card_run_that_fails_its_compare_does_not_pass(rig: Rig, script: Script) -> None:
    rig.world.scripts["candidate/syn-bale/syn_transpose/1"] = script
    assert rig.run([*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1)]) == DONE
    record = rig.records()[3]
    assert (record["kind"], record["status"], record["passed"]) == ("candidate", "ran", False)


def test_a_card_run_with_no_cpp_output_to_compare_does_not_pass(rig: Rig) -> None:
    rig.world.scripts["cpp/syn-bale/syn_transpose/0"] = Script(output=None)
    assert rig.run([*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1)]) == DONE
    cpp, card = rig.records()[2:4]
    assert cpp["passed"] is True
    assert (card["status"], card["passed"]) == ("ran", False)


def test_a_card_run_is_compared_against_its_items_cpp_outputs(rig: Rig) -> None:
    rig.world.scripts["cpp/syn-bale/syn_transpose/0"] = Script(output=REFERENCE_OUT)
    rig.world.scripts["candidate/syn-bale/syn_transpose/1"] = Script(output=REFERENCE_OUT)
    rig.run([*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1)])
    calls = [(want, got) for name, want, got in rig.world.compares if name == "candidate/syn-bale/syn_transpose/1"]
    assert calls == [({OUT: REFERENCE_OUT}, {OUT: REFERENCE_OUT})]
    assert rig.records()[3]["passed"] is True


@pytest.mark.parametrize(("output", "passed", "vs_reference"), [(EXPECTED, True, False), (REFERENCE_OUT, False, True)],
                         ids=["matches-cpp", "matches-reference"])
def test_a_candidate_is_also_judged_against_the_reference_that_ran(rig: Rig, output: bytes, passed: bool,
                                                                   vs_reference: bool) -> None:
    rig.world.scripts[f"reference/{PAIRS}/{ELTWISE}/0"] = Script(output=REFERENCE_OUT)
    rig.world.scripts[f"candidate/{PAIRS}/{ELTWISE}/3"] = Script(output=output)
    steps = [*qualification(rig.tracy), *item_steps(rig, PAIRS, ELTWISE, "tt", 3, reference=True)]
    assert rig.run(steps) == DONE
    record = rig.records()[4]
    assert (record["kind"], record["status"], record["passed"]) == ("candidate", "ran", passed)
    assert [stat["passed"] for stat in record["vs_reference"]] == [vs_reference]
    wants = [want for name, want, _ in rig.world.compares if name == f"candidate/{PAIRS}/{ELTWISE}/3"]
    assert {OUT: REFERENCE_OUT} in wants and {OUT: EXPECTED} in wants


def test_a_candidate_without_a_reference_holds_no_vs_reference(rig: Rig) -> None:
    rig.run([*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1)])
    record = rig.records()[3]
    assert record["kind"] == "candidate"
    assert record["vs_reference"] is None


# ---------------------------------------------------------------------------
# run_steps: device timing and profiler files


def test_a_references_timing_is_not_self_reported_and_a_candidates_is(rig: Rig) -> None:
    rig.run([*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1)])
    reference, candidate_record = rig.records()[1], rig.records()[3]
    assert reference["device_timing"]["self_reported"] is False
    assert candidate_record["device_timing"]["self_reported"] is True


def test_the_device_timing_record_holds_the_programs_and_their_total(rig: Rig) -> None:
    rig.world.scripts["candidate/syn-bale/syn_transpose/1"] = Script(timing=timing(1000, 2500))
    rig.run([*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1)])
    recorded = rig.records()[3]["device_timing"]
    expected = {"source": "tt-metal device profiler", "build": "SYNTHETIC-build", "arch": "SYNTHETIC-arch",
                "aiclk_mhz": 1000, "self_reported": True, "kernel_ns_total": 3500}
    assert {key: recorded[key] for key in expected} == expected
    program_keys = ("device_id", "op_name", "call_count", "cores", "kernel_cycles", "kernel_ns", "fw_ns")
    assert [{key: program[key] for key in program_keys} for program in recorded["programs"]] == [
        {"device_id": 0, "op_name": "SYNTHETIC-op-0", "call_count": 1, "cores": 1, "kernel_cycles": 1000,
         "kernel_ns": 1000, "fw_ns": 1100},
        {"device_id": 0, "op_name": "SYNTHETIC-op-1", "call_count": 2, "cores": 2, "kernel_cycles": 2500,
         "kernel_ns": 2500, "fw_ns": 2600},
    ]


def test_a_card_run_with_no_timing_records_none(rig: Rig) -> None:
    rig.world.scripts["candidate/syn-bale/syn_transpose/1"] = Script(timing=None)
    rig.run([*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1)])
    assert rig.records()[3]["device_timing"] is None


def test_each_card_runs_profiler_files_are_kept_under_the_report(rig: Rig) -> None:
    rig.run([*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1)])
    assert [artifact for artifact, _, _ in rig.world.profiles] == rig.world.card_artifacts
    destinations = [dest for _, dest, _ in rig.world.profiles]
    assert len(set(destinations)) == 2
    for dest in destinations:
        assert dest.resolve().is_relative_to((rig.report / "profiler").resolve()), dest
    records = rig.records()
    assert [records[1]["profile"], records[3]["profile"]] == [kept for _, _, kept in rig.world.profiles]


# ---------------------------------------------------------------------------
# run_steps: programs already run


def test_an_identical_program_is_recorded_as_such_and_not_run_again(rig: Rig) -> None:
    for order in (1, 2):
        rig.world.scripts[f"candidate/syn-bale/syn_transpose/{order}"] = Script(files=dict(FILES_A))
    assert rig.run([*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1, 2)]) == DONE
    record = rig.records()[4]
    assert (record["step"], record["status"], record["same_as"]) == (5, "skipped", 4)
    assert "candidate/syn-bale/syn_transpose/2" not in rig.ran("build") + rig.ran("run_card")


SAME_FILES_NOT_SKIPPED = {
    "other-item": (Script(files=dict(FILES_A)), RANDPERM),
    "other-harness": (Script(files=dict(FILES_A), harness={"harness.cpp": "SYNTHETIC other harness"}), TRANSPOSE),
    "other-language": (Script(files=dict(FILES_A), language="tt"), TRANSPOSE),
}


@pytest.mark.parametrize(("later", "item"), list(SAME_FILES_NOT_SKIPPED.values()), ids=list(SAME_FILES_NOT_SKIPPED))
def test_the_same_files_under_another_key_run_again(rig: Rig, later: Script, item: str) -> None:
    rig.world.scripts["candidate/syn-bale/syn_transpose/1"] = Script(files=dict(FILES_A))
    rig.world.scripts[f"candidate/syn-bale/{item}/2"] = later
    if item == TRANSPOSE:
        steps = [*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1, 2)]
    else:
        steps = [*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1),
                 *item_steps(rig, BALE, item, "ttkh", 2)]
    assert rig.run(steps) == DONE
    record = rig.records()[-1]
    assert (record["kind"], record["status"], record["same_as"]) == ("candidate", "ran", None)
    assert f"candidate/syn-bale/{item}/2" in rig.ran("run_card")


def test_a_skipped_cpp_duplicate_provides_its_outputs_to_its_own_item(rig: Rig) -> None:
    qualify_files = {"main.cpp": f"SYNTHETIC {Q_CPP}"}
    rig.world.scripts[f"cpp/{KH}/loopback/0"] = Script(files=dict(qualify_files))
    steps = [*qualification(rig.tracy), *item_steps(rig, KH, "loopback", "ttkh", 3)]
    assert rig.run(steps) == DONE
    cpp, card = rig.records()[2:4]
    assert (cpp["status"], cpp["same_as"]) == ("skipped", 1)
    assert (card["status"], card["passed"]) == ("ran", True)
    assert f"cpp/{KH}/loopback/0" not in rig.ran("build") + rig.ran("run_cpu")
    wants = [want for name, want, _ in rig.world.compares if name == f"candidate/{KH}/loopback/3"]
    assert {OUT: EXPECTED} in wants


def test_a_skipped_reference_duplicate_provides_its_outputs_for_vs_reference(rig: Rig) -> None:
    rig.world.scripts[f"cpp/{KH}/loopback/0"] = Script(files={"main.cpp": f"SYNTHETIC {Q_CPP}"})
    rig.world.scripts[f"reference/{KH}/loopback/0"] = Script(files={"main.cpp": f"SYNTHETIC {Q_REF}"}, language="tt")
    rig.world.scripts[f"candidate/{KH}/loopback/3"] = Script(output=WRONG)
    steps = [*qualification(rig.tracy), *item_steps(rig, KH, "loopback", "ttkh", 3, reference=True)]
    assert rig.run(steps) == DONE
    reference, card = rig.records()[3:5]
    assert (reference["kind"], reference["status"], reference["same_as"]) == ("reference", "skipped", 2)
    assert rig.ran("run_card") == [Q_REF, f"candidate/{KH}/loopback/3"]
    assert card["status"] == "ran"
    assert [stat["passed"] for stat in card["vs_reference"]] == [False]


# ---------------------------------------------------------------------------
# run_steps: errors that let the run continue


def test_a_candidate_error_skips_the_step_and_the_run_continues(rig: Rig) -> None:
    rig.world.scripts["candidate/syn-bale/syn_transpose/1"] = Script(program_error="SYNTHETIC files changed")
    assert rig.run([*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1, 2)]) == DONE
    skipped, after = rig.records()[3:5]
    assert (skipped["status"], skipped["reason"]) == ("skipped", "SYNTHETIC files changed")
    assert (after["status"], after["passed"]) == ("ran", True)


def test_a_build_error_is_recorded_and_the_run_continues(rig: Rig) -> None:
    tail = "SYNTHETIC host.cpp:3:1: error: expected ';'\nSYNTHETIC\t1 error generated."
    rig.world.scripts["candidate/syn-bale/syn_transpose/1"] = Script(build_error=tail)
    assert rig.run([*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1, 2)]) == DONE
    failed, after = rig.records()[3:5]
    assert failed["status"] == "build-failed"
    assert "SYNTHETIC host.cpp:3:1: error: expected ';'" in failed["reason"]
    assert "candidate/syn-bale/syn_transpose/1" not in rig.ran("run_card")
    assert (after["status"], after["passed"]) == ("ran", True)
    assert len(rig.summary()) == len(rig.records()) + 1


# ---------------------------------------------------------------------------
# run_steps: stops


def test_an_exception_from_a_card_run_stops_the_run(rig: Rig) -> None:
    rig.world.scripts["candidate/syn-bale/syn_transpose/1"] = Script(card_error="SYNTHETIC device timeout")
    steps = [*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1, 2)]
    assert rig.run(steps) == STOPPED
    records = rig.records()
    assert [record["kind"] for record in records] == ["cpp", "reference", "cpp", "candidate", "stop"]
    assert records[3]["status"] == "error"
    assert "SYNTHETIC device timeout" in records[3]["reason"]
    assert (records[4]["step"], records[4]["status"]) == (None, "card-error")
    assert "candidate/syn-bale/syn_transpose/2" not in rig.ran("program")


def test_a_quarantine_marker_stops_the_run_after_the_steps_record(rig: Rig) -> None:
    markers = ("SYNTHETIC-marker-a", "SYNTHETIC-marker-b")
    rig.world.scripts["candidate/syn-bale/syn_transpose/1"] = Script(markers=markers)
    steps = [*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1, 2)]
    assert rig.run(steps) == STOPPED
    records = rig.records()
    assert [record["kind"] for record in records] == ["cpp", "reference", "cpp", "candidate", "stop"]
    assert records[3]["status"] == "ran"
    assert (records[4]["step"], records[4]["status"]) == (None, "quarantine")
    assert all(marker in records[4]["reason"] for marker in markers)
    assert "candidate/syn-bale/syn_transpose/2" not in rig.ran("program")


def test_the_quarantine_check_follows_every_card_run(rig: Rig) -> None:
    rig.run([*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1, 2)])
    kinds = [event for event, _ in rig.world.events if event in ("run_card", "quarantined")]
    assert kinds.count("run_card") == 3
    for index, event in enumerate(kinds):
        if event == "run_card":
            assert kinds[index + 1:index + 2] == ["quarantined"], kinds


@pytest.mark.parametrize("script", [Script(output=WRONG), Script(exit_code=1)], ids=["wrong-output", "exit-1"])
def test_a_failed_qualification_stops_the_run(rig: Rig, script: Script) -> None:
    rig.world.scripts[Q_REF] = script
    steps = [*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1)]
    assert rig.run(steps) == QUALIFY_FAILED
    records = rig.records()
    assert [record["kind"] for record in records] == ["cpp", "reference", "stop"]
    assert (records[1]["status"], records[1]["passed"]) == ("ran", False)
    assert (records[2]["step"], records[2]["status"]) == (None, "qualification-failed")
    assert rig.ran("program") == [Q_CPP, Q_REF]


@pytest.mark.parametrize(
    ("name", "script", "kinds"),
    [
        (Q_CPP, Script(build_error="SYNTHETIC compiler tail"), ["cpp", "stop"]),
        (Q_CPP, Script(exit_code=1), ["cpp", "stop"]),
        (Q_REF, Script(build_error="SYNTHETIC compiler tail"), ["cpp", "reference", "stop"]),
        (Q_REF, Script(timing=None), ["cpp", "reference", "stop"]),
    ],
    ids=["cpp-build-failed", "cpp-exit-1", "reference-build-failed", "reference-without-timing"],
)
def test_a_qualification_that_did_not_run_and_pass_stops_before_any_candidate(
    rig: Rig, name: str, script: Script, kinds: list[str]
) -> None:
    rig.world.scripts[name] = script
    steps = [*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1)]
    assert rig.run(steps) == QUALIFY_FAILED
    records = rig.records()
    assert [record["kind"] for record in records] == kinds
    assert (records[-1]["step"], records[-1]["status"]) == (None, "qualification-failed")
    assert not [step for step in rig.ran("program") if step.startswith("candidate/")]
    assert not [step for step in rig.ran("run_card") if step.startswith("candidate/")]


@pytest.mark.parametrize(
    ("script", "status"),
    [
        (Script(output=WRONG), "ran"),
        (Script(build_error="SYNTHETIC compiler tail"), "build-failed"),
        (Script(program_error="SYNTHETIC trial changed"), "skipped"),
    ],
    ids=["wrong-output", "build-failed", "refused"],
)
def test_a_failing_translation_of_the_qualification_item_is_judged_like_any_other(
    rig: Rig, script: Script, status: str
) -> None:
    # Two translations of the qualification's own item follow it, as plan_steps orders them (no second C++ step).
    translations = [step for step in item_steps(rig, *QUALIFY, 1, 2) if step.kind == "candidate"]
    steps = [*qualification(rig.tracy), *translations]
    rig.world.scripts[f"candidate/{QUALIFY[0]}/{QUALIFY[1]}/1"] = script
    assert rig.run(steps) == DONE
    records = rig.records()
    assert [record["kind"] for record in records] == ["cpp", "reference", "candidate", "candidate"]
    assert records[2]["status"] == status and records[2]["passed"] is not True
    assert (records[3]["status"], records[3]["passed"]) == ("ran", True)
    assert stops(records) == []


def test_a_step_with_fewer_than_its_budget_left_is_not_started(rig: Rig) -> None:
    rig.world.scripts[Q_CPP] = Script(wall_s=100.0)
    # SYNTHETIC: at step 2 the clock reads 100, and 100 + 50 passes the deadline of 120.
    assert rig.run(qualification(rig.tracy), deadline=120.0, budget_s=50.0) == DEADLINE
    records = rig.records()
    assert [record["kind"] for record in records] == ["cpp", "stop"]
    assert (records[1]["step"], records[1]["status"]) == (None, "deadline")
    assert rig.ran("program") == [Q_CPP]


def test_a_step_with_exactly_its_budget_left_still_runs(rig: Rig) -> None:
    rig.world.scripts[Q_CPP] = Script(wall_s=100.0)
    # SYNTHETIC: at step 2 the clock reads 100, and 100 + 50 is the deadline, not past it.
    assert rig.run(qualification(rig.tracy), deadline=150.0, budget_s=50.0) == DONE
    assert stops(rig.records()) == []


def test_a_deadline_before_the_first_step_runs_nothing(rig: Rig) -> None:
    assert rig.run(qualification(rig.tracy), deadline=10.0, budget_s=BUDGET) == DEADLINE
    records = rig.records()
    assert [(record["step"], record["kind"], record["status"]) for record in records] == [(None, "stop", "deadline")]
    assert rig.world.events == []


# ---------------------------------------------------------------------------
# run_steps: pauses


def test_card_runs_pause_between_them_and_not_before_the_first(rig: Rig) -> None:
    rig.run([*qualification(rig.tracy), *item_steps(rig, BALE, TRANSPOSE, "ttkh", 1, 2)])
    assert rig.sleep.calls == [PAUSE, PAUSE]
    order = [event for event, _ in rig.world.events if event in ("run_card", "sleep")]
    assert order == ["run_card", "sleep", "run_card", "sleep", "run_card"]


def test_runs_on_the_host_cpu_need_no_pause(rig: Rig) -> None:
    steps: list[Any] = [rig.tracy.Step("cpp", BALE, TRANSPOSE, "ttkh"), rig.tracy.Step("cpp", BALE, RANDPERM, "ttkh")]
    assert rig.run(steps) == DONE
    assert rig.sleep.calls == []
