"""Tests for what the ttsim executor reads from a run: findings, kernel JIT errors, and the Watcher rerun (P4.11).

Bible: ttsim Facts (the finding line, its classes, the JIT abort, the hang,
Watcher), Harness Contract (kernel JIT errors surface during execution and
are reclassified as stage jit; a timeout returns a hang diagnostic plus the
Watcher dump), Result Record (Diagnostic stages), Agent Rules 2 and 6.
Owner-queue choices applied under the owner's standing direction: OQ-036
option (a) (the class lists below) and OQ-037 option (b) (a hung run is
rerun once with Watcher). Plan: plans/p4-ttsim.md, P4.11. The captures are
tests/executors/fixtures/ttsim/ (P4.9; README.md there).

The contract these tests fix:

- Findings. ttsim prints a finding on stdout as one line
  `[<cycle>] ERROR: <class>: <function>: <message>`; every such line
  counts, in order. UndefinedBehavior, NonContractualBehavior, and
  UnpredictableValueUsed are undefined behavior (sim_ub True).
  UnimplementedFunctionality, UnsupportedFunctionality,
  UntestedFunctionality, SystemError, ConfigurationError, AssertionFailure,
  and any class not listed are gaps: sim_gap is the first gap class in
  stdout order. sim_ub is False when no undefined behavior appeared (the
  executor always checks) and sim_gap None when no gap did. Each finding is
  also one Diagnostic(stage="run", severity="error", code=<class>,
  message="<class>: <function>: <message>"), in order, with printable ASCII
  only (any other character becomes "?").
- Kernel JIT errors. Each `<path>:<line>:<column>: <severity>: <message>`
  line of a "<processor> build failed. Log:" block becomes one
  Diagnostic(stage="jit") with the severity as printed, the file relative to
  the workdir when it lies inside it and as printed otherwise, the line, the
  column, and the message. The same log in stdout and stderr counts once. A
  line of that form outside such a block (a host program's own print) is
  not a jit diagnostic.
- Hang and Watcher. When the sandbox reports a hang, the program runs once
  more with TT_METAL_WATCHER=1 added and everything else the same, under the
  same limits, in a fresh copy of the workdir's pre-run files at
  <workdir>/@ttsim/watcher/ with that copy's own cache and logs, so the
  scored run's streams and output files are never touched. The copy is
  made fresh, and only when every pre-run file still holds the bytes it
  held before the scored run (their sha256 is taken before it); when one
  changed or vanished there is no rerun and no note. From
  <copy logs>/generated/watcher/watcher.log the last complete dump's core
  lines with a nonzero kernel id, and the k_id lines of those ids, become one
  Diagnostic(stage="run", severity="note", code=lassi.core.stages
  WATCHER_CODE) naming the dump, at most WATCHER_BYTES_CAP bytes, cut on a
  line boundary with the count of lines left out. No complete dump, or no
  log, means no such note; the RunResult is otherwise the first run's.

Every test uses a FakeSandbox (tests/executors/ttsim_fakes.py) and a
PLACEHOLDER install; nothing is executed and no simulator or device opens.
Lines marked SYNTHETIC are made up here; the P4.9 captures are the
recorded runs. No value in this module is a measurement.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from ttsim_fakes import (
    CAPTURES,
    P49_CWD,
    RESERVED,
    SEEDED_KERNEL,
    WATCHER_NAME,
    Effect,
    FakeInstall,
    FakeSandbox,
    canned,
    capture,
    capture_status,
    capture_text,
    inside,
    make_artifact,
    make_install,
    same_path,
    watcher_log,
    writes_watcher_log,
)

from lassi.core import stages
from lassi.core.interfaces import Limits, RunResult
from lassi.core.record import Diagnostic

LIMITS = Limits(wall_s=30.0, memory_mb=4096, cpus=4)
INPUTS = ["--size", "8"]
UB_CLASSES = ("UndefinedBehavior", "NonContractualBehavior", "UnpredictableValueUsed")
GAP_CLASSES = (
    "UnimplementedFunctionality",
    "UnsupportedFunctionality",
    "UntestedFunctionality",
    "SystemError",
    "ConfigurationError",
    "AssertionFailure",
)
# A class no list names; ttsim may print more classes than P4.9 saw (the P4.9 lesson: the class is any word).
UNKNOWN_CLASS = "SyntheticFutureClass"
# What each P4.9 capture reads as: sim_ub, sim_gap, the run-stage findings (code, message), and its jit errors.
# The finding lines are the captures' own (stdout.txt, status.json sim_lines).
EXPECTED: dict[str, tuple[bool, str | None, list[tuple[str, str]], int]] = {
    "non-contractual-behavior": (
        True, None,
        [("NonContractualBehavior", "NonContractualBehavior: rv32_mem_rd: unaligned addr=0x16dfe2 size=4")], 0,
    ),
    "undefined-behavior": (
        True, None,
        [("UndefinedBehavior", "UndefinedBehavior: decode_and_execute_csrrs: Wormhole does not support Zicsr")], 0,
    ),
    "unsupported-functionality": (
        False, "UnsupportedFunctionality",
        [(
            "UnsupportedFunctionality",
            "UnsupportedFunctionality: decode_and_execute_fence: fence instructions do not enforce memory ordering "
            "on Wormhole and should not be used",
        )],
        0,
    ),
    "jit-error": (False, None, [], 1),
    "hang": (False, None, [], 0),
    "watcher-hang": (False, None, [], 0),
}
# The jit-error capture's compiler line names this file, line, and column, with this message.
JIT_FILE = f"{P49_CWD}/{SEEDED_KERNEL}"
JIT_MESSAGE = "'lassi_seeded_jit_error' was not declared in this scope"


@pytest.fixture
def ttsim() -> ModuleType:
    """Return the lassi.executors.ttsim module (imported here so each test shows a missing module clearly)."""
    from lassi.executors import ttsim

    return ttsim


@pytest.fixture(autouse=True)
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make tmp_path/runs the runs root, set the scratch and home roots beside it, and unset $LASSI_TOOLCHAINS."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("LASSI_RUNS_ROOT", str(runs))
    monkeypatch.setenv("LASSI_SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("LASSI_TOOLCHAINS", raising=False)
    return runs


@pytest.fixture
def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeInstall:
    """Return the PLACEHOLDER install the executor's checks accept (ttsim_fakes.make_install)."""
    return make_install(tmp_path, monkeypatch)


def run_with(
    ttsim: ModuleType,
    install: FakeInstall,
    artifact: Path,
    results: Sequence[Any],
    effects: Sequence[Effect | None] = (),
) -> tuple[RunResult, FakeSandbox]:
    """Run `artifact` through the executor on a FakeSandbox answering `results` in order; return both."""
    fake = FakeSandbox(results=list(results), effects=list(effects))
    result = ttsim.TtsimExecutor(toolchains=str(install.root), sandbox=fake).run(artifact, INPUTS, LIMITS)
    return result, fake


def run_stdout(ttsim: ModuleType, install: FakeInstall, runs_root: Path, stdout: str, **fields: Any) -> RunResult:
    """Run once on a sandbox that returns `stdout` (and any other SandboxResult field) with no hang."""
    result, fake = run_with(ttsim, install, make_artifact(runs_root), [canned(stdout=stdout, **fields)])
    assert len(fake.calls) == 1, "a run that did not hang runs once"
    return result


def finding(cls: str, function: str = "SYNTHETIC_function", message: str = "SYNTHETIC message") -> str:
    """Return one SYNTHETIC ttsim finding line in the captured format."""
    return f"[1234] ERROR: {cls}: {function}: {message}\n"


def run_diagnostics(result: RunResult) -> list[Diagnostic]:
    """Return the run-stage Diagnostics of a result, in order."""
    return [item for item in result.diagnostics if item.stage == "run"]


def jit_diagnostics(result: RunResult) -> list[Diagnostic]:
    """Return the jit-stage Diagnostics of a result, in order."""
    return [item for item in result.diagnostics if item.stage == "jit"]


def finding_diagnostic(cls: str, message: str) -> Diagnostic:
    """Return the Diagnostic one finding becomes."""
    return Diagnostic(stage="run", severity="error", code=cls, message=message)


def watcher_notes(result: RunResult) -> list[Diagnostic]:
    """Return the result's Watcher notes."""
    return [item for item in result.diagnostics if item.code == stages.WATCHER_CODE]


# ---------------------------------------------------------------------------
# The P4.9 captures


@pytest.mark.parametrize("name", CAPTURES)
def test_each_capture_reads_as_its_recorded_run(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path, name: str
) -> None:
    sim_ub, sim_gap, findings, jit_errors = EXPECTED[name]
    recorded = capture(name)
    # A hung capture is rerun once with Watcher; the rerun here writes no log, so no Watcher note comes of it.
    results = [recorded, recorded] if recorded.hang else [recorded]
    result, _ = run_with(ttsim, install, make_artifact(runs_root), results)
    status = capture_status(name)
    assert (result.exit_code, result.hang) == (status["rc"], bool(status["timed_out"]))
    assert result.stdout == capture_text(name, "stdout.txt"), "the stream is the run's, unchanged"
    assert (result.sim_ub, result.sim_gap) == (sim_ub, sim_gap)
    assert run_diagnostics(result) == [finding_diagnostic(cls, message) for cls, message in findings]
    assert len([item for item in jit_diagnostics(result) if item.severity == "error"]) == jit_errors


# ---------------------------------------------------------------------------
# Finding lines and their classes


@pytest.mark.parametrize("cls", UB_CLASSES)
def test_an_undefined_behavior_class_sets_sim_ub(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path, cls: str
) -> None:
    result = run_stdout(ttsim, install, runs_root, "SYNTHETIC host output\n" + finding(cls), returncode=1)
    assert (result.sim_ub, result.sim_gap) == (True, None)
    assert run_diagnostics(result) == [finding_diagnostic(cls, f"{cls}: SYNTHETIC_function: SYNTHETIC message")]


@pytest.mark.parametrize("cls", [*GAP_CLASSES, UNKNOWN_CLASS])
def test_a_gap_class_or_an_unlisted_one_is_a_gap_named_by_its_class(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path, cls: str
) -> None:
    result = run_stdout(ttsim, install, runs_root, finding(cls), returncode=1)
    assert (result.sim_ub, result.sim_gap) == (False, cls)
    assert run_diagnostics(result) == [finding_diagnostic(cls, f"{cls}: SYNTHETIC_function: SYNTHETIC message")]


def test_every_finding_counts_in_stdout_order_and_the_first_gap_names_the_gap(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path
) -> None:
    stdout = (
        "SYNTHETIC host output\n"
        + finding("UntestedFunctionality", "fn_a", "first")
        + finding("UnpredictableValueUsed", "fn_b", "second")
        + finding("ConfigurationError", "fn_c", "third")
    )
    result = run_stdout(ttsim, install, runs_root, stdout, returncode=1)
    assert (result.sim_ub, result.sim_gap) == (True, "UntestedFunctionality")
    assert [(item.code, item.message) for item in run_diagnostics(result)] == [
        ("UntestedFunctionality", "UntestedFunctionality: fn_a: first"),
        ("UnpredictableValueUsed", "UnpredictableValueUsed: fn_b: second"),
        ("ConfigurationError", "ConfigurationError: fn_c: third"),
    ]


def test_lines_that_are_not_findings_give_none(ttsim: ModuleType, install: FakeInstall, runs_root: Path) -> None:
    stdout = (
        capture_text("hang", "stdout.txt")
        + "[20000000] 46.2 seconds (433.2 KHz)\n"
        + "WARNING: SYNTHETIC: a tt-metal warning line\n"
        + "2026-09-26 14:10:12.547 | error    |          Fabric | SYNTHETIC log line (control_plane.cpp:1)\n"
    )
    result = run_stdout(ttsim, install, runs_root, stdout)
    assert (result.sim_ub, result.sim_gap, list(result.diagnostics)) == (False, None, [])


def test_a_finding_message_keeps_printable_ascii_only(ttsim: ModuleType, install: FakeInstall, runs_root: Path) -> None:
    line = finding("UndefinedBehavior", "rv32_mem_rd", "SYNTHETIC caf\N{LATIN SMALL LETTER E WITH ACUTE} \x07 end")
    result = run_stdout(ttsim, install, runs_root, line, returncode=1)
    (diagnostic,) = run_diagnostics(result)
    assert diagnostic.message == "UndefinedBehavior: rv32_mem_rd: SYNTHETIC caf? ? end"


# ---------------------------------------------------------------------------
# Kernel JIT errors


def test_the_jit_error_capture_gives_one_jit_error_as_printed(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path
) -> None:
    # The same compiler line is in stdout twice and in stderr once; it counts once. Its file lies outside this
    # test's workdir (the P4.9 step's), so it stays as printed.
    result, _ = run_with(ttsim, install, make_artifact(runs_root), [capture("jit-error")])
    (diagnostic,) = jit_diagnostics(result)
    assert (diagnostic.severity, diagnostic.file, diagnostic.line, diagnostic.column) == ("error", JIT_FILE, 6, 5)
    assert JIT_MESSAGE in diagnostic.message
    assert result.exit_code == 134, "the host aborts; the diagnostics, not the status, make the attempt S1"


@pytest.mark.skipif(os.name == "nt", reason="the kernel JIT prints POSIX paths; the rule is checked on a POSIX host")
def test_a_jit_error_inside_the_workdir_names_its_file_relative_to_the_workdir(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path
) -> None:
    artifact = make_artifact(runs_root)
    workdir = os.path.realpath(artifact.parent)
    recorded = capture("jit-error")
    moved = canned(
        returncode=recorded.returncode,
        stdout=recorded.stdout.replace(P49_CWD, workdir),
        stderr=recorded.stderr.replace(P49_CWD, workdir),
    )
    result, _ = run_with(ttsim, install, artifact, [moved])
    (diagnostic,) = jit_diagnostics(result)
    assert (diagnostic.file, diagnostic.line, diagnostic.column) == (SEEDED_KERNEL, 6, 5)


def test_jit_lines_keep_their_printed_severity_and_order(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path
) -> None:
    stdout = (
        "SYNTHETIC host output\n"
        "trisc1 build failed. Log: In file included from SYNTHETIC_includes.hpp:1,\n"
        "/SYNTHETIC/outside/kernels/k.cpp:3:9: warning: unused variable 'v' [-Wunused-variable]\n"
        "    3 |     int v = 0;\n"
        "/SYNTHETIC/outside/kernels/k.cpp:4:5: error: 'y' was not declared in this scope\n"
        "\n"
    )
    result = run_stdout(ttsim, install, runs_root, stdout, returncode=134)
    found = jit_diagnostics(result)
    places = [(item.severity, item.file, item.line, item.column) for item in found]
    assert places == [
        ("warning", "/SYNTHETIC/outside/kernels/k.cpp", 3, 9),
        ("error", "/SYNTHETIC/outside/kernels/k.cpp", 4, 5),
    ]
    assert "unused variable 'v'" in found[0].message and "'y' was not declared in this scope" in found[1].message


def test_a_compiler_style_line_outside_a_build_failed_log_is_not_a_jit_diagnostic(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path
) -> None:
    # A host program may print anything; only the JIT's own build log is read as kernel JIT output.
    stdout = "/SYNTHETIC/outside/kernels/k.cpp:4:5: error: SYNTHETIC host print that looks like a compiler line\n"
    result = run_stdout(ttsim, install, runs_root, stdout)
    assert jit_diagnostics(result) == []


# ---------------------------------------------------------------------------
# The Watcher rerun of a hang


HUNG = canned(returncode=137, hang=True, stdout="SYNTHETIC first run stdout\n", stderr="SYNTHETIC first run stderr\n")


def test_a_run_that_did_not_hang_is_not_rerun(ttsim: ModuleType, install: FakeInstall, runs_root: Path) -> None:
    result, fake = run_with(ttsim, install, make_artifact(runs_root), [canned(returncode=3), canned()])
    assert len(fake.calls) == 1 and WATCHER_NAME not in fake.calls[0].environment
    assert watcher_notes(result) == []


def test_a_hang_is_rerun_once_with_watcher_in_a_fresh_copy_of_the_pre_run_files(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path
) -> None:
    artifact = make_artifact(runs_root)
    workdir = Path(os.path.realpath(artifact.parent))
    copy = workdir / RESERVED / "watcher"
    seen: dict[str, Any] = {}

    def scored_run_writes(spec: Any, argv: list[str]) -> None:
        """Write an output file and a JIT cache file, as the scored run would."""
        (workdir / "out.bin").write_bytes(b"\x01")
        (workdir / RESERVED / "cache" / "first.elf").write_bytes(b"PLACEHOLDER\n")

    def rerun_looks(spec: Any, argv: list[str]) -> None:
        """Note what the rerun's workdir holds and whether its cache and logs exist, then write the log."""
        environment = dict(spec.environment)
        root = Path(spec.workdir)
        seen["files"] = {
            path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()
        }
        seen["ready"] = all(Path(environment[name]).is_dir() for name in ("TT_METAL_CACHE", "TT_METAL_LOGS_PATH"))
        writes_watcher_log(watcher_log())(spec, argv)

    result, fake = run_with(ttsim, install, artifact, [HUNG, HUNG], [scored_run_writes, rerun_looks])
    assert len(fake.calls) == 2, "one rerun, even though the rerun hangs too"
    first, second = fake.calls
    assert WATCHER_NAME not in first.environment and second.environment[WATCHER_NAME] == "1"
    moved = {"TT_METAL_CACHE", "TT_METAL_LOGS_PATH"}
    kept = {name: value for name, value in second.environment.items() if name not in {*moved, WATCHER_NAME}}
    assert kept == {name: value for name, value in first.environment.items() if name not in moved}
    assert same_path(second.spec.workdir, copy), "the rerun runs in <workdir>/@ttsim/watcher"
    for name in moved:
        assert inside(second.environment[name], copy), f"the copy's own {name}"
        assert not same_path(second.environment[name], first.environment[name])
    assert second.limits == first.limits == LIMITS
    assert same_path(second.argv[0], copy / artifact.name) and second.argv[1:] == first.argv[1:] == INPUTS
    assert (second.spec.hidden_roots, second.spec.harness, second.spec.toolchains) == (
        first.spec.hidden_roots, first.spec.harness, first.spec.toolchains,
    )
    assert seen["ready"], "the copy's cache and logs directories exist when the rerun starts"
    assert seen["files"] == {
        artifact.name: artifact.read_bytes(),
        "kernels/k.cpp": (workdir / "kernels" / "k.cpp").read_bytes(),
    }, "the copy holds the pre-run files only: no output of the scored run, no JIT cache"


def test_the_scored_run_stays_the_first_run(ttsim: ModuleType, install: FakeInstall, runs_root: Path) -> None:
    artifact = make_artifact(runs_root)
    workdir = Path(os.path.realpath(artifact.parent))

    def scored_run_writes(spec: Any, argv: list[str]) -> None:
        """Write the scored run's output file."""
        (workdir / "out.bin").write_bytes(b"\x01")

    rerun = canned(returncode=1, hang=False, stdout="SYNTHETIC rerun stdout\n" + finding("UndefinedBehavior"))
    effects = [scored_run_writes, writes_watcher_log(watcher_log())]
    result, _ = run_with(ttsim, install, artifact, [HUNG, rerun], effects)
    assert (result.exit_code, result.hang, result.stdout, result.stderr) == (
        137, True, HUNG.stdout, HUNG.stderr,
    )
    assert (result.sim_ub, result.sim_gap) == (False, None), "the rerun's findings are not the scored run's"
    assert sorted(result.output_files) == ["out.bin"], "nothing of the rerun's copy is an output file"
    assert [item.code for item in result.diagnostics] == [stages.WATCHER_CODE], "only the Watcher note is added"


def changes_a_pre_run_file(how: str, workdir: Path, artifact: Path) -> Effect:
    """Return an effect by which the scored run changes or removes one of its workdir's pre-run files."""

    def effect(spec: Any, argv: list[str]) -> None:
        """Change the pre-run file as `how` names, the way a program may rewrite files in its workdir."""
        if how == "rewrites-the-artifact":
            artifact.write_bytes(b"PLACEHOLDER artifact rewritten by the scored run\n")
        elif how == "rewrites-a-kernel":
            (workdir / "kernels" / "k.cpp").write_bytes(b"// SYNTHETIC kernel rewritten by the scored run\n")
        elif how == "removes-a-kernel":
            (workdir / "kernels" / "k.cpp").unlink()
        else:
            raise AssertionError(how)

    return effect


@pytest.mark.parametrize("how", ["rewrites-the-artifact", "rewrites-a-kernel", "removes-a-kernel"])
def test_a_pre_run_file_the_scored_run_changed_or_removed_skips_the_rerun(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path, how: str
) -> None:
    # Section 7 of the P4.11 design: the rerun runs only the program that was scored, so every pre-run file must
    # still hold the bytes it held before the scored run. A FakeSandbox with one result fails the test on a rerun.
    artifact = make_artifact(runs_root)
    workdir = Path(os.path.realpath(artifact.parent))
    result, fake = run_with(ttsim, install, artifact, [HUNG], [changes_a_pre_run_file(how, workdir, artifact)])
    assert len(fake.calls) == 1, "no rerun"
    assert watcher_notes(result) == [] and list(result.diagnostics) == []
    assert (result.exit_code, result.hang, result.stdout, result.stderr) == (137, True, HUNG.stdout, HUNG.stderr)


def test_the_rerun_copy_is_fresh_so_a_log_the_scored_run_planted_there_is_never_read(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path
) -> None:
    artifact = make_artifact(runs_root)
    workdir = Path(os.path.realpath(artifact.parent))
    planted = workdir / RESERVED / "watcher" / RESERVED / "logs" / "generated" / "watcher" / "watcher.log"

    def scored_run_plants(spec: Any, argv: list[str]) -> None:
        """Write a Watcher log where the rerun's copy will keep its logs, as a program in the workdir could."""
        planted.parent.mkdir(parents=True)
        planted.write_bytes(watcher_log().encode("utf-8"))

    result, fake = run_with(ttsim, install, artifact, [HUNG, HUNG], [scored_run_plants, None])
    assert len(fake.calls) == 2 and watcher_notes(result) == [], "the rerun wrote no log, so there is no note"


def test_the_watcher_note_condenses_the_last_complete_dump_of_the_capture(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path
) -> None:
    effects = [None, writes_watcher_log(watcher_log())]
    result, _ = run_with(ttsim, install, make_artifact(runs_root), [HUNG, HUNG], effects)
    (note,) = watcher_notes(result)
    assert (note.stage, note.severity, note.code) == ("run", "note", "watcher")
    message = note.message
    assert message.isascii() and len(message.encode("ascii")) <= ttsim.WATCHER_BYTES_CAP
    assert re.search(r"#\s*128\b", message), f"the note names dump #128, the last complete one: {message!r}"
    assert "core(x= 0,y= 0)" in message and "NSW" in message, "the hung worker core at a NOC semaphore wait"
    assert SEEDED_KERNEL in message, "k_id 1 mapped to the kernel's path"
    for left_out in ("core(x= 1,y= 0)", "idleth", "blank", "#127", "Legend"):
        assert left_out not in message, f"{left_out!r} is not a kept line of the last dump: {message!r}"


def test_the_watcher_cap_is_a_stated_positive_byte_count(ttsim: ModuleType) -> None:
    cap = ttsim.WATCHER_BYTES_CAP
    assert isinstance(cap, int) and not isinstance(cap, bool) and cap > 0


def test_a_later_partial_dump_leaves_the_last_complete_one(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path
) -> None:
    partial = "-----\nDump #129 at 131.168s\nDevice 0 worker core(x= 0,y= 0) virtual(x=18,y=18):  NSW,"
    effects = [None, writes_watcher_log(watcher_log() + partial)]
    result, _ = run_with(ttsim, install, make_artifact(runs_root), [HUNG, HUNG], effects)
    (note,) = watcher_notes(result)
    assert re.search(r"#\s*128\b", note.message) and "129" not in note.message, note.message


def test_a_log_with_no_complete_dump_gives_no_watcher_note(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path
) -> None:
    head = "".join(watcher_log().splitlines(keepends=True)[:100])
    assert "Dump #1 at" in head and "completed" not in head, "the head opens dump #1 and never closes it"
    result, fake = run_with(ttsim, install, make_artifact(runs_root), [HUNG, HUNG], [None, writes_watcher_log(head)])
    assert len(fake.calls) == 2 and watcher_notes(result) == []
    assert (result.exit_code, result.hang, result.stdout, list(result.diagnostics)) == (137, True, HUNG.stdout, [])


def test_no_watcher_log_gives_no_watcher_note(ttsim: ModuleType, install: FakeInstall, runs_root: Path) -> None:
    result, fake = run_with(ttsim, install, make_artifact(runs_root), [HUNG, HUNG])
    assert len(fake.calls) == 2 and watcher_notes(result) == []
    assert (result.exit_code, result.hang, result.stdout) == (137, True, HUNG.stdout)


def collapsed(text: str) -> str:
    """Return `text` with every run of whitespace written as one space."""
    return " ".join(text.split())


def synthetic_dump(count: int) -> tuple[str, list[str], list[str]]:
    """Return a SYNTHETIC Watcher log with one complete dump of `count` busy worker cores, and its kept lines.

    Each core line holds its own kernel id, and each id has its own k_id
    line, so every core line and every k_id line is kept (2 x count lines).
    """
    cores = [
        f"Device 0 worker core(x={index % 8:2d},y={index // 8:2d}) virtual(x=18,y=18):  NSW,   W,   W,   W,   W  "
        f"rmsg:H0G|Bnt h_id:  0 smsg:DDDD k_ids:{index + 1:3d}|  0|  0|  0|  0"
        for index in range(count)
    ]
    ids = [f"k_id[{index + 1:3d}]: SYNTHETIC/kernels/synthetic_{index + 1:04d}.cpp" for index in range(count)]
    lines = [
        "At 0.500s starting",
        "-----",
        "Dump #7 at 1.500s",
        *cores,
        "Device 0 idleth core(x= 0,y=15) virtual(x=21,y=17):   GW,   X,   X,   X,   X  rmsg:H0D|e h_id:  0 k_ids:  0",
        "k_id[  0]: blank",
        *ids,
        "Dump #7 completed at 1.600s",
    ]
    return "".join(f"{line}\n" for line in lines), cores, ids


def test_a_long_dump_is_capped_on_a_line_boundary_with_the_count_left_out(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path
) -> None:
    cap = ttsim.WATCHER_BYTES_CAP
    count = cap // 50 + 50
    log, cores, ids = synthetic_dump(count)
    result, _ = run_with(ttsim, install, make_artifact(runs_root), [HUNG, HUNG], [None, writes_watcher_log(log)])
    (note,) = watcher_notes(result)
    message = note.message
    assert len(message.encode("ascii")) <= cap
    assert re.search(r"#\s*7\b", message), message[:200]
    # Compared with runs of whitespace collapsed, so a note may keep the lines verbatim or with blanks collapsed.
    flat = collapsed(message)
    tokens = [f"core(x={index % 8:2d},y={index // 8:2d})" for index in range(count)]
    tokens += [f"synthetic_{index + 1:04d}.cpp" for index in range(count)]
    lines = [collapsed(line) for line in (*cores, *ids)]
    cut = [line for token, line in zip(tokens, lines, strict=True) if token in flat and line not in flat]
    assert cut == [], f"a kept line is cut in the middle: {cut[:1]}"
    shown = sum(1 for line in lines if line in flat)
    assert 0 < shown < len(lines), "the note shows the first kept lines and leaves the rest out"
    assert re.search(rf"\b{len(lines) - shown}\b", message), f"the note says {len(lines) - shown} lines were left out"
