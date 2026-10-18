"""Tests for tools/status.py."""

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import status  # noqa: E402


def fresh(tmp_path):
    """Copy a fixed baseline STATUS.md so tests do not track live plan state."""
    dst = tmp_path / "STATUS.md"
    shutil.copy(REPO / "tests" / "tools" / "fixtures" / "STATUS-baseline.md", dst)
    return dst


def test_repo_status_parses():
    st = status.load()
    assert not st.errors


def test_first_item_is_p0_plan(tmp_path):
    st = status.load(fresh(tmp_path))
    assert status.next_item(st)["id"] == "P0.0"


def test_lifecycle_and_advance(tmp_path):
    f = fresh(tmp_path)
    run = lambda *a: status.main(["--file", str(f), *a])  # noqa: E731
    assert run("set", "P0.0", "DONE") == 0
    assert run("add", "P0.1", "Result record", "--depends", "P0.0") == 0
    assert run("add", "P0.G", "Phase gate", "--depends", "P0.1") == 0
    assert status.next_item(status.load(f))["id"] == "P0.1"
    assert run("set", "P0.1", "ACTIVE") == 0
    assert status.next_item(status.load(f))["resume"] is True
    assert run("set", "P0.1", "DONE") == 0
    assert run("set", "P0.G", "OWNER") == 0
    item = status.next_item(status.load(f))
    assert item["kind"] == "advance" and item["phase"] == "P1"


def test_single_active_enforced(tmp_path):
    f = fresh(tmp_path)
    status.main(["--file", str(f), "add", "P0.1", "a"])
    status.main(["--file", str(f), "set", "P0.0", "ACTIVE"])
    try:
        status.main(["--file", str(f), "set", "P0.1", "ACTIVE"])
    except SystemExit as exc:
        assert "already ACTIVE" in str(exc)
    else:
        raise AssertionError("second ACTIVE accepted")


def test_none_when_everything_waits(tmp_path):
    f = fresh(tmp_path)
    status.main(["--file", str(f), "set", "P0.0", "OWNER"])
    assert status.next_item(status.load(f))["kind"] == "none"


def test_phase_add(tmp_path):
    """phase-add appends a NOT-STARTED row and refuses a repeat, a mismatched branch, a bad id, a bad name, a
    bad note, or an unknown state, leaving the file unchanged."""
    f = fresh(tmp_path)
    assert status.main(["--file", str(f), "phase-add", "P17", "Portable Stack", "p17-portable"]) == 0
    rows = [r for r in status.load(f).phases.rows if r[0] == "P17 Portable Stack"]
    assert rows == [["P17 Portable Stack", "p17-portable", "NOT-STARTED", "-"]]
    before = f.read_bytes()
    for args in (["P17", "Again", "p17-portable"], ["P18", "X", "p17-x"], ["17", "X", "p17-x"],
                 ["P18", "A | B", "p18-x"], ["P18", "A\nB", "p18-x"], ["P18", " ", "p18-x"],
                 ["P18", "X", "p18-x", "--state", "SOON"], ["P18", "X", "p18-x", "--note", "a\nb"]):
        try:
            status.main(["--file", str(f), "phase-add", *args])
        except SystemExit:
            continue
        raise AssertionError(f"phase-add {args} accepted")
    assert f.read_bytes() == before


def test_set_changes_a_title_and_dependencies(tmp_path):
    """set --title and --depends rewrite those cells; '-' clears the dependencies."""
    f = fresh(tmp_path)
    run = lambda *a: status.main(["--file", str(f), *a])  # noqa: E731
    assert run("add", "P0.1", "a", "--depends", "P0.0") == 0
    assert run("add", "P0.G", "Phase gate: A and B", "--depends", "P0.0,P0.1") == 0
    assert run("set", "P0.G", "READY", "--title", "Phase gate: A", "--depends", "P0.0") == 0
    assert [r for r in status.load(f).tasks.rows if r[0] == "P0.G"] == [["P0.G", "READY", "Phase gate: A", "P0.0", ""]]
    assert run("set", "P0.1", "BLOCKED", "--depends", "-", "--note", "moved") == 0
    assert [r for r in status.load(f).tasks.rows if r[0] == "P0.1"] == [["P0.1", "BLOCKED", "a", "-", "moved"]]


DONE_BEFORE_P5 = [(p, "DONE") for p in ("P0", "P1", "P2", "P4", "P17")]


def write_status(tmp_path, phases, tasks):
    """Write a STATUS.md with LF line endings from (phase, state) and (task, state, depends) rows."""
    lines = ["# Status", "", "## Phases", "", "| Phase | Branch | State | Note |", "| --- | --- | --- | --- |"]
    lines += [f"| {pid} Name | p{pid[1:]}-x | {state} | - |" for pid, state in phases]
    lines += ["", "## Tasks", "", "| ID | State | Title | Depends | Note |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {tid} | {state} | t | {depends} | |" for tid, state, depends in tasks]
    f = tmp_path / "STATUS.md"
    f.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return f


def test_ready_tasks_behind_an_owner_task_stall_the_phase(tmp_path):
    """P5's READY tasks wait on its OWNER install, so P5 is stalled and P12 is planned beside it (OQ-055, 9 (a))."""
    f = write_status(tmp_path, DONE_BEFORE_P5 + [("P5", "ACTIVE"), ("P12", "NOT-STARTED")],
                     [("P5.1", "DONE", "-"), ("P5.4", "OWNER", "P5.1"), ("P5.5", "READY", "P5.1,P5.4"),
                      ("P5.G", "READY", "P5.5")])
    st = status.load(f)
    assert not st.errors
    assert status.phase_stalled(st, "P5")
    item = status.next_item(st)
    assert item["kind"] == "advance" and item["phase"] == "P12"


def test_a_task_that_can_start_keeps_its_phase_from_stalling(tmp_path):
    """ACTIVE work, a READY task whose dependencies are DONE, or no DONE work at all keeps a phase unstalled;
    a chain of READY tasks behind a BLOCKED task does not."""
    cases = [
        ([("P5.1", "DONE", "-"), ("P5.2", "READY", "P5.1"), ("P5.3", "OWNER", "-")], False),
        ([("P5.1", "DONE", "-"), ("P5.2", "ACTIVE", "P5.1"), ("P5.3", "OWNER", "-")], False),
        ([("P5.1", "OWNER", "-"), ("P5.2", "READY", "P5.1")], False),
        ([("P5.1", "DONE", "-"), ("P5.2", "BLOCKED", "-"), ("P5.3", "READY", "P5.2"), ("P5.4", "READY", "P5.3")],
         True),
        ([("P5.1", "DONE", "-"), ("P5.2", "OWNER", "-"), ("P5.3", "BLOCKED", "-")], True),
    ]
    for tasks, stalled in cases:
        f = write_status(tmp_path, DONE_BEFORE_P5 + [("P5", "ACTIVE")], tasks)
        assert status.phase_stalled(status.load(f), "P5") is stalled, tasks


def test_a_phase_outside_the_work_order_is_never_advanced(tmp_path):
    """A NOT-STARTED phase outside WORK_ORDER is never advanced, even once every phase in the order is DONE: P11
    (the df dialect) and P18 (TurboQuant Tier B) are separate projects the owner deferred (2026-10-18, OQ-066)."""
    assert not {"P11", "P18"} & {pid for pid, _ in status.WORK_ORDER}
    every_phase_done = [(pid, "DONE") for pid, _ in status.WORK_ORDER]
    for outside in ("P11", "P18"):
        f = write_status(tmp_path, every_phase_done + [(outside, "NOT-STARTED")], [("P0.0", "DONE", "-")])
        assert status.next_item(status.load(f))["kind"] == "none", outside
    f = write_status(tmp_path, DONE_BEFORE_P5 + [("P5", "ACTIVE"), ("P11", "NOT-STARTED"), ("P12", "NOT-STARTED")],
                     [("P5.1", "DONE", "-"), ("P5.4", "OWNER", "P5.1")])
    item = status.next_item(status.load(f))
    assert item["kind"] == "advance" and item["phase"] == "P12"


def test_p12_waits_for_p5(tmp_path):
    """P12 is not planned while P5 has a task that can start."""
    f = write_status(tmp_path, DONE_BEFORE_P5 + [("P5", "ACTIVE"), ("P12", "NOT-STARTED")],
                     [("P5.1", "DONE", "-"), ("P5.2", "OWNER", "-"), ("P5.3", "READY", "P5.1")])
    item = status.next_item(status.load(f))
    assert item["kind"] == "task" and item["id"] == "P5.3"


def test_p16_follows_p12(tmp_path):
    """P16's part (a) is planned once P12 is stalled, not while P12 has no DONE work, and never while BLOCKED
    (OQ-055, decision 12 (b))."""
    tasks = [("P5.1", "DONE", "-"), ("P5.4", "OWNER", "P5.1")]
    phases = DONE_BEFORE_P5 + [("P5", "ACTIVE"), ("P12", "ACTIVE")]
    f = write_status(tmp_path, phases + [("P16", "NOT-STARTED")],
                     tasks + [("P12.1", "DONE", "-"), ("P12.2", "OWNER", "-")])
    item = status.next_item(status.load(f))
    assert item["kind"] == "advance" and item["phase"] == "P16"
    f = write_status(tmp_path, phases + [("P16", "NOT-STARTED")], tasks + [("P12.2", "OWNER", "-")])
    assert status.next_item(status.load(f))["kind"] == "none"
    f = write_status(tmp_path, phases + [("P16", "BLOCKED")], tasks + [("P12.1", "DONE", "-")])
    assert status.next_item(status.load(f))["kind"] == "none"


def test_save_writes_lf_line_endings(tmp_path):
    """A rewrite writes LF line endings on every platform, even over a CRLF copy (PHASE-NOTES, All Phases)."""
    f = fresh(tmp_path)
    f.write_bytes(f.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert status.main(["--file", str(f), "set", "P0.0", "ACTIVE"]) == 0
    data = f.read_bytes()
    assert b"\r" not in data and data.endswith(b"\n")
    assert [r for r in status.load(f).tasks.rows if r[0] == "P0.0"][0][1] == "ACTIVE"


def test_set_refuses_a_bad_title_or_dependency_and_changes_nothing(tmp_path):
    """An unknown dependency, the task itself, or an empty title is refused before anything is written."""
    f = fresh(tmp_path)
    status.main(["--file", str(f), "add", "P0.1", "a", "--depends", "P0.0"])
    before = f.read_bytes()
    for extra in (["--depends", "P0.9"], ["--depends", "P0.1"], ["--depends", "P0.0,P9.9"], ["--title", " "]):
        try:
            status.main(["--file", str(f), "set", "P0.1", "READY", *extra])
        except SystemExit:
            continue
        raise AssertionError(f"set {extra} accepted")
    assert f.read_bytes() == before
