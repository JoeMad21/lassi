#!/usr/bin/env python3
"""Read and update plans/STATUS.md, the single source of task and phase state.

STATUS.md holds two pipe tables that this tool parses and rewrites:

  ## Phases   columns: Phase | Branch | State | Note
  ## Tasks    columns: ID | State | Title | Depends | Note

Phase states: NOT-STARTED, ACTIVE, GATE-OWNER, DONE, BLOCKED.
Task states:  READY, ACTIVE, DONE, BLOCKED, OWNER.

Commands:
  summary [--json]                 counts and the next item
  next [--json]                    the next item: task, advance, or none
  set ID STATE [--note TEXT]       set a task state
  add ID TITLE [--depends A,B] [--state S] [--note TEXT]
  phase PN STATE [--note TEXT]     set a phase state
  check                            validate the file

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
STATUS = REPO / "plans" / "STATUS.md"
WORK_ORDER = ["P0", "P1", "P2", "P4", "P5", "P12", "P11"]
PHASE_STATES = {"NOT-STARTED", "ACTIVE", "GATE-OWNER", "DONE", "BLOCKED"}
TASK_STATES = {"READY", "ACTIVE", "DONE", "BLOCKED", "OWNER"}
TASK_ID = re.compile(r"^P(\d+)\.(\d+|G)$")
PHASE_ID = re.compile(r"^P(\d+)$")


@dataclass
class Table:
    """A parsed pipe table with its location in the file."""

    header: List[str]
    rows: List[List[str]]
    start: int
    end: int


@dataclass
class Status:
    """The parsed STATUS.md file."""

    lines: List[str]
    phases: Table
    tasks: Table
    errors: List[str] = field(default_factory=list)


def split_row(line: str) -> List[str]:
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def find_table(lines: List[str], heading: str) -> Table:
    """Locate the first pipe table after a '## heading' line."""
    try:
        h = next(i for i, ln in enumerate(lines) if ln.strip() == f"## {heading}")
    except StopIteration:
        raise SystemExit(f"status: section '## {heading}' not found") from None
    i = h + 1
    while i < len(lines) and not lines[i].lstrip().startswith("|"):
        i += 1
    start = i
    header = split_row(lines[i])
    i += 2  # skip separator
    rows = []
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        rows.append(split_row(lines[i]))
        i += 1
    return Table(header, rows, start, i)


def load(path: Path = STATUS) -> Status:
    lines = path.read_text(encoding="utf-8").splitlines()
    st = Status(lines, find_table(lines, "Phases"), find_table(lines, "Tasks"))
    validate(st)
    return st


def validate(st: Status) -> None:
    if st.phases.header != ["Phase", "Branch", "State", "Note"]:
        st.errors.append(f"phase header must be Phase | Branch | State | Note, got {st.phases.header}")
    if st.tasks.header != ["ID", "State", "Title", "Depends", "Note"]:
        st.errors.append(f"task header must be ID | State | Title | Depends | Note, got {st.tasks.header}")
    for r in st.phases.rows:
        if len(r) != 4 or not PHASE_ID.match(r[0].split()[0]) or r[2] not in PHASE_STATES:
            st.errors.append(f"bad phase row: {r}")
    seen = set()
    for r in st.tasks.rows:
        if len(r) != 5 or not TASK_ID.match(r[0]) or r[1] not in TASK_STATES:
            st.errors.append(f"bad task row: {r}")
            continue
        if r[0] in seen:
            st.errors.append(f"duplicate task id {r[0]}")
        seen.add(r[0])
    active = [r[0] for r in st.tasks.rows if len(r) == 5 and r[1] == "ACTIVE"]
    if len(active) > 1:
        st.errors.append(f"more than one ACTIVE task: {active}")


def render_table(t: Table) -> List[str]:
    out = ["| " + " | ".join(t.header) + " |", "| " + " | ".join("---" for _ in t.header) + " |"]
    for r in t.rows:
        out.append("| " + " | ".join(r) + " |")
    return out


def save(st: Status, path: Path = STATUS) -> None:
    lines = list(st.lines)
    # Replace the later table first so earlier indices stay valid.
    for t in sorted([st.phases, st.tasks], key=lambda x: x.start, reverse=True):
        lines[t.start:t.end] = render_table(t)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def phase_of(task_id: str) -> str:
    return "P" + TASK_ID.match(task_id).group(1)


def phase_state(st: Status) -> Dict[str, str]:
    return {r[0].split()[0]: r[2] for r in st.phases.rows}


def task_state(st: Status) -> Dict[str, str]:
    return {r[0]: r[1] for r in st.tasks.rows}


def deps_done(row: List[str], states: Dict[str, str]) -> bool:
    deps = [d.strip() for d in row[3].split(",") if d.strip() and d.strip() != "-"]
    return all(states.get(d) == "DONE" for d in deps)


def phase_stalled(st: Status, pid: str) -> bool:
    """A phase is stalled when it has DONE work and nothing READY or ACTIVE left."""
    rows = [r for r in st.tasks.rows if phase_of(r[0]) == pid]
    if not rows:
        return False
    states = {r[1] for r in rows}
    return "DONE" in states and not states & {"READY", "ACTIVE"}


def next_item(st: Status) -> dict:
    """Pick the next unit of work."""
    tstates = task_state(st)
    for r in st.tasks.rows:
        if r[1] == "ACTIVE":
            return {"kind": "task", "id": r[0], "title": r[2], "resume": True}
    for r in st.tasks.rows:
        if r[1] == "READY" and deps_done(r, tstates):
            return {"kind": "task", "id": r[0], "title": r[2], "resume": False}
    pstates = phase_state(st)
    for i, pid in enumerate(WORK_ORDER):
        if pstates.get(pid) != "NOT-STARTED":
            continue
        prev = WORK_ORDER[i - 1] if i else None
        if prev is None or pstates.get(prev) in {"DONE", "GATE-OWNER"} or phase_stalled(st, prev):
            return {"kind": "advance", "phase": pid,
                    "note": f"plan {pid}; branch from main if {prev} is merged, else from the {prev} branch"}
        break
    waiting = [r[0] for r in st.tasks.rows if r[1] in {"OWNER", "BLOCKED"}]
    return {"kind": "none", "reason": "no READY task with satisfied dependencies and no phase to advance",
            "waiting": waiting}


def summary(st: Status) -> dict:
    counts: Dict[str, int] = {}
    for r in st.tasks.rows:
        counts[r[1]] = counts.get(r[1], 0) + 1
    return {"phases": phase_state(st), "task_counts": counts, "next": next_item(st), "errors": st.errors}


def cmd_set(st: Status, tid: str, state: str, note: Optional[str]) -> None:
    if state not in TASK_STATES:
        raise SystemExit(f"status: unknown task state {state}")
    for r in st.tasks.rows:
        if r[0] == tid:
            if state == "ACTIVE":
                for other in st.tasks.rows:
                    if other[1] == "ACTIVE" and other[0] != tid:
                        raise SystemExit(f"status: {other[0]} is already ACTIVE")
            r[1] = state
            if note is not None:
                r[4] = note.replace("|", "/")
            return
    raise SystemExit(f"status: task {tid} not found")


def cmd_add(st: Status, tid: str, title: str, depends: str, state: str, note: str) -> None:
    if not TASK_ID.match(tid):
        raise SystemExit(f"status: bad task id {tid} (use P<phase>.<n> or P<phase>.G)")
    if any(r[0] == tid for r in st.tasks.rows):
        raise SystemExit(f"status: task {tid} exists")
    if state not in TASK_STATES:
        raise SystemExit(f"status: unknown task state {state}")
    st.tasks.rows.append([tid, state, title.replace("|", "/"), depends or "-", note.replace("|", "/")])


def cmd_phase(st: Status, pid: str, state: str, note: Optional[str]) -> None:
    if state not in PHASE_STATES:
        raise SystemExit(f"status: unknown phase state {state}")
    for r in st.phases.rows:
        if r[0].split()[0] == pid:
            r[2] = state
            if note is not None:
                r[3] = note.replace("|", "/")
            return
    raise SystemExit(f"status: phase {pid} not found")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Read and update plans/STATUS.md")
    p.add_argument("--file", default=str(STATUS), help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("summary", "next"):
        s = sub.add_parser(name)
        s.add_argument("--json", action="store_true")
    s = sub.add_parser("set")
    s.add_argument("id")
    s.add_argument("state")
    s.add_argument("--note")
    s = sub.add_parser("add")
    s.add_argument("id")
    s.add_argument("title")
    s.add_argument("--depends", default="-")
    s.add_argument("--state", default="READY")
    s.add_argument("--note", default="")
    s = sub.add_parser("phase")
    s.add_argument("id")
    s.add_argument("state")
    s.add_argument("--note")
    sub.add_parser("check")
    a = p.parse_args(argv)
    path = Path(a.file)
    st = load(path)
    if a.cmd in ("summary", "next"):
        data = summary(st) if a.cmd == "summary" else next_item(st)
        if a.json:
            print(json.dumps(data, indent=2))
        else:
            print(json.dumps(data))
        return 1 if st.errors else 0
    if a.cmd == "check":
        for e in st.errors:
            print("status: " + e, file=sys.stderr)
        return 1 if st.errors else 0
    if st.errors:
        for e in st.errors:
            print("status: " + e, file=sys.stderr)
        return 1
    if a.cmd == "set":
        cmd_set(st, a.id, a.state, a.note)
    elif a.cmd == "add":
        cmd_add(st, a.id, a.title, a.depends, a.state, a.note)
    elif a.cmd == "phase":
        cmd_phase(st, a.id, a.state, a.note)
    validate(st)
    if st.errors:
        for e in st.errors:
            print("status: " + e, file=sys.stderr)
        return 1
    save(st, path)
    print(json.dumps(next_item(st)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
