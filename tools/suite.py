#!/usr/bin/env python3
"""suite: start LASSI's full RNGD suite on alpha01 as one rx job, then follow, stop, and pull it.

The owner's tool, so the owner can launch the full suite unattended, with a help option on every command
(the owner's direction of 2026-10-26). `launch` checks the tree and the build
host, shows the plan, asks, and starts plans/runs/suite/run-suite.sh as one big rx job, which downloads the
models, builds the TT assets, runs plans/runs/rngd-sweep/run.sh once per model and recipe directory (one model
served at a time on RNGD cards, every TT program on ttsim, a simulator), and writes the tables. `plan` prints
the same plan and contacts nothing. Every child process runs with MSYS_NO_PATHCONV=1, so Git Bash never rewrites
an argument that looks like a POSIX path, and PYTHONIOENCODING=utf-8 (plans/LESSONS.md, alpha01). Standard
library only, apart from the repository's recipe loader. plans/runs/suite/README.md is the guide.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REMOTE_SCRIPT = "plans/runs/suite/run-suite.sh"
DEFAULT_TABLE = "plans/runs/suite/models.tsv"
DEFAULT_SWEEP = "projects/lassi-demo/sweep8"
DEFAULT_SLOT = "suite"
DEFAULT_NAME = "suite"
OUTPUT_DIR = "lassi-runs/suite"  # run-suite.sh writes $LASSI_RUNS_ROOT/suite/<job id>
FLOOR_GB = 60  # the gate's free-space floor for big jobs (tools/server/config.default.json, min_free_gb_big)
MARGIN_GB = 5  # run-night.sh's margin on the shared disk
TIMEOUT_MARGIN_S = 900  # the night of 2026-10-25: deadline 01:45, rx timeout at 02:00
GATE_MAX_TIMEOUT_S = 172800  # tools/server/config.default.json, job_timeout_max_s
CALL_MIN_S = 1800  # run.sh serves no model with under 30 minutes left
STOP_FILE = '"$LASSI_RUNS_ROOT/rngd-sweep/STOP"'  # run.sh's stop file, in the gate's environment
STOP_COMMAND = f'mkdir -p "$LASSI_RUNS_ROOT/rngd-sweep" && touch {STOP_FILE} && ls -l {STOP_FILE}'
CLEAR_COMMAND = f'rm -f {STOP_FILE} && echo "the STOP file is cleared"'
RUNNING = ("starting", "running")
JOB_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[A-Za-z0-9._-]{1,40}-[0-9a-f]{4}$")  # the gate's ID_RE
NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,40}$")  # a job name the gate puts in the id
SLOT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")  # the gate's SLOT_RE
WORD_RE = re.compile(r"^[A-Za-z0-9._-]+$")
GLOB_RE = re.compile(r"^[A-Za-z0-9._*?\[\]-]+$")
PATH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
# The gate refuses a command naming these while Tenstorrent silicon is present and disabled (gate.py, tt_guard).
TT_GUARD = re.compile(r"tt-metal|tt_metal|ttnn|metal_example|TT_METAL")
SILICON_MESSAGE = (
    "--silicon is refused: this launcher never runs model-generated programs on the Wormhole cards. On alpha01 the "
    "cards' IOMMU groups run in identity (pass-through) mode, in which a program that holds a card can make it read "
    "or write any host memory with ordinary memory stores the sandbox's system-call filter cannot see "
    "(docs/tt-silicon-iommu.md; OQ-078, open). Running candidates on silicon inside the correction loop also needs "
    "an owner-worded change to Agent Rule 9 and its prerequisites (OQ-099, open). Every TT program of this suite "
    "runs on ttsim, a simulator."
)


class Refusal(Exception):
    """A request the launcher will not carry out; `code` is the exit status (2 usage, 1 a failed check)."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


def now_epoch() -> int:
    """Return the current time in epoch seconds (tests replace it)."""
    return int(time.time())


def rip_stop_present() -> bool:
    """Return True when the repository's stop file for unattended runs, .rip-stop, exists."""
    return (REPO / ".rip-stop").exists()


# ---------------------------------------------------------------- US Eastern time


def _sunday(year: int, month: int, nth: int) -> datetime:
    """Return 02:00 on the nth Sunday of the month (naive)."""
    first = datetime(year, month, 1, 2)
    return first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (nth - 1))


def eastern_dst(local: datetime) -> bool:
    """Return True when the naive US Eastern clock time `local` falls in daylight time (EDT)."""
    naive = local.replace(tzinfo=None)
    return _sunday(naive.year, 3, 2) <= naive < _sunday(naive.year, 11, 1)


def eastern_zone(local: datetime) -> timezone:
    """Return EDT (UTC-4) or EST (UTC-5) for the naive US Eastern clock time `local`."""
    return timezone(timedelta(hours=-4), "EDT") if eastern_dst(local) else timezone(timedelta(hours=-5), "EST")


def to_eastern(epoch: int) -> datetime:
    """Return `epoch` as an aware US Eastern datetime."""
    utc = datetime.fromtimestamp(epoch, timezone.utc)
    daylight = utc.astimezone(timezone(timedelta(hours=-4), "EDT"))
    return daylight if eastern_dst(daylight) else utc.astimezone(timezone(timedelta(hours=-5), "EST"))


def show_time(epoch: int) -> str:
    """Return `epoch` as 'YYYY-MM-DD HH:MM EDT' (or EST)."""
    local = to_eastern(epoch)
    return f"{local:%Y-%m-%d %H:%M} {local.tzname()}"


def parse_deadline(text: str, now: int) -> int:
    """Return the epoch of --until: epoch seconds, 'HH:MM' (its next time after `now`), or 'YYYY-MM-DD HH:MM'.

    Clock times are US Eastern: EDT (UTC-4) until the first Sunday of November, EST (UTC-5) after it.
    """
    text = text.strip()
    if text.isdigit():
        return int(text)
    match = re.fullmatch(r"(?:(\d{4})-(\d{2})-(\d{2})[ T])?(\d{1,2}):(\d{2})", text)
    if not match:
        raise Refusal(f"--until {text!r}: give an Eastern clock time HH:MM, a date and time YYYY-MM-DD HH:MM, "
                      "or epoch seconds", 2)
    hour, minute = int(match[4]), int(match[5])
    try:
        if match[1]:
            days = [datetime(int(match[1]), int(match[2]), int(match[3]), hour, minute)]
        else:
            today = to_eastern(now).replace(tzinfo=None)
            day = datetime(today.year, today.month, today.day, hour, minute)
            days = [day, day + timedelta(days=1)]
    except ValueError as exc:
        raise Refusal(f"--until {text!r}: {exc}", 2) from None
    epochs = [int(day.replace(tzinfo=eastern_zone(day)).timestamp()) for day in days]
    return next((epoch for epoch in epochs if epoch > now), epochs[-1])


# ---------------------------------------------------------------- child processes


def child_env() -> dict[str, str]:
    """Return the environment of every child: this one plus MSYS_NO_PATHCONV=1 and PYTHONIOENCODING=utf-8."""
    env = dict(os.environ)
    env["MSYS_NO_PATHCONV"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_child(argv: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    """Run `argv` from the repository root with child_env(); capture its text output when `capture`."""
    try:
        if capture:
            return subprocess.run(argv, cwd=REPO, env=child_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, encoding="utf-8", errors="replace")
        return subprocess.run(argv, cwd=REPO, env=child_env())
    except FileNotFoundError:
        raise Refusal(f"{argv[0]} is not on PATH; run this from a shell where `{argv[0]}` works", 2) from None


def rx(args: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    """Run `uv run tools/rx.py <args>`."""
    return run_child(["uv", "run", "tools/rx.py", *args], capture)


def rx_ok(args: list[str]) -> subprocess.CompletedProcess:
    """Run rx with `args` and print its output; raise Refusal (1) when it fails."""
    result = rx(args)
    sys.stdout.write(result.stdout or "")
    if result.returncode != 0:
        sys.stderr.write(result.stderr or "")
        raise Refusal(f"`uv run tools/rx.py {' '.join(args)}` failed (exit {result.returncode}); see above", 1)
    return result


def git(*args: str) -> str:
    """Return the stripped output of `git -C <repo> <args>`; raise Refusal when git fails."""
    result = run_child(["git", "-C", str(REPO), *args])
    if result.returncode != 0:
        raise Refusal(f"git {' '.join(args)} failed: {(result.stderr or '').strip()}", 2)
    return (result.stdout or "").strip()


@dataclass(frozen=True)
class Tree:
    """The local tree's state: HEAD, its branch, uncommitted changes, and whether an origin branch holds HEAD."""

    head: str
    branch: str
    dirty: bool
    pushed: bool


def read_tree() -> Tree:
    """Read the tree's state with git (read-only)."""
    dirty = bool(git("status", "--porcelain"))
    head, branch = git("rev-parse", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD")
    return Tree(head=head, branch=branch, dirty=dirty, pushed=bool(git("branch", "-r", "--contains", "HEAD")))


def tree_label(tree: Tree) -> str:
    """Return how a run from `tree` may be reported."""
    if tree.dirty:
        return ("exploratory: the tree is dirty, so rx sends a snapshot commit; nothing from this run is ever "
                "[MEASURED]")
    if not tree.pushed:
        return "the commit is on no origin branch; push it before citing this run"
    return ("a clean commit on origin: development results on tt-pairs-v0 (OQ-083 (b)), cited only from the "
            "provenance that pull writes")


def check_tree(tree: Tree, allow_dirty: bool) -> None:
    """Refuse a dirty tree or an unpushed commit unless `allow_dirty`."""
    if allow_dirty:
        return
    if tree.dirty:
        raise Refusal("the working tree is dirty (git status --porcelain lists changes). Commit or stash them and "
                      "launch from a clean, pushed commit, for example from a clean worktree of it "
                      "(plans/LESSONS.md, alpha01), or pass --allow-dirty to run anyway: exploratory, never "
                      "[MEASURED]", 1)
    if not tree.pushed:
        raise Refusal(f"HEAD {tree.head[:12]} is on no origin branch (git branch -r --contains HEAD): push it "
                      "first, or pass --allow-dirty to run anyway", 1)


# ---------------------------------------------------------------- the plan


@dataclass(frozen=True)
class ModelRow:
    """One row of a model table: run.sh's six columns and the revision's size in bytes (None when unknown)."""

    slug: str
    model: str
    revision: str
    cards: str
    parser: str
    venv: str
    size: int | None


def read_table(path: Path, shown: str) -> list[ModelRow]:
    """Read a model table of six or seven tab-separated columns, skipping comments and blank lines."""
    if not path.is_file():
        raise Refusal(f"no model table {shown}", 2)
    rows = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.split("\t")
        if len(cells) not in (6, 7) or not WORD_RE.match(cells[0]):
            raise Refusal(f"{shown}: a row needs 6 or 7 tab-separated columns and a plain slug: {line!r}", 2)
        size = int(cells[6]) if len(cells) == 7 and cells[6].isdigit() else None
        rows.append(ModelRow(*cells[:6], size))
    return rows


def select_models(rows: list[ModelRow], text: str | None, shown: str) -> list[ModelRow]:
    """Return the rows --models names, in its order (every row when it is not given); refuse unknown names."""
    known = {row.slug: row for row in rows}
    if not text:
        return rows
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in known]
    if unknown:
        raise Refusal(f"unknown model(s) {', '.join(unknown)}; {shown} has: {', '.join(known)}", 2)
    if len(set(names)) != len(names):
        raise Refusal(f"--models names a model twice: {text}", 2)
    return [known[name] for name in names]


def repo_path(text: str, what: str) -> str:
    """Return `text` as a POSIX path relative to the repository root; refuse one outside it."""
    path = Path(text)
    full = (path if path.is_absolute() else REPO / path).resolve()
    try:
        shown = full.relative_to(REPO.resolve()).as_posix()
    except ValueError:
        raise Refusal(f"{what} {text} is outside the repository {REPO}; the job runs in a checkout of it", 2) from None
    if not PATH_RE.match(shown):
        raise Refusal(f"{what} {shown}: use only letters, digits, '.', '_', '-', and '/'", 2)
    return shown


def sweeps_of(args: argparse.Namespace) -> list[str]:
    """Return the recipe directories of --sweep, or the default one."""
    return list(args.sweep) if args.sweep else [DEFAULT_SWEEP]


def load_sweep(sweep: str, rows: list[ModelRow], only: list[str]) -> dict[str, list[str]]:
    """Load every recipe under `sweep` with the repository's loader; return each model's selected recipe names.

    A recipe that does not load, or a selected one whose model id is not the table's, is refused.
    """
    from lassi.core import runner  # noqa: F401  (importing the runner registers every component)
    from lassi.core.recipe import RecipeError, load_recipe

    root = REPO / sweep
    if not root.is_dir():
        raise Refusal(f"no recipe directory {sweep}", 2)
    loaded = {}
    for path in sorted(root.rglob("*.yaml")):
        try:
            loaded[path] = load_recipe(path)
        except RecipeError as exc:
            shown = path.relative_to(REPO).as_posix()
            raise Refusal(f"{shown} does not load with the repository's recipe loader: {exc}", 2) from None
    selected = {}
    for row in rows:
        names = []
        for path in sorted((root / row.slug).glob("*.yaml")):
            if only and not any(fnmatch.fnmatchcase(path.stem, glob) for glob in only):
                continue
            named = loaded[path].data.get("model", {}).get("id")
            if named != row.model:
                raise Refusal(f"{path.relative_to(REPO).as_posix()} names model {named}, but the table serves "
                              f"{row.model} as {row.slug}", 2)
            names.append(path.stem)
        selected[row.slug] = names
    return selected


@dataclass(frozen=True)
class Plan:
    """Everything launch does: the remote script's arguments, the rx job's options, and what they select."""

    rows: tuple[ModelRow, ...]
    table: str
    sweeps: tuple[str, ...]
    only: tuple[str, ...]
    recipes: dict[tuple[str, str], list[str]]  # (slug, sweep) -> recipe names, in run order
    deadline: int
    slot: str
    name: str
    skip_download: bool

    @property
    def count(self) -> int:
        """The number of recipes the job runs."""
        return sum(len(names) for names in self.recipes.values())

    @property
    def need_gb(self) -> int:
        """The free space the preflight requires: the gate's big-job floor and the margin."""
        return FLOOR_GB + MARGIN_GB

    @property
    def download_gb(self) -> int:
        """The most a download may add: every byte of each revision with a known size (0 with --skip-download)."""
        if self.skip_download:
            return 0
        return math.ceil(sum(row.size or 0 for row in self.rows) / 1e9)

    def timeout(self, now: int) -> int:
        """The rx job's timeout in seconds: the deadline plus TIMEOUT_MARGIN_S, counted from `now`."""
        return self.deadline - now + TIMEOUT_MARGIN_S

    def remote_command(self) -> str:
        """The command the job runs in its slot."""
        parts = ["bash", REMOTE_SCRIPT, "--deadline", str(self.deadline), "--table", self.table,
                 "--models", ",".join(row.slug for row in self.rows)]
        for sweep in self.sweeps:
            parts += ["--sweep", sweep]
        for glob in self.only:
            parts += ["--only", f'"{glob}"']
        if self.skip_download:
            parts.append("--skip-download")
        return " ".join(parts)

    def rx_args(self, now: int) -> list[str]:
        """The arguments of `uv run tools/rx.py` that start the job."""
        return ["job", "start", "--big", "--slot", self.slot, "--name", self.name, "--timeout",
                str(self.timeout(now)), "--", self.remote_command()]

    def rx_line(self, now: int) -> str:
        """The rx command as one line to paste into Git Bash."""
        args = self.rx_args(now)
        return ("MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8 uv run tools/rx.py " + " ".join(args[:-1]) + " "
                + shlex.quote(args[-1]))


def check_deadline(deadline: int, now: int, calls: int) -> None:
    """Refuse a deadline too close for the run.sh calls, or one whose job timeout passes the gate's limit."""
    seconds = deadline - now
    need = CALL_MIN_S * max(calls, 1)
    if seconds < need:
        raise Refusal(f"the deadline {show_time(deadline)} is {seconds} s away; {calls} run.sh call(s) need at least "
                      f"{need} s (run.sh serves no model with under 30 minutes left)", 2)
    if seconds + TIMEOUT_MARGIN_S > GATE_MAX_TIMEOUT_S:
        raise Refusal(f"the job timeout would be {seconds + TIMEOUT_MARGIN_S} s, past the gate's limit of "
                      f"{GATE_MAX_TIMEOUT_S} s (job_timeout_max_s); choose an earlier deadline", 2)


def check_words(args: argparse.Namespace) -> list[str]:
    """Check --only, --name, and --slot; return the globs."""
    for glob in args.only:
        if not GLOB_RE.match(glob):
            raise Refusal(f"--only {glob!r}: use letters, digits, '.', '_', '-', and the glob characters * ? [ ]", 2)
    if not NAME_RE.match(args.name):
        raise Refusal(f"--name {args.name!r}: 1 to 40 letters, digits, '.', '_', or '-'", 2)
    if not SLOT_RE.match(args.slot):
        raise Refusal(f"--slot {args.slot!r}: a letter or digit, then up to 80 letters, digits, '.', '_', or '-'", 2)
    return list(args.only)


def build_plan(args: argparse.Namespace) -> Plan:
    """Check every launch option against the repository and return the plan; raise Refusal on any problem."""
    if args.silicon:
        raise Refusal(SILICON_MESSAGE, 2)
    now = now_epoch()
    table = repo_path(args.table, "model table")
    rows = select_models(read_table(REPO / table, table), args.models, table)
    sweeps = [repo_path(sweep, "recipe directory") for sweep in sweeps_of(args)]
    only = check_words(args)
    recipes: dict[tuple[str, str], list[str]] = {}
    for sweep in sweeps:
        for slug, names in load_sweep(sweep, rows, only).items():
            if names:
                recipes[(slug, sweep)] = names
    for row in rows:
        if not any(slug == row.slug for slug, _ in recipes):
            matching = " matching --only" if only else ""
            raise Refusal(f"no recipe for {row.slug}{matching} in {', '.join(sweeps)} (each holds <slug>/*.yaml)", 2)
    order = {row.slug: index for index, row in enumerate(rows)}
    recipes = dict(sorted(recipes.items(), key=lambda item: (order[item[0][0]], sweeps.index(item[0][1]))))
    deadline = now + round(args.hours * 3600) if args.hours is not None else parse_deadline(args.until, now)
    check_deadline(deadline, now, len(recipes))
    plan = Plan(rows=tuple(rows), table=table, sweeps=tuple(sweeps), only=tuple(only), recipes=recipes,
                deadline=deadline, slot=args.slot, name=args.name, skip_download=args.skip_download)
    if TT_GUARD.search(plan.remote_command()):
        raise Refusal("the remote command names tt-metal words the gate refuses while silicon is disabled; rename "
                      "the recipe directory or table", 2)
    return plan


def render_plan(plan: Plan, tree: Tree, now: int) -> str:
    """Return the plan as the lines plan and launch print."""
    branch = "a detached HEAD" if tree.branch == "HEAD" else tree.branch
    lines = ["Plan (plan starts nothing; launch runs exactly this):",
             f"  tree         HEAD {tree.head[:12]} on {branch}: {'dirty' if tree.dirty else 'clean'}, "
             f"{'on origin' if tree.pushed else 'on no origin branch'}",
             f"  label        {tree_label(tree)}"]
    for index, row in enumerate(plan.rows):
        size = f"{row.size / 1e9:.1f} GB" if row.size else "size unknown: never downloaded, served only if cached"
        lines.append(f"  {'models' if index == 0 else '':12} {row.slug}: {row.model}@{row.revision} on {row.cards} "
                     f"RNGD card(s), venv {row.venv}, {size}")
    lines += [f"  table        {plan.table}",
              f"  sweeps       {', '.join(plan.sweeps)}",
              f"  only         {', '.join(plan.only) if plan.only else 'every recipe'}",
              f"  recipes      {plan.count} recipes over {len(plan.recipes)} run.sh call(s), one model served at a "
              "time, in this order:"]
    for (slug, sweep), names in plan.recipes.items():
        lines.append(f"               {slug} over {sweep}: {len(names)} ({', '.join(names)})")
    downloads = ("none (--skip-download; run.sh skips a model not in the cache)" if plan.skip_download else
                 f"up to {plan.download_gb} GB more for revisions not yet cached; each download runs only while "
                 f"{plan.need_gb} GB stay free after its missing bytes, and run.sh skips a model not cached")
    lines += [f"  deadline     {show_time(plan.deadline)} (epoch {plan.deadline}); now {show_time(now)}",
              f"  job timeout  {plan.timeout(now)} s (the deadline plus {TIMEOUT_MARGIN_S} s), recomputed at start",
              f"  free space   the preflight needs {plan.need_gb} GB free on the scratch disk (the gate's {FLOOR_GB} "
              f"GB big-job floor and a {MARGIN_GB} GB margin); downloads: {downloads}",
              "  devices      RNGD cards claimed only at 0.00 GiB, served on port 8123; every TT program on ttsim, a "
              "simulator; no Wormhole card",
              f"  output       lassi-wt/{plan.slot} (the job's checkout), {OUTPUT_DIR}/<job id> (pull fetches it), "
              "lassi-runs/rngd-sweep/<pass>, lassi-runs/runs/sweep-<pass>-*",
              f"  remote       {plan.remote_command()}",
              f"  rx command   {plan.rx_line(now)}"]
    return "\n".join(lines)


# ---------------------------------------------------------------- subcommands


def do_plan(args: argparse.Namespace) -> int:
    """Print the plan; contact nothing."""
    plan = build_plan(args)
    print(render_plan(plan, read_tree(), now_epoch()))
    return 0


def doctor_json(text: str) -> dict | None:
    """Return the JSON block `rx doctor` prints (a line '{' to the next line '}'), or None."""
    lines = text.splitlines()
    if "{" not in lines:
        return None
    start = lines.index("{")
    end = next((index for index in range(start, len(lines)) if lines[index] == "}"), None)
    if end is None:
        return None
    try:
        return json.loads("\n".join(lines[start:end + 1]))
    except ValueError:
        return None


def running_jobs() -> list[str]:
    """Return the ids `rx job status` lists as starting or running."""
    result = rx(["job", "status"])
    if result.returncode != 0:
        raise Refusal(f"rx job status failed (exit {result.returncode}): {(result.stderr or '').strip()[-300:]}", 1)
    found = []
    for line in (result.stdout or "").splitlines():
        words = line.split()
        if len(words) >= 2 and JOB_ID_RE.match(words[0]) and words[1] in RUNNING:
            found.append(words[0])
    return found


def preflight(plan: Plan) -> None:
    """Check the build host through rx doctor and rx job status; raise Refusal (1) with what to do."""
    result = rx(["doctor"])
    doctor = doctor_json(result.stdout or "")
    if doctor is None:
        tail = ((result.stdout or "") + (result.stderr or "")).strip()[-400:]
        raise Refusal(f"rx doctor did not answer (exit {result.returncode}): {tail}. Check that ssh to the build "
                      "host works (uv run tools/rx.py doctor), then launch again", 1)
    if doctor.get("stop"):
        raise Refusal("the gate's owner STOP is set on alpha01, so it refuses every job; the owner clears it", 1)
    if not (doctor.get("devices_enabled") or {}).get("rngd"):
        raise Refusal("the gate's rngd device class is disabled (rx doctor, devices_enabled); the owner enables it "
                      "in the gate's config.json on alpha01 for the window, then launch again", 1)
    free = float(doctor.get("scratch_free_gb") or 0)
    if free < plan.need_gb:
        raise Refusal(f"the scratch disk has {free} GB free, under the {plan.need_gb} GB the job needs (the gate's "
                      f"{FLOOR_GB} GB floor and a {MARGIN_GB} GB margin); free space first", 1)
    running = running_jobs()
    if running:
        raise Refusal(f"rx job status lists a running job: {', '.join(running)}. The gate runs one big job at a "
                      "time and rx job status does not say which jobs are big, so any running job stops the launch; "
                      "wait for it (uv run tools/suite.py wait <id>) or stop it first", 1)


def confirm() -> bool:
    """Ask before starting; True only on y or yes."""
    try:
        answer = input("Start this on alpha01? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def start_job(plan: Plan) -> int:
    """Start the job with rx and print how to watch, stop, and pull it."""
    now = now_epoch()
    check_deadline(plan.deadline, now, len(plan.recipes))
    result = rx_ok(plan.rx_args(now))
    started = next((line for line in reversed((result.stdout or "").splitlines()) if line.startswith("{")), "{}")
    job = json.loads(started).get("id")
    if not job:
        raise Refusal("rx job start printed no job id; check `uv run tools/suite.py status`", 1)
    tool = "uv run tools/suite.py"
    print(f"\nStarted {job} on alpha01 in slot {plan.slot}; it ends by {show_time(plan.deadline + TIMEOUT_MARGIN_S)}.")
    print(f"  watch: {tool} status {job}")
    print(f"         {tool} wait {job}")
    print(f"  stop:  {tool} stop    (the sweep stops before its next model; running recipes end at their time limit)")
    print(f"  pull:  {tool} pull {job}    (after it ends)")
    return 0


def do_launch(args: argparse.Namespace) -> int:
    """Check, show the plan, ask (unless --yes), and start the job."""
    plan = build_plan(args)
    if rip_stop_present():
        raise Refusal(".rip-stop exists at the repository root (the stop file for unattended runs); remove it if "
                      "you mean to launch", 1)
    tree = read_tree()
    check_tree(tree, args.allow_dirty)
    preflight(plan)
    print(render_plan(plan, tree, now_epoch()))
    if not args.yes and not confirm():
        print("Not started.")
        return 1
    return start_job(plan)


def do_status(args: argparse.Namespace) -> int:
    """Show recent jobs, or one job and the last lines of its log."""
    result = rx(["job", "status", *([args.job] if args.job else [])], capture=False)
    if args.job and args.lines > 0 and result.returncode == 0:
        result = rx(["job", "tail", args.job, "-n", str(args.lines)], capture=False)
    return result.returncode


def do_wait(args: argparse.Namespace) -> int:
    """Wait for a job through rx job wait; its exit status passes through (3: still running)."""
    words = ["job", "wait", args.job, "--timeout", str(args.timeout), "--interval", str(args.interval)]
    return rx(words, capture=False).returncode


def do_stop(args: argparse.Namespace) -> int:
    """Create (or with --clear remove) the sweep's STOP file through one rx exec."""
    return rx(["exec", "--", CLEAR_COMMAND if args.clear else STOP_COMMAND], capture=False).returncode


# ---------------------------------------------------------------- pull


def job_state(job: str) -> str:
    """Return the state `rx job status <job>` reports."""
    result = rx(["job", "status", job])
    for line in (result.stdout or "").splitlines():
        words = line.split()
        if len(words) >= 2 and words[0] == job:
            return words[1]
    raise Refusal(f"rx job status {job} listed no such job (exit {result.returncode}): "
                  f"{(result.stderr or '').strip()[-300:]}", 1)


def fetch_output(job: str) -> Path:
    """Pull the job's output directory into .rx/pulls/<job>, after moving aside what that folder held.

    `rx pull <job>` extracts the job record into the same folder (both archives are named by the job id), and an
    earlier pull may have left files there, so the folder is moved aside first (plans/LESSONS.md, alpha01).
    """
    cache = REPO / ".rx" / "pulls" / job
    if cache.exists():
        aside = cache.with_name(f"{job}.aside-{now_epoch()}")
        cache.rename(aside)
        print(f"moved .rx/pulls/{job} (the job record, or an earlier pull) aside to {aside.name}")
    rx_ok(["pull", "--path", f"{OUTPUT_DIR}/{job}"])
    if not cache.is_dir():
        raise Refusal(f"rx pull left no directory {cache}", 1)
    return cache


def copy_ascii(source: Path, target: Path) -> list[str]:
    """Copy every plain-ASCII file under `source` to `target`; return the files left out, with their sizes."""
    left_out = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        data = path.read_bytes()
        if not data.isascii():
            left_out.append(f"{relative.as_posix()} ({len(data)} bytes)")
            continue
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target / relative)
    return left_out


# Lines of the job's own log that say a recipe or a whole recipe directory was left out (run-suite.sh).
LEFT_OUT_MARKS = ("did not build as tracked", "SKIP ", "STOP LINE", "STOP file", "no more calls")


def keep_job_log(cache: Path, target: Path) -> list[str]:
    """Copy the job's own log (output.log of `rx pull <job>`) to `target`/output.log; return its left-out lines.

    run-suite.sh says there, and only there, when an asset did not build as tracked or a recipe directory had
    no recipe to run, so a recipe set that never ran shows in the pull. Bytes that are not plain ASCII (progress
    bars) become '?', so the copy is plain ASCII.
    """
    log = cache / "output.log"
    if not log.is_file():
        return []
    text = log.read_bytes().decode("utf-8", "replace").encode("ascii", "replace").decode("ascii")
    target.mkdir(parents=True, exist_ok=True)
    (target / "output.log").write_text(text, encoding="ascii", newline="\n")
    return [line.strip() for line in text.splitlines() if any(mark in line for mark in LEFT_OUT_MARKS)]


def read_tsv(path: Path) -> list[dict[str, str]]:
    """Return a TSV file's rows as dicts keyed by its header."""
    with path.open(encoding="ascii", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def print_counts(results: Path) -> None:
    """Print this job's rows of tables/passes.tsv (its passes, from calls.tsv) and the provenance's label."""
    night = results / "night"
    passes, calls = night / "tables" / "passes.tsv", night / "calls.tsv"
    if not passes.is_file():
        print("no tables/passes.tsv in the pull: the job ended before its tables")
        return
    stamps = {row["stamp"] for row in read_tsv(calls)} if calls.is_file() else None
    rows = [row for row in read_tsv(passes) if stamps is None or row["pass"] in stamps]
    columns = ("pass", "slug", "combo", "trials", "s5", "alignment_1", "with_divergence", "sim_gap_attempts")
    print("\nThis job's passes (copied from tables/passes.tsv; development results on tt-pairs-v0, TT programs on "
          "ttsim, a simulator):")
    print("\t".join(columns))
    totals: dict[str, list[int]] = {}
    for row in rows:
        print("\t".join(row.get(column, "-") for column in columns))
        entry = totals.setdefault(row["slug"], [0, 0, 0])
        for index, column in enumerate(("trials", "s5", "alignment_1")):
            entry[index] += int(row.get(column) or 0)
    for slug, (trials, s5, exact) in totals.items():
        print(f"{slug}: {trials} trials, {s5} at S5, {exact} at alignment 1.0")
    provenance = results / "provenance.json"
    if provenance.is_file():
        meta = json.loads(provenance.read_text(encoding="utf-8"))
        label = "exploratory (dirty tree), never [MEASURED]" if meta.get("dirty") else "clean commit"
        print(f"provenance: commit {meta.get('commit')}, {label}, state {meta.get('state')}, rc {meta.get('rc')}")


def default_results_name(job: str) -> str:
    """Return results/<name> for a job id: its name, date, and time, as in suite-20261026-170035."""
    parts = job.split("-")
    return "-".join([*parts[2:-1], parts[0], parts[1]])


def do_pull(args: argparse.Namespace) -> int:
    """Pull a finished job's provenance, its own log, and its output directory into results/<name>.

    Then print its counts and the lines of its log that say it left an asset or a recipe set out.
    """
    job = args.job
    if not JOB_ID_RE.match(job):
        raise Refusal(f"{job!r} is not an rx job id (like 20261025-170035-night-395c)", 2)
    name = args.name or default_results_name(job)
    if not WORD_RE.match(name):
        raise Refusal(f"--name {name!r}: use letters, digits, '.', '_', and '-'", 2)
    target = REPO / "results" / name
    if target.exists() and any(target.iterdir()):
        raise Refusal(f"results/{name} exists and is not empty; pass --name for a new folder", 2)
    state = job_state(job)
    if state in RUNNING:
        raise Refusal(f"job {job} is still {state}; pull it after it ends (uv run tools/suite.py wait {job})", 1)
    rx_ok(["pull", job, "--into", f"results/{name}"])
    skipped = keep_job_log(REPO / ".rx" / "pulls" / job, target)
    left_out = copy_ascii(fetch_output(job), target / "night")
    print(f"copied the job's output directory into results/{name}/night")
    for entry in left_out:
        print(f"left out (not plain ASCII): {entry}")
    print_counts(target)
    if skipped:
        print(f"\nLeft out by the job (its output.log, kept in results/{name}):")
        for line in skipped:
            print(f"  {line}")
    return 0


# ---------------------------------------------------------------- the parser

TOP_DESCRIPTION = """Start LASSI's full RNGD suite on alpha01 as one rx job, then follow, stop, and pull it.

launch checks the tree and the build host, prints the plan, asks, and starts plans/runs/suite/run-suite.sh as
one big rx job: it downloads the models, builds the TT assets, runs plans/runs/rngd-sweep/run.sh once per model
and recipe directory (one model served at a time on RNGD cards; every TT program on ttsim, a simulator, never a
Wormhole card), and writes the tables. plan prints the same plan and contacts nothing. Every child process runs
with MSYS_NO_PATHCONV=1 and PYTHONIOENCODING=utf-8."""

TOP_EPILOG = """Examples:
  uv run tools/suite.py plan --until 01:45
  uv run tools/suite.py launch --until 01:45
  uv run tools/suite.py launch --hours 6 --models gpt-oss-120b --only 'v1*' --yes
  uv run tools/suite.py status
  uv run tools/suite.py wait 20261026-170035-suite-ab12
  uv run tools/suite.py stop
  uv run tools/suite.py pull 20261026-170035-suite-ab12

Run it from the repository root of a clean, pushed commit (a clean worktree of the commit keeps other work out of
the job). `<command> --help` documents each command; plans/runs/suite/README.md is the guide."""

RUN_EPILOG = """Examples:
  uv run tools/suite.py {name} --until 01:45
  uv run tools/suite.py {name} --hours 5.5 --models llama33-70b
  uv run tools/suite.py {name} --until "2026-10-27 01:45" --sweep projects/lassi-demo/sweep7 --only 'all-*'
  uv run tools/suite.py {name} --until 01:45 --skip-download --slot night --name night2

Clock times are US Eastern (EDT, or EST after the first Sunday of November); HH:MM means its next time. Quote a
glob so the shell does not expand it."""


def add_run_options(parser: argparse.ArgumentParser) -> None:
    """Add the options plan and launch share."""
    parser.add_argument("--models", metavar="LIST", help="comma-separated model slugs from --table, served one at a "
                        "time in this order (default: every row; the default table holds llama33-70b and "
                        "gpt-oss-120b, the night's two rows); an unknown name is refused")
    parser.add_argument("--table", metavar="TSV", default=DEFAULT_TABLE, help="the model table: run.sh's six "
                        "columns and the revision's size in bytes (default: %(default)s)")
    parser.add_argument("--sweep", metavar="DIR", action="append", help="a recipe directory holding "
                        f"<slug>/<combo>-<half>.yaml; repeat for more, run in order (default: {DEFAULT_SWEEP}). "
                        "Every recipe in it must load with the repository's recipe loader")
    parser.add_argument("--only", metavar="GLOB", action="append", default=[], help="run only the recipes whose "
                        "name (<combo>-<half>, as v1ex-maxabs) matches GLOB; repeat for more (default: every recipe)")
    deadline = parser.add_mutually_exclusive_group(required=True)
    deadline.add_argument("--hours", type=float, metavar="H", help="the deadline H hours from now (this or --until)")
    deadline.add_argument("--until", metavar="WHEN", help="the deadline: an Eastern clock time HH:MM, a date and "
                          "time 'YYYY-MM-DD HH:MM', or epoch seconds (this or --hours). The rx job's timeout is the "
                          f"deadline plus {TIMEOUT_MARGIN_S} s")
    parser.add_argument("--skip-download", action="store_true", help="download nothing; run.sh skips a model "
                        "whose snapshot is not in the host's cache (default: download what is missing)")
    parser.add_argument("--name", default=DEFAULT_NAME, help="the rx job's name, part of its id (default: "
                        "%(default)s)")
    parser.add_argument("--slot", default=DEFAULT_SLOT, help="the rx worktree slot on alpha01 (default: "
                        "%(default)s)")
    parser.add_argument("--yes", action="store_true", help="launch: start without asking 'Start this on "
                        "alpha01? [y/N]' (plan ignores it)")
    parser.add_argument("--allow-dirty", action="store_true", help="launch even from a dirty tree or a commit on "
                        "no origin branch; such a run is exploratory, never [MEASURED] (default: refuse)")
    parser.add_argument("--silicon", action="store_true", help="refused, exit 2: model-generated programs do not "
                        "run on the Wormhole cards from this launcher (docs/tt-silicon-iommu.md, OQ-078, OQ-099)")


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Keep descriptions and examples as written, and wrap option help without breaking a path at a hyphen."""

    def _split_lines(self, text: str, width: int) -> list[str]:
        return textwrap.wrap(" ".join(text.split()), width, break_on_hyphens=False, break_long_words=False)


def subparser(sub: argparse._SubParsersAction, name: str, summary: str, text: str,
              epilog: str) -> argparse.ArgumentParser:
    """Add one subcommand with its summary, description (wrapped here), and examples."""
    wrapped = textwrap.fill(text, 110, break_on_hyphens=False, break_long_words=False)
    return sub.add_parser(name, help=summary, description=wrapped, epilog=epilog, formatter_class=HelpFormatter)


def add_follow_commands(sub: argparse._SubParsersAction) -> None:
    """Add status, wait, stop, and pull."""
    tool = "uv run tools/suite.py"
    status = subparser(sub, "status", "show recent rx jobs, or one job and the end of its log",
                       "Wrap `rx job status [ID]`; with an id, also `rx job tail ID -n LINES`.",
                       f"Examples:\n  {tool} status\n  {tool} status 20261026-170035-suite-ab12 --lines 80")
    status.add_argument("job", nargs="?", metavar="ID", help="an rx job id (default: list recent jobs)")
    status.add_argument("--lines", type=int, default=40, help="log lines to show for ID, 0 for none "
                        "(default: %(default)s)")
    wait = subparser(sub, "wait", "wait for a job to end (rx job wait)",
                     "Wrap `rx job wait ID`: poll until the job ends and print the end of its log. Exit status: "
                     "the job's, or 3 when it is still running at --timeout.",
                     f"Examples:\n  {tool} wait 20261026-170035-suite-ab12\n"
                     f"  {tool} wait 20261026-170035-suite-ab12 --timeout 600 --interval 60")
    wait.add_argument("job", metavar="ID", help="the rx job id")
    wait.add_argument("--timeout", type=int, default=3600, help="seconds to wait (default: %(default)s)")
    wait.add_argument("--interval", type=int, default=300, help="seconds between polls (default: %(default)s)")
    stop = subparser(sub, "stop", "stop the sweep early (its STOP file, through one rx exec)",
                     "Create $LASSI_RUNS_ROOT/rngd-sweep/STOP on alpha01 with one rx exec. run.sh stops before its "
                     "next model, run-suite.sh before its next call, a running download ends, and the job writes "
                     "its tables and ends; recipes already running end at their time limit. rx has no job stop, and "
                     "rx job kill ends only the job's runner, not its command (OQ-082). While the file exists a new "
                     "job refuses to start; --clear removes it.",
                     f"Examples:\n  {tool} stop\n  {tool} stop --clear")
    stop.add_argument("--clear", action="store_true", help="remove the STOP file instead (owner), so the next "
                      "launch can start")
    pull = subparser(sub, "pull", "pull a finished job into results/<name> and print its counts",
                     "Run `rx pull ID --into results/<name>` (provenance.json), then `rx pull --path "
                     f"{OUTPUT_DIR}/ID` and copy that directory to results/<name>/night, leaving out any file "
                     "that is not plain ASCII (download logs' progress bars), keep the job's own log as "
                     "results/<name>/output.log, and print the job's rows of tables/passes.tsv and the lines "
                     "of that log that say an asset or a recipe set was left out. Write "
                     "results/<name>/summary.md from them afterwards.",
                     f"Examples:\n  {tool} pull 20261026-170035-suite-ab12\n"
                     f"  {tool} pull 20261026-170035-suite-ab12 --name night-20261026")
    pull.add_argument("job", metavar="ID", help="the rx job id")
    pull.add_argument("--name", help="the results folder (default: the job's name, date, and time, as "
                      "suite-20261026-170035)")


def build_parser() -> argparse.ArgumentParser:
    """Return the launcher's argument parser."""
    tool = "uv run tools/suite.py"
    parser = argparse.ArgumentParser(prog=tool, description=TOP_DESCRIPTION, epilog=TOP_EPILOG,
                                     formatter_class=HelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    plan = subparser(sub, "plan", "print what launch would do; contact nothing",
                     "Check the options against the repository (the model table, each recipe directory, every "
                     "recipe loaded) and print the plan: the tree, the models, the recipes, the deadline in "
                     "Eastern time and epoch, the job timeout, the free space needed, the remote script's "
                     "command, and the exact rx command. Nothing is started and nothing contacts alpha01.",
                     RUN_EPILOG.format(name="plan"))
    add_run_options(plan)
    launch = subparser(sub, "launch", "check, show the plan, ask, and start the suite on alpha01",
                       "Preflight (.rip-stop absent; a clean, pushed tree unless --allow-dirty; rx doctor answers, "
                       "lists the rngd device class enabled and enough scratch space; rx job status shows no "
                       "running job), print the plan, ask 'Start this on alpha01? [y/N]' unless --yes, then start "
                       "`uv run tools/rx.py job start --big ...` and print the job id and the commands to watch, "
                       "stop, and pull it.", RUN_EPILOG.format(name="launch"))
    add_run_options(launch)
    add_follow_commands(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand; return its exit status."""
    args = build_parser().parse_args(argv)
    handlers = {"plan": do_plan, "launch": do_launch, "status": do_status, "wait": do_wait, "stop": do_stop,
                "pull": do_pull}
    try:
        return handlers[args.command](args)
    except Refusal as exc:
        print(f"suite: {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
