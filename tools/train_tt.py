#!/usr/bin/env python3
"""train_tt: train the small models on a Wormhole card on alpha01 as one rx job, then follow, stop, and pull it.

The owner's tool (task P17.38), so the owner can start the window's training without a session, with a help option
on every command. The owner (2026-10-26): "I would like to be able to train the smaller models by then, plan to use
the tenstorrents for that task"; OQ-100 (a) with (s2): training runs of tracked code on the cards, started by the
owner; one start runs a sequence of runs, each its own process, spaced, stopping at the first failure or hang.
`launch` checks the tree and the build host, shows the plan, asks, and starts plans/runs/train-tt/run.sh as one big
rx job, which installs or checks the tt-xla venv (CPU only), builds the records and fetches the bases (CPU only),
then runs `lassi --graphics off train <recipe> --silicon` once per recipe, in order. `install` starts the venv's
install alone. `plan` prints the plan and contacts nothing. Every child process runs with MSYS_NO_PATHCONV=1 and
PYTHONIOENCODING=utf-8 (plans/LESSONS.md, alpha01). plans/runs/train-tt/README.md is the guide.
"""

from __future__ import annotations

import argparse
import json
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
REMOTE_SCRIPT = "plans/runs/train-tt/run.sh"
DEFAULT_DIR = "projects/lassi-demo/train-tt"
# The window's order (OQ-102 (3a)): the smoke on the synthetic fixture first, then tt-train-v0.
DEFAULT_RECIPES = (
    "ttxla-smoke-fixture", "ttxla-lora-qwen2.5-0.5b", "ttxla-lora-qwen3-0.6b", "ttxla-lora-qwen3-1.7b",
    "ttxla-full-scratch-77m", "ttxla-prompt-tuning-qwen3-1.7b",
)
DEFAULT_SLOT = "train-tt"
DEFAULT_NAME = "train-tt"
INSTALL_NAME = "tt-xla-venv"
OUTPUT_DIR = "lassi-runs/train-tt"  # run.sh writes $LASSI_RUNS_ROOT/train-tt/<job id>
ALLOWED_ROOT = "/mnt/nvme10/joseph_ufl"  # the scratch root and its stop line in GiB (AGENTS.md, Remote Execution)
STOP_GIB = 345  # the stop line under the owner's lift of the scratch cap (OQ-074, 2026-10-23), as the MLIR installs ran
FLOOR_GB = 60  # the gate's free-space floor for big jobs (tools/server/config.default.json, min_free_gb_big)
MARGIN_GB = 5
TIMEOUT_MARGIN_S = 900  # the rx job's timeout is the deadline plus this
GATE_MAX_TIMEOUT_S = 172800  # tools/server/config.default.json, job_timeout_max_s
SETUP_MIN_S = 900  # PROJECTED: the venv check, the records, and the bases before the first run, when all are there
RUN_MARGIN_S = 600  # run.sh's margin beyond a recipe's trainer.timeout_s
SPACING_S = 120  # the default pause between two runs
INSTALL_TIMEOUT_S = 3600
STOP_FILE = '"$LASSI_RUNS_ROOT/train-tt/STOP"'
STOP_COMMAND = f'mkdir -p "$LASSI_RUNS_ROOT/train-tt" && touch {STOP_FILE} && ls -l {STOP_FILE}'
CLEAR_COMMAND = f'rm -f {STOP_FILE} && echo "the STOP file is cleared"'
RUNNING = ("starting", "running")
JOB_ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[A-Za-z0-9._-]{1,40}-[0-9a-f]{4}$")
NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,40}$")
SLOT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
WORD_RE = re.compile(r"^[A-Za-z0-9._-]+$")
PATH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
# The gate refuses a command naming these while Tenstorrent silicon is present and disabled (gate.py, tt_guard), and
# the device classes' patterns name the others; no command here may hold one.
GATE_WORDS = re.compile(r"tt-metal|tt_metal|ttnn|metal_example|TT_METAL|tt-smi|/dev/tenstorrent|tt-flash|tt-topology")


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


def eastern_zone(local: datetime) -> timezone:
    """Return EDT (UTC-4) or EST (UTC-5) for the naive US Eastern clock time `local`."""
    naive = local.replace(tzinfo=None)
    daylight = _sunday(naive.year, 3, 2) <= naive < _sunday(naive.year, 11, 1)
    return timezone(timedelta(hours=-4), "EDT") if daylight else timezone(timedelta(hours=-5), "EST")


def to_eastern(epoch: int) -> datetime:
    """Return `epoch` as an aware US Eastern datetime."""
    daylight = datetime.fromtimestamp(epoch, timezone.utc).astimezone(timezone(timedelta(hours=-4), "EDT"))
    return daylight.astimezone(eastern_zone(daylight))


def show_time(epoch: int) -> str:
    """Return `epoch` as 'YYYY-MM-DD HH:MM EDT' (or EST)."""
    local = to_eastern(epoch)
    return f"{local:%Y-%m-%d %H:%M} {local.tzname()}"


def parse_deadline(text: str, now: int) -> int:
    """Return the epoch of --until: epoch seconds, 'HH:MM' (its next time after `now`), or 'YYYY-MM-DD HH:MM'."""
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
    return {**os.environ, "MSYS_NO_PATHCONV": "1", "PYTHONIOENCODING": "utf-8"}


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


def check_tree(tree: Tree, allow_dirty: bool) -> None:
    """Refuse a dirty tree or an unpushed commit unless `allow_dirty`."""
    if allow_dirty:
        return
    if tree.dirty:
        raise Refusal("the working tree is dirty (git status --porcelain lists changes). Commit or stash them and "
                      "launch from a clean, pushed commit (a clean worktree of it, plans/LESSONS.md, alpha01), or "
                      "pass --allow-dirty to run anyway: exploratory, never [MEASURED]", 1)
    if not tree.pushed:
        raise Refusal(f"HEAD {tree.head[:12]} is on no origin branch (git branch -r --contains HEAD): push it "
                      "first, or pass --allow-dirty to run anyway", 1)


# ---------------------------------------------------------------- the plan


@dataclass(frozen=True)
class RecipeRow:
    """One recipe of the sequence: its name, its path in the repository, its trainer.timeout_s, base, and weights."""

    name: str
    path: str
    timeout_s: int
    base: str
    weights: str


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


def load_rows(directory: str, names: list[str]) -> list[RecipeRow]:
    """Load each named recipe with the train loader; refuse one that is missing, does not load, or is not ttxla's."""
    import lassi.train  # noqa: F401  (importing the package registers the trainers)
    from lassi.core.recipe import RecipeError, load_train_recipe

    rows = []
    for name in names:
        path = REPO / directory / f"{name}.yaml"
        if not path.is_file():
            known = ", ".join(sorted(item.stem for item in (REPO / directory).glob("*.yaml")))
            raise Refusal(f"no recipe {name} in {directory} (it has: {known or 'none'})", 2)
        try:
            data = load_train_recipe(path).data
        except RecipeError as exc:
            raise Refusal(f"{directory}/{name}.yaml does not load with the train loader: {exc}", 2) from None
        trainer = data.get("trainer", {})
        device = trainer.get("device", {})
        if trainer.get("kind") != "ttxla" or device.get("kind") != "tenstorrent":
            raise Refusal(f"{directory}/{name}.yaml binds trainer {trainer.get('kind')!r} on {device.get('kind')!r}; "
                          "this launcher runs the ttxla trainer on a tenstorrent card only", 2)
        rows.append(RecipeRow(name, f"{directory}/{name}.yaml", int(trainer["timeout_s"]), str(data["base_model"]),
                              str(data["weights"])))
    return rows


@dataclass(frozen=True)
class Plan:
    """Everything launch does: the recipes in order, the deadline, the spacing, and the rx job's slot and name."""

    rows: tuple[RecipeRow, ...]
    deadline: int
    spacing: int
    slot: str
    name: str

    def timeout(self, now: int) -> int:
        """The rx job's timeout in seconds: the deadline plus TIMEOUT_MARGIN_S, counted from `now`."""
        return self.deadline - now + TIMEOUT_MARGIN_S

    def remote_command(self) -> str:
        """The command the job runs in its slot, with the allowed root and stop line tt-xla.sh needs."""
        parts = [f"LASSI_ALLOWED_ROOT={ALLOWED_ROOT}", f"LASSI_STOP_GIB={STOP_GIB}", "bash", REMOTE_SCRIPT,
                 "--deadline", str(self.deadline), "--spacing", str(self.spacing)]
        for row in self.rows:
            parts += ["--recipe", f"{row.path}:{row.timeout_s}"]
        return " ".join(parts)

    def rx_args(self, now: int) -> list[str]:
        """The arguments of `uv run tools/rx.py` that start the job."""
        return ["job", "start", "--big", "--slot", self.slot, "--name", self.name, "--timeout", str(self.timeout(now)),
                "--", self.remote_command()]


def check_deadline(deadline: int, now: int, first: int) -> None:
    """Refuse a deadline too close for the first recipe, or one whose job timeout passes the gate's limit."""
    seconds = deadline - now
    need = SETUP_MIN_S + first + RUN_MARGIN_S
    if seconds < need:
        raise Refusal(f"the deadline {show_time(deadline)} is {seconds} s away; the first recipe needs at least "
                      f"{need} s (the setup, its trainer.timeout_s, and run.sh's margin)", 2)
    if seconds + TIMEOUT_MARGIN_S > GATE_MAX_TIMEOUT_S:
        raise Refusal(f"the job timeout would be {seconds + TIMEOUT_MARGIN_S} s, past the gate's limit of "
                      f"{GATE_MAX_TIMEOUT_S} s (job_timeout_max_s); choose an earlier deadline", 2)


def check_words(args: argparse.Namespace) -> None:
    """Check --name and --slot."""
    if not NAME_RE.match(args.name):
        raise Refusal(f"--name {args.name!r}: 1 to 40 letters, digits, '.', '_', or '-'", 2)
    if not SLOT_RE.match(args.slot):
        raise Refusal(f"--slot {args.slot!r}: a letter or digit, then up to 80 letters, digits, '.', '_', or '-'", 2)


def build_plan(args: argparse.Namespace) -> Plan:
    """Check every launch option against the repository and return the plan; raise Refusal on any problem."""
    now = now_epoch()
    check_words(args)
    directory = repo_path(args.dir, "recipe directory")
    names = [name.strip() for name in args.recipes.split(",") if name.strip()] if args.recipes else list(
        DEFAULT_RECIPES)
    if len(set(names)) != len(names):
        raise Refusal(f"--recipes names a recipe twice: {args.recipes}", 2)
    if args.spacing < 0:
        raise Refusal("--spacing must be whole seconds of at least 0", 2)
    rows = load_rows(directory, names)
    deadline = now + round(args.hours * 3600) if args.hours is not None else parse_deadline(args.until, now)
    check_deadline(deadline, now, rows[0].timeout_s)
    plan = Plan(rows=tuple(rows), deadline=deadline, spacing=args.spacing, slot=args.slot, name=args.name)
    if GATE_WORDS.search(plan.remote_command()):
        raise Refusal("the remote command names a word the gate refuses while silicon is present; rename the "
                      "recipe directory or recipes", 2)
    return plan


def render_plan(plan: Plan, tree: Tree, now: int) -> str:
    """Return the plan as the lines plan and launch print."""
    branch = "a detached HEAD" if tree.branch == "HEAD" else tree.branch
    label = ("exploratory: the tree is dirty, never [MEASURED]" if tree.dirty else
             "on no origin branch; push it before citing this run" if not tree.pushed else
             "a clean commit on origin, cited only from the provenance that pull writes")
    lines = ["Plan (plan starts nothing; launch runs exactly this):",
             f"  tree         HEAD {tree.head[:12]} on {branch}: {'dirty' if tree.dirty else 'clean'}; {label}",
             "  first        the tt-xla venv (toolchains/tt-xla.sh: kept when it checks out, installed when missing; "
             "CPU only), tt-train-v0's records, and each recipe's base (CPU only)",
             f"  recipes      {len(plan.rows)}, one process each, {plan.spacing} s apart, the sequence stopping at "
             "its first failure or hang (OQ-100 (s2)):"]
    start = now + SETUP_MIN_S
    for index, row in enumerate(plan.rows, start=1):
        end = start + row.timeout_s + RUN_MARGIN_S
        fits = "fits" if end <= plan.deadline else "starts only if the runs before it end early"
        lines.append(f"    {index}. {row.name}: {row.weights} on {row.base}; trainer.timeout_s {row.timeout_s} s; "
                     f"worst case ends {show_time(end)} ({fits}; PROJECTED)")
        start = end + plan.spacing
    lines += [f"  deadline     {show_time(plan.deadline)} (epoch {plan.deadline}); now {show_time(now)}",
              f"  job timeout  {plan.timeout(now)} s (the deadline plus {TIMEOUT_MARGIN_S} s), recomputed at start",
              f"  free space   the preflight needs {FLOOR_GB + MARGIN_GB} GB free on the scratch disk (the gate's "
              f"{FLOOR_GB} GB big-job floor and a {MARGIN_GB} GB margin)",
              "  devices      Wormhole card 0 through `lassi train --silicon`: one card, one run at a time under the "
              "cards' lock, polled first, quarantined after a hang, kill, or crash; each worker in its own empty "
              "network namespace (trainer.network none); no RNGD card, no GPU",
              f"  output       lassi-wt/{plan.slot} (the job's checkout), {OUTPUT_DIR}/<job id> (pull fetches it), "
              "lassi-runs/train/tt-<stamp>-<recipe> (the train trees and checkpoints, left on the host)",
              f"  remote       {plan.remote_command()}",
              "  rx command   MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8 uv run tools/rx.py "
              + " ".join(plan.rx_args(now)[:-1]) + " " + shlex.quote(plan.remote_command())]
    return "\n".join(lines)


# ---------------------------------------------------------------- the build host


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


def preflight(silicon: bool) -> None:
    """Check the build host through rx doctor and rx job status; `silicon` also needs the tt_silicon class enabled."""
    result = rx(["doctor"])
    doctor = doctor_json(result.stdout or "")
    if doctor is None:
        tail = ((result.stdout or "") + (result.stderr or "")).strip()[-400:]
        raise Refusal(f"rx doctor did not answer (exit {result.returncode}): {tail}. Check that ssh to the build "
                      "host works (uv run tools/rx.py doctor), then try again", 1)
    if doctor.get("stop"):
        raise Refusal("the gate's owner STOP is set on alpha01, so it refuses every job; the owner clears it", 1)
    if silicon and not (doctor.get("devices_enabled") or {}).get("tt_silicon"):
        raise Refusal("the gate's tt_silicon device class is disabled (rx doctor, devices_enabled); the owner enables "
                      "it in the gate's config.json on alpha01 for the window, then launch again", 1)
    free = float(doctor.get("scratch_free_gb") or 0)
    if free < FLOOR_GB + MARGIN_GB:
        raise Refusal(f"the scratch disk has {free} GB free, under the {FLOOR_GB + MARGIN_GB} GB the job needs (the "
                      f"gate's {FLOOR_GB} GB floor and a {MARGIN_GB} GB margin); free space first", 1)
    running = running_jobs()
    if running:
        raise Refusal(f"rx job status lists a running job: {', '.join(running)}. The gate runs one big job at a time; "
                      "wait for it (uv run tools/train_tt.py wait <id>) or stop it first", 1)


def confirm(question: str) -> bool:
    """Ask `question`; True only on y or yes."""
    try:
        answer = input(f"{question} [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def start_job(args_rx: list[str], deadline: int | None) -> str:
    """Start the job with rx; return its id and print how to watch, stop, and pull it."""
    result = rx_ok(args_rx)
    started = next((line for line in reversed((result.stdout or "").splitlines()) if line.startswith("{")), "{}")
    job = json.loads(started).get("id")
    if not job:
        raise Refusal("rx job start printed no job id; check `uv run tools/train_tt.py status`", 1)
    tool = "uv run tools/train_tt.py"
    ends = "" if deadline is None else f"; it ends by {show_time(deadline + TIMEOUT_MARGIN_S)}"
    print(f"\nStarted {job} on alpha01{ends}.")
    print(f"  watch: {tool} status {job}")
    print(f"         {tool} wait {job}")
    print(f"  stop:  {tool} stop    (before the next recipe; a running one ends at its trainer.timeout_s)")
    print(f"  pull:  {tool} pull {job}    (after it ends)")
    return job


# ---------------------------------------------------------------- subcommands


def do_plan(args: argparse.Namespace) -> int:
    """Print the plan; contact nothing."""
    print(render_plan(build_plan(args), read_tree(), now_epoch()))
    return 0


def do_launch(args: argparse.Namespace) -> int:
    """Check, show the plan, ask (unless --yes), and start the job."""
    plan = build_plan(args)
    if rip_stop_present():
        raise Refusal(".rip-stop exists at the repository root (the stop file for unattended runs); remove it if "
                      "you mean to launch", 1)
    tree = read_tree()
    check_tree(tree, args.allow_dirty)
    preflight(silicon=True)
    print(render_plan(plan, tree, now_epoch()))
    if not args.yes and not confirm("Start this training on alpha01's Wormhole card?"):
        print("Not started.")
        return 1
    now = now_epoch()
    check_deadline(plan.deadline, now, plan.rows[0].timeout_s)
    start_job(plan.rx_args(now), plan.deadline)
    return 0


def install_command() -> str:
    """The command of the install job: run.sh with --install-only, with the allowed root and stop line."""
    return f"LASSI_ALLOWED_ROOT={ALLOWED_ROOT} LASSI_STOP_GIB={STOP_GIB} bash {REMOTE_SCRIPT} --install-only"


def do_install(args: argparse.Namespace) -> int:
    """Start the tt-xla venv's install alone as a big rx job (CPU only, no card)."""
    check_words(args)
    if args.timeout <= 0 or args.timeout > GATE_MAX_TIMEOUT_S:
        raise Refusal(f"--timeout must be 1 to {GATE_MAX_TIMEOUT_S} seconds", 2)
    if rip_stop_present():
        raise Refusal(".rip-stop exists at the repository root; remove it if you mean to install", 1)
    check_tree(read_tree(), args.allow_dirty)
    preflight(silicon=False)
    words = ["job", "start", "--big", "--slot", args.slot, "--name", args.name, "--timeout", str(args.timeout), "--",
             install_command()]
    print(f"Install: {install_command()}\n  rx job start --big --slot {args.slot} --name {args.name} "
          f"--timeout {args.timeout}; CPU only: downloads, a venv, and checks; no card is touched")
    if not args.yes and not confirm("Start the tt-xla install on alpha01?"):
        print("Not started.")
        return 1
    start_job(words, None)
    return 0


def do_status(args: argparse.Namespace) -> int:
    """Show recent jobs, or one job and the last lines of its log."""
    result = rx(["job", "status", *([args.job] if args.job else [])], capture=False)
    if args.job and args.lines > 0 and result.returncode == 0:
        result = rx(["job", "tail", args.job, "-n", str(args.lines)], capture=False)
    return result.returncode


def do_wait(args: argparse.Namespace) -> int:
    """Wait for a job through rx job wait; its exit status passes through (3: still running)."""
    return rx(["job", "wait", args.job, "--timeout", str(args.timeout), "--interval", str(args.interval)],
              capture=False).returncode


def do_stop(args: argparse.Namespace) -> int:
    """Create (or with --clear remove) the sequence's STOP file through one rx exec."""
    return rx(["exec", "--", CLEAR_COMMAND if args.clear else STOP_COMMAND], capture=False).returncode


def job_state(job: str) -> str:
    """Return the state `rx job status <job>` reports."""
    result = rx(["job", "status", job])
    for line in (result.stdout or "").splitlines():
        words = line.split()
        if len(words) >= 2 and words[0] == job:
            return words[1]
    raise Refusal(f"rx job status {job} listed no such job (exit {result.returncode})", 1)


def copy_ascii(source: Path, target: Path) -> list[str]:
    """Copy every plain-ASCII file under `source` to `target`; return the files left out, with their sizes."""
    left_out = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        relative = path.relative_to(source)
        if not data.isascii():
            left_out.append(f"{relative.as_posix()} ({len(data)} bytes)")
            continue
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target / relative)
    return left_out


def do_pull(args: argparse.Namespace) -> int:
    """Pull a finished job's provenance and output directory into results/<name>, then print its summary."""
    job = args.job
    if not JOB_ID_RE.match(job):
        raise Refusal(f"{job!r} is not an rx job id (like 20261027-080012-train-tt-ab12)", 2)
    parts = job.split("-")
    name = args.name or "-".join([*parts[2:-1], parts[0], parts[1]])
    if not WORD_RE.match(name):
        raise Refusal(f"--name {name!r}: use letters, digits, '.', '_', and '-'", 2)
    target = REPO / "results" / name
    if target.exists() and any(target.iterdir()):
        raise Refusal(f"results/{name} exists and is not empty; pass --name for a new folder", 2)
    state = job_state(job)
    if state in RUNNING:
        raise Refusal(f"job {job} is still {state}; pull it after it ends (uv run tools/train_tt.py wait {job})", 1)
    rx_ok(["pull", job, "--into", f"results/{name}"])
    cache = REPO / ".rx" / "pulls" / job
    if cache.exists():  # rx pull extracts the job record there too, so it goes aside first (plans/LESSONS.md)
        cache.rename(cache.with_name(f"{job}.aside-{now_epoch()}"))
    rx_ok(["pull", "--path", f"{OUTPUT_DIR}/{job}"])
    if not cache.is_dir():
        raise Refusal(f"rx pull left no directory {cache}", 1)
    for entry in copy_ascii(cache, target / "window"):
        print(f"left out (not plain ASCII): {entry}")
    summary = target / "window" / "summary.tsv"
    print(f"copied the job's output into results/{name}/window")
    if summary.is_file():
        print(summary.read_text(encoding="ascii"))
    return 0


# ---------------------------------------------------------------- the parser

TOOL = "uv run tools/train_tt.py"
TOP_DESCRIPTION = """Train the small models on a Wormhole card on alpha01 as one rx job, then follow, stop, and pull it.

launch checks the tree and the build host, prints the plan, asks, and starts plans/runs/train-tt/run.sh as one big
rx job: it keeps or installs the tt-xla venv and fetches the records and the bases (CPU only), then runs `lassi
--graphics off train <recipe> --silicon` once per recipe, in order, each its own process, spaced, stopping at the
first failure or hang (OQ-100 (s2)). install starts the venv's install alone; plan prints the plan and contacts
nothing. Every child process runs with MSYS_NO_PATHCONV=1 and PYTHONIOENCODING=utf-8."""

TOP_EPILOG = f"""Examples:
  {TOOL} plan --until 12:00
  {TOOL} install
  {TOOL} launch --until 12:00
  {TOOL} launch --hours 6 --recipes ttxla-smoke-fixture,ttxla-lora-qwen2.5-0.5b --yes
  {TOOL} status
  {TOOL} wait 20261027-080012-train-tt-ab12
  {TOOL} stop
  {TOOL} pull 20261027-080012-train-tt-ab12

Run it from the repository root of a clean, pushed commit. `<command> --help` documents each command;
plans/runs/train-tt/README.md is the guide."""

RUN_EPILOG = """Examples:
  {tool} {name} --until 12:00
  {tool} {name} --until "2026-10-27 12:00" --recipes ttxla-smoke-fixture,ttxla-full-scratch-77m
  {tool} {name} --hours 7.5 --spacing 300 --slot train-tt2 --name train-tt2

Clock times are US Eastern (EDT, or EST after the first Sunday of November); HH:MM means its next time."""


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Keep descriptions and examples as written, and wrap option help without breaking a path at a hyphen."""

    def _split_lines(self, text: str, width: int) -> list[str]:
        return textwrap.wrap(" ".join(text.split()), width, break_on_hyphens=False, break_long_words=False)


def subparser(sub: argparse._SubParsersAction, name: str, summary: str, text: str,
              epilog: str) -> argparse.ArgumentParser:
    """Add one subcommand with its summary, description (wrapped here), and examples."""
    wrapped = textwrap.fill(text, 110, break_on_hyphens=False, break_long_words=False)
    return sub.add_parser(name, help=summary, description=wrapped, epilog=epilog, formatter_class=HelpFormatter)


def add_run_options(parser: argparse.ArgumentParser) -> None:
    """Add the options plan and launch share."""
    parser.add_argument("--recipes", metavar="LIST", help="comma-separated recipe names from --dir, run in this order "
                        f"(default: all six, {', '.join(DEFAULT_RECIPES)}); an unknown name is refused")
    parser.add_argument("--dir", default=DEFAULT_DIR, metavar="DIR", help="the recipe directory; every recipe must "
                        "load with the train loader and bind the ttxla trainer on a tenstorrent card "
                        "(default: %(default)s)")
    deadline = parser.add_mutually_exclusive_group(required=True)
    deadline.add_argument("--hours", type=float, metavar="H", help="the deadline H hours from now (this or --until)")
    deadline.add_argument("--until", metavar="WHEN", help="the deadline: an Eastern clock time HH:MM, a date and "
                          "time 'YYYY-MM-DD HH:MM', or epoch seconds (this or --hours). A recipe starts only while the "
                          "time left covers its trainer.timeout_s; the rx job's timeout is the deadline plus "
                          f"{TIMEOUT_MARGIN_S} s")
    parser.add_argument("--spacing", type=int, default=SPACING_S, metavar="S", help="seconds between two runs "
                        "(default: %(default)s)")
    parser.add_argument("--name", default=DEFAULT_NAME, help="the rx job's name, part of its id (default: %(default)s)")
    parser.add_argument("--slot", default=DEFAULT_SLOT, help="the rx worktree slot on alpha01 (default: %(default)s)")
    parser.add_argument("--yes", action="store_true", help="launch: start without asking (plan ignores it)")
    parser.add_argument("--allow-dirty", action="store_true", help="launch even from a dirty tree or a commit on no "
                        "origin branch; such a run is exploratory, never [MEASURED] (default: refuse)")


def add_follow_commands(sub: argparse._SubParsersAction) -> None:
    """Add status, wait, stop, and pull."""
    status = subparser(sub, "status", "show recent rx jobs, or one job and the end of its log",
                       "Wrap `rx job status [ID]`; with an id, also `rx job tail ID -n LINES`.",
                       f"Examples:\n  {TOOL} status\n  {TOOL} status 20261027-080012-train-tt-ab12 --lines 80")
    status.add_argument("job", nargs="?", metavar="ID", help="an rx job id (default: list recent jobs)")
    status.add_argument("--lines", type=int, default=40, help="log lines to show for ID, 0 for none "
                        "(default: %(default)s)")
    wait = subparser(sub, "wait", "wait for a job to end (rx job wait)",
                     "Wrap `rx job wait ID`: poll until the job ends and print the end of its log. Exit status: the "
                     "job's, or 3 when it is still running at --timeout.",
                     f"Examples:\n  {TOOL} wait 20261027-080012-train-tt-ab12\n"
                     f"  {TOOL} wait 20261027-080012-train-tt-ab12 --timeout 600 --interval 60")
    wait.add_argument("job", metavar="ID", help="the rx job id")
    wait.add_argument("--timeout", type=int, default=3600, help="seconds to wait (default: %(default)s)")
    wait.add_argument("--interval", type=int, default=300, help="seconds between polls (default: %(default)s)")
    stop = subparser(sub, "stop", "stop the sequence before its next recipe (its STOP file, through one rx exec)",
                     "Create $LASSI_RUNS_ROOT/train-tt/STOP on alpha01 with one rx exec. run.sh starts no further "
                     "recipe; a run already going ends at its trainer.timeout_s or when it finishes, since killing a "
                     "run that holds the card would quarantine it. While the file exists a new job refuses to start; "
                     "--clear removes it.", f"Examples:\n  {TOOL} stop\n  {TOOL} stop --clear")
    stop.add_argument("--clear", action="store_true", help="remove the STOP file instead (owner)")
    pull = subparser(sub, "pull", "pull a finished job into results/<name> and print its summary",
                     f"Run `rx pull ID --into results/<name>` (provenance.json), then `rx pull --path {OUTPUT_DIR}/ID` "
                     "and copy that directory to results/<name>/window, leaving out any file that is not plain "
                     "ASCII, and print summary.tsv. Write results/<name>/summary.md from them afterwards.",
                     f"Examples:\n  {TOOL} pull 20261027-080012-train-tt-ab12\n"
                     f"  {TOOL} pull 20261027-080012-train-tt-ab12 --name train-tt-20261027")
    pull.add_argument("job", metavar="ID", help="the rx job id")
    pull.add_argument("--name", help="the results folder (default: the job's name, date, and time)")


def build_parser() -> argparse.ArgumentParser:
    """Return the launcher's argument parser."""
    parser = argparse.ArgumentParser(prog=TOOL, description=TOP_DESCRIPTION, epilog=TOP_EPILOG,
                                     formatter_class=HelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    plan = subparser(sub, "plan", "print what launch would do; contact nothing",
                     "Load every recipe with the train loader and print the plan: the tree, the recipes in order with "
                     "their trainer.timeout_s and a PROJECTED worst-case end, the deadline in Eastern time and epoch, "
                     "the job timeout, the free space needed, the remote script's command, and the exact rx command. "
                     "Nothing is started and nothing contacts alpha01.", RUN_EPILOG.format(tool=TOOL, name="plan"))
    add_run_options(plan)
    launch = subparser(sub, "launch", "check, show the plan, ask, and start the training on alpha01",
                       "Preflight (.rip-stop absent; a clean, pushed tree unless --allow-dirty; rx doctor answers, no "
                       "owner STOP, the tt_silicon device class enabled, enough scratch space; rx job status shows no "
                       "running job), print the plan, ask unless --yes, then start `uv run tools/rx.py job start "
                       "--big ...` and print the job id and the commands to watch, stop, and pull it.",
                       RUN_EPILOG.format(tool=TOOL, name="launch"))
    add_run_options(launch)
    install = subparser(sub, "install", "start the tt-xla venv's install alone (CPU only, no card)",
                        "Preflight as launch does, without the tt_silicon class, then start `run.sh --install-only` "
                        "as a big rx job: toolchains/tt-xla.sh keeps a checked install or installs one (downloads "
                        "checked by size and sha256, a venv, the sfpi link); nothing touches a card.",
                        f"Examples:\n  {TOOL} install\n  {TOOL} install --yes --timeout 5400")
    install.add_argument("--timeout", type=int, default=INSTALL_TIMEOUT_S, help="the rx job's timeout in seconds "
                         "(default: %(default)s)")
    install.add_argument("--name", default=INSTALL_NAME, help="the rx job's name (default: %(default)s)")
    install.add_argument("--slot", default=DEFAULT_SLOT, help="the rx worktree slot (default: %(default)s)")
    install.add_argument("--yes", action="store_true", help="start without asking")
    install.add_argument("--allow-dirty", action="store_true", help="install even from a dirty or unpushed tree")
    add_follow_commands(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand; return its exit status."""
    args = build_parser().parse_args(argv)
    handlers = {"plan": do_plan, "launch": do_launch, "install": do_install, "status": do_status, "wait": do_wait,
                "stop": do_stop, "pull": do_pull}
    try:
        return handlers[args.command](args)
    except Refusal as exc:
        print(f"train_tt: {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
