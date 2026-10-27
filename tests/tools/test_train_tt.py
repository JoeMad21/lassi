"""Tests for tools/train_tt.py, the owner's launcher for training on the Wormhole cards, and its remote side.

Task P17.38. The owner (2026-10-26): "I would like to be able to train the smaller models by then, plan to use the
tenstorrents for that task"; OQ-100 (a) with (s2): training runs of tracked code on the cards, started by the
owner; one owner start may run a sequence of runs, each its own process, spaced, stopping at the first failure or
hang. The launcher wraps `uv run tools/rx.py`; the remote side is plans/runs/train-tt/run.sh. Every test here runs
on the workstation with fakes for rx and git: no test calls rx, ssh, or alpha01, and no device is touched. The
fake host's values (free space, job ids, commits) are PLACEHOLDER values, never measurements.

The interface these tests fix:

- `main(argv) -> int` with the subcommands plan, install, launch, status, wait, stop, and pull, each with --help
  and an examples epilog; argparse usage errors exit 2, a refused request returns 2, a failed preflight 1.
- plan and launch take --recipes (names in projects/lassi-demo/train-tt, default all six in the window's order),
  --dir, --hours or --until (required; US Eastern clock times, as tools/suite.py reads them), --spacing (seconds
  between runs), --name, --slot, --yes, and --allow-dirty. Every recipe loads with the train loader, binds the
  ttxla trainer on a tenstorrent device, and sets trainer.timeout_s; the job timeout is the deadline plus 900 s.
- launch refuses .rip-stop, a dirty or unpushed tree (unless --allow-dirty), a gate with STOP set, the tt_silicon
  device class disabled (the owner enables it for the window), too little scratch space, and any running job;
  it prints the plan, asks y/N unless --yes, and starts one big rx job running
  `LASSI_ALLOWED_ROOT=/mnt/nvme10/joseph_ufl LASSI_STOP_GIB=345 bash plans/runs/train-tt/run.sh --deadline <epoch>
  --spacing <s> --recipe <path>:<timeout_s> ...`, a command naming no word the gate's Tenstorrent guard refuses.
- install starts the venv install alone (run.sh --install-only), CPU only, without the tt_silicon class.
- stop creates $LASSI_RUNS_ROOT/train-tt/STOP with one rx exec (--clear removes it); pull copies the job's
  provenance and its output directory into results/<name>.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "train_tt.py"
SCRIPT = REPO / "plans" / "runs" / "train-tt" / "run.sh"
README = REPO / "plans" / "runs" / "train-tt" / "README.md"
RECIPE_DIR = "projects/lassi-demo/train-tt"
ORDER = (
    "ttxla-smoke-fixture", "ttxla-lora-qwen2.5-0.5b", "ttxla-lora-qwen3-0.6b", "ttxla-lora-qwen3-1.7b",
    "ttxla-full-scratch-77m", "ttxla-prompt-tuning-qwen3-1.7b",
)
# The window's start, 2026-10-27 04:00 EDT, as the fixed clock; its end is 12:00 EDT.
EDT = timezone(timedelta(hours=-4))
NOW = int(datetime(2026, 10, 27, 4, 0, tzinfo=EDT).timestamp())
END = int(datetime(2026, 10, 27, 12, 0, tzinfo=EDT).timestamp())
JOB_ID = "20261027-010012-train-tt-ab12"  # PLACEHOLDER job id in the gate's format
HEAD = "0123456789abcdef0123456789abcdef01234567"  # PLACEHOLDER commit
# The gate refuses a command naming these while Tenstorrent silicon is present (tools/server/gate.py), and the
# device classes' patterns (tools/server/config.default.json).
GATE_WORDS = re.compile(r"tt-metal|tt_metal|ttnn|metal_example|TT_METAL|tt-smi|/dev/tenstorrent|tt-flash|tt-topology")

RUN_OPTIONS = ("--recipes", "--dir", "--hours", "--until", "--spacing", "--name", "--slot", "--yes", "--allow-dirty")
SUBCOMMAND_OPTIONS = {
    "plan": tuple(option for option in RUN_OPTIONS if option != "--yes"),
    "launch": RUN_OPTIONS,
    "install": ("--timeout", "--name", "--slot", "--yes", "--allow-dirty"),
    "status": ("--lines",),
    "wait": ("--timeout", "--interval"),
    "stop": ("--clear",),
    "pull": ("--name",),
}


def load() -> ModuleType:
    """Import tools/train_tt.py, failing the test clearly while it does not exist (task P17.38)."""
    if not TOOL.is_file():
        pytest.fail("tools/train_tt.py does not exist yet (task P17.38)")
    spec = importlib.util.spec_from_file_location("lassi_tools_train_tt", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["lassi_tools_train_tt"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def tool(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Return tools/train_tt.py with the clock fixed at the window's start."""
    module = load()
    monkeypatch.setattr(module, "now_epoch", lambda: NOW)
    return module


class FakeHost:
    """Stand-in for subprocess.run: answers git and rx, records each call's argv and environment."""

    def __init__(self, *, dirty: bool = False, pushed: bool = True, free_gb: float = 200.0, tt: bool = True,
                 stop: bool = False, jobs: str = "", job_state: str = "done") -> None:
        self.dirty, self.pushed = dirty, pushed
        self.doctor = {"stop": stop, "scratch_free_gb": free_gb, "running_jobs": [],
                       "devices_enabled": {"rngd": False, "tt_silicon": tt}}
        self.jobs, self.job_state = jobs, job_state
        self.repo: Path | None = None
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        argv = [str(part) for part in argv]
        self.calls.append((argv, dict(kwargs.get("env") or {})))
        if argv[0] == "git":
            return self.git(argv)
        if argv[:3] == ["uv", "run", "tools/rx.py"]:
            return self.rx(argv[3:])
        raise AssertionError(f"unexpected child process {argv}")

    def done(self, argv: list[str], out: str = "", rc: int = 0) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(argv, rc, stdout=out, stderr="")

    def git(self, argv: list[str]) -> subprocess.CompletedProcess:
        words = argv[3:]
        if words == ["status", "--porcelain"]:
            return self.done(argv, " M lassi/cli.py\n" if self.dirty else "")
        if words == ["rev-parse", "HEAD"]:
            return self.done(argv, HEAD + "\n")
        if words == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return self.done(argv, "p16-adversarial\n")
        if words == ["branch", "-r", "--contains", "HEAD"]:
            return self.done(argv, "  origin/p16-adversarial\n" if self.pushed else "")
        raise AssertionError(f"unexpected git call {argv}")

    def rx(self, words: list[str]) -> subprocess.CompletedProcess:
        argv = ["uv", "run", "tools/rx.py", *words]
        if words == ["doctor"]:
            return self.done(argv, f"[rx] ssh ionx: PASS\n{json.dumps(self.doctor, indent=2)}\n[rx] Ready\n")
        if words == ["job", "status"]:
            return self.done(argv, self.jobs)
        if words[:2] == ["job", "status"]:
            return self.done(argv, f"{words[2]}  {self.job_state}  rc=0  slot=train-tt  start=x  end=y  cmd=bash\n")
        if words[:2] == ["job", "start"]:
            return self.done(argv, json.dumps({"id": JOB_ID, "state": "starting", "slot": "train-tt"}) + "\n")
        if words[:1] in (["exec"], ["job"]):
            return self.done(argv, "")
        if words[:2] == ["pull", "--path"] and self.repo is not None:
            root = self.repo / ".rx" / "pulls" / Path(words[2]).name
            root.mkdir(parents=True)
            (root / "summary.tsv").write_text("recipe\trc\nttxla-smoke-fixture\t0\n")
            (root / "progress.log").write_bytes(b"bar " + bytes([0xE2, 0x96, 0x88]) + b"\n")
            return self.done(argv, f"[rx] pulled 2 entries into {root}\n")
        if words[:1] == ["pull"]:
            return self.done(argv, "[rx] pulled\n")
        raise AssertionError(f"unexpected rx call {argv}")

    def rx_calls(self) -> list[list[str]]:
        return [argv[3:] for argv, _ in self.calls if argv[:3] == ["uv", "run", "tools/rx.py"]]

    def started(self) -> list[list[str]]:
        return [words for words in self.rx_calls() if words[:2] == ["job", "start"]]


@pytest.fixture()
def host(monkeypatch: pytest.MonkeyPatch, tool: ModuleType) -> FakeHost:
    """Route every child process of the launcher to a FakeHost."""
    fake = FakeHost()
    monkeypatch.setattr(tool.subprocess, "run", fake)
    monkeypatch.setattr(tool, "rip_stop_present", lambda: False)
    return fake


def use(monkeypatch: pytest.MonkeyPatch, tool: ModuleType, fake: FakeHost) -> FakeHost:
    """Route the launcher's children to `fake` instead."""
    monkeypatch.setattr(tool.subprocess, "run", fake)
    monkeypatch.setattr(tool, "rip_stop_present", lambda: False)
    return fake


# ---------------------------------------------------------------- help


def test_top_level_help_lists_every_subcommand(tool: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        tool.main(["--help"])
    assert info.value.code == 0
    text = capsys.readouterr().out
    for name in SUBCOMMAND_OPTIONS:
        assert name in text
    assert "example" in text.lower()


@pytest.mark.parametrize("name", sorted(SUBCOMMAND_OPTIONS))
def test_each_subcommand_help_lists_its_options_and_examples(
    tool: ModuleType, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    with pytest.raises(SystemExit) as info:
        tool.main([name, "--help"])
    assert info.value.code == 0
    text = capsys.readouterr().out
    for option in SUBCOMMAND_OPTIONS[name]:
        assert option in text, f"{name} --help does not list {option}"
    assert "example" in text.lower()


# ---------------------------------------------------------------- the plan


def test_the_default_plan_runs_the_six_recipes_in_order(
    tool: ModuleType, host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    assert tool.main(["plan", "--until", "12:00"]) == 0
    text = capsys.readouterr().out
    positions = [text.index(name) for name in ORDER]
    assert positions == sorted(positions), "the recipes in the window's order"
    assert "2026-10-27 12:00 EDT" in text and str(END) in text
    assert f"{END - NOW + 900} s" in text, "the job timeout is the deadline plus 900 s"
    assert host.rx_calls() == [], "plan contacts nothing"


def test_the_remote_command_names_each_recipe_with_its_timeout_and_no_gate_word(tool: ModuleType) -> None:
    args = tool.build_parser().parse_args(["plan", "--until", "12:00"])
    plan = tool.build_plan(args)
    command = plan.remote_command()
    assert command.startswith("LASSI_ALLOWED_ROOT=/mnt/nvme10/joseph_ufl LASSI_STOP_GIB=345 bash "
                              "plans/runs/train-tt/run.sh --deadline ")
    assert f"--deadline {END}" in command and "--spacing 120" in command
    recipes = re.findall(r"--recipe (\S+):(\d+)", command)
    assert [Path(path).stem for path, _ in recipes] == list(ORDER)
    assert all(path.startswith(f"{RECIPE_DIR}/") for path, _ in recipes)
    assert all(600 <= int(seconds) <= 7200 for _, seconds in recipes)
    assert not GATE_WORDS.search(command), command


def test_plan_takes_a_subset_in_the_order_given(tool: ModuleType, host: FakeHost,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    assert tool.main(["plan", "--hours", "3", "--recipes", "ttxla-full-scratch-77m,ttxla-smoke-fixture"]) == 0
    text = capsys.readouterr().out
    assert text.index("ttxla-full-scratch-77m") < text.index("ttxla-smoke-fixture")
    assert "ttxla-lora-qwen3-1.7b" not in text


@pytest.mark.parametrize(
    ("argv", "named"),
    [
        pytest.param(["--recipes", "no-such-recipe"], "no-such-recipe", id="unknown"),
        pytest.param(["--recipes", "ttxla-smoke-fixture,ttxla-smoke-fixture"], "twice", id="twice"),
        pytest.param(["--dir", "projects/train"], "ttxla", id="not-ttxla"),
    ],
)
def test_plan_refuses_a_recipe_it_cannot_run(
    tool: ModuleType, host: FakeHost, capsys: pytest.CaptureFixture[str], argv: list[str], named: str
) -> None:
    assert tool.main(["plan", "--until", "12:00", *argv]) == 2
    assert named in capsys.readouterr().err


def test_plan_refuses_a_deadline_too_close_for_the_first_recipe(
    tool: ModuleType, host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    assert tool.main(["plan", "--hours", "0.2"]) == 2
    assert "deadline" in capsys.readouterr().err


# ---------------------------------------------------------------- launch and install


def test_launch_with_yes_starts_one_big_job_with_the_plan(
    tool: ModuleType, host: FakeHost, capsys: pytest.CaptureFixture[str]
) -> None:
    assert tool.main(["launch", "--until", "12:00", "--yes"]) == 0
    (start,) = host.started()
    assert start[:2] == ["job", "start"] and "--big" in start
    assert start[start.index("--slot") + 1] == "train-tt" and start[start.index("--name") + 1] == "train-tt"
    assert start[start.index("--timeout") + 1] == str(END - NOW + 900)
    assert start[-1] == tool.build_plan(tool.build_parser().parse_args(["plan", "--until", "12:00"])).remote_command()
    assert all(env.get("MSYS_NO_PATHCONV") == "1" for argv, env in host.calls if argv[0] == "uv")
    out = capsys.readouterr().out
    assert JOB_ID in out and "train_tt.py stop" in out


def test_launch_asks_and_starts_nothing_on_no(
    tool: ModuleType, host: FakeHost, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert tool.main(["launch", "--until", "12:00"]) == 1
    assert host.started() == []


@pytest.mark.parametrize(
    ("fake", "named"),
    [
        pytest.param({"dirty": True}, "dirty", id="dirty"),
        pytest.param({"pushed": False}, "origin", id="unpushed"),
        pytest.param({"tt": False}, "tt_silicon", id="tt-disabled"),
        pytest.param({"free_gb": 10.0}, "GB", id="space"),
        pytest.param({"stop": True}, "STOP", id="gate-stop"),
        pytest.param({"jobs": f"{JOB_ID}  running  rc=-  slot=x\n"}, "running", id="running-job"),
    ],
)
def test_launch_preflight_refuses_and_starts_nothing(
    tool: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fake: dict[str, Any],
    named: str
) -> None:
    host = use(monkeypatch, tool, FakeHost(**fake))
    assert tool.main(["launch", "--until", "12:00", "--yes"]) == 1
    assert named in capsys.readouterr().err
    assert host.started() == []


def test_launch_refuses_while_rip_stop_exists(
    tool: ModuleType, host: FakeHost, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(tool, "rip_stop_present", lambda: True)
    assert tool.main(["launch", "--until", "12:00", "--yes"]) == 1
    assert ".rip-stop" in capsys.readouterr().err and host.started() == []


def test_install_starts_the_venv_install_alone_without_the_tt_silicon_class(
    tool: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = use(monkeypatch, tool, FakeHost(tt=False))
    assert tool.main(["install", "--yes"]) == 0
    (start,) = host.started()
    command = start[-1]
    assert command == ("LASSI_ALLOWED_ROOT=/mnt/nvme10/joseph_ufl LASSI_STOP_GIB=345 bash plans/runs/train-tt/run.sh "
                       "--install-only")
    assert start[start.index("--timeout") + 1] == "3600" and "--big" in start
    assert start[start.index("--name") + 1] == "tt-xla-venv"


# ---------------------------------------------------------------- follow, stop, pull


def test_stop_creates_the_stop_file_and_clear_removes_it(tool: ModuleType, host: FakeHost) -> None:
    assert tool.main(["stop"]) == 0
    assert tool.main(["stop", "--clear"]) == 0
    (first, second) = [words for words in host.rx_calls() if words[0] == "exec"]
    assert "touch" in first[-1] and "train-tt/STOP" in first[-1]
    assert "rm -f" in second[-1] and "train-tt/STOP" in second[-1]
    assert not GATE_WORDS.search(first[-1] + second[-1])


def test_status_and_wait_wrap_rx(tool: ModuleType, host: FakeHost) -> None:
    assert tool.main(["status"]) == 0
    assert tool.main(["wait", JOB_ID, "--timeout", "60", "--interval", "30"]) == 0
    assert ["job", "status"] in host.rx_calls()
    assert ["job", "wait", JOB_ID, "--timeout", "60", "--interval", "30"] in host.rx_calls()


def test_pull_fetches_the_provenance_and_the_output_directory(
    tool: ModuleType, host: FakeHost, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(tool, "REPO", tmp_path)
    host.repo = tmp_path
    (tmp_path / ".rx" / "pulls" / JOB_ID).mkdir(parents=True)  # the job record an earlier pull left
    assert tool.main(["pull", JOB_ID, "--name", "train-tt-window"]) == 0
    calls = host.rx_calls()
    assert ["pull", JOB_ID, "--into", "results/train-tt-window"] in calls
    assert ["pull", "--path", f"lassi-runs/train-tt/{JOB_ID}"] in calls
    copied = tmp_path / "results" / "train-tt-window" / "window"
    assert (copied / "summary.tsv").is_file() and not (copied / "progress.log").exists(), "plain ASCII only"


# ---------------------------------------------------------------- the remote script and the guide


def test_the_remote_script_parses_as_bash_and_holds_the_rules() -> None:
    assert SCRIPT.is_file(), "plans/runs/train-tt/run.sh does not exist yet (task P17.38)"
    text = SCRIPT.read_text(encoding="ascii")
    assert "\r" not in text
    assert "prlimit --pid $$ --core=1:1" in text, "the 1-byte core limit, read back as (1, 1)"
    assert "--silicon" in text and "--graphics off train" in text
    assert "STOP" in text and "toolchains/tt-xla.sh" in text
    assert "sudo" not in text
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available")
    done = subprocess.run([bash, "-n", "plans/runs/train-tt/run.sh"], cwd=REPO, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_the_remote_script_prints_its_help_and_refuses_without_a_deadline() -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available")
    shown = subprocess.run([bash, "plans/runs/train-tt/run.sh", "--help"], cwd=REPO, capture_output=True, text=True)
    assert shown.returncode == 0 and "--deadline" in shown.stdout and "--install-only" in shown.stdout
    refused = subprocess.run([bash, "plans/runs/train-tt/run.sh"], cwd=REPO, capture_output=True, text=True)
    assert refused.returncode == 2 and "--deadline" in refused.stderr


def test_the_guide_explains_launch_install_and_stop() -> None:
    assert README.is_file(), "plans/runs/train-tt/README.md does not exist yet (task P17.38)"
    text = README.read_text(encoding="ascii")
    for word in ("uv run tools/train_tt.py launch", "uv run tools/train_tt.py install", "train_tt.py stop",
                 "quarantine", "OQ-100"):
        assert word in text, word
