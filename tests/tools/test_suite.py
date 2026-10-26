"""Tests for tools/suite.py, the owner's launcher for the full RNGD suite on alpha01, and its remote side.

The launcher (the owner's direction of 2026-10-26: the full suite launched unattended, with a help option on
every command) wraps `uv run tools/rx.py`; the
remote side is plans/runs/suite/run-suite.sh, with its recipe helper plans/runs/suite/recipes.py. Every test
here runs on the workstation with fakes for rx and git: no test calls rx, ssh, or alpha01. The fake host's
values (free space, job ids, commits) are PLACEHOLDER values, never measurements.

The interface these tests fix:

- `main(argv) -> int` with the subcommands plan, launch, status, wait, stop, and pull; argparse usage errors
  exit 2, a refused request returns 2, a failed preflight returns 1.
- Every child process goes through `subprocess.run` with an environment that holds MSYS_NO_PATHCONV=1 and
  PYTHONIOENCODING=utf-8; rx is `uv run tools/rx.py ...` and git is `git -C <repo> ...`.
- `now_epoch()` is the clock (tests replace it); deadlines are US Eastern clock times or epoch seconds.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SUITE = REPO / "tools" / "suite.py"
HELPER = REPO / "plans" / "runs" / "suite" / "recipes.py"
SCRIPT = REPO / "plans" / "runs" / "suite" / "run-suite.sh"
SWEEP7 = "projects/lassi-demo/sweep7"

# 2026-10-25 20:00:00 EDT, the night window's start, as the fixed clock (UTC-4).
EDT = timezone(timedelta(hours=-4))
NOW = int(datetime(2026, 10, 25, 20, 0, tzinfo=EDT).timestamp())
JOB_ID = "20261026-170035-suite-ab12"  # PLACEHOLDER job id in the gate's format
HEAD = "0123456789abcdef0123456789abcdef01234567"  # PLACEHOLDER commit

LAUNCH_OPTIONS = ("--models", "--table", "--sweep", "--only", "--hours", "--until", "--skip-download", "--name",
                  "--slot", "--yes", "--allow-dirty", "--silicon")
PLAN_OPTIONS = tuple(option for option in LAUNCH_OPTIONS if option != "--yes")
SUBCOMMAND_OPTIONS = {
    "plan": PLAN_OPTIONS,
    "launch": LAUNCH_OPTIONS,
    "status": ("--lines",),
    "wait": ("--timeout", "--interval"),
    "stop": ("--clear",),
    "pull": ("--name",),
}


def load(path: Path, name: str) -> ModuleType:
    """Import the module at `path` under `name`."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def suite(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Return tools/suite.py with the clock fixed at NOW."""
    module = load(SUITE, "lassi_tools_suite")
    monkeypatch.setattr(module, "now_epoch", lambda: NOW)
    return module


class FakeHost:
    """Stand-in for subprocess.run: answers git and rx, records each call's argv and environment."""

    def __init__(self, repo: Path, *, dirty: bool = False, pushed: bool = True, free_gb: float = 300.0,
                 rngd: bool = True, stop: bool = False, jobs: str = "", job_state: str = "done") -> None:
        self.repo = repo
        self.dirty, self.pushed = dirty, pushed
        self.doctor = {"stop": stop, "scratch_free_gb": free_gb, "running_jobs": [],
                       "devices_enabled": {"rngd": rngd, "tt_silicon": False}}
        self.jobs, self.job_state = jobs, job_state
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
        assert argv[1:3] == ["-C", str(self.repo)], argv
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
            body = json.dumps(self.doctor, indent=2)
            return self.done(argv, f"[rx] settings: {{}}\n[rx] ssh ionx: PASS\n{body}\n[rx] Ready\n")
        if words == ["job", "status"]:
            return self.done(argv, self.jobs)
        if words[:2] == ["job", "status"]:
            return self.done(argv, f"{words[2]}  {self.job_state}  rc=0  slot=suite  start=x  end=y  cmd=bash\n")
        if words[:2] == ["job", "start"]:
            return self.done(argv, json.dumps({"id": JOB_ID, "state": "starting", "slot": "suite"}) + "\n")
        if words[:1] in (["exec"], ["job"]):
            return self.done(argv, "")
        if words[:1] == ["pull"]:
            return self.pull(argv, words)
        raise AssertionError(f"unexpected rx call {argv}")

    def pull(self, argv: list[str], words: list[str]) -> subprocess.CompletedProcess:
        if words[1] == "--path":
            root = self.repo / ".rx" / "pulls" / Path(words[2]).name
            (root / "tables").mkdir(parents=True)
            (root / "calls.tsv").write_text("call\tslug\tsweep\tstamp\trc\trecipes\tskip\n"
                                            "1\tllama33-70b\tprojects/lassi-demo/sweep8\t20261026-1712\t0\t10\t-\n")
            (root / "tables" / "passes.tsv").write_text(
                "pass\tslug\ttrials\ts5\talignment_1\twith_divergence\tcombo\tsim_gap_attempts\n"
                "20261025-1712\tllama33-70b\t15\t3\t2\t0\tex\t0\n"
                "20261026-1712\tllama33-70b\t15\t6\t5\t0\tv1ex\t1\n")
            (root / "sweep-1-llama33-70b.log").write_text("plain ascii log\n")
            (root / "fetch-gpt-oss-120b.log").write_bytes(b"progress " + bytes([0xE2, 0x96, 0x88]) + b"\n")
            return self.done(argv, f"[rx] pulled 5 entries into {root}\n")
        record = self.repo / ".rx" / "pulls" / words[1]  # rx extracts the job record here, named by the job id
        record.mkdir(parents=True)
        (record / "meta.json").write_text("{}")
        (record / "output.log").write_text("the job's log\n")
        into = self.repo / words[words.index("--into") + 1]
        into.mkdir(parents=True, exist_ok=True)
        (into / "provenance.json").write_text(json.dumps({"id": words[1], "commit": HEAD, "dirty": False,
                                                          "state": "done", "rc": 0}))
        return self.done(argv, f"[rx] wrote {into / 'provenance.json'}\n")

    def rx_calls(self) -> list[list[str]]:
        return [argv[3:] for argv, _ in self.calls if argv[:3] == ["uv", "run", "tools/rx.py"]]


@pytest.fixture()
def host(monkeypatch: pytest.MonkeyPatch, suite: ModuleType) -> FakeHost:
    """Route every child process of the launcher to a FakeHost over the real repository."""
    fake = FakeHost(REPO)
    monkeypatch.setattr(suite.subprocess, "run", fake)
    monkeypatch.setattr(suite, "rip_stop_present", lambda: False)
    return fake


def expected_remote(deadline: int, extra: str = "") -> str:
    """Return the remote command for the default models over sweep7."""
    return (f"bash plans/runs/suite/run-suite.sh --deadline {deadline} --table plans/runs/suite/models.tsv "
            f"--models llama33-70b,gpt-oss-120b --sweep {SWEEP7}{extra}")


# ---------------------------------------------------------------- help


def test_top_level_help_lists_every_subcommand(suite: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        suite.main(["--help"])
    assert info.value.code == 0
    text = capsys.readouterr().out
    for name in SUBCOMMAND_OPTIONS:
        assert name in text
    assert "example" in text.lower()


@pytest.mark.parametrize("name", sorted(SUBCOMMAND_OPTIONS))
def test_each_subcommand_help_lists_its_options(suite: ModuleType, capsys: pytest.CaptureFixture[str],
                                                name: str) -> None:
    with pytest.raises(SystemExit) as info:
        suite.main([name, "--help"])
    assert info.value.code == 0
    text = capsys.readouterr().out
    for option in SUBCOMMAND_OPTIONS[name]:
        assert option in text, f"{name} --help does not list {option}"
    assert "example" in text.lower()


def test_every_option_carries_help_text(suite: ModuleType) -> None:
    parser = suite.build_parser()
    subparsers = [action for action in parser._actions if action.choices and isinstance(action.choices, dict)]
    assert subparsers
    for name, sub in subparsers[0].choices.items():
        for action in sub._actions:
            if action.option_strings and action.dest != "help":
                assert action.help, f"{name} {action.option_strings} has no help text"


# ---------------------------------------------------------------- plan


def test_plan_prints_the_exact_rx_command(suite: ModuleType, host: FakeHost,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    assert suite.main(["plan", "--sweep", SWEEP7, "--hours", "6"]) == 0
    out = capsys.readouterr().out
    deadline = NOW + 6 * 3600
    remote = expected_remote(deadline)
    line = ("MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8 uv run tools/rx.py job start --big --slot suite --name suite "
            f"--timeout {6 * 3600 + 900} -- '{remote}'")
    assert line in out
    assert str(deadline) in out and "2026-10-26 02:00 EDT" in out
    assert "llama33-70b" in out and "gpt-oss-120b" in out
    assert "18 recipes" in out  # sweep7: 8 for Llama, 10 for gpt-oss-120b
    assert "65 GB" in out  # the gate's 60 GB big-job floor and the night's 5 GB margin
    assert host.rx_calls() == []  # plan never contacts the build host


def test_plan_changes_nothing_and_reports_a_dirty_tree(suite: ModuleType, host: FakeHost,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    host.dirty = True
    assert suite.main(["plan", "--sweep", SWEEP7, "--hours", "6"]) == 0
    out = capsys.readouterr().out
    assert "dirty" in out and "exploratory" in out
    assert host.rx_calls() == []
    for argv, _ in host.calls:
        assert argv[0] == "git" and argv[3] in ("status", "rev-parse", "branch")


def test_only_filter_reaches_the_remote_command_and_the_count(suite: ModuleType, host: FakeHost,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
    assert suite.main(["plan", "--sweep", SWEEP7, "--only", "base-*", "--hours", "6"]) == 0
    out = capsys.readouterr().out
    assert expected_remote(NOW + 6 * 3600, ' --only "base-*"') in out
    assert "4 recipes" in out


def test_the_default_sweep_is_sweep8(suite: ModuleType) -> None:
    args = suite.build_parser().parse_args(["plan", "--hours", "6"])
    assert suite.sweeps_of(args) == ["projects/lassi-demo/sweep8"]


def test_unknown_models_are_rejected(suite: ModuleType, host: FakeHost, capsys: pytest.CaptureFixture[str]) -> None:
    assert suite.main(["plan", "--sweep", SWEEP7, "--models", "llama33-70b,no-such-model", "--hours", "6"]) == 2
    err = capsys.readouterr().err
    assert "no-such-model" in err and "llama33-70b" in err and "gpt-oss-120b" in err


def test_a_missing_sweep_is_rejected(suite: ModuleType, host: FakeHost, capsys: pytest.CaptureFixture[str]) -> None:
    assert suite.main(["plan", "--sweep", "projects/lassi-demo/no-such-sweep", "--hours", "6"]) == 2
    assert "no-such-sweep" in capsys.readouterr().err


def test_a_recipe_that_does_not_load_is_rejected(suite: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                                 tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "plans" / "runs" / "suite").mkdir(parents=True)
    shutil.copy(REPO / "plans" / "runs" / "suite" / "models.tsv", tmp_path / "plans" / "runs" / "suite")
    broken = tmp_path / "sweepx" / "llama33-70b"
    broken.mkdir(parents=True)
    (broken / "base-maxabs.yaml").write_text("extends: no-such-parent.yaml\n")
    fake = FakeHost(tmp_path)
    monkeypatch.setattr(suite, "REPO", tmp_path)
    monkeypatch.setattr(suite.subprocess, "run", fake)
    assert suite.main(["plan", "--sweep", "sweepx", "--hours", "6"]) == 2
    assert "base-maxabs.yaml" in capsys.readouterr().err


# ---------------------------------------------------------------- deadline


def test_hours_and_until_are_exclusive(suite: ModuleType, host: FakeHost) -> None:
    with pytest.raises(SystemExit) as info:
        suite.main(["plan", "--hours", "6", "--until", "01:45"])
    assert info.value.code == 2


def test_a_deadline_is_required(suite: ModuleType, host: FakeHost) -> None:
    with pytest.raises(SystemExit) as info:
        suite.main(["plan"])
    assert info.value.code == 2


def test_until_reads_an_eastern_clock_time(suite: ModuleType) -> None:
    # 01:45 after 20:00 EDT on 2026-10-25 is 01:45 EDT on 2026-10-26.
    assert suite.parse_deadline("01:45", NOW) == int(datetime(2026, 10, 26, 1, 45, tzinfo=EDT).timestamp())
    # A clock time later the same evening stays on the same day.
    assert suite.parse_deadline("23:30", NOW) == int(datetime(2026, 10, 25, 23, 30, tzinfo=EDT).timestamp())
    # An epoch passes through, and a dated time in November is read as EST (UTC-5).
    assert suite.parse_deadline("1793000000", NOW) == 1793000000
    est = timezone(timedelta(hours=-5))
    assert suite.parse_deadline("2026-11-02 01:45", NOW) == int(datetime(2026, 11, 2, 1, 45, tzinfo=est).timestamp())


def test_the_job_timeout_is_the_deadline_plus_the_margin(suite: ModuleType, host: FakeHost,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    assert suite.main(["plan", "--sweep", SWEEP7, "--until", "01:45"]) == 0
    out = capsys.readouterr().out
    deadline = int(datetime(2026, 10, 26, 1, 45, tzinfo=EDT).timestamp())
    assert f"--deadline {deadline}" in out
    assert f"--timeout {deadline - NOW + 900}" in out
    assert "2026-10-26 01:45 EDT" in out


def test_a_deadline_too_close_is_refused(suite: ModuleType, host: FakeHost,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    assert suite.main(["plan", "--sweep", SWEEP7, "--hours", "0.25"]) == 2
    assert "deadline" in capsys.readouterr().err


def test_a_deadline_past_the_gate_limit_is_refused(suite: ModuleType, host: FakeHost,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    assert suite.main(["plan", "--sweep", SWEEP7, "--hours", "48"]) == 2
    assert "172800" in capsys.readouterr().err


# ---------------------------------------------------------------- silicon


@pytest.mark.parametrize("name", ["plan", "launch"])
def test_silicon_is_refused(suite: ModuleType, host: FakeHost, capsys: pytest.CaptureFixture[str],
                            name: str) -> None:
    assert suite.main([name, "--silicon", "--hours", "6"]) == 2
    err = capsys.readouterr().err
    assert "docs/tt-silicon-iommu.md" in err and "OQ-078" in err and "OQ-099" in err
    assert "Wormhole" in err
    assert host.rx_calls() == []


# ---------------------------------------------------------------- launch


def test_launch_sets_msys_no_pathconv_in_every_child(suite: ModuleType, host: FakeHost) -> None:
    assert suite.main(["launch", "--sweep", SWEEP7, "--hours", "6", "--yes"]) == 0
    assert host.calls
    for argv, env in host.calls:
        assert env.get("MSYS_NO_PATHCONV") == "1", argv
        assert env.get("PYTHONIOENCODING") == "utf-8", argv


def test_launch_starts_the_job_and_prints_how_to_follow_it(suite: ModuleType, host: FakeHost,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    assert suite.main(["launch", "--sweep", SWEEP7, "--hours", "6", "--yes"]) == 0
    remote = expected_remote(NOW + 6 * 3600)
    start = ["job", "start", "--big", "--slot", "suite", "--name", "suite", "--timeout", str(6 * 3600 + 900),
             "--", remote]
    assert start in host.rx_calls()
    assert host.rx_calls().index(["doctor"]) < host.rx_calls().index(start)
    out = capsys.readouterr().out
    assert JOB_ID in out
    for command in (f"suite.py status {JOB_ID}", f"suite.py wait {JOB_ID}", "suite.py stop", f"suite.py pull {JOB_ID}"):
        assert command in out


def test_launch_without_yes_asks_and_stops_on_no(suite: ModuleType, host: FakeHost,
                                                 monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture[str]) -> None:
    prompts: list[str] = []
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or "n")
    assert suite.main(["launch", "--sweep", SWEEP7, "--hours", "6"]) != 0
    assert prompts == ["Start this on alpha01? [y/N] "]
    assert not [call for call in host.rx_calls() if call[:2] == ["job", "start"]]
    assert "rx.py job start" in capsys.readouterr().out  # the plan was shown before the question


def test_launch_without_yes_starts_on_yes(suite: ModuleType, host: FakeHost,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    assert suite.main(["launch", "--sweep", SWEEP7, "--hours", "6"]) == 0
    assert [call for call in host.rx_calls() if call[:2] == ["job", "start"]]


def test_a_dirty_tree_is_refused_without_allow_dirty(suite: ModuleType, host: FakeHost,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    host.dirty = True
    assert suite.main(["launch", "--sweep", SWEEP7, "--hours", "6", "--yes"]) != 0
    assert "--allow-dirty" in capsys.readouterr().err
    assert not [call for call in host.rx_calls() if call[:2] == ["job", "start"]]


def test_an_unpushed_commit_is_refused_without_allow_dirty(suite: ModuleType, host: FakeHost,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    host.pushed = False
    assert suite.main(["launch", "--sweep", SWEEP7, "--hours", "6", "--yes"]) != 0
    assert "push" in capsys.readouterr().err
    assert not [call for call in host.rx_calls() if call[:2] == ["job", "start"]]


def test_allow_dirty_starts_and_labels_the_run_exploratory(suite: ModuleType, host: FakeHost,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    host.dirty = True
    assert suite.main(["launch", "--sweep", SWEEP7, "--hours", "6", "--yes", "--allow-dirty"]) == 0
    assert "exploratory" in capsys.readouterr().out
    assert [call for call in host.rx_calls() if call[:2] == ["job", "start"]]


@pytest.mark.parametrize(("change", "words"), [
    ({"rngd": False}, "rngd"),
    ({"free_gb": 40.0}, "65"),
    ({"stop": True}, "STOP"),
    ({"jobs": "20261026-160000-other-aaaa  running  rc=None  slot=x  start=s  end=None  cmd=bash x\n"},
     "20261026-160000-other-aaaa"),
])
def test_preflight_failures_stop_the_launch(suite: ModuleType, host: FakeHost, capsys: pytest.CaptureFixture[str],
                                            change: dict[str, Any], words: str) -> None:
    for key, value in change.items():
        if key == "jobs":
            host.jobs = value
        elif key == "free_gb":
            host.doctor["scratch_free_gb"] = value
        elif key == "rngd":
            host.doctor["devices_enabled"]["rngd"] = value
        else:
            host.doctor[key] = value
    assert suite.main(["launch", "--sweep", SWEEP7, "--hours", "6", "--yes"]) == 1
    assert words in capsys.readouterr().err
    assert not [call for call in host.rx_calls() if call[:2] == ["job", "start"]]


def test_a_rip_stop_file_stops_the_launch(suite: ModuleType, host: FakeHost, monkeypatch: pytest.MonkeyPatch,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(suite, "rip_stop_present", lambda: True)
    assert suite.main(["launch", "--sweep", SWEEP7, "--hours", "6", "--yes"]) == 1
    assert ".rip-stop" in capsys.readouterr().err
    assert host.rx_calls() == []


# ---------------------------------------------------------------- status, wait, stop, pull


def test_status_and_wait_wrap_rx(suite: ModuleType, host: FakeHost) -> None:
    assert suite.main(["status"]) == 0
    assert suite.main(["status", JOB_ID, "--lines", "0"]) == 0
    assert suite.main(["wait", JOB_ID, "--timeout", "60", "--interval", "30"]) == 0
    calls = host.rx_calls()
    assert ["job", "status"] in calls
    assert ["job", "status", JOB_ID] in calls
    assert ["job", "wait", JOB_ID, "--timeout", "60", "--interval", "30"] in calls


def test_stop_writes_the_sweep_stop_file(suite: ModuleType, host: FakeHost) -> None:
    assert suite.main(["stop"]) == 0
    assert suite.main(["stop", "--clear"]) == 0
    execs = [call for call in host.rx_calls() if call[0] == "exec"]
    assert len(execs) == 2
    assert execs[0][1] == "--" and 'touch "$LASSI_RUNS_ROOT/rngd-sweep/STOP"' in execs[0][2]
    assert 'rm -f "$LASSI_RUNS_ROOT/rngd-sweep/STOP"' in execs[1][2]


def test_pull_copies_ascii_files_and_prints_the_counts(suite: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                                       tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeHost(tmp_path)
    monkeypatch.setattr(suite, "REPO", tmp_path)
    monkeypatch.setattr(suite.subprocess, "run", fake)
    assert suite.main(["pull", JOB_ID]) == 0
    calls = fake.rx_calls()
    assert ["pull", JOB_ID, "--into", "results/suite-20261026-170035"] in calls
    assert ["pull", "--path", f"lassi-runs/suite/{JOB_ID}"] in calls
    night = tmp_path / "results" / "suite-20261026-170035" / "night"
    assert (night / "calls.tsv").is_file() and (night / "tables" / "passes.tsv").is_file()
    assert (night / "sweep-1-llama33-70b.log").is_file()
    assert not (night / "fetch-gpt-oss-120b.log").exists()
    assert not (night / "output.log").exists() and not (night / "meta.json").exists()  # the record stays apart
    out = capsys.readouterr().out
    assert "fetch-gpt-oss-120b.log" in out  # named as left out
    assert "20261026-1712" in out and "v1ex" in out
    assert "20261025-1712" not in out  # an earlier pass in the whole-root table is not this job's


def test_pull_refuses_a_running_job(suite: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                    capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeHost(tmp_path, job_state="running")
    monkeypatch.setattr(suite, "REPO", tmp_path)
    monkeypatch.setattr(suite.subprocess, "run", fake)
    assert suite.main(["pull", JOB_ID]) == 1
    assert "running" in capsys.readouterr().err
    assert not [call for call in fake.rx_calls() if call[0] == "pull"]


def test_pull_refuses_an_existing_results_folder(suite: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                                 tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeHost(tmp_path)
    monkeypatch.setattr(suite, "REPO", tmp_path)
    monkeypatch.setattr(suite.subprocess, "run", fake)
    (tmp_path / "results" / "taken").mkdir(parents=True)
    (tmp_path / "results" / "taken" / "summary.md").write_text("x\n")
    assert suite.main(["pull", JOB_ID, "--name", "taken"]) == 2
    assert "taken" in capsys.readouterr().err


# ---------------------------------------------------------------- the remote side


def test_helper_manifest_and_select(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    helper = load(HELPER, "lassi_suite_recipes")
    manifest = tmp_path / "recipes.tsv"
    resolved = tmp_path / "resolved"
    assert helper.main(["manifest", "--out", str(manifest), "--resolved", str(resolved), "--sweep", SWEEP7,
                        "--model", "llama33-70b", "--only", "all-*", "--only", "base-*"]) == 0
    rows = [line.split("\t") for line in manifest.read_text().splitlines()]
    assert rows[0][:4] == ["sweep", "slug", "recipe", "selected"]
    selected = sorted(row[2] for row in rows[1:] if row[3] == "yes")
    assert selected == ["all-maxabs", "all-pcc", "base-maxabs", "base-pcc"]
    assert len(rows) == 9  # every recipe of sweep7/llama33-70b is listed, selected or not
    assert len(list(resolved.rglob("*.yaml"))) == 4
    capsys.readouterr()
    assert helper.main(["needs", "--manifest", str(manifest)]) == 0
    assert capsys.readouterr().out.split() == ["context:tt-metalium-5280a9cf-full", "examples:tt-examples-v0",
                                               "retrieval:tt-metal-docs"]
    assert helper.main(["select", "--manifest", str(manifest), "--sweep", SWEEP7, "--model", "llama33-70b",
                        "--failed", "retrieval:tt-metal-docs"]) == 0
    count, pattern = capsys.readouterr().out.split()
    assert count == "2"
    assert pattern == "^(all-maxabs|all-pcc|ex-maxabs|ex-pcc|lk-maxabs|lk-pcc)$"


def test_helper_select_with_nothing_skipped_prints_an_empty_pattern(tmp_path: Path,
                                                                    capsys: pytest.CaptureFixture[str]) -> None:
    helper = load(HELPER, "lassi_suite_recipes")
    manifest = tmp_path / "recipes.tsv"
    assert helper.main(["manifest", "--out", str(manifest), "--resolved", str(tmp_path / "r"), "--sweep", SWEEP7,
                        "--model", "gpt-oss-120b"]) == 0
    capsys.readouterr()
    assert helper.main(["select", "--manifest", str(manifest), "--sweep", SWEEP7, "--model", "gpt-oss-120b"]) == 0
    assert capsys.readouterr().out == "10\t\n"


def bash() -> str | None:
    """Return a bash to check the script with, or None."""
    return shutil.which("bash")


@pytest.mark.skipif(bash() is None, reason="no bash on PATH")
def test_the_remote_script_parses() -> None:
    result = subprocess.run([bash(), "-n", "plans/runs/suite/run-suite.sh"], cwd=REPO, capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(bash() is None, reason="no bash on PATH")
def test_the_remote_script_help_lists_its_options() -> None:
    result = subprocess.run([bash(), "plans/runs/suite/run-suite.sh", "--help"], cwd=REPO, capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr
    for option in ("--deadline", "--models", "--sweep", "--only", "--table", "--skip-download", "--help"):
        assert option in result.stdout
    bad = subprocess.run([bash(), "plans/runs/suite/run-suite.sh", "--no-such-option"], cwd=REPO,
                         capture_output=True, text=True)
    assert bad.returncode == 2


def test_the_suite_table_keeps_run_sh_columns() -> None:
    rows = [line.split("\t") for line in (REPO / "plans" / "runs" / "suite" / "models.tsv").read_text().splitlines()
            if line and not line.startswith("#")]
    night = [line.split("\t") for line in (REPO / "plans" / "runs" / "rngd-sweep" / "models10.tsv").read_text()
             .splitlines() if line and not line.startswith("#")]
    assert [row[:6] for row in rows] == night
    assert all(len(row) == 7 and row[6].isdigit() for row in rows)
