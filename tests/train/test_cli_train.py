"""Tests for `lassi train` on the command line (task P17.8).

Bible: Readability Standards, Terminal Presentation (every command decides
graphics; with graphics on it prints the banner first; the training table
is P7's, Delivery), Repository Layout (cli.py), Training Module, Agent Rules
5 and 7; plans/p17-portable.md, task P17.8 (`lassi train` decides graphics
as every command does and prints plain lines; the live training table
stays P7's).

The contract these tests fix (lassi.cli):

- `lassi [--graphics on|off] train <train.yaml> [--runs-root P]
  [--train-id ID]` runs lassi.train.run.run_training with TrainOptions
  holding the runs root and the train id, prints only the plain line
  `train directory: <dir>` on stdout, and exits 0. The module docstring
  states that usage line.
- Graphics is decided in main() as for every command: a bad LASSI_GRAPHICS
  value exits 2 with `lassi train: <message>` before anything runs, and with
  graphics on the banner is printed to stderr first and is all it adds: no
  live table is built (the training table is P7's).
- A RecipeError or RunError exits 2 with `lassi train: <message>` on stderr,
  before any directory exists: with lassi's own registry a trainer that is
  not registered is refused naming trainer.kind, and an eval split is
  refused naming bench.split (Agent Rule 5).

The fake trainer is put in place of lassi.train.run's DEFAULT_REGISTRY, as
tests/present/test_present_cli.py does for the runner, and the recipes read
the committed SYNTHETIC fixture under tests/fixtures/train/ with the host's
cpu probe, which never refuses. No value in this module is a measurement.
"""

from __future__ import annotations

import importlib
import io
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from train_fakes import COMMITTED_FIXTURE, MISSING, Log, fake_registry, recipe_data, train_run, write_recipe

from lassi import cli

ESCAPE = "\x1b"
USAGE = "lassi [--graphics on|off] train <train.yaml> [--runs-root P] [--train-id ID]"


def banner() -> str:
    """Return lassi.present.banner.BANNER without leading and trailing newlines."""
    return importlib.import_module("lassi.present.banner").BANNER.strip("\n")


class FakeStream(io.StringIO):
    """A text stream whose isatty() answers False."""

    def isatty(self) -> bool:
        """Return False: no stream here is a terminal."""
        return False


@dataclass
class Session:
    """One `lassi` invocation: its exit status, stdout, and stderr."""

    code: int
    out: str
    err: str


def invoke(argv: Sequence[str]) -> Session:
    """Run lassi.cli.main(argv) with fake streams; fail the test when argparse exits (an unknown command)."""
    stdin, stdout, stderr = FakeStream(""), FakeStream(), FakeStream()
    saved = sys.stdin, sys.stdout, sys.stderr
    sys.stdin, sys.stdout, sys.stderr = stdin, stdout, stderr
    exited: SystemExit | None = None
    code = -1
    try:
        code = cli.main(list(argv))
    except SystemExit as exit_:
        exited = exit_
    finally:
        sys.stdin, sys.stdout, sys.stderr = saved
    if exited is not None:
        pytest.fail(f"argparse exited ({exited.code}) on {list(argv)}: {stderr.getvalue().strip()}")
    return Session(code=code, out=stdout.getvalue(), err=stderr.getvalue())


@pytest.fixture
def fake_registry_in_place(monkeypatch: pytest.MonkeyPatch) -> Log:
    """Put a registry holding the fake trainer in place of lassi.train.run's DEFAULT_REGISTRY; return its log."""
    log = Log()
    monkeypatch.setattr(train_run(), "DEFAULT_REGISTRY", fake_registry(log))
    return log


@pytest.fixture
def no_live_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the command builds the live inference table or ticks it."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("lassi train built a live table; the training table is P7's")

    monkeypatch.setattr(cli, "LiveTable", refuse)
    monkeypatch.setattr(cli, "ticking", refuse)


def committed_recipe(directory: Path, **changes: Any) -> Path:
    """Write a train recipe on the committed SYNTHETIC fixture with the fake trainer on the CPU."""
    return write_recipe(directory, recipe_data(data={"synthetic": COMMITTED_FIXTURE}, **changes))


def train_argv(recipe: Path, runs_root: Path, train_id: str = "cli") -> list[str]:
    """Return the `lassi train` arguments."""
    return ["train", str(recipe), "--runs-root", str(runs_root), "--train-id", train_id]


# ---------------------------------------------------------------------------
# The parser and the docstring


def test_the_parser_has_the_train_command() -> None:
    try:
        args = cli.build_parser().parse_args(["train", "t.yaml", "--runs-root", "/r", "--train-id", "x"])
    except SystemExit:
        pytest.fail("lassi has no train command yet (task P17.8)")
    assert (args.command, args.recipe, args.runs_root, args.train_id) == ("train", Path("t.yaml"), Path("/r"), "x")
    defaults = cli.build_parser().parse_args(["train", "t.yaml"])
    assert (defaults.runs_root, defaults.train_id) == (None, None)


def test_the_module_docstring_states_the_usage_line() -> None:
    assert USAGE in " ".join((cli.__doc__ or "").split()), "lassi.cli's docstring does not state `lassi train`"


# ---------------------------------------------------------------------------
# A run with graphics off and on


def test_lassi_train_prints_the_train_directory_and_exits_zero(
    tmp_path: Path, fake_registry_in_place: Log, no_live_table: None
) -> None:
    runs_root = tmp_path / "runs-root"
    session = invoke(["--graphics", "off", *train_argv(committed_recipe(tmp_path / "recipes"), runs_root)])
    train_dir = runs_root / "train" / "cli"
    assert (session.code, session.err) == (0, ""), session.err
    assert session.out == f"train directory: {train_dir}\n"
    assert ESCAPE not in session.out
    assert fake_registry_in_place.events.count("train") == 1
    assert (train_dir / "provenance.json").is_file()


def test_lassi_train_with_graphics_on_prints_only_the_banner_on_stderr(
    tmp_path: Path, fake_registry_in_place: Log, no_live_table: None
) -> None:
    runs_root = tmp_path / "runs-root"
    session = invoke(["--graphics", "on", *train_argv(committed_recipe(tmp_path / "recipes"), runs_root)])
    assert session.code == 0, session.err
    text = banner()
    assert session.err.lstrip("\n").startswith(text), session.err
    assert session.err.count(text) == 1
    assert session.err.replace(text, "", 1).strip() == "", "graphics on adds the banner and nothing else"
    assert session.out == f"train directory: {runs_root / 'train' / 'cli'}\n"


def test_lassi_graphics_on_in_the_environment_prints_the_banner_too(
    tmp_path: Path, fake_registry_in_place: Log, no_live_table: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LASSI_GRAPHICS", "on")
    session = invoke(train_argv(committed_recipe(tmp_path / "recipes"), tmp_path / "runs-root"))
    assert session.code == 0, session.err
    assert session.err.count(banner()) == 1


def test_a_bad_lassi_graphics_value_is_refused_before_the_train_run(
    tmp_path: Path, fake_registry_in_place: Log, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LASSI_GRAPHICS", "maybe")
    runs_root = tmp_path / "runs-root"
    session = invoke(train_argv(committed_recipe(tmp_path / "recipes"), runs_root))
    assert session.code == 2 and session.out == ""
    assert session.err.startswith("lassi train: ") and "LASSI_GRAPHICS" in session.err, session.err
    assert not runs_root.exists() and fake_registry_in_place.events == []


# ---------------------------------------------------------------------------
# Refusals exit 2 before any directory


def test_a_refused_lassi_train_exits_2_with_its_message_and_creates_no_directory(tmp_path: Path) -> None:
    # lassi's own registry: the trainer named here is registered by no task.
    recipe = committed_recipe(tmp_path / "recipes", trainer={"kind": "no-such-trainer", "device": {"kind": "cpu"}})
    runs_root = tmp_path / "runs-root"
    session = invoke(["--graphics", "off", *train_argv(recipe, runs_root)])
    assert session.code == 2 and session.out == "", session.out
    assert session.err.startswith("lassi train: "), session.err
    assert "trainer.kind" in session.err and "no-such-trainer" in session.err, session.err
    assert session.err.count("\n") == 1, "one line"
    assert not runs_root.exists()


def test_lassi_train_on_an_eval_split_exits_2_naming_bench_split(
    tmp_path: Path, fake_registry_in_place: Log
) -> None:
    data = recipe_data(data=MISSING, bench={"suite": "lassi-hecbench-10", "split": "eval"})
    recipe = write_recipe(tmp_path / "recipes", data)
    runs_root = tmp_path / "runs-root"
    session = invoke(["--graphics", "off", *train_argv(recipe, runs_root)])
    assert session.code == 2 and session.out == ""
    assert session.err.startswith("lassi train: ") and "bench.split" in session.err, session.err
    assert "Agent Rule 5" in session.err, session.err
    assert not runs_root.exists() and "train" not in fake_registry_in_place.events


def test_a_run_error_from_the_layer_exits_2(tmp_path: Path, fake_registry_in_place: Log) -> None:
    recipe = committed_recipe(tmp_path / "recipes", lora={"r": 8, "targets": "all-linear"})
    runs_root = tmp_path / "runs-root"
    session = invoke(["--graphics", "off", *train_argv(recipe, runs_root)])
    assert session.code == 2 and session.out == ""
    assert session.err.startswith("lassi train: ") and "lora" in session.err, session.err
    assert not runs_root.exists()
