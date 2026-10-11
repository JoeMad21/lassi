"""Tests for `lassi corpus` on the command line (task P5.11).

Bible: Repository Layout (cli.py: lassi run | score | settings | train |
corpus | export | report), Readability Standards, Terminal Presentation
(every command decides graphics once, at start: the --graphics option,
LASSI_GRAPHICS, off without an interactive terminal, the preset, the
prompt; with graphics on it prints the banner first; with graphics off a
command prints what it prints today; a bad LASSI_GRAPHICS value exits 2
with `lassi <command>: <message>` before the command starts). Plan:
plans/p5-ir.md, task P5.11 (`lassi corpus` joins lassi/cli.py with plain
output, deciding graphics as every command does) and the planning decision
"Driver" (task P5.12 adds the round-trip driver to `lassi corpus`).

The contract these tests fix (lassi.cli), the smallest surface that reads
what P5.11 writes:

- `lassi [--graphics on|off] corpus summary <dir>` reads the corpus tables
  under <dir> (written by lassi.corpus's write_corpus, write_verification,
  or write_pair_set) and prints plain ASCII lines on stdout, among them
  `records: <n>`, one `split <split>: <n>` line per split present, and one
  `status <code>: <n>` line per gate status failure code present. It
  exits 0. The module docstring states that usage line.
- A directory that does not exist, or one that holds no corpus table,
  exits 2 with one line `lassi corpus: <message>` on stderr naming the
  directory, and nothing on stdout.
- Graphics is decided in main() as for every command: with graphics on the
  banner goes to stderr once, first, and stdout is what graphics off
  prints; no live table is built; without an option, a variable, or a
  terminal nothing is asked and no banner prints; a bad LASSI_GRAPHICS
  value exits 2 with `lassi corpus: ` and the variable's name before
  anything is read.

The tables are SYNTHETIC records (tests/corpus/corpus_fixtures.py). No value
here is a measurement.
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
from corpus_fixtures import corpus, failed_at, make_record

from lassi import cli

ESCAPE = "\x1b"
USAGE = "lassi [--graphics on|off] corpus summary <dir>"


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
    """One `lassi` invocation: its exit status, stdout, stderr, and how much of stdin it read."""

    code: int
    out: str
    err: str
    stdin_read: int


def invoke(argv: Sequence[str]) -> Session:
    """Run lassi.cli.main(argv) with fake streams; fail the test when argparse exits (an unknown command)."""
    stdin, stdout, stderr = FakeStream("y\ny\n"), FakeStream(), FakeStream()
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
    return Session(code=code, out=stdout.getvalue(), err=stderr.getvalue(), stdin_read=stdin.tell())


@pytest.fixture(autouse=True)
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the graphics and gate variables and put the graphics preset under <tmp>/config."""
    for name in ("LASSI_GRAPHICS", "LASSI_CONFIG_DIR", "LASSI_SCRATCH", "LASSI_RUNS_ROOT", "LASSI_TOOLCHAINS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LASSI_CONFIG_DIR", str(tmp_path / "config"))


@pytest.fixture
def no_live_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the command builds the live inference table or ticks it."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("lassi corpus built a live table")

    monkeypatch.setattr(cli, "LiveTable", refuse)
    monkeypatch.setattr(cli, "ticking", refuse)


def corpus_dir(tmp_path: Path) -> Path:
    """Write a SYNTHETIC corpus table of three train units, one failed at build; return its directory."""
    out = tmp_path / "corpus-out"
    records = [make_record("reader"), make_record("compute", gate_status=failed_at("build")), make_record("writer")]
    corpus().write_corpus(records, out)
    return out


def verification_dir(tmp_path: Path) -> Path:
    """Write a SYNTHETIC verification table of one eval and one unassigned record; return its directory."""
    out = tmp_path / "verify-out"
    records = [make_record("held", item="held", split="eval"), make_record("open", item="open", split="unassigned")]
    corpus().write_verification(records, out)
    return out


def summary_argv(directory: Path) -> list[str]:
    """Return the `lassi corpus summary` arguments."""
    return ["corpus", "summary", str(directory)]


# ---------------------------------------------------------------------------
# The parser and the docstring


def test_the_parser_has_the_corpus_summary_command() -> None:
    try:
        args = cli.build_parser().parse_args(summary_argv(Path("d")))
    except SystemExit:
        pytest.fail("lassi has no corpus summary command yet (task P5.11)")
    assert args.command == "corpus"


def test_the_module_docstring_states_the_usage_line() -> None:
    assert USAGE in " ".join((cli.__doc__ or "").split()), "lassi.cli's docstring does not state `lassi corpus`"


# ---------------------------------------------------------------------------
# Plain output


def test_the_summary_of_a_corpus_table_prints_plain_counts(tmp_path: Path, no_live_table: None) -> None:
    session = invoke(["--graphics", "off", *summary_argv(corpus_dir(tmp_path))])
    assert (session.code, session.err) == (0, ""), session.err
    rows = session.out.splitlines()
    assert "records: 3" in rows and "split train: 3" in rows, session.out
    assert f"status {corpus().failure_code('build')}: 1" in rows, session.out
    assert session.out.isascii() and ESCAPE not in session.out


def test_the_summary_of_a_verification_table_counts_each_split(tmp_path: Path, no_live_table: None) -> None:
    session = invoke(["--graphics", "off", *summary_argv(verification_dir(tmp_path))])
    assert (session.code, session.err) == (0, ""), session.err
    rows = session.out.splitlines()
    assert {"records: 2", "split eval: 1", "split unassigned: 1"} <= set(rows), session.out


@pytest.mark.parametrize("kind", ["missing", "empty"])
def test_a_directory_with_no_corpus_table_exits_2_naming_it(tmp_path: Path, kind: str) -> None:
    directory = tmp_path / "nothing-here"
    if kind == "empty":
        directory.mkdir()
    session = invoke(["--graphics", "off", *summary_argv(directory)])
    assert session.code == 2 and session.out == "", session.out
    assert session.err.startswith("lassi corpus: ") and session.err.count("\n") == 1, session.err
    assert str(directory) in session.err or directory.as_posix() in session.err, session.err


@pytest.mark.parametrize("damage", ["not-parquet", "held-record"])
def test_a_table_that_does_not_read_back_exits_2(tmp_path: Path, damage: str) -> None:
    if damage == "not-parquet":
        out = corpus_dir(tmp_path)
        sorted(out.rglob("*.parquet"))[0].write_bytes(b"SYNTHETIC: not a Parquet file\n")
    else:
        out = verification_dir(tmp_path)
        (out / "verification").rename(out / "corpus")
    session = invoke(["--graphics", "off", *summary_argv(out)])
    assert session.code == 2 and session.out == "", session.out
    assert session.err.startswith("lassi corpus: ") and session.err.count("\n") == 1, session.err


# ---------------------------------------------------------------------------
# Graphics, decided as every command decides it


def test_graphics_on_prints_only_the_banner_first_on_stderr_and_the_same_stdout(
    tmp_path: Path, no_live_table: None
) -> None:
    directory = corpus_dir(tmp_path)
    plain = invoke(["--graphics", "off", *summary_argv(directory)])
    session = invoke(["--graphics", "on", *summary_argv(directory)])
    assert session.code == 0, session.err
    text = banner()
    assert session.err.lstrip("\n").startswith(text), session.err
    assert session.err.count(text) == 1
    assert session.err.replace(text, "", 1).strip() == "", "graphics on adds the banner and nothing else"
    assert session.out == plain.out


def test_lassi_graphics_on_in_the_environment_prints_the_banner_too(
    tmp_path: Path, no_live_table: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = corpus_dir(tmp_path)
    monkeypatch.setenv("LASSI_GRAPHICS", "on")
    session = invoke(summary_argv(directory))
    assert session.code == 0, session.err
    assert session.err.count(banner()) == 1


def test_without_a_terminal_nothing_is_asked_and_no_banner_prints(tmp_path: Path, no_live_table: None) -> None:
    session = invoke(summary_argv(corpus_dir(tmp_path)))
    assert (session.code, session.err, session.stdin_read) == (0, "", 0), session.err
    assert banner() not in session.out and ESCAPE not in session.out
    assert not (tmp_path / "config").exists() or not list((tmp_path / "config").rglob("settings.yaml"))


def test_a_bad_lassi_graphics_value_is_refused_before_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LASSI_GRAPHICS", "maybe")
    session = invoke(summary_argv(tmp_path / "never-read"))
    assert session.code == 2 and session.out == ""
    assert session.err.startswith("lassi corpus: ") and "LASSI_GRAPHICS" in session.err, session.err
    assert "never-read" not in session.err, "the command did not start"
