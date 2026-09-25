"""Tests for the LASSI-DF banner (task P4.7).

Bible: Readability Standards, Terminal Presentation (Banner and Rules: plain
ASCII glyphs, ANSI escape sequences only on a terminal) and the Naming list
(text is plain ASCII).

The contract these tests fix, in lassi/present/banner.py:

- BANNER: str, the LASSI-DF ASCII art. Every character is printable ASCII
  (space to tilde) or a newline; no line is wider than 80 columns; it holds
  no escape sequence. A test cannot read ASCII art back, so "spells
  LASSI-DF" is checked as art of at least three non-blank lines together
  with the literal text LASSI-DF in the banner (a caption line).
- print_banner(stream) -> None writes the banner to `stream` once, ending
  with a newline. On a stream that is not a terminal (stream.isatty() is
  False) it writes BANNER exactly, apart from leading and trailing newlines,
  with no escape sequence. On a terminal it may add ANSI color, but the
  text with the escape sequences removed is still BANNER.
"""

from __future__ import annotations

import importlib
import io
import re
from types import ModuleType

import pytest

ESCAPE = "\x1b"
# An ANSI control sequence: ESC [ parameters final-byte (color and cursor codes).
ANSI_SEQUENCE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
MAX_COLUMNS = 80


def banner() -> ModuleType:
    """Import and return lassi.present.banner."""
    return importlib.import_module("lassi.present.banner")


class Stream(io.StringIO):
    """A text stream that reports a chosen isatty()."""

    def __init__(self, tty: bool) -> None:
        """Start empty, reporting `tty` from isatty()."""
        super().__init__()
        self.tty = tty

    def isatty(self) -> bool:
        """Return the chosen answer."""
        return self.tty


def test_the_banner_is_printable_ascii_and_newlines() -> None:
    text = banner().BANNER
    assert isinstance(text, str) and text.strip(), "the banner is empty"
    odd = sorted({repr(char) for char in text if char != "\n" and not " " <= char <= "~"})
    assert not odd, f"characters outside printable ASCII: {odd}"


def test_no_banner_line_is_wider_than_80_columns() -> None:
    lines = banner().BANNER.split("\n")
    wide = [(number, len(line)) for number, line in enumerate(lines, 1) if len(line) > MAX_COLUMNS]
    assert not wide, f"lines over {MAX_COLUMNS} columns: {wide}"


def test_the_banner_holds_no_escape_sequence() -> None:
    assert ESCAPE not in banner().BANNER


def test_the_banner_is_art_that_names_lassi_df() -> None:
    text = banner().BANNER
    assert "LASSI-DF" in text
    art = [line for line in text.split("\n") if line.strip() and "LASSI-DF" not in line]
    assert len(art) >= 3, f"expected ASCII art of at least three lines, got {len(art)}"


def test_print_banner_writes_the_banner_once_and_plain_off_a_terminal() -> None:
    module = banner()
    stream = Stream(tty=False)
    module.print_banner(stream)
    written = stream.getvalue()
    assert ESCAPE not in written
    assert written.endswith("\n")
    assert written.strip("\n") == module.BANNER.strip("\n")


@pytest.mark.parametrize("tty", [False, True], ids=["pipe", "terminal"])
def test_print_banner_changes_no_glyph(tty: bool) -> None:
    module = banner()
    stream = Stream(tty=tty)
    module.print_banner(stream)
    plain = ANSI_SEQUENCE.sub("", stream.getvalue())
    assert ESCAPE not in plain, "an escape character outside an ANSI sequence"
    assert plain.strip("\n") == module.BANNER.strip("\n")
