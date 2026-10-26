"""Tests for the tt-host-v1 prompt set: tt-host-v0 with two generic harness statements (task P17.27).

tt-host-v1 is tt-host-v0 with exactly the edits below and nothing else, so a run on either set is comparable with
the other and tt-host-v0's results stay as they were. The two statements rest on the harness and the toolchains,
not on any item's answer (Agent Rule 5; plans/p17-portable.md, P17.27):

- lassi_io_free releases an array and zeroes it (assets/harness/c/lassi_io.h: memset of the whole struct), so a
  program that frees an input before the write that uses its rank or dims writes a rank-0 array.
- a cpp program is built by gcc-native alone (toolchains/gcc.pin), with no tt-metal header or library on its
  include path, so a cpp program that includes a tt-metalium or kernel API header cannot build; it has the C++
  standard library, OpenMP (-fopenmp), and lassi_io.h.
"""

from __future__ import annotations

import string
from pathlib import Path

import pytest

from lassi.prompts import render

REPO = Path(__file__).resolve().parents[2]
V0 = REPO / "assets" / "prompts" / "tt-host-v0"
V1 = REPO / "assets" / "prompts" / "tt-host-v1"
SET = "tt-host-v1"
MAX_LINES = 60

CPP_OLD = "- cpp: a C++ program for the host CPU, built by g++ 12 with -O3 -fopenmp.\n"
CPP_NEW = (
    "- cpp: a C++ program for the host CPU, built by g++ 12 with -O3 -fopenmp, with no tt-metal header or\n"
    "  library on its path: it uses the C++ standard library, OpenMP, and \"lassi_io.h\", and does any bf16\n"
    "  conversion, tiling, and tile math itself.\n"
)
FREE_OLD = "The harness provides lassi_io.h; never return it.\n"
FREE_NEW = (
    "The harness provides lassi_io.h; never return it.\n"
    "lassi_io_free zeroes the array it frees, so write each output before freeing an input whose dims it uses.\n"
)
EDITS: dict[str, list[tuple[str, str]]] = {
    "generate": [(CPP_OLD, CPP_NEW), (FREE_OLD, FREE_NEW)],
    "correct": [(CPP_OLD, CPP_NEW), (FREE_OLD, FREE_NEW)],
    "correct_mismatch": [(CPP_OLD, CPP_NEW), (FREE_OLD, FREE_NEW)],
}


def edited(name: str, text: str) -> str:
    """Return tt-host-v0's text of template `name` with tt-host-v1's edits applied; each old text occurs once."""
    for old, new in EDITS.get(name, []):
        assert text.count(old) == 1, f"{name}.txt: {old!r} occurs {text.count(old)} times in tt-host-v0"
        text = text.replace(old, new)
    return text


def test_v1_is_v0_with_exactly_the_edits() -> None:
    v0 = sorted(p.name for p in V0.iterdir())
    assert sorted(p.name for p in V1.iterdir()) == v0
    for name in v0:
        old = (V0 / name).read_bytes().decode("ascii")
        new = (V1 / name).read_bytes()
        assert new.isascii() and b"\r" not in new, name
        assert new.decode("ascii") == edited(name.removesuffix(".txt"), old), name


@pytest.mark.parametrize("name", sorted(EDITS))
def test_v1_states_the_free_rule_and_the_cpp_toolchain(name: str) -> None:
    text = (V1 / f"{name}.txt").read_text(encoding="ascii")
    flat = " ".join(text.split())
    assert "lassi_io_free zeroes the array it frees" in flat
    assert "write each output before freeing an input whose dims it uses" in flat
    assert "with no tt-metal header or library" in flat
    assert "uses the C++ standard library, OpenMP, and" in flat


@pytest.mark.parametrize("name", sorted(EDITS))
def test_v1_template_is_short_and_renders(name: str) -> None:
    text = (V1 / f"{name}.txt").read_text(encoding="ascii")
    assert len(text.splitlines()) <= MAX_LINES
    for _, new in EDITS[name]:
        assert max(len(line) for line in new.splitlines()) <= 110
    found = string.Template.pattern.findall(text)
    names = {named or braced for _, named, braced, _ in found if named or braced}
    assert render(SET, name, dict.fromkeys(names, "")).isascii()


def test_v0_is_unchanged_by_the_new_set() -> None:
    # tt-host-v0 keeps its text, so earlier runs' prompts stay reproducible (each recipe names its set).
    for name in EDITS:
        text = (V0 / f"{name}.txt").read_text(encoding="ascii")
        assert "lassi_io_free zeroes" not in text and "no tt-metal header" not in text
