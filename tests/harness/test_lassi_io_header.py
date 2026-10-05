"""Local checks of the C and C++ header lassi_io.h (task P4.3).

Bible: Harness Contract (programs read inputs and write outputs as binary
files through lassi_io.h), Frontend Rules (a C and C++ header among the
harness bindings), Repository Layout (assets/harness/, bindings under c/).

The contract these checks fix, in assets/harness/c/lassi_io.h: one
self-contained header, usable from C99 and from C++17, that reads and writes
one array file in the format of lassi/harness/lassi_io.py (stated in
test_lassi_io_format.py). It holds an include guard and declares what the
programs in test_lassi_io_remote.py use:

- the type lassi_io_array, usable without the `struct` keyword in C, with
  the fields name (NUL-terminated text), dtype (uint32_t), rank (uint32_t),
  dims (rank uint64_t values), and data (void *, the element bytes);
- int lassi_io_read(const char *path, lassi_io_array *out), 0 on success;
- int lassi_io_write(const char *path, const char *name, uint32_t dtype,
  uint32_t rank, const uint64_t *dims, const void *data), 0 on success;
  dims may be NULL for rank 0, and data may be NULL for an empty array;
- void lassi_io_free(lassi_io_array *array), which releases what a
  successful lassi_io_read allocated;
- size_t lassi_io_itemsize(uint32_t dtype), 0 for an unknown code;
- the constants LASSI_IO_<DTYPE> for the nine required dtypes, with the
  codes of lassi.harness.lassi_io.DTYPES.

On a refusal the read and write functions return nonzero and write one line
naming the file to stderr. Only the build host compiles and runs the header
(test_lassi_io_remote.py); these local checks read its text. No value here
is a measurement.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HEADER = REPO / "assets" / "harness" / "c" / "lassi_io.h"
FUNCTIONS = ["lassi_io_read", "lassi_io_write", "lassi_io_free", "lassi_io_itemsize"]
DTYPES = ["f32", "f16", "bf16", "i32", "u32", "i8", "u8", "i64", "f64"]


def header_text() -> str:
    """Return the header's text; fail when the file does not exist."""
    assert HEADER.is_file(), f"no header at {HEADER.relative_to(REPO).as_posix()}"
    return HEADER.read_bytes().decode("ascii")


def test_the_header_is_plain_ascii() -> None:
    assert HEADER.is_file(), f"no header at {HEADER.relative_to(REPO).as_posix()}"
    data = HEADER.read_bytes()
    odd = sorted({byte for byte in data if byte > 0x7E or (byte < 0x20 and byte not in (0x09, 0x0A))})
    assert not odd, f"bytes outside plain ASCII text: {odd}"


def test_the_header_has_an_include_guard() -> None:
    text = header_text()
    guarded = re.search(r"^\s*#\s*pragma\s+once\b", text, re.MULTILINE) or re.search(
        r"^\s*#\s*ifndef\s+(\w+)\s*\n\s*#\s*define\s+\1\b", text, re.MULTILINE
    )
    assert guarded, "the header has neither #pragma once nor an #ifndef/#define guard"


@pytest.mark.parametrize("function", FUNCTIONS)
def test_the_header_declares_the_functions_the_programs_use(function: str) -> None:
    assert re.search(rf"\b{function}\s*\(", header_text()), f"the header does not declare {function}"


def test_the_header_declares_the_array_type_without_the_struct_keyword() -> None:
    text = header_text()
    assert re.search(r"\btypedef\b[^;]*\blassi_io_array\s*;", text, re.DOTALL), (
        "the header has no typedef named lassi_io_array"
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_header_names_every_required_dtype(dtype: str) -> None:
    constant = f"LASSI_IO_{dtype.upper()}"
    assert re.search(rf"\b{constant}\b", header_text()), f"the header does not define {constant}"


def test_the_header_holds_the_format_magic() -> None:
    assert "LASSIIO" in header_text()
