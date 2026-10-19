"""Tests for lassi.harness write_inputs: the writer of a proposal's input files (task P16.5).

Bible: Harness Contract (lassi_io files; held-out input files: a program
reads @inputs/<name>.lassiio), Adversarial Setups (item 1; Input contract),
Agent Rule 6. Plan: plans/p16-adversarial.md, P16.5 (each run on a
proposal gets its @inputs written from the proposal by a new writer;
stage_inputs draws only the item's own inputs).

The contract these tests fix:

- lassi.harness exports write_inputs(arrays, out_dir) -> dict[str, Path],
  defined in lassi.harness.lassi_io beside generate_inputs. For each
  LassiArray, in the order given, it writes <out_dir>/<name>.lassiio (the
  array's own name and SUFFIX) holding the array as write_array writes it,
  and returns name -> path in that order: the paths program_args names
  under @inputs, so a program reads a proposal as it reads its held-out
  inputs.
- Round trip: for the arrays read back from generate_inputs' files it
  writes the same bytes, file for file, so neither check_inputs nor a
  program can tell a written set from a drawn one.
- Refusals, as generate_inputs refuses: LassiIOError (a ValueError) whose
  message names out_dir, raised before any file is written, for an out_dir
  that is not a directory, a file that already exists at one of the paths,
  a name used twice, and an array lassi_io cannot write (a bad name, a name
  over MAX_INPUT_NAME_BYTES, an unknown dtype, data whose length is not
  the shape's); out_dir is left as it was.

Every array here is SYNTHETIC; no value in this module is a measurement.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest

import lassi.harness as harness
from lassi.bench.registry import INPUTS_DIR
from lassi.harness import LassiArray, LassiIOError, check_inputs, generate_inputs, input_contract, read_array
from lassi.harness.lassi_io import MAX_INPUT_NAME_BYTES, SUFFIX

# A SYNTHETIC held-out input spec: a matrix, an integer vector, a scalar, and an empty array.
SPEC = [
    {"name": "a", "dtype": "f32", "shape": [3, 2], "dist": "uniform", "lo": -1.0, "hi": 1.0},
    {"name": "n", "dtype": "i64", "shape": [5], "dist": "integers", "lo": -7, "hi": 7},
    {"name": "h", "dtype": "bf16", "shape": [], "dist": "uniform", "lo": 0.0, "hi": 2.0},
    {"name": "e", "dtype": "u8", "shape": [0], "dist": "integers", "lo": 0, "hi": 4},
]
SEED = 5
X = LassiArray("x", "f32", (2,), struct.pack("<2f", 1.5, -2.0))
Y = LassiArray("y", "i32", (3,), struct.pack("<3i", 1, 2, 3))


def write_inputs() -> Any:
    """Return lassi.harness.write_inputs, failing with the task's name when it does not exist yet."""
    assert hasattr(harness, "write_inputs"), "lassi.harness has no write_inputs (task P16.5)"
    return harness.write_inputs


def drawn(tmp_path: Path) -> tuple[dict[str, Path], list[LassiArray]]:
    """Return generate_inputs' files for SPEC and SEED and the arrays read back from them, in SPEC order."""
    out = tmp_path / "drawn"
    out.mkdir()
    paths = generate_inputs(SPEC, SEED, out)
    return paths, [read_array(paths[entry["name"]]) for entry in SPEC]


def listing(directory: Path) -> dict[str, bytes]:
    """Return every file under `directory` by relative POSIX path, with its bytes."""
    files = [path for path in directory.rglob("*") if path.is_file()]
    return {path.relative_to(directory).as_posix(): path.read_bytes() for path in files}


def test_write_inputs_is_exported_from_lassi_harness() -> None:
    assert callable(write_inputs())
    assert "write_inputs" in harness.__all__
    assert write_inputs().__module__ == "lassi.harness.lassi_io"


def test_it_writes_the_bytes_generate_inputs_writes_for_the_same_arrays(tmp_path: Path) -> None:
    paths, arrays = drawn(tmp_path)
    out = tmp_path / "written"
    out.mkdir()
    written = write_inputs()(arrays, out)
    assert list(written) == [entry["name"] for entry in SPEC]
    for name, path in written.items():
        assert path.read_bytes() == paths[name].read_bytes(), f"{name}: the bytes differ from generate_inputs'"
    assert set(listing(out)) == {f"{entry['name']}{SUFFIX}" for entry in SPEC}


def test_each_file_is_named_as_program_args_names_an_input(tmp_path: Path) -> None:
    out = tmp_path / INPUTS_DIR
    out.mkdir()
    written = write_inputs()([Y, X], out)
    assert list(written) == ["y", "x"]
    assert written == {"y": out / f"y{SUFFIX}", "x": out / f"x{SUFFIX}"}
    assert read_array(written["x"]) == X and read_array(written["y"]) == Y


def test_a_written_set_passes_the_contract_its_arrays_fit(tmp_path: Path) -> None:
    _, arrays = drawn(tmp_path)
    out = tmp_path / "written"
    out.mkdir()
    written = write_inputs()(list(reversed(arrays)), out)
    assert check_inputs(input_contract(SPEC), list(written.values())) is None


@pytest.mark.parametrize(
    ("arrays", "setup"),
    [
        ([X, Y], "missing-dir"),
        ([X, Y], "existing-file"),
        ([X, LassiArray("x", "f32", (1,), struct.pack("<f", 0.0))], ""),
        ([X, LassiArray("y/z", "f32", (1,), struct.pack("<f", 0.0))], ""),
        ([X, LassiArray("y" * (MAX_INPUT_NAME_BYTES + 1), "f32", (1,), struct.pack("<f", 0.0))], ""),
        ([X, LassiArray("y", "f128", (1,), b"\x00" * 16)], ""),
        ([X, LassiArray("y", "f32", (4,), b"\x00" * 12)], ""),
    ],
    ids=["missing-dir", "existing-file", "repeated-name", "bad-name", "long-name", "unknown-dtype", "short-data"],
)
def test_a_refusal_writes_nothing(tmp_path: Path, arrays: list[LassiArray], setup: str) -> None:
    out = tmp_path / "out"
    if setup != "missing-dir":
        out.mkdir()
    if setup == "existing-file":
        (out / f"y{SUFFIX}").write_bytes(b"SYNTHETIC: an earlier file\n")
    before = listing(out) if out.is_dir() else None
    with pytest.raises(LassiIOError) as caught:
        write_inputs()(arrays, out)
    assert str(out) in str(caught.value), str(caught.value)
    assert (listing(out) if out.is_dir() else None) == before, "a refused set leaves the directory as it was"
