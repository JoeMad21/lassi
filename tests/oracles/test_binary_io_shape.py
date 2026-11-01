"""Tests for binary_io's shape rule, the optional recipe key `shape` (task P17.48).

Bible: Oracles (binary_io row), Harness Contract (lassi_io files in
row-major order), Agent Rule 3 (one generic key, no project code).
Plan: plans/spikes/ttmlir-runtime.md, the dated section on the TTIR arms on
ttsim.

The contract these tests fix:

- BinaryIOOracle takes `shape`, one of SHAPE_RULES == ("exact",
  "elements"), default "exact"; any other value raises ValueError naming
  oracle.shape. with_baseline and with_tolerance keep it.
- "exact" is the rule before the key: a shape mismatch fails the output
  with a note starting "shape mismatch:".
- "elements": arrays of one dtype whose shapes differ but hold the same
  number of elements are compared element by element in row-major order,
  as arrays of one shape are; a different element count still fails with a
  "shape mismatch:" note that gives both counts; a dtype mismatch still
  fails first.
- describe() says so under "elements" and is unchanged under "exact".

Every array here is SYNTHETIC, written for these tests; no value is a
measurement.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from pathlib import Path

import pytest

from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.record import OutputStats
from lassi.core.tolerance import Tolerance
from lassi.harness.lassi_io import LassiArray, write_array
from lassi.oracles import binary_io


def f32(*values: float, shape: Sequence[int], name: str = "c") -> LassiArray:
    """Return a SYNTHETIC f32 array of `values` with `shape`."""
    return LassiArray(name, "f32", tuple(shape), struct.pack(f"<{len(values)}f", *values))


def compared(reference: LassiArray, candidate: LassiArray, shape: str, threshold: float = 0.0) -> OutputStats:
    """Return compare_arrays under max_abs and `threshold` with the shape rule `shape`."""
    return binary_io.compare_arrays(reference, candidate, metric="max_abs", threshold=threshold, shape=shape)


def test_the_rules_and_the_default() -> None:
    assert binary_io.SHAPE_RULES == ("exact", "elements")
    built = binary_io.BinaryIOOracle(metric="max_abs", threshold=0.0)
    assert built.shape == "exact"


@pytest.mark.parametrize("value", ["flat", "", None, 1])
def test_another_rule_is_refused_naming_the_key(value: object) -> None:
    with pytest.raises(ValueError, match="oracle.shape"):
        binary_io.BinaryIOOracle(metric="max_abs", threshold=0.0, shape=value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape rule"):
        compared(f32(1.0, shape=(1,)), f32(1.0, shape=(1,)), value)  # type: ignore[arg-type]


def test_exact_fails_a_reshaped_output_as_before() -> None:
    stats = compared(f32(1.0, 2.0, 3.0, 4.0, shape=(4,)), f32(1.0, 2.0, 3.0, 4.0, shape=(2, 2)), "exact")
    assert stats.passed is False and stats.note is not None and stats.note.startswith("shape mismatch:")
    assert (stats.pcc, stats.max_abs, stats.max_ulp) == (None, None, None)


def test_elements_compares_the_same_elements_in_row_major_order() -> None:
    stats = compared(f32(1.0, 2.0, 3.0, 4.0, shape=(4,)), f32(1.0, 2.0, 3.0, 4.0, shape=(2, 2)), "elements")
    assert stats.passed is True and stats.note is None
    assert (stats.pcc, stats.max_abs, stats.max_ulp) == (1.0, 0.0, 0)


def test_elements_still_reads_the_values() -> None:
    # Row-major [[1, 2], [3, 4]] against [1, 2, 4, 3]: elements 2 and 3 differ by 1 each.
    stats = compared(f32(1.0, 2.0, 4.0, 3.0, shape=(4,)), f32(1.0, 2.0, 3.0, 4.0, shape=(2, 2)), "elements", 0.5)
    assert stats.passed is False and stats.max_abs == 1.0
    assert stats.note is not None and "threshold" in stats.note


def test_elements_fails_another_element_count_naming_both() -> None:
    stats = compared(f32(1.0, 2.0, 3.0, 4.0, shape=(4,)), f32(1.0, 2.0, 3.0, shape=(1, 3)), "elements")
    assert stats.passed is False and stats.note is not None
    assert stats.note.startswith("shape mismatch:") and "4 and 3 elements" in stats.note
    assert (stats.pcc, stats.max_abs, stats.max_ulp) == (None, None, None)


def test_elements_still_fails_a_dtype_mismatch() -> None:
    bf16 = LassiArray("c", "bf16", (2, 2), struct.pack("<4H", 0x3F80, 0x4000, 0x4040, 0x4080))
    stats = compared(f32(1.0, 2.0, 3.0, 4.0, shape=(4,)), bf16, "elements")
    assert stats.passed is False and stats.note is not None and stats.note.startswith("dtype mismatch:")


def test_the_rule_reaches_compare_and_survives_binding(tmp_path: Path) -> None:
    reference, candidate = tmp_path / "reference.lassiio", tmp_path / "candidate.lassiio"
    write_array(reference, "c", "f32", (4,), struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))
    write_array(candidate, "c", "f32", (2, 2), struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))
    files = ({"c.lassiio": reference}, {"c.lassiio": candidate})
    agreement = [OutputStats(name="c", pcc=1.0, max_abs=0.0, max_ulp=0, passed=True, note=None)]
    built = binary_io.BinaryIOOracle(metric="max_abs", threshold="from_baseline", shape="elements")
    bound = built.with_baseline(agreement)
    assert bound.shape == "elements" and bound.alignment(*files).mean == 1.0
    judged = built.with_tolerance(Tolerance(metric="max_abs", threshold=0.01))
    assert judged.shape == "elements" and judged.alignment(*files).mean == 1.0
    exact = binary_io.BinaryIOOracle(metric="max_abs", threshold=0.01)
    assert exact.alignment(*files).mean == 0.0, "the default rule is unchanged"


def test_describe_names_the_rule_only_when_it_is_elements() -> None:
    exact = binary_io.BinaryIOOracle(metric="pcc", threshold=0.97)
    elements = binary_io.BinaryIOOracle(metric="pcc", threshold=0.97, shape="elements")
    assert exact.describe() == "binary_io: pcc at least 0.97, per output"
    assert elements.describe().startswith(exact.describe()) and "element counts" in elements.describe()
