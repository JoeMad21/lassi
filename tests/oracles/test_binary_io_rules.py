"""Tests for the binary_io rules the P4.4 tests leave open: other float dtypes, edge arrays, thresholds, files.

Bible: Oracles (binary_io rules), Benchmark Suites (item tolerance). Plan:
plans/p4-ttsim.md, P4.4. The decisions these tests fix (task P4.4):

- f16 and f64 follow the f32 rules (NaN, infinity, and -0), with the ULP
  distance counted in their own format; integer dtypes match exactly.
- Equal empty arrays, and arrays with no finite pair whose NaNs and
  infinities all match, pass with pcc 1.0, max_abs 0.0, and max_ulp 0.
- A difference that overflows double (f64 only) leaves max_abs None and
  fails the output with a note.
- Exact match is written `{metric: max_abs, threshold: 0}`; there is no
  other form.
- Thresholds: pcc in [-1, 1]; max_abs and ulp finite and >= 0; anything
  else is refused, in a recipe section and in a suite manifest item alike.
- Output files: two files of one run that hold one array name, and a
  reference run with no output file, cannot serve as a reference
  (reference_problem; compare raises ValueError). On the candidate side a
  repeated name fails that output with a note naming both files.
- with_tolerance judges against the tolerance's own metric; with_baseline
  leaves a numeric threshold as it is; from_baseline fails a float output
  the agreement gives no value for, while an integer output passes only on
  exact equality, whatever the agreement holds.
- Two sides equal element for element give pcc exactly 1.0, so identical
  outputs pass a pcc threshold of 1.
- An int threshold a double cannot hold is refused as not finite, with the
  normal refusal (a ValueError), never an OverflowError.

Every array and manifest here is SYNTHETIC; each expected statistic is
computed by hand in the comment beside it. No value is a measurement.
"""

from __future__ import annotations

import math
import random
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.bench import load_suite
from lassi.core.record import OutputStats
from lassi.core.tolerance import Tolerance, threshold_problem
from lassi.harness.lassi_io import LassiArray, write_array
from lassi.oracles.binary_io import BinaryIOOracle, compare_arrays

HUGE = 1.0e30  # a max_abs threshold no finite difference here reaches, so only a rule can fail the output

# f16 bit patterns.
F16_ONE = 0x3C00  # 1.0
F16_ONE_NEXT = 0x3C01  # 1.0 + 2**-10
F16_MINUS_ZERO = 0x8000
F16_INF = 0x7C00
F16_NEG_INF = 0xFC00
F16_QNAN = 0x7E00
F16_NAN_NEG_PAYLOAD = 0xFE01

# f64 bit patterns.
F64_ONE = 0x3FF0000000000000
F64_ONE_NEXT = 0x3FF0000000000001  # 1.0 + 2**-52
F64_MINUS_ZERO = 0x8000000000000000
F64_INF = 0x7FF0000000000000
F64_QNAN = 0x7FF8000000000000
F64_NAN_PAYLOAD = 0xFFF8000000000001


def bits(dtype: str, *patterns: int, name: str = "c") -> LassiArray:
    """Return an array of `dtype` (f16 or f64) whose elements are the given bit patterns."""
    code = {"f16": "H", "f64": "Q"}[dtype]
    return LassiArray(name, dtype, (len(patterns),), struct.pack(f"<{len(patterns)}{code}", *patterns))


def values(dtype: str, *items: Any, name: str = "c", shape: tuple[int, ...] | None = None) -> LassiArray:
    """Return an array of `dtype` holding `items` (floats or ints, each exact in the dtype)."""
    code = {"f32": "f", "f64": "d", "u8": "B", "i64": "q"}[dtype]
    data = struct.pack(f"<{len(items)}{code}", *items)
    return LassiArray(name, dtype, shape if shape is not None else (len(items),), data)


def stats(reference: LassiArray, candidate: LassiArray, metric: str = "max_abs", threshold: float = HUGE) -> Any:
    """Return compare_arrays(reference, candidate) under `metric` and `threshold`."""
    return compare_arrays(reference, candidate, metric=metric, threshold=threshold)


def write_files(directory: Path, arrays: Mapping[str, LassiArray]) -> dict[str, Path]:
    """Write each array to `<directory>/<file>` and return file -> path."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for file, array in arrays.items():
        write_array(directory / file, array.name, array.dtype, array.shape, array.data)
        files[file] = directory / file
    return files


# ---------------------------------------------------------------------------
# f16 and f64


def test_f16_counts_ulps_in_f16_units() -> None:
    result = stats(bits("f16", F16_ONE), bits("f16", F16_ONE_NEXT))
    assert (result.max_ulp, result.max_abs) == (1, 2.0**-10)


def test_f16_follows_the_nan_infinity_and_minus_zero_rules() -> None:
    reference = bits("f16", F16_QNAN, F16_INF, F16_MINUS_ZERO, F16_ONE)
    candidate = bits("f16", F16_NAN_NEG_PAYLOAD, F16_INF, 0x0000, F16_ONE)
    result = stats(reference, candidate, "ulp", 0)
    assert result.passed is True, result
    assert (result.max_abs, result.max_ulp) == (0.0, 0)
    opposite = stats(bits("f16", F16_INF), bits("f16", F16_NEG_INF))
    assert opposite.passed is False and "inf" in (opposite.note or "")


def test_f64_counts_ulps_in_f64_units_and_matches_nans_and_signed_zeros() -> None:
    assert stats(bits("f64", F64_ONE), bits("f64", F64_ONE_NEXT)).max_ulp == 1
    reference = bits("f64", F64_QNAN, F64_MINUS_ZERO, F64_INF)
    candidate = bits("f64", F64_NAN_PAYLOAD, 0, F64_INF)
    assert stats(reference, candidate, "ulp", 0).passed is True
    mismatch = stats(bits("f64", F64_QNAN), bits("f64", F64_ONE))
    assert mismatch.passed is False and "NaN" in (mismatch.note or "")


def test_an_f64_difference_that_overflows_double_leaves_max_abs_unset_and_fails() -> None:
    result = stats(values("f64", 1.5e308), values("f64", -1.5e308), "pcc", -1)
    assert result.max_abs is None and result.passed is False
    assert "overflows" in (result.note or "")


@pytest.mark.parametrize("dtype", ["u8", "i64"])
def test_every_integer_dtype_matches_exactly(dtype: str) -> None:
    assert stats(values(dtype, 1, 2), values(dtype, 1, 2), "ulp", 0).passed is True
    differ = stats(values(dtype, 1, 2), values(dtype, 1, 3), "max_abs", HUGE)
    assert differ.passed is False and differ.max_abs == 1.0 and differ.max_ulp is None


# ---------------------------------------------------------------------------
# Edge arrays


@pytest.mark.parametrize("dtype", ["f32", "f64"])
def test_equal_empty_float_arrays_pass_with_pcc_one(dtype: str) -> None:
    empty = LassiArray("c", dtype, (0, 3), b"")
    result = stats(empty, empty, "pcc", 1)
    assert (result.pcc, result.max_abs, result.max_ulp, result.passed) == (1.0, 0.0, 0, True)


def test_equal_empty_integer_arrays_pass() -> None:
    empty = LassiArray("c", "i64", (0,), b"")
    result = stats(empty, empty, "max_abs", 0)
    assert (result.pcc, result.max_abs, result.max_ulp, result.passed) == (1.0, 0.0, None, True)


def test_arrays_of_matched_nans_and_infinities_only_pass_with_pcc_one() -> None:
    reference = bits("f64", F64_QNAN, F64_INF)
    result = stats(reference, bits("f64", F64_NAN_PAYLOAD, F64_INF), "pcc", 1)
    assert (result.pcc, result.max_abs, result.max_ulp, result.passed) == (1.0, 0.0, 0, True)


def test_the_first_unmatched_element_and_the_count_are_named() -> None:
    # Elements 1 and 2 hold an unmatched NaN and an unmatched infinity; the note names element 1 and the count 2.
    reference = values("f32", 1.0, math.nan, math.inf)
    candidate = values("f32", 1.0, 2.0, 3.0)
    note = stats(reference, candidate).note or ""
    assert "element 1" in note and "2 elements" in note


# ---------------------------------------------------------------------------
# Exact match and thresholds


def test_exact_match_is_max_abs_zero() -> None:
    exact = BinaryIOOracle(metric="max_abs", threshold=0)
    assert exact.threshold == 0
    assert stats(values("f32", 1.0, -0.0), values("f32", 1.0, 0.0), "max_abs", 0).passed is True
    assert stats(values("f32", 1.0), values("f32", 1.0 + 2.0**-23), "max_abs", 0).passed is False


@pytest.mark.parametrize(
    ("metric", "threshold"),
    [("pcc", 1.5), ("pcc", -1.01), ("max_abs", -0.1), ("ulp", -1), ("max_abs", math.inf), ("ulp", math.nan)],
    ids=["pcc-above-one", "pcc-below-minus-one", "negative-max-abs", "negative-ulp", "infinite", "nan"],
)
def test_a_threshold_outside_its_metrics_range_is_refused(metric: str, threshold: float) -> None:
    with pytest.raises(ValueError, match="oracle.threshold"):
        BinaryIOOracle(metric=metric, threshold=threshold)
    with pytest.raises(ValueError):
        compare_arrays(values("f32", 1.0), values("f32", 1.0), metric=metric, threshold=threshold)
    with pytest.raises(ValueError):
        Tolerance(metric=metric, threshold=threshold)


@pytest.mark.parametrize(
    ("metric", "threshold"), [("pcc", -1), ("pcc", 1), ("max_abs", 0), ("ulp", 0), ("ulp", 2.5)]
)
def test_thresholds_at_the_edges_of_their_ranges_are_accepted(metric: str, threshold: float) -> None:
    assert BinaryIOOracle(metric=metric, threshold=threshold).threshold == threshold


def manifest(directory: Path, tolerance: Any) -> Path:
    """Write a SYNTHETIC one-item manifest whose item declares `tolerance`, and return its path."""
    languages = {"cpu": {"dir": "src/a", "files": ["a.cpp"]}, "tt": {"dir": "src/b", "files": ["b.cpp"]}}
    data = {
        "suite": "binio-rules",
        "repo": "https://example.invalid/binio-rules",
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "items": {"vadd": {"split": "eval", "languages": languages, "tolerance": tolerance}},
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "binio-rules.yaml"
    path.write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    return path


def test_an_item_may_declare_exact_match(tmp_path: Path) -> None:
    suite = load_suite(manifest(tmp_path, {"metric": "max_abs", "threshold": 0}))
    assert suite.items["vadd"].tolerance == Tolerance(metric="max_abs", threshold=0)


@pytest.mark.parametrize(
    ("metric", "threshold"),
    [("pcc", 2), ("ulp", -0.5), ("pcc", True), ("max_abs", "from_baseline")],
    ids=["pcc-above-one", "negative-ulp", "bool", "from-baseline"],
)
def test_an_item_tolerance_outside_its_range_is_refused(tmp_path: Path, metric: str, threshold: Any) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        load_suite(manifest(tmp_path, {"metric": metric, "threshold": threshold}))


# ---------------------------------------------------------------------------
# Output files


def test_a_repeated_candidate_name_fails_that_output_naming_both_files(tmp_path: Path) -> None:
    reference = write_files(tmp_path / "ref", {"c.lassiio": values("f32", 1.0)})
    candidate = write_files(tmp_path / "cand", {"a.lassiio": values("f32", 1.0), "b.lassiio": values("f32", 1.0)})
    (result,) = BinaryIOOracle(metric="max_abs", threshold=0).compare(reference, candidate)
    assert result.passed is False
    assert "a.lassiio" in (result.note or "") and "b.lassiio" in (result.note or "")


def test_reference_files_that_repeat_a_name_or_hold_nothing_cannot_serve(tmp_path: Path) -> None:
    oracle = BinaryIOOracle(metric="max_abs", threshold=0)
    repeated = write_files(tmp_path / "ref", {"a.lassiio": values("f32", 1.0), "b.lassiio": values("f32", 2.0)})
    problem = oracle.reference_problem(repeated) or ""
    assert "a.lassiio" in problem and "b.lassiio" in problem
    assert "no output file" in (oracle.reference_problem({}) or "")
    with pytest.raises(ValueError, match="no output file"):
        oracle.compare({}, repeated)
    good = write_files(tmp_path / "good", {"c.lassiio": values("f32", 1.0)})
    assert oracle.reference_problem(good) is None


def test_with_tolerance_judges_under_the_tolerances_own_metric(tmp_path: Path) -> None:
    # 1, 2, 3 against 1, 2, 3.5: pcc is high (above 0.99) while max_abs is 0.5.
    reference = write_files(tmp_path / "ref", {"c.lassiio": values("f32", 1.0, 2.0, 3.0)})
    other = write_files(tmp_path / "other", {"c.lassiio": values("f32", 1.0, 2.0, 3.5)})
    recipe = BinaryIOOracle(metric="max_abs", threshold=0)
    assert recipe.alignment(reference, other).mean == 0.0
    assert recipe.with_tolerance(Tolerance(metric="pcc", threshold=0.99)).alignment(reference, other).mean == 1.0


def test_with_baseline_keeps_a_numeric_threshold(tmp_path: Path) -> None:
    reference = write_files(tmp_path / "ref", {"c.lassiio": values("f32", 1.0)})
    near = write_files(tmp_path / "near", {"c.lassiio": values("f32", 1.25)})
    agreement = [OutputStats(name="c", pcc=1.0, max_abs=0.0, max_ulp=0, passed=True, note=None)]
    bound = BinaryIOOracle(metric="max_abs", threshold=0.5).with_baseline(agreement)
    assert bound.threshold == 0.5 and bound.alignment(reference, near).mean == 1.0


def test_from_baseline_fails_an_output_the_agreement_gives_no_value_for(tmp_path: Path) -> None:
    arrays = {"c.lassiio": values("f32", 1.0), "d.lassiio": values("f32", 2.0, name="d")}
    reference = write_files(tmp_path / "ref", arrays)
    agreement = [OutputStats(name="c", pcc=1.0, max_abs=0.0, max_ulp=0, passed=True, note=None)]
    bound = BinaryIOOracle(metric="max_abs", threshold="from_baseline").with_baseline(agreement)
    result = {entry.name: entry for entry in bound.compare(reference, reference)}
    assert result["c"].passed is True
    assert result["d"].passed is False and "no max_abs threshold" in (result["d"].note or "")


# ---------------------------------------------------------------------------
# Audit fixes: identical sides, huge thresholds, integer outputs under from_baseline


def test_identical_non_constant_sides_give_pcc_exactly_one() -> None:
    two = values("f32", 1.0, 2.0)
    result = compare_arrays(two, two, metric="pcc", threshold=1)
    assert result.pcc == 1.0 and result.passed is True
    rng = random.Random(20260925)  # a fixed seed: the same SYNTHETIC arrays on every run
    for _ in range(200):
        items = [rng.uniform(-100.0, 100.0) for _ in range(rng.randint(2, 40))]
        array = values("f64", *items)
        assert compare_arrays(array, array, metric="pcc", threshold=1).pcc == 1.0, items


@pytest.mark.parametrize("threshold", [10**400, -(10**400), 10**5000], ids=["huge", "huge-negative", "past-digits"])
def test_an_int_threshold_a_double_cannot_hold_is_refused_as_not_finite(threshold: int) -> None:
    assert "finite" in threshold_problem("max_abs", threshold)
    with pytest.raises(ValueError, match="oracle.threshold"):
        BinaryIOOracle(metric="max_abs", threshold=threshold)
    with pytest.raises(ValueError, match="finite"):
        Tolerance(metric="ulp", threshold=threshold)


def test_an_integer_output_under_from_baseline_matches_exactly_whatever_the_agreement_holds(tmp_path: Path) -> None:
    # The agreement names only c; the i64 output n has no value there, yet it passes on exact equality.
    arrays = {"c.lassiio": values("f32", 1.0), "n.lassiio": values("i64", 3, 4, name="n")}
    reference = write_files(tmp_path / "ref", arrays)
    differ = write_files(tmp_path / "differ", {**arrays, "n.lassiio": values("i64", 3, 5, name="n")})
    agreement = [OutputStats(name="c", pcc=1.0, max_abs=0.0, max_ulp=0, passed=True, note=None)]
    bound = BinaryIOOracle(metric="ulp", threshold="from_baseline").with_baseline(agreement)
    same = {entry.name: entry for entry in bound.compare(reference, reference)}
    assert same["n"].passed is True and same["n"].note is None
    other = {entry.name: entry for entry in bound.compare(reference, differ)}
    assert other["n"].passed is False and "exactly" in (other["n"].note or "")
