"""Tests for the binary_io oracle: per-output statistics, pass rules, and its recipe keys (task P4.4).

Bible: Oracles (binary_io row), Harness Contract (lassi_io files, one named
array per file), Component Interfaces (Oracle; the capabilities rule),
Project Recipes (the lassi-df oracle line). Plan: plans/p4-ttsim.md, P4.4 and
the planning decision "binary_io (P4.4)".

The contract these tests fix:

- lassi.core.capabilities.ALIGNS_OUTPUT_FILES == "aligns_output_files": the
  capability of an Oracle that compares a run's output files, which the
  oracle stage reads to align output files instead of stdout.
- lassi.oracles.binary_io registers the Oracle `binary_io` (importing
  lassi.core.runner registers it too). Its class BinaryIOOracle declares
  exactly the capability ALIGNS_OUTPUT_FILES and the config keys `metric`
  and `threshold`, and is built as BinaryIOOracle(metric=..., threshold=...),
  which keeps them as its `metric` and `threshold` attributes (its `name` is
  "binary_io"). Both keys are required choices with no default: a missing one raises
  ValueError whose message names `oracle.metric` or `oracle.threshold`.
  METRICS is ("pcc", "max_abs", "ulp") and an unknown metric raises
  ValueError naming each of them. `threshold` is a number (an int or a
  float, never a bool) or FROM_BASELINE == "from_baseline"; anything else
  raises ValueError.
- compare_arrays(reference, candidate, *, metric, threshold) takes two
  lassi.harness.lassi_io.LassiArray values and returns a
  lassi.core.record.OutputStats (fields name, pcc, max_abs, max_ulp,
  passed, note; any other field has a default) for the reference's name.
- Rules for f32 and bf16 elements:
  - A NaN equals a NaN in the same position, whatever its bits; a NaN
    facing anything else fails the output, with a note naming NaN.
  - An infinity equals an infinity of the same sign in the same position;
    any other pairing with an infinity fails the output, with a note that
    names the infinity.
  - Matched NaN and infinity pairs are left out of the statistics.
  - -0 and +0 are equal: max-abs difference 0, ULP distance 0. The ULP
    distance counts representable values of the dtype between the two
    (bf16 in bf16 units), with -0 and +0 one point, so the smallest
    positive and negative subnormals are 2 apart.
  - max_abs is the largest absolute difference, computed in double; max_ulp
    the largest ULP distance, an int.
  - pcc is the Pearson correlation computed in double. When either array is
    constant (zero variance over the compared elements), pcc is 1.0 when the
    two arrays are equal element for element and 0.0 otherwise.
- Other dtypes (tested with i32): exact match counts. The output passes only
  when every element is equal, whatever the metric and threshold; max_abs
  is the exact largest difference as a float, pcc is computed in double as
  for floats, and max_ulp is None (it does not apply).
- A dtype or a shape mismatch fails the output with a note naming it, and
  its pcc, max_abs, and max_ulp are None.
- Pass rules: pcc passes when pcc >= threshold, max_abs when max_abs <=
  threshold, ulp when max_ulp <= threshold.
- BinaryIOOracle.compare(reference_files, candidate_files) takes two
  mappings of output file (relative path -> path of a lassi_io file), as
  RunResult.output_files holds them, reads every file, and returns one
  OutputStats per array name found on either side; arrays are matched by the
  name in their file header, not by file name. A name the candidate lacks is
  a missing output and one the reference lacks an extra output; each fails
  with a note saying so. A candidate file that is not a lassi_io file fails
  with a note naming the file, and nothing raises.
- BinaryIOOracle.align(reference, candidate) reads the two RunResults'
  output_files and returns the per-input value: 1.0 when every output
  passes, else 0.0.
- With threshold FROM_BASELINE, with_baseline(agreement) returns a copy
  whose threshold for each output is the agreement's value of the metric for
  that output (pcc, max_abs, or max_ulp of the OutputStats the baseline
  recorded); compare or align on an unbound copy raises ValueError naming
  from_baseline.

Every array here is SYNTHETIC and written for these tests; each expected
statistic is computed by hand from the bit patterns and values in the test
(the comment beside it shows the arithmetic). No value is a measurement.
"""

from __future__ import annotations

import importlib
import math
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lassi.core import capabilities
from lassi.core import record as record_module
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.interfaces import RunResult
from lassi.core.registry import DEFAULT_REGISTRY, RegistryError
from lassi.harness.lassi_io import LassiArray, write_array

MODULE = "lassi.oracles.binary_io"
NAME = "binary_io"
CAPABILITY = "aligns_output_files"
HUGE = 1.0e30  # a max_abs threshold no finite difference here reaches, so only a rule can fail the output

# f32 bit patterns.
F32_ONE = 0x3F800000  # 1.0
F32_ONE_NEXT = 0x3F800001  # 1.0 + 2**-23, one unit in the last place above 1.0
F32_TWO = 0x40000000  # 2.0
F32_MINUS_ZERO = 0x80000000
F32_PLUS_ZERO = 0x00000000
F32_MIN_SUB = 0x00000001  # the smallest positive subnormal, 2**-149
F32_MIN_SUB_NEG = 0x80000001  # its negative
F32_INF = 0x7F800000
F32_NEG_INF = 0xFF800000
F32_MAX = 0x7F7FFFFF  # the largest finite f32
F32_QNAN = 0x7FC00000
F32_QNAN_NEG_PAYLOAD = 0xFFC00001
F32_SNAN = 0x7F800001

# bf16 bit patterns: the upper 16 bits of the f32 of the same value.
BF16_ONE = 0x3F80  # 1.0
BF16_ONE_NEXT = 0x3F81  # 1.0078125 = 1 + 2**-7, one bf16 unit above 1.0
BF16_MINUS_ONE = 0xBF80
BF16_INF = 0x7F80
BF16_NEG_INF = 0xFF80
BF16_QNAN = 0x7FC0
BF16_NAN_NEG_PAYLOAD = 0xFFC1


# ---------------------------------------------------------------------------
# Names this task adds, looked up so a missing one fails its test with a clear message


def binary_io() -> ModuleType:
    """Return lassi.oracles.binary_io; fail the test clearly while it does not exist."""
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as error:
        if error.name != MODULE:
            raise
        pytest.fail(f"{MODULE} does not exist; task P4.4 adds the binary_io oracle")


def oracle_class() -> type:
    """Return the Oracle class registered as binary_io; fail the test clearly while it is not registered."""
    binary_io()
    try:
        return DEFAULT_REGISTRY.get("Oracle", NAME).factory
    except RegistryError as error:
        pytest.fail(f"no Oracle is registered as {NAME!r} ({error}); task P4.4 registers it")


def oracle(metric: str = "max_abs", threshold: Any = 0.0) -> Any:
    """Return a binary_io oracle built from a recipe section with `metric` and `threshold`."""
    return oracle_class()(metric=metric, threshold=threshold)


def output_stats(**fields: Any) -> Any:
    """Return lassi.core.record.OutputStats(**fields); fail the test clearly while the class is missing."""
    cls = getattr(record_module, "OutputStats", None)
    if cls is None:
        pytest.fail("lassi.core.record has no OutputStats; task P4.4 adds the per-output statistics record")
    return cls(**fields)


def compared(reference: LassiArray, candidate: LassiArray, metric: str = "max_abs", threshold: float = HUGE) -> Any:
    """Return compare_arrays(reference, candidate) under `metric` and `threshold`."""
    return binary_io().compare_arrays(reference, candidate, metric=metric, threshold=threshold)


# ---------------------------------------------------------------------------
# SYNTHETIC arrays


def f32(*values: float, name: str = "c", shape: Sequence[int] | None = None) -> LassiArray:
    """Return an f32 array of `values` (each exactly representable in f32)."""
    data = struct.pack(f"<{len(values)}f", *values)
    return LassiArray(name, "f32", tuple(shape) if shape is not None else (len(values),), data)


def f32_bits(*bits: int, name: str = "c") -> LassiArray:
    """Return an f32 array whose elements are the given 32-bit patterns."""
    return LassiArray(name, "f32", (len(bits),), struct.pack(f"<{len(bits)}I", *bits))


def bf16_bits(*bits: int, name: str = "c") -> LassiArray:
    """Return a bf16 array whose elements are the given 16-bit patterns."""
    return LassiArray(name, "bf16", (len(bits),), struct.pack(f"<{len(bits)}H", *bits))


def i32(*values: int, name: str = "c") -> LassiArray:
    """Return an i32 array of `values`."""
    return LassiArray(name, "i32", (len(values),), struct.pack(f"<{len(values)}i", *values))


def write_files(directory: Path, arrays: Mapping[str, LassiArray]) -> dict[str, Path]:
    """Write each array to `<directory>/<file>` and return file -> path, as RunResult.output_files holds them."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for file, array in arrays.items():
        path = directory / file
        write_array(path, array.name, array.dtype, array.shape, array.data)
        files[file] = path
    return files


def by_name(stats: Sequence[Any]) -> dict[str, Any]:
    """Return a list of OutputStats keyed by output name; a repeated name fails the test."""
    keyed = {entry.name: entry for entry in stats}
    assert len(keyed) == len(stats), f"an output name repeats in {stats!r}"
    return keyed


def run_with(files: Mapping[str, Path]) -> RunResult:
    """Return a SYNTHETIC clean RunResult holding `files` as its output files."""
    return RunResult(exit_code=0, hang=False, stdout="", stderr="", output_files=dict(files))


# ---------------------------------------------------------------------------
# Registration, capability, and recipe keys


def test_the_capability_constant_names_output_file_alignment() -> None:
    value = getattr(capabilities, "ALIGNS_OUTPUT_FILES", None)
    assert value == CAPABILITY, "lassi.core.capabilities.ALIGNS_OUTPUT_FILES must be 'aligns_output_files'"


def test_importing_the_runner_registers_binary_io_with_its_capability_and_keys() -> None:
    oracle_class()
    entry = DEFAULT_REGISTRY.get("Oracle", NAME)
    assert entry.capabilities == frozenset({CAPABILITY})
    assert entry.config_keys == frozenset({"metric", "threshold"})
    stdout_mask = DEFAULT_REGISTRY.get("Oracle", "stdout_mask")
    assert CAPABILITY not in stdout_mask.capabilities, "stdout_mask aligns stdout, not output files"


def test_the_metrics_are_pcc_max_abs_and_ulp_and_from_baseline_is_named() -> None:
    module = binary_io()
    assert tuple(module.METRICS) == ("pcc", "max_abs", "ulp")
    assert module.FROM_BASELINE == "from_baseline"


@pytest.mark.parametrize(
    ("config", "key"),
    [({"threshold": 0.99}, "oracle.metric"), ({"metric": "pcc"}, "oracle.threshold"), ({}, "oracle.metric")],
    ids=["no-metric", "no-threshold", "neither"],
)
def test_metric_and_threshold_are_required_choices_with_no_default(config: dict[str, Any], key: str) -> None:
    with pytest.raises(ValueError, match=key.replace(".", r"\.")):
        oracle_class()(**config)


def test_an_unknown_metric_is_refused_naming_the_metrics() -> None:
    with pytest.raises(ValueError) as caught:
        oracle("rmse", 0.1)
    for metric in ("pcc", "max_abs", "ulp"):
        assert metric in str(caught.value)


@pytest.mark.parametrize("threshold", ["loose", True, [0.1], {"value": 0.1}], ids=["word", "bool", "list", "mapping"])
def test_a_threshold_that_is_neither_a_number_nor_from_baseline_is_refused(threshold: Any) -> None:
    with pytest.raises(ValueError, match="threshold"):
        oracle("pcc", threshold)


@pytest.mark.parametrize(
    ("metric", "threshold"),
    [("pcc", 0.999), ("max_abs", 0.001), ("ulp", 4), ("pcc", "from_baseline")],
    ids=["pcc-float", "max-abs-float", "ulp-int", "from-baseline"],
)
def test_a_number_or_from_baseline_is_a_valid_threshold(metric: str, threshold: Any) -> None:
    built = oracle(metric, threshold)
    assert built.name == NAME
    assert (built.metric, built.threshold) == (metric, threshold)


# ---------------------------------------------------------------------------
# f32 and bf16 statistics


def test_f32_statistics_match_hand_computed_values() -> None:
    # 3.0 + 2**-22 is one f32 unit above 3.0 (units are 2**-22 in [2, 4)); 4.0 + 2**-20 is two units above 4.0
    # (units are 2**-21 in [4, 8)). The largest difference is 2**-20 = 9.5367431640625e-07.
    stats = compared(f32(1.0, 2.0, 3.0, 4.0), f32(1.0, 2.0, 3.0 + 2.0**-22, 4.0 + 2.0**-20))
    assert stats.name == "c"
    assert stats.max_abs == 2.0**-20
    assert stats.max_ulp == 2
    assert stats.passed is True and stats.note is None


def test_one_f32_unit_above_one() -> None:
    stats = compared(f32_bits(F32_ONE), f32_bits(F32_ONE_NEXT))
    assert stats.max_ulp == 1
    assert stats.max_abs == 2.0**-23


def test_f32_units_between_one_and_two() -> None:
    # 0x40000000 - 0x3F800000 = 0x00800000 = 8388608 units; the difference is 1.0.
    stats = compared(f32(1.0), f32(2.0))
    assert stats.max_ulp == 8388608
    assert stats.max_abs == 1.0


def test_bf16_statistics_count_bf16_units() -> None:
    # 0x3F81 is one bf16 unit above 1.0: 1 + 2**-7 = 1.0078125.
    stats = compared(bf16_bits(BF16_ONE), bf16_bits(BF16_ONE_NEXT))
    assert stats.max_ulp == 1
    assert stats.max_abs == 0.0078125


def test_bf16_units_across_zero() -> None:
    # -1.0 (0xBF80) lies 0x3F80 units below zero and 1.0 (0x3F80) as far above: 2 * 0x3F80 = 32512 units.
    stats = compared(bf16_bits(BF16_MINUS_ONE), bf16_bits(BF16_ONE))
    assert stats.max_ulp == 32512
    assert stats.max_abs == 2.0


@pytest.mark.parametrize(
    ("reference", "candidate", "units", "difference"),
    [
        (F32_MIN_SUB, F32_MIN_SUB_NEG, 2, 2.0**-148),
        (F32_MINUS_ZERO, F32_MIN_SUB, 1, 2.0**-149),
        (F32_PLUS_ZERO, F32_MIN_SUB_NEG, 1, 2.0**-149),
    ],
    ids=["min-subnormals", "minus-zero-to-min", "plus-zero-to-minus-min"],
)
def test_the_ulp_distance_counts_minus_and_plus_zero_as_one_point(
    reference: int, candidate: int, units: int, difference: float
) -> None:
    stats = compared(f32_bits(reference), f32_bits(candidate))
    assert stats.max_ulp == units
    assert stats.max_abs == difference


def test_minus_zero_equals_plus_zero() -> None:
    stats = compared(f32_bits(F32_MINUS_ZERO, F32_ONE), f32_bits(F32_PLUS_ZERO, F32_ONE), "ulp", 0)
    assert stats.max_abs == 0.0
    assert stats.max_ulp == 0
    assert stats.passed is True
    assert stats.pcc == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# NaN and infinity


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        (f32_bits(F32_ONE, F32_QNAN, F32_TWO), f32_bits(F32_ONE, F32_QNAN_NEG_PAYLOAD, F32_TWO)),
        (f32_bits(F32_ONE, F32_SNAN, F32_TWO), f32_bits(F32_ONE, F32_QNAN, F32_TWO)),
        (bf16_bits(BF16_ONE, BF16_QNAN, BF16_ONE_NEXT), bf16_bits(BF16_ONE, BF16_NAN_NEG_PAYLOAD, BF16_ONE_NEXT)),
    ],
    ids=["f32-quiet-nans", "f32-signaling-and-quiet", "bf16-nans"],
)
def test_a_nan_equals_a_nan_in_the_same_position_whatever_its_bits(
    reference: LassiArray, candidate: LassiArray
) -> None:
    stats = compared(reference, candidate, "ulp", 0)
    assert stats.passed is True, stats
    assert stats.max_abs == 0.0
    assert stats.max_ulp == 0


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        (f32_bits(F32_ONE, F32_QNAN), f32(1.0, 2.0)),
        (f32(1.0, 2.0), f32_bits(F32_ONE, F32_QNAN)),
        (f32_bits(F32_QNAN, F32_ONE), f32_bits(F32_ONE, F32_QNAN)),
        (bf16_bits(BF16_ONE, BF16_QNAN), bf16_bits(BF16_ONE, BF16_ONE)),
    ],
    ids=["nan-in-reference", "nan-in-candidate", "nans-in-other-positions", "bf16-nan-in-reference"],
)
def test_a_nan_that_faces_anything_but_a_nan_fails_with_a_note(reference: LassiArray, candidate: LassiArray) -> None:
    stats = compared(reference, candidate, "max_abs", HUGE)
    assert stats.passed is False
    assert stats.note and "nan" in stats.note.lower()


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        (f32_bits(F32_INF, F32_ONE), f32_bits(F32_INF, F32_ONE)),
        (f32_bits(F32_NEG_INF, F32_TWO), f32_bits(F32_NEG_INF, F32_TWO)),
        (bf16_bits(BF16_INF, BF16_ONE), bf16_bits(BF16_INF, BF16_ONE)),
    ],
    ids=["f32-plus", "f32-minus", "bf16-plus"],
)
def test_an_infinity_equals_an_infinity_of_the_same_sign(reference: LassiArray, candidate: LassiArray) -> None:
    stats = compared(reference, candidate, "ulp", 0)
    assert stats.passed is True, stats
    assert stats.max_abs == 0.0
    assert stats.max_ulp == 0


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        (f32_bits(F32_INF, F32_ONE), f32_bits(F32_NEG_INF, F32_ONE)),
        (f32_bits(F32_INF, F32_ONE), f32_bits(F32_MAX, F32_ONE)),
        (f32_bits(F32_ONE, F32_ONE), f32_bits(F32_INF, F32_ONE)),
        (bf16_bits(BF16_NEG_INF, BF16_ONE), bf16_bits(BF16_INF, BF16_ONE)),
    ],
    ids=["opposite-sign", "largest-finite", "infinity-in-candidate", "bf16-opposite-sign"],
)
def test_any_other_pairing_with_an_infinity_fails_with_a_note(reference: LassiArray, candidate: LassiArray) -> None:
    stats = compared(reference, candidate, "max_abs", HUGE)
    assert stats.passed is False
    assert stats.note and "inf" in stats.note.lower()


def test_matched_nan_and_infinity_pairs_are_left_out_of_the_statistics() -> None:
    # Left: 1, 2, 3 against 1, 3, 2. pcc = 1 / sqrt(2 * 2) = 0.5 (deviations -1, 0, 1 and -1, 1, 0); the largest
    # difference is 1.0; 2.0 (0x40000000) and 3.0 (0x40400000) are 0x400000 = 4194304 units apart.
    reference = f32_bits(F32_ONE, F32_QNAN, F32_TWO, 0x40400000, F32_INF)
    candidate = f32_bits(F32_ONE, F32_QNAN, 0x40400000, F32_TWO, F32_INF)
    stats = compared(reference, candidate, "max_abs", 1.0)
    assert stats.passed is True
    assert stats.pcc == pytest.approx(0.5, rel=1e-12)
    assert stats.max_abs == 1.0
    assert stats.max_ulp == 4194304


# ---------------------------------------------------------------------------
# PCC


@pytest.mark.parametrize(
    ("reference", "candidate", "expected"),
    [
        ((1.0, 2.0, 3.0), (1.0, 3.0, 2.0), 0.5),  # deviations -1, 0, 1 and -1, 1, 0: 1 / sqrt(2 * 2)
        ((1.0, 2.0, 3.0, 4.0), (2.0, 4.0, 6.0, 8.0), 1.0),  # a positive multiple
        ((1.0, 2.0, 3.0, 4.0), (4.0, 3.0, 2.0, 1.0), -1.0),  # reversed
    ],
    ids=["half", "one", "minus-one"],
)
def test_pcc_matches_hand_computed_values(
    reference: tuple[float, ...], candidate: tuple[float, ...], expected: float
) -> None:
    assert compared(f32(*reference), f32(*candidate)).pcc == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_pcc_is_computed_in_double() -> None:
    # Even integers above 2**24 are exact in f32 but not in f32 sums. In double every step is exact: means
    # 16777218, deviations -2, 0, 2 and -2, 2, 0, so pcc = 4 / sqrt(8 * 8) = 0.5.
    stats = compared(f32(16777216.0, 16777218.0, 16777220.0), f32(16777216.0, 16777220.0, 16777218.0))
    assert stats.pcc == pytest.approx(0.5, rel=1e-12)


@pytest.mark.parametrize(
    ("reference", "candidate", "expected"),
    [
        (f32(2.0, 2.0, 2.0), f32(2.0, 2.0, 2.0), 1.0),
        (f32(2.0, 2.0, 2.0), f32(3.0, 3.0, 3.0), 0.0),
        (f32(2.0, 2.0, 2.0), f32(1.0, 2.0, 3.0), 0.0),
        (f32(1.0, 2.0, 3.0), f32(2.0, 2.0, 2.0), 0.0),
        (f32_bits(F32_MINUS_ZERO, F32_MINUS_ZERO), f32_bits(F32_PLUS_ZERO, F32_PLUS_ZERO), 1.0),
        (f32(5.0), f32(5.0), 1.0),
    ],
    ids=["equal-constants", "other-constants", "constant-reference", "constant-candidate", "signed-zeros", "one-value"],
)
def test_pcc_is_one_for_equal_arrays_and_zero_otherwise_when_either_is_constant(
    reference: LassiArray, candidate: LassiArray, expected: float
) -> None:
    assert compared(reference, candidate).pcc == expected


# ---------------------------------------------------------------------------
# Pass rules


@pytest.mark.parametrize(
    ("metric", "reference", "candidate", "passing", "failing"),
    [
        ("pcc", f32(1.0, 2.0, 3.0), f32(1.0, 3.0, 2.0), 0.49, 0.51),  # pcc 0.5
        ("max_abs", bf16_bits(BF16_ONE), bf16_bits(BF16_ONE_NEXT), 0.0078125, 0.0078),  # max_abs 0.0078125
        ("ulp", bf16_bits(BF16_ONE), bf16_bits(BF16_ONE_NEXT), 1, 0),  # max_ulp 1
    ],
    ids=["pcc-at-least", "max-abs-at-most", "ulp-at-most"],
)
def test_each_metric_passes_within_its_threshold_and_fails_past_it(
    metric: str, reference: LassiArray, candidate: LassiArray, passing: float, failing: float
) -> None:
    assert compared(reference, candidate, metric, passing).passed is True
    assert compared(reference, candidate, metric, failing).passed is False


def test_integer_outputs_pass_on_exact_equality_whatever_the_threshold() -> None:
    same = compared(i32(1, 2, 3), i32(1, 2, 3), "pcc", 0.5)
    assert same.passed is True
    assert same.max_abs == 0.0
    assert same.max_ulp is None
    # 1, 2, 3 against 1, 2, 4: deviations -1, 0, 1 and -4/3, -1/3, 5/3; pcc = 3 / sqrt(2 * 14/3) = sqrt(27/28).
    for metric, threshold in (("max_abs", 5.0), ("pcc", -1.0), ("ulp", 1000)):
        differ = compared(i32(1, 2, 3), i32(1, 2, 4), metric, threshold)
        assert differ.passed is False, f"{metric} {threshold}: an integer output must match exactly"
        assert differ.max_abs == 1.0
        assert differ.max_ulp is None
        assert differ.pcc == pytest.approx(math.sqrt(27 / 28), rel=1e-12)


def test_a_dtype_mismatch_fails_with_a_note() -> None:
    stats = compared(f32(1.0), bf16_bits(BF16_ONE))
    assert stats.passed is False
    assert stats.note and "dtype" in stats.note.lower()
    assert (stats.pcc, stats.max_abs, stats.max_ulp) == (None, None, None)


def test_a_shape_mismatch_fails_with_a_note() -> None:
    stats = compared(f32(1.0, 2.0, 3.0, 4.0, shape=(4,)), f32(1.0, 2.0, 3.0, 4.0, shape=(2, 2)))
    assert stats.passed is False
    assert stats.note and "shape" in stats.note.lower()
    assert (stats.pcc, stats.max_abs, stats.max_ulp) == (None, None, None)


# ---------------------------------------------------------------------------
# Output files: compare and align


def test_outputs_are_matched_by_the_array_name_in_each_file(tmp_path: Path) -> None:
    reference = write_files(tmp_path / "reference", {"c.lassiio": f32(1.0, 2.0), "d.lassiio": f32(3.0, name="d")})
    candidate = write_files(tmp_path / "candidate", {"out1.lassiio": f32(3.0, name="d"), "out0.lassiio": f32(1.0, 2.0)})
    stats = by_name(oracle("max_abs", 0.0).compare(reference, candidate))
    assert set(stats) == {"c", "d"}
    assert all(entry.passed for entry in stats.values())


def test_a_missing_output_fails_with_a_note(tmp_path: Path) -> None:
    reference = write_files(tmp_path / "reference", {"c.lassiio": f32(1.0), "d.lassiio": f32(2.0, name="d")})
    candidate = write_files(tmp_path / "candidate", {"c.lassiio": f32(1.0)})
    stats = by_name(oracle("max_abs", 0.0).compare(reference, candidate))
    assert stats["c"].passed is True
    assert stats["d"].passed is False
    assert stats["d"].note and "missing" in stats["d"].note.lower()
    assert (stats["d"].pcc, stats["d"].max_abs, stats["d"].max_ulp) == (None, None, None)


def test_an_extra_output_fails_with_a_note(tmp_path: Path) -> None:
    reference = write_files(tmp_path / "reference", {"c.lassiio": f32(1.0)})
    candidate = write_files(tmp_path / "candidate", {"c.lassiio": f32(1.0), "e.lassiio": f32(9.0, name="e")})
    stats = by_name(oracle("max_abs", 0.0).compare(reference, candidate))
    assert stats["c"].passed is True
    assert stats["e"].passed is False
    assert stats["e"].note and "extra" in stats["e"].note.lower()


def test_a_candidate_file_that_is_not_lassi_io_fails_with_a_note_naming_it(tmp_path: Path) -> None:
    reference = write_files(tmp_path / "reference", {"c.lassiio": f32(1.0)})
    bad = tmp_path / "candidate" / "c.lassiio"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"SYNTHETIC: not a lassi_io file\n")
    built = oracle("max_abs", 0.0)
    assert built.align(run_with(reference), run_with({"c.lassiio": bad})) == 0.0
    stats = built.compare(reference, {"c.lassiio": bad})
    assert not any(entry.passed for entry in stats)
    assert any(entry.note and "c.lassiio" in entry.note for entry in stats), stats


def test_per_input_is_one_when_every_output_passes_and_zero_otherwise(tmp_path: Path) -> None:
    reference = write_files(tmp_path / "reference", {"c.lassiio": f32(1.0, 2.0), "d.lassiio": f32(3.0, name="d")})
    same = write_files(tmp_path / "same", {"c.lassiio": f32(1.0, 2.0), "d.lassiio": f32(3.0, name="d")})
    one_off = write_files(tmp_path / "one-off", {"c.lassiio": f32(1.0, 2.0), "d.lassiio": f32(3.5, name="d")})
    built = oracle("max_abs", 0.25)
    assert built.align(run_with(reference), run_with(same)) == 1.0
    assert built.align(run_with(reference), run_with(one_off)) == 0.0
    stats = by_name(built.compare(reference, one_off))
    assert (stats["c"].passed, stats["d"].passed) == (True, False)
    assert stats["d"].max_abs == 0.5


@pytest.mark.parametrize(
    ("metric", "agreement", "near", "far"),
    [
        # The agreement allows 2**-8 on c and nothing on d; near is 2**-9 off on c, far 2**-7 off on c.
        ("max_abs", {"c": (0.00390625, 32768), "d": (0.0, 0)}, 1.0 + 2.0**-9, 1.0 + 2.0**-7),
        # The agreement allows 2 units on c (units of 2**-23 in [1, 2)); near is 1 unit off, far 3 units off.
        ("ulp", {"c": (2.0**-22, 2), "d": (0.0, 0)}, 1.0 + 2.0**-23, 1.0 + 3 * 2.0**-23),
    ],
    ids=["max-abs", "ulp"],
)
def test_from_baseline_takes_each_outputs_threshold_from_the_recorded_agreement(
    tmp_path: Path, metric: str, agreement: dict[str, tuple[float, int]], near: float, far: float
) -> None:
    recorded = [
        output_stats(name=name, pcc=1.0, max_abs=max_abs, max_ulp=units, passed=True, note=None)
        for name, (max_abs, units) in agreement.items()
    ]
    reference = write_files(tmp_path / "reference", {"c.lassiio": f32(1.0), "d.lassiio": f32(2.0, name="d")})
    within = write_files(tmp_path / "within", {"c.lassiio": f32(near), "d.lassiio": f32(2.0, name="d")})
    past = write_files(tmp_path / "past", {"c.lassiio": f32(far), "d.lassiio": f32(2.0, name="d")})
    d_off = write_files(tmp_path / "d-off", {"c.lassiio": f32(1.0), "d.lassiio": f32(2.0 + 2.0**-22, name="d")})
    bound = oracle(metric, "from_baseline").with_baseline(recorded)
    assert bound.align(run_with(reference), run_with(within)) == 1.0
    assert bound.align(run_with(reference), run_with(past)) == 0.0
    assert bound.align(run_with(reference), run_with(d_off)) == 0.0, "an agreement of 0 on d allows no difference"


def test_from_baseline_refuses_to_align_before_an_agreement_is_bound(tmp_path: Path) -> None:
    reference = write_files(tmp_path / "reference", {"c.lassiio": f32(1.0)})
    unbound = oracle("max_abs", "from_baseline")
    with pytest.raises(ValueError, match="from_baseline"):
        unbound.align(run_with(reference), run_with(reference))
    with pytest.raises(ValueError, match="from_baseline"):
        unbound.compare(reference, reference)
