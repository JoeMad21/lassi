"""Tests for the stdout_mask oracle (task P1.7).

Bible: Oracles (stdout_mask row), Harness Contract, Component Interfaces
(Oracle row and rules), Result Record (Attempt.alignment).

The rule under test [DESIGN, P1.7]:

- Both stdouts are split into lines on "\\n" only.
- A line is a timing line when any of the item's masks (regular
  expressions, lassi.oracles.load_masks) matches it. Each timing line is
  replaced by one fixed marker line; no line is deleted, so a missing or an
  extra timing line is still a difference.
- per_input is 1.0 when the masked candidate equals the masked reference
  exactly (no whitespace or case normalization), else 0.0.
- mean is the arithmetic mean of per_input.

The public names these tests use, from `lassi.oracles`:

- `load_masks(suite)`: item name -> the item's mask regexes.
- `mask_stdout(text, masks)`: the text with every timing line replaced.
- `stdout_mask_score(reference, candidate, masks)`: 1.0 or 0.0.
- `alignment(per_input)`: the Result Record Alignment with its mean.

SYNTHETIC fixtures: every stdout below is written by hand in the print
formats of the pinned HeCBench sources (assets/bench/lassi-hecbench-10.yaml).
The values in them are invented, not program output, and no test reads them
as measurements. The asserted values (1.0, 0.0, and means) follow from the
rule above, not from any run.
"""

from __future__ import annotations

import pytest

from lassi.core.record import Alignment

SUITE = "lassi-hecbench-10"

# SYNTHETIC: matrix-rotate's stdout, in the print formats of its sources; values invented.
MATRIX_ROTATE = "Average kernel execution time: 1.000000 (s)\nPASS\n"
MATRIX_ROTATE_OTHER_TIME = "Average kernel execution time: 2.500000 (s)\nPASS\n"

# SYNTHETIC: pathfinder's stdout (no PASS/FAIL), with its two timing lines and two data rows; values invented.
PATHFINDER = (
    "Total kernel execution time: 0.100000 (s)\n"
    "Device offloading time = 0.200000 (s)\n"
    "3 1 4 1 5 \n"
    "2 7 1 8 2 \n"
)

# SYNTHETIC: bsearch's stdout (no PASS/FAIL) as the OpenMP reference words it, and as a CUDA-worded translation.
BSEARCH_OMP = "".join(f"Average device execution time (bs{n}) 1.5e-05 (s)\n" for n in range(1, 5))
BSEARCH_CUDA_WORDING = "".join(f"Average kernel execution time (bs{n}) 0.00042 (s)\n" for n in range(1, 5))


def masks(item: str) -> list[str]:
    """Return the item's masks from the suite's mask file under assets/harness/."""
    from lassi.oracles import load_masks

    return list(load_masks(SUITE)[item])


def score(reference: str, candidate: str, item: str) -> float:
    """Return stdout_mask's per-input value for one reference and candidate stdout of `item`."""
    from lassi.oracles import stdout_mask_score

    return stdout_mask_score(reference, candidate, masks(item))


# ---------------------------------------------------------------------------
# Masking


def test_mask_stdout_replaces_each_matching_line_and_keeps_every_other_line() -> None:
    from lassi.oracles import mask_stdout

    text = "setup done\nkernel time 1.5 ms\nresult 42\nkernel time 3.0 ms\n"
    masked = mask_stdout(text, [r"kernel time"])
    before, after = text.split("\n"), masked.split("\n")
    assert len(after) == len(before), "masking replaces lines; it never deletes or adds one"
    assert [after[0], after[2], after[4]] == [before[0], before[2], before[4]], "unmatched lines stay byte for byte"
    assert after[1] == after[3], "every timing line becomes the same marker line"
    assert after[1] != before[1] and after[3] != before[3], "a timing line is replaced"


def test_mask_stdout_masks_the_whole_timing_line_not_only_its_number() -> None:
    from lassi.oracles import mask_stdout

    one = mask_stdout("kernel time 1.5 ms (fast)\n", [r"kernel time"])
    two = mask_stdout("kernel time 99 ms (slow)\n", [r"kernel time"])
    assert one == two


# ---------------------------------------------------------------------------
# Exact per-input values on fixture stdout


def test_identical_stdout_scores_one() -> None:
    assert score(PATHFINDER, PATHFINDER, "pathfinder") == 1.0


def test_stdout_that_differs_only_in_timing_values_scores_one() -> None:
    assert score(MATRIX_ROTATE, MATRIX_ROTATE_OTHER_TIME, "matrix-rotate") == 1.0


def test_both_timing_lines_of_an_item_are_masked() -> None:
    candidate = PATHFINDER.replace("0.100000", "9.900000").replace("0.200000", "8.800000")
    assert score(PATHFINDER, candidate, "pathfinder") == 1.0


def test_a_different_non_timing_line_scores_zero() -> None:
    candidate = PATHFINDER.replace("2 7 1 8 2 ", "2 7 1 8 3 ")
    assert score(PATHFINDER, candidate, "pathfinder") == 0.0


def test_a_missing_timing_line_scores_zero() -> None:
    candidate = PATHFINDER.replace("Device offloading time = 0.200000 (s)\n", "")
    assert score(PATHFINDER, candidate, "pathfinder") == 0.0


def test_an_extra_line_scores_zero() -> None:
    assert score(PATHFINDER, PATHFINDER + "done\n", "pathfinder") == 0.0


def test_whitespace_is_compared_exactly() -> None:
    candidate = PATHFINDER.replace("3 1 4 1 5 \n", "3 1 4 1 5\n")
    assert score(PATHFINDER, candidate, "pathfinder") == 0.0


def test_empty_candidate_stdout_scores_zero() -> None:
    assert score(PATHFINDER, "", "pathfinder") == 0.0


def test_an_item_masks_the_timing_lines_of_both_of_its_languages() -> None:
    """bsearch words its timing lines differently in each language; both are timing lines of the item."""
    assert score(BSEARCH_OMP, BSEARCH_CUDA_WORDING, "bsearch") == 1.0


def test_score_is_a_float() -> None:
    value = score(MATRIX_ROTATE, MATRIX_ROTATE, "matrix-rotate")
    assert isinstance(value, float) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# per_input and mean


@pytest.mark.parametrize(
    ("per_input", "mean"),
    [([1.0], 1.0), ([0.0], 0.0), ([1.0, 0.0], 0.5), ([1.0, 0.0, 1.0, 1.0], 0.75)],
    ids=["one-pass", "one-fail", "half", "three-of-four"],
)
def test_alignment_holds_per_input_values_and_their_mean(per_input: list[float], mean: float) -> None:
    from lassi.oracles import alignment

    assert alignment(per_input) == Alignment(per_input=per_input, mean=mean)


def test_alignment_of_fixture_inputs_is_the_mean_of_their_scores() -> None:
    from lassi.oracles import alignment

    pairs = [
        (PATHFINDER, PATHFINDER),
        (PATHFINDER, PATHFINDER.replace("3 1 4", "3 1 9")),
        (PATHFINDER, PATHFINDER.replace("0.100000", "0.300000")),
        (PATHFINDER, ""),
    ]
    values = [score(reference, candidate, "pathfinder") for reference, candidate in pairs]
    assert alignment(values) == Alignment(per_input=[1.0, 0.0, 1.0, 0.0], mean=0.5)
