"""Tests for the passfail oracle (task P1.7).

Bible: Oracles (passfail row: read PASS/FAIL, confirmed by stdout_mask),
Component Interfaces (Oracle rules: a program's self-reported PASS is never
the only signal), Source Papers (LASSI quirk table, PASS/FAIL row).

The rule under test [DESIGN, P1.7], through
`lassi.oracles.passfail_score(reference, candidate, masks, reads_passfail=...)`:

- `reads_passfail` is True only when the manifest lists the target language
  under the item's `passfail` (assets/bench/lassi-hecbench-10.yaml); the
  oracle stage decides it, and these tests pass it directly.
- When it is False, the value is stdout_mask's alone.
- When it is True, the value is 1.0 only when the candidate's stdout holds
  the token PASS, holds no token FAIL, and stdout_mask scores 1.0; else 0.0.
  Tokens are whole words anywhere in a line (randomAccess's CUDA version
  prints "(PASS)." at the end of a line). A missing PASS is a fail, because
  colorwheel prints PASS but never FAIL (plans/PHASE-NOTES.md, P1 bench
  facts).

SYNTHETIC fixtures: every stdout below is written by hand in the print
formats of the pinned HeCBench sources. The values in them are invented, not
program output, and no test reads them as measurements.
"""

from __future__ import annotations

import pytest

SUITE = "lassi-hecbench-10"

# SYNTHETIC: layout prints PASS after each of its two timed kernels; values invented.
LAYOUT = (
    "Average kernel execution time (AoS): 1.5 (us)\nPASS\n"
    "Average kernel execution time (SoA): 2.5 (us)\nPASS\n"
)
LAYOUT_OTHER_TIMES = (
    "Average kernel execution time (AoS): 7.25 (us)\nPASS\n"
    "Average kernel execution time (SoA): 0.0003 (us)\nPASS\n"
)

# SYNTHETIC: dense-embedding with its PASS line; values invented.
DENSE_EMBEDDING = (
    "Number of rows in the embedding table: 10000\n"
    "Batch size: 8\n"
    "\n"
    "Embedding dimension: 64\n"
    "Average execution time of dense embedding kernel (k1): 1.000000 (us)\n"
    "Average execution time of dense embedding kernel (k2): 2.000000 (us)\n"
    "PASS\n"
)

# SYNTHETIC: colorwheel prints PASS when the results agree and a maximum error line (never FAIL) when not.
COLORWHEEL_PASS = "Start execution on a device\nAverage kernel execution time : 1.000000 (ms)\nPASS\n"
COLORWHEEL_NO_PASS = (
    "Start execution on a device\n"
    "Average kernel execution time : 1.000000 (ms)\n"
    "Maximum error between host and device results: 3\n"
)

# SYNTHETIC: atomicCost prints PASS or FAIL after each timed section; here the second section fails.
ATOMIC_COST_ONE_FAIL = (
    "\n\nEach thread sums up 1 elements\n"
    "Average execution time of WithAtomicOnGlobalMem: 1.000000 (us)\n"
    "Average execution time of WithoutAtomicOnGlobalMem: 2.000000 (us)\n"
    "PASS\n"
    "\n\nEach thread sums up 2 elements\n"
    "Average execution time of WithAtomicOnGlobalMem: 3.000000 (us)\n"
    "Average execution time of WithoutAtomicOnGlobalMem: 4.000000 (us)\n"
    "FAIL\n"
)

# SYNTHETIC: randomAccess's CUDA version reports PASS or FAIL inside its last line.
RANDOM_ACCESS_CUDA = (
    "Table size = 1024\n"
    "Main table size   = 2^10 = 1024 words\n"
    "Number of updates = 4096\n"
    "Average kernel execution time: 1.000000 (s)\n"
    "Found 0 errors in 1024 locations ({verdict}).\n"
)


def score(reference: str, candidate: str, item: str, reads_passfail: bool) -> float:
    """Return passfail's per-input value for one reference and candidate stdout of `item`."""
    from lassi.oracles import load_masks, passfail_score

    return passfail_score(reference, candidate, list(load_masks(SUITE)[item]), reads_passfail=reads_passfail)


def test_pass_with_matching_masked_stdout_scores_one() -> None:
    assert score(LAYOUT, LAYOUT_OTHER_TIMES, "layout", reads_passfail=True) == 1.0


def test_pass_whose_masked_stdout_differs_is_not_a_pass() -> None:
    candidate = DENSE_EMBEDDING.replace("Embedding dimension: 64", "Embedding dimension: 32")
    assert "PASS" in candidate
    assert score(DENSE_EMBEDDING, candidate, "dense-embedding", reads_passfail=True) == 0.0


def test_pass_with_an_extra_line_is_not_a_pass() -> None:
    assert score(LAYOUT, LAYOUT + "PASS\n", "layout", reads_passfail=True) == 0.0


@pytest.mark.parametrize(("reads_passfail", "expected"), [(True, 0.0), (False, 1.0)], ids=["read", "not-read"])
def test_a_missing_pass_is_a_fail_where_passfail_is_read(reads_passfail: bool, expected: float) -> None:
    """colorwheel never prints FAIL: identical output without PASS passes stdout_mask but not passfail."""
    assert score(COLORWHEEL_NO_PASS, COLORWHEEL_NO_PASS, "colorwheel", reads_passfail) == expected


def test_colorwheel_with_pass_scores_one() -> None:
    candidate = COLORWHEEL_PASS.replace("1.000000", "4.000000")
    assert score(COLORWHEEL_PASS, candidate, "colorwheel", reads_passfail=True) == 1.0


@pytest.mark.parametrize(("reads_passfail", "expected"), [(True, 0.0), (False, 1.0)], ids=["read", "not-read"])
def test_any_fail_token_is_a_fail_where_passfail_is_read(reads_passfail: bool, expected: float) -> None:
    """Identical output that holds PASS and FAIL: stdout_mask passes it; passfail does not."""
    assert score(ATOMIC_COST_ONE_FAIL, ATOMIC_COST_ONE_FAIL, "atomicCost", reads_passfail) == expected


@pytest.mark.parametrize(("verdict", "expected"), [("PASS", 1.0), ("FAIL", 0.0)])
def test_a_token_inside_a_line_is_read(verdict: str, expected: float) -> None:
    stdout = RANDOM_ACCESS_CUDA.format(verdict=verdict)
    candidate = stdout.replace("1.000000", "6.000000")
    assert score(stdout, candidate, "randomAccess", reads_passfail=True) == expected


def test_where_passfail_is_not_read_the_value_is_stdout_mask_alone() -> None:
    from lassi.oracles import load_masks, stdout_mask_score

    pairs = [
        (LAYOUT, LAYOUT_OTHER_TIMES, "layout"),
        (DENSE_EMBEDDING, DENSE_EMBEDDING.replace("Batch size: 8", "Batch size: 16"), "dense-embedding"),
        (COLORWHEEL_NO_PASS, COLORWHEEL_NO_PASS, "colorwheel"),
        (ATOMIC_COST_ONE_FAIL, ATOMIC_COST_ONE_FAIL, "atomicCost"),
    ]
    for reference, candidate, item in pairs:
        expected = stdout_mask_score(reference, candidate, list(load_masks(SUITE)[item]))
        assert score(reference, candidate, item, reads_passfail=False) == expected, item
