"""Tests for the interval and pass@k statistics of lassi.analysis.stats (task P2.8).

Bible: Evaluation Protocol (preamble: pass@k with Wilson 95% intervals;
Acceptance Criteria, the B0 Wilson criterion), Repository Layout
(`analysis/`: pass@k).

The contract these tests fix:

- `wilson_interval(successes: float, n: int, z: float = 1.959963984540054)
  -> tuple[float, float]` is the Wilson score interval without continuity
  correction. `successes` may be fractional: the pass@k rows use successes =
  p x n, with p the pass@k value. The bounds lie in [0, 1] and bracket
  successes / n. n = 0 raises ValueError whose message names n (the choice
  between ValueError and None is fixed here as ValueError: an interval over
  no trials is a caller error, and the metric rows give None with a note
  before they would call it).
- `pass_at_k(n: int, c: int, k: int) -> float | None` is the unbiased
  estimator 1 - C(n - c, k) / C(n, k) of Chen et al. 2021, arXiv:2107.03374,
  Sec. 2.1, for n samples of which c are correct. It returns None when
  n < k, and 1.0 when n - c < k.

Oracles:

- Wilson: the published worked values of Newcombe 1998, "Two-sided
  confidence intervals for the single proportion: comparison of seven
  methods", Statistics in Medicine 17:857-872, Table I, method 3 (score
  method, no continuity correction), each checked to 4 decimals. All four
  were recomputed from the Wilson formula before being used here and
  reproduce to 4 decimals; none was dropped. Two closed forms add exact
  checks: at successes = n the lower bound is n / (n + z^2), and at
  successes = 0 the upper bound is z^2 / (n + z^2). The interval is also
  symmetric: wilson(x, n) = (1 - high(n - x), 1 - low(n - x)).
- pass@k: values worked by hand from the binomial formula, each shown in its
  comment.

No value here is a measurement: every expected number is arithmetic on the
formula or a published worked example.
"""

from __future__ import annotations

import importlib
import math
from types import ModuleType

import pytest

MODULE = "lassi.analysis.stats"
Z95 = 1.959963984540054
FOUR_DECIMALS = 5e-5
EXACT = 1e-12

# Newcombe 1998, Statistics in Medicine 17:857-872, Table I, method 3: (successes, n, lower, upper).
NEWCOMBE_METHOD_3 = (
    (81, 263, 0.2553, 0.3662),
    (15, 148, 0.0624, 0.1605),
    (0, 20, 0.0000, 0.1611),
    (1, 29, 0.0061, 0.1718),
)

# pass@k worked by hand: (n, c, k, 1 - C(n - c, k) / C(n, k)).
PASS_AT_K = (
    (5, 2, 1, 0.4),  # 1 - C(3,1)/C(5,1) = 1 - 3/5
    (5, 1, 3, 0.6),  # 1 - C(4,3)/C(5,3) = 1 - 4/10
    (5, 2, 3, 0.9),  # 1 - C(3,3)/C(5,3) = 1 - 1/10
    (5, 0, 3, 0.0),  # 1 - C(5,3)/C(5,3)
    (5, 3, 3, 1.0),  # n - c = 2 < k: every 3-subset holds a correct sample
    (4, 1, 3, 0.75),  # 1 - C(3,3)/C(4,3) = 1 - 1/4
    (10, 3, 3, 17 / 24),  # 1 - C(7,3)/C(10,3) = 1 - 35/120
    (3, 1, 3, 1.0),  # n = k: the one 3-subset holds the correct sample
    (3, 0, 3, 0.0),  # n = k, nothing correct
)


def stats() -> ModuleType:
    """Import lassi.analysis.stats; fail the test clearly while task P2.8 has not added it."""
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as error:
        if error.name not in ("lassi.analysis", MODULE):
            raise
        pytest.fail(f"task P2.8 adds {MODULE}: {error}")


# ---------------------------------------------------------------------------
# Wilson score interval


@pytest.mark.parametrize("successes, n, lower, upper", NEWCOMBE_METHOD_3,
                         ids=[f"{x}-of-{n}" for x, n, _, _ in NEWCOMBE_METHOD_3])
def test_wilson_matches_newcombe_table_i_method_3(successes: int, n: int, lower: float, upper: float) -> None:
    """Newcombe 1998, Statistics in Medicine 17:857-872, Table I, method 3 (score, no continuity correction)."""
    low, high = stats().wilson_interval(successes, n)
    assert low == pytest.approx(lower, abs=FOUR_DECIMALS)
    assert high == pytest.approx(upper, abs=FOUR_DECIMALS)


def test_wilson_returns_a_pair_of_floats() -> None:
    interval = stats().wilson_interval(15, 148)
    assert isinstance(interval, tuple) and len(interval) == 2
    assert all(type(bound) is float for bound in interval)


def test_wilson_default_z_is_the_two_sided_95_percent_quantile() -> None:
    wilson = stats().wilson_interval
    assert wilson(81, 263) == wilson(81, 263, z=Z95)


@pytest.mark.parametrize("n", [1, 4, 10, 20, 263])
def test_wilson_all_successes_has_the_closed_form_lower_bound(n: int) -> None:
    low, high = stats().wilson_interval(n, n)
    assert low == pytest.approx(n / (n + Z95**2), abs=EXACT)
    assert high == pytest.approx(1.0, abs=EXACT)


@pytest.mark.parametrize("n", [1, 4, 10, 20, 263])
def test_wilson_no_successes_has_the_closed_form_upper_bound(n: int) -> None:
    low, high = stats().wilson_interval(0, n)
    assert low == pytest.approx(0.0, abs=EXACT)
    assert high == pytest.approx(Z95**2 / (n + Z95**2), abs=EXACT)


def test_wilson_uses_the_given_z() -> None:
    # successes = n = 4 with z = 1: lower = 4 / (4 + 1) = 0.8.
    low, high = stats().wilson_interval(4, 4, z=1.0)
    assert low == pytest.approx(0.8, abs=EXACT)
    assert high == pytest.approx(1.0, abs=EXACT)


@pytest.mark.parametrize("successes, n", [(0.6, 2), (1.5, 2), (2.25, 5), (7, 10), (3, 34)])
def test_wilson_is_symmetric_and_takes_fractional_successes(successes: float, n: int) -> None:
    wilson = stats().wilson_interval
    low, high = wilson(successes, n)
    mirror_low, mirror_high = wilson(n - successes, n)
    assert low == pytest.approx(1.0 - mirror_high, abs=EXACT)
    assert high == pytest.approx(1.0 - mirror_low, abs=EXACT)


def test_wilson_bounds_lie_in_the_unit_interval_and_bracket_the_rate() -> None:
    wilson = stats().wilson_interval
    for n in range(1, 41):
        for successes in range(n + 1):
            low, high = wilson(successes, n)
            assert 0.0 <= low <= high <= 1.0, f"{successes}/{n} gave ({low}, {high})"
            assert low - EXACT <= successes / n <= high + EXACT, f"{successes}/{n} gave ({low}, {high})"


def test_wilson_over_no_trials_is_a_clear_value_error() -> None:
    with pytest.raises(ValueError, match=r"\bn\b"):
        stats().wilson_interval(0, 0)


# ---------------------------------------------------------------------------
# pass@k


@pytest.mark.parametrize("n, c, k, expected", PASS_AT_K, ids=[f"n{n}-c{c}-k{k}" for n, c, k, _ in PASS_AT_K])
def test_pass_at_k_is_the_unbiased_estimator(n: int, c: int, k: int, expected: float) -> None:
    """Chen et al. 2021, arXiv:2107.03374, Sec. 2.1: pass@k = 1 - C(n - c, k) / C(n, k)."""
    value = stats().pass_at_k(n, c, k)
    assert type(value) is float, f"pass@k is a float, got {value!r}"
    assert value == pytest.approx(expected, abs=EXACT)


def test_pass_at_1_is_the_share_correct() -> None:
    pass_at_k = stats().pass_at_k
    for n in range(1, 9):
        for c in range(n + 1):
            assert pass_at_k(n, c, 1) == pytest.approx(c / n, abs=EXACT)


def test_pass_at_k_agrees_with_the_binomial_formula() -> None:
    pass_at_k = stats().pass_at_k
    for n in range(3, 11):
        for c in range(n + 1):
            expected = 1.0 - math.comb(n - c, 3) / math.comb(n, 3)
            assert pass_at_k(n, c, 3) == pytest.approx(expected, abs=EXACT), f"n={n} c={c}"


@pytest.mark.parametrize("n, c, k", [(2, 1, 3), (2, 2, 3), (0, 0, 1), (0, 0, 3), (4, 4, 5)])
def test_pass_at_k_is_none_when_n_is_below_k(n: int, c: int, k: int) -> None:
    assert stats().pass_at_k(n, c, k) is None
