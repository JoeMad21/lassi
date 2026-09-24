"""Interval and pass@k statistics for run metrics (task P2.8).

Bible: Evaluation Protocol (preamble: pass@k with Wilson 95% intervals;
Acceptance Criteria, the B0 Wilson criterion), Repository Layout
(`analysis/`).

- wilson_interval: the Wilson score interval without continuity correction
  (Newcombe 1998, Statistics in Medicine 17:857-872, method 3). It takes a
  fractional success count, since a pass@k row's successes are the sum of
  per-scenario pass@k values.
- pass_at_k: the unbiased estimator 1 - C(n - c, k) / C(n, k) of Chen et al.
  2021 (arXiv:2107.03374, Sec. 2.1) for n samples of which c are correct.
"""

from __future__ import annotations

import math

# The two-sided 95% quantile of the standard normal distribution.
Z95 = 1.959963984540054


def wilson_interval(successes: float, n: int, z: float = Z95) -> tuple[float, float]:
    """Return the Wilson score interval (low, high) of `successes` out of `n` at quantile `z`.

    `successes` may be fractional and must lie in [0, n]. The bounds are
    clamped to [0, 1]. Raises ValueError when n is not a positive int (an
    interval over no trials is a caller error), when successes lies outside
    [0, n], or when z is not a positive finite number.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"wilson_interval: n must be a positive int, got n = {n!r}")
    if isinstance(successes, bool) or not isinstance(successes, (int, float)) or not math.isfinite(successes):
        raise ValueError(f"wilson_interval: successes must be a finite number, got {successes!r}")
    if not 0 <= successes <= n:
        raise ValueError(f"wilson_interval: successes must lie in [0, n = {n}], got {successes!r}")
    if isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z) or z <= 0:
        raise ValueError(f"wilson_interval: z must be a positive finite number, got {z!r}")
    rate = successes / n
    z2 = z * z
    scale = 1.0 + z2 / n
    centre = (rate + z2 / (2 * n)) / scale
    half = z * math.sqrt(rate * (1.0 - rate) / n + z2 / (4 * n * n)) / scale
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Return the unbiased pass@k for `n` samples of which `c` are correct, or None when n < k.

    It is 1.0 when n - c < k, since every k-subset then holds a correct
    sample. Raises ValueError unless n and c are ints with 0 <= c <= n and k
    is an int of at least 1.
    """
    for name, value in (("n", n), ("c", c), ("k", k)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"pass_at_k: {name} must be an int, got {value!r}")
    if k < 1 or not 0 <= c <= n:
        raise ValueError(f"pass_at_k: need k >= 1 and 0 <= c <= n, got n = {n}, c = {c}, k = {k}")
    if n < k:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
