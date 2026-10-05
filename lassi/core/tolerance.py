"""Output tolerances: how far two runs' output arrays may differ (bible Oracles, binary_io row; Benchmark Suites).

A tolerance names one metric and a threshold. The metrics compare two
arrays of one output and are recorded per output in the Result Record
(lassi.core.record OutputStats):

- pcc: the Pearson correlation of the two arrays; an output is within a
  threshold t when pcc >= t, and t lies in [-1, 1];
- max_abs: the largest absolute difference of two elements; within t when
  max_abs <= t, and t is finite and >= 0;
- ulp: the largest distance of two elements in units in the last place of
  the dtype (OutputStats.max_ulp); within t when max_ulp <= t, and t is
  finite and >= 0.

The binary_io oracle (lassi.oracles.binary_io) takes its metric and a
threshold from its recipe section, and a suite manifest item may declare
the tolerance its two references must meet (lassi.bench.registry). Both
check a threshold with threshold_problem, so one rule holds for both.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

# The metrics, in the order messages list them.
METRICS = ("pcc", "max_abs", "ulp")
# The OutputStats field that holds each metric's value.
STAT_FIELDS: Mapping[str, str] = MappingProxyType({"pcc": "pcc", "max_abs": "max_abs", "ulp": "max_ulp"})


def _shown(value: object) -> str:
    """Return a value for a message: an int wider than 64 bits by its width, anything else by repr, cut at 60."""
    if isinstance(value, int) and not isinstance(value, bool) and value.bit_length() > 64:
        return f"<an int of {value.bit_length()} bits>"
    try:
        text = repr(value)
    except ValueError:  # a nested int past Python's digit limit
        return f"<a {type(value).__name__} too large to show>"
    return text if len(text) <= 60 else f"{text[:60]}... ({len(text)} characters)"


def metric_problem(metric: object) -> str:
    """Return why `metric` is not one of METRICS, or "" when it is."""
    if isinstance(metric, str) and metric in METRICS:
        return ""
    return f"the metric must be one of {', '.join(METRICS)}, got {_shown(metric)}"


def threshold_problem(metric: str, threshold: object) -> str:
    """Return why `threshold` is not a threshold of `metric` (one of METRICS), or "" when it is.

    A threshold is an int or a float, never a bool; it is finite (an int a
    double cannot hold is refused as not finite), in [-1, 1] for pcc and
    >= 0 for max_abs and ulp. It never raises: a huge int is compared, never
    converted, and messages show it by its width.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        return f"the threshold must be a number, got {_shown(threshold)}"
    finite = abs(threshold) <= sys.float_info.max if isinstance(threshold, int) else math.isfinite(threshold)
    if not finite:
        return f"the threshold must be finite, got {_shown(threshold)}"
    if metric == "pcc" and not -1 <= threshold <= 1:
        return f"a pcc threshold must lie in [-1, 1], got {_shown(threshold)}"
    if metric != "pcc" and threshold < 0:
        return f"a {metric} threshold must be >= 0, got {_shown(threshold)}"
    return ""


def within(metric: str, value: float, threshold: float) -> bool:
    """Return True when a metric's `value` meets `threshold`: pcc at least it, max_abs and ulp at most it."""
    return value >= threshold if metric == "pcc" else value <= threshold


@dataclass(frozen=True)
class Tolerance:
    """A metric and its threshold, checked when built: ValueError names the problem."""

    metric: str
    threshold: float

    def __post_init__(self) -> None:
        """Refuse a metric outside METRICS or a threshold threshold_problem refuses."""
        problem = metric_problem(self.metric) or threshold_problem(self.metric, self.threshold)
        if problem:
            raise ValueError(problem)
