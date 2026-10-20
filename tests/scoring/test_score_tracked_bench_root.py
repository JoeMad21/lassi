"""Tests for the bench root lassi score gives a profile for a tracked suite (task P16.8).

Bible: Benchmark Suites (adv-seeded-v0, a tracked suite), Project Recipes
(the score step). Plan: plans/p16-adversarial.md, P16.8.

The contract these tests fix:

- lassi.scoring.score_run find_bench_root returns the repository,
  lassi.bench.registry TRACKED_ROOT, as the bench root of a profile that
  declares reads_bench_sources (lassi) for trials of a tracked suite
  (adv-seeded-v0), with no bench root given and LASSI_SCRATCH unset, as the
  runner reads such a suite.
- A suite with fetched sources (tt-pairs-v0) still needs LASSI_SCRATCH: with
  it unset and no bench root given, find_bench_root raises ScoreError.

The trials are SYNTHETIC stand-ins that carry only bench_item.suite, the one
field find_bench_root reads; nothing here is a measurement.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from lassi.bench import registry
from lassi.scoring import score_run


def trials_of(suite: str) -> list[Any]:
    """Return two SYNTHETIC trial stand-ins of `suite`."""
    return [SimpleNamespace(bench_item=SimpleNamespace(suite=suite)) for _ in range(2)]


def test_a_tracked_suite_gets_the_repository_without_lassi_scratch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LASSI_SCRATCH", raising=False)
    root = score_run.find_bench_root(["lassi"], trials_of("adv-seeded-v0"), None)
    assert root == registry.TRACKED_ROOT.resolve()


def test_a_suite_with_fetched_sources_still_needs_lassi_scratch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LASSI_SCRATCH", raising=False)
    with pytest.raises(score_run.ScoreError, match="LASSI_SCRATCH"):
        score_run.find_bench_root(["lassi"], trials_of("tt-pairs-v0"), None)
