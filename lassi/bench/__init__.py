"""Benchmark suites: the item registry with splits, reference targets, and pinned sources.

See lassi.bench.registry. Eval items are refused to training (Agent Rule 5).
"""

from lassi.bench.registry import (
    Direction,
    EvalSplitError,
    LanguageSources,
    Suite,
    SuiteItem,
    load_suite,
    sources_dir,
)

__all__ = ["Direction", "EvalSplitError", "LanguageSources", "Suite", "SuiteItem", "load_suite", "sources_dir"]
