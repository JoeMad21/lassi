"""Benchmark suites: the item registry with splits, reference targets, and pinned sources.

See lassi.bench.registry. Eval and unassigned items are refused to training,
prompt tuning, and corpus harvest (Agent Rule 5).
"""

from lassi.bench.registry import (
    ComputeSource,
    ComputeUnit,
    Direction,
    EvalSplitError,
    LanguageSources,
    Suite,
    SuiteItem,
    SupportFile,
    load_suite,
    program_args,
    sources_dir,
    stage_inputs,
)

__all__ = [
    "ComputeSource",
    "ComputeUnit",
    "Direction",
    "EvalSplitError",
    "LanguageSources",
    "Suite",
    "SuiteItem",
    "SupportFile",
    "load_suite",
    "program_args",
    "sources_dir",
    "stage_inputs",
]
