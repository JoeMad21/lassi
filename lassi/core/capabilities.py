"""Capability declarations shared by every component.

Each component names the capabilities it provides (for example
`emits_warnings` or `supports_power`). Recipes are validated against these
declarations at load time, before any backend is constructed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from lassi.core.record import Alignment, OutputStats
    from lassi.core.tolerance import Tolerance

# The capability of an LLM backend whose model is unloaded before generated code runs, as upstream's notebook does
# for Ollama (bible Source Papers, LASSI quirk table, Ollama row). A backend that declares it provides unload();
# the runner asks it to unload at trial start and run_loop right before each run of an attempt, never by checking
# the backend's type.
UNLOAD_BEFORE_RUN = "unload_before_run"

# The capability of a ScoreProfile that also scores each attempt: it provides score_attempts(trial), one Score per
# attempt in attempt order. A scoring pass writes attempt scores only for a profile that declares it.
SCORES_ATTEMPTS = "scores_attempts"

# The capability of a ScoreProfile that reads the suite's fetched bench sources (such as a reference target). It is
# built as factory(bench_root=<root of those sources>) and every other profile as factory()
# (lassi.scoring.profiles.build_profile), so a scoring pass looks for a bench root only when a profile declares it.
READS_BENCH_SOURCES = "reads_bench_sources"

# The capability of an Executor that runs programs on a simulator (task P4.6). The stages and pages read it, never the
# executor's name: trial.md and run.md label the wall times of its runs as simulator wall time, not performance
# (Agent Rule 2), and run_loop adds the Harness Contract's hang diagnostic, whose text is the ttsim hint, to an attempt
# run of it that hung (lassi.core.stages HANG_DIAGNOSTIC). Its findings (undefined behavior, a gap, jit-stage
# diagnostics) come in RunResult's fields.
SIMULATOR = "simulator"

# The capability of an Oracle that compares a run's output files (RunResult.output_files, one lassi_io array per
# file) instead of its stdout; it provides the OutputFileOracle methods below. For such an Oracle the oracle stage
# aligns output files, and baseline and run_loop keep each run's output files in the binary store
# (lassi.core.store.BlobStore, RunInfo.outputs); for any other Oracle they keep none.
ALIGNS_OUTPUT_FILES = "aligns_output_files"


@runtime_checkable
class Component(Protocol):
    """Anything a recipe can bind: it has a registry name and capabilities."""

    name: str
    capabilities: frozenset[str]


def missing_capabilities(required: Iterable[str], component: Component) -> frozenset[str]:
    """Return the required capabilities that `component` does not declare."""
    return frozenset(required) - frozenset(component.capabilities)


def declares(component: object, capability: str) -> bool:
    """Return True when `component` names `capability` among its capabilities (none counts as no capabilities)."""
    return capability in frozenset(getattr(component, "capabilities", ()))


class Unloads(Protocol):
    """A backend that declares UNLOAD_BEFORE_RUN: it drops its model from memory when asked."""

    def unload(self) -> None:
        """Drop the model from memory now."""


def unload_before_run(backend: object) -> None:
    """Ask `backend` to unload its model when it declares UNLOAD_BEFORE_RUN; do nothing otherwise.

    The runner refuses, before any trial, a backend that declares the
    capability but has no unload(), so the call here always has a method.
    """
    if declares(backend, UNLOAD_BEFORE_RUN):
        cast(Unloads, backend).unload()


class OutputFileOracle(Protocol):
    """An Oracle that declares ALIGNS_OUTPUT_FILES: what the stages read from it (lassi.oracles.binary_io).

    Output files are given as RunResult.output_files holds them: each
    file's relative path -> the path of its bytes. `sides` names the two
    runs of a comparison in the notes, (reference, candidate) by default.
    """

    name: str

    @property
    def agreement_setting(self) -> str | None:
        """Return the recipe setting that makes this oracle read the references' agreement, or None when none does."""
        ...

    def describe(self) -> str:
        """Return a one-line description of what each output must meet."""
        ...

    def compare(
        self,
        reference_files: Mapping[str, Path],
        candidate_files: Mapping[str, Path],
        *,
        sides: tuple[str, str] = ...,
    ) -> list[OutputStats]:
        """Return one OutputStats per output found on either side."""
        ...

    def alignment(
        self,
        reference_files: Mapping[str, Path],
        candidate_files: Mapping[str, Path],
        *,
        sides: tuple[str, str] = ...,
    ) -> Alignment:
        """Return the candidate's Alignment against the reference, its per-output statistics included."""
        ...

    def reference_problem(self, files: Mapping[str, Path], side: str = ...) -> str | None:
        """Return why a reference run's output files cannot be compared against, or None when they can."""
        ...

    def with_baseline(self, agreement: Sequence[OutputStats] | None) -> OutputFileOracle:
        """Return a copy whose thresholds come from the references' recorded agreement."""
        ...

    def with_tolerance(self, tolerance: Tolerance) -> OutputFileOracle:
        """Return a copy that judges every output against `tolerance` (its metric and threshold)."""
        ...
