"""Capability declarations shared by every component.

Each component names the capabilities it provides (for example
`emits_warnings` or `supports_power`). Recipes are validated against these
declarations at load time, before any backend is constructed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from lassi.core.record import Alignment, Diagnostic, OutputStats
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
# The Diagnostic code of a Watcher dump that such an executor adds to a run that hung (task P4.11; the Harness
# Contract's hang diagnostic plus the Watcher dump): a run-stage note whose message is the condensed dump. It is here,
# beside SIMULATOR, so the executor and lassi.core.stages (which re-exports it as WATCHER_CODE and carries each such
# note into the run's error text after the hang diagnostic) share one name without importing each other.
WATCHER_CODE = "watcher"

# The capability of an Oracle that compares a run's output files (RunResult.output_files, one lassi_io array per
# file) instead of its stdout; it provides the OutputFileOracle methods below. For such an Oracle the oracle stage
# aligns output files, and baseline and run_loop keep each run's output files in the binary store
# (lassi.core.store.BlobStore, RunInfo.outputs); for any other Oracle they keep none.
ALIGNS_OUTPUT_FILES = "aligns_output_files"

# The capability of a Toolchain that reads a built program's host code for the Harness Contract's
# host-compute guard (task P4.12): it provides host_compute_guard(files, harness) (HostComputeGuard).
# compile_loop asks the target language's toolchain for a reading after every build that gave a
# program, by this capability, never by the toolchain's name.
HOST_COMPUTE_GUARD = "host_compute_guard"
# Every Diagnostic code of a guard reading starts with this prefix; compile_loop keeps such diagnostics
# out of every correction prompt (lassi.core.stages prompt_diagnostics).
GUARD_CODE_PREFIX = "guard-"
# A guard reads program text and never fails an attempt: its diagnostics are parse-stage warnings or notes.
GUARD_STAGE = "parse"
GUARD_SEVERITIES = frozenset({"warning", "note"})


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


@dataclass(frozen=True)
class HostComputeReading:
    """A guard's reading of one built program: host_compute (True, False, or None for not checked) and its notes.

    host_compute is True for a violation, False when the guard checked and
    found none, and None when it did not check. `diagnostics` are the
    guard's parse-stage warnings or notes, each with a code that starts
    with GUARD_CODE_PREFIX; the record itself checks nothing, and
    host_compute_reading refuses one outside that contract.
    """

    host_compute: bool | None
    diagnostics: tuple[Diagnostic, ...] = ()


class HostComputeGuard(Protocol):
    """A Toolchain that declares HOST_COMPUTE_GUARD."""

    def host_compute_guard(self, files: Mapping[str, str], harness: Mapping[str, str]) -> HostComputeReading:
        """Return the reading of the program built from `files` (the model's) and `harness` (the support files)."""
        ...


def host_compute_reading(
    toolchain: object, files: Mapping[str, str], harness: Mapping[str, str]
) -> HostComputeReading | None:
    """Return `toolchain`'s reading when it declares HOST_COMPUTE_GUARD, else None (not checked).

    The method of a toolchain that does not declare the capability is never
    called. A ValueError naming the toolchain (its `name`, else its class
    name) and the capability refuses a declared capability without a
    callable host_compute_guard, and a reading outside the contract: not a
    HostComputeReading, host_compute not a bool or None, diagnostics not a
    tuple, or an item that is not a Diagnostic, has a stage other than
    GUARD_STAGE, a severity outside GUARD_SEVERITIES, or a code that is not
    a str starting with GUARD_CODE_PREFIX. So a guard can never add an
    error, change a stage, or start a correction.
    """
    if not declares(toolchain, HOST_COMPUTE_GUARD):
        return None
    name = getattr(toolchain, "name", type(toolchain).__name__)
    method = getattr(toolchain, HOST_COMPUTE_GUARD, None)
    if not callable(method):
        raise ValueError(
            f"Toolchain {name!r} declares {HOST_COMPUTE_GUARD!r} but has no callable {HOST_COMPUTE_GUARD}()"
        )
    reading = method(files, harness)
    problem = _reading_problem(reading)
    if problem is not None:
        raise ValueError(f"Toolchain {name!r} declares {HOST_COMPUTE_GUARD!r}, but its reading {problem}")
    return cast(HostComputeReading, reading)


def _reading_problem(reading: object) -> str | None:
    """Return why `reading` is outside the guard contract (see host_compute_reading), or None when it is inside."""
    from lassi.core.record import Diagnostic  # lassi.core.record imports this module through lassi.core.interfaces

    if not isinstance(reading, HostComputeReading):
        return f"is a {type(reading).__name__}, not a HostComputeReading"
    if not (reading.host_compute is None or isinstance(reading.host_compute, bool)):
        return f"has host_compute {reading.host_compute!r}, not True, False, or None"
    if not isinstance(reading.diagnostics, tuple):
        return f"has diagnostics of type {type(reading.diagnostics).__name__}, not a tuple"
    for item in reading.diagnostics:
        if not isinstance(item, Diagnostic):
            return f"holds a {type(item).__name__} among its diagnostics, not a Diagnostic"
        if item.stage != GUARD_STAGE or item.severity not in GUARD_SEVERITIES:
            allowed = f"{GUARD_STAGE}-stage warnings or notes"
            return f"holds a {item.stage}-stage {item.severity}; a guard gives only {allowed}"
        if not isinstance(item.code, str) or not item.code.startswith(GUARD_CODE_PREFIX):
            return f"holds the diagnostic code {item.code!r}, which does not start with {GUARD_CODE_PREFIX!r}"
    return None


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
