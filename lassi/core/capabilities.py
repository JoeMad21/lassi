"""Capability declarations shared by every component.

Each component names the capabilities it provides (for example
`emits_warnings` or `supports_power`). Recipes are validated against these
declarations at load time, before any backend is constructed.
"""

from __future__ import annotations

from typing import Iterable, Protocol, cast, runtime_checkable

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
