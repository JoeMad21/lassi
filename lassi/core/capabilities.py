"""Capability declarations shared by every component.

Each component names the capabilities it provides (for example
`emits_warnings` or `supports_power`). Recipes are validated against these
declarations at load time, before any backend is constructed.
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable


@runtime_checkable
class Component(Protocol):
    """Anything a recipe can bind: it has a registry name and capabilities."""

    name: str
    capabilities: frozenset[str]


def missing_capabilities(required: Iterable[str], component: Component) -> frozenset[str]:
    """Return the required capabilities that `component` does not declare."""
    return frozenset(required) - frozenset(component.capabilities)
