"""The component registry and the capability check run when a recipe loads.

A recipe names components by registry name (bible, Design Principle 1). Each
component class declares, as class attributes, what the registry needs to
check a recipe without building anything (bible, Component Interfaces):

- `capabilities`: the capabilities it provides (required).
- `requires`: interface name -> capabilities it needs from every component
  bound to that interface (optional).
- `config_keys`: extra keys it accepts in its recipe section besides `kind`
  (optional).

The registry reads these attributes and never calls the class, so a recipe
whose components do not fit together fails before any backend is constructed.

The runner, never the registry, constructs components. A component bound by a
kind section is built as `factory(**binding.config)`; an LLM backend as
`factory(model_id)`, with keyword settings left at their defaults unless the
caller passes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

# The twelve interface names, as the Protocol classes in lassi/core/interfaces.py, in the bible's table order.
INTERFACES: tuple[str, ...] = (
    "LLMBackend",
    "Frontend",
    "Target",
    "IRLevel",
    "Toolchain",
    "Executor",
    "Oracle",
    "Profiler",
    "Agent",
    "Judge",
    "ScoreProfile",
    "Stage",
)

_Class = TypeVar("_Class", bound=type)


class RegistryError(ValueError):
    """A bad component declaration, an unknown component, or components that do not fit together."""


@dataclass(frozen=True)
class Entry:
    """One registered component: the class and what it declares."""

    interface: str
    name: str
    factory: type
    capabilities: frozenset[str]
    requires: Mapping[str, frozenset[str]]
    config_keys: frozenset[str]


@dataclass(frozen=True)
class Binding:
    """One component a recipe binds: `where` is its recipe key path, `config` its section's keys other than kind."""

    interface: str
    name: str
    where: str
    config: Mapping[str, Any]


class Registry:
    """Registered components by interface and name; registering reads class attributes and never constructs."""

    def __init__(self) -> None:
        """Start with no components registered."""
        self._entries: dict[str, dict[str, Entry]] = {interface: {} for interface in INTERFACES}

    def register(self, interface: str, name: str, factory: type) -> Entry:
        """Register `factory` as component `name` of `interface` and return its entry."""
        where = f"{interface} {name!r}"
        _require_interface(interface, where)
        if not isinstance(name, str) or not name:
            raise RegistryError(f"{where}: the name must be a non-empty string")
        if not isinstance(factory, type):
            raise RegistryError(f"{where}: the factory must be a class, not {factory!r}")
        if name in self._entries[interface]:
            raise RegistryError(f"{where} is already registered")
        entry = Entry(
            interface=interface,
            name=name,
            factory=factory,
            capabilities=_declared_capabilities(factory, where),
            requires=_declared_requires(factory, where),
            config_keys=_string_set(getattr(factory, "config_keys", ()), f"{where} config_keys"),
        )
        self._entries[interface][name] = entry
        return entry

    def get(self, interface: str, name: str) -> Entry:
        """Return the entry for `name` of `interface`; the error lists the names that are registered."""
        _require_interface(interface, f"component {name!r}")
        entry = self._entries[interface].get(name)
        if entry is None:
            names = self.names(interface)
            registered = f"registered: {', '.join(names)}" if names else "none registered"
            raise RegistryError(f"no {interface} component is registered as {name!r}; {registered}")
        return entry

    def names(self, interface: str) -> list[str]:
        """Return the registered names of `interface`, sorted."""
        _require_interface(interface, "names")
        return sorted(self._entries[interface])


DEFAULT_REGISTRY = Registry()


def register(interface: str, name: str) -> Callable[[_Class], _Class]:
    """Return a class decorator that registers the class in DEFAULT_REGISTRY and returns it unchanged."""

    def decorate(cls: _Class) -> _Class:
        DEFAULT_REGISTRY.register(interface, name, cls)
        return cls

    return decorate


def check_bindings(bindings: Sequence[Binding], registry: Registry) -> None:
    """Raise RegistryError on the first binding that does not fit; construct nothing.

    Three passes, each over the bindings in recipe order: every component is
    registered, every config key is one its component accepts, and every
    requirement a component declares is met by every component bound to the
    required interface.
    """
    entries: list[Entry] = []
    for binding in bindings:
        try:
            entries.append(registry.get(binding.interface, binding.name))
        except RegistryError as exc:
            raise RegistryError(f"{binding.where}: {exc}") from exc
    for binding, entry in zip(bindings, entries, strict=True):
        _check_config(binding, entry)
    for binding, entry in zip(bindings, entries, strict=True):
        _check_requires(binding, entry, bindings, entries)


def _check_config(binding: Binding, entry: Entry) -> None:
    """Raise RegistryError when the binding carries a config key its component does not accept."""
    for key in binding.config:
        if key not in entry.config_keys:
            keys = sorted(entry.config_keys)
            accepted = f"accepted keys: {', '.join(keys)}" if keys else "it accepts no config keys"
            raise RegistryError(
                f"{_config_path(binding.where, key)}: {entry.interface} {entry.name!r} ({binding.where}) "
                f"does not accept the key {key!r}; {accepted}"
            )


def _check_requires(binding: Binding, entry: Entry, bindings: Sequence[Binding], entries: Sequence[Entry]) -> None:
    """Raise RegistryError when a requirement of `entry` is not met by the components bound to that interface."""
    who = f"{entry.interface} {entry.name!r} ({binding.where})"
    for interface, needed in entry.requires.items():
        providers = [(b, e) for b, e in zip(bindings, entries, strict=True) if b.interface == interface]
        if not providers:
            wanted = _capability_phrase(interface, needed)
            raise RegistryError(f"{who} requires {wanted}, but the recipe binds no {interface}")
        for provider, provided in providers:
            for capability in sorted(needed):
                if capability not in provided.capabilities:
                    declared = ", ".join(sorted(provided.capabilities)) or "nothing"
                    raise RegistryError(
                        f"{who} requires capability {capability!r} from {interface} {provided.name!r} "
                        f"({provider.where}), which declares: {declared}"
                    )


def _capability_phrase(interface: str, needed: frozenset[str]) -> str:
    """Name what a component needs from an interface, for a requirement no bound component can meet."""
    if not needed:
        return f"the {interface} interface"
    names = ", ".join(repr(capability) for capability in sorted(needed))
    return f"{interface} {'capability' if len(needed) == 1 else 'capabilities'} {names}"


_KIND_SUFFIX = ".kind"


def _config_path(where: str, key: str) -> str:
    """Return the recipe path of a config key: the path where the key actually sits in the recipe.

    In a kind section the key sits beside `kind`, so where `executor.kind` and
    key `hots` give `executor.hots`. Any other where is kept whole, so
    `agents.generator` and `x` give `agents.generator.x`.
    """
    section = where[: -len(_KIND_SUFFIX)] if where.endswith(_KIND_SUFFIX) else where
    return f"{section}.{key}"


def _require_interface(interface: str, where: str) -> None:
    """Raise RegistryError unless `interface` is one of the twelve interface names."""
    if interface not in INTERFACES:
        raise RegistryError(f"{where}: unknown interface {interface!r}; interfaces: {', '.join(INTERFACES)}")


def _declared_capabilities(factory: type, where: str) -> frozenset[str]:
    """Return the class's required `capabilities` attribute as a frozenset."""
    if not hasattr(factory, "capabilities"):
        label = getattr(factory, "__name__", repr(factory))
        raise RegistryError(f"{where}: {label} declares no capabilities attribute")
    return _string_set(factory.capabilities, f"{where} capabilities")


def _declared_requires(factory: type, where: str) -> Mapping[str, frozenset[str]]:
    """Return the class's optional `requires` attribute, keyed by interface in table order and read-only."""
    requires = getattr(factory, "requires", {})
    if not isinstance(requires, Mapping):
        raise RegistryError(f"{where}: requires must map interface names to capabilities")
    unknown = sorted(str(key) for key in requires if key not in INTERFACES)
    if unknown:
        raise RegistryError(
            f"{where}: requires names unknown interfaces {', '.join(unknown)}; interfaces: {', '.join(INTERFACES)}"
        )
    ordered = {key: _string_set(requires[key], f"{where} requires[{key}]") for key in INTERFACES if key in requires}
    return MappingProxyType(ordered)


def _string_set(values: Iterable[str], where: str) -> frozenset[str]:
    """Return `values` as a frozenset, refusing a bare string or any item that is not a string."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise RegistryError(f"{where}: must be a collection of strings, not {values!r}")
    items = list(values)
    wrong = [item for item in items if not isinstance(item, str)]
    if wrong:
        raise RegistryError(f"{where}: every item must be a string, not {wrong[0]!r}")
    return frozenset(items)
