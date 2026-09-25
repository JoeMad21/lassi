"""Build a registered ScoreProfile by name, with the bench root it needs.

Bible: Component Interfaces (ScoreProfile; the capability rule), Design
Principle 1 (a recipe or a command names a component; the code never checks
for a particular one).

A profile that declares the capability READS_BENCH_SOURCES
(lassi.core.capabilities) reads the suite's fetched sources and is built as
`factory(bench_root=<root>)`; every other profile is built as `factory()`
and never sees a bench root. The decision reads only the class's declared
capabilities, so a scoring pass (lassi.scoring.score_run) and the runner
build every profile the same way, and a new profile needs no change here.
"""

from __future__ import annotations

from pathlib import Path

from lassi.core.capabilities import READS_BENCH_SOURCES
from lassi.core.interfaces import ScoreProfile
from lassi.core.registry import DEFAULT_REGISTRY, Registry

INTERFACE = "ScoreProfile"


def reads_bench_sources(name: str, registry: Registry = DEFAULT_REGISTRY) -> bool:
    """Return True when the registered ScoreProfile `name` declares READS_BENCH_SOURCES.

    Nothing is built. Raises RegistryError, listing the registered names,
    when no ScoreProfile is registered as `name`.
    """
    return READS_BENCH_SOURCES in registry.get(INTERFACE, name).capabilities


def build_profile(name: str, *, bench_root: Path | None, registry: Registry = DEFAULT_REGISTRY) -> ScoreProfile:
    """Build and return the registered ScoreProfile `name`.

    A profile that declares READS_BENCH_SOURCES is built as
    factory(bench_root=bench_root); any other as factory(), and `bench_root`
    is not passed to it. Raises RegistryError for an unknown name, and
    ValueError when a profile that reads bench sources gets no bench root.
    The profile's own errors (such as a ValueError for a profile file it
    refuses, or a bench root that is not a directory) propagate.
    """
    entry = registry.get(INTERFACE, name)
    if READS_BENCH_SOURCES not in entry.capabilities:
        return entry.factory()
    if bench_root is None:
        raise ValueError(f"the ScoreProfile {name!r} reads the suite's bench sources and needs a bench root")
    return entry.factory(bench_root=Path(bench_root))
