"""Tests for the recipe loader and the component registry (P0.3).

The loader is lassi/core/recipe.py and the registry is lassi/core/registry.py.
The tests follow the bible's Project Recipes section (the yaml blocks and the
Notes), Design Principles 1, 4, and 5, the capability rule in Component
Interfaces, the Config row of Readability Standards, and the LASSI quirk table
(the uncapped loop and the fence quirk are faithful toggles). Bible data (the
projects/base.yaml block, the project recipe blocks, the agent and judge
bindings block, and the interface names) is parsed from docs/BIBLE.md, so drift
between the bible and the code fails.

Fixture recipes live in tests/fixtures/recipes/. Every test passes that
directory as the only root, except the tests that reach projects/base.yaml
through the default roots. Components are fake classes defined here whose
constructors record the call and raise, so a load that constructs anything
fails. The golden resolved file and the pinned recipe_hash were computed from
the expected merged data below with the canonical YAML rule; no value in this
module is a measurement.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import importlib
import importlib.util
import re
import shutil
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
BIBLE = REPO / "docs" / "BIBLE.md"
FIXTURES = REPO / "tests" / "fixtures" / "recipes"
GOLDEN = FIXTURES / "expected" / "child.resolved.yaml"
PROJECTS = REPO / "projects"
BASE_YAML = PROJECTS / "base.yaml"
NEW_MODULES = ("lassi.core.recipe", "lassi.core.registry")
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)

# sha256 of canonical(EXPECTED_CHILD); computed once from the data below, never from a loader run.
CHILD_HASH = "f62fcef7d400815920bfea49f7a4efa64faedc01f5c8fb33a2f343d487a5d1c1"
CHILD_CHAIN = ("fixture-base", "parent", "child")

FENCE_TAG_DESCRIPTION = (
    "strip only the exact fence language tag; off keeps upstream's quirk that also drops a leading 'c' after cpp/c++"
)

# Keys of the bible's yaml blocks in Project Recipes, by their first comment line; guards the parser.
BIBLE_BLOCK_KEYS = (
    "projects/base.yaml",
    "projects/lassi-repro/recipe.yaml",
    "projects/lassi-ee/recipe.yaml",
    "projects/lassi-df/recipe.yaml",
    "projects/lassi-df/train.yaml",
    "agent and judge bindings, valid in any recipe",
)
BINDINGS_BLOCK = BIBLE_BLOCK_KEYS[-1]
BIBLE_RUN_RECIPES = ("lassi-repro", "lassi-ee", "lassi-df")

# Top-level keys of a run recipe (recipe.SCHEMA); train recipe keys are not among them.
RUN_RECIPE_KEYS = frozenset(
    "extends faithful fixes llm loop trials runs_root sandbox report bench directions prompts context toolchain"
    " stages executor oracle profiler adversary model arms metrics refine score agents judges".split()
)

# Keys that must be present and non-null in every resolved recipe (recipe.REQUIRED).
REQUIRED = (
    "llm.sampling.temperature",
    "llm.sampling.top_p",
    "loop.max_corrections",
    "trials.n",
    "runs_root",
    "sandbox.network",
    "sandbox.wall_s",
    "sandbox.mem_gb",
    "report.trial_md",
    "report.parquet",
    "bench.suite",
    "bench.split",
    "directions",
    "stages",
    "executor.kind",
)

# Kind sections and the interface each binds.
KIND_SECTIONS = {"executor": "Executor", "oracle": "Oracle", "profiler": "Profiler", "adversary": "Agent"}

# Names that belong to projects and must never appear in the shared loader or registry source.
PROJECT_NAMES = ("lassi-repro", "lassi-ee", "lassi-df", "hecbench", "qwen", "wizardcoder", "a100", "mi300x", "gpt-oss")

E_ACUTE = "\N{LATIN SMALL LETTER E WITH ACUTE}"

# The resolved mapping of child.yaml: fixture-base, then parent, then child, plus the materialized defaults.
EXPECTED_CHILD: dict[str, Any] = {
    "adversary": {"kind": "fuzzer", "budget": {"inputs": 64}},
    "agents": {"generator": {"model": "fixture-model"}, "fixer": {"model": "same_as_generator"}},
    "arms": ["fixture-arm"],
    "bench": {"suite": "fixture-suite", "split": "eval"},
    "context": [],
    "directions": [{"source": "omp", "target": "cuda"}],
    "executor": {"kind": "native"},
    "faithful": False,
    "fixes": {"fence_tag": True},
    "judges": {
        "equivalence": {"model": "judge-model", "rubric": "equivalence-v1", "use": "screen"},
        "efficiency": {"model": "judge-model", "rubric": "efficiency-v1", "mode": "predict", "use": "metric"},
    },
    "llm": {"sampling": {"temperature": 0.2, "top_p": 0.95, "max_tokens": 2048}},
    "loop": {"max_corrections": 3},
    "metrics": ["correct", "first_try"],
    "model": {"backend": "mock_llm", "id": "fixture-model"},
    "oracle": {"kind": "stdout_mask", "passfail": False},
    "profiler": {"kind": "timer", "interval_ms": 10},
    "prompts": "fixture-prompts",
    "report": {"trial_md": True, "parquet": True},
    "runs_root": "fixture-runs",
    "sandbox": {"network": False, "wall_s": 120, "mem_gb": 32},
    "score": "eval_metrics",
    "stages": ["generate", "compile_loop", "run_loop"],
    "toolchain": {"cuda": "fake_cc", "omp": "fake_cxx"},
    "trials": {"n": 2},
}

# (interface, name, where, config) for child.yaml, in the binding order of recipe._bindings.
EXPECTED_CHILD_BINDINGS: list[tuple[str, str, str, dict[str, Any]]] = [
    ("LLMBackend", "mock_llm", "model.backend", {}),
    ("Toolchain", "fake_cc", "toolchain.cuda", {}),
    ("Toolchain", "fake_cxx", "toolchain.omp", {}),
    ("Executor", "native", "executor.kind", {}),
    ("Oracle", "stdout_mask", "oracle.kind", {"passfail": False}),
    ("Profiler", "timer", "profiler.kind", {"interval_ms": 10}),
    ("Agent", "fuzzer", "adversary.kind", {"budget": {"inputs": 64}}),
    ("Stage", "generate", "stages[0]", {}),
    ("Stage", "compile_loop", "stages[1]", {}),
    ("Stage", "run_loop", "stages[2]", {}),
    ("ScoreProfile", "eval_metrics", "score", {}),
    ("Agent", "fixer", "agents.fixer", {}),
    ("Agent", "generator", "agents.generator", {}),
    ("Judge", "efficiency", "judges.efficiency", {}),
    ("Judge", "equivalence", "judges.equivalence", {}),
]

# A complete recipe that extends nothing; the required-key tests delete one key at a time.
STANDALONE: dict[str, Any] = {
    "llm": {"sampling": {"temperature": 0.2, "top_p": 0.9}},
    "loop": {"max_corrections": 10},
    "trials": {"n": 5},
    "runs_root": "fixture-runs",
    "sandbox": {"network": False, "wall_s": 60, "mem_gb": 4},
    "report": {"trial_md": True, "parquet": False},
    "bench": {"suite": "fixture-suite", "split": "eval"},
    "directions": [{"source": "omp", "target": "cuda"}],
    "stages": ["run_loop"],
    "executor": {"kind": "native"},
}


# ---------------------------------------------------------------------------
# Fake components


CONSTRUCTED: list[str] = []


class NeverBuilt:
    """Base for every fake component: constructing one records its name and fails the test."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        name = getattr(type(self), "name", type(self).__name__)
        CONSTRUCTED.append(name)
        raise AssertionError(f"component {name!r} was constructed while loading a recipe")


def fake(
    name: str,
    capabilities: Iterable[str] = (),
    *,
    requires: Mapping[str, Iterable[str]] | None = None,
    config_keys: Iterable[str] = (),
) -> type:
    """Return a fake component class with the class attributes the registry reads."""
    namespace: dict[str, Any] = {
        "__doc__": f"Fake component {name} for the recipe tests; never constructed.",
        "name": name,
        "capabilities": frozenset(capabilities),
        "requires": {interface: frozenset(caps) for interface, caps in (requires or {}).items()},
        "config_keys": frozenset(config_keys),
    }
    return type(f"Fake_{name}", (NeverBuilt,), namespace)


FAKES: tuple[tuple[str, type], ...] = (
    ("LLMBackend", fake("mock_llm", {"chat"})),
    ("Toolchain", fake("fake_cc", {"emits_warnings"})),
    ("Toolchain", fake("fake_cxx", {"emits_warnings", "openmp"})),
    ("Toolchain", fake("silent_cc")),
    ("Executor", fake("native", {"runs_code"}, config_keys={"host"})),
    ("Executor", fake("none", {"compile_only"})),
    ("Oracle", fake("stdout_mask", {"stdout_diff"}, config_keys={"passfail"})),
    ("Profiler", fake("timer", {"timing"}, config_keys={"interval_ms"})),
    ("Agent", fake("fuzzer", {"fuzzes_inputs"}, config_keys={"budget"})),
    ("Agent", fake("generator", {"generates"}, requires={"LLMBackend": {"chat"}})),
    ("Agent", fake("fixer", {"fixes_code"})),
    ("Agent", fake("reviewer", {"reviews"})),
    ("Judge", fake("equivalence", {"verdict"})),
    ("Judge", fake("efficiency", {"verdict"})),
    ("ScoreProfile", fake("eval_metrics", {"alignment"})),
    ("Stage", fake("generate", requires={"LLMBackend": {"chat"}})),
    ("Stage", fake("compile_loop", requires={"Toolchain": {"emits_warnings"}})),
    ("Stage", fake("run_loop", requires={"Executor": {"runs_code"}})),
    ("Stage", fake("profile", requires={"Profiler": {"timing"}})),
    ("Stage", fake("report")),
)


class CapabilitiesOnly(NeverBuilt):
    """Declares capabilities only; requires and config_keys take their defaults."""

    name = "capabilities_only"
    capabilities = ("verdict",)


class ListDeclared(NeverBuilt):
    """Declares every attribute as a list or tuple; the registry stores frozensets."""

    name = "list_declared"
    capabilities = ["b_cap", "a_cap"]
    requires = {"Executor": ["runs_code"], "Toolchain": ("emits_warnings",)}
    config_keys = ["depth", "width"]


class NoCapabilities(NeverBuilt):
    """Lacks the required capabilities attribute."""

    name = "no_capabilities"
    config_keys = ("depth",)


class BadRequires(NeverBuilt):
    """Requires something from an interface that does not exist."""

    name = "bad_requires"
    capabilities = ()
    requires = {"Compiler": {"optimizes"}}


class StringCapabilities(NeverBuilt):
    """Declares capabilities as one bare string instead of a collection of strings."""

    name = "string_capabilities"
    capabilities = "abc"


class NumberCapability(NeverBuilt):
    """Declares a capability that is not a string."""

    name = "number_capability"
    capabilities = [1]


class ListRequires(NeverBuilt):
    """Declares requires as a list instead of a mapping of interface names to capabilities."""

    name = "list_requires"
    capabilities = ()
    requires = ["Executor"]


class ReverseOrderSet(frozenset):
    """A frozenset that iterates in reverse sorted order, so code that forgets to sort it is caught on every run."""

    def __iter__(self) -> Iterator[Any]:
        return iter(sorted(frozenset.__iter__(self), reverse=True))


# ---------------------------------------------------------------------------
# Bible parsing helpers


def bible_lines() -> list[str]:
    """Return the lines of docs/BIBLE.md."""
    return BIBLE.read_text(encoding="utf-8").splitlines()


def bible_interface_names() -> list[str]:
    """Return the first-column names of the Component Interfaces table in the bible, in table order."""
    lines = bible_lines()
    start = lines.index("## Component Interfaces")
    names: list[str] = []
    in_rows = False
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not in_rows:
            in_rows = stripped.startswith("| ---")
            continue
        if not stripped:
            break
        names.append(stripped.strip("|").split("|")[0].strip())
    return names


def bible_recipe_blocks() -> dict[str, str]:
    """Map each yaml block in the bible's Project Recipes section, keyed by its first comment line, to its text."""
    lines = bible_lines()
    start = lines.index("## Project Recipes")
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("## "))
    blocks: dict[str, str] = {}
    cursor = start
    while True:
        try:
            begin = lines.index("```yaml", cursor, end) + 1
        except ValueError:
            return blocks
        stop = lines.index("```", begin)
        body = lines[begin:stop]
        blocks[body[0].lstrip("#").strip()] = "\n".join(body) + "\n"
        cursor = stop + 1


def bible_block_data(key: str) -> dict[str, Any]:
    """Return one bible recipe block parsed as YAML."""
    data = yaml.safe_load(bible_recipe_blocks()[key])
    assert isinstance(data, dict), key
    return data


# ---------------------------------------------------------------------------
# General helpers


def canonical(data: Mapping[str, Any]) -> str:
    """Return the canonical YAML text of a resolved mapping (load_recipe step 9)."""
    return yaml.safe_dump(data, sort_keys=True, default_flow_style=False, allow_unicode=False, width=4096)


def with_defaults(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of a non-faithful resolved mapping with faithful and fixes materialized as the loader does."""
    out = copy.deepcopy(dict(data))
    out.setdefault("faithful", False)
    out.setdefault("fixes", {"fence_tag": True})
    return out


def write(directory: Path, name: str, text: str) -> Path:
    """Write text as UTF-8 with LF newlines to directory/name, creating directories, and return the path."""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


def over_child(directory: Path, snippet: str, name: str = "leaf.yaml") -> Path:
    """Write a recipe that extends the child fixture by name and adds the snippet; return its path."""
    return write(directory, name, f"extends: child\n{snippet}\n")


def lookup(data: Mapping[str, Any], dotted: str) -> Any:
    """Return the value at a dotted key path."""
    value: Any = data
    for part in dotted.split("."):
        value = value[part]
    return value


def delete_path(data: dict[str, Any], dotted: str) -> None:
    """Delete the key at a dotted key path."""
    *parents, last = dotted.split(".")
    target = data
    for part in parents:
        target = target[part]
    del target[last]


def mentions_path(message: str, path: Path) -> bool:
    """Return True when a message names a path in native, forward-slash, or escaped form."""
    native = str(path)
    return any(form in message for form in (native, path.as_posix(), native.replace("\\", "\\\\")))


def assert_required_message(message: str, file_name: str, dotted: str) -> None:
    """Assert a required-choice failure names the file and path, says there is no value, and points at extends."""
    lowered = message.lower()
    assert file_name in message, message
    assert dotted in message, message
    assert "required" in lowered, message
    assert "no value" in lowered, message
    assert "extend" in lowered, message


def nested(depth: int, kind: str) -> str:
    """Return a flow-style value that nests lists (kind "list") or mappings `depth` deep around the scalar 1."""
    if kind == "list":
        return "[" * depth + "1" + "]" * depth
    return "{a: " * depth + "1" + "}" * depth


def alias_chain(depth: int) -> str:
    """Return an executor section whose host list nests `depth` deep only through aliases to its earlier items."""
    lines = ["executor:", "  kind: native", "  host:", "  - &a0 [1]"]
    lines += [f"  - &a{index} [*a{index - 1}]" for index in range(1, depth)]
    return "\n".join(lines)


def alias_fan_out(levels: int) -> str:
    """Return an executor section whose host list holds `levels` anchored lists, each ten aliases to the one before.

    Item k expands to more than 10**(k + 1) values, while the text grows by only
    about 50 bytes per level (the billion laughs pattern).
    """
    lines = ["executor:", "  kind: native", "  host:", f"  - &a0 [{', '.join(['1'] * 10)}]"]
    lines += [f"  - &a{index} [{', '.join([f'*a{index - 1}'] * 10)}]" for index in range(1, levels)]
    return "\n".join(lines)


def binding_tuples(bindings: Iterable[Any]) -> list[tuple[str, str, str, dict[str, Any]]]:
    """Return (interface, name, where, config) for each binding."""
    return [(b.interface, b.name, b.where, dict(b.config)) for b in bindings]


def bound_components(data: Mapping[str, Any]) -> dict[tuple[str, str], set[str]]:
    """Return each (interface, name) the loader's binding rules bind in data, with each kind section's extra keys."""
    found: dict[tuple[str, str], set[str]] = {}

    def bind(interface: str, name: str, keys: Iterable[str] = ()) -> None:
        found.setdefault((interface, name), set()).update(keys)

    if "model" in data:
        bind("LLMBackend", data["model"]["backend"])
    for name in data.get("toolchain", {}).values():
        bind("Toolchain", name)
    for section, interface in KIND_SECTIONS.items():
        if section in data:
            bind(interface, data[section]["kind"], set(data[section]) - {"kind"})
    for name in data.get("stages", []):
        bind("Stage", name)
    if "score" in data:
        bind("ScoreProfile", data["score"])
    for role in data.get("agents", {}):
        bind("Agent", role)
    for judge in data.get("judges", {}):
        bind("Judge", judge)
    return found


def permissive_registry(registry_module: ModuleType, *datas: Mapping[str, Any]) -> Any:
    """Return a Registry holding a fake for every component the given recipe data binds, requiring nothing."""
    found: dict[tuple[str, str], set[str]] = {}
    for data in datas:
        for key, keys in bound_components(data).items():
            found.setdefault(key, set()).update(keys)
    registry = registry_module.Registry()
    for (interface, name), keys in sorted(found.items()):
        registry.register(interface, name, fake(name, config_keys=keys))
    return registry


def value_lines_without_comment(text: str) -> list[str]:
    """Return YAML lines that carry a value but no trailing comment; mapping openers carry no value."""
    missing: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        body, _, comment = line.partition(" #")
        if body.rstrip().endswith(":"):
            continue
        if not comment.strip():
            missing.append(line)
    return missing


def value_line_count(text: str) -> int:
    """Return how many YAML lines carry a value, ignoring comments, blank lines, and mapping openers."""
    count = 0
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.partition(" #")[0].rstrip().endswith(":"):
            count += 1
    return count


def flow_style_lines(text: str) -> list[str]:
    """Return YAML lines that use flow style (braces or brackets) outside comments."""
    return [line for line in text.splitlines() if any(ch in line.partition(" #")[0] for ch in "{[")]


def leaf_count(data: Any) -> int:
    """Return the number of non-mapping leaves in nested mappings."""
    if isinstance(data, dict):
        return sum(leaf_count(value) for value in data.values())
    return 1


def public_defs(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    """Return public top-level classes and functions, plus the public methods of public classes."""
    found: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, *FUNCTION_NODES)) or node.name.startswith("_"):
            continue
        found.append((node.name, node))
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, FUNCTION_NODES) and not item.name.startswith("_"):
                    found.append((f"{node.name}.{item.name}", item))
    return found


def is_typed(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True when every parameter except self or cls, and the return value, are annotated."""
    args = node.args
    params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    params += [a for a in (args.vararg, args.kwarg) if a is not None]
    params = [p for p in params if p.arg not in ("self", "cls")]
    return node.returns is not None and all(p.annotation is not None for p in params)


def projects_imports(tree: ast.Module) -> list[int]:
    """Return the line numbers of absolute imports of the projects package."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            lines += [node.lineno for alias in node.names if alias.name.split(".")[0] == "projects"]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").split(".")[0] == "projects":
            lines.append(node.lineno)
    return lines


def module_source(name: str) -> str:
    """Return the source text of an importable module, failing when it does not exist."""
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin, f"missing module {name}"
    raw = Path(spec.origin).read_bytes()
    assert raw.isascii(), f"{name} has non-ASCII source text"
    return raw.decode("ascii")


# ---------------------------------------------------------------------------
# Fixtures


@pytest.fixture(autouse=True)
def clear_constructed() -> Iterator[None]:
    """Start and end every test with no recorded construction."""
    CONSTRUCTED.clear()
    yield
    CONSTRUCTED.clear()


@pytest.fixture(scope="module")
def recipe_module() -> ModuleType:
    """Import lassi.core.recipe."""
    return importlib.import_module("lassi.core.recipe")


@pytest.fixture(scope="module")
def registry_module() -> ModuleType:
    """Import lassi.core.registry."""
    return importlib.import_module("lassi.core.registry")


@pytest.fixture
def fakes(registry_module: ModuleType) -> Any:
    """Return a fresh Registry holding every fake component in FAKES."""
    registry = registry_module.Registry()
    for interface, cls in FAKES:
        registry.register(interface, cls.name, cls)
    return registry


@pytest.fixture
def default_registry(registry_module: ModuleType, recipe_module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Put a fresh, empty Registry in place of DEFAULT_REGISTRY for one test, so no test leaves components behind."""
    assert isinstance(registry_module.DEFAULT_REGISTRY, registry_module.Registry)
    assert recipe_module.DEFAULT_REGISTRY is registry_module.DEFAULT_REGISTRY
    fresh = registry_module.Registry()
    monkeypatch.setattr(registry_module, "DEFAULT_REGISTRY", fresh)
    monkeypatch.setattr(recipe_module, "DEFAULT_REGISTRY", fresh)
    return fresh


@pytest.fixture
def load(recipe_module: ModuleType, fakes: Any) -> Callable[..., Any]:
    """Return a loader: a str names a file under FIXTURES, a Path is used as given; roots and registry default to
    the fixture directory and the fake registry."""

    def load_one(target: str | Path, **kwargs: Any) -> Any:
        path = FIXTURES / target if isinstance(target, str) else target
        kwargs.setdefault("roots", [FIXTURES])
        kwargs.setdefault("registry", fakes)
        return recipe_module.load_recipe(path, **kwargs)

    return load_one


@pytest.fixture
def load_error(load: Callable[..., Any], recipe_module: ModuleType) -> Callable[..., str]:
    """Return a function that expects loading to raise RecipeError with nothing constructed; it returns the message."""

    def fail(target: str | Path, **kwargs: Any) -> str:
        with pytest.raises(recipe_module.RecipeError) as info:
            load(target, **kwargs)
        assert CONSTRUCTED == [], f"constructed during load: {CONSTRUCTED}"
        return str(info.value)

    return fail


def make_binding(registry_module: ModuleType, interface: str, name: str, where: str, **config: Any) -> Any:
    """Return a Binding with the given config."""
    return registry_module.Binding(interface=interface, name=name, where=where, config=config)


def check_error(registry_module: ModuleType, registry: Any, bindings: list[Any]) -> str:
    """Return the message of the RegistryError that check_bindings raises for the bindings."""
    with pytest.raises(registry_module.RegistryError) as info:
        registry_module.check_bindings(bindings, registry)
    assert CONSTRUCTED == []
    return str(info.value)


# ---------------------------------------------------------------------------
# Guards on the test's own parsers and helpers


def test_bible_recipe_blocks_are_found() -> None:
    assert list(bible_recipe_blocks()) == list(BIBLE_BLOCK_KEYS)


def test_bible_names_twelve_interfaces() -> None:
    names = bible_interface_names()
    assert len(names) == 12 and len(set(names)) == 12, names


def test_comment_scanner_flags_uncommented_values() -> None:
    text = "# header\nllm:\n  sampling:\n    top_p: 0.9  # why\n    temperature: 0.2\nitems: [a]  # why\n"
    assert value_lines_without_comment(text) == ["    temperature: 0.2"]
    assert value_line_count(text) == 3
    assert flow_style_lines(text) == ["items: [a]  # why"]


def test_expected_child_hash_is_the_sha256_of_its_canonical_yaml() -> None:
    assert hashlib.sha256(canonical(EXPECTED_CHILD).encode("ascii")).hexdigest() == CHILD_HASH


# ---------------------------------------------------------------------------
# Registry


def test_interfaces_match_bible_and_protocols(registry_module: ModuleType) -> None:
    interfaces = importlib.import_module("lassi.core.interfaces")
    assert isinstance(registry_module.INTERFACES, tuple)
    assert registry_module.INTERFACES == tuple(bible_interface_names())
    for name in registry_module.INTERFACES:
        assert getattr(getattr(interfaces, name), "_is_protocol", False), name


def test_errors_are_value_errors(registry_module: ModuleType, recipe_module: ModuleType) -> None:
    assert issubclass(registry_module.RegistryError, ValueError)
    assert issubclass(recipe_module.RecipeError, ValueError)


@pytest.mark.parametrize(
    ("module", "cls_name", "fields"),
    [
        pytest.param(
            "lassi.core.registry",
            "Entry",
            ["interface", "name", "factory", "capabilities", "requires", "config_keys"],
            id="entry",
        ),
        pytest.param("lassi.core.registry", "Binding", ["interface", "name", "where", "config"], id="binding"),
        pytest.param(
            "lassi.core.recipe",
            "Recipe",
            ["name", "path", "chain", "data", "canonical_yaml", "recipe_hash", "bindings"],
            id="recipe",
        ),
    ],
)
def test_records_are_frozen_dataclasses(module: str, cls_name: str, fields: list[str]) -> None:
    cls = getattr(importlib.import_module(module), cls_name)
    assert dataclasses.is_dataclass(cls)
    assert cls.__dataclass_params__.frozen, f"{cls_name} is not frozen"
    assert [f.name for f in dataclasses.fields(cls)] == fields


def test_register_reads_class_attributes_without_constructing(registry_module: ModuleType) -> None:
    registry = registry_module.Registry()
    entry = registry.register("Stage", "list_declared", ListDeclared)
    assert isinstance(entry, registry_module.Entry)
    assert (entry.interface, entry.name, entry.factory) == ("Stage", "list_declared", ListDeclared)
    assert isinstance(entry.capabilities, frozenset)
    assert entry.capabilities == frozenset({"a_cap", "b_cap"})
    assert dict(entry.requires) == {"Executor": frozenset({"runs_code"}), "Toolchain": frozenset({"emits_warnings"})}
    assert all(isinstance(caps, frozenset) for caps in entry.requires.values())
    assert isinstance(entry.config_keys, frozenset)
    assert entry.config_keys == frozenset({"depth", "width"})
    assert registry.get("Stage", "list_declared") == entry
    assert CONSTRUCTED == []


def test_register_defaults_requires_and_config_keys(registry_module: ModuleType) -> None:
    entry = registry_module.Registry().register("Judge", "capabilities_only", CapabilitiesOnly)
    assert entry.capabilities == frozenset({"verdict"})
    assert dict(entry.requires) == {}
    assert entry.config_keys == frozenset()


@pytest.mark.parametrize(
    ("interface", "cls", "needle"),
    [
        pytest.param("Compiler", CapabilitiesOnly, "Compiler", id="unknown-interface"),
        pytest.param("Stage", NoCapabilities, "capabilities", id="missing-capabilities"),
        pytest.param("Stage", BadRequires, "Compiler", id="bad-requires-key"),
        pytest.param("Stage", StringCapabilities, "capabilities", id="capabilities-bare-string"),
        pytest.param("Stage", NumberCapability, "capabilities", id="capability-not-a-string"),
        pytest.param("Stage", ListRequires, "requires", id="requires-not-a-mapping"),
    ],
)
def test_register_rejects_bad_declarations(registry_module: ModuleType, interface: str, cls: type, needle: str) -> None:
    registry = registry_module.Registry()
    with pytest.raises(registry_module.RegistryError) as info:
        registry.register(interface, "component", cls)
    assert needle in str(info.value)
    assert registry.names("Stage") == []
    assert CONSTRUCTED == []


@pytest.mark.parametrize(
    ("name", "factory", "needle"),
    [
        pytest.param(5, CapabilitiesOnly, "name", id="name-not-a-string"),
        pytest.param("", CapabilitiesOnly, "name", id="name-empty"),
        pytest.param("instance", SimpleNamespace(capabilities=("verdict",)), "class", id="factory-not-a-class"),
    ],
)
def test_register_rejects_bad_names_and_factories(
    registry_module: ModuleType, name: Any, factory: Any, needle: str
) -> None:
    # A name that is not a string would later break the sorted names() list and every get() error message.
    registry = registry_module.Registry()
    with pytest.raises(registry_module.RegistryError) as info:
        registry.register("Stage", name, factory)
    assert needle in str(info.value), str(info.value)
    assert registry.names("Stage") == []


def foreign_factories(registry_module: ModuleType) -> list[str]:
    """Return 'interface name module' for each DEFAULT_REGISTRY factory defined outside the lassi package."""
    registry = registry_module.DEFAULT_REGISTRY
    found: list[str] = []
    for interface in registry_module.INTERFACES:
        for name in registry.names(interface):
            module = registry.get(interface, name).factory.__module__
            if not module.startswith("lassi."):
                found.append(f"{interface} {name} {module}")
    return found


def test_default_registry_holds_only_package_components(registry_module: ModuleType) -> None:
    # No test fake leaks into the default registry: every factory comes from a module under lassi., whether or not
    # lassi.llm was imported first. It reads the real DEFAULT_REGISTRY, never the default_registry fixture's stand-in.
    assert foreign_factories(registry_module) == []
    importlib.import_module("lassi.llm")
    assert {"mock", "ollama", "openai_compat"} <= set(registry_module.DEFAULT_REGISTRY.names("LLMBackend"))
    assert foreign_factories(registry_module) == []


def test_register_rejects_duplicate_but_allows_same_name_elsewhere(registry_module: ModuleType) -> None:
    registry = registry_module.Registry()
    registry.register("Judge", "equivalence", CapabilitiesOnly)
    with pytest.raises(registry_module.RegistryError) as info:
        registry.register("Judge", "equivalence", CapabilitiesOnly)
    assert "Judge" in str(info.value) and "equivalence" in str(info.value)
    registry.register("Agent", "equivalence", CapabilitiesOnly)
    assert registry.names("Judge") == ["equivalence"]
    assert registry.names("Agent") == ["equivalence"]


def test_get_unknown_name_lists_sorted_registered_names(registry_module: ModuleType) -> None:
    registry = registry_module.Registry()
    registry.register("Oracle", "zeta_oracle", CapabilitiesOnly)
    registry.register("Oracle", "alpha_oracle", CapabilitiesOnly)
    with pytest.raises(registry_module.RegistryError) as info:
        registry.get("Oracle", "missing_oracle")
    message = str(info.value)
    assert "Oracle" in message and "missing_oracle" in message
    assert "alpha_oracle" in message and "zeta_oracle" in message
    assert message.index("alpha_oracle") < message.index("zeta_oracle")


def test_get_on_empty_interface_says_none_registered(registry_module: ModuleType) -> None:
    with pytest.raises(registry_module.RegistryError) as info:
        registry_module.Registry().get("Profiler", "timer")
    message = str(info.value)
    assert "Profiler" in message and "timer" in message and "none registered" in message


def test_names_are_sorted(registry_module: ModuleType) -> None:
    registry = registry_module.Registry()
    for name in ("zeta", "alpha", "mid"):
        registry.register("Stage", name, fake(name))
    assert registry.names("Stage") == ["alpha", "mid", "zeta"]
    assert registry.names("Judge") == []


def test_register_decorator_uses_default_registry(registry_module: ModuleType, default_registry: Any) -> None:
    cls = fake("decorator_probe", {"probe"})
    assert registry_module.register("ScoreProfile", "decorator_probe")(cls) is cls
    assert default_registry.get("ScoreProfile", "decorator_probe").factory is cls
    assert default_registry.names("ScoreProfile") == ["decorator_probe"]
    with pytest.raises(registry_module.RegistryError):
        registry_module.register("ScoreProfile", "decorator_probe")(cls)
    assert CONSTRUCTED == []


def test_check_bindings_accepts_satisfied_requirements(registry_module: ModuleType, fakes: Any) -> None:
    bindings = [
        make_binding(registry_module, "LLMBackend", "mock_llm", "model.backend"),
        make_binding(registry_module, "Toolchain", "fake_cc", "toolchain.cuda"),
        make_binding(registry_module, "Toolchain", "fake_cxx", "toolchain.omp"),
        make_binding(registry_module, "Executor", "native", "executor.kind", host="box1"),
        make_binding(registry_module, "Stage", "generate", "stages[0]"),
        make_binding(registry_module, "Stage", "compile_loop", "stages[1]"),
        make_binding(registry_module, "Stage", "run_loop", "stages[2]"),
    ]
    assert registry_module.check_bindings(bindings, fakes) is None
    assert CONSTRUCTED == []


def test_check_bindings_unregistered_component(registry_module: ModuleType, fakes: Any) -> None:
    message = check_error(
        registry_module, fakes, [make_binding(registry_module, "Oracle", "no_such_oracle", "oracle.kind")]
    )
    for part in ("oracle.kind", "Oracle", "no_such_oracle", "stdout_mask"):
        assert part in message, message


def test_check_bindings_unknown_config_key(registry_module: ModuleType, fakes: Any) -> None:
    binding = make_binding(registry_module, "Executor", "native", "executor.kind", hots="box1")
    message = check_error(registry_module, fakes, [binding])
    # The key's path is its real recipe path, beside kind: executor.hots, not executor.kind.hots.
    assert "executor.hots" in message and "executor.kind.hots" not in message, message
    for part in ("Executor", "native", "host"):
        assert part in message, message
    keyless = make_binding(registry_module, "Executor", "none", "executor.kind", host="box1")
    message = check_error(registry_module, fakes, [keyless])
    assert "executor.host" in message and "accepts no config keys" in message, message


@pytest.mark.parametrize(
    ("interface", "name", "where", "key_path"),
    [
        pytest.param("Executor", "native", "executor.kind", "executor.x", id="kind-section"),
        pytest.param("Oracle", "stdout_mask", "oracle.kind", "oracle.x", id="other-kind-section"),
        pytest.param("Agent", "generator", "agents.generator", "agents.generator.x", id="agent-role"),
        pytest.param("Toolchain", "fake_cc", "toolchain.cuda", "toolchain.cuda.x", id="toolchain-key"),
        pytest.param("Stage", "run_loop", "stages[3]", "stages[3].x", id="list-item"),
    ],
)
def test_check_bindings_names_the_config_key_path(
    registry_module: ModuleType, fakes: Any, interface: str, name: str, where: str, key_path: str
) -> None:
    # A key in a kind section sits beside kind; any other where keeps its last part (a role, a toolchain key).
    binding = make_binding(registry_module, interface, name, where, x=1)
    message = check_error(registry_module, fakes, [binding])
    assert message.startswith(f"{key_path}: "), message
    assert f"({where})" in message, message


def test_check_bindings_requirement_on_unbound_interface(registry_module: ModuleType, fakes: Any) -> None:
    bindings = [
        make_binding(registry_module, "Executor", "native", "executor.kind"),
        make_binding(registry_module, "Stage", "profile", "stages[1]"),
    ]
    message = check_error(registry_module, fakes, bindings)
    for part in ("profile", "stages[1]", "Profiler", "timing", "binds no Profiler"):
        assert part in message, message


def test_check_bindings_requirement_of_any_component(registry_module: ModuleType, fakes: Any) -> None:
    fakes.register("Stage", "judge_all", fake("judge_all", requires={"Judge": ()}))
    message = check_error(registry_module, fakes, [make_binding(registry_module, "Stage", "judge_all", "stages[0]")])
    assert "requires the Judge interface" in message and "binds no Judge" in message, message


def test_check_bindings_capability_mismatch_names_both(registry_module: ModuleType, fakes: Any) -> None:
    bindings = [
        make_binding(registry_module, "Executor", "none", "executor.kind"),
        make_binding(registry_module, "Stage", "run_loop", "stages[5]"),
    ]
    message = check_error(registry_module, fakes, bindings)
    for part in ("run_loop", "stages[5]", "runs_code", "Executor", "none", "executor.kind", "compile_only"):
        assert part in message, message


@pytest.mark.parametrize(
    ("cuda", "omp", "lacking_where"),
    [
        pytest.param("fake_cc", "silent_cc", "toolchain.omp", id="lacking-last"),
        pytest.param("silent_cc", "fake_cc", "toolchain.cuda", id="lacking-first"),
    ],
)
def test_check_bindings_checks_every_bound_component(
    registry_module: ModuleType, fakes: Any, cuda: str, omp: str, lacking_where: str
) -> None:
    bindings = [
        make_binding(registry_module, "Toolchain", cuda, "toolchain.cuda"),
        make_binding(registry_module, "Toolchain", omp, "toolchain.omp"),
        make_binding(registry_module, "Stage", "compile_loop", "stages[0]"),
    ]
    message = check_error(registry_module, fakes, bindings)
    for part in ("compile_loop", "stages[0]", "emits_warnings", "'silent_cc'", lacking_where, "nothing"):
        assert part in message, message
    assert "'fake_cc'" not in message, message


def test_check_bindings_lists_declared_capabilities_sorted(
    registry_module: ModuleType, fakes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes.register("Stage", "vectorize", fake("vectorize", requires={"Toolchain": {"autovec"}}))
    # fake_cxx iterates openmp first, so only code that sorts lists emits_warnings first, whatever the hash seed.
    reordered = dataclasses.replace(
        fakes.get("Toolchain", "fake_cxx"), capabilities=ReverseOrderSet({"emits_warnings", "openmp"})
    )
    registered_get = fakes.get

    def get(interface: str, name: str) -> Any:
        return reordered if (interface, name) == ("Toolchain", "fake_cxx") else registered_get(interface, name)

    monkeypatch.setattr(fakes, "get", get)
    bindings = [
        make_binding(registry_module, "Toolchain", "fake_cxx", "toolchain.omp"),
        make_binding(registry_module, "Stage", "vectorize", "stages[0]"),
    ]
    message = check_error(registry_module, fakes, bindings)
    assert "autovec" in message and "fake_cxx" in message, message
    assert "emits_warnings, openmp" in message, message


def test_check_bindings_reports_the_first_missing_capability(
    registry_module: ModuleType, fakes: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    needed = ("alpha_cap", "beta_cap", "gamma_cap", "zeta_cap")
    entry = fakes.register("Stage", "power_loop", fake("power_loop", requires={"Executor": needed}))
    # The requirement iterates zeta_cap first, so only code that sorts it reports alpha_cap, whatever the hash seed.
    reordered = dataclasses.replace(entry, requires={"Executor": ReverseOrderSet(needed)})
    registered_get = fakes.get

    def get(interface: str, name: str) -> Any:
        return reordered if (interface, name) == ("Stage", "power_loop") else registered_get(interface, name)

    monkeypatch.setattr(fakes, "get", get)
    bindings = [
        make_binding(registry_module, "Executor", "native", "executor.kind"),
        make_binding(registry_module, "Stage", "power_loop", "stages[0]"),
    ]
    message = check_error(registry_module, fakes, bindings)
    assert "alpha_cap" in message and "power_loop" in message and "native" in message, message
    for later in needed[1:]:
        assert later not in message, message


def test_check_bindings_checks_agent_requirements(registry_module: ModuleType, fakes: Any) -> None:
    message = check_error(
        registry_module, fakes, [make_binding(registry_module, "Agent", "generator", "agents.generator")]
    )
    for part in ("generator", "agents.generator", "LLMBackend", "chat"):
        assert part in message, message


def test_check_bindings_runs_the_checks_in_documented_order(registry_module: ModuleType, fakes: Any) -> None:
    unregistered_last = [
        make_binding(registry_module, "Executor", "native", "executor.kind", hots="box1"),
        make_binding(registry_module, "Stage", "no_such_stage", "stages[0]"),
    ]
    assert "no_such_stage" in check_error(registry_module, fakes, unregistered_last)
    config_last = [
        make_binding(registry_module, "Stage", "run_loop", "stages[0]"),
        make_binding(registry_module, "Executor", "none", "executor.kind", hots="box1"),
    ]
    assert "hots" in check_error(registry_module, fakes, config_last)
    two_unregistered = [
        make_binding(registry_module, "Stage", "first_missing", "stages[0]"),
        make_binding(registry_module, "Stage", "second_missing", "stages[1]"),
    ]
    assert "first_missing" in check_error(registry_module, fakes, two_unregistered)
    two_config_keys = [
        make_binding(registry_module, "Executor", "native", "executor.kind", hots="box1"),
        make_binding(registry_module, "Oracle", "stdout_mask", "oracle.kind", passfial=True),
    ]
    message = check_error(registry_module, fakes, two_config_keys)
    assert "executor.hots" in message and "passfial" not in message, message
    two_requirements = [
        make_binding(registry_module, "Executor", "none", "executor.kind"),
        make_binding(registry_module, "Stage", "run_loop", "stages[0]"),
        make_binding(registry_module, "Stage", "profile", "stages[1]"),
    ]
    message = check_error(registry_module, fakes, two_requirements)
    assert "stages[0]" in message and "stages[1]" not in message, message


# ---------------------------------------------------------------------------
# Extends chains and merging


def test_chain_resolves_with_child_winning(load: Callable[..., Any], recipe_module: ModuleType) -> None:
    recipe = load("child.yaml")
    assert isinstance(recipe, recipe_module.Recipe)
    assert isinstance(recipe.data, dict)
    assert recipe.data == EXPECTED_CHILD
    assert recipe.name == "child"
    assert recipe.chain == CHILD_CHAIN
    assert recipe.path.resolve() == (FIXTURES / "child.yaml").resolve()
    assert "extends" not in recipe.data
    assert CONSTRUCTED == []


MERGE_CASES = [
    pytest.param("llm.sampling.temperature", 0.2, id="nested-mapping-keeps-root-leaf"),
    pytest.param("llm.sampling.top_p", 0.95, id="child-leaf-wins"),
    pytest.param("llm.sampling.max_tokens", 2048, id="child-adds-leaf"),
    pytest.param("sandbox.mem_gb", 32, id="parent-leaf-wins-over-root"),
    pytest.param("sandbox.network", False, id="root-leaf-kept"),
    pytest.param("sandbox.wall_s", 120, id="scalar-of-new-type-replaces"),
    pytest.param("loop.max_corrections", 3, id="parent-scalar-kept"),
    pytest.param("directions", [{"source": "omp", "target": "cuda"}], id="list-of-mappings-replaced"),
    pytest.param("stages", ["generate", "compile_loop", "run_loop"], id="list-replaced"),
    pytest.param("context", [], id="empty-list-replaces"),
    pytest.param("oracle", {"kind": "stdout_mask", "passfail": False}, id="kind-section-merges"),
    pytest.param("toolchain", {"cuda": "fake_cc", "omp": "fake_cxx"}, id="parent-only-section"),
    pytest.param("runs_root", "fixture-runs", id="root-only-scalar"),
]


@pytest.mark.parametrize(("dotted", "expected"), MERGE_CASES)
def test_merge_rules(load: Callable[..., Any], dotted: str, expected: Any) -> None:
    assert lookup(load("child.yaml").data, dotted) == expected


def test_a_new_kind_replaces_the_inherited_kind_section(load: Callable[..., Any]) -> None:
    # The bible's tier-1 example swaps executor {kind: gpu, host: a100} for {kind: none}: the old host key belongs
    # to the old component, so it must not carry over and fail the new component's config check.
    recipe = load("kind-switch.yaml")
    assert recipe.data["executor"] == {"kind": "none"}
    executor = [binding for binding in recipe.bindings if binding.interface == "Executor"]
    assert [(binding.name, dict(binding.config)) for binding in executor] == [("none", {})]
    assert CONSTRUCTED == []


def test_the_same_kind_keeps_merging_its_section(load: Callable[..., Any]) -> None:
    assert load("kind-keep.yaml").data["executor"] == {"kind": "native", "host": "box1"}


def test_parent_loads_on_its_own(load: Callable[..., Any]) -> None:
    recipe = load("parent.yaml")
    assert recipe.chain == ("fixture-base", "parent")
    assert recipe.data["stages"] == ["generate", "compile_loop", "run_loop", "report"]
    assert recipe.data["directions"] == [{"source": "omp", "target": "cuda"}, {"source": "cuda", "target": "omp"}]


def test_relative_path_extends(load: Callable[..., Any]) -> None:
    recipe = load("sub/relative-child.yaml")
    assert recipe.name == "relative-child"
    assert recipe.chain == ("fixture-base", "parent", "relative-child")
    assert recipe.data["trials"] == {"n": 4}
    assert recipe.data["sandbox"] == {"network": False, "wall_s": "baseline_x10", "mem_gb": 32}


def test_name_lookup_finds_a_directory_recipe(load: Callable[..., Any]) -> None:
    recipe = load("dir-child.yaml")
    assert recipe.chain == ("fixture-base", "parent", "dir-parent", "dir-child")
    assert recipe.data["trials"] == {"n": 7}
    assert recipe.data["sandbox"]["mem_gb"] == 8


def test_directory_recipe_is_named_after_its_directory(load: Callable[..., Any]) -> None:
    recipe = load("dir-parent/recipe.yaml")
    assert recipe.name == "dir-parent"
    assert recipe.chain == ("fixture-base", "parent", "dir-parent")


def test_directory_recipe_reached_through_dotdot_keeps_its_name(
    load: Callable[..., Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The name is the directory's, however the path spells it; never the literal "..".
    project = tmp_path / "proj"
    write(project, "recipe.yaml", "extends: child\n")
    variant = write(project, "variants/fast.yaml", "extends: ../recipe.yaml\n")
    assert load(variant).chain == (*CHILD_CHAIN, "proj", "fast")
    assert load(project / "variants" / ".." / "recipe.yaml").name == "proj"
    monkeypatch.chdir(project / "variants")
    assert load(Path("../recipe.yaml")).name == "proj"


@pytest.mark.parametrize("value", [".", ".."])
def test_extends_dot_names_are_refused(load_error: Callable[..., str], tmp_path: Path, value: str) -> None:
    # As names these would find <root>/./recipe.yaml and <root>/../recipe.yaml, the second outside every root.
    root = tmp_path / "root"
    write(root, "recipe.yaml", "extends: child\n")
    write(tmp_path, "recipe.yaml", "extends: child\n")
    leaf = write(tmp_path / "elsewhere", "leaf.yaml", f"extends: '{value}'\n")
    message = load_error(leaf, roots=[root, FIXTURES])
    assert "leaf.yaml" in message and f"extends {value!r}" in message and "recipe name" in message, message


def test_name_lookup_order(load: Callable[..., Any], tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    write(first, "shared/recipe.yaml", "extends: child\ntrials: {n: 11}\n")
    write(first, "shared.yaml", "extends: child\ntrials: {n: 12}\n")
    write(second, "shared.yaml", "extends: child\ntrials: {n: 13}\n")
    leaf = write(tmp_path, "leaf.yaml", "extends: shared\n")
    roots = [first, second, FIXTURES]
    recipe = load(leaf, roots=roots)
    assert recipe.data["trials"] == {"n": 12}
    assert recipe.chain == (*CHILD_CHAIN, "shared", "leaf")
    (first / "shared.yaml").unlink()
    assert load(leaf, roots=roots).data["trials"] == {"n": 11}
    shutil.rmtree(first / "shared")
    assert load(leaf, roots=roots).data["trials"] == {"n": 13}


def test_names_are_looked_up_in_roots_not_beside_the_file(load: Callable[..., Any], tmp_path: Path) -> None:
    write(tmp_path, "child.yaml", "extends: child\ntrials: {n: 21}\n")
    recipe = load(write(tmp_path, "leaf.yaml", "extends: child\n"))
    assert recipe.data["trials"] == {"n": 2}
    assert recipe.chain == (*CHILD_CHAIN, "leaf")


def test_a_name_beside_the_file_but_not_in_roots_is_missing(load_error: Callable[..., str], tmp_path: Path) -> None:
    write(tmp_path, "neighbor.yaml", "extends: child\n")
    message = load_error(write(tmp_path, "leaf.yaml", "extends: neighbor\n"))
    assert "neighbor" in message
    assert mentions_path(message, FIXTURES / "neighbor.yaml"), message


@pytest.mark.parametrize("spelling", ["parent.yaml", "parent.yml", "./parent.yaml", "nested/parent.yaml"])
def test_paths_resolve_relative_to_the_extending_file(load: Callable[..., Any], tmp_path: Path, spelling: str) -> None:
    write(tmp_path, spelling, "extends: child\ntrials: {n: 31}\n")
    recipe = load(write(tmp_path, "leaf.yaml", f"extends: {spelling}\n"))
    assert recipe.data["trials"] == {"n": 31}
    assert recipe.chain[: len(CHILD_CHAIN)] == CHILD_CHAIN


def test_missing_parent_lists_the_paths_tried(load_error: Callable[..., str]) -> None:
    message = load_error("missing-parent.yaml")
    assert "missing-parent.yaml" in message
    assert mentions_path(message, FIXTURES / "no-such-recipe.yaml"), message
    assert mentions_path(message, FIXTURES / "no-such-recipe" / "recipe.yaml"), message


def test_missing_parent_lists_every_root(load_error: Callable[..., str], tmp_path: Path) -> None:
    roots = [tmp_path / "one", tmp_path / "two"]
    for root in roots:
        root.mkdir()
    message = load_error("missing-parent.yaml", roots=roots)
    for root in roots:
        assert mentions_path(message, root / "no-such-recipe.yaml"), message
        assert mentions_path(message, root / "no-such-recipe" / "recipe.yaml"), message


def test_missing_relative_parent(load_error: Callable[..., str], tmp_path: Path) -> None:
    message = load_error(write(tmp_path, "leaf.yaml", "extends: ../nowhere/gone.yaml\n"))
    assert "leaf.yaml" in message
    assert mentions_path(message, tmp_path / "../nowhere/gone.yaml"), message


def test_a_name_with_no_roots_to_search_is_missing(load_error: Callable[..., str], tmp_path: Path) -> None:
    message = load_error(write(tmp_path, "leaf.yaml", "extends: child\n"), roots=[])
    assert "leaf.yaml" in message and "child" in message and "no roots" in message, message


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("CHILD", id="name-case"),
        pytest.param("DIR-PARENT", id="directory-name-case"),
        pytest.param("./PARENT.yaml", id="path-case"),
        pytest.param("NESTED/parent.yaml", id="directory-case"),
        pytest.param("./nowhere/../parent.yaml", id="missing-directory-before-dotdot"),
    ],
)
def test_extends_must_match_the_disk_exactly(load_error: Callable[..., str], tmp_path: Path, value: str) -> None:
    # Windows and macOS would find these files anyway; Linux, where runs execute, would not.
    write(tmp_path, "parent.yaml", "extends: child\n")
    write(tmp_path, "nested/parent.yaml", "extends: child\n")
    message = load_error(write(tmp_path, "leaf.yaml", f"extends: {value}\n"))
    assert "leaf.yaml" in message and value in message and "no recipe file was found" in message, message


def test_extends_may_step_through_an_existing_directory(load: Callable[..., Any], tmp_path: Path) -> None:
    write(tmp_path, "nested/parent.yaml", "extends: child\ntrials: {n: 32}\n")
    write(tmp_path, "parent.yaml", "extends: child\ntrials: {n: 33}\n")
    recipe = load(write(tmp_path, "leaf.yaml", "extends: ./nested/../parent.yaml\n"))
    assert recipe.data["trials"] == {"n": 33}
    assert recipe.chain == (*CHILD_CHAIN, "parent", "leaf")


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("'sub\\relative-child.yaml'", id="backslash-path"),
        pytest.param("'sub\\relative-child'", id="backslash-name"),
        pytest.param("ABSOLUTE", id="absolute-path"),
    ],
)
def test_extends_is_a_portable_relative_spelling(load_error: Callable[..., str], tmp_path: Path, value: str) -> None:
    if value == "ABSOLUTE":
        value = (FIXTURES / "child.yaml").as_posix()
    message = load_error(write(tmp_path, "leaf.yaml", f"extends: {value}\n"))
    assert "leaf.yaml" in message and "extends" in message and "relative path" in message, message


def test_cycle_fails_naming_the_chain(load_error: Callable[..., str]) -> None:
    message = load_error("cycle-a.yaml")
    assert "cycle-a -> cycle-b -> cycle-a" in message, message
    # The file named is the one whose extends closes the loop.
    assert mentions_path(message.split(": extends")[0], FIXTURES / "cycle-b.yaml"), message


def test_self_extends_is_a_cycle(load_error: Callable[..., str], tmp_path: Path) -> None:
    path = write(tmp_path, "loop-self.yaml", "extends: ./loop-self.yaml\n")
    message = load_error(path)
    assert "loop-self -> loop-self" in message, message
    assert mentions_path(message.split(": extends")[0], path), message


def test_same_name_in_two_places_is_not_a_cycle(load: Callable[..., Any], tmp_path: Path) -> None:
    recipe = load(write(tmp_path / "other", "child.yaml", "extends: child\ntrials: {n: 41}\n"))
    assert recipe.chain == (*CHILD_CHAIN, "child")
    assert recipe.data["trials"] == {"n": 41}


@pytest.mark.parametrize("value", ["5", "[parent]", "{name: parent}", "true"])
def test_extends_must_be_a_string(load_error: Callable[..., str], tmp_path: Path, value: str) -> None:
    message = load_error(write(tmp_path, "leaf.yaml", f"extends: {value}\n"))
    assert "leaf.yaml" in message and "extends" in message


# ---------------------------------------------------------------------------
# Parsing


def test_duplicate_key_fixture_fails(load_error: Callable[..., str]) -> None:
    message = load_error("duplicate-key.yaml")
    assert "duplicate-key.yaml" in message and "mem_gb" in message


@pytest.mark.parametrize(
    ("text", "key"),
    [
        pytest.param("extends: child\ntrials: {n: 1}\ntrials: {n: 2}\n", "trials", id="top-level"),
        pytest.param("extends: child\nextends: parent\n", "extends", id="extends"),
        pytest.param(
            "extends: child\nagents:\n  fixer: {model: a, model: b}\n", "agents.fixer.model", id="inside-open-map"
        ),
        pytest.param(
            "extends: child\ndirections:\n  - {source: omp, source: c, target: d}\n",
            "directions[0].source",
            id="in-list",
        ),
        pytest.param(
            "extends: child\noracle:\n  <<: {kind: none_such}\n  <<: {kind: stdout_mask}\n", "oracle.<<", id="merge-key"
        ),
        pytest.param("extends: child\n<<: {faithful: true}\n<<: {faithful: false}\n", "<<", id="top-level-merge-key"),
    ],
)
def test_duplicate_keys_anywhere_fail(load_error: Callable[..., str], tmp_path: Path, text: str, key: str) -> None:
    message = load_error(write(tmp_path, "dup.yaml", text))
    assert "dup.yaml" in message and f"duplicate key {key}" in message, message


def test_one_merge_key_merges_as_yaml_says(load: Callable[..., Any], tmp_path: Path) -> None:
    # In a list of merged mappings the earlier one wins; a key written in the mapping itself beats a merged one.
    text = "extends: child\nprofiler:\n  <<: [{interval_ms: 5}, {kind: timer, interval_ms: 9}]\n"
    assert load(write(tmp_path, "merged.yaml", text)).data["profiler"] == {"kind": "timer", "interval_ms": 5}
    text = "extends: child\nprofiler:\n  <<: {kind: timer, interval_ms: 9}\n  interval_ms: 7\n"
    assert load(write(tmp_path, "merged.yaml", text)).data["profiler"] == {"kind": "timer", "interval_ms": 7}


def test_the_equals_key_reads_as_a_plain_string(load_error: Callable[..., str], tmp_path: Path) -> None:
    # yaml.safe_load reads the plain key = as the string "="; the strict parse must agree, so it is an unknown key.
    message = load_error(over_child(tmp_path, "=: 1", "equals.yaml"))
    assert "equals.yaml" in message and "unknown key =" in message and "not valid YAML" not in message, message


@pytest.mark.parametrize(
    "snippet",
    [
        pytest.param("prompts: !!python/object/apply:os.getcwd []", id="object-apply"),
        pytest.param("prompts: !!python/name:os.getcwd ''", id="name"),
    ],
)
def test_python_tags_are_refused(load_error: Callable[..., str], tmp_path: Path, snippet: str) -> None:
    # Only the safe loader refuses these tags; an unsafe one would call os.getcwd or load a function.
    message = load_error(over_child(tmp_path, snippet, "unsafe.yaml"))
    assert "unsafe.yaml" in message and "python/" in message and "not valid YAML" in message, message


DEEP_VALUES = [
    pytest.param(f"executor: {{kind: native, host: {nested(100, 'list')}}}", id="lists-100"),
    pytest.param(f"executor: {{kind: native, host: {nested(400, 'list')}}}", id="lists-400"),
    pytest.param(f"executor: {{kind: native, host: {nested(1000, 'list')}}}", id="lists-1000"),
    pytest.param(f"executor: {{kind: native, host: {nested(400, 'map')}}}", id="mappings-400"),
    pytest.param(alias_chain(400), id="aliases-400"),
]


@pytest.mark.parametrize("snippet", DEEP_VALUES)
def test_deep_nesting_fails_as_a_recipe_error(load_error: Callable[..., str], tmp_path: Path, snippet: str) -> None:
    message = load_error(over_child(tmp_path, snippet, "deep.yaml"))
    assert "deep.yaml" in message and "nest" in message, message


def test_alias_fan_out_fails_as_a_recipe_error(load_error: Callable[..., str], tmp_path: Path) -> None:
    # About 300 bytes whose aliases expand to over 100000 values; unchecked, each level costs ten times more.
    message = load_error(over_child(tmp_path, alias_fan_out(5), "fanout.yaml"))
    assert "fanout.yaml" in message and "executor.host" in message and "100000 values" in message, message


def test_moderate_alias_fan_out_loads_without_aliases(load: Callable[..., Any], tmp_path: Path) -> None:
    recipe = load(over_child(tmp_path, alias_fan_out(3), "fanout.yaml"))
    assert recipe.data["executor"]["host"][2] == [[[1] * 10] * 10] * 10
    assert "&" not in recipe.canonical_yaml and "*" not in recipe.canonical_yaml


def test_moderate_nesting_loads(load: Callable[..., Any], tmp_path: Path) -> None:
    recipe = load(over_child(tmp_path, f"executor: {{kind: native, host: {nested(60, 'list')}}}", "nested.yaml"))
    value = recipe.data["executor"]["host"]
    for _ in range(60):
        value = value[0]
    assert value == 1


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param("# only a comment\n", id="comment-only"),
        pytest.param("null\n", id="null-document"),
        pytest.param("- child\n- parent\n", id="top-level-list"),
        pytest.param("just text\n", id="top-level-scalar"),
        pytest.param("stages: [generate\n", id="unclosed-flow"),
        pytest.param("a: b: c\n", id="syntax-error"),
        pytest.param("trials: {n: !!int abc}\n", id="bad-int-tag"),
        pytest.param("runs_root: !!int ''\n", id="empty-int-tag"),
        pytest.param("faithful: !!bool maybe\n", id="bad-bool-tag"),
        pytest.param("runs_root: !!timestamp nope\n", id="bad-timestamp-tag"),
        pytest.param("!!int abc: 1\n", id="bad-key-tag"),
    ],
)
def test_malformed_files_fail(load_error: Callable[..., str], tmp_path: Path, text: str) -> None:
    broken = write(tmp_path, "broken.yaml", text)
    assert "broken.yaml" in load_error(broken)
    complete = {"extends": "broken.yaml", **STANDALONE}
    assert "broken.yaml" in load_error(write(tmp_path, "leaf.yaml", yaml.safe_dump(complete)))


def test_missing_file_fails(load_error: Callable[..., str], tmp_path: Path) -> None:
    assert "absent.yaml" in load_error(tmp_path / "absent.yaml")


def test_a_path_no_file_system_can_hold_fails_as_a_recipe_error(load_error: Callable[..., str], tmp_path: Path) -> None:
    # The OS refuses a NUL character in a path with ValueError, not OSError.
    message = load_error(tmp_path / "bad\x00name.yaml")
    assert "name.yaml" in message and "cannot read" in message, message


def test_non_utf8_file_fails(load_error: Callable[..., str], tmp_path: Path) -> None:
    path = tmp_path / "latin.yaml"
    path.write_bytes(b"extends: child\nprompts: caf\xe9\n")
    message = load_error(path)
    assert "latin.yaml" in message and "UTF-8" in message, message


# ---------------------------------------------------------------------------
# Unknown keys


@pytest.mark.parametrize(
    ("fixture", "dotted"),
    [
        pytest.param("unknown-top-key.yaml", "stagse", id="top"),
        pytest.param("unknown-nested-key.yaml", "sandbox.netwrok", id="nested"),
        pytest.param("unknown-fix.yaml", "fixes.fence_tga", id="fix"),
    ],
)
def test_unknown_key_fixtures_fail(load_error: Callable[..., str], fixture: str, dotted: str) -> None:
    message = load_error(fixture)
    assert fixture in message and dotted in message


UNKNOWN_KEYS = [
    pytest.param("llm: {sampling: {temprature: 0.3}}", "llm.sampling.temprature", id="llm-sampling"),
    pytest.param("llm: {temperature: 0.3}", "llm.temperature", id="llm"),
    pytest.param("loop: {max_correction: 3}", "loop.max_correction", id="loop"),
    pytest.param("trials: {count: 3}", "trials.count", id="trials"),
    pytest.param("report: {html: true}", "report.html", id="report"),
    pytest.param("bench: {splt: eval}", "bench.splt", id="bench"),
    pytest.param("model: {backend: mock_llm, id: m, name: x}", "model.name", id="model"),
    pytest.param("refine: {max_iter: 3}", "refine.max_iter", id="refine"),
    pytest.param("agents: {generator: {modle: x}}", "agents.generator.modle", id="agent"),
    pytest.param("judges: {equivalence: {rubrik: r}}", "judges.equivalence.rubrik", id="judge"),
    pytest.param("directions: [{source: omp, target: cuda, via: hip}]", "directions[0].via", id="direction"),
    pytest.param("base_model: some-model", "base_model", id="train-key"),
]


@pytest.mark.parametrize(("snippet", "dotted"), UNKNOWN_KEYS)
def test_unknown_keys_fail_with_dotted_path(
    load_error: Callable[..., str], tmp_path: Path, snippet: str, dotted: str
) -> None:
    message = load_error(over_child(tmp_path, snippet, "typo.yaml"))
    assert "typo.yaml" in message and dotted in message


def test_unknown_key_in_a_parent_names_that_file(load_error: Callable[..., str], tmp_path: Path) -> None:
    write(tmp_path, "typo-parent.yaml", "extends: child\nsandbox: {netwrok: true}\n")
    message = load_error(write(tmp_path, "leaf.yaml", "extends: typo-parent.yaml\n"))
    assert "typo-parent.yaml" in message and "sandbox.netwrok" in message


def test_kind_section_keys_become_component_config(load: Callable[..., Any], tmp_path: Path) -> None:
    recipe = load(over_child(tmp_path, "executor: {kind: native, host: box1}"))
    executor = [b for b in recipe.bindings if b.where == "executor.kind"]
    assert binding_tuples(executor) == [("Executor", "native", "executor.kind", {"host": "box1"})]


def test_unknown_config_key_fails_at_the_bindings_check(load_error: Callable[..., str]) -> None:
    message = load_error("unknown-config-key.yaml")
    assert "unknown-config-key.yaml" in message
    assert "oracle.treshold" in message and "oracle.kind.treshold" not in message, message
    for part in ("Oracle", "stdout_mask", "passfail"):
        assert part in message, message


# ---------------------------------------------------------------------------
# Types


TYPE_ERRORS = [
    pytest.param("faithful: 'true'", "faithful", id="faithful-str"),
    pytest.param("faithful: 1", "faithful", id="faithful-int"),
    pytest.param("fixes: {fence_tag: 0}", "fixes.fence_tag", id="fix-int"),
    pytest.param("fixes: [fence_tag]", "fixes", id="fixes-list"),
    pytest.param("llm: {sampling: 0.2}", "llm.sampling", id="sampling-scalar"),
    pytest.param("llm: {sampling: {temperature: true}}", "llm.sampling.temperature", id="temperature-bool"),
    pytest.param("llm: {sampling: {top_p: '0.9'}}", "llm.sampling.top_p", id="top-p-str"),
    pytest.param("llm: {sampling: {temperature: .nan}}", "llm.sampling.temperature", id="temperature-nan"),
    pytest.param("llm: {sampling: {max_tokens: 1.5}}", "llm.sampling.max_tokens", id="max-tokens-float"),
    pytest.param("llm: {sampling: {max_tokens: false}}", "llm.sampling.max_tokens", id="max-tokens-bool"),
    pytest.param("loop: {max_corrections: -1}", "loop.max_corrections", id="corrections-negative"),
    pytest.param("loop: {max_corrections: unlimited}", "loop.max_corrections", id="corrections-other-str"),
    pytest.param("loop: {max_corrections: Uncapped}", "loop.max_corrections", id="corrections-case"),
    pytest.param("loop: {max_corrections: '10'}", "loop.max_corrections", id="corrections-digit-str"),
    pytest.param("loop: {max_corrections: 2.5}", "loop.max_corrections", id="corrections-float"),
    pytest.param("loop: {max_corrections: true}", "loop.max_corrections", id="corrections-bool"),
    pytest.param("trials: {n: true}", "trials.n", id="n-bool"),
    pytest.param("trials: {n: '5'}", "trials.n", id="n-str"),
    pytest.param("trials: {n: 2.0}", "trials.n", id="n-float"),
    pytest.param("trials: {n: 0}", "trials.n", id="n-zero"),
    pytest.param("trials: {n: -2}", "trials.n", id="n-negative"),
    pytest.param("runs_root: 5", "runs_root", id="runs-root-int"),
    pytest.param("sandbox: {network: 'false'}", "sandbox.network", id="network-str"),
    pytest.param("sandbox: {wall_s: true}", "sandbox.wall_s", id="wall-bool"),
    pytest.param("sandbox: {wall_s: [60]}", "sandbox.wall_s", id="wall-list"),
    pytest.param("sandbox: {mem_gb: '16'}", "sandbox.mem_gb", id="mem-str"),
    pytest.param("sandbox: {mem_gb: .inf}", "sandbox.mem_gb", id="mem-inf"),
    pytest.param("sandbox: {wall_s: .nan}", "sandbox.wall_s", id="wall-nan"),
    pytest.param("report: {parquet: 1}", "report.parquet", id="parquet-int"),
    pytest.param("report: {trial_md: 1}", "report.trial_md", id="trial-md-int"),
    pytest.param("bench: {suite: 7}", "bench.suite", id="suite-int"),
    pytest.param("bench: {split: [eval]}", "bench.split", id="split-list"),
    pytest.param("directions: []", "directions", id="directions-empty"),
    pytest.param("directions: [omp-cuda]", "directions", id="directions-str-item"),
    pytest.param("directions: {source: omp, target: cuda}", "directions", id="directions-mapping"),
    pytest.param("directions: [{source: omp, target: 3}]", "directions[0].target", id="direction-target-int"),
    pytest.param("directions: [{source: 3, target: cuda}]", "directions[0].source", id="direction-source-int"),
    pytest.param("prompts: [a]", "prompts", id="prompts-list"),
    pytest.param("context: card-a", "context", id="context-str"),
    pytest.param("context: [card-a, 5]", "context[1]", id="context-int-item"),
    pytest.param("toolchain: {cuda: 5}", "toolchain.cuda", id="toolchain-int"),
    pytest.param("toolchain: [fake_cc]", "toolchain", id="toolchain-list"),
    pytest.param("stages: []", "stages", id="stages-empty"),
    pytest.param("stages: generate", "stages", id="stages-str"),
    pytest.param("stages: [generate, 3]", "stages[1]", id="stages-int-item"),
    pytest.param("executor: native", "executor", id="executor-str"),
    pytest.param("executor: {kind: 5}", "executor.kind", id="kind-int"),
    pytest.param("executor: {kind: [native]}", "executor.kind", id="kind-list"),
    # Kind-section config is plain data: a mixed set would hash differently per process, and pairs reload as lists.
    pytest.param("executor: {kind: native, host: !!set {alpha, 1}}", "executor.host", id="config-set"),
    pytest.param("executor: {kind: native, host: !!omap [{a: 1}]}", "executor.host[0]", id="config-omap"),
    pytest.param("executor: {kind: native, host: !!pairs [{a: null}]}", "executor.host[0]", id="config-pairs"),
    pytest.param("executor: {kind: native, host: !!binary aGk=}", "executor.host", id="config-binary"),
    pytest.param("executor: {kind: native, host: 2026-01-01}", "executor.host", id="config-date"),
    pytest.param("executor: {kind: native, host: [box1, .nan]}", "executor.host[1]", id="config-nan"),
    pytest.param("executor: {kind: native, 5: box1}", "executor.5", id="config-key-int"),
    pytest.param("model: {backend: [mock_llm]}", "model.backend", id="backend-list"),
    pytest.param("model: {id: 5}", "model.id", id="model-id-int"),
    pytest.param("arms: fixture-arm", "arms", id="arms-str"),
    pytest.param("arms: [1]", "arms[0]", id="arms-int-item"),
    pytest.param("metrics: correct", "metrics", id="metrics-str"),
    pytest.param("metrics: [true]", "metrics[0]", id="metrics-bool-item"),
    pytest.param("refine: {max_iters: 2.0}", "refine.max_iters", id="max-iters-float"),
    pytest.param("refine: {counter_start: '0.2'}", "refine.counter_start", id="counter-start-str"),
    pytest.param("refine: {counter_step: true}", "refine.counter_step", id="counter-step-bool"),
    pytest.param("score: 1", "score", id="score-int"),
    pytest.param("agents: [generator]", "agents", id="agents-list"),
    pytest.param("agents: {generator: fixture-model}", "agents.generator", id="agent-str"),
    pytest.param("agents: {reviewer: {model: m, enabled: 'no'}}", "agents.reviewer.enabled", id="agent-enabled-str"),
    pytest.param("agents: {fixer: {model: 5}}", "agents.fixer.model", id="agent-model-int"),
    pytest.param("judges: {equivalence: {rubric: 5}}", "judges.equivalence.rubric", id="judge-rubric-int"),
    pytest.param("judges: {equivalence: {model: m, rubric: r, use: 5}}", "judges.equivalence.use", id="judge-use-int"),
    pytest.param(
        "judges: {efficiency: {model: m, rubric: r, mode: 1, use: metric}}",
        "judges.efficiency.mode",
        id="judge-mode-int",
    ),
]


@pytest.mark.parametrize(("snippet", "dotted"), TYPE_ERRORS)
def test_wrong_types_fail_with_dotted_path(
    load_error: Callable[..., str], tmp_path: Path, snippet: str, dotted: str
) -> None:
    message = load_error(over_child(tmp_path, snippet, "bad.yaml"))
    # "must be" makes this a type complaint, so no later check (a registry lookup) can pass for the type rule.
    assert "bad.yaml" in message and dotted in message and "must be" in message, message


VALID_VALUES = [
    pytest.param("llm: {sampling: {temperature: 1, top_p: 1}}", "llm.sampling.temperature", 1, id="int-as-number"),
    pytest.param("loop: {max_corrections: 0}", "loop.max_corrections", 0, id="zero-corrections"),
    pytest.param("loop: {max_corrections: uncapped}", "loop.max_corrections", "uncapped", id="uncapped"),
    pytest.param("trials: {n: 1}", "trials.n", 1, id="one-trial"),
    pytest.param("sandbox: {wall_s: 90.5}", "sandbox.wall_s", 90.5, id="wall-float"),
    pytest.param("sandbox: {wall_s: baseline_x10}", "sandbox.wall_s", "baseline_x10", id="wall-str"),
    pytest.param("sandbox: {mem_gb: 0.5}", "sandbox.mem_gb", 0.5, id="mem-float"),
    pytest.param("refine: {counter_start: 0, counter_step: 0.2, max_iters: 10}", "refine.max_iters", 10, id="refine"),
    pytest.param("arms: []", "arms", [], id="arms-empty"),
    pytest.param("metrics: []", "metrics", [], id="metrics-empty"),
    pytest.param("agents: {reviewer: {model: m, enabled: false}}", "agents.reviewer.enabled", False, id="agent"),
    pytest.param("toolchain: {cuda: fake_cxx}", "toolchain.cuda", "fake_cxx", id="toolchain"),
    pytest.param("executor: {kind: native, host: box1}", "executor.host", "box1", id="executor-config"),
    pytest.param(
        "executor: {kind: native, host: {name: box1, ports: [1, 2.5, true]}}",
        "executor.host",
        {"name": "box1", "ports": [1, 2.5, True]},
        id="executor-config-nested",
    ),
    pytest.param("faithful: false", "faithful", False, id="faithful-false"),
]


@pytest.mark.parametrize(("snippet", "dotted", "expected"), VALID_VALUES)
def test_valid_values_load(load: Callable[..., Any], tmp_path: Path, snippet: str, dotted: str, expected: Any) -> None:
    assert lookup(load(over_child(tmp_path, snippet, "ok.yaml")).data, dotted) == expected


# ---------------------------------------------------------------------------
# Required choices


def test_standalone_recipe_loads(load: Callable[..., Any], tmp_path: Path) -> None:
    recipe = load(write(tmp_path, "standalone.yaml", yaml.safe_dump(STANDALONE)))
    assert recipe.data == with_defaults(STANDALONE)
    assert recipe.chain == ("standalone",)


@pytest.mark.parametrize("dotted", REQUIRED)
def test_each_required_key_must_be_stated(load_error: Callable[..., str], tmp_path: Path, dotted: str) -> None:
    data = copy.deepcopy(STANDALONE)
    delete_path(data, dotted)
    message = load_error(write(tmp_path, "partial.yaml", yaml.safe_dump(data)))
    assert_required_message(message, "partial.yaml", dotted)


@pytest.mark.parametrize("section", ["executor", "oracle", "profiler", "adversary"])
def test_kind_sections_need_a_kind(load_error: Callable[..., str], tmp_path: Path, section: str) -> None:
    data = copy.deepcopy(STANDALONE)
    data[section] = {"passfail": True}
    message = load_error(write(tmp_path, "kindless.yaml", yaml.safe_dump(data)))
    assert "kindless.yaml" in message and f"{section}.kind" in message


def test_missing_required_fixture_fails(load_error: Callable[..., str]) -> None:
    assert_required_message(load_error("missing-required.yaml"), "missing-required.yaml", "trials.n")


def test_null_choice_fixture_fails(load_error: Callable[..., str]) -> None:
    assert_required_message(load_error("null-choice.yaml"), "null-choice.yaml", "executor.kind")


NULL_VALUES = [
    pytest.param("sandbox: {mem_gb: null}", "sandbox.mem_gb", id="explicit-null-replaces-parent-value"),
    pytest.param("bench: {suite: }", "bench.suite", id="empty-value"),
    pytest.param("directions: ~", "directions", id="tilde"),
    pytest.param("llm: {sampling: {max_tokens: null}}", "llm.sampling.max_tokens", id="optional-key"),
    pytest.param("prompts: null", "prompts", id="top-level-optional"),
    pytest.param("profiler: null", "profiler", id="null-replaces-mapping"),
    pytest.param("agents: {fixer: {model: null}}", "agents.fixer.model", id="inside-open-map"),
    pytest.param("agents: {fixer: null}", "agents.fixer", id="open-map-value"),
    pytest.param("judges: {equivalence: null}", "judges.equivalence", id="judge-value"),
    # The registry would refuse this too, but with a message that is not a required choice.
    pytest.param("toolchain: {cuda: null}", "toolchain.cuda", id="toolchain-value"),
    pytest.param("faithful: null", "faithful", id="faithful"),
    pytest.param("fixes: {fence_tag: null}", "fixes.fence_tag", id="fix"),
    pytest.param("oracle: {passfail: null}", "oracle.passfail", id="kind-section-config"),
    pytest.param("adversary: {budget: {inputs: null}}", "adversary.budget.inputs", id="nested-kind-config"),
    pytest.param("stages: [generate, null]", "stages[1]", id="list-item"),
]


@pytest.mark.parametrize(("snippet", "dotted"), NULL_VALUES)
def test_null_leaves_fail_as_required_choices(
    load_error: Callable[..., str], tmp_path: Path, snippet: str, dotted: str
) -> None:
    assert_required_message(load_error(over_child(tmp_path, snippet, "nulls.yaml")), "nulls.yaml", dotted)


# Keys each list or open-map entry must state: a direction names both ends, an agent its model, and a judge its
# model, rubric, and use. Fixed sections (model, bench, refine) follow the general rule, "optional unless Required".
ENTRY_KEYS = [
    pytest.param("directions: [{source: omp}]", "directions[0].target", id="direction-no-target"),
    pytest.param(
        "directions: [{source: omp, target: cuda}, {target: omp}]", "directions[1].source", id="direction-no-source"
    ),
    pytest.param("agents: {reviewer: {}}", "agents.reviewer.model", id="agent-empty"),
    pytest.param("agents: {reviewer: {enabled: false}}", "agents.reviewer.model", id="agent-no-model"),
    pytest.param("judges: {readability: {use: screen}}", "judges.readability.model", id="judge-no-model"),
    pytest.param("judges: {readability: {model: m, use: screen}}", "judges.readability.rubric", id="judge-no-rubric"),
    pytest.param("judges: {readability: {model: m, rubric: r}}", "judges.readability.use", id="judge-no-use"),
]


@pytest.mark.parametrize(("snippet", "dotted"), ENTRY_KEYS)
def test_entries_state_every_key_not_marked_optional(
    load_error: Callable[..., str], tmp_path: Path, snippet: str, dotted: str
) -> None:
    assert_required_message(load_error(over_child(tmp_path, snippet, "entry.yaml")), "entry.yaml", dotted)


# ---------------------------------------------------------------------------
# Faithful mode and fixes


def test_fixes_table_and_uncapped(recipe_module: ModuleType) -> None:
    assert recipe_module.UNCAPPED == "uncapped"
    assert recipe_module.FIXES == {"fence_tag": FENCE_TAG_DESCRIPTION}


def test_defaults_are_materialized(load: Callable[..., Any]) -> None:
    data = load("child.yaml").data
    assert data["faithful"] is False
    assert data["fixes"] == {"fence_tag": True}


def test_faithful_overrides_a_capped_parent_and_an_explicit_fix(
    load: Callable[..., Any], recipe_module: ModuleType
) -> None:
    data = load("faithful-child.yaml").data
    assert data["faithful"] is True
    assert data["loop"] == {"max_corrections": recipe_module.UNCAPPED}
    assert data["fixes"] == {"fence_tag": False}


def test_faithful_overrides_values_in_the_same_file(load: Callable[..., Any], tmp_path: Path) -> None:
    text = "extends: parent\nfaithful: true\nloop: {max_corrections: 4}\nfixes: {fence_tag: true}\n"
    data = load(write(tmp_path, "faithful-own.yaml", text)).data
    assert data["loop"] == {"max_corrections": "uncapped"}
    assert data["fixes"] == {"fence_tag": False}


def test_faithful_creates_the_loop_section(load: Callable[..., Any], tmp_path: Path) -> None:
    data = copy.deepcopy(STANDALONE)
    del data["loop"]
    data["faithful"] = True
    recipe = load(write(tmp_path, "faithful-no-loop.yaml", yaml.safe_dump(data)))
    assert recipe.data["loop"] == {"max_corrections": "uncapped"}
    assert recipe.data["fixes"] == {"fence_tag": False}


def test_faithful_applies_to_the_resolved_value(load: Callable[..., Any], tmp_path: Path) -> None:
    data = load(write(tmp_path, "unfaithful.yaml", "extends: faithful-child\nfaithful: false\n")).data
    assert data["faithful"] is False
    assert data["loop"] == {"max_corrections": 3}
    assert data["fixes"] == {"fence_tag": True}
    text = "extends: faithful-child\nloop: {max_corrections: 6}\nfixes: {fence_tag: true}\n"
    inherited = load(write(tmp_path, "inherits-faithful.yaml", text)).data
    assert inherited["faithful"] is True
    assert inherited["loop"] == {"max_corrections": "uncapped"}
    assert inherited["fixes"] == {"fence_tag": False}


def test_explicit_fix_off_keeps_the_cap(load: Callable[..., Any]) -> None:
    data = load("fix-off.yaml").data
    assert data["faithful"] is False
    assert data["fixes"] == {"fence_tag": False}
    assert data["loop"] == {"max_corrections": 3}


# ---------------------------------------------------------------------------
# Canonical YAML, recipe_hash, and the resolved file


def test_canonical_yaml_and_hash_follow_the_rule(load: Callable[..., Any]) -> None:
    recipe = load("child.yaml")
    assert recipe.canonical_yaml == canonical(EXPECTED_CHILD)
    assert recipe.recipe_hash == hashlib.sha256(recipe.canonical_yaml.encode("ascii")).hexdigest()
    assert re.fullmatch("[0-9a-f]{64}", recipe.recipe_hash)


def test_canonical_yaml_keeps_long_values_on_one_line(load: Callable[..., Any], tmp_path: Path) -> None:
    prompts = " ".join(f"word{i}" for i in range(40))
    recipe = load(over_child(tmp_path, f"prompts: {prompts}", "long.yaml"))
    assert recipe.canonical_yaml == canonical(recipe.data)
    assert f"prompts: {prompts}\n" in recipe.canonical_yaml


def test_recipe_hash_is_pinned(load: Callable[..., Any]) -> None:
    assert load("child.yaml").recipe_hash == CHILD_HASH


def test_resolved_yaml_matches_golden(load: Callable[..., Any], recipe_module: ModuleType) -> None:
    assert recipe_module.resolved_yaml(load("child.yaml")) == GOLDEN.read_text(encoding="ascii")


def test_resolved_yaml_header_and_format(load: Callable[..., Any], recipe_module: ModuleType) -> None:
    recipe = load("child.yaml")
    text = recipe_module.resolved_yaml(recipe)
    header = (
        "# Resolved recipe child (chain: fixture-base -> parent -> child)\n"
        f"# recipe_hash: {CHILD_HASH} is the sha256 of the YAML below these comment lines\n"
    )
    assert text == header + recipe.canonical_yaml
    assert text.isascii() and "\r" not in text and text.endswith("\n")


def test_hash_is_stable_across_loads_and_directories(load: Callable[..., Any], tmp_path: Path) -> None:
    first, second = load("child.yaml"), load("child.yaml")
    assert (first.canonical_yaml, first.recipe_hash) == (second.canonical_yaml, second.recipe_hash)
    for name in CHILD_CHAIN:
        shutil.copy(FIXTURES / f"{name}.yaml", tmp_path / f"{name}.yaml")
    moved = load(tmp_path / "child.yaml", roots=[tmp_path])
    assert moved.data == EXPECTED_CHILD
    assert moved.recipe_hash == CHILD_HASH


def test_hash_covers_the_data_only(load: Callable[..., Any], tmp_path: Path) -> None:
    same = load(write(tmp_path, "same.yaml", "extends: child\n"))
    assert same.chain == (*CHILD_CHAIN, "same")
    assert same.recipe_hash == CHILD_HASH
    changed = load(over_child(tmp_path, "trials: {n: 3}", "changed.yaml"))
    assert changed.recipe_hash != CHILD_HASH


def test_golden_file_loads_back_as_a_recipe(load: Callable[..., Any]) -> None:
    recipe = load(GOLDEN)
    assert recipe.name == "child.resolved"
    assert recipe.chain == ("child.resolved",)
    assert recipe.data == EXPECTED_CHILD
    assert recipe.recipe_hash == CHILD_HASH


@pytest.mark.parametrize("fixture", ["child.yaml", "faithful-child.yaml", "fix-off.yaml", "dir-child.yaml"])
def test_resolved_file_reloads_to_the_same_data_and_hash(
    load: Callable[..., Any], recipe_module: ModuleType, tmp_path: Path, fixture: str
) -> None:
    recipe = load(fixture)
    path = write(tmp_path, "saved.resolved.yaml", recipe_module.resolved_yaml(recipe))
    back = load(path)
    assert back.chain == ("saved.resolved",)
    assert back.data == recipe.data
    assert back.recipe_hash == recipe.recipe_hash
    assert recipe_module.resolved_yaml(back).splitlines()[1:] == recipe_module.resolved_yaml(recipe).splitlines()[1:]


def test_non_ascii_values_hash_through_ascii_yaml(
    load: Callable[..., Any], recipe_module: ModuleType, tmp_path: Path
) -> None:
    prompts = f"caf{E_ACUTE}-prompts"
    recipe = load(over_child(tmp_path, f"prompts: {prompts}", "accented.yaml"))
    assert recipe.data["prompts"] == prompts
    assert recipe.canonical_yaml.isascii()
    assert recipe.recipe_hash == hashlib.sha256(canonical(recipe.data).encode("ascii")).hexdigest()
    back = load(write(tmp_path, "accented.resolved.yaml", recipe_module.resolved_yaml(recipe)))
    assert back.data["prompts"] == prompts
    assert back.recipe_hash == recipe.recipe_hash


# ---------------------------------------------------------------------------
# Bindings and capability validation at load time


def test_bindings_follow_the_documented_order(load: Callable[..., Any], registry_module: ModuleType) -> None:
    recipe = load("child.yaml")
    assert isinstance(recipe.bindings, tuple)
    assert all(isinstance(b, registry_module.Binding) for b in recipe.bindings)
    assert binding_tuples(recipe.bindings) == EXPECTED_CHILD_BINDINGS


def test_loading_constructs_nothing(load: Callable[..., Any]) -> None:
    for fixture in ("child.yaml", "faithful-child.yaml", "dir-child.yaml"):
        load(fixture)
    assert CONSTRUCTED == []


def test_capability_mismatch_fails_at_load_naming_both(load_error: Callable[..., str]) -> None:
    message = load_error("capability-mismatch.yaml")
    assert message.startswith(f"{FIXTURES / 'capability-mismatch.yaml'}: "), message
    for part in (
        "capability-mismatch.yaml",
        "runs_code",
        "run_loop",
        "stages[2]",
        "Executor",
        "none",
        "executor.kind",
        "compile_only",
    ):
        assert part in message, message


def test_requirement_on_an_unbound_interface_fails(load_error: Callable[..., str]) -> None:
    message = load_error("unbound-requirement.yaml")
    for part in ("unbound-requirement.yaml", "profile", "stages[1]", "Profiler", "timing"):
        assert part in message, message


def test_unregistered_component_fails(load_error: Callable[..., str]) -> None:
    message = load_error("unregistered.yaml")
    for part in ("unregistered.yaml", "oracle.kind", "Oracle", "no_such_oracle", "stdout_mask"):
        assert part in message, message


def test_default_registry_is_used_when_none_is_given(load_error: Callable[..., str], default_registry: Any) -> None:
    message = load_error("child.yaml", registry=None)
    assert "child.yaml" in message and "mock_llm" in message and "model.backend" in message


def test_components_registered_with_the_decorator_bind_by_default(
    recipe_module: ModuleType, registry_module: ModuleType, default_registry: Any
) -> None:
    for interface, cls in FAKES:
        registry_module.register(interface, cls.name)(cls)
    for recipe in (
        recipe_module.load_recipe(FIXTURES / "child.yaml", roots=[FIXTURES]),
        recipe_module.load_recipe(FIXTURES / "child.yaml", roots=[FIXTURES], registry=None),
    ):
        assert binding_tuples(recipe.bindings) == EXPECTED_CHILD_BINDINGS
    assert CONSTRUCTED == []


# ---------------------------------------------------------------------------
# projects/base.yaml and the default roots


def test_default_roots_is_the_projects_directory(recipe_module: ModuleType) -> None:
    roots = recipe_module.default_roots()
    assert isinstance(roots, tuple)
    assert [Path(root).resolve() for root in roots] == [PROJECTS.resolve()]


def test_project_base_matches_the_bible_block() -> None:
    assert BASE_YAML.is_file(), f"missing {BASE_YAML}"
    assert yaml.safe_load(BASE_YAML.read_text(encoding="utf-8")) == bible_block_data("projects/base.yaml")


def test_project_base_comments_every_value() -> None:
    text = BASE_YAML.read_text(encoding="utf-8")
    first = text.splitlines()[0]
    assert first.startswith("#") and "extend" in first.lower(), first
    assert value_lines_without_comment(text) == []
    assert flow_style_lines(text) == []
    assert value_line_count(text) == leaf_count(bible_block_data("projects/base.yaml"))


def test_project_base_is_ascii_with_lf_newlines() -> None:
    raw = BASE_YAML.read_bytes()
    assert raw.isascii()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")


def test_project_base_cannot_load_on_its_own(load_error: Callable[..., str]) -> None:
    message = load_error(BASE_YAML, roots=None)
    assert "base.yaml" in message
    assert any(dotted in message for dotted in ("bench.suite", "bench.split", "directions", "stages", "executor.kind"))


def test_repository_recipe_loads_through_the_default_roots(recipe_module: ModuleType, fakes: Any) -> None:
    fixture = FIXTURES / "project-base-user.yaml"
    recipe = recipe_module.load_recipe(fixture, registry=fakes)
    own = yaml.safe_load(fixture.read_text(encoding="utf-8"))
    assert own.pop("extends") == "base"
    expected = {**bible_block_data("projects/base.yaml"), **own}
    assert recipe.data == with_defaults(expected)
    assert recipe.chain == ("base", "project-base-user")
    assert recipe_module.load_recipe(fixture, roots=None, registry=fakes).recipe_hash == recipe.recipe_hash
    assert CONSTRUCTED == []


# ---------------------------------------------------------------------------
# The bible's recipe blocks fit the schema


def write_bible_recipes(directory: Path) -> None:
    """Write the bible's base block as base.yaml and each project recipe block as <project>.yaml."""
    blocks = bible_recipe_blocks()
    write(directory, "base.yaml", blocks["projects/base.yaml"])
    for project in BIBLE_RUN_RECIPES:
        write(directory, f"{project}.yaml", blocks[f"projects/{project}/recipe.yaml"])


@pytest.mark.parametrize("project", BIBLE_RUN_RECIPES)
def test_bible_project_recipes_fit_the_schema(
    recipe_module: ModuleType, registry_module: ModuleType, tmp_path: Path, project: str
) -> None:
    write_bible_recipes(tmp_path)
    own = bible_block_data(f"projects/{project}/recipe.yaml")
    registry = permissive_registry(registry_module, own)
    recipe = recipe_module.load_recipe(tmp_path / f"{project}.yaml", roots=[tmp_path], registry=registry)
    assert recipe.chain == ("base", project)
    for key, value in own.items():
        if key != "extends":
            assert recipe.data[key] == value, key
    faithful = own.get("faithful", False)
    base_loop = bible_block_data("projects/base.yaml")["loop"]
    assert recipe.data["faithful"] is faithful
    assert recipe.data["loop"] == ({"max_corrections": "uncapped"} if faithful else base_loop)
    assert recipe.data["fixes"] == {"fence_tag": not faithful}
    assert CONSTRUCTED == []


def test_bible_binding_block_fits_the_schema(
    recipe_module: ModuleType, registry_module: ModuleType, tmp_path: Path
) -> None:
    write_bible_recipes(tmp_path)
    block = bible_recipe_blocks()[BINDINGS_BLOCK]
    path = write(tmp_path, "bindings.yaml", "extends: lassi-df\n" + block)
    own = bible_block_data(BINDINGS_BLOCK)
    registry = permissive_registry(registry_module, bible_block_data("projects/lassi-df/recipe.yaml"), own)
    recipe = recipe_module.load_recipe(path, roots=[tmp_path], registry=registry)
    for key, value in own.items():
        assert recipe.data[key] == value, key
    adversary = [b for b in recipe.bindings if b.where == "adversary.kind"]
    assert binding_tuples(adversary) == [("Agent", "fuzzer", "adversary.kind", {"budget": {"inputs": 64}})]


def test_bible_train_recipe_is_not_a_run_recipe(load_error: Callable[..., str], tmp_path: Path) -> None:
    block = bible_recipe_blocks()["projects/lassi-df/train.yaml"]
    train_only = set(yaml.safe_load(block)) - RUN_RECIPE_KEYS
    assert train_only
    message = load_error(write(tmp_path, "train.yaml", block), roots=[tmp_path])
    assert "train.yaml" in message
    assert any(key in message for key in train_only), message


# ---------------------------------------------------------------------------
# Module hygiene (Agent Rule 3, Readability Standards)


@pytest.mark.parametrize("name", NEW_MODULES)
def test_new_modules_documented_typed_ascii(name: str) -> None:
    tree = ast.parse(module_source(name))
    assert ast.get_docstring(tree), f"{name} has no module docstring"
    undocumented = [qual for qual, node in public_defs(tree) if not ast.get_docstring(node)]
    assert not undocumented, f"{name}: no docstring on {undocumented}"
    untyped = [qual for qual, node in public_defs(tree) if isinstance(node, FUNCTION_NODES) and not is_typed(node)]
    assert not untyped, f"{name}: missing type hints on {untyped}"


@pytest.mark.parametrize("name", NEW_MODULES)
def test_new_modules_hold_no_project_code(name: str) -> None:
    source = module_source(name)
    assert projects_imports(ast.parse(source)) == [], f"{name} imports the projects package"
    lowered = source.lower()
    found = [word for word in PROJECT_NAMES if word in lowered]
    assert not found, f"{name} names projects: {found}"
