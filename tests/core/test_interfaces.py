"""Tests for the lassi package skeleton and the twelve component interfaces (P0.1).

The expected interface names come from the Component Interfaces table in
docs/BIBLE.md. The tests check that lassi/core/interfaces.py defines exactly
those Protocols, each documented and type hinted, that every interface can
declare capabilities through lassi/core/capabilities.py, and that no module
under lassi/ imports the projects package (Agent Rule 3, Design Principle 9).
"""

from __future__ import annotations

import ast
import importlib
import inspect
import typing
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
BIBLE = REPO / "docs" / "BIBLE.md"
PACKAGE_DIR = REPO / "lassi"
INTERFACES_MODULE = "lassi.core.interfaces"
EXPECTED_COUNT = 12


def bible_interface_names() -> list[str]:
    """Return the first-column names of the Component Interfaces table in the bible."""
    lines = BIBLE.read_text(encoding="utf-8").splitlines()
    start = lines.index("## Component Interfaces")
    names: list[str] = []
    in_rows = False
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not in_rows:
            if stripped.startswith("| ---"):
                in_rows = True
            continue
        if not stripped:
            break
        names.append(stripped.strip("|").split("|")[0].strip())
    return names


def find_projects_imports(source: str, filename: str = "<string>") -> list[str]:
    """Return 'file:line' entries for every absolute import of the projects package."""
    hits: list[str] = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "projects" or alias.name.startswith("projects."):
                    hits.append(f"{filename}:{node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level == 0 and (mod == "projects" or mod.startswith("projects.")):
                hits.append(f"{filename}:{node.lineno}")
    return hits


def public_methods(proto: type) -> dict[str, typing.Any]:
    """Return functions defined directly in a Protocol body whose names do not start with '_'."""
    return {
        name: value
        for name, value in vars(proto).items()
        if not name.startswith("_") and inspect.isfunction(value)
    }


@pytest.fixture(scope="module")
def interfaces() -> ModuleType:
    """Import lassi.core.interfaces."""
    return importlib.import_module(INTERFACES_MODULE)


@pytest.fixture(scope="module")
def capabilities() -> ModuleType:
    """Import lassi.core.capabilities."""
    return importlib.import_module("lassi.core.capabilities")


@pytest.fixture(scope="module")
def protocols(interfaces: ModuleType) -> dict[str, type]:
    """Map each bible interface name to the class of that name in the interfaces module."""
    return {name: getattr(interfaces, name) for name in bible_interface_names()}


def test_bible_table_names_twelve_interfaces() -> None:
    names = bible_interface_names()
    assert len(names) == EXPECTED_COUNT, names
    assert len(set(names)) == EXPECTED_COUNT, names


def test_lassi_imports_with_version() -> None:
    lassi = importlib.import_module("lassi")
    assert isinstance(lassi.__version__, str)
    assert lassi.__version__


def test_interfaces_module_defines_every_bible_interface(interfaces: ModuleType) -> None:
    missing = [name for name in bible_interface_names() if not hasattr(interfaces, name)]
    assert not missing, f"interfaces missing: {missing}"


def test_interfaces_are_protocols_defined_in_module(protocols: dict[str, type]) -> None:
    for name, proto in protocols.items():
        assert inspect.isclass(proto), name
        assert proto.__module__ == INTERFACES_MODULE, name
        assert getattr(proto, "_is_protocol", False), f"{name} is not a typing.Protocol"


def test_interfaces_module_defines_exactly_the_bible_protocols(interfaces: ModuleType) -> None:
    defined = {
        name
        for name, obj in vars(interfaces).items()
        if inspect.isclass(obj)
        and obj.__module__ == INTERFACES_MODULE
        and getattr(obj, "_is_protocol", False)
    }
    assert defined == set(bible_interface_names())
    assert len(defined) == EXPECTED_COUNT


def test_interfaces_have_docstrings(protocols: dict[str, type]) -> None:
    for name, proto in protocols.items():
        doc = proto.__dict__.get("__doc__")
        assert doc and doc.strip(), f"{name} has no docstring"


def test_interfaces_have_documented_type_hinted_methods(protocols: dict[str, type]) -> None:
    for name, proto in protocols.items():
        methods = public_methods(proto)
        assert methods, f"{name} defines no public method"
        for mname, func in methods.items():
            where = f"{name}.{mname}"
            assert func.__doc__ and func.__doc__.strip(), f"{where} has no docstring"
            sig = inspect.signature(func)
            for pname, param in sig.parameters.items():
                if pname == "self":
                    continue
                assert param.annotation is not inspect.Parameter.empty, f"{where} param {pname} unannotated"
            assert "return" in func.__annotations__, f"{where} has no return annotation"


def test_component_protocol_is_runtime_checkable(capabilities: ModuleType) -> None:
    component = capabilities.Component
    assert getattr(component, "_is_protocol", False)
    obj = SimpleNamespace(name="x", capabilities=frozenset({"emits_warnings"}))
    assert isinstance(obj, component)


def test_component_declares_name_and_capabilities(capabilities: ModuleType) -> None:
    annotations = capabilities.Component.__annotations__
    assert "name" in annotations
    assert "capabilities" in annotations


def test_every_interface_extends_component(protocols: dict[str, type], capabilities: ModuleType) -> None:
    component = capabilities.Component
    for name, proto in protocols.items():
        assert component in proto.__mro__, f"{name} does not extend Component"


def test_missing_capabilities_satisfied(capabilities: ModuleType) -> None:
    obj = SimpleNamespace(name="x", capabilities=frozenset({"emits_warnings"}))
    assert capabilities.missing_capabilities(["emits_warnings"], obj) == frozenset()


def test_missing_capabilities_reports_missing(capabilities: ModuleType) -> None:
    obj = SimpleNamespace(name="x", capabilities=frozenset({"emits_warnings"}))
    result = capabilities.missing_capabilities(["emits_warnings", "supports_power"], obj)
    assert result == frozenset({"supports_power"})


def test_scanner_flags_synthetic_projects_imports() -> None:
    source = "import os\nimport projects.x\nfrom projects import y\nfrom . import projects\n"
    assert find_projects_imports(source, "synthetic.py") == ["synthetic.py:2", "synthetic.py:3"]


def test_lassi_does_not_import_projects() -> None:
    assert PACKAGE_DIR.is_dir(), f"missing package directory {PACKAGE_DIR}"
    files = sorted(PACKAGE_DIR.rglob("*.py"))
    assert files, "no Python files under lassi/"
    hits: list[str] = []
    for path in files:
        rel = path.relative_to(REPO).as_posix()
        hits.extend(find_projects_imports(path.read_text(encoding="utf-8"), rel))
    assert not hits, "lassi/ imports projects at: " + ", ".join(hits)
