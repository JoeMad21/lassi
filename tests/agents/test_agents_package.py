"""Tests for the lassi.agents package: it exists and holds no project code (task P16.2).

Bible: Repository Layout (lassi/agents), Agents And Judges (agents and
judges are registered, versioned components that recipes bind to roles, so
adding an adversary never touches core code), Readability Standards, Agent
Rule 3 (projects are recipes; no project-specific code enters the lassi/
package). Plan: plans/p16-adversarial.md, P16.2; PHASE-NOTES P16, where it
stands (lassi/agents does not exist yet).

The contract these tests fix: lassi/agents/ is a package, importable as
lassi.agents, with a module docstring. Every module in it is plain ASCII
with LF line ends, has a module docstring, imports only the standard
library and lassi at module level, never imports the projects package, and
names no project: no recipe under projects/ (its directory or file name),
no bench suite under assets/bench/, and none of the project, model, or host
names the other package tests refuse. Later P16 tasks add the fuzzer and
the llm adversary here (P16.3, P16.4); these tests then cover their modules
too.

No value in this module is a measurement.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
AGENTS_DIR = REPO / "lassi" / "agents"
PROJECTS = REPO / "projects"
BENCH = REPO / "assets" / "bench"
PACKAGE = "lassi.agents"

# Project, model, and host names the other package tests refuse in shared code (tests/core/test_recipe.py,
# tests/llm/test_package.py), plus the device names of the demo recipes.
FIXED_NAMES = (
    "lassi-repro", "lassi-ee", "lassi-df", "lassi-demo", "hecbench", "qwen", "wizardcoder", "a100", "mi300x",
    "gpt-oss", "rngd", "alpha01",
)


def project_names() -> list[str]:
    """Return the names no lassi.agents module may hold: fixed names, recipe names, and bench suite names."""
    names = set(FIXED_NAMES)
    for path in PROJECTS.rglob("*.yaml"):
        if path != PROJECTS / "base.yaml":
            names.add(path.stem.lower())
    names.update(path.name.lower() for path in PROJECTS.iterdir() if path.is_dir())
    names.update(path.stem.lower() for path in BENCH.glob("*.yaml"))
    names.discard("recipe")  # projects/<name>/recipe.yaml: the directory names the project
    return sorted(names)


def found_modules() -> tuple[str, ...]:
    """Return lassi.agents and every module found under lassi/agents/, sorted; the package first."""
    others = sorted(
        f"{PACKAGE}.{path.stem}" for path in AGENTS_DIR.glob("*.py") if path.stem != "__init__"
    ) if AGENTS_DIR.is_dir() else []
    return (PACKAGE, *others)


def module_path(name: str) -> Path:
    """Return the source file of module `name` under lassi/agents/."""
    return AGENTS_DIR / "__init__.py" if name == PACKAGE else AGENTS_DIR / f"{name.rsplit('.', 1)[1]}.py"


def module_text(name: str) -> str:
    """Return the source text of module `name`, failing unless it exists and is plain ASCII with LF line ends."""
    path = module_path(name)
    assert path.is_file(), f"{path.relative_to(REPO).as_posix()} does not exist (task P16.2)"
    raw = path.read_bytes()
    assert raw.isascii(), f"{name} is not plain ASCII"
    assert b"\r" not in raw, f"{name} has a CR"
    return raw.decode("ascii")


def module_level_imports(tree: ast.Module) -> list[str]:
    """Return the top-level package of every absolute import at module level (not inside a function)."""
    roots: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.append(node.module.split(".")[0])
    return roots


def projects_imports(tree: ast.Module) -> list[int]:
    """Return the line numbers of every absolute import of the projects package, at any depth."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            lines += [node.lineno for alias in node.names if alias.name.split(".")[0] == "projects"]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").split(".")[0] == "projects":
            lines.append(node.lineno)
    return lines


def test_the_package_exists_and_imports() -> None:
    assert (AGENTS_DIR / "__init__.py").is_file(), "lassi/agents/__init__.py does not exist (task P16.2)"
    package = importlib.import_module(PACKAGE)
    assert hasattr(package, "__path__"), "lassi.agents is not a package"
    assert package.__doc__ and package.__doc__.strip(), "lassi.agents has no docstring"


def test_the_name_list_holds_the_committed_recipes_and_suites() -> None:
    names = project_names()
    assert {"lassi-repro", "lassi-demo", "rngd-cpu", "tt-pairs-v0", "lassi-hecbench-10"} <= set(names), names
    assert "base" not in names and "recipe" not in names


@pytest.mark.parametrize("name", found_modules())
def test_each_module_is_documented_ascii_and_imports_only_the_standard_library_and_lassi(name: str) -> None:
    tree = ast.parse(module_text(name))
    assert ast.get_docstring(tree), f"{name} has no module docstring"
    third_party = sorted(
        root for root in module_level_imports(tree) if root not in sys.stdlib_module_names and root != "lassi"
    )
    assert not third_party, f"{name} imports {third_party} at module level"


@pytest.mark.parametrize("name", found_modules())
def test_each_module_holds_no_project_code(name: str) -> None:
    source = module_text(name)
    assert projects_imports(ast.parse(source)) == [], f"{name} imports the projects package"
    lowered = source.lower()
    found = [word for word in project_names() if word in lowered]
    assert not found, f"{name} names projects: {found}"
