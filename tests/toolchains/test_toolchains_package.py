"""Tests for the lassi.toolchains package as a whole (P0.5).

The package holds the Toolchain components (bible Component Interfaces,
Toolchain row). Its __init__ re-exports CommandResult, the CommandRunner type,
subprocess_runner (the default runner), and STDERR_ATTACHMENT from _base, and imports the
nvcc and nvcpp adapter modules so that importing the package registers both
presets. Every module in the package is documented, typed, plain ASCII, uses
only the standard library, and names no project (Agent Rule 3, Readability
Standards). Preset names such as nvcc-sm80 are component names, not project
names. No value in this module is a measurement.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

from lassi import toolchains

PACKAGE_DIR = Path(toolchains.__file__).resolve().parent
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
PROJECT_NAMES = (
    "lassi-repro",
    "lassi-ee",
    "lassi-df",
    "hecbench",
    "entropy",
    "qwen",
    "wizardcoder",
    "a100",
    "mi300x",
    "gpt-oss",
)

# The modules the P0.5 contract names, plus any other module found in the package directory.
NAMED_MODULES = ("lassi.toolchains", "lassi.toolchains.nvcc", "lassi.toolchains.nvcpp")
FOUND_MODULES = tuple(
    "lassi.toolchains" if path.stem == "__init__" else f"lassi.toolchains.{path.stem}"
    for path in sorted(PACKAGE_DIR.glob("*.py"))
)
TOOLCHAIN_MODULES = tuple(dict.fromkeys(NAMED_MODULES + FOUND_MODULES))


def module_source(name: str) -> str:
    """Return the ASCII source text of an importable module, failing when it does not exist or is not ASCII."""
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin, f"missing module {name}"
    raw = Path(spec.origin).read_bytes()
    assert raw.isascii(), f"{name} has non-ASCII source text"
    return raw.decode("ascii")


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


def imported_roots(tree: ast.Module) -> set[str]:
    """Return the top-level package of every absolute import in a module."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def imported_submodules(tree: ast.Module) -> set[str]:
    """Return the names of lassi.toolchains submodules that a module imports, in any import spelling."""
    names: set[str] = set()
    prefix = "lassi.toolchains."
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name[len(prefix) :].split(".")[0] for alias in node.names if alias.name.startswith(prefix)}
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if (node.level == 0 and module == "lassi.toolchains") or (node.level == 1 and not module):
                names |= {alias.name for alias in node.names}
            elif node.level == 0 and module.startswith(prefix):
                names.add(module[len(prefix) :].split(".")[0])
            elif node.level == 1:
                names.add(module.split(".")[0])
    return names


# ---------------------------------------------------------------------------
# What the package defines


def test_named_modules_exist() -> None:
    missing = [name for name in NAMED_MODULES if importlib.util.find_spec(name) is None]
    assert not missing, missing


def test_package_imports_both_adapters() -> None:
    # Importing lassi.toolchains alone must register both presets, so __init__ imports nvcc and nvcpp.
    tree = ast.parse(module_source("lassi.toolchains"))
    assert {"nvcc", "nvcpp"} <= imported_submodules(tree), imported_submodules(tree)


def test_package_defines_the_runner_contract() -> None:
    assert toolchains.STDERR_ATTACHMENT == "compile.stderr"
    assert inspect.isclass(toolchains.CommandResult)
    assert callable(toolchains.subprocess_runner)
    assert hasattr(toolchains, "CommandRunner")
    params = list(inspect.signature(toolchains.subprocess_runner).parameters)
    assert params == ["argv", "cwd", "timeout_s"], params


def test_subprocess_runner_docstring_says_it_is_the_default() -> None:
    doc = inspect.getdoc(toolchains.subprocess_runner) or ""
    assert "default" in doc.lower(), doc


# ---------------------------------------------------------------------------
# Module hygiene (Agent Rule 3, Readability Standards, no new dependency)


@pytest.mark.parametrize("name", TOOLCHAIN_MODULES)
def test_toolchain_modules_documented_typed_ascii(name: str) -> None:
    source = module_source(name)
    tree = ast.parse(source)
    assert ast.get_docstring(tree), f"{name} has no module docstring"
    undocumented = [qual for qual, node in public_defs(tree) if not ast.get_docstring(node)]
    assert not undocumented, f"{name}: no docstring on {undocumented}"
    untyped = [qual for qual, node in public_defs(tree) if isinstance(node, FUNCTION_NODES) and not is_typed(node)]
    assert not untyped, f"{name}: missing type hints on {untyped}"
    assert "from __future__ import annotations" in source, name


@pytest.mark.parametrize("name", TOOLCHAIN_MODULES)
def test_toolchain_modules_hold_no_project_code(name: str) -> None:
    source = module_source(name)
    lowered = source.lower()
    found = [word for word in PROJECT_NAMES if word in lowered]
    assert not found, f"{name} names projects: {found}"
    assert "projects" not in imported_roots(ast.parse(source)), f"{name} imports the projects package"


@pytest.mark.parametrize("name", TOOLCHAIN_MODULES)
def test_toolchain_modules_use_only_the_standard_library(name: str) -> None:
    roots = imported_roots(ast.parse(module_source(name)))
    third_party = sorted(root for root in roots if root not in sys.stdlib_module_names and root != "lassi")
    assert not third_party, f"{name} imports {third_party}"
