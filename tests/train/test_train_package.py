"""Tests for the lassi.train package as a whole (tasks P17.8, P17.9): the registration rule and module hygiene.

Bible: Repository Layout (train/), Readability Standards (docstrings on
public interfaces, type hints, plain ASCII), Design Principle 9 (placement:
framework code beside its adapter), Agent Rule 3; PHASE-NOTES P0 and P17,
the registration rule: lassi.llm and lassi.train are imported eagerly so
that every component registers, so their modules import no framework at
module level, and a framework is imported only when a component is built.

The contract these tests fix:

- lassi.train, lassi.train.run, and lassi.train.data exist, and so do
  lassi.train.trl_trainer, lassi.train.checkpoint, and
  lassi.train.fixture_reward (task P17.9). Importing lassi.train and
  lassi.cli succeeds in a fresh interpreter in which every framework import
  fails (torch, transformers, trl, peft, accelerate, datasets,
  huggingface_hub), loading none of them, and registers exactly one
  Trainer, trl (lassi.train.trl_trainer); importing lassi.cli imports
  lassi.train, so a trainer that lassi.train registers is registered for
  `lassi train`.
- Every module under lassi/train/ imports only the standard library and
  lassi at module level; inside a function it may import only those
  framework packages (a trainer's framework, as P17.9 adds), never another
  third-party package.
- Every module is documented, typed (public functions and methods), plain
  ASCII with LF, starts with `from __future__ import annotations`, and
  names no project.

No value in this module is a measurement.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from train_fakes import REPO

TRAIN_DIR = REPO / "lassi" / "train"
NAMED_MODULES = (
    "lassi.train", "lassi.train.run", "lassi.train.data", "lassi.train.trl_trainer", "lassi.train.checkpoint",
    "lassi.train.fixture_reward",
)
FOUND_MODULES = tuple(
    "lassi.train" if path.stem == "__init__" else f"lassi.train.{path.stem}" for path in sorted(TRAIN_DIR.glob("*.py"))
)
TRAIN_MODULES = tuple(dict.fromkeys(NAMED_MODULES + FOUND_MODULES))
# The framework packages a trainer may import inside its functions, and only there.
FRAMEWORK_IMPORTS = frozenset({"torch", "transformers", "trl", "peft", "accelerate", "datasets", "huggingface_hub"})
PROJECT_NAMES = ("lassi-repro", "lassi-ee", "lassi-df", "hecbench", "qwen", "wizardcoder", "a100", "mi300x", "gpt-oss")
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)

# Run in a fresh interpreter: refuse every framework import, import lassi.train and lassi.cli, report what loaded.
BLOCKED_IMPORT_SCRIPT = """
import importlib.abc
import json
import sys

BLOCKED = set(json.loads(sys.argv[1]))


class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"blocked for the test: {fullname}")
        return None


sys.meta_path.insert(0, Refuse())
import lassi.train  # noqa: E402
import lassi.cli  # noqa: E402
from lassi.core.registry import DEFAULT_REGISTRY, INTERFACES  # noqa: E402

print(json.dumps({
    "loaded": sorted(name for name in sys.modules if name.split(".")[0] in BLOCKED),
    "modules": sorted(name for name in sys.modules if name == "lassi.train" or name.startswith("lassi.train.")),
    "interfaces": list(INTERFACES),
    "trainers": DEFAULT_REGISTRY.names("Trainer") if "Trainer" in INTERFACES else None,
}))
"""


def module_source(name: str) -> str:
    """Return the ASCII source text of a module, failing when it does not exist, is not ASCII, or holds a CR."""
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    if spec is None or not spec.origin:
        pytest.fail(f"{name} does not exist yet (tasks P17.8, P17.9)")
    raw = Path(spec.origin).read_bytes()
    assert raw.isascii(), f"{name} has non-ASCII source text"
    assert b"\r" not in raw, f"{name} has a CR"
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


def split_imports(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Return the top-level packages of the absolute imports outside every function, and of those inside one."""
    outside: set[str] = set()
    inside: set[str] = set()

    def visit(node: ast.AST, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            found = inside if in_function else outside
            if isinstance(child, ast.Import):
                found.update(alias.name.split(".")[0] for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
                found.add(child.module.split(".")[0])
            elif isinstance(child, ast.Call) and child.args and isinstance(child.args[0], ast.Constant):
                name = child.func.attr if isinstance(child.func, ast.Attribute) else getattr(child.func, "id", "")
                if name in ("import_module", "__import__") and isinstance(child.args[0].value, str):
                    found.add(child.args[0].value.split(".")[0])
            visit(child, in_function or isinstance(child, (*FUNCTION_NODES, ast.Lambda)))

    visit(tree, False)
    return outside, inside


def third_party(roots: set[str]) -> set[str]:
    """Return the roots that are neither standard library modules nor lassi."""
    return {root for root in roots if root not in sys.stdlib_module_names and root != "lassi"}


# ---------------------------------------------------------------------------
# The registration rule


def test_named_modules_exist() -> None:
    if importlib.util.find_spec("lassi.train") is None:
        pytest.fail("lassi.train does not exist yet (task P17.8)")
    missing = [name for name in NAMED_MODULES if importlib.util.find_spec(name) is None]
    assert not missing, f"missing modules (tasks P17.8, P17.9): {missing}"


def test_lassi_train_and_cli_import_with_every_framework_blocked() -> None:
    if importlib.util.find_spec("lassi.train") is None:
        pytest.fail("lassi.train does not exist yet (task P17.8)")
    result = subprocess.run(
        [sys.executable, "-c", BLOCKED_IMPORT_SCRIPT, json.dumps(sorted(FRAMEWORK_IMPORTS))],
        cwd=REPO, capture_output=True, text=True, timeout=300, check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["loaded"] == [], f"a framework was imported: {report['loaded']}"
    assert {"lassi.train", "lassi.train.run", "lassi.train.data", "lassi.train.trl_trainer"} <= set(
        report["modules"]
    ), report["modules"]
    assert report["interfaces"][-1] == "Trainer", report["interfaces"]
    assert report["trainers"] == ["trl"], "lassi.train registers the trl Trainer without a framework (task P17.9)"


def test_lassi_cli_imports_lassi_train() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import sys, lassi.cli; print('lassi.train' in sys.modules)"],
        cwd=REPO, capture_output=True, text=True, timeout=300, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True", "importing lassi.cli must import lassi.train, so its trainers register"


# ---------------------------------------------------------------------------
# Module hygiene


@pytest.mark.parametrize("name", TRAIN_MODULES)
def test_lassi_train_modules_import_only_the_standard_library_and_lassi_at_module_level(name: str) -> None:
    outside, _ = split_imports(ast.parse(module_source(name)))
    found = sorted(third_party(outside))
    assert not found, f"{name} imports {found} at module level, so lassi.train would need it to register its trainers"


@pytest.mark.parametrize("name", TRAIN_MODULES)
def test_lassi_train_modules_import_only_framework_packages_inside_functions(name: str) -> None:
    _, inside = split_imports(ast.parse(module_source(name)))
    found = sorted(third_party(inside) - FRAMEWORK_IMPORTS)
    assert not found, f"{name} imports {found} inside a function; only {sorted(FRAMEWORK_IMPORTS)} may be"


@pytest.mark.parametrize("name", TRAIN_MODULES)
def test_lassi_train_modules_are_documented_typed_and_ascii(name: str) -> None:
    source = module_source(name)
    tree = ast.parse(source)
    assert ast.get_docstring(tree), f"{name} has no module docstring"
    undocumented = [qual for qual, node in public_defs(tree) if not ast.get_docstring(node)]
    assert not undocumented, f"{name}: no docstring on {undocumented}"
    untyped = [qual for qual, node in public_defs(tree) if isinstance(node, FUNCTION_NODES) and not is_typed(node)]
    assert not untyped, f"{name}: missing type hints on {untyped}"
    assert "from __future__ import annotations" in source, name


@pytest.mark.parametrize("name", TRAIN_MODULES)
def test_lassi_train_modules_hold_no_project_code(name: str) -> None:
    source = module_source(name)
    found = [word for word in PROJECT_NAMES if word in source.lower()]
    assert not found, f"{name} names projects: {found}"
    outside, inside = split_imports(ast.parse(source))
    assert "projects" not in outside | inside, f"{name} imports the projects package"


def test_the_import_split_sees_imports_inside_functions() -> None:
    source = (
        "import os\nif os:\n    import json\ndef f():\n    import torch\n    from trl import x\n"
        "    importlib.import_module('peft')\n"
    )
    assert split_imports(ast.parse(source)) == ({"os", "json"}, {"torch", "trl", "peft"})
