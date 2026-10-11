"""Tests that lassi/corpus/ imports nothing of v0 and calls none of its scripts (task P5.11).

Bible: Review Of v0 (the Keep row's pieces move into lassi/corpus/ by
P5.2's migration map; v0's ClangIR chain is not rebuilt), Repository Layout
(corpus/ absorbs mlir-corpus-pipeline), Agent Rule 3. Plan: plans/p5-ir.md,
task P5.11 (lassi/corpus/ holds what P5.2's migration map keeps from v0,
each piece with tests; nothing imports v0 or calls its scripts);
plans/spikes/p5-v0-corpus.md, Results 5 (each piece moves as a rule P5.11
writes anew, with its test; not migrated: scripts/lower_one.sh and
lower_all.sh, tools/parse_snapshots.py's stream splitter,
scripts/build-llvm.sh and env/setup.sh, tools/mlir_to_python.py, and the
untracked training/ package).

The rule these tests fix: lassi/corpus/ is a package of Python modules,
and no module under lassi/ imports a v0 module (parse_snapshots,
build_parquet, mlir_to_python, mlir_corpus_pipeline, or the top-level
package training) or holds, in a string that is not a docstring, the name
of v0's repository, one of its scripts or tools, or its output and LLVM
trees. A docstring or comment may cite v0's source as the origin of a rule.
No value here is a measurement.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "lassi"
CORPUS = PACKAGE / "corpus"
# v0's Python modules at 3ccd280 (tools/) and its untracked package on the build host's copy (training/).
V0_MODULES = {"parse_snapshots", "build_parquet", "mlir_to_python", "mlir_corpus_pipeline"}
V0_TOP_LEVEL = {"training"}
# v0's repository, scripts, tools, and trees (its env/setup.sh and README name the trees).
V0_TEXT = (
    "mlir-corpus-pipeline",
    "mlir_corpus_pipeline",
    "lower_one.sh",
    "lower_all.sh",
    "build-llvm.sh",
    "env/setup.sh",
    "parse_snapshots",
    "build_parquet",
    "mlir_to_python",
    "corpus-alpha01",
    "llvm-cir",
)


def docstring_nodes(tree: ast.AST) -> set[int]:
    """Return the ids of the module, class, and function docstring nodes of `tree`."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                if isinstance(first.value.value, str):
                    found.add(id(first.value))
    return found


def v0_references(source: str, where: str) -> list[str]:
    """Return `<where>:<line> <what>` for each v0 import and each non-docstring string naming v0 in `source`."""
    tree = ast.parse(source)
    skip = docstring_nodes(tree)
    hits: list[str] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
        for name in names:
            parts = name.split(".")
            if V0_MODULES & set(parts) or parts[0] in V0_TOP_LEVEL:
                hits.append(f"{where}:{node.lineno} import {name}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            hits += [f"{where}:{node.lineno} {text!r}" for text in V0_TEXT if text in node.value]
    return hits


def test_the_scanner_finds_synthetic_v0_references_and_passes_docstrings() -> None:
    source = (
        '"""SYNTHETIC: moved from v0\'s tools/parse_snapshots.py:71 (mlir-corpus-pipeline)."""\n'
        "import subprocess\n"
        "from tools import build_parquet\n"
        "import training.data\n"
        "subprocess.run(['bash', 'scripts/lower_one.sh'])\n"
    )
    assert v0_references(source, "synthetic.py") == [
        "synthetic.py:3 import tools.build_parquet",
        "synthetic.py:4 import training.data",
        "synthetic.py:5 'lower_one.sh'",
    ]


def test_lassi_corpus_is_a_package_of_modules() -> None:
    assert (CORPUS / "__init__.py").is_file(), f"missing package {CORPUS.relative_to(REPO).as_posix()}/"
    assert sorted(CORPUS.rglob("*.py")), "lassi/corpus/ holds no module"


def test_nothing_in_lassi_imports_v0_or_names_its_scripts() -> None:
    assert (CORPUS / "__init__.py").is_file(), "lassi/corpus/ does not exist yet (task P5.11)"
    hits: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        where = path.relative_to(REPO).as_posix()
        hits += v0_references(path.read_bytes().decode("utf-8"), where)
    assert not hits, "v0 is imported or named at: " + "; ".join(hits)
