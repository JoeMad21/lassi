"""Fixtures for the lassi-2024 prompt set and context pack tests (task P1.3).

Bible: Source Papers (LASSI pipeline steps 2 to 4), Design Principles 3 and 4,
Readability Standards (Prompts row), Repository Layout (`assets/prompts`,
`assets/context`).

This module provides five fixtures:

- `manifests`: reads a tree's MANIFEST.yaml and checks the fields every
  manifest carries (Manifests.entries documents them).
- `upstream`: the pinned upstream checkout at `third_party/LASSI` and an
  independent oracle built here with `ast` and `json`, never by importing or
  running upstream code. Tests that need it skip with a reason that names
  tools/extract_lassi_assets.py when the checkout is absent or not at
  74b4681.
- `tool_module`: tools/extract_lassi_assets.py loaded as a module, for tests
  of its parsing helpers on synthetic input.
- `extractor`: runs tools/extract_lassi_assets.py in this process through its
  `main(argv)` with `--upstream <checkout>` and `--out <assets root>`, and
  returns the exit status and everything it printed.
- `extraction`: one extraction of the pinned checkout into a session
  temporary directory, run on first use. A failed extraction fails the test
  that asked for it (an assertion, not a setup error).

The oracle's view of what the stages use (the `required` literals) is the
set of non-blank string literals that are direct operands of the
assignments building a prompt, a compile command, or the report of a run
(`content_prompt`, `content`, the correction intro and outro,
`code_compiler`, `code_compiler_kwds`, `return_result`) inside the notebook
functions that build the summary and description prompts, assemble the
translation and correction prompts, report a run (the error text of the
execute-error prompt), and set up the experiment (each function's own body,
not the rest of its cell). Whitespace-only joiners are left to the stages.

The oracle names every fragment by its key alone (Upstream.value_of): a
dictionary key is `<dictionary>.<entry>`, and a notebook key is the stage
role NOTEBOOK_KEYS gives the literal at its position, in source order, among
the literals joined into one name. NOTEBOOK_KEYS is this suite's own
statement of the keys the stages read; the extractor keeps its own table,
and a key swapped there fails here.

No upstream text is copied here (OQ-018); identifiers such as cell function
names and dictionary attribute names are not upstream prose. Generated text
lives only under pytest's temporary directories.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = REPO / "third_party" / "LASSI"
UPSTREAM_PIN = "74b46812523f2ff79b53b6880a4521690d7478b0"
TOOL = REPO / "tools" / "extract_lassi_assets.py"
DICTIONARY_FILE = "prompt_dictionary.py"
NOTEBOOK_FILE = "LASSI_pipeline_v0.ipynb"

# The prompt set and the two context packs, each with the upstream dictionary entry the pack holds.
PROMPT_SET = "lassi-2024"
PACK_SOURCES = {
    "openmp-4.0-card": ("codeknowledge_dict", "omp"),
    "cuda-12.5-ch5": ("codeknowledge_dict", "cuda"),
}
TREES = (f"prompts/{PROMPT_SET}", *(f"context/{name}" for name in PACK_SOURCES))
MANIFEST = "MANIFEST.yaml"
KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SHA256_RE = re.compile(r"[0-9a-f]{64}")

# The notebook functions whose literals the stages use, and the names their prompt text is assigned to.
STAGE_FUNCTIONS = ("code_knowledge_llm", "code_description_llm", "auto_code_llmgeneration_pipeline", "execute_code")
SETUP_FUNCTION = "experimental_setup"
PROMPT_TARGETS = frozenset({
    "content_prompt", "content", "error_correction_prompt_intro", "error_correction_prompt_outro",
    "code_compiler", "code_compiler_kwds", "return_result",
})

# The stage-facing key of each notebook literal, per (function, assigned name): key i names the i-th non-blank
# literal joined with `+` into that name, in source order.
NOTEBOOK_KEYS: dict[tuple[str, str], tuple[str, ...]] = {
    ("code_knowledge_llm", "content_prompt"): (
        "summarize_context.intro", "summarize_context.omp_note", "summarize_context.outro"),
    ("code_description_llm", "content_prompt"): ("describe_source.intro",),
    ("auto_code_llmgeneration_pipeline", "content"): (
        "generate.context_open", "generate.context_close", "generate.summary_open", "generate.summary_close",
        "generate.context_lead", "generate.description_lead", "generate.no_context_lead", "generate.request_lead"),
    ("auto_code_llmgeneration_pipeline", "error_correction_prompt_intro"): (
        "correct.run_error_head", "correct.run_error_tail", "correct.compile_error_head",
        "correct.compile_error_tail"),
    ("auto_code_llmgeneration_pipeline", "error_correction_prompt_outro"): ("correct.outro",),
    ("execute_code", "return_result"): (
        "execute.ok", "execute.exit_lead", "execute.segfault", "execute.exception_lead", "execute.stderr_lead"),
    ("experimental_setup", "code_compiler"): ("setup.cuda.compiler", "setup.omp.compiler"),
    ("experimental_setup", "code_compiler_kwds"): ("setup.cuda.flags", "setup.omp.flags"),
}

SKIP_REASON = (
    "needs the upstream checkout at third_party/LASSI on commit 74b4681 for tools/extract_lassi_assets.py;"
    " check it out first (task P1.1)"
)


def git(*args: str, cwd: Path = REPO, check: bool = True) -> subprocess.CompletedProcess:
    """Run git with `args` in `cwd` and return the finished process (text mode)."""
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=check)


def checkout_at_pin(path: Path) -> bool:
    """Return True when `path` is its own git checkout whose HEAD is the upstream pin."""
    if not (path / ".git").exists():
        return False
    done = git("rev-parse", "HEAD", cwd=path, check=False)
    return done.returncode == 0 and done.stdout.strip() == UPSTREAM_PIN


def _operands(expr: ast.expr) -> list[ast.expr]:
    """Flatten a chain of `+` into its operands, left to right."""
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        return _operands(expr.left) + _operands(expr.right)
    return [expr]


@dataclass
class Upstream:
    """The pinned checkout and the values parsed from it with `ast` (the test's own oracle)."""

    path: Path
    dictionaries: dict[str, dict[str, str]] = field(default_factory=dict, repr=False)
    cells: dict[str, str] = field(default_factory=dict, repr=False)
    trees: dict[str, ast.Module] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.dictionaries = self._parse_dictionaries()
        notebook = json.loads((self.path / NOTEBOOK_FILE).read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            source = cell.get("source", "")
            self.cells[cell["id"]] = source if isinstance(source, str) else "".join(source)
            if cell["cell_type"] == "code":
                with contextlib.suppress(SyntaxError):
                    self.trees[cell["id"]] = ast.parse(self.cells[cell["id"]])

    def _parse_dictionaries(self) -> dict[str, dict[str, str]]:
        """Return {attribute name: {key: value}} for every dict literal assigned in prompt_dictionary.py."""
        tree = ast.parse((self.path / DICTIONARY_FILE).read_text(encoding="utf-8"))
        found: dict[str, dict[str, str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                target = node.targets[0]
                name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
                values: dict[str, str] = {}
                for key, value in zip(node.value.keys, node.value.values, strict=True):
                    assert isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
                    assert isinstance(value.value, str), f"{name}[{key.value!r}] is not a string"
                    values[str(key.value)] = value.value
                found[name] = values
        for name, key in PACK_SOURCES.values():
            present = key in found.get(name, {})
            assert present, f"the oracle found no {name}[{key!r}] in {DICTIONARY_FILE}"
        return found

    def dictionary_values(self, *, packs: bool) -> set[str]:
        """Return every dictionary value: only the pack dictionary's when `packs`, else all the others."""
        pack_dicts = {name for name, _ in PACK_SOURCES.values()}
        return {value for name, values in self.dictionaries.items() if (name in pack_dicts) == packs
                for value in values.values()}

    def cell_of(self, function: str) -> str:
        """Return the id of the code cell that defines `function` at its top level."""
        hits = [cell_id for cell_id, tree in self.trees.items()
                if any(isinstance(node, ast.FunctionDef) and node.name == function for node in tree.body)]
        assert len(hits) == 1, f"the oracle expects one cell defining {function}, found {hits}"
        return hits[0]

    def function_node(self, function: str) -> ast.FunctionDef:
        """Return the top-level definition of `function` in the one code cell that defines it."""
        body = self.trees[self.cell_of(function)].body
        return next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == function)

    def assigned(self, function: str, names: frozenset[str]) -> list[str]:
        """Return the non-blank string literals joined with `+` into any of `names` inside `function`, in order.

        Order is source order: assignment statements by position, then each
        statement's operands left to right.
        """
        statements: list[ast.Assign | ast.AugAssign] = []
        for node in ast.walk(self.function_node(function)):
            targets = node.targets if isinstance(node, ast.Assign) else (
                [node.target] if isinstance(node, ast.AugAssign) else [])
            if any(isinstance(target, ast.Name) and target.id in names for target in targets):
                statements.append(node)
        statements.sort(key=lambda node: (node.lineno, node.col_offset))
        return [operand.value for node in statements for operand in _operands(node.value)
                if isinstance(operand, ast.Constant) and isinstance(operand.value, str) and operand.value.strip()]

    def required(self) -> dict[str, set[str]]:
        """Return {cell id: literals the stages use} for the stage functions and experimental_setup."""
        found: dict[str, set[str]] = {}
        for function in (*STAGE_FUNCTIONS, SETUP_FUNCTION):
            literals = set(self.assigned(function, PROMPT_TARGETS))
            assert literals, f"the oracle found no prompt literals in {function}"
            found.setdefault(self.cell_of(function), set()).update(literals)
        return found

    def notebook_values(self) -> dict[str, tuple[str, str]]:
        """Return {key: (cell id, literal)} in NOTEBOOK_KEYS order; key i of a (function, name) is literal i."""
        found: dict[str, tuple[str, str]] = {}
        for (function, name), keys in NOTEBOOK_KEYS.items():
            literals = self.assigned(function, frozenset({name}))
            counted = len(literals) == len(keys)
            assert counted, f"{function} joins {len(literals)} literals into {name}; NOTEBOOK_KEYS names {len(keys)}"
            cell_id = self.cell_of(function)
            found.update((key, (cell_id, text)) for key, text in zip(keys, literals, strict=True))
        return found

    def value_of(self, entry: dict) -> str:
        """Return the upstream value a manifest entry's key names.

        A dictionary key `<dictionary>.<entry>` is split once, on the first
        dot. A notebook key must be in NOTEBOOK_KEYS, and the entry's cell
        must be the cell that defines the key's function.
        """
        key = entry["key"]
        if entry["source"] == DICTIONARY_FILE:
            name, _, item = key.partition(".")
            known = item in self.dictionaries.get(name, {})
            assert known, f"{key}: {DICTIONARY_FILE} has no dictionary {name!r} with an entry {item!r}"
            return self.dictionaries[name][item]
        values = self.notebook_values()
        assert key in values, f"{key}: not a notebook key the stages read (NOTEBOOK_KEYS)"
        cell_id, text = values[key]
        assert entry.get("cell") == cell_id, f"{key}: manifest cell {entry.get('cell')!r}, expected {cell_id!r}"
        return text

    def setup_literals(self) -> set[str]:
        """Return the compiler and flag literals of experimental_setup."""
        return self.required()[self.cell_of(SETUP_FUNCTION)]


class Manifests:
    """Reads a tree's MANIFEST.yaml and checks the shape every manifest shares."""

    def entries(self, tree: Path) -> list[dict]:
        """Return the entries of `<tree>/MANIFEST.yaml` after checking its text and each entry's fields.

        The manifest is plain ASCII YAML with LF newlines whose top level maps
        `entries` to a list. Each entry has `key` (one plain path segment; the
        file is `<key>.txt` beside the manifest), `source` (the upstream file),
        `cell` (the notebook cell id, for notebook entries only), and `sha256`
        (64 lowercase hex digits of the file's bytes). Keys are unique.
        """
        path = tree / MANIFEST
        assert path.is_file(), f"{path} is missing"
        data = path.read_bytes()
        assert data.isascii() and b"\r" not in data, f"{path} is not plain ASCII with LF newlines"
        loaded = yaml.safe_load(data.decode("ascii"))
        assert isinstance(loaded, dict) and isinstance(loaded.get("entries"), list), f"{path}: no entries list"
        entries = loaded["entries"]
        assert entries, f"{path}: the entries list is empty"
        for entry in entries:
            assert isinstance(entry, dict), f"{path}: an entry is not a mapping: {entry!r}"
            assert isinstance(entry.get("key"), str) and KEY_RE.fullmatch(entry["key"]), f"{path}: bad key {entry!r}"
            assert entry.get("source") in (DICTIONARY_FILE, NOTEBOOK_FILE), f"{path}: bad source {entry!r}"
            if entry["source"] == NOTEBOOK_FILE:
                assert isinstance(entry.get("cell"), str) and entry["cell"], f"{path}: {entry['key']} needs a cell id"
            else:
                assert entry.get("cell") is None, f"{path}: {entry['key']} is no notebook entry but names a cell"
            assert isinstance(entry.get("sha256"), str) and SHA256_RE.fullmatch(entry["sha256"]), (
                f"{path}: {entry['key']} has no sha256 of 64 lowercase hex digits")
        keys = [entry["key"] for entry in entries]
        assert len(keys) == len(set(keys)), f"{path}: repeated keys"
        return entries


@pytest.fixture(scope="session")
def manifests() -> Manifests:
    """Return the manifest reader shared by the extractor and loader tests."""
    return Manifests()


@pytest.fixture(scope="session")
def upstream() -> Upstream:
    """Return the pinned upstream checkout with its oracle, or skip naming the extractor."""
    if not checkout_at_pin(UPSTREAM_DIR):
        pytest.skip(SKIP_REASON)
    return Upstream(UPSTREAM_DIR)


def load_tool() -> ModuleType:
    """Load tools/extract_lassi_assets.py as a module; fail clearly when it does not exist yet."""
    assert TOOL.is_file(), "tools/extract_lassi_assets.py is missing (task P1.3 writes it)"
    spec = importlib.util.spec_from_file_location("extract_lassi_assets", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert callable(getattr(module, "main", None)), "tools/extract_lassi_assets.py has no main(argv)"
    return module


@pytest.fixture(scope="session")
def tool_module() -> ModuleType:
    """Return tools/extract_lassi_assets.py loaded as a module, for tests of its parsing helpers."""
    return load_tool()


class Extractor:
    """Runs the extractor's main(argv) in this process and captures what it prints."""

    def run(self, upstream: Path, out: Path) -> tuple[int, str]:
        """Extract from `upstream` into the assets root `out`; return (exit status, stdout and stderr)."""
        tool = load_tool()
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            try:
                status = tool.main(["--upstream", str(upstream), "--out", str(out)])
            except SystemExit as stop:
                if isinstance(stop.code, str):
                    print(stop.code, file=sys.stderr)
                    status = 1
                else:
                    status = stop.code or 0
        return int(status or 0), sink.getvalue()


@pytest.fixture(scope="session")
def extractor() -> Extractor:
    """Return the in-process runner of tools/extract_lassi_assets.py."""
    return Extractor()


@dataclass
class Extraction:
    """One extraction of the pinned checkout into `out`, run on first use."""

    upstream: Upstream
    out: Path
    runner: Extractor
    problem: str | None = None
    done: bool = False

    def root(self) -> Path:
        """Return the assets root the extractor wrote, running it once; fail the calling test if it failed."""
        if not self.done and self.problem is None:
            try:
                status, output = self.runner.run(self.upstream.path, self.out)
            except AssertionError as error:
                self.problem = str(error)
            else:
                if status != 0:
                    self.problem = f"the extractor exited {status}: {output[-2000:]}"
                else:
                    self.done = True
        if self.problem is not None:
            raise AssertionError(self.problem)
        return self.out


@pytest.fixture(scope="session")
def extraction(upstream: Upstream, extractor: Extractor, tmp_path_factory: pytest.TempPathFactory) -> Extraction:
    """Return the session's extraction of the pinned checkout (generated into a temporary assets root)."""
    return Extraction(upstream, tmp_path_factory.mktemp("assets"), extractor)
