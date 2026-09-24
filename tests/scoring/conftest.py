"""Fixtures for the Sim-T and Sim-L tests (task P1.8).

Bible: Source Papers (LASSI results table; quirk table, Sim-T row), Evaluation
Protocol (LASSI reproduction row), Repository Layout (`scoring/`).

This module provides these fixtures:

- `upstream_pin`: the upstream commit and checkout path, read from the pin
  manifest `assets/upstream/lassi.yaml` (task P1.1 owns it), so a pin change
  is one edit.
- `upstream_root`: the pinned upstream checkout at that path. Tests that need
  it skip, naming `uv run tools/fetch_upstream.py`, when it is not fetched.
  They fail when it sits at another commit or its notebook or sources are
  edited (`third_party/` is read-only).
- `upstream_mains`: the 20 upstream `*_main` sources. They are read the way the
  notebook reads them (text mode, universal newlines) as UTF-8, the encoding
  of the build host.
- `upstream`: upstream's similarity functions. Only the function definitions
  of the notebook code cell that defines `token_similarity` and
  `compare_lines_with_reordering` are taken, parsed with `ast`. Nothing else
  in the notebook is imported or run. They are compiled into a namespace that
  holds the standard modules they name (tokenize, difflib, BytesIO), a
  `tiktoken` stub that refuses any use (the tokenizer import is stubbed), and
  a `print` recorder that counts upstream's error-branch messages without
  keeping their text.

Upstream code runs only inside `UpstreamGuard`. While the guard is active,
every way to start a process or open a socket is replaced by a function that
records the attempt and raises. On exit the guard fails the test if anything
was attempted. Recording matters because upstream's tokenize helper catches
every Exception, which would swallow a raised error. The P1.9 replay needs
the same guard. It lives here until a shared test module holds it.

No upstream text is copied here (OQ-018). The upstream code runs only in
memory during the test.
"""

from __future__ import annotations

import ast
import builtins
import difflib
import importlib
import io
import json
import subprocess
import tokenize
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
PIN_MANIFEST = REPO / "assets" / "upstream" / "lassi.yaml"
NOTEBOOK = "LASSI_pipeline_v0.ipynb"
MAINS_PARTS = ("translated_code", "input_codes", "HeCBench")
APPS = (
    "atomicCost", "bsearch", "colorwheel", "dense-embedding", "entropy",
    "jacobi", "layout", "matrix-rotate", "pathfinder", "randomAccess",
)
EXTENSIONS = {"omp": "cpp", "cuda": "cu"}
SIM_T_NAME = "token_similarity"
SIM_L_NAME = "compare_lines_with_reordering"

# Every entry point that starts a process or opens a socket, by module.
GUARDED: dict[str, tuple[str, ...]] = {
    "subprocess": ("Popen", "run", "call", "check_call", "check_output", "getoutput", "getstatusoutput"),
    "os": (
        "system", "popen", "startfile", "fork", "forkpty", "posix_spawn", "posix_spawnp",
        "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
        "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
    ),
    "socket": (
        "socket", "create_connection", "create_server", "socketpair", "fromfd",
        "getaddrinfo", "gethostbyname", "gethostbyname_ex",
    ),
    "_socket": ("socket",),
    "_winapi": ("CreateProcess",),
    "_posixsubprocess": ("fork_exec",),
}


class GuardRefusal(RuntimeError):
    """Raised in place of a process start or socket open while the guard is active."""


class UpstreamGuard:
    """Context manager: refuse and record every process start and socket open, then fail if any was tried."""

    def __init__(self) -> None:
        self.attempts: list[str] = []
        self._patch: pytest.MonkeyPatch | None = None

    def _refuser(self, name: str) -> Callable[..., Any]:
        """Return a stand-in for `name` that records the attempt and raises GuardRefusal."""

        def refuse(*args: Any, **kwargs: Any) -> Any:
            self.attempts.append(name)
            raise GuardRefusal(f"upstream guard: {name} refused")

        return refuse

    def __enter__(self) -> UpstreamGuard:
        self._patch = pytest.MonkeyPatch()
        for module_name, names in GUARDED.items():
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            for name in names:
                if hasattr(module, name):
                    self._patch.setattr(module, name, self._refuser(f"{module_name}.{name}"))
        return self

    def __exit__(self, *exc: object) -> None:
        assert self._patch is not None
        self._patch.undo()
        if self.attempts:
            pytest.fail(f"upstream code tried to start a process or open a socket: {self.attempts}")


class _RefusedModule(types.ModuleType):
    """Stand-in for a module upstream imports but P1.8 never uses (the tokenizer import)."""

    def __getattr__(self, name: str) -> Any:
        raise GuardRefusal(f"upstream guard: {self.__name__}.{name} is stubbed and must not be used")


@dataclass
class Upstream:
    """Upstream's similarity functions, callable only under UpstreamGuard."""

    namespace: dict[str, Any]
    printed: list[int] = field(default_factory=list)

    def call(self, name: str, *args: Any) -> Any:
        """Call the upstream function `name` with `args` under the guard and return its result."""
        with UpstreamGuard():
            return self.namespace[name](*args)

    def sim_t(self, reference: str, candidate: str) -> float:
        """Return upstream's token_similarity(reference, candidate, "tokenize")."""
        return self.call(SIM_T_NAME, reference, candidate, "tokenize")

    def sim_l(self, reference: str, candidate: str) -> float:
        """Return upstream's compare_lines_with_reordering(reference, candidate)."""
        return self.call(SIM_L_NAME, reference, candidate)

    def prints(self) -> int:
        """Return how many times upstream code has called print so far (its error branches print)."""
        return len(self.printed)


def git(root: Path, *args: str) -> str:
    """Run git with `args` in `root` and return its stripped stdout (git only reads trees here)."""
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)
    return done.stdout.strip()


@dataclass(frozen=True)
class UpstreamPin:
    """The upstream commit and its checkout path (relative to the repository root), from the pin manifest."""

    commit: str
    path: str

    @property
    def hint(self) -> str:
        """Return the instruction that fetches the pinned upstream checkout."""
        return f"run `uv run tools/fetch_upstream.py` to fetch upstream LASSI at {self.commit[:7]} into {self.path}"

    def require(self, root: Path) -> Path:
        """Return `root` when it holds the pinned, unedited checkout; skip when it is absent, fail otherwise."""
        if not (root / ".git").exists():
            pytest.skip(f"{self.path} is not fetched; {self.hint}")
        head = git(root, "rev-parse", "HEAD")
        if head != self.commit:
            pytest.fail(f"{self.path} is at {head}, not the pin {self.commit}; {self.hint}")
        edited = git(root, "status", "--porcelain", "--", NOTEBOOK, MAINS_PARTS[0])
        if edited:
            pytest.fail(f"{self.path} is edited (third_party/ is read-only):\n{edited}")
        return root


def load_pin(manifest: Path = PIN_MANIFEST) -> UpstreamPin:
    """Read the upstream `commit` and `path` from the pin manifest."""
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    return UpstreamPin(commit=str(data["commit"]), path=str(data["path"]))


def similarity_defs(notebook: dict[str, Any]) -> ast.Module:
    """Return the function definitions of the one code cell that defines both similarity functions."""
    found: list[ast.Module] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue  # cells with notebook magics are not Python; the similarity cell is
        names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        if {SIM_T_NAME, SIM_L_NAME} <= names:
            found.append(tree)
    assert len(found) == 1, f"expected one notebook cell defining both similarity functions, found {len(found)}"
    body = [node for node in found[0].body if isinstance(node, ast.FunctionDef)]
    return ast.Module(body=body, type_ignores=[])


def load_upstream(root: Path) -> Upstream:
    """Compile upstream's similarity functions from the pinned notebook into a guarded namespace."""
    notebook_path = root / NOTEBOOK
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = compile(similarity_defs(notebook), filename=f"{notebook_path.as_posix()}:similarity-cell", mode="exec")
    upstream = Upstream(namespace={})
    upstream.namespace.update({
        "__builtins__": builtins,
        "__name__": "upstream_similarity",
        "difflib": difflib,
        "tokenize": tokenize,
        "BytesIO": io.BytesIO,
        "tiktoken": _RefusedModule("tiktoken"),
        "print": lambda *args, **kwargs: upstream.printed.append(1),
    })
    with UpstreamGuard():
        exec(code, upstream.namespace)
    for name in (SIM_T_NAME, SIM_L_NAME):
        assert callable(upstream.namespace.get(name)), f"the notebook cell did not define {name}"
    return upstream


@pytest.fixture(scope="session")
def upstream_pin() -> UpstreamPin:
    """Return the upstream pin read from assets/upstream/lassi.yaml."""
    return load_pin()


@pytest.fixture(scope="session")
def upstream_root(upstream_pin: UpstreamPin) -> Path:
    """Return the pinned, unedited upstream checkout; skip when it is not fetched."""
    return upstream_pin.require(REPO / upstream_pin.path)


@pytest.fixture(scope="session")
def upstream(upstream_root: Path) -> Upstream:
    """Return upstream's similarity functions loaded from the pinned notebook under the guard."""
    return load_upstream(upstream_root)


@pytest.fixture(scope="session")
def upstream_mains(upstream_root: Path) -> dict[tuple[str, str], str]:
    """Return the 20 upstream `*_main` sources keyed by (app, language), read as the notebook reads them."""
    base = upstream_root.joinpath(*MAINS_PARTS)
    mains: dict[tuple[str, str], str] = {}
    for app in APPS:
        for lang, ext in EXTENSIONS.items():
            path = base / app / f"{app}-{lang}_main.{ext}"
            assert path.is_file(), f"missing upstream source {path}"
            with open(path, encoding="utf-8") as handle:  # text mode, universal newlines, as upstream
                mains[(app, lang)] = handle.read()
    assert len(mains) == 20
    return mains


@pytest.fixture(scope="session")
def upstream_texts(upstream_root: Path) -> list[str]:
    """Return the string constants of prompt_dictionary.py and every notebook cell source (for OQ-018 checks)."""
    tree = ast.parse((upstream_root / "prompt_dictionary.py").read_text(encoding="utf-8"))
    texts = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    notebook = json.loads((upstream_root / NOTEBOOK).read_text(encoding="utf-8"))
    texts.extend("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
    return texts


@pytest.fixture
def guard() -> Iterator[type[UpstreamGuard]]:
    """Give a test the guard class (for the guard's own checks)."""
    yield UpstreamGuard

