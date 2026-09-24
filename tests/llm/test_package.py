"""Tests for the lassi.llm package as a whole (P0.4).

Importing lassi.llm registers the three LLM backends (bible Component
Interfaces, LLMBackend row) and re-exports ServingError, the backend classes,
and model_info. Every backend has the LLMBackend shape: class attributes
`name` and `capabilities`, an instance `model_id`, and `complete(messages,
sampling)`. model_info builds the Trial `model` field (bible Result Record),
which is how sampling parameters land in the record. The package docstring and
the registry docstring state the construction convention the runner relies
on. The package modules are documented, typed, plain ASCII, use only the
standard library, and name no project (Agent Rule 3, Readability Standards).
No value in this module is a measurement.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from lassi import llm
from lassi.core import record
from lassi.core.capabilities import Component
from lassi.core.interfaces import LLMBackend, Sampling
from lassi.core.registry import DEFAULT_REGISTRY

if TYPE_CHECKING:
    from conftest import StubServer

MockBackend = llm.MockBackend
OllamaBackend = llm.OllamaBackend
OpenAICompatBackend = llm.OpenAICompatBackend
ServingError = llm.ServingError
model_info = llm.model_info

REPO = Path(__file__).resolve().parents[2]
LLM_DIR = REPO / "lassi" / "llm"
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
PROJECT_NAMES = ("lassi-repro", "lassi-ee", "lassi-df", "hecbench", "qwen", "wizardcoder", "a100", "mi300x", "gpt-oss")

# The modules the P0.4 contract names, plus any other module found in lassi/llm/.
NAMED_MODULES = ("lassi.llm", "lassi.llm.mock", "lassi.llm.openai_compat", "lassi.llm.ollama", "lassi.llm._http")
FOUND_MODULES = tuple(
    "lassi.llm" if path.stem == "__init__" else f"lassi.llm.{path.stem}" for path in sorted(LLM_DIR.glob("*.py"))
)
LLM_MODULES = tuple(dict.fromkeys(NAMED_MODULES + FOUND_MODULES))

BACKENDS = {"mock": MockBackend, "openai_compat": OpenAICompatBackend, "ollama": OllamaBackend}

REGISTRY_PARAGRAPH = (
    "The runner, never the registry, constructs components. A component bound by a kind section is built as "
    "`factory(**binding.config)`; an LLM backend as `factory(model_id)`, with keyword settings left at their "
    "defaults unless the caller passes them."
)

SAMPLING = Sampling(0.2, 0.9, 1024)
TRIAL_ID = "fixture-project/fixture-arm/fixture-suite/omp-cuda/fixture-item/run01"
RECIPE_HASH = "0123456789abcdef" * 4


def build_backends(stub: StubServer) -> dict[str, Any]:
    """Return one instance of each backend; the HTTP backends point at the stub."""
    return {
        "mock": MockBackend("mock-fixture"),
        "openai_compat": OpenAICompatBackend("fixture/coder-a", base_url=f"{stub.url}/v1"),
        "ollama": OllamaBackend("coder:7b-q8_0", base_url=stub.url),
    }


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


def normalized(text: str | None) -> str:
    """Return text with every whitespace run collapsed to one space."""
    return " ".join((text or "").split())


# ---------------------------------------------------------------------------
# Registration and re-exports


def test_import_registers_all_three_backends() -> None:
    names = DEFAULT_REGISTRY.names("LLMBackend")
    assert {"mock", "openai_compat", "ollama"} <= set(names), names
    for name, cls in BACKENDS.items():
        entry = DEFAULT_REGISTRY.get("LLMBackend", name)
        assert entry.factory is cls
        assert entry.capabilities == cls.capabilities


def test_package_reexports() -> None:
    modules = {name: importlib.import_module(f"lassi.llm.{name}") for name in BACKENDS}
    assert llm.MockBackend is modules["mock"].MockBackend
    assert llm.OpenAICompatBackend is modules["openai_compat"].OpenAICompatBackend
    assert llm.OllamaBackend is modules["ollama"].OllamaBackend
    assert llm.ServingError is ServingError
    assert llm.model_info is model_info


def test_serving_error_is_a_runtime_error() -> None:
    assert issubclass(ServingError, RuntimeError)
    assert inspect.getdoc(ServingError)


# ---------------------------------------------------------------------------
# The LLMBackend shape


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_backend_has_the_llm_backend_shape(stub_server: StubServer, name: str) -> None:
    cls = BACKENDS[name]
    assert cls.name == name
    assert isinstance(cls.capabilities, frozenset) and "chat" in cls.capabilities
    assert all(isinstance(capability, str) for capability in cls.capabilities)
    assert inspect.getdoc(cls)
    params = list(inspect.signature(cls.complete).parameters)
    assert params == [*inspect.signature(LLMBackend.complete).parameters], params
    instance = build_backends(stub_server)[name]
    assert isinstance(instance, Component)
    assert isinstance(instance.model_id, str) and "model_id" in vars(instance)
    assert stub_server.requests == []


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_capabilities_are_class_attributes(name: str) -> None:
    cls = BACKENDS[name]
    assert "name" in vars(cls) and "capabilities" in vars(cls)


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_registry_factory_builds_a_backend_from_the_model_id_alone(name: str) -> None:
    # The construction convention: the runner builds an LLM backend as factory(model_id).
    backend = DEFAULT_REGISTRY.get("LLMBackend", name).factory("fixture/model-z")
    assert type(backend) is BACKENDS[name]
    assert backend.model_id == "fixture/model-z"


# ---------------------------------------------------------------------------
# model_info: the Trial model field


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_model_info_is_the_trial_model_field(stub_server: StubServer, name: str) -> None:
    backend = build_backends(stub_server)[name]
    info = model_info(backend, SAMPLING)
    assert info == record.ModelInfo(backend=name, id=backend.model_id, sampling=SAMPLING)
    bench = record.BenchItem(suite="fixture-suite", item="fixture-item", split="eval", direction="omp-cuda")
    # Synthetic provenance, required on every Trial; the commit is not a commit of this repository.
    provenance = record.Provenance(
        commit="0" * 40, dirty=False, device="none (compile only)", sdk=None, date="2026-09-23T12:34:56+00:00"
    )
    trial = record.Trial(
        trial_id=TRIAL_ID, recipe_hash=RECIPE_HASH, provenance=provenance, bench_item=bench, model=info
    )
    assert record.to_dict(trial)["model"] == {
        "backend": name,
        "id": backend.model_id,
        "sampling": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 1024},
    }
    assert stub_server.requests == []


def test_model_info_carries_other_sampling_values(stub_server: StubServer) -> None:
    backend = build_backends(stub_server)["openai_compat"]
    info = model_info(backend, Sampling(0.0, 1.0, 16384))
    assert record.to_dict(info) == {
        "backend": "openai_compat",
        "id": "fixture/coder-a",
        "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 16384},
    }


# ---------------------------------------------------------------------------
# The construction convention


def test_registry_docstring_states_the_construction_convention() -> None:
    registry = importlib.import_module("lassi.core.registry")
    assert normalized(REGISTRY_PARAGRAPH) in normalized(registry.__doc__)


def test_package_docstring_states_the_construction_convention() -> None:
    doc = normalized(llm.__doc__)
    assert normalized(REGISTRY_PARAGRAPH) in doc, doc
    assert "Construction sends no request." in doc, doc


# ---------------------------------------------------------------------------
# Module hygiene (Agent Rule 3, Readability Standards, no new dependency)


def test_named_modules_exist() -> None:
    missing = [name for name in NAMED_MODULES if importlib.util.find_spec(name) is None]
    assert not missing, missing


@pytest.mark.parametrize("name", LLM_MODULES)
def test_llm_modules_documented_typed_ascii(name: str) -> None:
    tree = ast.parse(module_source(name))
    assert ast.get_docstring(tree), f"{name} has no module docstring"
    undocumented = [qual for qual, node in public_defs(tree) if not ast.get_docstring(node)]
    assert not undocumented, f"{name}: no docstring on {undocumented}"
    untyped = [qual for qual, node in public_defs(tree) if isinstance(node, FUNCTION_NODES) and not is_typed(node)]
    assert not untyped, f"{name}: missing type hints on {untyped}"
    assert "from __future__ import annotations" in module_source(name), name


@pytest.mark.parametrize("name", LLM_MODULES)
def test_llm_modules_hold_no_project_code(name: str) -> None:
    source = module_source(name)
    lowered = source.lower()
    found = [word for word in PROJECT_NAMES if word in lowered]
    assert not found, f"{name} names projects: {found}"
    assert "projects" not in imported_roots(ast.parse(source)), f"{name} imports the projects package"


@pytest.mark.parametrize("name", LLM_MODULES)
def test_llm_modules_use_only_the_standard_library(name: str) -> None:
    roots = imported_roots(ast.parse(module_source(name)))
    third_party = sorted(root for root in roots if root not in sys.stdlib_module_names and root != "lassi")
    assert not third_party, f"{name} imports {third_party}"
