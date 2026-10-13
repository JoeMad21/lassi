"""Fakes, recipes, and bench sources for the `lassi run` tests of hf_local (task P17.4).

tests/llm/test_hf_local_runs.py (framework extra) and
tests/llm/test_hf_local_settings.py (no extra) run recipes that bind the real
hf_local class; every other component is a fake: the toolchain writes a
PLACEHOLDER artifact and compiles nothing, the executor is compile only and
runs nothing, and the cpu probe answers with SYNTHETIC facts. The bench
sources are SYNTHETIC. No value here is a measurement.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from tiny_hf import REPO_ID, REVISION

from lassi.bench import load_suite
from lassi.core import runner as runner_module
from lassi.core.devices import HostFacts
from lassi.core.interfaces import BuildResult, Limits, RunResult
from lassi.core.registry import Registry
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.stages import CompileLoopStage, GenerateStage

REPO = Path(__file__).resolve().parents[2]
SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
ITEM = "layout"
SEED = 7
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
CPU_NAME = "SYNTHETIC CPU model 9000"
CPU = {"kind": "cpu"}
SOURCES = {
    "omp": '#include <cstdio>\nint main() {\n#pragma omp target\n  { }\n  std::printf("done\\n");\n}\n',
    "cuda": "#include <cstdio>\n__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n",
}


def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Unset the gate and compile variables, point TMPDIR at a test directory, fake git; return a fresh HF_HOME."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    answers = {"rev-parse": FAKE_COMMIT + "\n", "status": ""}
    monkeypatch.setattr(runner_module, "_git", lambda *args: answers[args[0]])
    home = tmp_path / "hf-home"
    home.mkdir()
    monkeypatch.setenv("HF_HOME", str(home))
    return home


class FakeToolchain:
    """A Toolchain "nvcc-sm80" without PIN: writes the files and a PLACEHOLDER artifact; compiles nothing."""

    name = "nvcc-sm80"
    capabilities = frozenset({"diagnostics"})

    def build(self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None) -> BuildResult:
        """Write every file and a PLACEHOLDER artifact, and report a clean build."""
        for path, text in [*files.items(), *(harness or {}).items()]:
            target = Path(workdir) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode("utf-8"))
        artifact = Path(workdir) / "main"
        artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
        return BuildResult(artifact=artifact, diagnostics=[])


class CompileOnly:
    """The Executor "none" of these tests: compile only, runs nothing."""

    name = "none"
    capabilities = frozenset({"compile_only"})

    def device(self) -> str:
        """Name no device, as a compile-only executor does."""
        return "none (compile only)"

    def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        """Fail the test: nothing runs on the compile-only path."""
        raise AssertionError("the compile-only path ran an executor")


@dataclass
class FakeCpuProbe:
    """A cpu probe with SYNTHETIC facts that counts its calls."""

    kind: str = "cpu"
    calls: int = 0

    def probe(self) -> HostFacts:
        """Count the call and return the SYNTHETIC facts."""
        self.calls += 1
        return HostFacts(count=8, name=CPU_NAME, memory_bytes=4096, driver=None, runtime=None)


def make_registry(backend: type) -> Registry:
    """Return a test Registry: `backend` as hf_local, the fake toolchain and executor, and the real stages."""
    registry = Registry()
    registry.register("LLMBackend", "hf_local", backend)
    registry.register("Executor", "none", CompileOnly)
    registry.register("Toolchain", "nvcc-sm80", FakeToolchain)
    registry.register("Stage", "generate", GenerateStage)
    registry.register("Stage", "compile_loop", CompileLoopStage)
    return registry


def model_section(**changes: Any) -> dict[str, Any]:
    """Return the hf_local model section: id, revision, seed, and a cpu device, with `changes` applied."""
    section: dict[str, Any] = {"backend": "hf_local", "id": REPO_ID, "revision": REVISION, "seed": SEED}
    section["device"] = dict(CPU)
    section.update(changes)
    return section


def recipe_data(model: Mapping[str, Any], *, max_tokens: int = 8, trials: int = 1) -> dict[str, Any]:
    """Return a compile-only recipe for the layout item, omp to cuda, with one model call per trial.

    loop.max_corrections is 0, so compile_loop asks for no correction: a
    reply without usable files ends the trial at correction-cap after the
    generate call.
    """
    return {
        "extends": "base",
        "model": json.loads(json.dumps(dict(model))),
        "llm": {"sampling": {"max_tokens": max_tokens}},
        "loop": {"max_corrections": 0},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": "omp", "target": "cuda"}],
        "prompts": "p0-smoke",
        "toolchain": {"cuda": "nvcc-sm80"},
        "stages": ["generate", "compile_loop"],
        "executor": {"kind": "none"},
        "trials": {"n": trials},
    }


def write_bench(root: Path) -> Path:
    """Write the item's SYNTHETIC sources where the suite manifest lays them out; return `root`."""
    spec = load_suite(SUITE_MANIFEST).items[ITEM]
    for language, text in SOURCES.items():
        layout = spec.languages[language]
        path = root / layout.dir / layout.files[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def write_recipe(directory: Path, name: str, data: Mapping[str, Any]) -> Path:
    """Write `data` as the recipe <directory>/<name>.yaml (ASCII, LF) and return its path."""
    path = directory / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


def run(
    tmp_path: Path, registry: Registry, data: Mapping[str, Any], name: str, probe: FakeCpuProbe, run_id: str = "hf"
) -> Path:
    """Write the recipe `name` and run it with `registry` and the cpu probe; return the run directory."""
    options = RunOptions(
        runs_root=tmp_path / "runs-root",
        run_id=run_id,
        bench_root=write_bench(tmp_path / "bench"),
        registry=registry,
        probes={"cpu": probe},
    )
    return run_recipe(write_recipe(tmp_path, name, data), options)


def read_ascii(path: Path) -> str:
    """Return a file's text after checking it is ASCII with LF newlines."""
    data = path.read_bytes()
    assert data.isascii() and b"\r" not in data, path
    return data.decode("ascii")


def trial_id(project: str, number: int = 1) -> str:
    """Return the id of a project's trial `number` on the tiny model (its id's '/' becomes '--')."""
    return f"{project}/{REPO_ID.replace('/', '--')}/{SUITE}/omp-cuda/{ITEM}/run{number:02d}"
