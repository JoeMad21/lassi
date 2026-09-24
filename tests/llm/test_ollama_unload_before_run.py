"""Tests that runs of generated code unload an Ollama model first (task P1.6).

Bible: Source Papers (LASSI quirk table, Ollama row: unload before each
execution, for Ollama arms only), Component Interfaces (capability rule).

- The ollama backend declares the capability `unload_before_run`, in its
  class and in its registry entry.
- End to end through the runner, against the local stub server from
  conftest.py: a trial with [baseline, generate, compile_loop, run_loop] and
  an executor that runs programs sends POST /api/generate with keep_alive 0
  once at trial start, before the first chat request and before the
  reference runs, and once more right before the attempt's run. The stage
  asks through the capability and the backend's unload(), never by checking
  the backend's type.

Every request goes to the stub on 127.0.0.1. The model reply, the bench
sources, and the program output are SYNTHETIC; fake toolchains compile
nothing and the scripted executor runs nothing. No value here is a
measurement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from lassi.bench import load_suite
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Limits, RunResult
from lassi.core.record import make_trial_id
from lassi.core.registry import DEFAULT_REGISTRY, Registry, RegistryError
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.llm.ollama import OllamaBackend

if TYPE_CHECKING:
    from conftest import StubServer

REPO = Path(__file__).resolve().parents[2]
SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
ITEM = "layout"
# A stub model id with no ':' tag, since a trial id segment allows none; the stub serves exactly this id.
MODEL_ID = "coder-7b-q8_0"
CHAT_PATH = "/api/chat"
GENERATE_PATH = "/api/generate"
UNLOAD_BEFORE_RUN = "unload_before_run"
TOOLCHAINS = {"cuda": "nvcc-sm80", "omp": "nvcpp-cc80"}
STAGES = ["baseline", "generate", "compile_loop", "run_loop"]

# SYNTHETIC sources and program output.
SOURCES = {
    "omp": "int main() {\n  return 0;\n}\n",
    "cuda": "__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n",
}
GOOD_CODE = "int main() {\n  return 0;\n}\n"
STDOUT = "Average kernel execution time (AoS): 1.5 (us)\nPASS\nAverage kernel execution time (SoA): 2.5 (us)\nPASS\n"


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate and compile variables a test could inherit, and point TMPDIR at a test directory."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CPATH", raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))


@dataclass
class Seen:
    """What the scripted executor saw: per run, its kind and how many unload requests the stub had received."""

    runs: list[tuple[str, int]] = field(default_factory=list)


def fake_toolchain(registered_as: str) -> type:
    """Return a Toolchain class without PIN that writes the files and a PLACEHOLDER artifact; compiles nothing."""

    class FakeToolchain:
        """Writes the files and reports a build."""

        name = registered_as
        capabilities = frozenset({"diagnostics"})

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Write every file and a PLACEHOLDER artifact, and report a clean build."""
            workdir = Path(workdir)
            for path, text in [*files.items(), *(harness or {}).items()]:
                target = workdir / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            (workdir / "compile.stderr").write_bytes(b"")
            artifact = workdir / "main"
            artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=[], stderr_ref="compile.stderr")

    return FakeToolchain


def scripted_executor(stub: StubServer, seen: Seen) -> type:
    """Return an Executor class that runs programs and notes how many unloads preceded each run."""

    class ScriptedExecutor:
        """Records each run and returns a SYNTHETIC clean run; runs nothing."""

        name = "scripted"
        capabilities = frozenset({"runs_code", "sandboxed"})

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Note the run's kind and the unload count so far; return a clean run."""
            kind = "reference" if Path(artifact).parent.parent.name.startswith("baseline-") else "attempt"
            seen.runs.append((kind, len(stub.requests_to("POST", GENERATE_PATH))))
            return RunResult(exit_code=0, hang=False, stdout=STDOUT, stderr="", wall_s=0.5)

    return ScriptedExecutor


def stub_ollama(stub: StubServer) -> type:
    """Return an OllamaBackend subclass that the runner builds as factory(model_id), pointed at the stub."""

    class StubOllama(OllamaBackend):
        """The real ollama backend, with the stub's base URL."""

        def __init__(self, model_id: str) -> None:
            """Point the backend at the stub."""
            super().__init__(model_id, base_url=stub.url, timeout_s=30.0)

    return StubOllama


def registered_stage(name: str) -> type:
    """Return the Stage class registered as `name`; fail clearly while it is missing."""
    try:
        return DEFAULT_REGISTRY.get("Stage", name).factory
    except RegistryError as error:
        pytest.fail(f"no stage is registered as {name!r} ({error}); task P1.6 adds the run_loop stage")


def write_bench(root: Path) -> Path:
    """Write the item's SYNTHETIC source per language where the suite manifest lays it out; return `root`."""
    spec = load_suite(SUITE_MANIFEST).items[ITEM]
    for language, text in SOURCES.items():
        layout = spec.languages[language]
        path = root / layout.dir / layout.files[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def recipe() -> dict[str, Any]:
    """Return a p0-smoke recipe for layout, CUDA to OpenMP, with the ollama backend and the scripted executor."""
    return {
        "extends": "base",
        "model": {"backend": "ollama", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": "cuda", "target": "omp"}],
        "prompts": "p0-smoke",
        "toolchain": dict(TOOLCHAINS),
        "stages": list(STAGES),
        "executor": {"kind": "scripted"},
        "trials": {"n": 1},
    }


def serve(stub: StubServer) -> None:
    """Serve the model list, one chat reply holding the target's FILE block, and the unload reply."""
    stub.serve_models([MODEL_ID])
    reply = render_file_blocks({"main.cpp": GOOD_CODE})
    stub.reply(
        "POST",
        CHAT_PATH,
        json_body={
            "model": MODEL_ID,
            "message": {"role": "assistant", "content": reply},
            "done": True,
            "prompt_eval_count": 13,
            "eval_count": 21,
        },
    )
    stub.reply("POST", GENERATE_PATH, json_body={"model": MODEL_ID, "response": "", "done": True})


def test_the_ollama_backend_declares_unload_before_run() -> None:
    assert UNLOAD_BEFORE_RUN in OllamaBackend.capabilities
    assert UNLOAD_BEFORE_RUN in DEFAULT_REGISTRY.get("LLMBackend", "ollama").capabilities


def test_an_ollama_trial_unloads_at_trial_start_and_right_before_the_attempt_runs(
    tmp_path: Path, stub_server: StubServer
) -> None:
    serve(stub_server)
    seen = Seen()
    registry = Registry()
    registry.register("LLMBackend", "ollama", stub_ollama(stub_server))
    registry.register("Executor", "scripted", scripted_executor(stub_server, seen))
    for name in TOOLCHAINS.values():
        registry.register("Toolchain", name, fake_toolchain(name))
    for name in STAGES:
        registry.register("Stage", name, registered_stage(name))
    path = tmp_path / "ollama-unload.yaml"
    path.write_bytes(yaml.safe_dump(recipe(), sort_keys=False).encode("ascii"))
    bench = write_bench(tmp_path / "bench")
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=bench, registry=registry)
    run_dir = run_recipe(path, options)

    unloads = stub_server.requests_to("POST", GENERATE_PATH)
    assert [seen_request.body for seen_request in unloads] == [{"model": MODEL_ID, "keep_alive": 0}] * 2, (
        "one unload at trial start and one before the attempt's run"
    )
    calls = stub_server.calls()
    assert calls.index(("POST", GENERATE_PATH)) < calls.index(("POST", CHAT_PATH)), "the setup unload comes first"
    references = [count for kind, count in seen.runs if kind == "reference"]
    attempts = [count for kind, count in seen.runs if kind == "attempt"]
    assert references and all(count == 1 for count in references), "the reference runs follow the setup unload only"
    assert attempts == [2], "the attempt's run follows its own unload"
    trial_id = make_trial_id("ollama-unload", MODEL_ID, SUITE, "cuda-omp", ITEM, 1)
    trial = read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir))
    assert [attempt.stage_reached for attempt in trial.attempts] == ["S5"]
