"""Tests for served backends in `lassi run` (task P17.3): the recipe keys, the model check, the serving record.

Bible: Model Serving (Serving Rules), Project Recipes (Notes), Result Record
(Storage: provenance.json is authoritative), Agent Rule 12; plans/p17-portable.md,
task P17.3; plans/LESSONS.md (Audits: a refusal before any directory needs the
output built in memory first); PHASE-NOTES P0 (the runner note on HTTP
settings) and P3 (the trial-start unload sent before the model check).

The contract these tests fix, end to end through run_recipe:

- model.base_url, model.timeout_s, and model.api_key_env reach the backend:
  the runner builds it as factory(model.id, **config), the config being the
  model keys other than backend and id. The key is read from the variable
  api_key_env names and sent as a Bearer header; the recipe names only the
  variable.
- A setting the backend refuses (its ValueError, such as a base_url with a
  query or a password) is a RunError before any directory exists, and the
  message does not quote the value.
- For a backend that declares model_check (lassi.core.capabilities
  MODEL_CHECK), the runner reads the serving record with serving() before any
  directory exists: a model the server does not list, an unreachable server,
  a record JSON cannot hold, and a record that holds the API key are each a
  RunError before any directory, and no chat or unload request is sent
  (Ollama's trial-start unload included). The trials reuse the checked entry,
  so the model list is read once per run. A backend that declares
  model_check without a callable serving() is refused before any directory.
- provenance.json holds `serving` ({base_url, model, version, version_from})
  for such a backend and no `serving` key for any other; each trial's
  provenance, trial.json, and the Parquet trials table stay as they were.
  run.md shows one Server row right after the Driver row.
- The API key is written nowhere in the run tree, the console, or the log,
  even when the server's /server_info reply holds it.

Every component but the two served backends and the generate and
compile_loop stages is a fake in a test Registry: the fake toolchain writes a
PLACEHOLDER artifact and compiles nothing, and the compile-only executor runs
nothing. Every request goes to the stub on 127.0.0.1 (tests/llm/served_stubs.py).
The bench sources, the replies, the key, and every served value are
SYNTHETIC. No value in this module is a measurement.
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from served_stubs import (
    CHAT_PATH,
    LLAMA_BUILD_INFO,
    MODELS_PATH,
    OLLAMA_CHAT_PATH,
    OLLAMA_GENERATE_PATH,
    SGLANG_VERSION,
    VLLM_VERSION,
    serve_llama_server,
    serve_ollama,
    serve_sglang,
    serve_vllm,
    stable,
)

from lassi.bench import load_suite
from lassi.core import runner as runner_module
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Limits, Message, RunResult, Sampling
from lassi.core.parquet import read_run_parquet
from lassi.core.registry import Registry
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.stages import CompileLoopStage, GenerateStage
from lassi.llm import OllamaBackend, OpenAICompatBackend

if TYPE_CHECKING:
    from conftest import StubServer

REPO = Path(__file__).resolve().parents[2]
SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
ITEM = "layout"
MODEL_ID = "coder-fixture-a"
OTHER_ID = "coder-fixture-b"
OLLAMA_ID = "coder-7b-q8_0"
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
KEY_ENV = "LASSI_TEST_API_KEY"
SENTINEL_KEY = "lassi-test-sentinel-key-4e9b20d7"
# The stub replaces this marker with the Authorization header it received.
ECHO = "<<authorization>>"
# SYNTHETIC text that a refused base_url carries; no message may quote it.
URL_SECRET = "Q7ZR2LW9XV4TN8KP"
# SYNTHETIC bench sources and the reply, which the fake toolchain "builds".
SOURCES = {
    "omp": '#include <cstdio>\nint main() {\n#pragma omp target\n  { }\n  std::printf("done\\n");\n}\n',
    "cuda": "#include <cstdio>\n__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n",
}
GOOD_REPLY = render_file_blocks({"main.cu": "int main() { return 0; }\n"})
TRIAL_PROVENANCE_KEYS = {"commit", "dirty", "device", "sdk", "date", "device_records"}
SERVED = {
    "vllm": (serve_vllm, VLLM_VERSION, "GET /version version"),
    "sglang": (serve_sglang, SGLANG_VERSION, "GET /server_info version"),
    "llama-server": (serve_llama_server, LLAMA_BUILD_INFO, "GET /props build_info"),
}


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate, compile, and key variables, point TMPDIR at a test directory, and fake git."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(KEY_ENV, raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    answers = {"rev-parse": FAKE_COMMIT + "\n", "status": ""}
    monkeypatch.setattr(runner_module, "_git", lambda *args: answers[args[0]])


# ---------------------------------------------------------------------------
# Fake components


class FakeToolchain:
    """A Toolchain "nvcc-sm80" without PIN: writes the files and a PLACEHOLDER artifact; compiles nothing."""

    name = "nvcc-sm80"
    capabilities = frozenset({"diagnostics"})

    def build(self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None) -> BuildResult:
        """Write every file (and any harness file) and a PLACEHOLDER artifact, and report a clean build."""
        for path, text in [*files.items(), *(harness or {}).items()]:
            target = Path(workdir) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode("utf-8"))
        artifact = Path(workdir) / "main"
        artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
        return BuildResult(artifact=artifact, diagnostics=[])


class CompileOnly:
    """The Executor "none" of this module: compile only, runs nothing."""

    name = "none"
    capabilities = frozenset({"compile_only"})

    def device(self) -> str:
        """Name no device, as a compile-only executor does."""
        return "none (compile only)"

    def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        """Fail the test: nothing runs on the compile-only path."""
        raise AssertionError("the compile-only path ran an executor")


class PlainBackend:
    """An LLMBackend "plainllm" without model_check: answers with the SYNTHETIC reply."""

    name = "plainllm"
    capabilities = frozenset({"chat"})

    def __init__(self, model_id: str) -> None:
        """Keep the model id."""
        self.model_id = model_id

    def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
        """Return the SYNTHETIC reply that the fake toolchain builds."""
        return Completion(text=GOOD_REPLY, prompt_tokens=0, completion_tokens=0)


class CheckWithoutServing(PlainBackend):
    """An LLMBackend "checkless" that declares model_check and has check() but no serving()."""

    name = "checkless"
    capabilities = frozenset({"chat", "model_check"})

    def check(self) -> dict[str, Any]:
        """Return an entry for the model id."""
        return {"id": self.model_id}


def pinned(cls: type, base_url: str) -> type:
    """Return a subclass of a served backend that the runner builds as factory(model_id), pointed at `base_url`."""

    class Pinned(cls):  # type: ignore[misc, valid-type]
        """The real backend with a fixed base URL, for a recipe without server keys."""

        def __init__(self, model_id: str) -> None:
            """Point the backend at `base_url`."""
            super().__init__(model_id, base_url=base_url, timeout_s=30.0)

    return Pinned


def make_registry(**backends: type) -> Registry:
    """Return a test Registry: the served backends (or `backends` in their place), the fakes, and the real stages."""
    registry = Registry()
    chosen = {"openai_compat": OpenAICompatBackend, "ollama": OllamaBackend, **backends}
    for name, cls in chosen.items():
        registry.register("LLMBackend", name, cls)
    registry.register("LLMBackend", "plainllm", PlainBackend)
    registry.register("LLMBackend", "checkless", CheckWithoutServing)
    registry.register("Executor", "none", CompileOnly)
    registry.register("Toolchain", "nvcc-sm80", FakeToolchain)
    registry.register("Stage", "generate", GenerateStage)
    registry.register("Stage", "compile_loop", CompileLoopStage)
    return registry


# ---------------------------------------------------------------------------
# Recipes, bench sources, and runs


def model(backend: str = "openai_compat", model_id: str = MODEL_ID, **keys: Any) -> dict[str, Any]:
    """Return a model section for `backend` with the given server keys."""
    return {"backend": backend, "id": model_id, **keys}


def recipe_data(model_section: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compile-only recipe for the layout item, omp to cuda, with the given model section."""
    return {
        "extends": "base",
        "model": dict(model_section),
        "llm": {"sampling": {"max_tokens": 64}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": "omp", "target": "cuda"}],
        "prompts": "p0-smoke",
        "toolchain": {"cuda": "nvcc-sm80"},
        "stages": ["generate", "compile_loop"],
        "executor": {"kind": "none"},
        "trials": {"n": 1},
    }


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    """Write the item's SYNTHETIC sources where the suite manifest lays them out; return the bench root."""
    root = tmp_path / "bench"
    spec = load_suite(SUITE_MANIFEST).items[ITEM]
    for language, text in SOURCES.items():
        layout = spec.languages[language]
        path = root / layout.dir / layout.files[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def run(
    tmp_path: Path, bench: Path, data: Mapping[str, Any], registry: Registry | None = None, name: str = "served"
) -> Path:
    """Write the recipe `name` and run it with the test registry; return the run directory."""
    path = tmp_path / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    options = RunOptions(
        runs_root=tmp_path / "runs-root",
        run_id="served-run",
        bench_root=bench,
        registry=make_registry() if registry is None else registry,
    )
    return run_recipe(path, options)


def refused(tmp_path: Path, bench: Path, data: Mapping[str, Any], registry: Registry | None = None) -> RunError:
    """Run a recipe the runner must refuse; assert no directory exists under the runs root; return the error."""
    with pytest.raises(RunError) as info:
        run(tmp_path, bench, data, registry, "refused")
    assert not (tmp_path / "runs-root").exists(), "a refusal comes before any directory exists"
    return info.value


def read_ascii(path: Path) -> str:
    """Return a file's text after checking it is ASCII with LF newlines."""
    data = path.read_bytes()
    assert data.isascii() and b"\r" not in data, path
    return data.decode("ascii")


def manifest(run_dir: Path) -> dict[str, Any]:
    """Return the run's provenance.json."""
    return json.loads(read_ascii(run_dir / "provenance.json"))


def summary_rows(run_dir: Path) -> list[tuple[str, str]]:
    """Return run.md's summary table (the rows above the first ## heading) as (field, value) pairs, in order."""
    head = read_ascii(run_dir / "run.md").split("\n## ", 1)[0]
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in head.split("\n")]
    return [(row[0], row[1]) for row in rows if len(row) == 2][2:]


def error_texts(error: BaseException) -> list[str]:
    """Return every text an error shows: str, repr, args, and the formatted traceback with its chain."""
    texts = [str(error), repr(error), repr(error.args)]
    texts.append("".join(traceback.format_exception(type(error), error, error.__traceback__)))
    return texts


def serve(stub: StubServer, server: str, **kwargs: Any) -> dict[str, Any]:
    """Serve MODEL_ID in one server's shapes with the reply the fake toolchain builds; return the entry."""
    serve_function: Callable[..., dict[str, Any]] = SERVED[server][0]
    return serve_function(stub, MODEL_ID, reply=GOOD_REPLY, **kwargs)


def at_v1(stub: StubServer, **keys: Any) -> dict[str, Any]:
    """Return an openai_compat model section at the stub's /v1, with a short timeout and the given keys."""
    return model(base_url=f"{stub.url}/v1", timeout_s=30, **keys)


# ---------------------------------------------------------------------------
# The keys reach the backends


def test_server_keys_reach_openai_compat_through_the_runner(
    tmp_path: Path, bench: Path, stub_server: StubServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_ENV, SENTINEL_KEY)
    serve(stub_server, "vllm")
    run_dir = run(tmp_path, bench, recipe_data(at_v1(stub_server, api_key_env=KEY_ENV)))
    assert manifest(run_dir)["status"] == "complete"
    assert stub_server.requests_to("POST", CHAT_PATH), "the chat request reached the stub at base_url"
    assert all(seen.header("Authorization") == f"Bearer {SENTINEL_KEY}" for seen in stub_server.requests)
    assert stub_server.calls()[0] == ("GET", MODELS_PATH), "the model check comes first"
    assert len(stub_server.requests_to("GET", MODELS_PATH)) == 1, "the trials reuse the checked entry"


def test_timeout_s_reaches_the_backend(tmp_path: Path, bench: Path, stub_server: StubServer) -> None:
    card = serve(stub_server, "vllm")
    stub_server.reply("GET", MODELS_PATH, json_body={"object": "list", "data": [card]}, delay_s=5.0)
    started = time.monotonic()
    error = refused(tmp_path, bench, recipe_data(model(base_url=f"{stub_server.url}/v1", timeout_s=0.5)))
    assert time.monotonic() - started < 4.0, "the model check gave up after timeout_s, not the default 600 s"
    assert "model.id" in str(error), str(error)


def test_server_keys_reach_ollama_through_the_runner(tmp_path: Path, bench: Path, stub_server: StubServer) -> None:
    entry = serve_ollama(stub_server, OLLAMA_ID, reply=GOOD_REPLY)
    run_dir = run(tmp_path, bench, recipe_data(model("ollama", OLLAMA_ID, base_url=stub_server.url, timeout_s=30)))
    calls = stub_server.calls()
    assert calls[0] == ("GET", MODELS_PATH)
    assert ("POST", OLLAMA_GENERATE_PATH) in calls and ("POST", OLLAMA_CHAT_PATH) in calls
    expected = {"base_url": stub_server.url, "model": stable(entry), "version": None, "version_from": None}
    assert manifest(run_dir)["serving"] == expected


def test_the_model_check_comes_before_the_ollama_unload(tmp_path: Path, bench: Path, stub_server: StubServer) -> None:
    serve_ollama(stub_server, OLLAMA_ID, reply=GOOD_REPLY)
    registry = make_registry(ollama=pinned(OllamaBackend, stub_server.url))
    run(tmp_path, bench, recipe_data(model("ollama", OLLAMA_ID)), registry)
    calls = stub_server.calls()
    assert calls[0] == ("GET", MODELS_PATH)
    assert calls.index(("POST", OLLAMA_GENERATE_PATH)) > 0
    assert calls.count(("GET", MODELS_PATH)) == 1


def test_an_ollama_server_without_the_model_is_refused_before_any_unload(
    tmp_path: Path, bench: Path, stub_server: StubServer
) -> None:
    serve_ollama(stub_server, "other-7b-q8_0")
    registry = make_registry(ollama=pinned(OllamaBackend, stub_server.url))
    error = refused(tmp_path, bench, recipe_data(model("ollama", OLLAMA_ID)), registry)
    assert "model.id" in str(error) and OLLAMA_ID in str(error), str(error)
    assert stub_server.calls() == [("GET", MODELS_PATH)], "no unload or chat request reaches a server without the model"


# ---------------------------------------------------------------------------
# The serving record in provenance.json and run.md


@pytest.mark.parametrize("server", sorted(SERVED))
def test_provenance_json_records_the_serving_record(
    tmp_path: Path, bench: Path, stub_server: StubServer, server: str
) -> None:
    entry = serve(stub_server, server)
    run_dir = run(tmp_path, bench, recipe_data(at_v1(stub_server)))
    _, version, version_from = SERVED[server]
    expected = {
        "base_url": f"{stub_server.url}/v1",
        "model": stable(entry),
        "version": version,
        "version_from": version_from,
    }
    assert manifest(run_dir)["serving"] == expected
    trial_files = sorted(run_dir.rglob("trial.json"))
    assert trial_files
    for path in trial_files:
        assert set(json.loads(read_ascii(path))["provenance"]) == TRIAL_PROVENANCE_KEYS, "trial provenance unchanged"
    for row in read_run_parquet(run_dir / "parquet")["trials"]:
        assert not [column for column in row if "serving" in column], "the Parquet trials table is unchanged"


@pytest.mark.parametrize(
    ("server", "line"),
    [
        pytest.param("vllm", f"owned_by vllm; version {VLLM_VERSION}; max_model_len 32768", id="vllm"),
        pytest.param(
            "llama-server", f"owned_by llamacpp; version {LLAMA_BUILD_INFO}; meta.n_ctx 4096", id="llama-server"
        ),
    ],
)
def test_run_md_shows_the_server_row(
    tmp_path: Path, bench: Path, stub_server: StubServer, server: str, line: str
) -> None:
    serve(stub_server, server)
    rows = summary_rows(run(tmp_path, bench, recipe_data(at_v1(stub_server))))
    fields = [field for field, _ in rows]
    assert fields.count("Server") == 1, fields
    assert fields.index("Server") == fields.index("Driver") + 1, "the Server row follows the Driver row"
    assert dict(rows)["Server"] == line


def test_a_backend_without_model_check_writes_no_serving_record(tmp_path: Path, bench: Path) -> None:
    run_dir = run(tmp_path, bench, recipe_data(model("plainllm")))
    assert "serving" not in manifest(run_dir)
    assert "Server" not in dict(summary_rows(run_dir))


# ---------------------------------------------------------------------------
# Refusals before any directory


def test_a_model_the_server_does_not_list_is_refused_before_any_directory(
    tmp_path: Path, bench: Path, stub_server: StubServer
) -> None:
    serve_vllm(stub_server, OTHER_ID)
    message = str(refused(tmp_path, bench, recipe_data(at_v1(stub_server))))
    assert "model.id" in message and MODEL_ID in message and OTHER_ID in message, message
    assert [method for method, _ in stub_server.calls()] == ["GET"], "no chat or version request"


def test_an_unset_key_variable_is_refused_before_any_directory(
    tmp_path: Path, bench: Path, stub_server: StubServer
) -> None:
    # clean_environment unsets KEY_ENV: the key is read when the model check is sent, so nothing reaches the server.
    serve(stub_server, "vllm")
    message = str(refused(tmp_path, bench, recipe_data(at_v1(stub_server, api_key_env=KEY_ENV))))
    assert "model.id" in message and KEY_ENV in message, message
    assert stub_server.requests == []


def test_an_unreachable_server_is_refused_before_any_directory(tmp_path: Path, bench: Path, dead_port: int) -> None:
    data = recipe_data(model(base_url=f"http://127.0.0.1:{dead_port}/v1", timeout_s=5))
    assert "model.id" in str(refused(tmp_path, bench, data))


@pytest.mark.parametrize(
    "base_url",
    [
        pytest.param("http://127.0.0.1:{port}/v1?token=" + URL_SECRET, id="query"),
        pytest.param("http://fixture-user:" + URL_SECRET + "@127.0.0.1:{port}/v1", id="password"),
    ],
)
def test_a_setting_the_backend_refuses_is_a_run_error_before_any_directory(
    tmp_path: Path, bench: Path, stub_server: StubServer, base_url: str
) -> None:
    serve(stub_server, "vllm")
    error = refused(tmp_path, bench, recipe_data(model(base_url=base_url.format(port=stub_server.port))))
    assert "base_url" in str(error), str(error)
    for text in error_texts(error):
        assert URL_SECRET not in text, text
    assert stub_server.requests == [], "a refused setting sends nothing"


def test_a_backend_declaring_model_check_without_serving_is_refused(tmp_path: Path, bench: Path) -> None:
    message = str(refused(tmp_path, bench, recipe_data(model("checkless"))))
    assert "checkless" in message and "model_check" in message and "serving" in message, message


def test_a_serving_record_json_cannot_hold_is_refused_before_any_directory(
    tmp_path: Path, bench: Path, stub_server: StubServer
) -> None:
    serve(stub_server, "vllm")
    raw = (
        b'{"object": "list", "data": [{"id": "' + MODEL_ID.encode("ascii") + b'", "object": "model", '
        b'"owned_by": "vllm", "max_model_len": NaN}]}'
    )
    stub_server.reply("GET", MODELS_PATH, raw=raw)
    refused(tmp_path, bench, recipe_data(at_v1(stub_server)))
    assert stub_server.requests_to("POST", CHAT_PATH) == []


def test_a_record_holding_the_api_key_is_refused_before_any_directory(
    tmp_path: Path, bench: Path, stub_server: StubServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_ENV, SENTINEL_KEY)
    card = serve(stub_server, "vllm")
    listing = {"object": "list", "data": [{**card, "root": ECHO}]}
    stub_server.reply("GET", MODELS_PATH, json_body=listing, echo=(ECHO, "Authorization"))
    error = refused(tmp_path, bench, recipe_data(at_v1(stub_server, api_key_env=KEY_ENV)))
    for text in error_texts(error):
        assert SENTINEL_KEY not in text, text
    assert stub_server.requests_to("POST", CHAT_PATH) == []


# ---------------------------------------------------------------------------
# Never a key


def test_the_api_key_is_written_nowhere_in_the_run_tree(
    tmp_path: Path,
    bench: Path,
    stub_server: StubServer,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv(KEY_ENV, SENTINEL_KEY)
    serve(stub_server, "sglang", api_key=SENTINEL_KEY)
    run_dir = run(tmp_path, bench, recipe_data(at_v1(stub_server, api_key_env=KEY_ENV)))
    assert manifest(run_dir)["serving"]["version"] == SGLANG_VERSION, "the /server_info reply was read"
    assert stub_server.requests_to("GET", "/server_info")[0].header("Authorization") == f"Bearer {SENTINEL_KEY}"
    files = [path for path in (tmp_path / "runs-root").rglob("*") if path.is_file()]
    assert files
    for path in files:
        assert SENTINEL_KEY.encode("ascii") not in path.read_bytes(), path
    assert f"api_key_env: {KEY_ENV}" in read_ascii(run_dir / "recipe.resolved.yaml"), "the recipe names the variable"
    captured = capsys.readouterr()
    assert SENTINEL_KEY not in captured.out and SENTINEL_KEY not in captured.err
    assert all(SENTINEL_KEY not in logged.getMessage() for logged in caplog.records)
