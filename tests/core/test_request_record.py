"""Tests for the record of every model request (task P2.1).

Bible: Result Record, Readability Standards (Trial row), Component
Interfaces (Stage contract rules), Decision Log 2026-09-24 (faithful
generation entry, which left the system prompts and the context requests
out of the record "until a later phase").

The contract these tests fix, from the P2.1 acceptance criteria
(plans/p2-scoring.md):

- Trial gains the field `requests: list[Request] | None`, default None.
  None means not recorded: a trial.json written before this change has no
  `requests` key and loads with None. A trial the runner runs now records
  a list, which is empty when no model call was made (a baseline failure
  ends the trial first).
- Request is a frozen, keyword-only record class in lassi.core.record with
  these fields: `index` (int, its position in Trial.requests), `stage`
  (str, the registered name of the stage that sent it: summarize_context,
  describe_source, generate, compile_loop for a compile-error correction,
  run_loop for a run-error correction), `attempt_index` (int or None, the
  attempt its reply became; None for a context request), `messages`
  (list of RequestMessage: every message sent, in order, system messages
  included), `reply_ref` (TextRef: the reply in the text store as kept,
  each lone surrogate replaced by U+FFFD), and `diagnostics` (list of
  Diagnostic). RequestMessage has `role` (str) and `ref` (TextRef: the
  message text in the text store, by sha256). No message text is kept
  inline in trial.json.
- An attempt's request ends with the user message the attempt keeps as
  prompt_ref, and its reply is the attempt's response_text. A faithful
  trial with context holds 3 + final.corrections requests, and its system
  messages hold exactly two distinct texts: the general system prompt and
  the direction's system prompt.
- A lone surrogate in a context reply gives that request one parse-stage
  `invalid-text` warning; attempt 0 no longer carries it.
- trial.md shows every request in order: for each request, the sha256 of
  each message and of the reply, and each message text as a fenced block
  shows it (LF line breaks, non-ASCII as backslash escapes).
- The Parquet mirror gains the table `requests` (lassi.core.parquet
  TABLES): one row per request, in index order, with the key columns and
  `index`, `stage`, `attempt_index` (null for a context request),
  `message_roles` and `message_sha256` (lists in message order), and
  `reply_ref_sha256`. A trial whose requests were not recorded has no rows.

Most tests run on a SYNTHETIC fragment set (short marker texts under a
temporary assets root), so they need no upstream checkout. The tests of
the faithful lassi-2024 set compute every expected text from a fresh
extraction of the pinned upstream checkout into a temporary directory
(tools/extract_lassi_assets.py) and skip, naming that tool, when
third_party/LASSI is absent or not at the pin. No upstream text is copied
into this file (OQ-018): fragment keys and dictionary entry names are
identifiers, not upstream prose. Bench sources, model replies, compiler
output, and program runs are SYNTHETIC; the fake toolchain compiles
nothing and the scripted executor runs nothing. No value in this module
is a measurement.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

import lassi.prompts as prompts_module
import lassi.prompts.assets as prompt_assets
from lassi.bench import Direction, load_suite
from lassi.core import fragments, parquet
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Limits, Message, RunResult, Sampling
from lassi.core.record import (
    Attempt,
    BenchItem,
    Context,
    Diagnostic,
    Final,
    ModelInfo,
    Provenance,
    RunInfo,
    TextRef,
    ToolchainPins,
    Trial,
    arm_segment,
    json_text,
    make_trial_id,
)
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.stages import RUN_LOOP_KEYS
from lassi.core.store import TextStore, read_trial, trial_dir, write_trial
from lassi.core.trial_md import render_trial_md
from lassi.executors import NoneExecutor
from lassi.llm import MockBackend
from lassi.prompts import RecipeAssets, load_recipe_assets

REPO = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = REPO / "third_party" / "LASSI"
UPSTREAM_PIN = "74b46812523f2ff79b53b6880a4521690d7478b0"
TOOL = REPO / "tools" / "extract_lassi_assets.py"
SKIP_REASON = (
    "needs the upstream checkout at third_party/LASSI on commit 74b4681 so that tools/extract_lassi_assets.py "
    "can generate the lassi-2024 fragments; run uv run tools/fetch_upstream.py first"
)

SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
ITEM = "layout"
MODEL_ID = "scripted-fixture"
MOCK_ID = "mock-reference"
OMP_TO_CUDA = Direction("omp", "cuda")
CUDA_TO_OMP = Direction("cuda", "omp")
# run_loop reproduces fixes.execution_gate, which faithful: true turns off, so every faithful list names it.
FAITHFUL_STAGES = ("baseline", "summarize_context", "describe_source", "generate", "compile_loop", "run_loop")
TEMPLATE_SET = "p0-smoke"
TEMPLATE_STAGES = ("generate", "compile_loop")
# One fake toolchain, bound for whichever language is the target.
FAKE_TOOLCHAIN = "fake-cc"

# The faithful prompt set, the packs upstream picks per target, and upstream's dictionary entry name per direction.
UPSTREAM_SET = "lassi-2024"
UPSTREAM_PACKS = ("openmp-4.0-card", "cuda-12.5-ch5")
DIRECTION_KEY = {("omp", "cuda"): "OMP_to_CUDA", ("cuda", "omp"): "CUDA_to_OMP"}

# A SYNTHETIC fragment set for omp to cuda: every key the faithful stages read, each a short marker text.
SYNTHETIC_SET = "synthetic-fragments"
SYNTHETIC_PACK = "synthetic-cuda-pack"
_SYNTHETIC_KEYS = (*fragments.GENERATE_KEYS, *fragments.SUMMARY_KEYS, *fragments.DESCRIPTION_KEYS, *RUN_LOOP_KEYS)
SYNTHETIC_FRAGMENTS = {
    fragments.fragment_key(key, OMP_TO_CUDA): f"<SYNTHETIC {fragments.fragment_key(key, OMP_TO_CUDA)}>"
    for key in _SYNTHETIC_KEYS
}
SYNTHETIC_GENERAL_SYSTEM = SYNTHETIC_FRAGMENTS[fragments.GENERAL_SYSTEM]
SYNTHETIC_DIRECTION_SYSTEM = SYNTHETIC_FRAGMENTS[fragments.fragment_key(fragments.DIRECTION_SYSTEM, OMP_TO_CUDA)]

# SYNTHETIC bench sources; indentation holds runs of spaces, which the faithful generation prompt collapses.
OMP_SOURCE = '#include <cstdio>\nint main() {\n    std::printf("SYNTHETIC\\n");\n    return 0;\n}\n'
CUDA_SOURCE = "#include <cstdio>\n__global__ void kernel(int *out) {\n    out[threadIdx.x] = 1;\n}\nint main() {}\n"
SOURCES = {"omp": OMP_SOURCE, "cuda": CUDA_SOURCE}
# A file holding this marker makes the fake toolchain report a failed build.
BROKEN = "#error SYNTHETIC"

# SYNTHETIC model replies. The code replies are one untagged fence each, the form faithful extraction reads.
SUMMARY_REPLY = "SYNTHETIC summary reply."
DESCRIPTION_REPLY = "SYNTHETIC description reply."
BAD_CODE = BROKEN + " not translated yet\nint main() { return 1; }\n"
FIRST_FIX = "int main() { return 1; }\n"
SECOND_FIX = "int main() { return 0; }\n"

# SYNTHETIC compiler output of a failed build: the raw stderr attachment and its one parsed diagnostic.
RAW_STDERR = b'"main.x", line 1: error: SYNTHETIC compiler text\n'
PARSED_ERROR = Diagnostic(stage="compile", severity="error", code="synthetic", message="SYNTHETIC parsed diagnostic")

# SYNTHETIC program runs. The wall times are PLACEHOLDER values a RunResult needs, never measurements.
PLACEHOLDER_WALL_S = 0.5
REFERENCE_OK = RunResult(exit_code=0, hang=False, stdout="SYNTHETIC stdout\n", stderr="", wall_s=PLACEHOLDER_WALL_S)
RUN_FAILED = RunResult(exit_code=1, hang=False, stdout="", stderr="SYNTHETIC run error\n", wall_s=PLACEHOLDER_WALL_S)
RUN_OK = RunResult(exit_code=0, hang=False, stdout="SYNTHETIC stdout\n", stderr="", wall_s=PLACEHOLDER_WALL_S)

# The stages and attempts of the full scenario: generate, one compile-error correction, one run-error correction.
FULL_STAGES = ["summarize_context", "describe_source", "generate", "compile_loop", "run_loop"]
FULL_ATTEMPTS = [None, None, 0, 1, 2]
INVALID_TEXT = ("parse", "warning", "invalid-text")


def fenced(code: str) -> str:
    """Return a SYNTHETIC reply holding `code` in one untagged fence."""
    return "SYNTHETIC reply.\n```\n" + code + "```\n"


FULL_REPLIES = (SUMMARY_REPLY, DESCRIPTION_REPLY, fenced(BAD_CODE), fenced(FIRST_FIX), fenced(SECOND_FIX))


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate variables a test could inherit and point TMPDIR at a test directory."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))


# ---------------------------------------------------------------------------
# The contract's names, looked up so a missing one fails with a clear message


def require_requests_field() -> None:
    """Fail the test clearly while Trial has no `requests` field."""
    if "requests" not in [spec.name for spec in dataclasses.fields(Trial)]:
        pytest.fail("Trial has no requests field; task P2.1 adds Trial.requests (every model request) to the record")


def requests_of(trial: Trial) -> list[Any]:
    """Return trial.requests of a trial run now; fail clearly while the field is missing or reads not recorded."""
    require_requests_field()
    requests = trial.requests
    assert requests is not None, "a trial the runner runs records its requests; None means not recorded"
    return list(requests)


def text_ref(text: str) -> TextRef:
    """Return the TextRef the text store gives `text`, computed here with hashlib."""
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return TextRef(sha256=sha, path=f"texts/{sha[:2]}/{sha}.txt")


def refs_sent(messages: Sequence[Message]) -> list[tuple[str, TextRef]]:
    """Return the (role, text reference) of each message a backend received."""
    return [(message.role, text_ref(message.content)) for message in messages]


def refs_recorded(request: Any) -> list[tuple[str, TextRef]]:
    """Return the (role, text reference) of each message a Request records."""
    return [(message.role, message.ref) for message in request.messages]


def system_texts(requests: Sequence[Any], store: TextStore) -> set[str]:
    """Return the distinct texts of every system message across `requests`."""
    return {store.get(message.ref) for request in requests for message in request.messages if message.role == "system"}


def md_form(text: str) -> str:
    """Return `text` as trial.md shows fenced text: CRLF and lone CR as LF, non-ASCII as backslash escapes."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.encode("ascii", "backslashreplace").decode("ascii")


# ---------------------------------------------------------------------------
# Assets: the synthetic set, or a fresh extraction of the pinned upstream checkout


def write_tree(tree: Path, texts: Mapping[str, str]) -> None:
    """Write a manifest tree: one `<key>.txt` per entry and MANIFEST.yaml with each file's sha256."""
    tree.mkdir(parents=True)
    entries = []
    for key, text in texts.items():
        data = text.encode("ascii")
        (tree / f"{key}.txt").write_bytes(data)
        entries.append({"key": key, "source": "synthetic", "sha256": hashlib.sha256(data).hexdigest()})
    (tree / "MANIFEST.yaml").write_bytes(yaml.safe_dump({"entries": entries}).encode("ascii"))


@pytest.fixture
def synthetic_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a temporary assets root with the synthetic set and a cuda pack; the loader reads it by default."""
    root = tmp_path / "assets"
    write_tree(root / "prompts" / SYNTHETIC_SET, SYNTHETIC_FRAGMENTS)
    write_tree(root / "context" / SYNTHETIC_PACK, {"synthdict.cuda": "<SYNTHETIC cuda pack>"})
    monkeypatch.setattr(prompt_assets, "default_root", lambda: root)
    return root


def checkout_at_pin() -> bool:
    """Return True when third_party/LASSI is its own git checkout whose HEAD is the upstream pin."""
    if not (UPSTREAM_DIR / ".git").exists():
        return False
    done = subprocess.run(
        ["git", "-C", str(UPSTREAM_DIR), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return done.returncode == 0 and done.stdout.strip() == UPSTREAM_PIN


def load_extractor() -> ModuleType:
    """Load tools/extract_lassi_assets.py as a module."""
    spec = importlib.util.spec_from_file_location("extract_lassi_assets_for_p21", TOOL)
    assert spec and spec.loader, f"cannot load {TOOL}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def upstream_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a temporary assets root holding a fresh extraction of the pinned checkout, or skip naming the tool."""
    if not checkout_at_pin():
        pytest.skip(SKIP_REASON)
    out = tmp_path_factory.mktemp("lassi-assets")
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        status = load_extractor().main(["--upstream", str(UPSTREAM_DIR), "--out", str(out)])
    assert status == 0, f"tools/extract_lassi_assets.py exited {status}: {sink.getvalue()[-2000:]}"
    return out


@pytest.fixture
def lassi_assets(upstream_root: Path, monkeypatch: pytest.MonkeyPatch) -> RecipeAssets:
    """Point the asset loader and the prompt sets at the fresh extraction; return its fragments and both packs."""
    monkeypatch.setattr(prompt_assets, "default_root", lambda: upstream_root)
    monkeypatch.setattr(
        prompts_module, "default_roots", lambda: (upstream_root / "prompts", REPO / "assets" / "prompts")
    )
    return load_recipe_assets({"prompts": UPSTREAM_SET, "context": list(UPSTREAM_PACKS)}, root=upstream_root)


def write_bench(root: Path, sources: Mapping[str, str]) -> Path:
    """Write the item's SYNTHETIC source per language where the suite manifest lays it out; return `root`."""
    spec = load_suite(SUITE_MANIFEST).items[ITEM]
    for language, text in sources.items():
        layout = spec.languages[language]
        path = root / layout.dir / layout.files[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    """Return a bench root holding the item's SYNTHETIC source in both languages."""
    return write_bench(tmp_path / "bench", SOURCES)


def target_file(direction: Direction) -> str:
    """Return the item's one target file name for `direction`, from the suite manifest."""
    return load_suite(SUITE_MANIFEST).items[ITEM].languages[direction.target].files[0]


# ---------------------------------------------------------------------------
# Fake components


@dataclass
class Log:
    """The script the fakes follow and what they saw: replies and runs in order, and every model request."""

    replies: list[str] = field(default_factory=list)
    runs: list[RunResult] = field(default_factory=list)
    requests: list[list[Message]] = field(default_factory=list)


def scripted_backend(log: Log) -> type:
    """Return an LLMBackend class, registered as "scripted", that answers from `log.replies` in order."""

    class ScriptedBackend:
        """Records each request and answers with the next scripted reply."""

        name = "scripted"
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id, as every backend does."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Record the request and return the next scripted reply."""
            log.requests.append(list(messages))
            assert log.replies, "the backend was asked for more replies than the script holds"
            return Completion(text=log.replies.pop(0), prompt_tokens=0, completion_tokens=0)

    return ScriptedBackend


def recording_mock(log: Log) -> type:
    """Return the mock backend, registered as "mock", recording each request it answers in `log`."""

    class RecordingMock(MockBackend):
        """The mock backend, answering with the reference target; each request is recorded first."""

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Record the request and return the mock's reply."""
            log.requests.append(list(messages))
            return super().complete(messages, sampling)

    return RecordingMock


def scripted_executor(log: Log) -> type:
    """Return a sandboxed Executor class, registered as "scripted", that answers each run from `log.runs` in order."""

    class ScriptedExecutor:
        """Returns the next SYNTHETIC RunResult; runs nothing."""

        name = "scripted"
        capabilities = frozenset({"runs_code", "sandboxed"})

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Return the next scripted result."""
            assert log.runs, "the executor was asked for more runs than the script holds"
            return log.runs.pop(0)

    return ScriptedExecutor


class FakeToolchain:
    """Keeps a raw stderr attachment and reports a build: failed when a file holds BROKEN; compiles nothing."""

    capabilities = frozenset({"diagnostics"})

    def build(self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None) -> BuildResult:
        """Write the stderr attachment and return a failed build or a PLACEHOLDER artifact."""
        workdir = Path(workdir)
        broken = any(BROKEN in text for text in files.values())
        (workdir / "compile.stderr").write_bytes(RAW_STDERR if broken else b"")
        if broken:
            return BuildResult(artifact=None, diagnostics=[PARSED_ERROR], stderr_ref="compile.stderr")
        artifact = workdir / "PLACEHOLDER-artifact"
        artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
        return BuildResult(artifact=artifact, diagnostics=[], stderr_ref="compile.stderr")


def make_registry(log: Log, stages: Sequence[str]) -> Registry:
    """Return a test Registry: both backends, both executors, the fake toolchain, and the named real stages."""
    registry = Registry()
    registry.register("LLMBackend", "scripted", scripted_backend(log))
    registry.register("LLMBackend", "mock", recording_mock(log))
    registry.register("Executor", "none", NoneExecutor)
    registry.register("Executor", "scripted", scripted_executor(log))
    registry.register("Toolchain", FAKE_TOOLCHAIN, FakeToolchain)
    for name in stages:
        registry.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    return registry


# ---------------------------------------------------------------------------
# Recipes and runs


def recipe(
    direction: Direction = OMP_TO_CUDA,
    *,
    prompts: str = SYNTHETIC_SET,
    context: Sequence[str] | None = (SYNTHETIC_PACK,),
    backend: str = "scripted",
    model_id: str = MODEL_ID,
    executor: str = "none",
    **changes: Any,
) -> dict[str, Any]:
    """Return a faithful one-trial recipe for the item and `direction`; `changes` replace top-level keys."""
    data: dict[str, Any] = {
        "extends": "base",
        "faithful": True,
        "model": {"backend": backend, "id": model_id},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": direction.source, "target": direction.target}],
        "prompts": prompts,
        "toolchain": {direction.target: FAKE_TOOLCHAIN},
        "stages": list(FAITHFUL_STAGES),
        "executor": {"kind": executor},
        "trials": {"n": 1},
    }
    if context is not None:
        data["context"] = list(context)
    data.update(changes)
    return data


@dataclass
class Outcome:
    """One finished run: its directory, its one trial's id, the trial read back from the run tree, and the log."""

    run_dir: Path
    trial_id: str
    trial: Trial
    log: Log

    @property
    def store(self) -> TextStore:
        """Return the run's text store."""
        return TextStore(self.run_dir)


def run_one(tmp_path: Path, bench_root: Path, name: str, data: Mapping[str, Any], log: Log) -> Outcome:
    """Run the one-trial recipe `data`, saved as `<name>.yaml`, with the fakes following `log`; return the outcome."""
    registry = make_registry(log, data["stages"])
    runs_root = tmp_path / name / "runs-root"
    options = RunOptions(runs_root=runs_root, run_id="test-run", bench_root=bench_root, registry=registry)
    path = tmp_path / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    run_dir = run_recipe(path, options)
    direction = data["directions"][0]
    arm = arm_segment(data["model"]["id"])
    trial_id = make_trial_id(name, arm, SUITE, f"{direction['source']}-{direction['target']}", ITEM, 1)
    return Outcome(run_dir, trial_id, read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir)), log)


def run_full(tmp_path: Path, bench_root: Path, direction: Direction = OMP_TO_CUDA, **changes: Any) -> Outcome:
    """Run the full faithful scenario: a compile-error correction, then a run-error correction, then a clean run.

    The scripted executor runs the target reference cleanly, attempt 1
    fails its run, and attempt 2 runs clean, so the trial ends at S5 after
    two corrections and five model requests.
    """
    log = Log(replies=list(FULL_REPLIES), runs=[REFERENCE_OK, RUN_FAILED, RUN_OK])
    outcome = run_one(tmp_path, bench_root, "every-request", recipe(direction, executor="scripted", **changes), log)
    assert outcome.trial.final.corrections == 2, "the scenario has one compile-error and one run-error correction"
    assert outcome.trial.final.stage_reached == "S5", "the last attempt runs clean"
    assert len(log.requests) == 5 and not log.replies and not log.runs, "the script is used up exactly"
    return outcome


@pytest.fixture
def full_run(tmp_path: Path, bench: Path, synthetic_assets: Path) -> Outcome:
    """Return the full faithful scenario run on the synthetic set."""
    return run_full(tmp_path, bench)


# ---------------------------------------------------------------------------
# Every model call is recorded (synthetic set)


def test_every_model_call_is_one_request_in_order_with_its_stage_and_attempt(full_run: Outcome) -> None:
    requests = requests_of(full_run.trial)
    assert len(requests) == 3 + full_run.trial.final.corrections, "summary, description, generation, each correction"
    assert [request.index for request in requests] == list(range(len(requests)))
    assert [request.stage for request in requests] == FULL_STAGES
    assert [request.attempt_index for request in requests] == FULL_ATTEMPTS


def test_each_request_records_every_message_it_sent_by_text_store_hash(full_run: Outcome) -> None:
    requests = requests_of(full_run.trial)
    store = full_run.store
    for position, request in enumerate(requests):
        sent = full_run.log.requests[position]
        assert refs_recorded(request) == refs_sent(sent), f"request {position} records each message sent, in order"
        for message, original in zip(request.messages, sent, strict=True):
            assert store.get(message.ref) == original.content, f"request {position}: the store holds each message"


def test_each_request_records_its_reply_and_links_to_the_attempt_it_became(full_run: Outcome) -> None:
    trial, store = full_run.trial, full_run.store
    requests = requests_of(trial)
    for position, (request, reply) in enumerate(zip(requests, FULL_REPLIES, strict=True)):
        assert request.reply_ref == text_ref(reply), f"request {position} records its reply by text-store hash"
        assert store.get(request.reply_ref) == reply
    assert store.get(requests[0].reply_ref) == trial.context.knowledge_summary
    assert store.get(requests[1].reply_ref) == trial.context.source_description
    for request in requests[2:]:
        attempt = trial.attempts[request.attempt_index]
        assert request.messages[-1].role == "user"
        assert request.messages[-1].ref == attempt.prompt_ref, "an attempt's prompt_ref is its request's user message"
        assert store.get(request.reply_ref) == attempt.response_text
    assert all(request.diagnostics == [] for request in requests), "clean replies carry no request diagnostics"


def test_a_faithful_trial_sends_two_distinct_system_prompts_and_records_both(full_run: Outcome) -> None:
    requests = requests_of(full_run.trial)
    assert system_texts(requests, full_run.store) == {SYNTHETIC_GENERAL_SYSTEM, SYNTHETIC_DIRECTION_SYSTEM}
    store = full_run.store
    general = [request.stage for request in requests if SYNTHETIC_GENERAL_SYSTEM in system_texts([request], store)]
    assert general == ["summarize_context", "describe_source"], "the context requests send the general system prompt"


def test_trial_json_keeps_each_message_by_hash_not_inline(full_run: Outcome) -> None:
    require_requests_field()
    raw = (trial_dir(full_run.run_dir, full_run.trial_id) / "trial.json").read_text(encoding="utf-8")
    assert "requests" in json.loads(raw), "trial.json holds Trial.requests"
    for text in (SYNTHETIC_GENERAL_SYSTEM, SYNTHETIC_DIRECTION_SYSTEM):
        assert text not in raw, "a system prompt is kept once in the text store, never inline in trial.json"
        assert text_ref(text).sha256 in raw, "trial.json names each system message by its sha256"


def test_a_trial_that_ends_before_any_model_call_records_no_requests(tmp_path: Path, synthetic_assets: Path) -> None:
    bench_root = write_bench(tmp_path / "bench-broken", {**SOURCES, "cuda": BROKEN + "\n" + CUDA_SOURCE})
    log = Log()
    outcome = run_one(tmp_path, bench_root, "baseline-ends", recipe(), log)
    reason = outcome.trial.final.end_reason
    assert reason is not None and reason.code == "baseline-compile", "the target reference does not build"
    assert log.requests == [], "no model is asked"
    assert requests_of(outcome.trial) == [], "recorded, and none was made: an empty list, never None"


def test_a_template_set_records_each_request_as_its_one_user_message(
    tmp_path: Path, bench: Path, synthetic_assets: Path
) -> None:
    name = target_file(OMP_TO_CUDA)
    log = Log(replies=[render_file_blocks({name: BAD_CODE}), render_file_blocks({name: SECOND_FIX})])
    data = recipe(faithful=False, prompts=TEMPLATE_SET, context=None, stages=list(TEMPLATE_STAGES))
    outcome = run_one(tmp_path, bench, "template-requests", data, log)
    trial = outcome.trial
    assert trial.final.corrections == 1 and len(log.requests) == 2
    requests = requests_of(trial)
    assert len(requests) == 1 + trial.final.corrections, "generate and each correction; a template set has no context"
    assert [request.stage for request in requests] == ["generate", "compile_loop"]
    assert [request.attempt_index for request in requests] == [0, 1]
    for position, request in enumerate(requests):
        assert refs_recorded(request) == refs_sent(log.requests[position])
        assert refs_recorded(request) == [("user", trial.attempts[position].prompt_ref)]


# ---------------------------------------------------------------------------
# A context reply's invalid-text warning rides on its request


def test_a_context_reply_invalid_text_warning_is_recorded_with_its_request(
    tmp_path: Path, bench: Path, synthetic_assets: Path
) -> None:
    # json.loads turns the escape "\\ud83d" in an HTTP reply body into a lone surrogate, which is not Unicode text.
    summary = "SYNTHETIC summary " + chr(0xD83D) + " end"
    description = "SYNTHETIC description " + chr(0xDC00) + chr(0xDC01) + " end"
    log = Log(replies=[summary, description, fenced(SECOND_FIX)])
    outcome = run_one(tmp_path, bench, "invalid-text", recipe(), log)
    trial, store = outcome.trial, outcome.store
    replaced = chr(0xFFFD)
    assert trial.context.knowledge_summary == "SYNTHETIC summary " + replaced + " end"
    requests = requests_of(trial)
    notes = [[(item.stage, item.severity, item.code) for item in request.diagnostics] for request in requests]
    assert notes == [[INVALID_TEXT], [INVALID_TEXT], []], "one warning on each context request whose reply held one"
    assert all(item.message for request in requests for item in request.diagnostics), "each warning says what happened"
    assert store.get(requests[0].reply_ref) == trial.context.knowledge_summary, "the reply is kept as stored"
    assert store.get(requests[1].reply_ref) == trial.context.source_description
    codes = [item.code for item in trial.attempts[0].diagnostics]
    assert "invalid-text" not in codes, "attempt 0 no longer carries a context reply's warning"


# ---------------------------------------------------------------------------
# trial.md and the Parquet mirror


def test_trial_md_shows_every_request_in_order(full_run: Outcome) -> None:
    requests = requests_of(full_run.trial)
    store = full_run.store
    page = (trial_dir(full_run.run_dir, full_run.trial_id) / "trial.md").read_text(encoding="ascii")
    position = 0
    for request in requests:
        for ref in [*(message.ref for message in request.messages), request.reply_ref]:
            found = page.find(ref.sha256, position)
            assert found >= 0, f"request {request.index}: trial.md shows sha256 {ref.sha256} after the previous one"
            position = found + len(ref.sha256)
    for text in {store.get(message.ref) for request in requests for message in request.messages}:
        assert md_form(text) in page, f"trial.md shows the message text {text[:60]!r}"


def test_the_parquet_mirror_has_a_requests_table_with_one_row_per_request(full_run: Outcome) -> None:
    assert "requests" in parquet.TABLES, "lassi.core.parquet.TABLES has no requests table; task P2.1 adds it"
    requests = requests_of(full_run.trial)
    written = read_run_parquet_rows(full_run)
    assert written == parquet.trial_rows([full_run.trial])["requests"], "the file holds what trial_rows flattens"
    assert [row["index"] for row in written] == [request.index for request in requests]
    for row, request in zip(written, requests, strict=True):
        assert row["trial_id"] == full_run.trial_id
        assert row["stage"] == request.stage
        assert row["attempt_index"] == request.attempt_index
        assert row["message_roles"] == [message.role for message in request.messages]
        assert row["message_sha256"] == [message.ref.sha256 for message in request.messages]
        assert row["reply_ref_sha256"] == request.reply_ref.sha256


def read_run_parquet_rows(outcome: Outcome) -> list[dict[str, Any]]:
    """Return the requests rows of the outcome's trial from the run's Parquet mirror."""
    rows = parquet.read_run_parquet(outcome.run_dir / "parquet")["requests"]
    return [row for row in rows if row["trial_id"] == outcome.trial_id]


# ---------------------------------------------------------------------------
# A trial.json written before this change


def p1_record(store: TextStore) -> Trial:
    """Return a trial as P1 records it, every text SYNTHETIC: a reference run, context, and one built attempt.

    As P1 wrote it, attempt 0 carries the invalid-text warning of a context
    reply that held a lone surrogate.
    """
    code = SECOND_FIX
    warning = Diagnostic(
        stage="parse",
        severity="warning",
        code="invalid-text",
        message="SYNTHETIC: Trial.context.knowledge_summary holds 1 U+FFFD replacement character(s)",
    )
    attempt = Attempt(
        index=0,
        prompt_ref=store.put("SYNTHETIC generation prompt\n"),
        response_text=fenced(code),
        files={target_file(OMP_TO_CUDA): code},
        stage_reached="S4",
        diagnostics=[warning],
    )
    return Trial(
        trial_id=make_trial_id("lassi-repro", MODEL_ID, SUITE, OMP_TO_CUDA.name, ITEM, 1),
        recipe_hash="0123456789abcdef" * 4,
        toolchain_pins=ToolchainPins(cuda="PLACEHOLDER-cuda-pin"),
        provenance=Provenance(
            commit="0123456789abcdef0123456789abcdef01234567",
            dirty=False,
            device="none (compile only)",
            sdk=None,
            date="2026-09-24T00:00:00+00:00",
        ),
        bench_item=BenchItem(suite=SUITE, item=ITEM, split="eval", direction=OMP_TO_CUDA.name),
        model=ModelInfo(
            backend="scripted", id=MODEL_ID, sampling=Sampling(temperature=0.0, top_p=1.0, max_tokens=4096)
        ),
        reference_run=RunInfo(exit_code=0, hang=False, wall_s=PLACEHOLDER_WALL_S, stdout_ref=store.put("SYNTHETIC\n")),
        context=Context(knowledge_summary="SYNTHETIC summary " + chr(0xFFFD), source_description="SYNTHETIC"),
        attempts=[attempt],
        final=Final(stage_reached="S4", corrections=0, wall_s=PLACEHOLDER_WALL_S),
    )


def test_a_trial_json_written_before_requests_were_recorded_loads_as_not_recorded(tmp_path: Path) -> None:
    require_requests_field()
    run_root = tmp_path / "run"
    store = TextStore(run_root)
    trial = p1_record(store)
    out = write_trial(trial, run_root, store)
    data = json.loads((out / "trial.json").read_text(encoding="utf-8"))
    assert "requests" in data, "trial.json holds every Trial field"
    del data["requests"]
    (out / "trial.json").write_bytes(json_text(data).encode("ascii"))
    loaded = read_trial(out, store)
    assert loaded.requests is None, "a trial.json without requests reads as not recorded (None), never as []"
    assert loaded == dataclasses.replace(trial, requests=None), "every other field loads as written"
    notes = [(item.stage, item.severity, item.code) for item in loaded.attempts[0].diagnostics]
    assert notes == [INVALID_TEXT], "a P1 record keeps its warning where P1 wrote it"
    assert parquet.trial_rows([loaded])["requests"] == [], "a trial whose requests were not recorded has no rows"
    assert render_trial_md(loaded, store).isascii(), "trial.md still renders an older record"


# ---------------------------------------------------------------------------
# The faithful lassi-2024 set (upstream extraction)


def upstream_systems(assets: RecipeAssets, direction: Direction) -> set[str]:
    """Return the general system prompt and the direction's system prompt of the extracted set."""
    key = DIRECTION_KEY[(direction.source, direction.target)]
    return {assets.fragments["system_prompt_dict.general_system"], assets.fragments[f"system_prompt_dict.{key}"]}


@pytest.mark.parametrize("direction", [OMP_TO_CUDA, CUDA_TO_OMP], ids=["omp-cuda", "cuda-omp"])
def test_a_faithful_mock_trial_holds_three_plus_corrections_requests_and_two_system_prompts(
    tmp_path: Path, bench: Path, lassi_assets: RecipeAssets, direction: Direction
) -> None:
    log = Log()
    data = recipe(direction, prompts=UPSTREAM_SET, context=UPSTREAM_PACKS, backend="mock", model_id=MOCK_ID)
    outcome = run_one(tmp_path, bench, "faithful-mock", data, log)
    trial = outcome.trial
    assert trial.final.stage_reached == "S4" and len(log.requests) == 3, "the mock answers with the reference"
    requests = requests_of(trial)
    assert len(requests) == 3 + trial.final.corrections
    assert [request.stage for request in requests] == ["summarize_context", "describe_source", "generate"]
    for position, request in enumerate(requests):
        assert refs_recorded(request) == refs_sent(log.requests[position]), f"request {position}, system included"
    assert system_texts(requests, outcome.store) == upstream_systems(lassi_assets, direction)


def test_a_faithful_scripted_trial_with_corrections_records_every_request(
    tmp_path: Path, bench: Path, lassi_assets: RecipeAssets
) -> None:
    outcome = run_full(tmp_path, bench, prompts=UPSTREAM_SET, context=UPSTREAM_PACKS)
    requests = requests_of(outcome.trial)
    assert len(requests) == 3 + outcome.trial.final.corrections
    assert [request.stage for request in requests] == FULL_STAGES
    assert [request.attempt_index for request in requests] == FULL_ATTEMPTS
    for position, request in enumerate(requests):
        assert refs_recorded(request) == refs_sent(outcome.log.requests[position]), f"request {position}"
        assert request.reply_ref == text_ref(FULL_REPLIES[position])
    assert system_texts(requests, outcome.store) == upstream_systems(lassi_assets, OMP_TO_CUDA)
