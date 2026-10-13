"""Tests for a model request past the model's context: the trial ends at context-exceeded, the run goes on (P17.4).

Bible: Model Serving (Serving Rules: a prompt that exceeds the model's context
fails the trial and is never truncated), Component Interfaces (LLMBackend),
Result Record (final.end_reason), Evaluation Protocol (LASSI score
profile); Decision Log 2026-09-28; plans/p17-portable.md, task P17.4; the
P17.4 design (section 8).

The contract these tests fix:

- lassi.core.interfaces.ContextExceeded(prompt_tokens, max_tokens, context)
  is what a backend raises for a request whose prompt tokens plus max_tokens
  pass its context. It keeps the three numbers, its message names them and
  says nothing was truncated, and it is neither a ValueError nor a
  ServingError, so no existing handler catches it by accident.
  LLMBackend.complete's docstring names it.
- lassi.core.record END_REASONS gains `context-exceeded`, which
  lassi.core.stages names CONTEXT_EXCEEDED.
- Each stage that asks the model ends the trial with that code when its one
  model call raises ContextExceeded: generate, summarize_context, and
  describe_source, and the correction of compile_loop and run_loop. The
  message names the stage and holds the error's text. Every attempt and
  request made before stays as it was; no attempt or request is added for
  the refused call, and no attempt is built or run again. No later stage
  runs, and the run goes on to the next trial and ends with status complete.
- The end reason round-trips through trial.json, trial.md, and the Parquet
  trials table.
- The lassi profile reads a trial with no attempt that ended at
  context-exceeded as a failed trial, not a baseline end: correct, first_try,
  and compiled are 0 when the target reference ran, and the similarity
  components are null with the no-attempt note. With attempts, the output
  that stands is scored as for any trial.

Every component is a fake in a test Registry beside the real stages: the
backend answers from a script or raises ContextExceeded with SYNTHETIC
numbers, the toolchain writes a PLACEHOLDER artifact (a file holding #error
fails), and the executor returns SYNTHETIC runs. No model, compiler, or
program runs. No value in this module is a measurement.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

import lassi.prompts.assets as prompt_assets
from lassi.bench import Direction, load_suite
from lassi.core import fragments
from lassi.core import record as record_module
from lassi.core import runner as runner_module
from lassi.core import stages as stages_module
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Limits, LLMBackend, Message, RunResult, Sampling
from lassi.core.parquet import read_run_parquet
from lassi.core.record import (
    BenchItem,
    Diagnostic,
    EndReason,
    Final,
    ModelInfo,
    Provenance,
    RunInfo,
    Trial,
    make_trial_id,
)
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.llm import ServingError

REPO = Path(__file__).resolve().parents[2]
SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
ITEM = "layout"
MODEL_ID = "scripted-fixture"
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
CONTEXT_EXCEEDED = "context-exceeded"
# SYNTHETIC numbers of a refused request.
PROMPT_TOKENS, MAX_TOKENS, CONTEXT = 900, 64, 512
REFUSE = "<refuse>"
SOURCES = {
    "omp": '#include <cstdio>\nint main() {\n#pragma omp target\n  { }\n  std::printf("done\\n");\n}\n',
    "cuda": "#include <cstdio>\n__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n",
}
GOOD = render_file_blocks({"main.cu": "int main() { return 0; }\n"})
BROKEN = render_file_blocks({"main.cu": "#error SYNTHETIC broken attempt\nint main() { return 0; }\n"})
COMPILE_ERROR = Diagnostic(stage="compile", severity="error", code="SYNTHETIC-error", message="SYNTHETIC: #error")
FAILED_RUN = RunResult(exit_code=1, hang=False, stdout="SYNTHETIC output\nFAIL\n", stderr="", wall_s=0.25)
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=MAX_TOKENS)
FRAGMENT_SET = "synthetic-fragments"
OMP_TO_CUDA = Direction("omp", "cuda")
FRAGMENTS = {
    fragments.fragment_key(key, OMP_TO_CUDA): f"<{key}>"
    for key in (*fragments.GENERATE_KEYS, *fragments.SUMMARY_KEYS, *fragments.DESCRIPTION_KEYS)
}


def context_exceeded() -> type:
    """Return lassi.core.interfaces.ContextExceeded, failing the test clearly while it does not exist."""
    module = importlib.import_module("lassi.core.interfaces")
    if not hasattr(module, "ContextExceeded"):
        pytest.fail("lassi.core.interfaces has no ContextExceeded yet (task P17.4)")
    return module.ContextExceeded


def require_the_code() -> None:
    """Fail the test clearly while END_REASONS lacks context-exceeded."""
    if CONTEXT_EXCEEDED not in record_module.END_REASONS:
        pytest.fail(f"lassi.core.record.END_REASONS has no {CONTEXT_EXCEEDED!r} yet (task P17.4)")


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate and compile variables, point TMPDIR at a test directory, and fake git."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    answers = {"rev-parse": FAKE_COMMIT + "\n", "status": ""}
    monkeypatch.setattr(runner_module, "_git", lambda *args: answers[args[0]])


# ---------------------------------------------------------------------------
# Fakes


@dataclass
class Log:
    """The backend's script and what the fakes saw: requests, builds, and runs."""

    script: list[str] = field(default_factory=list)
    requests: list[list[Message]] = field(default_factory=list)
    builds: int = 0
    runs: int = 0


def scripted_backend(log: Log) -> type:
    """Return an LLMBackend "scripted" that answers from log.script in order; REFUSE raises ContextExceeded."""
    refusal = context_exceeded()

    class Scripted:
        """Answers each request with the next scripted reply, or refuses it as past the context."""

        name = "scripted"
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Record the request; raise ContextExceeded for REFUSE, else return the reply."""
            log.requests.append(list(messages))
            assert log.script, "the backend was asked for more replies than the script holds"
            reply = log.script.pop(0)
            if reply == REFUSE:
                raise refusal(PROMPT_TOKENS, MAX_TOKENS, CONTEXT)
            return Completion(text=reply, prompt_tokens=0, completion_tokens=0)

    return Scripted


def fake_toolchain(log: Log) -> type:
    """Return a Toolchain "nvcc-sm80" without PIN: a file holding #error fails, anything else builds a PLACEHOLDER."""

    class FakeToolchain:
        """Counts builds, writes the files, and reports the build; compiles nothing."""

        name = "nvcc-sm80"
        capabilities = frozenset({"diagnostics"})

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Write every file; fail with one compile error when a file holds #error, else write an artifact."""
            log.builds += 1
            for path, text in [*files.items(), *(harness or {}).items()]:
                target = Path(workdir) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            if any("#error" in text for text in files.values()):
                return BuildResult(artifact=None, diagnostics=[COMPILE_ERROR])
            artifact = Path(workdir) / "main"
            artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=[])

    return FakeToolchain


class CompileOnly:
    """The Executor "none": compile only, runs nothing."""

    name = "none"
    capabilities = frozenset({"compile_only"})

    def device(self) -> str:
        """Name no device, as a compile-only executor does."""
        return "none (compile only)"

    def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        """Fail the test: nothing runs on the compile-only path."""
        raise AssertionError("the compile-only path ran an executor")


def failing_executor(log: Log) -> type:
    """Return a sandboxed Executor "failing" whose every run exits 1; it runs nothing."""

    class Failing:
        """Counts runs and returns a SYNTHETIC failed run."""

        name = "failing"
        capabilities = frozenset({"runs_code", "sandboxed"})

        def device(self) -> str:
            """Name the fake's SYNTHETIC device."""
            return "SYNTHETIC device"

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Count the run and return the failed run."""
            log.runs += 1
            return FAILED_RUN

    return Failing


def make_registry(log: Log) -> Registry:
    """Return a test Registry: the scripted backend, the fake toolchain, both executors, and the real stages."""
    registry = Registry()
    registry.register("LLMBackend", "scripted", scripted_backend(log))
    registry.register("Toolchain", "nvcc-sm80", fake_toolchain(log))
    registry.register("Executor", "none", CompileOnly)
    registry.register("Executor", "failing", failing_executor(log))
    for name in ("generate", "compile_loop", "run_loop", "summarize_context", "describe_source"):
        registry.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    return registry


# ---------------------------------------------------------------------------
# Recipes, bench sources, assets, and runs


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
def fragment_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a temporary assets root with a SYNTHETIC fragment set and a cuda pack; the loader reads it."""
    root = tmp_path / "assets"
    write_tree(root / "prompts" / FRAGMENT_SET, FRAGMENTS)
    write_tree(root / "context" / "cuda-pack", {"packdict.cuda": "<cuda pack>"})
    monkeypatch.setattr(prompt_assets, "default_root", lambda: root)
    return root


def write_bench(root: Path) -> Path:
    """Write the item's SYNTHETIC sources where the suite manifest lays them out; return `root`."""
    spec = load_suite(SUITE_MANIFEST).items[ITEM]
    for language, text in SOURCES.items():
        layout = spec.languages[language]
        path = root / layout.dir / layout.files[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def template_recipe(stages: Sequence[str], executor: str = "none", **changes: Any) -> dict[str, Any]:
    """Return a p0-smoke recipe for the layout item, omp to cuda, with the scripted backend."""
    data: dict[str, Any] = {
        "extends": "base",
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": MAX_TOKENS}},
        "loop": {"max_corrections": 5},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": "omp", "target": "cuda"}],
        "prompts": "p0-smoke",
        "toolchain": {"cuda": "nvcc-sm80"},
        "stages": list(stages),
        "executor": {"kind": executor},
        "trials": {"n": 1},
    }
    data.update(changes)
    return data


def fragment_recipe(stages: Sequence[str]) -> dict[str, Any]:
    """Return a recipe on the SYNTHETIC fragment set with the cuda pack, as tests/core/test_fragment_prompts.py has."""
    return {
        "extends": "base",
        "faithful": False,
        "fixes": {"fence_tag": False, "prompt_spaces": False},
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": MAX_TOKENS}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": "omp", "target": "cuda"}],
        "prompts": FRAGMENT_SET,
        "context": ["cuda-pack"],
        "stages": list(stages),
        "executor": {"kind": "none"},
        "trials": {"n": 1},
    }


def run(tmp_path: Path, log: Log, data: Mapping[str, Any], name: str = "context-run") -> Path:
    """Write the recipe `name` and run it with the fakes; return the run directory."""
    path = tmp_path / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    options = RunOptions(
        runs_root=tmp_path / "runs-root", run_id="r", bench_root=write_bench(tmp_path / "bench"),
        registry=make_registry(log),
    )
    return run_recipe(path, options)


def read(run_dir: Path, name: str = "context-run", number: int = 1) -> Trial:
    """Return trial `number` of the run, read back from its trial.json."""
    return read_trial(trial_dir(run_dir, make_trial_id(name, MODEL_ID, SUITE, "omp-cuda", ITEM, number)),
                      TextStore(run_dir))


def assert_ended_at(trial: Trial, stage: str) -> None:
    """Assert the trial ended at context-exceeded, with a message naming `stage` and the refused request's numbers."""
    reason = trial.final.end_reason
    assert reason is not None and reason.code == CONTEXT_EXCEEDED, reason
    assert stage in reason.message, reason.message
    for number in (PROMPT_TOKENS, MAX_TOKENS, CONTEXT):
        assert str(number) in reason.message, reason.message


# ---------------------------------------------------------------------------
# The error and the code


def test_context_exceeded_is_its_own_error() -> None:
    refusal = context_exceeded()
    error = refusal(PROMPT_TOKENS, MAX_TOKENS, CONTEXT)
    assert (error.prompt_tokens, error.max_tokens, error.context) == (PROMPT_TOKENS, MAX_TOKENS, CONTEXT)
    message = str(error)
    assert all(str(number) in message for number in (PROMPT_TOKENS, MAX_TOKENS, CONTEXT)), message
    assert "truncated" in message, "the message says nothing was truncated"
    assert issubclass(refusal, Exception)
    assert not issubclass(refusal, (ValueError, ServingError)), "no existing handler catches it by accident"
    assert inspect.getdoc(refusal)
    assert "ContextExceeded" in (inspect.getdoc(LLMBackend.complete) or ""), "the contract names the error"


def test_the_end_code_joins_the_record() -> None:
    require_the_code()
    assert getattr(stages_module, "CONTEXT_EXCEEDED", None) == CONTEXT_EXCEEDED
    reason = EndReason(code=CONTEXT_EXCEEDED, message="generate: SYNTHETIC: the request was refused")
    assert record_module.from_json(EndReason, record_module.to_json(reason)) == reason


# ---------------------------------------------------------------------------
# Each stage that asks the model


def test_generate_refused_ends_the_trial_before_any_attempt(tmp_path: Path) -> None:
    log = Log(script=[REFUSE, REFUSE])
    run_dir = run(tmp_path, log, template_recipe(["generate", "compile_loop"], trials={"n": 2}))
    for number in (1, 2):
        trial = read(run_dir, number=number)
        assert_ended_at(trial, "generate")
        assert (trial.attempts, trial.requests) == ([], []), "a refused request records no attempt and no request"
        assert (trial.final.stage_reached, trial.final.corrections) == (None, 0)
    assert log.builds == 0, "no later stage ran"
    assert len(log.requests) == 2, "the run went on to the second trial"
    provenance = (run_dir / "provenance.json").read_bytes().decode("ascii")
    assert '"status": "complete"' in provenance
    assert (run_dir / "run.md").is_file()
    rows = read_run_parquet(run_dir / "parquet")["trials"]
    assert [row["final_end_reason_code"] for row in rows] == [CONTEXT_EXCEEDED] * 2
    trial_md = trial_dir(run_dir, make_trial_id("context-run", MODEL_ID, SUITE, "omp-cuda", ITEM, 1)) / "trial.md"
    assert CONTEXT_EXCEEDED in trial_md.read_bytes().decode("ascii")


def test_summarize_context_refused_ends_the_trial(tmp_path: Path, fragment_assets: Path) -> None:
    log = Log(script=[REFUSE])
    run_dir = run(tmp_path, log, fragment_recipe(["summarize_context", "describe_source", "generate"]))
    trial = read(run_dir)
    assert_ended_at(trial, "summarize_context")
    assert trial.context == record_module.Context(), "no context field was filled"
    assert (trial.attempts, trial.requests) == ([], [])
    assert len(log.requests) == 1, "no later stage asked the model"


def test_describe_source_refused_keeps_the_summary(tmp_path: Path, fragment_assets: Path) -> None:
    log = Log(script=["SYNTHETIC summary", REFUSE])
    run_dir = run(tmp_path, log, fragment_recipe(["summarize_context", "describe_source", "generate"]))
    trial = read(run_dir)
    assert_ended_at(trial, "describe_source")
    assert trial.context.knowledge_summary == "SYNTHETIC summary"
    assert trial.context.source_description == record_module.Context().source_description, "left unfilled"
    assert trial.requests is not None and [request.stage for request in trial.requests] == ["summarize_context"]
    assert trial.attempts == []
    assert len(log.requests) == 2


def test_a_compile_correction_refused_keeps_earlier_attempts(tmp_path: Path) -> None:
    log = Log(script=[BROKEN, REFUSE])
    run_dir = run(tmp_path, log, template_recipe(["generate", "compile_loop"]))
    trial = read(run_dir)
    assert_ended_at(trial, "compile_loop")
    (attempt,) = trial.attempts
    assert attempt.index == 0 and any(item.severity == "error" for item in attempt.diagnostics), attempt
    assert log.builds == 1, "attempt 0 is not built again"
    assert trial.requests is not None and [request.stage for request in trial.requests] == ["generate"]
    assert trial.final.corrections == 0


def test_a_later_correction_refused_keeps_every_earlier_attempt(tmp_path: Path) -> None:
    log = Log(script=[BROKEN, BROKEN, REFUSE])
    run_dir = run(tmp_path, log, template_recipe(["generate", "compile_loop"]))
    trial = read(run_dir)
    assert_ended_at(trial, "compile_loop")
    assert [attempt.index for attempt in trial.attempts] == [0, 1]
    assert log.builds == 2
    assert trial.requests is not None and [request.attempt_index for request in trial.requests] == [0, 1]
    assert trial.final.corrections == 1


def test_a_run_correction_refused_keeps_earlier_attempts(tmp_path: Path) -> None:
    log = Log(script=[GOOD, REFUSE])
    run_dir = run(tmp_path, log, template_recipe(["generate", "compile_loop", "run_loop"], executor="failing"))
    trial = read(run_dir)
    assert_ended_at(trial, "run_loop")
    (attempt,) = trial.attempts
    assert attempt.stage_reached == "S4" and attempt.run.exit_code == 1, attempt
    assert (log.builds, log.runs) == (1, 1), "attempt 0 is neither built nor run again"
    assert trial.requests is not None and [request.stage for request in trial.requests] == ["generate"]


# ---------------------------------------------------------------------------
# How the lassi profile reads it


def ran() -> RunInfo:
    """Return a SYNTHETIC reference run that exited 0."""
    return RunInfo(exit_code=0, hang=False, wall_s=0.5, stdout_truncated=False, stderr_truncated=False,
                   workdir_incomplete=False)


def ended_trial(attempts: Sequence[Any] = ()) -> Trial:
    """Return a SYNTHETIC trial that ended at context-exceeded after `attempts`, its reference run done."""
    require_the_code()
    reason = EndReason(code=CONTEXT_EXCEEDED, message="generate: SYNTHETIC: the request was refused")
    return Trial(
        trial_id=make_trial_id("context-scoring", MODEL_ID, SUITE, "omp-cuda", ITEM, 1),
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(commit=FAKE_COMMIT, dirty=False, device="SYNTHETIC device", sdk=None,
                              date="2026-10-06T00:00:00+00:00"),
        bench_item=BenchItem(suite=SUITE, item=ITEM, split="eval", direction="omp-cuda"),
        model=ModelInfo(backend="scripted", id=MODEL_ID, sampling=SAMPLING),
        reference_run=ran(),
        requests=[],
        attempts=list(attempts),
        final=Final(stage_reached=None, corrections=0, end_reason=reason),
    )


def test_the_lassi_profile_reads_context_exceeded_as_a_failed_trial(tmp_path: Path) -> None:
    from lassi.scoring.lassi_profile import BASELINE_ENDS, LassiProfile

    assert CONTEXT_EXCEEDED not in BASELINE_ENDS, "the model was asked, so it is not a baseline end"
    (tmp_path / "bench").mkdir()
    profile = LassiProfile(bench_root=tmp_path / "bench")
    score = profile.score(ended_trial())
    found = {name: score.components[name] for name in ("correct", "first_try", "compiled", "compiled_first_try")}
    assert found == {"correct": 0.0, "first_try": 0.0, "compiled": 0.0, "compiled_first_try": 0.0}, found
    assert score.scalar == 0.0
    no_attempt = profile.profile.notes["no_attempt"]
    for name in ("sim_t", "sim_t_c", "sim_l"):
        assert score.components[name] is None and score.notes[name] == no_attempt, (name, score.notes)
    assert score.components["cap_hit"] == 0.0
