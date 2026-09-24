"""Tests for the run_loop stage: execution gate, stale output, run flags, and the unload request (task P1.6).

Bible: Source Papers (LASSI quirk table, loop and Ollama rows), Component
Interfaces (Executor rules; capability rule), Execution Backends (none),
Result Record (Attempt.run, Diagnostic, Final.end_reason), Sandbox, Design
Principle 4, Agent Rules 4 and 6.

The contract these tests fix, from the P1.6 acceptance criteria
(plans/p1-faithful.md) and the P1 phase notes ("For P1.6 (from P1.5)",
"Baseline on this host", "Reference stdout", "Run flags"):

- A stage registered as `run_loop` runs each compiling attempt through the
  bound executor when the executor declares `runs_code`, with the item's run
  arguments. It records Attempt.run (exit code, hang flag, wall time, and
  the stdout by text reference); a clean run is S5. A run error is fed back
  to the model with upstream's execute-error prompt (fragment set: the
  previous code, correct.run_error_head, the setup.<target>.compiler
  fragment, one space, the setup.<target>.flags fragment,
  correct.run_error_tail, upstream's report of the run, and correct.outro,
  every LF removed under the prompt_newlines quirk), and run corrections
  share one count with compile corrections. Upstream's report is
  execute_code's return_result: the execute.exit_lead fragment, the return
  code as Popen gives it (a death by signal N, which the sandbox reports as
  128 + N, is -N), one space, the execute.segfault fragment for -11, and,
  when the run wrote any stderr, the execute.stderr_lead fragment and the
  stderr read in text mode. A template set keeps this project's wording.
- Model-generated code runs only in the sandbox (Agent Rule 6): the runner
  refuses an executor that runs programs but does not declare `sandboxed`
  when a listed stage runs attempts.
- A new named fix `execution_gate` in lassi.core.recipe.FIXES, on by
  default and off under `faithful: true`, reproduced by run_loop. Off: a
  compiling attempt runs only while its correction count is at most 7. A
  compiling attempt after 8 or more corrections ends the trial unexecuted;
  when an earlier attempt ran, its stdout stands as the trial's output and
  the last attempt carries a run-stage warning Diagnostic, code
  `stale-output`, naming that attempt; when none ran, the trial ends with
  final.end_reason `upstream-crash` (a new code in END_REASONS), no stdout,
  and no alignment. On: the recipe's cap applies and no stale output is
  used. Every faithful behavior is keyed on a fix or loop.max_corrections,
  never on the faithful flag, so `faithful: false` with every fix off and a
  numeric cap behaves as `faithful: true` except for the cap.
- With executor `none`, run_loop records nothing, and a faithful trial ends
  at its first compiling attempt, past the gate too, with no end reason
  (upstream reads no run output when it does not execute).
- The attempt run limit: with sandbox.wall_s `baseline_x10`, ten times
  Trial.reference_run.wall_s, but never less than RUN_WALL_FLOOR_S (a
  positive number in the module that defines run_loop), which also applies
  when no reference ran; memory from sandbox.mem_gb; cpus RUN_CPUS (an int
  of at least 1 in the same module).
- RunResult's stdout_truncated, stderr_truncated, and workdir_incomplete
  become run-stage warning Diagnostics on the attempt, with codes
  `stdout-truncated`, `stderr-truncated`, and `workdir-incomplete`.
- A backend declaring the capability `unload_before_run` is asked to unload
  (its `unload()` method) once at trial start, before anything else, as the
  notebook's setup does, and right before each run of an attempt; not before
  the baseline's reference run, which the notebook runs without one. A
  backend without the capability is never asked, and no stage checks a
  backend's type (the unloading backend here is not the Ollama class).
- End to end, [baseline, summarize_context, describe_source, generate,
  compile_loop, run_loop, oracle] under `faithful: false`, every fix off, a
  numeric cap, oracle {kind: stdout_mask, passfail: true}, and the
  lassi-2024 prompt set fills Attempt.alignment from the run's stdout
  against Trial.reference_run's stdout.
- A hung run is a run error fed back like any other. Before any directory
  exists, the runner refuses a sandbox.wall_s it cannot apply when the
  executor runs programs, and a backend declaring `unload_before_run`
  without unload().

The fragment-set tests compute expected prompts from a fresh extraction of
the pinned upstream checkout into a temporary directory
(tools/extract_lassi_assets.py) and skip, naming the tool, when
third_party/LASSI is absent or not at the pin. No upstream text is copied
into this file (OQ-018). Model replies, compiler stderr, program output, wall
times, and bench sources are SYNTHETIC texts and values written here; fake
toolchains compile nothing and the scripted executor runs nothing. No value
in this module is a measurement.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

import lassi.prompts as prompts_module
import lassi.prompts.assets as prompt_assets
from lassi.bench import Direction, load_suite
from lassi.core import record as record_module
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Limits, Message, RunResult, Sampling
from lassi.core.recipe import FIXES, load_recipe
from lassi.core.record import Alignment, Attempt, Diagnostic, Trial, make_trial_id
from lassi.core.registry import DEFAULT_REGISTRY, Registry, RegistryError
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.executors.workdir import build_dir
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
FRAGMENT_SET = "lassi-2024"
TEMPLATE_SET = "p0-smoke"
OMP_TO_CUDA = Direction("omp", "cuda")
CUDA_TO_OMP = Direction("cuda", "omp")
DIRECTIONS = [CUDA_TO_OMP, OMP_TO_CUDA]
DIRECTION_IDS = ["cuda-omp", "omp-cuda"]
TOOLCHAINS = {"cuda": "nvcc-sm80", "omp": "nvcpp-cc80"}
PACK_OF = {"omp": "openmp-4.0-card", "cuda": "cuda-12.5-ch5"}
# Upstream's dictionary entry name for each direction: `<SOURCE>_to_<TARGET>` in upstream's language spelling.
DIRECTION_KEY = {("omp", "cuda"): "OMP_to_CUDA", ("cuda", "omp"): "CUDA_to_OMP"}
RUN_STAGES = ["baseline", "summarize_context", "describe_source", "generate", "compile_loop", "run_loop"]
ORACLE_STAGES = [*RUN_STAGES, "oracle"]
ORACLE = {"kind": "stdout_mask", "passfail": True}
GATE_FIX = "execution_gate"
UPSTREAM_CRASH = "upstream-crash"
STALE_OUTPUT = "stale-output"
UNLOAD_BEFORE_RUN = "unload_before_run"
# The fragment keys of upstream's report of a failed run (execute_code's return_result), by role.
EXIT_LEAD = "execute.exit_lead"
SEGFAULT_NOTE = "execute.segfault"
STDERR_LEAD = "execute.stderr_lead"
# The return code Popen gives a death by SIGSEGV, for which upstream's report adds its segfault note.
POPEN_SEGFAULT = -11
# The RunResult flags and the code of the run-stage warning each becomes.
RUN_FLAGS = {
    "stdout_truncated": "stdout-truncated",
    "stderr_truncated": "stderr-truncated",
    "workdir_incomplete": "workdir-incomplete",
}
FLOOR_NAME = "RUN_WALL_FLOOR_S"
CPUS_NAME = "RUN_CPUS"
COMPILED = "S4"
RAN_CLEAN = "S5"
# The run arguments the suite manifest lists for layout.
LAYOUT_ARGS = ["1"]

# SYNTHETIC bench sources.
OMP_SOURCE = "#include <cstdio>\nint main() {\n    int  n  =  4;\n    std::printf(\"n   = %d\\n\", n);\n}\n"
CUDA_SOURCE = "#include <cstdio>\n__global__ void kernel(int *out) {\n    out[threadIdx.x]  =  1;\n}\n"
SOURCES = {"omp": OMP_SOURCE, "cuda": CUDA_SOURCE}

# SYNTHETIC model replies. BAD_CODE holds an #error line, which the fake toolchains refuse.
SUMMARY_REPLY = "Synthetic summary of the pack."
DESCRIPTION_REPLY = "Synthetic description of the source."
BAD_CODE = "#error SYNTHETIC not translated yet\nint main() {\n    return 1;\n}\n"
GOOD_CODE = "int main() {\n    return 0;\n}\n"

# SYNTHETIC compiler output for a build of a file holding #error.
RAW_STDERR = '"main.x", line 1: error: SYNTHETIC-RAW-STDERR compiler   text\n  #error SYNTHETIC not translated yet\n'
PARSED = Diagnostic(stage="compile", severity="error", code="synthetic", message="SYNTHETIC parsed diagnostic")

# SYNTHETIC program output in layout's print format (values invented), and SYNTHETIC wall times in seconds.
LAYOUT = "Average kernel execution time (AoS): 1.5 (us)\nPASS\nAverage kernel execution time (SoA): 2.5 (us)\nPASS\n"
LAYOUT_OTHER_TIMES = LAYOUT.replace("1.5 (us)", "3.75 (us)").replace("2.5 (us)", "0.5 (us)")
LAYOUT_FAIL = LAYOUT_OTHER_TIMES.replace("PASS", "FAIL")
RUN_STDERR = "SYNTHETIC-RUN-STDERR: fault in   step\nsecond line of the run's stderr\n"
RUN_ERROR_EXIT = 3
REFERENCE_WALL_S = 1.25
ATTEMPT_WALL_S = 0.5


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate and compile variables a test could inherit, and point TMPDIR at a test directory."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CPATH", raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))


# ---------------------------------------------------------------------------
# Upstream extraction (fragment-set tests only)


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
    spec = importlib.util.spec_from_file_location("extract_lassi_assets_for_p16", TOOL)
    assert spec and spec.loader, f"cannot load {TOOL}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def assets_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
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
def lassi_assets(assets_root: Path, monkeypatch: pytest.MonkeyPatch) -> RecipeAssets:
    """Point the asset loader and the prompt sets at the fresh extraction; return its fragments and both packs."""
    monkeypatch.setattr(prompt_assets, "default_root", lambda: assets_root)
    monkeypatch.setattr(prompts_module, "default_roots", lambda: (assets_root / "prompts", REPO / "assets" / "prompts"))
    return load_recipe_assets({"prompts": FRAGMENT_SET, "context": list(PACK_OF.values())}, root=assets_root)


# ---------------------------------------------------------------------------
# Names this task adds, looked up so a missing one fails its test with a clear message


def require_fixes(*names: str) -> None:
    """Fail the test clearly when a named fix is not in lassi.core.recipe.FIXES yet."""
    missing = [name for name in names if name not in FIXES]
    if missing:
        pytest.fail(f"lassi.core.recipe.FIXES has no {', '.join(missing)}; task P1.6 adds it")


def registered_stage(name: str) -> type:
    """Return the Stage class registered as `name` in DEFAULT_REGISTRY; fail clearly while it is missing."""
    try:
        return DEFAULT_REGISTRY.get("Stage", name).factory
    except RegistryError as error:
        pytest.fail(f"no stage is registered as {name!r} ({error}); task P1.6 adds the run_loop stage")


def run_loop_constant(name: str) -> Any:
    """Return the constant `name` of the module that defines run_loop; fail clearly while it is missing."""
    module = sys.modules[registered_stage("run_loop").__module__]
    if not hasattr(module, name):
        pytest.fail(f"{module.__name__} has no {name}; task P1.6 adds it beside run_loop")
    return getattr(module, name)


def wall_floor() -> float:
    """Return RUN_WALL_FLOOR_S, which must be a positive number."""
    value = run_loop_constant(FLOOR_NAME)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        pytest.fail(f"{FLOOR_NAME} must be a number above zero, got {value!r}")
    return float(value)


def run_cpus() -> int:
    """Return RUN_CPUS, which must be an int of at least 1."""
    value = run_loop_constant(CPUS_NAME)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        pytest.fail(f"{CPUS_NAME} must be an int of at least 1, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Fake components


@dataclass
class Run:
    """One run the scripted executor was asked for: `kind` is "reference" or "attempt"."""

    kind: str
    artifact: Path
    inputs: list[str]
    limits: Limits


@dataclass
class Log:
    """The script the fakes follow and what they saw.

    `replies` are the model's replies in order. `reference` is what every
    baseline run returns, and `attempt_runs` what the runs of attempts return,
    in order; a run past the script fails the test. `events` lists, in order,
    "unload", "complete", "build", "run:reference", and "run:attempt".
    """

    replies: list[str] = field(default_factory=list)
    reference: RunResult = field(
        default_factory=lambda: RunResult(exit_code=0, hang=False, stdout=LAYOUT, stderr="", wall_s=REFERENCE_WALL_S)
    )
    attempt_runs: list[RunResult] = field(default_factory=list)
    builds: list[Path] = field(default_factory=list)
    runs: list[Run] = field(default_factory=list)
    requests: list[tuple[list[Message], Sampling]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


def fake_toolchain(registered_as: str, log: Log) -> type:
    """Return a Toolchain class without PIN: a file holding `#error` fails, anything else builds a PLACEHOLDER.

    The raw stderr goes to `compile.stderr` in the workdir, named by
    BuildResult.stderr_ref, as build() keeps it. It compiles and runs nothing.
    """

    class FakeToolchain:
        """Writes the files, keeps a raw stderr attachment, and reports a build; compiles nothing."""

        name = registered_as
        capabilities = frozenset({"diagnostics"})

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Record the build, write every file and the stderr attachment, and return the result."""
            workdir = Path(workdir)
            log.events.append("build")
            log.builds.append(workdir)
            for path, text in [*files.items(), *(harness or {}).items()]:
                target = workdir / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            failed = any("#error" in text for text in files.values())
            (workdir / "compile.stderr").write_bytes(RAW_STDERR.encode("ascii") if failed else b"")
            if failed:
                return BuildResult(artifact=None, diagnostics=[PARSED], stderr_ref="compile.stderr")
            artifact = workdir / "main"
            artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=[], stderr_ref="compile.stderr")

    return FakeToolchain


def scripted_executor(log: Log) -> type:
    """Return an Executor class that runs programs: baseline runs get the reference, attempts the script in order."""

    class ScriptedExecutor:
        """Records each run and returns its SYNTHETIC RunResult; runs nothing."""

        name = "scripted"
        capabilities = frozenset({"runs_code", "sandboxed"})

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Record the run and return the reference result or the next scripted attempt result."""
            artifact = Path(artifact)
            if artifact.parent.parent.name.startswith("baseline-"):
                log.events.append("run:reference")
                log.runs.append(Run("reference", artifact, list(inputs), limits))
                return log.reference
            log.events.append("run:attempt")
            log.runs.append(Run("attempt", artifact, list(inputs), limits))
            assert log.attempt_runs, (
                f"the executor was asked to run {artifact}, but the script holds no more attempt runs; "
                "the execution gate or the correction cap should have stopped the loop before this run"
            )
            return log.attempt_runs.pop(0)

    return ScriptedExecutor


class CompileOnlyExecutor:
    """A compile-only Executor registered as "none"; being asked to run anything fails the test."""

    name = "none"
    capabilities = frozenset({"compile_only"})

    def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        """Fail the test: a compile-only executor is never asked to run a program."""
        raise AssertionError(f"a compile-only executor was asked to run {artifact}")


def scripted_backend(log: Log, *, unloading: bool) -> type:
    """Return an LLMBackend class, registered as "scripted", that answers from `log.replies` in order.

    With `unloading` it declares `unload_before_run` and logs each unload;
    without it, it has an unload() method that fails the test when called.
    """

    class ScriptedBackend:
        """Records each request and answers with the next scripted reply."""

        name = "scripted"
        capabilities = frozenset({"chat", UNLOAD_BEFORE_RUN}) if unloading else frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id, as every backend does."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Record the request and return the next scripted reply."""
            log.events.append("complete")
            log.requests.append((list(messages), sampling))
            assert log.replies, "the backend was asked for more replies than the script holds"
            return Completion(text=log.replies.pop(0), prompt_tokens=0, completion_tokens=0)

        def unload(self) -> None:
            """Log the unload, or fail the test when the backend does not declare unload_before_run."""
            assert unloading, "a backend that does not declare unload_before_run was asked to unload"
            log.events.append("unload")

    return ScriptedBackend


def make_registry(log: Log, stages: Sequence[str], *, unloading: bool = False) -> Registry:
    """Return a test Registry: the scripted backend, both executors, fake toolchains, the real oracle and stages."""
    registry = Registry()
    registry.register("LLMBackend", "scripted", scripted_backend(log, unloading=unloading))
    registry.register("Executor", "none", CompileOnlyExecutor)
    registry.register("Executor", "scripted", scripted_executor(log))
    for name in TOOLCHAINS.values():
        registry.register("Toolchain", name, fake_toolchain(name, log))
    registry.register("Oracle", ORACLE["kind"], DEFAULT_REGISTRY.get("Oracle", ORACLE["kind"]).factory)
    for name in stages:
        registry.register("Stage", name, registered_stage(name))
    return registry


# ---------------------------------------------------------------------------
# Recipes, bench sources, replies, and runs


def write_bench(root: Path) -> Path:
    """Write the item's SYNTHETIC source per language where the suite manifest lays it out; return `root`."""
    spec = load_suite(SUITE_MANIFEST).items[ITEM]
    for language, text in SOURCES.items():
        layout = spec.languages[language]
        path = root / layout.dir / layout.files[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def target_file(direction: Direction) -> str:
    """Return the item's one target file name for `direction`."""
    return load_suite(SUITE_MANIFEST).items[ITEM].languages[direction.target].files[0]


def fragment_recipe(direction: Direction = CUDA_TO_OMP, *, faithful: bool = True, **changes: Any) -> dict[str, Any]:
    """Return a lassi-2024 recipe for layout and `direction` with the scripted executor; `changes` add keys."""
    data: dict[str, Any] = {
        "extends": "base",
        "faithful": faithful,
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": direction.source, "target": direction.target}],
        "prompts": FRAGMENT_SET,
        "context": list(PACK_OF.values()),
        "toolchain": dict(TOOLCHAINS),
        "stages": list(RUN_STAGES),
        "executor": {"kind": "scripted"},
        "trials": {"n": 1},
    }
    data.update(copy.deepcopy(changes))
    return data


def all_fixes_off(**overrides: bool) -> dict[str, bool]:
    """Return every named fix turned off, the execution gate included, except those `overrides` set."""
    require_fixes(GATE_FIX)
    return {**{name: False for name in FIXES}, **overrides}


def write_recipe(directory: Path, name: str, data: Mapping[str, Any]) -> Path:
    """Write `data` as the recipe `<directory>/<name>.yaml` and return its path."""
    path = directory / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


def fenced(code: str) -> str:
    """Return a SYNTHETIC reply holding `code` in one untagged fenced block (read as "\\n" + code)."""
    return f"Synthetic reply.\n```\n{code}```\nEnd of reply.\n"


def blocks(code: str, direction: Direction = CUDA_TO_OMP) -> str:
    """Return a SYNTHETIC reply holding `code` as the target's one FILE block (the fence_tag fix's reply form)."""
    return render_file_blocks({target_file(direction): code})


def clean_run(stdout: str = LAYOUT_OTHER_TIMES, **flags: bool) -> RunResult:
    """Return a SYNTHETIC run that exited 0, with `flags` set on the RunResult."""
    return RunResult(exit_code=0, hang=False, stdout=stdout, stderr="", wall_s=ATTEMPT_WALL_S, **flags)


def failed_run(stdout: str = LAYOUT_FAIL, exit_code: int = RUN_ERROR_EXIT, stderr: str = RUN_STDERR) -> RunResult:
    """Return a SYNTHETIC run that exited nonzero with `stderr` (RUN_STDERR by default)."""
    return RunResult(exit_code=exit_code, hang=False, stdout=stdout, stderr=stderr, wall_s=ATTEMPT_WALL_S)


def attempt_stdout(index: int) -> str:
    """Return a distinct SYNTHETIC stdout for the run of attempt `index`, which prints FAIL."""
    return f"SYNTHETIC output of attempt {index}\nFAIL\n"


@dataclass
class Outcome:
    """One finished run: its directory, its one trial read back from the run tree, its id, and the log."""

    run_dir: Path
    trial_id: str
    trial: Trial
    log: Log

    @property
    def store(self) -> TextStore:
        """Return the run's text store."""
        return TextStore(self.run_dir)

    def messages(self, index: int) -> list[tuple[str, str]]:
        """Return model request `index` as (role, content) pairs."""
        return [(message.role, message.content) for message in self.log.requests[index][0]]

    def prompt(self, index: int) -> str:
        """Return the stored prompt of attempt `index`."""
        ref = self.trial.attempts[index].prompt_ref
        assert ref is not None, f"attempt {index} has no prompt"
        return self.store.get(ref)

    def stdout(self, index: int) -> str:
        """Return the stored run stdout of attempt `index`."""
        ref = self.trial.attempts[index].run.stdout_ref
        assert ref is not None, f"attempt {index} kept no run stdout"
        return self.store.get(ref)

    def attempt_runs(self) -> list[Run]:
        """Return the runs of attempts, in order."""
        return [run for run in self.log.runs if run.kind == "attempt"]

    def stages(self) -> list[str]:
        """Return every attempt's stage reached, in order."""
        return [attempt.stage_reached for attempt in self.trial.attempts]


def run_one(
    tmp_path: Path, name: str, data: Mapping[str, Any], log: Log, *, unloading: bool = False
) -> Outcome:
    """Run a one-trial recipe with the fakes following `log`; return the outcome."""
    registry = make_registry(log, data["stages"], unloading=unloading)
    bench = write_bench(tmp_path / f"{name}-bench")
    options = RunOptions(runs_root=tmp_path / name / "runs-root", run_id="test-run", bench_root=bench,
                         registry=registry)
    run_dir = run_recipe(write_recipe(tmp_path, name, data), options)
    direction = data["directions"][0]
    trial_id = make_trial_id(name, MODEL_ID, SUITE, f"{direction['source']}-{direction['target']}", ITEM, 1)
    return Outcome(run_dir, trial_id, read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir)), log)


def context_replies(*codes: str) -> list[str]:
    """Return the two context replies followed by one faithful fenced reply per code."""
    return [SUMMARY_REPLY, DESCRIPTION_REPLY, *[fenced(code) for code in codes]]


# ---------------------------------------------------------------------------
# Expected prompts and diagnostics


def direction_system(fragments: Mapping[str, str], direction: Direction) -> str:
    """Return the direction's system prompt fragment."""
    return fragments[f"system_prompt_dict.{DIRECTION_KEY[(direction.source, direction.target)]}"]


def setup_text(fragments: Mapping[str, str], direction: Direction) -> str:
    """Return upstream's compiler text, one space, and its flag text for the direction's target."""
    return fragments[f"setup.{direction.target}.compiler"] + " " + fragments[f"setup.{direction.target}.flags"]


def compile_error_prompt(fragments: Mapping[str, str], direction: Direction, code: str) -> str:
    """Return upstream's correction prompt for a compile error of `code` (RAW_STDERR), every LF removed."""
    text = (
        code + fragments["correct.compile_error_head"] + setup_text(fragments, direction)
        + fragments["correct.compile_error_tail"] + RAW_STDERR + fragments["correct.outro"]
    )
    return text.replace("\n", "")


def run_report(fragments: Mapping[str, str], return_code: int, stderr: str) -> str:
    """Return upstream's report of a failed run (execute_code's return_result) for a Popen return code and stderr.

    The lead, the return code, one space, the segfault note for -11, then,
    when `stderr` (already read in text mode) is not empty, the stderr lead
    and the stderr.
    """
    text = fragments[EXIT_LEAD] + str(return_code) + " "
    if return_code == POPEN_SEGFAULT:
        text += fragments[SEGFAULT_NOTE]
    if stderr:
        text += fragments[STDERR_LEAD] + stderr
    return text


def execute_error_prompt(fragments: Mapping[str, str], direction: Direction, code: str, report: str) -> str:
    """Return upstream's execute-error prompt for `code` and the run report `report`, every LF removed."""
    text = (
        code + fragments["correct.run_error_head"] + setup_text(fragments, direction)
        + fragments["correct.run_error_tail"] + report + fragments["correct.outro"]
    )
    return text.replace("\n", "")


def assert_execute_error_prompt(
    outcome: Outcome,
    fragments: Mapping[str, str],
    direction: Direction,
    request: int,
    attempt: int,
    code: str,
    report: str | None = None,
) -> None:
    """Check that model request `request`, stored as attempt `attempt`'s prompt, is the execute-error prompt.

    `report` defaults to upstream's report of failed_run(): RUN_ERROR_EXIT
    and RUN_STDERR.
    """
    report = run_report(fragments, RUN_ERROR_EXIT, RUN_STDERR) if report is None else report
    (system_role, system), (user_role, sent) = outcome.messages(request)
    assert (system_role, system) == ("system", direction_system(fragments, direction))
    assert user_role == "user"
    assert sent == execute_error_prompt(fragments, direction, code, report), (
        "the previous code, the run-error head, upstream's compiler and flags, the tail, upstream's report of the "
        "run, and the outro, every line feed removed"
    )
    assert outcome.prompt(attempt) == sent, "the attempt keeps the prompt it was asked with"


def warnings_with_code(attempt: Attempt, code: str) -> list[Diagnostic]:
    """Return the attempt's diagnostics with `code`."""
    return [diagnostic for diagnostic in attempt.diagnostics if diagnostic.code == code]


def names_attempt(message: str, index: int) -> bool:
    """Return True when `message` names attempt `index`: the phrase "attempt <index>", the index a whole number."""
    return re.search(rf"\battempt {index}(?![0-9])", message, re.IGNORECASE) is not None


def end_code(trial: Trial) -> str | None:
    """Return final.end_reason's code, or None when the trial ended normally."""
    reason = trial.final.end_reason
    return None if reason is None else reason.code


# ---------------------------------------------------------------------------
# Names: the stage, the fix, and the end code


def test_run_loop_is_a_registered_stage_that_reproduces_the_execution_gate(tmp_path: Path) -> None:
    stage = registered_stage("run_loop")
    require_fixes(GATE_FIX)
    assert isinstance(FIXES[GATE_FIX], str) and FIXES[GATE_FIX].strip(), "the fix needs a description"
    assert GATE_FIX in set(getattr(stage, "reproduces", ())), "run_loop names the fix it reproduces"
    registry = make_registry(Log(), RUN_STAGES)
    plain = load_recipe(write_recipe(tmp_path, "plain", fragment_recipe(faithful=False)), registry=registry)
    faithful = load_recipe(write_recipe(tmp_path, "faithful", fragment_recipe()), registry=registry)
    assert (plain.data["fixes"][GATE_FIX], faithful.data["fixes"][GATE_FIX]) == (True, False), (
        "fixes.execution_gate: on by default, off under faithful"
    )


def test_upstream_crash_is_a_fixed_end_code() -> None:
    assert UPSTREAM_CRASH in record_module.END_REASONS, f"END_REASONS is {record_module.END_REASONS}"
    reason = record_module.EndReason(code=UPSTREAM_CRASH, message="SYNTHETIC: no attempt ran")
    assert reason.code == UPSTREAM_CRASH


# ---------------------------------------------------------------------------
# Executor none: nothing runs


def test_with_executor_none_run_loop_records_nothing_and_the_trial_ends_at_its_first_compiling_attempt(
    tmp_path: Path, lassi_assets: RecipeAssets
) -> None:
    log = Log(replies=context_replies(BAD_CODE, GOOD_CODE, GOOD_CODE))
    outcome = run_one(tmp_path, "none-executor", fragment_recipe(executor={"kind": "none"}), log)
    assert outcome.stages() == ["S1", COMPILED]
    assert log.replies == [fenced(GOOD_CODE)], "no correction is asked after the first compiling attempt"
    assert log.runs == [], "a compile-only executor runs nothing"
    for attempt in outcome.trial.attempts:
        assert (attempt.run.exit_code, attempt.run.hang, attempt.run.stdout_ref) == (None, None, None)
        assert [item for item in attempt.diagnostics if item.stage == "run"] == []
    assert (outcome.trial.final.stage_reached, outcome.trial.final.corrections) == (COMPILED, 1)
    assert end_code(outcome.trial) is None


def test_with_executor_none_a_compiling_attempt_past_the_gate_ends_the_trial_normally(
    tmp_path: Path, lassi_assets: RecipeAssets
) -> None:
    log = Log(replies=context_replies(*[BAD_CODE] * 8, GOOD_CODE))
    outcome = run_one(tmp_path, "none-past-gate", fragment_recipe(executor={"kind": "none"}), log)
    assert outcome.stages() == ["S1"] * 8 + [COMPILED]
    assert end_code(outcome.trial) is None, "upstream reads no run output when it does not execute, so no crash"
    assert warnings_with_code(outcome.trial.attempts[-1], STALE_OUTPUT) == []


# ---------------------------------------------------------------------------
# Running an attempt, and the execute-error prompt


def test_a_compiling_attempt_runs_through_the_executor_and_its_run_is_recorded(
    tmp_path: Path, lassi_assets: RecipeAssets
) -> None:
    log = Log(replies=context_replies(GOOD_CODE), attempt_runs=[clean_run()])
    outcome = run_one(tmp_path, "one-run", fragment_recipe(), log)
    (run,) = outcome.attempt_runs()
    assert run.artifact.parent.resolve() == build_dir(outcome.run_dir, outcome.trial_id, 0).resolve(), (
        "the executor runs the artifact compile_loop built for the attempt"
    )
    assert run.inputs == LAYOUT_ARGS, "with the item's run arguments"
    (attempt,) = outcome.trial.attempts
    assert (attempt.run.exit_code, attempt.run.hang, attempt.run.wall_s) == (0, False, ATTEMPT_WALL_S)
    assert outcome.stdout(0) == LAYOUT_OTHER_TIMES, "the run's stdout, kept in the text store"
    assert attempt.stage_reached == RAN_CLEAN, "a clean run is S5 on the stage ladder"
    assert (outcome.trial.final.stage_reached, outcome.trial.final.corrections) == (RAN_CLEAN, 0)
    assert end_code(outcome.trial) is None


@pytest.mark.parametrize("direction", DIRECTIONS, ids=DIRECTION_IDS)
def test_a_run_error_gets_upstreams_execute_error_prompt(
    tmp_path: Path, lassi_assets: RecipeAssets, direction: Direction
) -> None:
    log = Log(replies=context_replies(GOOD_CODE, GOOD_CODE), attempt_runs=[failed_run(), clean_run()])
    outcome = run_one(tmp_path, "run-error", fragment_recipe(direction), log)
    assert_execute_error_prompt(outcome, lassi_assets.fragments, direction, 3, 1, "\n" + GOOD_CODE)
    first, second = outcome.trial.attempts
    assert (first.stage_reached, first.run.exit_code) == (COMPILED, RUN_ERROR_EXIT), "a run error stays S4"
    assert outcome.stdout(0) == LAYOUT_FAIL, "a failed run's stdout is kept too"
    assert (second.stage_reached, second.run.exit_code) == (RAN_CLEAN, 0)
    assert (outcome.trial.final.stage_reached, outcome.trial.final.corrections) == (RAN_CLEAN, 1)
    assert end_code(outcome.trial) is None


# (the executor's exit status in the shell's form, the return code upstream's Popen would give)
SIGNAL_STATUSES = [(139, POPEN_SEGFAULT), (134, -6), (2, 2)]


@pytest.mark.parametrize(("status", "return_code"), SIGNAL_STATUSES, ids=["sigsegv", "sigabrt", "exit-2"])
def test_upstreams_run_report_gives_the_popen_return_code_and_the_stderr_in_text_mode(
    tmp_path: Path, lassi_assets: RecipeAssets, status: int, return_code: int
) -> None:
    stderr = "SYNTHETIC-RUN-STDERR first\r\nsecond\rthird\n"
    log = Log(replies=context_replies(GOOD_CODE, GOOD_CODE), attempt_runs=[failed_run(exit_code=status, stderr=stderr),
                                                                           clean_run()])
    outcome = run_one(tmp_path, "run-report", fragment_recipe(), log)
    report = run_report(lassi_assets.fragments, return_code, "SYNTHETIC-RUN-STDERR first\nsecond\nthird\n")
    assert_execute_error_prompt(outcome, lassi_assets.fragments, CUDA_TO_OMP, 3, 1, "\n" + GOOD_CODE, report)
    assert outcome.trial.attempts[0].run.exit_code == status, "the record keeps the executor's own status"


def test_upstreams_run_report_has_no_stderr_lead_when_the_run_wrote_no_stderr(
    tmp_path: Path, lassi_assets: RecipeAssets
) -> None:
    log = Log(replies=context_replies(GOOD_CODE, GOOD_CODE), attempt_runs=[failed_run(stderr=""), clean_run()])
    outcome = run_one(tmp_path, "run-report-quiet", fragment_recipe(), log)
    report = run_report(lassi_assets.fragments, RUN_ERROR_EXIT, "")
    assert_execute_error_prompt(outcome, lassi_assets.fragments, CUDA_TO_OMP, 3, 1, "\n" + GOOD_CODE, report)
    assert lassi_assets.fragments[STDERR_LEAD].replace("\n", "") not in outcome.prompt(1)


def test_run_errors_and_compile_errors_share_the_correction_count(tmp_path: Path, lassi_assets: RecipeAssets) -> None:
    fragments = lassi_assets.fragments
    replies = context_replies(BAD_CODE, GOOD_CODE, BAD_CODE, GOOD_CODE)
    log = Log(replies=replies, attempt_runs=[failed_run(), clean_run()])
    outcome = run_one(tmp_path, "shared-count", fragment_recipe(), log)
    assert outcome.stages() == ["S1", COMPILED, "S1", RAN_CLEAN]
    assert outcome.prompt(1) == compile_error_prompt(fragments, CUDA_TO_OMP, "\n" + BAD_CODE)
    assert_execute_error_prompt(outcome, fragments, CUDA_TO_OMP, 4, 2, "\n" + GOOD_CODE)
    assert outcome.prompt(3) == compile_error_prompt(fragments, CUDA_TO_OMP, "\n" + BAD_CODE)
    assert outcome.trial.final.corrections == 3, "one count for compile and run corrections"
    assert log.replies == []


def test_the_template_prompt_set_runs_a_compiling_attempt_too(tmp_path: Path) -> None:
    # The fragment-set runs skip without third_party/LASSI; this one keeps run_loop covered end to end.
    data = fragment_recipe(prompts=TEMPLATE_SET, stages=["baseline", "generate", "compile_loop", "run_loop"])
    del data["context"]
    log = Log(replies=[fenced(GOOD_CODE)], attempt_runs=[clean_run()])
    outcome = run_one(tmp_path, "template-run", data, log)
    (attempt,) = outcome.trial.attempts
    assert (attempt.stage_reached, attempt.run.exit_code) == (RAN_CLEAN, 0)
    assert outcome.stdout(0) == LAYOUT_OTHER_TIMES


# ---------------------------------------------------------------------------
# The execution gate and stale output (fixes.execution_gate off)


def test_a_compiling_attempt_after_seven_corrections_still_runs(tmp_path: Path, lassi_assets: RecipeAssets) -> None:
    log = Log(replies=context_replies(*[BAD_CODE] * 7, GOOD_CODE), attempt_runs=[clean_run()])
    outcome = run_one(tmp_path, "gate-boundary", fragment_recipe(), log)
    assert outcome.stages() == ["S1"] * 7 + [RAN_CLEAN], "the gate lets a correction count of 7 run"
    assert len(outcome.attempt_runs()) == 1
    assert end_code(outcome.trial) is None


def test_a_compiling_attempt_after_eight_corrections_ends_unexecuted_with_the_earlier_stdout_as_stale_output(
    tmp_path: Path, lassi_assets: RecipeAssets
) -> None:
    runs = [failed_run(stdout=attempt_stdout(index)) for index in range(8)]
    log = Log(replies=context_replies(*[GOOD_CODE] * 9), attempt_runs=runs)
    outcome = run_one(tmp_path, "gate-stale", fragment_recipe(stages=ORACLE_STAGES, oracle=ORACLE), log)
    attempts = outcome.trial.attempts
    assert outcome.stages() == [COMPILED] * 9
    assert len(outcome.attempt_runs()) == 8, "attempts 0 to 7 ran; attempt 8 is past the gate"
    last = attempts[8]
    assert (last.run.exit_code, last.run.stdout_ref) == (None, None), "the last attempt ends the trial unexecuted"
    assert outcome.stdout(7) == attempt_stdout(7), "attempt 7's stdout stands as the trial's output"
    (stale,) = warnings_with_code(last, STALE_OUTPUT)
    assert (stale.stage, stale.severity) == ("run", "warning")
    assert names_attempt(stale.message, 7), f"the warning names attempt 7: {stale.message!r}"
    assert all(warnings_with_code(attempt, STALE_OUTPUT) == [] for attempt in attempts[:8])
    assert attempts[7].alignment == Alignment(per_input=[0.0], mean=0.0), "the stale stdout is aligned"
    assert last.alignment == Alignment(), "the unexecuted attempt has no alignment"
    assert (outcome.trial.final.stage_reached, outcome.trial.final.corrections) == (COMPILED, 8)
    assert end_code(outcome.trial) is None
    assert log.replies == []


def test_stale_output_names_an_earlier_run_before_a_series_of_compile_errors(
    tmp_path: Path, lassi_assets: RecipeAssets
) -> None:
    log = Log(replies=context_replies(GOOD_CODE, *[BAD_CODE] * 7, GOOD_CODE), attempt_runs=[failed_run()])
    outcome = run_one(tmp_path, "gate-stale-early", fragment_recipe(), log)
    assert outcome.stages() == [COMPILED] + ["S1"] * 7 + [COMPILED]
    assert len(outcome.attempt_runs()) == 1
    last = outcome.trial.attempts[8]
    assert last.run.stdout_ref is None
    (stale,) = warnings_with_code(last, STALE_OUTPUT)
    assert (stale.stage, stale.severity) == ("run", "warning")
    assert names_attempt(stale.message, 0), f"the warning names attempt 0: {stale.message!r}"
    assert outcome.stdout(0) == LAYOUT_FAIL
    assert end_code(outcome.trial) is None


def test_with_no_earlier_run_a_compiling_attempt_past_the_gate_ends_with_upstream_crash(
    tmp_path: Path, lassi_assets: RecipeAssets
) -> None:
    log = Log(replies=context_replies(*[BAD_CODE] * 8, GOOD_CODE))
    outcome = run_one(tmp_path, "gate-crash", fragment_recipe(stages=ORACLE_STAGES, oracle=ORACLE), log)
    assert outcome.stages() == ["S1"] * 8 + [COMPILED]
    assert outcome.attempt_runs() == [], "attempt 8 is past the gate and no attempt before it compiled"
    reason = outcome.trial.final.end_reason
    assert reason is not None and reason.code == UPSTREAM_CRASH, "the notebook raises here"
    assert isinstance(reason.message, str) and reason.message.strip()
    assert all(attempt.run.stdout_ref is None for attempt in outcome.trial.attempts), "no stdout"
    assert all(attempt.alignment == Alignment() for attempt in outcome.trial.attempts), "no alignment"
    assert outcome.trial.final.alignment is None
    assert all(warnings_with_code(attempt, STALE_OUTPUT) == [] for attempt in outcome.trial.attempts)
    assert outcome.trial.final.corrections == 8


# ---------------------------------------------------------------------------
# Fixes on: the recipe's cap applies and no stale output is used


def test_with_fixes_on_a_compiling_attempt_past_eight_corrections_runs(tmp_path: Path) -> None:
    # The template set keeps this case covered when the upstream checkout is absent.
    stages = ["baseline", "generate", "compile_loop", "run_loop"]
    data = fragment_recipe(faithful=False, prompts=TEMPLATE_SET, stages=stages)
    del data["context"]
    log = Log(replies=[*[blocks(BAD_CODE)] * 9, blocks(GOOD_CODE)], attempt_runs=[clean_run()])
    outcome = run_one(tmp_path, "fixes-on-no-gate", data, log)
    assert outcome.stages() == ["S1"] * 9 + [RAN_CLEAN], "with the fix on every compiling attempt runs"
    assert outcome.trial.final.corrections == 9, "within projects/base.yaml's cap of 10"
    assert all(warnings_with_code(attempt, STALE_OUTPUT) == [] for attempt in outcome.trial.attempts)
    assert end_code(outcome.trial) is None


def test_with_fixes_on_the_cap_counts_run_corrections(tmp_path: Path, lassi_assets: RecipeAssets) -> None:
    replies = [SUMMARY_REPLY, DESCRIPTION_REPLY, *[blocks(GOOD_CODE)] * 3]
    log = Log(replies=replies, attempt_runs=[failed_run() for _ in range(3)])
    outcome = run_one(tmp_path, "fixes-on-run-cap", fragment_recipe(faithful=False, loop={"max_corrections": 2}),
                      log)
    assert outcome.stages() == [COMPILED] * 3
    assert len(outcome.attempt_runs()) == 3
    assert end_code(outcome.trial) == "correction-cap", "a run error remains when the cap stops the loop"
    assert outcome.trial.final.corrections == 2


def test_with_fixes_on_a_cap_hit_after_a_run_uses_no_stale_output(tmp_path: Path, lassi_assets: RecipeAssets) -> None:
    replies = [SUMMARY_REPLY, DESCRIPTION_REPLY, blocks(GOOD_CODE), blocks(BAD_CODE), blocks(BAD_CODE)]
    log = Log(replies=replies, attempt_runs=[failed_run()])
    outcome = run_one(tmp_path, "fixes-on-no-stale", fragment_recipe(faithful=False, loop={"max_corrections": 2}),
                      log)
    assert outcome.stages() == [COMPILED, "S1", "S1"]
    assert end_code(outcome.trial) == "correction-cap"
    assert all(warnings_with_code(attempt, STALE_OUTPUT) == [] for attempt in outcome.trial.attempts)


# ---------------------------------------------------------------------------
# Every faithful behavior is keyed on a fix or the cap, never on the faithful flag


def gate_log() -> Log:
    """Return a log whose attempts 0 to 7 compile and fail their runs, and whose attempt 8 compiles."""
    runs = [failed_run(stdout=attempt_stdout(index)) for index in range(8)]
    return Log(replies=context_replies(*[GOOD_CODE] * 9), attempt_runs=runs)


def test_all_fixes_off_with_a_numeric_cap_behaves_as_faithful_through_the_gate(
    tmp_path: Path, lassi_assets: RecipeAssets
) -> None:
    faithful = run_one(tmp_path, "faithful-true", fragment_recipe(), gate_log())
    data = fragment_recipe(faithful=False, fixes=all_fixes_off(), loop={"max_corrections": 10})
    unfaithful = run_one(tmp_path, "fixes-off", data, gate_log())
    assert unfaithful.log.requests == faithful.log.requests, "the same messages and sampling, in order"
    assert unfaithful.trial.attempts == faithful.trial.attempts
    assert unfaithful.trial.reference_run == faithful.trial.reference_run
    assert replace(unfaithful.trial.final, wall_s=None) == replace(faithful.trial.final, wall_s=None)
    runs = [[(run.kind, run.inputs, run.limits) for run in outcome.log.runs] for outcome in (faithful, unfaithful)]
    assert runs[0] == runs[1]
    assert len(warnings_with_code(faithful.trial.attempts[8], STALE_OUTPUT)) == 1


def test_all_fixes_off_stops_at_the_numeric_cap_before_the_gate(tmp_path: Path, lassi_assets: RecipeAssets) -> None:
    faithful = run_one(tmp_path, "faithful-true", fragment_recipe(), gate_log())
    capped_log = gate_log()
    capped_log.replies = capped_log.replies[:6]
    data = fragment_recipe(faithful=False, fixes=all_fixes_off(), loop={"max_corrections": 3})
    capped = run_one(tmp_path, "fixes-off-capped", data, capped_log)
    assert capped.log.requests == faithful.log.requests[:6], "upstream's requests up to the cap"
    assert capped.trial.attempts == faithful.trial.attempts[:4]
    assert end_code(capped.trial) == "correction-cap"
    assert capped.trial.final.corrections == 3


# ---------------------------------------------------------------------------
# The attempt run limit


def limit_recipe(**changes: Any) -> dict[str, Any]:
    """Return the faithful recipe with sandbox wall_s baseline_x10 and mem_gb 3 (3072 MB)."""
    return fragment_recipe(sandbox={"network": False, "wall_s": "baseline_x10", "mem_gb": 3}, **changes)


def test_the_attempt_run_limit_is_ten_times_the_reference_wall_time(tmp_path: Path, lassi_assets: RecipeAssets) -> None:
    floor, cpus = wall_floor(), run_cpus()
    reference = RunResult(exit_code=0, hang=False, stdout=LAYOUT, stderr="", wall_s=2 * floor)
    log = Log(replies=context_replies(GOOD_CODE), reference=reference, attempt_runs=[clean_run()])
    outcome = run_one(tmp_path, "limit-x10", limit_recipe(), log)
    (run,) = outcome.attempt_runs()
    assert run.limits.wall_s == pytest.approx(20 * floor), "baseline_x10: ten times the reference run's wall time"
    assert (run.limits.memory_mb, run.limits.cpus) == (3072, cpus), "memory from sandbox.mem_gb, cpus RUN_CPUS"


def test_the_attempt_run_limit_has_a_floor_for_a_tiny_reference_wall_time(
    tmp_path: Path, lassi_assets: RecipeAssets
) -> None:
    floor = wall_floor()
    reference = RunResult(exit_code=0, hang=False, stdout=LAYOUT, stderr="", wall_s=floor / 1000)
    log = Log(replies=context_replies(GOOD_CODE), reference=reference, attempt_runs=[clean_run()])
    outcome = run_one(tmp_path, "limit-floor", limit_recipe(), log)
    (run,) = outcome.attempt_runs()
    assert run.limits.wall_s == pytest.approx(floor)


def test_the_attempt_run_limit_is_the_floor_when_no_reference_ran(tmp_path: Path, lassi_assets: RecipeAssets) -> None:
    floor = wall_floor()
    stages = ["summarize_context", "describe_source", "generate", "compile_loop", "run_loop"]
    data = limit_recipe(faithful=False, fixes=all_fixes_off(baseline_both=True), loop={"max_corrections": 10},
                        stages=stages)
    log = Log(replies=context_replies(GOOD_CODE), attempt_runs=[clean_run()])
    outcome = run_one(tmp_path, "limit-missing", data, log)
    assert outcome.trial.reference_run.wall_s is None, "no baseline stage, so no reference run"
    (run,) = outcome.attempt_runs()
    assert run.limits.wall_s == pytest.approx(floor)
    assert run.limits.memory_mb == 3072


# ---------------------------------------------------------------------------
# Run flags


@pytest.mark.parametrize("flag", [None, *RUN_FLAGS], ids=["no-flag", *RUN_FLAGS])
def test_run_flags_become_run_stage_warnings(tmp_path: Path, lassi_assets: RecipeAssets, flag: str | None) -> None:
    flags = {} if flag is None else {flag: True}
    log = Log(replies=context_replies(GOOD_CODE), attempt_runs=[clean_run(**flags)])
    outcome = run_one(tmp_path, "run-flags", fragment_recipe(), log)
    (attempt,) = outcome.trial.attempts
    for name, code in RUN_FLAGS.items():
        found = warnings_with_code(attempt, code)
        if name == flag:
            assert [(item.stage, item.severity) for item in found] == [("run", "warning")], f"{name} is recorded"
        else:
            assert found == [], f"{code} without {name}"


# ---------------------------------------------------------------------------
# Unloading the model before runs


def test_a_backend_declaring_unload_before_run_is_asked_at_trial_start_and_before_each_attempt_run(
    tmp_path: Path, lassi_assets: RecipeAssets
) -> None:
    log = Log(replies=context_replies(GOOD_CODE, GOOD_CODE), attempt_runs=[failed_run(), clean_run()])
    run_one(tmp_path, "unload", fragment_recipe(), log, unloading=True)
    assert log.events == [
        "unload",  # the notebook's setup unload, before the baseline
        "build",
        "run:reference",  # the notebook runs the reference with no unload of its own
        "complete",
        "complete",
        "complete",
        "build",
        "unload",
        "run:attempt",
        "complete",
        "build",
        "unload",
        "run:attempt",
    ]


def test_a_backend_without_unload_before_run_is_never_asked_to_unload(
    tmp_path: Path, lassi_assets: RecipeAssets
) -> None:
    log = Log(replies=context_replies(GOOD_CODE, GOOD_CODE), attempt_runs=[failed_run(), clean_run()])
    outcome = run_one(tmp_path, "no-unload", fragment_recipe(), log)
    assert "unload" not in log.events
    assert outcome.trial.final.stage_reached == RAN_CLEAN


def test_with_executor_none_the_backend_is_asked_to_unload_once_at_trial_start(
    tmp_path: Path, lassi_assets: RecipeAssets
) -> None:
    log = Log(replies=context_replies(GOOD_CODE))
    run_one(tmp_path, "unload-none", fragment_recipe(executor={"kind": "none"}), log, unloading=True)
    assert log.events == ["unload", "build", "complete", "complete", "complete", "build"]


# ---------------------------------------------------------------------------
# End to end: every fix off, a numeric cap, the oracle


@pytest.mark.parametrize(
    ("candidate", "expected"), [(LAYOUT_OTHER_TIMES, 1.0), (LAYOUT_FAIL, 0.0)], ids=["matches", "prints-fail"]
)
def test_end_to_end_all_fixes_off_with_a_cap_aligns_the_run_against_the_reference_run(
    tmp_path: Path, lassi_assets: RecipeAssets, candidate: str, expected: float
) -> None:
    data = fragment_recipe(
        faithful=False, fixes=all_fixes_off(), loop={"max_corrections": 10}, stages=ORACLE_STAGES, oracle=ORACLE
    )
    log = Log(replies=context_replies(BAD_CODE, GOOD_CODE), attempt_runs=[clean_run(stdout=candidate)])
    outcome = run_one(tmp_path, "end-to-end", data, log)
    resolved = yaml.safe_load((outcome.run_dir / "recipe.resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["faithful"] is False and resolved["fixes"] == all_fixes_off()
    assert resolved["loop"]["max_corrections"] == 10 and resolved["stages"] == ORACLE_STAGES
    reference = outcome.trial.reference_run
    assert reference.stdout_ref is not None and outcome.store.get(reference.stdout_ref) == LAYOUT
    first, second = outcome.trial.attempts
    assert (first.stage_reached, second.stage_reached) == ("S1", RAN_CLEAN)
    assert outcome.stdout(1) == candidate
    assert first.alignment == Alignment(), "an attempt that never ran is not aligned"
    assert second.alignment == Alignment(per_input=[expected], mean=expected), (
        "the run's stdout, aligned against Trial.reference_run's stdout"
    )
    assert end_code(outcome.trial) is None


# ---------------------------------------------------------------------------
# A hang, and the runner's refusals (template set, so these run without the upstream checkout)


def template_data(**changes: Any) -> dict[str, Any]:
    """Return a p0-smoke recipe with fixes on, [baseline, generate, compile_loop, run_loop], and `changes`."""
    data = fragment_recipe(
        faithful=False, prompts=TEMPLATE_SET, stages=["baseline", "generate", "compile_loop", "run_loop"], **changes
    )
    del data["context"]
    return data


def test_a_hung_run_is_a_run_error_fed_back_with_the_run_stderr(tmp_path: Path) -> None:
    hung = RunResult(exit_code=None, hang=True, stdout="", stderr=RUN_STDERR, wall_s=ATTEMPT_WALL_S)
    log = Log(replies=[blocks(GOOD_CODE), blocks(GOOD_CODE)], attempt_runs=[hung, clean_run()])
    outcome = run_one(tmp_path, "hung-run", template_data(), log)
    first, second = outcome.trial.attempts
    assert (first.stage_reached, first.run.hang, second.stage_reached) == (COMPILED, True, RAN_CLEAN)
    errors = [item for item in first.diagnostics if item.stage == "run" and item.severity == "error"]
    assert [item.code for item in errors] == ["run-error"] and "wall limit" in errors[0].message
    assert RUN_STDERR in outcome.prompt(1), "the correction prompt carries the run's stderr"


def assert_refused(tmp_path: Path, name: str, data: Mapping[str, Any], match: str) -> None:
    """Check that the runner refuses the recipe with RunError matching `match` before any run directory exists."""
    log = Log(replies=[blocks(GOOD_CODE)], attempt_runs=[clean_run()])
    registry = make_registry(log, data["stages"])
    runs_root = tmp_path / name / "runs-root"
    run_options = RunOptions(runs_root=runs_root, run_id="test-run", bench_root=write_bench(tmp_path / f"{name}-bench"),
                             registry=registry)
    with pytest.raises(runner_module.RunError, match=match):
        run_recipe(write_recipe(tmp_path, name, data), run_options)
    assert not runs_root.exists() and log.events == []


@pytest.mark.parametrize("wall", ["baseline_x5", 0], ids=["unknown-rule", "zero"])
def test_the_runner_refuses_a_run_wall_limit_it_cannot_apply(tmp_path: Path, wall: Any) -> None:
    data = template_data(sandbox={"network": False, "wall_s": wall, "mem_gb": 3})
    assert_refused(tmp_path, "bad-wall", data, r"sandbox\.wall_s")


def test_a_compile_only_run_never_reads_the_wall_limit(tmp_path: Path) -> None:
    data = template_data(sandbox={"network": False, "wall_s": "baseline_x5", "mem_gb": 3}, executor={"kind": "none"})
    log = Log(replies=[blocks(GOOD_CODE)])
    outcome = run_one(tmp_path, "none-bad-wall", data, log)
    assert outcome.stages() == [COMPILED]


def test_the_runner_refuses_a_backend_declaring_unload_before_run_without_unload(tmp_path: Path) -> None:
    class NoUnload:
        """Declares unload_before_run but has no unload()."""

        name = "scripted"
        capabilities = frozenset({"chat", UNLOAD_BEFORE_RUN})

        def __init__(self, model_id: str) -> None:
            """Keep the model id."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Fail the test: a refused run asks no model."""
            raise AssertionError("a refused run asked the model")

    data = template_data()
    log = Log()
    registry = Registry()
    registry.register("LLMBackend", "scripted", NoUnload)
    registry.register("Executor", "scripted", scripted_executor(log))
    for name in TOOLCHAINS.values():
        registry.register("Toolchain", name, fake_toolchain(name, log))
    for name in data["stages"]:
        registry.register("Stage", name, registered_stage(name))
    runs_root = tmp_path / "no-unload" / "runs-root"
    options = RunOptions(runs_root=runs_root, run_id="test-run", bench_root=write_bench(tmp_path / "no-unload-bench"),
                         registry=registry)
    with pytest.raises(runner_module.RunError, match="unload"):
        run_recipe(write_recipe(tmp_path, "no-unload", data), options)
    assert not runs_root.exists() and log.events == []


def unsandboxed_registry(log: Log, stages: Sequence[str]) -> Registry:
    """Return make_registry's registry with an executor that runs programs but does not declare `sandboxed`."""
    base = scripted_executor(log)

    class UnsandboxedExecutor(base):  # type: ignore[misc, valid-type]
        """The scripted executor without the `sandboxed` capability."""

        capabilities = frozenset({"runs_code"})

    registry = make_registry(log, stages)
    registry.register("Executor", "unsandboxed", UnsandboxedExecutor)
    return registry


def test_the_runner_refuses_an_unsandboxed_executor_for_a_stage_that_runs_attempts(tmp_path: Path) -> None:
    data = template_data(executor={"kind": "unsandboxed"})
    log = Log(replies=[blocks(GOOD_CODE)], attempt_runs=[clean_run()])
    runs_root = tmp_path / "unsandboxed" / "runs-root"
    options = RunOptions(runs_root=runs_root, run_id="test-run", bench_root=write_bench(tmp_path / "unsandboxed-bench"),
                         registry=unsandboxed_registry(log, data["stages"]))
    with pytest.raises(runner_module.RunError, match=r"run_loop.*sandboxed"):
        run_recipe(write_recipe(tmp_path, "unsandboxed", data), options)
    assert not runs_root.exists() and log.events == []


def test_an_unsandboxed_executor_may_still_run_the_baseline_without_run_loop(tmp_path: Path) -> None:
    # Baseline runs only bench references, never model output, so the sandbox rule of run_loop does not apply.
    data = {**template_data(executor={"kind": "unsandboxed"}), "stages": ["baseline", "generate", "compile_loop"]}
    log = Log(replies=[blocks(GOOD_CODE)])
    bench = write_bench(tmp_path / "unsandboxed-baseline-bench")
    options = RunOptions(runs_root=tmp_path / "unsandboxed-baseline" / "runs-root", run_id="test-run",
                         bench_root=bench, registry=unsandboxed_registry(log, data["stages"]))
    run_recipe(write_recipe(tmp_path, "unsandboxed-baseline", data), options)
    assert log.runs and {run.kind for run in log.runs} == {"reference"}, "only references ran"
