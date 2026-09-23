"""Tests for the stage runner and the `lassi run` compile-only path (P0.11).

The runner (lassi/core/runner.py) loads a recipe, builds its components, runs
the recipe's stages over every direction, item, and trial (generate, then the
compile loop with the correction bound from the recipe; lassi/core/stages.py),
and writes the run tree: the resolved recipe, the toolchain pins, provenance,
run.md, one trial.json and one trial.md per trial, and the Parquet mirror
(bible Component Interfaces, Stage contract rules; Result Record; Project
Recipes; Readability Standards, Run and Config rows; Design Principles 3 and 5;
Agent Rules 1, 7, 10, and 12). `lassi run <recipe>` (lassi/cli.py) is its
command line, and tests/fixtures/recipes/p0-smoke.yaml is the P0 gate recipe.
build_toolchain (P0.15), the public function that builds one registered
toolchain with its pinned compiler and clean environment, is checked against
the runner's own construction, which goes through it.

No test here runs a compiler, a sandbox, or the network. The end-to-end runs
use the real mock backend, stages, and none executor, with a fake toolchain
registered as "nvcc-sm80" in a test Registry; the fake writes a placeholder
artifact unless a source holds an `#error` line. The pinned-compiler tests
build the real presets against a temporary toolchains root holding empty
placeholder executables, with subprocess.Popen replaced so that no process
starts (git alone, which the runner may call for provenance, runs for real).
The bench sources are small synthetic files, not HeCBench sources. No value in
this module is a measurement.

Every trial carries a copy of the run manifest in Trial.provenance (P0.18):
commit, dirty, and device as provenance.json records them, sdk from its
driver, and date from its started_utc. The tests check that copy against
provenance.json in each trial's record, trial.json, trial.md, and Parquet row,
so the two can never disagree, and that the stages never change it.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import errno
import importlib.util
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from lassi import cli
from lassi import prompts as prompts_module
from lassi.bench import Direction, load_suite, sources_dir
from lassi.core import record as record_module
from lassi.core import registry as registry_module
from lassi.core import runner as runner_module
from lassi.core import stages
from lassi.core.files import parse_file_blocks, render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Message, Sampling
from lassi.core.parquet import read_run_parquet
from lassi.core.recipe import Recipe, RecipeError, load_recipe, resolved_yaml
from lassi.core.record import (
    BenchItem,
    Diagnostic,
    ModelInfo,
    RunInfo,
    ToolchainPins,
    Trial,
    make_trial_id,
    to_json,
    unified_diff,
)
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.stages import CompileLoopStage, GenerateStage, RunContext, diagnostic_line
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.core.trial_md import PLACEHOLDER, fmt_provenance
from lassi.executors import NoneExecutor, SandboxUnavailableError
from lassi.executors.workdir import build_dir
from lassi.llm import MockBackend, model_info
from lassi.prompts import render
from lassi.toolchains import CommandResult, EnvRunner, NvccSm80, NvcppCc80
from lassi.toolchains.pins import read_pin

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "tests" / "fixtures" / "recipes" / "p0-smoke.yaml"
SUITE_MANIFEST = REPO / "assets" / "bench" / "lassi-hecbench-10.yaml"
PYPROJECT = REPO / "pyproject.toml"

SUITE = "lassi-hecbench-10"
ITEM = "layout"
MOCK_ID = "mock-reference"
SMOKE_TRIAL = "p0-smoke/mock-reference/lassi-hecbench-10/omp-cuda/layout/run01"
LOOP_TRIAL = "loop-test/scripted-fixture/lassi-hecbench-10/omp-cuda/layout/run01"
NVHPC_TRIAL = "omp-target/mock-reference/lassi-hecbench-10/cuda-omp/layout/run01"
# The stage an attempt that built an artifact reaches: S4, Compiles, on the bible's stage ladder (Training Module,
# Reward Function). Source-level translation has no verify or lower step, so no attempt here records S2 or S3.
COMPILED = "S4"
# The sampling a p0-smoke trial records: temperature and top_p from projects/base.yaml, max_tokens from the fixture.
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)

# The pinned executables under a toolchains root as toolchains/cuda.pin and toolchains/nvhpc.pin name them
# (PREFIX_NAME, then bin/nvcc or COMPILER_SUBDIR/nvc++; PHASE-NOTES, P0.7), and the CUDA prefix nvc++ needs.
NVCC_BIN = "cuda@12.6.3/bin/nvcc"
NVCPP_BIN = "nvhpc@24.11/Linux_x86_64/24.11/compilers/bin/nvc++"
CUDA_PREFIX = "cuda@12.6.3"

# Synthetic bench sources standing in for the layout item; the mock backend replies with the CUDA one.
FAKE_OMP_SOURCE = '#include <cstdio>\nint main() {\n#pragma omp target\n  { }\n  std::printf("done\\n");\n}\n'
FAKE_CUDA_SOURCE = "#include <cstdio>\n__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n"

# Scripted model replies: a file the fake toolchain refuses, one it accepts, and a reply with no FILE block.
BAD_SOURCE = "#error not translated yet\nint main() { return 0; }\n"
GOOD_SOURCE = "int main() { return 0; }\n"
BAD_REPLY = render_file_blocks({"main.cu": BAD_SOURCE})
GOOD_REPLY = render_file_blocks({"main.cu": GOOD_SOURCE})
NO_FILE_REPLY = "I cannot translate this program without more context.\n"
# The fake toolchain's diagnostics for BAD_SOURCE as correct.txt's $diagnostics shows them, one line each:
# `<severity> <file>:<line>:<column> [<code>] <message>`, with absent parts omitted (P0.11 contract, Prompts).
BAD_DIAGNOSTIC_LINES = (
    "error main.cu:1:1 [fake-error] the source has an #error line",
    "warning the fake toolchain built nothing",
)

# The keys and values the P0.11 contract lists for tests/fixtures/recipes/p0-smoke.yaml.
SMOKE_DATA: dict[str, Any] = {
    "extends": "base",
    "model": {"backend": "mock", "id": MOCK_ID},
    "llm": {"sampling": {"max_tokens": 4096}},
    "bench": {"suite": SUITE, "split": "eval"},
    "directions": [{"source": "omp", "target": "cuda"}],
    "prompts": "p0-smoke",
    "toolchain": {"cuda": "nvcc-sm80"},
    "stages": ["generate", "compile_loop"],
    "executor": {"kind": "none"},
    "trials": {"n": 1},
}

NEW_MODULES = ("lassi.core.runner", "lassi.core.stages", "lassi.cli", "lassi.prompts", "lassi.toolchains.pins")
PROJECT_NAMES = ("lassi-repro", "lassi-ee", "lassi-df", "hecbench", "qwen", "wizardcoder", "a100", "mi300x", "gpt-oss")
ALLOWED_THIRD_PARTY = frozenset({"yaml", "pyarrow"})
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
RUN_MD_COLUMNS = ["trial", "direction", "item", "stage reached", "corrections", "wall_s", "trial.md"]


# ---------------------------------------------------------------------------
# Environment, bench sources, and files


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate variables and the compile variables a test could inherit; each test sets what it needs.

    TMPDIR, which a pinned compile requires, points at a test directory, as
    the gate points it at the scratch disk.
    """
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CPATH", raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))


def write_bench(root: Path) -> Path:
    """Write the synthetic layout sources under `root` as the manifest lays them out, and return `root`."""
    for relative, text in (("src/layout-omp/main.cpp", FAKE_OMP_SOURCE), ("src/layout-cuda/main.cu", FAKE_CUDA_SOURCE)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    """Return a bench root holding the synthetic layout sources in both languages."""
    return write_bench(tmp_path / "bench")


def read_ascii(path: Path) -> str:
    """Return a file's text after checking that it is plain ASCII with LF newlines only."""
    data = path.read_bytes()
    assert data.isascii(), f"{path} is not ASCII"
    assert b"\r" not in data, f"{path} has a CR"
    return data.decode("ascii")


def smoke_data(**changes: Any) -> dict[str, Any]:
    """Return a copy of SMOKE_DATA with top-level keys replaced by `changes`; a None value drops the key."""
    data = copy.deepcopy(SMOKE_DATA)
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    return data


def nvhpc_data() -> dict[str, Any]:
    """Return a smoke recipe for the CUDA to OpenMP direction, built by the nvc++ preset alone."""
    return smoke_data(directions=[{"source": "cuda", "target": "omp"}], toolchain={"omp": "nvcpp-cc80"})


def write_recipe(directory: Path, name: str, data: Mapping[str, Any]) -> Path:
    """Write `data` as the recipe `<directory>/<name>.yaml` and return its path."""
    path = directory / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


def generate_fields() -> dict[str, str]:
    """Return the generate.txt fields for the omp to cuda direction of the synthetic layout item."""
    return {
        "source_language": "omp",
        "target_language": "cuda",
        "source_files": render_file_blocks({"main.cpp": FAKE_OMP_SOURCE}),
        "target_files": "main.cu",
    }


def normalized(text: str | None) -> str:
    """Return text with every whitespace run collapsed to one space."""
    return " ".join((text or "").split())


def table_cells(line: str) -> list[str]:
    """Return the stripped cells of one Markdown table row."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


# ---------------------------------------------------------------------------
# Fake components


@dataclass
class BuildLog:
    """What the fake toolchains were asked to build, in order: registered name, workdir, and files."""

    builds: list[tuple[str, Path, dict[str, str]]] = field(default_factory=list)

    def workdirs(self) -> list[Path]:
        """Return the workdir of every build, resolved, in order."""
        return [workdir.resolve() for _, workdir, _ in self.builds]


def fake_diagnostics(path: str, text: str) -> list[Diagnostic]:
    """Return the fake toolchain's diagnostics for one file: an error at its first #error line and a warning."""
    for number, line in enumerate(text.split("\n"), start=1):
        if line.startswith("#error"):
            error = Diagnostic(
                stage="compile",
                severity="error",
                code="fake-error",
                file=path,
                line=number,
                column=1,
                message="the source has an #error line",
            )
            return [error, Diagnostic(stage="compile", severity="warning", message="the fake toolchain built nothing")]
    return []


def fake_toolchain(registered_as: str, log: BuildLog) -> type:
    """Return a Toolchain class without PIN, to register as `registered_as`; it records builds in `log`.

    A build whose files hold no `#error` line writes a placeholder artifact
    file, `main`, in the workdir; otherwise it returns the fake diagnostics and
    no artifact. It runs no command.
    """

    class FakeToolchain:
        """A Toolchain without PIN, built as factory(); it writes the files and compiles nothing."""

        name = registered_as
        capabilities = frozenset({"diagnostics"})

        def build(self, files: Mapping[str, str], workdir: Path) -> BuildResult:
            """Record the build, write the files under the fresh `workdir`, and return the artifact or diagnostics."""
            workdir = Path(workdir)
            assert workdir.is_dir() and not any(workdir.iterdir()), f"{workdir} is not a fresh build directory"
            log.builds.append((registered_as, workdir, dict(files)))
            for path, text in files.items():
                target = workdir / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            diagnostics = [item for path, text in sorted(files.items()) for item in fake_diagnostics(path, text)]
            if diagnostics:
                return BuildResult(artifact=None, diagnostics=diagnostics)
            artifact = workdir / "main"
            artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=[])

    return FakeToolchain


class PinnedFake:
    """A Toolchain class with PIN and PIN_BIN, as the compiler presets declare them; it must never build here."""

    name = "nvcc-sm80"
    capabilities = frozenset({"diagnostics"})
    PIN = "cuda"
    PIN_BIN = "bin/nvcc"

    def __init__(self, *, executable: str, runner: Callable[..., CommandResult]) -> None:
        """Keep the pinned executable and the command runner."""
        self.executable = executable
        self.runner = runner

    def build(self, files: Mapping[str, str], workdir: Path) -> BuildResult:
        """Fail the test: a toolchain whose pinned executable is missing must never be asked to build."""
        raise AssertionError("a toolchain without its pinned executable was asked to build")


@dataclass
class Script:
    """The replies a scripted backend gives, in order, and every request it received."""

    replies: list[str]
    requests: list[tuple[list[Message], Sampling]] = field(default_factory=list)


def scripted_backend(script: Script) -> type:
    """Return an LLMBackend class, registered by the tests as "scripted", that answers from `script` in order."""

    class ScriptedBackend:
        """Answers each request with the next reply of the script; a request past its end fails the test."""

        name = "scripted"
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id, as every backend does."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Record the request and return the next scripted reply."""
            script.requests.append((list(messages), sampling))
            assert script.replies, "the backend was asked for more replies than the script holds"
            return Completion(text=script.replies.pop(0), prompt_tokens=0, completion_tokens=0)

    return ScriptedBackend


def recording_stage(stage_class: type, label: str, events: list[tuple[str, Any]]) -> type:
    """Return a Stage class that wraps `stage_class` and records each construction and call in `events`.

    A construction appends (label, context) and a call appends (label, trial_id).
    """

    class RecordingStage:
        """Builds the real stage on the same context and records what the runner does with it."""

        capabilities = stage_class.capabilities
        requires = stage_class.requires
        prompt_fields = stage_class.prompt_fields

        def __init__(self, *, context: RunContext) -> None:
            """Record the context and build the real stage on it."""
            events.append((label, context))
            self._stage = stage_class(context=context)

        def __call__(self, trial: Trial) -> Trial:
            """Record the trial id and run the real stage."""
            events.append((label, trial.trial_id))
            return self._stage(trial)

        def describe(self) -> str:
            """Return the real stage's description."""
            return self._stage.describe()

    return RecordingStage


def make_registry(
    log: BuildLog,
    *,
    backend: type | None = None,
    toolchain: type | None = None,
    stage_classes: Mapping[str, type] | None = None,
) -> Registry:
    """Return a test Registry: the real mock backend, stages, and none executor, and fake toolchains.

    "nvcc-sm80" is `toolchain` when given, else a fake without PIN; "nvcpp-cc80"
    is always a fake without PIN. `backend`, when given, is registered under its
    name, and `stage_classes` replace the real stages by name.
    """
    registry = Registry()
    registry.register("LLMBackend", "mock", MockBackend)
    if backend is not None:
        registry.register("LLMBackend", backend.name, backend)
    chosen = {"generate": GenerateStage, "compile_loop": CompileLoopStage, **(stage_classes or {})}
    for name, stage_class in chosen.items():
        registry.register("Stage", name, stage_class)
    registry.register("Executor", "none", NoneExecutor)
    registry.register("Toolchain", "nvcc-sm80", toolchain or fake_toolchain("nvcc-sm80", log))
    registry.register("Toolchain", "nvcpp-cc80", fake_toolchain("nvcpp-cc80", log))
    return registry


def run(recipe: Path, runs_root: Path, bench_root: Path, registry: Registry, run_id: str = "test-run") -> Path:
    """Run `recipe` with the given runs root, bench root, registry, and run id; return the run directory."""
    options = RunOptions(runs_root=runs_root, run_id=run_id, bench_root=bench_root, registry=registry)
    return run_recipe(recipe, options)


def load_trial(run_dir: Path, trial_id: str) -> Trial:
    """Read one trial back from the run tree through the run's text store."""
    return read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir))


def attempt_dir(run_dir: Path, trial_id: str, attempt: int) -> Path:
    """Return where an attempt builds, resolved: its workdir.build_dir under the run directory.

    The runner passes the run directory as the stages' build root, so a
    second run of a recipe (same trial ids) never meets the first one's builds.
    """
    return build_dir(run_dir, trial_id, attempt).resolve()


# A commit id the faked git reports; not a commit of this repository.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def patch_git(monkeypatch: pytest.MonkeyPatch, head: str | None, porcelain: str | None) -> None:
    """Make the runner's git report `head` for rev-parse and `porcelain` for status; None means git failed."""
    answers = {"rev-parse": head, "status": porcelain}
    monkeypatch.setattr(runner_module, "_git", lambda *args: answers[args[0]])


class NoGit:
    """Stands in for the runner's EnvRunner when git is not installed: every command fails to start."""

    def __init__(self, env: Mapping[str, str]) -> None:
        """Ignore the environment."""

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Raise what starting a missing executable raises."""
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", argv[0])


def summary_rows(run_md: str) -> dict[str, str]:
    """Return run.md's summary table (the rows above the first ## heading) as field -> value."""
    head = run_md.split("\n## ", 1)[0]
    rows = [table_cells(line) for line in head.split("\n") if line.startswith("|")]
    return {field: value for field, value in rows[2:]}


class StepToolchain:
    """A Toolchain without PIN that answers each build from a list of steps, then builds an artifact.

    A step is an exception to raise or a BuildResult with no artifact to
    return; `builds` counts the calls. It runs no command.
    """

    name = "nvcc-sm80"
    capabilities = frozenset({"diagnostics"})

    def __init__(self, steps: Sequence[BaseException | BuildResult]) -> None:
        """Keep the steps to play, in order."""
        self.steps = list(steps)
        self.builds = 0

    def build(self, files: Mapping[str, str], workdir: Path) -> BuildResult:
        """Play the next step, or write a placeholder artifact once the steps are used up."""
        self.builds += 1
        if self.steps:
            step = self.steps.pop(0)
            if isinstance(step, BaseException):
                raise step
            return step
        artifact = Path(workdir) / "main"
        artifact.write_bytes(b"PLACEHOLDER artifact of the step toolchain\n")
        return BuildResult(artifact=artifact, diagnostics=[])


def step_toolchain_class(steps: Sequence[BaseException | BuildResult]) -> type:
    """Return a StepToolchain subclass the runner builds as factory(), playing `steps`."""

    class Steps(StepToolchain):
        """A StepToolchain with fixed steps."""

        def __init__(self) -> None:
            """Play the fixed steps."""
            super().__init__(steps)

    return Steps


def probe_stage(action: Callable[[RunContext], None]) -> type:
    """Return a Stage class with no capabilities that calls `action(context)` and returns the trial unchanged."""

    class Probe:
        """Runs a test action on the trial's context."""

        capabilities: frozenset[str] = frozenset()
        requires: dict[str, set[str]] = {}

        def __init__(self, *, context: RunContext) -> None:
            """Keep the context."""
            self.context = context

        def __call__(self, trial: Trial) -> Trial:
            """Run the action and return the trial as it came."""
            action(self.context)
            return trial

        def describe(self) -> str:
            """Return a one-line description."""
            return "probe: a test action"

    return Probe


# ---------------------------------------------------------------------------
# A fake subprocess.Popen


@dataclass(frozen=True)
class Spawned:
    """One command the fake subprocess.Popen was asked to start: argv, working directory, and environment."""

    argv: list[str]
    cwd: Path | None
    env: dict[str, str] | None


class FinishedProcess:
    """What the fake subprocess.Popen returns: a process that has already exited; nothing was started."""

    # A pid no process has, so a kill (only after a timeout, which never happens here) finds nothing.
    pid = 2**31 - 1

    def __init__(self, argv: list[str], returncode: int, output: tuple[bytes, bytes], text: bool) -> None:
        """Keep the exit status and the output, as text when the caller asked for text."""
        self.args = argv
        self.returncode = returncode
        self.stdin = self.stdout = self.stderr = None
        decoded = tuple(data.decode("utf-8", errors="replace") for data in output)
        self._output: tuple[Any, Any] = decoded if text else output

    def communicate(self, input: Any = None, timeout: float | None = None) -> tuple[Any, Any]:
        """Return the stdout and stderr the fake was given."""
        return self._output

    def poll(self) -> int:
        """Return the exit status."""
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        """Return the exit status."""
        return self.returncode

    def kill(self) -> None:
        """Do nothing: no process was started."""

    def terminate(self) -> None:
        """Do nothing: no process was started."""

    def send_signal(self, signal: int) -> None:
        """Do nothing: no process was started."""

    def __enter__(self) -> FinishedProcess:
        """Return self, as Popen does."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Do nothing on exit."""


class Processes:
    """Stands in for subprocess.Popen: records every command and starts none, except git, which runs for real.

    A command whose executable lies under a watched directory is a compile:
    when it has a working directory, the fake writes a placeholder `main`
    there, as a compiler that succeeded would. `returncode`, `stdout`, and
    `stderr` set what every faked command returns.
    """

    def __init__(self) -> None:
        """Start with no calls and no watched directory; faked commands exit 0 with no output."""
        self.calls: list[Spawned] = []
        self.watched: list[Path] = []
        self.returncode = 0
        self.stdout = b""
        self.stderr = b""

    def watch(self, directory: Path) -> None:
        """Count commands whose executable lies under `directory` as compiles."""
        self.watched.append(directory)

    def compiles(self) -> list[Spawned]:
        """Return the calls whose executable lies under a watched directory, in order."""
        return [call for call in self.calls if self._is_watched(call.argv[0])]

    def _is_watched(self, executable: str) -> bool:
        """Return True when `executable` lies under a watched directory."""
        path = os.path.normcase(os.path.abspath(executable))
        roots = [os.path.normcase(os.path.abspath(directory)) + os.sep for directory in self.watched]
        return any(path.startswith(root) for root in roots)

    def spawn(self, argv: list[str], kwargs: Mapping[str, Any]) -> FinishedProcess:
        """Record one command and return a finished process; a watched command gets a placeholder `main`."""
        cwd = None if kwargs.get("cwd") is None else Path(kwargs["cwd"])
        env = None if kwargs.get("env") is None else dict(kwargs["env"])
        self.calls.append(Spawned(argv=argv, cwd=cwd, env=env))
        if cwd is not None and self._is_watched(argv[0]):
            (cwd / "main").write_bytes(b"PLACEHOLDER artifact of a faked compile\n")
        text = any(kwargs.get(key) for key in ("text", "universal_newlines", "encoding", "errors"))
        return FinishedProcess(argv, self.returncode, (self.stdout, self.stderr), text)


@pytest.fixture
def processes(monkeypatch: pytest.MonkeyPatch) -> Processes:
    """Replace subprocess.Popen for one test: git runs for real, every other command is recorded and faked."""
    fake = Processes()
    real_popen = subprocess.Popen

    def popen(args: Any, *more: Any, **kwargs: Any) -> Any:
        items = [args] if isinstance(args, (str, bytes, os.PathLike)) else list(args)
        argv = [os.fsdecode(item) for item in items]
        if Path(argv[0]).stem.lower() == "git":
            return real_popen(args, *more, **kwargs)
        return fake.spawn(argv, kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen)
    return fake


@pytest.fixture
def parent_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set os.environ as a build host might have it and return the environment a compile may get.

    HOME and TMPDIR point at test directories, the locale is not C, and
    variables that change a compile silently are set; the compile environment
    is the parent's PATH, LANG=C, LC_ALL=C, HOME, and TMPDIR, nothing else.
    """
    home, tmpdir = tmp_path / "home", tmp_path / "tmp"
    home.mkdir()
    tmpdir.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.setenv("LANG", "de_DE.UTF-8")
    monkeypatch.setenv("LC_ALL", "de_DE.UTF-8")
    monkeypatch.setenv("NVCC_PREPEND_FLAGS", "-DLEAKED_PREPEND")
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "-DLEAKED_APPEND")
    monkeypatch.setenv("CPATH", str(tmp_path / "leaked-include"))
    return {"PATH": os.environ["PATH"], "LANG": "C", "LC_ALL": "C", "HOME": str(home), "TMPDIR": str(tmpdir)}


def pinned_root(base: Path, *executables: str) -> Path:
    """Return a toolchains root with the pinned CUDA prefix and an empty placeholder file at each executable path."""
    root = base / "toolchains"
    (root / CUDA_PREFIX).mkdir(parents=True, exist_ok=True)
    for relative in executables:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return root


# ---------------------------------------------------------------------------
# The fixture recipe


def test_smoke_recipe_loads_with_the_default_registry_and_binds_the_p0_components() -> None:
    recipe = load_recipe(SMOKE)
    bound = [(binding.interface, binding.name) for binding in recipe.bindings]
    assert bound == [
        ("LLMBackend", "mock"),
        ("Toolchain", "nvcc-sm80"),
        ("Executor", "none"),
        ("Stage", "generate"),
        ("Stage", "compile_loop"),
    ]
    assert recipe.chain == ("base", "p0-smoke")
    assert recipe.data["model"] == {"backend": "mock", "id": MOCK_ID}
    assert recipe.data["llm"]["sampling"]["max_tokens"] == 4096


def test_smoke_recipe_holds_exactly_the_contract_values(tmp_path: Path) -> None:
    expected = load_recipe(write_recipe(tmp_path, "p0-smoke", SMOKE_DATA))
    actual = load_recipe(SMOKE)
    assert actual.canonical_yaml == expected.canonical_yaml
    assert actual.recipe_hash == expected.recipe_hash


def test_smoke_recipe_comments_every_value_in_plain_ascii() -> None:
    text = read_ascii(SMOKE)
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.endswith(":"):
            continue
        assert "#" in stripped, f"p0-smoke.yaml: no comment on {line!r}"


# ---------------------------------------------------------------------------
# End to end: the mock backend, a fake toolchain, and the none executor


@dataclass
class SmokeRun:
    """One run of p0-smoke with the test registry: where it wrote, what was built, and what it printed."""

    run_dir: Path
    runs_root: Path
    log: BuildLog
    registry: Registry
    output: str


@pytest.fixture
def smoke_run(tmp_path: Path, bench: Path, capsys: pytest.CaptureFixture[str]) -> SmokeRun:
    """Run p0-smoke once with the test registry, run id "smoke", and a runs root under tmp_path."""
    log = BuildLog()
    registry = make_registry(log)
    runs_root = tmp_path / "runs-root"
    run_dir = run(SMOKE, runs_root, bench, registry, run_id="smoke")
    return SmokeRun(run_dir=run_dir, runs_root=runs_root, log=log, registry=registry, output=capsys.readouterr().out)


def test_mock_run_writes_the_run_tree(smoke_run: SmokeRun) -> None:
    run_dir = smoke_run.run_dir
    assert run_dir.resolve() == (smoke_run.runs_root / "runs" / "smoke").resolve()
    for name in ("recipe.resolved.yaml", "toolchains.json", "provenance.json", "run.md"):
        assert (run_dir / name).is_file(), name
    assert (run_dir / "parquet").is_dir()
    trial_path = trial_dir(run_dir, SMOKE_TRIAL)
    assert (trial_path / "trial.json").is_file()
    assert (trial_path / "trial.md").is_file()
    toolchains = json.loads(read_ascii(run_dir / "toolchains.json"))
    assert toolchains == {"nvcc-sm80": {"languages": ["cuda"], "executable": None, "environment": None, "pins": {}}}


def test_mock_trial_compiles_with_no_corrections(smoke_run: SmokeRun) -> None:
    trial = load_trial(smoke_run.run_dir, SMOKE_TRIAL)
    assert trial.recipe_hash == load_recipe(SMOKE, registry=smoke_run.registry).recipe_hash
    assert trial.bench_item == BenchItem(suite=SUITE, item=ITEM, split="eval", direction="omp-cuda")
    assert trial.model == ModelInfo(backend="mock", id=MOCK_ID, sampling=SAMPLING)
    assert trial.toolchain_pins == ToolchainPins()
    (attempt,) = trial.attempts
    assert attempt.index == 0
    assert attempt.stage_reached == COMPILED
    assert attempt.response_text == render_file_blocks({"main.cu": FAKE_CUDA_SOURCE})
    assert attempt.files == {"main.cu": FAKE_CUDA_SOURCE}
    assert attempt.diagnostics == []
    assert attempt.diff_from_previous == ""
    assert attempt.run == RunInfo(), "nothing runs in a compile-only trial, so RunInfo stays unset"
    final = trial.final
    assert (final.stage_reached, final.corrections, final.alignment, final.score) == (COMPILED, 0, None, None)
    assert isinstance(final.wall_s, float) and final.wall_s >= 0.0


def test_mock_trial_builds_once_in_a_fresh_attempt_directory(smoke_run: SmokeRun) -> None:
    ((name, workdir, files),) = smoke_run.log.builds
    assert name == "nvcc-sm80"
    assert files == {"main.cu": FAKE_CUDA_SOURCE}
    assert workdir.resolve() == attempt_dir(smoke_run.run_dir, SMOKE_TRIAL, 0)


def test_mock_trial_prompt_is_the_rendered_generate_template(smoke_run: SmokeRun) -> None:
    trial = load_trial(smoke_run.run_dir, SMOKE_TRIAL)
    ref = trial.attempts[0].prompt_ref
    assert ref is not None
    prompt = TextStore(smoke_run.run_dir).get(ref)
    assert prompt == render("p0-smoke", "generate", generate_fields())
    assert FAKE_CUDA_SOURCE not in prompt, "the reference target must never reach the prompt"


def test_mock_run_parquet_reads_back_one_trial_with_null_run_fields(smoke_run: SmokeRun) -> None:
    tables = read_run_parquet(smoke_run.run_dir / "parquet")
    (row,) = tables["trials"]
    assert row["trial_id"] == SMOKE_TRIAL
    assert (row["final_stage_reached"], row["final_corrections"], row["attempt_count"]) == (COMPILED, 0, 1)
    (attempt_row,) = tables["attempts"]
    assert (attempt_row["run_exit_code"], attempt_row["run_hang"], attempt_row["run_wall_s"]) == (None, None, None)


def test_resolved_recipe_is_saved_and_reloads_to_the_same_hash(smoke_run: SmokeRun) -> None:
    recipe = load_recipe(SMOKE, registry=smoke_run.registry)
    saved = smoke_run.run_dir / "recipe.resolved.yaml"
    assert read_ascii(saved) == resolved_yaml(recipe)
    assert load_recipe(saved, registry=smoke_run.registry).recipe_hash == recipe.recipe_hash
    assert load_trial(smoke_run.run_dir, SMOKE_TRIAL).recipe_hash == recipe.recipe_hash


@pytest.mark.parametrize(("porcelain", "dirty"), [("", False), (" M lassi/core/runner.py\n", True)])
def test_provenance_records_the_commit_the_dirty_flag_the_host_and_the_times(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch, porcelain: str, dirty: bool
) -> None:
    patch_git(monkeypatch, FAKE_COMMIT + "\n", porcelain)
    registry = make_registry(BuildLog())
    before = datetime.now(timezone.utc).replace(microsecond=0)
    run_dir = run(SMOKE, tmp_path / "runs-root", bench, registry)
    after = datetime.now(timezone.utc)
    data = json.loads(read_ascii(run_dir / "provenance.json"))
    assert data["recipe_hash"] == load_recipe(SMOKE, registry=registry).recipe_hash
    assert (data["commit"], data["dirty"]) == (FAKE_COMMIT, dirty)
    assert (data["host"], data["python"]) == (platform.node(), platform.python_version())
    assert (data["status"], data["executor"], data["device"], data["driver"]) == (
        "complete",
        "none",
        "none (compile only)",
        None,
    )
    assert data["pins"] == {}, "the fake toolchain has no pin"
    started, finished = (datetime.fromisoformat(data[key]) for key in ("started_utc", "finished_utc"))
    assert started.utcoffset() == finished.utcoffset() == timedelta(0), "times are UTC"
    assert before <= started <= finished <= after


def test_provenance_and_run_md_show_no_commit_when_git_is_unavailable(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_module, "EnvRunner", NoGit)
    run_dir = run(SMOKE, tmp_path / "runs-root", bench, make_registry(BuildLog()))
    data = json.loads(read_ascii(run_dir / "provenance.json"))
    assert (data["commit"], data["dirty"]) == (None, None)
    summary = summary_rows(read_ascii(run_dir / "run.md"))
    # Provenance is not a measurement, so run.md shows an unknown value as "-", as trial.md does, never PLACEHOLDER.
    assert (summary["Commit"], summary["Dirty"], summary["Driver"]) == ("-", "-", "-")
    assert PLACEHOLDER not in summary.values()


def test_run_md_shows_the_run_the_resolved_recipe_the_pins_and_the_trials(smoke_run: SmokeRun) -> None:
    text = read_ascii(smoke_run.run_dir / "run.md")
    lines = text.split("\n")
    recipe = load_recipe(SMOKE, registry=smoke_run.registry)
    assert "# Run smoke" in lines
    headings = ["## Resolved recipe", "## Toolchain pins", "## Trials"]
    for heading in headings:
        assert heading in lines, heading
    positions = [lines.index(heading) for heading in headings]
    assert positions == sorted(positions)
    head = "\n".join(lines[: positions[0]])
    assert recipe.recipe_hash in head and "p0-smoke" in head
    resolved = "\n".join(lines[positions[0] : positions[1]])
    assert "```yaml" in resolved and recipe.canonical_yaml in resolved
    trials = lines[positions[2] :]
    header_index = next(index for index, line in enumerate(trials) if line.startswith("|"))
    assert [cell.lower() for cell in table_cells(trials[header_index])] == RUN_MD_COLUMNS
    above = [line for line in trials[:header_index] if line.strip()]
    assert above[-1] == f"### Arm {MOCK_ID}", "each arm's table is headed by its model id"


def test_run_md_summary_matches_provenance(smoke_run: SmokeRun) -> None:
    provenance = json.loads(read_ascii(smoke_run.run_dir / "provenance.json"))
    summary = summary_rows(read_ascii(smoke_run.run_dir / "run.md"))
    recipe_hash = load_recipe(SMOKE, registry=smoke_run.registry).recipe_hash
    assert summary == {
        "Recipe": "p0-smoke",
        "Recipe hash": f"`{recipe_hash}`",
        "Commit": fmt_provenance(provenance["commit"]),
        "Dirty": fmt_provenance(provenance["dirty"]),
        "Executor": "none",
        "Device": "none (compile only)",
        "Driver": "-",
        "Started (UTC)": provenance["started_utc"],
        "Finished (UTC)": provenance["finished_utc"],
        "Trials": "1",
    }


def test_run_md_links_every_trial_md_relatively(smoke_run: SmokeRun) -> None:
    text = read_ascii(smoke_run.run_dir / "run.md")
    link = f"{SMOKE_TRIAL}/trial.md"
    rows = [line for line in text.split("\n") if link in line]
    assert len(rows) == 1, rows
    assert {"omp-cuda", "layout", COMPILED, "0"} <= set(table_cells(rows[0]))
    assert (smoke_run.run_dir / link).is_file()
    assert str(smoke_run.run_dir) not in text and smoke_run.run_dir.as_posix() not in text


def test_run_files_are_ascii_with_lf(smoke_run: SmokeRun) -> None:
    trial_path = trial_dir(smoke_run.run_dir, SMOKE_TRIAL)
    names = ("recipe.resolved.yaml", "toolchains.json", "provenance.json", "run.md")
    for path in [*(smoke_run.run_dir / name for name in names), trial_path / "trial.json", trial_path / "trial.md"]:
        read_ascii(path)


def test_run_prints_one_line_per_trial_and_the_run_directory(smoke_run: SmokeRun) -> None:
    lines = smoke_run.output.splitlines()
    assert sum(SMOKE_TRIAL in line for line in lines) == 1, lines
    assert any(str(smoke_run.run_dir) in line for line in lines), lines


def test_trials_follow_direction_then_item_then_run_order(tmp_path: Path, bench: Path) -> None:
    log = BuildLog()
    data = smoke_data(
        directions=[{"source": "omp", "target": "cuda"}, {"source": "cuda", "target": "omp"}],
        toolchain={"cuda": "nvcc-sm80", "omp": "nvcpp-cc80"},
        trials={"n": 2},
    )
    run_dir = run(write_recipe(tmp_path, "order-test", data), tmp_path / "runs-root", bench, make_registry(log))
    expected = [
        make_trial_id("order-test", MOCK_ID, SUITE, direction, ITEM, number)
        for direction in ("omp-cuda", "cuda-omp")
        for number in (1, 2)
    ]
    assert ["/".join(workdir.parts[-8:-2]) for workdir in log.workdirs()] == expected
    assert [name for name, _, _ in log.builds] == ["nvcc-sm80", "nvcc-sm80", "nvcpp-cc80", "nvcpp-cc80"]
    for trial_id in expected:
        trial = load_trial(run_dir, trial_id)
        assert trial.final.stage_reached == COMPILED
    assert load_trial(run_dir, expected[2]).attempts[0].files == {"main.cpp": FAKE_OMP_SOURCE}
    assert sorted(row["trial_id"] for row in read_run_parquet(run_dir / "parquet")["trials"]) == sorted(expected)


@pytest.mark.parametrize(("loop", "expected"), [(None, 10), (3, 3), ("uncapped", None)])
def test_runner_builds_each_stage_per_trial_on_a_fresh_context(
    tmp_path: Path, bench: Path, loop: int | str | None, expected: int | None
) -> None:
    events: list[tuple[str, Any]] = []
    classes = {
        "generate": recording_stage(GenerateStage, "generate", events),
        "compile_loop": recording_stage(CompileLoopStage, "compile_loop", events),
    }
    data = smoke_data(trials={"n": 2})
    if loop is not None:
        data["loop"] = {"max_corrections": loop}
    runs_root = tmp_path / "runs-root"
    registry = make_registry(BuildLog(), stage_classes=classes)
    run_dir = run(write_recipe(tmp_path, "context-test", data), runs_root, bench, registry)
    first, second = (make_trial_id("context-test", MOCK_ID, SUITE, "omp-cuda", ITEM, n) for n in (1, 2))
    calls = [(label, value) for label, value in events if isinstance(value, str)]
    assert calls == [("generate", first), ("compile_loop", first), ("generate", second), ("compile_loop", second)]
    contexts = [value for _, value in events if not isinstance(value, str)]
    assert len(contexts) == 4, "one stage object per stage and trial"
    assert contexts[0] is not contexts[2], "each trial runs on a fresh RunContext"
    for context in contexts:
        assert context.max_corrections == expected
        assert (context.prompts, context.item, context.direction) == ("p0-smoke", ITEM, Direction("omp", "cuda"))
        assert context.sampling == SAMPLING and context.backend.model_id == MOCK_ID
        assert context.recipe.name == "context-test" and context.suite.name == SUITE
        assert Path(context.sources_root).resolve() == bench.resolve()
        assert set(context.toolchains) == {"cuda"} and context.toolchains["cuda"].name == "nvcc-sm80"
        assert isinstance(context.executor, NoneExecutor)
        assert isinstance(context.store, TextStore) and context.store.root.resolve() == run_dir.resolve()
        assert Path(context.build_root).resolve() == run_dir.resolve()


def test_final_wall_s_is_the_monotonic_time_the_stages_took(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only the runner's own time binding is scripted: subprocess and pytest keep the real clock.
    ticks = iter([100.0, 102.5])
    monkeypatch.setattr(runner_module, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    run_dir = run(SMOKE, tmp_path / "runs-root", bench, make_registry(BuildLog()))
    assert load_trial(run_dir, SMOKE_TRIAL).final.wall_s == 2.5
    row = next(line for line in read_ascii(run_dir / "run.md").split("\n") if f"{SMOKE_TRIAL}/trial.md" in line)
    assert table_cells(row)[RUN_MD_COLUMNS.index("wall_s")] == "2.5"


SYNTH_MANIFEST = """\
suite: synth-two
repo: https://example.invalid/synth-two
commit: 0123456789abcdef0123456789abcdef01234567
items:
  zeta:
    split: eval
    languages:
      omp: {dir: src/zeta-omp, files: [main.cpp]}
      cuda: {dir: src/zeta-cuda, files: [a.cu, b.cu]}
  alpha:
    split: eval
    languages:
      omp: {dir: src/alpha-omp, files: [main.cpp]}
      cuda: {dir: src/alpha-cuda, files: [a.cu, b.cu]}
"""


def test_items_run_in_sorted_order_and_the_prompt_lists_every_target_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A synthetic suite whose manifest lists its items out of order, each with two target files.
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "synth-two.yaml").write_bytes(SYNTH_MANIFEST.encode("ascii"))
    monkeypatch.setattr(runner_module, "BENCH_DIR", manifests)
    sources = tmp_path / "synth-bench"
    for item in ("alpha", "zeta"):
        for relative, text in ((f"{item}-omp/main.cpp", FAKE_OMP_SOURCE), (f"{item}-cuda/a.cu", GOOD_SOURCE)):
            (sources / "src" / relative).parent.mkdir(parents=True, exist_ok=True)
            (sources / "src" / relative).write_bytes(text.encode("ascii"))
        (sources / "src" / f"{item}-cuda" / "b.cu").write_bytes(b"int b;\n")
    log = BuildLog()
    data = smoke_data(bench={"suite": "synth-two", "split": "eval"})
    run_dir = run(write_recipe(tmp_path, "two-items", data), tmp_path / "runs-root", sources, make_registry(log))
    ids = [make_trial_id("two-items", MOCK_ID, "synth-two", "omp-cuda", item, 1) for item in ("alpha", "zeta")]
    assert log.workdirs() == [attempt_dir(run_dir, trial_id, 0) for trial_id in ids]
    assert [sorted(files) for _, _, files in log.builds] == [["a.cu", "b.cu"], ["a.cu", "b.cu"]]
    for trial_id in ids:
        trial = load_trial(run_dir, trial_id)
        ref = trial.attempts[0].prompt_ref
        assert ref is not None and "a.cu, b.cu" in TextStore(run_dir).get(ref)
        assert trial.final.stage_reached == COMPILED


def test_a_second_run_of_the_same_recipe_builds_in_its_own_run_directory(tmp_path: Path, bench: Path) -> None:
    runs_root = tmp_path / "runs-root"
    log = BuildLog()
    registry = make_registry(log)
    first = run(SMOKE, runs_root, bench, registry, run_id="first")
    second = run(SMOKE, runs_root, bench, registry, run_id="second")
    assert log.workdirs() == [attempt_dir(run_dir, SMOKE_TRIAL, 0) for run_dir in (first, second)]
    for run_dir in (first, second):
        assert load_trial(run_dir, SMOKE_TRIAL).final.stage_reached == COMPILED


def test_provenance_is_written_before_the_trials_with_status_running(tmp_path: Path, bench: Path) -> None:
    seen: list[dict[str, Any]] = []

    def peek(context: RunContext) -> None:
        seen.append(json.loads((Path(context.build_root) / "provenance.json").read_text(encoding="ascii")))

    registry = make_registry(BuildLog(), stage_classes={"peek": probe_stage(peek)})
    recipe = write_recipe(tmp_path, "peek", smoke_data(stages=["generate", "compile_loop", "peek"]))
    run_dir = run(recipe, tmp_path / "runs-root", bench, registry)
    (during,) = seen
    assert (during["status"], during["finished_utc"]) == ("running", None)
    assert "commit" in during and during["started_utc"]
    done = json.loads(read_ascii(run_dir / "provenance.json"))
    assert done["status"] == "complete" and done["finished_utc"] is not None
    closing = ("status", "finished_utc")
    assert {key: value for key, value in done.items() if key not in closing} == {
        key: value for key, value in during.items() if key not in closing
    }, "the final manifest changes only its status and finish time"


def test_a_trial_that_raises_leaves_provenance_with_status_failed(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_git(monkeypatch, FAKE_COMMIT, "")

    def fail(context: RunContext) -> None:
        raise RuntimeError("fixture: a stage failed")

    registry = make_registry(BuildLog(), stage_classes={"fail": probe_stage(fail)})
    recipe = write_recipe(tmp_path, "failing", smoke_data(stages=["generate", "compile_loop", "fail"]))
    with pytest.raises(RuntimeError, match="fixture: a stage failed"):
        run(recipe, tmp_path / "runs-root", bench, registry, run_id="failing")
    run_dir = tmp_path / "runs-root" / "runs" / "failing"
    data = json.loads(read_ascii(run_dir / "provenance.json"))
    assert (data["status"], data["commit"], data["dirty"]) == ("failed", FAKE_COMMIT, False)
    assert data["finished_utc"] is not None
    assert not (run_dir / "run.md").exists()


# ---------------------------------------------------------------------------
# Trial provenance: every trial's copy of the run manifest (P0.18)


def manifest_copy(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Trial provenance a manifest gives: commit, dirty, device, driver as sdk, started_utc as date."""
    return {
        "commit": manifest["commit"],
        "dirty": manifest["dirty"],
        "device": manifest["device"],
        "sdk": manifest["driver"],
        "date": manifest["started_utc"],
    }


def assert_trials_carry_the_manifest(run_dir: Path, trial_ids: Sequence[str]) -> dict[str, Any]:
    """Assert that each trial's record, trial.json, and Parquet row hold the manifest's copy; return the copy."""
    expected = manifest_copy(json.loads(read_ascii(run_dir / "provenance.json")))
    rows = {row["trial_id"]: row for row in read_run_parquet(run_dir / "parquet")["trials"]}
    assert sorted(rows) == sorted(trial_ids)
    for trial_id in trial_ids:
        assert load_trial(run_dir, trial_id).provenance == record_module.Provenance(**expected), trial_id
        raw = json.loads(read_ascii(trial_dir(run_dir, trial_id) / "trial.json"))
        assert raw["provenance"] == expected, trial_id
        assert {key: rows[trial_id][f"provenance_{key}"] for key in expected} == expected, trial_id
    return expected


def test_every_trial_carries_a_copy_of_the_run_manifest(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_git(monkeypatch, FAKE_COMMIT + "\n", " M lassi/core/runner.py\n")
    data = smoke_data(
        directions=[{"source": "omp", "target": "cuda"}, {"source": "cuda", "target": "omp"}],
        toolchain={"cuda": "nvcc-sm80", "omp": "nvcpp-cc80"},
        trials={"n": 2},
    )
    run_dir = run(write_recipe(tmp_path, "copy-test", data), tmp_path / "runs-root", bench, make_registry(BuildLog()))
    ids = [
        make_trial_id("copy-test", MOCK_ID, SUITE, direction, ITEM, number)
        for direction in ("omp-cuda", "cuda-omp")
        for number in (1, 2)
    ]
    expected = assert_trials_carry_the_manifest(run_dir, ids)
    date = expected.pop("date")
    assert expected == {"commit": FAKE_COMMIT, "dirty": True, "device": "none (compile only)", "sdk": None}
    assert datetime.fromisoformat(date).utcoffset() == timedelta(0), "the date is the run's start time in UTC"


def test_trial_provenance_is_null_where_the_manifest_is_when_git_is_unavailable(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner_module, "EnvRunner", NoGit)
    run_dir = run(SMOKE, tmp_path / "runs-root", bench, make_registry(BuildLog()))
    expected = assert_trials_carry_the_manifest(run_dir, [SMOKE_TRIAL])
    assert (expected["commit"], expected["dirty"]) == (None, None)


class DevicelessExecutor:
    """An Executor that may run programs but names no device, so the manifest's device is what the runner makes of it.

    Nothing runs on the compile-only path, so run() is never called.
    """

    capabilities = frozenset({"runs_code"})

    def run(self, artifact: Path, inputs: Sequence[str], limits: Any) -> Any:
        """Fail the test: nothing runs on the compile-only path."""
        raise AssertionError("the compile-only path ran an executor")


def test_trial_provenance_matches_the_manifest_for_an_executor_that_names_no_device(
    tmp_path: Path, bench: Path
) -> None:
    registry = make_registry(BuildLog())
    registry.register("Executor", "deviceless", DevicelessExecutor)
    recipe = write_recipe(tmp_path, "deviceless", smoke_data(executor={"kind": "deviceless"}))
    run_dir = run(recipe, tmp_path / "runs-root", bench, registry)
    assert_trials_carry_the_manifest(run_dir, ["deviceless/mock-reference/lassi-hecbench-10/omp-cuda/layout/run01"])


def test_the_final_manifest_keeps_the_provenance_every_trial_copied(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An executor could name its device partway through a run; the final manifest still holds the device the
    # trials copied, because it is the first manifest with only its status and finish time changed.
    names = iter(["fixture-device-first", "fixture-device-later", "fixture-device-last"])
    monkeypatch.setattr(runner_module, "_device", lambda executor: next(names))
    run_dir = run(SMOKE, tmp_path / "runs-root", bench, make_registry(BuildLog()))
    assert assert_trials_carry_the_manifest(run_dir, [SMOKE_TRIAL])["device"] == "fixture-device-first"
    assert summary_rows(read_ascii(run_dir / "run.md"))["Device"] == "fixture-device-first"


def test_trial_md_in_the_run_tree_shows_the_manifest_provenance(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_git(monkeypatch, FAKE_COMMIT + "\n", "")
    run_dir = run(SMOKE, tmp_path / "runs-root", bench, make_registry(BuildLog()))
    manifest = json.loads(read_ascii(run_dir / "provenance.json"))
    md = read_ascii(trial_dir(run_dir, SMOKE_TRIAL) / "trial.md")
    block = (
        "## Provenance\n\n"
        "| Field | Value |\n"
        "| --- | --- |\n"
        f"| commit | {FAKE_COMMIT} |\n"
        "| dirty | false |\n"
        "| device | none (compile only) |\n"
        "| sdk | - |\n"
        f"| date | {manifest['started_utc']} |\n"
    )
    assert block in md
    assert md.index(block) < md.index("## Attempt 0\n")


class RecordingExecutor:
    """An Executor with one config key, `host`, that records the value it was built with; it runs nothing."""

    capabilities = frozenset({"compile_only"})
    config_keys = frozenset({"host"})
    built: list[str] = []

    def __init__(self, *, host: str) -> None:
        """Record the host the recipe gave."""
        RecordingExecutor.built.append(host)

    def run(self, artifact: Path, inputs: Sequence[str], limits: Any) -> Any:
        """Fail the test: nothing runs on the compile-only path."""
        raise AssertionError("the compile-only path ran an executor")


def test_the_executor_is_built_with_its_recipe_config(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(RecordingExecutor, "built", [])
    registry = make_registry(BuildLog())
    registry.register("Executor", "recording", RecordingExecutor)
    recipe = write_recipe(tmp_path, "configured", smoke_data(executor={"kind": "recording", "host": "bench-host-1"}))
    run_dir = run(recipe, tmp_path / "runs-root", bench, registry)
    assert RecordingExecutor.built == ["bench-host-1"]
    assert load_trial(run_dir, "configured/mock-reference/lassi-hecbench-10/omp-cuda/layout/run01").attempts


# ---------------------------------------------------------------------------
# Runs root, run id, and bench root


def test_recipe_roots_option_resolves_extends(tmp_path: Path, bench: Path) -> None:
    roots = tmp_path / "recipe-roots"
    roots.mkdir()
    base = (REPO / "projects" / "base.yaml").read_text(encoding="ascii")
    assert "max_corrections: 10" in base
    (roots / "base.yaml").write_bytes(base.replace("max_corrections: 10", "max_corrections: 0").encode("ascii"))
    script = Script([BAD_REPLY, GOOD_REPLY])
    registry = make_registry(BuildLog(), backend=scripted_backend(script))
    data = smoke_data(model={"backend": "scripted", "id": "scripted-fixture"})
    options = RunOptions(
        runs_root=tmp_path / "runs-root", run_id="rooted", bench_root=bench, registry=registry, roots=[roots]
    )
    run_dir = run_recipe(write_recipe(tmp_path, "loop-test", data), options)
    trial = load_trial(run_dir, LOOP_TRIAL)
    assert len(trial.attempts) == 1 and len(script.requests) == 1, "the cap of 0 comes from the given roots"
    assert load_recipe(run_dir / "recipe.resolved.yaml", registry=registry).data["loop"]["max_corrections"] == 0


def test_runs_root_option_wins_over_the_environment(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_root = tmp_path / "env-root"
    monkeypatch.setenv("LASSI_RUNS_ROOT", str(env_root))
    run_dir = run(SMOKE, tmp_path / "option-root", bench, make_registry(BuildLog()))
    assert run_dir.resolve() == (tmp_path / "option-root" / "runs" / "test-run").resolve()
    assert not env_root.exists()


def test_runs_root_falls_back_to_the_environment_then_the_recipe(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = make_registry(BuildLog())
    env_root = tmp_path / "env-root"
    monkeypatch.setenv("LASSI_RUNS_ROOT", str(env_root))
    from_env = run_recipe(SMOKE, RunOptions(run_id="from-env", bench_root=bench, registry=registry))
    assert from_env.resolve() == (env_root / "runs" / "from-env").resolve()
    monkeypatch.delenv("LASSI_RUNS_ROOT")
    recipe_root = tmp_path / "recipe-root"
    recipe = write_recipe(tmp_path, "rooted", smoke_data(runs_root=recipe_root.as_posix()))
    from_recipe = run_recipe(recipe, RunOptions(run_id="from-recipe", bench_root=bench, registry=registry))
    assert from_recipe.resolve() == (recipe_root / "runs" / "from-recipe").resolve()


def test_run_id_defaults_to_a_utc_timestamp(tmp_path: Path, bench: Path) -> None:
    before = datetime.now(timezone.utc).replace(microsecond=0)
    options = RunOptions(runs_root=tmp_path / "runs-root", bench_root=bench, registry=make_registry(BuildLog()))
    run_dir = run_recipe(SMOKE, options)
    after = datetime.now(timezone.utc)
    assert run_dir.parent.name == "runs"
    assert re.fullmatch(r"[0-9]{8}-[0-9]{6}", run_dir.name), run_dir.name
    stamp = datetime.strptime(run_dir.name, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    assert before <= stamp <= after, "the run id is the UTC start time"


def test_bench_root_defaults_to_the_fetched_sources_under_lassi_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    write_bench(sources_dir(scratch, load_suite(SUITE_MANIFEST)))
    monkeypatch.setenv("LASSI_SCRATCH", str(scratch))
    options = RunOptions(runs_root=scratch / "lassi-runs", run_id="scratch", registry=make_registry(BuildLog()))
    run_dir = run_recipe(SMOKE, options)
    assert load_trial(run_dir, SMOKE_TRIAL).final.stage_reached == COMPILED


# ---------------------------------------------------------------------------
# The correction loop


@dataclass
class LoopRun:
    """One run of the smoke recipe with the scripted backend: the trial, the script, the builds, and the roots."""

    trial: Trial
    script: Script
    log: BuildLog
    run_dir: Path
    runs_root: Path

    def prompt(self, index: int) -> str:
        """Return the stored prompt text of attempt `index`."""
        ref = self.trial.attempts[index].prompt_ref
        assert ref is not None, f"attempt {index} has no prompt"
        return TextStore(self.run_dir).get(ref)


def run_loop(tmp_path: Path, bench: Path, replies: Sequence[str], loop: int | str | None = None) -> LoopRun:
    """Run the smoke recipe with the scripted backend answering `replies`; `loop` sets loop.max_corrections.

    With `loop` None the recipe keeps projects/base.yaml's cap of 10.
    """
    script = Script(list(replies))
    log = BuildLog()
    data = smoke_data(model={"backend": "scripted", "id": "scripted-fixture"})
    if loop is not None:
        data["loop"] = {"max_corrections": loop}
    runs_root = tmp_path / "runs-root"
    registry = make_registry(log, backend=scripted_backend(script))
    run_dir = run(write_recipe(tmp_path, "loop-test", data), runs_root, bench, registry, run_id="loop")
    return LoopRun(load_trial(run_dir, LOOP_TRIAL), script, log, run_dir, runs_root)


def test_correction_loop_fixes_an_error_with_one_correction(tmp_path: Path, bench: Path) -> None:
    result = run_loop(tmp_path, bench, [BAD_REPLY, GOOD_REPLY])
    first, second = result.trial.attempts
    assert (first.index, first.stage_reached, first.files) == (0, "S1", {"main.cu": BAD_SOURCE})
    assert first.diagnostics == fake_diagnostics("main.cu", BAD_SOURCE)
    assert first.diff_from_previous == ""
    assert (second.index, second.stage_reached, second.files) == (1, COMPILED, {"main.cu": GOOD_SOURCE})
    assert second.response_text == GOOD_REPLY and second.diagnostics == []
    assert second.diff_from_previous == unified_diff(first.files, second.files) != ""
    assert (result.trial.final.stage_reached, result.trial.final.corrections) == (COMPILED, 1)
    assert result.script.replies == [] and len(result.script.requests) == 2
    assert result.log.workdirs() == [attempt_dir(result.run_dir, LOOP_TRIAL, index) for index in (0, 1)]


def test_correction_prompt_carries_the_files_and_the_diagnostics(tmp_path: Path, bench: Path) -> None:
    result = run_loop(tmp_path, bench, [BAD_REPLY, GOOD_REPLY])
    (generate_messages, generate_sampling), (correct_messages, correct_sampling) = result.script.requests
    assert generate_messages == [Message("user", result.prompt(0))]
    assert result.prompt(0) == render("p0-smoke", "generate", generate_fields())
    assert generate_sampling == correct_sampling == SAMPLING
    last = correct_messages[-1]
    assert (last.role, last.content) == ("user", result.prompt(1))
    prompt_lines = result.prompt(1).split("\n")
    for line in BAD_DIAGNOSTIC_LINES:
        assert line in prompt_lines, line
    fields = {
        "target_language": "cuda",
        "files": render_file_blocks({"main.cu": BAD_SOURCE}),
        "diagnostics": "\n".join(BAD_DIAGNOSTIC_LINES),
        "target_files": "main.cu",
    }
    assert result.prompt(1) == render("p0-smoke", "correct", fields)


def test_max_corrections_zero_stops_after_the_first_attempt(tmp_path: Path, bench: Path) -> None:
    result = run_loop(tmp_path, bench, [BAD_REPLY], loop=0)
    (attempt,) = result.trial.attempts
    assert attempt.stage_reached == "S1"
    assert attempt.diagnostics == fake_diagnostics("main.cu", BAD_SOURCE)
    assert (result.trial.final.stage_reached, result.trial.final.corrections) == ("S1", 0)
    assert len(result.script.requests) == 1


def test_correction_cap_counts_corrections(tmp_path: Path, bench: Path) -> None:
    result = run_loop(tmp_path, bench, [BAD_REPLY] * 3, loop=2)
    assert [attempt.stage_reached for attempt in result.trial.attempts] == ["S1", "S1", "S1"]
    assert (result.trial.final.stage_reached, result.trial.final.corrections) == ("S1", 2)
    assert result.script.replies == [] and len(result.log.builds) == 3


def test_uncapped_loop_runs_past_the_default_cap_until_an_attempt_compiles(tmp_path: Path, bench: Path) -> None:
    result = run_loop(tmp_path, bench, [BAD_REPLY] * 12 + [GOOD_REPLY], loop="uncapped")
    stages_reached = [attempt.stage_reached for attempt in result.trial.attempts]
    assert stages_reached == ["S1"] * 12 + [COMPILED]
    assert (result.trial.final.stage_reached, result.trial.final.corrections) == (COMPILED, 12)
    assert result.script.replies == []


def test_reply_without_a_file_block_is_s0_and_never_compiled(tmp_path: Path, bench: Path) -> None:
    result = run_loop(tmp_path, bench, [NO_FILE_REPLY], loop=0)
    (attempt,) = result.trial.attempts
    assert (attempt.stage_reached, attempt.files) == ("S0", {})
    assert attempt.diagnostics == parse_file_blocks(NO_FILE_REPLY, ["main.cu"]).diagnostics
    assert result.log.builds == []
    assert (result.trial.final.stage_reached, result.trial.final.corrections) == ("S0", 0)


def test_a_file_block_error_is_s0_even_with_other_files(tmp_path: Path, bench: Path) -> None:
    reply = render_file_blocks({"other.cu": GOOD_SOURCE})
    result = run_loop(tmp_path, bench, [reply], loop=0)
    (attempt,) = result.trial.attempts
    parsed = parse_file_blocks(reply, ["main.cu"])
    assert attempt.stage_reached == "S0"
    assert (attempt.files, attempt.diagnostics) == (parsed.files, parsed.diagnostics)
    assert result.log.builds == []


def test_an_s0_attempt_is_corrected(tmp_path: Path, bench: Path) -> None:
    result = run_loop(tmp_path, bench, [NO_FILE_REPLY, GOOD_REPLY])
    first, second = result.trial.attempts
    assert (first.stage_reached, second.stage_reached) == ("S0", COMPILED)
    assert second.diff_from_previous == unified_diff({}, {"main.cu": GOOD_SOURCE})
    assert (result.trial.final.stage_reached, result.trial.final.corrections) == (COMPILED, 1)
    (workdir,) = result.log.workdirs()
    assert workdir == attempt_dir(result.run_dir, LOOP_TRIAL, 1)
    (missing,) = parse_file_blocks(NO_FILE_REPLY, ["main.cu"]).diagnostics
    assert f"error main.cu [missing-file] {missing.message}" in result.prompt(1).split("\n")


def test_a_file_name_too_long_to_write_is_a_bad_path_fed_back_to_the_model(tmp_path: Path, bench: Path) -> None:
    long_name = "k" * 300 + ".cuh"
    reply = GOOD_REPLY + "\n" + f"```cuda\n// FILE: {long_name}\nint k;\n```\n"
    result = run_loop(tmp_path, bench, [reply, GOOD_REPLY])
    first, second = result.trial.attempts
    assert first.stage_reached == "S0" and first.files == {"main.cu": GOOD_SOURCE}
    assert [(item.code, item.file) for item in first.diagnostics] == [("bad-path", long_name)]
    assert second.stage_reached == COMPILED
    assert len(result.log.builds) == 1, "the refused attempt is never built"


def test_a_lone_surrogate_in_a_reply_becomes_a_replacement_character_and_a_warning(tmp_path: Path, bench: Path) -> None:
    # json.loads turns the escape "\\ud83d" in an HTTP reply body into a lone surrogate, which is not Unicode text.
    source = "// note " + chr(0xD83D) + "\n" + GOOD_SOURCE
    reply = f"```cuda\n// FILE: main.cu\n{source}```\n"
    result = run_loop(tmp_path, bench, [reply])
    (attempt,) = result.trial.attempts
    assert chr(0xD83D) not in attempt.response_text
    assert attempt.files == {"main.cu": source.replace(chr(0xD83D), chr(0xFFFD))}
    assert [(item.severity, item.code) for item in attempt.diagnostics] == [("warning", "invalid-text")]
    assert attempt.stage_reached == COMPILED
    assert (result.run_dir / "run.md").is_file()


def test_each_stage_declares_the_prompt_fields_it_renders(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rendered: list[tuple[str, set[str]]] = []
    real_render = stages.render

    def recording_render(set_name: str, prompt_name: str, fields: Mapping[str, str]) -> str:
        rendered.append((prompt_name, set(fields)))
        return real_render(set_name, prompt_name, fields)

    monkeypatch.setattr(stages, "render", recording_render)
    run_loop(tmp_path, bench, [BAD_REPLY, GOOD_REPLY])
    assert rendered == [
        ("generate", set(GenerateStage.prompt_fields["generate"])),
        ("correct", set(CompileLoopStage.prompt_fields["correct"])),
    ]


# ---------------------------------------------------------------------------
# Stages called directly


def stage_context(
    tmp_path: Path, bench: Path, backend: Any, toolchain: Any, max_corrections: int | None = 10
) -> RunContext:
    """Return a RunContext for the omp to cuda direction of the synthetic layout item."""
    recipe = load_recipe(write_recipe(tmp_path, "stage-test", SMOKE_DATA), registry=make_registry(BuildLog()))
    return RunContext(
        recipe=recipe,
        backend=backend,
        sampling=SAMPLING,
        toolchains={"cuda": toolchain},
        executor=NoneExecutor(),
        store=TextStore(tmp_path / "store"),
        suite=load_suite(SUITE_MANIFEST),
        sources_root=bench,
        item=ITEM,
        direction=Direction("omp", "cuda"),
        build_root=tmp_path / "builds",
        prompts="p0-smoke",
        max_corrections=max_corrections,
    )


def stage_provenance() -> record_module.Provenance:
    """Return the synthetic Provenance of every trial the stage tests build: FAKE_COMMIT, clean, compile only."""
    return record_module.Provenance(
        commit=FAKE_COMMIT, dirty=False, device="none (compile only)", sdk=None, date="2026-09-23T12:34:56+00:00"
    )


def fresh_trial(context: RunContext) -> Trial:
    """Return a Trial with no attempts for the context's item, direction, and backend."""
    trial_id = make_trial_id("stage-test", context.backend.model_id, SUITE, context.direction.name, ITEM, 1)
    return Trial(
        trial_id=trial_id,
        recipe_hash=context.recipe.recipe_hash,
        provenance=stage_provenance(),
        bench_item=context.suite.bench_item(ITEM, context.direction),
        model=model_info(context.backend, SAMPLING),
    )


def assert_one_line(description: str) -> None:
    """Fail unless a stage description is one non-empty line."""
    assert isinstance(description, str) and description.strip() and "\n" not in description, description


def test_stage_classes_declare_what_the_registry_checks() -> None:
    assert GenerateStage.capabilities == frozenset({"generates"})
    assert GenerateStage.requires == {"LLMBackend": {"chat"}}
    assert CompileLoopStage.capabilities == frozenset({"compiles"})
    assert CompileLoopStage.requires == {"Toolchain": {"diagnostics"}}
    assert DEFAULT_REGISTRY.get("Stage", "generate").factory is GenerateStage
    assert DEFAULT_REGISTRY.get("Stage", "compile_loop").factory is CompileLoopStage


def test_stage_semantics_and_the_stage_construction_are_documented() -> None:
    doc = normalized(stages.__doc__)
    for stage in ("S0", "S1", "S2", "S3", "S4", "S5"):
        assert stage in doc, stage
    assert "factory(context=" in normalized(registry_module.__doc__)


def test_run_context_is_a_frozen_dataclass(tmp_path: Path, bench: Path) -> None:
    context = stage_context(tmp_path, bench, MockBackend(MOCK_ID), fake_toolchain("nvcc-sm80", BuildLog())())
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.item = "other"  # type: ignore[misc]


def test_generate_stage_is_pure_and_appends_attempt_zero(tmp_path: Path, bench: Path) -> None:
    reference = {"main.cu": FAKE_CUDA_SOURCE}
    log = BuildLog()
    backend = MockBackend(MOCK_ID).with_reference(reference)
    context = stage_context(tmp_path, bench, backend, fake_toolchain("nvcc-sm80", log)())
    trial = fresh_trial(context)
    before = to_json(trial)
    stage = GenerateStage(context=context)
    after = stage(trial)
    assert to_json(trial) == before and trial.attempts == []
    (attempt,) = after.attempts
    assert (attempt.index, attempt.stage_reached, attempt.files) == (0, "S1", reference)
    assert attempt.diagnostics == [] and attempt.diff_from_previous == ""
    assert attempt.response_text == render_file_blocks(reference)
    assert attempt.prompt_ref is not None
    assert context.store.get(attempt.prompt_ref) == render("p0-smoke", "generate", generate_fields())
    assert log.builds == []
    assert_one_line(stage.describe())


def test_generate_stage_gives_s0_for_a_reply_without_file_blocks(tmp_path: Path, bench: Path) -> None:
    script = Script([NO_FILE_REPLY])
    backend = scripted_backend(script)("scripted-fixture")
    context = stage_context(tmp_path, bench, backend, fake_toolchain("nvcc-sm80", BuildLog())())
    (attempt,) = GenerateStage(context=context)(fresh_trial(context)).attempts
    assert (attempt.stage_reached, attempt.files) == ("S0", {})
    assert attempt.diagnostics == parse_file_blocks(NO_FILE_REPLY, ["main.cu"]).diagnostics
    ((messages, sampling),) = script.requests
    assert attempt.prompt_ref is not None
    assert messages == [Message("user", context.store.get(attempt.prompt_ref))]
    assert sampling == SAMPLING


def test_compile_loop_stage_is_pure_and_annotates_the_last_attempt(tmp_path: Path, bench: Path) -> None:
    reference = {"main.cu": FAKE_CUDA_SOURCE}
    log = BuildLog()
    backend = MockBackend(MOCK_ID).with_reference(reference)
    context = stage_context(tmp_path, bench, backend, fake_toolchain("nvcc-sm80", log)())
    generated = GenerateStage(context=context)(fresh_trial(context))
    before = to_json(generated)
    stage = CompileLoopStage(context=context)
    compiled = stage(generated)
    assert to_json(generated) == before
    assert [attempt.stage_reached for attempt in generated.attempts] == ["S1"]
    (attempt,) = compiled.attempts
    original = generated.attempts[0]
    assert attempt.stage_reached == COMPILED
    assert (attempt.prompt_ref, attempt.response_text, attempt.files) == (
        original.prompt_ref,
        original.response_text,
        original.files,
    )
    assert attempt.run == RunInfo()
    assert log.workdirs() == [build_dir(context.build_root, generated.trial_id, 0).resolve()]
    assert_one_line(stage.describe())


def test_compile_loop_stage_stays_pure_through_a_correction(tmp_path: Path, bench: Path) -> None:
    script = Script([BAD_REPLY, GOOD_REPLY])
    log = BuildLog()
    backend = scripted_backend(script)("scripted-fixture")
    context = stage_context(tmp_path, bench, backend, fake_toolchain("nvcc-sm80", log)())
    generated = GenerateStage(context=context)(fresh_trial(context))
    before = to_json(generated)
    corrected = CompileLoopStage(context=context)(generated)
    assert to_json(generated) == before
    assert [attempt.stage_reached for attempt in corrected.attempts] == ["S1", COMPILED]
    trial_id = generated.trial_id
    expected = [build_dir(context.build_root, trial_id, index).resolve() for index in (0, 1)]
    assert log.workdirs() == expected
    assert generated.provenance == corrected.provenance == stage_provenance(), "stages never change the provenance"


def test_compile_loop_stage_uncapped_context_keeps_correcting(tmp_path: Path, bench: Path) -> None:
    script = Script([BAD_REPLY] * 12 + [GOOD_REPLY])
    backend = scripted_backend(script)("scripted-fixture")
    context = stage_context(tmp_path, bench, backend, fake_toolchain("nvcc-sm80", BuildLog())(), None)
    corrected = CompileLoopStage(context=context)(GenerateStage(context=context)(fresh_trial(context)))
    assert len(corrected.attempts) == 13 and corrected.attempts[-1].stage_reached == COMPILED


def compile_with_steps(
    tmp_path: Path, bench: Path, steps: Sequence[BaseException | BuildResult]
) -> tuple[Trial, StepToolchain, Script]:
    """Generate and compile the good reply twice over with a StepToolchain playing `steps`; return the result."""
    script = Script([GOOD_REPLY, GOOD_REPLY])
    toolchain = StepToolchain(steps)
    context = stage_context(tmp_path, bench, scripted_backend(script)("scripted-fixture"), toolchain)
    trial = CompileLoopStage(context=context)(GenerateStage(context=context)(fresh_trial(context)))
    return trial, toolchain, script


@pytest.mark.parametrize("code", [errno.ENAMETOOLONG, errno.EINVAL], ids=["name-too-long", "invalid-name"])
def test_a_file_name_the_filesystem_refuses_is_an_unwritable_error_fed_back(
    tmp_path: Path, bench: Path, code: int
) -> None:
    trial, toolchain, script = compile_with_steps(tmp_path, bench, [OSError(code, os.strerror(code), "x.cu")])
    first, second = trial.attempts
    assert first.stage_reached == "S1"
    assert [(item.severity, item.code) for item in first.diagnostics] == [("error", "unwritable")]
    assert second.stage_reached == COMPILED
    assert (toolchain.builds, len(script.requests)) == (2, 2)


def test_an_os_error_that_is_not_about_a_name_stops_the_trial(tmp_path: Path, bench: Path) -> None:
    with pytest.raises(OSError, match="No space"):
        compile_with_steps(tmp_path, bench, [OSError(errno.ENOSPC, "No space left on device")])


def test_a_failed_build_without_an_error_gets_a_no_artifact_error(tmp_path: Path, bench: Path) -> None:
    silent = BuildResult(artifact=None, diagnostics=[])
    trial, toolchain, script = compile_with_steps(tmp_path, bench, [silent])
    first, second = trial.attempts
    assert first.stage_reached == "S1"
    assert [(item.stage, item.severity, item.code) for item in first.diagnostics] == [
        ("compile", "error", "no-artifact")
    ]
    assert second.stage_reached == COMPILED and len(script.requests) == 2


@pytest.mark.parametrize(
    ("diagnostic", "line"),
    [
        (
            Diagnostic(stage="compile", severity="error", file="main.cu", line=3, column=7, code="c", message="m"),
            "error main.cu:3:7 [c] m",
        ),
        (
            Diagnostic(stage="compile", severity="error", file="main.cu", line=3, code="c", message="m"),
            "error main.cu:3 [c] m",
        ),
        (
            Diagnostic(stage="compile", severity="error", file="main.cu", column=7, code="c", message="m"),
            "error main.cu [c] m",
        ),
        (Diagnostic(stage="compile", severity="note", line=3, message="m"), "note m"),
        (
            Diagnostic(stage="compile", severity="warning", file="a.cu", line=1, message="first\nsecond\r\nthird"),
            "warning a.cu:1 first second third",
        ),
    ],
    ids=["full", "no-column", "column-without-line", "no-file", "multi-line-message"],
)
def test_diagnostic_line_leaves_out_absent_parts_and_stays_one_line(diagnostic: Diagnostic, line: str) -> None:
    assert diagnostic_line(diagnostic) == line


def test_stage_labels_follow_the_bible_stage_ladder() -> None:
    bible = (REPO / "docs" / "BIBLE.md").read_text(encoding="utf-8")
    ladder = dict(re.findall(r"^\| (S[0-5]) \| ([^|]+?) \|", bible, flags=re.MULTILINE))
    assert sorted(ladder) == ["S0", "S1", "S2", "S3", "S4", "S5"]
    assert ladder[stages.NO_OUTPUT] == "No extractable output"
    assert ladder[stages.PARSED] == "Parses"
    assert ladder[stages.COMPILED].startswith("Compiles")
    assert stages.COMPILED == COMPILED


# ---------------------------------------------------------------------------
# Refusals


def test_refuses_arms_without_a_model(tmp_path: Path, bench: Path) -> None:
    # The real stage list: generate needs an LLMBackend, which only model binds, so the loader refuses it first.
    data = smoke_data(model=None, arms=["arm-a", "arm-b"])
    recipe = write_recipe(tmp_path, "arms-only", data)
    with pytest.raises(RunError, match="arms but no model; the model registry that maps arms to backends"):
        run(recipe, tmp_path / "runs-root", bench, make_registry(BuildLog()))
    assert not (tmp_path / "runs-root").exists()


@pytest.mark.parametrize(
    ("model", "message"),
    [(None, "names no model"), ({"backend": "mock"}, "model.id has no value")],
    ids=["no-model", "no-model-id"],
)
def test_refuses_an_incomplete_model_section_with_a_run_error(
    tmp_path: Path, bench: Path, model: dict[str, str] | None, message: str
) -> None:
    recipe = write_recipe(tmp_path, "no-model", smoke_data(model=model))
    with pytest.raises(RunError, match=message):
        run(recipe, tmp_path / "runs-root", bench, make_registry(BuildLog()))


def test_refuses_arms_beside_a_model(tmp_path: Path, bench: Path) -> None:
    recipe = write_recipe(tmp_path, "model-and-arms", smoke_data(arms=["arm-a"]))
    with pytest.raises(RunError, match="both model and arms"):
        run(recipe, tmp_path / "runs-root", bench, make_registry(BuildLog()))
    assert not (tmp_path / "runs-root").exists()


def test_refuses_a_recipe_without_max_tokens(tmp_path: Path, bench: Path) -> None:
    recipe = write_recipe(tmp_path, "no-max-tokens", smoke_data(llm=None))
    with pytest.raises(RunError, match="max_tokens"):
        run(recipe, tmp_path / "runs-root", bench, make_registry(BuildLog()))


def test_refuses_a_stage_the_runner_does_not_implement(tmp_path: Path, bench: Path) -> None:
    data = smoke_data(stages=["generate", "compile_loop", "unimplemented_stage"])
    recipe = write_recipe(tmp_path, "later-stage", data)
    with pytest.raises(RecipeError, match="unimplemented_stage"):
        run_recipe(recipe, RunOptions(runs_root=tmp_path / "runs-root", run_id="x", bench_root=bench))


def assert_refused_before_anything_runs(
    tmp_path: Path, bench: Path, name: str, data: Mapping[str, Any], match: str, registry: Registry | None = None
) -> None:
    """Run the recipe `data` and assert a RunError matching `match`, with no run tree and no model call."""
    script = Script([])
    chosen = registry or make_registry(BuildLog(), backend=scripted_backend(script))
    with pytest.raises(RunError, match=match):
        run(write_recipe(tmp_path, name, data), tmp_path / "runs-root", bench, chosen)
    assert not (tmp_path / "runs-root").exists(), "a refused run creates no directory"
    assert script.requests == [], "a refused run asks no model"


def scripted_data(**changes: Any) -> dict[str, Any]:
    """Return smoke_data(**changes) answered by the scripted backend, so any model call is recorded."""
    return smoke_data(**{"model": {"backend": "scripted", "id": "scripted-fixture"}, **changes})


@pytest.mark.parametrize(
    "stage_list",
    [
        ["compile_loop"],
        ["compile_loop", "generate"],
        ["generate", "generate"],
        ["generate", "compile_loop", "generate"],
    ],
    ids=["compile-alone", "compile-first", "generate-twice", "generate-after-compile"],
)
def test_refuses_a_stage_order_that_cannot_run(tmp_path: Path, bench: Path, stage_list: list[str]) -> None:
    data = scripted_data(stages=stage_list)
    assert_refused_before_anything_runs(tmp_path, bench, "bad-order", data, r"stages\[[0-9]+\]")


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"faithful": True}, r"fixes\.fence_tag is off \(faithful: true\)"),
        ({"fixes": {"fence_tag": False}}, r"fixes\.fence_tag is off \(faithful: false\)"),
        ({"report": {"trial_md": False}}, r"report\.trial_md is false"),
        ({"report": {"parquet": False}}, r"report\.parquet is false"),
        ({"context": ["openmp-card"]}, "sets context, which this runner does not carry out"),
        ({"metrics": ["pass-at-1"]}, "sets metrics, which this runner does not carry out"),
    ],
    ids=["faithful", "fence-tag-off", "no-trial-md", "no-parquet", "context", "metrics"],
)
def test_refuses_a_recipe_choice_the_stages_cannot_honor(
    tmp_path: Path, bench: Path, changes: dict[str, Any], match: str
) -> None:
    assert_refused_before_anything_runs(tmp_path, bench, "unhonored", scripted_data(**changes), match)


def prompt_sets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point lassi.prompts at a temporary root holding broken prompt sets and return it.

    "half" has generate.txt but no correct.txt, and "stray-field" uses a
    placeholder no stage fills; "full" is a copy of p0-smoke.
    """
    root = tmp_path / "prompt-sets"
    smoke = REPO / "assets" / "prompts" / "p0-smoke"
    generate, correct = ((smoke / name).read_bytes() for name in ("generate.txt", "correct.txt"))
    for name, files in (
        ("half", {"generate.txt": generate}),
        ("stray-field", {"generate.txt": generate + b"$context_pack\n", "correct.txt": correct}),
        ("full", {"generate.txt": generate, "correct.txt": correct}),
    ):
        (root / name).mkdir(parents=True)
        for file, data in files.items():
            (root / name / file).write_bytes(data)
    monkeypatch.setattr(prompts_module, "default_roots", lambda: (root,))
    return root


@pytest.mark.parametrize(
    ("prompt_set", "match"),
    [
        ("no-such-set", "no prompt set 'no-such-set'"),
        ("half", r"no prompt file .*correct\.txt"),
        ("stray-field", r"\$context_pack has no field value"),
    ],
    ids=["missing-set", "missing-correct", "stray-placeholder"],
)
def test_refuses_a_prompt_set_a_stage_cannot_render(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch, prompt_set: str, match: str
) -> None:
    prompt_sets(tmp_path, monkeypatch)
    data = scripted_data(prompts=prompt_set)
    assert_refused_before_anything_runs(tmp_path, bench, "prompt-check", data, match)


def test_a_complete_prompt_set_from_another_root_runs(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt_sets(tmp_path, monkeypatch)
    run_dir = run(
        write_recipe(tmp_path, "full-set", smoke_data(prompts="full")),
        tmp_path / "rr",
        bench,
        make_registry(BuildLog()),
    )
    assert (
        load_trial(run_dir, "full-set/mock-reference/lassi-hecbench-10/omp-cuda/layout/run01").final.stage_reached
        == COMPILED
    )


def test_generate_only_skips_the_correct_prompt_check(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt_sets(tmp_path, monkeypatch)
    data = smoke_data(prompts="half", stages=["generate"])
    run_dir = run(write_recipe(tmp_path, "half-set", data), tmp_path / "rr", bench, make_registry(BuildLog()))
    trial = load_trial(run_dir, "half-set/mock-reference/lassi-hecbench-10/omp-cuda/layout/run01")
    assert trial.final.stage_reached == "S1"


@pytest.mark.parametrize(
    ("directions", "toolchain"),
    [
        ([{"source": "omp", "target": "cuda"}, {"source": "omp", "target": "cuda"}], {"cuda": "nvcc-sm80"}),
        (
            [{"source": "omp", "target": "cuda"}, {"source": "OMP", "target": "CUDA"}],
            {"cuda": "nvcc-sm80", "CUDA": "nvcc-sm80"},
        ),
    ],
    ids=["same", "same-but-for-case"],
)
def test_refuses_trial_ids_that_repeat(
    tmp_path: Path, bench: Path, directions: list[dict[str, str]], toolchain: dict[str, str]
) -> None:
    data = scripted_data(directions=directions, toolchain=toolchain)
    assert_refused_before_anything_runs(tmp_path, bench, "twice", data, "would share one trial directory")


def test_refuses_a_missing_bench_root_and_names_fetch_bench(tmp_path: Path) -> None:
    with pytest.raises(RunError, match="fetch_bench"):
        run(SMOKE, tmp_path / "runs-root", tmp_path / "not-fetched", make_registry(BuildLog()))


def test_refuses_missing_scratch_sources_and_names_fetch_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LASSI_SCRATCH", str(tmp_path / "scratch"))
    options = RunOptions(runs_root=tmp_path / "scratch" / "lassi-runs", run_id="x", registry=make_registry(BuildLog()))
    with pytest.raises(RunError, match="fetch_bench"):
        run_recipe(SMOKE, options)


@pytest.mark.parametrize("source", ["option", "environment", "recipe"])
def test_refuses_a_relative_runs_root(
    source: str, tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    relative = "relative-runs"
    recipe, runs_root = SMOKE, None
    if source == "option":
        runs_root = Path(relative)
    elif source == "environment":
        monkeypatch.setenv("LASSI_RUNS_ROOT", relative)
    else:
        recipe = write_recipe(tmp_path, "relative-root", smoke_data(runs_root=relative))
    options = RunOptions(runs_root=runs_root, run_id="x", bench_root=bench, registry=make_registry(BuildLog()))
    with pytest.raises(RunError):
        run_recipe(recipe, options)
    assert not (tmp_path / relative).exists()


def test_refuses_a_runs_root_inside_the_repository(bench: Path) -> None:
    inside = REPO / "runs-root-inside-repository-test"
    try:
        with pytest.raises(RunError):
            run(SMOKE, inside, bench, make_registry(BuildLog()))
        assert not inside.exists(), "a refused runs root must not be created"
    finally:
        if inside.exists():
            shutil.rmtree(inside)


def test_refuses_a_run_directory_that_resolves_inside_the_repository(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A stand-in repository named "runs": its parent passes as a runs root, but <root>/runs/<id> lies inside it.
    stand_in = tmp_path / "outer" / "runs"
    stand_in.mkdir(parents=True)
    monkeypatch.setattr(runner_module, "REPO", stand_in)
    with pytest.raises(RunError, match="the run directory .* is inside the repository"):
        run(SMOKE, tmp_path / "outer", bench, make_registry(BuildLog()))
    assert list(stand_in.iterdir()) == [], "nothing is written into the repository"


def test_refuses_a_runs_root_outside_lassi_scratch(
    tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LASSI_SCRATCH", str(tmp_path / "scratch"))
    with pytest.raises(RunError, match=r"outside \$LASSI_SCRATCH"):
        run(SMOKE, tmp_path / "elsewhere", bench, make_registry(BuildLog()))
    assert not (tmp_path / "elsewhere").exists()


def test_refuses_a_pinned_compile_without_tmpdir(
    tmp_path: Path, bench: Path, processes: Processes, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = pinned_root(tmp_path, NVCC_BIN)
    processes.watch(root)
    monkeypatch.delenv("TMPDIR")
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="x", bench_root=bench, toolchains_root=root)
    with pytest.raises(RunError, match="TMPDIR is not set"):
        run_recipe(SMOKE, options)
    assert processes.compiles() == [] and not (tmp_path / "runs-root").exists()


def test_refuses_a_missing_pinned_executable_and_names_its_install_script(tmp_path: Path, bench: Path) -> None:
    root = tmp_path / "toolchains"
    root.mkdir()
    registry = make_registry(BuildLog(), toolchain=PinnedFake)
    options = RunOptions(
        runs_root=tmp_path / "runs-root", run_id="x", bench_root=bench, toolchains_root=root, registry=registry
    )
    with pytest.raises(RunError) as refused:
        run_recipe(SMOKE, options)
    message = str(refused.value)
    assert "toolchains/cuda.sh" in message
    assert "cuda@12.6.3" in message and "nvcc" in message


def test_refuses_a_missing_nvhpc_compiler_and_names_its_install_script(
    tmp_path: Path, bench: Path, processes: Processes
) -> None:
    root = pinned_root(tmp_path)
    processes.watch(root)
    recipe = write_recipe(tmp_path, "omp-target", nvhpc_data())
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="x", bench_root=bench, toolchains_root=root)
    with pytest.raises(RunError, match="toolchains/nvhpc.sh"):
        run_recipe(recipe, options)
    assert processes.compiles() == []


def test_refuses_a_pinned_toolchain_without_a_toolchains_root(
    tmp_path: Path, bench: Path, processes: Processes
) -> None:
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="x", bench_root=bench)
    with pytest.raises(RunError, match="LASSI_TOOLCHAINS"):
        run_recipe(SMOKE, options)
    assert [call.argv for call in processes.calls if "nvcc" in call.argv[0]] == []


# ---------------------------------------------------------------------------
# Pins and the compile environment


def test_read_pin_parses_the_committed_pin_files() -> None:
    cuda, nvhpc = read_pin("cuda"), read_pin("nvhpc")
    assert (cuda["NAME"], cuda["VERSION"], cuda["PREFIX_NAME"]) == ("cuda", "12.6.3", "cuda@12.6.3")
    assert cuda["INSTALL_FLAGS"] == "--silent --toolkit --no-opengl-libs --no-man-page --no-drm --override"
    assert cuda["MD5"] == "29d297908c72b810c9ceaa5177142abd", "a trailing comment is not part of the value"
    assert cuda["EXPECT_VERSION"] == "Cuda compilation tools, release 12.6, V12.6.85"
    assert (nvhpc["NAME"], nvhpc["VERSION"], nvhpc["PREFIX_NAME"]) == ("nvhpc", "24.11", "nvhpc@24.11")
    assert nvhpc["COMPILER_SUBDIR"] == "Linux_x86_64/24.11/compilers/bin"
    assert nvhpc["CUDA_HOME_FROM"] == CUDA_PREFIX
    assert f"{cuda['PREFIX_NAME']}/bin/nvcc" == NVCC_BIN
    assert f"{nvhpc['PREFIX_NAME']}/{nvhpc['COMPILER_SUBDIR']}/nvc++" == NVCPP_BIN


def test_compiler_presets_declare_their_pins() -> None:
    assert (NvccSm80.PIN, NvccSm80.PIN_BIN) == ("cuda", "bin/nvcc")
    assert (NvcppCc80.PIN, NvcppCc80.PIN_BIN) == ("nvhpc", "{COMPILER_SUBDIR}/nvc++")


def test_env_runner_passes_exactly_the_given_environment(
    tmp_path: Path, processes: Processes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NVCC_PREPEND_FLAGS", "-DLEAKED_PREPEND")
    executable = tmp_path / "bin" / "fakecc"
    processes.watch(tmp_path / "bin")
    env = {"PATH": "/pinned/bin", "LANG": "C", "LC_ALL": "C"}
    argv = [str(executable), "-o", "main", "main.cu"]
    result = EnvRunner(env)(argv, tmp_path, 30.0)
    assert result == CommandResult(returncode=0, stdout="", stderr="")
    (call,) = processes.compiles()
    assert call.argv == argv
    assert call.cwd is not None and call.cwd.resolve() == tmp_path.resolve()
    assert call.env == env


def test_env_runner_returns_the_status_and_the_decoded_output(tmp_path: Path, processes: Processes) -> None:
    processes.watch(tmp_path / "bin")
    processes.returncode = 2
    processes.stdout = b"built\n"
    processes.stderr = b"main.cu(3): error: bad \xff byte\n"
    result = EnvRunner({"PATH": "/pinned/bin"})([str(tmp_path / "bin" / "fakecc")], tmp_path, 30.0)
    assert result == CommandResult(returncode=2, stdout="built\n", stderr="main.cu(3): error: bad \ufffd byte\n")


def test_pinned_nvcc_builds_with_the_pinned_executable_and_a_clean_environment(
    tmp_path: Path, bench: Path, processes: Processes, parent_env: dict[str, str]
) -> None:
    root = pinned_root(tmp_path, NVCC_BIN)
    processes.watch(root)
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="pinned", bench_root=bench, toolchains_root=root)
    run_dir = run_recipe(SMOKE, options)
    (call,) = processes.compiles()
    assert Path(call.argv[0]) == root / NVCC_BIN
    assert "main.cu" in call.argv
    assert call.env == parent_env
    trial = load_trial(run_dir, SMOKE_TRIAL)
    assert trial.toolchain_pins == ToolchainPins(cuda="12.6.3")
    assert trial.final.stage_reached == COMPILED
    toolchains = json.loads(read_ascii(run_dir / "toolchains.json"))
    assert set(toolchains) == {"nvcc-sm80"}
    entry = toolchains["nvcc-sm80"]
    assert (entry["languages"], entry["executable"], entry["environment"]) == (
        ["cuda"],
        str(root / NVCC_BIN),
        parent_env,
    )
    assert set(entry["pins"]) == {"cuda"} and entry["pins"]["cuda"] == read_pin("cuda")
    assert json.loads(read_ascii(run_dir / "provenance.json"))["pins"] == {"cuda": "12.6.3"}
    run_md_lines = read_ascii(run_dir / "run.md").split("\n")
    pin_rows = [table_cells(line) for line in run_md_lines if "cuda@12.6.3" in line and line.startswith("|")]
    assert pin_rows == [["nvcc-sm80", "cuda", "cuda", "12.6.3", "cuda@12.6.3"]]


def test_nvhpc_build_gets_nvhpc_cuda_home_and_records_both_pins(
    tmp_path: Path, bench: Path, processes: Processes, parent_env: dict[str, str]
) -> None:
    root = pinned_root(tmp_path, NVCPP_BIN)
    processes.watch(root)
    recipe = write_recipe(tmp_path, "omp-target", nvhpc_data())
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="nvhpc", bench_root=bench, toolchains_root=root)
    run_dir = run_recipe(recipe, options)
    (call,) = processes.compiles()
    assert Path(call.argv[0]) == root / NVCPP_BIN
    assert call.env is not None
    env = dict(call.env)
    assert Path(env.pop("NVHPC_CUDA_HOME")) == root / CUDA_PREFIX
    assert env == parent_env
    trial = load_trial(run_dir, NVHPC_TRIAL)
    assert trial.toolchain_pins == ToolchainPins(cuda="12.6.3", nvhpc="24.11")
    assert trial.final.stage_reached == COMPILED
    pins = json.loads(read_ascii(run_dir / "toolchains.json"))["nvcpp-cc80"]["pins"]
    assert (pins["nvhpc"]["VERSION"], pins["cuda"]["VERSION"]) == ("24.11", "12.6.3")
    assert json.loads(read_ascii(run_dir / "provenance.json"))["pins"] == {"cuda": "12.6.3", "nvhpc": "24.11"}


def test_each_trial_records_the_pins_of_the_toolchain_that_builds_its_target(
    tmp_path: Path, bench: Path, processes: Processes, parent_env: dict[str, str]
) -> None:
    root = pinned_root(tmp_path, NVCC_BIN, NVCPP_BIN)
    processes.watch(root)
    data = smoke_data(
        directions=[{"source": "omp", "target": "cuda"}, {"source": "cuda", "target": "omp"}],
        toolchain={"cuda": "nvcc-sm80", "omp": "nvcpp-cc80"},
    )
    recipe = write_recipe(tmp_path, "both-ways", data)
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="both", bench_root=bench, toolchains_root=root)
    run_dir = run_recipe(recipe, options)
    to_cuda, to_omp = (make_trial_id("both-ways", MOCK_ID, SUITE, name, ITEM, 1) for name in ("omp-cuda", "cuda-omp"))
    assert load_trial(run_dir, to_cuda).toolchain_pins == ToolchainPins(cuda="12.6.3")
    assert load_trial(run_dir, to_omp).toolchain_pins == ToolchainPins(cuda="12.6.3", nvhpc="24.11")
    assert json.loads(read_ascii(run_dir / "provenance.json"))["pins"] == {"cuda": "12.6.3", "nvhpc": "24.11"}


def test_every_default_toolchain_declares_a_pin_the_runner_can_read() -> None:
    # A Toolchain registered without PIN would compile with whatever compiler PATH finds (Agent Rule 10).
    names = DEFAULT_REGISTRY.names("Toolchain")
    assert names
    for name in names:
        factory = DEFAULT_REGISTRY.get("Toolchain", name).factory
        assert isinstance(getattr(factory, "PIN", None), str) and isinstance(getattr(factory, "PIN_BIN", None), str), (
            name
        )
        pin = read_pin(factory.PIN)
        assert pin["VERSION"] and pin["PREFIX_NAME"], name
        factory.PIN_BIN.format_map(pin)


# ---------------------------------------------------------------------------
# build_toolchain: one toolchain built as the runner builds it (P0.15)


def both_toolchains_recipe(directory: Path) -> Recipe:
    """Return a loaded recipe that binds nvcc-sm80 for cuda and nvcpp-cc80 for omp."""
    data = smoke_data(
        directions=[{"source": "omp", "target": "cuda"}, {"source": "cuda", "target": "omp"}],
        toolchain={"cuda": "nvcc-sm80", "omp": "nvcpp-cc80"},
    )
    return load_recipe(write_recipe(directory, "both-ways", data))


def built_fields(built: Any) -> tuple[Any, ...]:
    """Return what a built toolchain holds, with the toolchain object reduced to its class, executable, and runner.

    The runner is reduced to its class and its environment (EnvRunner.env),
    so two separately built toolchains compare equal when they were built
    the same way.
    """
    toolchain = built.toolchain
    runner = getattr(toolchain, "runner", None)
    return (
        built.name,
        type(toolchain),
        getattr(toolchain, "executable", None),
        type(runner),
        getattr(runner, "env", None),
        built.executable,
        built.environment,
        built.pins,
    )


def test_build_toolchain_builds_each_pinned_preset_as_the_runner_does(
    tmp_path: Path, parent_env: dict[str, str]
) -> None:
    root = pinned_root(tmp_path, NVCC_BIN, NVCPP_BIN)
    built_by_runner = runner_module._toolchains(both_toolchains_recipe(tmp_path), DEFAULT_REGISTRY, root)
    from_recipe = {built.name: built for built in built_by_runner}
    for name, language in (("nvcc-sm80", "cuda"), ("nvcpp-cc80", "omp")):
        built = runner_module.build_toolchain(name, root)
        assert isinstance(built, runner_module.BuiltToolchain), name
        assert isinstance(from_recipe[name], runner_module.BuiltToolchain), name
        assert built.languages == (), "a toolchain built outside a recipe builds no recipe language"
        assert from_recipe[name].languages == (language,)
        assert built_fields(built) == built_fields(from_recipe[name]), name


def test_build_toolchain_gives_the_pinned_executable_the_clean_environment_and_the_pins(
    tmp_path: Path, parent_env: dict[str, str]
) -> None:
    root = pinned_root(tmp_path, NVCC_BIN, NVCPP_BIN)
    nvcc = runner_module.build_toolchain("nvcc-sm80", root)
    assert nvcc.name == "nvcc-sm80"
    assert isinstance(nvcc.toolchain, NvccSm80) and isinstance(nvcc.toolchain.runner, EnvRunner)
    assert nvcc.executable == nvcc.toolchain.executable == str(root / NVCC_BIN)
    assert nvcc.environment == nvcc.toolchain.runner.env == parent_env
    assert nvcc.pins == {"cuda": read_pin("cuda")}
    nvcpp = runner_module.build_toolchain("nvcpp-cc80", root, DEFAULT_REGISTRY)
    assert isinstance(nvcpp.toolchain, NvcppCc80) and isinstance(nvcpp.toolchain.runner, EnvRunner)
    assert nvcpp.executable == nvcpp.toolchain.executable == str(root / NVCPP_BIN)
    assert nvcpp.environment is not None and nvcpp.toolchain.runner.env == nvcpp.environment
    environment = dict(nvcpp.environment)
    assert Path(environment.pop("NVHPC_CUDA_HOME")) == root / CUDA_PREFIX
    assert environment == parent_env
    assert nvcpp.pins == {"nvhpc": read_pin("nvhpc"), "cuda": read_pin("cuda")}


def test_the_runner_builds_each_bound_toolchain_through_build_toolchain(
    tmp_path: Path, parent_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # One construction path: _toolchains calls the public function, so the capture tool builds the same thing.
    root = pinned_root(tmp_path, NVCC_BIN, NVCPP_BIN)
    original = runner_module.build_toolchain
    signature = inspect.signature(original)
    calls: list[tuple[str, Any, Any]] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        calls.append((bound.arguments["name"], bound.arguments["root"], bound.arguments["registry"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(runner_module, "build_toolchain", spy)
    built = runner_module._toolchains(both_toolchains_recipe(tmp_path), DEFAULT_REGISTRY, root)
    assert sorted(item.name for item in built) == ["nvcc-sm80", "nvcpp-cc80"]
    assert sorted(calls, key=lambda call: call[0]) == [
        ("nvcc-sm80", root, DEFAULT_REGISTRY),
        ("nvcpp-cc80", root, DEFAULT_REGISTRY),
    ]


def test_build_toolchain_builds_a_toolchain_without_a_pin_as_its_bare_factory() -> None:
    registry = make_registry(BuildLog())
    built = runner_module.build_toolchain("nvcc-sm80", None, registry)
    assert isinstance(built, runner_module.BuiltToolchain)
    assert type(built.toolchain) is registry.get("Toolchain", "nvcc-sm80").factory
    assert (built.name, built.languages, built.executable, built.environment, built.pins) == (
        "nvcc-sm80",
        (),
        None,
        None,
        {},
    )


def test_build_toolchain_registry_defaults_to_the_default_registry() -> None:
    parameters = inspect.signature(runner_module.build_toolchain).parameters
    assert list(parameters)[:3] == ["name", "root", "registry"]
    assert parameters["registry"].default is DEFAULT_REGISTRY


def test_build_toolchain_refuses_what_the_runner_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RunError, match="LASSI_TOOLCHAINS"):
        runner_module.build_toolchain("nvcc-sm80", None)
    with pytest.raises(RunError, match="absolute"):
        runner_module.build_toolchain("nvcc-sm80", Path("relative-toolchains"))
    empty = tmp_path / "empty-toolchains"
    empty.mkdir()
    with pytest.raises(RunError, match="toolchains/cuda.sh"):
        runner_module.build_toolchain("nvcc-sm80", empty)
    root = pinned_root(tmp_path, NVCC_BIN)
    with pytest.raises(RunError, match="toolchains/nvhpc.sh"):
        runner_module.build_toolchain("nvcpp-cc80", root)
    monkeypatch.delenv("TMPDIR")
    with pytest.raises(RunError, match="TMPDIR is not set"):
        runner_module.build_toolchain("nvcc-sm80", root)


# ---------------------------------------------------------------------------
# The command line


def test_cli_run_returns_zero_and_prints_the_run_directory(
    tmp_path: Path,
    bench: Path,
    processes: Processes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = pinned_root(tmp_path, NVCC_BIN)
    processes.watch(root)
    monkeypatch.setenv("LASSI_TOOLCHAINS", str(root))
    runs_root = tmp_path / "runs-root"
    argv = ["run", str(SMOKE), "--runs-root", str(runs_root), "--run-id", "cli", "--bench-root", str(bench)]
    assert cli.main(argv) == 0
    run_dir = runs_root / "runs" / "cli"
    assert (run_dir / "run.md").is_file()
    assert load_trial(run_dir, SMOKE_TRIAL).final.stage_reached == COMPILED
    assert str(run_dir) in capsys.readouterr().out
    assert len(processes.compiles()) == 1


def test_cli_returns_two_and_prints_a_recipe_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "absent-recipe.yaml"
    assert cli.main(["run", str(missing), "--runs-root", str(tmp_path / "runs-root")]) == 2
    captured = capsys.readouterr()
    assert "absent-recipe.yaml" in captured.out + captured.err


def test_cli_returns_two_and_prints_a_run_error(
    tmp_path: Path, processes: Processes, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = pinned_root(tmp_path, NVCC_BIN)
    processes.watch(root)
    monkeypatch.setenv("LASSI_TOOLCHAINS", str(root))
    argv = ["run", str(SMOKE), "--runs-root", str(tmp_path / "runs-root"), "--bench-root", str(tmp_path / "absent")]
    assert cli.main(argv) == 2
    captured = capsys.readouterr()
    assert "fetch_bench" in captured.out + captured.err
    assert processes.compiles() == []


def test_cli_returns_two_and_prints_a_sandbox_error(
    tmp_path: Path,
    bench: Path,
    processes: Processes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = pinned_root(tmp_path, NVCC_BIN)
    processes.watch(root)
    monkeypatch.setenv("LASSI_TOOLCHAINS", str(root))

    def unavailable(self: object, *args: object, **kwargs: object) -> None:
        raise SandboxUnavailableError("fixture: no isolation on this host")

    monkeypatch.setattr(NoneExecutor, "__init__", unavailable)
    argv = ["run", str(SMOKE), "--runs-root", str(tmp_path / "runs-root"), "--bench-root", str(bench)]
    assert cli.main(argv) == 2
    captured = capsys.readouterr()
    assert "fixture: no isolation on this host" in captured.out + captured.err


@pytest.mark.parametrize("command", ["train", "corpus", "export", "report"])
def test_cli_adds_no_other_subcommand(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    try:
        code = cli.main([command])
    except SystemExit as exit_:
        code = exit_.code
    assert code == 2


def test_pyproject_declares_the_lassi_command() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    assert "[project.scripts]" in text
    section = text.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    assert 'lassi = "lassi.cli:main"' in section


# ---------------------------------------------------------------------------
# Module hygiene (Agent Rule 3, Readability Standards, no new dependency)


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
    params += [arg for arg in (args.vararg, args.kwarg) if arg is not None]
    params = [param for param in params if param.arg not in ("self", "cls")]
    return node.returns is not None and all(param.annotation is not None for param in params)


def imported_roots(tree: ast.Module) -> set[str]:
    """Return the top-level package of every absolute import in a module."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("name", NEW_MODULES)
def test_new_modules_are_documented_typed_and_ascii(name: str) -> None:
    source = module_source(name)
    tree = ast.parse(source)
    assert ast.get_docstring(tree), f"{name} has no module docstring"
    undocumented = [qual for qual, node in public_defs(tree) if not ast.get_docstring(node)]
    assert not undocumented, f"{name}: no docstring on {undocumented}"
    untyped = [qual for qual, node in public_defs(tree) if isinstance(node, FUNCTION_NODES) and not is_typed(node)]
    assert not untyped, f"{name}: missing type hints on {untyped}"
    assert "from __future__ import annotations" in source, name


@pytest.mark.parametrize("name", NEW_MODULES)
def test_new_modules_hold_no_project_code_and_no_new_dependency(name: str) -> None:
    source = module_source(name)
    found = [word for word in PROJECT_NAMES if word in source.lower()]
    assert not found, f"{name} names projects: {found}"
    roots = imported_roots(ast.parse(source))
    assert "projects" not in roots, f"{name} imports the projects package"
    third_party = sorted(root for root in roots if root not in sys.stdlib_module_names and root != "lassi")
    assert set(third_party) <= ALLOWED_THIRD_PARTY, f"{name} imports {third_party}"


@pytest.mark.parametrize("module", ["lassi.core.runner", "lassi.cli"])
def test_importing_the_runner_registers_every_component(module: str) -> None:
    code = (
        "import json\n"
        f"import {module}\n"
        "from lassi.core.registry import DEFAULT_REGISTRY as registry\n"
        "names = ('LLMBackend', 'Toolchain', 'Executor', 'Stage')\n"
        "print(json.dumps({name: registry.names(name) for name in names}))\n"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    names = json.loads(done.stdout)
    assert {"mock", "openai_compat", "ollama"} <= set(names["LLMBackend"])
    assert {"nvcc-sm80", "nvcpp-cc80"} <= set(names["Toolchain"])
    assert {"none", "native"} <= set(names["Executor"])
    assert {"generate", "compile_loop"} <= set(names["Stage"])
