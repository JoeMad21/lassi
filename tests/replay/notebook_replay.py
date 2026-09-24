"""Harness for the notebook replay (task P1.9): upstream's pipeline and our faithful stages on one script.

Bible: Build Roadmap (P1 row, Gate), Source Papers (LASSI pipeline; quirk
table), Design Principle 4, Agent Rules 1, 4, and 6.

A case is one scenario of tests/fixtures/replay/scenarios.json in one
direction. Both sides get the same scripted inputs: the recorded replies (a
synthetic recording read by the `replay` LLMBackend), the compile outcomes
in order (the target reference's first), and the run outcomes in order (the
reference run's first). Each side yields its Decisions: every message sent,
the target reference's build and run, each attempt's extracted block and
whether it compiled and ran, the final correction count, how the trial
ended, Sim-T and Sim-L of the last block against the reference (read as text
mode reads it), the stdout that stands as the output, and the fence-quirk
hits.

The notebook's side (run_notebook) takes the function definitions it needs
from the code cells of the pinned notebook, read with `git show <pin>:<path>`
into a temporary directory, and compiles them with the PromptDictionary class
of the pinned prompt_dictionary.py into a fresh namespace per case. Stubbed
there: the LLM call (its Ollama helper answers from the recording), the
unload helper (recorded, never run), compile and execute (the namespace's
`subprocess` is NotebookProcesses, which follows the script, writes a
placeholder program for a compile that succeeds, and starts nothing, so
upstream's own compile_code and execute_code build its results and reports),
and the tokenizer import (a one-token-per-character stand-in, never compared).
Its correction count, similarity values, last block, and output are the
pipeline's own locals, read when its frame returns or raises. It runs in its
own working directory, as the notebook does, with relative paths.

Our side (run_ours) runs `lassi run`'s run_recipe on a faithful: true recipe
listing baseline, summarize_context, describe_source, generate,
compile_loop, and run_loop, with the real stages, the real replay backend
(a subclass that also records each request), fake toolchains that follow the
script and keep the scripted stderr as the raw compile.stderr attachment, and
a scripted executor that runs programs in the sandbox's name only. The
runner's git query for provenance would start a process, so it is answered
here; provenance is not compared.

Both sides run inside UpstreamGuard: while it is active every way to start a
process or open a socket is refused and recorded, and the test fails on exit
if either side tried one. The guard mirrors tests/scoring/conftest.py (P1.8);
a test checks that the two GUARDED tables agree.

No upstream text is copied into this file or the fixtures (OQ-018); upstream
code runs only in memory during the test. Nothing here is a measurement.
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import difflib
import importlib
import io
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
import tokenize
import types
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.bench import Direction, load_suite
from lassi.core import runner as runner_module
from lassi.core.interfaces import BuildResult, Completion, Limits, Message, RunResult, Sampling
from lassi.core.record import Diagnostic, Trial
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.store import TextStore, read_trial
from lassi.llm.replay import ReplayBackend
from lassi.scoring import sim_l, sim_t

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "replay"
SCENARIOS = FIXTURES / "scenarios.json"
COMPILER_FIXTURES = REPO / "tests" / "toolchains" / "fixtures"
PIN_MANIFEST = REPO / "assets" / "upstream" / "lassi.yaml"
SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
NOTEBOOK = "LASSI_pipeline_v0.ipynb"
DICTIONARY_FILE = "prompt_dictionary.py"
FETCH_HINT = "run `uv run tools/fetch_upstream.py` to fetch upstream LASSI at the pin into third_party/LASSI"

# Upstream names the harness reads (identifiers only; no upstream text).
PIPELINE = "auto_code_llmgeneration_pipeline"
DICTIONARY_CLASS = "PromptDictionary"
GENERATOR = "code_generator_llm"
LLM_CALL = "ollama_llm"
UNLOAD_CALL = "unload_ollama_model"
# The notebook functions taken from its cells; the LLM call and the unload are stubbed instead.
TAKEN = (
    "code_knowledge_llm", "code_description_llm", GENERATOR, "compile_code", "execute_code",
    "token_count", "tokenize_by_tiktoken", "tokenize_by_tokenize", "tokenize_code", "token_similarity",
    "find_line_in_lines", "compare_lines_with_reordering", PIPELINE,
)
# The pipeline's locals the harness reads when its frame exits.
COUNTER = "self_correction_counter"
SIM_T_LOCAL = "similarity_tokens"
SIM_L_LOCAL = "similarity_lines"
OUTPUT_LOCAL = "exe_output"
# The notebook's spelling of each language, and the file extension each target is saved with.
NOTEBOOK_LANGUAGE = {"omp": "OMP", "cuda": "CUDA"}
EXTENSION = {"omp": "cpp", "cuda": "cu"}
# The notebook's LLM source whose helpers the stubs replace, and a context size it passes along unused.
LLM_SOURCE = "ollama"
CONTEXT_SIZE = 16384

MODEL_ID = "replay-fixture"
FRAGMENT_SET = "lassi-2024"
PACKS = ("openmp-4.0-card", "cuda-12.5-ch5")
TOOLCHAINS = {"cuda": "nvcc-sm80", "omp": "nvcpp-cc80"}
STAGES = ("baseline", "summarize_context", "describe_source", "generate", "compile_loop", "run_loop")
DIRECTIONS = (Direction("cuda", "omp"), Direction("omp", "cuda"))
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
COMPILED_STAGES = ("S4", "S5")
FENCE_QUIRK = "fence-quirk"
STALE_OUTPUT = "stale-output"
COMPLETE = "complete"
UPSTREAM_CRASH = "upstream-crash"
BASELINE_COMPILE = "baseline-compile"
BASELINE_RUN = "baseline-run"
# The status a failed scripted compile exits with, the shell's signal base, and a SYNTHETIC wall time.
COMPILE_FAILURE_STATUS = 2
SHELL_SIGNAL_BASE = 128
SYNTHETIC_WALL_S = 0.25
SYNTHETIC_ERROR = "SYNTHETIC compile error of the replay's scripted toolchain"
PLACEHOLDER_PROGRAM = b"PLACEHOLDER program written by the replay's scripted compile; never run\n"

# ---------------------------------------------------------------------------
# The guard (mirrors tests/scoring/conftest.py, task P1.8)

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
GUARD_FAILURE = "tried to start a process or open a socket"


class GuardRefusal(RuntimeError):
    """Raised in place of a process start or socket open while the guard is active."""


class UpstreamGuard:
    """Context manager: refuse and record every process start and socket open, then fail if any was tried."""

    def __init__(self, side: str = "the replay") -> None:
        """Name the side the guard watches, for the failure message."""
        self.side = side
        self.attempts: list[str] = []
        self._patch: pytest.MonkeyPatch | None = None

    def _refuser(self, name: str) -> Callable[..., Any]:
        """Return a stand-in for `name` that records the attempt and raises GuardRefusal."""

        def refuse(*args: Any, **kwargs: Any) -> Any:
            self.attempts.append(name)
            raise GuardRefusal(f"replay guard: {name} refused")

        return refuse

    def __enter__(self) -> UpstreamGuard:
        """Replace every guarded entry point."""
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
        """Restore every entry point, then fail the test when anything was tried."""
        assert self._patch is not None
        self._patch.undo()
        if self.attempts:
            pytest.fail(f"{self.side} {GUARD_FAILURE}: {self.attempts}")


class RefusedModule(types.ModuleType):
    """Stand-in for a module the notebook imports but the replay never uses."""

    def __getattr__(self, name: str) -> Any:
        """Refuse every attribute."""
        raise GuardRefusal(f"replay guard: {self.__name__}.{name} is stubbed and must not be used")


# ---------------------------------------------------------------------------
# Scenarios


class ScriptOverrun(BaseException):
    """A side asked for more compiles or runs than its scenario scripts.

    A BaseException, so the notebook's `except Exception` around its program
    run cannot swallow it.
    """


@dataclass(frozen=True)
class CompileOutcome:
    """One scripted compile: whether it builds a program, and the raw stderr it prints."""

    ok: bool
    stderr: bytes

    def text(self) -> str:
        """Return the stderr as a pipe read in text mode gives it (the notebook's subprocess.run(text=True))."""
        return text_mode(self.stderr.decode("utf-8"))


@dataclass(frozen=True)
class RunOutcome:
    """One scripted program run: its exit status or signal, and its stdout and stderr."""

    stdout: str
    stderr: str
    exit: int | None = None
    signal: int | None = None

    @property
    def popen_code(self) -> int:
        """Return the return code Popen gives: -N for a death by signal N."""
        return -self.signal if self.signal is not None else int(self.exit or 0)

    @property
    def shell_code(self) -> int:
        """Return the exit status an executor that runs programs reports: 128 + N for a death by signal N."""
        return SHELL_SIGNAL_BASE + self.signal if self.signal is not None else int(self.exit or 0)


@dataclass(frozen=True)
class Expected:
    """The decisions a scenario's script leads to, derived by hand in scenarios.json."""

    attempts: int
    compiled: tuple[int, ...]
    ran: tuple[int, ...]
    corrections: int | None
    end: str
    fence_quirk: int
    output_from: int | None
    stale_output: bool


@dataclass(frozen=True)
class Case:
    """One scenario in one direction: its programs, recording, script, and expected decisions."""

    scenario: str
    what: str
    direction: Direction
    item: str
    source: str
    reference: str
    recording: Path
    compiles: tuple[CompileOutcome, ...]
    runs: tuple[RunOutcome, ...]
    expected: Expected

    @property
    def id(self) -> str:
        """Return the case id: the scenario name and the direction name."""
        return f"{self.scenario}-{self.direction.name}"

    def script(self) -> Script:
        """Return a fresh script of this case's outcomes for one side."""
        return Script(list(self.compiles), list(self.runs))


@dataclass
class Script:
    """The scripted outcomes one side consumes in order; asking past the end raises ScriptOverrun."""

    compiles: list[CompileOutcome]
    runs: list[RunOutcome]

    def next_compile(self) -> CompileOutcome:
        """Return the next scripted compile outcome."""
        if not self.compiles:
            raise ScriptOverrun("a side asked for a compile past the end of the scenario's script")
        return self.compiles.pop(0)

    def next_run(self) -> RunOutcome:
        """Return the next scripted run outcome."""
        if not self.runs:
            raise ScriptOverrun("a side asked for a program run past the end of the scenario's script")
        return self.runs.pop(0)


def load_manifest(path: Path = SCENARIOS) -> dict[str, Any]:
    """Read scenarios.json, which must be ASCII JSON labeled synthetic."""
    raw = path.read_bytes()
    assert raw.isascii(), f"{path} is not plain ASCII"
    data = json.loads(raw.decode("ascii"))
    assert data.get("synthetic") is True, f'{path} must be labeled "synthetic": true'
    return data


def load_cases(path: Path = SCENARIOS) -> list[Case]:
    """Return every scenario of `path` in both directions, in file order."""
    data = load_manifest(path)
    kinds = data["compile_kinds"]
    runs = {name: RunOutcome(**spec) for name, spec in data["runs"].items()}
    cases = []
    for scenario in data["scenarios"]:
        program = data["programs"][scenario["program"]]
        expect = scenario["expect"]
        expected = Expected(
            attempts=expect["attempts"],
            compiled=tuple(expect["compiled"]),
            ran=tuple(expect["ran"]),
            corrections=expect["corrections"],
            end=expect["end"],
            fence_quirk=expect["fence_quirk"],
            output_from=expect["output_from"],
            stale_output=expect["stale_output"],
        )
        for direction in DIRECTIONS:
            compiles = tuple(compile_outcome(kinds[kind], direction.target) for kind in scenario["compiles"])
            cases.append(Case(
                scenario=scenario["name"],
                what=scenario["what"],
                direction=direction,
                item=data["item"],
                source=program[direction.source],
                reference=program[direction.target],
                recording=FIXTURES / scenario["recording"],
                compiles=compiles,
                runs=tuple(runs[name] for name in scenario["runs"]),
                expected=expected,
            ))
    return cases


def compile_outcome(kind: Mapping[str, Any], target: str) -> CompileOutcome:
    """Return one scripted compile outcome for the target language: its success flag and the captured stderr file."""
    return CompileOutcome(ok=kind["ok"], stderr=(COMPILER_FIXTURES / kind["stderr"][target]).read_bytes())


def text_mode(text: str) -> str:
    """Return `text` as a file opened in text mode reads it: each CRLF and each lone CR becomes LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def read_text_mode(path: Path) -> str:
    """Read a file as the notebook reads its sources: text mode, universal newlines (UTF-8 here)."""
    with open(path, encoding="utf-8") as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# Events and decisions


@dataclass(frozen=True)
class Compiled:
    """One compile a side asked for: the source as text mode reads it, and whether it built."""

    text: str
    ok: bool


@dataclass(frozen=True)
class Ran:
    """One program run a side asked for: its arguments, its return code as Popen gives it, and its stdout."""

    args: tuple[str, ...]
    code: int
    stdout: str


@dataclass
class SideLog:
    """What one side did, in order: model requests, compiles and runs, extracted blocks before the tag cut."""

    messages: list[tuple[tuple[str, str], ...]] = field(default_factory=list)
    events: list[Compiled | Ran] = field(default_factory=list)
    before_cut: list[str] = field(default_factory=list)
    unloads: int = 0


@dataclass(frozen=True)
class AttemptDecision:
    """One attempt: its extracted block, whether it compiled, and its run (None when it did not run)."""

    block: str
    compiled: bool
    run: Ran | None


@dataclass
class Decisions:
    """Everything one side decided in one case (see the module docstring).

    `stale_output` says the last attempt compiled and did not run while an
    earlier run's stdout stands as the output: ours reads it from its
    stale-output warning, the notebook's side from that state of its loop,
    which has no warning. `problems` lists where our record and our fakes
    disagree (always empty on the notebook's side).
    """

    messages: list[tuple[tuple[str, str], ...]]
    baseline: tuple[bool, int | None]
    attempts: list[AttemptDecision]
    corrections: int | None
    end: str
    sim_t: float | None
    sim_l: float | None
    output: str | None
    fence_quirk: int
    stale_output: bool = False
    problems: list[str] = field(default_factory=list)

    def codes(self) -> str:
        """Return one letter per attempt: x compile error, c compiled and not run, r run error, R clean run."""
        letters = []
        for attempt in self.attempts:
            if not attempt.compiled:
                letters.append("x")
            elif attempt.run is None:
                letters.append("c")
            else:
                letters.append("R" if attempt.run.code == 0 else "r")
        return " ".join(letters) or "-"


def split_events(events: Sequence[Compiled | Ran]) -> tuple[tuple[bool, int | None], list[AttemptDecision]]:
    """Return the reference's (built, run code) and one AttemptDecision per later compile, from a side's events."""
    assert events and isinstance(events[0], Compiled), f"the first event must be the reference's compile: {events}"
    reference_run: int | None = None
    rest = list(events[1:])
    if rest and isinstance(rest[0], Ran):
        reference_run = rest.pop(0).code
    attempts: list[AttemptDecision] = []
    for event in rest:
        if isinstance(event, Compiled):
            attempts.append(AttemptDecision(block=event.text, compiled=event.ok, run=None))
        else:
            assert attempts and attempts[-1].compiled and attempts[-1].run is None, f"a run with no build: {events}"
            attempts[-1] = dataclasses.replace(attempts[-1], run=event)
    return (events[0].ok, reference_run), attempts


def is_quirk(before: str, after: str) -> bool:
    """Return True when a tag cut removed text and left text on the block's first line (a fence-quirk hit)."""
    assert before.endswith(after), "the tag cut must only remove a prefix of the extracted block"
    removed = before[: len(before) - len(after)]
    return bool(removed) and bool(after.split("\n", 1)[0].strip())


# ---------------------------------------------------------------------------
# The notebook's side


@dataclass(frozen=True)
class Notebook:
    """The pinned notebook's taken functions and the prompt dictionary, compiled and ready to run."""

    functions: types.CodeType
    dictionary: types.CodeType
    pipeline_parameters: tuple[str, ...]


def pinned_bytes(checkout: Path, commit: str, path: str) -> bytes:
    """Return the bytes of `path` at `commit` in the upstream checkout (git show: line-ending settings never matter)."""
    done = subprocess.run(["git", "-C", str(checkout), "show", f"{commit}:{path}"], capture_output=True, check=True)
    return done.stdout


def taken_definitions(notebook: Mapping[str, Any]) -> ast.Module:
    """Return the notebook's definitions of every TAKEN function, each from the one code cell that defines it."""
    found: dict[str, ast.FunctionDef] = {}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        try:
            tree = ast.parse("".join(cell.get("source", [])))
        except SyntaxError:
            continue  # cells with notebook magics are not Python; no taken function lives in one
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in TAKEN:
                assert node.name not in found, f"the notebook defines {node.name} twice"
                found[node.name] = node
    missing = [name for name in TAKEN if name not in found]
    assert not missing, f"the pinned notebook defines none of {missing}"
    return ast.Module(body=[found[name] for name in TAKEN], type_ignores=[])


def load_notebook(checkout: Path, commit: str, workdir: Path) -> Notebook:
    """Copy the pinned notebook and prompt dictionary into `workdir` and compile what the replay runs from them."""
    workdir.mkdir(parents=True, exist_ok=True)
    for name in (NOTEBOOK, DICTIONARY_FILE):
        (workdir / name).write_bytes(pinned_bytes(checkout, commit, name))
    cells = json.loads((workdir / NOTEBOOK).read_bytes().decode("utf-8"))
    functions = taken_definitions(cells)
    pipeline = next(node for node in functions.body if node.name == PIPELINE)
    parameters = tuple(argument.arg for argument in pipeline.args.args)
    tree = ast.parse((workdir / DICTIONARY_FILE).read_bytes().decode("utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == DICTIONARY_CLASS]
    assert len(classes) == 1, f"expected one {DICTIONARY_CLASS} class in {DICTIONARY_FILE}"
    return Notebook(
        functions=compile(functions, f"{(workdir / NOTEBOOK).as_posix()}:taken", "exec"),
        dictionary=compile(ast.Module(body=classes, type_ignores=[]), (workdir / DICTIONARY_FILE).as_posix(), "exec"),
        pipeline_parameters=parameters,
    )


class _FakeProcess:
    """A finished program as Popen would hand it over: its stdout and stderr to read, and its return code."""

    def __init__(self, outcome: RunOutcome) -> None:
        """Keep the scripted outcome's streams and Popen's return code."""
        self.stdout = io.StringIO(outcome.stdout)
        self.stderr = io.StringIO(outcome.stderr)
        self._code = outcome.popen_code

    def wait(self) -> int:
        """Return the scripted return code."""
        return self._code


class NotebookProcesses:
    """The `subprocess` of the notebook's namespace: its compiles and runs follow the script; nothing starts."""

    PIPE = subprocess.PIPE

    def __init__(self, script: Script, log: SideLog) -> None:
        """Follow `script` and record into `log`."""
        self.script = script
        self.log = log

    def run(self, argv: Sequence[str], **kwargs: Any) -> Any:
        """Answer a compile (any argv with -o) from the script; any other call (a permission change) is a no-op."""
        argv = list(argv)
        if "-o" not in argv:
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        at = argv.index("-o")
        source, output = Path(argv[at - 1]), Path(argv[at + 1])
        outcome = self.script.next_compile()
        self.log.events.append(Compiled(text=read_text_mode(source), ok=outcome.ok))
        if outcome.ok:
            output.write_bytes(PLACEHOLDER_PROGRAM)
        status = 0 if outcome.ok else COMPILE_FAILURE_STATUS
        return types.SimpleNamespace(returncode=status, stdout="", stderr=outcome.text())

    def Popen(self, argv: Sequence[str], **kwargs: Any) -> _FakeProcess:  # the name the notebook calls
        """Answer a program run from the script with a finished process."""
        outcome = self.script.next_run()
        self.log.events.append(Ran(args=tuple(argv[1:]), code=outcome.popen_code, stdout=outcome.stdout))
        return _FakeProcess(outcome)


class _CharacterEncoding:
    """Tokenizer stand-in: one token per character. Its counts are never compared or reported."""

    _pat_str = ""
    _mergeable_ranks: dict[bytes, int] = {}
    _special_tokens: dict[str, int] = {}

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        """Return one token per character."""
        return [ord(char) for char in text]

    def decode(self, tokens: Sequence[int]) -> str:
        """Return the characters back."""
        return "".join(chr(token) for token in tokens)


CHARACTER_TOKENIZER = types.SimpleNamespace(
    get_encoding=lambda name: _CharacterEncoding(),
    Encoding=lambda **kwargs: _CharacterEncoding(),
)


class _NoColor:
    """The notebook's text colors, all empty (they only decorate prints)."""

    def __getattr__(self, name: str) -> str:
        """Return an empty string for every color name."""
        return ""


def _refused(name: str) -> Callable[..., Any]:
    """Return a callable that fails when the notebook calls `name`, which the replay never needs."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"the notebook called {name}, which the replay does not provide")

    return refuse


def notebook_namespace(notebook: Notebook, log: SideLog, script: Script, recording: Path) -> dict[str, Any]:
    """Return a fresh namespace holding the notebook's taken functions, its prompt dictionary, and the stubs."""
    replay = ReplayBackend(MODEL_ID, recording=recording)

    def llm(model: str, num_ctx: int, system_prompt: str, content_prompt: str) -> str:
        messages = (Message("system", system_prompt), Message("user", content_prompt))
        log.messages.append(tuple((message.role, message.content) for message in messages))
        return replay.complete(messages, SAMPLING).text

    def unload(model: str) -> None:
        log.unloads += 1

    namespace: dict[str, Any] = {
        "__builtins__": builtins, "__name__": "lassi_notebook_replay",
        "print": lambda *args, **kwargs: None, "display": _refused("display"), "Markdown": _refused("Markdown"),
        "color": _NoColor(), "re": re, "os": os, "time": time, "datetime": datetime, "json": json,
        "shlex": shlex, "difflib": difflib, "tokenize": tokenize, "BytesIO": io.BytesIO,
        "subprocess": NotebookProcesses(script, log), "tiktoken": CHARACTER_TOKENIZER,
        "requests": RefusedModule("requests"), "ollama": RefusedModule("ollama"), "OpenAI": _refused("OpenAI"),
    }
    exec(notebook.dictionary, namespace)
    exec(notebook.functions, namespace)
    namespace["prompts"] = namespace[DICTIONARY_CLASS]()
    namespace[LLM_CALL] = llm
    namespace[UNLOAD_CALL] = unload
    generator = namespace[GENERATOR]

    def recording_generator(*args: Any, **kwargs: Any) -> tuple[str, str]:
        response, extracted = generator(*args, **kwargs)
        log.before_cut.append(extracted)
        return response, extracted

    namespace[GENERATOR] = recording_generator
    return namespace


def call_keeping_locals(function: Callable[..., Any], arguments: Mapping[str, Any]) -> tuple[Any, Exception | None,
                                                                                           dict[str, Any]]:
    """Call `function`; return its result, the Exception it raised (or None), and its locals when its frame exited."""
    target = function.__code__
    kept: dict[str, Any] = {}

    def profiler(frame: types.FrameType, event: str, arg: Any) -> None:
        if event == "return" and frame.f_code is target:
            kept.update(frame.f_locals)

    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        return function(**arguments), None, kept
    except Exception as error:  # the notebook's own failures are decisions to compare
        return None, error, kept
    finally:
        sys.setprofile(previous)


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    """Run the block with `path` as the working directory, as the notebook runs in its own directory."""
    before = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(before)


def pipeline_arguments(case: Case, prompts: Any, fragments: Mapping[str, str], paths: Mapping[str, str]) -> dict:
    """Return the pipeline's arguments for `case`, as the notebook's driver cell builds them (by parameter name)."""
    source, target = case.direction.source, case.direction.target
    key = f"{NOTEBOOK_LANGUAGE[source]}_to_{NOTEBOOK_LANGUAGE[target]}"
    item = load_suite(SUITE_MANIFEST).items[case.item]
    return {
        "this_llm_systemcode_prompt": prompts.get_system_prompt(key),
        "this_llm_prompt": prompts.get_codetranslate_prompt(key),
        "llm_source": LLM_SOURCE,
        "model": MODEL_ID,
        "num_ctx": CONTEXT_SIZE,
        "use_context": True,
        "input_context": prompts.get_codeknowledge(target),
        "input_code": read_text_mode(Path(paths["source"])),
        "target_code": read_text_mode(Path(paths["reference"])),
        "translated_code_extension": EXTENSION[target],
        "target_lang": NOTEBOOK_LANGUAGE[target],
        "code_compiler": fragments[f"setup.{target}.compiler"],
        "code_compiler_kwds": fragments[f"setup.{target}.flags"],
        "directory": paths["directory"],
        "file_name": paths["file_name"],
        "target_file_path": paths["reference"],
        "display_response": False,
        "test_with_error": False,
        "generate_test_case": False,
        "yes_execute": True,
        "exe_input_parameters": list(item.run_args),
        "meta_data_file": "",
    }


def write_notebook_inputs(case: Case, root: Path) -> dict[str, str]:
    """Write the case's source and reference under `root`; return the relative paths the pipeline gets."""
    source, target = case.direction.source, case.direction.target
    paths = {
        "source": f"source/{case.item}-{source}_main.{EXTENSION[source]}",
        "reference": f"reference/{case.item}-{target}_main.{EXTENSION[target]}",
        "directory": "generated",
        "file_name": f"{case.item}-{source}_main_replay_{NOTEBOOK_LANGUAGE[target]}",
    }
    for role, text in (("source", case.source), ("reference", case.reference)):
        path = root / paths[role]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return paths


def run_notebook(
    case: Case, notebook: Notebook, fragments: Mapping[str, str], root: Path, *, processes: Any = None
) -> Decisions:
    """Run the notebook's pipeline for `case` in `root` under the guard and return its Decisions.

    `processes` replaces the namespace's scripted `subprocess` (a guard test
    hands it the real module).
    """
    root.mkdir(parents=True, exist_ok=True)
    log, script = SideLog(), case.script()
    paths = write_notebook_inputs(case, root)
    with working_directory(root), UpstreamGuard("the notebook's side"):
        namespace = notebook_namespace(notebook, log, script, case.recording)
        if processes is not None:
            namespace["subprocess"] = processes
        arguments = pipeline_arguments(case, namespace["prompts"], fragments, paths)
        assert set(arguments) == set(notebook.pipeline_parameters), "the harness must pass every pipeline parameter"
        result, error, kept = call_keeping_locals(namespace[PIPELINE], arguments)
    return notebook_decisions(case, log, result, error, kept, namespace[PIPELINE].__code__)


def notebook_end(
    result: Any, error: Exception | None, pipeline: types.CodeType, baseline: tuple[bool, int | None]
) -> str:
    """Return how the pipeline ended: complete, a baseline failure, or the crash past the execution gate."""
    if error is not None:
        frame = error.__traceback__
        while frame is not None and frame.tb_next is not None:
            frame = frame.tb_next
        if isinstance(error, UnboundLocalError) and frame is not None and frame.tb_frame.f_code is pipeline:
            return UPSTREAM_CRASH
        raise error
    if result is True:
        return COMPLETE
    assert result is False, f"the pipeline returned {result!r}"
    return BASELINE_COMPILE if not baseline[0] else BASELINE_RUN


def notebook_decisions(
    case: Case, log: SideLog, result: Any, error: Exception | None, kept: Mapping[str, Any], pipeline: types.CodeType
) -> Decisions:
    """Return the notebook side's Decisions from its log, its result, and its pipeline locals."""
    baseline, attempts = split_events(log.events)
    end = notebook_end(result, error, pipeline, baseline)
    assert len(log.before_cut) == len(attempts), "each generator call must be followed by one compile"
    quirks = sum(is_quirk(before, attempt.block) for before, attempt in zip(log.before_cut, attempts, strict=True))
    output = kept.get(OUTPUT_LOCAL)
    return Decisions(
        messages=list(log.messages),
        baseline=baseline,
        attempts=attempts,
        corrections=kept.get(COUNTER) if attempts else None,
        end=end,
        sim_t=kept.get(SIM_T_LOCAL),
        sim_l=kept.get(SIM_L_LOCAL),
        output=output,
        fence_quirk=quirks,
        stale_output=bool(attempts) and attempts[-1].compiled and attempts[-1].run is None and output is not None,
    )


# ---------------------------------------------------------------------------
# Our side


def replay_class(recording: Path, log: SideLog) -> type:
    """Return the replay backend bound to `recording`, recording each request into `log` before it answers."""

    class ScenarioReplay(ReplayBackend):
        """The real replay backend, built by the runner as factory(model_id), with the case's recording."""

        def __init__(self, model_id: str) -> None:
            """Read the case's recording."""
            super().__init__(model_id, recording=recording)

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Record the request, then answer it as the replay backend does."""
            log.messages.append(tuple((message.role, message.content) for message in messages))
            return super().complete(messages, sampling)

    return ScenarioReplay


def toolchain_class(registered_as: str, script: Script, log: SideLog, on_build: Callable[[], None] | None) -> type:
    """Return a Toolchain class without PIN that follows the script and keeps the raw stderr attachment."""

    class ScriptedToolchain:
        """Writes the files and the scripted stderr as compile.stderr; builds a placeholder program on success."""

        name = registered_as
        capabilities = frozenset({"diagnostics"})

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Follow the next scripted compile outcome."""
            if on_build is not None:
                on_build()
            workdir = Path(workdir)
            assert len(files) == 1, f"the replay's items have one file per language, got {sorted(files)}"
            for path, text in [*files.items(), *(harness or {}).items()]:
                (workdir / path).parent.mkdir(parents=True, exist_ok=True)
                (workdir / path).write_bytes(text.encode("utf-8"))
            outcome = script.next_compile()
            log.events.append(Compiled(text=text_mode(next(iter(files.values()))), ok=outcome.ok))
            (workdir / "compile.stderr").write_bytes(outcome.stderr)
            if not outcome.ok:
                error = Diagnostic(stage="compile", severity="error", code="scripted", message=SYNTHETIC_ERROR)
                return BuildResult(artifact=None, diagnostics=[error], stderr_ref="compile.stderr")
            artifact = workdir / "main"
            artifact.write_bytes(PLACEHOLDER_PROGRAM)
            return BuildResult(artifact=artifact, diagnostics=[], stderr_ref="compile.stderr")

    return ScriptedToolchain


def executor_class(script: Script, log: SideLog) -> type:
    """Return an Executor class that runs programs in name only: every run follows the script."""

    class ScriptedExecutor:
        """Declares runs_code and sandboxed; returns scripted results and runs nothing."""

        name = "scripted"
        capabilities = frozenset({"runs_code", "sandboxed"})

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Return the next scripted run outcome, a death by signal N in the shell's form (128 + N)."""
            outcome = script.next_run()
            log.events.append(Ran(args=tuple(inputs), code=outcome.popen_code, stdout=outcome.stdout))
            return RunResult(
                exit_code=outcome.shell_code, hang=False, stdout=outcome.stdout, stderr=outcome.stderr,
                wall_s=SYNTHETIC_WALL_S,
            )

    return ScriptedExecutor


def our_registry(case: Case, script: Script, log: SideLog, on_build: Callable[[], None] | None) -> Registry:
    """Return the registry our side runs with: the real stages and replay backend, scripted toolchains and executor."""
    registry = Registry()
    registry.register("LLMBackend", "replay", replay_class(case.recording, log))
    registry.register("Executor", "scripted", executor_class(script, log))
    for name in TOOLCHAINS.values():
        registry.register("Toolchain", name, toolchain_class(name, script, log, on_build))
    for stage in STAGES:
        registry.register("Stage", stage, DEFAULT_REGISTRY.get("Stage", stage).factory)
    return registry


def recipe_data(case: Case) -> dict[str, Any]:
    """Return the faithful recipe of one case: every fix off, uncapped, the lassi-2024 prompts and both packs."""
    return {
        "extends": "base",
        "faithful": True,
        "model": {"backend": "replay", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": SAMPLING.max_tokens}},
        "bench": {"suite": SUITE, "split": "eval", "items": [case.item]},
        "directions": [{"source": case.direction.source, "target": case.direction.target}],
        "prompts": FRAGMENT_SET,
        "context": list(PACKS),
        "toolchain": dict(TOOLCHAINS),
        "stages": list(STAGES),
        "executor": {"kind": "scripted"},
        "trials": {"n": 1},
    }


def write_bench(case: Case, root: Path) -> Path:
    """Write the case's programs where the suite manifest lays out the item; return the bench root."""
    spec = load_suite(SUITE_MANIFEST).items[case.item]
    texts = {case.direction.source: case.source, case.direction.target: case.reference}
    for language, text in texts.items():
        layout = spec.languages[language]
        path = root / layout.dir / layout.files[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def run_ours(case: Case, root: Path, *, on_build: Callable[[], None] | None = None) -> Decisions:
    """Run our faithful stages for `case` through run_recipe in `root` under the guard; return their Decisions.

    `on_build` runs at the start of every scripted build (a guard test uses it).
    """
    root.mkdir(parents=True, exist_ok=True)
    log, script = SideLog(), case.script()
    registry = our_registry(case, script, log, on_build)
    bench = write_bench(case, root / "bench")
    recipe = root / "replay.yaml"
    recipe.write_bytes(yaml.safe_dump(recipe_data(case), sort_keys=False).encode("ascii"))
    options = RunOptions(runs_root=root / "runs-root", run_id="replay", bench_root=bench, registry=registry)
    warm_platform_cache()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner_module, "_git_state", lambda: (None, None))
        with UpstreamGuard("our side"):
            run_dir = run_recipe(recipe, options)
    found = sorted(run_dir.rglob("trial.json"))
    assert len(found) == 1, f"expected one trial under {run_dir}, found {len(found)}"
    store = TextStore(run_dir)
    return our_decisions(case, read_trial(found[0].parent, store), store, log)


def warm_platform_cache() -> None:
    """Ask for the host description once before the guard: on some hosts the first call starts a process."""
    platform.platform()
    platform.node()


def our_decisions(case: Case, trial: Trial, store: TextStore, log: SideLog) -> Decisions:
    """Return our side's Decisions from its fakes' log and its trial record, noting where the two disagree."""
    baseline, attempts = split_events(log.events)
    problems = record_problems(case, trial, attempts)
    ran = [attempt for attempt in trial.attempts if attempt.run.stdout_ref is not None]
    last = trial.attempts[-1] if trial.attempts else None
    reference = text_mode(case.reference)
    final_block = last.files.get(target_file(case), "") if last is not None else None
    return Decisions(
        messages=list(log.messages),
        baseline=baseline,
        attempts=attempts,
        corrections=trial.final.corrections if trial.attempts else None,
        end=trial.final.end_reason.code if trial.final.end_reason is not None else COMPLETE,
        sim_t=None if final_block is None else sim_t(reference, final_block),
        sim_l=None if final_block is None else sim_l(reference, final_block),
        output=store.get(ran[-1].run.stdout_ref) if ran else None,
        fence_quirk=sum(d.code == FENCE_QUIRK for attempt in trial.attempts for d in attempt.diagnostics),
        stale_output=last is not None and any(d.code == STALE_OUTPUT for d in last.diagnostics),
        problems=problems,
    )


def target_file(case: Case) -> str:
    """Return the item's one target file name."""
    return load_suite(SUITE_MANIFEST).items[case.item].languages[case.direction.target].files[0]


def record_problems(case: Case, trial: Trial, attempts: Sequence[AttemptDecision]) -> list[str]:
    """Return every way the trial record disagrees with what our fakes saw (blocks, builds, runs)."""
    if len(trial.attempts) != len(attempts):
        return [f"the record holds {len(trial.attempts)} attempts, the fakes saw {len(attempts)} builds"]
    problems = []
    for recorded, seen in zip(trial.attempts, attempts, strict=True):
        block = text_mode(recorded.files.get(target_file(case), ""))
        if block != seen.block:
            problems.append(f"attempt {recorded.index}: the recorded file differs from the built one")
        if (recorded.stage_reached in COMPILED_STAGES) != seen.compiled:
            problems.append(f"attempt {recorded.index}: stage {recorded.stage_reached}, built {seen.compiled}")
        if (recorded.run.exit_code is not None) != (seen.run is not None):
            problems.append(f"attempt {recorded.index}: the record and the executor disagree on whether it ran")
    return problems


# ---------------------------------------------------------------------------
# Comparison and the report


def first_difference(label: str, want: str, got: str) -> str:
    """Describe where two texts first differ, with a short window of each around that point."""
    shared = min(len(want), len(got))
    at = next((i for i in range(shared) if want[i] != got[i]), shared)
    return (
        f"{label} differs at character {at} (notebook length {len(want)}, ours {len(got)}): "
        f"notebook {want[max(at - 30, 0):at + 30]!r}, ours {got[max(at - 30, 0):at + 30]!r}"
    )


def message_differences(notebook: Sequence[tuple[tuple[str, str], ...]], ours: Sequence[tuple[tuple[str, str], ...]]
                        ) -> list[str]:
    """Return how the two sides' model requests differ: count, roles, or the first differing character."""
    problems = []
    if len(notebook) != len(ours):
        problems.append(f"model calls: the notebook made {len(notebook)}, ours {len(ours)}")
    for call, (want, got) in enumerate(zip(notebook, ours, strict=False), start=1):
        if [role for role, _ in want] != [role for role, _ in got]:
            problems.append(f"call {call}: roles {[r for r, _ in want]} in the notebook, {[r for r, _ in got]} ours")
            continue
        for (role, want_text), (_, got_text) in zip(want, got, strict=True):
            if want_text != got_text:
                problems.append(first_difference(f"call {call} {role} message", want_text, got_text))
    return problems


def attempt_differences(notebook: Sequence[AttemptDecision], ours: Sequence[AttemptDecision]) -> list[str]:
    """Return how the two sides' attempts differ: count, extracted block, compiled, ran."""
    problems = []
    if len(notebook) != len(ours):
        problems.append(f"attempts: the notebook made {len(notebook)}, ours {len(ours)}")
    for index, (want, got) in enumerate(zip(notebook, ours, strict=False)):
        if want.block != got.block:
            problems.append(first_difference(f"attempt {index} extracted block", want.block, got.block))
        if want.compiled != got.compiled:
            problems.append(f"attempt {index}: compiled {want.compiled} in the notebook, {got.compiled} ours")
        if want.run != got.run:
            problems.append(f"attempt {index}: run {want.run} in the notebook, {got.run} ours")
    return problems


def differences(notebook: Decisions, ours: Decisions) -> list[str]:
    """Return every difference between the notebook's decisions and ours, and any record problem of ours."""
    problems = [*ours.problems, *message_differences(notebook.messages, ours.messages)]
    problems += attempt_differences(notebook.attempts, ours.attempts)
    for name in ("baseline", "corrections", "end", "sim_t", "sim_l", "output", "fence_quirk", "stale_output"):
        want, got = getattr(notebook, name), getattr(ours, name)
        if want != got:
            problems.append(f"{name}: {want!r} in the notebook, {got!r} ours")
    return problems


def expectation_problems(expected: Expected, decisions: Decisions) -> list[str]:
    """Return how one side's decisions differ from the scenario's hand-derived expectations."""
    compiled = tuple(i for i, attempt in enumerate(decisions.attempts) if attempt.compiled)
    ran = tuple(i for i, attempt in enumerate(decisions.attempts) if attempt.run is not None)
    got = {
        "attempts": len(decisions.attempts), "compiled": compiled, "ran": ran,
        "corrections": decisions.corrections, "end": decisions.end, "fence_quirk": decisions.fence_quirk,
        "stale_output": decisions.stale_output,
    }
    problems = [
        f"{name}: expected {getattr(expected, name)!r}, got {value!r}"
        for name, value in got.items() if getattr(expected, name) != value
    ]
    if expected.output_from is None:
        if decisions.output is not None:
            problems.append("output: expected none to stand, but a run's stdout does")
    else:
        attempt = decisions.attempts[expected.output_from] if expected.output_from < len(decisions.attempts) else None
        if attempt is None or attempt.run is None or decisions.output != attempt.run.stdout:
            problems.append(f"output: expected attempt {expected.output_from}'s stdout to stand")
    return problems


@dataclass(frozen=True)
class CaseResult:
    """One row of the replay report."""

    scenario: str
    direction: str
    codes: str
    corrections: int | None
    end: str
    sims_equal: bool
    fence_quirk: int
    same: bool


RESULTS: list[CaseResult] = []

REPORT_TITLE = "Notebook replay: a check of decision logic on synthetic fixtures, not a measurement"


def render_report(results: Sequence[CaseResult]) -> str:
    """Return the replay report: a decision table per scenario and the fence-quirk hit count, labeled synthetic."""
    lines = [
        REPORT_TITLE,
        "Replies, compiler outcomes, and program runs are scripted synthetic fixtures (tests/fixtures/replay);",
        "no value below was measured. Attempts: x compile error, c compiled and not run, r run error, R clean run.",
        "",
        "| scenario | direction | attempts | corrections | end | Sim-T, Sim-L | fence-quirk hits | notebook = ours |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in results:
        corrections = "-" if row.corrections is None else str(row.corrections)
        sims = "equal" if row.sims_equal else "DIFFER"
        same = "yes" if row.same else "NO"
        lines.append(
            f"| {row.scenario} | {row.direction} | {row.codes} | {corrections} | {row.end} | {sims} | "
            f"{row.fence_quirk} | {same} |"
        )
    total = sum(row.fence_quirk for row in results)
    lines += [
        "",
        f"Fence-quirk hits on the synthetic fixtures: {total} (a decision-logic check, not the Evaluation Protocol's "
        "fence-quirk replay count and never a measurement)",
    ]
    return "\n".join(lines)
