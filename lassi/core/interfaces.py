"""The twelve component interfaces that carry every project difference.

Each interface is a Protocol; recipes bind concrete components to them by
registry name. Contract rules (bible, Component Interfaces):

- Toolchains return diagnostics parsed into Diagnostic records; raw stderr is
  kept as an attachment and never consumed downstream.
- Executors enforce wall time, memory, and CPU limits and return exit status,
  stdout, stderr, output files, and a hang flag.
- Oracles never trust a program's self-reported PASS as the only signal.
- Stages are pure over the trial record: read fields, append an attempt or
  annotation, return. Side effects go through components.
- Every component declares its capabilities (see `capabilities.Component`).

Trial, Attempt, and Diagnostic are the Result Record types in
`lassi.core.record`; they are referenced here by name only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Protocol, Sequence

from lassi.core.capabilities import Component

if TYPE_CHECKING:
    from lassi.core.record import Attempt, Diagnostic, Trial


@dataclass(frozen=True)
class Message:
    """One chat message sent to a model."""

    role: str
    content: str


@dataclass(frozen=True)
class Sampling:
    """Sampling parameters; recorded in the Trial `model` field."""

    temperature: float
    top_p: float
    max_tokens: int


@dataclass(frozen=True)
class Completion:
    """Model output text with token counts."""

    text: str
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class Module:
    """IR text at a named level, tagged `structured` or `low` for hub modules."""

    level: str
    text: str
    tag: str = ""


@dataclass(frozen=True)
class BuildResult:
    """Toolchain output: the artifact path (None on failure) and diagnostics."""

    artifact: Path | None
    diagnostics: list[Diagnostic]
    stderr_ref: str = ""


@dataclass(frozen=True)
class Limits:
    """Resource limits an executor must enforce."""

    wall_s: float
    memory_mb: int
    cpus: int


@dataclass(frozen=True)
class RunResult:
    """What an executor returns for one run of an artifact.

    `stdout_truncated` and `stderr_truncated` are True when the executor
    kept only part of that stream because it passed the output cap
    (lassi.toolchains._base OUTPUT_CAP_BYTES). `workdir_incomplete` is True
    when the executor returned only part of what the run wrote in its
    workdir, because it passed a cap or limit (lassi.executors.sandbox), so
    output_files may lack files the program wrote.
    """

    exit_code: int | None
    hang: bool
    stdout: str
    stderr: str
    output_files: Mapping[str, Path] = field(default_factory=dict)
    wall_s: float = 0.0
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    workdir_incomplete: bool = False


@dataclass(frozen=True)
class Verdict:
    """A judge's verdict, score, and rationale; always reported as [JUDGED]."""

    verdict: str
    score: float
    rationale: str


@dataclass(frozen=True)
class Score:
    """Named score components, the scalar derived from them, and notes on the components.

    A component or the scalar is None when it was not measured or not
    computed. `notes` maps a component name to a plain ASCII note, such as
    why that component is None or which interpreter computed it; it is empty
    unless the profile writes one.
    """

    components: Mapping[str, float | None]
    scalar: float | None
    notes: Mapping[str, str] = field(default_factory=dict)


class LLMBackend(Component, Protocol):
    """Serves a model: messages plus sampling in, text plus token counts out."""

    def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
        """Return the model's completion for `messages` under `sampling`."""
        ...


class Frontend(Component, Protocol):
    """Raises source code to a hub-level module tagged `structured` or `low`."""

    def raise_source(self, files: Mapping[str, str], flags: Sequence[str]) -> Module:
        """Return the hub-level module for the given source files and flags."""
        ...


class Target(Component, Protocol):
    """Lowers a hub module to a device and builds the artifact."""

    def build(self, module: Module, workdir: Path) -> BuildResult:
        """Lower, emit, and build `module` under `workdir`."""
        ...


class IRLevel(Component, Protocol):
    """Parses, verifies, and normalizes IR at one level."""

    def parse(self, text: str) -> Module:
        """Parse `text` into a module at this level."""
        ...

    def verify(self, module: Module) -> list[Diagnostic]:
        """Return verifier diagnostics for `module`; empty means valid."""
        ...

    def normalize(self, module: Module) -> Module:
        """Return `module` in canonical custom assembly form."""
        ...


class Toolchain(Component, Protocol):
    """Compiles files into an artifact with parsed diagnostics."""

    def build(self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None) -> BuildResult:
        """Write `files` under `workdir`, build them, and parse diagnostics.

        `harness` holds the bench item's support files (relative path ->
        text), written beside `files`; a model file never replaces one. A
        stage passes it only for an item that has support files.
        """
        ...


class Executor(Component, Protocol):
    """Runs an artifact on inputs under enforced limits."""

    def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        """Run `artifact` with `inputs` and return its RunResult."""
        ...


class Oracle(Component, Protocol):
    """Compares a run against the reference and returns alignment in [0, 1]."""

    def align(self, reference: RunResult, candidate: RunResult) -> float:
        """Return how closely `candidate` matches `reference`, in [0, 1]."""
        ...


class Profiler(Component, Protocol):
    """Traces a run: timing, and power where telemetry exists."""

    def start(self) -> None:
        """Begin tracing."""
        ...

    def stop(self) -> Mapping[str, float]:
        """End tracing and return measured values such as `runtime_s`."""
        ...


class Agent(Component, Protocol):
    """A role bound to a backend, tools, and a budget."""

    def act(self, trial: Trial) -> Attempt | Mapping[str, str]:
        """Return a new attempt or an annotation for `trial`."""
        ...


class Judge(Component, Protocol):
    """Scores a candidate against evidence with a rubric."""

    def judge(self, candidate: Attempt, evidence: Mapping[str, str], rubric: str) -> Verdict:
        """Return the verdict, score, and rationale for `candidate`."""
        ...


class ScoreProfile(Component, Protocol):
    """Turns a trial into score components, a scalar, and notes on its null components.

    A built profile may declare `trial_components`, the names of its trial
    Score's components in order; `lassi run` offers them as metrics a
    recipe may name (lassi.scoring.run_scoring). One that declares none
    offers none.
    """

    def score(self, trial: Trial) -> Score:
        """Return the score breakdown for `trial`."""
        ...


class Stage(Component, Protocol):
    """One pure step of the pipeline over the trial record."""

    def __call__(self, trial: Trial) -> Trial:
        """Return `trial` with this stage's attempt or annotation appended."""
        ...

    def describe(self) -> str:
        """Return a one-line description of the stage for run.md."""
        ...
