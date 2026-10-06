"""The thirteen component interfaces that carry every project difference.

Each interface is a Protocol; recipes bind concrete components to them by
registry name. Contract rules (bible, Component Interfaces):

- Toolchains return diagnostics parsed into Diagnostic records; raw stderr is
  kept as an attachment and never consumed downstream.
- Executors enforce wall time, memory, and CPU limits and return exit status,
  stdout, stderr, output files, and a hang flag, with a simulator's findings
  when they report any (RunResult); each names the device its programs run
  on (device()).
- Oracles never trust a program's self-reported PASS as the only signal.
- LLM backends never truncate a request: one that knows its model's context
  raises ContextExceeded, defined here, before generating (task P17.4).
- Stages are pure over the trial record: read fields, append an attempt or
  annotation, return. Side effects go through components.
- A Trainer writes only under the output directory it is given and reads a
  bench item only through TrainData.bench_item, which asks the bench
  registry for purpose train (Agent Rule 5; task P17.8).
- Every component declares its capabilities (see `capabilities.Component`).

Trial, Attempt, Diagnostic, and DeviceRecord are Result Record types in
`lassi.core.record`, FrameworkBuild is in `lassi.core.devices`, and Suite
and SuiteItem are in `lassi.bench.registry`; they are referenced here by
name only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

from lassi.core.capabilities import Component

if TYPE_CHECKING:
    from lassi.bench.registry import Suite, SuiteItem
    from lassi.core.devices import FrameworkBuild
    from lassi.core.record import Attempt, DeviceRecord, Diagnostic, Trial


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

    The last three fields carry a simulator's findings (task P4.6); the
    defaults read as no finding, so an executor that reports none builds
    its RunResult as before. `sim_ub` is True when the simulator reported
    undefined behavior, False when the executor checked and found none, and
    None when it did not check. `sim_gap` names the class of a simulator gap
    (such as "UnimplementedFunctionality"), a program the simulator cannot
    run, and is None otherwise. `diagnostics` holds what the executor parsed
    from the run, kernel JIT messages among them with stage "jit"; the
    stages read these fields, never the executor's name (lassi.core.stages).
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
    sim_ub: bool | None = None
    sim_gap: str | None = None
    diagnostics: list[Diagnostic] = field(default_factory=list)


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


class ContextExceeded(Exception):
    """A model request whose prompt tokens plus max_tokens pass the model's context; nothing was truncated.

    An LLMBackend raises it before generating anything (task P17.4), and
    the stages end the trial at the end reason context-exceeded instead of
    failing the run. It keeps prompt_tokens, max_tokens, and context. It is
    neither a ValueError nor a ServingError, so no handler written for those
    catches it by accident.
    """

    def __init__(self, prompt_tokens: int, max_tokens: int, context: int) -> None:
        """Keep the three numbers; the message names them."""
        super().__init__(
            f"the prompt holds {prompt_tokens} tokens and max_tokens asks for {max_tokens} more, past the model's "
            f"context of {context} tokens; nothing was truncated"
        )
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.context = context


class LLMBackend(Component, Protocol):
    """Serves a model: messages plus sampling in, text plus token counts out."""

    def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
        """Return the model's completion for `messages` under `sampling`.

        A request past the model's context is never truncated: a backend that
        knows its context raises ContextExceeded before generating.
        """
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
    """Runs an artifact on inputs under enforced limits, and names the device its programs run on.

    An executor that runs programs on a pinned install may also declare
    `pins`, pin name -> the pin file's pairs, as a built toolchain's pins
    are held (task P4.11); the runner records their versions in the
    toolchain_pins of the trials it serves and in provenance.json
    (lassi.core.runner). An executor without the attribute declares none.
    """

    def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        """Run `artifact` with `inputs` and return its RunResult."""
        ...

    def device(self) -> str:
        """Return the device the programs run on, starting no process and never using the sandbox (task P4.5).

        The result is one non-empty line of printable ASCII with no leading
        or trailing blank. The runner (lassi.core.runner) asks each bound
        executor once, before the run directory exists, and refuses any
        other result except None, which it records as no device (null). An
        executor without device() is read as naming "none (compile only)"
        when it declares compile_only, and no device (null) otherwise.
        """
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


@dataclass(frozen=True)
class TrainData:
    """The data a Trainer trains on, with its data split hash (task P17.8; lassi.train.data).

    `source` is "synthetic" or "bench". `split_hash` is the sha256 the train
    layer took over the data's identity. `records` holds the synthetic
    records in file order (() for bench). `suite` and `items` hold the bench
    suite and the names of the selected items, each of which already passed
    Suite.item for purpose train (None and () for synthetic data). A Trainer
    reads a bench item only through bench_item (Agent Rule 5); that is its
    contract, since `suite` itself answers any purpose.
    """

    source: str
    split_hash: str
    records: tuple[Mapping[str, Any], ...]
    suite: Suite | None
    items: tuple[str, ...]

    def bench_item(self, name: str) -> SuiteItem:
        """Return bench item `name` through Suite.item for purpose train.

        An eval or unassigned item raises EvalSplitError (Agent Rule 5); a
        train item outside `items`, which the split hash does not cover, and
        synthetic data, which has no suite, raise ValueError.
        """
        if self.suite is None:
            raise ValueError(f"the train data is {self.source}, not bench, so it has no bench item {name!r}")
        found = self.suite.item(name, purpose="train")
        if name not in self.items:
            raise ValueError(f"bench item {name!r} is not among the selected items {list(self.items)}")
        return found


@dataclass(frozen=True)
class TrainJob:
    """What a Trainer is given (task P17.8).

    `recipe` is the resolved train recipe (treat it as read-only) and
    `recipe_yaml` the text of recipe.resolved.yaml; `data` the TrainData;
    `device` the DeviceRecord of trainer.device as probed; `provenance` the
    train directory's provenance.json as first written; and `out_dir` the
    train directory's output/, created empty, the only place the Trainer
    writes.
    """

    recipe: Mapping[str, Any]
    recipe_yaml: str
    data: TrainData
    device: DeviceRecord
    provenance: Mapping[str, Any]
    out_dir: Path


@dataclass(frozen=True)
class TrainResult:
    """What a Trainer returns: the steps it ran and its checkpoint directories.

    `checkpoints` are paths relative to the job's out_dir with "/"
    separators, each an existing directory strictly inside it.
    """

    steps: int
    checkpoints: tuple[str, ...]


class Trainer(Component, Protocol):
    """Trains a model from a train recipe on one device and writes its checkpoints (bible Training Module).

    Declared class attributes, read by the loader and the train layer without
    building anything: `methods`, `weight_modes`, and `data_sources`,
    collections of the names it carries out (the Training Module's methods
    and weight modes; synthetic and bench), and `packages`, the distribution
    names whose installed versions are recorded as its framework pins. It
    declares the capability takes_device and is built as factory(**config)
    from its trainer section, device included; building it writes nothing
    and loads no model.
    """

    @staticmethod
    def framework() -> FrameworkBuild:
        """Return the framework build it trains on, read from build metadata only; called on the class."""
        ...

    def train(self, job: TrainJob) -> TrainResult:
        """Run the steps, writing every output under job.out_dir only; return the steps run and the checkpoints."""
        ...
