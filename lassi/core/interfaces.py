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
- Frontends raise FrontendRefusal, defined here, for a source outside their
  published subset, carrying one parse-stage Diagnostic per construct
  (task P12.4); they return a module only for a source they raise whole.
- Stages are pure over the trial record: read fields, append an attempt or
  annotation, return. Side effects go through components.
- A Trainer writes only under the output directory it is given and reads a
  bench item only through TrainData.bench_item, which reads the suite's
  train-only view (Agent Rule 5; tasks P17.8, P17.9). Before any directory
  exists, its check(recipe, data) refuses every recipe key, value, and
  record it does not carry out (task P17.9).
- A Profiler brackets one attempt run with start() and stop() and returns
  a Profile (task P17.7); power values come only from a profiler that
  declares supports_power.
- An adversary is an Agent that declares proposes_inputs and provides
  propose(view) (lassi.core.capabilities PROPOSES_INPUTS, Adversary; task
  P16.2). Its AdversaryView, defined here, holds the item's input contract,
  the source program and the candidate as text, the budget left, and
  earlier proposals' outcomes, and never a held-out input, the item's
  seed, or a reference output. A Proposal, defined here, is data (named
  arrays), never code, and is checked against the contract before any run.
  An outcome's code is a short token, never free text. The three types
  check their fields' exact types and take no subclass.
- Every component declares its capabilities (see `capabilities.Component`).

Trial, Attempt, Diagnostic, DeviceRecord, and Profile are Result Record types in
`lassi.core.record`, FrameworkBuild is in `lassi.core.devices`, and
SuiteItem and TrainView are in `lassi.bench.registry`; they are referenced
here by name only. InputContract, ContractInput, and LassiArray are
lassi.harness types, which the adversary's records check against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence, final

from lassi.core.capabilities import Component
from lassi.harness.contract import ContractInput, InputContract
from lassi.harness.lassi_io import LassiArray

if TYPE_CHECKING:
    from lassi.bench.registry import SuiteItem, TrainView
    from lassi.core.devices import FrameworkBuild
    from lassi.core.record import Attempt, DeviceRecord, Diagnostic, Profile, Trial


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


# An outcome's code [DESIGN] (task P16.2): a short token, never free text, so no run output, statistic, or seed reaches
# an adversary through it. Every lassi.harness CONTRACT_REASONS entry is one; task P16.5 adds the stage's own codes.
_OUTCOME_CODE = re.compile(r"[a-z][a-z0-9-]{0,63}")
# The fields of a ContractInput other than shape, with the exact types input_contract gives them (lassi.harness).
_CONTRACT_INPUT_TYPES: tuple[tuple[str, tuple[type, ...]], ...] = (
    ("name", (str,)),
    ("dtype", (str,)),
    ("lo", (int, float)),
    ("hi", (int, float)),
    ("allow_nan", (bool,)),
    ("allow_inf", (bool,)),
)


@final
@dataclass(frozen=True)
class Proposal:
    """One input set an adversary proposes: named arrays, data only, never code (task P16.2).

    `arrays` is a tuple of LassiArray (lassi.harness), each a name, a dtype,
    a shape, and the element bytes, as one lassi_io file holds them; a list
    is stored as a tuple. Anything else is refused with a TypeError naming
    `arrays`: program text, bytes, a mapping, a callable, a code object,
    None, a lone array, an instance of a LassiArray subclass, which could
    carry methods, and an array whose name or dtype is not exactly a str or
    whose shape holds anything but ints (a bool, a float, or text). A
    proposal may lie outside the item's contract (a repeated, missing, or
    extra input, another dtype or shape, an element out of range, or no
    array at all): check_inputs refuses such a set before any run and the
    refusal is counted, so a Proposal refuses only what is not data. The
    class takes no subclass, which could add a field.
    """

    arrays: tuple[LassiArray, ...]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Refuse every subclass (see the class docstring)."""
        _refuse_subclass("Proposal", cls)

    def __post_init__(self) -> None:
        """Refuse `arrays` unless it is a tuple or list of LassiArray holding data only; keep it as a tuple."""
        arrays = _exact_items(self.arrays, LassiArray, "Proposal.arrays")
        for index, array in enumerate(arrays):
            _check_array_fields(array, f"Proposal.arrays[{index}]")
        object.__setattr__(self, "arrays", arrays)


@final
@dataclass(frozen=True)
class ProposalOutcome:
    """What became of one earlier proposal, as its adversary may see it: the proposal and a code (task P16.2).

    `code` is a short token, exactly a str of lower-case letters, digits,
    and hyphens that starts with a letter and has at most 64 characters: the
    check_inputs reason (lassi.harness CONTRACT_REASONS) for a proposal
    refused before any run, or the adversary stage's own code for one that
    ran (task P16.5 fixes those). It is never free text, and the outcome
    carries no run result, output file, or statistic, so no reference output
    reaches an adversary through it. A `proposal` that is not exactly a
    Proposal and a `code` that is not exactly a str are refused with a
    TypeError, and a str that is not such a token with a ValueError, the
    message naming the field. The class takes no subclass, which could add
    a field.
    """

    proposal: Proposal
    code: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Refuse every subclass (see the class docstring)."""
        _refuse_subclass("ProposalOutcome", cls)

    def __post_init__(self) -> None:
        """Refuse a proposal that is not a Proposal and a code that is not a short token."""
        if type(self.proposal) is not Proposal:
            raise TypeError(f"ProposalOutcome.proposal must be a Proposal, not {type(self.proposal).__name__}")
        if type(self.code) is not str:
            raise TypeError(f"ProposalOutcome.code must be a str, not {type(self.code).__name__}")
        if _OUTCOME_CODE.fullmatch(self.code) is None:
            raise ValueError(
                f"ProposalOutcome.code must be a short token (lower-case letters, digits, and hyphens, starting with "
                f"a letter, at most 64 characters), never free text, not {self.code[:80]!r}"
            )


@final
@dataclass(frozen=True)
class AdversaryView:
    """All an adversary sees of one trial (task P16.2): the contract, the programs, its budget, and its outcomes.

    `contract` is the item's InputContract (lassi.harness), exactly that
    type, its inputs a non-empty tuple of exactly ContractInput whose fields
    have the exact types input_contract gives them (name and dtype a str,
    shape a tuple of ints, lo and hi an int or a float, the two flags a
    bool); `source` and `candidate` map each file's relative path to its
    text, both exactly str, for the source program and the candidate, and
    are kept as read-only copies; `budget_left` is how many input sets the
    adversary may still propose, exactly an int of at least 0 (a bool is
    refused); `outcomes` holds earlier proposals' outcomes (ProposalOutcome)
    in order, a list kept as a tuple. A held-out input, the item's seed, and
    a reference output have no field here, and anything else in a field
    (the held-out input spec, an array, file bytes, a run result, a subclass
    that adds a field) is refused with a TypeError or a ValueError naming
    the field (Harness Contract: held-out inputs never appear in a prompt).
    The class takes no subclass, which could add a field.

    Known cases, not complete: the checks bound what each field can hold,
    not what a text says, so a file's text could quote anything; the stage
    that builds the view (task P16.5) gives it the source program's and the
    candidate's files only.
    """

    contract: InputContract
    source: Mapping[str, str]
    candidate: Mapping[str, str]
    budget_left: int
    outcomes: tuple[ProposalOutcome, ...]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Refuse every subclass (see the class docstring)."""
        _refuse_subclass("AdversaryView", cls)

    def __post_init__(self) -> None:
        """Check every field (see the class docstring) and keep read-only copies of the files and outcomes."""
        _check_contract(self.contract, "AdversaryView.contract")
        for name in ("source", "candidate"):
            object.__setattr__(self, name, _text_files(getattr(self, name), f"AdversaryView.{name}"))
        budget = self.budget_left
        if type(budget) is not int:
            raise TypeError(f"AdversaryView.budget_left must be an int, not {type(budget).__name__}")
        if budget < 0:
            raise ValueError(f"AdversaryView.budget_left must be at least 0, not {budget}")
        object.__setattr__(self, "outcomes", _exact_items(self.outcomes, ProposalOutcome, "AdversaryView.outcomes"))


def _refuse_subclass(base: str, cls: type) -> None:
    """Raise TypeError for `cls`, a subclass of `base`, which takes none: a subclass could add an unchecked field."""
    raise TypeError(f"{base} takes no subclass ({cls.__name__}): a subclass could add a field its checks never see")


def _exact_items(items: object, kind: type, where: str) -> tuple[Any, ...]:
    """Return `items` as a tuple when it is a tuple or list whose every item is exactly of `kind`; else TypeError."""
    if not isinstance(items, (tuple, list)):
        raise TypeError(f"{where} must be a tuple or list of {kind.__name__}, not {type(items).__name__}")
    for index, item in enumerate(items):
        if type(item) is not kind:
            raise TypeError(f"{where}[{index}] must be a {kind.__name__}, not {type(item).__name__}")
    return tuple(items)


def _check_array_fields(array: LassiArray, where: str) -> None:
    """Refuse an array whose name or dtype is not exactly a str or whose shape holds anything but exactly ints."""
    for name in ("name", "dtype"):
        value = getattr(array, name)
        if type(value) is not str:
            raise TypeError(f"{where}.{name} must be a str, not {type(value).__name__}")
    others = [type(dim).__name__ for dim in array.shape if type(dim) is not int]
    if others:
        raise TypeError(f"{where}.shape must hold ints, not {others[0]}")


def _check_contract(contract: object, where: str) -> None:
    """Refuse a contract that is not exactly an InputContract of exactly ContractInput (AdversaryView's docstring)."""
    if type(contract) is not InputContract:
        raise TypeError(f"{where} must be an InputContract, not {type(contract).__name__}")
    inputs = contract.inputs
    if type(inputs) is not tuple or not inputs:
        given = "an empty tuple" if type(inputs) is tuple else type(inputs).__name__
        raise TypeError(f"{where}.inputs must be a non-empty tuple of ContractInput, not {given}")
    for index, spec in enumerate(inputs):
        at = f"{where}.inputs[{index}]"
        if type(spec) is not ContractInput:
            raise TypeError(f"{at} must be a ContractInput, not {type(spec).__name__}")
        for name, kinds in _CONTRACT_INPUT_TYPES:
            value = getattr(spec, name)
            if type(value) not in kinds:
                wanted = " or ".join(kind.__name__ for kind in kinds)
                raise TypeError(f"{at}.{name} must be {wanted}, not {type(value).__name__}")
        if type(spec.shape) is not tuple or any(type(dim) is not int for dim in spec.shape):
            raise TypeError(f"{at}.shape must be a tuple of ints, not {type(spec.shape).__name__} {spec.shape!r}")


def _text_files(files: object, where: str) -> Mapping[str, str]:
    """Return a read-only copy of `files` when it maps exactly str paths to exactly str text; else TypeError."""
    if not isinstance(files, Mapping):
        raise TypeError(f"{where} must map each file's path (str) to its text (str), not {type(files).__name__}")
    for path, text in files.items():
        if type(path) is not str:
            raise TypeError(f"{where} has a path of type {type(path).__name__}; each path is a str")
        if type(text) is not str:
            raise TypeError(f"{where}[{path!r}] is a {type(text).__name__}; each file is given as its text (str)")
    return MappingProxyType(dict(files))


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


class FrontendRefusal(Exception):
    """A source outside a frontend's published subset: one Diagnostic at stage parse per construct it refuses.

    Every Frontend raises it, and only it, for a source outside its subset,
    instead of returning a module (bible Frontend Rules: anything else fails
    with a readable diagnostic that is counted in corpus statistics). It is
    built from Diagnostic records (lassi.core.record) alone and keeps them,
    in the order given, as the tuple `diagnostics`, its own copy. Each is at
    stage "parse", one of the Result Record's stages, so the record gains no
    field or stage. Its message names each diagnostic's place as
    `<file>:<line>:<column>`, as much of it as the diagnostic holds, with
    its code and message, on one line of ASCII.

    It refuses an empty sequence, and a diagnostic at any other stage, with
    a ValueError that names the stage, and an item that is not a Diagnostic
    with a TypeError. It is neither a ValueError nor a ContextExceeded, so
    no handler written for those catches it by accident.
    """

    def __init__(self, diagnostics: Sequence[Diagnostic]) -> None:
        """Keep a copy of `diagnostics` after checking each one; the message names every place and reason."""
        from lassi.core.record import Diagnostic  # lassi.core.record imports this module, so import it here

        items = tuple(diagnostics)
        if not items:
            raise ValueError("a FrontendRefusal needs at least one diagnostic")
        for index, item in enumerate(items):
            if not isinstance(item, Diagnostic):
                raise TypeError(f"item {index} of a FrontendRefusal is a {type(item).__name__}, not a Diagnostic")
            if item.stage != "parse":
                raise ValueError(
                    f"diagnostic {index} of a FrontendRefusal is at stage {item.stage!r}; a refusal holds "
                    "diagnostics at stage 'parse' only"
                )
        super().__init__(_refusal_message(items))
        self.diagnostics = items


def _refusal_message(diagnostics: tuple[Diagnostic, ...]) -> str:
    """Return a FrontendRefusal's message: how many constructs, then each one's place, code, and message."""
    parts = []
    for item in diagnostics:
        place = ""
        if item.file is not None:
            place = item.file
            if item.line is not None:
                place += f":{item.line}" + (f":{item.column}" if item.column is not None else "")
        elif item.line is not None:
            place = f"line {item.line}" + (f", column {item.column}" if item.column is not None else "")
        fields = [field for field in (place, item.code, item.message) if field]
        parts.append(": ".join(fields))
    count = len(diagnostics)
    head = f"the source is outside the frontend's subset ({count} construct{'' if count == 1 else 's'})"
    text = f"{head}: {'; '.join(parts)}"
    return "".join(char if " " <= char <= "~" else char.encode("unicode_escape").decode("ascii") for char in text)


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
        """Return the hub-level module for the given source files and flags.

        A source outside the frontend's published subset raises
        FrontendRefusal, defined here, with one parse-stage Diagnostic per
        construct outside it, and no module is returned.
        """
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
    """Traces a run: timing, and power where telemetry exists (task P17.7).

    run_loop calls start() right before one attempt run and stop() right
    after it, in a finally block, so stop() also runs when the executor
    raises. One profiler is built per run and profiles one window after
    another. A profiler that declares supports_power reads power telemetry
    and declares takes_device; one that does not leaves the power fields of
    its Profile null.
    """

    def start(self) -> None:
        """Begin tracing one window; a second start() before stop() is a ValueError."""
        ...

    def stop(self) -> Profile:
        """End the window and return its Profile; stop() without start() is a ValueError."""
        ...


class Agent(Component, Protocol):
    """A role bound to a backend, tools, and a budget.

    An adversary (bible Agents And Judges, the Adversary role; task P16.2)
    is an Agent that declares the capability proposes_inputs
    (lassi.core.capabilities PROPOSES_INPUTS) and provides propose(view)
    (lassi.core.capabilities Adversary). Its AdversaryView holds the item's
    input contract, the source program and the candidate as text, the
    budget left, and earlier proposals' outcomes, and never a held-out
    input, the item's seed, or a reference output; it returns Proposals,
    which are data, never code. A recipe's adversary section binds it at
    adversary.kind and requires the capability (lassi.core.recipe).
    """

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
    records in file order (() for bench). `suite` is the bench suite's
    train-only view (lassi.bench.registry TrainView, from Suite.train_view();
    task P17.9), which holds no eval or unassigned item (Agent Rule 5), and
    `items` the names of the selected items (None and () for synthetic
    data). A Trainer reads a bench item through bench_item.
    """

    source: str
    split_hash: str
    records: tuple[Mapping[str, Any], ...]
    suite: TrainView | None
    items: tuple[str, ...]

    def bench_item(self, name: str) -> SuiteItem:
        """Return bench item `name` through the train-only view.

        An eval or unassigned item raises EvalSplitError (Agent Rule 5); a
        train item outside `items`, which the split hash does not cover, and
        synthetic data, which has no suite, raise ValueError.
        """
        if self.suite is None:
            raise ValueError(f"the train data is {self.source}, not bench, so it has no bench item {name!r}")
        found = self.suite.item(name)
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
    and loads no model. The train layer then calls check(recipe, data)
    before any directory exists (task P17.9): a Trainer refuses there every
    recipe key, value, and record it does not carry out, rather than train
    without it.
    """

    @staticmethod
    def framework() -> FrameworkBuild:
        """Return the framework build it trains on, read from build metadata only; called on the class."""
        ...

    def check(self, recipe: Mapping[str, Any], data: TrainData) -> None:
        """Raise ValueError naming any recipe key, value, or record it does not carry out; write and load nothing."""
        ...

    def train(self, job: TrainJob) -> TrainResult:
        """Run the steps, writing every output under job.out_dir only; return the steps run and the checkpoints."""
        ...
