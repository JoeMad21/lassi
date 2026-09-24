"""The Result Record: one Trial per run of one bench item, one Attempt per generation.

The same record drives the correction prompt, the metrics, the RL reward, and
the corpus harvest (bible Result Record, Design Principle 2). Every class here
is a frozen, keyword-only dataclass whose field names and order are those of
the bible's Result Record block. Values are checked when a record is built and
a bad value raises ValueError naming the field and the value: every field must
match its type hint (a bool is never a number, an int in a float field is
stored as a float, and a float must be finite), then each class checks its own
rules.

This module also holds the trial naming rule (bible Readability Standards,
Naming), the unified diff between two attempts' files, and strict JSON
conversion for every record class.
"""

from __future__ import annotations

import dataclasses
import difflib
import functools
import json
import math
import re
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, NamedTuple, NoReturn

from lassi.core.interfaces import Sampling

STAGES = ("S0", "S1", "S2", "S3", "S4", "S5")
DIAGNOSTIC_STAGES = ("parse", "verify", "lower", "compile", "jit", "run")
SEVERITIES = ("error", "warning", "note")
TOOLCHAIN_PIN_NAMES = ("llvm", "polygeist", "tt_mlir", "tt_metal", "ttsim", "furiosa_sdk", "cuda", "nvhpc", "rocm")
# The run flags RunInfo records, named as the RunResult flags they copy, in field order.
RUN_FLAG_NAMES = ("stdout_truncated", "stderr_truncated", "workdir_incomplete")
# The fixed codes of Final.end_reason: why a trial ended early. baseline-compile and baseline-run end a trial
# before any model call (a reference program did not build, or its run exited nonzero or hung); correction-cap
# means an error remained when loop.max_corrections stopped the correction loop; upstream-crash means that, with
# fixes.execution_gate off, a compiling attempt came past upstream's execution gate when no earlier attempt had run,
# where upstream's notebook raises (it reads run output that was never set).
END_REASONS = ("baseline-compile", "baseline-run", "correction-cap", "upstream-crash")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_ID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_DRIVE = re.compile(r"[A-Za-z]:")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_RUN = re.compile(r"run([0-9]{2,})")
_NO_FINAL_NEWLINE = "\\ No newline at end of file\n"


# ---------------------------------------------------------------------------
# Validation helpers


def _fail(owner: str, name: str, value: object, rule: str) -> NoReturn:
    """Raise the ValueError for one bad field value."""
    raise ValueError(f"{owner}.{name} {rule}, got {value!r}")


def _check_choice(owner: str, name: str, value: object, choices: tuple[str, ...]) -> None:
    """Require value to be one of choices."""
    if value not in choices:
        _fail(owner, name, value, f"must be one of {', '.join(choices)}")


def _check_sha256(owner: str, name: str, value: object) -> None:
    """Require value to be 64 lowercase hex characters."""
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        _fail(owner, name, value, "must be 64 lowercase hex characters")


def _check_non_negative(owner: str, name: str, value: int) -> None:
    """Require an integer value to be zero or more."""
    if value < 0:
        _fail(owner, name, value, "must be >= 0")


def _check_non_empty(owner: str, name: str, value: object) -> None:
    """Require value to be a non-empty string."""
    if not isinstance(value, str) or not value:
        _fail(owner, name, value, "must be a non-empty string")


def _check_utc(owner: str, name: str, value: str) -> None:
    """Require value to be an ISO 8601 time with a UTC offset of zero, such as 2026-09-23T12:34:56+00:00."""
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        _fail(owner, name, value, "must be an ISO 8601 time in UTC")
    if moment.utcoffset() != timedelta(0):
        _fail(owner, name, value, "must be an ISO 8601 time in UTC")


def _check_unit(owner: str, name: str, value: float | None) -> None:
    """Require a float value, when set, to lie in [0, 1]."""
    if value is not None and not 0.0 <= value <= 1.0:
        _fail(owner, name, value, "must be in [0, 1]")


# ---------------------------------------------------------------------------
# Field types


def _convert(hint: Any, value: Any, where: str, build: Callable[[type, Any, str], Any]) -> Any:
    """Return `value` as the type `hint` names, or raise ValueError naming `where` and the value.

    `X | None`, `list[X]`, and `dict[str, X]` are handled here, and a record
    class is handed to `build`. Lists and dicts come back as new containers.
    """
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        if value is None:
            return None
        (inner,) = [arg for arg in typing.get_args(hint) if arg is not type(None)]
        return _convert(inner, value, where, build)
    if dataclasses.is_dataclass(hint):
        return build(hint, value, where)
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{where}: expected a list, got {value!r}")
        (item_hint,) = typing.get_args(hint)
        return [_convert(item_hint, item, f"{where}[{i}]", build) for i, item in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise ValueError(f"{where}: expected a dict with string keys, got {value!r}")
        _, item_hint = typing.get_args(hint)
        return {key: _convert(item_hint, item, f"{where}[{key!r}]", build) for key, item in value.items()}
    return _scalar(hint, value, where)


def _scalar(hint: Any, value: Any, where: str) -> Any:
    """Check one str, int, float, or bool value; a bool is never a number, and a float must be finite."""
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if hint is float and number:
        return _finite_float(value, where)
    if hint is int and number and isinstance(value, int):
        return value
    if hint in (str, bool) and isinstance(value, hint):
        return value
    raise ValueError(f"{where}: expected {getattr(hint, '__name__', hint)}, got {value!r}")


def _finite_float(value: int | float, where: str) -> float:
    """Return a number as a float; NaN, infinity, and ints too large for a float raise ValueError."""
    try:
        result = float(value)
    except OverflowError:
        result = math.inf
    if not math.isfinite(result):
        raise ValueError(f"{where}: expected a finite number, got {value!r}")
    return result


def _instance_of(cls: type, value: Any, where: str) -> Any:
    """Return `value` when it is an instance of the record class `cls`; raise ValueError otherwise."""
    if not isinstance(value, cls):
        raise ValueError(f"{where}: expected {cls.__name__}, got {value!r}")
    return value


@functools.lru_cache(maxsize=None)
def _field_hints(cls: type) -> dict[str, Any]:
    """Return the resolved type hint of every field of a record class."""
    return typing.get_type_hints(cls)


def _checked_fields(obj: Any, where: str) -> dict[str, Any]:
    """Return the fields of a dataclass instance, each checked against its type hint, keyed by name."""
    hints = _field_hints(type(obj))
    return {
        spec.name: _convert(hints[spec.name], getattr(obj, spec.name), f"{where}.{spec.name}", _instance_of)
        for spec in dataclasses.fields(obj)
    }


def _check_fields(obj: Any) -> None:
    """Check every field of a record against its type hint and store the checked values.

    An int in a float field is stored as a float, and lists and dicts are
    stored as copies, so a record shares no container with its caller.
    """
    for name, value in _checked_fields(obj, type(obj).__name__).items():
        object.__setattr__(obj, name, value)


# ---------------------------------------------------------------------------
# Record classes


@dataclass(frozen=True, kw_only=True)
class TextRef:
    """A text kept once in the text store: its sha256 and its path under the store root."""

    sha256: str
    path: str

    def __post_init__(self) -> None:
        """Check the hash and that the path is a relative POSIX path inside the store."""
        _check_fields(self)
        _check_sha256("TextRef", "sha256", self.sha256)
        path = self.path
        if "\\" in path or _DRIVE.match(path) or _CONTROL.search(path):
            _fail("TextRef", "path", path, "must be a POSIX path with no drive, backslash, or control character")
        segments = path.split("/")
        if "" in segments or ".." in segments:
            _fail("TextRef", "path", path, "must be a relative path with no empty or '..' segment")


@dataclass(frozen=True, kw_only=True)
class Diagnostic:
    """One parsed toolchain, verifier, or runtime message."""

    stage: str
    severity: str
    code: str | None = None
    file: str | None = None
    line: int | None = None
    column: int | None = None
    message: str

    def __post_init__(self) -> None:
        """Check the field types, then the stage and severity against the bible's lists."""
        _check_fields(self)
        _check_choice("Diagnostic", "stage", self.stage, DIAGNOSTIC_STAGES)
        _check_choice("Diagnostic", "severity", self.severity, SEVERITIES)


@dataclass(frozen=True, kw_only=True)
class RunInfo:
    """How one run of the built artifact ended; None means not run or not measured.

    `stdout_truncated`, `stderr_truncated`, and `workdir_incomplete` copy the
    RunResult flags of the same names: the executor kept only part of the
    stream, or returned only part of what the run wrote in its workdir. A
    stage that records a run records each flag as a bool, so a run whose
    output was kept whole reads False; None means not recorded, as for a run
    that did not happen or a trial.json written before the flags existed.
    """

    exit_code: int | None = None
    hang: bool | None = None
    sim_ub: bool | None = None
    wall_s: float | None = None
    stdout_ref: TextRef | None = None
    outputs_ref: TextRef | None = None
    stdout_truncated: bool | None = None
    stderr_truncated: bool | None = None
    workdir_incomplete: bool | None = None

    def __post_init__(self) -> None:
        """Check the field types; the wall time must be finite."""
        _check_fields(self)


@dataclass(frozen=True, kw_only=True)
class Alignment:
    """Oracle alignment per input and its mean, each in [0, 1]."""

    per_input: list[float] = field(default_factory=list)
    mean: float | None = None

    def __post_init__(self) -> None:
        """Check the field types and that every value lies in [0, 1]."""
        _check_fields(self)
        for value in self.per_input:
            _check_unit("Alignment", "per_input", value)
        _check_unit("Alignment", "mean", self.mean)


@dataclass(frozen=True, kw_only=True)
class Profile:
    """Profiler measurements; None means not measured."""

    runtime_s: float | None = None
    avg_power_w: float | None = None
    energy_j: float | None = None

    def __post_init__(self) -> None:
        """Check the field types; every measurement must be finite."""
        _check_fields(self)


@dataclass(frozen=True, kw_only=True)
class Guards:
    """Guard outcomes: host compute, harness tampering, oracle access; None means not checked."""

    host_compute: bool | None = None
    harness_tamper: bool | None = None
    oracle_access: bool | None = None

    def __post_init__(self) -> None:
        """Check that every outcome is a bool or None."""
        _check_fields(self)


@dataclass(frozen=True, kw_only=True)
class ScoreBreakdown:
    """Named score components (None when not measured) and the scalar derived from them."""

    components: dict[str, float | None] = field(default_factory=dict)
    scalar: float | None = None

    def __post_init__(self) -> None:
        """Check the field types; every component and the scalar must be finite."""
        _check_fields(self)


@dataclass(frozen=True, kw_only=True)
class RequestMessage:
    """One message of a model request: its chat role and its text in the text store, by reference."""

    role: str
    ref: TextRef

    def __post_init__(self) -> None:
        """Check the field types and that the role is a non-empty string."""
        _check_fields(self)
        _check_non_empty("RequestMessage", "role", self.role)


@dataclass(frozen=True, kw_only=True)
class Request:
    """One model call of a trial, in Trial.requests at position `index`.

    `stage` is the registered name of the stage that sent it (a correction
    names the loop stage that asked for it). `attempt_index` is the attempt
    the reply became, and None for a request whose reply fills Trial.context.
    `messages` holds every message sent, in order, system messages included,
    each by text-store reference; `reply_ref` is the reply as kept, each lone
    surrogate replaced by U+FFFD. `diagnostics` holds what was noted about a
    reply that no attempt carries, such as a context reply's `invalid-text`
    warning. Every field but `diagnostics` is required, so a missing value is
    never read as a context request or an empty call.
    """

    index: int
    stage: str
    attempt_index: int | None
    messages: list[RequestMessage]
    reply_ref: TextRef
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Check the field types, the indexes, the stage name, and that at least one message was sent."""
        _check_fields(self)
        _check_non_negative("Request", "index", self.index)
        _check_non_empty("Request", "stage", self.stage)
        if self.attempt_index is not None:
            _check_non_negative("Request", "attempt_index", self.attempt_index)
        if not self.messages:
            _fail("Request", "messages", self.messages, "must hold at least one message")


@dataclass(frozen=True, kw_only=True)
class Attempt:
    """One generation: index 0 is the initial one, then one per correction."""

    index: int
    prompt_ref: TextRef | None = None
    response_text: str = ""
    files: dict[str, str] = field(default_factory=dict)
    diff_from_previous: str = ""
    stage_reached: str
    diagnostics: list[Diagnostic] = field(default_factory=list)
    run: RunInfo = field(default_factory=RunInfo)
    alignment: Alignment = field(default_factory=Alignment)
    profile: Profile = field(default_factory=Profile)
    guards: Guards = field(default_factory=Guards)
    score: ScoreBreakdown = field(default_factory=ScoreBreakdown)

    def __post_init__(self) -> None:
        """Check the field types, the index, and the stage reached."""
        _check_fields(self)
        _check_non_negative("Attempt", "index", self.index)
        _check_choice("Attempt", "stage_reached", self.stage_reached, STAGES)


@dataclass(frozen=True, kw_only=True)
class ToolchainPins:
    """The pin of every toolchain a run used; None means the toolchain was not used."""

    llvm: str | None = None
    polygeist: str | None = None
    tt_mlir: str | None = None
    tt_metal: str | None = None
    ttsim: str | None = None
    furiosa_sdk: str | None = None
    cuda: str | None = None
    nvhpc: str | None = None
    rocm: str | None = None

    def __post_init__(self) -> None:
        """Check that every pin is a string or None."""
        _check_fields(self)


@dataclass(frozen=True, kw_only=True)
class Provenance:
    """A trial's copy of its run manifest, so a trial read outside its run tree still says where it came from.

    The run's provenance.json stays authoritative (bible Result Record,
    Storage); the runner fills this copy from that manifest, key by key:
    commit from "commit", dirty from "dirty", device from "device", sdk from
    "driver", and date from "started_utc". The types are what the manifest
    holds: commit and dirty are None when git is unavailable, device is None
    when the executor names no device, sdk is None until an executor reports
    an SDK or driver version, and date (the run's start, ISO 8601 UTC with
    seconds) is always set. A known commit is a full git object id (40 or 64
    lowercase hex characters), a known device or sdk is a non-empty string,
    and date must parse as an ISO 8601 time in UTC. Every field is required,
    so a missing value is never read as unknown.
    """

    commit: str | None
    dirty: bool | None
    device: str | None
    sdk: str | None
    date: str

    def __post_init__(self) -> None:
        """Check the field types the run manifest gives, a known commit, device, and sdk, and the date."""
        _check_fields(self)
        if self.commit is not None and not _GIT_OBJECT_ID.fullmatch(self.commit):
            _fail("Provenance", "commit", self.commit, "must be 40 or 64 lowercase hex characters when known")
        for name in ("device", "sdk"):
            if getattr(self, name) is not None:
                _check_non_empty("Provenance", name, getattr(self, name))
        _check_utc("Provenance", "date", self.date)


@dataclass(frozen=True, kw_only=True)
class BenchItem:
    """The benchmark item a trial works on."""

    suite: str
    item: str
    split: str
    direction: str

    def __post_init__(self) -> None:
        """Check that every field is a non-empty string."""
        _check_fields(self)
        for name in ("suite", "item", "split", "direction"):
            _check_non_empty("BenchItem", name, getattr(self, name))


@dataclass(frozen=True, kw_only=True)
class ModelInfo:
    """The model arm: serving backend, model id, and sampling parameters."""

    backend: str
    id: str
    sampling: Sampling

    def __post_init__(self) -> None:
        """Check the field types, including the sampling values, which interfaces.Sampling leaves unchecked."""
        _check_fields(self)
        object.__setattr__(self, "sampling", Sampling(**_checked_fields(self.sampling, "ModelInfo.sampling")))


@dataclass(frozen=True, kw_only=True)
class Context:
    """Context given to the model: the knowledge summary and the source description."""

    knowledge_summary: str = ""
    source_description: str = ""

    def __post_init__(self) -> None:
        """Check that both texts are strings."""
        _check_fields(self)


@dataclass(frozen=True, kw_only=True)
class EndReason:
    """Why a trial ended early: one of END_REASONS and a message that says what happened."""

    code: str
    message: str

    def __post_init__(self) -> None:
        """Check the field types, that the code is one of END_REASONS, and that the message is not empty."""
        _check_fields(self)
        _check_choice("EndReason", "code", self.code, END_REASONS)
        _check_non_empty("EndReason", "message", self.message)


@dataclass(frozen=True, kw_only=True)
class Final:
    """The outcome of the whole trial; None means not reached or not measured.

    `alignment` is the alignment mean of the attempt whose output stands
    (standing_attempt), None when no attempt ran or that one was not
    aligned. `end_reason` is None when the trial ended normally, and
    otherwise says why it ended early (EndReason).
    """

    stage_reached: str | None = None
    alignment: float | None = None
    score: float | None = None
    corrections: int = 0
    wall_s: float | None = None
    end_reason: EndReason | None = None

    def __post_init__(self) -> None:
        """Check the field types, the stage, the correction count, and that the alignment lies in [0, 1]."""
        _check_fields(self)
        if self.stage_reached is not None:
            _check_choice("Final", "stage_reached", self.stage_reached, STAGES)
        _check_non_negative("Final", "corrections", self.corrections)
        _check_unit("Final", "alignment", self.alignment)


@dataclass(frozen=True, kw_only=True)
class Trial:
    """One run of one bench item by one arm; stages return new Trials and never mutate one.

    `provenance` is required: a Trial without it raises TypeError naming the
    field, and a trial.json without it fails from_dict with a ValueError.
    `reference_run` is the target reference's run from the baseline stage
    (exit status, hang flag, wall time, stdout by reference, and the run
    flags); it stays all None when the reference was not run, as under a
    compile-only executor. `requests` holds every model call in the order
    sent (Request); None means not recorded, as in a trial.json written
    before requests were, and the runner starts every trial with an empty
    list, so a trial that asked no model records [].
    """

    trial_id: str
    recipe_hash: str
    toolchain_pins: ToolchainPins = field(default_factory=ToolchainPins)
    provenance: Provenance
    bench_item: BenchItem
    model: ModelInfo
    reference_run: RunInfo = field(default_factory=RunInfo)
    context: Context = field(default_factory=Context)
    requests: list[Request] | None = None
    attempts: list[Attempt] = field(default_factory=list)
    final: Final = field(default_factory=Final)

    def __post_init__(self) -> None:
        """Check the field types, the id, the recipe hash, the id against the bench item, and the list orders.

        Each attempt's and each request's index must be its position, and a
        request's attempt_index must name an attempt of the trial.
        """
        _check_fields(self)
        parsed = parse_trial_id(self.trial_id)
        _check_sha256("Trial", "recipe_hash", self.recipe_hash)
        segments = {"suite": parsed.bench, "direction": parsed.direction, "item": parsed.item}
        for name, segment in segments.items():
            value = getattr(self.bench_item, name)
            if value != segment:
                _fail("Trial", f"bench_item.{name}", value, f"must equal {segment!r} from trial_id {self.trial_id!r}")
        for position, attempt in enumerate(self.attempts):
            if attempt.index != position:
                _fail("Trial", f"attempts[{position}].index", attempt.index, f"must be {position}")
        for position, request in enumerate(self.requests or []):
            if request.index != position:
                _fail("Trial", f"requests[{position}].index", request.index, f"must be {position}")
            if request.attempt_index is not None and request.attempt_index >= len(self.attempts):
                where = f"requests[{position}].attempt_index"
                _fail("Trial", where, request.attempt_index, f"must name one of the {len(self.attempts)} attempt(s)")

    def with_attempt(self, attempt: Attempt) -> Trial:
        """Return a new Trial with `attempt` appended; its index must equal the current attempt count."""
        expected = len(self.attempts)
        if attempt.index != expected:
            raise ValueError(f"Trial.with_attempt: attempt.index must be {expected}, got {attempt.index!r}")
        return dataclasses.replace(self, attempts=[*self.attempts, attempt])

    def with_request(self, request: Request) -> Trial:
        """Return a new Trial with `request` appended; its index must equal the current request count.

        A trial whose requests were not recorded (None) starts its list with
        this request.
        """
        requests = self.requests or []
        if request.index != len(requests):
            raise ValueError(f"Trial.with_request: request.index must be {len(requests)}, got {request.index!r}")
        return dataclasses.replace(self, requests=[*requests, request])


# ---------------------------------------------------------------------------
# Readings


def standing_attempt(trial: Trial) -> Attempt | None:
    """Return the attempt whose output stands as the trial's output, or None when no attempt ran.

    It is the last attempt that ran: the last one whose Attempt.run holds
    stdout (run.stdout_ref is set), whether or not later attempts exist that
    did not run, so stale output past the execution gate stands (bible
    Oracles). Final.alignment is its alignment mean.
    """
    ran = [attempt for attempt in trial.attempts if attempt.run.stdout_ref is not None]
    return ran[-1] if ran else None


# ---------------------------------------------------------------------------
# Naming


class TrialId(NamedTuple):
    """The parts of a trial_id `<project>/<arm>/<bench>/<direction>/<item>/run<NN>`."""

    project: str
    arm: str
    bench: str
    direction: str
    item: str
    run: int


def parse_trial_id(trial_id: str) -> TrialId:
    """Split a trial_id into its parts; raise ValueError naming the id when it breaks the naming rule."""
    if not isinstance(trial_id, str):
        raise ValueError(f"invalid trial_id {trial_id!r}: must be a string")
    parts = trial_id.split("/")
    if len(parts) != 6:
        raise ValueError(f"invalid trial_id {trial_id!r}: expected 6 '/'-separated segments, got {len(parts)}")
    for part in parts[:5]:
        if not _SEGMENT.fullmatch(part):
            raise ValueError(f"invalid trial_id {trial_id!r}: segment {part!r} must match {_SEGMENT.pattern}")
    run = _RUN.fullmatch(parts[5])
    if run is None:
        raise ValueError(f"invalid trial_id {trial_id!r}: last segment {parts[5]!r} must be 'run' and 2+ digits")
    return TrialId(*parts[:5], run=int(run.group(1)))


def make_trial_id(project: str, arm: str, bench: str, direction: str, item: str, run: int) -> str:
    """Return the trial_id for these parts, with the run written as run<NN>; raise ValueError if invalid."""
    segments = [project, arm, bench, direction, item]
    if not all(isinstance(segment, str) for segment in segments):
        raise ValueError(f"make_trial_id: segments must be strings, got {segments!r}")
    if not isinstance(run, int) or isinstance(run, bool):
        raise ValueError(f"make_trial_id: run must be an int, got {run!r}")
    trial_id = "/".join([*segments, f"run{run:02d}"])
    parse_trial_id(trial_id)
    return trial_id


def arm_segment(model_id: str) -> str:
    """Return the trial_id arm segment for a model id: each '/' becomes '--', as the Hugging Face cache spells it.

    A served model id such as "furiosa-ai/Llama-3.1-8B-Instruct" holds '/', which
    a trial_id segment may not; the record keeps the model id itself unchanged.
    """
    return model_id.replace("/", "--")


# ---------------------------------------------------------------------------
# Unified diff


def _split_lines(text: str) -> list[str]:
    """Split text after each LF only, keeping the LF; a last line without LF is kept as is."""
    lines = [line + "\n" for line in text.split("\n")]
    lines[-1] = lines[-1][:-1]
    if not lines[-1]:
        lines.pop()
    return lines


def _file_diff(path: str, previous: str | None, current: str | None) -> list[str]:
    """Return the unified diff lines for one path; None stands for an absent file."""
    before = _split_lines(previous) if previous is not None else []
    after = _split_lines(current) if current is not None else []
    from_file = f"a/{path}" if previous is not None else "/dev/null"
    to_file = f"b/{path}" if current is not None else "/dev/null"
    if not before and not after:
        # An empty file was added or removed: difflib writes nothing, so write the headers alone.
        return [f"--- {from_file}\n", f"+++ {to_file}\n"]
    lines = []
    for line in difflib.unified_diff(before, after, fromfile=from_file, tofile=to_file, n=3):
        lines.append(line if line.endswith("\n") else line + "\n" + _NO_FINAL_NEWLINE)
    return lines


def unified_diff(previous: Mapping[str, str], current: Mapping[str, str]) -> str:
    """Return the unified diff from `previous` to `current` files, path by path in sorted order.

    Headers are `--- a/<path>` and `+++ b/<path>`, with `/dev/null` for an added
    or removed file, and three lines of context; an added or removed empty file
    shows as its two header lines alone. A line without a final newline is
    followed by `\\ No newline at end of file`, as git does. Returns "" when
    nothing changed.
    """
    lines: list[str] = []
    for path in sorted(set(previous) | set(current)):
        before, after = previous.get(path), current.get(path)
        if before != after:
            lines += _file_diff(path, before, after)
    return "".join(lines)


# ---------------------------------------------------------------------------
# JSON


def to_dict(obj: Any) -> dict[str, Any]:
    """Return a record as plain dicts, lists, and scalars, with keys in field order."""
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(f"to_dict needs a record instance, got {obj!r}")
    return dataclasses.asdict(obj)


def json_text(data: Any) -> str:
    """Return data as record JSON text: two-space indent, ASCII only, no NaN, and a final LF."""
    return json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def to_json(obj: Any) -> str:
    """Return a record as JSON text; see json_text for the format."""
    return json_text(to_dict(obj))


def from_json(cls: type, text: str) -> Any:
    """Rebuild a record of class `cls` from JSON text written by to_json."""
    return from_dict(cls, json.loads(text))


def from_dict(cls: type, data: Mapping[str, Any]) -> Any:
    """Rebuild a record of class `cls` from plain data, strictly.

    Unknown keys and missing required keys raise ValueError naming the key and
    the class. Nested records, `X | None`, `list[X]`, and `dict[str, X]` are
    rebuilt from the type hints. A JSON int is accepted for a float and turned
    into a float; a bool is never accepted as a number, and NaN, infinity, and
    ints too large for a float are rejected.
    """
    return _record_from(cls, data, cls.__name__)


def _record_from(cls: type, data: Any, where: str) -> Any:
    """Build one record of class `cls` from a mapping found at `where`."""
    if not isinstance(data, Mapping):
        raise ValueError(f"{where}: expected an object for {cls.__name__}, got {data!r}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = [key for key in data if key not in fields]
    if unknown:
        raise ValueError(f"{where}: unknown key {unknown[0]!r} for {cls.__name__}")
    for name, spec in fields.items():
        required = spec.default is dataclasses.MISSING and spec.default_factory is dataclasses.MISSING
        if required and name not in data:
            raise ValueError(f"{where}: missing required key {name!r} for {cls.__name__}")
    hints = _field_hints(cls)
    values = {name: _convert(hints[name], value, f"{where}.{name}", _record_from) for name, value in data.items()}
    try:
        return cls(**values)
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from error
