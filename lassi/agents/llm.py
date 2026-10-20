"""The llm adversary: input sets a model proposes in a fixed declarative form (task P16.4).

Bible: Agents And Judges (the Agent interface: a role, an LLM backend,
versioned prompts, and a budget in tokens and steps), Adversarial Setups
(item 1; the adversary rule), Harness Contract (held-out inputs never in a
prompt), Component Interfaces (LLMBackend; ContextExceeded), Model Serving
(Serving Rules), Project Recipes (the adversary section; Notes, Replay
recording), Agent Rules 6, 12, and 14. [DESIGN]

LLMAdversary is registered as the Agent `llm` and declares proposes_inputs
and reports_log. A recipe binds it as

    adversary: {kind: llm, budget: {inputs: N, tokens: T, steps: S},
                backend: <LLMBackend>, model: <id>, prompts: <set>,
                sampling: {temperature: t, top_p: p, max_tokens: m},
                recording: tests/fixtures/<...>.json}

Every key is a required choice with no default, and each refusal is a
ValueError naming the key, so the runner refuses the recipe before any
directory exists (lassi.core.runner _check_agent). budget.inputs is the most
input sets proposed in one trial (the adversary stage's), budget.steps the
most model calls, and budget.tokens the most prompt plus completion tokens,
each an int of at least 1. backend names a registered LLMBackend that
replays a recording, one that lists `recording` in its config_keys (the
replay backend), built as factory(model, recording=<file>). A backend that
declares needs_reference (the mock) is refused, since an adversary never
gets the item's reference, and so is any backend that takes no recording:
the llm adversary binds scripted backends only, and a served or in-process
model is task P16.11 (OQ-068). sampling holds temperature (a finite number
of at least 0), top_p (in (0, 1]), and max_tokens (an int of at least 1).
prompts names
a prompt set holding propose.txt and malformed.txt (assets/prompts/
adversary-v0/ is the one written here). recording is required: the loader checks it
under the replay recording rule (lassi.core.recipe recording_file: a path
under tests/fixtures/) and enters adversary.recording_sha256, the sha256 of
its bytes, beside it; this class resolves the path itself with
recording_file, and refuses a backend whose recording_sha256, the sha256 of
the bytes it read, is not that one.

propose(view) asks the model, one step per call, with the messages
render_messages builds from the AdversaryView alone: the input contract, the
source program's and the candidate's files, the budget left, and the
earlier outcomes, each beside the JSON of this adversary's own proposal
there. The view holds no held-out input, no item seed, and no reference
output, so no prompt does. The reply is read strictly (parse_reply): a reply
that breaks the form (a ReplyError) gives no proposal and counts as a
malformed reply, and the model is asked again within the same propose()
call with malformed.txt, while steps and tokens are left. A reply of no
proposal stops the search; one of more input sets than view.budget_left is
cut to that many, and the rest are counted as dropped, since the stage
refuses a reply past the budget left. Each proposal the stage gets is data,
checked by check_inputs before any run; the reply's text outside its JSON
block is never read, and nothing the model claims becomes a finding (Agent
Rule 14): only binary_io confirms a divergence.

Budget: before each call, a spent step budget stops the search at stop
"steps" and a spent token budget at "tokens"; a call's tokens count once it
returns, so the total can pass budget.tokens by the last call's tokens. A
ContextExceeded from the backend counts as a step with no tokens and stops
the search at "context-exceeded". After a stop, propose() returns no
proposal. log() returns the AgentLog (lassi.core.record) the stage puts in
AdversaryRecord.agent_log: steps, the backend's token counts (a scripted
backend's are approximations, never measurements), malformed replies,
dropped proposals, and the stop.

The reply form (adversary-v0): exactly one fenced block, opened by a line
"```json" and closed by a line "```", and no other line that starts with
three backticks; it holds one JSON object {"proposals": [...]} (duplicate
keys and the NaN and Infinity literals refused), each proposal
{"inputs": [...]}, and each input {"name", "dtype", "shape", and exactly one
of "values" (every element), "fill" (one value for every element), or
"pattern" (values repeated, and cut, to the element count)}, with no other
key. A value is a JSON integer for an integer dtype, within the dtype's own
range, and for a float dtype a JSON number, read as the nearest double and
rounded to the dtype to nearest with ties to even, as generate_inputs rounds
a double (one past the format's largest finite value is refused), or the
string "nan" (the format's quiet NaN),
"inf", or "-inf". A dtype lassi_io does not know, a negative dim, or a shape
of more than MAX_INPUT_ELEMENTS elements gives the array no data, so the
stage refuses it as unreadable or too-many-elements; everything else the
form allows, inside the contract or not (another dtype or shape, a value
out of range, a NaN the contract refuses, a repeated, extra, or missing
input), is passed on as data for check_inputs to refuse.

A ReplyError's message is model-facing text: malformed.txt shows it to the
model as $reason, so the reasons are part of what the model sees, and they
are versioned with this file, not with the prompt set.

Known cases, not complete: the backend is built from the default registry
with only its recording as a setting; the model's messages and replies are
not stored, only the log's counts; a recording that runs out of replies
raises the replay backend's ServingError, which fails the run; the token
budget can be passed by one call; a malformed reply spends a step without
any proposal; a JSON number is rounded twice, to the nearest double and
then to the dtype, so a literal with more digits than a double holds can
round otherwise than its exact value would, and the integer literal -0 is
+0 (-0.0 keeps its sign); and a trailing space after the closing fence
breaks the form, while one after the opening fence does not.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from lassi import llm as _llm  # noqa: F401  (registers the LLMBackends this adversary may bind)
from lassi.core.capabilities import PROPOSES_INPUTS, REPORTS_LOG
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import AdversaryView, ContextExceeded, Message, Proposal, Sampling
from lassi.core.recipe import recording_file
from lassi.core.record import AgentLog
from lassi.core.registry import DEFAULT_REGISTRY, register
from lassi.harness.contract import MAX_INPUT_ELEMENTS, InputContract
from lassi.harness.lassi_io import _BITS_CODE, _FLOATS, _INTS, DTYPES, LassiArray, _round_bits, element_count
from lassi.prompts import prompt_dir, render

# The prompts of a prompt set this adversary renders: the request, and the follow-up after a malformed reply.
PROPOSE_PROMPT = "propose"
MALFORMED_PROMPT = "malformed"
# The keys of the adversary section's budget and sampling mappings, each required.
BUDGET_KEYS = ("inputs", "tokens", "steps")
SAMPLING_KEYS = ("temperature", "top_p", "max_tokens")
# The reply form's keys: of the top object, of a proposal, and of an input (one of DATA_KEYS besides the first three).
_TOP_KEYS = frozenset({"proposals"})
_PROPOSAL_KEYS = frozenset({"inputs"})
DATA_KEYS = ("values", "fill", "pattern")
_INPUT_KEYS = frozenset({"name", "dtype", "shape", *DATA_KEYS})
# The strings a float value may be, besides a JSON number.
_SPECIAL_FLOATS = ("nan", "inf", "-inf")
# The line that opens the reply's block, and the start of every fence line.
_OPEN = "```json"
_FENCE = "```"


class ReplyError(ValueError):
    """A reply that breaks the adversary-v0 form; the message says how, and the model is shown it."""


# ---------------------------------------------------------------------------
# The reply form


def parse_reply(text: str) -> tuple[Proposal, ...]:
    """Return the proposals of a reply in the adversary-v0 form; raise ReplyError when it breaks the form.

    The module docstring states the form and how each input becomes an array.
    """
    return tuple(proposal for proposal, _ in _parse(text))


def _parse(text: str) -> list[tuple[Proposal, str]]:
    """Return each proposal of the reply with the compact JSON of its proposal object, in order."""
    found = []
    for index, entry in enumerate(_proposal_objects(_block(text))):
        where = f"proposals[{index}]"
        if not isinstance(entry, dict) or set(entry) != _PROPOSAL_KEYS:
            raise ReplyError(f"{where} must be an object with exactly the key 'inputs'")
        inputs = entry["inputs"]
        if not isinstance(inputs, list):
            raise ReplyError(f"{where}.inputs must be a list of input objects")
        arrays = tuple(_array(item, f"{where}.inputs[{position}]") for position, item in enumerate(inputs))
        found.append((Proposal(arrays), json.dumps(entry, sort_keys=True, separators=(",", ":"))))
    return found


def _block(text: str) -> str:
    """Return the text inside the reply's one ```json block; ReplyError unless there is exactly one fence pair."""
    lines = text.split("\n")
    fences = [index for index, line in enumerate(lines) if line.lstrip().startswith(_FENCE)]
    if len(fences) != 2:
        raise ReplyError(
            f"the reply must hold exactly one fenced block, opened by a line {_OPEN} and closed by a line {_FENCE}, "
            f"and no other fence; it holds {len(fences)} fence line(s)"
        )
    first, last = fences
    if lines[first].rstrip() != _OPEN:
        raise ReplyError(f"the fenced block must open with a line that is exactly {_OPEN}")
    if lines[last].rstrip("\r") != _FENCE:
        raise ReplyError(f"the fenced block must close with a line that is exactly {_FENCE}")
    return "\n".join(lines[first + 1 : last])


def _proposal_objects(body: str) -> list[Any]:
    """Return the proposals list of the block's JSON object; ReplyError for bad JSON or another top object."""
    try:
        top = json.loads(body, object_pairs_hook=_unique_keys, parse_constant=_no_constant)
    except ReplyError:
        raise
    except ValueError as error:
        raise ReplyError(f"the fenced block is not valid JSON: {error}") from None
    except RecursionError:
        raise ReplyError("the fenced block nests too deeply to read; a proposal object is three levels deep") from None
    if not isinstance(top, dict) or set(top) != _TOP_KEYS:
        raise ReplyError("the fenced block must hold one JSON object with exactly the key 'proposals'")
    if not isinstance(top["proposals"], list):
        raise ReplyError("'proposals' must be a list")
    return top["proposals"]


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object, refusing a key given twice."""
    found: dict[str, Any] = {}
    for key, value in pairs:
        if key in found:
            raise ReplyError(f"the JSON object gives the key {key!r} twice")
        found[key] = value
    return found


def _no_constant(name: str) -> Any:
    """Refuse the NaN and Infinity literals, which JSON does not have."""
    raise ReplyError(f"{name} is not JSON; write the string \"nan\", \"inf\", or \"-inf\" for a float dtype")


def _array(item: object, where: str) -> LassiArray:
    """Return the LassiArray of one input object at `where` (the module docstring's encoding rules)."""
    if not isinstance(item, dict):
        raise ReplyError(f"{where} must be an input object")
    extra = sorted(set(item) - _INPUT_KEYS)
    if extra:
        raise ReplyError(f"{where} has the key {extra[0]!r}; an input has name, dtype, shape, and one of {DATA_KEYS}")
    for key in ("name", "dtype", "shape"):
        if key not in item:
            raise ReplyError(f"{where} has no {key!r}")
    given = [key for key in DATA_KEYS if key in item]
    if len(given) != 1:
        raise ReplyError(f"{where} must give exactly one of 'values', 'fill', and 'pattern', not {len(given)}")
    name, dtype, shape = item["name"], item["dtype"], item["shape"]
    if not isinstance(name, str) or not isinstance(dtype, str):
        raise ReplyError(f"{where}: name and dtype must be strings")
    if not isinstance(shape, list) or any(type(dim) is not int for dim in shape):
        raise ReplyError(f"{where}.shape must be a list of integers")
    (key,) = given
    data = _data(item[key], key, dtype, shape, f"{where}.{key}")
    return LassiArray(name, dtype, tuple(shape), data)


def _data(given: Any, key: str, dtype: str, shape: list[int], where: str) -> bytes:
    """Return the element bytes `given` under `key` describes; empty when lassi_io could not hold the array."""
    values = [given] if key == "fill" else given
    if key != "fill" and not isinstance(given, list):
        raise ReplyError(f"{where} must be a list of values")
    if key == "pattern" and not given:
        raise ReplyError(f"{where} must hold at least one value")
    for position, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ReplyError(f"{where}[{position}]: a value is a JSON number or a string, not {json.dumps(value)}")
    count = None if dtype not in DTYPES or min(shape, default=0) < 0 else element_count(shape, MAX_INPUT_ELEMENTS)
    if count is None:
        return b""  # the stage refuses the array: unreadable, or too-many-elements (module docstring)
    encoded = [_encode(value, dtype, f"{where}[{position}]") for position, value in enumerate(values)]
    if key == "values":
        return b"".join(encoded)
    return b"".join(encoded[position % len(encoded)] for position in range(count))


def _encode(value: int | float | str, dtype: str, where: str) -> bytes:
    """Return one element's little-endian bytes; ReplyError for a value the dtype cannot hold."""
    if dtype in _INTS:
        code, signed = _INTS[dtype]
        bits = 8 * DTYPES[dtype].itemsize
        low, high = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
        if type(value) is not int or not low <= value <= high:
            shown = json.dumps(value)
            raise ReplyError(f"{where}: {dtype} values are JSON integers from {low} to {high}, not {shown}")
        return struct.pack(f"<{code}", value)
    fmt = _FLOATS[dtype]
    return struct.pack(f"<{_BITS_CODE[fmt.width]}", _float_bits(value, dtype, where))


def _float_bits(value: int | float | str, dtype: str, where: str) -> int:
    """Return the bit pattern of a float value in `dtype`'s format (module docstring)."""
    fmt = _FLOATS[dtype]
    infinity = ((1 << fmt.exp_bits) - 1) << fmt.frac_bits
    if isinstance(value, str):
        if value not in _SPECIAL_FLOATS:
            raise ReplyError(f"{where}: {dtype} string values are {', '.join(_SPECIAL_FLOATS)}, not {value!r}")
        if value == "nan":
            return infinity | (1 << (fmt.frac_bits - 1))
        return infinity | ((1 << (fmt.width - 1)) if value == "-inf" else 0)
    try:
        number = float(value)
        if not math.isfinite(number):
            raise OverflowError
        return _round_bits(number, fmt)
    except OverflowError:
        raise ReplyError(f"{where}: {json.dumps(value)} is past the largest finite {dtype} value") from None


# ---------------------------------------------------------------------------
# The prompts


def render_messages(view: AdversaryView, prompts: str, shown: Sequence[tuple[Proposal, str]]) -> list[Message]:
    """Return the request's messages: one user message, prompt set `prompts`' propose.txt filled from `view` alone.

    `shown` pairs each proposal this adversary made, in order, with the JSON
    of its proposal object; outcome i shows that JSON when the outcome's
    proposal is shown[i]'s. Nothing outside `view` and `shown` is read, so no
    held-out input, item seed, or reference output reaches the prompt.
    """
    fields = {
        "contract": _contract_lines(view.contract),
        "source_files": _files(view.source),
        "candidate_files": _files(view.candidate),
        "budget_left": str(view.budget_left),
        "outcomes": _outcome_lines(view, shown),
    }
    return [Message(role="user", content=render(prompts, PROPOSE_PROMPT, fields))]


def _contract_lines(contract: InputContract) -> str:
    """Return one line per contract input: its name, dtype, shape, range, and whether NaN and infinity may pass."""
    lines = []
    for item in contract.inputs:
        dims = ", ".join(str(dim) for dim in item.shape)
        nan = "NaN allowed" if item.allow_nan else "NaN not allowed"
        inf = "infinity allowed" if item.allow_inf else "infinity not allowed"
        lines.append(
            f"- {item.name}: {item.dtype}, shape [{dims}], each element in [{item.lo!r}, {item.hi!r}); {nan}; {inf}"
        )
    return "\n".join(lines)


def _files(files: Mapping[str, str]) -> str:
    """Return the files as FILE blocks in sorted path order, or "None." for no file."""
    return render_file_blocks(dict(files)) if files else "None.\n"


def _outcome_lines(view: AdversaryView, shown: Sequence[tuple[Proposal, str]]) -> str:
    """Return one line per earlier outcome, each with this adversary's own JSON for it when it made it."""
    if not view.outcomes:
        return "None yet."
    lines = []
    for index, outcome in enumerate(view.outcomes):
        lines.append(f"- proposal {index}: {outcome.code}")
        if index < len(shown) and shown[index][0] == outcome.proposal:
            lines.append(f"  proposed: {shown[index][1]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The Agent


@register("Agent", "llm")
class LLMAdversary:
    """The llm adversary: a model's proposals, read strictly, within a budget of inputs, tokens, and steps.

    Built from the recipe's adversary section (module docstring); every key
    is a required choice, and a missing or bad one is a ValueError naming it.
    `backend` holds the built LLMBackend.
    """

    name: ClassVar[str] = "llm"
    capabilities: ClassVar[frozenset[str]] = frozenset({PROPOSES_INPUTS, REPORTS_LOG})
    config_keys: ClassVar[frozenset[str]] = frozenset(
        {"budget", "backend", "model", "sampling", "prompts", "recording", "recording_sha256"}
    )

    def __init__(
        self,
        *,
        budget: Mapping[str, Any] | None = None,
        backend: str | None = None,
        model: str | None = None,
        sampling: Mapping[str, Any] | None = None,
        prompts: str | None = None,
        recording: str | None = None,
        recording_sha256: str | None = None,
    ) -> None:
        """Check every setting and build the backend; a ValueError names the first bad or missing one."""
        self.budget = _budget(budget)
        self.model = _text(model, "adversary.model")
        self.sampling = _sampling(sampling)
        self.prompts = _prompts(prompts)
        self.backend = _backend(_text(backend, "adversary.backend"), self.model, recording, recording_sha256)
        self._shown: list[tuple[Proposal, str]] = []
        self._steps = self._prompt_tokens = self._completion_tokens = self._malformed = self._dropped = 0
        self._stop: str | None = None

    def propose(self, view: AdversaryView) -> list[Proposal]:
        """Return the model's next input sets, at most view.budget_left; none once the search has stopped."""
        if type(view) is not AdversaryView:
            raise TypeError(f"LLMAdversary.propose takes an AdversaryView, not {type(view).__name__}")
        if self._stop is not None or view.budget_left == 0:
            return []
        messages = render_messages(view, self.prompts, self._shown)
        while True:
            if self._steps >= self.budget["steps"]:
                return self._stopped("steps")
            if self._prompt_tokens + self._completion_tokens >= self.budget["tokens"]:
                return self._stopped("tokens")
            self._steps += 1
            try:
                completion = self.backend.complete(messages, self.sampling)
            except ContextExceeded:
                return self._stopped("context-exceeded")
            self._prompt_tokens += completion.prompt_tokens
            self._completion_tokens += completion.completion_tokens
            try:
                found = _parse(completion.text)
            except ReplyError as error:
                self._malformed += 1
                fields = {"reason": str(error), "budget_left": str(view.budget_left)}
                again = Message(role="user", content=render(self.prompts, MALFORMED_PROMPT, fields))
                messages = [*messages, Message(role="assistant", content=completion.text), again]
                continue
            if not found:
                return self._stopped("no-proposal")
            kept = found[: view.budget_left]
            self._dropped += len(found) - len(kept)
            self._shown += kept
            return [proposal for proposal, _ in kept]

    def log(self) -> AgentLog:
        """Return this trial's AgentLog: steps, token counts, malformed replies, dropped proposals, and the stop."""
        return AgentLog(
            steps=self._steps,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            malformed_replies=self._malformed,
            dropped_proposals=self._dropped,
            stop=self._stop,
        )

    def _stopped(self, stop: str) -> list[Proposal]:
        """Stop the search at `stop` and propose nothing."""
        self._stop = stop
        return []


def _budget(budget: object) -> dict[str, int]:
    """Return the checked budget mapping: inputs, tokens, and steps, each an int of at least 1."""
    if budget is None:
        raise ValueError("adversary.budget is a required choice with no default: set {inputs: N, tokens: T, steps: S}")
    if not isinstance(budget, Mapping):
        raise ValueError(f"adversary.budget must be a mapping {{inputs, tokens, steps}}, not {budget!r}")
    extra = sorted(str(key) for key in budget if key not in BUDGET_KEYS)
    if extra:
        raise ValueError(f"adversary.budget.{extra[0]} is not an llm budget key; it takes {', '.join(BUDGET_KEYS)}")
    found = {}
    for key in BUDGET_KEYS:
        value = budget.get(key)
        if value is None:
            raise ValueError(f"adversary.budget.{key} is a required choice with no default: set an int of at least 1")
        if type(value) is not int or value < 1:
            raise ValueError(f"adversary.budget.{key} must be an int of at least 1, not {value!r}")
        found[key] = value
    return found


def _sampling(sampling: object) -> Sampling:
    """Return the checked Sampling: temperature at least 0, top_p in (0, 1], and max_tokens an int of at least 1."""
    if sampling is None:
        raise ValueError(
            "adversary.sampling is a required choice with no default: set {temperature, top_p, max_tokens}"
        )
    if not isinstance(sampling, Mapping):
        raise ValueError(f"adversary.sampling must be a mapping {{temperature, top_p, max_tokens}}, not {sampling!r}")
    extra = sorted(str(key) for key in sampling if key not in SAMPLING_KEYS)
    if extra:
        raise ValueError(f"adversary.sampling.{extra[0]} is not a sampling key; it takes {', '.join(SAMPLING_KEYS)}")
    for key in SAMPLING_KEYS:
        if sampling.get(key) is None:
            raise ValueError(f"adversary.sampling.{key} is a required choice with no default")
    temperature, top_p, max_tokens = (sampling[key] for key in SAMPLING_KEYS)
    if not _number(temperature) or temperature < 0:
        raise ValueError(f"adversary.sampling.temperature must be a finite number of at least 0, not {temperature!r}")
    if not _number(top_p) or not 0 < top_p <= 1:
        raise ValueError(f"adversary.sampling.top_p must be a number in (0, 1], not {top_p!r}")
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError(f"adversary.sampling.max_tokens must be an int of at least 1, not {max_tokens!r}")
    return Sampling(temperature=float(temperature), top_p=float(top_p), max_tokens=max_tokens)


def _number(value: object) -> bool:
    """Return True for a finite int or float that is not a bool."""
    return type(value) in (int, float) and math.isfinite(value)


def _text(value: object, key: str) -> str:
    """Return `value` when it is a non-empty str; ValueError naming `key` otherwise."""
    if value is None:
        raise ValueError(f"{key} is a required choice with no default")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string, not {value!r}")
    return value


def _prompts(prompts: object) -> str:
    """Return the prompt set name when the set holds propose.txt and malformed.txt; ValueError naming the key."""
    name = _text(prompts, "adversary.prompts")
    try:
        folder = prompt_dir(name)
    except ValueError as error:
        raise ValueError(f"adversary.prompts: {error}") from None
    for prompt in (PROPOSE_PROMPT, MALFORMED_PROMPT):
        if not (folder / f"{prompt}.txt").is_file():
            raise ValueError(f"adversary.prompts {name!r} has no {prompt}.txt in {folder}")
    return name


def _backend(name: str, model: str, recording: object, recording_sha256: object) -> Any:
    """Build the scripted LLMBackend `name` for `model` over its checked recording; refuse any other backend."""
    try:
        entry = DEFAULT_REGISTRY.get("LLMBackend", name)
    except ValueError as error:
        raise ValueError(f"adversary.backend: {error}") from None
    if "needs_reference" in entry.capabilities:
        raise ValueError(
            f"adversary.backend {name!r} needs the item's reference target, which an adversary never gets; bind a "
            "backend that replies on its own, such as replay"
        )
    if "recording" not in entry.config_keys:
        raise ValueError(
            f"adversary.backend {name!r} replays no recording: the llm adversary binds a scripted backend only, such "
            "as replay; a served or in-process model is task P16.11 (OQ-068)"
        )
    if recording is None:
        if recording_sha256 is not None:
            raise ValueError("adversary.recording_sha256 is set without adversary.recording; name the recording")
        raise ValueError(f"adversary.recording is a required choice for LLMBackend {name!r}: name its recording")
    if recording_sha256 is None:
        raise ValueError(
            "adversary.recording_sha256 is a required choice with adversary.recording: the sha256 of the recording's "
            "bytes, which the recipe loader enters"
        )
    file = recording_file(recording, "adversary.recording")
    try:
        backend = entry.factory(model, recording=str(file))
    except OSError as error:
        raise ValueError(f"adversary.recording {recording!r} cannot be read ({error.strerror or error})") from None
    read = getattr(backend, "recording_sha256", None)
    if read != recording_sha256:
        raise ValueError(
            f"adversary.recording_sha256 is {recording_sha256}, but LLMBackend {name!r} read the bytes of "
            f"adversary.recording {recording!r} with the sha256 {read}; the recording changed since the recipe "
            "loaded, so restore those bytes or load the recipe again"
        )
    return backend
