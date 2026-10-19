"""The fuzzer adversary: in-contract input sets from boundary values, patterns, and fresh draws (task P16.3).

Bible: Adversarial Setups (item 1: an adversary, an LLM agent or a fuzzer,
proposes inputs within the item's input contract; its precedent, metamorphic
fuzzing and differential testing of translated HPC programs), Harness
Contract (held-out input files: each input's own PCG stream and the
rounding; Input contracts), Project Recipes (the bindings block's adversary
line), Component Interfaces (the adversary bullet), Agent Rules 1, 5, and 10.

Fuzzer is registered as the Agent `fuzzer` and declares proposes_inputs, so
a recipe's adversary section binds it: `adversary: {kind: fuzzer, budget:
{inputs: N}, seed: S}`. Both keys are required choices with no default and
enter the resolved recipe and its hash: budget.inputs, an int of at least 1,
is the most input sets it proposes in one trial, and seed, an int in
[0, 2**64), is the only source of its randomness. propose(view) reads only
its AdversaryView, which holds no held-out input, no item seed, and no
reference output. It never reads the source or the candidate text, and its
proposals are a function of its config and the contract alone: the trial's
position is the number of earlier outcomes, so the stage may ask for one set
at a time or many and gets the same sets.

Strategies [DESIGN, task P16.3]; every value is one the contract allows, so
each proposal is inside the contract by construction:

1. Boundary and special values. Each input has a list of values, each kept
   once, in this order, and each only when the contract allows it. A float
   dtype: its lowest value, its largest value below hi, +0 and -0, a quiet
   NaN (allow_nan), +infinity and -infinity (allow_inf), the smallest
   subnormal of each sign, the second-lowest value and the second-largest
   below hi, the smallest normal and the largest subnormal of each sign, +1
   and -1, then a negative quiet NaN and a signalling NaN (allow_nan). An
   integer dtype: lo, hi - 1, 0, lo + 1, hi - 2, 1, -1, the midpoint of
   [lo, hi - 1], and the edges of the narrow integer types, 2**b - 1, 2**b,
   -2**b, and -2**b - 1 for b of 7, 8, 15, 16, 31, and 32. Boundary
   proposal r fills every input with a constant array of its value r
   (cycling through a shorter list), so the constant arrays are these.
2. Patterned arrays, for a contract with an input of two or more elements
   that allows two or more values: lowest and largest below hi alternating;
   the input's boundary and special values in turn; a sweep of 16 evenly
   spaced values from the lowest to the largest below hi, or as many as the
   input has elements when fewer (at least 2), up and then down (for a
   float dtype, evenly spaced by ordering key, so the values spread over the
   binades between lo and hi rather than evenly by value);
   the lowest value everywhere with the largest below hi in the last
   element; then, without end, seeded mixes, each element one of the
   input's boundary and special values drawn from the input's own Pcg32
   stream, seeded from the fuzzer's seed. A pattern is repeated, or cut,
   to the input's size.
3. Fresh draws. Draw k is the input set generate_inputs writes for the
   contract read as a held-out input spec (each input's name, dtype, shape,
   lo, and hi, dist uniform for a float dtype and integers for an integer
   one) with the seed draw_seed(seed, k): each input's own PCG stream and
   the same rounding, the same bytes on every host. It draws over the
   contract's range, a declared one included, never a NaN or an infinity.

Schedule: the three strategies take turns, boundary, pattern, fresh draw,
until the longest boundary list among the inputs with elements is used up, then
pattern and fresh draw take turns; with no patterned arrays, boundary and
fresh draw take turns, then fresh draws alone. A trial ends at budget.inputs
sets.

Seeds (Agent Rule 5): a fresh draw's seed is draw_seed(seed, k), derived
from the fuzzer's own seed and k alone through sha256, never the seed
itself, and a seeded mix's seed comes the same way under another tag. The
item's seed is never read: the view holds none, and this module imports no
bench registry, trial record, stage, recipe, runner, or store. The fresh
draws share generate_inputs' own draw code and the boundary values
check_inputs' own key range (lassi.harness.lassi_io, through the private
helpers imported below), so neither can drift from the harness. Under the
P16 plan's Freeze decision this module, the seed, and the budget are frozen
before the first adversary run on a set. The proposals' bytes depend on
those lassi_io helpers as well as on this module, so the fuzzer's source
files, as the Freeze decision's record hashes them, are this file and
lassi/harness/lassi_io.py; the known-answer digest of
tests/agents/test_fuzzer.py pins the bytes the two give over one fixed
contract.

Known cases, not complete: the strategies bound each element alone, so no
proposal targets a relation between elements or inputs (a sorted array, an
index below another input's length); a NaN is one of three bit patterns,
never another payload; the fuzzer does not adapt to its outcomes, and a
contract that admits few input sets gets repeated ones, since no proposal
is checked against the earlier ones. Such a contract (each input of one or
two values, or with no elements) can also get a proposal equal to the
item's held-out inputs by chance: the fuzzer never sees them, so it cannot
avoid them. A budget below three times the longest boundary list (two
times with no patterned arrays) ends the trial before the later boundary
values get their own constant arrays: a full-range i64 vector at a budget
of 64 gets constant arrays of its first 22 boundary values of 31, so its
last nine edges, from -2**16 - 1 on, reach it only through the specials pattern, which
holds as many of them as the input has elements, and the seeded mixes.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from lassi.core.capabilities import PROPOSES_INPUTS
from lassi.core.interfaces import AdversaryView, Proposal
from lassi.core.registry import register
from lassi.harness.contract import ContractInput, InputContract
from lassi.harness.lassi_io import (
    _FLOATS,
    LassiArray,
    Pcg32,
    _from_key,
    _input_plans,
    _integer_data,
    _key_range,
    _shown,
    _uniform_data,
    stream_of,
)

_M64 = (1 << 64) - 1
# The struct code of one element: an integer dtype's value, or a float dtype's bit pattern by its width.
_INT_CODES = {"i8": "b", "u8": "B", "i32": "i", "u32": "I", "i64": "q"}
_BITS_CODES = {16: "H", 32: "I", 64: "Q"}
# The narrow integer types' bit counts whose edges join an integer input's boundary values.
_EDGE_BITS = (7, 8, 15, 16, 31, 32)
# The most values a sweep pattern takes, evenly spaced from the lowest to the largest below hi.
_SWEEP_POINTS = 16
# The fixed patterns, in their turn; seeded mixes follow them without end.
_ALTERNATING, _SPECIALS, _SWEEP_UP, _SWEEP_DOWN, _SPIKE = range(5)
_FIXED_PATTERNS = 5
# The strategies of the schedule, in their turn.
_BOUNDARY, _PATTERN, _FRESH = "boundary", "pattern", "fresh"
# The sha256 tags of the derived seeds: fresh draws and seeded mixes.
_FRESH_TAG = b"lassi.agents.fuzzer fresh draw"
_MIX_TAG = b"lassi.agents.fuzzer seeded mix"


@dataclass(frozen=True)
class _Input:
    """One contract input as the fuzzer fills it.

    `words` are its boundary and special values as elements: a value for an
    integer dtype and a bit pattern for a float dtype, each packed with
    `code`. `first` and `last` order its lowest value and its largest below
    hi (a value, or a float dtype's ordering key); `word` maps such an
    order to an element.
    """

    spec: ContractInput
    size: int
    code: str
    first: int
    last: int
    words: tuple[int, ...]

    def word(self, order: int) -> int:
        """Return the element of an order in [first, last]: the value itself, or the float dtype's bit pattern."""
        fmt = _FLOATS.get(self.spec.dtype)
        return order if fmt is None else _from_key(order, fmt)


def _unique(words: Iterable[int | None]) -> tuple[int, ...]:
    """Return the words other than None, each kept once, in their first order."""
    return tuple(dict.fromkeys(word for word in words if word is not None))


def _float_input(spec: ContractInput, size: int) -> _Input:
    """Return a float input's fill: its ordering keys' range and its boundary and special bit patterns."""
    fmt = _FLOATS[spec.dtype]
    first, last = _key_range(float(spec.lo), float(spec.hi), fmt)
    sign = 1 << (fmt.width - 1)
    infinity = ((1 << fmt.exp_bits) - 1) << fmt.frac_bits
    quiet = infinity | (1 << (fmt.frac_bits - 1))
    normal, one = 1 << fmt.frac_bits, fmt.bias << fmt.frac_bits

    def finite(key: int) -> int | None:
        return _from_key(key, fmt) if first <= key <= last else None

    nan = [quiet] if spec.allow_nan else []
    infinities = [infinity, sign | infinity] if spec.allow_inf else []
    words = [finite(first), finite(last), finite(0), sign if first <= 0 <= last else None, *nan, *infinities]
    words += [finite(1), finite(-1), finite(first + 1), finite(last - 1)]
    words += [finite(normal), finite(-normal), finite(normal - 1), finite(1 - normal), finite(one), finite(-one)]
    if spec.allow_nan:
        words += [sign | quiet, infinity | 1]
    return _Input(spec, size, _BITS_CODES[fmt.width], first, last, _unique(words))


def _int_input(spec: ContractInput, size: int) -> _Input:
    """Return an integer input's fill: [lo, hi - 1] and its boundary values that lie in it."""
    lo, last = int(spec.lo), int(spec.hi) - 1
    values = [lo, last, 0, lo + 1, last - 1, 1, -1, lo + (last - lo) // 2]
    for bits in _EDGE_BITS:
        values += [(1 << bits) - 1, 1 << bits, -(1 << bits), -(1 << bits) - 1]
    words = _unique(value if lo <= value <= last else None for value in values)
    return _Input(spec, size, _INT_CODES[spec.dtype], lo, last, words)


def _inputs(contract: InputContract) -> tuple[_Input, ...]:
    """Return the fill of every contract input, in the contract's order."""
    found = []
    for spec in contract.inputs:
        size = math.prod(spec.shape)
        found.append(_float_input(spec, size) if spec.dtype in _FLOATS else _int_input(spec, size))
    return tuple(found)


def _tile(words: Sequence[int], size: int, code: str) -> bytes:
    """Return `words` repeated to `size` elements, packed little-endian with struct code `code`."""
    if size == 0:
        return b""
    cycle = struct.pack(f"<{len(words)}{code}", *words)
    whole, part = divmod(size, len(words))
    return cycle * whole + cycle[: part * struct.calcsize(f"<{code}")]


def _derive(tag: bytes, seed: int, index: int) -> int:
    """Return a seed in [0, 2**64) from sha256 over `tag`, `seed`, and `index`, re-hashed while it equals `seed`."""
    counter = 0
    while True:
        message = b"%s|%d|%d|%d" % (tag, seed, index, counter)
        value = int.from_bytes(hashlib.sha256(message).digest()[:8], "little")
        if value != seed:
            return value
        counter += 1


def _check_seed(seed: object, label: str) -> int:
    """Return `seed` when it is an int in [0, 2**64) (a bool is not); else ValueError naming `label`."""
    if seed is None:
        raise ValueError(f"{label} is a required choice with no default: set an int in [0, 2**64)")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _M64:
        raise ValueError(f"{label} must be an int in [0, 2**64), not {_shown(seed)} ({type(seed).__name__})")
    return seed


def draw_seed(seed: int, index: int) -> int:
    """Return the seed of the fuzzer's fresh draw `index` of a trial, derived from its own `seed` and `index` alone.

    The result is an int in [0, 2**64), the first 8 bytes, little-endian, of
    sha256 over a tag, the seed, and the index, never the seed itself (the
    hash is taken again with a counter in that case). `seed` is an int in
    [0, 2**64) and `index` an int of at least 0; ValueError names a bad one.
    """
    _check_seed(seed, "seed")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"index must be an int of at least 0, not {_shown(index)}")
    return _derive(_FRESH_TAG, seed, index)


def _entry(position: int, boundaries: int, patterned: bool) -> tuple[str, int]:
    """Return the strategy of a trial position and that strategy's own index (the module's Schedule)."""
    streams = (_BOUNDARY, _PATTERN, _FRESH) if patterned else (_BOUNDARY, _FRESH)
    if position < len(streams) * boundaries:
        return streams[position % len(streams)], position // len(streams)
    rest = streams[1:]
    offset = position - len(streams) * boundaries
    return rest[offset % len(rest)], boundaries + offset // len(rest)


def _boundary_data(item: _Input, index: int) -> bytes:
    """Return an input's boundary proposal `index`: a constant array of its value `index`, cycling."""
    return _tile((item.words[index % len(item.words)],), item.size, item.code)


def _pattern_data(item: _Input, index: int, seed: int) -> bytes:
    """Return an input's pattern `index`: a fixed pattern (the module's Strategies, 2), then seeded mixes."""
    if item.size == 0:
        return b""
    low, high = item.word(item.first), item.word(item.last)
    if index == _ALTERNATING:
        return _tile((low, high), item.size, item.code)
    if index == _SPECIALS:
        return _tile(item.words, item.size, item.code)
    if index in (_SWEEP_UP, _SWEEP_DOWN):
        span, points = item.last - item.first, max(2, min(_SWEEP_POINTS, item.size))
        sweep = [item.word(item.first + span * point // (points - 1)) for point in range(points)]
        return _tile(sweep if index == _SWEEP_UP else sweep[::-1], item.size, item.code)
    if index == _SPIKE:
        return _tile((low,), item.size - 1, item.code) + _tile((high,), min(item.size, 1), item.code)
    rng = Pcg32(_derive(_MIX_TAG, seed, index - _FIXED_PATTERNS), stream_of(item.spec.name))
    count = len(item.words)
    mixed = [item.words[(rng.next_u32() * count) >> 32] for _ in range(item.size)]
    return struct.pack(f"<{item.size}{item.code}", *mixed)


def fresh_draw(contract: InputContract, seed: int) -> tuple[LassiArray, ...]:
    """Return the arrays generate_inputs writes for `contract` read as a held-out input spec and `seed`.

    The spec holds each input's name, dtype, shape, lo, and hi, with dist
    uniform for a float dtype and integers for an integer one; the arrays
    are drawn by generate_inputs' own code, in the contract's order, and
    nothing is written.
    """
    spec = [
        {
            "name": item.name,
            "dtype": item.dtype,
            "shape": list(item.shape),
            "dist": "uniform" if item.dtype in _FLOATS else "integers",
            "lo": item.lo,
            "hi": item.hi,
        }
        for item in contract.inputs
    ]
    arrays = []
    for plan in _input_plans("fuzzer fresh draw", spec):
        rng = Pcg32(seed, stream_of(plan.name))
        data = _uniform_data(plan, rng) if plan.dist == "uniform" else _integer_data(plan, rng)
        arrays.append(LassiArray(plan.name, plan.dtype.name, plan.dims, data))
    return tuple(arrays)


@register("Agent", "fuzzer")
class Fuzzer:
    """The fuzzer adversary: boundary and special values, constant and patterned arrays, and fresh draws.

    Built from the recipe section `adversary: {kind: fuzzer, budget:
    {inputs: N}, seed: S}` as Fuzzer(budget={"inputs": N}, seed=S). Both are
    required choices with no default: None, a missing key, is refused like
    any other bad value, with a ValueError naming adversary.budget.inputs,
    adversary.budget, another key under it, or adversary.seed. Its
    strategies, the boundary and special values of each input as constant
    arrays, patterned arrays, and fresh draws through generate_inputs'
    streams from seeds derived from its own seed, take turns as the module
    docstring's Schedule says. Known cases, not complete: the module
    docstring lists what no strategy covers.
    """

    name: ClassVar[str] = "fuzzer"
    capabilities: ClassVar[frozenset[str]] = frozenset({PROPOSES_INPUTS})
    config_keys: ClassVar[frozenset[str]] = frozenset({"budget", "seed"})

    def __init__(self, *, budget: Mapping[str, Any] | None = None, seed: int | None = None) -> None:
        """Keep budget.inputs and the seed, both required; a ValueError names a bad or missing one."""
        self.inputs = _budget_inputs(budget)
        self.seed = _check_seed(seed, "adversary.seed")

    def propose(self, view: AdversaryView) -> list[Proposal]:
        """Return the trial's next input sets: at most view.budget_left, and none past budget.inputs in the trial.

        The trial's position is len(view.outcomes); the sets are the
        schedule's entries from there, a function of the config and the
        contract alone. An empty result proposes nothing more.
        """
        if type(view) is not AdversaryView:
            raise TypeError(f"Fuzzer.propose takes an AdversaryView, not {type(view).__name__}")
        start = len(view.outcomes)
        count = max(0, min(view.budget_left, self.inputs - start))
        if count == 0:
            return []
        items = _inputs(view.contract)
        filled = [item for item in items if item.size > 0]
        boundaries = max((len(item.words) for item in filled), default=1)
        patterned = any(item.size >= 2 and len(item.words) >= 2 for item in items)
        found = []
        for position in range(start, start + count):
            strategy, index = _entry(position, boundaries, patterned)
            if strategy == _FRESH:
                found.append(Proposal(fresh_draw(view.contract, draw_seed(self.seed, index))))
            else:
                found.append(Proposal(tuple(self._array(item, strategy, index) for item in items)))
        return found

    def _array(self, item: _Input, strategy: str, index: int) -> LassiArray:
        """Return one input's array of a boundary or pattern proposal `index`."""
        if strategy == _BOUNDARY:
            data = _boundary_data(item, index)
        else:
            data = _pattern_data(item, index, self.seed)
        return LassiArray(item.spec.name, item.spec.dtype, item.spec.shape, data)


def _budget_inputs(budget: object) -> int:
    """Return budget.inputs from the adversary section's budget mapping; ValueError names what is wrong."""
    if budget is None:
        raise ValueError("adversary.budget.inputs is a required choice with no default: set an int of at least 1")
    if not isinstance(budget, Mapping):
        raise ValueError(f"adversary.budget must be a mapping {{inputs: N}}, not {_shown(budget)}")
    extra = sorted(str(key) for key in budget if key != "inputs")
    if extra:
        raise ValueError(f"adversary.budget.{extra[0]} is not a fuzzer budget key; the budget takes inputs only")
    inputs = budget.get("inputs")
    if inputs is None:
        raise ValueError("adversary.budget.inputs is a required choice with no default: set an int of at least 1")
    if isinstance(inputs, bool) or not isinstance(inputs, int) or inputs < 1:
        raise ValueError(f"adversary.budget.inputs must be an int of at least 1, not {_shown(inputs)}")
    return inputs
