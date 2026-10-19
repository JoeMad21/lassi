"""Tests for the fuzzer adversary: a registered Agent that proposes in-contract input sets from its own seed (P16.3).

Bible: Adversarial Setups (item 1: an adversary, an LLM agent or a fuzzer,
proposes inputs within the item's input contract; its precedent, metamorphic
fuzzing and differential testing of translated HPC programs; the adversary
rule), Harness Contract (held-out input files: each input's own PCG stream,
the rounding, the same bytes on every host; Input contracts), Project
Recipes (the bindings block's adversary line: a fuzzer with a budget of 64
input sets and, from task P16.3, a seed), Component Interfaces (the
adversary bullet, task P16.2), Agent Rules 1, 5, and 10. Plan:
plans/p16-adversarial.md, P16.3, and its Freeze decision (the fuzzer's
code, seed, and budget are frozen before P16.10's first adversary run).

The contract these tests fix:

- lassi.agents.fuzzer defines Fuzzer, registered in DEFAULT_REGISTRY as the
  Agent "fuzzer" when lassi.agents is imported. It declares the capability
  proposes_inputs, so a recipe's adversary section binds it at
  adversary.kind, and accepts exactly the config keys budget and seed.
- Fuzzer(budget={"inputs": N}, seed=S) takes both as required choices with
  no default. A missing one is refused with a ValueError naming
  budget.inputs or seed, and so is a value that is not one: a budget that
  is not a mapping, a budget.inputs that is not an int of at least 1 (a
  bool is not one), any other key under budget (named), and a seed that is
  not an int in [0, 2**64).
- propose(view) reads only its AdversaryView, which holds no held-out
  input, no item seed, and no reference output. One call returns at most
  view.budget_left Proposals, and over a trial the fuzzer proposes at most
  budget.inputs, however the stage splits its budget across calls (the
  trial's earlier proposals are view.outcomes).
- Every proposal is inside the contract by construction: it passes
  check_inputs over a grid of contracts (GRID: every dtype; rank 0, empty,
  and multi-dim shapes; declared ranges; allow_nan and allow_inf) and seeds
  (SEEDS): each grid contract with one of the four seeds in turn, and,
  marked slow, each contract of MULTI_SEED_CASES with every seed.
- Its strategies each appear within the bindings block's budget of 64:
  boundary and special values the contract allows (for each input with an
  element: its lowest value and its largest value below hi; for a float
  dtype, +0 and -0 when 0 lies in [lo, hi), the smallest subnormal of each
  sign that lies in it, a NaN when allow_nan is set, and both infinities
  when allow_inf is set; for an integer dtype, 0 when it lies in [lo, hi));
  constant arrays (an input of two or more elements, all one bit pattern);
  patterned arrays (an input of two or more elements holding two or more
  bit patterns, in a proposal that is not a fresh draw); and fresh draws,
  the first two among them.
- A fresh draw is the input set generate_inputs writes for the contract
  read as a held-out input spec (fuzzer_support.draw_spec) with the seed
  draw_seed(seed, k), for the fuzzer's k-th fresh draw of a trial, counted
  from 0. lassi.agents.fuzzer.draw_seed(seed, index) derives it from the
  fuzzer's own seed and the index alone: an int in [0, 2**64), never the
  seed itself, distinct over indices and over seeds. An item's seed is
  never read: the view holds none, and the module imports neither the
  bench registry nor the trial record, the stages, the recipe, or the
  runner. tests/agents/test_fuzzer_held_out.py checks the derived seeds
  against every manifest item's seed and the proposals against
  tt-pairs-v0's held-out inputs.
- Later fresh draws keep to the same rule. A long trial (LONG_BUDGET, four
  times the bindings block's 64) over each contract of FAR_DRAW_CASES, the
  strategy cases and two wide scalar contracts, whose schedule with no
  patterns turns to fresh draws alone, holds fresh draws 0 to 63, each
  generate_inputs with draw_seed(seed, k); and none of its proposals is the
  input set generate_inputs writes for the contract with the fuzzer's own
  seed or with any manifest item's seed (fuzzer_support.manifest_seeds), so
  no draw of the trial reuses either.
- The proposals are a function of the config and the view: the same from a
  new object and on a second trial, the same when the stage asks for one
  set at a time or several, and the same under another PYTHONHASHSEED. One
  seed gives the same bytes on every host: KNOWN_ANSWER_SHA256 pins them
  over a fixed contract (KAT_SPEC, KAT_DECLARED, KAT_SEED, a budget of
  LONG_BUDGET, so the digest also pins the schedule's later patterns and
  fresh draws).
- The module and class docstrings list the strategies, as known cases, not
  complete.

Values: KNOWN_ANSWER_SHA256 is a known-answer value of a deterministic
function, not a measurement, recorded from the first run in which every
other test here passed. A value starting with PLACEHOLDER fails its test,
printing the digest the run gives. A change to the fuzzer's strategies
changes it, and under the Freeze decision such a change after the first
adversary run is measured only on a fresh seeded set. It was recorded on
Windows only; "every host" is seen once this file also passes in the local
distro, which P16.10's freeze record waits for. Every contract,
program text, and seed here is SYNTHETIC. No value in this module is a
measurement.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import math
import os
import struct
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fuzzer_support import (
    BUDGET,
    FLOATS,
    INTS,
    ArraysKey,
    arrays_key,
    build,
    drive,
    fresh_draw,
    fuzzer_module,
    manifest_seeds,
    view,
)

from lassi.core.interfaces import Proposal
from lassi.core.registry import DEFAULT_REGISTRY, Binding, RegistryError, check_bindings
from lassi.harness import (
    ContractInput,
    InputContract,
    InputContractError,
    LassiIOError,
    check_inputs,
    input_contract,
    write_array,
)

REPO = Path(__file__).resolve().parents[2]
SOURCE_FILE = REPO / "lassi" / "agents" / "fuzzer.py"
KIND = "fuzzer"
CAPABILITY = "proposes_inputs"
CONFIG_KEYS = frozenset({"budget", "seed"})
SEEDS = (0, 1, 12345, 2**64 - 1)
# Modules through which an item, its seed, or its held-out inputs could be read; the fuzzer imports none of them.
FORBIDDEN_IMPORTS = (
    "lassi.bench",
    "lassi.core.record",
    "lassi.core.stages",
    "lassi.core.recipe",
    "lassi.core.runner",
    "lassi.core.store",
)
STRATEGY_WORDS = ("boundary", "special", "constant", "pattern", "fresh draw", "known cases, not complete")

WIDTH = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64, "i8": 8, "u8": 8, "i32": 32, "u32": 32, "i64": 64}
FRACTION = {"f16": 10, "bf16": 7, "f32": 23, "f64": 52}
# The largest finite value of each float dtype, except f64, whose [-max, max) is wider than a double holds.
TOP = {"f16": 65504.0, "bf16": (2 - 2**-7) * 2.0**127, "f32": (2 - 2**-23) * 2.0**127, "f64": 2.0**1000}
# A subnormal of each float dtype, exact in it, and its smallest positive subnormal.
SUBNORMAL = {"f16": 2.0**-20, "bf16": 2.0**-130, "f32": 2.0**-140, "f64": 2.0**-1070}
SMALLEST = {"f16": 2.0**-24, "bf16": 2.0**-133, "f32": 2.0**-149, "f64": 2.0**-1074}
INT_RANGE = {"i8": (-128, 128), "u8": (0, 256), "i32": (-(2**31), 2**31), "u32": (0, 2**32), "i64": (-(2**63), 2**63)}

# ---------------------------------------------------------------------------
# The grid of SYNTHETIC contracts; every float bound is exact in its dtype (exact_bits checks it).


def entry(name: str, dtype: str, shape: tuple[int, ...], lo: float | int, hi: float | int) -> dict[str, Any]:
    """Return one held-out input spec entry (a generate_inputs entry)."""
    dist = "uniform" if dtype in FLOATS else "integers"
    return {"name": name, "dtype": dtype, "shape": list(shape), "dist": dist, "lo": lo, "hi": hi}


def declare(name: str, dtype: str, lo: float | int, hi: float | int, **flags: bool) -> dict[str, Any]:
    """Return one entry of a declared `contract` manifest key."""
    return {"name": name, "dtype": dtype, "lo": lo, "hi": hi, **flags}


@dataclass(frozen=True)
class Case:
    """One contract of the grid: its id, its held-out input spec, and its declared contract key, if any."""

    id: str
    spec: tuple[dict[str, Any], ...]
    declared: tuple[dict[str, Any], ...] | None = None

    @property
    def contract(self) -> InputContract:
        """Return the contract input_contract reads from the spec and the declaration."""
        declared = None if self.declared is None else list(self.declared)
        return input_contract(list(self.spec), declared, where=self.id)


def float_cases(dtype: str) -> list[Case]:
    """Return the grid's contracts of one float dtype."""
    top, tiny = TOP[dtype], SUBNORMAL[dtype]
    return [
        Case(f"{dtype}-scalar", (entry("x", dtype, (), -1.0, 1.0),)),
        Case(f"{dtype}-vector-lo-zero", (entry("x", dtype, (5,), 0.0, 1.0),)),
        Case(f"{dtype}-matrix-hi-zero", (entry("x", dtype, (2, 3), -2.0, 0.0),)),
        Case(f"{dtype}-empty-multi-dim", (entry("x", dtype, (3, 0, 2), 0.0, 4.0),)),
        Case(f"{dtype}-subnormals", (entry("x", dtype, (4,), -tiny, tiny),)),
        Case(f"{dtype}-finite-range", (entry("x", dtype, (6,), -top, top),)),
        Case(
            f"{dtype}-declared-nan",
            (entry("x", dtype, (4,), 0.5, 1.0),),
            (declare("x", dtype, -8.0, 8.0, allow_nan=True),),
        ),
        Case(
            f"{dtype}-declared-inf",
            (entry("x", dtype, (2, 2), 0.0, 1.0),),
            (declare("x", dtype, 0.0, 1.0, allow_inf=True),),
        ),
        Case(
            f"{dtype}-declared-scalar-nan-inf",
            (entry("x", dtype, (), 1.0, 2.0),),
            (declare("x", dtype, -4.0, 4.0, allow_nan=True, allow_inf=True),),
        ),
    ]


def int_cases(dtype: str) -> list[Case]:
    """Return the grid's contracts of one integer dtype."""
    low, high = INT_RANGE[dtype]
    signed = low < 0
    return [
        Case(f"{dtype}-scalar-full-range", (entry("n", dtype, (), low, high),)),
        Case(f"{dtype}-vector", (entry("n", dtype, (5,), 0, 10),)),
        Case(f"{dtype}-matrix", (entry("n", dtype, (2, 2), -3 if signed else 1, 3 if signed else 4),)),
        Case(f"{dtype}-empty", (entry("n", dtype, (0,), 0, 2),)),
        Case(
            f"{dtype}-extremes",
            (entry("bottom", dtype, (3, 4), low, low + 2), entry("top", dtype, (2,), high - 2, high)),
        ),
        Case(
            f"{dtype}-declared",
            (entry("n", dtype, (3,), 0, 4),),
            (declare("n", dtype, -100 if signed else 0, 100 if signed else 200),),
        ),
    ]


MIXED = (
    Case(
        "mixed",
        (
            entry("x", "f32", (3,), -1.0, 1.0),
            entry("n", "i32", (), 0, 100),
            entry("w", "bf16", (2, 2), 0.0, 1.0),
            entry("e", "u8", (0,), 0, 1),
        ),
    ),
    Case(
        "mixed-declared",
        (entry("p", "f64", (2,), 0.0, 1.0), entry("q", "f16", (3,), -1.0, 1.0), entry("k", "i64", (2,), -5, 5)),
        (
            declare("p", "f64", 0.0, 1.0, allow_nan=True),
            declare("q", "f16", -2.0, 2.0, allow_inf=True),
            declare("k", "i64", -5, 5),
        ),
    ),
    # Each input holds one value only: i32 7, and the f32 1.0 (the next f32 is 1 + 2**-23, hi itself).
    Case("single-values", (entry("c", "i32", (3,), 7, 8), entry("one", "f32", (), 1.0, 1.0 + 2.0**-23))),
)
GRID = tuple([case for d in FLOATS for case in float_cases(d)] + [case for d in INTS for case in int_cases(d)])
GRID += MIXED
CASES = {case.id: case for case in GRID}
# Contracts where every strategy can show: each holds an input of two or more elements, and every input's value
# space is wide (at least 200**3 input sets), so no patterned set coincides with one of 64 fresh draws by chance.
STRATEGY_CASES = (
    "f32-vector-lo-zero",
    "f16-matrix-hi-zero",
    "bf16-finite-range",
    "f64-declared-nan",
    "f32-declared-inf",
    "i8-declared",
    "u8-declared",
    "i32-declared",
    "u32-declared",
    "i64-declared",
    "mixed",
    "mixed-declared",
)
# Contracts checked at every seed (slow): several inputs, declared flags, subnormals, and a full integer range.
MULTI_SEED_CASES = ("mixed", "mixed-declared", "f16-subnormals", "i64-scalar-full-range")
DETERMINISM_CASES = ("mixed", "mixed-declared", "f16-subnormals")
# The budget of a long trial: four times the bindings block's, so every trial over FAR_DRAW_CASES reaches fresh
# draws far past the first two (at least FAR_DRAWS of them; the long-trial tests check it).
LONG_BUDGET = 4 * BUDGET
FAR_DRAWS = BUDGET
# Contracts whose long trial is checked draw by draw: the strategy cases, and two scalar contracts with wide value
# spaces whose schedule, with no patterns, turns to fresh draws alone once their boundary values are used up.
FAR_DRAW_CASES = (*STRATEGY_CASES, "f32-scalar", "i64-scalar-full-range")

# The fixed contract of the known-answer digest: six inputs, every float dtype, two integer dtypes, a declaration.
KAT_SPEC = (
    entry("x", "f32", (2, 3), -1.0, 1.0),
    entry("h", "f16", (4,), 0.0, 2.0),
    entry("b", "bf16", (), -0.5, 0.5),
    entry("d", "f64", (3,), 0.0, 1.0),
    entry("n", "i32", (), -8, 8),
    entry("u", "u8", (2, 2), 0, 200),
)
KAT_DECLARED = (
    declare("x", "f32", -4.0, 4.0, allow_nan=True, allow_inf=True),
    declare("h", "f16", 0.0, 2.0),
    declare("b", "bf16", -0.5, 0.5),
    declare("d", "f64", 0.0, 1.0, allow_nan=True),
    declare("n", "i32", -8, 8),
    declare("u", "u8", 0, 200),
)
KAT_SEED = 20261019
# The known-answer sha256 of the KAT proposals (digest()), recorded from the first run in which every other test here
# passed (task P16.3): a known-answer value of a deterministic function, not a measurement.
KNOWN_ANSWER_SHA256 = "c307d33a7cbc4b6fe00ec562fbd1166b69eb957c63d525a847b21a41d197d9b5"


def seed_for(case_id: str) -> int:
    """Return the seed the fast tests give a grid contract: the seeds in turn, over the grid's order."""
    return SEEDS[list(CASES).index(case_id) % len(SEEDS)]


_PROPOSALS: dict[tuple[str, int, int], list[Proposal]] = {}


def proposals_of(case_id: str, seed: int, budget: int = BUDGET) -> list[Proposal]:
    """Return one trial's proposals of a Fuzzer with `budget` (64 unless given) and `seed` over a grid contract.

    The trial is cached by contract, seed, and budget.
    """
    key = (case_id, seed, budget)
    if key not in _PROPOSALS:
        _PROPOSALS[key] = drive(build(budget, seed), CASES[case_id].contract, budget=budget)
    return _PROPOSALS[key]


# ---------------------------------------------------------------------------
# Reading elements, independently of lassi.harness


def bit_patterns(dtype: str, data: bytes) -> list[int]:
    """Return the elements of `data` as unsigned bit patterns of the dtype's width."""
    width = WIDTH[dtype]
    code = {8: "B", 16: "H", 32: "I", 64: "Q"}[width]
    return list(struct.unpack(f"<{len(data) * 8 // width}{code}", data))


def element_value(dtype: str, bits: int) -> float | int:
    """Return the value of one element's bit pattern: a two's-complement int, or the float it encodes."""
    width = WIDTH[dtype]
    if dtype in INTS:
        return bits - (1 << width) if dtype.startswith("i") and bits >> (width - 1) else bits
    if dtype == "bf16":
        return struct.unpack("<f", (bits << 16).to_bytes(4, "little"))[0]
    code = {"f16": "<e", "f32": "<f", "f64": "<d"}[dtype]
    return struct.unpack(code, bits.to_bytes(width // 8, "little"))[0]


def exact_bits(dtype: str, value: float) -> int:
    """Return the bit pattern of `value` in a float dtype, failing unless `value` is exact there."""
    if dtype == "bf16":
        bits = struct.unpack("<I", struct.pack("<f", value))[0] >> 16
    else:
        pack, unpack = {"f16": ("<e", "<H"), "f32": ("<f", "<I"), "f64": ("<d", "<Q")}[dtype]
        bits = struct.unpack(unpack, struct.pack(pack, value))[0]
    assert element_value(dtype, bits) == value, f"{value!r} is not exact in {dtype}; the grid's bounds are exact"
    return bits


def order_key(dtype: str, bits: int) -> int:
    """Return an int that orders a float dtype's finite bit patterns as their values (both zeros give 0)."""
    sign = 1 << (WIDTH[dtype] - 1)
    return -(bits & (sign - 1)) if bits & sign else bits


def from_order_key(dtype: str, key: int) -> int:
    """Return the bit pattern of an order key (0 gives +0)."""
    return key if key >= 0 else (1 << (WIDTH[dtype] - 1)) | -key


def value_test(dtype: str, value: float | int) -> Callable[[int], bool]:
    """Return a test that an element holds `value` (a NaN holds none; -0 and +0 hold 0)."""

    def holds(bits: int) -> bool:
        return element_value(dtype, bits) == value

    return holds


def bits_test(pattern: int) -> Callable[[int], bool]:
    """Return a test that an element is exactly the bit pattern `pattern`."""

    def holds(bits: int) -> bool:
        return bits == pattern

    return holds


def nan_test(dtype: str) -> Callable[[int], bool]:
    """Return a test that an element is a NaN of a float dtype, of either sign and any payload."""
    sign = 1 << (WIDTH[dtype] - 1)
    infinity = ((1 << (WIDTH[dtype] - 1 - FRACTION[dtype])) - 1) << FRACTION[dtype]

    def holds(bits: int) -> bool:
        return bits & (sign - 1) > infinity

    return holds


def wanted(spec: ContractInput) -> dict[str, Callable[[int], bool]]:
    """Return the boundary and special values one input must hold in some proposal, each as a test on an element.

    Nothing for an input without elements. An integer input: its lowest
    value lo, its largest value hi - 1, and 0 when [lo, hi) holds it. A float
    input: its lowest value lo, its largest value below hi, +0 and -0 when 0
    lies in [lo, hi), the smallest subnormal of each sign that lies in it, a
    NaN when allow_nan is set, and +infinity and -infinity when allow_inf is.
    """
    dtype = spec.dtype
    if math.prod(spec.shape) == 0:
        return {}
    if dtype in INTS:
        found = {"lowest": value_test(dtype, spec.lo), "largest below hi": value_test(dtype, spec.hi - 1)}
        if spec.lo <= 0 < spec.hi:
            found["zero"] = value_test(dtype, 0)
        return found
    sign = 1 << (WIDTH[dtype] - 1)
    below_hi = from_order_key(dtype, order_key(dtype, exact_bits(dtype, float(spec.hi))) - 1)
    found = {
        "lowest": value_test(dtype, element_value(dtype, exact_bits(dtype, float(spec.lo)))),
        "largest below hi": bits_test(below_hi),
    }
    if spec.lo <= 0.0 < spec.hi:
        found["+0"] = bits_test(0)
        found["-0"] = bits_test(sign)
    if spec.lo <= SMALLEST[dtype] < spec.hi:
        found["smallest positive subnormal"] = bits_test(1)
    if spec.lo <= -SMALLEST[dtype] < spec.hi:
        found["smallest negative subnormal"] = bits_test(sign | 1)
    infinity = ((1 << (WIDTH[dtype] - 1 - FRACTION[dtype])) - 1) << FRACTION[dtype]
    if spec.allow_nan:
        found["a NaN"] = nan_test(dtype)
    if spec.allow_inf:
        found["+infinity"] = bits_test(infinity)
        found["-infinity"] = bits_test(sign | infinity)
    return found


def matching_patterns(spec: ContractInput, proposals: Sequence[Proposal]) -> list[list[int]]:
    """Return, per proposal, the bit patterns of its array named and typed as `spec` (empty when it has none)."""
    return [
        [bits for array in proposal.arrays if (array.name, array.dtype) == (spec.name, spec.dtype)
         for bits in bit_patterns(spec.dtype, array.data)]
        for proposal in proposals
    ]


def constant_inputs(proposal: Proposal) -> list[str]:
    """Return the names of the proposal's arrays of two or more elements that hold one bit pattern."""
    return [
        array.name for array in proposal.arrays
        if len(patterns := bit_patterns(array.dtype, array.data)) >= 2 and len(set(patterns)) == 1
    ]


def patterned_inputs(proposal: Proposal) -> list[str]:
    """Return the names of the proposal's arrays of two or more elements that hold two or more bit patterns."""
    return [array.name for array in proposal.arrays if len(set(bit_patterns(array.dtype, array.data))) >= 2]


def refusals(contract: InputContract, proposals: Sequence[Proposal], root: Path) -> dict[int, str]:
    """Write each proposal as lassi_io files in `root` and check it; return each refusal by proposal index.

    Files are named by proposal and position, never by array name, so a
    repeated input stays two files and check_inputs sees it.
    """
    found: dict[int, str] = {}
    for index, proposal in enumerate(proposals):
        paths = []
        try:
            for position, array in enumerate(proposal.arrays):
                path = root / f"{index}-{position}.lassiio"
                write_array(path, array.name, array.dtype, array.shape, array.data)
                paths.append(path)
            check_inputs(contract, paths)
        except InputContractError as error:
            found[index] = f"{error.reason}: {error}"
        except LassiIOError as error:
            found[index] = f"not a lassi_io array: {error}"
    return found


def digest(contract: InputContract, proposals: Sequence[Proposal]) -> str:
    """Return the sha256 over the proposals in order, each array in the contract's order: header line, then bytes."""
    hasher = hashlib.sha256()
    for index, proposal in enumerate(proposals):
        arrays = {array.name: array for array in proposal.arrays}
        for name in contract.names:
            array = arrays[name]
            header = f"{index} {array.name} {array.dtype} {list(array.shape)} {len(array.data)}\n"
            hasher.update(header.encode("ascii"))
            hasher.update(array.data)
    return hasher.hexdigest()


def kat_digest() -> str:
    """Return the digest of one trial of a Fuzzer with a budget of LONG_BUDGET and KAT_SEED over the KAT contract."""
    contract = input_contract(list(KAT_SPEC), list(KAT_DECLARED), where="kat")
    return digest(contract, drive(build(LONG_BUDGET, KAT_SEED), contract, budget=LONG_BUDGET))


@pytest.fixture(scope="module")
def draws(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str, int, int], list[ArraysKey]]:
    """Return first(case_id, seed, count): the fuzzer's first `count` fresh draws over a grid contract (cached)."""
    root = tmp_path_factory.mktemp("fresh-draws")
    cache: dict[tuple[str, int], list[ArraysKey]] = {}

    def first(case_id: str, seed: int, count: int) -> list[ArraysKey]:
        made = cache.setdefault((case_id, seed), [])
        draw_seed = fuzzer_module().draw_seed
        while len(made) < count:
            directory = root / f"{case_id}-{seed}-{len(made)}"
            made.append(fresh_draw(CASES[case_id].contract, draw_seed(seed, len(made)), directory))
        return made[:count]

    return first


# ---------------------------------------------------------------------------
# Registration, binding, and config


def test_the_registered_agent_fuzzer_is_the_fuzzer_and_declares_proposes_inputs() -> None:
    module = fuzzer_module()
    found = DEFAULT_REGISTRY.get("Agent", KIND)
    assert found.factory is module.Fuzzer
    assert CAPABILITY in found.capabilities
    assert "takes_device" not in found.capabilities
    assert found.config_keys == CONFIG_KEYS


def test_importing_lassi_agents_registers_the_fuzzer() -> None:
    code = (
        "import lassi.agents\n"
        "from lassi.core.registry import DEFAULT_REGISTRY\n"
        "factory = DEFAULT_REGISTRY.get('Agent', 'fuzzer').factory\n"
        "print(factory.__module__, factory.__name__)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=300)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["lassi.agents.fuzzer", "Fuzzer"]


def test_the_adversary_section_binds_it_with_a_budget_and_a_seed_and_refuses_another_key() -> None:
    fuzzer_module()
    needs = frozenset({CAPABILITY})
    config = {"budget": {"inputs": BUDGET}, "seed": 7}
    check_bindings([Binding("Agent", KIND, "adversary.kind", config, needs)], DEFAULT_REGISTRY)
    extra = Binding("Agent", KIND, "adversary.kind", {**config, "seeds": 7}, needs)
    with pytest.raises(RegistryError, match=r"adversary\.seeds"):
        check_bindings([extra], DEFAULT_REGISTRY)


def test_it_takes_only_its_budget_and_seed_and_propose_reads_only_the_view() -> None:
    module = fuzzer_module()
    assert set(inspect.signature(module.Fuzzer).parameters) == CONFIG_KEYS
    assert list(inspect.signature(module.Fuzzer.propose).parameters) == ["self", "view"]


@pytest.mark.parametrize(
    "config",
    [{"seed": 7}, {"budget": None, "seed": 7}, {"budget": {}, "seed": 7}, {"budget": {"inputs": None}, "seed": 7}],
    ids=["no-budget", "null-budget", "empty-budget", "null-inputs"],
)
def test_a_missing_budget_inputs_is_refused_naming_it(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"budget\.inputs"):
        fuzzer_module().Fuzzer(**config)


@pytest.mark.parametrize(
    "config", [{"budget": {"inputs": 4}}, {"budget": {"inputs": 4}, "seed": None}], ids=["no-seed", "null-seed"]
)
def test_a_missing_seed_is_refused_naming_it(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"\bseed\b"):
        fuzzer_module().Fuzzer(**config)


@pytest.mark.parametrize("value", [True, 0, -1, 2.0, "64", [64]], ids=repr)
def test_a_budget_inputs_that_is_not_an_int_of_at_least_1_is_refused_naming_it(value: Any) -> None:
    with pytest.raises(ValueError, match=r"budget\.inputs"):
        fuzzer_module().Fuzzer(budget={"inputs": value}, seed=7)


@pytest.mark.parametrize("value", [64, "64", [64]], ids=repr)
def test_a_budget_that_is_not_a_mapping_is_refused_naming_it(value: Any) -> None:
    with pytest.raises(ValueError, match=r"\bbudget\b"):
        fuzzer_module().Fuzzer(budget=value, seed=7)


def test_another_key_under_budget_is_refused_naming_it() -> None:
    with pytest.raises(ValueError, match=r"budget\.tokens"):
        fuzzer_module().Fuzzer(budget={"inputs": 4, "tokens": 100}, seed=7)


@pytest.mark.parametrize("value", [True, -1, 2**64, 1.5, "7"], ids=repr)
def test_a_seed_that_is_not_an_int_in_0_to_2_64_is_refused_naming_it(value: Any) -> None:
    with pytest.raises(ValueError, match=r"\bseed\b"):
        fuzzer_module().Fuzzer(budget={"inputs": 4}, seed=value)


# ---------------------------------------------------------------------------
# The budget


@pytest.mark.parametrize("left", [0, 1, 3, BUDGET])
def test_one_call_returns_at_most_the_views_budget_left(left: int) -> None:
    batch = list(build(BUDGET, 5).propose(view(CASES["mixed"].contract, left)))
    assert len(batch) <= left


def test_one_call_returns_at_most_budget_inputs_when_offered_more() -> None:
    batch = list(build(5, 5).propose(view(CASES["mixed"].contract, 1000)))
    assert len(batch) <= 5


def test_over_a_trial_it_proposes_at_most_budget_inputs_whatever_the_stage_offers() -> None:
    agent, contract = build(5, 5), CASES["mixed"].contract
    got: list[Proposal] = []
    for _ in range(1000):
        batch = list(agent.propose(view(contract, 1000, got)))
        if not batch:
            break
        got += batch
        assert len(got) <= 5, f"{len(got)} proposals in one trial, past budget.inputs 5"
    assert len(got) <= 5


# ---------------------------------------------------------------------------
# Inside the contract by construction


@pytest.mark.parametrize("case_id", list(CASES))
def test_every_proposal_passes_check_inputs(case_id: str, tmp_path: Path) -> None:
    proposals = proposals_of(case_id, seed_for(case_id))
    assert proposals, "the fuzzer proposes at least one input set"
    assert refusals(CASES[case_id].contract, proposals, tmp_path) == {}


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_every_proposal_passes_check_inputs_over_the_multi_seed_contracts_at_every_seed(
    seed: int, tmp_path: Path
) -> None:
    found = {}
    for case_id in MULTI_SEED_CASES:
        root = tmp_path / case_id
        root.mkdir()
        refused = refusals(CASES[case_id].contract, proposals_of(case_id, seed), root)
        if refused:
            found[case_id] = refused
    assert found == {}


# ---------------------------------------------------------------------------
# The strategies


@pytest.mark.parametrize("case_id", list(CASES))
def test_the_boundary_and_special_values_the_contract_allows_each_appear(case_id: str) -> None:
    proposals = proposals_of(case_id, seed_for(case_id))
    missing = {}
    for spec in CASES[case_id].contract.inputs:
        seen = {bits for patterns in matching_patterns(spec, proposals) for bits in patterns}
        absent = [label for label, holds in wanted(spec).items() if not any(holds(bits) for bits in seen)]
        if absent:
            missing[spec.name] = absent
    assert missing == {}, f"values the contract allows that no proposal holds: {missing}"


@pytest.mark.parametrize("case_id", STRATEGY_CASES)
def test_constant_arrays_appear(case_id: str) -> None:
    assert any(constant_inputs(proposal) for proposal in proposals_of(case_id, seed_for(case_id)))


@pytest.mark.parametrize("case_id", STRATEGY_CASES)
def test_the_first_two_fresh_draws_through_generate_inputs_streams_appear(
    case_id: str, draws: Callable[[str, int, int], list[ArraysKey]]
) -> None:
    seed = seed_for(case_id)
    found = {arrays_key(proposal.arrays) for proposal in proposals_of(case_id, seed)}
    absent = [index for index, drawn in enumerate(draws(case_id, seed, 2)) if drawn not in found]
    assert absent == [], f"fresh draws {absent} (generate_inputs with draw_seed(seed, k)) are not proposed"


@pytest.mark.parametrize("case_id", STRATEGY_CASES)
def test_patterned_arrays_that_are_no_fresh_draw_appear(
    case_id: str, draws: Callable[[str, int, int], list[ArraysKey]]
) -> None:
    seed = seed_for(case_id)
    fresh = set(draws(case_id, seed, BUDGET))
    others = [proposal for proposal in proposals_of(case_id, seed) if arrays_key(proposal.arrays) not in fresh]
    assert any(patterned_inputs(proposal) for proposal in others)


def test_the_docstrings_list_the_strategies_as_known_cases_not_complete() -> None:
    module = fuzzer_module()
    text = " ".join(f"{module.__doc__ or ''} {module.Fuzzer.__doc__ or ''}".lower().split())
    assert [word for word in STRATEGY_WORDS if word not in text] == []


# ---------------------------------------------------------------------------
# Seeds: its own, never the item's


def test_draw_seed_derives_distinct_seeds_from_the_fuzzers_own_seed_and_the_index_alone() -> None:
    draw_seed = fuzzer_module().draw_seed
    assert list(inspect.signature(draw_seed).parameters) == ["seed", "index"]
    for seed in SEEDS:
        derived = [draw_seed(seed, index) for index in range(BUDGET)]
        assert all(type(value) is int and 0 <= value < 2**64 for value in derived), derived
        assert derived == [draw_seed(seed, index) for index in range(BUDGET)]
        assert len(set(derived)) == BUDGET, "every draw of a trial has its own seed"
        assert seed not in derived, "a draw never uses the configured seed itself"
    assert len({draw_seed(seed, 0) for seed in range(256)}) == 256


@pytest.mark.parametrize("case_id", FAR_DRAW_CASES)
def test_a_long_trial_holds_its_fresh_draws_through_generate_inputs_streams_far_past_the_first_two(
    case_id: str, draws: Callable[[str, int, int], list[ArraysKey]]
) -> None:
    seed = seed_for(case_id)
    found = {arrays_key(proposal.arrays) for proposal in proposals_of(case_id, seed, LONG_BUDGET)}
    absent = [index for index, drawn in enumerate(draws(case_id, seed, FAR_DRAWS)) if drawn not in found]
    assert absent == [], f"fresh draws {absent} (generate_inputs with draw_seed(seed, k)) are not in the long trial"


@pytest.mark.parametrize("case_id", FAR_DRAW_CASES)
def test_no_proposal_of_a_long_trial_is_drawn_with_its_own_seed_or_a_manifest_items_seed(
    case_id: str, tmp_path: Path
) -> None:
    seed, contract = seed_for(case_id), CASES[case_id].contract
    assert manifest_seeds(), "the manifests declare held-out seeds (tt-pairs-v0 does)"
    targets = {used: fresh_draw(contract, used, tmp_path / str(used)) for used in sorted({seed} | manifest_seeds())}
    found = {arrays_key(proposal.arrays) for proposal in proposals_of(case_id, seed, LONG_BUDGET)}
    hits = [used for used, drawn in targets.items() if drawn in found]
    assert hits == [], f"proposals equal to generate_inputs over the contract at seeds {hits} (fuzzer seed {seed})"


def test_the_module_imports_neither_the_bench_registry_nor_the_trial_record() -> None:
    assert SOURCE_FILE.is_file(), f"{SOURCE_FILE.relative_to(REPO).as_posix()} does not exist (task P16.3)"
    tree = ast.parse(SOURCE_FILE.read_text(encoding="ascii"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names += [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
    found = sorted({name for name in names for bad in FORBIDDEN_IMPORTS if name == bad or name.startswith(f"{bad}.")})
    assert found == []


# ---------------------------------------------------------------------------
# Determinism: one seed, the same bytes


@pytest.mark.parametrize("case_id", DETERMINISM_CASES)
def test_the_proposals_are_a_function_of_the_config_and_the_view(case_id: str) -> None:
    contract = CASES[case_id].contract
    agent = build(BUDGET, 99)
    first = drive(agent, contract)
    assert drive(agent, contract) == first, "a second trial with the same object"
    assert drive(build(BUDGET, 99), contract) == first, "a new object"
    assert drive(build(BUDGET, 99), contract, step=1) == first, "asked for one set at a time"
    assert drive(build(BUDGET, 99), contract, step=7) == first, "asked for seven sets at a time"


def test_another_seed_gives_other_proposals() -> None:
    contract = CASES["mixed"].contract
    assert drive(build(BUDGET, 1), contract) != drive(build(BUDGET, 2), contract)


def test_the_bytes_do_not_depend_on_the_hash_seed() -> None:
    here = kat_digest()
    code = (
        "import importlib.util, sys\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "spec = importlib.util.spec_from_file_location('fuzzer_kat_run', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name] = module\n"
        "spec.loader.exec_module(module)\n"
        "print(module.kat_digest())\n"
    )
    for hash_seed in ("0", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        argv = [sys.executable, "-c", code, __file__, str(Path(__file__).parent)]
        result = subprocess.run(argv, capture_output=True, text=True, cwd=REPO, env=env, timeout=300)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == here, f"PYTHONHASHSEED={hash_seed} gives other bytes"


def test_one_seed_gives_the_known_answer_bytes() -> None:
    got = kat_digest()
    if KNOWN_ANSWER_SHA256.startswith("PLACEHOLDER"):
        pytest.fail(
            f"KNOWN_ANSWER_SHA256 is a PLACEHOLDER; record the known-answer value only from a run in which every "
            f"other test here passes. This run gives {got}"
        )
    assert got == KNOWN_ANSWER_SHA256
