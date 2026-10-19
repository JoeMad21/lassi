"""Shared helpers for the tests of the fuzzer adversary (task P16.3); not a test module.

- fuzzer_module imports lassi.agents.fuzzer inside each test, so a missing
  module fails each test on its own instead of the whole collection.
- build makes a Fuzzer from the two config keys a recipe's adversary
  section gives it (Project Recipes, the bindings block's adversary line):
  budget, a mapping whose key inputs is the most input sets it proposes in
  a trial, and seed.
- view builds the AdversaryView a stage would give it (lassi.core.interfaces;
  task P16.2): the contract, SYNTHETIC source and candidate texts, the
  budget left, and the earlier proposals as outcomes coded CODE, a token of
  the outcome form. It holds no held-out input, no item seed, and no
  reference output, since the type has no field for any of them.
- drive asks for one trial's proposals as the adversary stage would (task
  P16.5 builds the stage): each call offers the budget less the proposals
  so far, at most `step` of it when given, until an empty result or a spent
  budget (Component Interfaces, the adversary bullet: propose returns at
  most view.budget_left Proposals, and an empty result proposes nothing
  more).
- draw_spec reads a contract as a held-out input spec (generate_inputs
  entries: each input's name, dtype, shape, lo, and hi, dist uniform for a
  float dtype and integers for an integer one), and fresh_draw writes the
  input set generate_inputs gives for it and a seed and reads it back, as
  an ArraysKey, the arrays sorted by name; arrays_key gives a proposal's.
- manifest_seeds reads every manifest under assets/bench/ through the bench
  registry, each item for purpose eval, and returns the held-out seeds its
  items declare. Only the tests read them; the fuzzer never does.

Every program text here is SYNTHETIC. No value in this module is a
measurement.
"""

from __future__ import annotations

import functools
import importlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from lassi.bench import load_suite
from lassi.core.interfaces import AdversaryView, Proposal, ProposalOutcome
from lassi.harness import InputContract, LassiArray, generate_inputs, read_array

MODULE = "lassi.agents.fuzzer"
MANIFESTS = Path(__file__).resolve().parents[2] / "assets" / "bench"
# The budget of the bindings block's adversary line (Project Recipes: a fuzzer with a budget of 64 input sets).
BUDGET = 64
# The outcome code the driven trials give every earlier proposal: a token of the outcome form (task P16.2).
CODE = "synthetic-ran"
FLOATS = ("f32", "f16", "bf16", "f64")
INTS = ("i8", "u8", "i32", "u32", "i64")
SOURCE = {"main.cpp": "// SYNTHETIC source program\nint main() { return 0; }\n"}
CANDIDATE = {"main.cpp": "// SYNTHETIC candidate program\nint main() { return 1; }\n"}

ArraysKey = tuple[tuple[str, str, tuple[int, ...], bytes], ...]


def fuzzer_module() -> ModuleType:
    """Import and return lassi.agents.fuzzer, the module under test."""
    return importlib.import_module(MODULE)


def build(inputs: int = BUDGET, seed: int = 0) -> Any:
    """Return a Fuzzer configured as an adversary section would configure it: budget.inputs and seed."""
    return fuzzer_module().Fuzzer(budget={"inputs": inputs}, seed=seed)


def view(contract: InputContract, budget_left: int, earlier: Sequence[Proposal] = ()) -> AdversaryView:
    """Return the AdversaryView of one trial: `contract`, SYNTHETIC programs, the budget left, the earlier outcomes."""
    outcomes = tuple(ProposalOutcome(proposal, CODE) for proposal in earlier)
    return AdversaryView(
        contract=contract, source=SOURCE, candidate=CANDIDATE, budget_left=budget_left, outcomes=outcomes
    )


def drive(agent: Any, contract: InputContract, budget: int = BUDGET, step: int | None = None) -> list[Proposal]:
    """Ask `agent` for one trial's proposals as the adversary stage would, and return them in order.

    Each call offers budget_left, the budget less the proposals so far (at
    most `step` of it when given), with the earlier proposals as outcomes;
    the trial ends at an empty result or a spent budget. Every call must
    return exactly Proposals, no more than it was offered.
    """
    got: list[Proposal] = []
    while len(got) < budget:
        left = budget - len(got)
        offered = left if step is None else min(step, left)
        batch = list(agent.propose(view(contract, offered, got)))
        assert len(batch) <= offered, f"one call returned {len(batch)} proposals, past budget_left {offered}"
        wrong = sorted({type(item).__name__ for item in batch if type(item) is not Proposal})
        assert not wrong, f"propose returned {wrong}, not Proposals"
        if not batch:
            break
        got += batch
    return got


def draw_spec(contract: InputContract) -> list[dict[str, Any]]:
    """Return `contract` read as a held-out input spec: generate_inputs entries over its own ranges."""
    return [
        {
            "name": spec.name,
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "dist": "uniform" if spec.dtype in FLOATS else "integers",
            "lo": spec.lo,
            "hi": spec.hi,
        }
        for spec in contract.inputs
    ]


def arrays_key(arrays: Iterable[LassiArray]) -> ArraysKey:
    """Return arrays as a comparable key: (name, dtype, shape, data) of each, sorted by name."""
    return tuple(sorted((array.name, array.dtype, tuple(array.shape), bytes(array.data)) for array in arrays))


def fresh_draw(contract: InputContract, seed: int, directory: Path) -> ArraysKey:
    """Write generate_inputs(draw_spec(contract), seed) into the new directory `directory`; return its arrays' key."""
    directory.mkdir()
    paths = generate_inputs(draw_spec(contract), seed, directory)
    return arrays_key(read_array(path) for path in paths.values())


@functools.cache
def manifest_seeds() -> frozenset[int]:
    """Return the held-out seed of every item of every manifest under assets/bench/, each item read for eval."""
    seeds: set[int] = set()
    for path in sorted(MANIFESTS.glob("*.yaml")):
        suite = load_suite(path)
        for name in sorted(suite.items):
            item = suite.item(name, purpose="eval")
            if item.seed is not None:
                seeds.add(item.seed)
    return frozenset(seeds)
