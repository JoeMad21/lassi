"""Tests that the fuzzer never reproduces an item's held-out inputs and never derives an item's seed (task P16.3).

Bible: Adversarial Setups (item 1; confirmed counterexamples never join the
held-out inputs of an eval or unassigned item), Harness Contract (reward and
evaluation use held-out inputs that never appear in a prompt; held-out input
files: a seed gives the same bytes on every host), Benchmark Suites (held-out
inputs; split rules), Agent Rule 5. Plan: plans/p16-adversarial.md, P16.3
("no proposal equals the item's held-out inputs"; fresh draws "from seeds
derived from its own seed, never the item's"), and its Constraints (items are
read only through the bench registry, for purpose eval; an adversary never
sees a held-out input or an item's seed).

- Each tt-pairs-v0 item is read with Suite.item(purpose="eval"), and its
  held-out inputs are generated with generate_inputs(item.inputs, item.seed,
  <a temporary directory>), never changed and never shown to the fuzzer. No
  item declares a `contract` key, so the item's contract read as a spec
  (fuzzer_support.draw_spec) is exactly its held-out input spec (checked
  first): a fuzzer that drew with its configured seed itself would then
  reproduce the held-out inputs byte for byte whenever that seed is the
  item's. So the fuzzer is configured with the item's own seed, the case of
  a derivation that reuses the configured seed, and a budget of 32, and no
  proposed array equals the item's held-out array of its name, so no
  proposal equals the held-out set. A derivation that ignores the configured
  seed (draw k at seed k + 1, say) is left to the seed tests: the last test
  here, and in tests/agents/test_fuzzer.py
  test_draw_seed_derives_distinct_seeds_from_the_fuzzers_own_seed_and_the_index_alone
  and the long-trial tests.
- For loopback, the smallest item (51200 elements), the whole trial is
  compared, and the fuzzer's first fresh draw must be among its 32
  proposals, so the comparison reaches the strategy that could reproduce
  the inputs. For the other items the trial is asked for one set at a time,
  as a stage may ask, up to and including the first fresh draw, which must
  come within the budget: their fresh draws are larger, 65536 elements for
  eltwise_sfpu, 131072 for eltwise_binary, and 819200 (a 640 x 640 pair)
  for each matmul item, 16 times loopback's 51200, so whole trials of them
  are not drawn here. Their later draws use draw_seed(seed, k) for k
  of 1 and above, which the seed test below and the long-trial tests in
  tests/agents/test_fuzzer.py cover. The two matmul items are marked slow.
- No derived seed, draw_seed(seed, k) for k below 256 (four times the
  bindings block's budget of 64, past the fresh draws of those long
  trials), equals the seed of any item of any manifest under assets/bench/,
  for the fuzzer's seeds 0 to 255 and every manifest item's seed: a
  derivation that maps small seeds to small seeds (seed + k) would collide
  with one.

Nothing here runs a program, trains, tunes a prompt, or harvests; items are
read for purpose eval only, and the held-out inputs are written only to a
temporary directory (Agent Rule 5). No value in this module is a
measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fuzzer_support import (
    MANIFESTS,
    ArraysKey,
    arrays_key,
    build,
    draw_spec,
    drive,
    fresh_draw,
    fuzzer_module,
    manifest_seeds,
    view,
)

from lassi.bench import SuiteItem, load_suite
from lassi.core.interfaces import Proposal
from lassi.harness import LassiArray, generate_inputs, read_array

TT_PAIRS_MANIFEST = MANIFESTS / "tt-pairs-v0.yaml"
WHOLE_TRIAL_ITEMS = ("loopback",)
FIRST_DRAW_ITEMS = (
    pytest.param("eltwise_binary"),
    pytest.param("eltwise_sfpu"),
    pytest.param("matmul_multi_core", marks=pytest.mark.slow),
    pytest.param("matmul_single_core", marks=pytest.mark.slow),
)
HELD_OUT_BUDGET = 32
DRAW_INDICES = 256


def held_out(name: str, directory: Path) -> tuple[SuiteItem, dict[str, LassiArray]]:
    """Read tt-pairs-v0 item `name` for eval and generate its held-out inputs into `directory`; return both.

    Fails unless the item's contract read as a spec is its held-out input
    spec, the premise that makes the item's own seed the hardest case.
    """
    item = load_suite(TT_PAIRS_MANIFEST).item(name, purpose="eval")
    assert item.contract is not None and item.seed is not None
    assert draw_spec(item.contract) == [dict(entry) for entry in item.inputs], "the contract is the held-out spec"
    directory.mkdir()
    paths = generate_inputs(item.inputs, item.seed, directory)
    return item, {input_name: read_array(path) for input_name, path in paths.items()}


def equal_arrays(proposals: list[Proposal], held: dict[str, LassiArray]) -> list[tuple[int, str]]:
    """Return (proposal index, input name) for every proposed array equal to the held-out array of its name."""
    return [
        (index, array.name)
        for index, proposal in enumerate(proposals)
        for array in proposal.arrays
        if array == held.get(array.name)
    ]


def through(agent: Any, item: SuiteItem, last: ArraysKey) -> list[Proposal]:
    """Ask `agent` for one set at a time over the item's contract, up to and including the proposal `last`.

    The trial's budget is HELD_OUT_BUDGET; it also ends at an empty result
    or a spent budget, and every call must return at most one Proposal.
    """
    got: list[Proposal] = []
    while len(got) < HELD_OUT_BUDGET:
        batch = list(agent.propose(view(item.contract, 1, got)))
        assert len(batch) <= 1 and all(type(proposal) is Proposal for proposal in batch), batch
        if not batch:
            break
        got += batch
        if arrays_key(batch[0].arrays) == last:
            break
    return got


@pytest.mark.parametrize("name", WHOLE_TRIAL_ITEMS)
def test_no_proposal_of_a_whole_trial_equals_the_items_held_out_inputs_at_the_items_own_seed(
    name: str, tmp_path: Path
) -> None:
    item, held = held_out(name, tmp_path / "held-out")
    proposals = drive(build(HELD_OUT_BUDGET, item.seed), item.contract, budget=HELD_OUT_BUDGET)
    first_draw = fresh_draw(item.contract, fuzzer_module().draw_seed(item.seed, 0), tmp_path / "first-draw")
    assert first_draw in {arrays_key(proposal.arrays) for proposal in proposals}, "the first fresh draw is compared"
    assert equal_arrays(proposals, held) == [], "proposed arrays equal to the item's held-out arrays"
    assert arrays_key(held.values()) not in {arrays_key(proposal.arrays) for proposal in proposals}


@pytest.mark.parametrize("name", FIRST_DRAW_ITEMS)
def test_no_proposal_up_to_the_first_fresh_draw_equals_the_items_held_out_inputs_at_the_items_own_seed(
    name: str, tmp_path: Path
) -> None:
    item, held = held_out(name, tmp_path / "held-out")
    first_draw = fresh_draw(item.contract, fuzzer_module().draw_seed(item.seed, 0), tmp_path / "first-draw")
    proposals = through(build(HELD_OUT_BUDGET, item.seed), item, first_draw)
    assert proposals and arrays_key(proposals[-1].arrays) == first_draw, "the first fresh draw comes in the budget"
    assert equal_arrays(proposals, held) == [], "proposed arrays equal to the item's held-out arrays"
    assert arrays_key(held.values()) not in {arrays_key(proposal.arrays) for proposal in proposals}


def test_no_derived_seed_is_the_seed_of_any_manifest_item() -> None:
    seeds = manifest_seeds()
    assert seeds, "the manifests declare held-out seeds (tt-pairs-v0 does)"
    draw_seed = fuzzer_module().draw_seed
    hits = sorted(
        (seed, index)
        for seed in sorted(set(range(256)) | seeds)
        for index in range(DRAW_INDICES)
        if draw_seed(seed, index) in seeds
    )
    assert hits == [], f"(fuzzer seed, draw index) pairs whose derived seed is a manifest item's seed: {hits}"
