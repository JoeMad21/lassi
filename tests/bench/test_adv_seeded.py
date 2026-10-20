"""Tests for the seeded set: the suite adv-seeded-v0 and its seeded faulty candidates (task P16.8).

Bible: Benchmark Suites (a new row; split rules; manifest keys; held-out
inputs; Item tolerance), Harness Contract (lassi_io.h), Adversarial Setups
(item 1), Agent Rules 3 and 5. Plan: plans/p16-adversarial.md, P16.8, and
its planning decisions "Gate subjects (P16.8)", "Freeze (P16.10)", and
"Gate reading (P16.G)".

The contract these tests fix:

- assets/bench/adv-seeded-v0.yaml loads with load_suite as a tracked suite
  (no repo, no commit, no pin). Its six items (pair_max, piecewise_gate,
  piecewise_ops, prefix_sum, scale_i64, signed_root) are all unassigned and
  each has one language, cpp, tracked at assets/bench/adv-seeded-v0/<item>/cpp
  with the one file <item>.cpp and the support file lassi_io.h tracked at
  assets/harness/c/lassi_io.h; held-out inputs of one input x and one output
  y, with the seed, the input spec, the contract, and the tolerance the plan's
  table gives. Its held-out inputs pass its contract.
- No reference and no candidate uses a thread or an OpenMP construct, since
  toolchains/gcc.pin builds with -fopenmp and a thread-order reduction could
  make a control diverge.
- tests/fixtures/adversary/seeded/ holds the candidates
  (candidates/<item>/<id>.cpp), the replay recording seeded.json (synthetic,
  completions only, no messages), index.json naming each candidate in
  recording order, break-contract.json (an llm adversary recording), and a
  README. Each candidate differs from its item's reference; every file under
  candidates/ is one of the index's entries; the recording's completions are
  the FILE blocks (render_file_blocks) of the candidates in index order, one
  per trial of the runner's order (directions in recipe order, items sorted,
  runs 1 to trials.n), so every candidate is some trial's reply and no trial
  gets two; each entry's sha256 is its completion's; each class has at least
  two candidates.
- Each witness, written in the adversary-v0 input form and read with
  lassi.agents.llm parse_reply, passes check_inputs against its item's
  contract (staged_inputs writes the files, as the adversary stage does),
  and lies outside the held-out inputs' range (it fails the held-out spec
  read as a contract), so the divergence it shows is the adversary's finding.
- break-contract.json's first reply holds exactly the plan's eight
  proposals, each refused by check_inputs for every adv-seeded-v0 item; its
  second reply proposes nothing.

Every program, candidate, recording, and witness here is SYNTHETIC: written
by hand, never model output, no upstream text. No value in this module is a
measurement; the witness and held-out values are the plan's choices.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from lassi import agents  # noqa: F401  (registers the Agents the recipes bind)
from lassi.agents.llm import parse_reply
from lassi.bench import Direction, Suite, load_suite, registry
from lassi.core import runner  # noqa: F401  (importing the runner registers every component)
from lassi.core.adversary_stage import staged_inputs
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import Proposal
from lassi.core.recipe import load_recipe
from lassi.harness import LassiArray, generate_inputs
from lassi.harness.contract import ContractInput, InputContract, InputContractError, check_inputs, input_contract
from lassi.llm import ReplayBackend

REPO = Path(__file__).resolve().parents[2]
SUITE = "adv-seeded-v0"
MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
SEEDED = REPO / "tests" / "fixtures" / "adversary" / "seeded"
RECORDING = "tests/fixtures/adversary/seeded/seeded.json"
PROJECTS = REPO / "projects" / "adversary-seeded"
HEADER = "assets/harness/c/lassi_io.h"
SPLIT = "unassigned"
ELEMENTS = 64
# The plan's table: per item, its held-out input x (dtype, dist, lo, hi), seed, and contract for x.
HELD_OUT: dict[str, tuple[dict[str, Any], int, tuple[Any, ...]]] = {
    "pair_max": ({"dtype": "f32", "dist": "uniform", "lo": -1.0, "hi": 1.0}, 101, ("f32", -1000.0, 1000.0, True, True)),
    "piecewise_gate": (
        {"dtype": "f32", "dist": "uniform", "lo": 2.0, "hi": 10.0}, 102, ("f32", -1000.0, 1000.0, False, False),
    ),
    "piecewise_ops": (
        {"dtype": "f32", "dist": "uniform", "lo": 1.0, "hi": 10.0}, 103, ("f32", -1000.0, 1000.0, False, False),
    ),
    "prefix_sum": ({"dtype": "f32", "dist": "uniform", "lo": 0.0, "hi": 1.0}, 104, ("f32", -1e8, 1e8, False, False)),
    "scale_i64": (
        {"dtype": "i32", "dist": "integers", "lo": 0, "hi": 100}, 105, ("i32", -1000000, 1000000, False, False),
    ),
    "signed_root": (
        {"dtype": "f32", "dist": "uniform", "lo": 1.0, "hi": 100.0}, 106, ("f32", -10000.0, 10000.0, False, False),
    ),
}
ITEMS = tuple(sorted(HELD_OUT))
# The tolerance of every float item, and of scale_i64 (exact).
FLOAT_TOLERANCE = ("max_abs", 0.001)
EXACT_TOLERANCE = ("max_abs", 0)
# The fault class of each item's two candidates (the plan's classes; known cases, not complete).
CLASSES = {
    "pair_max": "special-value",
    "piecewise_gate": "boundary",
    "piecewise_ops": "arithmetic",
    "prefix_sum": "precision",
    "scale_i64": "narrowing",
    "signed_root": "sign",
}
INDEX_KEYS = {"synthetic", "suite", "recording", "candidates"}
ENTRY_KEYS = {"id", "item", "run", "class", "fault", "trigger", "witness", "file", "sha256"}
# A thread or an OpenMP construct; `omp_` only at a word start, so an identifier such as comp_x is not one.
THREADED = (
    r"#\s*pragma\s+omp",
    r"\bomp_",
    r"<thread>",
    r"\bstd::thread\b",
    r"\bstd::async\b",
    r"<atomic>",
    r"<future>",
    r"<omp\.h>",
)
# The f32 quiet NaN the adversary-v0 form writes for "nan".
NAN_BITS = 0x7FC00000


def text_of(path: Path) -> str:
    """Return a tracked file's text, failing unless it is plain ASCII with LF line ends."""
    raw = path.read_bytes()
    assert b"\r" not in raw, f"{path} has a CR"
    return raw.decode("ascii")


def json_of(path: Path) -> Any:
    """Return a fixture's JSON value, read as plain ASCII."""
    return json.loads(text_of(path))


def suite() -> Suite:
    """Load the adv-seeded-v0 manifest."""
    return load_suite(MANIFEST)


def reference(item: str) -> str:
    """Return the text of an item's tracked reference program, <item>.cpp."""
    return text_of(REPO / "assets" / "bench" / SUITE / item / "cpp" / f"{item}.cpp")


def index() -> dict[str, Any]:
    """Return the seeded set's index.json."""
    return json_of(SEEDED / "index.json")


def entries() -> list[dict[str, Any]]:
    """Return the index's candidate entries, in recording order."""
    return index()["candidates"]


def candidate_path(entry: dict[str, Any]) -> Path:
    """Return the file an index entry names: a repository path, or a path under the seeded directory."""
    file = entry["file"]
    return REPO / file if file.startswith("tests/") else SEEDED / file


def completions(name: str) -> list[str]:
    """Return the reply texts of a recording in the seeded directory, checking its synthetic form."""
    data = json_of(SEEDED / name)
    assert set(data) == {"synthetic", "completions"} and data["synthetic"] is True, f"{name}: {sorted(data)}"
    for position, entry in enumerate(data["completions"]):
        assert set(entry) == {"text"}, f"{name}: completion {position} carries {sorted(entry)}, not reply text only"
    return [entry["text"] for entry in data["completions"]]


def trial_order(recipe: str) -> list[tuple[str, str, int]]:
    """Return the runner's trial order for a recipe: directions in recipe order, items sorted, runs 1 to trials.n."""
    data = load_recipe(PROJECTS / f"{recipe}.yaml").data
    loaded = suite()
    selected = data["bench"].get("items")
    in_split = [name for name, item in loaded.items.items() if item.split == data["bench"]["split"]]
    items = sorted(in_split if selected is None else selected)
    runs = range(1, data["trials"]["n"] + 1)
    return [(f"{d['source']}-{d['target']}", item, run) for d in data["directions"] for item in items for run in runs]


def f32_array(*values: float, name: str = "x", count: int = ELEMENTS) -> LassiArray:
    """Return a SYNTHETIC f32 array of `count` elements repeating `values`."""
    elements = [values[position % len(values)] for position in range(count)]
    return LassiArray(name, "f32", (count,), struct.pack(f"<{count}f", *elements))


def f32_nan_pattern(other: float) -> LassiArray:
    """Return the f32 array of 64 elements alternating the quiet NaN and `other`."""
    pair = struct.pack("<I", NAN_BITS) + struct.pack("<f", other)
    return LassiArray("x", "f32", (ELEMENTS,), pair * (ELEMENTS // 2))


def i32_fill(value: int) -> LassiArray:
    """Return the i32 array of 64 elements that holds `value` each."""
    return LassiArray("x", "i32", (ELEMENTS,), struct.pack(f"<{ELEMENTS}i", *([value] * ELEMENTS)))


# The plan's witnesses per item, as arrays, one per candidate (their order between the two runs is not fixed).
WITNESSES = {
    "pair_max": [f32_nan_pattern(1.0), f32_nan_pattern(1.0)],
    "piecewise_gate": [f32_array(0.0), f32_array(1.0)],
    "piecewise_ops": [f32_array(-5.0), f32_array(-5.0)],
    "prefix_sum": [f32_array(16777216.0, 1.0), f32_array(16777216.0, 1.0)],
    "scale_i64": [i32_fill(-1000000), i32_fill(999999)],
    "signed_root": [f32_array(-4.0), f32_array(-4.0)],
}


def proposals_of(block: dict[str, Any]) -> tuple[Proposal, ...]:
    """Return the proposals of a reply holding `block` as its one ```json block, read with parse_reply."""
    return parse_reply(f"```json\n{json.dumps(block)}\n```")


def witness_proposal(entry: dict[str, Any]) -> Proposal:
    """Return an entry's witness as the one proposal of a reply {"proposals": [witness]}."""
    (found,) = proposals_of({"proposals": [entry["witness"]]})
    return found


def refusal(contract: InputContract, found: Proposal, directory: Path) -> str | None:
    """Stage `found` under the new `directory` as the stage does and check it; return the refusal reason or None."""
    try:
        check_inputs(contract, staged_inputs(found.arrays, directory))
    except InputContractError as error:
        return error.reason
    return None


# ---------------------------------------------------------------------------
# The manifest and its programs


def test_the_manifest_is_a_tracked_suite_of_six_unassigned_items() -> None:
    loaded = suite()
    assert loaded.name == SUITE
    assert (loaded.repo, loaded.commit, loaded.pin) == (None, None, None)
    assert loaded.tracked is True
    assert tuple(sorted(loaded.items)) == ITEMS
    assert {item.split for item in loaded.items.values()} == {SPLIT}


@pytest.mark.parametrize("name", ITEMS)
def test_each_item_is_one_tracked_cpp_program_with_lassi_io_h(name: str) -> None:
    item = suite().items[name]
    assert set(item.languages) == {"cpp"}
    cpp = item.languages["cpp"]
    assert cpp.tracked is True
    assert cpp.dir == f"assets/bench/{SUITE}/{name}/cpp"
    assert cpp.files == (f"{name}.cpp",)
    assert dict(cpp.support) == {"lassi_io.h": registry.SupportFile(path=HEADER, tracked=True)}
    assert dict(item.support) == {} and item.run_args == ()


@pytest.mark.parametrize("name", ITEMS)
def test_each_item_has_the_planned_held_out_inputs_tolerance_and_contract(name: str) -> None:
    item = suite().items[name]
    spec, seed, (dtype, lo, hi, allow_nan, allow_inf) = HELD_OUT[name]
    assert [dict(entry) for entry in item.inputs] == [{"name": "x", "shape": [ELEMENTS], **spec}]
    assert item.outputs == ("y",)
    assert item.seed == seed
    assert item.contract == InputContract((ContractInput("x", dtype, (ELEMENTS,), lo, hi, allow_nan, allow_inf),))
    assert item.tolerance is not None
    expected = EXACT_TOLERANCE if name == "scale_i64" else FLOAT_TOLERANCE
    assert (item.tolerance.metric, item.tolerance.threshold) == expected


@pytest.mark.parametrize("name", ITEMS)
def test_the_held_out_inputs_pass_the_contract(name: str, tmp_path: Path) -> None:
    item = suite().items[name]
    written = generate_inputs(item.inputs, item.seed, tmp_path)
    assert item.contract is not None
    check_inputs(item.contract, list(written.values()))


@pytest.mark.parametrize("name", ITEMS)
def test_each_reference_reads_and_writes_through_lassi_io_h(name: str) -> None:
    text = reference(name)
    assert suite().reference_target(name, Direction("cpp", "cpp"), Path("unused"), purpose="eval") == {
        f"{name}.cpp": text
    }
    assert '#include "lassi_io.h"' in text
    assert "lassi_io_read" in text and "lassi_io_write" in text


def program_files() -> list[Path]:
    """Return every reference program and every candidate file of the seeded set."""
    references = [REPO / "assets" / "bench" / SUITE / name / "cpp" / f"{name}.cpp" for name in ITEMS]
    candidates = sorted((SEEDED / "candidates").rglob("*.cpp"))
    assert candidates, f"no candidate under {SEEDED / 'candidates'}"
    return [*references, *candidates]


def test_no_reference_and_no_candidate_uses_a_thread_or_openmp() -> None:
    for path in program_files():
        text = text_of(path)
        found = [pattern for pattern in THREADED if re.search(pattern, text)]
        assert not found, f"{path.relative_to(REPO).as_posix()} matches {found}"


# ---------------------------------------------------------------------------
# The candidates and the index


def test_the_index_names_the_suite_and_the_recording() -> None:
    data = index()
    assert set(data) == INDEX_KEYS
    assert data["synthetic"] is True
    assert (data["suite"], data["recording"]) == (SUITE, RECORDING)
    assert data["candidates"], "the index lists no candidate"


def test_each_index_entry_names_its_fault_trigger_witness_and_file() -> None:
    ids = []
    for entry in entries():
        assert set(entry) == ENTRY_KEYS, sorted(entry)
        assert entry["item"] in ITEMS and entry["run"] in (1, 2)
        assert entry["class"] == CLASSES[entry["item"]], entry["id"]
        for key in ("id", "fault", "trigger"):
            assert isinstance(entry[key], str) and entry[key].strip(), f"{entry['id']}: {key}"
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), entry["id"]
        parts = PurePosixPath(entry["file"]).parts
        assert parts[-3:] == ("candidates", entry["item"], f"{entry['id']}.cpp"), entry["file"]
        assert candidate_path(entry).is_file(), entry["file"]
        ids.append(entry["id"])
    assert len(set(ids)) == len(ids), "candidate ids are distinct"


def test_every_candidate_file_is_an_index_entry() -> None:
    on_disk = {path.resolve() for path in (SEEDED / "candidates").rglob("*") if path.is_file()}
    listed = {candidate_path(entry).resolve() for entry in entries()}
    assert on_disk == listed


def test_each_candidate_differs_from_its_reference() -> None:
    # Past their first lines (the candidate's first line names it SYNTHETIC), so one that lost its fault fails.
    for entry in entries():
        candidate = text_of(candidate_path(entry)).split("\n", 1)[1]
        assert candidate != reference(entry["item"]).split("\n", 1)[1], entry["id"]


def test_each_class_has_at_least_two_candidates() -> None:
    counts = Counter(entry["class"] for entry in entries())
    assert set(counts) == set(CLASSES.values())
    assert all(count >= 2 for count in counts.values()), counts


# ---------------------------------------------------------------------------
# The recording


def test_the_recording_is_the_candidates_file_blocks_in_index_order() -> None:
    texts = completions("seeded.json")
    listed = entries()
    assert len(texts) == len(listed)
    for text, entry in zip(texts, listed, strict=True):
        assert text == render_file_blocks({f"{entry['item']}.cpp": text_of(candidate_path(entry))}), entry["id"]
        assert hashlib.sha256(text.encode("ascii")).hexdigest() == entry["sha256"], entry["id"]


@pytest.mark.parametrize("recipe", ["seeded-fuzzer", "seeded-llm"])
def test_every_candidate_is_one_trials_reply_and_no_trial_gets_two(recipe: str) -> None:
    data = load_recipe(PROJECTS / f"{recipe}.yaml").data
    assert data["model"]["recording"] == RECORDING
    order = trial_order(recipe)
    listed = entries()
    assert len(completions("seeded.json")) == len(order), "one completion per trial"
    assert [(entry["item"], entry["run"]) for entry in listed] == [(item, run) for _, item, run in order]
    assert len({(entry["item"], entry["run"]) for entry in listed}) == len(listed), "no trial gets two"


def test_the_replay_backend_reads_the_recording() -> None:
    backend = ReplayBackend("adv-seeded-replay", recording=SEEDED / "seeded.json")
    digest = hashlib.sha256((SEEDED / "seeded.json").read_bytes()).hexdigest()
    assert backend.recording_sha256 == digest


# ---------------------------------------------------------------------------
# The witnesses


def test_every_witness_passes_its_items_contract(tmp_path: Path) -> None:
    loaded = suite()
    for position, entry in enumerate(entries()):
        contract = loaded.items[entry["item"]].contract
        assert contract is not None
        found = witness_proposal(entry)
        assert refusal(contract, found, tmp_path / f"witness{position:02d}") is None, entry["id"]


def test_every_witness_lies_outside_the_held_out_inputs_range(tmp_path: Path) -> None:
    loaded = suite()
    for position, entry in enumerate(entries()):
        held_out = input_contract(loaded.items[entry["item"]].inputs)
        reason = refusal(held_out, witness_proposal(entry), tmp_path / f"witness{position:02d}")
        assert reason in ("out-of-range", "nan", "infinity"), f"{entry['id']}: {reason}"


def test_the_witnesses_are_the_plans() -> None:
    found: dict[str, list[bytes]] = {}
    for entry in entries():
        (array,) = witness_proposal(entry).arrays
        assert (array.name, array.shape) == ("x", (ELEMENTS,)), entry["id"]
        found.setdefault(entry["item"], []).append(array.dtype.encode("ascii") + array.data)
    expected = {
        item: sorted(array.dtype.encode("ascii") + array.data for array in arrays) for item, arrays in WITNESSES.items()
    }
    assert {item: sorted(blobs) for item, blobs in found.items()} == expected


# ---------------------------------------------------------------------------
# The break-contract recording


def zeros(dtype: str, count: int, name: str = "x") -> LassiArray:
    """Return a SYNTHETIC array of `count` zero elements of a 4- or 8-byte dtype."""
    width = 8 if dtype == "f64" else 4
    return LassiArray(name, dtype, (count,), bytes(width * count))


# The plan's eight proposals, as parse_reply reads them; an array lassi_io cannot hold has no data.
BREAK_CONTRACT = (
    (),
    (zeros("f32", ELEMENTS, name="zz"),),
    (LassiArray("x", "f128", (ELEMENTS,), b""),),
    (zeros("f32", 3),),
    (LassiArray("x", "f32", (2000000,), b""),),
    (zeros("f64", ELEMENTS),),
    (zeros("f32", ELEMENTS), zeros("f32", ELEMENTS)),
    (f32_array(200000000.0),),
)


def test_break_contract_holds_the_plans_eight_proposals_then_none() -> None:
    first, second = completions("break-contract.json")
    assert [found.arrays for found in parse_reply(first)] == list(BREAK_CONTRACT)
    assert parse_reply(second) == ()


def test_every_break_contract_proposal_is_refused_for_every_item(tmp_path: Path) -> None:
    first, _ = completions("break-contract.json")
    found = parse_reply(first)
    assert found, "the first reply proposes nothing"
    for name, item in sorted(suite().items.items()):
        assert item.contract is not None
        for position, proposal in enumerate(found):
            reason = refusal(item.contract, proposal, tmp_path / f"{name}-{position:02d}")
            assert reason is not None, f"proposal {position} passes {name}'s contract"


# ---------------------------------------------------------------------------
# The README


def test_the_readme_labels_the_set_synthetic_and_cites_the_classes() -> None:
    text = text_of(SEEDED / "README.md")
    assert "SYNTHETIC" in text
    for name in sorted(set(CLASSES.values())):
        assert name in text, f"the README does not name the class {name}"
    for source in ("DeMillo", "Offutt", "McKeeman", "Laguna"):
        assert source in text, f"the README does not cite {source}"
    for file in ("seeded.json", "index.json", "break-contract.json", "candidates/"):
        assert file in text, f"the README does not describe {file}"
