"""Tests for the llm adversary's prompts: adversary-v0 and render_messages (task P16.4, design section 5).

Bible: Agents And Judges (versioned prompts), Harness Contract (held-out
inputs never appear in a prompt), Design Principle 3 (prompts are versioned
data files), Readability Standards (Prompts row), Agent Rule 5. Plan:
plans/p16-adversarial.md, P16.4 ("versioned prompts written here ... show
the contract, the source program, and the candidate, and never a held-out
input, the item's seed, or a reference output (test over the rendered
messages)"), and its Constraints (an adversary never sees a held-out input,
an item's seed, or a reference output).

The contract these tests fix:

- assets/prompts/adversary-v0/ holds propose.txt and malformed.txt, plain
  ASCII with LF line ends, string.Template files. propose.txt's
  placeholders are exactly $contract, $source_files, $candidate_files,
  $budget_left, and $outcomes, and it shows the reply form (a "```json"
  block, the keys "proposals", "inputs", "name", "dtype", "shape",
  "values", "fill", and "pattern", the strings "nan", "inf", and "-inf")
  and that {"proposals": []} stops the search. malformed.txt's are exactly
  $reason and $budget_left.
- lassi.agents.llm render_messages(view, prompts, shown) returns one user
  Message, propose.txt of the set `prompts` rendered (lassi.prompts.render)
  from the view alone: $contract is one line per input, "- <name>:
  <dtype>, shape [<dims>], each element in [<lo>, <hi>)" then "; NaN
  allowed" or "; NaN not allowed" and "; infinity allowed" or "; infinity
  not allowed", lo and hi written with repr; $source_files and
  $candidate_files are lassi.core.files render_file_blocks of the view's
  files (each a fenced block whose first line is "// FILE: <path>", in
  sorted path order); $budget_left is the view's; $outcomes is "None yet."
  or one line per outcome, "- proposal <i>: <code>", followed by the line
  "  proposed: <json>" when i < len(shown) and shown[i][0] equals the
  outcome's proposal. `shown` holds one (Proposal, compact JSON) pair per
  proposal the agent made, in order. Lines are joined with LF, with no
  trailing LF.
- The rendered messages hold no held-out input value, no item seed, and no
  held-out spec key such as `dist`: the view has no field for any of them.
  Checked with adv-stage's held-out spec and seed 11 (as
  tests/core/adversary_support.py writes them; restated here, since that
  module lies outside this directory) and with the tt-pairs-v0 item
  loopback, read through the bench registry for purpose eval, its held-out
  inputs generated into a temporary directory only (Agent Rule 5).

Every contract, program text, and JSON text here is SYNTHETIC. No value in
this module is a measurement.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Sequence
from pathlib import Path
from string import Template
from typing import Any

import pytest
from llm_support import CANDIDATE, CONTRACT, PROMPTS, REPO, SOURCE, llm_module, view

from lassi.bench import load_suite
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import AdversaryView, Message, Proposal, ProposalOutcome
from lassi.harness import ContractInput, InputContract, LassiArray, generate_inputs, input_contract, read_array
from lassi.prompts import render

PROMPT_DIR = REPO / "assets" / "prompts" / PROMPTS
PROPOSE_PLACEHOLDERS = {"contract", "source_files", "candidate_files", "budget_left", "outcomes"}
MALFORMED_PLACEHOLDERS = {"reason", "budget_left"}
REPLY_KEYS = ("proposals", "inputs", "name", "dtype", "shape", "values", "fill", "pattern")
X_LINE = "- x: f32, shape [4], each element in [-1000.0, 1000.0); NaN not allowed; infinity not allowed"
# adv-stage's held-out spec, seed, and declared contract (tests/core/adversary_support.py HELD_OUT, ITEM_SEED, _item).
HELD_OUT = {"name": "x", "dtype": "f32", "shape": [4], "dist": "uniform", "lo": 0.0, "hi": 10.0}
ITEM_SEED = 11
DECLARED = [{"name": "x", "dtype": "f32", "lo": -1000.0, "hi": 1000.0}]
TT_PAIRS = REPO / "assets" / "bench" / "tt-pairs-v0.yaml"
JSON_0 = '{"inputs":[{"dtype":"f32","fill":500,"name":"x","shape":[4]}]}'
JSON_1 = '{"inputs":[{"dtype":"f32","name":"x","shape":[4],"values":[1,2,3,4]}]}'


def placeholders(text: str) -> set[str]:
    """Return the names of a string.Template's placeholders; fail on a `$` that is not one."""
    found = set()
    for match in Template.pattern.finditer(text):
        assert match.group("invalid") is None, f"a '$' that is not a placeholder at {match.start()}"
        name = match.group("named") or match.group("braced")
        if name:
            found.add(name)
    return found


def prompt_text(name: str) -> str:
    """Return the text of adversary-v0's prompt `name`, failing unless it is plain ASCII with LF line ends."""
    path = PROMPT_DIR / f"{name}.txt"
    assert path.is_file(), f"{path.relative_to(REPO).as_posix()} does not exist (task P16.4)"
    raw = path.read_bytes()
    assert raw.isascii(), f"{name}.txt is not plain ASCII"
    assert b"\r" not in raw, f"{name}.txt has a CR"
    return raw.decode("ascii")


def x(*values: float) -> Proposal:
    """Return a proposal of the f32 array x of `values`."""
    return Proposal((LassiArray("x", "f32", (len(values),), struct.pack(f"<{len(values)}f", *values)),))


def outcome_view(outcomes: Sequence[tuple[Proposal, str]], budget_left: int = 2, **files: Any) -> AdversaryView:
    """Return a view of CONTRACT whose outcomes are the (proposal, code) pairs given."""
    return AdversaryView(
        contract=files.get("contract", CONTRACT),
        source=files.get("source", SOURCE),
        candidate=files.get("candidate", CANDIDATE),
        budget_left=budget_left,
        outcomes=tuple(ProposalOutcome(proposal, code) for proposal, code in outcomes),
    )


def rendered(shown_view: AdversaryView, shown: Sequence[tuple[Proposal, str]] = ()) -> str:
    """Return the content of the one user message render_messages gives; fail unless there is exactly one."""
    messages = llm_module().render_messages(shown_view, PROMPTS, list(shown))
    assert isinstance(messages, list) and len(messages) == 1, f"render_messages gave {messages!r}"
    (message,) = messages
    assert type(message) is Message and message.role == "user", message
    return message.content


def expected(shown_view: AdversaryView, contract: Sequence[str], outcomes: Sequence[str]) -> str:
    """Return propose.txt rendered with the design's fields for `shown_view`, given its contract and outcome lines."""
    fields = {
        "contract": "\n".join(contract),
        "source_files": render_file_blocks(dict(shown_view.source)),
        "candidate_files": render_file_blocks(dict(shown_view.candidate)),
        "budget_left": str(shown_view.budget_left),
        "outcomes": "\n".join(outcomes) if outcomes else "None yet.",
    }
    return render(PROMPTS, "propose", fields)


def shortest_f32(value: float) -> str:
    """Return the shortest %g text that reads back as the same f32 value."""
    target = struct.pack("<f", value)
    for digits in range(1, 18):
        text = f"{value:.{digits}g}"
        if struct.pack("<f", float(text)) == target:
            return text
    return repr(value)


def value_texts(array: LassiArray) -> set[str]:
    """Return the texts an element of a float array could be shown as: repr, its shortest form, and %.6g."""
    code = {"f32": "f", "f64": "d", "f16": "e"}.get(array.dtype)
    if code is None:
        assert array.dtype == "bf16", array.dtype
        words = struct.unpack(f"<{len(array.data) // 2}H", array.data)
        floats = [struct.unpack("<f", struct.pack("<I", word << 16))[0] for word in words]
    else:
        floats = list(struct.unpack(f"<{len(array.data) // struct.calcsize(code)}{code}", array.data))
    texts = set()
    for value in floats:
        texts.update({repr(value), shortest_f32(value), f"{value:.6g}"})
    return texts


def assert_holds_no_held_out_data(text: str, arrays: Sequence[LassiArray]) -> None:
    """Fail when `text` shows a held-out element value (in a form at least 5 characters long) or the data's bytes."""
    for array in arrays:
        shown = sorted(item for item in value_texts(array) if len(item) >= 5 and item in text)
        assert not shown, f"the rendered message shows held-out values of {array.name}: {shown[:5]}"
        assert array.data[:32].hex() not in text, "the rendered message shows held-out bytes as hex"
        assert array.data[:32].decode("latin-1") not in text, "the rendered message shows held-out bytes"


def assert_holds_no_spec_key(text: str) -> None:
    """Fail when `text` shows a held-out spec's `dist` key in any form a mapping would print it."""
    for form in ('"dist"', "'dist'", "dist:", "dist="):
        assert form not in text, f"the rendered message shows the held-out spec key {form}"


# ---------------------------------------------------------------------------
# The prompt files


def test_the_prompt_set_holds_the_two_prompts_as_plain_ascii() -> None:
    assert PROMPT_DIR.is_dir(), "assets/prompts/adversary-v0/ does not exist (task P16.4)"
    for name in ("propose", "malformed"):
        assert prompt_text(name).strip(), f"{name}.txt is empty"


def test_propose_has_exactly_the_designed_placeholders() -> None:
    assert placeholders(prompt_text("propose")) == PROPOSE_PLACEHOLDERS


def test_malformed_has_exactly_the_designed_placeholders() -> None:
    assert placeholders(prompt_text("malformed")) == MALFORMED_PLACEHOLDERS


def test_propose_shows_the_reply_form_and_the_stop() -> None:
    text = prompt_text("propose")
    assert "```json" in text, "the reply form is a json block"
    assert '{"proposals": []}' in text, "the prompt says the empty proposal list stops the search"
    missing = [key for key in REPLY_KEYS if f'"{key}"' not in text]
    assert not missing, f"propose.txt does not show the reply keys {missing}"
    for special in ('"nan"', '"inf"', '"-inf"'):
        assert special in text, f"propose.txt does not show the float string {special}"


def test_both_prompts_render_with_their_fields_alone() -> None:
    fields = {name: f"SYNTHETIC-{name}" for name in PROPOSE_PLACEHOLDERS}
    text = render(PROMPTS, "propose", fields)
    assert all(value in text for value in fields.values())
    answer = render(PROMPTS, "malformed", {"reason": "SYNTHETIC-reason", "budget_left": "3"})
    assert "SYNTHETIC-reason" in answer and "3" in answer


# ---------------------------------------------------------------------------
# render_messages


def test_a_first_view_renders_the_contract_the_files_the_budget_and_no_outcome() -> None:
    first = view(4)
    text = rendered(first)
    assert text == expected(first, [X_LINE], [])
    assert X_LINE in text.splitlines()
    assert "None yet." in text.splitlines()
    assert render_file_blocks(SOURCE) in text and render_file_blocks(CANDIDATE) in text
    assert "// FILE: main.cpp" in text


def test_the_budget_left_is_the_views() -> None:
    shown_view = view(37)
    assert rendered(shown_view) == expected(shown_view, [X_LINE], [])
    assert "37" in rendered(shown_view)


def test_each_file_is_a_file_block_in_sorted_path_order() -> None:
    source = {"src/b.cpp": "// SYNTHETIC b\n", "a.h": "// SYNTHETIC a\n"}
    candidate = {"main.cpp": "// SYNTHETIC candidate without a final newline", "util/x.h": "// SYNTHETIC x\n"}
    shown_view = view(4, source=source, candidate=candidate)
    text = rendered(shown_view)
    assert text == expected(shown_view, [X_LINE], [])
    for path in (*source, *candidate):
        assert f"// FILE: {path}" in text.splitlines()
    assert text.index("// FILE: a.h") < text.index("// FILE: src/b.cpp")


@pytest.mark.parametrize(
    ("spec", "line"),
    [
        pytest.param(
            ContractInput("n", "i32", (1,), 0, 10),
            "- n: i32, shape [1], each element in [0, 10); NaN not allowed; infinity not allowed",
            id="integer",
        ),
        pytest.param(
            ContractInput("y", "f64", (), 0.1, 2.5, allow_nan=True, allow_inf=True),
            "- y: f64, shape [], each element in [0.1, 2.5); NaN allowed; infinity allowed",
            id="scalar-with-both-flags",
        ),
        pytest.param(
            ContractInput("z", "f16", (8,), -1e-05, 65504.0, allow_nan=True),
            "- z: f16, shape [8], each element in [-1e-05, 65504.0); NaN allowed; infinity not allowed",
            id="nan-only",
        ),
        pytest.param(
            ContractInput("w", "bf16", (3,), -1.0, 0.0, allow_inf=True),
            "- w: bf16, shape [3], each element in [-1.0, 0.0); NaN not allowed; infinity allowed",
            id="infinity-only",
        ),
    ],
)
def test_each_contract_input_is_one_line(spec: ContractInput, line: str) -> None:
    contract = InputContract((CONTRACT.inputs[0], spec))
    shown_view = outcome_view([], budget_left=4, contract=contract)
    text = rendered(shown_view)
    assert text == expected(shown_view, [X_LINE, line], [])
    lines = text.splitlines()
    assert lines.index(X_LINE) + 1 == lines.index(line), "one line per input, in the contract's order"


def test_each_outcome_is_one_line_of_its_code() -> None:
    shown_view = outcome_view([(x(5000, 5000, 5000, 5000), "out-of-range"), (x(1, 2, 3, 4), "agreed")])
    text = rendered(shown_view)
    outcomes = ["- proposal 0: out-of-range", "- proposal 1: agreed"]
    assert text == expected(shown_view, [X_LINE], outcomes)
    assert "None yet." not in text.splitlines()


def test_the_agents_own_proposal_is_shown_as_its_json() -> None:
    first, second = x(500, 500, 500, 500), x(1, 2, 3, 4)
    shown_view = outcome_view([(first, "diverged"), (second, "agreed")])
    text = rendered(shown_view, [(first, JSON_0), (second, JSON_1)])
    outcomes = ["- proposal 0: diverged", f"  proposed: {JSON_0}", "- proposal 1: agreed", f"  proposed: {JSON_1}"]
    assert text == expected(shown_view, [X_LINE], outcomes)


def test_an_outcome_whose_proposal_is_not_the_shown_one_gets_no_json() -> None:
    first, other = x(500, 500, 500, 500), x(7, 7, 7, 7)
    shown_view = outcome_view([(other, "agreed"), (first, "diverged")])
    text = rendered(shown_view, [(first, JSON_0), (first, JSON_0)])
    outcomes = ["- proposal 0: agreed", "- proposal 1: diverged", f"  proposed: {JSON_0}"]
    assert text == expected(shown_view, [X_LINE], outcomes)


def test_shown_proposals_past_the_outcomes_are_not_rendered() -> None:
    first, second = x(500, 500, 500, 500), x(1, 2, 3, 4)
    shown_view = outcome_view([(first, "diverged")])
    text = rendered(shown_view, [(first, JSON_0), (second, JSON_1)])
    assert text == expected(shown_view, [X_LINE], ["- proposal 0: diverged", f"  proposed: {JSON_0}"])
    assert JSON_1 not in text


# ---------------------------------------------------------------------------
# Held-out inputs, the item's seed, and the spec never appear


def test_the_adv_stage_held_out_inputs_and_seed_never_appear(tmp_path: Path) -> None:
    contract = input_contract([HELD_OUT], declared=DECLARED, where="adv-stage/scale")
    assert contract == CONTRACT, "the restated contract is adv-stage's"
    (tmp_path / "held-out").mkdir()
    files = generate_inputs([HELD_OUT], ITEM_SEED, tmp_path / "held-out")
    arrays = [read_array(path) for path in files.values()]
    text = rendered(view(4, contract=contract))
    assert_holds_no_held_out_data(text, arrays)
    assert_holds_no_spec_key(text)
    assert re.search(rf"(?<![\w.]){ITEM_SEED}(?![\w.])", text) is None, "the rendered message shows the item's seed"


def test_a_tt_pairs_item_read_for_eval_shows_its_contract_and_never_its_held_out_inputs(tmp_path: Path) -> None:
    item = load_suite(TT_PAIRS).item("loopback", purpose="eval")
    assert item.contract is not None and item.seed is not None
    (tmp_path / "held-out").mkdir()
    files = generate_inputs(list(item.inputs), item.seed, tmp_path / "held-out")
    arrays = [read_array(path) for path in files.values()]
    text = rendered(view(4, contract=item.contract))
    line = "- input: bf16, shape [51200], each element in [0.0, 100.0); NaN not allowed; infinity not allowed"
    assert line in text.splitlines(), "the contract is shown"
    assert_holds_no_held_out_data(text, arrays)
    assert_holds_no_spec_key(text)
