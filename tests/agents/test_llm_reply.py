"""Tests for the llm adversary's reply form: parse_reply and ReplyError (task P16.4, design section 4).

Bible: Agents And Judges (the Agent interface), Adversarial Setups (item 1;
the adversary rule), Harness Contract (lassi_io dtypes; held-out input
files: the rounding), Agent Rules 6 and 14. Plan: plans/p16-adversarial.md,
P16.4 ("its reply is read strictly as declarative proposals (the task fixes
the form), a malformed reply gives no proposal and a counted refusal").

The contract these tests fix:

- lassi.agents.llm defines parse_reply(text) -> tuple[Proposal, ...] and
  ReplyError, a ValueError whose message is the reason shown to the model.
- The reply holds exactly one fenced block: a line that is exactly
  "```json", trailing spaces allowed, opens it, and the next line that is
  exactly "```" closes it. No other line may start with "```" anywhere in
  the reply (before, inside, or after the block). Text outside the block is
  ignored. Anything else is a ReplyError: no block, two blocks, an unclosed
  block, another fence, "```JSON", a bare "```" opener.
- The block is JSON: duplicate keys and the NaN, Infinity, and -Infinity
  literals are refused. The top level is an object with exactly the key
  "proposals", a list (possibly empty). Each proposal is an object with
  exactly the key "inputs", a list of input objects. Each input object has
  exactly "name" (a str), "dtype" (a str), "shape" (a list of ints, a bool
  refused, a negative int passed through), and exactly one of "values" (a
  list), "fill" (one value), and "pattern" (a non-empty list).
- Encoding to LassiArray(name, dtype, tuple(shape), data): an array whose
  dtype is not a lassi_io DTYPES key, whose shape has a negative dim, or
  whose element count passes MAX_INPUT_ELEMENTS (lassi.harness.contract) is
  passed with empty data, its values not read. values encode in order
  (the count may differ from the shape's); fill repeats one value
  element-count times; pattern repeats and cuts its list to the element
  count.
- A value for an integer dtype is a JSON integer (a bool refused) inside
  the dtype's range (i8 -128..127, u8 0..255, i32, u32, i64); anything else
  is a ReplyError. A value for a float dtype is a JSON int or float (a bool
  refused) rounded to nearest, ties to even, without overflowing the format,
  or one of the strings "nan" (the quiet NaN: sign 0, exponent all ones,
  top fraction bit), "inf", "-inf"; a finite value whose rounding overflows
  the format is a ReplyError, and -0.0 keeps its sign. f32, f64, and f16 are
  compared with struct; bf16 with known bit patterns.
- Everything else the form allows (another dtype or shape than the
  contract's, a value outside its range, a NaN it refuses, a repeated,
  extra, or missing input) is data, passed through for check_inputs.

Every reply here is SYNTHETIC. No value in this module is a measurement:
the bit patterns are those of the IEEE 754 binary16 and binary32 formats and
of bfloat16, the top half of binary32.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import pytest
from llm_support import inputs, llm_module, reply

from lassi.core.interfaces import Proposal
from lassi.harness import LassiArray
from lassi.harness.contract import MAX_INPUT_ELEMENTS

# bfloat16's largest finite value, (2 - 2**-7) * 2**127, and the halfway point past it that rounds up to overflow.
BF16_MAX = (2 - 2**-7) * 2.0**127
BF16_OVERFLOW_TIE = (2 - 2**-8) * 2.0**127
F32_MAX = 3.4028234663852886e38
F64_MAX = 1.7976931348623157e308
# The integer dtypes' ranges and struct codes.
INT_RANGES = {
    "i8": (-128, 127, "b"),
    "u8": (0, 255, "B"),
    "i32": (-(2**31), 2**31 - 1, "i"),
    "u32": (0, 2**32 - 1, "I"),
    "i64": (-(2**63), 2**63 - 1, "q"),
}
# The quiet NaN and the infinities of each float dtype, as bit patterns.
SPECIALS = {
    "f16": (0x7E00, 0x7C00, 0xFC00),
    "bf16": (0x7FC0, 0x7F80, 0xFF80),
    "f32": (0x7FC00000, 0x7F800000, 0xFF800000),
    "f64": (0x7FF8000000000000, 0x7FF0000000000000, 0xFFF0000000000000),
}
BITS_CODES = {"f16": "H", "bf16": "H", "f32": "I", "f64": "Q"}


def parse(text: str) -> tuple[Any, ...]:
    """Return parse_reply(text)."""
    return llm_module().parse_reply(text)


def refused(text: str) -> str:
    """Return the message of the ReplyError parse_reply raises for `text`; it must be a ValueError with a reason."""
    module = llm_module()
    assert issubclass(module.ReplyError, ValueError), "ReplyError must be a ValueError"
    with pytest.raises(module.ReplyError) as info:
        module.parse_reply(text)
    message = str(info.value)
    assert message.strip(), "a ReplyError's message is the reason shown to the model; it may not be empty"
    return message


def block(body: str) -> str:
    """Return a reply that is one json block holding `body`."""
    return f"```json\n{body}\n```"


def one_array(**item: Any) -> LassiArray:
    """Return the one array of the one proposal parse_reply reads from a reply holding the input object `item`."""
    (found,) = parse(reply([inputs(**item)]))
    (array,) = found.arrays
    return array


def bits(array: LassiArray) -> list[int]:
    """Return a float array's elements as bit patterns."""
    code = BITS_CODES[array.dtype]
    return list(struct.unpack(f"<{len(array.data) // struct.calcsize(code)}{code}", array.data))


def proposal_body(item: Any) -> str:
    """Return the JSON text of a reply block whose one proposal holds the one input object `item`."""
    return json.dumps({"proposals": [{"inputs": [item]}]})


# ---------------------------------------------------------------------------
# The fenced block


def test_one_block_with_text_around_it_parses() -> None:
    text = reply(
        [inputs(fill=1.0)],
        before="SYNTHETIC: I think the candidate is wrong.\nUse `x` here.\n",
        after="SYNTHETIC text after the block ``` inline.\n",
    )
    (found,) = parse(text)
    assert type(found) is Proposal
    assert found.arrays == (LassiArray("x", "f32", (4,), struct.pack("<4f", 1, 1, 1, 1)),)


def test_the_result_is_a_tuple_of_proposals_in_order() -> None:
    found = parse(reply([inputs(fill=1)], [inputs(fill=2)], [inputs(fill=3)]))
    assert type(found) is tuple and all(type(item) is Proposal for item in found)
    assert [item.arrays[0].data for item in found] == [struct.pack("<4f", *[value] * 4) for value in (1, 2, 3)]


def test_an_empty_proposal_list_gives_no_proposal() -> None:
    assert parse(block('{"proposals": []}')) == ()


@pytest.mark.parametrize("spaces", [" ", "   "])
def test_trailing_spaces_on_the_opening_line_are_allowed(spaces: str) -> None:
    assert parse(f"```json{spaces}\n" + '{"proposals": []}\n```') == ()


def test_a_closing_fence_at_the_end_of_the_reply_needs_no_newline() -> None:
    assert parse('```json\n{"proposals": []}\n```') == ()
    assert parse('```json\n{"proposals": []}\n```\n') == ()


def test_a_line_that_holds_a_fence_but_does_not_start_with_one_is_not_a_fence() -> None:
    body = proposal_body(inputs(name="a```b", fill=0))
    (found,) = parse(block(body))
    assert found.arrays[0].name == "a```b"


FENCE_REFUSALS = [
    pytest.param("SYNTHETIC reply with no block at all.", id="no-block"),
    pytest.param("", id="empty-reply"),
    pytest.param(block('{"proposals": []}') + "\n" + block('{"proposals": []}'), id="two-blocks"),
    pytest.param('```json\n{"proposals": []}\n', id="unclosed"),
    pytest.param('```JSON\n{"proposals": []}\n```', id="upper-case-tag"),
    pytest.param('```\n{"proposals": []}\n```', id="untagged-opener"),
    pytest.param('```json5\n{"proposals": []}\n```', id="other-tag"),
    pytest.param(' ```json\n{"proposals": []}\n ```', id="indented-fences"),
    pytest.param('```\nSYNTHETIC\n```\n```json\n{"proposals": []}\n```', id="fence-before"),
    pytest.param('```json\n{"proposals": []}\n```\n```cpp\nint main() {}\n```', id="fence-after"),
    pytest.param('```json\n{"proposals":\n```python\n[]}\n```', id="fence-inside"),
    pytest.param('```json\n{"proposals": []}\n```   ', id="closing-line-with-spaces"),
    pytest.param("```json\n```", id="empty-block"),
]


@pytest.mark.parametrize("text", FENCE_REFUSALS)
def test_a_reply_without_exactly_one_json_block_is_refused(text: str) -> None:
    refused(text)


# ---------------------------------------------------------------------------
# The JSON form


JSON_REFUSALS = [
    pytest.param('{"proposals": [', id="does-not-parse"),
    pytest.param('{"proposals": [], "proposals": []}', id="duplicate-top-key"),
    pytest.param("[]", id="top-level-list"),
    pytest.param("{}", id="no-proposals-key"),
    pytest.param('{"proposals": [], "claim": "SYNTHETIC"}', id="extra-top-key"),
    pytest.param('{"proposals": {}}', id="proposals-not-a-list"),
    pytest.param('{"proposals": [[]]}', id="proposal-not-an-object"),
    pytest.param('{"proposals": [{}]}', id="proposal-without-inputs"),
    pytest.param('{"proposals": [{"inputs": [], "why": "SYNTHETIC"}]}', id="proposal-extra-key"),
    pytest.param('{"proposals": [{"inputs": {}}]}', id="inputs-not-a-list"),
    pytest.param('{"proposals": [{"inputs": [[]]}]}', id="input-not-an-object"),
]


@pytest.mark.parametrize("body", JSON_REFUSALS)
def test_a_block_outside_the_json_form_is_refused(body: str) -> None:
    refused(block(body))


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_the_nan_and_infinity_literals_are_refused(literal: str) -> None:
    body = '{"proposals": [{"inputs": [{"name": "x", "dtype": "f32", "shape": [1], "fill": %s}]}]}' % literal
    refused(block(body))


@pytest.mark.parametrize("depth", [1000, 5000])
def test_a_block_that_nests_too_deeply_is_refused_not_raised(depth: int) -> None:
    # A degenerate reply that opens thousands of lists would make json.loads raise RecursionError; it is a malformed
    # reply like any other, so the agent counts it and asks again rather than failing the run.
    message = refused(block('{"proposals": ' + "[" * depth))
    assert "nest" in message, message


def test_a_duplicate_key_in_an_input_object_is_refused() -> None:
    body = '{"proposals": [{"inputs": [{"name": "x", "name": "y", "dtype": "f32", "shape": [1], "fill": 0}]}]}'
    refused(block(body))


INPUT_REFUSALS = [
    pytest.param({"dtype": "f32", "shape": [4], "fill": 0}, id="no-name"),
    pytest.param({"name": "x", "shape": [4], "fill": 0}, id="no-dtype"),
    pytest.param({"name": "x", "dtype": "f32", "fill": 0}, id="no-shape"),
    pytest.param({"name": "x", "dtype": "f32", "shape": [4]}, id="no-data-key"),
    pytest.param({"name": "x", "dtype": "f32", "shape": [4], "fill": 0, "values": [0]}, id="fill-and-values"),
    pytest.param({"name": "x", "dtype": "f32", "shape": [4], "fill": 0, "pattern": [0]}, id="fill-and-pattern"),
    pytest.param({"name": "x", "dtype": "f32", "shape": [4], "values": [0], "pattern": [0]}, id="values-and-pattern"),
    pytest.param({"name": "x", "dtype": "f32", "shape": [4], "fill": 0, "seed": 1}, id="extra-key"),
    pytest.param({"name": 5, "dtype": "f32", "shape": [4], "fill": 0}, id="name-not-a-str"),
    pytest.param({"name": "x", "dtype": 32, "shape": [4], "fill": 0}, id="dtype-not-a-str"),
    pytest.param({"name": "x", "dtype": "f32", "shape": 4, "fill": 0}, id="shape-not-a-list"),
    pytest.param({"name": "x", "dtype": "f32", "shape": [4.0], "fill": 0}, id="shape-float-dim"),
    pytest.param({"name": "x", "dtype": "f32", "shape": [True], "fill": 0}, id="shape-bool-dim"),
    pytest.param({"name": "x", "dtype": "f32", "shape": ["4"], "fill": 0}, id="shape-str-dim"),
    pytest.param({"name": "x", "dtype": "f32", "shape": [4], "values": 0}, id="values-not-a-list"),
    pytest.param({"name": "x", "dtype": "f32", "shape": [4], "pattern": 0}, id="pattern-not-a-list"),
    pytest.param({"name": "x", "dtype": "f32", "shape": [4], "pattern": []}, id="empty-pattern"),
    pytest.param({"name": "x", "dtype": "f32", "shape": [4], "fill": [1]}, id="fill-a-list"),
    pytest.param({"name": "x", "dtype": "f32", "shape": [4], "fill": None}, id="fill-null"),
]


@pytest.mark.parametrize("item", INPUT_REFUSALS)
def test_an_input_object_outside_the_form_is_refused(item: dict[str, Any]) -> None:
    refused(block(proposal_body(item)))


# ---------------------------------------------------------------------------
# Encoding: values, fill, and pattern


def test_values_encode_in_order() -> None:
    assert one_array(values=[1, -2.5, 3, 0.25]) == LassiArray("x", "f32", (4,), struct.pack("<4f", 1, -2.5, 3, 0.25))


def test_fill_repeats_one_value_over_the_element_count() -> None:
    assert one_array(shape=[2, 3], fill=7) == LassiArray("x", "f32", (2, 3), struct.pack("<6f", *[7.0] * 6))
    assert one_array(dtype="i32", shape=[3], fill=-5) == LassiArray("x", "i32", (3,), struct.pack("<3i", -5, -5, -5))


def test_a_scalar_holds_one_element_and_an_empty_shape_none() -> None:
    assert one_array(shape=[], fill=2) == LassiArray("x", "f32", (), struct.pack("<f", 2))
    assert one_array(shape=[0], fill=2) == LassiArray("x", "f32", (0,), b"")
    assert one_array(shape=[3, 0], pattern=[1, 2]) == LassiArray("x", "f32", (3, 0), b"")


def test_pattern_repeats_and_cuts_to_the_element_count() -> None:
    assert one_array(shape=[7], pattern=[1, 2, 3]).data == struct.pack("<7f", 1, 2, 3, 1, 2, 3, 1)
    assert one_array(shape=[2], pattern=[1, 2, 3, 4, 5]).data == struct.pack("<2f", 1, 2)
    assert one_array(dtype="u8", shape=[2, 2], pattern=[9]).data == bytes([9, 9, 9, 9])


def test_a_values_count_other_than_the_shape_is_passed_through() -> None:
    # The stage then refuses the set as unreadable; the reply form does not.
    assert one_array(shape=[4], values=[1, 2, 3]) == LassiArray("x", "f32", (4,), struct.pack("<3f", 1, 2, 3))
    assert one_array(shape=[2], values=[1, 2, 3]) == LassiArray("x", "f32", (2,), struct.pack("<3f", 1, 2, 3))


# ---------------------------------------------------------------------------
# Encoding: integer dtypes


@pytest.mark.parametrize("dtype", sorted(INT_RANGES))
def test_an_integer_dtype_takes_its_whole_range(dtype: str) -> None:
    lo, hi, code = INT_RANGES[dtype]
    assert one_array(dtype=dtype, shape=[3], values=[lo, hi, 0 if lo <= 0 else lo]).data == struct.pack(
        f"<3{code}", lo, hi, 0 if lo <= 0 else lo
    )


@pytest.mark.parametrize("dtype", sorted(INT_RANGES))
@pytest.mark.parametrize("side", ["below", "above"])
def test_an_integer_outside_its_dtype_is_refused(dtype: str, side: str) -> None:
    lo, hi, _ = INT_RANGES[dtype]
    value = lo - 1 if side == "below" else hi + 1
    refused(block(proposal_body(inputs(dtype=dtype, shape=[1], values=[value]))))
    refused(block(proposal_body(inputs(dtype=dtype, shape=[1], fill=value))))
    refused(block(proposal_body(inputs(dtype=dtype, shape=[1], pattern=[0, value]))))


@pytest.mark.parametrize("value", [1.0, 2.5, True, False, "1", "nan", "inf", None, [1]])
@pytest.mark.parametrize("dtype", ["i8", "i32", "u32", "i64"])
def test_an_integer_dtype_takes_only_json_integers(dtype: str, value: Any) -> None:
    refused(block(proposal_body(inputs(dtype=dtype, shape=[1], values=[value]))))


# ---------------------------------------------------------------------------
# Encoding: float dtypes


F32_VALUES = [0.1, 1 / 3, -2.5, 1e-45, -1e-46, 1 + 2**-24, 1 + 3 * 2**-24, 16777217, F32_MAX, -F32_MAX, 0, 7]
F64_VALUES = [0.1, -0.0, 5e-324, F64_MAX, -F64_MAX, 2**53 + 1, 0, 1e300]
F16_VALUES = [0.1, 1.0, -2.0, 65504, -65504, 2**-24, 2**-25, 3 * 2**-25, 1 + 2**-11, 1 + 3 * 2**-11, 0.333, 3]


@pytest.mark.parametrize(("dtype", "code", "values"), [("f32", "f", F32_VALUES), ("f64", "d", F64_VALUES),
                                                        ("f16", "e", F16_VALUES)])
def test_a_float_dtype_rounds_to_nearest_ties_to_even_as_struct_does(
    dtype: str, code: str, values: list[float]
) -> None:
    expected = struct.pack(f"<{len(values)}{code}", *values)
    assert one_array(dtype=dtype, shape=[len(values)], values=values).data == expected


@pytest.mark.parametrize(
    ("value", "pattern"),
    [
        pytest.param(1.0, 0x3F80, id="one"),
        pytest.param(-2.0, 0xC000, id="minus-two"),
        pytest.param(0.1, 0x3DCD, id="a-tenth-rounds-up"),
        pytest.param(1 + 2**-8, 0x3F80, id="tie-to-even-down"),
        pytest.param(1 + 3 * 2**-8, 0x3F82, id="tie-to-even-up"),
        pytest.param(1 + 2**-8 + 2**-20, 0x3F81, id="past-the-tie"),
        pytest.param(BF16_MAX, 0x7F7F, id="largest"),
        pytest.param(-BF16_MAX, 0xFF7F, id="lowest"),
        pytest.param(2.0**-133, 0x0001, id="smallest-subnormal"),
        pytest.param(2.0**-134, 0x0000, id="half-the-smallest-subnormal-ties-to-zero"),
        pytest.param(0.0, 0x0000, id="zero"),
        pytest.param(-0.0, 0x8000, id="minus-zero"),
        pytest.param(3, 0x4040, id="a-json-integer"),
    ],
)
def test_bf16_rounds_to_its_known_bit_patterns(value: float, pattern: int) -> None:
    assert bits(one_array(dtype="bf16", shape=[1], values=[value])) == [pattern]


@pytest.mark.parametrize(
    ("value", "pattern"),
    [
        pytest.param(1.0, 0x3C00, id="one"),
        pytest.param(0.1, 0x2E66, id="a-tenth"),
        pytest.param(65504, 0x7BFF, id="largest"),
        pytest.param(2.0**-24, 0x0001, id="smallest-subnormal"),
        pytest.param(2.0**-25, 0x0000, id="tie-to-even-zero"),
        pytest.param(3 * 2.0**-25, 0x0002, id="tie-to-even-two"),
        pytest.param(-0.0, 0x8000, id="minus-zero"),
    ],
)
def test_f16_rounds_to_its_known_bit_patterns(value: float, pattern: int) -> None:
    assert bits(one_array(dtype="f16", shape=[1], values=[value])) == [pattern]


@pytest.mark.parametrize("dtype", sorted(SPECIALS))
def test_the_nan_and_infinity_strings_give_the_quiet_nan_and_the_infinities(dtype: str) -> None:
    assert bits(one_array(dtype=dtype, shape=[3], values=["nan", "inf", "-inf"])) == list(SPECIALS[dtype])
    assert bits(one_array(dtype=dtype, shape=[2], fill="nan")) == [SPECIALS[dtype][0]] * 2
    assert bits(one_array(dtype=dtype, shape=[3], pattern=["-inf", 0])) == [SPECIALS[dtype][2], 0, SPECIALS[dtype][2]]


@pytest.mark.parametrize("dtype", sorted(SPECIALS))
def test_minus_zero_keeps_its_sign(dtype: str) -> None:
    sign = 1 << (struct.calcsize(BITS_CODES[dtype]) * 8 - 1)
    assert bits(one_array(dtype=dtype, shape=[2], values=[-0.0, 0.0])) == [sign, 0]


FLOAT_REFUSALS = [
    pytest.param("f32", "1e39", id="f32-1e39"),
    pytest.param("f32", "-3.5e38", id="f32-past-the-lowest"),
    pytest.param("f32", str(10**39), id="f32-huge-integer"),
    pytest.param("f16", "65520", id="f16-tie-past-the-largest"),
    pytest.param("f16", "70000", id="f16-70000"),
    pytest.param("bf16", repr(BF16_OVERFLOW_TIE), id="bf16-tie-past-the-largest"),
    pytest.param("bf16", repr(F32_MAX), id="bf16-f32-largest"),
    pytest.param("f64", "1e309", id="f64-past-the-largest"),
    pytest.param("f64", str(10**400), id="f64-huge-integer"),
    pytest.param("f32", "true", id="bool"),
    pytest.param("f32", '"NaN"', id="string-NaN"),
    pytest.param("f32", '"Infinity"', id="string-Infinity"),
    pytest.param("f32", '"1.5"', id="string-number"),
    pytest.param("f32", '"+inf"', id="string-plus-inf"),
    pytest.param("f64", "null", id="null"),
    pytest.param("bf16", "{}", id="object"),
]


@pytest.mark.parametrize(("dtype", "value"), FLOAT_REFUSALS)
def test_a_float_value_the_format_cannot_hold_is_refused(dtype: str, value: str) -> None:
    body = '{"proposals": [{"inputs": [{"name": "x", "dtype": "%s", "shape": [1], "values": [%s]}]}]}'
    refused(block(body % (dtype, value)))


def test_the_largest_finite_values_are_accepted() -> None:
    assert bits(one_array(dtype="f32", shape=[1], values=[F32_MAX])) == [0x7F7FFFFF]
    assert bits(one_array(dtype="f16", shape=[1], values=[65504])) == [0x7BFF]
    assert bits(one_array(dtype="bf16", shape=[1], values=[BF16_MAX])) == [0x7F7F]
    assert bits(one_array(dtype="f64", shape=[1], values=[F64_MAX])) == [0x7FEFFFFFFFFFFFFF]


# ---------------------------------------------------------------------------
# Passed through with empty data: an unknown dtype, a negative dim, a shape past the cap


@pytest.mark.parametrize("dtype", ["f128", "F32", "float", "int", ""])
def test_an_unknown_dtype_is_passed_with_empty_data(dtype: str) -> None:
    assert one_array(dtype=dtype, values=[1, 2, 3, 4]) == LassiArray("x", dtype, (4,), b"")


def test_an_unknown_dtype_reads_no_value() -> None:
    # Each value is of the form (a number or a string) but would be refused if read for any dtype.
    assert one_array(dtype="f128", values=[1e39, "SYNTHETIC", 10**400]).data == b""
    assert one_array(dtype="f128", fill="SYNTHETIC").data == b""
    assert one_array(dtype="f128", pattern=[300, -1]).data == b""


@pytest.mark.parametrize("shape", [[-1], [4, -2], [-3, 0]])
def test_a_negative_dim_is_passed_through_with_empty_data(shape: list[int]) -> None:
    assert one_array(shape=shape, fill=1) == LassiArray("x", "f32", tuple(shape), b"")
    assert one_array(shape=shape, fill=1e39).data == b"", "the values of such an array are not read"


def test_a_shape_at_the_element_cap_is_encoded_and_one_past_it_is_not() -> None:
    assert one_array(dtype="u8", shape=[MAX_INPUT_ELEMENTS], fill=1).data == b"\x01" * MAX_INPUT_ELEMENTS
    assert one_array(dtype="u8", shape=[1024, MAX_INPUT_ELEMENTS // 1024], fill=0).data == bytes(MAX_INPUT_ELEMENTS)
    past = MAX_INPUT_ELEMENTS + 1
    assert one_array(dtype="u8", shape=[past], fill=1) == LassiArray("x", "u8", (past,), b"")
    assert one_array(dtype="u8", shape=[1024, 1025], fill=1).data == b""
    assert one_array(dtype="f32", shape=[2000000], fill=0) == LassiArray("x", "f32", (2000000,), b"")
    assert one_array(dtype="i8", shape=[past], fill=300).data == b"", "the values of such an array are not read"


# ---------------------------------------------------------------------------
# Passed through as data: everything check_inputs refuses


def test_out_of_contract_data_is_passed_through() -> None:
    assert one_array(fill=5000).data == struct.pack("<4f", *[5000.0] * 4), "a value outside [lo, hi) is data"
    assert one_array(dtype="i32", values=[1, 2, 3, 4]).dtype == "i32", "another dtype is data"
    assert one_array(shape=[3], values=[1, 2, 3]).shape == (3,), "another shape is data"
    assert one_array(name="bad name/", fill=0).name == "bad name/", "a name lassi_io refuses is data"


def test_repeated_extra_and_missing_inputs_are_passed_through() -> None:
    (repeated, extra, empty) = parse(
        reply([inputs(fill=1), inputs(fill=2)], [inputs(fill=1), inputs(name="w", fill=1)], [])
    )
    assert [array.name for array in repeated.arrays] == ["x", "x"]
    assert [array.name for array in extra.arrays] == ["x", "w"]
    assert empty.arrays == ()
