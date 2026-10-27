"""Tests for lassi.prompts.dialect_defs, the dialect definitions a verify error's correction carries (task P17.41).

plans/spikes/mlir-issues.md, Half 1: of 27 attempts, 23 ended at ttmlir-opt's
parse or verify step, with types and ops the dialect does not have and the
ops' assembly formats not followed, and the same parser message often came
back three times, so one message was not enough feedback. The definitions
tree holds the exact definition of every op, type, and attribute of the
pinned dialect (tools/build_ttkernel_context.py); these tests fix how a
tree is read, which names an error gives (its message, and the line of the
model's file it points at), and the nearest names of a name the dialect
lacks. The definitions and messages are SYNTHETIC, in the forms the spike's
records show; nothing is upstream text.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lassi.prompts import dialect_defs
from lassi.prompts.dialect_defs import (
    DialectDefs,
    error_names,
    line_names,
    load_defs,
    nearest_names,
    parse_defs,
)

DEFS_TEXT = (
    "// SYNTHETIC preamble before the first block\n"
    "// ==== ttkernel.TensorAccessorArgs ====\n"
    'def TTKernel_TensorAccessorArgsOp : TTKernel_Op<"TensorAccessorArgs"> {\n'
    "  let hasCustomAssemblyFormat = 1;\n"
    "}\n"
    "// ==== ttkernel.TensorAccessor ====\n"
    'def TTKernel_TensorAccessorOp : TTKernel_Op<"TensorAccessor"> {\n}\n'
    "// ==== ttkernel.get_interleaved_addr_gen_fast ====\n"
    'def TTKernel_GetInterleavedAddrGenFastOp : TTKernel_Op<"get_interleaved_addr_gen_fast"> {\n}\n'
    "// ==== ttkernel.get_compile_time_arg_val ====\n"
    'def TTKernel_GetCompileArgValOp : TTKernel_Op<"get_compile_time_arg_val"> {\n}\n'
    "// ==== !ttkernel.TensorAccessorArgs ====\n"
    'def TTKernel_TensorAccessorArgs : TTKernel_Type<"TensorAccessorArgs", "TensorAccessorArgs"> {\n}\n'
    "// ==== !ttkernel.TensorAccessor ====\n"
    'def TTKernel_TensorAccessor : TTKernel_Type<"TensorAccessor", "TensorAccessor"> {\n}\n'
    "// ==== !ttkernel.interleaved_addr_gen_fast ====\n"
    'def TTKernel_InterleavedAddrGenFast : TTKernel_Type<"InterleavedAddrGenFast", "interleaved_addr_gen_fast"> {\n}\n'
    "// ==== #ttkernel.thread ====\n"
    'def TTKernel_ThreadTypeAttr : EnumAttr<TTKernel_Dialect, TTKernel_ThreadType, "thread"> {\n}\n'
)


def defs() -> DialectDefs:
    """Return the SYNTHETIC definitions."""
    return DialectDefs(name="syn-defs", dialect="ttkernel", definitions=parse_defs(DEFS_TEXT))


def test_parse_defs_reads_each_block_by_its_name_exactly() -> None:
    parsed = parse_defs(DEFS_TEXT)
    assert list(parsed)[:2] == ["ttkernel.TensorAccessorArgs", "ttkernel.TensorAccessor"]
    assert parsed["ttkernel.TensorAccessorArgs"] == (
        'def TTKernel_TensorAccessorArgsOp : TTKernel_Op<"TensorAccessorArgs"> {\n'
        "  let hasCustomAssemblyFormat = 1;\n}"
    )
    assert "#ttkernel.thread" in parsed and "!ttkernel.TensorAccessor" in parsed
    assert dialect_defs.BLOCK_HEAD.format(name="x.y") == "// ==== x.y ===="


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("// no block here\n", "no block"),
        ("// ==== ttkernel.a ====\nA\n// ==== ttkernel.a ====\nB\n", "twice"),
        ("// ==== ttkernel.a ====\n\n", "empty"),
    ],
)
def test_parse_defs_refuses_a_bad_tree(text: str, why: str) -> None:
    with pytest.raises(ValueError, match=why):
        parse_defs(text)


def test_the_dialect_is_read_from_the_names() -> None:
    assert dialect_defs.dialect_of(parse_defs(DEFS_TEXT)) == "ttkernel"
    with pytest.raises(ValueError, match="one dialect"):
        dialect_defs.dialect_of(parse_defs("// ==== a.x ====\nA\n// ==== b.y ====\nB\n"))


@pytest.mark.parametrize(
    ("message", "names"),
    [
        ("unknown  type `tensor_accessor_args` in dialect `ttkernel`", ["!ttkernel.tensor_accessor_args"]),
        ("custom op 'ttkernel.InterleavedAddrGenFast' is unknown", ["ttkernel.InterleavedAddrGenFast"]),
        ("'ttkernel.cb_push_back' op requires attribute 'ttkernel.thread'", ["ttkernel.cb_push_back",
                                                                           "ttkernel.thread"]),
        ("expected ')'", []),
        ("undefined symbol alias id '0'", []),
        ("unknown  type `tile` in dialect `ttcore`", []),
    ],
)
def test_error_names_reads_the_names_a_message_gives(message: str, names: list[str]) -> None:
    assert error_names("ttkernel", message) == names


@pytest.mark.parametrize(
    ("line", "names"),
    [
        ('  %a = ttkernel.TensorAccessorArgs(%cta, %crta, null, "", "")', ["ttkernel.TensorAccessorArgs"]),
        ("  %a = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.tensor_accessor_args",
         ["ttkernel.get_compile_time_arg_val", "!ttkernel.tensor_accessor_args"]),
        ("  %a = ttkernel.get_compile_time_arg_val 0 : (i32) -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>",
         ["ttkernel.get_compile_time_arg_val", "!ttkernel.cb"]),
        ('  "ttkernel.cb_push_back"(%cb, %n) : (!ttkernel.cb<2, i32>, i32) -> ()',
         ["ttkernel.cb_push_back", "!ttkernel.cb"]),
        ("func.func @k() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {", ["#ttkernel.thread"]),
        ("  %c0 = arith.constant 0 : i32", []),
    ],
)
def test_line_names_reads_the_ops_and_types_a_line_holds(line: str, names: list[str]) -> None:
    assert line_names("ttkernel", line) == names


def test_nearest_names_finds_the_dialects_spelling_of_a_borrowed_name() -> None:
    found = nearest_names("!ttkernel.tensor_accessor_args", defs(), 4)
    assert set(found[:2]) == {"!ttkernel.TensorAccessorArgs", "ttkernel.TensorAccessorArgs"}, found
    found = nearest_names("ttkernel.InterleavedAddrGenFast", defs(), 4)
    assert found[0] == "!ttkernel.interleaved_addr_gen_fast", found
    assert "ttkernel.get_interleaved_addr_gen_fast" in found
    assert nearest_names("ttkernel.zzzzqqqq", defs(), 4) == ()
    assert len(nearest_names("!ttkernel.tensor_accessor", defs(), 1)) == 1


def test_load_defs_reads_a_manifest_tree(tmp_path: Path) -> None:
    import hashlib

    import yaml

    tree = tmp_path / "context" / "syn-defs"
    tree.mkdir(parents=True)
    data = DEFS_TEXT.encode("ascii")
    (tree / "syn_defs.ttkernel.txt").write_bytes(data)
    manifest = {"generator": "tools/build_ttkernel_context.py",
                "entries": [{"key": "syn_defs.ttkernel", "sha256": hashlib.sha256(data).hexdigest()}]}
    (tree / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest), encoding="ascii")
    loaded = load_defs("syn-defs", root=tmp_path)
    assert loaded.name == "syn-defs" and loaded.dialect == "ttkernel"
    assert loaded.definitions == parse_defs(DEFS_TEXT)
    (tree / "syn_defs.ttkernel.txt").write_bytes(data + b"x")
    with pytest.raises(ValueError, match="sha256"):
        load_defs("syn-defs", root=tmp_path)
    with pytest.raises(ValueError, match="no definitions tree"):
        load_defs("missing", root=tmp_path)
