"""Tests for tools/build_ttkernel_reference.py, the ttkernel API reference (task P17.44).

The tool renders the pinned dialect's sources (the sources of
tools/build_ttkernel_context.py) into one context pack, every type,
attribute, and op with its textual form, by rule: an op as its
assemblyFormat prints it with each operand's type spelled from its
constraint, an op whose format holds a non-literal attribute or that has
no format in MLIR's generic form, and a hand-written parser's format
comment. The pack loads with lassi.prompts.assets load_tree, and only its
manifest is tracked. The .td text here is SYNTHETIC; no network is used.
When the gitignored reference is built in this checkout, it is checked
against its tracked manifest and for the forms the 2026-10-26 card got
wrong.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from lassi.prompts import assets
from lassi.toolchains import pins

REPO = Path(__file__).resolve().parents[2]
OPS = '''// SYNTHETIC license line
include "x.td"

def IndexLike: AnyTypeOf<[Index, I32]>;

//===----------------------------------------------------------------------===//
// TTKernel CB operations
//===----------------------------------------------------------------------===//

def TTKernel_CBReserveBackOp : TTKernel_Op<"cb_reserve_back"> {
    let summary = "CBReserveBack call.";
    let arguments = (ins TTKernel_CB:$cb, I32:$numPages);
    let assemblyFormat = [{
      `(` $cb `,` $numPages `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_GetNocAddrOp : TTKernel_Op<"tensor_accessor.get_noc_addr"> {
    let summary = "SYNTHETIC first sentence. SYNTHETIC second sentence.";
    let arguments = (ins TTKernel_TensorAccessor:$acc, IndexLike:$id, Optional<I8>:$noc);
    let results = (outs TTKernel_NocAddr:$nocAddr);
    let assemblyFormat = [{
      `(` $acc `,` $id (`,` $noc^)? `)` attr-dict `:` functional-type(operands, results)
    }];
}

//===----------------------------------------------------------------------===//
// TTKernel Compile and runtime arguments operations
//===----------------------------------------------------------------------===//

def TTKernel_ArgResultType : AnyTypeOf<[I32, TTKernel_CB]>;

def TTKernel_GetCompileArgValOp : TTKernel_Op<"get_compile_time_arg_val"> {
    let summary = "Get compile-time arg value.";
    let arguments = (ins I32Attr:$arg_index);
    let results = (outs TTKernel_ArgResultType:$arg_val);
    let assemblyFormat = [{
      `(` $arg_index `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_ReduceInitOp : TTKernel_InitOp<"reduce_init"> {
    let summary = "Init function";
    let arguments = (ins TTKernel_CB:$in_cb, TTKernel_ReduceTypeAttr:$reduce_type, UnitAttr:$full_fp32);
    let assemblyFormat = [{
      `(` $in_cb `,` $reduce_type `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_MatmulBlockInitOp : TTKernel_InitOp<"mm_block_init"> {
    let summary = "Matmul init function";
    let arguments = (ins TTKernel_CB:$in0_cb, I32:$transpose);
}

def TTKernel_TensorAccessorArgsOp : TTKernel_Op<"TensorAccessorArgs"> {
    let summary = "TensorAccessorArgs";
    let arguments = (ins Optional<I32>:$cta_base, Optional<TTKernel_TensorAccessorArgs>:$prev_args);
    let results = (outs TTKernel_TensorAccessorArgs:$result);
    let hasCustomAssemblyFormat = 1;
}
'''
TYPES = '''// SYNTHETIC license
def TTKernel_CB : TTKernel_Type<"CB", "cb"> {
    let summary = "TTKernel cb";
    let parameters = (ins "int64_t":$num_elements, "Type":$element_type);
    let assemblyFormat = "`<` $num_elements `,` $element_type `>`";
}

def TTKernel_NocAddr : TTKernel_Type<"NocAddr", "noc_addr"> {
    let summary = "TTKernel noc address";
}

def TTKernel_TensorAccessor : TTKernel_Type<"TensorAccessor", "TensorAccessor"> {
  let summary = "TensorAccessor type";
}

def TTKernel_TensorAccessorArgs : TTKernel_Type<"TensorAccessorArgs", "TensorAccessorArgs"> {
  let summary = "TensorAccessorArgs type";
}

def TTKernel_ReduceTypeAttr : EnumAttr<TTKernel_Dialect, TTKernel_ReduceType, "reduce_type"> {
  let assemblyFormat = "`<` $value `>`";
}
'''
ENUMS = '''// SYNTHETIC license
def TTKernel_ReduceSum : I32EnumAttrCase<"Sum", 0, "reduce_sum">;
def TTKernel_ReduceMax : I32EnumAttrCase<"Max", 1, "reduce_max">;

def TTKernel_ReduceType : I32EnumAttr<"ReduceType", "TTKernel Reduce Types",
                          [
                            TTKernel_ReduceSum,
                            TTKernel_ReduceMax
                          ]> {
  let genSpecializedAttr = 0;
}
'''
CPP = '''// SYNTHETIC
//===----------------------------------------------------------------------===//
// TensorAccessorArgsOp custom assembly format
//===----------------------------------------------------------------------===//
// Format:
// - SYNTHETIC: TensorAccessorArgs(%cta) or TensorAccessorArgs(prev = %prev)

void TensorAccessorArgsOp::print() {}
'''
GENERATED = REPO / "assets" / "context" / "ttkernel-ref-5f396fd6"


def tool() -> ModuleType:
    """Import tools/build_ttkernel_reference.py as a module."""
    path = REPO / "tools" / "build_ttkernel_reference.py"
    spec = importlib.util.spec_from_file_location("build_ttkernel_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tree(tmp_path: Path, ops: str = OPS) -> Path:
    """Return a SYNTHETIC fetched tt-mlir tree at the pin holding the dialect's four sources."""
    context = tool().context_tool()
    root = tmp_path / "tt-mlir"
    types, enums = context.TYPE_SOURCES
    for path, text in {context.OPS_SOURCE: ops, types: TYPES, enums: ENUMS, context.CPP_SOURCE: CPP}.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(text.encode("ascii"))
    record = f"tt-mlir {pins.read_pin('tt-mlir')['COMMIT']} x\n"
    (root / context.FETCH_RECORD).write_text(record, encoding="ascii")
    return root


def reference(tmp_path: Path) -> str:
    """Return the reference the tool builds from the SYNTHETIC tree."""
    module = tool()
    built = module.build(tree(tmp_path), pins.read_pin("tt-mlir"))
    return built[f"context/{module.REF}"][f"{module.KEY}.txt"].decode("ascii")


def test_an_op_is_its_custom_form_with_each_operand_type_spelled(tmp_path: Path) -> None:
    lines = reference(tmp_path).splitlines()
    assert "ttkernel.cb_reserve_back(%cb, %numPages) : (!ttkernel.cb<...>, i32) -> ()" in lines, (
        "a summary that only repeats the op's name is left out")
    assert ("%nocAddr = ttkernel.tensor_accessor.get_noc_addr(%acc, %id[, %noc]) : (!ttkernel.TensorAccessor, "
            "index | i32[, i8]) -> !ttkernel.noc_addr  // SYNTHETIC first sentence.") in lines


def test_an_integer_attribute_is_a_literal_and_a_constraint_alias_is_resolved(tmp_path: Path) -> None:
    lines = reference(tmp_path).splitlines()
    assert ("%arg_val = ttkernel.get_compile_time_arg_val(<arg_index>) : () -> i32 | !ttkernel.cb<...> where "
            "<arg_index> is an integer literal  // Get compile-time arg value.") in lines


def test_a_non_literal_attribute_or_no_format_gives_the_generic_form(tmp_path: Path) -> None:
    lines = reference(tmp_path).splitlines()
    assert ('"ttkernel.reduce_init"(%in_cb) <{reduce_type = #ttkernel.reduce_type<reduce_sum | reduce_max>, '
            "[full_fp32]}> : (!ttkernel.cb<...>) -> ()  // Init function") in lines
    assert ('"ttkernel.mm_block_init"(%in0_cb, %transpose) : (!ttkernel.cb<...>, i32) -> ()  '
            "// Matmul init function") in lines


def test_a_hand_written_parser_gives_its_format_comment(tmp_path: Path) -> None:
    text = reference(tmp_path)
    assert ("ttkernel.TensorAccessorArgs: its parser is hand-written (lib/Dialect/TTKernel/IR/TTKernelOps.cpp:5-6): "
            "Format: - SYNTHETIC: TensorAccessorArgs(%cta) or TensorAccessorArgs(prev = %prev); operands "
            "%cta_base: [i32], %prev_args: [!ttkernel.TensorAccessorArgs]; result !ttkernel.TensorAccessorArgs") in text


def test_types_attributes_and_sections_are_listed(tmp_path: Path) -> None:
    lines = reference(tmp_path).splitlines()
    assert ("!ttkernel.cb<num_elements, element_type>  where num_elements is int64_t, element_type is Type  "
            "// TTKernel cb") in lines
    assert "!ttkernel.noc_addr  // TTKernel noc address" in lines
    assert "#ttkernel.reduce_type<reduce_sum | reduce_max>" in lines
    assert lines.index("TTKernel CB operations:") < lines.index("TTKernel Compile and runtime arguments operations:")


def test_a_hand_written_parser_without_its_comment_is_refused(tmp_path: Path) -> None:
    module = tool()
    root = tree(tmp_path)
    cpp = root / module.context_tool().CPP_SOURCE
    cpp.write_bytes(b"// SYNTHETIC, no format comment\n")
    with pytest.raises(ValueError, match="TensorAccessorArgs"):
        module.build(root, pins.read_pin("tt-mlir"))


def test_build_writes_a_pack_the_loader_accepts_and_the_same_tree_gives_the_same_bytes(tmp_path: Path) -> None:
    module = tool()
    pin = pins.read_pin("tt-mlir")
    out = tmp_path / "assets"
    module.write(module.build(tree(tmp_path / "a"), pin), out)
    texts = assets.load_tree(out / "context" / module.REF)
    assert list(texts) == [module.KEY] and module.KEY.endswith(".ttkernel")
    assert assets.pack_text(module.REF, texts).startswith("// The ttkernel dialect of tt-mlir")
    assert module.compare(module.build(tree(tmp_path / "b"), pin), out) == []
    (out / "context" / module.REF / f"{module.KEY}.txt").write_bytes(b"SYNTHETIC changed\n")
    assert module.compare(module.build(tree(tmp_path / "b"), pin), out) != []
    assert module.main(["check", "--tree", str(tree(tmp_path / "c")), "--out", str(out)]) == 1


def test_a_tree_at_another_commit_is_refused(tmp_path: Path) -> None:
    module = tool()
    root = tree(tmp_path)
    (root / module.context_tool().FETCH_RECORD).write_text("tt-mlir 0000000 x\n", encoding="ascii")
    with pytest.raises(ValueError, match="not tt-mlir"):
        module.build(root, pins.read_pin("tt-mlir"))
    assert module.main(["build", "--tree", str(root), "--out", str(tmp_path / "out")]) == 2


def test_the_tracked_manifest_reads_the_sources_the_definitions_tree_reads() -> None:
    # The reference and the definitions tree (tools/build_ttkernel_context.py) render the same pinned files, so the
    # job's build on the build host reads exactly the bytes these digests name.
    module = tool()
    manifest = yaml.safe_load((GENERATED / "MANIFEST.yaml").read_text(encoding="ascii"))
    defs = yaml.safe_load((REPO / "assets" / "context" / "ttkernel-defs-5f396fd6" / "MANIFEST.yaml").read_text(
        encoding="ascii"))
    assert manifest["generator"] == module.GENERATOR and manifest["commit"] == pins.read_pin("tt-mlir")["COMMIT"]
    assert manifest["entries"][0]["key"] == module.KEY
    assert manifest["entries"][0]["sources"] == defs["entries"][0]["sources"]


def test_the_built_reference_matches_its_manifest_and_spells_the_pinned_forms() -> None:
    text_path = GENERATED / "ttkernel_ref.ttkernel.txt"
    if not text_path.is_file():
        pytest.skip("the reference is not built here; run tools/build_ttkernel_reference.py build --tree <tt-mlir>")
    text = assets.pack_text("ttkernel-ref-5f396fd6", assets.load_tree(GENERATED))
    assert text.isascii()
    lines = text.splitlines()
    # The forms the 2026-10-26 card got wrong (plans/spikes/mlir-issues.md, Half 1).
    assert ("%arg_val = ttkernel.get_compile_time_arg_val(<arg_index>) : () -> i32 | !ttkernel.cb<...> | "
            "!ttkernel.l1_addr where <arg_index> is an integer literal  // Get compile-time arg value.") in lines
    assert "!ttkernel.TensorAccessorArgs  // TensorAccessorArgs type" in lines
    assert "tensor_accessor_args" not in text.replace("tensor_accessor_args_", "")
    assert any(line.startswith("ttkernel.TensorAccessorArgs: its parser is hand-written") for line in lines)
    assert ("%result = ttkernel.TensorAccessor(%args, %bank_base_address_in, %page_size_in) : "
            "(!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor  // MakeTensorAccessorFromArgs"
            ) in lines
