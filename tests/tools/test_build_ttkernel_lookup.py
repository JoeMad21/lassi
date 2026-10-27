"""Tests for task P17.41's changes to tools/build_ttkernel_context.py: the pack and the definitions tree.

plans/spikes/mlir-issues.md, Half 1: the dialect pack is cut at 40,960 bytes,
which left about 100 ops out, among them get_noc_addr_from_bank_id, the
bank-based form a DRAM read needs; and most attempts ended at the verify
step on names and assembly formats the model did not know. The fixes these
tests hold:

- every data-movement op (the ops of the argument, NOC, CB, TensorAccessor,
  and multicast sections of TTKernelOps.td, and the helpers
  DATA_MOVEMENT_HELPERS) is carried whole, after CORE_OPS and before every
  other op, and a limit too small to hold them is refused, never cut;
- an op with a custom assembly format (hasCustomAssemblyFormat) carries the
  format its parser reads, from the format comment of TTKernelOps.cpp at the
  pin, and an op whose format comment is missing is refused;
- `build` also writes the definitions tree DEFS (lassi.prompts.dialect_defs):
  every op, type, and attribute of the dialect by name, its definition exact
  (descriptions included), an op with the constraints it names and its custom
  format, an enum attribute with its enum; only its manifest is tracked;
- the tracked pack's manifest cuts none of the data-movement ops the spike
  names.

The .td and .cpp text is SYNTHETIC; no network is used.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest
import yaml
from test_build_ttkernel_context import REPO, tool, tree

from lassi.prompts.dialect_defs import load_defs
from lassi.toolchains import pins

SECTIONED = '''// SYNTHETIC license line
def IndexLike: AnyTypeOf<[Index, I32]>;

//===----------------------------------------------------------------------===//
// TTKernel SFPU operations
//===----------------------------------------------------------------------===//

def TTKernel_ZebraOp : TTKernel_SFPUOp<"zebra"> {
    let summary = "Zebra op, SYNTHETIC compute op that is no core op.";
    let description = [{
      SYNTHETIC description the pack leaves out and the definitions tree keeps.
    }];
    let arguments = (ins I32:$a);
    let assemblyFormat = [{ `(` $a `)` attr-dict `:` functional-type(operands, results) }];
}

//===----------------------------------------------------------------------===//
// TTKernel NOC operations
//===----------------------------------------------------------------------===//

def TTKernel_NocZapOp : TTKernel_Op<"noc_zap"> {
    let summary = "NocZap, SYNTHETIC data-movement op that is no core op.";
    let arguments = (ins IndexLike:$x, I32:$y);
    let assemblyFormat = [{ `(` $x `,` $y `)` attr-dict `:` functional-type(operands, results) }];
}

//===----------------------------------------------------------------------===//
// TTKernel TensorAccessor operations
//===----------------------------------------------------------------------===//

def TTKernel_ArgsZapOp : TTKernel_Op<"ArgsZap", [AttrSizedOperandSegments]> {
    let summary = "ArgsZap, SYNTHETIC op with a custom assembly format.";
    let arguments = (ins Optional<I32>:$base);
    let hasCustomAssemblyFormat = 1;
}
'''
CUSTOM_CPP = '''// SYNTHETIC ops source
//===----------------------------------------------------------------------===//
// ArgsZapOp custom assembly format
//===----------------------------------------------------------------------===//
// Format:
// - ArgsZap(%base) {attr-dict}

void ArgsZapOp::print(::mlir::OpAsmPrinter &p) {}
'''
TYPES = '''// SYNTHETIC license
class TTKernel_Type<string name, string typeMnemonic> : TypeDef<TTKernel_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def TTKernel_Zap : TTKernel_Type<"Zap", "zap"> {
    let summary = "SYNTHETIC zap type";
}

def TTKernel_ThreadTypeAttr : EnumAttr<TTKernel_Dialect, TTKernel_ThreadType, "thread"> {
  let assemblyFormat = "`<` $value `>`";
}
'''
ENUMS = '''// SYNTHETIC license
def TTKernel_ThreadTypeNoc : I32EnumAttrCase<"Noc", 0, "noc">;

def TTKernel_ThreadType : I32EnumAttr<"ThreadType", "SYNTHETIC thread types",
                           [TTKernel_ThreadTypeNoc]> {
  let cppNamespace = "::syn";
}
'''
# The data-movement ops the spike names (plans/spikes/mlir-issues.md, Half 1, the fixes): argument ops, the
# bank-based NOC address forms, noc_async_read and write and their barriers, and the circular-buffer ops.
SPIKE_DATA_MOVEMENT = (
    "get_arg_val", "get_common_arg_val", "get_compile_time_arg_val", "get_noc_addr", "get_noc_addr_from_bank_id",
    "TensorAccessorArgs", "TensorAccessor", "tensor_accessor.get_noc_addr", "get_interleaved_addr_gen_fast",
    "interleaved_addr_gen_fast.get_noc_addr", "noc_async_read", "noc_async_read_tile", "noc_async_write",
    "noc_async_write_tile", "noc_async_read_barrier", "noc_async_write_barrier", "cb_reserve_back", "cb_push_back",
    "cb_wait_front", "cb_pop_front", "get_write_ptr", "get_read_ptr", "get_tile_size",
)


def sectioned_tree(tmp_path: Path, module: ModuleType) -> Path:
    """Return a SYNTHETIC tree whose ops have section headers, a custom format, a type, and an enum attribute."""
    root = tree(tmp_path, module, ops=SECTIONED, cpp=CUSTOM_CPP)
    (root / module.TYPE_SOURCES[0]).write_bytes(TYPES.encode("ascii"))
    (root / module.TYPE_SOURCES[1]).write_bytes(ENUMS.encode("ascii"))
    return root


def test_data_movement_ops_are_read_by_section_and_by_name() -> None:
    module = tool()
    found = module.data_movement_ops(SECTIONED)
    assert found == ("noc_zap", "ArgsZap"), found
    for helper in module.DATA_MOVEMENT_HELPERS:
        assert module.data_movement_ops(f'def TTKernel_XOp : TTKernel_Op<"{helper}"> {{\n}}\n') == (helper,)
    assert set(module.DATA_MOVEMENT_SECTIONS) >= {"TTKernel NOC operations", "TTKernel CB operations",
                                                   "TTKernel Compile and runtime arguments operations"}


def test_the_pack_keeps_data_movement_ops_before_the_rest_and_refuses_to_cut_one(tmp_path: Path) -> None:
    module = tool()
    root, pin = sectioned_tree(tmp_path, module), pins.read_pin("tt-mlir")
    built = module.build(root, pin, module.LIMIT_BYTES)
    ops = built[f"context/{module.PACK}"]["ttkernel_ops.ttkernel.txt"].decode("ascii")
    assert ops.index("ttkernel.noc_zap") < ops.index("ttkernel.zebra"), "data movement before the rest"
    outcomes = []
    for limit in range(400, 3000, 10):
        try:
            built = module.build(root, pin, limit)
        except ValueError as error:
            assert "data-movement" in str(error), error
            outcomes.append("refused")
            continue
        ops = built[f"context/{module.PACK}"]["ttkernel_ops.ttkernel.txt"].decode("ascii")
        assert "ttkernel.noc_zap" in ops and "ttkernel.ArgsZap" in ops, "every data-movement op is kept whole"
        outcomes.append("built")
    assert outcomes[0] == "refused" and outcomes[-1] == "built"
    assert "refused" not in outcomes[outcomes.index("built"):], "past the smallest limit that holds them, all build"


def test_a_custom_format_op_carries_its_format_from_the_ops_source() -> None:
    module = tool()
    blocks = dict(module.op_blocks(SECTIONED, CUSTOM_CPP))
    assert "customAssemblyFormat" in blocks["ArgsZap"], blocks["ArgsZap"]
    assert f"{module.CPP_SOURCE}:5-6" in blocks["ArgsZap"] and "ArgsZap(%base) {attr-dict}" in blocks["ArgsZap"]
    assert "customAssemblyFormat" not in blocks["noc_zap"]
    with pytest.raises(ValueError, match="ArgsZap"):
        module.op_blocks(SECTIONED, "// SYNTHETIC ops source with no format comment\n")


def test_build_writes_a_definitions_tree_of_every_op_type_and_attribute(tmp_path: Path) -> None:
    module = tool()
    out = tmp_path / "assets"
    module.write(module.build(sectioned_tree(tmp_path, module), pins.read_pin("tt-mlir"), module.LIMIT_BYTES), out)
    loaded = load_defs(module.DEFS, root=out)
    assert loaded.dialect == "ttkernel"
    assert list(loaded.definitions) == ["ttkernel.zebra", "ttkernel.noc_zap", "ttkernel.ArgsZap", "!ttkernel.zap",
                                        "#ttkernel.thread"]
    zebra = loaded.definitions["ttkernel.zebra"]
    assert zebra.startswith('def TTKernel_ZebraOp : TTKernel_SFPUOp<"zebra"> {') and zebra.endswith("}")
    assert "SYNTHETIC description the pack leaves out" in zebra, "a definition is exact, its description included"
    assert "def IndexLike: AnyTypeOf<[Index, I32]>;" in loaded.definitions["ttkernel.noc_zap"], (
        "a constraint the op names is carried with it")
    assert "// - ArgsZap(%base) {attr-dict}" in loaded.definitions["ttkernel.ArgsZap"]
    thread = loaded.definitions["#ttkernel.thread"]
    assert 'I32EnumAttrCase<"Noc", 0, "noc">' in thread and 'I32EnumAttr<"ThreadType"' in thread
    manifest = yaml.safe_load((out / "context" / module.DEFS / "MANIFEST.yaml").read_bytes())
    assert manifest["generator"] == module.GENERATOR and manifest["names"] == {"ops": 3, "types": 1, "attributes": 1}


def test_the_tracked_definitions_manifest_names_the_pin_and_this_tool() -> None:
    module = tool()
    text = (REPO / "assets" / "context" / module.DEFS / "MANIFEST.yaml").read_text(encoding="ascii")
    assert f"commit: {pins.read_pin('tt-mlir')['COMMIT']}" in text and f"generator: {module.GENERATOR}" in text
    assert module.CPP_SOURCE in text


def test_the_tracked_pack_cuts_no_data_movement_op() -> None:
    module = tool()
    manifest = yaml.safe_load((REPO / "assets" / "context" / module.PACK / "MANIFEST.yaml").read_bytes())
    cut = set(manifest["cut_by_limit"])
    assert not cut & set(SPIKE_DATA_MOVEMENT), sorted(cut & set(SPIKE_DATA_MOVEMENT))
    assert not cut & set(module.DATA_MOVEMENT_HELPERS)
    assert not [name for name in cut if name.startswith(("noc_", "cb_", "get_noc", "tensor_accessor."))], cut
