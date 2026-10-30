"""Tests for tools/build_ttkernel_reference.py over the TTIR and D2M dialects (task P17.47).

The tool, generalized over a dialect (DIALECTS), renders TTIR and D2M from
the pinned TableGen sources as it renders ttkernel (task P17.44, whose
tests stay in test_build_ttkernel_reference.py), with the rules the larger
dialects need: an op's arguments, results, and format from the class it
derives from (a `dag` field and !con included), ops grouped by family (the
section and the class), attributes a format leaves to attr-dict in { },
type($x), an enum parameter's cases, an enum attribute with no format as
MLIR prints it, operandSegmentSizes and regions in a generic form, the
ttcore types and attributes the ops name, and a hand-written parser's
comment inside its parse function. The .td and C++ text here is SYNTHETIC;
no network is used. When the gitignored references are built in this
checkout, each is checked against its tracked manifest and for forms of the
pinned dialects.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from lassi.prompts import assets
from lassi.toolchains import pins

REPO = Path(__file__).resolve().parents[2]
TTIR_BASE = '''// SYNTHETIC
def TTIR_Dialect : Dialect { let name = "ttir"; }
class TTIR_Op<string mnemonic, list<Trait> traits = []> : Op<TTIR_Dialect, mnemonic, traits>;
def AnyRankedTensorOrMemRef: AnyTypeOf<[AnyRankedTensor, AnyNon0RankedMemRef]>;
'''
TTIR_OPS = '''// SYNTHETIC
include "x.td"

//===----------------------------------------------------------------------===//
// TTIR top level named ops
//===----------------------------------------------------------------------===//

class TTIR_NamedOp<string mnemonic, list<Trait> traits = []> :
    TTIR_Op<mnemonic, [Pure] # traits>;

class TTIR_ElementwiseUnaryOp<string mnemonic, list<Trait> traits = []> :
    TTIR_NamedOp<mnemonic, traits> {
    let summary = "Base class for elementwise operations with one input tensor.";
    let description = [{
      SYNTHETIC: %result = ttir.abs(%input) : tensor<4x4xf32> -> tensor<4x4xf32>
    }];
    dag commonArgs = (ins AnyRankedTensor:$input);
    let arguments = commonArgs;
    let results = (outs AnyRankedTensor:$result);
}

def TTIR_AbsOp: TTIR_ElementwiseUnaryOp<"abs"> {
    let summary = "Elementwise absolute value operation.";
}

def TTIR_FullOp : TTIR_NamedOp<"full"> {
    let summary = "SYNTHETIC full.";
    let arguments = (ins DenseI32ArrayAttr:$shape, AnyAttrOf<[F32Attr, I32Attr]>:$fill_value);
    let results = (outs AnyRankedTensor:$result);
}

def TTIR_TypecastOp: TTIR_ElementwiseUnaryOp<"typecast"> {
    let summary = "Elementwise type casting operation.";
    let arguments = !con(commonArgs,
                         (ins DefaultValuedAttr<BoolAttr, "false">:$conservative_folding));
}

def TTIR_EmptyOp : TTIR_Op<"empty"> {
    let summary = "Empty tensor allocation operation.";
    let results = (outs AnyRankedTensor:$result);
    let assemblyFormat = "`(` `)` attr-dict `:` type($result)";
}

def TTIR_RmsNormOp : TTIR_NamedOp<"rms_norm", [AttrSizedOperandSegments]> {
    let summary = "RMS normalization operation";
    let arguments = (ins AnyRankedTensor:$input, Optional<AnyRankedTensor>:$weight,
                         // SYNTHETIC comment, with a comma
                         OptionalAttr<F32Attr>:$epsilon);
    let results = (outs AnyRankedTensor:$result);
}

def TTIR_ToLayoutOp : TTIR_Op<"to_layout"> {
    let arguments = (ins AnyRankedTensorOrMemRef:$input, OptionalAttr<TTCore_MetalLayoutAttr>:$layout);
    let results = (outs AnyRankedTensor:$result);
}

def TTIR_ReduceOp : TTIR_NamedOp<"reduce"> {
    let arguments = (ins AnyRankedTensor:$input, TTIR_ReduceDimAttr:$dim);
    let results = (outs AnyRankedTensor:$result);
}
'''
TTIR_TYPES = '''// SYNTHETIC
def TTIR_Semaphore : TTIR_Type<"Semaphore", "semaphore"> {
    let summary = "TTIR semaphore type.";
}
'''
TTIR_ATTRS = '''// SYNTHETIC
def TTIR_ReduceDimAttr : EnumAttr<TTIR_Dialect, TTIR_ReduceDim, "reduce_dim">;

def TTIR_ThreadAttr : AttrDef<TTIR_Dialect, "Thread", [], "::mlir::Attribute"> {
  let mnemonic = "thread";
  let summary = "Thread information for a generic op.";
  let parameters = (ins "ThreadType":$threadType, OptionalParameter<"SymbolRefAttr">:$kernelSymbol);
  let assemblyFormat = [{ `<` $threadType (`,` $kernelSymbol^)? `>` }];
}

def TTIR_ShouldHoistAttr : AttrDef<TTIR_Dialect, "ShouldHoist", [], "::mlir::Attribute"> {
  let mnemonic = "should_hoist";
  let summary = "SYNTHETIC marker";
}
'''
TTIR_ENUMS = '''// SYNTHETIC
def TTIR_ReduceDimRow : I32EnumAttrCase<"R", 0>;
def TTIR_ReduceDimCol : I32EnumAttrCase<"C", 1>;
def TTIR_ReduceDim : I32EnumAttr<"ReduceDim", "TTIR ReduceDim", [TTIR_ReduceDimRow, TTIR_ReduceDimCol]> {
  let genSpecializedAttr = 0;
}
def TTIR_ThreadTypeCompute : I32EnumAttrCase<"Compute", 0, "compute">;
def TTIR_ThreadType : I32EnumAttr<"ThreadType", "TTIR ThreadType", [
                                  TTIR_ThreadTypeCompute,
                                ]> {
  let genSpecializedAttr = 1;
}
'''
TTCORE_TYPES = '''// SYNTHETIC
class TTCore_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<TTCore_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

def TTCore_GridAttr : TTCore_Attr<"Grid", "grid"> {
  let summary = "TT grid attribute";
  let parameters = (ins ArrayRefParameter<"int64_t">:$shape);
  let assemblyFormat = "`<` custom<DimensionList>($shape) `>`";
}

def TTCore_ChipDescAttr : TTCore_Attr<"ChipDesc", "chip_desc"> {
  let summary = "SYNTHETIC named by no op";
  let parameters = (ins "int":$x);
  let assemblyFormat = "`<` $x `>`";
}

def TTCore_MetalLayoutAttr : TTCore_Attr<"MetalLayout", "metal_layout"> {
  let summary = "SYNTHETIC layout";
  let parameters = (ins
    // SYNTHETIC: the logical shape, a comment with a comma
    ArrayRefParameter<"int64_t">:$logical_shape,
    AttrParameter<"OOBVal", "fill">:$oob_val);
  let assemblyFormat = "`<` `logical_shape` `=` custom<DimensionList>($logical_shape) `,` $oob_val `>`";
}

def TTCore_MemorySpaceAttr : EnumAttr<TTCore_Dialect, TTCore_MemorySpace, "memory_space"> {
  let assemblyFormat = "`<` $value `>`";
}

def TTCore_Tile : TTCore_Type<"Tile", "tile"> {
    let summary = "TT tile";
    let parameters = (ins ArrayRefParameter<"int64_t">:$shape, "DataType":$dataType);
    let assemblyFormat = "`<` custom<DimensionList>($shape) `,` $dataType `>`";
}
'''
TTCORE_ENUMS = '''// SYNTHETIC
def TTCore_Float32 : I32EnumAttrCase<"Float32", 0, "f32">;
def TTCore_BFloat16 : I32EnumAttrCase<"BFloat16", 2, "bf16">;
def TTCore_DataType : I32EnumAttr<"DataType", "TT DataTypes", [TTCore_Float32, TTCore_BFloat16]> {}
def TTCore_Undef : I32EnumAttrCase<"Undef", 0, "undef">;
def TTCore_Zero : I32EnumAttrCase<"Zero", 1, "zero">;
def TTCore_OOBVal : I32EnumAttr<"OOBVal", "TT OOBVal", [TTCore_Undef, TTCore_Zero]> {}
def TTCore_DeviceL1 : I32EnumAttrCase<"DeviceL1", 3, "l1">;
def TTCore_MemorySpace : I32EnumAttr<"MemorySpace", "TT MemorySpace", [TTCore_DeviceL1]> {}
'''
D2M_BASE = '''// SYNTHETIC
class D2M_Op<string mnemonic, list<Trait> traits = []> :
        Op<D2M_Dialect, mnemonic, traits>;
def AnyRankedTensorOrMemRef: AnyTypeOf<[AnyRankedTensor, AnyNon0RankedMemRef]>;
'''
D2M_OPS = r'''// SYNTHETIC
def D2M_GenericOp
    : D2M_Op<
          "generic",
          [AttrSizedOperandSegments, NoTerminator]> {
  let summary = "Generically dispatch work to a grid of cores (D2M).";
  let arguments = (ins Variadic<AnyRankedTensorOrMemRef>:$inputs,
      Variadic<AnyRankedTensorOrMemRef>:$outputs, TTCore_GridAttr:$grid,
      D2M_ThreadArrayAttr:$threads);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region VariadicRegion<AnyRegion>:$regions);
  let assemblyFormat = [{ attr-dict `\n`
    ` ` `ins` `(` $inputs (`:` type($inputs)^)? `)` `\n`
    `outs` `(` $outputs  `:` type($outputs) `)` ` `  $regions (`:`  type($results)^ )?
    }];
}

def D2M_SpatialOp : D2M_Op<"spatial"> {
  let summary = "SYNTHETIC region op.";
  let arguments = (ins AnyRankedTensor:$input);
  let regions = (region SizedRegion<1>:$body);
}
'''
D2M_REGION_OPS = '''// SYNTHETIC
//===----------------------------------------------------------------------===//
// D2M Generic Region Math Ops
//===----------------------------------------------------------------------===//

class D2M_GenericRegionOp<string mnemonic, list<Trait> traits = []> :
    D2M_Op<mnemonic, traits> {}

class D2M_GenericRegionComputeFPUOrSFPUBinary<string mnemonic, list<Trait> traits = []> :
    D2M_GenericRegionOp<mnemonic, traits> {
  let extraClassDeclaration = [{
    bool isFPUBinaryOp() { return true; }
}];
}

def TileOrScalar : AnyTypeOf<[TTCore_Tile, AnyFloat]>;

def D2M_TileAddOp : D2M_GenericRegionComputeFPUOrSFPUBinary<"tile_add", [Pure]> {
    let summary = "D2M Tile Add Op";
    let arguments = (ins TTCore_Tile:$lhs, TileOrScalar:$rhs);
    let results = (outs TTCore_Tile:$result);
}

//===----------------------------------------------------------------------===//
// D2M Generic Region Control Ops
//===----------------------------------------------------------------------===//

def D2M_WaitOp : D2M_GenericRegionOp<"wait"> {
  let summary = "Wait from circular buffer.";
  let arguments = (ins D2M_CB:$cb);
  let results = (outs AnyRankedTensorOrMemRef:$result);
  let assemblyFormat = [{ $cb attr-dict `:` type($cb) `->` type($result) }];
}

class D2M_IndexOp<string mnemonic> : D2M_GenericRegionOp<mnemonic> {
  let extraClassDeclaration = [{
    int64_t getDimValue() { return 0; }
}];
  let results = (outs Index:$result);
}

def D2M_IterIndexOp : D2M_IndexOp<"iter_index"> {
    let arguments = (ins I64Attr:$dim);
    let assemblyFormat = [{ `(` $dim `)` attr-dict `:` type($result) }];
}

def D2M_RemoteLoadOp : D2M_GenericRegionOp<"remote_load"> {
    let summary = "D2M Remote/Aliased Load Op";
    let arguments = (ins Optional<AnyRankedTensorOrMemRef>:$localBuffer,
        Arg<AnyRankedTensorOrMemRef, "the reference to load from",
        [MemRead]>:$memref,
        Variadic<Index>:$indices);
    let results = (outs Optional<AnyRankedTensorOrMemRef>:$result);
    let hasCustomAssemblyFormat = 1;
}
'''
D2M_TYPES = '''// SYNTHETIC
def D2M_CB : D2M_Type<"CB", "cb"> {
    let summary = "D2M circular buffer type.";
    let parameters = (ins "ShapedType":$underlying);
    let assemblyFormat = "`<` $underlying `>`";
}
'''
D2M_ATTRS = '''// SYNTHETIC
def D2M_ThreadAttr : AttrDef<D2M_Dialect, "Thread", [], "::mlir::Attribute"> {
  let mnemonic = "thread";
  let parameters = (ins "ThreadType":$threadType);
  let assemblyFormat = [{ `<` $threadType `>` }];
}

def D2M_ThreadArrayAttr : TypedArrayAttrBase<D2M_ThreadAttr, "">;
'''
D2M_ENUMS = '''// SYNTHETIC
def D2M_ThreadTypeCompute : I32EnumAttrCase<"Compute", 0, "compute">;
def D2M_ThreadTypeUnified : I32EnumAttrCase<"Unified", 2, "unified">;
def D2M_ThreadType : I32EnumAttr<"ThreadType", "D2M ThreadType", [
  D2M_ThreadTypeCompute,
  D2M_ThreadTypeUnified,
]> {}
'''
D2M_CPP = '''//===----------------------------------------------------------------------===//
// RemoteLoadOp Custom Assembly Format
//===----------------------------------------------------------------------===//

ParseResult RemoteLoadOp::parse(OpAsmParser &parser, OperationState &result) {
  // SYNTHETIC first comment
  // Format: SYNTHETIC: %localBuffer %memref[...]
  //         SYNTHETIC: or %memref[...] into %cb
  OpAsmParser::UnresolvedOperand localBuffer;
}
'''


def tool() -> ModuleType:
    """Import tools/build_ttkernel_reference.py as a module."""
    path = REPO / "tools" / "build_ttkernel_reference.py"
    spec = importlib.util.spec_from_file_location("build_ttkernel_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def texts(name: str) -> dict[str, str]:
    """Return {source path: SYNTHETIC text} for every source of dialect `name` (ttir or d2m)."""
    module = tool()
    sources = module.DIALECTS[name]
    by_name = {
        "TTIROps.td": TTIR_OPS, "TTIROpsTypes.td": TTIR_TYPES, "TTIROpsAttrs.td": TTIR_ATTRS,
        "TTIROpsEnums.td": TTIR_ENUMS, "TTIRBase.td": TTIR_BASE, "TTCoreOpsTypes.td": TTCORE_TYPES,
        "TTCoreOpsEnums.td": TTCORE_ENUMS, "D2MOps.td": D2M_OPS, "D2MGenericRegionOps.td": D2M_REGION_OPS,
        "D2MOpsTypes.td": D2M_TYPES, "D2MOpsAttrs.td": D2M_ATTRS, "D2MOpsEnums.td": D2M_ENUMS,
        "D2MBase.td": D2M_BASE, "D2MGenericRegionOps.cpp": D2M_CPP,
    }
    return {path: by_name[Path(path).name] for path in sources.sources}


def tree(tmp_path: Path, name: str) -> Path:
    """Return a SYNTHETIC fetched tt-mlir tree at the pin holding dialect `name`'s sources."""
    root = tmp_path / f"tt-mlir-{name}"
    for path, text in texts(name).items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(text.encode("ascii"))
    record = f"tt-mlir {pins.read_pin('tt-mlir')['COMMIT']} x\n"
    (root / tool().context_tool().FETCH_RECORD).write_text(record, encoding="ascii")
    return root


def reference(tmp_path: Path, name: str) -> list[str]:
    """Return the lines of the reference the tool builds for dialect `name` from its SYNTHETIC tree."""
    module = tool()
    sources = module.DIALECTS[name]
    built = module.build(tree(tmp_path, name), pins.read_pin("tt-mlir"), sources)
    return built[f"context/{sources.ref}"][f"{sources.key}.txt"].decode("ascii").splitlines()


def test_the_dialects_are_named_and_ttkernel_keeps_its_pack_and_rules() -> None:
    module = tool()
    assert sorted(module.DIALECTS) == ["d2m", "ttir", "ttkernel"]
    assert (module.REF, module.KEY) == ("ttkernel-ref-5f396fd6", "ttkernel_ref.ttkernel")
    assert [(d.ref, d.key) for d in (module.TTIR, module.D2M)] == [
        ("ttir-ref-5f396fd6", "ttir_ref.ttir"), ("d2m-ref-5f396fd6", "d2m_ref.d2m")]
    assert not module.TTKERNEL.extended and not module.TTKERNEL.families and not module.TTKERNEL.foreign
    assert all(d.extended and d.families for d in (module.TTIR, module.D2M))


def test_an_op_takes_its_arguments_results_and_form_from_its_class(tmp_path: Path) -> None:
    lines = reference(tmp_path, "ttir")
    assert '%result = "ttir.abs"(%input) : (tensor<...>) -> tensor<...>  // Elementwise absolute value operation.' \
        in lines, "a dag field the class names gives the arguments"
    assert ('%result = "ttir.typecast"(%input) <{[conservative_folding = true | false]}> : (tensor<...>) -> '
            "tensor<...>  // Elementwise type casting operation.") in lines, "!con joins the class's dag and the def's"
    assert "%result = ttir.empty() : tensor<...>  // Empty tensor allocation operation." in lines


def test_attributes_and_segments_in_the_generic_form(tmp_path: Path) -> None:
    lines = reference(tmp_path, "ttir")
    assert ('%result = "ttir.full"() <{shape = array<i32: <integer>, ...>, fill_value = <float> : f32 | <integer> : '
            "i32}> : () -> tensor<...>  // SYNTHETIC full.") in lines, "AnyAttrOf is an attribute"
    assert ('%result = "ttir.rms_norm"(%input[, %weight]) <{[epsilon = <float> : f32], operandSegmentSizes = '
            "array<i32: <count of %input>, <count of %weight>>}> : (tensor<...>[, tensor<...>]) -> tensor<...>  "
            "// RMS normalization operation") in lines, "a comment inside the arguments is no argument"
    assert ('%result = "ttir.to_layout"(%input) <{[layout = #ttcore.metal_layout<...>]}> : (tensor<...> | '
            "memref<...>) -> tensor<...>") in lines
    assert '%result = "ttir.reduce"(%input) <{dim = #ttir<reduce_dim R | C>}> : (tensor<...>) -> tensor<...>' \
        in lines, "an enum attribute with no format of its own prints as #dialect<mnemonic case>"


def test_ops_are_grouped_by_family_and_descriptions_are_left_out(tmp_path: Path) -> None:
    lines = reference(tmp_path, "ttir")
    unary = ("TTIR top level named ops, class TTIR_ElementwiseUnaryOp (Base class for elementwise operations with "
             "one input tensor.):")
    named = "TTIR top level named ops, class TTIR_NamedOp:"
    assert lines.index(unary) < lines.index(named) < lines.index("TTIR top level named ops:")
    assert lines[lines.index(unary) + 1].startswith('%result = "ttir.abs"')
    assert lines[lines.index(unary) + 2].startswith('%result = "ttir.typecast"')
    assert lines[lines.index(named) + 1].startswith('%result = "ttir.full"')
    assert "ttir.abs(%input)" not in "\n".join(lines), "a description's example is never rendered"


def test_types_attributes_and_the_ttcore_ones_the_ops_name(tmp_path: Path) -> None:
    lines = reference(tmp_path, "ttir")
    assert "!ttir.semaphore  // TTIR semaphore type." in lines
    assert "#ttir<reduce_dim R | C>" in lines
    assert ("#ttir.thread<threadType[, kernelSymbol]>  where threadType is compute, kernelSymbol is "
            'OptionalParameter<"SymbolRefAttr">  // Thread information for a generic op.') in lines
    assert "#ttir.should_hoist  // SYNTHETIC marker" in lines
    assert ("#ttcore.metal_layout<logical_shape = custom<DimensionList>(logical_shape), oob_val>  where "
            'logical_shape is ArrayRefParameter<"int64_t">, oob_val is undef | zero  // SYNTHETIC layout') in lines
    assert ('!ttcore.tile<custom<DimensionList>(shape), dataType>  where shape is ArrayRefParameter<"int64_t">, '
            "dataType is f32 | bf16  // TT tile") in lines
    assert "#ttcore.memory_space<l1>" in lines
    assert not any("chip_desc" in line for line in lines), "a ttcore def no op names and none kept is left out"


def test_d2m_custom_forms_with_type_directives_attr_dict_and_regions(tmp_path: Path) -> None:
    lines = reference(tmp_path, "d2m")
    assert ("%results = d2m.generic {grid = #ttcore.grid<...>, threads = [#d2m.thread<...>, ...]} ins (%inputs...[: "
            "tensor<...> | memref<...>...]) outs (%outputs... : tensor<...> | memref<...>...) {...}[: "
            "tensor<...>...]  // Generically dispatch work to a grid of cores (D2M).") in lines
    assert '"d2m.spatial"(%input) ({...}) : (tensor<...>) -> ()  // SYNTHETIC region op.' in lines
    assert "%result = d2m.wait %cb : !d2m.cb<...> -> tensor<...> | memref<...>  // Wait from circular buffer." \
        in lines
    assert "%result = d2m.iter_index(<dim>) : index where <dim> is an integer literal" in lines, (
        "a class's results after a code block closed at the line's start still count")
    assert ('%result = "d2m.tile_add"(%lhs, %rhs) : (!ttcore.tile<...>, !ttcore.tile<...> | <float type>) -> '
            "!ttcore.tile<...>") in lines, "a summary that is only the dialect, the name, and Op is left out"
    assert "D2M Generic Region Math Ops, class D2M_GenericRegionComputeFPUOrSFPUBinary:" in lines
    assert lines.index("Ops:") < lines.index("D2M Generic Region Control Ops, class D2M_GenericRegionOp:")
    assert "#d2m.thread<threadType>  where threadType is compute | unified" in lines


def test_a_hand_written_parser_gives_the_comment_in_its_parse_function(tmp_path: Path) -> None:
    lines = reference(tmp_path, "d2m")
    assert ("d2m.remote_load: its parser is hand-written (lib/Dialect/D2M/IR/D2MGenericRegionOps.cpp:7-8): Format: "
            "SYNTHETIC: %localBuffer %memref[...] SYNTHETIC: or %memref[...] into %cb; operands %localBuffer: "
            "[tensor<...> | memref<...>], %memref: tensor<...> | memref<...>, %indices: index...; result "
            "([tensor<...> | memref<...>])  // D2M Remote/Aliased Load Op") in lines


def test_a_hand_written_parser_with_no_comment_is_refused(tmp_path: Path) -> None:
    module = tool()
    root = tree(tmp_path, "d2m")
    (root / module.D2M.cpp[0]).write_bytes(b"// SYNTHETIC, no format comment\n")
    with pytest.raises(ValueError, match="d2m.remote_load"):
        module.build(root, pins.read_pin("tt-mlir"), module.D2M)


def test_main_builds_and_checks_each_dialect_and_the_loader_reads_the_pack(tmp_path: Path) -> None:
    module = tool()
    out = tmp_path / "assets"
    for name in ("ttir", "d2m"):
        root = tree(tmp_path, name)
        assert module.main(["build", "--tree", str(root), "--dialect", name, "--out", str(out)]) == 0
        assert module.main(["check", "--tree", str(root), "--dialect", name, "--out", str(out)]) == 0
        sources = module.DIALECTS[name]
        loaded = assets.load_tree(out / "context" / sources.ref)
        assert list(loaded) == [sources.key]
        assert assets.pack_text(sources.ref, loaded).startswith(f"// The {name} dialect of tt-mlir")
    assert module.main(["check", "--tree", str(tree(tmp_path, "ttir")), "--dialect", "d2m", "--out",
                        str(out)]) == 2, "a tree without the dialect's sources is refused"


def test_fetch_downloads_every_dialects_sources_checked_by_blob(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    module = tool()
    context = module.context_tool()
    pin = pins.read_pin("tt-mlir")
    data = {path: f"// SYNTHETIC {path}\n".encode("ascii") for d in module.DIALECTS.values() for path in d.sources}
    listing = {"tree": [{"path": path, "type": "blob", "sha": context.blob_sha1(body)} for path, body in data.items()]}

    def get(url: str) -> bytes:
        """Serve the SYNTHETIC listing and files."""
        if "/git/trees/" in url:
            return json.dumps(listing).encode("ascii")
        return data[url.split(f"/{pin['COMMIT']}/", 1)[1]]

    monkeypatch.setattr(context, "_get", get)
    dest = tmp_path / "fetched"
    assert module.fetch(dest, pin) == len(data)
    assert all((dest / path).read_bytes() == body for path, body in data.items())
    context.check_tree(dest, pin)
    data[module.TTIR.ops[0]] = b"// SYNTHETIC tampered\n"
    with pytest.raises(ValueError, match="blob sha1"):
        module.fetch(tmp_path / "again", pin)


@pytest.mark.parametrize("name", ["ttir", "d2m"])
def test_the_tracked_manifest_names_the_dialects_sources_and_a_prompt_sized_reference(name: str) -> None:
    module = tool()
    sources = module.DIALECTS[name]
    manifest = yaml.safe_load((REPO / "assets" / "context" / sources.ref / "MANIFEST.yaml").read_text(
        encoding="ascii"))
    assert manifest["generator"] == module.GENERATOR and manifest["commit"] == pins.read_pin("tt-mlir")["COMMIT"]
    assert manifest["language"] == name and [entry["key"] for entry in manifest["entries"]] == [sources.key]
    assert [row["path"] for row in manifest["entries"][0]["sources"]] == list(sources.sources)
    # The tt-metalium pack's limit, which summarize_context carried (tools/build_ttkernel_context.py LIMIT_BYTES).
    assert manifest["bytes"] <= module.context_tool().LIMIT_BYTES


@pytest.mark.parametrize(("name", "expected"), [
    ("ttir", ['%result = "ttir.add"(%lhs, %rhs) : (tensor<...>, tensor<...>) -> tensor<...>  // Elementwise addition '
              "operation.",
              "%result = ttir.empty() : tensor<...>  // Empty tensor allocation operation.",
              '%result = "ttir.matmul"(%a, %b) <{[transpose_a = true | false], [transpose_b = true | false]}> : '
              "(tensor<...>, tensor<...>) -> tensor<...>  // Matrix multiplication operation."]),
    ("d2m", ["%result = d2m.wait %cb : !d2m.cb<...> -> tensor<...> | memref<...>  // Wait from circular buffer.",
             "%result = d2m.empty() : tensor<...>  // Empty tensor allocation operation (D2M).",
             '%result = "d2m.tile_add"(%lhs, %rhs) : (!ttcore.tile<...>, !ttcore.tile<...> | <float type> | '
             "<integer type>) -> !ttcore.tile<...>"]),
])
def test_the_built_reference_matches_its_manifest_and_spells_the_pinned_forms(name: str, expected: list[str]) -> None:
    sources = tool().DIALECTS[name]
    generated = REPO / "assets" / "context" / sources.ref
    if not (generated / f"{sources.key}.txt").is_file():
        pytest.skip(f"the reference is not built here; run tools/build_ttkernel_reference.py build --dialect {name} "
                    "--tree <tt-mlir>")
    text = assets.pack_text(sources.ref, assets.load_tree(generated))
    assert text.isascii()
    lines = text.splitlines()
    for line in expected:
        assert line in lines
    digest = hashlib.sha256(text.encode("ascii")).hexdigest()
    manifest = yaml.safe_load((generated / "MANIFEST.yaml").read_text(encoding="ascii"))
    assert manifest["entries"][0]["sha256"] == digest
