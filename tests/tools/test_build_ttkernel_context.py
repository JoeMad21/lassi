"""Tests for tools/build_ttkernel_context.py, the ttkernel dialect pack and worked example (task P17.36).

The tool reads tt-mlir at the pin (a fetched tree whose lassi-fetch.txt
names the commit), condenses TTKernelOps.td into one block per op (CORE_OPS
first, the rest in file order, each kept whole until --limit-bytes), adds
the types and enums, and copies tt-mlir's two TTKernel translate lit tests
as one worked example. The trees it writes load with the project's own
loaders (lassi.prompts.assets load_tree, lassi.prompts.examples load_pack),
and only their manifests are tracked. The .td and .mlir text here is
SYNTHETIC; no network is used.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from lassi.prompts import assets, examples
from lassi.toolchains import pins

REPO = Path(__file__).resolve().parents[2]
OPS = '''// SYNTHETIC license line
def TTKernel_ZebraOp : TTKernel_Op<"zebra"> {
    let summary = "Zebra op.";
    let description = [{
      SYNTHETIC description that the pack leaves out.
    }];
    let arguments = (ins I32:$a,
                         I32:$b);
    let assemblyFormat = [{
      `(` $a `,` $b `)` attr-dict `:` functional-type(operands, results)
    }];
}

def TTKernel_GetArgValOp : TTKernel_Op<"get_arg_val", [Pure]> {
    let summary = "Get runtime arg value.";
    let arguments = (ins IndexLike:$arg_index);
    let results = (outs TTKernel_ArgResultType:$arg_val);
}
'''


def tool() -> ModuleType:
    """Import tools/build_ttkernel_context.py as a module."""
    path = REPO / "tools" / "build_ttkernel_context.py"
    spec = importlib.util.spec_from_file_location("build_ttkernel_context", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tree(tmp_path: Path, module: ModuleType, ops: str = OPS, cpp: str = "// SYNTHETIC ops source\n",
         types: str = "// SYNTHETIC license\ndef TTKernel_T : X;\n") -> Path:
    """Return a SYNTHETIC fetched tt-mlir tree at the pin: `ops`, `cpp`, and `types` as the dialect's sources."""
    root = tmp_path / "tt-mlir"
    files = {module.OPS_SOURCE: ops, **{path: types for path in module.TYPE_SOURCES}, module.CPP_SOURCE: cpp}
    files.update({path: f"// SYNTHETIC lit test {path}\nfunc.func @k() -> () {{\n}}\n"
                  for path in module.EXAMPLE_FILES})
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(text.encode("ascii"))  # line feeds as fetch writes them, on every host
    (root / module.FETCH_RECORD).write_text(f"tt-mlir {pins.read_pin('tt-mlir')['COMMIT']} x\n", encoding="ascii")
    return root


def test_op_blocks_keep_the_signature_and_leave_out_the_description() -> None:
    blocks = dict(tool().op_blocks(OPS))
    assert list(blocks) == ["zebra", "get_arg_val"]
    zebra = blocks["zebra"]
    assert zebra.startswith("ttkernel.zebra  (TTKernel_ZebraOp)")
    assert "arguments: (ins I32:$a, I32:$b)" in zebra
    assert "assemblyFormat: `(` $a `,` $b `)` attr-dict `:` functional-type(operands, results)" in zebra
    assert "SYNTHETIC description" not in zebra
    assert "results: (outs TTKernel_ArgResultType:$arg_val)" in blocks["get_arg_val"]


def test_build_writes_trees_the_loaders_accept_with_core_ops_first(tmp_path: Path) -> None:
    module = tool()
    out = tmp_path / "assets"
    module.write(module.build(tree(tmp_path, module), pins.read_pin("tt-mlir"), module.LIMIT_BYTES), out)
    texts = assets.load_tree(out / "context" / module.PACK)
    assert list(texts) == ["ttkernel_ops.ttkernel", "ttkernel_types.ttkernel"]
    ops = texts["ttkernel_ops.ttkernel"]
    assert ops.index("ttkernel.get_arg_val") < ops.index("ttkernel.zebra"), "CORE_OPS first"
    loaded = examples.load_pack(out / "examples" / module.EXAMPLES)
    assert [example.name for example in loaded] == [module.EXAMPLE_NAME]
    assert [path for path, _ in loaded[0].files] == ["ttkernel_noc.mlir", "ttkernel_compute.mlir"]
    assert module.compare(module.build(tree(tmp_path / "again", module), pins.read_pin("tt-mlir"),
                                       module.LIMIT_BYTES), out) == [], "the same tree gives the same bytes"


def test_a_limit_cuts_whole_ops_and_names_them(tmp_path: Path) -> None:
    module = tool()
    root, pin = tree(tmp_path, module), pins.read_pin("tt-mlir")
    seen = []
    for limit in range(400, 2400, 20):
        built = module.build(root, pin, limit)
        manifest = built[f"context/{module.PACK}"]["MANIFEST.yaml"].decode("ascii")
        ops = built[f"context/{module.PACK}"]["ttkernel_ops.ttkernel.txt"].decode("ascii")
        types = built[f"context/{module.PACK}"]["ttkernel_types.ttkernel.txt"]
        seen.append(("ttkernel.get_arg_val" in ops, "ttkernel.zebra" in ops))
        if any(seen[-1]):
            assert len(ops.encode("ascii")) + len(types) <= limit, "kept ops stay within the limit"
        if "ttkernel.zebra" not in ops:
            assert "- zebra" in manifest, "a cut op is named"
    assert (True, False) in seen, "a limit keeps the core op whole and cuts the next"
    assert (False, True) not in seen, "ops are kept in fill order"


def test_a_tree_at_another_commit_is_refused(tmp_path: Path) -> None:
    module = tool()
    root = tree(tmp_path, module)
    (root / module.FETCH_RECORD).write_text("tt-mlir 0000000 x\n", encoding="ascii")
    with pytest.raises(ValueError, match="not tt-mlir"):
        module.build(root, pins.read_pin("tt-mlir"), module.LIMIT_BYTES)


def test_the_tracked_manifests_name_the_pin_and_this_tool() -> None:
    module = tool()
    for path in (REPO / "assets" / "context" / module.PACK / "MANIFEST.yaml",
                 REPO / "assets" / "examples" / module.EXAMPLES / "MANIFEST.yaml"):
        text = path.read_text(encoding="ascii")
        assert f"commit: {pins.read_pin('tt-mlir')['COMMIT']}" in text and f"generator: {module.GENERATOR}" in text
