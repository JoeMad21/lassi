"""Tests for tools/build_ttkernel_rag.py, the tt-mlir tests retrieval index (task P17.44).

The tool selects ttkernel MLIR from tt-mlir's lit tests at the pin by the
corpus spec assets/rag/tt-mlir-tests/corpus.yaml and builds a BM25 index and
manifest in lassi.prompts.retrieval's and lassi.prompts.rag_corpus's
formats, so the runner loads it as it loads the tt-metal documentation
index. Agent Rule 5: every item's upstream kernels (read from the suite's
fetched sources, each checked against its sha256) and tracked files are
references for the run check, and a test that names an item, its example
directory, or a kernel's file stem is excluded whole. The .mlir and kernel
text here is SYNTHETIC; no network is used.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from lassi.bench import load_suite
from lassi.prompts import rag_corpus
from lassi.prompts.retrieval import load_index
from lassi.toolchains import pins

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "assets" / "rag" / "tt-mlir-tests" / "corpus.yaml"
MANIFEST = SPEC.parent / "MANIFEST.yaml"
SUITE = load_suite(REPO / "assets" / "bench" / "tt-pairs-v0-ttkernel.yaml")
REFERENCE = "\n".join([
    "void kernel_main() {",
    "    uint32_t synthetic_src_addr = get_arg_val<uint32_t>(0);",
    "    uint32_t synthetic_tiles = get_arg_val<uint32_t>(1);",
    "    cb_reserve_back(synthetic_cb, synthetic_tiles);",
    "    noc_async_read_tile(synthetic_tile_id, synthetic_accessor, synthetic_write_ptr);",
    "}",
]) + "\n"
OPS = """// RUN: ttmlir-opt %s
func.func @synthetic_reader() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : i32
  %cb = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<2, !ttcore.tile<32x32, bf16>>
  ttkernel.noc_async_read_tile(%c0, %acc, %c0) : (i32, !ttkernel.TensorAccessor, i32) -> ()
  func.return
}
"""
# Three lines of REFERENCE, kept as a comment's text in a test, which the literal run check reads.
COPY = "// RUN: ttmlir-opt %s\n" + "".join(f"// {line}\n" for line in REFERENCE.splitlines()[1:4]) + OPS
NAMED = "// RUN: ttmlir-opt %s\n// SYNTHETIC: a ttkernel module of loopback_dram_copy\n" + OPS
FILES = {
    "test/ttmlir/Dialect/TTKernel/ops.mlir": OPS,
    "test/ttmlir/Conversion/TTKernelToEmitC/copied.mlir": COPY,
    "test/ttmlir/Translate/TTKernel/named.mlir": NAMED,
    "test/ttmlir/Dialect/TTKernel/invalid.mlir": OPS,
    "test/ttmlir/Dialect/D2M/not_selected.mlir": OPS,
}


def tool() -> ModuleType:
    """Import tools/build_ttkernel_rag.py as a module."""
    path = REPO / "tools" / "build_ttkernel_rag.py"
    spec = importlib.util.spec_from_file_location("build_ttkernel_rag", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fetched(tmp_path: Path, corpus: str = "tt-mlir-tests") -> Path:
    """Return a SYNTHETIC tree as `build_ttkernel_rag.py fetch` leaves it at the pin, holding FILES."""
    root = tmp_path / "tree"
    for path, text in FILES.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(text.encode("ascii"))
    commit = pins.read_pin("tt-mlir")["COMMIT"]
    (root / "lassi-fetch.txt").write_bytes(f"{commit}\ncorpus {corpus}: SYNTHETIC fetch\n".encode("ascii"))
    return root


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Return the tool with references() giving one SYNTHETIC upstream kernel, as the suite's sources would."""
    loaded = tool()
    monkeypatch.setattr(loaded, "references", lambda suite, bench: {"tt_metal/synthetic/kernels/reader.cpp": REFERENCE})
    return loaded


def build(module: ModuleType, tmp_path: Path) -> tuple[dict, bytes]:
    """Return (manifest document, index bytes) of the SYNTHETIC tree under the real corpus spec."""
    spec = rag_corpus.load_corpus_spec(SPEC)
    return module.build(fetched(tmp_path), SPEC, spec, pins.read_pin("tt-mlir"), tmp_path / "bench")


def test_the_spec_selects_the_ttkernel_tests_and_leaves_out_the_negative_ones() -> None:
    spec = rag_corpus.load_corpus_spec(SPEC)
    assert spec.pin == "tt-mlir" and spec.suite == "assets/bench/tt-pairs-v0-ttkernel.yaml"
    assert all(pattern.startswith("test/ttmlir/") and pattern.endswith(".mlir") for pattern in spec.include)
    assert not any("D2M" in pattern for pattern in spec.include)
    assert {"test/ttmlir/Dialect/TTKernel/invalid.mlir", "test/ttmlir/Translate/TTKernel/ttkernel_noc.mlir",
            "test/ttmlir/Translate/TTKernel/ttkernel_compute.mlir"} <= set(spec.exclude)


def test_an_items_name_marks_a_file_but_a_longer_identifier_does_not() -> None:
    module = tool()
    names = module.identities(SUITE)
    assert {"eltwise_binary", "eltwise_sfpu", "loopback", "matmul_multi_core", "matmul_single_core"} <= set(names)
    assert {"loopback_dram_copy", "reader_single_core_mm", "read_tiles", "tiles_add"} <= set(names)
    assert "mm" not in names, "a stem under MIN_NAME characters is too common a word"
    found = module.identity_exclusions({"a.mlir": b"// a module for eltwise_sfpu\n",
                                        "b.mlir": b"ttkernel.noc_async_read_tile(%a, %b, %c)\n",
                                        "loopback/c.mlir": b"func.func @k()\n"}, names)
    assert set(found) == {"a.mlir", "loopback/c.mlir"}
    assert found["a.mlir"] == "tier-a-name: names eltwise_sfpu, of the item eltwise_sfpu (Agent Rule 5)"


def test_three_reference_lines_in_filecheck_lines_mark_a_file_and_two_do_not() -> None:
    module = tool()
    lines = REFERENCE.splitlines()[1:4]
    three = "".join(f"// CHECK{'-NEXT' if index else ''}: {line.strip()}\n" for index, line in enumerate(lines))
    two = "".join(f"// CHECK: {line.strip()}\n" for line in lines[:2]) + "// CHECK: SYNTHETIC other line\n"
    refs = {"tt_metal/synthetic/kernels/reader.cpp": REFERENCE}
    found = module.comment_runs({"three.mlir": three.encode(), "two.mlir": two.encode()}, refs, 3)
    assert list(found) == ["three.mlir"]


def test_the_references_must_be_the_items_own_upstream_kernels(tmp_path: Path) -> None:
    module = tool()
    with pytest.raises(ValueError, match="is not under"):
        module.references(SUITE, tmp_path / "empty")
    for entry in (e for item in SUITE.items.values() for spec in item.languages.values()
                  for e in spec.support.values() if not e.tracked):
        (tmp_path / "stand-in" / entry.path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "stand-in" / entry.path).write_bytes(b"// SYNTHETIC stand-in\n")
    with pytest.raises(ValueError, match="sha256"):
        module.references(SUITE, tmp_path / "stand-in")


def test_build_excludes_named_listed_and_copied_text_and_indexes_the_rest(module: ModuleType, tmp_path: Path) -> None:
    document, data = build(module, tmp_path)
    excluded = {row["path"]: row["reason"] for row in document["excluded"]}
    assert excluded["test/ttmlir/Translate/TTKernel/named.mlir"].startswith(
        "listed: tier-a-name: names loopback_dram_copy, of the item loopback")
    assert excluded["test/ttmlir/Dialect/TTKernel/invalid.mlir"].startswith("listed: a negative test")
    assert excluded["test/ttmlir/Conversion/TTKernelToEmitC/copied.mlir"].startswith(
        "listed: tier-a-run in its comments: 3 line(s) of a reference run, first line 2 from "
        "tt_metal/synthetic/kernels/reader.cpp"), "code quoted in comments or CHECK lines is a copy too"
    assert all(not path.startswith("test/ttmlir/Dialect/D2M") for path in excluded)
    assert [row["path"] for row in document["files"]][-1] == "test/ttmlir/Dialect/TTKernel/ops.mlir"
    assert document["generator"] == module.GENERATOR and document["source"]["pin"] == "tt-mlir"
    assert document["tier_a"]["references"] == [
        {"path": "tt_metal/synthetic/kernels/reader.cpp", "sha256": hashlib.sha256(REFERENCE.encode()).hexdigest()}]
    text = data.decode("utf-8")
    assert "synthetic_tiles" not in text and "loopback_dram_copy" not in text, "no reference run or named file"
    assert "ttkernel.get_compile_time_arg_val(0)" in text


def test_main_builds_and_checks_an_index_the_retriever_loads(module: ModuleType, tmp_path: Path,
                                                             capsys: pytest.CaptureFixture[str]) -> None:
    index, manifest = tmp_path / "out" / "index.json", tmp_path / "out" / "MANIFEST.yaml"
    args = ["--tree", str(fetched(tmp_path)), "--bench-root", str(tmp_path / "bench"), "--index", str(index),
            "--manifest", str(manifest)]
    assert module.main(["build", *args]) == 0
    document = rag_corpus.read_manifest(manifest)
    assert manifest.read_text(encoding="ascii").startswith(module.MANIFEST_HEADER)
    assert load_index(index, sha256=document["index"]["sha256"]).search("get_compile_time_arg_val", 1)
    assert module.main(["check", *args]) == 0
    index.write_bytes(index.read_bytes() + b" ")
    assert module.main(["check", *args]) == 1
    capsys.readouterr()


def test_an_index_inside_the_repository_is_refused(module: ModuleType, tmp_path: Path,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    inside = REPO / "assets" / "rag" / "tt-mlir-tests" / "index.json"
    assert module.main(["build", "--tree", str(fetched(tmp_path)), "--bench-root", str(tmp_path / "bench"),
                        "--index", str(inside), "--manifest", str(tmp_path / "MANIFEST.yaml")]) == 2
    assert not inside.exists() and "outside the repository" in capsys.readouterr().err


def test_a_tree_for_another_corpus_or_commit_is_refused(module: ModuleType, tmp_path: Path) -> None:
    spec = rag_corpus.load_corpus_spec(SPEC)
    with pytest.raises(ValueError, match="corpus tt-mlir-tests"):
        module.build(fetched(tmp_path, corpus="tt-metal-docs"), SPEC, spec, pins.read_pin("tt-mlir"), tmp_path)
    assert module.main(["build", "--tree", str(tmp_path / "nothing"), "--bench-root", str(tmp_path),
                        "--index", str(tmp_path / "i.json"), "--manifest", str(tmp_path / "m.yaml")]) == 2


def test_the_tracked_manifest_is_this_spec_and_records_every_items_kernel_as_a_reference() -> None:
    document = rag_corpus.read_manifest(MANIFEST)
    assert document["generator"] == tool().GENERATOR and document["corpus"] == "tt-mlir-tests"
    assert document["corpus_sha256"] == hashlib.sha256(SPEC.read_bytes()).hexdigest(), "rebuild after a spec edit"
    assert document["source"] == {"pin": "tt-mlir", "repo": pins.read_pin("tt-mlir")["URL"],
                                  "commit": pins.read_pin("tt-mlir")["COMMIT"]}
    recorded = {row["path"]: row["sha256"] for row in document["tier_a"]["references"]}
    for item in SUITE.items.values():
        for spec in item.languages.values():
            for entry in spec.support.values():
                if not entry.tracked:
                    assert recorded[entry.path] == entry.sha256, entry.path
    listed = yaml.safe_load(SPEC.read_text(encoding="ascii"))["exclude"]
    assert {row["path"] for row in listed} <= {row["path"] for row in document["excluded"]}
    assert all(row["path"].startswith("test/ttmlir/") for row in document["files"])
