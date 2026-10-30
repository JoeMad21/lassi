"""Tests for tools/build_ttkernel_rag.py over the TTIR and D2M corpora (task P17.47).

The tool, generalized over a dialect (CORPORA), builds tt-mlir-ttir-tests
and tt-mlir-d2m-tests from tt-mlir's own lit tests at the pin as it builds
tt-mlir-tests (task P17.44, whose tests stay in test_build_ttkernel_rag.py),
with the same Agent Rule 5 references and exclusions: every item's upstream
kernels and the suite's tracked files are references, a test that names an
item, its example directory, or a kernel's file stem goes whole, and so does
one quoting a reference run in its comments. A negative test and a test lit
does not run go whole too, and a run of closing braces is no copy. The .mlir
and kernel text here is SYNTHETIC; no network is used. The tracked manifests
are checked against their specs and for every item's kernel as a reference.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from lassi.bench import load_suite
from lassi.prompts import rag_corpus
from lassi.prompts.retrieval import load_index
from lassi.toolchains import pins

REPO = Path(__file__).resolve().parents[2]
SUITE = load_suite(REPO / "assets" / "bench" / "tt-pairs-v0-ttkernel.yaml")
REFERENCE = "\n".join([
    "void kernel_main() {",
    "    uint32_t synthetic_src_addr = get_arg_val<uint32_t>(0);",
    "    for (uint32_t i = 0; i < synthetic_n; i++) {",
    "        for (uint32_t j = 0; j < synthetic_m; j++) {",
    "            cb_reserve_back(synthetic_cb, synthetic_tiles);",
    "        }",
    "    }",
    "}",
]) + "\n"
TTIR = """// RUN: ttmlir-opt %s
func.func @synthetic_add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}
"""
# Three closing braces in a row, as nested loops end, in the code and in CHECK lines: no copy of the reference.
NESTED = """// RUN: ttmlir-opt %s | FileCheck %s
func.func @synthetic_nest(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: scf.for
  // CHECK: }
  // CHECK: }
  // CHECK: }
  scf.for %i = %c0 to %c2 step %c1 {
    scf.for %j = %c0 to %c3 step %c1 {
      %1 = "ttir.exp"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    }
  }
  return %arg0 : tensor<64x128xf32>
}
"""
COPY = "// RUN: ttmlir-opt %s\n" + "".join(f"// CHECK: {line.strip()}\n" for line in REFERENCE.splitlines()[1:4]) + TTIR
NAMED = "// RUN: ttmlir-opt %s\n// SYNTHETIC: a TTIR module of loopback_dram_copy\n" + TTIR
NEGATIVE = TTIR + "// expected-error @+1 {{SYNTHETIC: invalid}}\n"
NOT_RUN = "// UNSUPPORTED: true\n" + TTIR
FILES = {
    "test/ttmlir/Dialect/TTIR/ops.mlir": TTIR,
    "test/ttmlir/Dialect/TTIR/nested/nest.mlir": NESTED,
    "test/ttmlir/Dialect/TTIR/negative.mlir": NEGATIVE,
    "test/ttmlir/Dialect/TTIR/not_run.mlir": NOT_RUN,
    "test/ttmlir/Conversion/TTIRToD2M/copied.mlir": COPY,
    "test/ttmlir/Conversion/TTIRToD2M/named.mlir": NAMED,
    "test/ttmlir/Conversion/StableHLOToTTIR/not_selected.mlir": TTIR,
    "test/ttmlir/Dialect/D2M/not_selected.mlir": TTIR,
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


def fetched(tmp_path: Path, corpus: str) -> Path:
    """Return a SYNTHETIC tree as `build_ttkernel_rag.py fetch` leaves it at the pin for `corpus`, holding FILES."""
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


@pytest.mark.parametrize(("dialect", "corpus", "roots"), [
    ("ttir", "tt-mlir-ttir-tests", ["test/ttmlir/Dialect/TTIR/", "test/ttmlir/Conversion/TTIRToD2M/"]),
    ("d2m", "tt-mlir-d2m-tests", ["test/ttmlir/Dialect/D2M/", "test/ttmlir/Conversion/D2MToTTKernel/",
                                  "test/ttmlir/Conversion/D2MToTTMetal/"]),
])
def test_each_dialects_spec_selects_its_own_tests_with_the_ttkernel_corpus_rules(dialect: str, corpus: str,
                                                                                  roots: list[str]) -> None:
    module = tool()
    assert module.CORPORA[dialect] == corpus and module.spec_of(dialect) == REPO / "assets" / "rag" / corpus / (
        "corpus.yaml")
    spec = rag_corpus.load_corpus_spec(module.spec_of(dialect))
    ttkernel = rag_corpus.load_corpus_spec(module.SPEC)
    assert spec.corpus == corpus and spec.pin == "tt-mlir" and spec.doxygen is None
    assert [pattern.removesuffix("**/*.mlir") for pattern in spec.include] == roots
    assert (spec.suite, spec.examples, spec.run_lines, spec.max_file_overlap) == (
        ttkernel.suite, ttkernel.examples, ttkernel.run_lines, ttkernel.max_file_overlap), "Rule 5 as P17.44's"
    assert (spec.max_lines, spec.max_chars, spec.min_piece_lines, spec.k1, spec.b) == (
        ttkernel.max_lines, ttkernel.max_chars, ttkernel.min_piece_lines, ttkernel.k1, ttkernel.b)
    assert spec.exclude == {}


def test_negative_tests_and_tests_lit_does_not_run_are_named() -> None:
    found = tool().negative_exclusions({
        "a.mlir": TTIR.encode(), "b.mlir": NEGATIVE.encode(), "c.mlir": NOT_RUN.encode(),
        "d.mlir": b"// RUN: not ttmlir-opt %s\n", "e.mlir": b"// RUN: ttmlir-opt --verify-diagnostics %s\n",
        "f.mlir": b"// XFAIL: *\n", "pad_negative.mlir": b"// a negative pad, valid IR\n",
    })
    assert set(found) == {"b.mlir", "c.mlir", "d.mlir", "e.mlir", "f.mlir"}, "a file's name alone marks nothing"
    assert found["b.mlir"].startswith("negative-test: ") and found["c.mlir"].startswith("not-run: an UNSUPPORTED")


def test_a_run_of_closing_braces_is_no_copy_but_three_reference_lines_are() -> None:
    module = tool()
    refs = {"tt_metal/synthetic/kernels/reader.cpp": REFERENCE}
    assert module.comment_runs({"nest.mlir": NESTED.encode()}, refs, 3) == {}
    assert list(module.comment_runs({"copied.mlir": COPY.encode()}, refs, 3)) == ["copied.mlir"]


@pytest.mark.parametrize("dialect", ["ttir"])
def test_build_excludes_named_copied_negative_and_unrun_tests_and_indexes_the_rest(module: ModuleType, dialect: str,
                                                                                   tmp_path: Path) -> None:
    spec_path = module.spec_of(dialect)
    spec = rag_corpus.load_corpus_spec(spec_path)
    document, data = module.build(fetched(tmp_path, spec.corpus), spec_path, spec, pins.read_pin("tt-mlir"),
                                  tmp_path / "bench")
    excluded = {row["path"]: row["reason"] for row in document["excluded"]}
    assert excluded["test/ttmlir/Conversion/TTIRToD2M/named.mlir"].startswith(
        "listed: tier-a-name: names loopback_dram_copy, of the item loopback")
    assert excluded["test/ttmlir/Conversion/TTIRToD2M/copied.mlir"].startswith("listed: tier-a-run in its comments")
    assert excluded["test/ttmlir/Dialect/TTIR/negative.mlir"].startswith("listed: negative-test: ")
    assert excluded["test/ttmlir/Dialect/TTIR/not_run.mlir"].startswith("listed: not-run: ")
    indexed = [row["path"] for row in document["files"]]
    assert indexed == ["test/ttmlir/Dialect/TTIR/nested/nest.mlir", "test/ttmlir/Dialect/TTIR/ops.mlir"]
    assert not any("not_selected" in path for path in [*indexed, *excluded])
    text = "\n".join(chunk["text"] for chunk in json.loads(data)["chunks"])
    assert "synthetic_tiles" not in text and "loopback_dram_copy" not in text and "expected-error" not in text
    assert '"ttir.add"(%arg0, %arg1)' in text and '"ttir.exp"(%arg0)' in text


def test_main_with_a_dialect_builds_and_checks_an_index_the_retriever_loads(module: ModuleType, tmp_path: Path,
                                                                            capsys: pytest.CaptureFixture[str]) -> None:
    index, manifest = tmp_path / "out" / "index.json", tmp_path / "out" / "MANIFEST.yaml"
    args = ["--dialect", "ttir", "--tree", str(fetched(tmp_path, "tt-mlir-ttir-tests")), "--bench-root",
            str(tmp_path / "bench"), "--index", str(index), "--manifest", str(manifest)]
    assert module.main(["build", *args]) == 0
    document = rag_corpus.read_manifest(manifest)
    assert document["corpus"] == "tt-mlir-ttir-tests" and document["generator"] == module.GENERATOR
    assert load_index(index, sha256=document["index"]["sha256"]).search("ttir.add", 1)
    assert module.main(["check", *args]) == 0
    assert module.main(["check", *args[:1], "d2m", *args[2:]]) == 2, "a fetch for another corpus is refused"
    capsys.readouterr()


@pytest.mark.parametrize("corpus", ["tt-mlir-ttir-tests", "tt-mlir-d2m-tests"])
def test_the_tracked_manifest_is_its_spec_and_keeps_every_items_answer_out(corpus: str) -> None:
    module = tool()
    spec_path = REPO / "assets" / "rag" / corpus / "corpus.yaml"
    spec = rag_corpus.load_corpus_spec(spec_path)
    document = rag_corpus.read_manifest(spec_path.parent / "MANIFEST.yaml")
    assert document["generator"] == module.GENERATOR and document["corpus"] == corpus
    assert document["corpus_sha256"] == hashlib.sha256(spec_path.read_bytes()).hexdigest(), "rebuild after a spec edit"
    assert document["source"] == {"pin": "tt-mlir", "repo": pins.read_pin("tt-mlir")["URL"],
                                  "commit": pins.read_pin("tt-mlir")["COMMIT"]}
    recorded = {row["path"]: row["sha256"] for row in document["tier_a"]["references"]}
    for item in SUITE.items.values():
        for language in item.languages.values():
            for entry in language.support.values():
                if not entry.tracked:
                    assert recorded[entry.path] == entry.sha256, entry.path
    roots = [pattern.removesuffix("**/*.mlir") for pattern in spec.include]
    names = module.identities(SUITE)
    for row in document["files"]:
        assert any(row["path"].startswith(root) for root in roots), row["path"]
        assert not any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", row["path"])
                       for name in names), row["path"]
    assert document["files"] and document["index"]["files"] == len(document["files"])
    reasons = {row["reason"].removeprefix("listed: ").split(":", 1)[0] for row in document["excluded"]}
    assert reasons <= {"negative-test", "not-run", "tier-a-name", "tier-a-run in its comments"}
