"""Tests for the bench registry and the lassi-hecbench-10 manifest (P0.8).

The manifest pins HeCBench by commit and lists items with their split and
their source files per language (bible Benchmark Suites). The registry loads
it, returns an item's reference target for a direction, and refuses eval
items to training (Agent Rule 5). Sources are never read from git: tests use
files written under tmp_path. No value here is a measurement.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lassi import bench
from lassi.core import record

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "assets" / "bench" / "lassi-hecbench-10.yaml"


def write(path: Path, text: str) -> Path:
    """Write `text` to `path` with LF newlines, creating directories, and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


def small_manifest(tmp_path: Path, split: str = "eval") -> Path:
    """Write a one-item manifest and return its path."""
    return write(
        tmp_path / "demo-suite.yaml",
        "suite: demo-suite\n"
        "repo: https://example.invalid/demo.git\n"
        f"commit: {'a' * 40}\n"
        "items:\n"
        "  saxpy:\n"
        f"    split: {split}\n"
        "    languages:\n"
        "      omp: {dir: src/saxpy-omp, files: [main.cpp]}\n"
        "      cuda: {dir: src/saxpy-cuda, files: [main.cu, kernel.h]}\n",
    )


def test_manifest_pins_hecbench_and_lists_an_eval_pair() -> None:
    suite = bench.load_suite(MANIFEST)
    assert suite.name == "lassi-hecbench-10"
    assert suite.repo == "https://github.com/zjin-lcf/HeCBench"
    assert re.fullmatch(r"[0-9a-f]{40}", suite.commit)
    assert suite.items, "at least one app"
    for item in suite.items.values():
        assert item.split == "eval", "lassi-hecbench-10 is eval only (bible Benchmark Suites)"
        assert {"omp", "cuda"} <= set(item.languages)
        for spec in item.languages.values():
            assert spec.dir.startswith("src/") and spec.files


def test_layout_is_listed_with_its_hecbench_paths() -> None:
    layout = bench.load_suite(MANIFEST).items["layout"]
    assert layout.languages["omp"].dir == "src/layout-omp"
    assert layout.languages["omp"].files == ("main.cpp",)
    assert layout.languages["cuda"].dir == "src/layout-cuda"
    assert layout.languages["cuda"].files == ("main.cu",)


def test_reference_target_per_direction(tmp_path: Path) -> None:
    suite = bench.load_suite(small_manifest(tmp_path))
    root = tmp_path / "sources"
    write(root / "src/saxpy-cuda/main.cu", "// cuda main\n")
    write(root / "src/saxpy-cuda/kernel.h", "// header\n")
    write(root / "src/saxpy-omp/main.cpp", "// omp main\n")
    to_cuda = bench.Direction(source="omp", target="cuda")
    to_omp = bench.Direction(source="cuda", target="omp")
    assert to_cuda.name == "omp-cuda"
    assert suite.reference_target("saxpy", to_cuda, root, purpose="eval") == {
        "main.cu": "// cuda main\n",
        "kernel.h": "// header\n",
    }
    assert suite.reference_target("saxpy", to_omp, root, purpose="eval") == {"main.cpp": "// omp main\n"}
    assert suite.source_files("saxpy", to_cuda, root, purpose="eval") == {"main.cpp": "// omp main\n"}


def test_eval_items_are_refused_to_training(tmp_path: Path) -> None:
    suite = bench.load_suite(small_manifest(tmp_path))
    with pytest.raises(bench.EvalSplitError, match="saxpy"):
        suite.item("saxpy", purpose="train")
    with pytest.raises(bench.EvalSplitError):
        suite.reference_target("saxpy", bench.Direction("omp", "cuda"), tmp_path, purpose="train")
    assert suite.item("saxpy", purpose="eval").name == "saxpy"


def test_train_items_serve_both_purposes(tmp_path: Path) -> None:
    suite = bench.load_suite(small_manifest(tmp_path, split="train"))
    assert suite.item("saxpy", purpose="train").split == "train"
    assert suite.item("saxpy", purpose="eval").split == "train"


def test_bad_purpose_unknown_item_and_unknown_language_fail(tmp_path: Path) -> None:
    suite = bench.load_suite(small_manifest(tmp_path))
    with pytest.raises(ValueError, match="purpose"):
        suite.item("saxpy", purpose="tune")
    with pytest.raises(ValueError, match="nosuch"):
        suite.item("nosuch", purpose="eval")
    with pytest.raises(ValueError, match="hip"):
        suite.reference_target("saxpy", bench.Direction("omp", "hip"), tmp_path, purpose="eval")


def test_bench_item_record_matches_the_result_record(tmp_path: Path) -> None:
    suite = bench.load_suite(small_manifest(tmp_path))
    item = suite.bench_item("saxpy", bench.Direction("omp", "cuda"))
    assert item == record.BenchItem(suite="demo-suite", item="saxpy", split="eval", direction="omp-cuda")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("suite: demo-suite", "suite: other"), "suite"),
        ((f"commit: {'a' * 40}", "commit: main"), "commit"),
        (("split: eval", "split: holdout"), "split"),
        (("dir: src/saxpy-omp", "dir: ../saxpy-omp"), "dir"),
        (("files: [main.cpp]", "files: [/etc/passwd]"), "files"),
        (("repo: https://example.invalid/demo.git\n", "repo: https://example.invalid/demo.git\nextra: 1\n"), "extra"),
    ],
)
def test_manifest_validation(tmp_path: Path, change: tuple[str, str], message: str) -> None:
    path = small_manifest(tmp_path)
    old, new = change
    path.write_bytes(path.read_bytes().decode("utf-8").replace(old, new).encode("utf-8"))
    with pytest.raises(ValueError, match=message):
        bench.load_suite(path)


def test_sources_live_under_scratch_never_in_the_repo(tmp_path: Path) -> None:
    suite = bench.load_suite(MANIFEST)
    dest = bench.sources_dir(tmp_path / "scratch", suite)
    assert dest == tmp_path / "scratch" / "bench" / f"lassi-hecbench-10@{suite.commit}"
    with pytest.raises(ValueError, match="repository"):
        bench.sources_dir(REPO / "tmp-scratch", suite)


def test_fetch_script_needs_the_scratch_root_and_lists_only_item_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("fetch_bench", REPO / "tools" / "fetch_bench.py")
    assert spec and spec.loader
    fetch_bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch_bench)
    monkeypatch.delenv("LASSI_SCRATCH", raising=False)
    assert fetch_bench.main([str(MANIFEST)]) == 2
    suite = bench.load_suite(MANIFEST)
    assert fetch_bench.sparse_dirs(suite) == ["src/layout-cuda", "src/layout-omp"]
    assert fetch_bench.missing_files(suite, tmp_path) == ["src/layout-omp/main.cpp", "src/layout-cuda/main.cu"]
