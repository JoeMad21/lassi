"""Tests for the bench registry and the lassi-hecbench-10 manifest (P0.8, P1.2).

The manifest pins HeCBench by commit and lists items with their split and
their source files per language (bible Benchmark Suites). The registry loads
it, returns an item's reference target for a direction, and refuses eval
items to training (Agent Rule 5). Sources are never read from git: tests use
files written under tmp_path. No value here is a measurement.

P1.2 adds, per language, the sha256 of each model-facing file
(`sha256: {<file>: <hex>}`, LanguageSources.sha256), and per item its run
arguments (`run_args`, SuiteItem.run_args), the languages that print
PASS/FAIL (`passfail`, SuiteItem.passfail), and its support files
(`support: {<build-dir name>: <path in the sources>}`, SuiteItem.support),
which Suite.support_files reads from the pinned sources. tools/fetch_bench.py
fetches the support files with the item files and refuses a fetch whose
files differ from their sha256. The ten-app content of the real manifest is
checked in tests/bench/test_hecbench10.py.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path
from types import ModuleType

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


def fetch_tool() -> ModuleType:
    """Import tools/fetch_bench.py by path and return the module."""
    spec = importlib.util.spec_from_file_location("fetch_bench", REPO / "tools" / "fetch_bench.py")
    assert spec and spec.loader
    fetch_bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch_bench)
    return fetch_bench


def test_fetch_script_needs_the_scratch_root_and_lists_only_item_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch_bench = fetch_tool()
    monkeypatch.delenv("LASSI_SCRATCH", raising=False)
    assert fetch_bench.main([str(MANIFEST)]) == 2
    suite = bench.load_suite(MANIFEST)
    dirs = sorted({spec.dir for item in suite.items.values() for spec in item.languages.values()})
    assert fetch_bench.sparse_dirs(suite) == dirs
    assert "src/layout-omp/main.cpp" in fetch_bench.missing_files(suite, tmp_path)


# ---------------------------------------------------------------------------
# P1.2: sha256, run arguments, PASS/FAIL languages, and support files

OMP_TEXT = "// omp main\n"
CUDA_TEXT = "// cuda main\n"
HELPER_TEXT = "// helper header\n"


def digest(text: str) -> str:
    """Return the sha256 hex digest of `text` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def full_manifest(tmp_path: Path, split: str = "eval") -> Path:
    """Write a one-item manifest with sha256, run_args, passfail, and support, and return its path."""
    return write(
        tmp_path / "demo-suite.yaml",
        "suite: demo-suite\n"
        "repo: https://example.invalid/demo.git\n"
        f"commit: {'a' * 40}\n"
        "items:\n"
        "  saxpy:\n"
        f"    split: {split}\n"
        '    run_args: ["100", ""]\n'
        "    passfail: [cuda]\n"
        "    support: {helper.h: src/common/helper.h}\n"
        "    languages:\n"
        f"      omp: {{dir: src/saxpy-omp, files: [main.cpp], sha256: {{main.cpp: {digest(OMP_TEXT)}}}}}\n"
        f"      cuda: {{dir: src/saxpy-cuda, files: [main.cu], sha256: {{main.cu: {digest(CUDA_TEXT)}}}}}\n",
    )


def write_sources(root: Path, cuda_text: str = CUDA_TEXT) -> Path:
    """Write the full manifest's sources under `root` (the CUDA main as `cuda_text`) and return `root`."""
    write(root / "src/saxpy-omp/main.cpp", OMP_TEXT)
    write(root / "src/saxpy-cuda/main.cu", cuda_text)
    write(root / "src/common/helper.h", HELPER_TEXT)
    return root


def test_manifest_records_sha256_run_args_passfail_and_support(tmp_path: Path) -> None:
    item = bench.load_suite(full_manifest(tmp_path)).items["saxpy"]
    assert dict(item.languages["omp"].sha256) == {"main.cpp": digest(OMP_TEXT)}
    assert dict(item.languages["cuda"].sha256) == {"main.cu": digest(CUDA_TEXT)}
    assert tuple(item.run_args) == ("100", "")
    assert frozenset(item.passfail) == {"cuda"}
    assert dict(item.support) == {"helper.h": "src/common/helper.h"}


def test_support_files_are_read_from_the_pinned_sources_and_refused_to_training(tmp_path: Path) -> None:
    root = write_sources(tmp_path / "sources")
    suite = bench.load_suite(full_manifest(tmp_path))
    assert suite.support_files("saxpy", root, purpose="eval") == {"helper.h": HELPER_TEXT}
    with pytest.raises(bench.EvalSplitError):
        suite.support_files("saxpy", root, purpose="train")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("{main.cpp: " + digest(OMP_TEXT), "{main.cpp: " + "g" * 64), "sha256"),
        (("{main.cpp: " + digest(OMP_TEXT), "{other.cpp: " + digest(OMP_TEXT)), "sha256"),
        (('run_args: ["100", ""]', "run_args: 100"), "run_args"),
        (("passfail: [cuda]", "passfail: [hip]"), "hip"),
        (("support: {helper.h: src/common/helper.h}", "support: {helper.h: ../helper.h}"), "support"),
        (("passfail: [cuda]", "passfail: [[cuda]]"), "passfail"),
        (("support: {helper.h: src/common/helper.h}", "support: {main.cu: src/common/helper.h}"), "main.cu"),
    ],
)
def test_new_manifest_fields_are_validated(tmp_path: Path, change: tuple[str, str], message: str) -> None:
    path = full_manifest(tmp_path)
    old, new = change
    text = path.read_bytes().decode("utf-8")
    assert old in text
    path.write_bytes(text.replace(old, new).encode("utf-8"))
    with pytest.raises(ValueError, match=message):
        bench.load_suite(path)


def test_fetch_lists_support_files_as_files_to_fetch(tmp_path: Path) -> None:
    suite = bench.load_suite(full_manifest(tmp_path))
    fetch_bench = fetch_tool()
    missing = fetch_bench.missing_files(suite, tmp_path / "empty")
    assert sorted(missing) == ["src/common/helper.h", "src/saxpy-cuda/main.cu", "src/saxpy-omp/main.cpp"]
    sparse = fetch_bench.sparse_dirs(suite)
    assert any(entry == "src/common/helper.h" or "src/common/helper.h".startswith(f"{entry}/") for entry in sparse)


def test_fetch_refuses_files_whose_sha256_differs_from_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fetch_bench = fetch_tool()
    manifest = full_manifest(tmp_path)
    suite = bench.load_suite(manifest)
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("LASSI_SCRATCH", str(scratch))
    dest = bench.sources_dir(scratch, suite)
    served = {"cuda": "// edited upstream of the pin\n"}
    fetched: list[Path] = []

    def fake_fetch(suite_arg: bench.Suite, dest_arg: Path, *rest: object) -> None:
        """Stand in for the sparse shallow checkout: write the sources, with the CUDA main as served."""
        fetched.append(Path(dest_arg))
        write_sources(Path(dest_arg), served["cuda"])

    monkeypatch.setattr(fetch_bench, "fetch", fake_fetch)
    monkeypatch.setattr(fetch_bench, "_head", lambda path: suite.commit)
    # A kept checkout at the pinned commit whose CUDA main differs is not accepted, and neither is a fetch of it.
    write_sources(dest, served["cuda"])
    assert fetch_bench.main([str(manifest)]) == 1
    assert "src/saxpy-cuda/main.cu" in capsys.readouterr().err
    served["cuda"] = CUDA_TEXT
    assert fetch_bench.main([str(manifest)]) == 0
    assert fetched, "the checkout was fetched again"
    assert (dest / "src/saxpy-cuda/main.cu").read_bytes() == CUDA_TEXT.encode("utf-8")
