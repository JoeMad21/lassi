"""Tests for tools/fetch_bench.py on a Tier A style manifest (task P4.13; bible Benchmark Suites).

The contract these tests fix (tools/fetch_bench.py's module docstring):

- Tracked languages (files in this repository) are never fetched; each
  language's upstream support files are, and their sha256 is checked with
  the language files' digests.
- A suite that names its source's pin (`pin: tt-metal`) is copied from the
  installed pinned tree, $LASSI_TOOLCHAINS/<PREFIX_NAME>, when that tree's
  lassi-install.txt names the pin's NAME and COMMIT on its first line; the
  copy writes lassi-fetch.txt, whose first line is the commit, and needs no
  network. Otherwise the sparse shallow git fetch runs, as for any suite.
- Either way the commit and every sha256 are verified; a file whose digest
  differs is named on stderr and the status is 1. A destination that
  already holds the commit with every file present and matching is kept.

The installed tree, its install record, and the kernel text are SYNTHETIC
stand-ins in a temporary directory; nothing is fetched. No value here is a
measurement.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from lassi import bench
from lassi.bench import registry
from lassi.toolchains import pins

REPO = Path(__file__).resolve().parents[2]
PIN = pins.read_pin("tt-metal")
KERNEL = "// SYNTHETIC kernel at the pin\n"
KERNEL_UPSTREAM = "tt_metal/programming_examples/vadd/kernels/compute/add.cpp"
KERNEL_NAME = "vadd/kernels/compute/add.cpp"


def digest(text: str) -> str:
    """Return the sha256 hex digest of `text` as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_tool() -> ModuleType:
    """Import tools/fetch_bench.py by path and return the module."""
    spec = importlib.util.spec_from_file_location("fetch_bench", REPO / "tools" / "fetch_bench.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(tmp_path: Path, *, pin: bool = True) -> Path:
    """Write a SYNTHETIC tracked-language manifest with one upstream kernel; pinned to tt-metal when `pin`."""
    header = {"tracked": "assets/harness/c/lassi_io.h"}
    data: dict[str, Any] = {
        "suite": "tier-a-fetch",
        "repo": PIN["URL"],
        "commit": PIN["COMMIT"],
        "items": {
            "vadd": {
                "split": "unassigned",
                "seed": 1,
                "inputs": [{"name": "a", "dtype": "f32", "shape": [4], "dist": "uniform", "lo": 0.0, "hi": 1.0}],
                "outputs": ["c"],
                "tolerance": {"metric": "max_abs", "threshold": 0},
                "unpack_to_dest": "SYNTHETIC note",
                "languages": {
                    "tt": {
                        "dir": "assets/bench/tier-a-fetch/vadd/tt",
                        "tracked": True,
                        "files": ["vadd.cpp"],
                        "support": {
                            "lassi_io.h": dict(header),
                            KERNEL_NAME: {"upstream": KERNEL_UPSTREAM, "sha256": digest(KERNEL)},
                        },
                    },
                    "cpp": {
                        "dir": "assets/bench/tier-a-fetch/vadd/cpp",
                        "tracked": True,
                        "files": ["vadd.cpp"],
                        "support": {"lassi_io.h": dict(header)},
                    },
                },
            }
        },
    }
    if pin:
        data["pin"] = "tt-metal"
    path = tmp_path / "manifests" / "tier-a-fetch.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    return path


def installed_tree(toolchains: Path, *, commit: str = PIN["COMMIT"], kernel: str = KERNEL) -> Path:
    """Write a SYNTHETIC installed tree with its install record and the kernel; return the tree."""
    tree = toolchains / PIN["PREFIX_NAME"]
    (tree / KERNEL_UPSTREAM).parent.mkdir(parents=True, exist_ok=True)
    (tree / KERNEL_UPSTREAM).write_bytes(kernel.encode("ascii"))
    (tree / "lassi-install.txt").write_bytes(f"{PIN['NAME']} {commit} {PIN['URL']}\n".encode("ascii"))
    return tree


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Set LASSI_SCRATCH and LASSI_TOOLCHAINS to temporary directories; return them."""
    scratch, toolchains = tmp_path / "scratch", tmp_path / "toolchains"
    scratch.mkdir()
    toolchains.mkdir()
    monkeypatch.setenv("LASSI_SCRATCH", str(scratch))
    monkeypatch.setenv("LASSI_TOOLCHAINS", str(toolchains))
    return scratch, toolchains


def no_git(tool: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Replace the git fetch with a recorder that fetches nothing; return the list of destinations it was asked for."""
    calls: list[Path] = []
    monkeypatch.setattr(tool, "fetch", lambda suite, dest: calls.append(Path(dest)))
    return calls


def test_tracked_languages_are_never_fetched_and_language_support_is(tmp_path: Path) -> None:
    tool = fetch_tool()
    suite = bench.load_suite(manifest(tmp_path))
    assert tool.listed_files(suite) == [KERNEL_UPSTREAM]
    assert tool.sparse_dirs(suite) == [KERNEL_UPSTREAM]
    assert tool.missing_files(suite, tmp_path / "empty") == [KERNEL_UPSTREAM]
    dest = tmp_path / "dest"
    (dest / KERNEL_UPSTREAM).parent.mkdir(parents=True)
    (dest / KERNEL_UPSTREAM).write_bytes(b"// edited\n")
    assert tool.mismatched_files(suite, dest) == [KERNEL_UPSTREAM]
    (dest / KERNEL_UPSTREAM).write_bytes(KERNEL.encode("ascii"))
    assert tool.mismatched_files(suite, dest) == []


def test_a_pinned_suite_is_copied_from_the_installed_tree(
    tmp_path: Path, env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    scratch, toolchains = env
    tool = fetch_tool()
    calls = no_git(tool, monkeypatch)
    path = manifest(tmp_path)
    tree = installed_tree(toolchains)
    assert tool.main([str(path)]) == 0, capsys.readouterr().err
    assert calls == [], "no git fetch when the installed tree holds the commit"
    dest = registry.sources_dir(scratch, bench.load_suite(path))
    assert (dest / KERNEL_UPSTREAM).read_bytes() == KERNEL.encode("ascii")
    record = (dest / "lassi-fetch.txt").read_bytes().decode("ascii").splitlines()
    assert record[0] == PIN["COMMIT"] and str(tree) in record[1]
    assert "copied" in capsys.readouterr().out
    # Idempotent: a second call keeps the destination and copies nothing.
    monkeypatch.setattr(tool, "copy_from_tree", lambda *args: pytest.fail("copied again"))
    assert tool.main([str(path)]) == 0
    assert "already holds" in capsys.readouterr().out


def test_a_copy_whose_sha256_differs_is_refused(
    tmp_path: Path, env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, toolchains = env
    tool = fetch_tool()
    no_git(tool, monkeypatch)
    installed_tree(toolchains, kernel="// SYNTHETIC kernel edited after the pin\n")
    assert tool.main([str(manifest(tmp_path))]) == 1
    assert KERNEL_UPSTREAM in capsys.readouterr().err


@pytest.mark.parametrize("case", ["other-commit", "no-tree", "no-toolchains-root", "no-pin"])
def test_without_a_matching_installed_tree_the_git_fetch_runs(
    tmp_path: Path, env: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    _, toolchains = env
    tool = fetch_tool()
    calls = no_git(tool, monkeypatch)
    if case == "other-commit":
        installed_tree(toolchains, commit="c" * 40)
    elif case == "no-toolchains-root":
        installed_tree(toolchains)
        monkeypatch.delenv("LASSI_TOOLCHAINS")
    elif case == "no-pin":
        installed_tree(toolchains)
    status = tool.main([str(manifest(tmp_path, pin=case != "no-pin"))])
    assert len(calls) == 1, "the sparse shallow git fetch ran"
    assert status == 1, "the stand-in fetch fetched nothing, so the files are missing"
