"""Tests for a tracked suite: a manifest with no upstream, no fetch, and no network (task P16.8).

Bible: Benchmark Suites (manifest keys; the sources rule), Agent Rules 3
and 5. Plan: plans/p16-adversarial.md, P16.8 ("The suite runs with no
network and no upstream fetch: the task settles its repo and commit keys and
its sources root").

The contract these tests fix (lassi.bench.registry):

- load_suite takes `repo` and `commit` together or not at all. A manifest
  with neither is a tracked suite: every language of every item must be
  `tracked: true`, every language support entry must be `{tracked: ...}`
  (no upstream entry), no item may declare item-level `support` (it reads
  the fetched sources), and `pin` is refused. Each refusal is a ValueError
  naming the file and the key. One of repo and commit without the other is
  refused naming the missing key.
- Suite.repo and Suite.commit are None for a tracked suite, and the new
  property Suite.tracked (a bool) says so; a suite with repo and commit is
  not tracked, even when every language of it is (tt-pairs-v0 reads its
  kernels from the pinned sources).
- sources_dir(scratch, suite) raises ValueError for a tracked suite: it has
  no fetched sources.
- A tracked suite's items read their files from TRACKED_ROOT, whatever root
  a caller gives.
- TrainView.commit is `str | None`, and None for a tracked suite.

The manifests and tracked files are SYNTHETIC, written under tmp_path, with
lassi.bench.registry TRACKED_ROOT pointed at a temporary repository as
tests/bench/test_tier_a_registry.py does. No value here is a measurement.
"""

from __future__ import annotations

import copy
import inspect
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi import bench
from lassi.bench import registry

REPO = Path(__file__).resolve().parents[2]
SUITE = "mini-suite"
HEADER = "// SYNTHETIC lassi_io.h stand-in\n"
PROGRAM = "// SYNTHETIC tracked C++ program\n"
TRAIN_PROGRAM = "// SYNTHETIC tracked C++ program of a train item\n"
COMMIT = "c" * 40
URL = "https://example.invalid/tracked.git"


def language(item: str) -> dict[str, Any]:
    """Return the tracked cpp language entry of a SYNTHETIC item, with its tracked lassi_io.h."""
    return {
        "dir": f"assets/fixture/{item}/cpp",
        "tracked": True,
        "files": [f"{item}.cpp"],
        "support": {"lassi_io.h": {"tracked": "assets/fixture/lassi_io.h"}},
    }


def manifest_data() -> dict[str, Any]:
    """Return the SYNTHETIC tracked manifest: no repo, no commit, an unassigned item and a train item."""
    return {
        "suite": SUITE,
        "items": {
            "vadd": {
                "split": "unassigned",
                "seed": 7,
                "inputs": [{"name": "x", "dtype": "f32", "shape": [4], "dist": "uniform", "lo": 0.0, "hi": 1.0}],
                "outputs": ["y"],
                "tolerance": {"metric": "max_abs", "threshold": 0},
                "languages": {"cpp": language("vadd")},
            },
            "scale": {"split": "train", "languages": {"cpp": language("scale")}},
        },
    }


def write_manifest(directory: Path, data: dict[str, Any]) -> Path:
    """Write `data` as <directory>/<suite>.yaml with LF newlines and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{data['suite']}.yaml"
    path.write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    return path


@pytest.fixture
def tracked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the registry's TRACKED_ROOT at a temporary repository holding the SYNTHETIC tracked files."""
    root = tmp_path / "repo"
    for path, text in {
        "assets/fixture/lassi_io.h": HEADER,
        "assets/fixture/vadd/cpp/vadd.cpp": PROGRAM,
        "assets/fixture/scale/cpp/scale.cpp": TRAIN_PROGRAM,
    }.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(text.encode("ascii"))
    monkeypatch.setattr(registry, "TRACKED_ROOT", root)
    return root


def load(directory: Path, data: dict[str, Any]) -> bench.Suite:
    """Write and load a manifest."""
    return bench.load_suite(write_manifest(directory, data))


# ---------------------------------------------------------------------------
# A tracked suite loads


def test_a_manifest_with_neither_repo_nor_commit_is_a_tracked_suite(tmp_path: Path, tracked: Path) -> None:
    suite = load(tmp_path / "manifests", manifest_data())
    assert (suite.repo, suite.commit, suite.pin) == (None, None, None)
    assert suite.tracked is True
    assert sorted(suite.items) == ["scale", "vadd"]


def test_tracked_is_a_bool_property_of_suite(tmp_path: Path, tracked: Path) -> None:
    assert isinstance(inspect.getattr_static(bench.Suite, "tracked"), property)
    assert type(load(tmp_path / "manifests", manifest_data()).tracked) is bool


@pytest.mark.parametrize("manifest", ["lassi-hecbench-10", "tt-pairs-v0"])
def test_a_suite_with_repo_and_commit_is_not_tracked(manifest: str) -> None:
    suite = bench.load_suite(REPO / "assets" / "bench" / f"{manifest}.yaml")
    assert isinstance(suite.repo, str) and isinstance(suite.commit, str)
    assert suite.tracked is False


def test_a_tracked_suite_reads_its_files_from_the_repository_whatever_the_root(
    tmp_path: Path, tracked: Path
) -> None:
    suite = load(tmp_path / "manifests", manifest_data())
    elsewhere = tmp_path / "no-such-root"
    to_cpp = bench.Direction("cpp", "cpp")
    assert suite.reference_target("vadd", to_cpp, elsewhere, purpose="eval") == {"vadd.cpp": PROGRAM}
    assert suite.source_files("vadd", to_cpp, elsewhere, purpose="eval") == {"vadd.cpp": PROGRAM}
    assert suite.support_files("vadd", elsewhere, purpose="eval", language="cpp") == {"lassi_io.h": HEADER}


# ---------------------------------------------------------------------------
# What a manifest may and may not say


def edit_top(key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    """Return a change that sets the manifest's top-level `key` to `value`."""

    def apply(data: dict[str, Any]) -> None:
        data[key] = value

    return apply


def edit_vadd(path: str, value: Any) -> Callable[[dict[str, Any]], None]:
    """Return a change that sets the key `path` (levels joined by " > ") of item vadd; None deletes it."""

    def apply(data: dict[str, Any]) -> None:
        *parents, last = path.split(" > ")
        node: Any = data["items"]["vadd"]
        for part in parents:
            node = node[part]
        if value is None:
            del node[last]
        else:
            node[last] = value

    return apply


@pytest.mark.parametrize(
    ("change", "key"),
    [
        (edit_top("repo", URL), "commit"),
        (edit_top("commit", COMMIT), "repo"),
        (edit_vadd("languages > cpp > tracked", False), "tracked"),
        (edit_vadd("languages > cpp > tracked", None), "tracked"),
        (
            edit_vadd("languages > cpp > support > kernel.h", {"upstream": "src/kernel.h", "sha256": "0" * 64}),
            "support",
        ),
        (edit_vadd("support", {"common.h": "src/common.h"}), "support"),
        (edit_top("pin", "tt-metal"), "pin"),
    ],
    ids=[
        "repo-without-commit",
        "commit-without-repo",
        "language-not-tracked",
        "language-without-tracked",
        "upstream-support-entry",
        "item-level-support",
        "pin",
    ],
)
def test_a_tracked_suite_refuses_what_reads_fetched_sources(
    tmp_path: Path, tracked: Path, change: Callable[[dict[str, Any]], None], key: str
) -> None:
    load(tmp_path / "good", manifest_data())
    data = copy.deepcopy(manifest_data())
    change(data)
    path = write_manifest(tmp_path / "bad", data)
    with pytest.raises(ValueError) as refused:
        bench.load_suite(path)
    assert_names(str(refused.value), path, key)


def assert_names(message: str, path: Path, key: str) -> None:
    """Assert that a refusal names the manifest file and, apart from that path, the key."""
    assert str(path) in message, f"the refusal does not name the file: {message}"
    assert key in message.replace(str(path), ""), f"the refusal does not name {key!r}: {message}"


@pytest.mark.parametrize("missing", ["repo", "commit"])
def test_one_of_repo_and_commit_alone_is_refused_for_an_untracked_suite_too(
    tmp_path: Path, missing: str
) -> None:
    data = {
        "suite": "fetched-fixture",
        "repo": URL,
        "commit": COMMIT,
        "items": {"saxpy": {"split": "eval", "languages": {"omp": {"dir": "src/saxpy-omp", "files": ["main.cpp"]}}}},
    }
    load(tmp_path / "good", copy.deepcopy(data))
    del data[missing]
    path = write_manifest(tmp_path / "bad", data)
    with pytest.raises(ValueError) as refused:
        bench.load_suite(path)
    assert_names(str(refused.value), path, missing)


# ---------------------------------------------------------------------------
# Where a tracked suite's sources are


def test_sources_dir_refuses_a_tracked_suite(tmp_path: Path, tracked: Path) -> None:
    suite = load(tmp_path / "manifests", manifest_data())
    with pytest.raises(ValueError, match="no fetched sources") as refused:
        bench.sources_dir(tmp_path / "scratch", suite)
    assert SUITE in str(refused.value), str(refused.value)
    assert not (tmp_path / "scratch").exists(), "nothing is created"


# ---------------------------------------------------------------------------
# The train view of a tracked suite


def test_train_view_commit_may_be_none() -> None:
    hint = typing.get_type_hints(registry.TrainView)["commit"]
    assert set(typing.get_args(hint)) == {str, type(None)}, hint


def test_the_train_view_of_a_tracked_suite_has_no_commit(tmp_path: Path, tracked: Path) -> None:
    view = load(tmp_path / "manifests", manifest_data()).train_view()
    assert (view.name, view.commit) == (SUITE, None)
    assert sorted(view.train_items) == ["scale"] and dict(view.held) == {"vadd": "unassigned"}
    with pytest.raises(bench.EvalSplitError):
        view.item("vadd")
