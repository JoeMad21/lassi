"""Tests for tools/fetch_bench.py on a tracked suite: nothing to fetch, no network, nothing written (task P16.8).

Bible: Benchmark Suites (the sources rule). Plan: plans/p16-adversarial.md,
P16.8 ("The suite runs with no network and no upstream fetch").

The contract these tests fix: `fetch_bench.py <manifest>` for a tracked
suite (a manifest with no repo and no commit, every language tracked;
lassi.bench.registry) prints

    fetch_bench: <suite> is a tracked suite: every language is in the repository, so there is nothing to fetch

and exits 0. It starts no process (subprocess.run, subprocess.Popen), opens
no socket, copies and fetches nothing, and writes nothing under
$LASSI_SCRATCH or the working directory. Both a SYNTHETIC tracked manifest
written under tmp_path and the committed assets/bench/adv-seeded-v0.yaml are
checked.

No value here is a measurement.
"""

from __future__ import annotations

import importlib.util
import socket
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SYNTHETIC_SUITE = "mini-suite"
SEEDED_SUITE = "adv-seeded-v0"


def fetch_tool() -> ModuleType:
    """Import tools/fetch_bench.py by path and return the module."""
    spec = importlib.util.spec_from_file_location("fetch_bench", REPO / "tools" / "fetch_bench.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_manifest(directory: Path) -> Path:
    """Write a SYNTHETIC tracked manifest (no repo, no commit) whose one language is the real tracked lassi_io.h."""
    data = {
        "suite": SYNTHETIC_SUITE,
        "items": {
            "vadd": {
                "split": "unassigned",
                "languages": {
                    "cpp": {
                        "dir": "assets/harness/c",
                        "tracked": True,
                        "files": ["lassi_io.h"],
                    }
                },
            }
        },
    }
    directory.mkdir(parents=True)
    path = directory / f"{SYNTHETIC_SUITE}.yaml"
    path.write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    return path


def refuse(what: str) -> Any:
    """Return a stand-in that fails the test when called: the tool may not `what`."""

    def stand_in(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"fetch_bench tried to {what} for a tracked suite")

    return stand_in


@pytest.fixture
def sealed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Refuse every process start and socket, set LASSI_SCRATCH to an empty directory, and return it."""
    monkeypatch.setattr(subprocess, "run", refuse("start a process"))
    monkeypatch.setattr(subprocess, "Popen", refuse("start a process"))
    monkeypatch.setattr(socket, "socket", refuse("open a socket"))
    monkeypatch.setattr(socket, "create_connection", refuse("open a connection"))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("LASSI_SCRATCH", str(scratch))
    monkeypatch.delenv("LASSI_TOOLCHAINS", raising=False)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return scratch


@pytest.mark.parametrize("which", ["synthetic", "adv-seeded-v0"])
def test_a_tracked_suite_has_nothing_to_fetch(
    tmp_path: Path, sealed: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], which: str
) -> None:
    tool = fetch_tool()
    monkeypatch.setattr(tool, "fetch", refuse("fetch"))
    monkeypatch.setattr(tool, "copy_from_tree", refuse("copy from an installed tree"))
    if which == "synthetic":
        manifest, suite = synthetic_manifest(tmp_path / "manifests"), SYNTHETIC_SUITE
    else:
        manifest, suite = REPO / "assets" / "bench" / f"{SEEDED_SUITE}.yaml", SEEDED_SUITE
    assert manifest.is_file(), f"no manifest at {manifest}"
    status = tool.main([str(manifest)])
    out = capsys.readouterr().out
    expected = (
        f"fetch_bench: {suite} is a tracked suite: every language is in the repository, so there is nothing to fetch"
    )
    assert status == 0, out
    assert expected in out.splitlines(), out
    assert list(sealed.iterdir()) == [], "nothing is written under $LASSI_SCRATCH"
    assert list((tmp_path / "work").iterdir()) == [], "nothing is written in the working directory"
