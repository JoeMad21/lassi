"""Fixtures for the notebook replay (task P1.9).

Bible: Build Roadmap (P1 row, Gate), Source Papers (LASSI pipeline; quirk
table), Design Principle 4, Agent Rules 1, 4, and 6.

- `upstream_checkout`: the pinned upstream checkout named by
  assets/upstream/lassi.yaml. Tests that need it skip, naming
  `uv run tools/fetch_upstream.py`, when it is absent, and fail when it sits
  at another commit.
- `notebook`: the notebook's taken functions and prompt dictionary, read from
  the pin with `git show` into a temporary directory and compiled
  (notebook_replay.load_notebook).
- `assets_root`: a fresh extraction of the lassi-2024 fragments and both
  context packs into a temporary directory by tools/extract_lassi_assets.py;
  its failure names the tool.
- `replay_env`: points the asset and prompt loaders at that extraction,
  clears the host's gate variables, sets TMPDIR under the test's directory,
  and gives the fragments.

After the session, the terminal summary shows the replay report
(notebook_replay.render_report): a decision table per scenario and the
fence-quirk hit count, labeled a check of decision logic on synthetic
fixtures, not a measurement.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from notebook_replay import FETCH_HINT, PACKS, PIN_MANIFEST, REPO, RESULTS, Notebook, load_notebook, render_report

import lassi.prompts as prompts_module
import lassi.prompts.assets as prompt_assets
from lassi.prompts import load_recipe_assets

EXTRACTOR = REPO / "tools" / "extract_lassi_assets.py"
FRAGMENT_SET = "lassi-2024"


@dataclass(frozen=True)
class ReplayEnv:
    """What a replay case needs besides its scenario: the compiled notebook and the extracted fragments."""

    notebook: Notebook
    fragments: dict[str, str]


def _pin() -> tuple[str, str]:
    """Return the upstream commit and checkout path from the pin manifest."""
    data = yaml.safe_load(PIN_MANIFEST.read_bytes().decode("utf-8"))
    return str(data["commit"]), str(data["path"])


@pytest.fixture(scope="session")
def upstream_pin() -> tuple[str, Path]:
    """Return the pinned commit and the checkout's path."""
    commit, path = _pin()
    return commit, REPO / path


@pytest.fixture(scope="session")
def upstream_checkout(upstream_pin: tuple[str, Path]) -> Path:
    """Return the pinned upstream checkout; skip when it is not fetched, fail when it is off the pin."""
    commit, root = upstream_pin
    if not (root / ".git").exists():
        pytest.skip(f"{root} is not fetched; {FETCH_HINT}")
    done = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    head = done.stdout.strip()
    if head != commit:
        pytest.fail(f"{root} is at {head}, not the pin {commit}; {FETCH_HINT}")
    return root


@pytest.fixture(scope="session")
def notebook(upstream_pin: tuple[str, Path], upstream_checkout: Path, tmp_path_factory: pytest.TempPathFactory
             ) -> Notebook:
    """Return the notebook's taken functions and prompt dictionary, read from the pin into a temporary directory."""
    commit, _ = upstream_pin
    return load_notebook(upstream_checkout, commit, tmp_path_factory.mktemp("pinned-notebook"))


def _extractor() -> ModuleType:
    """Load tools/extract_lassi_assets.py as a module."""
    spec = importlib.util.spec_from_file_location("extract_lassi_assets_for_replay", EXTRACTOR)
    assert spec and spec.loader, f"cannot load {EXTRACTOR}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def assets_root(upstream_checkout: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a temporary assets root holding a fresh extraction of the pinned checkout."""
    out = tmp_path_factory.mktemp("lassi-assets")
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        status = _extractor().main(["--upstream", str(upstream_checkout), "--out", str(out)])
    assert status == 0, f"tools/extract_lassi_assets.py exited {status}: {sink.getvalue()[-2000:]}"
    return out


@pytest.fixture
def replay_env(notebook: Notebook, assets_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ReplayEnv:
    """Point the loaders at the fresh extraction and give the fragments; clear the host's gate variables."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
                 "CPATH"):
        monkeypatch.delenv(name, raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.setattr(prompt_assets, "default_root", lambda: assets_root)
    monkeypatch.setattr(prompts_module, "default_roots", lambda: (assets_root / "prompts", REPO / "assets" / "prompts"))
    assets = load_recipe_assets({"prompts": FRAGMENT_SET, "context": list(PACKS)}, root=assets_root)
    return ReplayEnv(notebook=notebook, fragments=dict(assets.fragments))


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Show the replay report after the session when any replay case ran."""
    if not RESULTS:
        return
    terminalreporter.write_sep("-", "notebook replay (synthetic fixtures, not a measurement)")
    for line in render_report(RESULTS).splitlines():
        terminalreporter.write_line(line)
