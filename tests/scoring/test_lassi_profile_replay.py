"""Tests for the lassi score profile on notebook replay records (task P2.7).

Bible: Evaluation Protocol (LASSI reproduction row; LASSI Paper Metrics),
Source Papers (LASSI quirk table, Sim-T and fence rows), Component
Interfaces (ScoreProfile), Build Roadmap (P1 row, the replay gate).

The P2.7 acceptance criteria ask that Sim-T and Sim-L compare the reference
target read in text mode with the last attempt's target file, as the P1.9
replay does, tested on a replay scenario's record. Here a replay case (a
scenario of tests/fixtures/replay/scenarios.json in one direction) runs both
sides of the P1.9 harness (tests/replay/notebook_replay.py): the pinned
notebook's pipeline, whose own Sim-T and Sim-L locals are the oracle, and our
faithful stages through run_recipe, which write the trial record. The lassi
profile, built as `factory(bench_root=<the case's bench root>)`, scores that
record, and its values must equal the notebook's exactly:

- sim_t and sim_l equal the notebook's token (Python tokenize) and line
  similarities; sim_t_c is sim_t_c of the text-mode reference against the
  record's last target file; each carries the interpreter version in its
  note.
- self_corr equals the notebook's final correction count, and fence_quirk
  its fence-quirk hits.
- correct is 0.0 where no clean run stands: the stale output of a failed run
  (stale-output-past-gate) and no output at all (crash-past-gate-no-run).

The scenarios cover a first-try success, a fence-quirk hit, stale output
past the execution gate, the crash past the gate with no earlier run, and a
reference with CRLF line ends (spaces-in-source), in both directions. On
these fixtures the extracted block starts one line below the reference, so
no positioned token matches and Sim-T comes out the same for a text-mode
and a binary read; test_lassi_profile.py holds the CRLF check that tells
the two apart. Tests skip, naming `uv run tools/fetch_upstream.py`, when the
pinned upstream checkout is not fetched.

Every reply, program, and run outcome here is a SYNTHETIC replay fixture; no
value in this module is a measurement, and no upstream text is copied into
it (OQ-018). Upstream code runs only inside the replay harness's guard.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

import lassi.prompts as prompts_module
import lassi.prompts.assets as prompt_assets
from lassi.core.registry import DEFAULT_REGISTRY, RegistryError
from lassi.core.store import TextStore, read_trial
from lassi.prompts import load_recipe_assets
from lassi.scoring.similarity import sim_t_c

REPO = Path(__file__).resolve().parents[2]
REPLAY_DIR = REPO / "tests" / "replay"
EXTRACTOR = REPO / "tools" / "extract_lassi_assets.py"
PROFILE_NAME = "lassi"
SCENARIOS = (
    "first-try-success", "fence-cuda", "stale-output-past-gate", "crash-past-gate-no-run", "spaces-in-source",
)
# Scenarios where no clean run stands as the output, so correct is 0.0 whatever the oracle would say.
NOT_CORRECT = ("stale-output-past-gate", "crash-past-gate-no-run")


def _replay_module() -> ModuleType:
    """Import the P1.9 replay harness from tests/replay (it is a test helper module, not a package)."""
    if str(REPLAY_DIR) not in sys.path:
        sys.path.insert(0, str(REPLAY_DIR))
    return importlib.import_module("notebook_replay")


replay = _replay_module()
CASES = [case for case in replay.load_cases() if case.scenario in SCENARIOS]


def profile_class() -> type:
    """Return the class registered as ScoreProfile `lassi`; fail the test clearly while none is registered."""
    importlib.import_module("lassi.scoring")
    try:
        return DEFAULT_REGISTRY.get("ScoreProfile", PROFILE_NAME).factory
    except RegistryError as error:
        pytest.fail(f"task P2.7 registers the ScoreProfile {PROFILE_NAME!r} when lassi.scoring is imported: {error}")


# ---------------------------------------------------------------------------
# The replay's environment, as tests/replay/conftest.py builds it


@dataclass(frozen=True)
class ReplaySetup:
    """The compiled notebook, the fresh asset extraction, and its fragments."""

    notebook: Any
    assets_root: Path
    fragments: dict[str, str]


def _extract(checkout: Path, out: Path) -> None:
    """Extract the lassi-2024 fragments and both context packs from `checkout` into `out` with the P1 tool."""
    spec = importlib.util.spec_from_file_location("extract_lassi_assets_for_scoring", EXTRACTOR)
    assert spec and spec.loader, f"cannot load {EXTRACTOR}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        status = module.main(["--upstream", str(checkout), "--out", str(out)])
    assert status == 0, f"tools/extract_lassi_assets.py exited {status}: {sink.getvalue()[-2000:]}"


@pytest.fixture(scope="module")
def replay_setup(tmp_path_factory: pytest.TempPathFactory) -> ReplaySetup:
    """Return the replay's notebook and a fresh asset extraction; skip when the pinned checkout is absent."""
    pin = yaml.safe_load(replay.PIN_MANIFEST.read_bytes().decode("utf-8"))
    commit, checkout = str(pin["commit"]), REPO / str(pin["path"])
    if not (checkout / ".git").exists():
        pytest.skip(f"{checkout} is not fetched; {replay.FETCH_HINT}")
    done = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True)
    if done.stdout.strip() != commit:
        pytest.fail(f"{checkout} is at {done.stdout.strip()}, not the pin {commit}; {replay.FETCH_HINT}")
    notebook = replay.load_notebook(checkout, commit, tmp_path_factory.mktemp("pinned-notebook"))
    assets_root = tmp_path_factory.mktemp("lassi-assets")
    _extract(checkout, assets_root)
    assets = load_recipe_assets({"prompts": replay.FRAGMENT_SET, "context": list(replay.PACKS)}, root=assets_root)
    return ReplaySetup(notebook=notebook, assets_root=assets_root, fragments=dict(assets.fragments))


@pytest.fixture
def replay_env(replay_setup: ReplaySetup, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ReplaySetup:
    """Point the loaders at the fresh extraction and clear the host's gate variables for one case."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
                 "CPATH"):
        monkeypatch.delenv(name, raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    root = replay_setup.assets_root
    monkeypatch.setattr(prompt_assets, "default_root", lambda: root)
    monkeypatch.setattr(prompts_module, "default_roots", lambda: (root / "prompts", REPO / "assets" / "prompts"))
    return replay_setup


def our_trial(root: Path) -> Any:
    """Return the one trial record our side wrote under `root` (run_ours's runs root), read with its text store."""
    found = sorted((root / "runs-root").rglob("trial.json"))
    assert len(found) == 1, f"expected one trial under {root}, found {len(found)}"
    run_dir = next(parent for parent in found[0].parents if (parent / "provenance.json").is_file())
    return read_trial(found[0].parent, TextStore(run_dir))


# ---------------------------------------------------------------------------
# The test


def test_the_replay_scenarios_are_present() -> None:
    present = {case.scenario for case in CASES}
    assert present == set(SCENARIOS), f"replay scenarios missing: {sorted(set(SCENARIOS) - present)}"
    assert len(CASES) == 2 * len(SCENARIOS), "each scenario runs in both directions"


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_the_profile_scores_a_replay_record_as_the_notebook_does(case: Any, replay_env: ReplaySetup,
                                                                 tmp_path: Path) -> None:
    notebook = replay.run_notebook(case, replay_env.notebook, replay_env.fragments, tmp_path / "notebook")
    replay.run_ours(case, tmp_path / "ours")
    trial = our_trial(tmp_path / "ours")
    score = profile_class()(bench_root=tmp_path / "ours" / "bench").score(trial)

    assert score.components["sim_t"] == notebook.sim_t, "Sim-T equals the notebook's token similarity"
    assert score.components["sim_l"] == notebook.sim_l, "Sim-L equals the notebook's line similarity"
    last_file = trial.attempts[-1].files.get(replay.target_file(case), "")
    assert score.components["sim_t_c"] == sim_t_c(replay.text_mode(case.reference), last_file)
    version = f"python {platform.python_version()}"
    for name in ("sim_t", "sim_t_c", "sim_l"):
        assert version in score.notes.get(name, "").lower(), f"{name} carries the interpreter version ({version})"

    assert score.components["self_corr"] == float(notebook.corrections)
    assert score.components["fence_quirk"] == float(notebook.fence_quirk)
    if case.scenario in NOT_CORRECT:
        assert score.components["correct"] == 0.0 and score.scalar == 0.0
