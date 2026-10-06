"""Fixtures shared by the `lassi train` tests (task P17.8).

Every test starts with none of the gate's variables (LASSI_RUNS_ROOT,
LASSI_SCRATCH, LASSI_TOOLCHAINS) and no graphics variable, with the graphics
preset directory under the test's own directory, and with git answering a
SYNTHETIC commit and a clean tree through lassi.core.runner's git helper,
which the train layer shares with the runner.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from train_fakes import FAKE_COMMIT

from lassi.core import runner as runner_module


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate and graphics variables, put the preset under <tmp>/config, and fake git."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "LASSI_GRAPHICS", "LASSI_CONFIG_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LASSI_CONFIG_DIR", str(tmp_path / "config"))
    answers = {"rev-parse": FAKE_COMMIT + "\n", "status": ""}
    monkeypatch.setattr(runner_module, "_git", lambda *args: answers[args[0]])
