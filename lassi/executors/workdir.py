"""Per-trial build directories under the runs root (bible Sandbox; Agent Rule 7).

Each attempt of a trial builds in its own fresh directory,
`<runs root>/<trial_id segments>/attempt<NN>/build`, so no attempt sees
another's files and toolchains get the fresh workdir they expect. The runs
root is $LASSI_RUNS_ROOT, which the gate sets under /mnt/nvme10; build
directories never sit inside the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

from lassi.core.record import parse_trial_id

_REPO = Path(__file__).resolve().parents[2]


def runs_root() -> Path:
    """Return $LASSI_RUNS_ROOT; raise ValueError when it is not set or not absolute."""
    value = os.environ.get("LASSI_RUNS_ROOT", "")
    if not value:
        raise ValueError("LASSI_RUNS_ROOT is not set; the gate sets it on the build host")
    if not Path(value).is_absolute():
        raise ValueError(f"LASSI_RUNS_ROOT must be an absolute path, got {value!r}")
    return Path(value)


def build_dir(root: Path, trial_id: str, attempt: int) -> Path:
    """Return the build directory of one attempt of a trial under `root`; nothing is created."""
    parse_trial_id(trial_id)
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ValueError(f"attempt must be an integer >= 0, got {attempt!r}")
    path = Path(root).joinpath(*trial_id.split("/"), f"attempt{attempt:02d}", "build")
    resolved = path.resolve()
    if resolved == _REPO or _REPO in resolved.parents:
        raise ValueError(f"build directories must live outside the repository, not under {_REPO}")
    return path


def fresh_build_dir(root: Path, trial_id: str, attempt: int) -> Path:
    """Create and return the attempt's build directory; raise FileExistsError when it already exists."""
    path = build_dir(root, trial_id, attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    return path
