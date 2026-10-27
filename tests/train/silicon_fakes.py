"""The SYNTHETIC Wormhole host of the `lassi train --silicon` tests (task P17.38); not a test module.

Rig puts one test's silicon run together: the tt_silicon tests' SYNTHETIC
host tree (tests/executors/tt_silicon_fakes.py write_host: eight n300 cards
with serials, firmware, PCI addresses, the 1 GiB pool, and the hugetlbfs
mount), their fake flock, owners, and clock, $LASSI_SCRATCH set to the
test's directory (so the lock and the quarantine marker go under
<test dir>/locks and the runs root lies inside the scratch root), the real
tenstorrent probe on that tree beside the train layer's SYNTHETIC cpu probe,
and the lassi.train.silicon SiliconHost holding those seams. No card, node,
or process is touched, and no value here is a measurement.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from train_fakes import REPO, Setup, fake_probes, recipe_data

TT_DEVICE = {"kind": "tenstorrent", "indices": [0]}
LOCK = "tt-silicon.lock"
MARKER = "tt-silicon.quarantine"


def tt_fakes() -> ModuleType:
    """Load tests/executors/tt_silicon_fakes.py once, under a name of its own."""
    name = "tt_silicon_fakes_for_train"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "tests" / "executors" / "tt_silicon_fakes.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def silicon() -> ModuleType:
    """Import lassi.train.silicon, failing the test clearly while it does not exist (task P17.38)."""
    try:
        return importlib.import_module("lassi.train.silicon")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.train.silicon does not exist yet (task P17.38): {error}")


class Rig:
    """One test's SYNTHETIC host, its fakes, and the layer's options for a silicon run."""

    def __init__(self, setup: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
        """Write the host tree under the test directory and point $LASSI_SCRATCH at the test directory."""
        fakes = tt_fakes()
        self.setup = setup
        self.fakes = fakes
        self.host = fakes.write_host(setup.root / "host")
        self.flock = fakes.FakeFlock()
        self.clock = fakes.Clock()
        self.owners = fakes.Owners(self.host.root)
        self.locks = setup.root / "locks"
        self.seen: list[dict[str, Any]] = []
        monkeypatch.setenv("LASSI_SCRATCH", str(setup.root))

    def host_seams(self) -> Any:
        """Return the SiliconHost of this rig: its root, access check, flock, uid, owners, and clock."""
        return silicon().SiliconHost(
            root=self.host.root, access=lambda path: True, flock=self.flock, uid=self.fakes.OWN_UID,
            owner=self.owners, clock=self.clock, sleep=self.clock.sleep,
        )

    def probes(self) -> dict[str, Any]:
        """Return the real tenstorrent probe on the host tree, beside the SYNTHETIC cpu, rocm, and cuda probes."""
        devices = importlib.import_module("lassi.executors.devices")
        probe = devices.TenstorrentProbe(root=self.host.root, access=lambda path: True)
        return {**fake_probes(self.setup.log), "tenstorrent": probe}

    def run(self, registry: Any, *, flag: bool = True, **changes: Any) -> Path:
        """Run a recipe of the fake Trainer on card 0, with top-level `changes`, through run_training."""
        trainer = {"kind": "fake", "device": TT_DEVICE, "steps": 2, "network": "none"}
        data = recipe_data(**{"trainer": trainer, **changes})
        return self.setup.run(data, registry=registry, probes=self.probes(), silicon=flag,
                              silicon_host=self.host_seams())

    def watching(self) -> Callable[[Any], None]:
        """Return a hook that records in self.seen, inside train(), whether the lock is held and the marker's text."""

        def look(job: Any) -> None:
            marker = self.locks / MARKER
            self.seen.append({
                "lock_held": self.flock.held(self.locks / LOCK),
                "marker": marker.read_text(encoding="ascii") if marker.exists() else None,
            })

        return look
