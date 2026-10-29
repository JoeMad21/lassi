"""Every tracked suite_sha256 equals the sha256 of the suite file it names (the window of 2026-10-27).

P17.42 (37c0cf4) changed assets/bench/tt-pairs-v0.yaml and left its old hash in
three manifests; every test passed, and on the build host the kh examples'
rebuild rewrote that line, so run-suite.sh left out every sweep9 recipe. This
test reads each manifest that records a suite hash and checks it against the
file's bytes, so such a change fails here first.

No value here is a measurement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SUITE = REPO / "assets" / "bench" / "tt-pairs-v0.yaml"
MANIFESTS = (
    REPO / "assets" / "examples" / "tt-examples-v0" / "MANIFEST.yaml",
    REPO / "assets" / "examples" / "tt-kh-examples-v0" / "MANIFEST.yaml",
    REPO / "assets" / "train" / "tt-train-v0" / "manifest.json",
)


def found_values(data: object) -> list[str]:
    """Return every suite_sha256 value at any depth of a manifest's data."""
    if isinstance(data, dict):
        own = [data["suite_sha256"]] if "suite_sha256" in data else []
        return own + [value for child in data.values() for value in found_values(child)]
    if isinstance(data, list):
        return [value for child in data for value in found_values(child)]
    return []


def recorded(path: Path) -> list[str]:
    """Return the suite_sha256 values a manifest records."""
    text = path.read_text(encoding="utf-8")
    return found_values(json.loads(text) if path.suffix == ".json" else yaml.safe_load(text))


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda path: path.parent.name)
def test_the_recorded_suite_hash_is_the_suite_files(path: Path) -> None:
    assert recorded(path) == [hashlib.sha256(SUITE.read_bytes()).hexdigest()], (
        f"{path} records a stale suite_sha256; a change to {SUITE.name} must update it in the same commit")


def test_no_other_tracked_manifest_records_a_suite_hash() -> None:
    found = {path for pattern in ("assets/**/MANIFEST.yaml", "assets/**/manifest.json")
             for path in REPO.glob(pattern) if "suite_sha256" in path.read_text(encoding="utf-8")}
    assert found == set(MANIFESTS)
