"""Tests for the checkpoint pins lassi train records (2026-10-23; Agent Rule 10; lassi.train.run, lassi.core.digest).

The contract: provenance.json as first written holds checkpoint_sha256
null, and after a complete run it maps each reported checkpoint to its
content digest, lassi.core.digest tree_sha256 of the checkpoint directory,
the pin a run recipe gives hf_local as model.checkpoint_sha256. A failed
run leaves it null. The fake trainer is SYNTHETIC; no value here is a
measurement.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from train_fakes import Setup, provenance, recipe_data

from lassi.core.digest import tree_sha256


def test_a_complete_run_records_each_checkpoints_digest(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    train_dir = setup.run()
    record = provenance(train_dir)
    assert record["checkpoints"] == ["checkpoint-1"]
    assert record["checkpoint_sha256"] == {"checkpoint-1": tree_sha256(train_dir / "output" / "checkpoint-1")}
    first = setup.log.seen[0]["provenance"]
    assert "checkpoint_sha256" in first and first["checkpoint_sha256"] is None


def test_a_failed_run_records_no_digest(tmp_path: Path) -> None:
    setup = Setup(tmp_path)
    data = recipe_data(trainer={"kind": "fake", "device": {"kind": "cpu"}, "fail": True})
    with pytest.raises(RuntimeError):
        setup.run(data)
    record = provenance(setup.train_dir())
    assert record["status"] == "failed" and record["checkpoint_sha256"] is None
