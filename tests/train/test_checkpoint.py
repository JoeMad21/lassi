"""Tests for the trainer-neutral checkpoint records (task P17.9; lassi.train.checkpoint).

Bible: Training Module (Compute: every checkpoint records the device, the
framework pins, and the resolved train recipe; Safeguards: checkpoints log
data split hashes), Agent Rules 1 and 10; plans/p17-portable.md, task P17.9
("Each run writes a checkpoint directory holding per-step losses and reward
components, the resolved train recipe, the device record, and the pins");
PHASE-NOTES P17, the copy-helper candidate.

The contract these tests fix:

- RECORDS_JSON is "checkpoint.json", STEPS_JSONL "steps.jsonl", and
  PROVENANCE_KEYS the keys copied from the job's provenance: train_id,
  trainer, recipe_hash, commit, dirty, device_records, driver,
  framework_pins, and data (whose split_hash is the data split hash).
- write_records(job, checkpoint, trainer_records) writes, into a checkpoint
  directory strictly inside job.out_dir, recipe.resolved.yaml with the bytes
  of job.recipe_yaml and checkpoint.json as lassi.core.record json_text of
  {each PROVENANCE_KEYS value, "trainer_records": trainer_records}: two-space
  indent, ASCII, no NaN, a final LF. A checkpoint outside job.out_dir, or
  job.out_dir itself, is a ValueError and nothing is written.
- write_steps(checkpoint, steps) writes steps.jsonl: one JSON object per
  step, in order, ASCII, each line ending in LF. A non-finite number (NaN,
  an infinity) is written as null and its key listed in that step's
  "nonfinite"; nothing is bridged (Agent Rule 1).

The module imports no framework, so these tests run without the extra. The
job is SYNTHETIC; no value in this module is a measurement.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from trl_smoke import train_module

from lassi.core.interfaces import TrainData, TrainJob
from lassi.core.record import DeviceRecord, json_text

PROVENANCE_KEYS = (
    "train_id", "trainer", "recipe_hash", "commit", "dirty", "device_records", "driver", "framework_pins", "data",
)
RECIPE_YAML = "# Resolved recipe train (chain: train)\n# recipe_hash: SYNTHETIC\nbase_model: fixture:tiny-causal-lm\n"
SPLIT_HASH = "0" * 63 + "1"
DEVICE = {
    "key": "trainer.device", "kind": "cpu", "indices": [], "name": "SYNTHETIC CPU model 9000", "count": 8,
    "memory_bytes": 4096, "driver": None, "runtime": None, "framework": "torch",
    "framework_version": "2.99.0+synthetic-cpu",
}
# A SYNTHETIC provenance.json as the layer first writes it; the keys not in PROVENANCE_KEYS are never copied.
PROVENANCE: dict[str, Any] = {
    "train_id": "t1", "status": "running", "recipe": "train", "recipe_path": "/SYNTHETIC/train.yaml",
    "recipe_chain": ["train"], "recipe_hash": "a" * 64, "commit": "0123456789abcdef0123456789abcdef01234567",
    "dirty": False, "host": "SYNTHETIC-host", "platform": "SYNTHETIC-platform", "python": "3.10.99",
    "trainer": "trl", "device_records": [DEVICE], "driver": None,
    "framework_pins": {"build": {"name": "torch", "version": "2.99.0+synthetic-cpu", "cuda": None, "hip": None},
                       "packages": {"trl": "9.9.9", "peft": None}},
    "data": {"source": "synthetic", "name": "sft-smoke.jsonl", "record_count": 4, "split_hash": SPLIT_HASH},
    "started_utc": "2026-10-06T00:00:00+00:00", "finished_utc": None, "steps": None, "checkpoints": None,
}
TRAINER_RECORDS: dict[str, Any] = {
    "method": "sft", "weights": "full", "global_step": 2,
    "base_model": {"name": "fixture:tiny-causal-lm", "files": {"config.json": "b" * 64}, "seed": 7},
}


def checkpoint_module() -> Any:
    """Import lassi.train.checkpoint."""
    return train_module("checkpoint")


def job(tmp_path: Path) -> TrainJob:
    """Return a SYNTHETIC TrainJob whose out_dir is <tmp>/train/output, created empty."""
    out_dir = tmp_path / "train" / "output"
    out_dir.mkdir(parents=True)
    data = TrainData(
        source="synthetic", split_hash=SPLIT_HASH,
        records=({"prompt": "SYNTHETIC prompt", "completion": "SYNTHETIC completion"},), suite=None, items=(),
    )
    return TrainJob(
        recipe={"base_model": "fixture:tiny-causal-lm"}, recipe_yaml=RECIPE_YAML, data=data,
        device=DeviceRecord(**DEVICE), provenance=PROVENANCE, out_dir=out_dir,
    )


def checkpoint_dir(train_job: TrainJob, name: str = "checkpoint-2") -> Path:
    """Create and return a checkpoint directory under the job's out_dir."""
    path = Path(train_job.out_dir) / name
    path.mkdir(parents=True)
    return path


def test_the_record_names() -> None:
    module = checkpoint_module()
    assert (module.RECORDS_JSON, module.STEPS_JSONL) == ("checkpoint.json", "steps.jsonl")
    assert tuple(module.PROVENANCE_KEYS) == PROVENANCE_KEYS


def test_write_records_copies_the_recipe_bytes_and_the_provenance_keys(tmp_path: Path) -> None:
    train_job = job(tmp_path)
    checkpoint = checkpoint_dir(train_job)
    checkpoint_module().write_records(train_job, checkpoint, TRAINER_RECORDS)
    assert (checkpoint / "recipe.resolved.yaml").read_bytes() == RECIPE_YAML.encode("ascii")
    expected = {**{key: PROVENANCE[key] for key in PROVENANCE_KEYS}, "trainer_records": TRAINER_RECORDS}
    raw = (checkpoint / "checkpoint.json").read_bytes()
    assert raw == json_text(expected).encode("ascii"), "checkpoint.json is json_text of the copied keys"
    assert raw.isascii() and b"\r" not in raw and raw.endswith(b"\n")
    written = json.loads(raw)
    assert written["data"]["split_hash"] == SPLIT_HASH
    assert "host" not in written and "status" not in written
    assert sorted(item.name for item in checkpoint.iterdir()) == ["checkpoint.json", "recipe.resolved.yaml"]


def test_write_records_accepts_a_nested_checkpoint(tmp_path: Path) -> None:
    train_job = job(tmp_path)
    checkpoint = checkpoint_dir(train_job, "steps/checkpoint-2")
    checkpoint_module().write_records(train_job, checkpoint, TRAINER_RECORDS)
    assert (checkpoint / "checkpoint.json").is_file()


@pytest.mark.parametrize("where", ["sibling", "parent", "out-dir-itself", "escape"])
def test_write_records_refuses_a_checkpoint_outside_the_out_dir(where: str, tmp_path: Path) -> None:
    train_job = job(tmp_path)
    out_dir = Path(train_job.out_dir)
    target = {
        "sibling": out_dir.parent / "checkpoint-2",
        "parent": out_dir.parent,
        "out-dir-itself": out_dir,
        "escape": out_dir / ".." / "checkpoint-2",
    }[where]
    target.mkdir(parents=True, exist_ok=True)
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    with pytest.raises(ValueError):
        checkpoint_module().write_records(train_job, target, TRAINER_RECORDS)
    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert after == before, "a refused checkpoint is left as it was"


def read_steps(checkpoint: Path) -> list[dict[str, Any]]:
    """Return steps.jsonl's objects, refusing NaN and Infinity, after checking ASCII and LF."""
    raw = (checkpoint / "steps.jsonl").read_bytes()
    assert raw.isascii() and b"\r" not in raw and raw.endswith(b"\n")

    def no_constant(name: str) -> Any:
        raise AssertionError(f"steps.jsonl holds {name}")

    return [json.loads(line, parse_constant=no_constant) for line in raw.decode("ascii").split("\n")[:-1]]


def test_write_steps_writes_one_object_per_step_in_order(tmp_path: Path) -> None:
    checkpoint = checkpoint_dir(job(tmp_path))
    steps = [
        {"step": 1, "loss": 5.5, "grad_norm": 0.25, "learning_rate": 0.001},
        {"step": 2, "loss": 5.25, "grad_norm": 0.125, "learning_rate": 0.0005, "reward_record": {"mean": None}},
    ]
    checkpoint_module().write_steps(checkpoint, steps)
    found = read_steps(checkpoint)
    assert [{key: value for key, value in line.items() if key != "nonfinite"} for line in found] == steps
    assert all(line.get("nonfinite", []) == [] for line in found)


def test_write_steps_writes_a_non_finite_value_as_null_and_names_it(tmp_path: Path) -> None:
    checkpoint = checkpoint_dir(job(tmp_path))
    steps = [
        {"step": 1, "loss": math.nan, "grad_norm": math.inf, "learning_rate": 0.001},
        {"step": 2, "loss": 1.5, "grad_norm": -math.inf, "learning_rate": 0.0005},
    ]
    checkpoint_module().write_steps(checkpoint, steps)
    first, second = read_steps(checkpoint)
    assert (first["loss"], first["grad_norm"], first["learning_rate"]) == (None, None, 0.001)
    assert sorted(first["nonfinite"]) == ["grad_norm", "loss"]
    assert (second["loss"], second["grad_norm"]) == (1.5, None)
    assert second["nonfinite"] == ["grad_norm"]
