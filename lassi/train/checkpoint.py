"""The trainer-neutral checkpoint records (task P17.9; bible Training Module, Compute and Safeguards).

Every checkpoint a Trainer writes holds, beside its weights:

- recipe.resolved.yaml: the bytes of the train directory's resolved train
  recipe (TrainJob.recipe_yaml);
- checkpoint.json (RECORDS_JSON): lassi.core.record json_text (two-space
  indent, ASCII, no NaN, a final LF) of the PROVENANCE_KEYS values copied
  from the train directory's provenance.json (the device record, the
  framework pins, and the data with its split hash among them), plus the
  Trainer's own record under "trainer_records";
- steps.jsonl (STEPS_JSONL): one JSON object per optimizer step, in order,
  ASCII with LF. A non-finite number is written as null and its key listed
  in that step's "nonfinite"; nothing is bridged (Agent Rule 1).
  finite_record applies the same rule to any flat record, such as TRL's
  final log entry.

write_records refuses a checkpoint that is not strictly inside the job's
out_dir, the only place a Trainer writes (lassi.core.interfaces Trainer).
This module imports only the standard library and lassi, so any backend can
use it and the fast suite tests it without a framework.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lassi.core.interfaces import TrainJob
from lassi.core.record import json_text

RECORDS_JSON = "checkpoint.json"
STEPS_JSONL = "steps.jsonl"
# The resolved train recipe's file name, as the train directory names it (lassi.train.run RESOLVED_RECIPE).
RESOLVED_RECIPE = "recipe.resolved.yaml"
# The provenance.json keys every checkpoint.json copies, in this order.
PROVENANCE_KEYS: tuple[str, ...] = (
    "train_id", "trainer", "recipe_hash", "commit", "dirty", "device_records", "driver", "framework_pins", "data",
)
# The key of a step that lists its non-finite values, each written as null.
NONFINITE = "nonfinite"


def write_records(job: TrainJob, checkpoint: Path, trainer_records: Mapping[str, Any]) -> None:
    """Write recipe.resolved.yaml and checkpoint.json into `checkpoint`, a directory strictly inside job.out_dir.

    A checkpoint outside job.out_dir, or job.out_dir itself, raises
    ValueError before anything is written; so does a provenance without one
    of PROVENANCE_KEYS, or a record that is not JSON without NaN.
    """
    base = Path(job.out_dir).resolve()
    found = Path(checkpoint).resolve()
    if base not in found.parents:
        raise ValueError(f"the checkpoint {checkpoint} is not a directory strictly inside {job.out_dir}")
    missing = [key for key in PROVENANCE_KEYS if key not in job.provenance]
    if missing:
        raise ValueError(f"the job's provenance lacks {', '.join(missing)}")
    records = {**{key: job.provenance[key] for key in PROVENANCE_KEYS}, "trainer_records": trainer_records}
    text = json_text(records)
    (found / RESOLVED_RECIPE).write_bytes(job.recipe_yaml.encode("ascii"))
    (found / RECORDS_JSON).write_bytes(text.encode("ascii"))


def finite_record(step: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of a step's mapping with each non-finite float as None, its keys listed under NONFINITE."""
    line: dict[str, Any] = {}
    nonfinite: list[str] = []
    for key, value in step.items():
        if isinstance(value, float) and not math.isfinite(value):
            line[key] = None
            nonfinite.append(key)
        else:
            line[key] = value
    if nonfinite:
        line[NONFINITE] = nonfinite
    return line


def write_steps(checkpoint: Path, steps: Sequence[Mapping[str, Any]]) -> None:
    """Write steps.jsonl into `checkpoint`: one JSON object per step, in order, ASCII, each line ending in LF."""
    lines = [json.dumps(finite_record(step), ensure_ascii=True, allow_nan=False) + "\n" for step in steps]
    (Path(checkpoint) / STEPS_JSONL).write_bytes("".join(lines).encode("ascii"))
