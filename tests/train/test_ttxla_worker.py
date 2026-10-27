"""Tests for the ttxla worker's framework-free parts (task P17.38): the job file, the batch order, and padding.

Bible: Training Module (Weight Modes, Compute, Safeguards), Agent Rules 1,
6, and 10; plans/spikes/tt-training.md, Finding 3, step 2 (the worker).

The worker (lassi/train/ttxla_worker.py) runs as a script under the tt-xla
venv's own interpreter (`python -I`, so it imports nothing from lassi and
nothing from the working directory). At module level it imports the
standard library alone; torch, torch_xla, transformers, and peft only inside
functions. These tests fix:

- read_job(path) returns the job the trainer wrote, refusing (Refusal, the
  worker's status 3) a file that is not the job format lassi-ttxla-job-1
  or lacks a key, naming it, and a backend other than xla or cpu (cpu is
  for the CPU-loop test alone; the trainer always writes xla).
- batch_order(count, steps, batch_size, seed) gives `steps` batches of
  `batch_size` record indices: each pass over the records a permutation
  from the seed, in order, the same for the same arguments.
- padded(ids, labels, length, pad_id) pads on the right to `length`: the
  attention mask is 1 over the ids and 0 over the padding, and the labels
  -100 over the padding; a sequence longer than `length` is refused,
  never cut.
- main(argv) with a job file that does not parse exits 3 and writes no
  result anywhere but the job's own result file.

Nothing here imports a framework or touches a device. No value here is a
measurement.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from train_fakes import REPO

WORKER = REPO / "lassi" / "train" / "ttxla_worker.py"


def worker() -> ModuleType:
    """Import lassi.train.ttxla_worker, failing the test clearly while it does not exist (task P17.38)."""
    try:
        return importlib.import_module("lassi.train.ttxla_worker")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.train.ttxla_worker does not exist yet (task P17.38): {error}")


def job(tmp_path: Path, **changes: object) -> dict[str, object]:
    """Return a minimal valid job under tmp_path, with `changes`."""
    values: dict[str, object] = {
        "format": "lassi-ttxla-job-1", "backend": "xla",
        "base": {"kind": "fixture", "path": str(tmp_path / "base"), "tokenizer": None},
        "weights": "full", "lora_r": None, "virtual_tokens": None, "steps": 2, "seed": 7, "batch_size": 1,
        "learning_rate": 0.001, "max_length": 32, "gradient_checkpointing": False, "chat_format": False,
        "source": "data.synthetic", "dtype": "bfloat16", "network": "none",
        "compile_options": {"tt_enable_torch_fx_fusion_pass": False, "tt_legacy_compile": True},
        "records": [{"prompt": "SYNTHETIC", "completion": " a"}],
        "checkpoint": str(tmp_path / "checkpoint-2"), "steps_file": str(tmp_path / "steps.jsonl"),
        "result_file": str(tmp_path / "result.json"), "stage_file": str(tmp_path / "stage"),
    }
    values.update(changes)
    return values


def write_job(tmp_path: Path, values: dict[str, object]) -> Path:
    """Write the job as tmp_path/job.json; return its path."""
    path = tmp_path / "job.json"
    path.write_bytes(json.dumps(values).encode("ascii"))
    return path


def test_read_job_returns_the_job(tmp_path: Path) -> None:
    values = job(tmp_path)
    assert worker().read_job(write_job(tmp_path, values)) == values


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        pytest.param({"format": "other"}, "format", id="format"),
        pytest.param({"backend": "cuda"}, "backend", id="backend"),
        pytest.param({"steps": None}, "steps", id="steps"),
        pytest.param({"records": []}, "records", id="no-records"),
        pytest.param({"network": "host"}, "network", id="network"),
    ],
)
def test_read_job_refuses_a_bad_job_naming_the_key(tmp_path: Path, changes: dict[str, object], named: str) -> None:
    values = job(tmp_path, **changes)
    if changes.get("steps", 0) is None:
        values.pop("steps")
    with pytest.raises(worker().Refusal) as refused:
        worker().read_job(write_job(tmp_path, values))
    assert named in str(refused.value)


def test_an_xla_run_refuses_any_network_interface_but_the_loopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = worker()
    monkeypatch.setattr(module.socket, "if_nameindex", lambda: [(1, "lo"), (2, "eth0"), (3, "docker0")])
    with pytest.raises(module.Refusal) as refused:
        module.check_network(job(tmp_path))
    assert "docker0, eth0" in str(refused.value)
    monkeypatch.setattr(module.socket, "if_nameindex", lambda: [(1, "lo")])
    module.check_network(job(tmp_path))
    monkeypatch.setattr(module.socket, "if_nameindex", lambda: [(2, "eth0")])
    module.check_network(job(tmp_path, backend="cpu"))  # the CPU-loop test alone runs on the host's network


def test_batch_order_permutes_each_pass_from_the_seed() -> None:
    order = worker().batch_order(5, 6, 2, 7)
    assert len(order) == 6 and all(len(batch) == 2 for batch in order)
    flat = [index for batch in order for index in batch]
    assert sorted(flat[:5]) == [0, 1, 2, 3, 4] and sorted(flat[5:10]) == [0, 1, 2, 3, 4], "each pass covers all"
    assert order == worker().batch_order(5, 6, 2, 7)
    assert order != worker().batch_order(5, 6, 2, 8)


def test_padded_pads_on_the_right_and_masks_the_padding() -> None:
    ids, mask, labels = worker().padded([5, 6, 7], [-100, 6, 7], 5, 0)
    assert (ids, mask, labels) == ([5, 6, 7, 0, 0], [1, 1, 1, 0, 0], [-100, 6, 7, -100, -100])
    with pytest.raises(worker().Refusal):
        worker().padded([1, 2, 3], [1, 2, 3], 2, 0)


def test_the_worker_refuses_a_job_that_does_not_parse_with_status_3(tmp_path: Path) -> None:
    path = tmp_path / "job.json"
    path.write_bytes(b"not json")
    done = subprocess.run([sys.executable, "-I", str(WORKER), str(path)], capture_output=True, text=True,
                          timeout=120, check=False)
    assert done.returncode == 3, done.stderr
    assert "job" in (done.stdout + done.stderr).lower()


def test_the_worker_names_the_module_it_needs_no_project_or_host() -> None:
    text = WORKER.read_text(encoding="ascii")
    assert "import lassi" not in text and "from lassi" not in text, "the worker runs in the tt-xla venv alone"
