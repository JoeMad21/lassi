"""Fakes for the tests of the ttxla Trainer and its worker (task P17.38); not a test module.

- write_venv(root, ...) writes a SYNTHETIC tt-xla install under
  <root>/<PREFIX_NAME> as toolchains/tt-xla.sh would leave it, without any
  package: its install record, lassi-install.txt, whose first line names
  tt-xla, the wheel's version, and its sha256, and the venv's
  site-packages holding one dist-info directory per distribution, each with
  a METADATA file (Name, Version, and for pjrt-plugin-tt the Summary line
  the real wheel carries, which names the tt-xla, tt-mlir, and tt-metal
  commits). importlib.metadata reads them by path.
- worker_script(mode) returns the text of a fake worker: a Python script
  run with the test interpreter in place of the venv's, which reads the
  job file the trainer wrote, and then, by `mode`: "complete" writes one
  steps line per step (the second step's loss NaN; task P17.42's step
  time, token counts, model FLOPs, and compile flag, the first step a
  compile step), a checkpoint directory
  with SYNTHETIC weight and tokenizer files, and result.json, and exits 0;
  "refuse" writes result.json with status refused and exits 3; "error"
  prints a SYNTHETIC traceback line and exits 1; "crash" exits 139, the
  status a death by SIGSEGV gets through a shell; "hang" sleeps far past
  any test's limit. It records the job file's path and its environment in
  seen.json beside the job file, so a test reads what the worker got.
- write_unshare(directory) writes a stand-in for unshare, run with the test
  interpreter in place of the namespace prefix: it records its flags and
  the command after "--" in unshare.json and runs that command, passing on
  its status, so a test sees that the worker ran under the namespace.

The commits are the ones the real wheel's metadata names
(plans/spikes/tt-training.md, Command 4); every version string marked
SYNTHETIC, every serial, and every loss is SYNTHETIC. No value here is a
measurement.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from lassi.toolchains.pins import read_pin

# The wheel's metadata Summary, as plans/spikes/tt-training.md, Command 4, read it from the real wheel.
WHEEL_SUMMARY = (
    "commit=ff3a67dde32ad3311258674bd3fa260e11a20e98, tt-mlir-commit=c7b8bd7784307cce46577bca1c2d33fdab31fccc, "
    "tt-metal-commit=5280a9cfb00998fd49667a29523d03aee905c129, built-date=2026-02-21, build-type=release"
)
# A Summary naming another tt-metal commit (SYNTHETIC), which the trainer refuses.
OTHER_SUMMARY = WHEEL_SUMMARY.replace("5280a9cfb00998fd49667a29523d03aee905c129", "0" * 40)
# SYNTHETIC versions of the other distributions.
OTHER_DISTRIBUTIONS = {
    "torch": "2.9.0+cpu",
    "torch-xla": "2.9.0+git061c1e7",
    "transformers": "4.57.1",
    "peft": "0.18.1",
}


def pin() -> dict[str, str]:
    """Return toolchains/tt-xla.pin as the repository's pin reader reads it."""
    return read_pin("tt-xla")


def site_packages(venv: Path, minor: str) -> Path:
    """Return the venv's site-packages directory for Python `minor` (such as 3.11)."""
    return venv / "lib" / f"python{minor}" / "site-packages"


def write_dist_info(site: Path, name: str, version: str, summary: str | None = None) -> Path:
    """Write <site>/<name>-<version>.dist-info/METADATA in the wheel layout; return the directory."""
    directory = site / f"{name.replace('-', '_')}-{version}.dist-info"
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["Metadata-Version: 2.4", f"Name: {name}", f"Version: {version}"]
    if summary is not None:
        lines.append(f"Summary: {summary}")
    (directory / "METADATA").write_bytes(("\n".join(lines) + "\n").encode("ascii"))
    return directory


def write_venv(root: Path, *, summary: str = WHEEL_SUMMARY, version: str | None = None, record: bool = True,
               others: Mapping[str, str] = OTHER_DISTRIBUTIONS) -> Path:
    """Write the SYNTHETIC tt-xla install under `root` (the toolchains root); return its prefix."""
    values = pin()
    prefix = root / values["PREFIX_NAME"]
    venv = prefix / values["VENV_DIR"]
    site = site_packages(venv, values["PYTHON_MINOR"])
    write_dist_info(site, "pjrt-plugin-tt", version or values["VERSION"], summary)
    for name, other in others.items():
        write_dist_info(site, name, other)
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    if record:
        first = f"tt-xla {values['VERSION']} {values['WHEEL_SHA256']}"
        (prefix / "lassi-install.txt").write_bytes(f"{first}\nSYNTHETIC rest of the record\n".encode("ascii"))
    return prefix


WORKER_TEMPLATE = '''\
"""A fake ttxla worker for the tests (SYNTHETIC); mode {mode!r}."""
import json
import os
import sys
import time
from pathlib import Path

job_path = Path(sys.argv[-1])
job = json.loads(job_path.read_text(encoding="ascii"))
(job_path.parent / "seen.json").write_text(
    json.dumps({{"argv": sys.argv, "environment": dict(os.environ), "job": str(job_path)}}), encoding="ascii"
)
Path(job["stage_file"]).write_text("checked", encoding="ascii")
mode = {mode!r}
if mode == "refuse":
    Path(job["result_file"]).write_text(json.dumps({{
        "status": "refused", "stage": "checked",
        "message": "data.synthetic record 2 is 999 tokens, more than trainer.max_length 64 (SYNTHETIC)",
    }}), encoding="ascii")
    sys.exit(3)
if mode == "error":
    print("Traceback (SYNTHETIC): RuntimeError: the worker failed", flush=True)
    sys.exit(1)
if mode == "crash":
    Path(job["stage_file"]).write_text("device", encoding="ascii")
    os._exit(139)
if mode == "hang":
    Path(job["stage_file"]).write_text("device", encoding="ascii")
    time.sleep(3600)
with open(job["steps_file"], "w", encoding="ascii") as steps:
    for step in range(1, job["steps"] + 1):
        loss = float("nan") if step == 2 else 1.0 / step
        steps.write(json.dumps({{
            "step": step, "loss": loss, "learning_rate": job["learning_rate"],
            "step_time_s": 10.0 if step == 1 else 0.5, "tokens": 40, "tokens_computed": 64,
            "model_flop": 64000, "compiles": 1 if step == 1 else 0,
            "compile_step": step == 1,
        }}) + "\\n")
checkpoint = Path(job["checkpoint"])
checkpoint.mkdir(parents=True)
(checkpoint / "adapter_model.safetensors").write_bytes(b"SYNTHETIC weights")
(checkpoint / "tokenizer.json").write_text("{{}}", encoding="ascii")
Path(job["stage_file"]).write_text("saved", encoding="ascii")
Path(job["result_file"]).write_text(json.dumps({{
    "status": "complete", "stage": "saved", "message": None, "global_step": job["steps"],
    "train_runtime_s": 1.5, "device": "SYNTHETIC xla:0", "parameters": {{"total": 10, "trainable": 2}},
    "tokens": {{"max_record": 12, "records": [12, 10]}}, "versions": {{"torch": "SYNTHETIC"}}, "notes": [],
    "model_flops": {{"convention": "palm-model-flops", "flop_per_token": 1000}},
}}), encoding="ascii")
'''


def worker_script(mode: str) -> str:
    """Return the fake worker's source for `mode` (complete, refuse, error, crash, or hang)."""
    assert mode in ("complete", "refuse", "error", "crash", "hang"), mode
    return WORKER_TEMPLATE.format(mode=mode)


def write_worker(directory: Path, mode: str) -> Path:
    """Write the fake worker for `mode` as <directory>/fake_worker_<mode>.py; return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"fake_worker_{mode}.py"
    path.write_bytes(worker_script(mode).encode("ascii"))
    return path


UNSHARE_TEMPLATE = '''\
"""A stand-in for unshare (SYNTHETIC): it records its flags and the command after --, then runs that command."""
import json
import subprocess
import sys
from pathlib import Path

args = sys.argv[1:]
cut = args.index("--")
Path({record!r}).write_text(json.dumps({{"flags": args[:cut], "command": args[cut + 1:]}}), encoding="ascii")
sys.exit(subprocess.run(args[cut + 1:], check=False).returncode)
'''


def write_unshare(directory: Path) -> Path:
    """Write the stand-in for unshare as <directory>/fake_unshare.py; it records into unshare.json beside it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "fake_unshare.py"
    record = (directory / "unshare.json").as_posix()
    path.write_bytes(UNSHARE_TEMPLATE.format(record=record).encode("ascii"))
    return path


def unshare_record(directory: Path) -> dict[str, list[str]]:
    """Return what the stand-in for unshare recorded: its flags and the command it ran."""
    return json.loads((directory / "unshare.json").read_text(encoding="ascii"))


def seen(job_dir: Path) -> dict[str, object]:
    """Return what the fake worker recorded beside its job file: its argv, environment, and job path."""
    return json.loads((job_dir / "seen.json").read_text(encoding="ascii"))
