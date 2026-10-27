"""The tt-xla backend, registered as Trainer "ttxla": sft on one Tenstorrent card through a worker (task P17.38).

Bible: Training Module (Weight Modes, Compute, Safeguards), Component
Interfaces (Trainer), Toolchain Pins, Agent Rules 1, 5, 6, 7, 9, and 10;
plans/spikes/tt-training.md, Finding 3 (the design); OQ-100 (a) with (s2),
OQ-101, OQ-102.

The training stack is the tt-xla wheel pjrt-plugin-tt with PyTorch/XLA,
installed by toolchains/tt-xla.sh into its own venv,
$LASSI_TOOLCHAINS/<PREFIX_NAME of toolchains/tt-xla.pin>/<VENV_DIR>, on
CPython 3.11 with torch 2.9.0, beside LASSI's Python 3.10 environment. So
the steps run in a worker process, lassi/train/ttxla_worker.py, under the
venv's own interpreter with `-I`; LASSI's side imports no framework at all.
Everything that runs on the card is tracked code: the wheel, PyTorch, and
the worker, never model output (Agent Rule 6).

Declarations. Method sft alone: no reward, so RL never touches silicon
(Agent Rule 9). Weight modes full, lora (PEFT LoraConfig, lora.r, every
linear layer but the output layer, as the trl trainer's), and
prompt_tuning (PEFT PromptTuningConfig, trainer.virtual_tokens, RANDOM
initialization). Data sources synthetic and corpus. packages, the venv's
distributions, whose versions the train layer reads from the venv's
site-packages by path (package_paths(); lassi.train.run), never from
LASSI's environment. config_keys: steps, seed, batch_size, learning_rate,
max_length (every sequence is padded to it, one static shape on the card;
a record longer is refused, never cut), and timeout_s (the worker's wall
limit, 60 to 86400 seconds; past it the run is a hang), each required with
no default; network, required with no default, whose one value is none
(below); gradient_checkpointing and chat_format (bools, off when unset), and
virtual_tokens (prompt_tuning only).

network none: the worker runs in its own empty network namespace,
`unshare --user --map-root-user --net --` (NETWORK_NAMESPACE, unshare by
path; the user and network namespaces of the sandbox's unshare, without
its mounts, which a training run that writes its checkpoint and reads the
venv and the Hub cache cannot keep). The wheel's libtt_metal.so is
Tracy-enabled, and the Tracy client starts at library load, listens on TCP
8086 to 8105 on every interface, and broadcasts on UDP every 3 s, with no
variable that turns the broadcast off (plans/spikes/tt-perf.md, the
training finding; inferred, not observed), so on the build host's shared
network no worker may have one. check() refuses, before any directory
exists, a host where that namespace cannot be made (network_namespace_problem
runs `true` under it), and the worker refuses to touch a card while it sees
any network interface but the loopback. The device profiler stays off: the
worker's environment, built from nothing, names no profiler variable.

framework() reads the install: its record, lassi-install.txt, whose first
line names tt-xla, the wheel's version, and its sha256, and the venv's
pjrt-plugin-tt metadata, by path: its version must be the pin's and its
Summary must name the pin's tt-xla, tt-mlir, and tt-metal commits, so a
wheel built on another tt-metal is refused (Agent Rule 10). It returns
FrameworkBuild(pjrt-plugin-tt, <version>, None, None), which the
tenstorrent device kind needs (lassi.core.devices).

Building checks the settings, naming trainer.<key>; trainer.device, which
must be one Tenstorrent card (kind tenstorrent, one index: one card per
run); HF_HOME (Agent Rule 7); and an absolute LASSI_SCRATCH, under which
the worker keeps its tt-metal cache and home. check() refuses what this
trainer does not carry out, as the module docstring of lassi.train.bases
and the trl trainer's describe: the base (fixture, scratch, or Hub at a
revision, its snapshot already in $HF_HOME/hub), a scratch base with an
adapter, lora and virtual_tokens out of place, and records whose keys are
not exactly prompt and completion, or whose values are not strings.

train(job) writes output/worker/job.json (worker_job: the resolved base,
the settings, the records, and where to write), builds the worker's
environment from nothing (worker_environment), and runs the worker once
through lassi.toolchains' EnvRunner, the runner the stage runner runs git
with (exactly that environment, a session of its own, killed at the wall
limit), under the nice and ionice of a background job; the worker inherits
lassi train's core limit and refuses to touch a card unless it reads it as
exactly 1 byte (Agent Rule 7; plans/runs/train-tt/run.sh holds it). Its
output is kept in output/worker/worker.log, its steps in
output/worker/steps.jsonl as they come. On an interrupt or a SIGTERM to
lassi train, the worker's session is killed before the exception goes on.
The ending is read as the tt_silicon executor reads a program's
(lassi.train.silicon):

- status 0: output/checkpoint-<steps>/ must hold the weights the worker
  saved (the adapter for lora and prompt_tuning, the full weights
  otherwise, with the tokenizer); this module then writes its records
  through lassi.train.checkpoint: recipe.resolved.yaml, checkpoint.json
  (with this trainer's own record: method, weights, base_model as
  lassi.train.bases records it, peft, the settings, global_step, the
  worker's result, and performance, lassi.analysis.tt_rates
  training_summary over the steps without a compile and the worker's
  model_flops, naming the card: task P17.42), and steps.jsonl, each
  non-finite value null and named;
- status 3, a refusal the worker wrote (a record longer than max_length, a
  chat format without a template): WorkerRefusal, a ValueError;
- any other status from 1 to 127: WorkerError, a RuntimeError naming the
  stage the worker reached and the end of its log; both are
  silicon.CleanEnding, so the card's quarantine marker is removed;
- a hang past timeout_s, a death by signal, or a status outside 0 to 127:
  silicon.CardEnding naming it, after the worker is killed, so the marker
  stays; nothing is retried.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lassi.analysis.tt_rates import training_summary
from lassi.core.capabilities import TAKES_DEVICE
from lassi.core.devices import FrameworkBuild, parse_device
from lassi.core.interfaces import TrainData, TrainJob, TrainResult
from lassi.core.registry import register
from lassi.toolchains import EnvRunner
from lassi.toolchains.pins import read_pin
from lassi.train import bases
from lassi.train import checkpoint as records
from lassi.train.silicon import CardEnding, CleanEnding

TRAINER = "ttxla"
PIN = "tt-xla"
# The worker script, run under the venv's interpreter; the record of an install; the distribution of the wheel.
WORKER = Path(__file__).resolve().with_name("ttxla_worker.py")
INSTALL_RECORD = "lassi-install.txt"
WHEEL = "pjrt-plugin-tt"
# The worker's status for a refusal it wrote to result.json; its job format.
REFUSED_STATUS = 3
JOB_FORMAT = "lassi-ttxla-job-1"
# The worker's wall limit's range, in seconds.
TIMEOUT_RANGE = (60, 86400)
# The worker's files under output/worker/.
WORK_DIR = "worker"
# The one LoRA target, PEFT's task type, and the soft prompt's initialization, as the trl trainer's.
LORA_TARGETS = "all-linear"
TASK_TYPE = "CAUSAL_LM"
PROMPT_TUNING_INIT = "RANDOM"
# The dtype the worker trains in, and the tt backend's compile options, as tt-blacksmith's recipe at the wheel
# passes them (blacksmith/models/torch/huggingface/hf_models.py at 002ebe47).
DTYPE = "bfloat16"
COMPILE_OPTIONS = {"tt_enable_torch_fx_fusion_pass": False, "tt_legacy_compile": True}
# The background job's priorities: ionice best effort at its lowest level, and nice 10, as the gate runs jobs.
NICENESS = ("ionice", "-c", "2", "-n", "7", "nice", "-n", "10")
# trainer.network's one value, and the prefix that runs the worker in its own empty network namespace: a new user
# namespace (the current user mapped to root inside, which an unprivileged user may make, as the sandbox's unshare
# does) holding a new network namespace with only a loopback device, down. The wheel's libtt_metal.so is
# Tracy-enabled, and the Tracy client starts at library load, listens on TCP 8086 to 8105 on every interface, and
# broadcasts on UDP every 3 s, with no variable that turns the broadcast off (plans/spikes/tt-perf.md, the training
# finding; inferred from the wheel's strings and Tracy's source, not observed), so no worker may reach a network.
NETWORK_NONE = "none"
NETWORK_NAMESPACE = ("unshare", "--user", "--map-root-user", "--net", "--")
# The most seconds the namespace probe may take.
NAMESPACE_PROBE_S = 30.0
_SEED_LIMIT = 2**32


class WorkerRefusal(CleanEnding, ValueError):
    """A record or setting the worker refused before any step (its status 3); the card was left as it was."""


class WorkerError(CleanEnding, RuntimeError):
    """The worker failed with a status from 1 to 127 and neither hung nor was killed; the card's state is known."""


def _pin() -> dict[str, str]:
    """Return toolchains/tt-xla.pin."""
    return read_pin(PIN)


def install_prefix() -> Path:
    """Return $LASSI_TOOLCHAINS/<PREFIX_NAME>; ValueError when LASSI_TOOLCHAINS is unset or relative."""
    root = os.environ.get("LASSI_TOOLCHAINS", "")
    if not root or not Path(root).is_absolute():
        raise ValueError(
            "LASSI_TOOLCHAINS is unset or not absolute; the ttxla trainer runs in the tt-xla venv that "
            "toolchains/tt-xla.sh installs there"
        )
    return Path(root) / _pin()["PREFIX_NAME"]


def venv_dir() -> Path:
    """Return the tt-xla venv: <install prefix>/<VENV_DIR>."""
    return install_prefix() / _pin()["VENV_DIR"]


def venv_python() -> Path:
    """Return the venv's interpreter, which runs the worker."""
    return venv_dir() / "bin" / "python"


def site_packages() -> Path:
    """Return the venv's site-packages: lib/python<PYTHON_MINOR>/site-packages."""
    return venv_dir() / "lib" / f"python{_pin()['PYTHON_MINOR']}" / "site-packages"


def read_wheel(site: Path, pin: Mapping[str, str]) -> FrameworkBuild:
    """Return the wheel's FrameworkBuild from its metadata in `site`; ValueError for a wheel the pin does not name."""
    import importlib.metadata

    found = next(iter(importlib.metadata.distributions(name=WHEEL, path=[str(site)])), None)
    if found is None:
        raise ValueError(f"{site} holds no {WHEEL}; install it with toolchains/tt-xla.sh")
    if found.version != pin["VERSION"]:
        raise ValueError(f"{site} holds {WHEEL} {found.version}, not the pinned {pin['VERSION']} (Agent Rule 10)")
    summary = found.metadata.get("Summary") or ""
    for key, value in (("commit", pin["TT_XLA_COMMIT"]), ("tt-mlir-commit", pin["TT_MLIR_COMMIT"]),
                       ("tt-metal-commit", pin["TT_METAL_COMMIT"])):
        if f"{key}={value}" not in [part.strip() for part in summary.split(",")]:
            raise ValueError(
                f"{WHEEL} {found.version} in {site} does not name {key}={value} in its metadata ({summary!r}); a wheel "
                "built on another tt-metal or tt-mlir commit is refused (Agent Rule 10)"
            )
    return FrameworkBuild(name=WHEEL, version=found.version, cuda=None, hip=None)


def _count(key: str, value: Any, low: int = 1, limit: int | None = None) -> int:
    """Return trainer.<key> as an integer from `low` and below `limit`; ValueError naming it otherwise."""
    is_int = isinstance(value, int) and not isinstance(value, bool)
    if not is_int or value < low or (limit is not None and value >= limit):
        upper = "" if limit is None else f" and at most {limit - 1}"
        raise ValueError(f"trainer.{key} must be an integer of at least {low}{upper}, not {value!r}")
    return value


def _flag(key: str, value: Any) -> bool:
    """Return trainer.<key> as a bool, False when unset; ValueError naming it for anything else."""
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"trainer.{key} must be true or false, not {value!r}")
    return bool(value)


def _rate(value: Any) -> float:
    """Return trainer.learning_rate as a finite number above 0."""
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not is_number or not math.isfinite(value) or value <= 0:
        raise ValueError(f"trainer.learning_rate must be a finite number above 0, not {value!r}")
    return float(value)


def _card(device: Any) -> int:
    """Return the one Tenstorrent card trainer.device names; ValueError naming trainer.device otherwise."""
    spec = parse_device(device, "trainer.device")
    if spec.kind != "tenstorrent" or len(spec.indices) != 1:
        raise ValueError(
            f"trainer.device {spec.kind} {list(spec.indices)}: the ttxla trainer trains on one Tenstorrent card, "
            "kind tenstorrent with one index (one card per run)"
        )
    return spec.indices[0]


def _scratch() -> Path:
    """Return $LASSI_SCRATCH; ValueError when it is unset or relative (Agent Rule 7)."""
    value = os.environ.get("LASSI_SCRATCH", "")
    if not value or not Path(value).is_absolute():
        raise ValueError("LASSI_SCRATCH is unset or not absolute; the worker keeps its tt-metal cache and home there "
                         "(Agent Rule 7)")
    return Path(value)


def worker_command(
    python: Path, worker: Path, job: Path, wrappers: Sequence[str], namespace: Sequence[str]
) -> list[str]:
    """Return the wrappers, the network namespace, then the venv's interpreter with -I on the worker and its job."""
    return [*wrappers, *namespace, str(python), "-I", str(worker), str(job)]


def network_prefix(posix: bool | None = None, which: Any = None) -> list[str]:
    """Return NETWORK_NAMESPACE with unshare by path; FileNotFoundError off Linux or without unshare on PATH.

    `posix` None means this host's os.name; `which` None means shutil.which.
    """
    is_posix = os.name == "posix" if posix is None else posix
    if not is_posix:
        raise FileNotFoundError("the worker's empty network namespace needs Linux's unshare (util-linux), which this "
                                "host does not have")
    found = (shutil.which if which is None else which)(NETWORK_NAMESPACE[0])
    if not found:
        raise FileNotFoundError("unshare (util-linux) is not on PATH, so the worker's empty network namespace cannot "
                                "be made")
    return [found, *NETWORK_NAMESPACE[1:]]


def network_namespace_problem() -> str | None:
    """Return why the worker's empty network namespace cannot be made on this host, or None when it can.

    It runs `true` under network_prefix() through lassi.toolchains' EnvRunner,
    with a bare PATH: a user namespace that this user may not make, or a
    network namespace inside it that the host refuses, ends that command
    with a non-zero status, and its stderr says why. No device is touched.
    """
    try:
        prefix = network_prefix()
    except FileNotFoundError as error:
        return str(error)
    result = EnvRunner({"PATH": "/usr/bin:/bin"})([*prefix, "true"], Path("/"), NAMESPACE_PROBE_S)
    if result.returncode != 0:
        return f"`{' '.join([*prefix, 'true'])}` ended with status {result.returncode}: {result.stderr.strip()[-400:]}"
    return None


def _network(value: Any) -> str:
    """Return trainer.network, which must be NETWORK_NONE; ValueError naming it otherwise (it has no default)."""
    if value != NETWORK_NONE:
        raise ValueError(
            f"trainer.network must be {NETWORK_NONE!r}, with no default: the worker runs in its own empty network "
            "namespace, since the wheel's Tracy client listens on TCP 8086 to 8105 and broadcasts on UDP from library "
            f"load (plans/spikes/tt-perf.md); got {value!r}"
        )
    return value


@register("Trainer", TRAINER)
class TtxlaTrainer:
    """Trains sft (full, lora, or prompt_tuning weights) on one Tenstorrent card through the tt-xla wheel."""

    name = TRAINER
    capabilities = frozenset({TAKES_DEVICE})
    methods = ("sft",)
    weight_modes = ("full", "lora", "prompt_tuning")
    data_sources = ("synthetic", "corpus")
    packages = (
        "pjrt-plugin-tt", "torch", "torch-xla", "jax", "jaxlib", "transformers", "peft", "accelerate", "tokenizers",
        "huggingface-hub", "safetensors", "numpy",
    )
    config_keys = frozenset({
        "steps", "seed", "batch_size", "learning_rate", "max_length", "timeout_s", "gradient_checkpointing",
        "chat_format", "virtual_tokens", "network",
    })

    @staticmethod
    def framework() -> FrameworkBuild:
        """Return the venv's wheel build after checking the install record; ValueError naming what is wrong."""
        pin = _pin()
        record = install_prefix() / INSTALL_RECORD
        expected = f"{pin['NAME']} {pin['VERSION']} {pin['WHEEL_SHA256']}"
        try:
            first = record.read_bytes().decode("ascii", errors="replace").splitlines()[0]
        except (OSError, IndexError):
            first = None
        if first != expected:
            raise ValueError(
                f"{record} does not begin with {expected!r}, so the tt-xla install is missing or another pin's; "
                "install it with toolchains/tt-xla.sh"
            )
        return read_wheel(site_packages(), pin)

    @staticmethod
    def package_paths() -> list[Path]:
        """Return the venv's site-packages, where the train layer reads the framework pins by path."""
        return [site_packages()]

    def __init__(
        self,
        *,
        device: Any = None,
        steps: Any = None,
        seed: Any = None,
        batch_size: Any = None,
        learning_rate: Any = None,
        max_length: Any = None,
        timeout_s: Any = None,
        gradient_checkpointing: Any = None,
        chat_format: Any = None,
        virtual_tokens: Any = None,
        network: Any = None,
    ) -> None:
        """Check the settings, the device, HF_HOME, and LASSI_SCRATCH; import nothing and read no file."""
        self.steps = _count("steps", steps)
        self.seed = _count("seed", seed, 0, _SEED_LIMIT)
        self.batch_size = _count("batch_size", batch_size)
        self.learning_rate = _rate(learning_rate)
        self.max_length = _count("max_length", max_length, 2)
        self.timeout_s = _count("timeout_s", timeout_s, TIMEOUT_RANGE[0], TIMEOUT_RANGE[1] + 1)
        self.gradient_checkpointing = _flag("gradient_checkpointing", gradient_checkpointing)
        self.chat_format = _flag("chat_format", chat_format)
        self.virtual_tokens = None if virtual_tokens is None else _count("virtual_tokens", virtual_tokens)
        self.network = _network(network)
        self.card = _card(device)
        if not os.environ.get("HF_HOME"):
            raise ValueError("HF_HOME is not set or is empty; the bases are read from $HF_HOME/hub (Agent Rule 7)")
        _scratch()

    def check(self, recipe: Mapping[str, Any], data: TrainData) -> None:
        """Raise ValueError naming any recipe key, value, or record this trainer does not carry out."""
        base = bases.parse_base(recipe.get("base_model"))
        weights = recipe.get("weights")
        _check_lora(weights, recipe.get("lora"))
        if (weights == "prompt_tuning") != (self.virtual_tokens is not None):
            need = ("is required for prompt_tuning weights" if self.virtual_tokens is None
                    else f"is set for {weights} weights, which ignore it")
            raise ValueError(f"trainer.virtual_tokens, the soft prompt's length, {need}")
        if base.kind == "scratch" and weights != "full":
            raise ValueError(f"base_model {base.name!r} is a scratch base, built with random weights, so it trains "
                             f"full weights only; {weights} would train an adapter on random weights")
        bases.check_base(base, os.environ.get("HF_HOME", ""))
        source = f"data.{data.source}"
        for number, record in enumerate(data.records, start=1):
            if sorted(record) != ["completion", "prompt"]:
                raise ValueError(f"{source} record {number} has the keys {sorted(record)}; sft takes exactly prompt, "
                                 "completion")
            for key, value in record.items():
                if not isinstance(value, str):
                    raise ValueError(f"{source} record {number}: {key} is {value!r}, not a string")
        problem = network_namespace_problem()
        if problem is not None:
            raise ValueError(f"trainer.network {self.network}: the worker's empty network namespace cannot be made "
                             f"on this host, so the worker is not started: {problem}")

    def worker_job(self, recipe: Mapping[str, Any], data: TrainData, work: Path, checkpoint: Path) -> dict[str, Any]:
        """Return the worker's job: the resolved base, the settings, the records, and the files it writes."""
        hf_home = os.environ["HF_HOME"]
        base = bases.check_base(bases.parse_base(recipe["base_model"]), hf_home)
        lora = recipe.get("lora")
        return {
            "format": JOB_FORMAT,
            "backend": "xla",
            "base": _resolved_base(base, hf_home),
            "weights": recipe["weights"],
            "lora_r": None if lora is None else lora["r"],
            "virtual_tokens": self.virtual_tokens,
            "steps": self.steps,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_length": self.max_length,
            "gradient_checkpointing": self.gradient_checkpointing,
            "chat_format": self.chat_format,
            "source": f"data.{data.source}",
            "network": self.network,
            "dtype": DTYPE,
            "compile_options": dict(COMPILE_OPTIONS),
            "records": [dict(record) for record in data.records],
            "checkpoint": str(checkpoint),
            "steps_file": str(work / "steps.jsonl"),
            "result_file": str(work / "result.json"),
            "stage_file": str(work / "stage"),
        }

    def worker_environment(self, work: Path) -> dict[str, str]:
        """Return the worker's whole environment, built from nothing (the module docstring; the spike, Finding 3)."""
        scratch = _scratch()
        home = scratch / "ttxla" / "home"
        venv = venv_dir()
        return {
            "PATH": os.pathsep.join([str(venv / "bin"), "/usr/bin", "/bin"]),
            "HOME": str(home),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "TMPDIR": str(work / "tmp"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONNOUSERSITE": "1",
            "HF_HOME": os.environ["HF_HOME"],
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PJRT_DEVICE": "TT",
            "XLA_STABLEHLO_COMPILE": "1",
            "TT_VISIBLE_DEVICES": str(self.card),
            "TT_METAL_CACHE": str(scratch / "ttxla" / "cache"),
            "TT_METAL_LOGS_PATH": str(work / "logs"),
            "TTXLA_LOGGER_LEVEL": "INFO",
            **_windows_root(),
        }

    def train(self, job: TrainJob) -> TrainResult:
        """Run the worker once on the card and write output/checkpoint-<steps>/'s records; return the result."""
        out_dir = Path(job.out_dir)
        work = out_dir / WORK_DIR
        name = f"checkpoint-{self.steps}"
        checkpoint = out_dir / name
        try:
            payload = self.worker_job(job.recipe, job.data, work, checkpoint)
            environment = self.worker_environment(work)
            made = (work / "tmp", work / "logs", Path(environment["HOME"]), Path(environment["TT_METAL_CACHE"]))
            for directory in made:
                directory.mkdir(parents=True, exist_ok=True)
            job_path = work / "job.json"
            job_path.write_bytes(json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n")
            python = venv_python()
            if not python.is_file():
                raise FileNotFoundError(f"{python} is not a file; install the venv with toolchains/tt-xla.sh")
            command = worker_command(python, WORKER, job_path, _niceness(), network_prefix())
        except (OSError, ValueError) as error:
            raise WorkerError(f"the ttxla worker was not started: {type(error).__name__}: {error}") from error
        status, hung = _run(command, environment, work, self.timeout_s)
        result = _read_json(work / "result.json")
        _settle(status, hung, result, work, self.timeout_s)
        return self._records(job, payload, result, checkpoint, name)

    def _records(self, job: TrainJob, payload: Mapping[str, Any], result: Mapping[str, Any], checkpoint: Path,
                 name: str) -> TrainResult:
        """Check the worker's checkpoint and write its records through lassi.train.checkpoint."""
        step = result.get("global_step")
        if step != self.steps or not checkpoint.is_dir():
            raise WorkerError(f"the ttxla worker reported global_step {step!r} and left {checkpoint} "
                              f"{'present' if checkpoint.is_dir() else 'missing'}; trainer.steps is {self.steps}")
        steps = [json.loads(line) for line in _lines(Path(payload["steps_file"]))]
        own = self._trainer_records(job.recipe, result, steps)
        records.write_records(job, checkpoint, own)
        records.write_steps(checkpoint, steps)
        return TrainResult(steps=self.steps, checkpoints=(name,))

    def _trainer_records(
        self, recipe: Mapping[str, Any], result: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Return this trainer's own checkpoint record (the module docstring)."""
        hf_home = os.environ["HF_HOME"]
        base = bases.check_base(bases.parse_base(recipe["base_model"]), hf_home)
        worker = {key: value for key, value in result.items() if key not in ("status", "message")}
        return {
            "method": recipe["method"],
            "weights": recipe["weights"],
            "base_model": bases.base_record(base, hf_home, self.seed),
            "peft": _peft_record(recipe, self.virtual_tokens),
            "reward": None,
            "settings": {
                "steps": self.steps, "seed": self.seed, "batch_size": self.batch_size,
                "learning_rate": self.learning_rate, "max_length": self.max_length, "timeout_s": self.timeout_s,
                "gradient_checkpointing": self.gradient_checkpointing, "chat_format": self.chat_format,
                "virtual_tokens": self.virtual_tokens, "card": self.card, "network": self.network, "dtype": DTYPE,
                "compile_options": dict(COMPILE_OPTIONS),
            },
            "global_step": result.get("global_step"),
            "worker": records.finite_record(worker),
            "performance": training_summary(
                steps, result.get("model_flops"), f"tenstorrent card {self.card} ({result.get('device')})"
            ),
        }


def _check_lora(weights: Any, lora: Any) -> None:
    """Refuse lora unless the weights are lora (then required), a missing lora.r, and targets other than all-linear."""
    if (weights == "lora") != (lora is not None):
        need = "is required for lora weights" if lora is None else f"is set for {weights} weights, which ignore it"
        raise ValueError(f"lora {need}")
    if lora is None:
        return
    if "r" not in lora:
        raise ValueError("lora.r is a required choice with no value")
    if lora.get("targets") != LORA_TARGETS:
        raise ValueError(f"lora.targets {lora.get('targets')!r}: the ttxla trainer carries out {LORA_TARGETS} only "
                         "(every linear layer but the output layer)")


def _peft_record(recipe: Mapping[str, Any], virtual_tokens: int | None) -> dict[str, Any] | None:
    """Return checkpoint.json's peft record: None for full weights, the LoRA or the prompt-tuning settings."""
    if recipe["weights"] == "lora":
        return {"r": recipe["lora"]["r"], "target_modules": LORA_TARGETS, "task_type": TASK_TYPE}
    if recipe["weights"] == "prompt_tuning":
        return {"type": "prompt_tuning", "num_virtual_tokens": virtual_tokens, "prompt_tuning_init": PROMPT_TUNING_INIT,
                "task_type": TASK_TYPE}
    return None


def _resolved_base(base: bases.Base, hf_home: str) -> dict[str, Any]:
    """Return the base as the worker reads it: local directories only, nothing on the Hub."""
    if base.kind == "fixture":
        return {"kind": "fixture", "path": str(base.path), "tokenizer": None}
    assert base.repo is not None and base.revision is not None
    snapshot = str(bases.snapshot_dir(hf_home, base.repo, base.revision))
    if base.kind == "hub":
        return {"kind": "hub", "path": snapshot, "tokenizer": None}
    return {"kind": "scratch", "path": str(base.path), "tokenizer": snapshot}


def _windows_root() -> dict[str, str]:
    """Return SYSTEMROOT on Windows, where only the tests run the worker and Python cannot start without it."""
    return {"SYSTEMROOT": os.environ.get("SYSTEMROOT", "")} if os.name == "nt" else {}


def _niceness() -> list[str]:
    """Return the worker's priority wrappers on a POSIX host (ionice and nice by path), none elsewhere."""
    if os.name != "posix":
        return []
    found = {tool: shutil.which(tool) for tool in ("ionice", "nice")}
    missing = [tool for tool, path in found.items() if path is None]
    if missing:
        raise FileNotFoundError(f"{', '.join(missing)} is not on PATH; the worker runs niced as a background job")
    return [found.get(word) or word for word in NICENESS]


def _run(command: list[str], environment: Mapping[str, str], work: Path, timeout_s: int) -> tuple[int, bool]:
    """Run the worker once through lassi.toolchains' EnvRunner; return (status, hung) and write worker.log.

    EnvRunner gives the worker exactly `environment`, runs it in a session of
    its own, and kills that session when it runs past timeout_s (status -1
    with its "timed out" line, read here as a hang) or when an exception
    (an interrupt, or a SIGTERM to lassi train) comes while it waits, which
    then goes on. The worker inherits lassi train's core limit and refuses
    to touch a card unless it reads (1, 1).
    """
    with _terminated_on_sigterm():
        result = EnvRunner(environment)(command, work, timeout_s)
    log = f"{result.stdout}{result.stderr}"
    (work / "worker.log").write_bytes(log.encode("utf-8", errors="replace"))
    hung = result.returncode == -1 and result.stderr.rstrip().endswith(f"timed out after {timeout_s:g} s")
    return result.returncode, hung


class _Terminated(BaseException):
    """A SIGTERM to lassi train while the worker runs; it kills the worker and ends the run."""


@contextmanager
def _terminated_on_sigterm() -> Iterator[None]:
    """Turn a SIGTERM into _Terminated while the block runs (main thread only), so the worker is killed first."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def raise_terminated(number: int, frame: Any) -> None:
        raise _Terminated(f"signal {number}")

    previous = signal.signal(signal.SIGTERM, raise_terminated)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _settle(status: int, hung: bool, result: Mapping[str, Any], work: Path, timeout_s: int) -> None:
    """Raise for any ending but a complete run, as the module docstring reads the endings."""
    stage = _stage(work)
    if hung:
        raise CardEnding(f"the ttxla worker hung at stage {stage!r} and was killed",
                         f"a hang at the wall limit, trainer.timeout_s {timeout_s} s (exit status {status}); the "
                         f"worker had reached stage {stage!r} and was killed")
    if status < 0 or status > 127:
        ending = f"a death by signal, or a status outside 0 to 127 (exit status {status}) at stage {stage!r}"
        raise CardEnding(f"the ttxla worker ended with {ending}", ending)
    if status == REFUSED_STATUS and result.get("status") == "refused":
        raise WorkerRefusal(str(result.get("message")))
    if status != 0 or result.get("status") != "complete":
        tail = "\n".join(_lines(work / "worker.log")[-20:])
        raise WorkerError(f"the ttxla worker failed with exit status {status} at stage {stage!r}; the end of "
                          f"{work / 'worker.log'}:\n{tail}")


def _stage(work: Path) -> str:
    """Return the last stage the worker wrote, or "not started"."""
    try:
        return (work / "stage").read_bytes().decode("ascii", errors="replace").strip() or "not started"
    except OSError:
        return "not started"


def _lines(path: Path) -> list[str]:
    """Return a text file's non-empty lines, [] when it cannot be read."""
    try:
        return [line for line in path.read_bytes().decode("utf-8", errors="replace").splitlines() if line.strip()]
    except OSError:
        return []


def _read_json(path: Path) -> dict[str, Any]:
    """Return a JSON object file's content, {} when it is missing or not an object."""
    try:
        found = json.loads(path.read_bytes().decode("ascii", errors="replace"))
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}

