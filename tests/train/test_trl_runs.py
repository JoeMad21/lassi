"""Slow runs of the trl Trainer on the CPU: sft, dpo, and grpo with a tiny model built from a config (task P17.9).

Bible: Training Module (Algorithms, Weight Modes, Compute: every checkpoint
records the device, the framework pins, and the resolved train recipe;
Safeguards: checkpoints log data split hashes), Model Serving (Serving
Rules: the hub library is set offline before any load), Agent Rules 1, 5,
6, 7, and 10; plans/p17-portable.md, task P17.9:

- B1: "A registered Trainer runs a few steps each, on the CPU, with a tiny
  model built from a config and synthetic data under `tests/fixtures/train/`:
  sft and dpo, with full and lora weights, and grpo, with in-process
  generation and a fixture reward, executing nothing."
- B2: "Each run writes a checkpoint directory holding per-step losses and
  reward components, the resolved train recipe, the device record, and the
  pins. [...] The tests are marked slow."

The contract these tests fix, end to end through lassi.train.run
run_training with lassi's registry and the host's cpu probe:

- Each smoke recipe (tests/train/trl_smoke.py) completes with steps 2 and
  one checkpoint, output/checkpoint-2/, holding the weights of its mode
  (full: model.safetensors and config.json; lora: adapter_model.safetensors
  and adapter_config.json with r 8 on every linear layer but the output
  layer, which is all-linear, and task type CAUSAL_LM; every lora_B tensor
  has moved from its start at 0), the tokenizer, recipe.resolved.yaml with
  the train directory's bytes, checkpoint.json, and steps.jsonl.
- checkpoint.json holds provenance.json's train_id, trainer, recipe_hash,
  commit, dirty, device_records, driver, framework_pins (torch's CPU build;
  trl 1.14.1, peft 0.21.2, accelerate 1.15.0, transformers 5.18.0), and
  data (with the split hash), plus trainer_records: method, weights,
  base_model {name, files (sha256 of each base file), seed}, peft (null for
  full weights; r 8, all-linear, and task type CAUSAL_LM for lora), reward
  (null for sft and dpo; the fixture rule for grpo), training_args (TRL's
  arguments as run: two steps, the batch size, the learning rate, the seed
  as seed and data_seed, use_cpu, no saves, no reports), global_step, and
  train_summary (TRL's final log entry).
- steps.jsonl holds one line per optimizer step, in order, each with TRL's
  logged step, a finite loss, a grad_norm above 0, learning_rate, and the
  method's own entries (dpo's rewards/margins; grpo's reward and
  reward_std); each grpo line adds "reward_record", the step's fixture
  reward record (lassi.train.fixture_reward step_rewards). TRL logs its own
  scalar under "reward", so lassi's record takes another key.
- grpo's null target is skipped, never read as 0: exactly one step's record
  holds the null record's group (skipped_null 2, count 2), its mean is the
  mean of the scored values, and TRL's logged reward for that step equals
  it (TRL 1.14.1 marks the row unscorable with one reward function).
- grpo runs nothing: no process is started during the steps, and TRL is
  given one reward function, lassi's.
- One seed repeats every step: two grpo runs give the same losses, gradient
  norms, and reward records.
- No connection leaves the host (only loopback is allowed), the hub
  library's offline flag is on after a run, nothing is written into the
  working directory, and the committed fixtures are unchanged.
- `lassi train` runs the grpo smoke from the command line and exits 0.
- framework() reads torch's build metadata only, never the device runtime.

These tests need the cpu extra and skip without it, naming it
(tests/tiny_hf.py NEEDS_EXTRA): run them as `uv run --extra cpu pytest
tests/train/test_trl_runs.py`. Each run is two steps of a 2-layer model with
hidden size 16 on 4 SYNTHETIC records; the budget is under 30 s per test
and under 2 minutes for the file. The weights are random and no value in
this module is a measurement.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import multiprocessing.process
import os
import socket
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from tiny_hf import NEEDS_EXTRA
from train_fakes import write_recipe
from trl_smoke import (
    BASE_DIR,
    BASE_MODEL,
    BATCH_SIZE,
    GROUP_SIZE,
    LEARNING_RATE,
    LORA,
    MAX_COMPLETION_LENGTH,
    METHODS,
    PINS,
    SEED,
    STEPS,
    SYNTHETIC_DIR,
    TRAINER,
    WEIGHT_MODES,
    base_hashes,
    smoke_recipe,
    trl_trainer,
)

pytestmark = pytest.mark.slow

torch = pytest.importorskip("torch", reason=NEEDS_EXTRA)
transformers = pytest.importorskip("transformers", reason=NEEDS_EXTRA)
trl = pytest.importorskip("trl", reason=NEEDS_EXTRA)
pytest.importorskip("peft", reason=NEEDS_EXTRA)

import lassi.cli  # noqa: E402  (registers the host's device probes and every trainer)
from lassi.core.devices import FrameworkBuild  # noqa: E402
from lassi.train import run as train_run  # noqa: E402

CHECKPOINT = f"checkpoint-{STEPS}"
PROVENANCE_KEYS = (
    "train_id", "trainer", "recipe_hash", "commit", "dirty", "device_records", "driver", "framework_pins", "data",
)
TRAINER_RECORD_KEYS = {
    "method", "weights", "base_model", "peft", "reward", "training_args", "global_step", "train_summary",
}
FIXTURE_REWARD = {"source": "fixture", "rule": "target_fraction", "function_count": 1}
LOOPBACK = ("127.0.0.1", "::1", "localhost")
SMOKE = [(method, weights) for method in METHODS for weights in WEIGHT_MODES]


@pytest.fixture(autouse=True)
def contained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Run each test with HF_HOME under tmp, an empty working directory, the hub flag off, and no outbound socket.

    Yields the refused connection attempts; the test fails at teardown if
    any was made, or if the working directory or the committed fixtures
    changed.
    """
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    constants = importlib.import_module("huggingface_hub.constants")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    attempts: list[str] = []
    real_connect = socket.socket.connect

    def connect(self: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else address
        if host in LOOPBACK:
            return real_connect(self, address)
        attempts.append(repr(address))
        raise OSError(f"test guard: no connection leaves the host ({address!r})")

    def create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        attempts.append(repr(address))
        raise OSError(f"test guard: no connection leaves the host ({address!r})")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    before = fixture_listing()
    yield attempts
    assert attempts == [], f"a connection was attempted: {attempts}"
    assert sorted(item.name for item in workdir.iterdir()) == [], "a run wrote into the working directory"
    assert fixture_listing() == before, "a run changed the committed fixtures"


def fixture_listing() -> dict[str, str]:
    """Return the sha256 of every file under tests/fixtures/train/, by relative path."""
    return {
        path.relative_to(SYNTHETIC_DIR).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(SYNTHETIC_DIR.rglob("*")) if path.is_file()
    }


def run_smoke(root: Path, method: str, weights: str, train_id: str = "t1") -> Path:
    """Write the smoke recipe for `method` and `weights` under `root` and run it; return the train directory."""
    trl_trainer()
    path = write_recipe(root / "recipes", smoke_recipe(method, weights), name=f"{method}-{weights}")
    options = train_run.TrainOptions(runs_root=root / "runs-root", train_id=train_id, roots=[root / "recipes"])
    return train_run.run_training(path, options)


def read_json(path: Path) -> Any:
    """Return a JSON record file after checking it is ASCII with LF only and a final LF; NaN is refused."""
    raw = path.read_bytes()
    assert raw.isascii() and b"\r" not in raw and raw.endswith(b"\n"), path.name
    return json.loads(raw.decode("ascii"), parse_constant=refuse_constant)


def refuse_constant(name: str) -> Any:
    """Fail on NaN, Infinity, or -Infinity in a record."""
    raise AssertionError(f"a record holds {name}")


def read_steps(checkpoint: Path) -> list[dict[str, Any]]:
    """Return the objects of a checkpoint's steps.jsonl, after the ASCII and LF checks."""
    raw = (checkpoint / "steps.jsonl").read_bytes()
    assert raw.isascii() and b"\r" not in raw and raw.endswith(b"\n")
    return [json.loads(line, parse_constant=refuse_constant) for line in raw.decode("ascii").split("\n")[:-1]]


def linear_layers() -> set[str]:
    """Return the names of the fixture model's linear layers but the output layer: what all-linear covers."""
    config = transformers.AutoConfig.from_pretrained(str(BASE_DIR), local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_config(config)
    output = model.get_output_embeddings()
    return {
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and module is not output
    }


# ---------------------------------------------------------------------------
# Each method and weight mode writes a checkpoint (B1, B2)


@pytest.mark.parametrize(("method", "weights"), SMOKE, ids=[f"{m}-{w}" for m, w in SMOKE])
def test_each_method_and_weight_mode_writes_a_checkpoint(method: str, weights: str, tmp_path: Path) -> None:
    train_dir = run_smoke(tmp_path, method, weights)
    provenance = read_json(train_dir / "provenance.json")
    assert (provenance["status"], provenance["steps"], provenance["checkpoints"]) == ("complete", STEPS, [CHECKPOINT])
    assert provenance["trainer"] == TRAINER
    assert sorted(item.name for item in train_dir.iterdir()) == ["output", "provenance.json", "recipe.resolved.yaml"]
    checkpoint = train_dir / "output" / CHECKPOINT
    names = {item.name for item in checkpoint.iterdir()}
    weights_files = {"full": {"model.safetensors", "config.json"}, "lora": {"adapter_model.safetensors",
                                                                             "adapter_config.json"}}[weights]
    assert weights_files | {"tokenizer.json", "recipe.resolved.yaml", "checkpoint.json", "steps.jsonl"} <= names
    if weights == "lora":
        assert "model.safetensors" not in names, "lora weights save the adapter, not the whole model"
        adapter = json.loads((checkpoint / "adapter_config.json").read_bytes())
        assert adapter["r"] == LORA["r"]
        assert set(adapter["target_modules"]) == linear_layers(), "all-linear: every linear layer but the output"
        assert adapter["task_type"] == "CAUSAL_LM"
        tensors = importlib.import_module("safetensors.torch").load_file(str(checkpoint / "adapter_model.safetensors"))
        lora_b = [tensor for name, tensor in tensors.items() if "lora_B" in name]
        assert lora_b and all(bool(tensor.any()) for tensor in lora_b), "lora_B starts at 0; training moves it"
    assert (checkpoint / "recipe.resolved.yaml").read_bytes() == (train_dir / "recipe.resolved.yaml").read_bytes()
    check_records(read_json(checkpoint / "checkpoint.json"), provenance, method, weights)
    check_steps(read_steps(checkpoint), method)


def check_records(records: dict[str, Any], provenance: dict[str, Any], method: str, weights: str) -> None:
    """Fail unless checkpoint.json copies provenance.json's keys and holds the trainer's records."""
    assert set(records) == {*PROVENANCE_KEYS, "trainer_records"}, sorted(records)
    for key in PROVENANCE_KEYS:
        assert records[key] == provenance[key], key
    assert records["data"]["split_hash"] and records["device_records"][0]["key"] == "trainer.device"
    pins = records["framework_pins"]
    assert pins["build"] == {"name": "torch", "version": torch.__version__, "cuda": None, "hip": None}
    packages = pins["packages"]
    assert packages["torch"] == torch.__version__ and packages["torch"].startswith(PINS["torch"])
    assert {name: packages[name] for name in ("transformers", "trl", "peft", "accelerate")} == {
        name: PINS[name] for name in ("transformers", "trl", "peft", "accelerate")
    }
    own = records["trainer_records"]
    assert set(own) == TRAINER_RECORD_KEYS, sorted(own)
    assert (own["method"], own["weights"], own["global_step"]) == (method, weights, STEPS)
    assert own["base_model"] == {"name": BASE_MODEL, "files": base_hashes(), "seed": SEED}
    assert (own["peft"] is None) == (weights == "full")
    if weights == "lora":
        assert own["peft"] == {"r": LORA["r"], "target_modules": "all-linear", "task_type": "CAUSAL_LM"}
    assert own["reward"] == (FIXTURE_REWARD if method == "grpo" else None)
    check_training_args(own["training_args"], method)
    assert {"train_loss", "train_runtime"} <= set(own["train_summary"]), own["train_summary"]


def check_training_args(args: dict[str, Any], method: str) -> None:
    """Fail unless TRL's recorded arguments are the smoke settings on the CPU with no saves and no reports."""
    assert args["max_steps"] == STEPS
    assert args["per_device_train_batch_size"] == BATCH_SIZE[method]
    assert args["learning_rate"] == LEARNING_RATE
    assert (args["seed"], args["data_seed"]) == (SEED, SEED)
    assert args["use_cpu"] is True
    assert (args["logging_steps"], args["save_strategy"]) == (1, "no")
    assert args["report_to"] in ([], "none", ["none"]), args["report_to"]
    if method == "grpo":
        assert (args["num_generations"], args["max_completion_length"]) == (GROUP_SIZE, MAX_COMPLETION_LENGTH)
        assert args["use_vllm"] is False


def check_steps(steps: list[dict[str, Any]], method: str) -> None:
    """Fail unless steps.jsonl holds one line per step, in order, with a finite loss and the method's entries."""
    assert [line["step"] for line in steps] == list(range(1, STEPS + 1)), steps
    for line in steps:
        assert isinstance(line["loss"], float) and math.isfinite(line["loss"]), line
        assert {"grad_norm", "learning_rate"} <= set(line), sorted(line)
        assert isinstance(line["grad_norm"], float) and line["grad_norm"] > 0, line
        assert not line.get("nonfinite"), line
        if method == "dpo":
            assert "rewards/margins" in line, sorted(line)
        if method == "grpo":
            assert {"reward", "reward_std", "reward_record"} <= set(line), sorted(line)
            record = line["reward_record"]
            assert set(record) == {"values", "count", "skipped_null", "mean", "components"}, record
            assert len(record["values"]) == BATCH_SIZE["grpo"]
            assert set(record["components"]) == {"target_fraction"}
        else:
            assert "reward_record" not in line


# ---------------------------------------------------------------------------
# grpo: a null reward is skipped, nothing is executed, one seed repeats a run


def test_grpo_skips_a_null_reward_and_never_reads_it_as_zero(tmp_path: Path) -> None:
    steps = read_steps(run_smoke(tmp_path, "grpo", "full") / "output" / CHECKPOINT)
    with_null = [line for line in steps if line["reward_record"]["skipped_null"]]
    assert len(with_null) == 1, [line["reward_record"] for line in steps]
    (line,) = with_null
    record = line["reward_record"]
    scored = [value for value in record["values"] if value is not None]
    assert (record["skipped_null"], record["count"], len(scored)) == (GROUP_SIZE, 2, 2), record
    assert all(0.0 <= value <= 1.0 for value in scored), record
    assert record["mean"] == pytest.approx(sum(scored) / len(scored), abs=1e-12)
    assert record["components"]["target_fraction"] == {"count": 2, "mean": pytest.approx(record["mean"], abs=1e-12)}
    assert line["reward"] == pytest.approx(record["mean"], abs=1e-6), "TRL's reward leaves the null out too"
    for other in steps:
        if other is not line:
            assert other["reward_record"]["skipped_null"] == 0
            assert other["reward"] == pytest.approx(other["reward_record"]["mean"], abs=1e-6)


def test_grpo_executes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cls = trl_trainer().TRLTrainer
    started: list[str] = []
    trainers: list[Any] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        started.append(repr(args)[:200])
        raise AssertionError(f"a process was started during the steps (Agent Rule 6): {args!r}"[:300])

    original_train = cls.train
    original_init = trl.GRPOTrainer.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        trainers.append(self)
        original_init(self, *args, **kwargs)

    def guarded(self: Any, job: Any) -> Any:
        with monkeypatch.context() as patch:
            patch.setattr(subprocess.Popen, "__init__", refuse)
            patch.setattr(multiprocessing.process.BaseProcess, "start", refuse)
            for name in ("system", "popen", "startfile", "spawnv", "spawnve", "execv", "execve", "posix_spawn",
                         "posix_spawnp", "fork"):
                if hasattr(os, name):
                    patch.setattr(os, name, refuse)
            return original_train(self, job)

    monkeypatch.setattr(cls, "train", guarded)
    monkeypatch.setattr(trl.GRPOTrainer, "__init__", init)
    train_dir = run_smoke(tmp_path, "grpo", "lora")
    assert read_json(train_dir / "provenance.json")["status"] == "complete"
    assert started == []
    (trainer,) = trainers
    (reward,) = trainer.reward_funcs
    assert callable(reward) and not isinstance(reward, torch.nn.Module), "the reward is a function, not a model"
    assert reward.__module__.startswith("lassi.train."), reward.__module__


def test_a_rerun_with_the_same_seed_repeats_every_step(tmp_path: Path) -> None:
    first = read_steps(run_smoke(tmp_path, "grpo", "full", train_id="first") / "output" / CHECKPOINT)
    second = read_steps(run_smoke(tmp_path, "grpo", "full", train_id="second") / "output" / CHECKPOINT)
    keys = ("step", "loss", "grad_norm", "reward", "reward_record")
    assert [{key: line[key] for key in keys} for line in second] == [{key: line[key] for key in keys} for line in first]


def test_the_hub_library_is_offline_after_a_run(tmp_path: Path) -> None:
    run_smoke(tmp_path, "sft", "lora")
    assert importlib.import_module("huggingface_hub.constants").HF_HUB_OFFLINE is True


# ---------------------------------------------------------------------------
# The command line, the framework build, and the fixture tokenizer


def test_lassi_train_runs_the_grpo_smoke_from_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trl_trainer()
    recipe = write_recipe(tmp_path / "recipes", smoke_recipe("grpo", "lora"), name="grpo-lora")
    runs_root = tmp_path / "runs-root"
    argv = ["--graphics", "off", "train", str(recipe), "--runs-root", str(runs_root), "--train-id", "g1"]
    code = lassi.cli.main(argv)
    out = capsys.readouterr().out
    train_dir = runs_root / "train" / "g1"
    assert code == 0
    assert f"train directory: {train_dir}\n" in out, out
    assert (train_dir / "output" / CHECKPOINT / "checkpoint.json").is_file()


def test_framework_reads_the_torch_build(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("framework() initialized the device runtime (OQ-002)")

    monkeypatch.setattr(torch.cuda, "is_available", refuse)
    monkeypatch.setattr(torch.cuda, "device_count", refuse)
    build = trl_trainer().TRLTrainer.framework()
    assert build == FrameworkBuild(name="torch", version=torch.__version__, cuda=torch.version.cuda,
                                   hip=torch.version.hip)


def test_the_fixture_tokenizer_gives_each_byte_its_own_id() -> None:
    config = json.loads((BASE_DIR / "tokenizer_config.json").read_bytes())
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(BASE_DIR / "tokenizer.json"), eos_token=config["eos_token"], pad_token=config["pad_token"]
    )
    text = "SYNTHETIC abc {}\n"
    ids = tokenizer(text)["input_ids"]
    assert ids == [2 + byte for byte in text.encode("ascii")]
    assert tokenizer.decode(ids) == text
    assert (len(tokenizer), tokenizer.eos_token_id, tokenizer.pad_token_id) == (258, 1, 0)
