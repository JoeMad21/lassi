"""The TRL backend on PyTorch, registered as Trainer "trl": sft, dpo, and grpo on the CPU (task P17.9).

Bible: Training Module (Algorithms, Weight Modes, Compute, Safeguards),
Component Interfaces (Trainer), Project Recipes (train.yaml), Agent Rules 1,
5, 6, 7, and 10.

Declarations. methods sft, dpo, and grpo; weight_modes full and lora;
data_sources synthetic (bench items as training examples are P7's and P8's);
packages, the distributions whose installed versions the layer records as
the framework pins; and config_keys, the trainer section's settings: steps,
seed, batch_size, learning_rate, and max_completion_length (grpo only). The
code picks no default for any of them. The loader refuses a method, weight
mode, or data source outside the declarations, naming it.

Building (factory(**config)) checks, in order: the settings, naming
trainer.<key>; trainer.device, which must be the CPU until task P17.13
(OQ-040); HF_HOME, which must be set (Agent Rule 7: the framework libraries
keep their caches and config under it); and each framework module, found
with importlib.util.find_spec only, naming the cpu extra when one is
missing. Building imports no framework and reads no file.

check(recipe, data) refuses, before any directory exists, every recipe key,
value, and record this trainer does not carry out, naming it: a base_model
that is not `fixture:<name>`, one directory under the synthetic fixture
directory holding BASE_FILES (a Hub base needs a revision pin, Agent Rule
10; P7); lora unless the weights are lora, and then targets other than
all-linear (P7); rollout unless the method is grpo, and then an engine other
than transformers (vllm is P8's), a group_size below 2, or a
trainer.batch_size that is not a whole number of groups;
trainer.max_completion_length unless the method is grpo; and a record whose
keys are not exactly the method's FIELDS or whose values are not strings
(grpo's target may be null).

train(job) sets the hub library offline (lassi.llm.hf_local hub_offline),
then imports the framework, builds the base model from its config with
random weights under transformers.set_seed(seed) (nothing is downloaded),
and runs TRL's SFTTrainer, DPOTrainer, or GRPOTrainer on the CPU for
trainer.steps steps, each of trainer.batch_size records for sft and dpo.
dpo with full weights is given a copy of the model made right after the
seeded build as its reference model; with lora weights TRL disables the
adapter instead. grpo generates in process (use_vllm False), and each of
its steps is trainer.batch_size completions, rollout.group_size for each of
trainer.batch_size / rollout.group_size records (TRL counts
per_device_train_batch_size in completions for grpo); it scores them with
one reward function, the fixture reward
(lassi.train.fixture_reward), which reads text only and runs nothing (Agent
Rule 6); a null reward reaches TRL as None, which TRL leaves out of the
group baseline, and the step's reward record never counts it as 0 (Agent
Rule 1). The checkpoint, output/checkpoint-<steps>/, holds the weights (the
adapter for lora), the tokenizer, and the records of lassi.train.checkpoint.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from lassi.core.capabilities import TAKES_DEVICE
from lassi.core.devices import FrameworkBuild, parse_device
from lassi.core.interfaces import TrainData, TrainJob, TrainResult
from lassi.core.registry import register
from lassi.llm.hf_local import hub_offline
from lassi.train import checkpoint as records
from lassi.train import data as train_data
from lassi.train.fixture_reward import COMPONENT, fixture_reward, step_rewards

# The framework modules a run imports, in the order a missing one is named.
FRAMEWORK_MODULES = ("torch", "transformers", "trl", "peft", "accelerate", "datasets")
# What a missing framework module's message tells the user to run.
EXTRA_HINT = "run `uv sync --extra cpu`, and tests with `uv run --extra cpu ...`"
# The files a base fixture directory holds: the model config and the tokenizer.
BASE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
# Each method's record fields: TRL's standard, non-conversational prompt-completion and preference formats.
FIELDS = {"sft": ("prompt", "completion"), "dpo": ("prompt", "chosen", "rejected"), "grpo": ("prompt", "target")}
# The one LoRA target and PEFT task type, the one rollout engine, and TRL's smallest grpo group (grpo_config.py:1124).
LORA_TARGETS = "all-linear"
LORA_TASK_TYPE = "CAUSAL_LM"
ROLLOUT_ENGINE = "transformers"
MIN_GROUP_SIZE = 2
# The reward record a grpo checkpoint holds under trainer_records.reward; TRL logs its own scalar under "reward",
# so each step's fixture reward record goes under REWARD_RECORD.
FIXTURE_REWARD = {"source": "fixture", "rule": COMPONENT, "function_count": 1}
REWARD_RECORD = "reward_record"
_FIXTURE = re.compile(r"fixture:([A-Za-z0-9][A-Za-z0-9._-]*)")
_HUB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
_SEED_LIMIT = 2**32


def _count(key: str, value: Any, low: int = 1, limit: int | None = None) -> int:
    """Return trainer.<key> as an integer of at least `low` and below `limit`; ValueError naming it otherwise."""
    is_int = isinstance(value, int) and not isinstance(value, bool)
    if not is_int or value < low or (limit is not None and value >= limit):
        upper = "" if limit is None else f" and at most {limit - 1}"
        raise ValueError(f"trainer.{key} must be an integer of at least {low}{upper}, not {value!r}")
    return value


def _rate(value: Any) -> float:
    """Return trainer.learning_rate as a finite number above 0 (an int is accepted, a bool is not)."""
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not is_number or not math.isfinite(value) or value <= 0:
        raise ValueError(f"trainer.learning_rate must be a finite number above 0, not {value!r}")
    return value


def _check_environment(device: Any) -> None:
    """Refuse a device other than the CPU, an unset HF_HOME, and a missing framework module, in that order."""
    spec = parse_device(device, "trainer.device")
    if spec.kind != "cpu":
        raise ValueError(
            f"trainer.device kind {spec.kind}: the trl trainer runs on the CPU only until task P17.13, and no GPU "
            "host is planned (OQ-040)"
        )
    if not os.environ.get("HF_HOME"):
        raise ValueError("HF_HOME is not set or is empty; the framework libraries keep their caches under it "
                         "(Agent Rule 7)")
    for module in FRAMEWORK_MODULES:
        if importlib.util.find_spec(module) is None:
            raise ValueError(f"the trl trainer needs {module}, which installs only with a framework extra: "
                             f"{EXTRA_HINT}")


@register("Trainer", "trl")
class TRLTrainer:
    """Trains sft and dpo (full or lora weights) and grpo with TRL on PyTorch, on the CPU (task P17.9)."""

    name = "trl"
    capabilities = frozenset({TAKES_DEVICE})
    methods = ("sft", "dpo", "grpo")
    weight_modes = ("full", "lora")
    data_sources = ("synthetic",)
    packages = (
        "torch", "transformers", "trl", "peft", "accelerate", "datasets", "tokenizers", "huggingface-hub",
        "safetensors",
    )
    config_keys = frozenset({"steps", "seed", "batch_size", "learning_rate", "max_completion_length"})

    @staticmethod
    def framework() -> FrameworkBuild:
        """Return torch's build read from its version metadata, never a device runtime call (OQ-002)."""
        try:
            torch = importlib.import_module("torch")
        except ImportError as error:
            raise ImportError(f"the trl trainer needs PyTorch, which installs only with a framework extra: "
                              f"{EXTRA_HINT}") from error
        return FrameworkBuild(name="torch", version=torch.__version__, cuda=torch.version.cuda, hip=torch.version.hip)

    def __init__(
        self,
        *,
        device: Any = None,
        steps: Any = None,
        seed: Any = None,
        batch_size: Any = None,
        learning_rate: Any = None,
        max_completion_length: Any = None,
    ) -> None:
        """Check the settings, then the device, HF_HOME, and the framework; import nothing and read no file."""
        self.steps = _count("steps", steps)
        self.seed = _count("seed", seed, 0, _SEED_LIMIT)
        self.batch_size = _count("batch_size", batch_size)
        self.learning_rate = _rate(learning_rate)
        self.max_completion_length = (
            None if max_completion_length is None else _count("max_completion_length", max_completion_length)
        )
        _check_environment(device)

    def check(self, recipe: Mapping[str, Any], data: TrainData) -> None:
        """Raise ValueError naming any recipe key, value, or record this trainer does not carry out."""
        _check_base(recipe.get("base_model"))
        _check_lora(recipe.get("weights"), recipe.get("lora"))
        method = recipe.get("method")
        _check_rollout(method, recipe.get("rollout"), self.batch_size)
        if method == "grpo" and self.max_completion_length is None:
            raise ValueError("trainer.max_completion_length, grpo's generation length, is required for grpo")
        if method != "grpo" and self.max_completion_length is not None:
            raise ValueError(f"trainer.max_completion_length is set for {method}, which would ignore it")
        _check_records(method, data)

    def train(self, job: TrainJob) -> TrainResult:
        """Run the steps on the CPU and write output/checkpoint-<steps>/ with its records; return the result."""
        hub_offline()
        import datasets
        import peft
        import transformers
        import trl

        recipe = job.recipe
        base = _base_dir(recipe["base_model"])
        model, tokenizer = _load_base(transformers, base, self.seed)
        peft_config = _peft_config(peft, recipe)
        dataset = datasets.Dataset.from_list([dict(record) for record in job.data.records])
        rewards: dict[int, dict[str, Any]] = {}
        args = _arguments(trl, self, recipe, Path(job.out_dir))
        trainer = _trainer(trl, recipe["method"], model, args, dataset, tokenizer, peft_config, rewards)
        trainer.train()
        step = trainer.state.global_step
        name = f"checkpoint-{step}"
        checkpoint = Path(job.out_dir) / name
        trainer.save_model(str(checkpoint))
        own = _trainer_records(recipe, base, self.seed, peft_config, args, trainer.state)
        records.write_records(job, checkpoint, own)
        records.write_steps(checkpoint, _steps(trainer.state.log_history, rewards, recipe["method"] == "grpo"))
        return TrainResult(steps=step, checkpoints=(name,))


def _base_dir(name: Any) -> Path:
    """Return the fixture directory `fixture:<name>` names; ValueError naming base_model for anything else."""
    if isinstance(name, str) and _HUB_ID.fullmatch(name):
        raise ValueError(
            f"base_model {name!r}: loading a base model from the Hub needs a revision pin (Agent Rule 10), which "
            "is P7's; use a fixture:<name> base built from its config"
        )
    match = _FIXTURE.fullmatch(name) if isinstance(name, str) else None
    root = train_data.SYNTHETIC_DIR
    if match is None:
        raise ValueError(f"base_model {name!r} is not fixture:<name>, one directory under {root}")
    path = root / match.group(1)
    if path.resolve().parent != root.resolve():
        raise ValueError(f"base_model {name!r} names a path outside {root}")
    return path


def _check_base(name: Any) -> None:
    """Refuse a base_model that is not a fixture directory holding BASE_FILES, naming base_model."""
    path = _base_dir(name)
    missing = [file for file in BASE_FILES if not (path / file).is_file()]
    if missing:
        raise ValueError(f"base_model {name!r}: {path} lacks {', '.join(missing)}; no base fixture is there")


def _check_lora(weights: Any, lora: Any) -> None:
    """Refuse lora unless the weights are lora (then required), a missing lora.r, and targets but all-linear."""
    if (weights == "lora") != (lora is not None):
        need = "is required for lora weights" if lora is None else f"is set for {weights} weights, which ignore it"
        raise ValueError(f"lora {need}")
    if lora is None:
        return
    if "r" not in lora:
        raise ValueError("lora.r is a required choice with no value")
    targets = lora.get("targets")
    if targets != LORA_TARGETS:
        raise ValueError(
            f"lora.targets {targets!r}: the trl trainer carries out {LORA_TARGETS} only (every linear layer but "
            "the output layer); other targets are P7's"
        )


def _check_rollout(method: Any, rollout: Any, batch_size: int) -> None:
    """Refuse rollout unless the method is grpo (then required), an engine but transformers, and a bad group."""
    if (method == "grpo") != (rollout is not None):
        need = "is required for grpo" if rollout is None else f"is set for {method}, which ignores it"
        raise ValueError(f"rollout {need}")
    if rollout is None:
        return
    engine = rollout.get("engine")
    if engine != ROLLOUT_ENGINE:
        raise ValueError(
            f"rollout.engine {engine!r}: the trl trainer generates in process ({ROLLOUT_ENGINE}); vllm is P8's"
        )
    group = rollout.get("group_size")
    if not isinstance(group, int) or isinstance(group, bool) or group < MIN_GROUP_SIZE:
        raise ValueError(f"rollout.group_size must be an integer of at least {MIN_GROUP_SIZE}, not {group!r}")
    if batch_size % group:
        raise ValueError(
            f"trainer.batch_size {batch_size} is not a whole number of groups of rollout.group_size {group}"
        )


def _check_records(method: Any, data: TrainData) -> None:
    """Refuse a record whose keys are not the method's fields or whose values are not strings, naming its number."""
    fields = FIELDS[method]
    for number, record in enumerate(data.records, start=1):
        if sorted(record) != sorted(fields):
            raise ValueError(
                f"data.synthetic record {number} has the keys {sorted(record)}; {method} takes exactly "
                f"{', '.join(fields)}"
            )
        for key, value in record.items():
            nullable = method == "grpo" and key == "target"
            if not isinstance(value, str) and not (nullable and value is None):
                raise ValueError(f"data.synthetic record {number}: {key} is {value!r}, not a string")


def _load_base(transformers: ModuleType, base: Path, seed: int) -> tuple[Any, Any]:
    """Return the base model built from its config with random weights under the seed, and its tokenizer."""
    config = transformers.AutoConfig.from_pretrained(str(base), local_files_only=True)
    settings = json.loads((base / "tokenizer_config.json").read_bytes().decode("ascii"))
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(base / "tokenizer.json"),
        eos_token=settings["eos_token"],
        pad_token=settings["pad_token"],
        model_max_length=settings["model_max_length"],
    )
    transformers.set_seed(seed)
    model = transformers.AutoModelForCausalLM.from_config(config)
    return model, tokenizer


def _peft_config(peft: ModuleType, recipe: Mapping[str, Any]) -> Any:
    """Return None for full weights, or PEFT's LoraConfig with lora.r on every linear layer for lora weights."""
    if recipe["weights"] != "lora":
        return None
    return peft.LoraConfig(r=recipe["lora"]["r"], target_modules=LORA_TARGETS, task_type=LORA_TASK_TYPE)


def _arguments(trl: ModuleType, trainer: TRLTrainer, recipe: Mapping[str, Any], out_dir: Path) -> Any:
    """Return TRL's SFTConfig, DPOConfig, or GRPOConfig: the settings on the CPU, logging every step, no saves."""
    common = {
        "output_dir": str(out_dir),
        "max_steps": trainer.steps,
        "per_device_train_batch_size": trainer.batch_size,
        "learning_rate": trainer.learning_rate,
        "seed": trainer.seed,
        "data_seed": trainer.seed,
        "use_cpu": True,
        "logging_strategy": "steps",
        "logging_steps": 1,
        "save_strategy": "no",
        "eval_strategy": "no",
        "report_to": "none",
        "disable_tqdm": True,
    }
    method = recipe["method"]
    if method == "sft":
        return trl.SFTConfig(**common)
    if method == "dpo":
        return trl.DPOConfig(**common)
    return trl.GRPOConfig(
        num_generations=recipe["rollout"]["group_size"],
        max_completion_length=trainer.max_completion_length,
        use_vllm=False,
        **common,
    )


def _reward_function(rewards: dict[int, dict[str, Any]]) -> Any:
    """Return grpo's one reward function: the fixture reward per completion, each step's record kept in `rewards`."""

    def target_fraction(
        prompts: Any, completions: Any, completion_ids: Any, target: Any, trainer_state: Any, **kwargs: Any
    ) -> list[float | None]:
        """Score each completion against its record's target; None, never 0, where the target is null."""
        found = [fixture_reward(goal, completion) for goal, completion in zip(target, completions, strict=True)]
        rewards[trainer_state.global_step + 1] = step_rewards(found)
        return [reward.value for reward in found]

    return target_fraction


def _trainer(
    trl: ModuleType, method: str, model: Any, args: Any, dataset: Any, tokenizer: Any, peft_config: Any,
    rewards: dict[int, dict[str, Any]],
) -> Any:
    """Return TRL's trainer for the method; dpo with full weights gets a copy of the seeded model as reference."""
    common = {"model": model, "args": args, "train_dataset": dataset, "processing_class": tokenizer,
              "peft_config": peft_config}
    if method == "sft":
        return trl.SFTTrainer(**common)
    if method == "dpo":
        reference = copy.deepcopy(model) if peft_config is None else None
        return trl.DPOTrainer(ref_model=reference, **common)
    return trl.GRPOTrainer(reward_funcs=[_reward_function(rewards)], **common)


def _trainer_records(
    recipe: Mapping[str, Any], base: Path, seed: int, peft_config: Any, args: Any, state: Any
) -> dict[str, Any]:
    """Return the trainer's own checkpoint record: method, weights, base model, peft, reward, arguments, summary."""
    files = {file: hashlib.sha256((base / file).read_bytes()).hexdigest() for file in BASE_FILES}
    peft = None if peft_config is None else {
        "r": recipe["lora"]["r"], "target_modules": LORA_TARGETS, "task_type": LORA_TASK_TYPE,
    }
    return {
        "method": recipe["method"],
        "weights": recipe["weights"],
        "base_model": {"name": recipe["base_model"], "files": files, "seed": seed},
        "peft": peft,
        "reward": dict(FIXTURE_REWARD) if recipe["method"] == "grpo" else None,
        "training_args": json.loads(json.dumps(args.to_dict())),
        "global_step": state.global_step,
        "train_summary": records.finite_record(state.log_history[-1]),
    }


def _steps(history: list[dict[str, Any]], rewards: dict[int, dict[str, Any]], grpo: bool) -> list[dict[str, Any]]:
    """Return TRL's per-step log entries (those with a loss), each grpo one with its step's fixture reward record."""
    steps = []
    for entry in history:
        if "loss" not in entry:
            continue
        line = dict(entry)
        if grpo:
            line[REWARD_RECORD] = rewards[entry["step"]]
        steps.append(line)
    return steps
