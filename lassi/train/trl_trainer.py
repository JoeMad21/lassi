"""The TRL backend on PyTorch, registered as Trainer "trl": sft, dpo, and grpo on the CPU (task P17.9; 2026-10-23).

Bible: Training Module (Algorithms, Weight Modes, Compute, Safeguards),
Component Interfaces (Trainer), Project Recipes (train.yaml), Agent Rules 1,
5, 6, 7, and 10; OQ-076 (the owner's techniques: full training, frozen
weights, LoRA, and a model of our own).

Declarations. methods sft, dpo, and grpo; weight_modes full, lora, and
prompt_tuning, the frozen-weights mode (every base weight frozen, a PEFT
prompt-tuning soft prompt of trainer.virtual_tokens embeddings trained;
sft only); data_sources synthetic and corpus (bench items as training
examples are P7's and P8's); packages, the distributions whose installed
versions the layer records as the framework pins; and config_keys, the
trainer section's settings: steps, seed, batch_size, learning_rate, and
max_completion_length (grpo only), with no default, and four optional
ones: max_length (sft only, TRL's sequence limit) and
gradient_checkpointing, which stay at TRL's defaults when unset;
virtual_tokens (prompt_tuning only); and chat_format (sft only): true wraps
each record as one user turn (the prompt) and one assistant turn (the
completion), so TRL renders it with the tokenizer's chat template, the
format hf_local serves a model in; false or unset trains on prompt +
completion as plain text. The loader refuses a method, weight mode, or data source outside the
declarations, naming it.

Building (factory(**config)) checks, in order: the settings, naming
trainer.<key>; trainer.device, which must be the CPU until task P17.13
(OQ-040; the owner's direction of 2026-10-23 keeps training on alpha01's
CPU); HF_HOME, which must be set (Agent Rule 7: the framework libraries
keep their caches and config under it); and each framework module, found
with importlib.util.find_spec only, naming the cpu extra when one is
missing. Building imports no framework and reads no file.

check(recipe, data) refuses, before any directory exists, every recipe key,
value, and record this trainer does not carry out, naming it: a base_model
outside the three kinds of lassi.train.bases (fixture:<name>,
scratch:<name>, and <owner>/<name>@<revision>), one whose files are
missing, and a Hub snapshot $HF_HOME/hub lacks (the separate prepare step,
tools/prepare_train.py, downloads it; nothing is downloaded here or in
train()); a scratch base with weights other than full; lora unless the
weights are lora, and then targets other than all-linear (P7);
trainer.virtual_tokens unless the weights are prompt_tuning (then
required), and prompt_tuning with a method other than sft; rollout unless
the method is grpo, and then an engine other than transformers (vllm is
P8's), a group_size below 2, or a trainer.batch_size that is not a whole
number of groups; trainer.max_completion_length unless the method is grpo;
trainer.max_length and trainer.chat_format unless the method is sft; and a
record whose keys are
not exactly the method's FIELDS or whose values are not strings (grpo's
target may be null).

train(job) sets the hub library offline (lassi.llm.hf_local hub_offline),
then imports the framework, sets the seed (transformers.set_seed) and builds
or loads the base (lassi.train.bases load_base: a fixture or scratch base
from its config with random weights, a Hub base from the cache in
float32), refuses an sft record whose tokens with TRL's end token pass
TRL's max_length (check_lengths; nothing is truncated), and runs TRL's
SFTTrainer, DPOTrainer, or GRPOTrainer on the CPU for trainer.steps steps,
each of trainer.batch_size records for sft and dpo. dpo with full weights
is given a copy of the model made right after the seeded build as its
reference model; with lora weights TRL disables the adapter instead. grpo
generates in process (use_vllm False), and each of its steps is
trainer.batch_size completions, rollout.group_size for each of
trainer.batch_size / rollout.group_size records (TRL counts
per_device_train_batch_size in completions for grpo); it scores them with
one reward function, the fixture reward (lassi.train.fixture_reward), which
reads text only and runs nothing (Agent Rule 6); a null reward reaches TRL
as None, which TRL leaves out of the group baseline, and the step's reward
record never counts it as 0 (Agent Rule 1). The checkpoint,
output/checkpoint-<steps>/, holds the weights (the adapter for lora, the
soft prompt for prompt_tuning), the tokenizer, and the records of
lassi.train.checkpoint, with the torch thread count the steps ran with.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from lassi.core.capabilities import TAKES_DEVICE
from lassi.core.devices import FrameworkBuild, parse_device
from lassi.core.interfaces import TrainData, TrainJob, TrainResult
from lassi.core.registry import register
from lassi.llm.hf_local import hub_offline
from lassi.train import bases
from lassi.train import checkpoint as records
from lassi.train.fixture_reward import COMPONENT, fixture_reward, step_rewards

# The framework modules a run imports, in the order a missing one is named.
FRAMEWORK_MODULES = ("torch", "transformers", "trl", "peft", "accelerate", "datasets")
# What a missing framework module's message tells the user to run.
EXTRA_HINT = "run `uv sync --extra cpu`, and tests with `uv run --extra cpu ...`"
# Each method's record fields: TRL's standard, non-conversational prompt-completion and preference formats.
FIELDS = {"sft": ("prompt", "completion"), "dpo": ("prompt", "chosen", "rejected"), "grpo": ("prompt", "target")}
# The one LoRA target and PEFT task type, the one rollout engine, and TRL's smallest grpo group (grpo_config.py:1124).
LORA_TARGETS = "all-linear"
LORA_TASK_TYPE = "CAUSAL_LM"
# The frozen-weights mode: PEFT's prompt tuning, its soft prompt initialized at random (PEFT's PromptTuningInit), for
# the one method that carries it out.
PROMPT_TUNING = "prompt_tuning"
PROMPT_TUNING_INIT = "RANDOM"
ROLLOUT_ENGINE = "transformers"
MIN_GROUP_SIZE = 2
# The reward record a grpo checkpoint holds under trainer_records.reward; TRL logs its own scalar under "reward",
# so each step's fixture reward record goes under REWARD_RECORD.
FIXTURE_REWARD = {"source": "fixture", "rule": COMPONENT, "function_count": 1}
REWARD_RECORD = "reward_record"
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


def _flag(key: str, value: Any) -> bool | None:
    """Return trainer.<key> as a bool, or None when unset; ValueError naming it for anything else (1 is no bool)."""
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"trainer.{key} must be true or false, not {value!r}")
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
    """Trains sft (full, lora, or prompt_tuning weights), dpo, and grpo with TRL on PyTorch, on the CPU."""

    name = "trl"
    capabilities = frozenset({TAKES_DEVICE})
    methods = ("sft", "dpo", "grpo")
    weight_modes = ("full", "lora", PROMPT_TUNING)
    data_sources = ("synthetic", "corpus")
    packages = (
        "torch", "transformers", "trl", "peft", "accelerate", "datasets", "tokenizers", "huggingface-hub",
        "safetensors",
    )
    config_keys = frozenset({
        "steps", "seed", "batch_size", "learning_rate", "max_completion_length", "max_length",
        "gradient_checkpointing", "virtual_tokens", "chat_format",
    })

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
        max_length: Any = None,
        gradient_checkpointing: Any = None,
        virtual_tokens: Any = None,
        chat_format: Any = None,
    ) -> None:
        """Check the settings, then the device, HF_HOME, and the framework; import nothing and read no file."""
        self.steps = _count("steps", steps)
        self.seed = _count("seed", seed, 0, _SEED_LIMIT)
        self.batch_size = _count("batch_size", batch_size)
        self.learning_rate = _rate(learning_rate)
        self.max_completion_length = (
            None if max_completion_length is None else _count("max_completion_length", max_completion_length)
        )
        self.max_length = None if max_length is None else _count("max_length", max_length)
        self.gradient_checkpointing = _flag("gradient_checkpointing", gradient_checkpointing)
        self.virtual_tokens = None if virtual_tokens is None else _count("virtual_tokens", virtual_tokens)
        self.chat_format = _flag("chat_format", chat_format)
        _check_environment(device)

    def check(self, recipe: Mapping[str, Any], data: TrainData) -> None:
        """Raise ValueError naming any recipe key, value, or record this trainer does not carry out."""
        base = bases.check_base(bases.parse_base(recipe.get("base_model")), os.environ.get("HF_HOME", ""))
        method, weights = recipe.get("method"), recipe.get("weights")
        _check_lora(weights, recipe.get("lora"))
        _check_prompt_tuning(method, weights, self.virtual_tokens)
        if base.kind == "scratch" and weights != "full":
            raise ValueError(
                f"base_model {base.name!r} is a scratch base, built with random weights, so it trains full weights "
                f"only; {weights} would train an adapter on random weights"
            )
        _check_rollout(method, recipe.get("rollout"), self.batch_size)
        if method == "grpo" and self.max_completion_length is None:
            raise ValueError("trainer.max_completion_length, grpo's generation length, is required for grpo")
        if method != "grpo" and self.max_completion_length is not None:
            raise ValueError(f"trainer.max_completion_length is set for {method}, which would ignore it")
        if method != "sft" and self.max_length is not None:
            raise ValueError(f"trainer.max_length is set for {method}; the trl trainer sizes sft sequences by it only")
        if method != "sft" and self.chat_format is not None:
            raise ValueError(f"trainer.chat_format is set for {method}; the trl trainer formats sft records only")
        _check_records(method, data)

    def train(self, job: TrainJob) -> TrainResult:
        """Run the steps on the CPU and write output/checkpoint-<steps>/ with its records; return the result."""
        hub_offline()
        import datasets
        import peft
        import torch
        import transformers
        import trl

        recipe = job.recipe
        hf_home = os.environ["HF_HOME"]
        base = bases.check_base(bases.parse_base(recipe["base_model"]), hf_home)
        model, tokenizer = bases.load_base(transformers, torch, base, self.seed, hf_home)
        peft_config = _peft_config(peft, recipe, self.virtual_tokens)
        args = _arguments(trl, self, recipe, Path(job.out_dir))
        items = _records_for(tokenizer, job.data.records, bool(self.chat_format))
        if recipe["method"] == "sft":
            check_lengths(tokenizer, items, args.max_length, f"data.{job.data.source}")
        dataset = datasets.Dataset.from_list(items)
        rewards: dict[int, dict[str, Any]] = {}
        trainer = _trainer(trl, recipe["method"], model, args, dataset, tokenizer, peft_config, rewards)
        trainer.train()
        step = trainer.state.global_step
        name = f"checkpoint-{step}"
        checkpoint = Path(job.out_dir) / name
        trainer.save_model(str(checkpoint))
        base_model = bases.base_record(base, hf_home, self.seed)
        own = {**_trainer_records(recipe, base_model, self.virtual_tokens, args, trainer.state),
               "threads": int(torch.get_num_threads())}
        records.write_records(job, checkpoint, own)
        records.write_steps(checkpoint, _steps(trainer.state.log_history, rewards, recipe["method"] == "grpo"))
        return TrainResult(steps=step, checkpoints=(name,))


def chat_record(record: Mapping[str, str]) -> dict[str, list[dict[str, str]]]:
    """Return an sft record as TRL's conversational prompt-completion: one user turn and one assistant turn."""
    return {"prompt": [{"role": "user", "content": record["prompt"]}],
            "completion": [{"role": "assistant", "content": record["completion"]}]}


def _records_for(tokenizer: Any, items: Sequence[Mapping[str, Any]], chat: bool) -> list[dict[str, Any]]:
    """Return the records as TRL gets them: as written, or as chat turns (chat_format), which need a chat template."""
    if not chat:
        return [dict(record) for record in items]
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("trainer.chat_format is true, but the base's tokenizer has no chat template")
    return [chat_record(record) for record in items]


def check_lengths(tokenizer: Any, items: Sequence[Mapping[str, Any]], max_length: int | None, source: str) -> None:
    """Refuse an sft record whose tokens, with TRL's end token, pass max_length; None checks nothing.

    TRL tokenizes prompt + completion (rendered with the chat template for a
    record of chat turns) and may append the end token, then cuts the
    sequence at max_length (keep_start); the trl trainer truncates no record,
    so one that would be cut is a ValueError naming it (`source` names the
    data source) and the two numbers. The end token is always counted.
    """
    if max_length is None:
        return
    for number, record in enumerate(items, start=1):
        if isinstance(record["prompt"], list):
            rendered = tokenizer.apply_chat_template(record["prompt"] + record["completion"], tokenize=True,
                                                     return_dict=True)
            count = len(rendered["input_ids"]) + 1
        else:
            count = len(tokenizer(record["prompt"] + record["completion"])["input_ids"]) + 1
        if count > max_length:
            raise ValueError(
                f"{source} record {number} is {count} tokens with TRL's end token, more than max_length "
                f"{max_length}; TRL would cut it, and no record is truncated: raise trainer.max_length or split "
                "the record"
            )


def _check_prompt_tuning(method: Any, weights: Any, virtual_tokens: int | None) -> None:
    """Refuse trainer.virtual_tokens unless the weights are prompt_tuning (then required), and prompt tuning off sft."""
    if (weights == PROMPT_TUNING) != (virtual_tokens is not None):
        need = ("is required for prompt_tuning weights" if virtual_tokens is None
                else f"is set for {weights} weights, which ignore it")
        raise ValueError(f"trainer.virtual_tokens, the soft prompt's length, {need}")
    if weights == PROMPT_TUNING and method != "sft":
        raise ValueError(f"weights prompt_tuning is carried out for sft only; {method} with a soft prompt is P7's")


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
    source = f"data.{data.source}"
    for number, record in enumerate(data.records, start=1):
        if sorted(record) != sorted(fields):
            raise ValueError(
                f"{source} record {number} has the keys {sorted(record)}; {method} takes exactly {', '.join(fields)}"
            )
        for key, value in record.items():
            nullable = method == "grpo" and key == "target"
            if not isinstance(value, str) and not (nullable and value is None):
                raise ValueError(f"{source} record {number}: {key} is {value!r}, not a string")


def _peft_config(peft: ModuleType, recipe: Mapping[str, Any], virtual_tokens: int | None) -> Any:
    """Return None for full weights, PEFT's LoraConfig (lora.r, all-linear), or its PromptTuningConfig."""
    if recipe["weights"] == "lora":
        return peft.LoraConfig(r=recipe["lora"]["r"], target_modules=LORA_TARGETS, task_type=LORA_TASK_TYPE)
    if recipe["weights"] == PROMPT_TUNING:
        return peft.PromptTuningConfig(
            num_virtual_tokens=virtual_tokens, task_type=LORA_TASK_TYPE, prompt_tuning_init=PROMPT_TUNING_INIT
        )
    return None


def _peft_record(recipe: Mapping[str, Any], virtual_tokens: int | None) -> dict[str, Any] | None:
    """Return checkpoint.json's peft record: None for full weights, the LoRA or the prompt-tuning settings."""
    if recipe["weights"] == "lora":
        return {"r": recipe["lora"]["r"], "target_modules": LORA_TARGETS, "task_type": LORA_TASK_TYPE}
    if recipe["weights"] == PROMPT_TUNING:
        return {"type": PROMPT_TUNING, "num_virtual_tokens": virtual_tokens, "prompt_tuning_init": PROMPT_TUNING_INIT,
                "task_type": LORA_TASK_TYPE}
    return None


def _arguments(trl: ModuleType, trainer: TRLTrainer, recipe: Mapping[str, Any], out_dir: Path) -> Any:
    """Return TRL's SFTConfig, DPOConfig, or GRPOConfig: the settings on the CPU, logging every step, no saves.

    gradient_checkpointing and, for sft, max_length are passed only when the
    recipe sets them, so an unset one stays at TRL's default, which
    checkpoint.json records with every other argument.
    """
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
    if trainer.gradient_checkpointing is not None:
        common["gradient_checkpointing"] = trainer.gradient_checkpointing
    method = recipe["method"]
    if method == "sft":
        sized = {} if trainer.max_length is None else {"max_length": trainer.max_length}
        return trl.SFTConfig(**common, **sized)
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
    recipe: Mapping[str, Any], base_model: Mapping[str, Any], virtual_tokens: int | None, args: Any, state: Any
) -> dict[str, Any]:
    """Return the trainer's own checkpoint record: method, weights, base model, peft, reward, arguments, summary."""
    return {
        "method": recipe["method"],
        "weights": recipe["weights"],
        "base_model": dict(base_model),
        "peft": _peft_record(recipe, virtual_tokens),
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
