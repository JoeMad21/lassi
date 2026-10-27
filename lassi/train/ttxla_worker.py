"""The ttxla worker: sft steps on one Tenstorrent card through PyTorch/XLA, run in the tt-xla venv (task P17.38).

Bible: Training Module (Weight Modes, Compute, Safeguards), Agent Rules 1,
6, 7, and 10; plans/spikes/tt-training.md, Finding 3, step 2.

    <tt-xla venv>/bin/python -I ttxla_worker.py <job.json>

lassi.train.ttxla_trainer writes the job and runs this script with the
venv's own interpreter, isolated (-I), so it imports no LASSI module and
nothing from the working directory: at module level the standard library alone,
and torch, torch_xla, transformers, and peft only inside functions. It is
tracked code, never model output (Agent Rule 6), and reads only local
files: the base from its directory or Hub snapshot, offline.

In order, writing each stage's name to the job's stage file:

1. checked: the job (read_job: the format lassi-ttxla-job-1, every key,
   the backend xla, or cpu for the CPU-loop test alone, and the network
   none); for xla, no network interface in view but the loopback (the
   trainer runs the worker in its own empty network namespace, since the
   wheel's Tracy client listens and broadcasts from library load) and, on a
   POSIX host, RLIMIT_CORE inherited at exactly (1, 1).
2. encoded: each record tokenized as the trl trainer formats it: the
   prompt and completion as plain text with the end token appended, or,
   with chat_format, one user turn and one assistant turn through the
   tokenizer's chat template; the prompt's tokens and the padding are
   masked out of the loss (-100), so only the completion trains, as TRL
   trains a prompt-completion record. A record longer than max_length, or
   with no completion token, and chat_format without a chat template are
   refused: nothing is ever cut.
3. loaded: the base built on the CPU: a fixture or scratch base from its
   config with random weights under the seed, a Hub base from its
   snapshot; then PEFT's LoRA (r, all-linear, CAUSAL_LM) or prompt tuning
   (virtual_tokens, RANDOM), gradient checkpointing when asked, and the
   dtype.
4. device: written before the first device call. For xla it sets the
   device type TT, takes torch_xla.device(), moves the model there, and
   compiles it with torch.compile's tt backend and the job's options, as
   tt-blacksmith's recipe at the wheel does; for cpu nothing is compiled.
5. training: `steps` steps of batch_size records in batch_order, every
   sequence padded to max_length (one static shape): zero the gradients,
   the forward pass, the cross-entropy over the shifted completion tokens,
   the backward pass, a sync, torch's AdamW step, a sync, then one JSON
   line in the steps file (step_line: step, loss, learning_rate,
   label_tokens, elapsed_s; a non-finite loss written as NaN, which the
   trainer records as null). No interim checkpoint. Task P17.42 (the
   owner's answer of 2026-10-26 to OQ-106,
   option (a)) adds to each line: step_time_s, the host's high-resolution
   monotonic time (time.perf_counter) from the step's start to after its
   second sync (torch_xla.sync with wait, so the step's graphs have run on
   the card); tokens, the batch's
   non-pad tokens; tokens_computed, batch_size x the sequence the card runs
   (max_length plus any virtual tokens); model_flop, the PaLM model FLOPs
   of those tokens (model_flops); compiles, the graph compiles the step
   added by the backend's compile counter (torch_xla's CompileTime metric;
   None without one); and compile_step, true when it compiled, or, without
   a counter, for the first COMPILE_RULE_STEPS steps by rule. No profiler
   runs in training: the environment the trainer builds names none, and the
   worker has no network.
6. saved: the model back on the CPU, saved with its tokenizer into the
   checkpoint directory in safetensors (the adapter for lora and
   prompt_tuning, the full weights otherwise).

Then result.json: status complete, global_step, train_runtime_s, the
device, the parameter counts, the token counts, model_flops (the
convention, its formula, and its parts; task P17.42), the optimizer's
settings, the framework versions, and notes. A refusal writes status refused and its
message and exits 3 (REFUSED); a failure writes status failed and exits 1
with its traceback. Nothing here is retried; the trainer reads the ending.
"""

from __future__ import annotations

import json
import os
import random
import socket
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

JOB_FORMAT = "lassi-ttxla-job-1"
REFUSED = 3
IGNORE_INDEX = -100
BACKENDS = ("xla", "cpu")
WEIGHTS = ("full", "lora", "prompt_tuning")
# Every key a job holds and its JSON type; the optional ones may also be null.
REQUIRED: dict[str, type | tuple[type, ...]] = {
    "format": str, "backend": str, "base": dict, "weights": str, "steps": int, "seed": int, "batch_size": int,
    "learning_rate": (int, float), "max_length": int, "gradient_checkpointing": bool, "chat_format": bool,
    "source": str, "dtype": str, "records": list, "checkpoint": str, "steps_file": str, "result_file": str,
    "stage_file": str, "network": str,
}
# The one network setting: the worker runs in its own empty network namespace (lassi.train.ttxla_trainer).
NETWORK_NONE = "none"
OPTIONAL: dict[str, type | tuple[type, ...]] = {"lora_r": int, "virtual_tokens": int, "compile_options": dict}
# Task P17.42: without a compile counter, the first steps are flagged as compile steps by rule (the first step
# compiles the forward and backward graphs, the second the optimizer's once its state exists).
COMPILE_RULE_STEPS = 2
# The model FLOP convention every step's model_flop follows (PaLM, arXiv:2204.02311, Appendix B; Kaplan et al. 2020).
PALM_CONVENTION = "palm-model-flops"
PALM_FORMULA = (
    "PaLM model FLOPs per token (arXiv:2204.02311, Appendix B): 6 per trained matmul weight (forward, input gradient, "
    "weight gradient), 4 per frozen matmul weight (no weight gradient), plus 12 x layers x heads x head_dim x sequence "
    "for the attention matmuls, which hold no weights; the matmul weights are those of every Linear module, the output "
    "projection included and the embedding lookup excluded; recompute (gradient checkpointing) is not counted"
)
# The distributions whose versions result.json records.
VERSIONED = ("torch", "torch-xla", "pjrt-plugin-tt", "jax", "transformers", "peft", "accelerate", "tokenizers",
             "safetensors", "numpy")


class Refusal(Exception):
    """A job, record, or host the worker will not train on; it exits REFUSED and touches no card."""


def read_job(path: Path) -> dict[str, Any]:
    """Return the job file's content, checked; Refusal naming the key that is missing or wrong."""
    try:
        job = json.loads(Path(path).read_bytes().decode("ascii"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise Refusal(f"the job file {path} cannot be read as an ASCII JSON job: {error}") from None
    if not isinstance(job, dict) or job.get("format") != JOB_FORMAT:
        raise Refusal(f"the job file {path} is not of the format {JOB_FORMAT}")
    for key, kind in REQUIRED.items():
        if key not in job or not isinstance(job[key], kind) or (kind is int and isinstance(job[key], bool)):
            raise Refusal(f"the job's {key} is missing or not a {getattr(kind, '__name__', kind)}")
    for key, kind in OPTIONAL.items():
        if job.get(key) is not None and not isinstance(job[key], kind):
            raise Refusal(f"the job's {key} is not a {getattr(kind, '__name__', kind)} or null")
    if job["backend"] not in BACKENDS:
        raise Refusal(f"the job's backend {job['backend']!r} is none of {', '.join(BACKENDS)}")
    if job["network"] != NETWORK_NONE:
        raise Refusal(f"the job's network {job['network']!r} is not {NETWORK_NONE!r}; the worker never has a network")
    if job["weights"] not in WEIGHTS or (job["weights"] == "lora") != (job.get("lora_r") is not None):
        raise Refusal(f"the job's weights {job['weights']!r} and lora_r {job.get('lora_r')!r} do not agree")
    if (job["weights"] == "prompt_tuning") != (job.get("virtual_tokens") is not None):
        raise Refusal("the job's virtual_tokens is set exactly when its weights are prompt_tuning")
    if not job["records"] or not all(isinstance(item, dict) for item in job["records"]):
        raise Refusal("the job's records must be a non-empty list of objects")
    return job


def batch_order(count: int, steps: int, batch_size: int, seed: int) -> list[list[int]]:
    """Return `steps` batches of `batch_size` record indices: pass after pass, each a seeded permutation."""
    order: list[list[int]] = []
    pool: list[int] = []
    passes = 0
    while len(order) < steps:
        batch: list[int] = []
        while len(batch) < batch_size:
            if not pool:
                pool = list(range(count))
                random.Random(seed * 1_000_003 + passes).shuffle(pool)
                passes += 1
            batch.append(pool.pop(0))
        order.append(batch)
    return order


def padded(ids: list[int], labels: list[int], length: int, pad_id: int) -> tuple[list[int], list[int], list[int]]:
    """Return (input ids, attention mask, labels) padded on the right to `length`; Refusal for a longer sequence."""
    if len(ids) > length or len(labels) != len(ids):
        raise Refusal(f"a sequence of {len(ids)} tokens does not fit max_length {length}; nothing is cut")
    pad = length - len(ids)
    return ids + [pad_id] * pad, [1] * len(ids) + [0] * pad, labels + [IGNORE_INDEX] * pad


def stage(job: dict[str, Any], name: str) -> None:
    """Write the stage's name to the job's stage file and print it."""
    Path(job["stage_file"]).write_text(name, encoding="ascii")
    print(f"ttxla-worker stage {name}", flush=True)


def check_network(job: dict[str, Any]) -> None:
    """Refuse an xla run that sees any network interface but the loopback: it must be in an empty namespace.

    The wheel's Tracy client listens and broadcasts from library load
    (lassi.train.ttxla_trainer, NETWORK_NAMESPACE), so before any framework
    is imported the worker lists the interfaces it can see.
    """
    if job["backend"] != "xla":
        return
    others = sorted(name for _, name in socket.if_nameindex() if name != "lo")
    if others:
        raise Refusal(f"the worker sees the network interfaces {', '.join(others)}; trainer.network none runs it in "
                      "its own empty network namespace, and it never loads the training stack with a network")


def check_host(job: dict[str, Any]) -> None:
    """Refuse an xla run without its empty network namespace, or on a POSIX host whose core limit is not 1 byte."""
    check_network(job)
    if job["backend"] != "xla" or os.name != "posix":
        return
    import resource

    limit = resource.getrlimit(resource.RLIMIT_CORE)
    if limit != (1, 1):
        raise Refusal(f"RLIMIT_CORE is {limit}, not (1, 1); a crash could store a core on the root filesystem")


def _chat_ids(tokenizer: Any, messages: list[dict[str, str]], prompt: bool) -> list[int]:
    """Return the token ids of `messages` through the chat template; `prompt` adds the assistant turn's start."""
    found = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=prompt, return_dict=True)
    return list(found["input_ids"])


def encode(tokenizer: Any, record: dict[str, str], chat: bool) -> tuple[list[int], list[int], bool]:
    """Return (ids, labels, prefix kept) of one record; the prompt's tokens are masked out of the labels."""
    if chat:
        if not getattr(tokenizer, "chat_template", None):
            raise Refusal("trainer.chat_format is true, but the base's tokenizer has no chat template")
        user = [{"role": "user", "content": record["prompt"]}]
        prompt_ids = _chat_ids(tokenizer, user, True)
        ids = _chat_ids(tokenizer, [*user, {"role": "assistant", "content": record["completion"]}], False)
    else:
        prompt_ids = list(tokenizer(record["prompt"])["input_ids"])
        ids = list(tokenizer(record["prompt"] + record["completion"])["input_ids"])
        if tokenizer.eos_token_id is not None and (not ids or ids[-1] != tokenizer.eos_token_id):
            ids.append(tokenizer.eos_token_id)
    cut = min(len(prompt_ids), len(ids))
    return ids, [IGNORE_INDEX] * cut + ids[cut:], ids[:cut] == prompt_ids


def encode_all(tokenizer: Any, job: dict[str, Any], notes: list[str]) -> list[tuple[list[int], list[int]]]:
    """Encode every record; Refusal for one longer than max_length or with no completion token."""
    encoded = []
    for number, record in enumerate(job["records"], start=1):
        ids, labels, kept = encode(tokenizer, record, job["chat_format"])
        where = f"{job['source']} record {number}"
        if len(ids) > job["max_length"]:
            raise Refusal(f"{where} is {len(ids)} tokens, more than trainer.max_length {job['max_length']}; no "
                          "record is cut: raise trainer.max_length or split the record")
        if all(label == IGNORE_INDEX for label in labels[1:]):
            raise Refusal(f"{where} has no completion token to train on")
        if not kept:
            notes.append(f"{where}: the prompt's tokens are not a prefix of the whole record's; its first "
                         f"{len(ids) - sum(label != IGNORE_INDEX for label in labels)} tokens are masked")
        encoded.append((ids, labels))
    return encoded


def load_tokenizer(transformers: Any, base: dict[str, Any]) -> Any:
    """Return the base's tokenizer from local files only: a fixture's tokenizer.json, or its directory or snapshot."""
    if base["kind"] == "fixture":
        directory = Path(base["path"])
        settings = json.loads((directory / "tokenizer_config.json").read_text(encoding="ascii"))
        return transformers.PreTrainedTokenizerFast(
            tokenizer_file=str(directory / "tokenizer.json"), eos_token=settings["eos_token"],
            pad_token=settings["pad_token"], model_max_length=settings["model_max_length"],
        )
    path = base["tokenizer"] or base["path"]
    return transformers.AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def load_model(transformers: Any, peft: Any, torch: Any, job: dict[str, Any], tokenizer: Any) -> Any:
    """Return the base with its weight mode applied, on the CPU in the job's dtype (the module docstring, step 3)."""
    base = job["base"]
    transformers.set_seed(job["seed"])
    if base["kind"] == "hub":
        model = transformers.AutoModelForCausalLM.from_pretrained(base["path"], local_files_only=True,
                                                                   trust_remote_code=False)
    else:
        config = transformers.AutoConfig.from_pretrained(base["path"], local_files_only=True)
        if len(tokenizer) > config.vocab_size:
            raise Refusal(f"the tokenizer has {len(tokenizer)} entries, more than the config's {config.vocab_size}")
        model = transformers.AutoModelForCausalLM.from_config(config)
    model.config.use_cache = False
    if job["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if job["weights"] == "lora":
        model = peft.get_peft_model(model, peft.LoraConfig(r=job["lora_r"], target_modules="all-linear",
                                                           task_type="CAUSAL_LM"))
    elif job["weights"] == "prompt_tuning":
        model = peft.get_peft_model(model, peft.PromptTuningConfig(
            num_virtual_tokens=job["virtual_tokens"], task_type="CAUSAL_LM", prompt_tuning_init="RANDOM"))
    return model.to(getattr(torch, job["dtype"]))


def palm_flop_per_token(frozen: int, trained: int, layers: int, heads: int, head_dim: int, sequence: int) -> int:
    """Return PaLM's model FLOPs per token (PALM_FORMULA): 6 per trained weight, 4 per frozen one, 12LHQT."""
    return 6 * trained + 4 * frozen + 12 * layers * heads * head_dim * sequence


def matmul_weights(model: Any, linear: type) -> tuple[int, int]:
    """Return (frozen, trained) element counts of the weights of every `linear` module in `model`, each counted once."""
    seen: set[int] = set()
    frozen = trained = 0
    for module in model.modules():
        weight = getattr(module, "weight", None) if isinstance(module, linear) else None
        if weight is None or id(weight) in seen:
            continue
        seen.add(id(weight))
        if weight.requires_grad:
            trained += weight.numel()
        else:
            frozen += weight.numel()
    return frozen, trained


def attention_shape(config: Any) -> tuple[int, int, int]:
    """Return (layers, heads, head_dim) from a transformers config; head_dim is hidden_size // heads when unset."""
    layers, heads = int(config.num_hidden_layers), int(config.num_attention_heads)
    head_dim = getattr(config, "head_dim", None) or int(config.hidden_size) // heads
    return layers, heads, int(head_dim)


def model_flops(model: Any, config: Any, job: dict[str, Any], linear: type) -> dict[str, Any]:
    """Return result.json's model_flops: the convention, its formula, its parts, and the FLOPs per token run."""
    sequence = job["max_length"] + (job.get("virtual_tokens") or 0)
    frozen, trained = matmul_weights(model, linear)
    layers, heads, head_dim = attention_shape(config)
    return {
        "convention": PALM_CONVENTION, "formula": PALM_FORMULA, "frozen_weights": frozen, "trained_weights": trained,
        "layers": layers, "heads": heads, "head_dim": head_dim, "sequence": sequence,
        "flop_per_token": palm_flop_per_token(frozen, trained, layers, heads, head_dim, sequence),
    }


def compile_counter(job: dict[str, Any]) -> Callable[[], int | None]:
    """Return a function giving the backend's graph compiles so far: 0 on the CPU, None when it cannot be read."""
    if job["backend"] != "xla":
        return lambda: 0
    try:
        import torch_xla.debug.metrics as metrics
    except ImportError:
        return lambda: None

    def count() -> int | None:
        try:
            data = metrics.metric_data("CompileTime")
        except Exception:
            return None
        return 0 if data is None else int(data[0])

    return count


def step_line(
    *, step: int, loss: float, learning_rate: float, label_tokens: int, elapsed_s: float, step_time_s: float,
    tokens: int, tokens_computed: int, flop_per_token: int, compiles: int | None,
) -> dict[str, Any]:
    """Return one steps.jsonl line (the module docstring, step 5); compile_step by count, else by rule."""
    compile_step = compiles > 0 if compiles is not None else step <= COMPILE_RULE_STEPS
    return {
        "step": step, "loss": loss, "learning_rate": learning_rate, "label_tokens": label_tokens,
        "elapsed_s": elapsed_s, "step_time_s": step_time_s, "tokens": tokens, "tokens_computed": tokens_computed,
        "model_flop": flop_per_token * tokens_computed, "compiles": compiles, "compile_step": compile_step,
    }


def device_backend(torch: Any, job: dict[str, Any]) -> tuple[Any, Any]:
    """Return (device, sync) of the job's backend: the TT device through torch_xla, or the CPU."""
    if job["backend"] == "cpu":
        return torch.device("cpu"), (lambda: None)
    import torch_xla
    import torch_xla.runtime as xr

    xr.set_device_type("TT")
    return torch_xla.device(), (lambda: torch_xla.sync(wait=True))


def train_steps(torch: Any, job: dict[str, Any], model: Any, runner: Any, batches: list[Any], device: Any,
                sync: Any, flop_per_token: int) -> tuple[Any, float]:
    """Run the job's steps (the module docstring, step 5); return the optimizer and the seconds they took."""
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=job["learning_rate"])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    virtual = job.get("virtual_tokens") or 0
    count = compile_counter(job)
    model.train()
    start = time.monotonic()
    with open(job["steps_file"], "w", encoding="ascii") as steps_out:
        for step, (ids, mask, labels) in enumerate(batches, start=1):
            begin, before = time.perf_counter(), count()
            optimizer.zero_grad()
            logits = runner(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            if virtual:
                logits = logits[:, virtual:, :]
            shifted = logits[:, :-1, :].contiguous()
            target = labels.to(device)[:, 1:].contiguous()
            loss = loss_fn(shifted.view(-1, shifted.size(-1)), target.view(-1))
            loss.backward()
            sync()
            optimizer.step()
            sync()
            seconds, after = time.perf_counter() - begin, count()
            line = step_line(
                step=step, loss=float(loss.item()), learning_rate=job["learning_rate"],
                label_tokens=int((labels[:, 1:] != IGNORE_INDEX).sum()), elapsed_s=round(time.monotonic() - start, 3),
                step_time_s=round(seconds, 6), tokens=int(mask.sum()),
                tokens_computed=int(ids.numel()) + virtual * int(ids.shape[0]), flop_per_token=flop_per_token,
                compiles=None if before is None or after is None else after - before,
            )
            steps_out.write(json.dumps(line) + "\n")
            steps_out.flush()
            stage(job, f"step {step}")
    return optimizer, time.monotonic() - start


def batch_tensors(
    torch: Any, job: dict[str, Any], encoded: list[tuple[list[int], list[int]]], pad_id: int
) -> list[Any]:
    """Return each step's (input ids, attention mask, labels) tensors, padded to max_length, in batch_order."""
    batches = []
    for indices in batch_order(len(encoded), job["steps"], job["batch_size"], job["seed"]):
        rows = [padded(encoded[index][0], encoded[index][1], job["max_length"], pad_id) for index in indices]
        batches.append(tuple(torch.tensor([row[part] for row in rows], dtype=torch.long) for part in range(3)))
    return batches


def versions() -> dict[str, str | None]:
    """Return the installed version of each VERSIONED distribution, None when it is not installed."""
    import importlib.metadata

    found: dict[str, str | None] = {}
    for name in VERSIONED:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def run(job: dict[str, Any]) -> dict[str, Any]:
    """Train as the module docstring says and return result.json's content."""
    check_host(job)
    stage(job, "checked")
    import peft
    import torch
    import transformers

    notes: list[str] = []
    tokenizer = load_tokenizer(transformers, job["base"])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise Refusal("the base's tokenizer has neither a pad token nor an end token to pad with")
    encoded = encode_all(tokenizer, job, notes)
    stage(job, "encoded")
    model = load_model(transformers, peft, torch, job, tokenizer)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    flops = model_flops(model, base.config, job, torch.nn.Linear)
    batches = batch_tensors(torch, job, encoded, pad_id)
    stage(job, "loaded")
    stage(job, "device")
    device, sync = device_backend(torch, job)
    model.to(device)
    options = job.get("compile_options")
    runner = torch.compile(model, backend="tt", options=options) if job["backend"] == "xla" else model
    stage(job, "training")
    optimizer, seconds = train_steps(torch, job, model, runner, batches, device, sync, flops["flop_per_token"])
    stage(job, "saving")
    model.to("cpu")
    model.save_pretrained(job["checkpoint"], safe_serialization=True)
    tokenizer.save_pretrained(job["checkpoint"])
    stage(job, "saved")
    lengths = [len(ids) for ids, _ in encoded]
    return {
        "status": "complete", "message": None, "stage": "saved", "global_step": job["steps"],
        "train_runtime_s": round(seconds, 3), "device": str(device), "backend": job["backend"],
        "python": sys.version.split()[0],
        "parameters": {"total": sum(p.numel() for p in model.parameters()),
                       "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)},
        "tokens": {"max_record": max(lengths), "records": lengths,
                   "label_tokens": [sum(label != IGNORE_INDEX for label in labels) for _, labels in encoded]},
        "model_flops": flops,
        "optimizer": {"name": "torch.optim.AdamW", **{key: _plain(value) for key, value in optimizer.defaults.items()
                                                      if key in ("lr", "betas", "eps", "weight_decay")}},
        "versions": versions(), "notes": notes,
    }


def _plain(value: Any) -> Any:
    """Return an optimizer setting as plain JSON: a tuple as a list, a tensor or number as a float."""
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return float(value)


def write_result(job: dict[str, Any], result: dict[str, Any]) -> None:
    """Write result.json beside the job."""
    Path(job["result_file"]).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="ascii")


def _last_stage(job: dict[str, Any]) -> str:
    """Return the last stage written, or "not started"."""
    try:
        return Path(job["stage_file"]).read_text(encoding="ascii").strip() or "not started"
    except OSError:
        return "not started"


def main(argv: list[str]) -> int:
    """Run the worker on argv's one job file; return 0, REFUSED (3), or 2 for a bad invocation."""
    if len(argv) != 1:
        print("usage: python -I ttxla_worker.py <job.json>", file=sys.stderr)
        return 2
    try:
        job = read_job(Path(argv[0]))
    except Refusal as error:
        print(f"ttxla worker: refused: {error}", file=sys.stderr)
        return REFUSED
    try:
        result = run(job)
    except Refusal as error:
        write_result(job, {"status": "refused", "message": str(error), "stage": _last_stage(job)})
        print(f"ttxla worker: refused: {error}", file=sys.stderr)
        return REFUSED
    except Exception as error:
        write_result(job, {"status": "failed", "message": f"{type(error).__name__}: {error}",
                           "stage": _last_stage(job)})
        traceback.print_exc()
        return 1
    write_result(job, result)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
