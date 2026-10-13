"""A tiny chat model and tokenizer, built in memory, for the hf_local tests (task P17.4); nothing is downloaded.

Bible: Model Serving (Serving Rules), Agent Rules 7 and 10;
plans/p17-portable.md, task P17.4 ("tested on the CPU with a tiny model built
in the test from a config (no download)").

The helpers here import torch, tokenizers, and transformers inside their
functions only, so a test module that needs the framework extra imports this
module freely and skips with NEEDS_EXTRA when the framework is absent
(`pytest.importorskip("torch", reason=NEEDS_EXTRA)`).

- tiny_tokenizer() trains a byte-level BPE tokenizer in memory on a few
  SYNTHETIC code lines, with three special tokens and a ChatML-like chat
  template (CHAT_TEMPLATE). A byte-level tokenizer is needed: Transformers
  reloads a saved tokenizer as the class the model type names, and a
  word-level one did not survive that reload (P17.4 design, section 11).
- build_snapshot() saves a randomly initialized Qwen2-architecture model of
  two layers and that tokenizer into a fake HF_HOME hub cache, at
  `hub/models--<org>--<name>/snapshots/<revision>/`, the layout a named fetch
  step writes, and then overwrites generation_config.json with sampling
  defaults (TRAP_DEFAULTS by default) that hf_local must never let reach
  generate().
- reference_decode() is an independent decoding loop (one full forward pass
  per step, argmax at temperature 0, else one multinomial draw from the
  softmax of logits / temperature) that the backend's outputs are compared
  against.

Every weight is random and every text SYNTHETIC; no value made here is a
measurement.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# The skip reason of every test that needs the framework extra (bible Toolchain Pins: run them with --extra).
NEEDS_EXTRA = "needs a framework extra: uv run --extra cpu pytest ..."

REPO_ID = "lassi-test/tiny-chat"
# SYNTHETIC full commit ids (40 lowercase hexadecimal characters); no repository has them.
REVISION = "0123456789abcdef0123456789abcdef01234567"
OTHER_REVISION = "89abcdef0123456789abcdef0123456789abcdef"

END_TOKEN = "<|im_end|>"
PAD_TOKEN = "<|endoftext|>"
START_TOKEN = "<|im_start|>"
SPECIAL_TOKENS = (PAD_TOKEN, START_TOKEN, END_TOKEN)
VOCAB_SIZE = 300

# A ChatML-like template: each message as <|im_start|>role\ncontent<|im_end|>\n, then the assistant's opening.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

# SYNTHETIC lines the tokenizer is trained on; plain ASCII.
CORPUS = (
    "int main() { return 0; }",
    "#pragma omp parallel for",
    "__global__ void k() { }",
    "system user assistant",
    "translate the program below",
)

# Sampling defaults in the checkpoint's generation_config.json that would change greedy and sampled output if any
# of them reached generate(): the trap the P17.4 design (section 7) names.
TRAP_DEFAULTS: Mapping[str, Any] = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 2,
    "repetition_penalty": 1.5,
    "no_repeat_ngram_size": 1,
    "min_p": 0.3,
}


def repo_dir(hf_home: Path, repo_id: str = REPO_ID) -> Path:
    """Return the hub cache directory of `repo_id` under `hf_home`: hub/models--<org>--<name>."""
    return Path(hf_home) / "hub" / ("models--" + repo_id.replace("/", "--"))


def snapshot_dir(hf_home: Path, repo_id: str = REPO_ID, revision: str = REVISION) -> Path:
    """Return the snapshot directory of `repo_id` at `revision` under `hf_home`."""
    return repo_dir(hf_home, repo_id) / "snapshots" / revision


def tiny_tokenizer(chat_template: str | None = CHAT_TEMPLATE) -> Any:
    """Return a byte-level BPE PreTrainedTokenizerFast trained in memory on CORPUS, with `chat_template` (or none)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    core = Tokenizer(models.BPE())
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    core.train_from_iterator(list(CORPUS), trainer)
    options: dict[str, Any] = {"tokenizer_object": core, "eos_token": END_TOKEN, "pad_token": PAD_TOKEN}
    if chat_template is not None:
        options["chat_template"] = chat_template
    return PreTrainedTokenizerFast(**options)


def build_snapshot(
    hf_home: Path,
    repo_id: str = REPO_ID,
    revision: str = REVISION,
    *,
    context: int = 64,
    generation_defaults: Mapping[str, Any] | None = TRAP_DEFAULTS,
    end_ids: Sequence[int] | None = None,
    chat_template: str | None = CHAT_TEMPLATE,
    seed: int = 0,
) -> Path:
    """Save a tiny model and tokenizer as the snapshot of `repo_id` at `revision` under `hf_home`; return its path.

    The model is a Qwen2-architecture causal LM (2 layers, hidden size 16)
    with max_position_embeddings `context`, initialized under
    torch.manual_seed(`seed`). generation_config.json is then written as
    bytes with `generation_defaults` and eos_token_id `end_ids` (default:
    the tokenizer's end token). refs/main names `revision`, as a hub
    download leaves it, so a loader that ignored the revision would read the
    last snapshot built.
    """
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    tokenizer = tiny_tokenizer(chat_template)
    end = tokenizer.convert_tokens_to_ids(END_TOKEN)
    pad = tokenizer.convert_tokens_to_ids(PAD_TOKEN)
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=context,
        eos_token_id=end,
        pad_token_id=pad,
    )
    model = Qwen2ForCausalLM(config)
    target = snapshot_dir(hf_home, repo_id, revision)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    write_generation_config(target, generation_defaults, [end] if end_ids is None else list(end_ids))
    refs = repo_dir(hf_home, repo_id) / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_bytes(revision.encode("ascii"))
    return target


def write_generation_config(snapshot: Path, defaults: Mapping[str, Any] | None, end_ids: Sequence[int]) -> None:
    """Write the snapshot's generation_config.json as ASCII bytes: `defaults` plus eos_token_id `end_ids`."""
    data: dict[str, Any] = dict(defaults or {})
    data["eos_token_id"] = list(end_ids) if len(end_ids) != 1 else end_ids[0]
    (Path(snapshot) / "generation_config.json").write_bytes(json.dumps(data, sort_keys=True).encode("ascii"))


def load_reference(snapshot: Path) -> tuple[Any, Any]:
    """Return (tokenizer, model) loaded straight from the snapshot directory, independently of hf_local."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True, dtype="auto")
    return tokenizer, model.eval()


def chat_ids(tokenizer: Any, messages: Sequence[tuple[str, str]]) -> list[int]:
    """Return the prompt ids the tokenizer's chat template gives for (role, content) pairs, generation prompt on."""
    encoded = tokenizer.apply_chat_template(
        [{"role": role, "content": content} for role, content in messages],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    )
    ids = encoded["input_ids"]
    return [int(item) for item in (ids[0] if ids and isinstance(ids[0], list) else ids)]


def reference_decode(
    model: Any, input_ids: Sequence[int], *, temperature: float, seed: int, max_new: int, stops: Sequence[int]
) -> list[int]:
    """Return the ids a plain decoding loop appends to `input_ids`, an end id included when one stops it.

    torch.manual_seed(`seed`) first; then, for at most `max_new` steps, one
    full forward pass over the ids so far, and the next id is the argmax of
    the last logits at temperature 0, else one torch.multinomial draw from
    softmax(logits / temperature). No other sampling setting applies.
    """
    import torch

    torch.manual_seed(seed)
    ids = torch.tensor([list(input_ids)], dtype=torch.long)
    found: list[int] = []
    with torch.inference_mode():
        for _ in range(max_new):
            logits = model(input_ids=ids).logits[:, -1, :].float()
            if temperature == 0:
                chosen = logits.argmax(dim=-1, keepdim=True)
            else:
                chosen = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)
            found.append(int(chosen[0, 0]))
            ids = torch.cat([ids, chosen], dim=1)
            if found[-1] in stops:
                break
    return found
