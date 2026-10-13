"""Tests for the committed training fixtures: the smoke data and the tiny base model (task P17.9).

Bible: Training Module, Agent Rules 5 and 10; plans/p17-portable.md, task
P17.9 ("a tiny model built from a config and synthetic data under
tests/fixtures/train/"); tests/fixtures/train/README.md.

The contract these tests fix:

- sft-smoke.jsonl, dpo-smoke.jsonl, and grpo-smoke.jsonl load through
  lassi.train.data load_synthetic from the default fixture directory, four
  records each, with exactly the method's fields: sft {prompt, completion},
  dpo {prompt, chosen, rejected}, grpo {prompt, target}. Every value is a
  string, except that grpo's target may be null; every prompt is marked
  SYNTHETIC; grpo-smoke holds exactly one null target and its other targets
  are the lowercase alphabet. Each prompt and completion stays far below the
  model's 256 positions (one token per byte).
- tiny-causal-lm/ holds config.json, tokenizer.json, and
  tokenizer_config.json, each plain ASCII JSON with LF only and a final LF.
  The tokenizer is the README's rule: a BPE model with no merges whose
  vocabulary is <|pad|> 0, <|endoftext|> 1, and the 256 ByteLevel symbols
  in byte order at ids 2 to 257 (the ByteLevel table, rebuilt here in
  pure Python), both special tokens marked special, a ByteLevel
  pre-tokenizer and decoder without a prefix space. config.json is a tiny
  causal LM (hidden size 16, 2 layers) whose vocab_size, eos_token_id,
  pad_token_id, and max_position_embeddings agree with the tokenizer files.
- The fixture directory lies outside assets/bench, and no smoke text names a
  bench item (Agent Rule 5).

These tests import no framework. Every text is SYNTHETIC; no value in this
module is a measurement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from train_fakes import REPO
from trl_smoke import BASE_DIR, BASE_FILES, FIELDS, FIXTURES, SYNTHETIC_DIR, TARGET

from lassi.bench import load_suite
from lassi.train import data as train_data

PAD, END = "<|pad|>", "<|endoftext|>"
CONTEXT = 256
# The grpo smoke's longest completion (trainer.max_completion_length in the smoke recipe).
MAX_COMPLETION = 8


def byte_symbols() -> list[str]:
    """Return the ByteLevel symbol of each byte 0 to 255, in byte order (the bytes_to_unicode table)."""
    printable = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    codes = list(printable)
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codes.append(256 + extra)
            extra += 1
    table = dict(zip(printable, codes, strict=True))
    return [chr(table[byte]) for byte in range(256)]


def read_json(path: Path) -> Any:
    """Return a committed JSON fixture after checking it is ASCII with LF only and a final LF."""
    raw = path.read_bytes()
    assert raw.isascii(), f"{path.name} is not plain ASCII"
    assert b"\r" not in raw and raw.endswith(b"\n"), f"{path.name} is not LF-only with a final LF"
    return json.loads(raw.decode("ascii"))


# ---------------------------------------------------------------------------
# The smoke data


@pytest.mark.parametrize("method", list(FIXTURES))
def test_each_smoke_fixture_loads_with_the_methods_fields(method: str) -> None:
    data = train_data.load_synthetic(FIXTURES[method], train_data.SYNTHETIC_DIR)
    records = [dict(record) for record in data.records]
    assert len(records) == 4
    for number, record in enumerate(records, start=1):
        assert tuple(record) == FIELDS[method], (number, record)
        for key, value in record.items():
            nullable = method == "grpo" and key == "target"
            assert isinstance(value, str) or (nullable and value is None), (number, key, value)
        assert "SYNTHETIC" in record["prompt"], number


def test_the_grpo_smoke_holds_exactly_one_null_target() -> None:
    data = train_data.load_synthetic(FIXTURES["grpo"], train_data.SYNTHETIC_DIR)
    targets = [record["target"] for record in data.records]
    assert targets.count(None) == 1, targets
    assert [target for target in targets if target is not None] == [TARGET] * 3


@pytest.mark.parametrize("method", list(FIXTURES))
def test_the_smoke_texts_stay_far_below_the_context(method: str) -> None:
    raw = (SYNTHETIC_DIR / FIXTURES[method]).read_bytes()
    for line in raw.decode("ascii").splitlines():
        record = json.loads(line)
        prompt = len(record["prompt"].encode("ascii"))
        answers = [len((record[key] or "").encode("ascii")) for key in FIELDS[method][1:]]
        longest = MAX_COMPLETION if method == "grpo" else max(answers)
        assert prompt + longest + 2 <= CONTEXT // 2, record


def test_no_smoke_text_names_a_bench_item() -> None:
    names = set()
    for manifest in sorted((REPO / "assets" / "bench").glob("*.yaml")):
        names.update(load_suite(manifest).items)
    assert names, "no suite manifest was read"
    text = " ".join((SYNTHETIC_DIR / name).read_bytes().decode("ascii") for name in FIXTURES.values()).lower()
    found = sorted(name for name in names if len(name) > 3 and name.lower() in text)
    assert not found, f"the smoke fixtures name bench items: {found}"
    assert REPO / "assets" / "bench" not in [SYNTHETIC_DIR, *SYNTHETIC_DIR.parents]


# ---------------------------------------------------------------------------
# The tiny base model


def test_the_base_model_directory_holds_the_three_files() -> None:
    assert BASE_DIR.parent == SYNTHETIC_DIR
    assert sorted(item.name for item in BASE_DIR.iterdir()) == sorted(BASE_FILES)
    for name in BASE_FILES:
        read_json(BASE_DIR / name)


def test_the_tokenizer_is_the_byte_level_rule_with_no_merges() -> None:
    tokenizer = read_json(BASE_DIR / "tokenizer.json")
    model = tokenizer["model"]
    assert model["type"] == "BPE"
    assert model["merges"] == []
    expected = {PAD: 0, END: 1, **{symbol: 2 + byte for byte, symbol in enumerate(byte_symbols())}}
    assert model["vocab"] == expected
    assert len(model["vocab"]) == 258 and sorted(model["vocab"].values()) == list(range(258))
    specials = {entry["content"]: (entry["id"], entry["special"]) for entry in tokenizer["added_tokens"]}
    assert specials == {PAD: (0, True), END: (1, True)}
    for part in ("pre_tokenizer", "decoder"):
        assert tokenizer[part]["type"] == "ByteLevel", part
    assert tokenizer["pre_tokenizer"]["add_prefix_space"] is False


def test_the_tokenizer_config_names_the_fast_class_and_the_special_tokens() -> None:
    config = read_json(BASE_DIR / "tokenizer_config.json")
    assert config["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert (config["eos_token"], config["pad_token"], config["model_max_length"]) == (END, PAD, CONTEXT)


def test_the_model_config_is_tiny_and_agrees_with_the_tokenizer() -> None:
    config = read_json(BASE_DIR / "config.json")
    vocab = read_json(BASE_DIR / "tokenizer.json")["model"]["vocab"]
    assert config["vocab_size"] == len(vocab) == 258
    assert (config["eos_token_id"], config["pad_token_id"]) == (vocab[END], vocab[PAD]) == (1, 0)
    assert config["max_position_embeddings"] == CONTEXT
    sizes = {key: config[key] for key in (
        "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
    )}
    assert sizes == {
        "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
        "num_key_value_heads": 1,
    }
    assert config["tie_word_embeddings"] is False
    assert isinstance(config["model_type"], str) and config["model_type"]
