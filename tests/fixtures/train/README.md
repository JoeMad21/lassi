# Training fixtures (tasks P17.8 and P17.9)

Every file here is SYNTHETIC and lies outside every bench suite: no text is
taken from a bench item, so no eval or unassigned item can reach training
(Agent Rule 5). `lassi train` reads `data.synthetic` from this directory
(lassi.train.data SYNTHETIC_DIR), as plain ASCII JSON Lines split on LF.
Nothing here is a measurement.

## Data

- `layer-smoke.jsonl`: two `{prompt, completion}` records the train layer
  tests read (task P17.8).
- `sft-smoke.jsonl`: four `{prompt, completion}` records for the sft smoke
  run (task P17.9).
- `dpo-smoke.jsonl`: four `{prompt, chosen, rejected}` records for the dpo
  smoke run.
- `grpo-smoke.jsonl`: four `{prompt, target}` records for the grpo smoke
  run. Three targets are the lowercase alphabet; one is null, the stand-in
  for a reward that cannot be scored, which the trainer skips and never
  reads as 0. The fixture reward (lassi.train.fixture_reward) scores a
  completion by the fraction of its characters that occur in the target; it
  reads text only and runs nothing (Agent Rule 6).

Each text is short, so a prompt and its completion stay far below the
model's 256 positions.

## The base model: `fixture:tiny-causal-lm`

`tiny-causal-lm/` holds what the trl trainer builds its base model from, with
random weights and nothing downloaded:

- `config.json`: a causal LM config of model_type qwen2 (the architecture
  tests/tiny_hf.py builds): vocab_size 258, hidden_size 16,
  intermediate_size 32, 2 hidden layers, 2 attention heads, 1 key-value head,
  max_position_embeddings 256, eos_token_id 1, pad_token_id 0, untied
  embeddings.
- `tokenizer.json`: a byte-level BPE tokenizer with no merges, fully
  determined by this rule: id 0 is the special token `<|pad|>`, id 1 the
  special token `<|endoftext|>`, and ids 2 to 257 are the 256 ByteLevel
  alphabet symbols in byte order: byte b has id 2 + b, and its symbol is the
  ByteLevel table's (the bytes 0x21 to 0x7e, 0xa1 to 0xac, and 0xae to 0xff
  map to the code point of the same value, and the other 68 bytes, in
  order, to U+0100 onward). The pre-tokenizer and the decoder are ByteLevel
  without a prefix space. The file is written with JSON ASCII escapes, so
  the symbols above 0x7f appear as `\uXXXX` and the file is plain ASCII. It
  needs no training corpus.
- `tokenizer_config.json`: the tokenizer class PreTrainedTokenizerFast, eos
  token `<|endoftext|>`, pad token `<|pad|>`, model_max_length 256.

A run pins this base by content: its checkpoint records the sha256 of each
of the three files and the trainer's seed, which together give the initial
weights.
