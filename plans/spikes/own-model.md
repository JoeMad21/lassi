# Spike: a model of our own, trained from scratch

- Request: the owner's OQ-076 answer of 2026-10-23, in the owner's words: "Full model training, frozen weights, LORA, and potentially training our own model as far reaching goal." This note covers the last of the four: what a first own model would be, and what it needs.
- Status: planning note. Nothing here was run as a training or evaluation run. Every compute, time, and size figure is PROJECTED from the arithmetic shown. The parameter counts, token counts, tokenizer size, and Hub header readings are workstation readings on 2026-10-23 and 2026-10-24 from a dirty tree, logged with their commands and outputs under Readings below; they are exploratory, never [MEASURED], and every recipe that uses one labels it PROJECTED.
- Constraints taken from the session (2026-10-23): training runs on alpha01's CPU, in the owner's window only, as a niced background job (the owner's direction that replaced the RTX 3080 plan); downloads into HF_HOME happen only as a separate prepare command in that window (tools/prepare_train.py); trained models are served through hf_local on alpha01's CPU (OQ-076 (3a)); a base meant for a later RNGD build uses an architecture furiosa-llm supports (Readings, item 6).

## What exists now: the from-scratch baseline

`lassi train projects/train/tt-scratch-qwen3-77m.yaml` builds the model of assets/train/scratch/qwen3-77m/config.json with random weights under the trainer's seed (lassi.train.bases, `base_model: scratch:<name>`) and trains it on tt-train-v0 with full weights.

- Architecture: Qwen3ForCausalLM, hidden 384, MLP 1,024, 12 layers, 6 query heads of 64 and 2 key-value heads, 4,096 positions, tied embeddings. Parameters: 77,228,928 (Readings, item 3; the config's own arithmetic gives the same). The embedding is 151,936 x 384 = 58.3M of them, three quarters, because the baseline borrows Qwen3's vocabulary (tokenizer-pin.json pins the tokenizer, Qwen/Qwen3-0.6B at c1899de; 151,669 entries, Readings, item 4).
- Data: tt-train-v0, 8 records, 13,027 tokens a pass in the chat format (Readings, item 5). A compute-optimal budget for 77M parameters is about 20 tokens a parameter, 1.5B tokens (Hoffmann et al. 2022, the "Chinchilla" ratio), so the baseline sees about one part in 119,000 of what its size calls for. PROJECTED outcome: it memorizes the 8 programs and passes no Tier A item; it checks that the from-scratch path (config, seeded initialization, pinned tokenizer, checkpoint pinning, hf_local serving) works end to end, and it is a floor for later comparisons.
- Compute: 640 steps (80 passes), about 5.4e14 FLOP; 45 minutes at an assumed 0.2 TFLOP/s on alpha01's CPU (PROJECTED; the recipe's comments hold the arithmetic).

## A first real own model

### Size

About 100M parameters, small enough to train on one accelerator in hours and to serve on a CPU in hf_local:

- a code tokenizer of 32,768 entries (below), so the embedding is 32,768 x 768 = 25.2M rather than three quarters of the model;
- Qwen3ForCausalLM or LlamaForCausalLM with hidden 768, 12 layers, 12 query heads of 64 and 4 key-value heads, MLP 2,048: per layer 768 x 768 x 2 + 768 x 256 x 2 + 3 x 768 x 2,048 = 6.29M, 75.5M for 12 layers, 100.7M with the tied embedding (PROJECTED arithmetic; a config would be written and counted on the meta device before any run).

### Data

A compute-optimal pass for about 100M parameters is about 2B tokens (PROJECTED), more than 150,000 times tt-train-v0. Candidate sources, each needing the owner's decision on licenses and each decontaminated under Agent Rule 5 by the same rules tools/build_tt_train.py applies (Tier A example directories excluded, task duplicates excluded, no 3-line run of any eval or unassigned item's tracked reference, no byte-identical or near-copy file of an eval or unassigned item's upstream files), extended to every suite's held items, not only tt-pairs-v0's:

1. Permissively licensed public code in the target languages (C, C++, MLIR, Python, Fortran, Rust, C#), for example the deduplicated subsets of The Stack v2 by license; the bulk of the 2B tokens.
2. The pinned trees LASSI already builds: tt-metal (examples, tests, kernels, documentation), tt-mlir, LLVM and MLIR (including their test suites, which are MLIR in custom assembly form), each at its toolchain pin; tens of millions of tokens (not counted).
3. LASSI's own trajectories once P7 harvests them (train split only, Agent Rule 5).

Storage (PROJECTED): about 4 bytes a token of raw text, 8 GB for 2B tokens, plus 4 GB of uint16 token ids for a 32K vocabulary: within the 150G cap on alpha01's scratch root but a real share of it, so the corpus needs its own space check and a backup decision.

### Tokenizer

Two choices, each with a cost:

- Borrow Qwen3's 151,936-entry tokenizer, as the baseline does: no tokenizer work, and the chat template hf_local needs comes with it, but at hidden 768 the embedding is 116.7M (PROJECTED arithmetic), more than the rest of a 100M model.
- Train a byte-level BPE of 32,768 entries on the corpus with the tokenizers library, with the ChatML special tokens and a chat template written here: fewer tokens a program for code, and parameters spent on layers; it becomes a pinned artifact of its own (its training corpus hash, its files' sha256), and furiosa-llm would compile the model with it.

Recommendation: train the tokenizer, after a spike that compares its tokens per tt-train-v0 record against Qwen3's (13,027 a pass, Readings, item 5).

### Compute

Training FLOPs are about 6 x N x D: 6 x 100.7M x 2.0B = 1.2e18 FLOP (PROJECTED).

| Host | Assumed sustained rate (PROJECTED) | Time (PROJECTED) | Status |
| --- | --- | --- | --- |
| alpha01's CPU, one process | 0.5 TFLOP/s | about 28 days | the only host training may use today; impractical for this size |
| one MI300X | 40 percent of its bf16 dense peak, about 0.5 PFLOP/s | about 40 minutes | no access for now (OQ-040, OQ-002) |
| AMD Developer Cloud, one MI300X | as above | about 40 minutes of the roughly 50 hours of credit | named in Risks And Questions; needs the owner's action and a second gate |
| the workstation's RTX 3080 | about 12 TFLOP/s | about 28 hours | the owner set it aside on 2026-10-23 |

The peak figures are vendor specification numbers, not read or measured in this session; the rates are assumptions, and the first run's TRL train_runtime measures the real one. On alpha01's CPU the same rate buys a model of about 10M parameters on about 200M tokens: 6 x 10M x 200M = 1.2e16 FLOP, about 7 hours at 0.5 TFLOP/s (PROJECTED), too small to write tt-metal host programs.

### What it needs

1. A GPU host or cloud credit (OQ-040 reopened, or the AMD Developer Cloud credit behind a second gate): the decisive need. Without one, the own model stays at the 77M baseline.
2. The owner's decision on the data sources and their licenses, and on whether the corpus may hold public code that Tier B or later suites might draw on (decontamination must cover every suite, so a later suite's held items are checked against a frozen corpus before it is used).
3. A corpus tool on the pattern of tools/build_tt_train.py (pinned sources, a tracked manifest with every source's sha256, generated records outside git, the Agent Rule 5 checks as tests), and a `text` record format for pretraining beside sft's prompt and completion; the trl trainer has no pretraining method today (methods sft, dpo, grpo), so pretraining or continued pretraining on raw text is a new method behind the Trainer interface (bible Training Module, Algorithms: full weights for continued pretraining).
4. A tokenizer spike and a pinned tokenizer artifact (above).
5. An evaluation plan: perplexity on a decontaminated held-out slice, then Tier A pass rates through hf_local like every other configuration, labeled exploratory until a clean-commit run.

## Next steps

1. Run the four projects/train recipes in the owner's window on alpha01's CPU (plans/runs/train-window/run.sh; the prepare step first), and record TRL's train_runtime for each: it replaces the assumed rates above with measured ones.
2. Serve each checkpoint through hf_local on Tier A and record the scores, the scratch baseline included.
3. Ask the owner (owner queue) for the GPU host or credit and the data decision; until then, the own model is the baseline.

## Readings

Workstation readings, Windows 10, Git Bash, from a dirty tree on branch p16-adversarial; exploratory, never [MEASURED] (Agent Rule 1). The helper scripts ran from the session's scratch directory and are not tracked; their logic is given in full.

1. Hub revisions (2026-10-23). `curl -sS https://huggingface.co/api/models/Qwen/<name>` returned `"sha"`: Qwen3-0.6B `c1899de289a04d12100db370d81485cdf75e47ca`, Qwen3-1.7B `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, both `"architectures": ["Qwen3ForCausalLM"]`. `curl -sSI https://huggingface.co/Qwen/<name>/resolve/<sha>/config.json` returned `HTTP/1.1 307` with `X-Repo-Commit: c1899de289a04d12100db370d81485cdf75e47ca` and `X-Repo-Commit: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, each the requested revision.
2. Download sizes (2026-10-23), from `curl -sSI` of each file at its revision: Qwen3-0.6B model.safetensors `X-Linked-Size: 1503300328`; Qwen3-1.7B model-00001-of-00002.safetensors `X-Linked-Size: 3441185608` and model-00002-of-00002.safetensors `X-Linked-Size: 622329984`; Qwen3-0.6B tokenizer.json 11,422,654 bytes as downloaded (`ls -la`).
3. Parameter counts (2026-10-23), with the cpu extra installed for the reading and removed after it (`uv sync`): `uv run --extra cpu python -c` building each model on `torch.device("meta")` with `AutoModelForCausalLM.from_config` (the scratch config) or `Qwen3ForCausalLM(Qwen3Config(**config.json))` (the two Hub configs, fetched at their revisions) and summing `p.numel()` printed `Qwen3ForCausalLM 77228928`, `Qwen3-0.6B 596049920`, `Qwen3-1.7B 1720574976`. No weight was made or loaded.
4. Tokenizer size (2026-10-23): `len(PreTrainedTokenizerFast(tokenizer_file=<Qwen3-0.6B tokenizer.json at c1899de>))` printed `tokenizer entries 151669`.
5. Token counts (2026-10-24, after the audit's rebuild to 8 records, records sha256 86bc79319ffbe8a77620ef4bc76e43bcbdbacaaf6a6150e30623a2f7e7db8518): `uv run --no-project --with tokenizers==0.22.1 python -I count_tokens.py <tokenizer.json> <records>`, which encodes each record as `<|im_start|>user\n` + prompt + `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` + completion + `<|im_end|>\n` (the Qwen3 chat template's rendering of one turn each) with `Tokenizer.from_file` and counts the ids, printed `chat 2791, 781, 885, 1244, 1466, 1475, 1327, 3058` and `total chat tokens 13027, longest 3058`. The token-weighted mean causal position, the sum of squares over twice the total, is 1,004.
6. furiosa-llm's documentation, v2026.1.0, read on 2026-10-24 through a web fetch of each page:
   - developer.furiosa.ai/v2026.1.0/en/overview/supported_models.html lists for text generation LlamaForCausalLM, Exaone4ForCausalLM, Qwen2ForCausalLM, and Qwen3ForCausalLM (with Qwen3Model and Qwen3ForSequenceClassification for pooling), and does not mention LoRA.
   - developer.furiosa.ai/v2026.1.0/en/furiosa_llm/model-preparation.html lists no architectures and mentions neither LoRA nor prompt tuning.
   - developer.furiosa.ai/v2026.1.0/en/furiosa_llm/reference/artifact_builder.html: ArtifactBuilder's model argument is `model_id_or_path: str | Path`, "The HuggingFace model id or a local path. This corresponds to pretrained_model_name_or_path in HuggingFace Transformers." The page does not mention LoRA, adapters, PEFT, merging, or prompt embeddings.
   - developer.furiosa.ai/v2026.1.0/en/furiosa_llm/furiosa-llm-serve.html: the OpenAI-compatible server "hosts a single model" with /v1/completions, /v1/chat/completions, /v1/embeddings (input a string or an array), score, rerank, models, version, and metrics; no endpoint takes a LoRA adapter or prompt embeddings, and neither is mentioned.
   - developer.furiosa.ai/v2026.1.0/en/furiosa_llm/reference/llm.html: the LLM class's generate() and chat() take text prompts, generate() optionally prompt_token_ids; no input takes embedding vectors, and LoRA, adapters, and prompt embeddings are not mentioned.
   - Inferred from those pages, not stated by them: with no documented adapter input, a LoRA checkpoint reaches RNGD only merged into full weights and saved as a local model directory for ArtifactBuilder; and with only text and token-id inputs, a soft prompt (input embeddings) cannot be served by furiosa-llm, so a prompt-tuned model is evaluated through hf_local only. A later RNGD spike checks both on the cards.
