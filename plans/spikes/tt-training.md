# Spike: training small language models on alpha01's Wormhole n300 cards at the tt-metal pin

- Owner direction (2026-10-26, the owner's words, from the session brief): "I would like to be able to train the
  smaller models by then, plan to use the tenstorrents for that task." "By then" is the owner window of Tuesday
  2026-10-27, 04:00 to 12:00 EDT. Earlier: OQ-076 ("I want Full model training, frozen weights, LORA, and potentially
  training our own model as far reaching goal"; then "don't train on the 3080. Just train if possible during our
  testing window.").
- Bible: Training Module (Weight Modes, Training Data, Compute, Safeguards), Agent Rules 5, 6, 7, 9, 10, 13, 17;
  Execution Backends (the tt_silicon row); Toolchain Pins.
- Status: sources only. No rx command of any kind ran, nothing touched alpha01, and no device command ran anywhere.
  Upstream trees, CI records, package indexes, and Hugging Face model files were read from the workstation
  (Windows, Git Bash) on 2026-10-26. One wheel was downloaded into a session scratch directory and read as a zip
  archive; nothing from it was executed.
- Pins: tt-metal 5280a9cfb00998fd49667a29523d03aee905c129 (toolchains/tt-metal.pin; built with TARGETS tt_metal
  hw_toolchain, WITH_PYTHON_BINDINGS OFF, TT-NN configured but not built), UMD 0450f1be, ttsim v1.3.4, tt-mlir
  5f396fd6 (toolchains/tt-mlir.pin, 0.9.0.dev20260221). Host facts from earlier spikes: eight Wormhole n300 cards,
  tenstorrent driver 2.8.0, firmware bundle 19.5.0.0, 32 1 GiB hugepages, IOMMU domain identity
  (plans/spikes/tt-silicon-executor.md, plans/spikes/tt-card-dma.md).
- Citations:
  - `metal <file>:<line>`: tenstorrent/tt-metal at 5280a9cf; `train <file>:<line>` is `metal tt-train/<file>`.
  - `xla <file>:<line>`: tenstorrent/tt-xla at ff3a67dde32ad3311258674bd3fa260e11a20e98 (the commit the wheel below
    names).
  - `blacksmith <file>:<line>`: tenstorrent/tt-blacksmith at 002ebe47570399626581046a0c653cade7e78203
    (2026-02-24T16:53:08Z).
  - `wheel`: pjrt_plugin_tt-0.9.0.dev20260223001148-cp311-cp311-linux_x86_64.whl from
    https://pypi.eng.aws.tenstorrent.com/pjrt-plugin-tt/.
  - Repository files at the working tree of branch p9-ee.
- Conventions: upstream code is paraphrased, never quoted, as the sibling spikes do. Each finding says whether it was
  read or inferred. Every number not read from a source is PROJECTED arithmetic. Upstream CI outcomes and upstream
  documentation figures are upstream readings, never LASSI measurements, and nothing here is a LASSI measurement.

## Question

Can LASSI train Qwen2.5-0.5B, Qwen3-0.6B, and Qwen3-1.7B (and the 77M scratch model) on alpha01's Wormhole n300 cards,
at the project's tt-metal pin, in the owner window of 2026-10-27?

1. Which training stacks exist for Wormhole at or near the pin, what each needs built, and whether each runs on one
   n300 and on several.
2. Which of the three bases each stack can train, at what precision, whether training state fits one n300, and what
   is unknown.
3. The smallest path to a first training run in the window: what to build first, what data, what LASSI code is
   missing, and what can be tested on Windows with fakes.
4. The risks on alpha01.

Why it matters: the owner asked for it by name for the next window. The tasks that depend on it are the
Tenstorrent training tasks still to be planned (a Trainer backend, a device kind, the pin, the recipes), the matrix's
training combinations (OQ-076; P17.19's four recipes now run only on alpha01's CPU), and P7's training plan.

## Classification

- Questions 1 and 2 are factual: the pinned sources, the upstream packages, and their CI records answer them, apart
  from what only a run on the cards can show (marked unknown).
- Question 3 is factual for what exists and what is missing; which stack to use is the owner's choice (OQ-101 draft).
- Question 4 is factual for the risks; whether training may run on silicon at all is the owner's choice, because
  Agent Rule 9 as written allows only evaluation runs there (OQ-100 draft), and the host steps before the window and
  the disk are access questions (OQ-102 draft).

## Plain answer

- Agent Rule 9 says "silicon runs are evaluation runs a human starts". A training run is not an evaluation run, and
  the bible's Training Module puts training on alpha01's CPU only. Training on a card therefore needs the owner's
  decision and a rule change before the window (OQ-100 draft). This is the first thing the owner needs to know.
- A stack built at exactly the pin exists without any source build: the tt-xla wheel pjrt-plugin-tt
  0.9.0.dev20260223001148, whose metadata names tt-metal 5280a9cf, bundles TT-NN built at that commit and trains
  PyTorch models through PyTorch/XLA (read). It runs the Hugging Face Qwen2 and Qwen3 code as written, so all three
  bases load correctly in it (inferred from how it ingests models; not run). Upstream trained Qwen with LoRA on one
  Wormhole chip through it, at 128 tokens (read: tt-blacksmith's recipes at that wheel and the report in its pull
  request 464; not a LASSI run).
- tt-train (the pin's own C++ training framework) needs TT-NN, the Python bindings, and tt-train built, which the
  pinned install does not have. Its Llama-style model cannot load any of the three Qwen bases correctly (read), so
  it serves the own-model goal (a Llama-architecture model trained from scratch), not the Qwen bases, without new
  model code.
- What is likely in the window, if OQ-100 allows it: LoRA on Qwen2.5-0.5B, Qwen3-0.6B, and Qwen3-1.7B, and full
  training of the 77M scratch model, on one chip of one card (PROJECTED). Full training of Qwen3-1.7B does not fit one
  chip (about 13.8 GB of state against 12 GiB, PROJECTED; tt-xla's own test list records it as out of DRAM, read).
  Full training of the 0.5B and 0.6B bases fits on paper only with recompute and is marked a known failure in
  tt-xla's own training tests at that commit (read). tt-train-v0's longest record is 3,058 tokens and no record is
  cut; nothing upstream trained Qwen at that length through tt-xla, so a first run on tt-train-v0 may fail to compile
  or run out of memory (unknown).

## Commands and outputs

All ran on the workstation on 2026-10-26, from the session's scratch directory. Outputs are trimmed where marked.

1. The pinned tree's file list:

   ```
   gh api "repos/tenstorrent/tt-metal/git/trees/5280a9cfb00998fd49667a29523d03aee905c129?recursive=1"
   truncated False, 17701 entries, 803 under tt-train/
   ```

   Then 634 text files under tt-train/ (every blob but .png, .ipynb, and .safetensors) were fetched with
   `curl -sSfL https://raw.githubusercontent.com/tenstorrent/tt-metal/5280a9cf.../tt-train/<path>`, and the root
   CMakeLists.txt, build_metal.sh, create_venv.sh, cmake/project_options.cmake, tt_metal/python_env/requirements-dev.txt,
   tt_metal/jit_build/build.cpp, and the 150 files of .github/workflows the same way.

2. Upstream CI at the pin (the PR Gate run that plans/spikes/p4-tt-pins.md cites):

   ```
   gh api "repos/tenstorrent/tt-metal/actions/runs/22095847892/jobs?per_page=100"
   build / Build Release ubuntu 22.04 | 2026-02-17T11:01:38Z | 2026-02-17T11:07:52Z | success
   build-tt-train | 2026-02-17T11:07:54Z | 2026-02-17T11:12:09Z | success
   [... trimmed ...]
   ```

   The release build uses a shared ccache (metal .github/workflows/build-artifact.yaml:352-468), so these times say
   nothing about a cold build on alpha01.

3. tt-xla's tt-mlir pin around the nightly:

   ```
   gh api "repos/tenstorrent/tt-xla/commits?path=third_party/CMakeLists.txt&since=2026-02-15T00:00:00Z&until=2026-02-26T00:00:00Z"
   04826b15d2 2026-02-23T17:08:16Z Upgrade to python 3.12 (#3122)
   8add68dfa9 2026-02-20T20:23:05Z Uplift third_party/tt-mlir to c7b8bd7784307cce46577bca1c2d33fdab31fccc 2026-02-20 (#3387)
   [... two earlier uplifts trimmed ...]
   TT_MLIR_VERSION at 8add68df: c7b8bd7784307cce46577bca1c2d33fdab31fccc
   TT_MLIR_VERSION at 04826b15: 96090c423381c82f0cfa298f86e794430db09a8a
   gh api "repos/tenstorrent/tt-mlir/compare/c7b8bd77...5f396fd6"
   ahead ahead_by=7 behind_by=0
   ```

   So LASSI's tt-mlir pin is 7 commits after the wheel's tt-mlir; plans/spikes/p4-tt-pins.md already read that both
   commits pin tt-metal 5280a9cf.

4. The package index and the wheel:

   ```
   curl -sS https://pypi.eng.aws.tenstorrent.com/pjrt-plugin-tt/
   pjrt_plugin_tt-0.9.0.dev20260221001203-cp311-cp311-linux_x86_64.whl
   pjrt_plugin_tt-0.9.0.dev20260222001106-cp311-cp311-linux_x86_64.whl
   pjrt_plugin_tt-0.9.0.dev20260223001148-cp311-cp311-linux_x86_64.whl#sha256=fbbe6241869e9646633604b213903b9795f9dc2f9192f875a569799689b0c4db
   pjrt_plugin_tt-0.9.0.dev20260224001247-cp312-cp312-linux_x86_64.whl
   curl -sSI .../pjrt_plugin_tt-0.9.0.dev20260223001148-cp311-cp311-linux_x86_64.whl
   Content-Length: 121492577
   Last-Modified: Mon, 23 Feb 2026 00:12:24 GMT
   sha256sum pjrt.whl
   fbbe6241869e9646633604b213903b9795f9dc2f9192f875a569799689b0c4db
   uv run --no-project python -I <scripts>/whl_inspect.py <dl-wheel>/pjrt.whl   (zipfile only)
   members 6660, uncompressed bytes 453606668
   Version: 0.9.0.dev20260223001148
   Summary: commit=ff3a67dde32ad3311258674bd3fa260e11a20e98, tt-mlir-commit=c7b8bd7784307cce46577bca1c2d33fdab31fccc, tt-metal-commit=5280a9cfb00998fd49667a29523d03aee905c129, built-date=2026-02-21, build-type=release
   Requires-Python: >=3.11, <3.12
   Requires-Dist: jax==0.7.1, jaxlib==0.7.1, torch @ .../torch-2.9.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl,
     torch-xla @ https://pypi.eng.aws.tenstorrent.com/torch-xla/torch_xla-2.9.0%2Bgit061c1e7-cp311-cp311-linux_x86_64.whl,
     [... requests, loguru, networkx, graphviz, seaborn, pyyaml, click, pandas trimmed ...]
   286792400 pjrt_plugin_tt/lib/libTTMLIRCompiler.so
   38636240 pjrt_plugin_tt/lib/_ttnncpp.so
   14214624 pjrt_plugin_tt/lib/libtt_metal.so
   6365568 pjrt_plugin_tt/lib/_ttnn.so
   3704624 pjrt_plugin_tt/lib/libdevice.so
   [... trimmed; tt-metal/ttnn 27261736 and tt-metal/tt_metal 20877789 bytes of kernel sources ...]
   pjrt_plugin_tt/tt-metal/runtime/sfpi-version.cmake: SFPI_version "7.25.0", SFPI_hash 6a8883c4...
   (no sfpi toolchain in the wheel)
   pjrt_plugin_tt/__init__.py:53-102: sets TT_METAL_RUNTIME_ROOT to the wheel's tt-metal directory unless the user set it
   curl -sSI .../torch_xla-2.9.0%2Bgit061c1e7-cp311-cp311-linux_x86_64.whl  -> Content-Length: 107174326
   curl -sSI https://download.pytorch.org/whl/cpu/torch-2.9.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl -> Content-Length: 184462550
   ```

   The wheel's last-modified time, 00:12:24 on 2026-02-23, is before tt-xla's tt-mlir change at 17:08:16 that day,
   which agrees with the metadata.

5. tt-blacksmith, the upstream training recipes for tt-xla:

   ```
   gh api "repos/tenstorrent/tt-blacksmith/commits?until=2026-02-25T00:00:00Z&per_page=3"
   002ebe4757 2026-02-24T16:53:08Z [TT-XLA][PyTorch] Training Workload for Qwen 3 4B Instruct 2507 (#464)
   bd82953d2c 2026-02-23T07:59:09Z Uplift tt-xla to 0.9.0.dev20260223001148 2026-02-23 (#463)
   9fda494f1d 2026-02-21T07:38:02Z Uplift tt-xla to 0.9.0.dev20260221001203 2026-02-21 (#461)
   blacksmith env/xla_requirements.txt: pjrt-plugin-tt==0.9.0.dev20260223001148, transformers==4.57.1, peft (unpinned), datasets, ...
   PR 463 check runs (head e0662dbc):
   uplift-test / test 'single_chip and uplift and n150 and torch' (runs-on: n150-high-memory) | success | 07:02:48Z | 07:58:56Z
   basic-test / test 'single_chip and push and n300 and torch' (runs-on: n300) | success
   [... jax and data/tensor-parallel jobs, all success, trimmed ...]
   PR 464 (merged 2026-02-24T16:53:08Z): basic-test jobs success; uplift-test skipped; body: "Model: Qwen 3 4B Instruct
   2507", "Strategy: LoRA", "Framework: PyTorch", "Hardware: N150 / Blackhole QuietBox"
   ```

6. tt-xla's own training tests at the wheel's commit:

   ```
   curl .../tt-xla/ff3a67dd.../tests/runner/test_config/torch/test_config_training_single_device.yaml
   Counter({'KNOWN_FAILURE_XFAIL': 249, 'NOT_SUPPORTED_SKIP': 168, 'EXPECTED_PASSING': 1, 'EXCLUDE_MODEL': 1})
   :42  qwen_2_5/causal_lm/pytorch-0.5B-single_device-training: KNOWN_FAILURE_XFAIL
   :950 qwen_3/causal_lm/pytorch-0_6B-single_device-training: KNOWN_FAILURE_XFAIL
   :1498 qwen_3/causal_lm/pytorch-1_7B-single_device-training: NOT_SUPPORTED_SKIP, FAILED_RUNTIME,
         "Out of Memory: Not enough space to allocate 1048576000 B DRAM buffer across 12 banks"
   :1256 (another model) "... bank size is only 1073741792 B"
   [... trimmed ...]
   ```

7. Hugging Face, the three bases (configs and safetensors headers by HTTP range request):

   ```
   curl -sS https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B  -> sha 060db6499f32faf8b98477b0a26969ef7d8b9987
   Qwen2.5-0.5B config: Qwen2ForCausalLM, hidden 896, intermediate 4864, 24 layers, 14 heads, 2 KV heads, vocab 151936, tied, rope_theta 1e6, rms_norm_eps 1e-6
   Qwen3-0.6B at c1899de2: Qwen3ForCausalLM, hidden 1024, intermediate 3072, 28 layers, 16 heads, 8 KV heads, head_dim 128, vocab 151936, tied
   Qwen3-1.7B at 70d244cc: Qwen3ForCausalLM, hidden 2048, intermediate 6144, 28 layers, 16 heads, 8 KV heads, head_dim 128, vocab 151936, tied
   Qwen2.5-0.5B model.safetensors header: 290 tensors, 494032768 elements, BF16; layer 0 holds self_attn.q_proj.bias, k_proj.bias, v_proj.bias
   Qwen3-0.6B model.safetensors header: layer 0 holds self_attn.q_norm.weight [128], k_norm.weight [128], q_proj.weight [2048, 1024]; lm_head.weight stored beside embed_tokens
   ```

8. The other frontends:

   ```
   gh api "repos/tenstorrent/tt-torch/commits?per_page=3"
   f5e3955f8a 2025-11-14T03:08:29Z Aknezevic/ultralytics (#1245)
   c621c04130 2025-10-06T13:31:42Z Disable scheduled jobs (uplift, nightly, weekly) (#1244)
   blacksmith env/ffe_requirements.txt: tt-forge-fe 0.4.0.dev20250923, tt-tvm 0.4.0.dev20250923
   ```

## Sources

Read on 2026-10-26.

- https://github.com/tenstorrent/tt-metal/tree/5280a9cfb00998fd49667a29523d03aee905c129: tt-train/ (README.md,
  INSTALLING.md, docs/INSTALLING_TTML.md, docs/TTML_ONBOARDING.md, docs/MEMORY_TRACKING.md,
  docs/DISTRIBUTED_TRAINING.md, configs/, sources/ttml/, sources/examples/, tests/, tools/weight-utils/), CMakeLists.txt,
  cmake/project_options.cmake, build_metal.sh, create_venv.sh, tt_metal/jit_build/build.cpp,
  tt_metal/python_env/requirements-dev.txt, .github/workflows/ (pr-gate.yaml, build-artifact.yaml,
  all-post-commit-workflows.yaml, tt-train-post-commit.yaml). The CI run
  https://github.com/tenstorrent/tt-metal/actions/runs/22095847892.
- https://github.com/tenstorrent/tt-xla at ff3a67dd and 8add68df and 04826b15: third_party/CMakeLists.txt,
  tests/runner/test_config/torch/test_config_training_single_device.yaml, tests/runner/test_utils.py,
  tests/runner/test_models.py, tests/torch/training/test_basic.py, docs/src/getting_started.md.
- https://pypi.eng.aws.tenstorrent.com/pjrt-plugin-tt/ and /torch-xla/ (the index pages and HEAD requests), and the
  wheel itself, read as a zip archive.
- https://github.com/tenstorrent/tt-blacksmith at 002ebe47: env/xla_requirements.txt, env/ffe_requirements.txt,
  blacksmith/experiments/torch/qwen/ (README.md, configs.py, single_chip/*.yaml, test_qwen_finetuning.py),
  tests/training_test_cases.py, tests/test_all.py, blacksmith/tools/cli.py, .github/workflows/test-matrix-presets/;
  pull requests https://github.com/tenstorrent/tt-blacksmith/pull/463 and /pull/464 and their check runs.
- https://github.com/tenstorrent/tt-forge (README, the frontends and the training example), and
  https://github.com/tenstorrent/tt-torch (commit history).
- https://docs.tenstorrent.com/docs-test/core/latest/_sources/aibs/wormhole/specifications.md.txt (n150s: 1 ASIC,
  12GB GDDR6; n300s: 2 ASICs, 128 Tensix cores, 64 per ASIC, 24GB GDDR6, 131 TFLOPs FP16, 300W).
- https://huggingface.co/Qwen/Qwen2.5-0.5B at 060db649, Qwen/Qwen3-0.6B at c1899de2, Qwen/Qwen3-1.7B at 70d244cc:
  config.json and the model.safetensors headers.
- Repository: lassi/train/ (bases.py, trl_trainer.py, run.py, data.py, checkpoint.py), lassi/core/interfaces.py
  (Trainer), lassi/core/record.py (DEVICE_KINDS), lassi/core/devices.py, lassi/executors/tt_silicon.py, lassi/cli.py,
  pyproject.toml, assets/train/tt-train-v0/, tools/build_tt_train.py, projects/train/, assets/train/scratch/qwen3-77m/,
  toolchains/tt-metal.pin, toolchains/tt-mlir.pin, results/p4-tt-install/summary.md, plans/spikes/p4-tt-pins.md,
  plans/spikes/tt-silicon-executor.md, plans/spikes/tt-card-dma.md, plans/spikes/own-model.md, plans/OWNER-QUEUE.md
  (OQ-070 to OQ-097), plans/LESSONS.md, docs/BIBLE.md.

## Findings

### 1. Training stacks at or near the pin

#### 1a. tt-train (ttml), in the pinned tree

- What it is (read): a C++ autograd framework over TT-NN with a nanobind Python module, `ttml` (train README.md;
  docs/TTML_ONBOARDING.md). Optimizers AdamW, MorehAdamW, a fused AdamW, an AdamW with fp32 master weights and moments
  (sources/ttml/optimizers/adamw_full_precision.hpp), SGD; DDP, tensor parallel (FSDP-style), and pipeline parallel
  over a device mesh, one process per host.
- Models (read): GPT-2, a Llama-style transformer (RMSNorm, RoPE with optional NTK scaling, grouped-query attention,
  SwiGLU MLP), an MLP, and linear regression (sources/ttml/models/). Configs ship for nanoGPT, GPT-2 S to XL,
  TinyLlama, Llama 3.2 1B, Llama 3 8B and 70B (configs/model_configs/).
- Hugging Face weights (read): the Llama model loads `model.safetensors` files from a directory and maps the Hugging
  Face Llama names onto its own (sources/ttml/models/llama.cpp:325-693): embeddings, norms, q_proj, k_proj and
  v_proj (concatenated into one kv_linear), o_proj, gate_proj, up_proj, down_proj, with the q and k rows permuted for
  its interleaved RoPE (:551, :569). A tensor it does not know is skipped and listed in a printed note
  (:641-642). The gsm8k example at the pin fine-tunes GPT-2 weights fetched with hf_hub_download, though its README
  names TinyLlama (sources/examples/gsm8k_finetune/gsm8k_finetune.py:368, :412; README.md).
- Full fine-tuning (read): yes, the normal path (the examples and tests).
- Frozen layers (read): Tensor.set_requires_grad is exposed to Python (sources/ttml/nanobind/nb_autograd.cpp:81), and
  AdamW skips a tensor that does not require a gradient (sources/ttml/optimizers/adamw.cpp:34, :134). Freezing chosen
  layers is a few lines of Python (inferred). Prompt tuning (a soft prompt) is not provided (read: no such module).
- LoRA (read): a LoRALinearLayer exists (sources/ttml/modules/lora_linear_module.cpp, .hpp), but no model, example,
  test, or Python binding uses it: of the 634 text files fetched, only CMakeLists.txt:43 names it. Wiring it into the
  Llama blocks is new code (inferred).
- Memory tools (read): a memory-efficient runner that recomputes each block in backward (models/llama.cpp:238-240),
  fused cross-entropy forward and backward kernels (sources/ttml/ops/losses.cpp:72, :96), and a fused flash-style SDPA
  kernel, though the Llama attention calls the composite SDPA that keeps the full attention matrix
  (sources/ttml/modules/grouped_query_attention.cpp:50; sources/ttml/ops/scaled_dot_product_attention.cpp:157-188,
  :244-300).
- What it needs built (read): tt-metal configured with BUILD_TT_TRAIN ON (cmake/project_options.cmake:16, off by
  default; CMakeLists.txt:311) and, for the Python module, WITH_PYTHON_BINDINGS ON (:5): the ttml library links
  TT::Metalium and TTNN::TTNN (_ttnncpp.so) and Python3::Python (sources/ttml/CMakeLists.txt:160, :206, :297-298),
  and the ttml package imports the ttnn Python module first (sources/ttml/ttml/__init__.py:13). The pinned install
  builds neither TT-NN nor the bindings, so tt-train needs a new build of TT-NN, its Python module, and tt-train.
  Its configure step fetches its own CPM packages (Boost 1.86.0 by hash, yaml-cpp, googletest, nanobind 2.10.2,
  xtensor, flatbuffers, and others; cmake/dependencies.cmake) and downloads two tokenizer files from Hugging Face at
  `main`, unpinned, into the source tree's data/ (sources/ttml/CMakeLists.txt:396-426), which a LASSI build would
  have to pre-place and record (Agent Rule 10; inferred). The tt-metal venv at the pin installs torch 2.7.1 CPU and
  transformers 4.53.0 among many development packages (tt_metal/python_env/requirements-dev.txt:35, :105).
- Build time and space: not stated upstream for a cold build. The upstream CI times above use a shared ccache. The
  pinned install's own build (676 Ninja steps, tt_metal and hw_toolchain and ten examples) took 3 minutes 44 seconds
  in all at 32 jobs and added 3.0 GiB (results/p4-tt-install/summary.md). At the pin TT-NN has about 1,118 host .cpp
  files outside kernel and test directories, its nanobind layer 249 more, and tt-train's ttml 140, against about 238
  for tt_metal's host library (counted from the tree listing). PROJECTED: 30 to 90 minutes at 32 jobs and 5 to 10 GB
  more under $LASSI_TOOLCHAINS (the build tree, the TT-NN products, and a Python venv with torch); the TurboQuant
  checkout's build_Release with TT-NN and bindings was 1.4 GB (plans/spikes/p4-tt-pins.md, Disk estimate), a
  different commit.
- Devices (read): tt-train's C++ tests run in upstream post-commit CI on wormhole_b0 N300 runners
  (.github/workflows/all-post-commit-workflows.yaml:263-273; tt-train-post-commit.yaml). The docs give mesh [1, 1]
  for one device (N150) and [1, 2] for an N300's two chips (docs/TTML_ONBOARDING.md:894-895), and an n300 DDP config
  ships (configs/training_configs/training_shakespeare_nanogpt_ddp_n300.yaml). Several cards need the fabric and a
  mesh graph descriptor that matches the hardware (docs/DISTRIBUTED_TRAINING.md:36-44, :295-316), and a run stopped
  with Ctrl+C may leave the fabric needing a card reset with the SMI tool (:319-327).

#### 1b. TT-NN with PyTorch, without tt-train

- Read: TT-NN at the pin is an op library with Python bindings and torch interop; it has backward ops (eltwise
  unary, binary, and ternary backward, embedding_backward, and the moreh_* backward ops for linear, matmul, softmax,
  layer norm, NLL loss) and optimizer ops (moreh_adam, moreh_adamw, moreh_sgd) under ttnn/cpp/ttnn/operations/, but
  no autograd engine. tt-train is the autograd built on it.
- Inferred: training through TT-NN alone means writing every backward by hand. Not a path for this window.

#### 1c. tt-xla (pjrt-plugin-tt) with PyTorch/XLA, and tt-blacksmith

- Read: tt-xla is a PJRT plugin; PyTorch models reach it through torch-xla and JAX models through jit, as StableHLO
  for tt-mlir (the wheel's README). tt-torch is deprecated in its favor.
- The wheel at the pin (read, Command 4): 0.9.0.dev20260223001148, cp311 only, 121,492,577 bytes, sha256
  fbbe6241...; its metadata names tt-xla ff3a67dd, tt-mlir c7b8bd77, and tt-metal 5280a9cf, LASSI's pin. It bundles
  tt-metal and TT-NN built at that commit (libtt_metal.so, libdevice.so, _ttnncpp.so, _ttnn.so) and the kernel
  sources, so it needs no source build. It requires Python 3.11 exactly, torch 2.9.0+cpu, torch-xla
  2.9.0+git061c1e7 (Tenstorrent's build, 107,174,326 bytes), and jax 0.7.1. The nightlies of 2026-02-21 and
  2026-02-22 carry the same tt-mlir (Command 3; inferred from the dates).
- sfpi (read): the wheel carries no RISC-V toolchain, only its version file (sfpi 7.25.0). The pinned tt-metal looks
  for sfpi under <runtime root>/runtime/sfpi, then /opt/tenstorrent/sfpi (metal tt_metal/jit_build/build.cpp:140-154),
  and the wheel sets the runtime root to its own directory unless TT_METAL_RUNTIME_ROOT is set (wheel
  pjrt_plugin_tt/__init__.py:53-102). On alpha01 /opt/tenstorrent/sfpi is the root-owned 7.48.0 package
  (plans/spikes/p4-tt-pins.md), so the wheel would build kernels with 7.48.0 unless the pinned sfpi 7.25.0 is put in
  front (inferred). Two ways: a link from the venv's pjrt_plugin_tt/tt-metal/runtime/sfpi to the pinned install's
  runtime/sfpi, or TT_METAL_RUNTIME_ROOT set to the pinned install, which is the same commit (both inferred; the
  first keeps the wheel's own layout).
- Training at that version (read): tt-xla's push suite holds test_qwen3_backward, a full backward pass of Qwen3-0.6B
  on one device at batch 1 with a short input, checking only that the loss is positive and a gradient exists
  (xla tests/torch/training/test_basic.py:19-55); its dual-chip twin is an expected failure, the loss coming out as
  -1.98e38 (:57-62, tt-xla issue 3069). tt-xla's model-test list for single-device training holds 419 entries, of
  which 1 is expected to pass (an autoencoder), 249 are known failures (the test runs and is then marked xfail; xla
  tests/runner/test_utils.py:98-106, :530-532), and 168 are skipped (Command 6): Qwen2.5-0.5B and Qwen3-0.6B are known
  failures, Qwen3-1.7B, Gemma 2B, and others are out of DRAM, and one Gemma entry hangs (:1503-1517).
- Upstream recipes (read): tt-blacksmith pinned this wheel on 2026-02-23 (Command 5). Its PyTorch Qwen experiment is
  a plain training loop: a Hugging Face model in bf16 on the XLA device, PEFT LoRA (r 4, alpha 8, q_proj and v_proj),
  torch AdamW, max_length 128, batch 8 (blacksmith experiments/torch/qwen/test_qwen_finetuning.py:92-188;
  single_chip/test_qwen_finetuning.yaml for Qwen2.5-0.5B, test_qwen_1-5b_finetuning.yaml for Qwen2.5-1.5B). Its
  configs know only lora and adapters (configs.py:21). PR 464's author reports Qwen3-4B-Instruct LoRA training on an
  N150 (with result plots), and the environment file at its merge pins that wheel; that the author ran exactly that
  wheel is inferred. The uplift job that holds the Qwen case passed on an n150 runner for PR 463, but the Qwen case
  names a config path that does not exist at that commit (tests/training_test_cases.py:126-138 against the tree), and
  the loader asserts the file exists (blacksmith/tools/cli.py:13), so whether that case ran is not established; the
  job logs are past retention (inferred from the dates).
- What it needs built: nothing from source. A venv with uv-managed CPython 3.11 and the wheels above, under
  $LASSI_TOOLCHAINS. PROJECTED: about 0.42 GB of the three big wheels plus their dependencies to download, 2.5 to
  4 GB installed.
- Devices (read): upstream runs the torch single-chip jobs on n150 and n300 runners and the data- and
  tensor-parallel jobs on n300 runners (Command 5; blacksmith .github/workflows/test-matrix-presets/). Unknown: which
  chips the PJRT client opens on a host with eight n300 cards, and whether TT_VISIBLE_DEVICES, which UMD reads
  (plans/spikes/tt-silicon-executor.md, Finding 1), limits it to one card.

#### 1d. Other frontends

- tt-torch (read): deprecated; its scheduled jobs were disabled on 2025-10-06 and its last commit is from 2025-11-14,
  so no tt-torch build pins tt-metal 5280a9cf (inferred).
- TT-Forge-FE, now TT-Forge ONNX (read): TVM-based, single chip only; tt-blacksmith's ffe environment at 002ebe47
  pinned tt-forge-fe 0.4.0.dev20250923, five months before the pin, so a different tt-metal (inferred).
- Summary:

| Stack | Pin match | Needs built | Qwen2.5-0.5B | Qwen3-0.6B | Qwen3-1.7B | Full / frozen / LoRA | n300 single / multi |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tt-train (ttml) | the pinned tree itself | TT-NN, bindings, tt-train (PROJECTED 30 to 90 min, 5 to 10 GB) | loads wrong (biases skipped) | refused (q width) | loads wrong (q/k norms skipped) | yes / yes (requires_grad) / layer exists, unwired | CI on N300; [1, 2] DDP config; cards need fabric |
| tt-xla wheel 0.9.0.dev20260223001148 | tt-metal 5280a9cf by its metadata; tt-mlir 7 commits before LASSI's | nothing (a Python 3.11 venv) | loads as written; LoRA recipe upstream at 128 tokens; full training a known failure | loads as written; full backward in push CI; full training a known failure | loads as written; full training out of DRAM | full (unproven for Qwen) / PEFT prompt tuning (untested) / PEFT LoRA (upstream recipes) | single chip in CI; dual chip known failure |
| TT-NN alone | the pinned tree | TT-NN, bindings | no autograd | no autograd | no autograd | hand-written backward only | n/a |
| tt-torch, TT-Forge ONNX | no | n/a | n/a | n/a | n/a | n/a | n/a |

### 2. The three bases per stack, precision, and memory

- Architectures (read, Command 7): Qwen2.5-0.5B is Qwen2ForCausalLM with biases on q, k, and v; Qwen3-0.6B and
  Qwen3-1.7B are Qwen3ForCausalLM with an RMSNorm on each head's q and k and head_dim 128 set apart from
  hidden / heads. All use RMSNorm (eps 1e-6), RoPE (theta 1e6), grouped-query attention, SwiGLU, a 151,936-entry
  vocabulary, and tied embeddings.
- tt-train (read): head_dim is fixed at embedding_dim / num_heads (models/llama.cpp:160) and the query projection is
  embedding_dim wide (modules/grouped_query_attention.cpp:20-21), the Llama block builds attention without biases
  (modules/llama_block.cpp:60), and RMSNorm defaults to eps 1e-5 (modules/rms_norm_module.hpp:17, :23). So:
  - Qwen2.5-0.5B: the shapes fit (14 x 64 = 896; kv 2 x 2 x 64 = 256 rows), but its 72 bias tensors are skipped with
    a note (llama.cpp:641-642), so the loaded model is not Qwen2.5 (inferred: wrong outputs, not an error).
  - Qwen3-0.6B: q_proj is 2,048 x 1,024 while tt-train's query projection is 1,024 x 1,024, so the loader throws a
    shape mismatch (llama.cpp:363).
  - Qwen3-1.7B: the shapes fit (16 x 128 = 2,048), but the q and k norms are skipped, so the loaded model is not
    Qwen3 (inferred).
  - A Qwen model written in Python from ttml's exposed ops (linear, rmsnorm, rope, grouped_heads_creation,
    scaled_dot_product_attention, embedding, silu, cross_entropy_loss; sources/ttml/nanobind/nb_ops.cpp:149-369) is
    possible but new, untested code (inferred).
  - The own-model goal is native: assets/train/scratch/qwen3-77m without its q and k norms is a Llama-style model of
    77,227,392 parameters (77,228,928 less 12 layers x 2 x 64, PROJECTED arithmetic), with every dimension a multiple
    of 32 as tt-train requires (llama.cpp:123-135) and the Qwen3 tokenizer's vocabulary. It is no longer
    Qwen3ForCausalLM, so serving it needs a converter to LlamaForCausalLM safetensors that undoes the RoPE permutation
    and splits kv_linear; tt-train only reads safetensors (sources/ttml/serialization/safetensors.cpp) and its
    weight-utils convert Hugging Face to tt-train, not back (tools/weight-utils/README.md) (read; the converter is
    inferred work).
- tt-xla: the Hugging Face model code runs unchanged, so biases, q/k norms, and head_dim are as Qwen wrote them
  (inferred from the ingestion path; read that upstream loads Qwen3 through it, Finding 1c).
- Precision: both stacks keep weights and activations in bf16 on the device (tt-train's MEMORY_TRACKING.md:152 and
  its fp32 optimizer variant; blacksmith's dtype torch.bfloat16). tt-train offers fp32 master weights and moments
  (AdamWFullPrecision); a PyTorch AdamW on bf16 parameters keeps bf16 moments (inferred).
- DRAM (read): an n300 has two Wormhole chips and 24 GB GDDR6 (Tenstorrent's specification page); tt-xla's OOM
  messages show 12 DRAM banks of 1,073,741,792 bytes per chip (Command 6), about 12.0 GiB per chip; tt-train's
  memory tool assumes 12 GB per device (scripts/analyze_memory.py:420). Training on one n300 means one chip, or both
  chips through DDP or tensor parallel over the card's internal ethernet link.
- Upstream anchor (an upstream reading, not ours): tt-train's memory guide shows TinyLlama training with
  MorehAdamW at a 2,048-token sequence peaking at 8,278.65 MB with per-block recompute and 15,666.65 MB without
  (docs/MEMORY_TRACKING.md:147-190): weights 1,863.25 MB in bf16, optimizer state 3,708.76 MB, gradients
  1,663.07 MB.

Memory arithmetic, all PROJECTED, batch 1, sequence S = 3,072 (the multiple of 32 above tt-train-v0's longest
record, 3,058 tokens in the Qwen3 chat format; Qwen2.5's template adds a default system turn, so its longest record
must be counted and may need 3,104):

- Static state: full training with bf16 weights, gradients, and two bf16 moments is 8 bytes a parameter; with fp32
  master weights and moments, 16; LoRA and prompt tuning keep the base frozen at 2 bytes a parameter, plus an
  adapter or soft prompt under 0.15 GB.

| Base | Parameters | Full, 8 B | Full, 16 B | LoRA or prompt tuning |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | 494,032,768 (header) | 3.95 GB | 7.90 GB | 0.99 GB |
| Qwen3-0.6B | 596,049,920 (plans/spikes/own-model.md) | 4.77 GB | 9.54 GB | 1.19 GB |
| Qwen3-1.7B | 1,720,574,976 (plans/spikes/own-model.md) | 13.76 GB | 27.5 GB | 3.44 GB |
| scratch qwen3-77m | 77,228,928 | 0.62 GB | 1.24 GB | n/a (full only) |

- Logits: S x 151,936 x 2 bytes = 0.93 GB, the same again for their gradient, and twice each if the loss upcasts to
  fp32: 1.9 to 5.6 GB at batch 1, for every base.
- Attention matrices kept for backward, if materialized: heads x S^2 x 2 bytes per layer, 0.30 GB for Qwen3 (16
  heads), 0.26 GB for Qwen2.5 (14), 0.11 GB for the scratch model (6); over all layers 8.5 GB (Qwen3, 28 layers),
  6.3 GB (Qwen2.5, 24), 1.4 GB (scratch, 12).
- Other kept activations, roughly S x (4 hidden + 2 query width + 4 intermediate) x 2 bytes a layer: 3.7 GB
  (Qwen2.5-0.5B), 3.5 GB (Qwen3-0.6B), 6.3 GB (Qwen3-1.7B) over all layers.
- With per-block recompute (tt-train's memory-efficient runner; Hugging Face gradient checkpointing) only the block
  inputs stay, S x hidden x 2 bytes a layer (0.13 to 0.35 GB in all), plus one block's working set, about 0.5 to
  1 GB.

| Base and mode | With recompute | Without recompute | One chip (12 GiB) |
| --- | --- | --- | --- |
| Qwen2.5-0.5B full, 8 B | 6.4 to 10.7 GB | 15.8 to 19.6 GB | fits only with recompute, marginal at its high end |
| Qwen2.5-0.5B LoRA | 3.6 to 7.8 GB | 13.0 to 16.7 GB | fits only with recompute |
| Qwen3-0.6B full, 8 B | 7.3 to 11.6 GB | 18.6 to 22.4 GB | marginal, only with recompute |
| Qwen3-0.6B LoRA | 3.8 to 8.1 GB | 15.2 to 18.9 GB | fits only with recompute |
| Qwen3-1.7B full, 8 B | over 13.8 GB of state alone | - | no; two chips with tensor parallel at about 6.9 GB of state each, unproven |
| Qwen3-1.7B LoRA or prompt tuning | 6.3 to 10.5 GB | 20.3 to 24.0 GB | fits only with recompute |
| scratch 77M full | 3.0 to 7.3 GB | 4.4 to 8.1 GB | fits either way |

Each row is the static state above, plus the logits term, plus either the block inputs and one block's working set
(with recompute) or the attention matrices and the other kept activations (without).

- At 128 tokens, the length upstream trained Qwen at, the logits and activations shrink 24-fold and every row but
  Qwen3-1.7B full training fits one chip (PROJECTED).
- Unknown: whether tt-mlir keeps the attention matrices (a fused attention would not); whether torch-xla's gradient
  checkpointing survives tt-mlir's compilation (in PyTorch/XLA a recompute can be merged away unless
  torch_xla.utils.checkpoint marks it; inferred, not read for tt-mlir); how much of each chip's DRAM the runtime
  reserves; and how long tt-mlir takes to compile a 24- to 28-layer training graph at 3,072 tokens.
- Compute (PROJECTED): a pass over tt-train-v0 padded to 8 x 3,072 = 24,576 tokens is about 1.5e14 FLOP for
  Qwen3-0.6B full training with recompute (6N plus attention, times 4/3), about 2.5 minutes at a sustained
  1 TFLOP/s and 15 seconds at 10, against a peak of about 65 TFLOP/s FP16 per chip (half of the card's 131). The
  sustained rate on these stacks is not known; compile time is likely the larger cost in the window.

### 3. The smallest path to a first training run in the window

The path below assumes OQ-100 allows training on a card and OQ-101 picks the tt-xla wheel for the window.

Before the window (the workstation; nothing on alpha01):

1. Pins (new files, tested as text): toolchains/tt-xla.pin naming pjrt-plugin-tt 0.9.0.dev20260223001148 with its
   sha256 and size, torch 2.9.0+cpu, torch-xla 2.9.0+git061c1e7, jax and jaxlib 0.7.1, transformers 4.57.1 (the
   version tt-blacksmith ran with), a pinned peft, CPython 3.11 from uv, and the expected tt-metal commit; and
   toolchains/tt-xla.sh, which makes the venv at $LASSI_TOOLCHAINS/tt-xla@0.9.0.dev20260223001148 with
   `uv python install` and `uv pip install --require-hashes` under the scratch root only, links the pinned sfpi 7.25.0
   into the wheel's runtime directory, refuses a wheel whose metadata does not name tt-metal 5280a9cf, and writes an
   install record. No sudo, no system package, nothing outside $LASSI_TOOLCHAINS (Agent Rule 7).
2. A Trainer backend behind lassi/train's interface, registered for example as Trainer `ttxla`
   (lassi/train/ttxla_trainer.py), which:
   - declares method sft only (no reward, so Agent Rule 9's reward clause holds by construction), weight modes lora,
     full, and prompt_tuning, and data sources synthetic and corpus;
   - takes a new device kind, for example `tenstorrent` with the card's /dev/tenstorrent number as its one index (the
     card key OQ-072 (7) proposed, with the card's serial recorded), with no default; lassi/core/record.py DEVICE_KINDS holds only cpu, cuda, and rocm today, so the
     kind, its probe (sysfs card facts as lassi/executors/tt_silicon.py reads them, no node opened, OQ-002), and the
     DeviceRecord rules are new;
   - reuses the trl trainer's checks that carry over (base kinds, the max_length refusal so no record is cut,
     chat_format) and refuses trainer.max_length below the longest record;
   - runs the steps in a worker process with the tt-xla venv's own interpreter (`python -I`), since LASSI's
     environment is Python 3.10 with torch 2.14.1 and the wheel needs Python 3.11 with torch 2.9.0; the worker gets a
     JSON job file (records, the base's snapshot path, settings), sets the hub libraries offline, runs a plain
     PyTorch loop on the XLA device as tt-blacksmith does (PEFT LoRA or prompt tuning, or full weights), syncs every
     step, and writes output/checkpoint-<steps>/ in the same layout the trl trainer writes (PEFT adapter or
     safetensors, tokenizer, steps.jsonl) through lassi.train.checkpoint, so hf_local serves it unchanged on the CPU
     (OQ-076 (3a); inferred);
   - sets the worker's environment: TT_METAL_CACHE and TMPDIR under the scratch root (tt-metal's JIT clears other
     entries in its cache root; plans/LESSONS.md, alpha01), HF_HOME, HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE, no
     telemetry, TT_VISIBLE_DEVICES set to the card, the 1-byte core limit read back as (1, 1), nice and ionice;
   - records framework pins from the venv's dist-info files read by path (importlib.metadata can read another
     site-packages without importing it; inferred), since run.py's `packages` reads LASSI's own environment only, so
     run.py needs a small change to take a Trainer's external environment.
3. The silicon guard: `lassi train --silicon`, a flag no recipe can set and no unattended session passes, as
   `lassi run --silicon` is (lassi/cli.py:34-37, :152-153); without it a tenstorrent device is refused before any
   directory exists. The run takes the tt_silicon executor's lock ($LASSI_SCRATCH/locks/tt-silicon.lock) for its whole
   length and honours its quarantine marker (lassi/executors/tt_silicon.py:62-139, :223-226; OQ-075 (2a), (3a)), so a
   training run and a silicon evaluation run never overlap and nothing retries after a hang.
4. Recipes under projects/train/: a smoke recipe on the synthetic fixture (tests/fixtures/train) at a short length,
   then one per technique on tt-train-v0: LoRA on Qwen/Qwen2.5-0.5B@060db6499f32faf8b98477b0a26969ef7d8b9987 (the
   closest to upstream's single-chip recipe), LoRA on Qwen3-0.6B and Qwen3-1.7B at their pinned revisions, full
   weights on Qwen2.5-0.5B or Qwen3-0.6B, prompt tuning on Qwen3-1.7B, and the scratch model with full weights.
   Batch 1, max_length 3,072 (3,104 for Qwen2.5 if its count needs it), gradient checkpointing on, each with its
   PROJECTED time and memory in its comments.
5. Data (read): tt-train-v0 (8 records, 52,215 bytes, records sha256 86bc7931...; assets/train/tt-train-v0/manifest.json)
   is OQ-076 (1a)'s data set: tt-metal's programming examples at the pin without the five Tier A items and their
   duplicates, enforced by tools/build_tt_train.py and checked again by lassi train (lassi/train/data.py
   load_corpus). The records are generated on the host under $LASSI_SCRATCH/train-data
   (`uv run tools/build_tt_train.py build --tree $LASSI_TOOLCHAINS/tt-metal@5280a9cf`), never tracked. tt-metal's
   documentation is not yet a corpus; the RAG pack (assets/rag/tt-metal-docs) is a separate asset.

Tests on Windows with fakes (no device, no venv):

- registration and declarations; every config refusal naming its key (device kind, card index, no default);
- check(): base kinds and pins, weight modes, max_length against the longest record, no truncation;
- the worker's job file and its command line and environment, built deterministically, with a fake venv path;
- a fake worker script that writes steps.jsonl and a checkpoint, so train() and the records are tested end to end;
- the device probe against a fake sysfs tree (card type n300, serial, firmware, driver), and its refusals;
- `lassi train` refusing a tenstorrent device without --silicon, before any directory exists;
- the shared lock and quarantine (a second holder refused at once, release after an error, quarantine after a hang
  or a status of 128 or more), as the tt_silicon executor's tests do;
- framework pins read from a fake venv's dist-info, and the refusal of a wheel whose metadata names another
  tt-metal commit;
- with the cpu extra (`uv run --extra cpu pytest ...`): the worker's loop itself on the CPU with the fixture model,
  LoRA and full and prompt tuning, a few steps each, since only the device differs on the card (inferred; the worker
  would import only torch, transformers, and peft, which LASSI's cpu extra also installs at other versions);
- pin and script text: hashes, sizes, no sudo, every path under the scratch root.

In the window (alpha01; each batch only after the owner's yes in the session, Agent Rule 17; commands an agent
would run, not run here):

```
uv run tools/rx.py doctor
uv run tools/rx.py exec -- 'cat /sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages; ls -la /dev/hugepages-1G; ls /dev/shm | grep -c TT_UMD_LOCK; df -B1 /mnt/nvme10; du -sh /mnt/nvme10/joseph_ufl'
uv run tools/rx.py job start --big --name tt-xla-venv -- 'bash toolchains/tt-xla.sh'
uv run tools/rx.py job start --big --name tt-train-smoke -- 'uv run lassi --graphics off train projects/train/ttxla-smoke-fixture.yaml --silicon'
uv run tools/rx.py job wait <id> --timeout 600
uv run tools/rx.py job start --big --name tt-lora-q25 -- 'uv run lassi --graphics off train projects/train/ttxla-lora-qwen2.5-0.5b.yaml --silicon'
uv run tools/rx.py pull <id> --into results/<name>
```

- The poll keeps the device path and the SMI tool's name out of the command text, which the gate refuses
  (plans/LESSONS.md, alpha01), and waits until no other account's file is in /dev/hugepages-1G or /dev/shm and the
  pool is free (the owner's direction of 2026-10-23).
- The smoke recipe is the smallest first training run on a card: if tt-train-v0 at 3,072 tokens fails to compile or
  runs out of memory, the smoke still trains on the card and its failure point is the finding.
- A PROJECTED timeline for 04:00 to 12:00: checks and the install (if not done before) 20 to 40 minutes; the smoke
  20 to 40 minutes, most of it compiling; Qwen2.5-0.5B LoRA 30 to 90 minutes; then the other recipes one at a
  time, each stopping the sequence at its first hang or failure.
- Every run is exploratory until it runs from a clean commit and its provenance names the card, the driver, the
  firmware, the wheel, and the pins (Agent Rules 1 and 10).

The tt-train route later (for the own-model goal; not this window): a toolchains/tt-train.sh that fetches the pinned
tree again into its own prefix (for example $LASSI_TOOLCHAINS/tt-metal-train@5280a9cf, so the installed pin is
untouched), pre-places pinned tokenizer files under tt-train/data, configures with the pin's flags plus
WITH_PYTHON_BINDINGS ON and BUILD_TT_TRAIN ON, and builds the TT-NN library, its Python module, and the ttml targets
with -j $LASSI_JOBS as one big job; then a Trainer `ttml` with the same device kind, guard, and lock. Unknown: whether
tt-train builds with ENABLE_DISTRIBUTED OFF, the pin's setting.

### 4. Risks on alpha01

- Agent Rule 9 (read): "one placement at a time" holds if each training run is one process holding one card for its
  whole length under the executor's lock, with nothing else on silicon meanwhile. "Never looped device opens" holds
  if nothing restarts a run automatically: a failed or hung run stops the sequence and quarantines (OQ-075 (3a)).
  But each process start makes UMD open every card's node for a device-info read, at least four times with
  TT_VISIBLE_DEVICES set (plans/spikes/tt-silicon-executor.md, Finding 1), so starts should be few and spaced, and a
  script of many short runs would look like the opens the rule forbids (inferred). "RL rewards never execute on
  silicon" holds with sft only. "Silicon runs are evaluation runs a human starts" does not hold for training as
  written (OQ-100 draft).
- Trusted code (read; Agent Rule 6): the training stack, PyTorch, and LASSI's worker are tracked upstream and LASSI
  code, never model output, so they run outside the sandbox as the 2026-10-23 smoke did; OQ-078's host-memory
  finding concerns model output and does not block training (inferred). The data is upstream text that is never
  executed.
- Hangs and resets (read and inferred): upstream tt-xla lists one training model that hangs (Command 6), tt-train's
  guide says a stopped fabric run may need a card reset (docs/DISTRIBUTED_TRAINING.md:319-327), and alpha01 has no
  SMI tool while the gate refuses its name, so no agent can reset a card; a hung card stays quarantined until the
  owner resets it or the host reboots. Single-chip runs avoid the fabric (inferred). Under OQ-075 (3a) a compile
  abort (status 134) quarantines every card, as OQ-084 describes for kernel JIT failures; tt-mlir compiles on the
  host before any device program runs, so a compile failure may abort the process without touching the card
  (inferred), and the owner may want training's quarantine read the same way or not.
- The host (read): alpha01 went down during the window of 2026-10-25 two minutes after a CPU training recipe started
  beside RNGD serving, and the cause is still asked (OQ-085, OPEN). Training on the cards puts the compile and data
  work on the host CPU, niced (inferred). The cards' previous user may take minutes to release them (the owner,
  2026-10-23).
- Hugepages (read, plans/spikes/tt-silicon-executor.md, Finding 3): each n300 a process opens takes two 1 GiB pages
  of the host's 32 for as long as it runs; if the PJRT client opens every card, that is 16 pages (inferred).
- The scratch disk: the brief's figure of about 62 GB free against the gate's 60 GB floor for big jobs predates
  the owner's removal of about 73 GB of orphan model blobs on 2026-10-26; the free space is read again with rx
  doctor before the install. The tt-xla venv (2.5 to 4 GB), the JIT and compile caches (about 1 GB), and the
  checkpoints (a full Qwen3-0.6B in bf16 1.2 GB, adapters a few MB) need about 5 to 10 GB (PROJECTED); the tt-train
  build adds 5 to 10 GB more.
- Pins (read): the wheel's tt-mlir is c7b8bd77, 7 commits before LASSI's tt-mlir pin, with the same tt-metal; the
  wheel's binaries were built by Tenstorrent's CI, not by LASSI's pinned clang-20 build, so the same source commit
  is a different binary; Python 3.11, torch 2.9.0, and transformers 4.57.1 are new pins beside LASSI's Python 3.10,
  torch 2.14.1, and transformers 5.18.0 (pyproject.toml). Each is a Decision Log entry (Agent Rule 10).
- Agent Rule 13 does not bear: no RNGD card is used; trained models are served through hf_local on the CPU
  (OQ-076 (3a)). Agent Rule 5 holds through tt-train-v0's manifest checks; any new or split corpus goes through the
  same tool and tests.

## Confidence

- High: what each stack contains and needs at the pins (read at the commits named), the wheel's tt-metal commit (its
  own metadata, sha256 checked), the three bases' architectures (their configs and safetensors headers), and the
  loader mismatches in tt-train (read in the loader and module code).
- Medium: that the wheel works with the pinned sfpi and alpha01's driver 2.8.0 and firmware 19.5.0.0 (the pinned
  tt-metal ran Tier A on that host on 2026-10-23, results/tt-silicon-smoke-20261023; the wheel's binaries are
  another build of that commit); that LoRA on the three bases trains on one chip at short lengths (upstream recipes
  and CI at that wheel, at 128 tokens).
- Low: anything at tt-train-v0's 3,072 tokens through tt-xla (no upstream evidence at that length; recompute under
  tt-mlir unknown), the build time of TT-NN and tt-train on alpha01, sustained training throughput, compile times,
  and which chips the PJRT client opens on an eight-card host.

## Consequences for the plan

- No Tenstorrent training task can start until OQ-100 is answered; until then training stays on alpha01's CPU, which
  itself waits on OQ-085.
- If OQ-100 allows it and OQ-101 picks (a) or (c), a task set: the pin and install script; the device kind and
  probe; the `ttxla` Trainer and worker with the `lassi train --silicon` guard and the shared lock; the recipes; and a
  window run script. All but the window run are workstation work with fakes; the CPU-loop test needs the cpu extra.
- The window of 2026-10-27 can then aim at the smoke and LoRA on Qwen2.5-0.5B as its first training runs on a card,
  and report plainly which of full training, prompt tuning, LoRA on the Qwen3 bases, and the scratch model ran.
- The tt-train route stays a later task for the own-model goal, after the disk is freed and its build approved.
- The bible's Training Module, the Agent Rules (if OQ-100 changes Rule 9), Toolchain Pins, and Result Record
  (DEVICE_KINDS) change only with the code and the owner's answers; the factual edit below records what is read now.

## Proposed bible edit and Decision Log entry

Facts only; the choices stay [OPEN] and point to the drafts. In docs/BIBLE.md, Training Module, Compute, insert after
the "Training host and recipes" bullet:

```
- [OPEN] Training on the Tenstorrent Wormholes (the owner's direction of 2026-10-26; plans/spikes/tt-training.md; OQ-100, OQ-101, OQ-102). Read at the pins: tt-train, in the pinned tt-metal tree at 5280a9cf, is a C++ autograd framework with a Python module, ttml, that links TT-NN and imports its Python module, so it needs TT-NN, the Python bindings, and tt-train built (BUILD_TT_TRAIN and WITH_PYTHON_BINDINGS), none of which the pinned install builds; its models are GPT-2 and a Llama-style transformer (RMSNorm, RoPE, grouped-query attention, SwiGLU) loaded from safetensors, with head_dim fixed at embedding_dim / num_heads and no attention biases or q/k norms, so Qwen2.5-0.5B loads without its q/k/v biases, Qwen3-1.7B without its q/k norms, and Qwen3-0.6B is refused; it trains full weights and frozen tensors, recomputes per block, and has an unused LoRA layer. The tt-xla wheel pjrt-plugin-tt 0.9.0.dev20260223001148 (sha256 fbbe6241869e9646633604b213903b9795f9dc2f9192f875a569799689b0c4db) names tt-metal 5280a9cf and tt-mlir c7b8bd77 in its metadata, bundles TT-NN built at that commit, needs Python 3.11, torch 2.9.0, and torch-xla 2.9.0+git061c1e7, and takes sfpi from /opt/tenstorrent/sfpi unless its runtime root provides one; upstream trained Qwen with PEFT LoRA through it on one Wormhole chip at 128 tokens (tt-blacksmith 002ebe47), while tt-xla's single-device training list at that commit marks Qwen2.5-0.5B and Qwen3-0.6B as known failures and Qwen3-1.7B as out of DRAM. A Wormhole chip has 12 DRAM banks of about 1 GiB; full training with bf16 state takes about 8 bytes a parameter, so Qwen3-1.7B's full weights (about 13.8 GB, PROJECTED) do not fit one chip. Agent Rule 9 allows only evaluation runs on silicon, so nothing trains on a card until OQ-100 is answered.
```

Decision Log: insert at the top, newest first, and raise the count sentence from "One hundred and eighty-two" to
"One hundred and eighty-three", adding "and one on 2026-10-26" after "twelve on 2026-10-25" (or raising the
2026-10-26 count if other entries of that date land first):

```
| 2026-10-26 | What training on the Wormhole cards needs at the tt-metal pin (Training Module, Compute, the Tenstorrent bullet; plans/spikes/tt-training.md; OQ-100 to OQ-102). tt-train at 5280a9cf needs TT-NN, the Python bindings, and tt-train built, which the pinned install does not have, and its Llama-style model cannot load Qwen2.5 (q/k/v biases skipped) or Qwen3 (q/k norms skipped; Qwen3-0.6B's 2048-wide query projection refused) as trained. The tt-xla wheel pjrt-plugin-tt 0.9.0.dev20260223001148 is built on tt-metal 5280a9cf and tt-mlir c7b8bd77 by its own metadata, bundles TT-NN, and needs Python 3.11 with torch 2.9.0 and torch-xla 2.9.0+git061c1e7; upstream trained Qwen with LoRA through it on one chip at 128 tokens, and its own training list marks the three bases as known failures or out of DRAM. Full training of Qwen3-1.7B does not fit one 12 GiB chip (PROJECTED). Training on silicon waits on OQ-100, since Agent Rule 9 allows only evaluation runs there | The owner asked on 2026-10-26 to train the smaller models on the Tenstorrent cards in the window of 2026-10-27. Read from the pinned tt-metal tree (tt-train/), the wheel's metadata (sha256 checked on the workstation), tt-xla at ff3a67dd, tt-blacksmith at 002ebe47 and its pull requests 463 and 464, and the three bases' configs and safetensors headers; nothing ran on alpha01 or a device, and every memory figure is arithmetic. The stack, the rule, and the host steps are the owner's (OQ-100 to OQ-102) |
```

## Owner-queue drafts

```
## OQ-100 Training On The Wormhole Cards: Agent Rule 9 And The Training Host
State: OPEN
Kind: policy
Blocks: the Tenstorrent training tasks (to be planned from plans/spikes/tt-training.md); training in the window of 2026-10-27, 04:00 to 12:00 EDT
Evidence: plans/spikes/tt-training.md (Plain answer; Findings 3 and 4); docs/BIBLE.md, Agent Rules 6 and 9 and Training Module (Compute, the Training host and recipes bullet; Safeguards); plans/spikes/tt-silicon-executor.md, Finding 1; OQ-075 (2a), (3a), (6a); OQ-078; OQ-084; OQ-085
Question: You asked on 2026-10-26 to train the smaller models on the Tenstorrent cards by the window of 2026-10-27. Agent Rule 9 says "silicon runs are evaluation runs a human starts", and the bible puts training on alpha01's CPU only, so a training run on a card needs your decision and a rule change first. A training run would be trusted code (the upstream training stack and LASSI's trainer, never model output, so Agent Rule 6 and OQ-078 do not apply), sft only (no reward, so RL never touches silicon), one process holding one card under the tt_silicon executor's lock for its whole length. Each process start makes UMD open every card's node for a device-info read, and a hung card needs a reset no agent can do (no SMI tool on the host, and the gate refuses its name). OQ-085, alpha01 going down during the last CPU training, is still open. May training run on the cards, and how is it started?
Options:
- (a) Amend Agent Rule 9 to: "Tenstorrent silicon: one placement at a time, never looped device opens (they have taken alpha01's network down twice). RL rewards never execute on silicon; silicon runs are evaluation runs, or training runs of tracked code, that a human starts." Training on a card then takes `lassi train --silicon` (a flag no recipe can set and no unattended session passes), one card per run, the executor's lock, no automatic retry, and the quarantine of OQ-075 (3a) after a hang, kill, or crash. Consequence: a rule change in the bible, AGENTS.md, and the master copy, with a Decision Log entry.
- (b) As (a), but for the window of 2026-10-27 only, as an exploratory exception recorded here; the rule text stays.
- (c) Keep the rule: training stays on alpha01's CPU once OQ-085 is answered, and the cards stay evaluation-only.
- And under (a) or (b): (s1) one human start per training run; or (s2) one human start for a sequence of runs, each its own process, spaced, the sequence stopping at the first failure or hang.
Recommendation: (a) with (s2): one placement at a time and no looped opens still hold, a person still starts the work, and a window can run the techniques you asked for in turn. Please also answer OQ-085 before any training on alpha01.
Answer:
```

```
## OQ-101 The Tenstorrent Training Stack And Its Pins
State: OPEN
Kind: decision
Blocks: the Tenstorrent training tasks; toolchains/tt-xla.pin and toolchains/tt-train.sh (new)
Evidence: plans/spikes/tt-training.md, Findings 1 and 2 (the wheel's metadata, sha256 fbbe6241...; tt-xla's training test list at ff3a67dd; tt-blacksmith at 002ebe47 and pull requests 463 and 464; tt-train at 5280a9cf)
Question: Which stack trains on the cards? Both are built on the project's tt-metal pin, 5280a9cf. Neither has trained a Qwen base at tt-train-v0's 3,058-token records upstream.
Options:
- (a) tt-xla: the prebuilt wheel pjrt-plugin-tt 0.9.0.dev20260223001148 (121,492,577 bytes), whose metadata names tt-metal 5280a9cf and tt-mlir c7b8bd77 (7 commits before LASSI's tt-mlir pin), in its own venv under $LASSI_TOOLCHAINS with Python 3.11, torch 2.9.0, torch-xla 2.9.0+git061c1e7, transformers 4.57.1, and peft; it bundles TT-NN, so nothing is built from source. The Hugging Face Qwen code runs as written, so all three bases and the scratch model load correctly; LoRA and prompt tuning through PEFT, full weights through plain PyTorch. Consequence: new pins beside LASSI's Python 3.10 environment, and a worker process in that venv; the JIT must be given the pinned sfpi 7.25.0 instead of the host's 7.48.0; upstream trained Qwen this way at 128 tokens with LoRA, and its own training list marks the bases' full training as known failures (Qwen3-1.7B out of DRAM).
- (b) tt-train (ttml): built from the pinned tree with TT-NN, the Python bindings, and tt-train, as one CPU job (PROJECTED 30 to 90 minutes and 5 to 10 GB); its own Llama and GPT-2 models, tested on N300 upstream, with per-block recompute and fused loss and attention kernels. Consequence: none of the three Qwen bases loads correctly without new model code, LoRA is an unwired layer, and a converter is needed to serve its checkpoints; a Llama-architecture model of our own from scratch (the 77M config without q/k norms, Qwen3 tokenizer) is native.
- (c) Both: (a) for the window of 2026-10-27, (b) afterwards for the own-model goal.
- (d) Neither until a later pin.
Recommendation: (c): (a) is the only stack that can load the three bases correctly without a build before Tuesday, and (b) is the pin's own framework and the better fit for a model of our own.
Answer:
```

```
## OQ-102 Before The Window Of 2026-10-27: Sequence Length
State: OPEN
Kind: access
Blocks: the training runs of the window of 2026-10-27
Evidence: plans/spikes/tt-training.md, Findings 2 to 4; the install runs as one CPU-only job before the window under the owner's direction of 2026-10-26 ("If they don't require the hardware accelerators, you can build them now."), after the free space is read with rx doctor (the owner removed about 73 GB of orphan model blobs on 2026-10-26)
Question: tt-train-v0's longest record is 3,058 tokens and no record is ever cut; upstream trained Qwen at 128 tokens, and at 3,072 a chip needs recompute (PROJECTED), which is unproven on this stack. What if tt-train-v0 does not compile or fit?
Options:
- (3a) Run the smoke recipe on the synthetic fixture first, then tt-train-v0, and report any tt-train-v0 failure as found. (3b) Also allow a tt-train-v1 corpus that splits each record by file, built and audited like tt-train-v0 (Agent Rule 5), as a later task. (3c) tt-train-v0 only.
Recommendation: (3a), with (3b) as a later task if 3,072 tokens fails.
Answer:
```

## Recommendation

Ask the owner OQ-100 first, at the top of the next message: as written, Agent Rule 9 forbids training on the cards.
If the owner allows it, use the tt-xla wheel for the window (OQ-101 (c)). Its metadata names the project's tt-metal
commit, it needs no source build, and it loads all three Qwen bases as Qwen wrote them. Build the `ttxla` Trainer, its
worker, the tenstorrent device kind, the `lassi train --silicon` guard, and the recipes on the workstation with fakes
and a CPU-loop test. In the window, run the fixture smoke and then LoRA on Qwen2.5-0.5B on one chip of one card, each
started by a person, under the executor's lock, with no retry after a failure. Tell the owner now that full training
of Qwen3-1.7B does not fit one chip, and that tt-train-v0's 3,058-token records go beyond anything upstream ran
through that stack. Keep tt-train for the own-model goal once the disk is freed and its build is approved.
