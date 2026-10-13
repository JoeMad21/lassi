# P17 gate part (a): the hardware-free half on alpha01's CPU

Gate part (a), verbatim (bible Build Roadmap, P17 row, Gate column): "(a) Without a GPU, on alpha01's CPU: a recipe
that binds hf_local on PyTorch CPU runs one HeCBench item in both directions with every trial past the baseline, and
provenance records the model's device probe and framework pins; `lassi train` runs sft, dpo, and grpo briefly on the
CPU with a tiny model on synthetic data, each checkpoint recording its device and pins; a recipe that names a GPU is
refused on a host without one. Part (a)'s pass opens a pull request for the hardware-free half (OQ-038)."

Verdict: PASS, 12 of 12 checks (verdict.txt).

## Provenance

[MEASURED] provenance.json, the gate's record of rx job 20261010-235501-p17-gate-a-8980:

- commit 7f920d71f7558b3b849dc78f1e90b58eceda0afe, the tip of branch p17-portable (provenance.json records the
  slot's branch name, detached-7f920d71), dirty false, snapshot_of null; slot
  desktop-8r113ei-detached-7f920d71; command `bash plans/runs/p17-gate-a/run.sh` as a big rx job with
  --timeout 21600;
- host alpha01 (Ubuntu 22.04.5 LTS, kernel 6.6.29+main+3.0.0r1-amd64-gio-epilmore-dev+, Python 3.10.12); start
  2026-10-10T23:55:01-07:00, end 2026-10-11T00:05:00-07:00, rc 0; scratch free 321.7 GB before and 320.6 GB after;
- the owner approved the launch in the working session of 2026-10-11 (Agent Rule 17; OQ-047, "Run with speed check
  (Recommended)"), and the two pulls of this folder on 2026-10-12 (OQ-047); `rx doctor` before the launch showed no
  running job, scratch free 321.7 GB, and no stop (OQ-047);
- toolchain pins: provenance.json's pins holds the first 4,000 characters of each of the cuda, gcc, hipcc, nvhpc,
  tt-metal, and ttsim pin files (the gate cuts each there, so tt-metal.pin's tail is not in it); the run's
  own pins (hf-run-provenance.json) are cuda 12.6.3 and nvhpc 24.11; the framework pins are torch 2.14.1+cpu,
  transformers 5.18.0, trl 1.14.1, peft 0.21.2, accelerate 1.15.0 (freeze.txt, and each train-*/provenance.json);
- devices: the host CPU, "AMD EPYC 9534 64-Core Processor" (each device record), for the model (model.device, kind
  cpu, torch 2.14.1+cpu), the trainer (trainer.device, kind cpu), and the omp target ("host CPU (native)"); the cuda
  target ran compile only ("none (compile only)"). Every device record names the CPU. The four recipes that name
  a GPU opened no GPU device node by the batch's strace check, which matches /dev/kfd, /dev/dri*, and /dev/nvidia*
  (plans/runs/p17-gate-a/gate.py, GPU_NODE; an open relative to a directory descriptor and an alias such as
  /dev/char/<major>:<minor> are not resolved, and no RNGD or Tenstorrent node is checked). No resolved recipe of
  the batch binds an RNGD card or a Tenstorrent device: p17-cpu-hf's chain passes through rngd-compile and rngd-cpu,
  whose RNGD-served model it overrides with hf_local on the CPU (hf-run-provenance.json, hf-run-recipe.resolved.yaml).

batch.txt records the batch's header: nproc 16 in the job, OMP_NUM_THREADS=16, a core limit of 1 byte, and du of the
scratch root 102,752,868 KiB before, under the batch's 115 GiB stop line (120,586,240 KiB).

## Checks

| Check (verdict.txt) | Evidence | Outcome |
| --- | --- | --- |
| env synced from the lock with the cpu extra and the five pins | freeze.txt (64 lines), stdout.txt | pass |
| model fetched once at its revision and matching the pin | model.json: Qwen/Qwen2.5-Coder-0.5B-Instruct at ea3f2471cf1b1f0db85067f1ef93848e38e88c25, 10 files, 999,604,233 bytes, sha256 of each listed file matching | pass |
| bench sources, upstream, and the slot's prompt files | stdout.txt: fetch-bench, fetch-upstream, extract-assets rc 0; 59 prompt files | pass |
| hf_local on cpu: both directions past the baseline, device record and pins | trials.json, hf-run-provenance.json, hf-run-recipe.resolved.yaml (recipe p17-cpu-hf, hash 8fce8899...) | pass |
| lassi train sft: checkpoint with device and pins | train-sft/ (checks.json, provenance.json, checkpoint-2/) | pass |
| lassi train dpo: checkpoint with device and pins | train-dpo/ | pass |
| lassi train grpo: checkpoint with device and pins | train-grpo/ | pass |
| p17-gpu-hf-cuda refused before any directory, no node opened | gpu-refusals.json: rc 2, "torch 2.14.1+cpu is not a CUDA build" | pass |
| p17-gpu-hf-rocm refused before any directory, no node opened | gpu-refusals.json: rc 2, "torch 2.14.1+cpu is not a HIP build" | pass |
| p17-gpu-exec-cuda refused before any directory, no node opened | gpu-refusals.json: rc 2, "/dev/nvidiactl does not exist on this host"; strace shows one newfstatat of /dev/nvidiactl, no open | pass |
| p17-gpu-exec-rocm refused before any directory, no node opened | gpu-refusals.json: rc 2, "/dev/kfd exists but this user may not read and write it"; strace shows newfstatat and access of /dev/kfd, no open | pass |
| slot tree clean after the batch | stdout.txt: 0 paths | pass |

The two HeCBench trials (trials.json; item matrix-rotate of lassi-hecbench-10, one run each): cuda-omp ended at
correction-cap after 3 attempts at stage S1, its reference run exiting 0; omp-cuda ended normally after 1 attempt at
stage S4, with no reference run, since the cuda target is compile only. Both are past the baseline, which is what the
gate asks; neither outcome is a quality measurement of the model, a 0.5B model chosen to fit the CPU.

Each training run took 2 steps on 4 synthetic records with the tiny fixture model fixture:tiny-causal-lm
(train-*/provenance.json and recipe.resolved.yaml), bf16 on the CPU. [MEASURED] train_runtime 0.0548 s (sft),
0.0773 s (dpo), and 0.1278 s (grpo) (stdout.txt): smoke timings of a tiny model, not a benchmark.

The refusals, case by case (gpu-refusals.json), each exiting 2 before any run directory existed:

- p17-gpu-exec-cuda asks for cuda 0 for the cuda target's executor and is the gate's literal case: alpha01 has no
  NVIDIA GPU ("/dev/nvidiactl does not exist on this host").
- p17-gpu-hf-cuda and p17-gpu-hf-rocm ask for cuda 0 and rocm 0 for the model and are refused by the framework
  build ("torch 2.14.1+cpu is not a CUDA build", "... not a HIP build").
- p17-gpu-exec-rocm asks for rocm 0 for the executor and is refused by an access(2) check of /dev/kfd: alpha01 has
  MI300Xs this user may not open (OQ-002, OQ-040), so the host is not GPU-less; the gate's "without one" holds as
  "without a usable one" here (PHASE-NOTES P17).

## Measured on the CPU

[MEASURED] speed.json, the speed step (load 6.647 s; dtype bfloat16; 16 torch threads): decode of 256 tokens after a
45-token prompt in 7.668 s (33.39 tokens/s), and prefill of 12,291 tokens in 13.751 s (893.85 tokens/s), on alpha01's
CPU under this job. These are CPU timings of one run for sizing, not a benchmark. The approval request (OQ-047)
described the speed check's long prompt as about 8,000 tokens; the prefill prompt was 12,291 tokens.

Space (sizes.txt, du-new.json): HF_HOME 0 to 976,248 KiB (the model), the venv 1,180,184 KiB, the batch's directory
26,000 KiB, and the scratch root 102,752,868 to 103,772,604 KiB; the new scratch entries are hf and
.cache/lassi-p17-torchinductor.

## Files

Pulled with `rx pull --path lassi-runs/p17-gate-a/20261010-235501-p17-gate-a-8980/p17-gate-a-report --into
results/p17-gate-a` and `rx pull 20261010-235501-p17-gate-a-8980 --into results/p17-gate-a` (provenance.json);
files.txt lists the report's 29 files, all ASCII. The raw step outputs and strace logs stay on alpha01 under the run
directory named in batch.txt.
