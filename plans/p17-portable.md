# P17 Portable Stack

DRAFT, written 2026-09-28 on branch p4-ttsim (HEAD d35b0a4) while P4 is ACTIVE; its tasks are not in plans/STATUS.md. When `uv run tools/status.py next` returns advance P17, re-read this plan against the tree and the bible (P4.8 changes the runner, stages, and CLI, and P4.11 changes lassi/executors, sandbox.py among them), correct every file and line reference, register the tasks, and commit it as `P17.0: plan phase`. Branch `p17-portable`, base: main if P4 is merged, else the p4-ttsim tip (AGENTS.md, Phase Planning).

## Scope

P17 builds what the P17 row of the bible's Build Roadmap names: the owner's request of 2026-09-27 to run LASSI on traditional hardware and to ready the stack for regular ML frameworks on other devices (Purpose And Scope). That is a device-neutral layer, with explicit device selection and capability probes recorded in provenance (Agent Rules 1 and 10; Result Record, provenance); framework-neutral serving (Model Serving; Component Interfaces, LLMBackend); a gpu executor for CUDA and HIP in the sandbox, plus hipcc (Execution Backends, gpu rows; Sandbox; Toolchain Pins); the timing, nvml, and rocm_smi profilers (Component Interfaces, Profiler; Result Record, Attempt.profile); a trainer interface on PyTorch (Training Module); and CI on the CPU. The gate is the Gate column of the P17 row: part (a) without a GPU (P17.11), whose pass opens a pull request for the hardware-free half (OQ-038), and part (b) on a GPU host (P17.G), which waits for OQ-040.

Relation to other phases: P17 builds the layer that P7 to P10 use and takes none of their scope. P7 (sft, rft, dpo with RNGD sampling, export) and P8 (grpo and gspo with vLLM rollouts at scale) train through P17's trainer interface. P9 keeps the LASSI-EE stages and power protocol and measures with P17's rocm_smi profiler. P10 runs the reproduction matrix on P17's gpu executor. P3 keeps RNGD serving and uses P17's recipe keys for servers.

Constraints for every task:

- Hardware: no GPU is reachable. The MI300Xs refuse /dev/kfd (OQ-002), and there is no NVIDIA host (OQ-003); see Environment State. P17.12, P17.13, and P17.G start BLOCKED on OQ-040. Nothing on alpha01 opens /dev/kfd or a /dev/dri node, so sandbox device tests use a harmless stand-in node. No RNGD card is used.
- The scope is fixed at planning. Each task gets one review round and one commit audit. [MEASURED] needs a clean-commit rx id, and rx output stays under 120 lines (the plans/runs/p0-retrospective.md practice, as in P4). A later finding goes into PHASE-NOTES or the owner queue.
- Frameworks stay out of the default environment. They are optional extras (P17.1 names them), imported only when a component is built. A plain `uv sync` keeps today's control plane, and lassi.llm still registers hf_local (PHASE-NOTES P0, registration rule).
- Agent Rule 5: training data is synthetic fixtures, never a bench item, and `lassi train` refuses eval and unassigned splits (OQ-025). Agent Rule 6: generated code runs only in the sandbox; the grpo smoke reads a fixture reward and executes nothing. Agent Rule 7: HF_HOME, the torch and Triton caches, downloads, and checkpoints live under the scratch root or the runs root; run `rx doctor` and du before any install. Agent Rule 12: API keys and HF tokens are named only by environment variable.
- Scratch sizes are PROJECTED until P17.1 measures them: the CPU framework extras about 1 GB per environment, and the gate model under 2 GB. A step that would pass 115G stops and queues candidates for the owner.
- Unattended sessions run with graphics off.

Decisions taken at planning (the named task records each in the bible with a Decision Log entry):

- Explicit devices (P17.2). A recipe names each device as a kind (cpu, cuda, rocm) and indices, with no default. The run probes each named device before any directory exists and refuses one the host lacks; it never falls back to the CPU. The probe's record goes into provenance.json and each trial's provenance: kind, name, count, memory, driver and runtime versions, and framework and version. lassi/core/runner.py:1789 at d35b0a4 keeps `driver` null until an executor reports one. ROCm PyTorch reaches its GPUs through torch's cuda device type, so kind rocm is checked against a HIP build of the framework.
- Placement (Design Principle 9). Device-neutral types live in lassi/core and import no framework. Framework probes sit beside their adapters (lassi/llm/hf_local, lassi/train). Vendor telemetry and device files are read only in lassi/executors and lassi/profilers.
- One GPU, one role (P17.5). The runner refuses a run in which hf_local and the gpu executor name the same GPU, unless the backend declares unload_before_run. This carries Agent Rule 13's intent to GPUs.
- Trainer (P17.8). A Trainer Protocol joins lassi/core/interfaces.py as the thirteenth interface, since the Training Module already calls lassi/train an interface that verl or OpenRLHF can replace. A new interface is the owner's to confirm at P17's planning review (Component Interfaces; Agent Rule 3).
- Gate drivers. Part (a) uses tests/fixtures/recipes/p17-cpu-hf.yaml, modeled on projects/lassi-demo/rngd-cpu.yaml, with hf_local on the CPU and the model P17.1 pins by revision, plus `lassi train` smoke recipes. Part (b) runs the same recipes with a GPU kind.
- CI (P17.10). A new workflow file sits beside text-policy.yml, which stays untouched. Making it a required check is the owner's repository setting.
- Owner reviews. P17 comes right after P4, so this plan's commit brings the items flagged for review in the next phase (plans/LESSONS.md, 2026-09-27) to the owner as one review item.

Owner queue: OQ-038 (the work order and the partial pull request) and OQ-039 (frameworks and devices) are applied with option (a) under the owner's 2026-09-27 direction and flagged for review. OQ-040 (a GPU host) stays OPEN and blocks P17.12, P17.13, and P17.G. PHASE-NOTES P17 items, by task: registration and extras, P17.4 and P17.9; the sandbox's private /dev and /sys, P17.5 and P17.12; AMD on alpha01, P17.1 and P17.6; caches, P17.4, P17.9, and P17.11; the work order, P17.11.

## Tasks

### P17.1 Spike: framework pins, environments, and device assumptions
- Bible: Toolchain Pins, Repository Layout (placement rules), Training Module (Compute; question 6), Host Facts (ROCm), Risks And Questions (question 15), Agent Rules 7 and 10.
- Accept: `plans/spikes/p17-frameworks.md` gives the following, each with its commands, outputs, and sources:
  - a PyTorch, Transformers, TRL, PEFT, and Accelerate set that installs together under Python 3.10 for the CPU, and the CUDA and ROCm wheel builds that pair with it (from index metadata);
  - whether one uv lock can hold the CPU flavor beside the GPU flavors, or the GPU flavors need their own environment;
  - the measured size of the CPU set in a scratch venv;
  - the `/v1/models` and usage replies of vLLM, SGLang, and llama.cpp's server at pinned versions (from docs or source);
  - whether alpha01's hipcc builds gfx942 code without a GPU, and which variables its environment script sets (read, never sourced); held until the owner's review of OQ-039 at P17's planning, since the bible records OQ-002's answer as "no agent action on AMD for now"; without that review P17.1 is DONE with this item recorded as deferred to P17.6;
  - a small instruct model pinned by revision for gate part (a), with its size;
  - every place in lassi/ that assumes RNGD, Tenstorrent, or a device by name, with file:line.
  The framework pins join Toolchain Pins and question 15 is answered, or its hipcc half recorded as deferred to P17.6, with a Decision Log entry.
- Files: `plans/spikes/p17-frameworks.md`, `docs/BIBLE.md`.
- Remote: `rx doctor`; read-only `rx exec`; one `rx run` that installs the CPU set in a scratch venv and measures it. Depends: none.

### P17.2 Device layer: selection, probes, provenance
- Bible: Result Record (provenance, Storage), Project Recipes (Notes), Component Interfaces (contract rules), Agent Rules 1, 3, and 10.
- Accept:
  - Any component that takes a device section (hf_local's model, a gpu executor's config, a train recipe) gets one shape: kind and indices. It resolves into the recipe hash. A missing or unknown kind, or duplicate indices, is refused at load with the key named (tests).
  - A probe seam returns the device record. With fake probes, a named device the host lacks is refused before any directory exists, and nothing falls back to the CPU (tests).
  - provenance.json and each trial's provenance record the device record. Recipes without a device section keep their resolved recipes and hashes (golden tests unchanged), and older trial.json files load (tests).
  - Bible edits (Result Record, Project Recipes) with a Decision Log entry.
- Files: `lassi/core/` (a devices module, recipe, runner, record, trial_md), `tests/core/`, `docs/BIBLE.md`.
- Remote: none. Depends: none.

### P17.3 Served backends: recipe keys and serving provenance
- Bible: Model Serving (Serving Rules), Project Recipes (Notes), Agent Rule 12; PHASE-NOTES P0 (HTTP settings).
- Accept:
  - model.base_url, model.timeout_s, and model.api_key_env reach openai_compat and ollama through the runner (today a backend is built from its id alone, lassi/core/runner.py:465 at d35b0a4). Values enter the recipe hash, a key value in api_key_env is refused, and recipes without these keys keep their hashes (tests).
  - Stub servers answer in the reply shapes P17.1 recorded for vLLM, SGLang, and llama.cpp's server. The backend confirms the id before the first request, and provenance records the server's entry fields (max_model_len among them) and any version the server reports, never a key (tests).
  - The Serving Rules bullet for other frameworks is written from the code, with a Decision Log entry.
- Files: `lassi/llm/`, `lassi/core/runner.py`, `lassi/core/recipe.py`, `tests/llm/`, `tests/core/`, `docs/BIBLE.md`.
- Remote: none. Depends: P17.1, P17.2.

### P17.4 hf_local on Transformers and PyTorch
- Bible: Model Serving, Component Interfaces (LLMBackend), Repository Layout (llm/), Risks And Questions (row 1), Agent Rules 7, 10, and 12.
- Accept:
  - A registered LLMBackend hf_local does the following, tested on the CPU with a tiny model built in the test from a config (no download): it loads a model by id and pinned revision from HF_HOME with the hub offline, on the recipe's device (P17.2); applies the tokenizer's chat template; returns the tokenizer's token counts; samples under Sampling with a recorded seed; and fails, never truncates, a prompt longer than the model's context.
  - Without the extra, lassi.llm still registers hf_local, and building it fails with a message that names the extra (test).
  - The extra and its pins go into pyproject.toml and uv.lock, from P17.1. Provenance records the framework versions and the device record. Bible edits with a Decision Log entry.
- Files: `lassi/llm/hf_local.py`, `lassi/llm/__init__.py`, `pyproject.toml`, `uv.lock`, `tests/llm/`, `docs/BIBLE.md`.
- Remote: none. Depends: P17.1, P17.2.

### P17.5 gpu executor and the sandbox's device access
- Bible: Execution Backends (gpu rows), Sandbox, Component Interfaces (Executor, device()), Agent Rules 6, 7, and 13.
- Accept:
  - A registered Executor gpu takes a vendor (nvidia or amd) and device indices, and declares runs_code and sandboxed. It runs only through the sandbox, with one table of device nodes, /sys paths, and visibility variables per vendor (unverified until P17.12) and nothing else. The sandbox's fixed lists gain those names for this executor only (Sandbox edit, Decision Log). Native and ttsim runs see no GPU node (tests).
  - device() names the GPU, its driver, and its index from procfs or sysfs files, starting no process (tests with fake roots). A host without the vendor's nodes is refused before any directory exists. So is one GPU named both by hf_local and by the executor, unless the backend unloads before each run (tests).
  - A remote test on alpha01 uses a harmless stand-in node: a listed node is visible in the sandbox and an unlisted one is not. No AMD node is opened.
- Files: `lassi/executors/`, `lassi/core/runner.py`, `tests/executors/`, `docs/BIBLE.md`.
- Remote: one `rx run` of the remote tests. Depends: P17.2.

### P17.6 hipcc toolchain, compile-only on alpha01
- Bible: Toolchain Pins (a host compiler pinned by version and path), Execution Backends (gpu AMD row), Component Interfaces (Toolchain), Host Facts (ROCm).
- Accept: a registered Toolchain builds HIP sources for gfx942 with alpha01's hipcc.
  - It is pinned by version and path in `toolchains/hipcc.pin`, with the variables the build needs, and --version is checked in the compile sandbox.
  - Diagnostics parse from fixtures captured on alpha01.
  - A remote test builds a small HIP program kept in the tests, compile-only.
  - It is registered OWNER until the owner's review of OQ-039 confirms AMD work without GPU access (OQ-002), and it goes BLOCKED on OQ-040 if hipcc needs a GPU. Gate part (a) needs no hipcc, so P17.11 does not wait for it. If P17.1 deferred its hipcc item, P17.6 first answers it and records it under question 15.
  - It reconciles the ROCm environment guidance: the P17 notes say never to source rocm_env.sh in a pin, while the bible's Host Facts and the gpu (AMD) key settings say to source it; the fix is a bible edit with a Decision Log entry.
- Files: `lassi/toolchains/`, `toolchains/hipcc.pin`, `tests/toolchains/`, `docs/BIBLE.md`.
- Remote: `rx run` for the fixture capture and the remote tests. Depends: P17.1.

### P17.7 Profilers: timing, nvml, rocm_smi
- Bible: Component Interfaces (Profiler; supports_power), Result Record (Attempt.profile), Project Recipes (the lassi-ee profiler line), Design Principle 9.
- Accept:
  - Registered Profilers timing, nvml, and rocm_smi return runtime_s, plus avg_power_w and energy_j where telemetry exists, sampling at the recipe's interval. nvml and rocm_smi read telemetry through a source seam and are tested with fake sources. Idle subtraction and the LASSI-EE windows stay in P9.
  - The runner binds `profiler`, which it refuses today (lassi/core/runner.py:270 at d35b0a4). run_loop profiles each attempt run and fills Attempt.profile. A power profiler whose telemetry or device the run lacks is refused before any directory exists, and recipes without a profiler keep their hashes (tests).
  - Bible edits with a Decision Log entry.
- Files: `lassi/profilers/`, `lassi/core/` (runner, stages, recipe), `tests/profilers/`, `tests/core/`, `docs/BIBLE.md`.
- Remote: none. Depends: P17.2, P17.5.

### P17.8 Trainer interface and the train recipe
- Bible: Training Module, Component Interfaces, Project Recipes (train.yaml), Repository Layout (train/, cli.py), Agent Rules 5 and 10.
- Accept:
  - A Trainer Protocol joins `lassi/core/interfaces.py` and the registry's interface list (lassi/core/registry.py:37-48).
  - `lassi train <train.yaml>` loads a train recipe with the bible's keys plus a device section. It refuses eval and unassigned splits, and refuses any key the layer does not carry out yet, naming the key. Before any step it writes the resolved train recipe, the device record, the framework pins, and the data split hash (tests with a fake trainer).
  - Bible edits (Component Interfaces, which then has thirteen interfaces; Project Recipes) with a Decision Log entry.
  - It is registered READY: the new interface is flagged for the owner's review at P17's planning (Agent Rule 3), and a change the owner asks for there becomes a task before P17.8 starts or after it.
- Files: `lassi/core/interfaces.py`, `lassi/core/registry.py`, `lassi/train/`, `lassi/cli.py`, `tests/train/`, `docs/BIBLE.md`.
- Remote: none. Depends: P17.2.

### P17.9 TRL backend: sft, dpo, and grpo on PyTorch
- Bible: Training Module (Algorithms, Weight Modes, Compute, Safeguards), Agent Rules 5, 6, and 10.
- Accept:
  - A registered Trainer runs a few steps each, on the CPU, with a tiny model built from a config and synthetic data under `tests/fixtures/train/`: sft and dpo, with full and lora weights, and grpo, with in-process generation and a fixture reward, executing nothing.
  - Each run writes a checkpoint directory holding per-step losses and reward components, the resolved train recipe, the device record, and the pins. The train extra's pins go into pyproject.toml and uv.lock. The tests are marked slow.
  - rft, gspo, ppo, adversarial, merge, and export are refused as not carried out (P7, P8, P16).
- Files: `lassi/train/`, `pyproject.toml`, `uv.lock`, `tests/train/`, `tests/fixtures/train/`, `docs/BIBLE.md`.
- Remote: none. Depends: P17.1, P17.8.

### P17.10 CI on the CPU
- Bible: Attribution Policy (Enforcement); AGENTS.md Repository Checks.
- Accept: a workflow file beside text-policy.yml (which stays unchanged) installs the CPU extras and runs the device-neutral tests of P17.2 to P17.9, remote tests excluded, on a Linux runner with no GPU. A push of the phase branch runs it green (run URL in the task note), and the text-policy check passes on the new file.
- Files: `.github/workflows/` (a new file). Remote: none (GitHub). Depends: P17.4, P17.9.

### P17.11 Gate part (a): recipes and the hardware-free evidence
- Bible: Build Roadmap (P17 row, Gate part (a)), Agent Rules 1, 7, and 10; AGENTS.md Phase Gate and Results.
- Accept, from one clean commit on alpha01:
  - `rx doctor` and du are recorded. The model from P17.1 is fetched once under HF_HOME at its revision, in a named fetch step.
  - `lassi run tests/fixtures/recipes/p17-cpu-hf.yaml` (hf_local on cpu; one HeCBench item in both directions; cuda: none, omp: native) exits 0 with every trial past the baseline. provenance.json records the model's device record and framework pins.
  - The sft, dpo, and grpo smoke recipes run through `lassi train` on the CPU and exit 0, each checkpoint with its records. `lassi run` of each p17-gpu-*.yaml recipe is refused before any directory exists, naming the missing device.
  - `rx pull` writes `results/p17-gate-a/` with provenance.json, and summary.md quotes gate part (a) and names every device.
  - A pull request `P17 Portable Stack, part (a)` goes from p17-portable to main (OQ-038). The phase stays ACTIVE with only OWNER or BLOCKED tasks left (the GPU tasks, and P17.6 while it waits for the owner's review of OQ-039), so P5 may start.
- Files: `tests/fixtures/recipes/p17-cpu-hf.yaml`, `tests/fixtures/recipes/p17-gpu-*.yaml`, the sft, dpo, and grpo smoke recipes under `tests/fixtures/recipes/`, `results/p17-gate-a/`.
- Remote: `rx doctor`, `rx run`, `rx job start` and `rx job wait`, `rx pull`. Depends: P17.3, P17.4, P17.5, P17.7, P17.9, P17.10.

### P17.12 Spike on the first GPU host (BLOCKED: GPU host, OQ-040)
- Bible: Execution Backends (gpu rows), Sandbox, Environment State, Host Facts, Agent Rules 6 and 7.
- Accept: `plans/spikes/p17-gpu-host.md`, from a clean commit on the host OQ-040 names, gives:
  - the GPU inventory: model, count, driver, runtime, and any other tenant's use;
  - the device nodes, /sys and /proc paths, and variables that the CUDA or HIP runtime and PyTorch need inside the sandbox, measured, not read from docs;
  - the probe's record as a fixture;
  - whether power telemetry is available;
  - the gate class the owner enabled.
  Environment State gains the host's row, with a Decision Log entry.
- Files: `plans/spikes/p17-gpu-host.md`, the probe fixture under `tests/fixtures/`, `docs/BIBLE.md`.
- Remote: `rx devcheck`, `rx doctor`, `rx run`. Depends: P17.5.

### P17.13 GPU runs: executor, profilers, hf_local, training (BLOCKED: GPU host, OQ-040)
- Bible: Execution Backends, Sandbox, Model Serving, Training Module (Compute), Agent Rules 1, 6, and 13.
- Accept:
  - The gpu executor's device table is set from P17.12 (Sandbox edit, Decision Log).
  - Remote tests on the GPU host show that a CUDA or HIP program runs clean in the sandbox with no other GPU visible, the vendor profiler reads power during the run, hf_local generates on the GPU, and the TRL smoke runs on the GPU.
  - Captured probe fixtures replace the fakes wherever they differ.
- Files: `lassi/executors/`, `tests/fixtures/`, `docs/BIBLE.md`.
- Remote: `rx run` of the remote tests on the GPU host. Depends: P17.4, P17.7, P17.9, P17.12.

### P17.G Phase gate (BLOCKED: GPU host, OQ-040)
- Bible: Build Roadmap (P17 row, Gate column), Agent Rules 1 and 10; AGENTS.md Phase Gate and Results.
- Accept, from one clean commit:
  - Part (a) reruns on alpha01, as in P17.11.
  - Part (b) runs on the GPU host: the p17-gpu recipe on cuda or rocm exits 0 with every trial past the baseline; the gpu executor runs the item's references in the sandbox, with the profiler's runtime and power; and the training smoke runs on the GPU.
  - `results/p17-gate/` names every device, driver, runtime, and framework pin in its provenance.
  - P17 is set DONE, and the pull request `P17 Portable Stack` goes to main.
- Files: `results/p17-gate/`, `plans/STATUS.md`.
- Remote: `rx doctor`, `rx run`, `rx job start` and `rx job wait`, `rx pull` on both hosts. Depends: P17.11, P17.13.
