# P17 Portable Stack

Branch `p17-portable`. Base branch: main at 0dcc261 (P4 DONE, PR 4 merged). Code references name functions and constants as they stand at 0dcc261.

## Scope

P17 builds what the P17 row of the bible's Build Roadmap names: the owner's request of 2026-09-27 to run LASSI on traditional hardware and to ready the stack for regular ML frameworks on other devices (Purpose And Scope). That is a device-neutral layer, with explicit device selection and capability probes recorded in provenance (Agent Rules 1 and 10; Result Record, provenance); framework-neutral serving (Model Serving; Component Interfaces, LLMBackend); a gpu executor for CUDA and HIP in the sandbox, plus hipcc (Execution Backends, gpu rows; Sandbox; Toolchain Pins); the timing, nvml, and rocm_smi profilers (Component Interfaces, Profiler; Result Record, Attempt.profile); a trainer interface on PyTorch (Training Module); and CI on the CPU.

Gate (the P17 row's Gate column, verbatim): "(a) Without a GPU, on alpha01's CPU: a recipe that binds hf_local on PyTorch CPU runs one HeCBench item in both directions with every trial past the baseline, and provenance records the model's device probe and framework pins; `lassi train` runs sft, dpo, and grpo briefly on the CPU with a tiny model on synthetic data, each checkpoint recording its device and pins; a recipe that names a GPU is refused on a host without one. Part (a)'s pass opens a pull request for the hardware-free half (OQ-038). (b) On a GPU host: the same recipe on cuda or rocm, one item's references run by the gpu executor in the sandbox with the vendor profiler's runtime and power, and the training runs on the GPU". Part (a) runs as P17.11; part (b) runs as P17.G, which waits for OQ-040.

Relation to other phases: P17 builds the layer that P7 to P10 use and takes none of their scope. P7 (sft, rft, dpo with RNGD sampling, export) and P8 (grpo and gspo with vLLM rollouts at scale) train through P17's trainer interface. P9 keeps the LASSI-EE stages and power protocol and measures with P17's rocm_smi profiler. P10 runs the reproduction matrix on P17's gpu executor. P3 keeps RNGD serving and uses P17's recipe keys for servers.

Constraints for every task:

- Hardware: no GPU is reachable. The MI300Xs refuse /dev/kfd (OQ-002), and there is no NVIDIA host (OQ-003); see Environment State. P17.12, P17.13, and P17.G are BLOCKED on OQ-040 with no host planned: the owner chose the MI300X on alpha01 (2026-10-05), then said on 2026-10-06 that there is no MI300X access for now and to focus on Tenstorrent and CPU, and on 2026-10-07 that access may come in the future, so the phase stalls after gate part (a) and its pull request (OQ-038). Nothing on alpha01 opens /dev/kfd or a /dev/dri node, a probe included: alpha01's MI300X nodes exist, so a GPU device is refused there from the framework build and file metadata alone, and sandbox device tests use a harmless stand-in node. No RNGD card is used.
- The scope is fixed at planning. [MEASURED] needs a clean-commit rx id, with clean or dirty read from the rx record, never from the run tree's own provenance; rx output stays under 120 lines, and every rx call sets PYTHONIOENCODING=utf-8 (plans/runs/p0-retrospective.md; PHASE-NOTES, All Phases). A later finding goes into PHASE-NOTES or the owner queue.
- Frameworks stay out of the default environment. They are optional extras (P17.1 names them), imported only when a component is built. A plain `uv sync` keeps today's control plane, and lassi.llm still registers hf_local (PHASE-NOTES P0, registration rule).
- Agent Rule 5: training data is synthetic fixtures, never a bench item, and `lassi train` reads a bench item only through the bench registry for purpose train, which refuses eval and unassigned items (lassi/bench/registry.py, Suite.item; OQ-025). Agent Rule 6: generated code runs only in the sandbox; the grpo smoke reads a fixture reward and executes nothing. Agent Rule 7: HF_HOME, the torch and Triton caches, downloads, and checkpoints live under the scratch root or the runs root; run `rx doctor` and du before any install. Agent Rule 12: API keys and HF tokens are named only by environment variable.
- Scratch sizes are PROJECTED until P17.1 measures them: the CPU framework extras about 1 GB per environment, and the gate model under 2 GB. A step that would pass 115G stops and queues candidates for the owner.
- Unattended sessions run with graphics off (`--graphics off` or LASSI_GRAPHICS=off).
- Working practices (plans/runs/p4-retrospective.md, Changes for P17 and later): each task gets one review round, its lenses run in parallel, and one commit audit; a second round only for a confirmed code defect, and a text blocker is fixed and committed with no re-review (the commit body lists it). A task's design maps each component to an acceptance bullet and takes the smallest design that meets them; a larger option goes to PHASE-NOTES. Limits text opens with "known cases, not complete", and a case missing from it is advisory. A PHASE-NOTES hint says what to do and points to the bible, never restating behavior. A value read from upstream source cites the line of code that sets or compares it, never a comment. Multi-item work gets a shared framework first, then one agent per item in its own rx slot. One multi-agent run at a time, of about 15 agents at most unless the brief says why. After a task's evidence commit, the next item comes from `tools/status.py next`, and a stretch that ends at a usage limit gets its run report at the next session's start.

Decisions taken at planning (the named task records each in the bible with a Decision Log entry):

- Explicit devices (P17.2). A recipe names each device as a kind (cpu, cuda, rocm) and indices, with no default. The run probes each named device before any directory exists and refuses one the host lacks; it never falls back to the CPU. The probe's record goes into provenance.json and each trial's provenance: kind, name, count, memory, driver and runtime versions, and framework and version. Today _provenance (lassi/core/runner.py) writes `driver` as null, and _trial_provenance copies it into each trial's provenance.sdk. ROCm PyTorch reaches its GPUs through torch's cuda device type, so kind rocm is checked against a HIP build of the framework.
- Placement (Design Principle 9). Device-neutral types live in lassi/core and import no framework. Framework probes sit beside their adapters (lassi/llm/hf_local, lassi/train). Vendor telemetry and device files are read only in lassi/executors and lassi/profilers.
- One GPU, one role (P17.5). The runner refuses a run in which hf_local and a gpu executor name the same GPU, unless the backend declares unload_before_run. This carries Agent Rule 13's intent to GPUs.
- Trainer (P17.8). A Trainer Protocol joins lassi/core/interfaces.py as the thirteenth interface, since the Training Module already calls lassi/train an interface that verl or OpenRLHF can replace. The owner confirmed it (OQ-043, entry 14; Component Interfaces; Agent Rule 3).
- Gate drivers. Part (a) uses tests/fixtures/recipes/p17-cpu-hf.yaml, modeled on projects/lassi-demo/rngd-cpu.yaml (OpenMP built by the proxy nvcpp-multicore, so not faithful), with hf_local on the CPU, the model P17.1 pins by revision, and executors per language (cuda: none, omp: native; Project Recipes, Notes), plus `lassi train` smoke recipes. Part (b) runs the same recipes with a GPU kind.
- CI (P17.10). A new workflow file sits beside text-policy.yml, which stays untouched. Making it a required check is the owner's repository setting.
- Owner reviews. OQ-043 brought the choices applied without the owner's answer back at this planning (PHASE-NOTES P5 and P17, owner review first): OQ-027 to OQ-033, OQ-036 to OQ-039, OQ-041, and OQ-042, plus three choices of this plan, the Trainer Protocol (P17.8), AMD work without GPU access (P17.1's hipcc item, P17.6), and Design Principle 2's wording against OQ-041 (a). The owner ratified every entry on 2026-10-05, with two conditions on OQ-028: an RL training data loader excludes alignment_missing attempts (a P7 note), and the guard's not-run reasons and not-checked rates are recorded (task P17.14) (bible Decision Log, 2026-10-05).

Owner queue: OQ-038 (the work order and the partial pull request) and OQ-039 (frameworks and devices: NVIDIA CUDA kept in the device layer, no Apple MPS or Intel XPU now) stand with option (a) by the owner's answers of 2026-10-05, and both are CLOSED. OQ-040 (a GPU host) is answered with option (a), the MI300X on alpha01, and stays OPEN until the owner's render group grant shows as rocm_gpu enabled in rx doctor; it blocks P17.12, P17.13, and P17.G.

PHASE-NOTES P17 items, by task: the owner request and the work order, this plan and P17.11 (the P17 row and WORK_ORDER exist since 2026-09-28); the owner review, OQ-043, with P17.1, P17.6, and P17.8; the missing seams, P17.2 (driver), P17.3 (HTTP settings, also PHASE-NOTES P0), P17.4 (hf_local), P17.7 (profiler binding), and P17.8 and P17.9 (lassi/train); the registration rule, P17.4, P17.7, and P17.9; the sandbox's /dev, /sys, and allowlist, P17.5, P17.12, and P17.13; AMD on alpha01, P17.1, P17.5, and P17.6; caches, P17.1, P17.4, P17.9, and P17.11; offline tests with a tiny model, P17.4, P17.9, and P17.10; synthetic data, P17.8 and P17.9; the gate-wording note (a GPU kind refused on a host whose nodes exist but refuse access), P17.2 and P17.11. Not taken: PHASE-NOTES P4's Windows console follow-up for the live table (P4.8 recheck) is outside the P17 row; P17.8 adds `lassi train` to lassi/cli.py without touching the live table, so the follow-up waits for P7's training table.

## Tasks

### P17.1 Spike: framework pins, environments, and device assumptions
- Bible: Toolchain Pins, Repository Layout (placement rules), Training Module (Compute), Host Facts (ROCm), Risks And Questions (questions 6 and 15), Agent Rules 7 and 10.
- Accept: `plans/spikes/p17-frameworks.md` gives the following, each with its commands, outputs, and sources:
  - a PyTorch, Transformers, TRL, PEFT, and Accelerate set that installs together under Python 3.10 for the CPU, and the CUDA and ROCm wheel builds that pair with it (from index metadata);
  - whether one uv lock can hold the CPU flavor beside the GPU flavors, or the GPU flavors need their own environment;
  - the measured size of the CPU set in a scratch venv;
  - the `/v1/models` and usage replies of vLLM, SGLang, and llama.cpp's server at pinned versions (from docs or source);
  - whether alpha01's hipcc builds gfx942 code without a GPU, and which variables its environment script sets (read, never sourced); the owner allowed this AMD work, which opens no GPU device (OQ-043, entry 15; OQ-039 (a));
  - a small instruct model pinned by revision for gate part (a), with its size;
  - every place in lassi/ that assumes RNGD, Tenstorrent, or a device by name, with file:line at the spike's commit.
  The framework pins join Toolchain Pins and question 15 is answered, with a Decision Log entry.
- Files: `plans/spikes/p17-frameworks.md`, `docs/BIBLE.md`.
- Remote: `rx doctor`; read-only `rx exec`; one `rx run` that installs the CPU set in a scratch venv and measures it. Depends: none.

### P17.2 Device layer: selection, probes, provenance
- Bible: Result Record (provenance, Storage), Project Recipes (Notes), Component Interfaces (contract rules), Agent Rules 1, 3, and 10.
- Accept:
  - Any component that takes a device section (hf_local's model, a gpu executor's config, a train recipe) gets one shape: kind and indices. It resolves into the recipe hash. A missing or unknown kind, or duplicate indices, is refused at load with the key named (tests).
  - A probe seam returns the device record. With fake probes, a named device the host lacks is refused before any directory exists, and nothing falls back to the CPU. No probe opens a device node, so a GPU kind on a host whose nodes exist but refuse access (alpha01, OQ-002) is refused from the framework build and file metadata (tests with a fake root).
  - provenance.json and each trial's provenance record the device record, its driver and runtime versions among them (the `driver` that _provenance writes as null today). Recipes without a device section keep their resolved recipes and hashes (golden tests unchanged), and older trial.json files load (tests).
  - Bible edits (Result Record, Project Recipes) with a Decision Log entry.
- Files: `lassi/core/` (a devices module, recipe, runner, record, trial_md), `tests/core/`, `docs/BIBLE.md`.
- Remote: none. Depends: none.

### P17.3 Served backends: recipe keys and serving provenance
- Bible: Model Serving (Serving Rules), Project Recipes (Notes), Agent Rule 12; PHASE-NOTES P0 (HTTP settings).
- Accept:
  - model.base_url, model.timeout_s, and model.api_key_env reach openai_compat and ollama through the runner. Today the recipe loader's model section takes only backend and id (lassi/core/recipe.py), and _prepare (lassi/core/runner.py) builds a backend as factory(settings.model_id). Values enter the recipe hash, a key value in api_key_env is refused, and recipes without these keys keep their hashes (tests).
  - Stub servers answer in the reply shapes P17.1 recorded for vLLM, SGLang, and llama.cpp's server. The backend confirms the id before the first request, and provenance records the server's entry fields (max_model_len among them) and any version the server reports, never a key (tests).
  - The Serving Rules bullet for other frameworks is written from the code, with a Decision Log entry.
- Files: `lassi/llm/`, `lassi/core/runner.py`, `lassi/core/recipe.py`, `tests/llm/`, `tests/core/`, `docs/BIBLE.md`.
- Remote: none. Depends: P17.1, P17.2.

### P17.4 hf_local on Transformers and PyTorch
- Bible: Model Serving (Serving Rules), Component Interfaces (LLMBackend), Repository Layout (llm/), Risks And Questions (row 1), Agent Rules 7, 10, and 12.
- Accept:
  - A registered LLMBackend hf_local does the following, tested on the CPU with a tiny model built in the test from a config (no download): it loads a model by id and by the revision the recipe pins (a model key that reaches the backend as P17.3's keys do and enters the recipe hash) from HF_HOME with the hub offline, on the recipe's device (P17.2); applies the tokenizer's chat template; returns the tokenizer's token counts; samples under Sampling with a recorded seed; and fails, never truncates, a prompt longer than the model's context.
  - Without the extra, lassi.llm still registers hf_local (its docstring lists four backends today, lassi/llm/__init__.py), and building it fails with a message that names the extra (test).
  - The extra and its pins go into pyproject.toml and uv.lock, from P17.1. Provenance records the framework versions and the device record. Bible edits with a Decision Log entry.
- Files: `lassi/llm/hf_local.py`, `lassi/llm/__init__.py`, `lassi/core/recipe.py`, `pyproject.toml`, `uv.lock`, `tests/llm/`, `docs/BIBLE.md`.
- Remote: none. Depends: P17.1, P17.2, P17.3.

### P17.5 gpu executor and the sandbox's device access
- Bible: Execution Backends (gpu rows), Sandbox, Component Interfaces (Executor, device()), Project Recipes (Notes, executor forms), Agent Rules 6, 7, and 13.
- Accept:
  - A registered Executor gpu takes a vendor (nvidia or amd) and device indices, and declares runs_code and sandboxed. It runs only through the sandbox, with one table per vendor of device nodes, /sys paths, and visibility variables (unverified until P17.12). Today the private /dev binds six fixed nodes, the setup covers /sys/class, /sys/bus, and every /sys/devices entry but system (lassi.executors.sandbox, _SETUP_TEMPLATE), and ENVIRONMENT_NAMES holds no visibility variable. Only runs the gpu executor starts expose its table's nodes and /sys paths, never by default; the visibility variables join ENVIRONMENT_NAMES, as TT_METAL_NAMES did for ttsim (P4.11), and only the gpu executor sets them (Sandbox edit, Decision Log). Native and ttsim runs see no GPU node (tests).
  - device() names the GPU, its driver, and its index from procfs or sysfs files, starting no process (tests with fake roots). A host without the vendor's nodes is refused before any directory exists. So is one GPU named both by hf_local and by a bound gpu executor (with executors per language, any of them), unless the backend unloads before each run (tests).
  - A remote test on alpha01 uses a harmless stand-in node: a listed node is visible in the sandbox and an unlisted one is not. No AMD node is opened.
- Files: `lassi/executors/` (a gpu module, sandbox.py), `lassi/core/runner.py`, `tests/executors/`, `docs/BIBLE.md`.
- Remote: one `rx run` of the remote tests. Depends: P17.2.

### P17.6 hipcc toolchain, compile-only on alpha01
- Bible: Toolchain Pins (a host compiler pinned by version and path), Execution Backends (gpu AMD row), Component Interfaces (Toolchain), Sandbox, Host Facts (ROCm).
- Accept: a registered Toolchain builds HIP sources for gfx942 with alpha01's hipcc.
  - It is pinned by version and path in `toolchains/hipcc.pin`, as `toolchains/gcc.pin` pins the host g++, with the variables the build needs. They reach the compile sandbox only through the pin and the sandbox's fixed allowlist (lassi.toolchains.pins PREFIX_VARIABLES feeds ENVIRONMENT_NAMES today; a new name is a Sandbox edit with a Decision Log entry), and --version is checked in the compile sandbox.
  - Diagnostics parse from fixtures captured on alpha01.
  - A remote test builds a small HIP program kept in the tests, compile-only.
  - It is registered READY: the owner allowed AMD work that opens no GPU device (OQ-043, entry 15; OQ-039 (a)). It goes BLOCKED on OQ-040 if hipcc needs a GPU. Gate part (a) needs no hipcc, so P17.11 does not wait for it.
  - It reconciles the ROCm environment guidance: the P17 notes say never to source rocm_env.sh in a pin, while the bible's Host Facts and the gpu (AMD) key settings say to source it; the fix is a bible edit with a Decision Log entry.
- Files: `lassi/toolchains/`, `toolchains/hipcc.pin`, `tests/toolchains/`, `docs/BIBLE.md`.
- Remote: `rx run` for the fixture capture and the remote tests. Depends: P17.1.

### P17.7 Profilers: timing, nvml, rocm_smi
- Bible: Component Interfaces (Profiler; supports_power), Result Record (Attempt.profile), Project Recipes (the lassi-ee profiler line), Design Principle 9.
- Accept:
  - Registered Profilers timing, nvml, and rocm_smi return runtime_s, plus avg_power_w and energy_j where telemetry exists, sampling at the recipe's interval. nvml and rocm_smi read telemetry through a source seam and are tested with fake sources. Idle subtraction and the LASSI-EE windows stay in P9.
  - The runner binds `profiler`, which it refuses today (lassi/core/runner.py, _NOT_CARRIED_OUT, checked in _check_carried_out; the recipe loader already binds the key to Profiler). run_loop profiles each attempt run and fills Attempt.profile. A power profiler is refused before any directory exists when the run lacks its telemetry or when no executor whose runs it profiles runs on its device (with executors per language, each profiled language's executor), and recipes without a profiler keep their hashes (tests).
  - Bible edits with a Decision Log entry.
- Files: `lassi/profilers/`, `lassi/core/` (runner, stages, recipe), `tests/profilers/`, `tests/core/`, `docs/BIBLE.md`.
- Remote: none. Depends: P17.2, P17.5.

### P17.8 Trainer interface and the train recipe
- Bible: Training Module, Component Interfaces, Project Recipes (train.yaml), Repository Layout (train/, cli.py), Readability Standards (Terminal Presentation), Agent Rules 5 and 10.
- Accept:
  - A Trainer Protocol joins `lassi/core/interfaces.py` and the registry's INTERFACES (lassi/core/registry.py), and the tests that count twelve interfaces follow (tests/core/test_interfaces.py, tests/core/test_recipe.py).
  - `lassi train <train.yaml>` loads a train recipe with the bible's keys plus a device section. It refuses a bench split of eval or unassigned at load and reads any bench item through the bench registry for purpose train (Suite.item, EvalSplitError), and it refuses any key the layer does not carry out yet, naming the key. Before any step it writes the resolved train recipe, the device record, the framework pins, and the data split hash under the runs root, by the run tree's location rules (lassi/core/runner.py, _check_location) (tests with a fake trainer).
  - `lassi train` decides graphics as every command does (lassi/cli.py) and prints plain lines; the live training table stays P7's.
  - Bible edits (Component Interfaces, which then has thirteen interfaces; Project Recipes) with a Decision Log entry.
  - The owner confirmed the thirteenth interface (OQ-043, entry 14; Agent Rule 3).
- Files: `lassi/core/interfaces.py`, `lassi/core/registry.py`, `lassi/train/`, `lassi/cli.py`, `tests/core/`, `tests/train/`, `docs/BIBLE.md`.
- Remote: none. Depends: P17.2.

### P17.9 TRL backend: sft, dpo, and grpo on PyTorch
- Bible: Training Module (Algorithms, Weight Modes, Compute, Safeguards), Agent Rules 5, 6, and 10.
- Accept:
  - A registered Trainer runs a few steps each, on the CPU, with a tiny model built from a config and synthetic data under `tests/fixtures/train/`: sft and dpo, with full and lora weights, and grpo, with in-process generation and a fixture reward, executing nothing.
  - Each run writes a checkpoint directory holding per-step losses and reward components, the resolved train recipe, the device record, and the pins. The train extra's pins go into pyproject.toml and uv.lock. The tests are marked slow.
  - Every method, weight mode, and train recipe key or value that the Training Module and Project Recipes (train.yaml) name beyond sft, dpo, and grpo with full and lora weights is refused as not carried out, naming it (P7, P8, P16).
- Files: `lassi/train/`, `pyproject.toml`, `uv.lock`, `tests/train/`, `tests/fixtures/train/`, `docs/BIBLE.md`.
- Remote: none. Depends: P17.1, P17.8.

### P17.10 CI on the CPU
- Bible: Attribution Policy (Enforcement); AGENTS.md Repository Checks.
- Accept: a workflow file beside text-policy.yml (which stays unchanged) installs the CPU extras and runs the device-neutral tests of P17.2 to P17.9, remote tests excluded, on a Linux runner with no GPU. A push of the phase branch runs it green (run URL in the task note), and the text-policy check passes on the new file.
- Files: `.github/workflows/` (a new file). Remote: none (GitHub). Depends: P17.4, P17.9.

### P17.14 Guard not-run reasons and not-checked rates
- Added at planning from the owner's OQ-028 answer (2026-10-05); numbered last to keep the other task numbers.
- Bible: Result Record (Guards), Harness Contract (the CPU -> TT guard), Evaluation Protocol (Run Metrics), Agent Rule 1.
- Accept:
  - Every attempt whose Attempt.guards.host_compute is null records why: not built, the build gave no program, the target language's toolchain declares no guard, or the guard ran and could not read the host text or found no output write (today's guard-not-checked note). Records written before this task load unchanged, and trial.md and the Parquet attempts table show the reason (tests).
  - Run metrics report the guard's not-checked rate per run and per target language, with the count of each reason, in run.md and the metric tables; a target whose toolchain declares no guard reports every attempt not checked for that reason (tests).
  - Bible edits (Result Record, Run Metrics) with a Decision Log entry.
- Files: `lassi/core/` (record, stages, trial_md, parquet), `lassi/analysis/`, `tests/core/`, `tests/analysis/`, `docs/BIBLE.md`.
- Remote: none. Depends: none.

### P17.15 The sandbox's Tenstorrent exposure: every card in the cluster
- Added on 2026-10-23 from the owner's OQ-072 answer ("I want the sandbox to have access to all Tenstorrent cards in the cluster. It needs to happen."), with plans/spikes/tt-silicon-executor.md as its evidence; the settings OQ-075 asks are explicit, with no default. Built on branch p16-adversarial, which carries P17's code.
- Bible: Execution Backends (the tt_silicon row), Sandbox, Agent Rules 6, 7, and 9.
- Accept:
  - A Tenstorrent exposure in the sandbox spec (beside the gpu executor's DeviceExposure):
    - every /dev/tenstorrent/<n> node, bound at its own path;
    - read-only: /sys/bus/pci/slots, each card's PCI sysfs directory, /sys/module/tenstorrent/version, and the 1 GiB nr_hugepages file;
    - one writable hugetlbfs bind at /dev/hugepages-1G of the run's own hugepage directory (OQ-075 (1));
    - a mount-check exception for exactly that one mount;
    - an optional seccomp rule refusing the driver's reset ioctl (OQ-075 (4)).
  - A run without the exposure keeps the setup script byte for byte (test).
  - Windows tests of the script text and the spec checks; remote tests with stand-in nodes and directories on Linux, opening no device.
- Files: lassi/executors/sandbox.py, tests/executors/.
- Host: Windows, then alpha01 with stand-ins. Depends: none.

### P17.16 The tt_silicon executor
- Bible: Execution Backends (the tt_silicon row), Component Interfaces (Executor), Agent Rules 2, 6, 9, and 10.
- Accept: a registered Executor `tt_silicon` that declares runs_code and sandboxed and runs a TT host program on the cluster's Wormhole cards through P17.15's exposure.
  - Settings: hugepages mode (OQ-075 (1)) and reset filter (OQ-075 (4)), each a required choice with no default.
  - Environment: the ttsim row's settings without TT_METAL_SIMULATOR (OQ-075 (8)).
  - Lock: one cross-process lock for all cards under $LASSI_SCRATCH/locks (OQ-075 (2)).
  - Hugepages: a per-run hugepage directory, created and emptied before and after each run.
  - Quarantine: after a hang, kill, or crash, a marker refuses every later run until a human removes it (OQ-075 (3)).
  - Human start: the runner refuses the executor unless the command line names it with a flag that a recipe cannot set (OQ-075 (6)), and training and reward stages refuse it.
  - Provenance records every card's serial and firmware, and its results name silicon as the device; wall times are never performance.
  - Windows tests with fakes.
- Files: lassi/executors/, lassi/core/runner.py, lassi/cli.py, tests/executors/, tests/core/, docs/BIBLE.md.
- Host: Windows. Depends: P17.15.

### P17.17 Silicon qualification and Tier A with the cards in the loop
- Accept: in the owner's window, runs the owner starts (Agent Rule 9), with results under results/ with provenance:
  - the loopback reference through tt_silicon, the qualification;
  - the two translations that passed on ttsim on 2026-10-23, Llama-3.1-8B's cpp -> tt eltwise_binary and EXAONE-4-32B's matmul_single_core, from their stored replies;
  - Tier A cpp -> tt with tt_silicon as the TT executor and RNGD-served models.
- Files: projects/lassi-demo/, results/.
- Host: alpha01 in the owner's window. Depends: P17.16; OQ-075; OQ-078 for the two stored translations and Tier A, which are model output (plans/spikes/tt-card-dma.md: under iommu=pt a program holding a card node can reach host memory).

### P17.18 tt-train-v0: training records from tt-metal's non-test examples
- Added on 2026-10-23 from the owner's OQ-076 answer, option (1a): train on tt-metal's example programs at the pin, without the five Tier A programs.
- Bible: Training Module (a new Training Data subsection, Safeguards), Project Recipes (Notes, train recipes), Benchmark Suites (tt-pairs-v0: no tt-metal text is tracked), Agent Rules 1, 5, and 10.
- Accept:
  - tools/build_tt_train.py fetches tt_metal/programming_examples at 5280a9cf from GitHub, each file checked against its git blob sha1 (or reads an install of the pin), into a directory outside the repository, and builds one SFT record {prompt, completion} per included example: the task in this project's words and the files to write, then the example's host program and kernels as FILE blocks.
  - The spec, assets/train/tt-train-v0/dataset.yaml, lists every directory under the examples root as included (task, where it was read from, an optional inclusion judgment) or excluded (reason); a code file no listed directory covers is refused.
  - Agent Rule 5, enforced by the build and its tests: every eval and unassigned item's example directory excluded; each duplicate of a Tier A task excluded with its reason, whatever its route (NoC_tile_transfer and profiler/test_noc_event_profiler as loopback's, the matmul and elementwise-add variants as the matmul items' and eltwise_binary's); no record holds a run of 3 consecutive non-blank lines of a tracked Tier A program (test); no completion file is a copy or a near copy of a Tier A upstream file.
  - Only the spec and the generated manifest (the pin, every source file's sha256, the references and parameters, every exclusion, the records' sha256, size, and count) are tracked; the records go under $LASSI_SCRATCH/train-data, never into git.
  - A data source data.corpus (lassi.train.data load_corpus) reads them only when their sha256 and count are the manifest's, and refuses a manifest that does not exclude every eval and unassigned item of its suite; the split hash covers the corpus name, the records' sha256, and the manifest's.
  - The proposed bible text, with its Decision Log row, goes to the main session for bible-sync.
- Files: tools/build_tt_train.py, assets/train/tt-train-v0/, lassi/train/data.py, lassi/train/run.py, lassi/core/recipe.py (the data section and DATA_SOURCES), tests/tools/test_build_tt_train.py, tests/train/test_corpus_data.py, tests/core/test_train_recipe.py.
- Host: Windows (the fetch reads GitHub; the records are built on the workstation for the tests and on alpha01 by plans/runs/train-window/run.sh in a window). Depends: none.

### P17.19 Trainer modes on the CPU: pinned Hub bases, full, LoRA, prompt tuning, from scratch
- Added on 2026-10-23 from the owner's OQ-076 answer ("Full model training, frozen weights, LORA, and potentially training our own model as far reaching goal"); CPU only, by the owner's direction of 2026-10-23 ("At this point, don't train on the 3080. Just train if possible during our testing window."): training runs on alpha01's CPU in the owner's windows only, and the trl trainer's device stays kind cpu.
- Bible: Training Module (Weight Modes, Compute, Safeguards), Project Recipes (the train.yaml block's weights comment, Notes), Model Serving (Serving Rules, the in-process model), Agent Rules 1, 5, 7, 10, and 17.
- Accept:
  - base_model takes `<owner>/<name>@<40-hex revision>` (a Hub id without one is refused, Agent Rule 10) and `scratch:<name>` (a config under assets/train/scratch/ with its pinned tokenizer, random weights under the seed, full weights only) beside `fixture:<name>`; Hub snapshots load from $HF_HOME/hub only, and check() refuses a missing snapshot naming the prepare step.
  - A separate prepare command, tools/prepare_train.py, is the one place a base is downloaded (config, tokenizer, and safetensors files at the pinned revision), never inside a training run.
  - The weight mode prompt_tuning (the frozen-weights mode: PEFT prompt tuning with trainer.virtual_tokens, sft only) beside full and lora; optional trainer.max_length (an sft record TRL would cut is refused before any step), trainer.gradient_checkpointing, and trainer.chat_format (train in the chat template hf_local serves with), each refused with a bad value; each is a declared option the loader or check() refuses when it does not apply.
  - Four recipes under projects/train/ for alpha01's CPU, one per technique (full weights on Qwen3-0.6B, LoRA and prompt tuning on Qwen3-1.7B, from scratch on a 77M-parameter Qwen3-style config), each base pinned by a full commit id, with PROJECTED time and memory arithmetic in its comments; each passes lassi train's refusals up to the device probe without the framework, and every refusal with a stand-in framework and cache (tests).
  - Every checkpoint's content digest in provenance.json (checkpoint_sha256), and hf_local serving a checkpoint by model.checkpoint and model.checkpoint_sha256 beside the base's model.id and model.revision (full weights from the directory, lora and prompt_tuning as an adapter on the pinned base), refusing a digest mismatch, a mismatched or fixture base, and a prompt_tuning checkpoint whose peft record names no soft prompt length; recipes without the keys load as before (tests).
  - plans/runs/train-window/run.sh runs the four recipes in a window as one big job under a deadline, with the 1-byte core limit read back; plans/spikes/own-model.md plans the own-model goal.
  - The proposed bible text, with its Decision Log rows, goes to the main session for bible-sync (the train.yaml block's weights comment gains prompt_tuning, which tests/core/test_train_recipe.py checks).
- Files: lassi/train/bases.py, lassi/train/prepare.py, lassi/train/trl_trainer.py, lassi/train/run.py, lassi/llm/hf_local.py, lassi/core/digest.py, lassi/core/recipe.py (WEIGHT_MODES, the model section's checkpoint keys), tools/prepare_train.py, assets/train/scratch/, projects/train/, plans/runs/train-window/run.sh, plans/spikes/own-model.md, tests/train/, tests/llm/, tests/core/.
- Host: Windows for the code and its tests; the runs on alpha01's CPU in the owner's window (Agent Rule 17), started by the owner's approval, never on the workstation. Depends: P17.18.

### P17.21 The API reference: the context stage under a template prompt set
- Added on 2026-10-23 from the owner's direction ("Okay, get that built and added."). The pack tt-metalium-5280a9cf is built by tools/build_tt_context.py from tt-metal at the pin; only its manifest and the head file are tracked.
- Bible: Component Interfaces (the stages), Harness Contract, Design Principle 3, Agent Rule 5.
- Accept: a template set that holds summarize_context.txt and knowledge.txt may serve context packs; summarize_context sends the pack under it; the summary goes before every later prompt; a recipe without the pack keeps every prompt byte for byte and its recipe hash (golden tests); the pack names no Tier A file (test).
- Files: lassi/core/stages.py, lassi/core/runner.py, assets/prompts/tt-host-v0/, tools/build_tt_context.py, tests/.
- Host: Windows. Depends: none.

### P17.22 Retrieval over the tt-metal docs in generation and corrections
- Added on 2026-10-23 from the owner's direction ("Ready a RAG of Tenstorrent documentation as well in case this is not sufficient.").
- Bible: Component Interfaces (the stages), Project Recipes, Harness Contract, Agent Rules 4 and 5.
- Accept: an opt-in recipe section `retrieval` (index, k, max_chars, prompts), refused under faithful: true and with a fragment set; passages from a BM25 index over tt-metal's docs at the pin, queried from the kernels and source for generate and from the reported errors for a correction (compiler-guided lookup); passages are dropped whole past max_chars, never cut; each request's passages recorded; a recipe without it keeps every prompt and its hash; the corpus excludes Tier A files (test).
- Files: lassi/core/stages.py, lassi/core/runner.py, lassi/core/recipe.py, lassi/prompts/retrieval.py, tools/build_tt_rag.py, assets/rag/, assets/prompts/tt-host-v0/, tests/.
- Host: Windows. Depends: none.

### P17.23 The feature matrix in the owner's windows
- Added on 2026-10-23 from the owner's direction ("do separate tests for all combinations of features added to base LASSI (RAG, API Reference, training, fuzzing, returning benchmark and tolerance reports)").
- Accept: one recipe per combination of the features built so far, per model, run by plans/runs/rngd-sweep/run.sh in the owner's windows, with tables per combination and provenance under results/; a feature not yet built is listed with its reason in the window's run report, never silently left out. The evening window of 2026-10-23 runs the mismatch report and the fuzzer (four combinations, projects/lassi-demo/sweep5; plans/runs/2026-10-23-evening-window.md); the API reference, retrieval, and training join as P17.21, P17.22, and P17.19 land.
- Files: projects/lassi-demo/, plans/runs/rngd-sweep/, plans/runs/window/, results/.
- Host: Windows, then alpha01 in the owner's windows. Depends: P17.21, P17.22 for their combinations.

### P17.24 Classify the cpp -> tt build errors across the sweep records
- Added on 2026-10-25: the first of the four corrections proposed after the passes of 2026-10-23 and 2026-10-24 (63 of the 65 cpp -> tt trials of those passes ended at the correction cap at S1, a build error; one passed, and one ended at S0, no extractable output; results/rngd-sweep5-20261024, results/rngd-matrix-short-20261024), and the owner's "launch as much as you can".
- Bible: Result Record (the diagnostics), Agent Rules 1 and 5.
- Accept: a tool that reads trial.json records under a runs root and buckets every compile-stage error of every cpp -> tt attempt by a fixed rule (the compiler's message with identifiers and paths abstracted), per model, combination, item, and attempt index, and whether a bucket recurs across corrections; it reads only records, runs nothing, and prints counts, never prompt or reference text; its report for the night's runs goes to results/ with the provenance of the runs it read.
- Files: tools/, tests/tools/, results/.
- Host: Windows for the tool and its tests; alpha01 in a window to read the records (one rx run). Depends: none.

### P17.25 Restarts: a fresh generation after N failed corrections (opt-in)
- Added on 2026-10-25 from the same evidence: in the sweep passes of 2026-10-23 and 2026-10-24 the three cpp -> tt passes came at corrections 0, 0, and 4 (the 4 in the fourth pass, rx 20261022-222829-rngd-sweep4-chain-cf48), every tt -> cpp pass at 0, and no other cpp -> tt trial reached a clean run before the cap (the fifth pass's tables, which hold the earlier passes' rows).
- Bible: Component Interfaces (the loop), Project Recipes (Loop opt-ins), Agent Rules 1 and 4.
- Accept: an opt-in loop key (restart_after, an int from 1 to the cap) under which, after that many corrections without a clean run, the next request is a fresh generation from generate.txt instead of a correction, the count and cap unchanged; refused under faithful: true; off when absent, so every recipe keeps its prompts and hash; each restart recorded as a request of its own kind; tests first.
- Files: lassi/core/stages.py, lassi/core/recipe.py, lassi/core/runner.py, tests/core/.
- Host: Windows. Depends: none.

### P17.26 Declaration lookup: the API pack's declarations for the identifiers a build error names (opt-in)
- Added on 2026-10-25: the API reference reaches the model as its own summary of the pack, which can lose exact signatures; a correction can carry the exact declarations instead.
- Bible: Component Interfaces (the context blocks), Project Recipes, Harness Contract, Agent Rule 5.
- Accept: an opt-in recipe key under which a compile correction under a template set gets, in a templated block, the pack's declarations of each tt-metal identifier its listed errors name (whole declarations, skipped whole past a character budget, never cut); queries only from the errors the prompt shows; refused under faithful: true and without a pack; off when absent with prompts byte for byte unchanged; tests first.
- Files: lassi/core/stages.py, lassi/core/runner.py, lassi/core/recipe.py, assets/prompts/tt-host-v0/, tests/core/.
- Host: Windows. Depends: none.

### P17.27 bf16 rounding in tt -> cpp outputs
- Added on 2026-10-25: tt-host-v0's generate.txt says "A bf16 value is the upper 16 bits of the f32 of the same value", which holds for reading inputs, while the C++ references round to nearest, ties to even, when they write bf16 (assets/bench/tt-pairs-v0/eltwise_sfpu/cpp/eltwise_sfpu.cpp:28-37); a candidate that truncates is off by one bf16 unit and misses a max_abs tolerance. A hypothesis until the generated code is read.
- Bible: Harness Contract, Agent Rule 5 (a generic statement about the format, never an item's answer; the tt-pairs-v0 items stay unassigned until P7, so the statement must rest on tt-metal's own bf16 conversion at the pin, not on item outcomes or an item's reference).
- Accept: P17.24's tool, or a reading of the records, shows whether the tt -> cpp candidates that ran clean and missed the tolerance truncate when they write bf16; if they do, a new template set (tt-host-v0 stays as it is, so earlier results stay comparable) states the rounding generically, with golden tests.
- Files: assets/prompts/, tests/prompts/.
- Host: Windows. Depends: P17.24.
- Amended 2026-10-25 (plans/spikes/p17-tt-cpp-misses.md): the records show no one-unit bf16 miss; five of the ten alignment-0.0 candidates write a rank-0 output after lassi_io_free, and the build errors read name tt-metalium and kernel API headers that gcc-native does not hold. Bible: Harness Contract (lassi_io.h), Toolchain Pins (gcc.pin), OQ-083 (b) (tt-pairs-v0 a development set). Accept: tt-host-v1 is tt-host-v0 plus two generic statements in generate.txt, correct.txt, and correct_mismatch.txt (a cpp program has no tt-metal header or library on its path and does any bf16 conversion, tiling, and tile math itself; lassi_io_free zeroes the array, so each output is written before an input whose dims it uses is freed), tt-host-v0 unchanged, and a test pins v1 = v0 plus the edits.

### P17.28 Reasoning effort as a recipe key, and a reply cut at max_tokens as the empty answer
- Added on 2026-10-25 in the owner's window: at the server's default reasoning effort, every one of gpt-oss-20b's 18 matrix recipes in the relaunched run ended on an HTTP 200 reply with a null content beside its reasoning, a ServingError that stopped each recipe run, with no trial finished (plans/runs/2026-10-25-window.md). The Serving Rules' gpt-oss line asks for a fixed, recorded effort per arm; OQ-080 recommends option (a) and is unanswered, so the key is opt-in and option (b) stays open.
- Bible: Model Serving (Serving Rules, the gpt-oss line), Project Recipes (Notes, a Reasoning effort bullet), Component Interfaces (LLMBackend), Agent Rules 1, 3, 4, and 10.
- Accept, tests first: llm.sampling.reasoning_effort, one of low, medium, high, absent by default and never materialized, so every existing recipe keeps its resolved recipe and hash and every request body its bytes; any other value refused at load naming the key, and any value under faithful: true, set or inherited; the runner passes it to a backend that declares the capability reasoning_effort (openai_compat, replay) and refuses it with any other before any directory exists, reading the capability, never a name; openai_compat sends a top-level reasoning_effort only when set; a reply whose content is JSON null with finish_reason length is the empty answer with the server's token counts, every other null or missing content stays a ServingError, and the reasoning is never returned or recorded as a reply; the record schema unchanged.
- Files: lassi/core/capabilities.py, lassi/core/recipe.py, lassi/core/runner.py, lassi/llm/openai_compat.py, lassi/llm/replay.py, tests/llm/, .github/workflows/cpu-tests.yml, docs/BIBLE.md; the window's projects/lassi-demo/sweep6/gpt-oss-20b-low/ and plans/runs/rngd-sweep/models9.tsv.
- Host: Windows. Depends: none.

### P17.29 Worked examples: complete host programs at the pin in the prompt (opt-in)
- Added on 2026-10-25 under the owner's answer to OQ-083, option (b) (tt-pairs-v0 a development set): the build-error report (results/build-errors-20261025, results/rngd-matrix-20261025) shows the models writing the TT-Metalium host API from memory, an older one, and they never see a host program that compiles at the pin.
- Bible: Project Recipes (Notes, an Examples bullet), Component Interfaces (the context blocks bullet), Result Record (Trial.requests), Agent Rules 1, 4, 5, and 10.
- Accept, tests first: tools/build_tt_examples.py fetches or reads tt-metal's programming examples at the pin and builds tt-examples-v0 from a tracked spec listing only directories of tt-train-v0's include list (never a Tier A item or a duplicate), with that data set's near-copy checks; only the spec and MANIFEST.yaml are tracked; a check mode verifies the fragments. The recipe section `examples` {pack, max_chars, prompts} is opt-in, never materialized, refused under faithful: true and by the runner with a fragment set, a missing or unrendering examples.txt or example.txt, or a pack that fails its manifest; the block goes before generation and, when listed, compile corrections, each example whole or skipped, with one examples note on the Request.
- Files: tools/build_tt_examples.py, lassi/prompts/examples.py, lassi/core/recipe.py, lassi/core/stages.py, lassi/core/runner.py, assets/examples/tt-examples-v0/, assets/prompts/tt-host-v0/, tests/core/, tests/tools/, docs/BIBLE.md.
- Host: Windows, the pack built and checked on the build host before a run. Depends: P17.26.

### P17.30 Exact declarations at generation, nearest names, and the full API pack (opt-in)
- Added on 2026-10-25 from the same evidence and OQ-083 (b): the 40 KiB pack cut mesh_workload.hpp, buffer.hpp, program.hpp, device.hpp, and others, and lookup carried declarations only at compile corrections and only for names that exist.
- Bible: Project Recipes (Notes, the Lookup bullet), Component Interfaces (the context blocks bullet), Result Record (Trial.requests), Agent Rules 1, 4, and 10.
- Accept, tests first: tools/build_tt_context.py --name and --limit-bytes build a fuller pack, tt-metalium-5280a9cf-full at 131072 bytes, with its tracked manifest, the default pack's bytes unchanged; lookup.prompts may list generate, with a required lookup.core of header parts whose declarations the generation prompt carries through core_api.txt (words that name no error), refused by the runner for a header the pack lacks or a template set without core_api.txt; lookup.nearest (1 to 8, with correct) adds, for an error identifier the pack does not declare, its nearest declared names (undeclared.txt); every key absent by default and refused under faithful: true.
- Files: tools/build_tt_context.py, assets/context/tt-metalium-5280a9cf-full/MANIFEST.yaml, lassi/core/recipe.py, lassi/core/stages.py, lassi/core/runner.py, assets/prompts/tt-host-v0/, tests/core/, tests/tools/, tests/prompts/, docs/BIBLE.md.
- Host: Windows, the full pack built and checked on the build host before a run. Depends: P17.26.

### P17.31 The development split for tt-pairs-v0 (OQ-083 (b))
- Added on 2026-10-25: OQ-083's answer (b) makes tt-pairs-v0 a development set; the bible records the decision, while the manifest's splits and lassi.bench.registry SPLITS (train, eval, unassigned) are unchanged until this task.
- Bible: Benchmark Suites (splits), Agent Rule 5.
- Accept: a split value for development items, which the registry gives to evaluation and to prompt and feature development and still refuses to training and corpus harvest; tt-pairs-v0's five items assigned to it; every recipe that reads them with `split: unassigned` moved to it; tests first; a plan for the held-out Tier A set that evaluation needs.
- Files: lassi/bench/registry.py, assets/bench/tt-pairs-v0.yaml, projects/, tests/, docs/BIBLE.md.
- Host: Windows. Depends: none.

### P17.32 The night window of 2026-10-25: the sweep with the new features on 70B-class models
- Added on 2026-10-25 for the owner's window of 20:00 to 02:00 ("We must significantly improve results"): projects/lassi-demo/sweep7 (base, ex, lk, all, and gpt-oss-120b's lowall) on Llama-3.3-70B-Instruct-FP8-dynamic and gpt-oss-120b (OQ-087), 3 samples per item at cap 8, cpp -> tt only (OQ-086), run by plans/runs/night/run-night.sh.
- Accept: the run in the window, its tables, per-trial feature counts, and the classifier's report under results/ with provenance, labeled development results (OQ-083 (b)), and the run report plans/runs/2026-10-25-night.md.
- Files: projects/lassi-demo/sweep7/, plans/runs/night/, plans/runs/rngd-sweep/models10.tsv, results/, plans/runs/.
- Host: alpha01 in the owner's window. Depends: P17.29, P17.30.

### P17.33 The C++ -> TT host skeleton (opt-in)
- Added on 2026-10-25 after the owner's "I cannot overstate how important that we get some meaningful results": in the C++ -> TT trials of 2026-10-25, 79 of 82 ended at the correction cap at S1 (results/rngd-matrix-20261025/summary.md); the host program is the model's only file, since the harness provides the kernels. OQ-096 (skeleton (a)): item-agnostic only, from the pinned headers and tt-metal programming examples outside the five tt-pairs-v0 items, never derived from or compared to their reference host programs. tt-pairs-v0 is a development set (OQ-083 (b)).
- Bible: Project Recipes (Notes: Worked examples, and a new Skeleton bullet), Component Interfaces (the context blocks), Result Record (Trial.requests), Harness Contract (the CPU -> TT guard limits), Repository Layout, Agent Rules 1, 3, 4, 5, and 10.
- Accept: assets/skeletons/tt-host-skeleton-v0/ holds host.cpp (a complete host program at the pin with named BLANK blocks of valid placeholder code), block.txt, and MANIFEST.yaml (sha256 and bytes); a recipe section `skeleton: {name, prompts}` (refused under faithful: true; validated by name against the manifest; its sha256 in the resolved recipe and hash; a direction whose target is not the skeleton's language refused) renders the block in the listed prompts; a recipe without it keeps its hash; the runner refuses a skeleton changed after load before any directory exists; host.cpp compiles at the pin with the ttmetal-host command line from a clean commit on alpha01 (an exploratory compile from a dirty snapshot passed: rx 20261025-191541-desktop-8r113ei-p16-adversarial-d4c2, never [MEASURED]).
- Files: assets/skeletons/, lassi/prompts/skeletons.py, lassi/core/recipe.py, lassi/core/stages.py, lassi/core/runner.py, tests/core/, tests/prompts/, tests/toolchains/, lassi/toolchains/ttmetal_guard.py (its limits docstring), plans/runs/night/skeleton_build.sh, plans/spikes/p17-skeleton-build.md, docs/BIBLE.md.
- Host: Windows; alpha01 for the compile from a clean commit, with the owner's yes in a working session (Agent Rule 17); tonight's one approved build (OQ-092's answer, recorded on branch p9-ee) ran from a dirty snapshot. Depends: P17.29.

### P17.34 The next window's sweep (sweep8)
- Added on 2026-10-25: projects/lassi-demo/sweep8 on both night models: cpp -> tt on tt-host-v1 with the worked examples (control), with the skeleton, and with every feature plus the skeleton; tt -> cpp on tt-host-v0 (control) and tt-host-v1 (P17.27). Every recipe corrects a wrong output until the cap (OQ-098), and sets the simulator-gap correction once P17.35 lands.
- Accept: every sweep8 recipe loads (tests/core/test_faithful_bindings.py), a night script runs them with provenance, and the run's tables come back as development results (OQ-083 (b)).
- Launcher (the owner's direction of 2026-10-26: the full suite launched unattended, with a help option on every command): tools/suite.py (plan, launch, status, wait, stop, pull) and its remote side plans/runs/suite/run-suite.sh, which runs the sweep with run-night.sh's safety steps; launch asks before it starts a job unless --yes, and --silicon is refused.
- Files: projects/lassi-demo/sweep8/, plans/runs/night/, plans/runs/, tools/suite.py, tests/tools/test_suite.py.
- Host: Windows; alpha01 in the next owner window. Depends: P17.27, P17.33, P17.35.

### P17.35 Simulator gaps: reported as not verifiable, and fed back (OQ-098)
- Added on 2026-10-25 (OQ-098, simulator gaps (a)): a simulator gap today ends the trial with sim-gap and no correction (lassi/core/stages.py). The owner chose: record the attempt as built but not verifiable on ttsim (never a fail, listed for a later human-started hardware check, Agent Rule 9) and, as a named opt-in, tell the model which construct the simulator cannot run and continue under the same cap.
- Bible: Project Recipes (Notes: the loop's opt-ins), Result Record, Evaluation Protocol (Reporting Rules), Agent Rules 1, 2, 4, and 9.
- Accept: an opt-in loop key (refused under faithful: true; off when absent, so earlier recipes keep their hashes) feeds a simulator gap back with its class under the same count and cap, through the template set's run-error correction; the sim-gap attempt keeps its record; a report lists every attempt that hit a gap as built but not verifiable on ttsim, counted apart from failures; tests with fakes.
- Files: lassi/core/stages.py, lassi/core/recipe.py, plans/runs/rngd-sweep/tabulate.py, tests/core/, docs/BIBLE.md.
- Host: Windows. Depends: none.

### P17.36 The cpp -> ttkernel path: the model writes ttkernel MLIR, tt-mlir emits the kernels
- Added on 2026-10-26 (the owner, 2026-10-26: "Get the MLIR lowering done and try GPT-OSS with MLIR lowering to tenstorrent MLIR and emit"; the ttkernel level first, since it needs no pin change). The model writes each kernel an item's tracked TT host program launches as a ttkernel MLIR module; the pinned tt-mlir verifies, lowers to EmitC, and translates it to Metalium kernel C++; the tracked host program builds around it and runs on ttsim.
- Bible: IR Strategy (the ttkernel row), Level Rules, Toolchain Pins (the joint pin), Benchmark Suites, Project Recipes (Notes), Harness Contract, Agent Rules 2, 5, 6, and 10.
- Accept: a language ttkernel with the Toolchain ttkernel-ttmetal (tt-mlir's own lit-test commands at 5f396fd6, each tool error a parsed Diagnostic fed back by the correction loop, any model file other than a kernels/**.mlir module refused); the sibling suite tt-pairs-v0-ttkernel with the host program and lassi_io.h as sha256-pinned support; the template set tt-mlir-v0 (the host program shown read-only, never the upstream kernels) with the ttkernel dialect pack and one worked example from tt-mlir's tests, only their manifests tracked; the recipe projects/lassi-demo/mlir1/gpt-oss-120b/ttkernel-maxabs.yaml (the max_abs half, loopback alone; the pcc half with P17.37); tests with fakes for the tt-mlir tools; and plans/runs/mlir/smoke.sh for alpha01. At the pin a ttkernel kernel gets a circular buffer only from an argument, so until P17.37 the one recipe runs loopback alone.
- Files: lassi/toolchains/ttkernel_build.py, lassi/bench/registry.py, lassi/core/ (runner, stages, recipe), assets/bench/tt-pairs-v0-ttkernel.yaml, assets/prompts/tt-mlir-v0/, assets/context/ttkernel-5f396fd6/MANIFEST.yaml, assets/examples/ttkernel-examples-5f396fd6/MANIFEST.yaml, tools/build_ttkernel_context.py, projects/lassi-demo/mlir1/, plans/runs/mlir/smoke.sh, tests/, docs/BIBLE.md.
- Host: Windows; the smoke and the recipes on alpha01 in an owner window. Depends: P17.27.

### P17.37 ttkernel host programs that pass each circular buffer index as an argument
- Added on 2026-10-26 (P17.36's finding): at tt-mlir 5f396fd6 the only ttkernel ops whose result may be a circular buffer are get_arg_val, get_common_arg_val, and get_compile_time_arg_val (TTKernelOps.td:2756-2800), and the tracked host programs of eltwise_binary, eltwise_sfpu, matmul_single_core, and matmul_multi_core pass no circular buffer index as an argument, so no ttkernel kernel can name their buffers.
- Bible: Level Rules, Benchmark Suites (tt-pairs-v0-ttkernel), Harness Contract, Agent Rule 5.
- Accept, under OQ-103 (a): for each of those items, a tracked host program for the ttkernel language that appends each kernel's circular buffer indices as compile-time arguments at fixed, documented positions, otherwise identical to the item's host program (a diff test), still building and passing with the upstream kernels on ttsim (an owner window); the sibling manifest names it with its sha256; the tt-mlir-v0 prompt shows it; the max_abs recipe then holds every item of its half again, and a pcc recipe for the matmuls joins it.
- Files: assets/bench/tt-pairs-v0/<item>/ttkernel/, assets/bench/tt-pairs-v0-ttkernel.yaml, projects/lassi-demo/mlir1/, tests/.
- Host: Windows; the ttsim check on alpha01 in an owner window. Depends: P17.36.

### P17.39 C++ -> TT with the kernels: the model writes the host program and every kernel it launches
- Added on 2026-10-26 (the owner, 2026-10-26: "I want to start having the model write kernels as well as host side code"). Until now cpp -> tt on tt-pairs-v0 has the model write the host program alone, around the item's upstream kernels as support files; in the window of 2026-10-26 gpt-oss-120b passed every output of 30 of 30 such trials with the host skeleton (sweep8's v1exsk and v1allsk) and of 2 of 14 without it (v1ex) (rx 20261026-055605-window-119f from commit 3d4e0e98, clean, on alpha01, pass 20261026-0626; every TT program on ttsim v1.3.4 with tt-metal 5280a9cf, a simulator; a development set, OQ-083 (b)). This task adds the arm in which the model writes the whole TT-Metalium program, with the item's tt program on ttsim as the reference.
- Bible: Benchmark Suites (tt-pairs-v0; the support and `reference:` keys; TT tiers; kernels in a prompt), Harness Contract (kernel sources written uncompiled; kernel JIT errors), ttsim Facts (the ttsim executor's kernel JIT reading), Project Recipes (Notes: the Examples, Lookup, and Skeleton bullets; the tt-kh-v0 set), Agent Rules 2, 5, 6, and 10.
- Accept: the sibling suite tt-pairs-v0-kh (assets/bench/tt-pairs-v0-kh.yaml), tt-pairs-v0's five items unchanged plus a language ttkh whose files are the item's tracked TT host program's name and then each kernel at the path that program launches it, support lassi_io.h alone with its sha256, and `reference: tt`; ttmetal-host builds a ttkh attempt as written (the host program compiled, each model kernel written uncompiled at its path); a kernel JIT error comes back to the model as a parsed jit-stage diagnostic (file, line, column, message) in compile_loop's correction under the cap, and a failed kernel build whose log holds no compiler error line as one kernel-build-failed jit-stage error holding the log's lines (lassi.executors.ttsim), both shown by SYNTHETIC captured-style tests through the runner with fakes; the template set tt-kh-v0 in tt-host-v1's style (the host program and every kernel as FILE blocks; the kernel kinds, circular buffers, kernel_main, the "api/..." includes, and the argument, NoC, and tile calls at the pin; it never uses kernel_files or host_files, so it never shows an item's kernels or host program); the worked examples tt-kh-examples-v0 (tt-metal's hello-world compute and data movement examples beside tt-examples-v0's two, all from tt-train-v0's include list, built by tools/build_tt_examples.py --spec) and the kernels pack tt-metalium-5280a9cf-kh (tools/build_tt_context.py --kernels: the guide with its compute kernel sections, the fuller pack's host API, and the declarations of every header under tt_metal/hw/inc/api at the pin, serving ttkh), only their spec and manifests tracked; the item-agnostic skeleton tt-kh-skeleton-v0 (OQ-096 (a)): the host skeleton that built at the pin, byte for byte, and a reader, a compute, and a writer kernel skeleton at the paths it launches, written from the pinned kernel headers (lassi.prompts.skeletons reads kernel skeletons after host.cpp and block.txt); the recipes projects/lassi-demo/sweep9/gpt-oss-120b/{kh-ex,kh-exsk,kh-allsk}-{maxabs,pcc}.yaml (sweep8's gpt-oss-120b settings, 3 samples, cap 8, loop.correct_on_mismatch and loop.correct_on_sim_gap true, no adversary stage); and run-suite.sh build steps for the two new assets and the suite's sources. Unverified until a window run: that the kernel skeletons and the kernel calls the prompt names compile at the pin (no kernel compiles off the build host), that the JIT prints a model kernel's path inside the run's directory, and every result of the arm.
- Files: assets/bench/tt-pairs-v0-kh.yaml, assets/prompts/tt-kh-v0/, assets/examples/tt-kh-examples-v0/ (spec.yaml, MANIFEST.yaml), assets/context/tt-metalium-5280a9cf-kh/MANIFEST.yaml, assets/context/tt-kernel-api-5280a9cf.head.txt, assets/skeletons/tt-kh-skeleton-v0/, lassi/executors/ttsim.py, lassi/prompts/skeletons.py, lassi/core/stages.py, tools/build_tt_context.py, projects/lassi-demo/sweep9/, plans/runs/suite/run-suite.sh, .gitignore, tests/, docs/BIBLE.md.
- Host: Windows; the sweep9 recipes on alpha01 in an owner window (tools/suite.py launch --sweep projects/lassi-demo/sweep9). Depends: P17.33, P17.34, P17.35.

### P17.40 bale-v0: histo, ig, and triangle, C++ <-> OpenMP
- Added on 2026-10-26 (the owner, 2026-10-26: "I want to add the BALE benchmark (not really a benchmark but you get the idea) into the dataset" and "BALE needs to be ready for tonight", for the window of 2026-10-27 at 04:00; the owner's answers to plans/spikes/bale.md, section 12: scope (a), histo, ig, and triangle in serial C++ <-> OpenMP C++ on the host CPU, as benchmark items for testing; every item unassigned, a development set, until the owner assigns it; fetch only, no bale text tracked, each upstream file named by path and sha256 at the pinned commit). bale (github.com/jdevinney/bale at 1b8f673) holds no program LASSI can use as a reference as written, so each item pairs two programs written here on the lassi_io harness that follow bale's serial C kernel.
- Bible: Benchmark Suites (a bale-v0 row and paragraph; fetch_bench), Harness Contract (the OpenMP-construct tag), Component Interfaces (the static_tags capability), Execution Backends (the native row), Project Recipes (Notes: the faithful bullet), Toolchain Pins, Repository Layout, Risks And Questions (question 10), Agent Rules 1, 3, 4, 5, 6, and 10.
- Accept: assets/bench/bale-v0.yaml loads as a fetched suite pinned at 1b8f673b, its three items unassigned with seeds, held-out inputs (each at most 2^20 elements), outputs, and an exact tolerance (max_abs 0), and two tracked languages per item, cpp and omp, each with lassi_io.h and bale's serial C file `{upstream: src/other_serial/C/<item>.c, sha256}`; tools/fetch_bench.py lists exactly those three files with their sha256; the six programs are plain ASCII, take the item's input files and then its output files, write each output under its declared name, name the bale file they follow and its copyright holder, and copy no bale text; no cpp program holds an OpenMP construct or a thread, and every omp program holds one; the held-out input digests are recorded (tests/bench/test_held_out_contract.py). A Toolchain capability static_tags (parse-stage guard- tags appended after each build that gave a program, never in a prompt, refused outside that contract) and the Toolchain gcc-openmp (gcc-native's build plus the OpenMP-construct tag, guard-openmp-none; refused in faithful recipes). The template set cpu-omp-v0 (generate, correct, correct_mismatch) that says what cpp and omp mean and shows no upstream file. The recipes projects/lassi-demo/bale1/gpt-oss-120b/cpp-to-omp.yaml and omp-to-cpp.yaml with sweep8's gpt-oss-120b settings, 3 samples, a cap of 8, loop.correct_on_mismatch, baseline_both, binary_io max_abs 0, no adversary stage; they load, pass the runner's checks before any directory with fetched sources, and `tools/suite.py plan --sweep projects/lassi-demo/bale1` accepts them. On a host with the sandbox and the pinned g++ (the LASSI distro or alpha01), tests/bench/test_bale_remote.py: both references of each item build, run clean, agree byte for byte, and match a Python model; then the two recipes run in the owner's window with the fetch first, as development results.
- Files: assets/bench/bale-v0.yaml, assets/bench/bale-v0/, assets/prompts/cpu-omp-v0/, lassi/core/capabilities.py, lassi/core/stages.py, lassi/core/recipe.py, lassi/toolchains/ (gcc.py, openmp_tag.py, __init__.py), projects/lassi-demo/bale1/, tests/bench/ (test_bale.py, test_bale_remote.py, test_held_out_contract.py), tests/core/ (test_static_tags.py, test_bale1_recipes.py), tests/toolchains/test_gcc_openmp.py, plans/runs/suite/run-suite.sh (the fetch step), docs/BIBLE.md.
- Host: Windows for the manifest, programs, tag, prompts, recipes, and tests (no compiler runs there, and the LASSI distro does not exist yet, so no program was built locally); alpha01 for the references' remote check and the recipes, in the owner's window. Depends: P17.34 (the suite launcher).

### P17.11 Gate part (a): recipes and the hardware-free evidence
- Bible: Build Roadmap (P17 row, Gate part (a)), Agent Rules 1, 7, and 10; AGENTS.md Phase Gate and Results.
- Accept, from one clean commit on alpha01, with graphics off:
  - `rx doctor` and du are recorded. The model from P17.1 is fetched once under HF_HOME at its revision, in a named fetch step. The bench sources are fetched (tools/fetch_bench.py), and the slot's prompt and context files are generated first (tools/extract_lassi_assets.py; plans/LESSONS.md, alpha01).
  - `lassi run tests/fixtures/recipes/p17-cpu-hf.yaml` (hf_local on cpu; one HeCBench item whose references pass under the proxy (PHASE-NOTES P1, multicore proxy), in both directions; executor {cuda: none, omp: native}; omp built by nvcpp-multicore) exits 0 with every trial past the baseline. provenance.json records the model's device record and framework pins.
  - The sft, dpo, and grpo smoke recipes run through `lassi train` on the CPU and exit 0, each checkpoint with its records. `lassi run` of each p17-gpu-*.yaml recipe is refused before any directory exists, naming the device it lacks, with no device node opened (OQ-002).
  - `rx pull` writes `results/p17-gate-a/` with provenance.json, and summary.md quotes gate part (a) and names every device.
  - A pull request `P17 Portable Stack, part (a)` goes from p17-portable to main (OQ-038). The phase stays ACTIVE with only BLOCKED tasks left (the GPU tasks, and P17.6 if hipcc needs a GPU), so P5 may start.
- Files: `tests/fixtures/recipes/p17-cpu-hf.yaml`, `tests/fixtures/recipes/p17-gpu-*.yaml`, the sft, dpo, and grpo smoke recipes under `tests/fixtures/recipes/`, `results/p17-gate-a/`.
- Remote: `rx doctor`, `rx run`, `rx job start` and `rx job wait`, `rx pull`. Depends: P17.3, P17.4, P17.5, P17.7, P17.9, P17.10, P17.14.

### P17.12 Spike on the first GPU host (BLOCKED: GPU host, OQ-040)
- Bible: Execution Backends (gpu rows), Sandbox, Environment State, Host Facts, Agent Rules 6 and 7.
- Accept: `plans/spikes/p17-gpu-host.md`, from a clean commit on a GPU host, if one is ever granted (OQ-040: none is planned), gives:
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
  - Remote tests on the GPU host show that a CUDA or HIP program (on an AMD host, built with P17.6's hipcc) runs clean in the sandbox with no other GPU visible, the vendor profiler reads power during the run, hf_local generates on the GPU, and the TRL smoke runs on the GPU.
  - Captured probe fixtures replace the fakes wherever they differ.
- Files: `lassi/executors/`, `tests/fixtures/`, `docs/BIBLE.md`.
- Remote: `rx run` of the remote tests on the GPU host. Depends: P17.4, P17.7, P17.9, P17.12.

### P17.G Phase gate (BLOCKED: GPU host, OQ-040)
- Bible: Build Roadmap (P17 row, Gate column), Agent Rules 1 and 10; AGENTS.md Phase Gate and Results.
- Accept, from one clean commit:
  - Part (a) reruns on alpha01, as in P17.11.
  - Part (b) runs on the GPU host: the p17-gpu recipe on cuda or rocm exits 0 with every trial past the baseline; the gpu executor, bound per language, runs the item's references in the sandbox, with the profiler's runtime and power; and the training smoke runs on the GPU.
  - `results/p17-gate/` names every device, driver, runtime, and framework pin in its provenance.
  - P17 is set DONE, and the pull request `P17 Portable Stack` goes to main.
- Files: `results/p17-gate/`, `plans/STATUS.md`.
- Remote: `rx doctor`, `rx run`, `rx job start` and `rx job wait`, `rx pull` on both hosts. Depends: P17.11, P17.13.
