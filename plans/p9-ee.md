# P9 LASSI-EE

Branch `p9-ee`, from p16-adversarial at e43a1e0 (P16 is not merged; it waits on OQ-067 to OQ-069). Opened by the owner on 2026-10-25 without a GPU host (OQ-088, option (a), with the owner's addition that the energy profiler be device-agnostic). Code references name files as they stand at e43a1e0.

## Scope

P9 builds what the bible's Build Roadmap P9 row names (EE stages, the LASSI-EE power protocol, the equivalence judge in the pipeline) from Source Papers, LASSI-EE (stages 0 to 4, the judge, the stopping counter, the power protocol, n = 30); Project Recipes (the lassi-ee recipe, the agent and judge bindings block, Notes on profilers and on the port label); Component Interfaces (the Profiler and Judge rows and contract rules); Result Record (Profiles); Oracles (llm_judge); Agents And Judges (Judge interface, Judge Types, Judge rules); Benchmark Suites (lassi-ee-22); and Evaluation Protocol (the LASSI-EE row, Acceptance Criteria, Reporting Rules). Under OQ-088 the first run is a labeled CPU port: C++/OpenMP same-language refactoring for energy on alpha01's EPYC 9534 through the native executor in the sandbox, energy from RAPL where this account can read it, else wall time labeled not energy, with RNGD-served generators. GPU runs (the HIP arm on MI300X, the CUDA arm on an NVIDIA host) are a later gate part. The P9 row's Scope and Gate text predate OQ-088; the edit that records the two gate parts is part of applying OQ-088 (bible-sync), not a P9 task.

Gate for this plan (OQ-088, as the owner opened it): part (a), the CPU port of LASSI-EE runs end to end on alpha01 with energy (or labeled wall time) reported per Evaluation Protocol; part (b), the LASSI-EE GPU acceptance criteria, stays for a GPU host.

Decisions taken at planning (each task records its own in the bible with a Decision Log entry):

- One energy interface. The Profiler stays the interface (Agent Rule 3). P9.1 adds one telemetry seam beside lassi.profilers.power PowerSource that also takes cumulative energy counters, and every backend declares what it can measure on this host (energy, power, runtime only, or nothing, with the reason). A backend that cannot measure refuses when built (TelemetryUnavailable naming why); nothing falls back silently: wall time is its own backend that a recipe names (AGENTS.md Owner Queue, choice rule).
- Judges come from P15. P9 consumes P15's Judge interface and equivalence judge and builds no judge component of its own: the judge stage and compare_report depend on P15.2 (provisional ID, the P15 task that delivers the Judge interface; its planner works on p15-judges at the same time). Until then P9's tests use a fake Judge behind lassi/core/interfaces.py Judge.
- The CPU-port suite. HeCBench at the lassi-hecbench-10 pin, 692cba3, holds `-omp` directories for 18 of lassi-ee-22's 22 apps (GitHub tree of src/ at 692cba3, read 2026-10-25): randomAccess, all-pairs-distance, colorwheel, chacha20, entropy, murmurhash3, floydwarshall, layout, threadfence, dense-embedding, jacobi, matrix-rotate, bsearch, keogh, extrema, pathfinder, lid-driven-cavity, and xsbench. marchingCubes, segment-reduce, and jaccard have cuda, hip, and sycl only; miniMDock is not in HeCBench (plans/spikes/p5-v0-corpus.md, section 4). The app list is the paper's Table I as that spike records it. Draft owner item A decides the four.
- Shared host. RAPL reads a whole package, other users' load included; the protocol's idle windows and drift report expose it, and every CPU-port number names the package, the cores the program ran on, and that the host was shared.

Constraints for every task: Agent Rules 1, 2, 5 (lassi-ee-22 is eval only: no prompt tuning on its items; prompts are written from the paper's text), 6, 9 (no silicon power reading in an unattended session; model output on silicon refused until OQ-078), 10, 14 (the judge screens and never overrides PASS/FAIL or a measurement), and 17 (every rx command waits for the owner's yes, batched per window). HeCBench text is never tracked: manifests hold paths and sha256 only. Shared records follow PHASE-NOTES All Phases, Stacked phase branches (the hub is p5-ir). Remote tests are marked `remote`; sandboxed native runs go to alpha01 in a window, since the LASSI distro waits on install batch 1 (P5.15, OQ-056).

## Tasks

### P9.1 The energy seam and the CPU backends (rapl, wall)
- Bible: Component Interfaces (Profiler contract rules), Result Record (Profiles), Project Recipes (Notes, Profilers), Design Principle 9, Agent Rules 1 and 2.
- Accept: a registered energy profiler (proposed kind `energy`, key `backend` required with no default) samples through one seam that takes power (W) or a cumulative counter (J, with wrap at max_energy_range_uj); backend `rapl` reads <root>/sys/class/powercap package domains (energy_uj) for one package named in the device section (kind cpu); backend `wall` gives runtime only and labels its Profile "wall time, not energy". Tests on Windows on a synthetic powercap tree: counter wrap, an unreadable energy_uj (mode 0400) refuses at build naming the file, a failed read nulls the window, no bridging. The runner accepts kind cpu beside an executor that runs on the host CPU (native) and still refuses simulators. Bible edits (Profiler row and rules, Profiles, Notes) with a Decision Log entry.
- Files: lassi/profilers/energy.py, lassi/profilers/rapl.py, lassi/profilers/__init__.py, lassi/core/stages.py, lassi/core/runner.py, tests/profilers/, docs/BIBLE.md.
- Host: Windows. Remote: none. Depends: none.

### P9.2 Spike: CPU energy telemetry on alpha01, and the excess-files inventory
- Bible: Host Facts, Agent Rules 1, 7, and 17; AGENTS.md Remote Execution (the standing inventory).
- Accept: plans/spikes/p9-cpu-energy.md records, with commands and outputs: the kernel release, the powercap tree and each energy_uj's mode and owner, max_energy_range_uj, perf_event_paranoid, /sys/bus/event_source/devices/power, the package-to-CPU map, and one read of two counters 1 s apart if readable; the Linux source lines (at alpha01's release) that register AMD RAPL in powercap and set the file mode. Verdict: `rapl` or `wall` for the CPU port, recorded in Host Facts. The same batch inventories excess files under /mnt/nvme10/joseph_ufl with du sizes and files one owner-queue item (PHASE-NOTES All Phases); nothing is deleted.
- Files: plans/spikes/p9-cpu-energy.md, docs/BIBLE.md, plans/OWNER-QUEUE.md.
- Host: Windows (source reading), alpha01 (one read-only `rx exec`, owner's yes). Depends: none.

### P9.3 Spike: Tenstorrent power telemetry at the pins
- Bible: Execution Backends (tt_silicon row, the pins), Agent Rule 9.
- Accept: plans/spikes/p9-tt-power.md names, from tt-kmd 2.8.0 and tt-smi or luwen source read locally, which file or call reports Wormhole n300 board power, its unit and update interval, and whether a read opens /dev/tenstorrent/<n> or messages the ARC firmware; verdict whether sampling at 10 ms is a looped device access under Agent Rule 9. No probe of the cards.
- Files: plans/spikes/p9-tt-power.md. Host: Windows. Depends: none.

### P9.4 AMD and NVIDIA backends on the seam
- Bible: Component Interfaces (Profiler; rocm_smi and nvml text), Execution Backends (gpu rows), OQ-002.
- Accept: `rocm_smi` (amdgpu hwmon power1_average) and `nvml` become backends of the seam with their existing sources reused and their registered kinds kept (existing tests pass unchanged); each declares power and its device kind; tests with fakes only. No GPU node opened, nothing new read on alpha01. Bible edit with a Decision Log entry.
- Files: lassi/profilers/rocm_smi.py, lassi/profilers/nvml.py, lassi/profilers/energy.py, tests/profilers/, docs/BIBLE.md.
- Host: Windows. Depends: P9.1.

### P9.5 Tenstorrent and Cerebras backends
- Bible: Component Interfaces (Profiler), Execution Backends (tt_silicon, ttsim, cerebras_sim rows), Agent Rules 2 and 9.
- Accept: backend `tt` reads what P9.3 names, built and tested with fakes on a synthetic tree, refused beside ttsim (simulator) and refused in a run without `--silicon`; backend `cerebras` declares it cannot measure ("the fabric simulator reports no power; hardware access, OQ-004") and refuses when built. Device kinds for both are added only if P9.3 needs them (Result Record edit). Bible edit with a Decision Log entry.
- Files: lassi/profilers/tt_smi.py, lassi/profilers/cerebras.py, lassi/core/record.py (if kinds), tests/profilers/, docs/BIBLE.md.
- Host: Windows. Depends: P9.1, P9.3.

### P9.6 The LASSI-EE power protocol
- Bible: Source Papers, LASSI-EE (power bullet), Evaluation Protocol (Acceptance Criteria, LASSI-EE), Result Record (Profiles), PHASE-NOTES P17 (P17.7 candidates for P9).
- Accept: profiler.idle_window_s becomes a key (integer seconds, no default; the load refusal goes); every profiled run gets pre-run and post-run idle windows, idle subtraction, negative samples clamped to 0, and an idle-drift value; the baseline's reference runs are profiled (Trial.reference_run.profile). New Profile fields reach trial.md and the Parquet attempts table; nulls follow the existing rules. Tests with fake sources and a fake clock on Windows. Result Record and Component Interfaces edits with a Decision Log entry.
- Files: lassi/profilers/energy.py, lassi/core/record.py, lassi/core/stages.py, lassi/core/parquet.py, lassi/core/trial_md.py, lassi/core/recipe.py, tests/, docs/BIBLE.md.
- Host: Windows. Depends: P9.1.

### P9.7 The lassi-ee-22 manifest and its CPU-port references
- Bible: Benchmark Suites (lassi-ee-22, split rules), Oracles (stdout_mask, passfail), Harness Contract, Agent Rule 5.
- Accept: assets/bench/lassi-ee-22.yaml at 692cba3 lists the 22 apps, split eval; the 18 `-omp` apps carry files and sha256 (cuda and hip entries recorded for part (b)); run_args from each app's Makefile run target at the pin ([DESIGN]); masks in assets/harness/masks/lassi-ee-22.yaml with the render test; passfail per language. tools/fetch_bench.py fetches it and refuses a changed file. Remote test: every omp reference builds with gcc-native and runs twice in the sandbox with identical masked stdout; failures are named per app.
- Files: assets/bench/lassi-ee-22.yaml, assets/harness/masks/lassi-ee-22.yaml, tests/bench/, docs/BIBLE.md.
- Host: Windows; alpha01 in a window for the remote test (`rx run`). Depends: none.

### P9.8 Spike: the paper's protocol details
- Bible: Source Papers, LASSI-EE.
- Accept: plans/spikes/p9-ee-protocol.md answers from arXiv:2505.02184 v3 (quoted with section): how the stopping counter ends a trial beside max_iters and maps to temperature; what each stage's prompt asks; how a refactor's energy is measured (repetitions, which value is fed back); the Stage 4 selection rule and report; what energy-reduction@k conditions on. Each answer the paper leaves open is a choice listed for the stages' [DESIGN] entries.
- Files: plans/spikes/p9-ee-protocol.md. Host: Windows. Depends: none.

### P9.9 EE stages 0 to 2 and the lassi-ee-2026 prompts
- Bible: Source Papers, LASSI-EE (stages 0 to 2), Project Recipes (lassi-ee stages list), Agents And Judges (Planner), Agent Rules 3 and 5.
- Accept: stages `profile`, `vanilla`, and `plan` are registered generic stages (proposed lassi/core/ee_stages.py) and `summarize_context` is reused; prompts assets/prompts/lassi-ee-2026/ are written here from the paper's text, with an OpenMP-on-CPU variant, never tuned on a lassi-ee-22 item; mock-LLM tests on Windows run stages 0 to 2 on a synthetic omp program and record each attempt's profile.
- Files: lassi/core/ee_stages.py, assets/prompts/lassi-ee-2026/, tests/core/, docs/BIBLE.md.
- Host: Windows. Depends: P9.6, P9.8.

### P9.10 Stages 3 and 4: refine_loop, select, compare_report
- Bible: Source Papers, LASSI-EE (stages 3 and 4, the stopping counter), Project Recipes (refine section), Judge Types (efficiency, explain mode).
- Accept: refine_loop applies refine.counter_start, counter_step, and max_iters (required, no default) with the counter as temperature, resets on a measured improvement, feeds back measured energy, and self-corrects compile and run errors; select picks the best by measured energy among oracle-passing candidates; compare_report calls the bound explain-mode judge through P15's interface, its text tagged [JUDGED]. Mock and fake-judge tests on Windows, including the counter sequence and a tie.
- Files: lassi/core/ee_stages.py, tests/core/, docs/BIBLE.md.
- Host: Windows. Depends: P9.9, P15.2.

### P9.11 The equivalence judge in the pipeline
- Bible: Oracles (llm_judge), Agents And Judges (interfaces, Judge rules), Agent Rule 14.
- Accept: the `judge` stage screens each candidate through P15's equivalence judge from source, candidate (comments stripped), and both stdouts on identical inputs; VALID or INVALID is recorded beside passfail and never replaces it; the runner refuses a judge of the generator's model family; judge agreement with PASS/FAIL is a per-trial record. Fake-judge tests on Windows.
- Files: lassi/core/ee_stages.py, lassi/core/recipe.py, tests/core/, docs/BIBLE.md.
- Host: Windows. Depends: P9.9, P15.2.

### P9.12 The ee score profile, metrics, recipes, and a mock dry run
- Bible: Evaluation Protocol (LASSI-EE row, Acceptance Criteria, Reporting Rules), LASSI Score Profile, Project Recipes (lassi-ee recipe; port label note).
- Accept: score `ee` and metrics energy_reduction, power_change, runtime_change (conditional and unconditional), energy_reduction_at_k (k = 1, 3, 5, Wilson intervals), idle drift, and judge agreement, with the paper's values alongside and every CPU-port value labeled "CPU port" (and "wall time, not energy" under `wall`); projects/lassi-ee/recipe.yaml as the bible gives it and projects/lassi-ee/cpu-port.yaml (omp -> omp, gcc-native, native, the energy profiler). Remote test: a mock-LLM dry run of cpu-port.yaml on 2 apps x 1 trial passes end to end on alpha01.
- Files: lassi/scoring/, lassi/analysis/, projects/lassi-ee/, tests/, docs/BIBLE.md.
- Host: Windows; alpha01 in a window for the dry run. Depends: P9.6, P9.7, P9.10, P9.11.

### P9.13 The CPU-port run on alpha01 with RNGD generators (OWNER)
- Bible: Model Serving (Serving Rules), Agent Rules 8, 13, and 17; Evaluation Protocol.
- Accept: in an owner window, a pilot (1 app x 1 trial) measures wall time per trial and projects the run (PROJECTED); then the scale draft item C chose runs with the models draft item B chose, cards claimed under Agent Rule 8 and released after; run tree under $LASSI_RUNS_ROOT; results/p9-cpu-port-<date>/ holds provenance.json from `rx pull`, summary.md, and the metric tables.
- Host: alpha01 (owner window; a big job). Depends: P9.2, P9.12. Waits on draft items B and C and a window.

### P9.14 GPU and device energy runs (BLOCKED)
- Bible: Project Baselines, LASSI-EE; Execution Backends; Evaluation Protocol (Acceptance Criteria: MI300X labeled a HIP-arm port).
- Accept: the lassi-ee recipe's HIP arm on MI300X with rocm_smi and the protocol; the CUDA arm with nvml; tt power on human-started silicon reference runs only; cerebras on hardware. Blockers: MI300X access (OQ-002, OQ-040), an NVIDIA host (OQ-003), Agent Rule 9 and OQ-078 for TT, Cerebras hardware (OQ-004).
- Host: a GPU host; devices. Depends: P9.4, P9.5, P9.12.

### P9.G Phase gate, part (a)
- Accept: the CPU port of LASSI-EE runs end to end on alpha01 with energy (or labeled wall time) reported per Evaluation Protocol (P9.13's results, with idle drift per trial and conditional and unconditional values); evidence under results/p9-gate-a/ with provenance. Pass opens the pull request `P9 LASSI-EE` for part (a); part (b), the GPU acceptance criteria, stays with P9.14, and the phase stays ACTIVE.
- Host: Windows (evidence); alpha01 runs come from P9.13. Depends: P9.13.

## Order

Tonight, hardware-free on Windows: P9.1, P9.3, P9.8 at once; then P9.4, P9.5, P9.6; P9.7's manifest half; P9.9. P9.10 and P9.11 start when P15.2 is DONE. One owner window batches P9.2's probe, P9.7's remote test, and P9.12's dry run; a later window runs P9.13.

## Owner queue (filed 2026-10-25 as OQ-089, answered: A (a), B (a), C (a); the alpha01 batch as OQ-092)

- A. lassi-ee-22's CPU port: the four apps without an OpenMP version at 692cba3, and the suite's pin. Recommendation: run 18 apps at 692cba3 and name the four as not ported.
- B. Generator and judge models on RNGD. Recommendation: generator gpt-oss-120b, equivalence judge llama-3.3-70b (another family, as Agents And Judges requires), coordinated with P15's choice.
- C. The CPU-port run's scale and host conditions: n = 30 on 18 apps is 540 trials; the pilot's projection decides how many windows; whether the run needs the package quiet. Recommendation: n = 30 on an owner-picked subset after the pilot, labeled a subset.
