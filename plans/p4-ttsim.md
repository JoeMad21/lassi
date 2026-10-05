# P4 ttsim Execution

Branch `p4-ttsim`, base `main` at 75eceef (P2 DONE by OQ-024; its pull request 3 is merged).

## Scope

P4 builds what the P4 row of the bible's Build Roadmap names: the ttsim executor and the native side it pairs with (Execution Backends; ttsim Facts), `lassi_io.h` and the binary_io oracle (Harness Contract; Oracles), the CPU -> TT guard (Harness Contract), the Tier A references the gate runs (Benchmark Suites), and terminal presentation (Readability Standards, Terminal Presentation; Decision Log 2026-09-24, OQ-023). tt-metal and ttsim are pinned jointly with tt-mlir (Toolchain Pins); tt-mlir itself is built in P5. The native executor exists since P0.10; P4 adds the C++ toolchain and per-language executors it needs. The gate is the Gate column of the P4 row, run exactly (P4.G); it names no owner review, so a pass sets P4 DONE.

Constraints for every task:

- The scope is fixed at planning, and the working changes of plans/runs/p0-retrospective.md (Changes from P1 on) apply: one review round per task and one commit audit, [MEASURED] only with a clean-commit rx id, a clean tree before every evidence run, rx output under 120 lines. A later finding goes into PHASE-NOTES under a later phase or into the owner queue. The one exception is OQ-024 (option b, applied at planning): the P2 review questions move to this phase as P4.15, which takes the plan to 16 tasks, one over the usual 5 to 15, by the owner's direction.
- Hardware: none, so no task starts BLOCKED. ttsim is a simulator on the CPU; tt_silicon is disabled in the gate and no Wormhole is on alpha01 (Environment State). Every simulator run in a trial sets the ttsim row's settings (Execution Backends) inside the sandbox, whose private /dev holds no Tenstorrent node; P4.9's measuring runs of upstream reference examples may run outside it while it measures what the sandbox must allow, and alpha01 has no Tenstorrent device node, so no step can open silicon (Agent Rule 9). Results name ttsim and its pin as the device, simulator wall times only size limits and are never shown as performance, and every ttsim pass is provisional; its silicon check (ttsim Facts) is a human-started run outside P4 (Agent Rule 2).
- Scratch: 94G of the 120G cap (rx 20260923-222319-exec-07d8, OQ-017). `rx doctor` and `du -sh /mnt/nvme10/joseph_ufl` before and after every install or long run, recorded in the evidence. Estimates, PROJECTED, not measured: the pinned tt-metal build 10 to 20G (P4.1 measures it), ttsim under 1G, P4 run trees under 1G. A step that would pass 115G stops and queues backup or deletion candidates (OQ-012's amd/ is the largest); agents never delete there.
- Big builds run as `rx job start --big`, one at a time; set the task back to READY with the job id in its note and take local tasks meanwhile. Nothing on the root filesystem (Agent Rule 7): downloads, CMake package caches, the sfpi toolchain, TMPDIR, and JIT caches live under the scratch root or the run's workdir.
- No model is served; the gate's driver is the mock. No tt-metal file enters a tracked file, results carry no source or model text (OQ-018 practice), and text is plain ASCII. P4.15 changes lassi/scoring and adds sim_t_tiktoken (PHASE-NOTES, All Phases). P4.4 and P4.6 also touch it: each new end reason that ends a trial at the baseline joins the lassi profile's baseline-end set, so correct and compiled stay null there as the LASSI Score Profile says, and df-v0's reading of a sim-gap attempt is stated. Unattended sessions run with graphics off.

Decisions taken at planning (the named task records each in the bible with a Decision Log entry):

- Joint pin (P4.1): one stated rule, the newest tt-mlir release (or main commit, if it has no releases) whose tt-metal pin a published ttsim release supports on Wormhole. The checkout at /mnt/nvme10/joseph_ufl/tt-metal belongs to another project (TurboQuant) and is only read, as a source of facts and, when its commit is the pin, of files to copy; the pinned build always goes under $LASSI_TOOLCHAINS (AGENTS.md, Remote Execution; PHASE-NOTES P4), so no build writes into that tree. If the rule conflicts with that checkout's other user (TurboQuant, the Tier B source) or with the scratch cap, P4.1 adds an owner-queue item and P4.2 goes OWNER.
- Executors per language (P4.5): a recipe binds an executor per language, as `toolchain` does, so a baseline runs the C++ reference natively and the TT reference on ttsim; the single-executor form stays valid with unchanged hashes. The host g++ is pinned by version and path (`toolchains/gcc.pin`), not installed.
- Simulator readings (P4.6): a kernel JIT failure leaves the attempt at S1 with jit-stage errors and is corrected like a compile error (S4 needs host and kernel JIT, Reward Function). UndefinedBehavior is a failed run fed back to the model. UnimplementedFunctionality and UnsupportedFunctionality end the trial with a new end reason, sim-gap. A reference run with UB ends at baseline-run and one with a gap at sim-gap. A timeout carries the Harness Contract's hang diagnostic.
- binary_io (P4.4): a candidate's output files are compared with the target reference's. `threshold: from_baseline` reads the source reference's agreement with the target reference, measured in the baseline with baseline_both on; items may declare exact match; any further tolerance is an explicit recipe value with no default. A pair whose references miss the item's declared tolerance ends at the baseline under a new end reason. Output files are stored by hash in a binary store beside the text store.
- Tier A (P4.13): upstream kernels are used unmodified from the pinned tt-metal, fetched under the scratch root; the TT host programs (inputs and outputs through lassi_io.h, no golden) and the C++ counterparts are written here and tracked. Each item's declared tolerance is its upstream example's own check (P4.9). OQ-025 is answered with option (d): each split stays `unassigned` until P7, refused to training as eval is, so nothing is picked and no item can later leave eval; the owner assigns the splits before any training run.
- Progress hook (P4.8; the bible leaves it [OPEN] for this plan): an in-process observer passed through RunOptions and RunContext. The runner and stages send it events (trial start, stage start, model request sent, attempt appended or updated, trial end), each carrying the immutable Trial. It writes nothing and nothing depends on it; an observer error drops the observer for the rest of the run and changes no record, file, or exit status. Rejected: polling trial.json (written only at trial end) and a progress file (presentation writes no file).
- Preset (P4.7): on a POSIX host the preset is never saved on the filesystem that holds `/` (Agent Rule 7; PHASE-NOTES P4); the answer then applies to that invocation only, and the prompt names LASSI_CONFIG_DIR. So on a Linux machine whose home is on `/`, the preset saves only with LASSI_CONFIG_DIR or LASSI_SCRATCH set; P4.7's Decision Log entry states this.
- Gate drivers: `tools/ttsim_smoke.py` for the example (P4.11) and a mock dry run of Tier A in both directions (P4.13), as P1.G used the mock. P4.11 and P4.13 record them in their Decision Log entries.

PHASE-NOTES P4 items, by task: terminal presentation, P4.7 and P4.8; the tt-metal checkout, P4.1 and P4.2; the ttsim environment, P4.9 and P4.11; unpack_to_dest, P4.9 and P4.13. Also taken: PHASE-NOTES P2, device in native runs (P4.5). Owner queue: OQ-025 (Tier A splits) is answered with option (d), every item unassigned until P7; P4.14 records it.

## Tasks

### P4.1 Spike: joint pin of tt-mlir, tt-metal, and ttsim; build budget
- Bible: Toolchain Pins, ttsim Facts, Execution Backends (ttsim row), Host Facts (existing assets, scratch cap), Risks And Questions (scratch cap).
- Accept: `plans/spikes/p4-tt-pins.md` gives, with commands, outputs, and sources: the existing checkout's commit, tree state, and size; the candidate tt-mlir commits and the tt-metal commit each pins; the ttsim release (version, sha256) for that tt-metal on Wormhole and whether ttsim issue #18 still stands; the host build prerequisites present or missing, each missing one with a user-space route; the build's disk estimate from the checkout's measured size. The joint pin joins Toolchain Pins with a Decision Log entry, or an owner-queue item names the conflict (planning decisions).
- Files: `plans/spikes/p4-tt-pins.md`, `docs/BIBLE.md`.
- Remote: `rx doctor`; read-only `rx exec` (du, git state, tool versions). Depends: none.

### P4.2 Install the pinned tt-metal and ttsim
- Bible: Toolchain Pins, Execution Backends (ttsim row), Agent Rules 7 and 10; AGENTS.md Remote Execution.
- Accept:
  - `toolchains/tt-metal.pin`, `tt-metal.sh`, `ttsim.pin`, and `ttsim.sh` record commit or version, the sha256 of every download, build flags, and install path; a rerun at the pin changes nothing and a mismatch is refused. The build includes the programming examples (the gate's example among them); ttsim's library has its soc_descriptor.yaml beside it.
  - From a clean commit: du before and after, and no file written to /tmp or /var/tmp (checked as P0.19 did); `results/p4-tt-install/` holds provenance.json and summary.md.
- Files: `toolchains/`, `results/p4-tt-install/`.
- Remote: `rx doctor`, du, `rx job start --big --name p4-tt-metal`, `rx job wait`, `rx pull`. Depends: P4.1.

### P4.3 lassi_io harness
- Bible: Harness Contract, Frontend Rules (harness bindings), Repository Layout (`assets/harness/`).
- Accept:
  - A C and C++ header under `assets/harness/c/` and a Python reader and writer in `lassi/` handle one binary file per named array (dtype including bf16, shape, little-endian data), in a format the Harness Contract records with a Decision Log entry.
  - Python round trips for every dtype, and refusals of a bad magic, version, dtype, size, or truncated file (tests); a remote test compiles a C++ program with the header in the sandbox and round trips its files with the Python side. A seeded generator writes an item's held-out input files, the same seed giving the same bytes (test).
- Files: `assets/harness/c/`, `lassi/` (module chosen by the task), `tests/`, `docs/BIBLE.md`.
- Remote: `rx run` of the remote test. Depends: none.

### P4.4 binary_io oracle over output files
- Bible: Oracles, Result Record (RunInfo.outputs_ref, reference_run, end_reason, Storage), Component Interfaces (Oracle), Project Recipes (lassi-df oracle line).
- Accept:
  - A registered Oracle `binary_io` gives per output PCC, max-abs, and ULP statistics (f32 and bf16, with NaN, infinity, and -0 rules stated), checked against hand-computed values, and per_input 1.0 or 0.0 per the planning decision; `metric` and `threshold` are required keys with no default (tests).
  - The oracle stage aligns output files for an oracle that declares the new capability; stdout_mask behavior and its tests are unchanged.
  - Output files stored by hash; the references' agreement recorded by the baseline; a pair outside its tolerance ends at the baseline; trial.md and the Parquet mirror show the statistics; an older trial.json loads unchanged (tests).
  - Bible edits (Oracles, Result Record, Harness Contract) with Decision Log entries.
- Files: `lassi/oracles/`, `lassi/core/` (record, store, stages, oracle_stage, trial_md, parquet), `lassi/scoring/lassi_profile.py` (its baseline-end set), `tests/`, `docs/BIBLE.md` (with Evaluation Protocol, LASSI Score Profile).
- Remote: none. Depends: P4.3.

### P4.5 Native C++ toolchain and executors per language
- Bible: Execution Backends (native row), Toolchain Pins, Component Interfaces (Toolchain, Executor), Project Recipes (Notes), Agent Rules 1 and 10.
- Accept:
  - A registered Toolchain builds C++ with the host g++ and the native row's flags, pinned by `toolchains/gcc.pin` and checked with --version in the compile sandbox before the first build; GCC diagnostics parse from stderr fixtures captured on alpha01 with tools/capture_toolchain_fixtures.py (tests).
  - `executor` binds per language; attempts use the target language's executor and each baseline reference its own language's; the sandboxed check runs per executor before any directory exists; single-executor recipes keep their golden resolved recipes and hashes (tests).
  - Each executor names its device (native: the host CPU); provenance.json and run.md record the device per language (test).
  - Bible edits (Project Recipes, Execution Backends, Result Record) with Decision Log entries.
- Files: `lassi/toolchains/`, `lassi/executors/`, `lassi/core/runner.py`, `lassi/core/recipe.py`, `toolchains/gcc.pin`, `tests/`, `docs/BIBLE.md`.
- Remote: `rx run` for the fixture capture and remote tests. Depends: none.

### P4.6 Record readings for simulator runs
- Bible: ttsim Facts, Harness Contract (jit stage, timeout), Result Record (RunInfo.sim_ub, end_reason), Reward Function (stage table), Component Interfaces (Executor), Agent Rule 2.
- Accept, with a fake simulator executor (local tests):
  - RunResult carries the simulator's findings (UB, a gap and its class, jit-stage Diagnostics); the baseline and run_loop read them per the planning decision; END_REASONS gains sim-gap; a hang carries the Harness Contract's diagnostic.
  - For an executor declaring the capability `simulator`, trial.md and run.md label wall times as simulator wall time, not performance.
  - Older records load unchanged; bible edits (Component Interfaces, Result Record, Reward Function) with Decision Log entries.
- Files: `lassi/core/` (interfaces, stages, record, trial_md, runner), `lassi/scoring/` (the baseline-end set; df-v0's reading of a sim-gap attempt), `tests/core/`, `tests/scoring/`, `docs/BIBLE.md` (with LASSI Score Profile, and one spelling of the sim-gap tag: the bible writes sim_gap, the record's end reasons use hyphens).
- Remote: none. Depends: none.

### P4.7 Graphics setting, preset, and banner
- Bible: Readability Standards (Terminal Presentation), Repository Layout (`present/`, `cli.py`), Agent Rule 7; AGENTS.md Operating Mode.
- Accept:
  - `lassi settings graphics on|off`, the global `--graphics on|off`, and LASSI_GRAPHICS resolve in the bible's order; the preset path follows the bible's order and the planning decision's refusal (tests with a fake terminal and a patched filesystem check).
  - Without an interactive terminal nothing is asked and graphics stay off unless turned on; with graphics off, `lassi run` of a mock recipe and `lassi score` print what they print today, line for line apart from times (tests).
  - With graphics on, the banner prints once per command in plain ASCII; tools/ output is unchanged. Bible edit (preset rule), Decision Log entry.
- Files: `lassi/present/`, `lassi/cli.py`, `tests/present/`, `docs/BIBLE.md`.
- Remote: none. Depends: none.

### P4.8 Progress hook and live inference table
- Bible: Readability Standards (Terminal Presentation), Component Interfaces (Stage contract rules), Design Principle 2.
- Accept:
  - The planning decision's hook: a mock run writes the same run tree, apart from recorded times and dates, and exits 0 with no observer, a recording one, and a raising one (tests). Bible edit closing the [OPEN] with a Decision Log entry.
  - The table is a pure rendering of a Trial snapshot and header values whose rows, columns, excerpt, and width follow the bible; snapshot tests on a clean pass, corrections, a cap hit, and a baseline end; ASCII glyphs, ANSI only on a terminal, no Code column without one.
  - `lassi run` with graphics on redraws about once a second and writes no file beyond what graphics off writes (test on a mock run).
- Files: `lassi/present/`, `lassi/core/runner.py`, `lassi/core/stages.py`, `tests/present/`, `tests/core/`, `docs/BIBLE.md`.
- Remote: none. Depends: P4.7.

### P4.9 Spike: ttsim runtime facts, unpack_to_dest, and Watcher
- Bible: ttsim Facts, Execution Backends (ttsim row), Harness Contract (jit stage, timeout, Watcher), Sandbox, Risks And Questions (simulator gaps; question 4).
- Accept: `plans/spikes/p4-ttsim-runtime.md`, from a clean commit, gives for the gate's example and each upstream Tier A example run on ttsim with the bible's settings: exit status and its own check's result, ttsim messages, peak memory, JIT cache size, and wall time (simulator, exploratory, sizing only); what tt-metal reads, writes, and needs from the environment at run time, and whether it runs in the P0.16 sandbox with the pinned tree read-only; captures of UndefinedBehavior, a gap, and a kernel JIT error (seeded reference edits, never model code) as fixtures; each Tier A kernel's use of unpack_to_dest; whether Watcher works under ttsim. Question 4 is answered and ttsim Facts extended, with a Decision Log entry.
- Files: `plans/spikes/p4-ttsim-runtime.md`, `tests/executors/fixtures/`, `docs/BIBLE.md`.
- Remote: several `rx run` calls, running references only. Depends: P4.2.

### P4.10 tt-metal host toolchain
- Bible: Component Interfaces (Toolchain), Toolchain Pins, Harness Contract (kernels compile at first launch), Repository Layout (`toolchains/`), Sandbox (compiles).
- Accept:
  - A registered Toolchain builds a TT host program against the pinned tt-metal in the compile sandbox, linked so it needs no loader variable, with kernel sources placed where the JIT finds them; the pin check (tt-metal commit, host compiler version) runs before the first build and refuses drift (tests).
  - Host compile errors parse to file and line from captured fixtures; remote tests build the gate's example and the upstream Tier A examples from the pinned sources.
- Files: `lassi/toolchains/`, `tests/toolchains/`, `docs/BIBLE.md`.
- Remote: `rx run` of the remote tests and the fixture capture. Depends: P4.2, P4.5.

### P4.11 ttsim executor and the smoke driver
- Bible: Execution Backends (ttsim row), ttsim Facts, Sandbox, Harness Contract, Agent Rules 2, 6, and 9.
- Accept:
  - A registered Executor `ttsim` (capabilities runs_code, sandboxed, simulator; the lassi-df block's executor keys) always sets the ttsim row's settings from the pinned install, with the JIT cache inside the workdir; those names join the sandbox's fixed allowlist (Sandbox edit, Decision Log). A missing library or descriptor, another arch, or more than one chip is refused before anything runs (tests).
  - It returns P4.6's findings parsed from P4.9's fixtures, adds the Watcher dump to a hang if P4.9 found Watcher works, and names ttsim and its pin as the device. A remote test shows that the sandboxed program sees no Tenstorrent device node.
  - `tools/ttsim_smoke.py <example>` builds the example with P4.10's toolchain, runs it through the executor, writes a summary with provenance under the runs root, and exits 0 only when the example passes its own check with exit status 0, no UB, and no gap.
- Files: `lassi/executors/`, `tools/ttsim_smoke.py`, `tests/executors/`, `docs/BIBLE.md`.
- Remote: `rx run` of the remote tests and one smoke run. Depends: P4.6, P4.9, P4.10.

### P4.12 CPU -> TT guard
- Bible: Harness Contract (CPU -> TT guard), Result Record (Attempt.guards), Reward Function (guard violation), Design Principle 8, Agent Rule 3.
- Accept:
  - A component, called for targets that declare it (no stage names a target), sets Attempt.guards.host_compute and tags a data-movement-only program with a warning Diagnostic, never a failure. Tests on hand-written host programs: a clean offload, a host loop writing outputs, a data-movement-only program, and a compute kernel beside a host loop that writes no output.
  - The detection method, its limits, and the reading of "must create a compute kernel" beside "tagged, not failed" join the Harness Contract with a Decision Log entry.
- Files: `lassi/` (module chosen by the task), `lassi/core/stages.py`, `tests/`, `docs/BIBLE.md`.
- Remote: one remote test if the method reads the run's JIT cache. Depends: P4.6, P4.10.

### P4.13 Tier A suite tt-pairs-v0
- Bible: Benchmark Suites (Tier A, split rules), Harness Contract, Oracles (binary_io), ttsim Facts (unpack_to_dest), Agent Rule 5.
- Accept:
  - `assets/bench/tt-pairs-v0.yaml` pins tt-metal at P4.2's commit and lists the Tier A items with the planning decision's TT and C++ versions, input specs, and a declared tolerance from P4.9; every split is `unassigned` (OQ-025), refused to training (test); tools/fetch_bench.py fetches the pinned files. Each item notes its unpack_to_dest check (P4.9).
  - Remote tests: each TT reference builds, runs clean on ttsim (no UB, no gap), and passes the guard; each C++ counterpart builds and runs clean natively; their binary_io agreement meets the item's tolerance.
  - `tests/fixtures/recipes/p4-tier-a-dry-run.yaml` (mock, both directions, baseline_both on, executors per language, binary_io from_baseline) loads locally (test).
  - Bible edits (Benchmark Suites: the unassigned split, each item's tolerance source, and the hand-written host programs beside the unmodified upstream kernels) with a Decision Log entry, which also records the gate's Tier A driver.
- Files: `assets/bench/`, `lassi/bench/registry.py`, `tools/fetch_bench.py`, `tests/`, `docs/BIBLE.md`.
- Remote: the fetch and `rx run` of the remote tests. Depends: P4.4, P4.5, P4.11, P4.12.

### P4.14 Record the owner's Tier A split answer (OQ-025)
- Bible: Benchmark Suites (split rules), Risks And Questions (question 10), Agent Rule 5.
- Accept: OQ-025 was answered with option (d): the manifest keeps every Tier A item `unassigned`, and a test checks that each is refused to training; the bible's question 10 already records the answer (Decision Log 2026-09-24), so this task confirms the manifest matches it and changes no recipe hash.
- Files: `assets/bench/tt-pairs-v0.yaml`, `tests/bench/`, `docs/BIBLE.md`.
- Remote: none. Depends: P4.13.

### P4.15 Carry the P2 review questions (OQ-024)
- Bible: Evaluation Protocol (LASSI Paper Metrics, LASSI Score Profile, Run Metrics), Reward Function (scoring decisions), Result Record; results/p2-gate/summary.md (Review questions for J); OQ-022 and OQ-024.
- Accept:
  - `plans/spikes/p4-p2-review.md` closes each of the eight review questions one way: settled by evidence (commands, outputs, sources) and recorded in the bible with a Decision Log entry; brought back as an owner-queue item with options and a recommendation, when it is a choice; or recorded as standing, with the reason. No reading, weight, or paper value changes without an owner answer.
  - Question 6: `sim_t_tiktoken` joins the lassi profile under its own name (the OQ-022 pending item): tiktoken pinned in pyproject.toml, its cl100k_base file cached under the scratch root for offline use on alpha01, the component in assets/scoring/lassi.yaml with its note naming the tokenizer and interpreter, and tests. A clean-commit `lassi score` of demo-rngd-cpu-1 reports its values beside sim_t and sim_t_c. Whether it is compared with the paper's Sim-T stays OQ-022's review.
  - Question 7 (OQ-021) is answered with option (c): assets/scoring/lassi-paper.yaml swaps `recount` and `recount_alternate` for omp-cuda within_10pct_rate, so the reference is the recomputed 24/32 and 23/32 from the printed Ratios is the alternate (a test pins it; the file's rule makes the change a Decision Log entry); the metric tables name the recount as the reference value and label the published value as shown for reference only (lassi.analysis tables and assets/scoring/lassi-paper.yaml as needed, with tests).
  - Bible edits with Decision Log entries; `results/p4-p2-review/` holds the scoring pass's provenance and summary.
- Files: `plans/spikes/p4-p2-review.md`, `lassi/scoring/`, `lassi/analysis/`, `assets/scoring/lassi.yaml`, `assets/scoring/lassi-paper.yaml`, `tests/analysis/`, `pyproject.toml`, `uv.lock`, `tests/scoring/`, `results/p4-p2-review/`, `plans/OWNER-QUEUE.md`, `docs/BIBLE.md`.
- Remote: `rx doctor`; an `rx run` that caches cl100k_base under the scratch root; `rx run` of the scoring pass. Depends: none.

### P4.G Phase gate
- Bible: Build Roadmap (P4 row, Gate column), ttsim Facts, Agent Rules 1 and 2; AGENTS.md Phase Gate and Results.
- Accept, from one clean commit on alpha01:
  - `rx doctor` and a du check, recorded; `rx run -- 'uv run tools/ttsim_smoke.py metal_example_add_2_integers_in_riscv'` exits 0: the example passes on ttsim.
  - `rx job start --name p4-tier-a -- 'uv run lassi run tests/fixtures/recipes/p4-tier-a-dry-run.yaml --run-id p4-gate-tier-a'` exits 0 with every trial past the baseline (Tier A references pass: each TT reference within its tolerance of its counterpart, no UB, no gap, guard clear) and, as an extra check reported beside the gate text (it tests the path, as P1.G's mock did), every mock candidate at S5 with alignment 1.0.
  - `rx pull` writes `results/p4-gate/` with provenance.json; summary.md quotes the gate text, names ttsim and its pin as the device of every TT run and the host CPU for native runs, shows no simulator timing as performance, marks each ttsim pass provisional until a silicon check, and lists each item's unpack_to_dest check.
  - P4 is set DONE; the pull request `P4 ttsim Execution` goes from `p4-ttsim` to `main`, its base main.
- Remote: `rx doctor`, `rx run`, `rx job start` and `rx job wait`, `rx pull`. Depends: P4.8, P4.11, P4.12, P4.13, P4.15.
