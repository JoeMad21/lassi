# P0 Core

Branch `p0-core`, base `main` at b092ea2.

## Scope

P0 builds the shared core named in the P0 row of the bible's Build Roadmap: the `lassi/` package skeleton (Repository Layout, Design Principles), the twelve Protocols (Component Interfaces), the Result record and `trial.md` (Result Record, Readability Standards), the recipe loader (Project Recipes), the openai_compat and ollama backends plus a mock backend (Model Serving, Component Interfaces), the sandbox (Execution Backends, Sandbox), and the compile-only path (Execution Backends, executor `none`; Harness Contract). The monorepo files, hooks, CI workflow, and tools under `tools/` and `tests/tools/` already exist (plans/PHASE-NOTES.md, P0 Core); P0 verifies and extends them and does not rewrite them. The gate is the Gate column of the P0 row, run exactly (task P0.G).

Constraints for every task:

- The origin is github.com/JoeMad21/lassi (OQ-005, answered 2026-09-23); push the phase branch after every task (AGENTS.md, Task Lifecycle step 8).
- Keep `du -sh /mnt/nvme10/joseph_ufl` under 120G as far as possible (owner cap, 2026-09-23; bible Decision Log). Check it before any install or big job, never delete files there, back files up to the workstation to save space (a large folder only with the owner's approval; removing a host copy after a backup also needs it), and recommend deletions in the owner queue.
- Remote tasks go through `uv run tools/rx.py` only. The first remote task runs `rx doctor`; if the gate is unreachable or not installed, set every remote task BLOCKED with that evidence and add an access item to the owner queue.
- No project-specific code in `lassi/` (Agent Rules 3); test fixtures and recipes live under `tests/` and `projects/`.
- Scope is frozen (owner, 2026-09-23): P0 ends with the tasks already in this plan (P0.17, P0.19, then P0.G). Add no new P0 task. Record anything new a P0 task finds in plans/PHASE-NOTES.md under the phase that should take it, or in the owner queue when it needs a decision.

## Tasks

### P0.1 Package skeleton and the twelve interfaces
- Bible: Repository Layout, Component Interfaces (table and contract rules), Design Principles 9, Readability Standards (Code).
- Accept: `uv run pytest -q tests/core/test_interfaces.py` passes and checks that `lassi/core/interfaces.py` defines exactly the Protocols named in the Component Interfaces table, each with a docstring and type-hinted methods; every component can declare capabilities; a test scans `lassi/` and fails on any import of `projects`; `uv run ruff check .` is clean.
- Files: `lassi/__init__.py`, `lassi/core/__init__.py`, `lassi/core/interfaces.py`, `lassi/core/capabilities.py`, `tests/core/test_interfaces.py`, `pyproject.toml` (package build config).
- Remote: none. Depends: none.

### P0.2 Result record, text store, trial.md, Parquet mirror
- Bible: Result Record, Readability Standards (Trial row, Naming), Design Principles 2 and 7.
- Accept: dataclasses for Trial, Attempt, Diagnostic round-trip through JSON with field names exactly as the Result Record block; `trial_id` is validated against the naming rule; large texts are stored once by sha256 and referenced; `render_trial_md` matches a committed golden file that contains each prompt, each attempt's code, the unified diff, parsed diagnostics, and the score breakdown; unmeasured score or profile values render as PLACEHOLDER; a run of trials aggregates to Hive-partitioned Parquet that reads back to the same rows; output is plain ASCII.
- Files: `lassi/core/record.py`, `lassi/core/store.py`, `lassi/core/trial_md.py`, `lassi/core/parquet.py`, `tests/core/test_record.py`, `tests/core/golden/trial.md`, `pyproject.toml` (pyyaml, pyarrow).
- Remote: none. Depends: P0.1.

### P0.3 Recipe loader
- Bible: Project Recipes (base.yaml, notes), Design Principles 1, 4, 5, Component Interfaces (capability validation), Readability Standards (Config).
- Accept: `extends` chains resolve with child-wins merge; `faithful: true` overrides `loop.max_corrections` per the recipe notes; the resolved recipe serializes to YAML and `recipe_hash` is its sha256 and is stable across runs; a recipe that asks for a capability no bound component declares fails at load with a message naming both, before any backend is constructed; an unknown key fails loudly; a required choice with no value fails with a clear message rather than defaulting. `projects/base.yaml` is committed matching the bible block, with a comment on every non-default value.
- Files: `lassi/core/recipe.py`, `lassi/core/registry.py`, `projects/base.yaml`, `tests/core/test_recipe.py`, `tests/fixtures/recipes/`.
- Remote: none. Depends: P0.1.

### P0.4 LLM backends: mock, openai_compat, ollama
- Bible: Component Interfaces (LLMBackend), Model Serving and Serving Rules (port, `/v1/models` check), Agent Rules 12, Risks And Questions (mock LLM mitigation).
- Accept: all three implement LLMBackend and return text plus token counts; the mock returns the reference target for a bench item wrapped in `// FILE:` blocks; openai_compat and ollama are tested against a local stdlib HTTP stub (no network) including a refusal when `/v1/models` lacks the expected id; API keys come only from environment variables and a test asserts no key is logged or written to the record; sampling parameters land in the Trial `model` field.
- Files: `lassi/llm/__init__.py`, `lassi/llm/mock.py`, `lassi/llm/openai_compat.py`, `lassi/llm/ollama.py`, `tests/llm/`.
- Remote: none. Depends: P0.1, P0.2.

### P0.5 FILE-block parser and nvcc, nvc++ toolchain adapters
- Bible: Harness Contract (FILE blocks, missing file), Component Interfaces (Toolchain contract rules), Result Record (Diagnostic).
- Accept: the parser extracts `// FILE: <relative path>` blocks, rejects absolute and `..` paths, and reports a missing expected file as a compile-stage error Diagnostic; nvcc and nvc++ adapters build command lines for the targets in the lassi-repro recipe and parse captured stderr fixtures into Diagnostic records with every field set; raw stderr is kept as an attachment only.
- Files: `lassi/core/files.py`, `lassi/toolchains/__init__.py`, `lassi/toolchains/nvcc.py`, `lassi/toolchains/nvcpp.py`, `tests/toolchains/fixtures/`, `tests/toolchains/test_diagnostics.py`.
- Remote: parser and adapter tests run locally on hand-written stderr fixtures and do not need P0.7; replacing them with fixtures captured on alpha01 (`rx run`) is a follow-up once P0.7 is done. Depends: P0.2.

### P0.6 Spike: nvcc and nvc++ on alpha01
- Bible: Execution Backends (executor `none`), Host Facts, Toolchain Pins.
- Accept: `plans/spikes/p0-nvcc.md` records `rx doctor` output and `rx exec -- 'command -v nvcc nvc++; ls /usr/local /opt; df -h /mnt/nvme10'` output with date; states which compilers exist, their versions and paths, and whether they cross-compile for sm_80 without a GPU; if the none-executor Status line changes, the bible is updated with a Decision Log entry.
- Files: `plans/spikes/p0-nvcc.md`, `docs/BIBLE.md` if changed.
- Remote: `rx doctor`, `rx exec`. Depends: none.

### P0.7 Pin CUDA and NVHPC under $LASSI_TOOLCHAINS
- Bible: Toolchain Pins, Execution Backends, Agent Rules 7 and 10.
- Accept: `toolchains/cuda.pin` and `toolchains/nvhpc.pin` record version, build or install flags, and install path; `toolchains/cuda.sh` and `toolchains/nvhpc.sh` install user-space (CUDA runfile toolkit-only, NVHPC tarball) under `$LASSI_TOOLCHAINS/<name>@<pin>` and are idempotent; `rx run -- 'nvcc --version; nvc++ --version'` through the pinned paths matches the pin files; the pin choice is a Decision Log entry. If P0.6 finds usable system installs, pin those instead and skip the install.
- Files: `toolchains/cuda.pin`, `toolchains/cuda.sh`, `toolchains/nvhpc.pin`, `toolchains/nvhpc.sh`, `docs/BIBLE.md`.
- Remote: `rx doctor` first (free space), then `rx job start --big` per install. Depends: P0.6.

### P0.8 Bench registry and one pinned HeCBench app
- Bible: Benchmark Suites (lassi-hecbench-10, split rules), Agent Rules 5, Design Principles 3.
- Accept: `assets/bench/lassi-hecbench-10.yaml` lists at least one app with HeCBench commit, OpenMP and CUDA source paths, and split `eval`; the registry loads it, returns a bench item with its reference target per direction, and raises when an eval item is requested for training; a fetch script materializes the pinned sources under `$LASSI_SCRATCH`, never into git.
- Files: `lassi/bench/__init__.py`, `lassi/bench/registry.py`, `assets/bench/lassi-hecbench-10.yaml`, `tools/fetch_bench.py` or `toolchains/hecbench.sh`, `tests/bench/`.
- Remote: `rx run` to fetch and verify the pinned sources. Depends: P0.1.

### P0.9 Spike: sandbox isolation on alpha01
- Bible: Sandbox, Agent Rules 6 and 7.
- Accept: `plans/spikes/p0-sandbox.md` records, for bubblewrap, `unshare -rn`, Apptainer, and `systemd-run --user`, the exact command, output, and whether it (a) runs unprivileged, (b) blocks network (a connect attempt fails), (c) enforces memory, CPU, and wall limits, (d) mounts the harness read-only, (e) runs as a separate uid or in a container; names the chosen mechanism with evidence. If none isolates the network, add an owner-queue item and set P0.10 OWNER.
- Files: `plans/spikes/p0-sandbox.md`, `docs/BIBLE.md` (Sandbox, Decision Log).
- Remote: `rx run`. Depends: none.

### P0.10 Sandbox and the none and native executors
- Bible: Sandbox, Execution Backends (none, native), Component Interfaces (Executor contract rules), Agent Rules 6.
- Accept: every executor runs generated code only through the sandbox module; the `none` executor returns a compile-only RunResult; local tests with a fake runner cover limits, hang flag, and exit status; `remote`-marked tests on alpha01 show a network connect fails, a memory hog is killed, a sleep past wall time sets `hang`, and a write to the harness mount fails; per-trial build dirs sit under `$LASSI_RUNS_ROOT`.
- Files: `lassi/executors/__init__.py`, `lassi/executors/sandbox.py`, `lassi/executors/none.py`, `lassi/executors/native.py`, `tests/executors/`.
- Remote: `rx run -- 'uv run pytest -q -m remote tests/executors'`. Depends: P0.1, P0.9.

### P0.11 Stage runner and `lassi run` compile-only path
- Bible: Component Interfaces (Stage contract rules), Result Record, Project Recipes, Readability Standards (Run row), Design Principles 5, Agent Rules 10.
- Accept: stages are pure over the Trial; the runner executes generate then compile loop with the correction bound from the recipe; `lassi run <recipe>` writes the resolved recipe, toolchain pins, one JSON and one `trial.md` per trial, `run.md`, and Parquet under `$LASSI_RUNS_ROOT`; a local test runs the mock backend against a fake toolchain end to end; a test fixture recipe (under `tests/fixtures/recipes/`) selects mock, one app, one direction, executor `none`.
- Files: `lassi/core/runner.py`, `lassi/core/stages.py`, `lassi/cli.py`, `tests/core/test_runner.py`, `tests/fixtures/recipes/p0-smoke.yaml`.
- Remote: none for tests. Depends: P0.2, P0.3, P0.4, P0.5, P0.8, P0.10.

### P0.12 Verify text-policy tooling, local half
- Bible: Attribution Policy, Enforcement items 1, 3, 5; Agent Rules 15 and 16.
- Accept: `uv run tools/check_setup.py` reports hooks installed and the pattern file present; `uv run tools/policy_canary.py local` reports both seeded commits blocked; `uv run tools/check_text_policy.py --history` passes on the branch; any gap found in `tests/tools/` gets a new test, never a weakened checker.
- Files: `tests/tools/` additions only if gaps are found.
- Remote: none. Depends: none.

### P0.13 Verify text-policy tooling, CI half
- Bible: Attribution Policy, Enforcement item 2.
- Accept: `uv run tools/policy_canary.py push`, then `status` shows the CI run on branch `p0-canary` failed, then `cleanup` removes it.
- Files: none (evidence goes under `results/p0-gate/` in P0.G).
- Unblocked 2026-09-23: origin exists (OQ-005), the Actions variable is set and main requires the text-policy check (OQ-007).
- Remote: GitHub only. Depends: P0.12.

### P0.14 Fix rx local-transport tests on Windows
- Bible: none beyond AGENTS.md Remote Execution and Repository Checks (the fast suite must pass).
- Accept: the five `tests/tools/test_rx_gate.py` tests that error at baseline b092ea2 with `rx: remote setup failed:` and empty output pass on the Windows workstation under Git Bash and still pass on Linux; the cause is written in the commit body; the gate server under `tools/server/` keeps every refusal it has now.
- Files: `tools/rx.py`, `tests/tools/test_rx_gate.py`.
- Remote: none. Depends: none.

### P0.15 Replace toolchain stderr fixtures with alpha01 captures
- Bible: Component Interfaces (Toolchain contract rules), Result Record (Diagnostic).
- Accept: every file in `tests/toolchains/fixtures/` is replaced by stderr captured on alpha01 with the pinned compilers under `LC_ALL=C` (see plans/spikes/p0-toolchains-verify.md), each with its rx id in the fixtures README; the expected Diagnostic lists and parsers are updated to match; the README no longer says PLACEHOLDER.
- Files: `tests/toolchains/fixtures/`, `tests/toolchains/test_diagnostics.py`, `lassi/toolchains/` if a format differs.
- Remote: `rx run` to capture. Depends: P0.7.

### P0.16 Sandbox hardening before native runs of generated code
- Bible: Sandbox, Agent Rules 5, 6, 7, 9, 12; plans/spikes/p0-sandbox-verify.md and the Known limits in lassi/executors/sandbox.py.
- Accept: `remote` tests on alpha01 show, inside the sandbox: only null, zero, full, random, urandom, tty, a private devpts, and shm exist under /dev (no accelerator or other host device node); every mount except the workdir and the private tmpfs mounts is read-only, checked from /proc/self/mountinfo, and setup fails closed otherwise; $HOME and the scratch root are hidden except the workdir, the harness, and the pinned toolchains; stdout and stderr are capped with a truncation flag; workdir disk use is capped; a crashing program leaves no core with the host handler; when the runner's timeout fires, no process of the sandbox survives. Local tests pin the new setup steps; the bible Sandbox bullet and Decision Log are updated.
- Files: `lassi/executors/sandbox.py`, `lassi/toolchains/_base.py` (runner kill), `tests/executors/`, `docs/BIBLE.md`.
- Remote: `rx run -- 'LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote tests/executors'`. Depends: P0.10. Any task that runs generated code on the native executor depends on P0.16.

### P0.17 Parse the compiler error formats the P0.15 probes found unread
- Bible: Component Interfaces (Toolchain contract rules), Result Record (Diagnostic).
- Seen only in exploratory dirty-tree probes during P0.15 (not reportable): nvlink in rx 20260923-104618-desktop-8r113ei-p0-core-173d, the ptxas file,line error in rx 20260923-104516-desktop-8r113ei-p0-core-5778, and `nvc++-Fatal-... TERMINATED by signal 11` in rx 20260923-104313-desktop-8r113ei-p0-core-2b22; `nvc++-Error-` appears in plans/spikes/p0-toolchains-verify.md.
- Accept: new scenarios in `tests/toolchains/fixtures/scenarios.json`, with sources, whose captures on alpha01 (from a clean commit, as in P0.15) show an nvlink undefined reference from an OpenMP target region, a ptxas error with a PTX file and line, and an nvc++ driver error line (`nvc++-Error-` or `nvc++-Fatal-`) when one can be produced from sources with the preset flags. Each capture parses into the Diagnostics the Toolchain contract calls for, with no fall back to the "exit-status" error. The fixtures README lists each with its rx id. A format that cannot be produced is recorded in the README with the attempts.
- Files: `tests/toolchains/fixtures/`, `tests/toolchains/test_diagnostics.py`, `lassi/toolchains/`.
- Remote: `rx run` to capture. Depends: P0.15.

### P0.18 Carry run provenance in each Trial
- Bible: Result Record (Trial provenance, OQ-008 decision of 2026-09-23), Agent Rules 1 and 2; AGENTS.md Results.
- Accept:
  - `Trial` in `lassi/core/record.py` gains `provenance` as the bible's Result Record defines it (commit, dirty, device, sdk, date), strictly typed like the other record fields.
  - The runner fills it from the same values it writes to the run's provenance.json, so the two can never disagree; a test checks every trial's copy against the manifest.
  - trial.json round-trips it. trial.md shows it. The Parquet trials table carries it as columns.
  - A record without provenance is refused, with a clear message.
- Files: `lassi/core/record.py`, `lassi/core/runner.py`, `lassi/core/trial_md.py`, `lassi/core/parquet.py`, `lassi/core/store.py` if trial.json changes, tests.
- Remote: none. Depends: P0.11. P0.G depends on it, since the gate run is the first to write records.

### P0.19 Install CUDA 12.6.3 from the redistributable archives
- Bible: Toolchain Pins (OQ-010 decision of 2026-09-23), Agent Rules 7 and 10.
- Accept:
  - `toolchains/cuda.sh` installs the toolkit components the pinned nvcc path and the nvcc-sm80 compiles need from NVIDIA's per-component redistributable archives for 12.6.3, each checked against the sha256 in NVIDIA's published redistrib manifest. No runfile or other installer runs, and nothing is written outside the scratch disk; the job checks this with a find over /tmp and /var/tmp.
  - `toolchains/cuda.pin` records the manifest URL and its checksum, the component list with versions and sha256, and the install layout. PREFIX_NAME and EXPECT_VERSION stay the same.
  - The install goes to a staging prefix first. It replaces `$LASSI_TOOLCHAINS/cuda@12.6.3` only after nvcc reports EXPECT_VERSION and the pinned nvcc compiles the P0.7 run 2 checks again: sm_80 for nvcc, and as NVHPC_CUDA_HOME for nvc++ -gpu=cc80. The runfile install is kept aside until the new one passes; agents may not delete it, so its removal goes to the owner queue.
  - The P0.15 fixture capture, rerun with the new install from a clean commit, gives byte-identical stderr for the 12 byte-stable scenarios. Any difference is recorded.
  - The pin change is a Decision Log entry.
- Files: `toolchains/cuda.sh`, `toolchains/cuda.pin`, `tests/toolchains/test_pins.py` (its checks of the runfile URL, MD5, and log trap change), `docs/BIBLE.md`, `plans/spikes/p0-cuda-redist.md`.
- Remote: `rx doctor` and `du -sh /mnt/nvme10/joseph_ufl` first; the archive downloads plus the staging prefix plus current use must stay under the 120G cap, else free space through the owner queue before the job. Then `rx job start --big`. Depends: P0.7.

### P0.20 Harden compiles of generated sources
- Bible: Sandbox, Toolchain Pins, Agent Rules 6, 7, 10, 12. See also plans/PHASE-NOTES.md (P0.20 scope) and the P0.16 spike handoff.
- Accept: compiles of model-generated sources cannot read files under $HOME, the scratch root, or the runs root other than the build directory and the pinned toolchains (a generated `#include` of an absolute path there fails, tested locally and on alpha01); files elsewhere keep host read permissions, as in the P0.16 sandbox, and the task states that limit; the compile environment no longer carries HOME; the runner refuses a TMPDIR outside $LASSI_SCRATCH, and each compile gets a private temporary directory inside its own view (under its build directory), so the rest of the scratch root stays hidden; the runner checks a pinned compiler's `--version` against its pin's EXPECT_VERSION before the first build; compiles run under prlimit --core=1 so a compiler crash stores no core with the host handler (tested with a crash stand-in, never by crashing a real compiler on purpose). The P0.15 fixture capture, rerun from a clean commit, still gives byte-identical stderr for its 12 byte-stable scenarios.
- Files: `lassi/core/runner.py`, `lassi/toolchains/_base.py`, `lassi/executors/sandbox.py` if the compile reuses its view, tests.
- Remote: `rx run` for the remote tests and the fixture recapture. Depends: P0.16. Needed before real model arms run (P3).

### P0.G Phase gate
- Bible: Build Roadmap, P0 row, Gate column; AGENTS.md Phase Gate and Results.
- Accept: (1) on alpha01 from a clean tree, `rx run -- 'uv run lassi run tests/fixtures/recipes/p0-smoke.yaml'` with the mock backend takes one HeCBench app end to end compile-only and reaches the compile stage with a built artifact; `rx pull` writes `results/p0-gate/provenance.json` and `summary.md` cites it; (2) `policy_canary.py local` output shows the seeded commit blocked locally; (3) `policy_canary.py push` and `status` show it blocked in CI. All three recorded under `results/p0-gate/`. Pass sets P0 DONE and opens a pull request `P0 Core` (needs origin).
- Files: `results/p0-gate/`.
- Remote: `rx run`, `rx pull`. Depends: P0.7, P0.11, P0.12, P0.13, P0.14, P0.18 (P0.14 and P0.18 are tracked in the STATUS note, since status.py cannot edit dependencies).
