# P1 Faithful LASSI

Branch `p1-faithful`, base `main` at c3cf248 (P0 merged, PR 1). The branch tip c9771d3 records the merge and closes OQ-017.

## Scope

P1 builds what the P1 row of the bible's Build Roadmap names: the LASSI stages (Source Papers, LASSI, pipeline steps 1 to 5; Component Interfaces, Stage row), the `lassi-2024` prompts and the two context packs (Repository Layout, `assets/`; Design Principle 3), the faithful toggles for the upstream quirks (Source Papers, LASSI quirk table; Design Principle 4; Project Recipes, Notes), the pinned HeCBench suite (Benchmark Suites, lassi-hecbench-10), the stdout_mask and passfail oracles (Oracles; Harness Contract), and Sim-T and Sim-L (Evaluation Protocol, LASSI reproduction row). P0 already provides the record, the recipe loader with the `faithful` override and the `fence_tag` fix, the mock, openai_compat, and ollama backends, the FILE-block parser, the nvcc-sm80 and nvcpp-cc80 toolchains in the compile sandbox, the bench registry with `layout`, the generate and compile_loop stages, and `lassi run`. The gate is the Gate column of the P1 row, run exactly (task P1.G).

Constraints for every task:

- The scope is fixed at planning (owner, 2026-09-23). A later finding goes into plans/PHASE-NOTES.md under a later phase, or into the owner queue. It never becomes a new P1 task.
- Hardware: only compile-only work on alpha01 (the pinned nvcc and nvc++, fetches, remote tests). No GPU, RNGD, or Tenstorrent device is used. Generated code is compiled in the compile sandbox and never executed; run_loop is tested with scripted executors.
- The working changes in plans/runs/p0-retrospective.md apply: one review round per task (contract, safety, and rules together) and one commit audit; an evidence checklist in each contract ([MEASURED] only with a clean-commit rx id, everything else labeled exploratory); before an evidence run, commit everything, confirm a clean tree, and record pytest's own exit status (PIPESTATUS); rx output under 120 lines; remote jobs run while local tasks proceed.
- Scratch: about 94G of the owner's 120G cap is in use (rx 20260923-222319-exec-07d8, after OQ-017). Run `rx doctor` and `du -sh /mnt/nvme10/joseph_ufl` before any fetch or clone there. Never delete files there.
- Upstream code runs only in the P1.8 equivalence test and the P1.9 replay, with every subprocess and network call stubbed and a guard that fails the test if one starts. Code from models, the mock, or fixtures is never executed (Agent Rule 6). No project-specific code enters `lassi/` (Agent Rule 3). Nothing in `third_party/` is edited.
- Upstream text: until P1.12 applies OQ-018, no string from `prompt_dictionary.py` or the notebook enters a tracked file (code, tests, fixtures, assets). `tools/extract_lassi_assets.py` generates it into gitignored trees; only manifests (keys, source cell ids, sha256) are committed.
- Tests that need `third_party/LASSI` or the generated text generate it into `tmp_path` through the extractor, or skip with a reason that names the tool. They never read files left by an earlier commit. The loader checks every generated file against its committed manifest and fails on a mismatch.

Decisions taken at planning (the named task records each one in the Decision Log):

- Recorded responses for the replay gate are scripted fixtures, labeled synthetic (P1.9). Upstream published no generated code (Source Papers, LASSI), and no model arm can serve before P3. The replay's fence-quirk hit count checks decision logic on synthetic input; it is not the Evaluation Protocol's fence-quirk replay count and is never reported as a measurement.
- Mock dry runs go through faithful fence stripping (P1.10): under faithful extraction the mock answers in one untagged fence, the form upstream's system prompts ask for.
- The resolved recipe carries its project name in an explicit `project` key (P1.10). This changes every recipe hash from P1 on.
- Faithful mode follows the notebook where the bible's pipeline summary differs: the baseline builds and runs only the target reference, and the correction prompt renders upstream's compiler text (P1.5).

PHASE-NOTES P1 items, by task: rerun ids, P1.10; p0-smoke to one item, P1.2; the upstream pin, P1.1; the HeCBench pin, the nine other apps, and `reference.h`, P1.1 and P1.2; CUDA components, P1.2; recorded responses, P1.9; compiler output facts, P1.5 and P1.9; the mock fence quirk, P1.10; run flags, P1.6; correction prompt diagnostics, P1.5; toolchain follow-ups, P1.11. Not taken: the furiosa backend choice (P3 scope).

Owner queue: OQ-018 (upstream LASSI text in the public repository) blocks only P1.12.

## Tasks

### P1.1 Pin upstream LASSI and check the HeCBench pin against its sources
- Bible: Source Papers (LASSI: upstream repository line; quirk table, entropy row), Repository Layout (`third_party/`), Benchmark Suites, Toolchain Pins, Agent Rules 4 and 10.
- Accept: `assets/upstream/lassi.yaml` pins upstream at 74b4681 and `uv run tools/fetch_upstream.py` checks it out into the gitignored `third_party/LASSI` (Decision Log, 2026-09-24: a manifest and a fetch tool replace the submodule, because the text-policy check refuses a staged gitlink; OQ-020). `rx run -- 'uv run tools/fetch_upstream.py && git -C third_party/LASSI rev-parse HEAD'` prints the same hash. `plans/spikes/p1-hecbench-pin.md` lists, for each of the 20 upstream `*_main` files, its blob id and whether HeCBench 7d2d3c5 holds it at `src/<app>-<omp|cuda>/main.*`, plus any HeCBench commits holding all 20, with commands and outputs. The spike names the source of each model-facing file and the HeCBench pin for support files; a pin change is a Decision Log entry.
- Files: `assets/upstream/lassi.yaml`, `tools/fetch_upstream.py`, `tests/bench/test_upstream_pin.py`, `.gitignore`, `plans/spikes/p1-hecbench-pin.md`, `docs/BIBLE.md` if the pin changes.
- Remote: `rx run` for the fetch check; a blobless HeCBench clone, locally or under `$LASSI_SCRATCH` after the space check. Depends: none.

### P1.2 Ten-app bench manifest, support files, and item selection
- Bible: Benchmark Suites (lassi-hecbench-10, split rules), Source Papers (LASSI quirk table: entropy and PASS/FAIL rows), Harness Contract, Design Principle 3, Agent Rule 5.
- Accept:
  - `assets/bench/lassi-hecbench-10.yaml` lists the 10 apps of upstream's `*_main` files in both languages, at the source and pin P1.1 chose; each model-facing file carries the sha256 of upstream's file. Each item records the run arguments of the `experimental_setup` docstring exactly (jacobi keeps `[""]`, one empty argument; the DNU entry is left out) and which languages print PASS/FAIL.
  - Support files land in every build directory of their item as harness files, and a model file with the same name cannot replace them (test). The known case is entropy's `reference.h` from pinned HeCBench. No app needs a CUDA component beyond the pinned archives.
  - A recipe can select items; `tests/fixtures/recipes/p0-smoke.yaml` selects `layout` only.
  - A `remote` test fetches the manifest and compiles all 20 reference programs with the pinned toolchains in the compile sandbox, reporting 20/20 or each failure. Evidence: run from a clean commit, its rx id in the P1.2 STATUS note, and `rx pull` into `results/p1-hecbench-compile/` with `provenance.json` and a `summary.md`.
- Files: `assets/bench/lassi-hecbench-10.yaml`, `lassi/bench/registry.py`, `tools/fetch_bench.py`, `lassi/core/recipe.py`, `lassi/toolchains/_base.py`, `tests/fixtures/recipes/p0-smoke.yaml`, `tests/bench/`, `tests/toolchains/`, `results/p1-hecbench-compile/`.
- Remote: `rx run` of the fetch and the remote test, then `rx pull`. Depends: P1.1.

### P1.3 lassi-2024 prompt set and context packs from pinned upstream
- Bible: Source Papers (LASSI pipeline steps 2 to 4), Design Principles 3 and 4, Readability Standards (Prompts row), Repository Layout (`assets/prompts`, `assets/context`).
- Accept:
  - `tools/extract_lassi_assets.py` refuses a checkout not at 74b4681. Without importing anything from upstream, it writes the `prompt_dictionary.py` values and the notebook literals the stages use (summary and description prompts, assembly wrappers, correction intro and outro, and the compiler and flag text of `experimental_setup`), parsed from the cells with `ast`, as named fragments under `assets/prompts/lassi-2024/`, and the packs under `assets/context/openmp-4.0-card/` and `assets/context/cuda-12.5-ch5/`. Each tree has a committed `MANIFEST.yaml` (key, source file and cell id, sha256). Rerunning it changes nothing.
  - `.gitignore` ignores those trees except `MANIFEST.yaml`: `git ls-files assets/prompts/lassi-2024 assets/context` lists only the manifests, and a test fails if any tracked file outside `third_party/` contains a generated fragment of 40 or more characters.
  - A test renders every fragment and compares it byte for byte with the value parsed from the pinned checkout.
  - The loader returns the packs and fragments a recipe names. It fails, naming the tool, when a file is missing or its sha256 differs from the manifest, and fails loudly on an unknown pack.
- Files: `tools/extract_lassi_assets.py`, `assets/prompts/lassi-2024/MANIFEST.yaml`, `assets/context/*/MANIFEST.yaml`, `lassi/prompts/`, `.gitignore`, `tests/prompts/`.
- Remote: none. Depends: P1.1.

### P1.4 Faithful generation: summarize_context, describe_source, generate
- Bible: Source Papers (LASSI pipeline steps 2 to 4; quirk table, fence row), Component Interfaces (Stage row and contract rules), Result Record (Trial.context, Diagnostic), Project Recipes (Notes), Design Principle 4.
- Accept:
  - summarize_context and describe_source send upstream's prompts as [system, user] with upstream's general system prompt, and fill `Trial.context`. `context` leaves the runner's not-carried-out list, and the runner passes the recipe's packs to the stages.
  - Faithful generate sends the direction's system prompt and the assembled prompt, with runs of spaces collapsed as upstream does (source indentation included). This quirk joins the bible's quirk table with a Decision Log entry.
  - Under faithful extraction the attempt's one target file is the first fenced block after upstream's tag stripping; a cuda-tagged fence yields text starting "uda", and the attempt carries a parse-stage warning Diagnostic with code `fence-quirk`, which a metric can count.
  - With fixes on, the P0 FILE-block path and its tests are unchanged. Each reproduced quirk is a named fix in `FIXES`, off under `faithful: true`; the runner refuses a fix turned off only when no bound stage reproduces its quirk.
- Files: `lassi/core/stages.py` or a new stage module, `lassi/core/recipe.py`, `lassi/core/runner.py`, `tests/core/`, `docs/BIBLE.md`.
- Remote: none. Depends: P1.3.

### P1.5 Baseline stage and the faithful correction loop
- Bible: Source Papers (LASSI pipeline steps 1 and 4; quirk table, loop row), Component Interfaces (Toolchain contract rules), Harness Contract, Result Record, Design Principle 4.
- Accept:
  - Faithful baseline builds only the target reference, and runs it when the executor runs programs, as the notebook does. A named fix, on outside faithful, builds and runs both programs. A failure ends the trial before any model call.
  - The Result Record gains `final.end_reason` (a fixed code and a message; this task uses `baseline-compile` and `baseline-run`), shown in `trial.md`.
  - Faithful correction prompt for a compile error: the previous code, upstream's compiler and flag text for the target language (the extracted `experimental_setup` literals, never our pinned executable path), the whole raw `compile.stderr` attachment, and the outro, with every newline removed; sent with the direction's system prompt; no cap. A test checks that the pinned toolchain flags equal upstream's flag text.
  - With fixes on, a correction prompt carries parsed diagnostics capped in count and bytes, and says how many it left out.
  - Bible edits with Decision Log entries: quirk table rows for the target-only baseline and the newline removal; a note on pipeline step 1; `final.end_reason`; a faithful exception to the raw-stderr Toolchain contract rule.
- Files: the stage module, `lassi/core/record.py`, `lassi/core/recipe.py`, `lassi/toolchains/_base.py` if the command text is exposed there, `tests/core/`, `docs/BIBLE.md`.
- Remote: none. Depends: P1.2, P1.4.

### P1.6 run_loop: execution gate, stale output, run flags, Ollama unload
- Bible: Source Papers (LASSI quirk table, loop and Ollama rows), Component Interfaces (Executor rules; capability rule), Execution Backends (none), Result Record (Attempt.run, Diagnostic), Sandbox.
- Accept:
  - With executor `none`, run_loop records nothing, and a faithful trial ends at its first compiling attempt.
  - With a scripted executor, a faithful trial runs a compiling attempt only while its correction count is at most 7; a run error gets upstream's execute-error prompt and shares the correction count with compile errors.
  - A compiling attempt after 8 or more corrections ends the trial unexecuted. If an earlier attempt ran, that attempt's stdout stands as the trial's output and the last attempt carries a run-stage warning Diagnostic with code `stale-output` naming it. If none ran, the trial ends with `final.end_reason` `upstream-crash`, no stdout, and no alignment (the notebook raises there).
  - With fixes on, the recipe's cap applies and no stale output is used.
  - Truncated stdout or stderr and an incomplete workdir copy-back become run-stage warning Diagnostics.
  - A backend declaring the capability `unload_before_run` is asked to unload at trial start (the notebook's setup unload) and before each execution; the ollama backend declares it (HTTP stub test), and a backend without it is never asked. No stage checks a backend's type.
- Files: the stage module, `lassi/core/capabilities.py`, `lassi/llm/ollama.py`, `tests/core/`, `tests/llm/`.
- Remote: none. Depends: P1.5.

### P1.7 Oracles: stdout_mask and passfail
- Bible: Oracles, Harness Contract, Component Interfaces (Oracle row and rules), Source Papers (LASSI quirk table, PASS/FAIL row), Repository Layout (`assets/harness/`).
- Accept:
  - `stdout_mask` [DESIGN]: per_input is 1.0 when stdout equals the reference stdout after the item's timing lines are masked, else 0.0; mean is the mean over inputs. Tests assert exact values on fixture stdout. The rule goes into the bible's Oracles section with a Decision Log entry.
  - Every item has masks under `assets/harness/`, derived from its sources' print statements.
  - `passfail` reads PASS or FAIL only where the manifest says the language prints it; a PASS whose masked stdout differs is not a pass; items without PASS/FAIL use stdout_mask alone.
  - `oracle` leaves the runner's not-carried-out list. The oracle stage fills alignment when a run produced stdout (the stale stdout of P1.6 included) and leaves it None for compile-only attempts.
- Files: `lassi/oracles/`, `assets/harness/`, the stage module, `lassi/core/runner.py`, `tests/oracles/`, `docs/BIBLE.md`.
- Remote: none. Depends: P1.2.

### P1.8 Sim-T and Sim-L
- Bible: Source Papers (LASSI results table; quirk table, Sim-T row), Evaluation Protocol (LASSI reproduction row), Repository Layout (`scoring/`).
- Accept:
  - The faithful Sim-T and Sim-L equal upstream's `token_similarity(..., "tokenize")` and `compare_lines_with_reordering`, taken from the pinned notebook cell with `ast` and run under the P1.9 guard. They match on every ordered pair of the 20 `*_main` files (a `slow` test) and on edge cases (empty text, tokenize errors) plus one pair per app in the fast suite.
  - A C-aware Sim-T exists under its own name, with a documented lexer. The interpreter version is recorded wherever a value is written. Wiring into metrics is P2's work.
- Files: `lassi/scoring/__init__.py`, `lassi/scoring/similarity.py`, `tests/scoring/`.
- Remote: none. Depends: P1.1.

### P1.9 Replay backend and the upstream notebook replay harness
- Bible: Build Roadmap (P1 row, Gate), Source Papers (LASSI pipeline; quirk table), Design Principle 4, Agent Rules 1, 4, and 6.
- Accept:
  - A `replay` LLMBackend returns recorded completions in order and fails when they run out.
  - `tests/replay/` runs the notebook's pipeline function (its cells from `third_party/LASSI`, in a temporary directory) and our faithful stages on the same recorded responses and outcomes. The notebook's LLM, compile, execute, unload, and tokenizer-import calls are stubbed; a test fails if either side starts a subprocess or opens a socket.
  - Per scenario it compares every message sent, each extracted block, which attempts compiled and ran, the final correction count, Sim-T, and Sim-L.
  - Scenarios, in both directions: first-try success; fences tagged cpp, c++, c, or cuda, and untagged; no fence, several fences; compile error then fix (stderr from `tests/toolchains/fixtures/`); run error then fix; a run, then compile errors past 8 corrections, ending unexecuted with stale output; 9 or more compile errors with no earlier run (the notebook raises UnboundLocalError; ours ends `upstream-crash`); baseline compile failure; baseline run failure; runs of spaces in the source.
  - Fixtures are scripted, plain ASCII, and labeled synthetic in a README; they copy no upstream source or text. Decision Log entry: the replay gate reads recorded responses as scripted synthetic fixtures.
  - The output shows a decision table per scenario and the fence-quirk hit count, labeled a check of decision logic on synthetic fixtures, not a measurement.
- Files: `lassi/llm/replay.py`, `tests/replay/`, `tests/fixtures/replay/`, `docs/BIBLE.md`.
- Remote: none. Depends: P1.6, P1.8.

### P1.10 lassi-repro recipes, project name, and the mock dry run
- Bible: Project Recipes (lassi-repro block and Notes), Design Principles 1, 4, and 5, Readability Standards (Naming, Config), Evaluation Protocol (Acceptance Criteria), Risks And Questions (mock LLM).
- Accept:
  - An explicit `project` key, inherited through `extends` (nearest file wins), names the first trial id segment; without it the recipe name does. The resolved recipe always writes `project`, so a rerun from `recipe.resolved.yaml` keeps the trial ids (test). The key and the hash change are a Decision Log entry.
  - `projects/lassi-repro/recipe.yaml` sets `project: lassi-repro`, loads, and matches the bible block except keys P1 cannot carry out, which are left out with a comment naming their phase (`metrics` P2, `arms` P3, executor `gpu` P10). It leaves `max_tokens` unset, with a comment naming P3 (upstream sets none).
  - `tests/fixtures/recipes/p1-dry-run.yaml` extends it with the mock backend, `max_tokens` for the mock, executor `none`, all 10 items, both directions, n = 1. Its trial ids start with `lassi-repro/`. The mock's faithful reply form (one untagged fence) is a Decision Log entry.
  - A local dry run with a fake toolchain reaches S4 at attempt 0 for all 20 trials; a reference that fails to compile ends its trial with `baseline-compile`.
- Files: `lassi/core/recipe.py`, `lassi/core/runner.py`, `lassi/llm/mock.py`, `projects/lassi-repro/recipe.yaml`, `tests/fixtures/recipes/p1-dry-run.yaml`, `tests/core/`, `docs/BIBLE.md`.
- Remote: none. Depends: P1.6, P1.7.

### P1.11 Toolchain follow-ups deferred from P0
- Bible: Component Interfaces (Toolchain contract rules), Result Record (Diagnostic).
- Accept: every item of the PHASE-NOTES P1 bullet "Toolchain follow-ups" is done: comments in `nvcc.py`, `nvcpp.py`, and `_stderr.py` cite the clean capture (rx 20260923-211958-desktop-8r113ei-p0-core-d221); a test ties the linker sample lines in `test_diagnostics.py` to the fixture files; the NVC++ backend line without a line number and the `<inline asm>` line parse to a file and line that index a built file, or to None.
- Files: `lassi/toolchains/`, `tests/toolchains/test_diagnostics.py`.
- Remote: none. Depends: none.

### P1.12 Apply OQ-018: upstream text in the repository
- Bible: Repository Layout, Design Principle 3, Attribution Policy.
- Accept: the answer is applied to both generated trees (dictionary and notebook fragments, and context packs): each part is either committed with a NOTICE naming upstream's commit and license, or stays generated and the bible's layout note says so (Decision Log). The commit passes the text-policy hooks.
- Files: `.gitignore`, `assets/prompts/lassi-2024/`, `assets/context/`, `docs/BIBLE.md`.
- Remote: none. Depends: P1.3. Starts OWNER (OQ-018). The gate does not wait for it: the text is byte-identical either way.

### P1.G Phase gate
- Bible: Build Roadmap (P1 row, Gate column), Evaluation Protocol (Acceptance Criteria, compile-only tier label); AGENTS.md Phase Gate and Results.
- Accept, from one clean commit on alpha01:
  - (1) `rx run -- 'uv run tools/fetch_upstream.py && uv run tools/extract_lassi_assets.py && uv run pytest -q -rs tests/replay'` exits 0 with no skipped test, and every scenario's decisions match upstream's.
  - (2) `rx run` (or `rx job` if it will pass 20 minutes) of `uv run tools/fetch_upstream.py && uv run tools/extract_lassi_assets.py && uv run tools/fetch_bench.py assets/bench/lassi-hecbench-10.yaml && uv run lassi run tests/fixtures/recipes/p1-dry-run.yaml` ends with 20 of 20 trials (10 apps, 2 directions) at S4 on attempt 0 and 20 of 20 baseline target references compiled.
  - `rx pull` writes `results/p1-gate/replay/` and `results/p1-gate/dry-run/` with provenance. `summary.md` cites both, says the replay responses are synthetic and its fence-quirk count is not a reproduction metric, and labels the dry run a compile-stage run with the mock, never compared with the paper.
  - On a pass, set P1 DONE and open the pull request `P1 Faithful LASSI` from `p1-faithful` to `main`.
- Remote: `rx doctor`, `rx run` or `rx job`, `rx pull`. Depends: P1.9, P1.10, P1.11.
