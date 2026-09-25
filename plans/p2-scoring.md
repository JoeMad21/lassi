# P2 Scoring

Branch `p2-scoring`, base `p1-faithful` at 5d5fd0c (P1 DONE; its pull request 2 is open, not merged).

## Scope

P2 builds what the P2 row of the bible's Build Roadmap names: the `df-v0` profile (Training Module, Reward Function) and the `lassi` profile (Evaluation Protocol, LASSI reproduction row; Project Recipes, the lassi-repro `metrics` line), both behind the ScoreProfile interface (Component Interfaces; Repository Layout, `scoring/` and `analysis/`), filling the score fields of the Result Record and run as metrics over P1 trajectories before any reward use (Reward Function, scoring decisions). It also takes the record readings the scores rest on, deferred from P1 in plans/PHASE-NOTES.md (mapping below). The gate is the Gate column of the P2 row, run exactly (P2.G). It is an owner review, so the phase ends GATE-OWNER and work moves to P4 while the review waits (PHASE-NOTES, P0 and P2).

Constraints for every task:

- The scope is fixed at planning (owner, 2026-09-23). A later finding goes into plans/PHASE-NOTES.md under a later phase, or into the owner queue. It never becomes a new P2 task.
- Hardware: none. Remote work is the CPU-only proxy run of P2.5 (the sandboxed native executor, as in the P1 CPU demo) and the gate's scoring of run trees already on alpha01. No RNGD, GPU, or Tenstorrent device is used and no model is served.
- The working changes in plans/runs/p0-retrospective.md apply: one review round per task and one commit audit; an evidence checklist in each contract ([MEASURED] only with a clean-commit rx id, everything else labeled exploratory); before an evidence run, commit everything, confirm a clean tree, and record the command's own exit status; rx output under 120 lines; remote jobs run while local tasks proceed.
- Upstream text (OQ-018): score outputs, the review packet, and every file under `results/` carry no prompt, context, notebook, source, or model text. Plain ASCII throughout.
- Every lassi-hecbench-10 item is eval only (Benchmark Suites). Scoring them as metrics is what the Reward Function asks for before any reward use; nothing here trains, tunes prompts, or harvests (Agent Rule 5). A weight change after the review is a Decision Log entry.
- A score is a value computed over a logged run: it carries the scoring commit and the source run's provenance (Agent Rule 1). A scoring pass from a dirty tree is exploratory.

Decisions taken at planning (the named task records each one in the Decision Log; the P2.G review item puts each to J):

- The gate reviews run demo-rngd-cpu-1 (results/demo-rngd-cpu-run; rx job 20260924-093127-demo-rngd-cpu-1ff6, clean 2f052e6), rescored from a clean P2 commit. It is the only P1 trajectory set whose records reach every term of both profiles: real model replies, S1 to S5, compiler warnings, program runs, oracle alignment, corrections, and cap hits. The mock dry run p1-gate-dry-run (every trial S4 on attempt 0 with the reference as the reply) and the compile-only demo-rngd-1 (no run, no alignment) reach only the stage and warning terms; the gate scores both as a load check, not as the review. The packet states the run's limits: a demo model, not an arm; not faithful (a correction cap of 5 and the -mp=multicore proxy); CUDA to OpenMP only; one trial per app; records that predate P2.1 and P2.2.
- Scoring a finished run writes a new tree under the runs root and never changes the run tree it reads (P2.9).
- The output that stands is the last attempt that ran, stale output included, as the P1.9 replay reads it; final.alignment is its alignment mean (P2.3).
- `correct` in the lassi profile needs the output that stands to come from a clean run (exit 0, no hang) with the oracle at 1.0; the stdout_mask rule itself is unchanged (P2.7; PHASE-NOTES P2, stdout_mask).
- df-v0's W counts the compile- and jit-stage warnings of an attempt at S4 or above, unique by (code, file, line, column); parse- and run-stage pipeline warnings (fence-quirk, stale-output, the truncation flags) are not code warnings (P2.6).
- df-v0's final.score is the multi-turn value (Algorithms, Episodes); attempt 0's R is kept as the single-turn component (P2.6).
- Profile weights and the paper's reference values live in YAML under `assets/scoring/`, a Repository Layout edit (P2.6, P2.8).

PHASE-NOTES P2 items, by task: prompts under a fragment set, P2.1; reference run flags and the base.yaml wall_s comment, P2.2; the stage ladder reading and final alignment, P2.3; stdout_mask and jacobi, P2.5 and P2.7; the C-aware Sim-T, P2.7; the gate packet, P2.G. Not taken, moved in PHASE-NOTES: the hint wording in compile diagnostics and the Attempt artifact field (P3), and generated asset trees (P5).

Owner queue: no new item blocks a task. P2.G adds one review item. OQ-018 still blocks only P1.12.

## Tasks

### P2.1 Record every model request
- Bible: Result Record, Readability Standards (Trial row), Component Interfaces (Stage contract rules), Decision Log 2026-09-24 (faithful generation entry).
- Accept:
  - Every model call of a trial (summarize_context, describe_source, generate, each correction) is recorded with each message it sent, system messages included, by text-store hash, and with its reply. A faithful mock trial holds 3 + corrections requests with two distinct system prompts among them (test).
  - trial.md shows every request in order; the Parquet mirror gains a requests table.
  - A context reply's invalid-text warning is recorded with its request, no longer on attempt 0 (test).
  - A trial.json written before this change still loads, with the new field read as not recorded (test on a P1 fixture record).
  - Bible Result Record edit with a Decision Log entry.
- Files: `lassi/core/record.py`, `stages.py`, `runner.py`, `trial_md.py`, `parquet.py`, `tests/core/`, `docs/BIBLE.md`.
- Remote: none. Depends: none.

### P2.2 Record the reference run's flags
- Bible: Result Record (reference_run, Attempt.run), Sandbox (output caps, attempt run limits), Oracles (stdout_mask rules).
- Accept:
  - RunInfo gains stdout_truncated, stderr_truncated, and workdir_incomplete (None when not recorded). The baseline fills them for Trial.reference_run and run_loop for Attempt.run; the run-stage warnings of P1.6 stay.
  - The oracle stage never aligns against a reference stdout marked truncated: those alignments stay unset and each affected attempt gets one run-stage warning naming the cut (test).
  - trial.md and the Parquet reference_run and attempt columns show the flags; an older trial.json loads with them None (test).
  - The sandbox.wall_s comment in `projects/base.yaml` names the 30 s floor and the no-reference case.
  - Bible edit (Result Record, Sandbox) with a Decision Log entry.
- Files: `lassi/core/record.py`, `stages.py`, `oracle_stage.py`, `trial_md.py`, `parquet.py`, `projects/base.yaml`, `tests/core/`, `tests/oracles/`, `docs/BIBLE.md`.
- Remote: none. Depends: none (touches P2.1's files; do them one after the other).

### P2.3 Readings the scores rest on: stage ladder and final alignment
- Bible: Reward Function (stage table), Harness Contract (a missing file is a build error), Result Record (Attempt.stage_reached, final), Oracles (stale output rule), Source Papers (quirk table, loop row).
- Accept:
  - With fixes.fence_tag on, a reply whose only FILE-block problem is a missing expected file is S1 with a compile-stage error Diagnostic `missing-file`, fed back to the model; a reply with no usable block, or with any other block error, stays S0. A test covers each case; the bible records both readings with a Decision Log entry.
  - One shared helper names the attempt whose output stands (None when no attempt ran), and the runner sets final.alignment to that attempt's alignment mean. Tests: a clean finish, stale output, upstream-crash, a run error at the cap, and a compile-only trial (None). trial.md and the Parquet final_alignment show it.
- Files: `lassi/core/stages.py`, `lassi/core/runner.py`, `lassi/core/record.py`, `tests/core/`, `docs/BIBLE.md`.
- Remote: none. Depends: none.

### P2.4 Spike: the paper's LASSI metric definitions
- Bible: Source Papers (LASSI results table, fidelity findings), Evaluation Protocol (LASSI reproduction row), Reporting Rules.
- Accept: `plans/spikes/p2-lassi-metrics.md` gives, citing the page, section, or table of arXiv:2407.01638 and the pinned notebook cell where one applies: what counted as correct output; the denominator of each percentage in the paper's results table (correct output, within 10% or faster, first try, Sim-T >= 0.6); how runtime ratios and first try were counted; what the notebook stores per trial. The findings go into the Evaluation Protocol with a Decision Log entry. Whatever the sources leave open is named as open, never guessed.
- Files: `plans/spikes/p2-lassi-metrics.md`, `docs/BIBLE.md`.
- Remote: none. Depends: none.

### P2.5 Spike: stability of the reference output under the proxy
- Bible: Oracles (stdout_mask rules), Harness Contract (CUDA to OpenMP proxy), Sandbox (native runs, attempt run limits).
- Accept:
  - `tests/fixtures/recipes/p2-proxy-mock.yaml`: the mock, CUDA to OpenMP, all ten items, nvcpp-multicore, the native executor, stdout_mask with passfail, n = 3. It loads locally (test).
  - An rx job runs it from a clean commit. `plans/spikes/p2-proxy-stability.md` gives the commands, the job id, and, per item, whether the masked stdout of its three reference runs and three mock candidate runs agree. jacobi's CUDA reference is recorded as not checkable without a GPU (OQ-003; P10).
  - Any item whose masked reference stdout varies is listed in the bible's Oracles section with a Decision Log entry and marked in every metrics table and packet that reports it.
  - `rx pull` into `results/p2-proxy-stability/` with provenance.json and summary.md. The mock candidate is the reference program again, so its passes check the path and are never model results.
- Files: `tests/fixtures/recipes/p2-proxy-mock.yaml`, `plans/spikes/p2-proxy-stability.md`, `results/p2-proxy-stability/`, `docs/BIBLE.md`.
- Remote: `rx doctor` and a `du` check first; `rx job start --big --name p2-proxy` (CPU only; atomicCost's OpenMP reference needs about 15 GB); `rx job wait` between local tasks; `rx pull`. Depends: none. Start it first so P2.1 to P2.4 proceed while it runs.

### P2.6 df-v0 score profile
- Bible: Reward Function (table, formula, scoring decisions), Algorithms (Episodes), Component Interfaces (ScoreProfile), Result Record (Attempt.score, guards, final.score, Diagnostic), Repository Layout (`scoring/`).
- Accept:
  - `assets/scoring/df-v0.yaml` holds every weight the Reward Function states; the code holds none (test: a changed YAML weight changes the score).
  - A registered ScoreProfile `df-v0` gives each attempt its components (stage base, W, alignment term, guard) and R. Tests with hand-computed values: each stage S0 to S5; W deduplicated and capped at 10; warnings of an attempt that did not compile not counted; A only at S5; any guard True gives R = -1, guards None count as not checked; a stale-output last attempt.
  - The trial's components hold attempt 0's R (single turn) and the multi-turn value; final.score is the multi-turn value.
  - The W reading and the final.score choice join the Reward Function, and `assets/scoring/` joins the Repository Layout, with a Decision Log entry.
- Files: `lassi/scoring/`, `assets/scoring/df-v0.yaml`, `tests/scoring/`, `docs/BIBLE.md`.
- Remote: none. Depends: P2.3.

### P2.7 lassi score profile
- Bible: Evaluation Protocol (LASSI reproduction row; Acceptance Criteria, compile-only label), Project Recipes (lassi-repro `metrics` line), Source Papers (quirk table, Sim-T and fence rows), Oracles, Harness Contract (the proxy checks outputs, never runtime).
- Accept:
  - A registered ScoreProfile `lassi` gives each trial the components the lassi-repro metrics line and the Evaluation Protocol row name, as P2.4 defines them: correct (per the planning decision), within_10pct (None with its reason while no timing profiler exists), first_try, sim_t, sim_t_c, sim_l, self_corr, cap_hit, fence_quirk, and the compile-stage components compiled and compiled_first_try. Its scalar is correct. The paper criterion (manual inspection) is never computed; it is None and labeled.
  - Sim-T and Sim-L compare the reference target read in text mode with the last attempt's target file, as the P1.9 replay does (test on a replay scenario's record); each value carries the interpreter version.
  - Tests on hand-built trials: a clean pass; an oracle match from a crashed run (correct 0.0); stale output; compile-only (correct None); a cap hit; a fence-quirk hit.
  - Bible: the profile's definitions (Evaluation Protocol) and the C-aware Sim-T design choices now documented only in `lassi/scoring/similarity.py`, with Decision Log entries.
- Files: `lassi/scoring/`, `assets/scoring/lassi.yaml`, `tests/scoring/`, `docs/BIBLE.md`.
- Remote: none. Depends: P2.3, P2.4.

### P2.8 Run metrics per arm and direction
- Bible: Evaluation Protocol (preamble, LASSI and LASSI-DF rows, Acceptance Criteria), Reporting Rules, Readability Standards (Run row), Repository Layout (`analysis/`), Source Papers (LASSI results table).
- Accept:
  - `lassi/analysis/` gives Wilson 95% intervals (tests against published worked values), pass@1 and pass@3 from n trials per scenario (None with the reason when n < k), the stage-reached distribution, compile, run, and correct rates, the correction distribution, cap hits, fence-quirk hits, and the Sim-T >= 0.6 rate. The population each interval covers is recorded in the Evaluation Protocol with a Decision Log entry.
  - One metrics table per arm and direction, in Markdown and Parquet, shows the paper's values alongside from `assets/scoring/lassi-paper.yaml`, each citing its bible table. A compile-only run is labeled a compile-stage reproduction with no comparison to the paper's correctness values; runtime values are PLACEHOLDER.
- Files: `lassi/analysis/`, `assets/scoring/lassi-paper.yaml`, `tests/analysis/`, `docs/BIBLE.md`.
- Remote: none. Depends: P2.6, P2.7.

### P2.9 lassi score: score a finished run and write the review packet
- Bible: Design Principles 2, 5, and 7; Result Record (Storage); Readability Standards (Trial and Run rows); Agent Rules 1 and 10; AGENTS.md Results.
- Accept:
  - `lassi score <run dir> --profile <name> [--profile <name>] [--score-id ID] [--bench-root P]` loads every trial.json of the run strictly (records from P1 commits included) and writes `<runs root>/scores/<score id>/`: provenance.json (scoring commit, dirty flag, date, interpreter version, sha256 of each profile file, a copy of the source run's provenance.json and recipe hash), per-attempt and per-trial components for each profile (Parquet), metrics.md (P2.8), and review.md.
  - review.md has one section per trial: each attempt's stage, diagnostics by code, severity, and place, run exit status and hang flag, alignment, and every component of every profile, plus the fields the source records lack (for a tree older than P2.2, no reference-run flags, with the reference stdout size in bytes). It is plain ASCII and holds no prompt, context, source, or model text (test).
  - The source run tree is unchanged byte for byte, and scoring twice gives identical scores (tests); a local end-to-end test scores a fixture run tree.
- Files: `lassi/cli.py`, `lassi/scoring/` or `lassi/analysis/`, `tests/scoring/` or `tests/analysis/`.
- Remote: none (the gate runs it on alpha01). Depends: P2.2, P2.8.

### P2.10 Recipes bind score and metrics in lassi run
- Bible: Project Recipes (lassi-repro and lassi-df blocks, Notes), Component Interfaces (ScoreProfile, capability rule), Design Principles 1 and 5, Readability Standards (Run row).
- Accept:
  - The runner no longer refuses `score` and `metrics`. `score` binds a registered ScoreProfile that fills Attempt.score and final.score; each `metrics` name is checked at load against the metrics the registered profiles and analyses provide, and an unknown name is refused before any directory is created (test).
  - run.md gains the P2.8 tables per arm and direction.
  - `projects/lassi-repro/recipe.yaml` carries the bible block's metrics line and drops its P2 comment; the recipe hash change is a Decision Log entry, and the golden resolved recipes follow.
  - A local mock dry run of `tests/fixtures/recipes/p1-dry-run.yaml` writes scores and metrics tables (test).
- Files: `lassi/core/runner.py`, `lassi/core/recipe.py`, `projects/lassi-repro/recipe.yaml`, `tests/core/`, `tests/fixtures/recipes/`, `docs/BIBLE.md`.
- Remote: none. Depends: P2.9.

### P2.G Phase gate
- Bible: Build Roadmap (P2 row, Gate column), Reward Function (scoring decisions), Evaluation Protocol (Reporting Rules); AGENTS.md Phase Gate and Results.
- Accept, from one clean commit on alpha01:
  - `rx doctor` and a `du` check. `rx run -- 'uv run lassi score $LASSI_RUNS_ROOT/runs/demo-rngd-cpu-1 --profile df-v0 --profile lassi --score-id p2-gate'` exits 0 with all 10 trials scored.
  - Load check, not part of the review: `lassi score` over p1-gate-dry-run and demo-rngd-1 exits 0 with every trial scored.
  - `rx pull` writes `results/p2-gate/` with provenance.json, review.md, and metrics.md. summary.md quotes the gate text, cites the scoring run's and the source run's provenance, states the run's limits (planning decisions above) and the P2.5 finding for its items, and lists the review questions: each planning decision above and any weight J wants changed.
  - An owner-queue review item (Kind: review, Blocks: none) points to `results/p2-gate/`. P2 is set GATE-OWNER, and the pull request `P2 Scoring` goes from `p2-scoring` to `main`, its body noting that it stacks on the open P1 pull request 2. Work moves to P4.
- Remote: `rx doctor`, `rx run` (at most two), `rx pull --path` for the packet. Depends: P2.1, P2.2, P2.5, P2.9, P2.10.
