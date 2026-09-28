# Spike: The P2 Review Questions (task P4.15, OQ-024)

Task P4.15 (plans/p4-ttsim.md) carries the eight review questions of results/p2-gate/summary.md ("Review questions for J"), which OQ-024's answer moved to P4 without naming a change. This spike closes each one: settled by evidence, brought back as an owner-queue item with options and a recommendation, or recorded as standing, with the reason. On 2026-09-26 the owner, unable to monitor, directed that every owner-queue item not answered personally takes its recommendation, recorded on the item and marked for review later (plans/LESSONS.md, Owner directions); items OQ-028 to OQ-033 record that direction.

## Method and provenance

- Checked state: commit 1ccf1fb on branch p4-ttsim. A citation `1ccf1fb:<path>:<line>` names a line at that commit; bible citations name a section, since the lines move.
- Local checks ran on the workstation (Windows 10) with `C:/dev/lassi/.venv/Scripts/python.exe`, Python 3.10.21 (below: `$PY`), on 2026-09-26 (UTC about 20:00 to 21:00). They are exploratory: no logged run, never [MEASURED].
- Imports are pinned to the commit under test: `git archive 1ccf1fb | tar -x -C <export>`, the scripts of plans/spikes/p4-p2-review/ copied to `<export>/plans/spikes/p4-p2-review/`, and every command run from `<export>` with `PYTHONPATH=<export>` (python -m pytest imports `lassi` from `<export>/lassi`; the package is installed editable from the working tree, which other tasks were editing). `<after>` is the same export with the files P4.15 changes copied over it.
- [MEASURED] values come from logged runs only: the P2 gate's scoring pass (rx 20260924-174659-desktop-8r113ei-p2-scoring-e027, clean a10fdd0, 2026-09-24, UTC 2026-09-25 00:47, Python 3.10.12, alpha01 host CPU, no accelerator; results/p2-gate/score/provenance.json), and P4.15's clean-commit scoring pass, results/p4-p2-review/ (PLACEHOLDER until it runs).
- demo-rngd-cpu-1's records were read from a workstation copy of its `rx pull` (its provenance.json: run_id demo-rngd-cpu-1, commit 2f052e6, dirty false, host alpha01, Python 3.10.12, executor native, recipe hash 209990b1d6f8; sha256 of that file a2473ff0ac6fe6649a2f11140ba5a32bbd945d7931d750d03a367dcd28947f4e). The run: rx job 20260924-093127-demo-rngd-cpu-1ff6 (results/demo-rngd-cpu-run/summary.md).
- The upstream checkout is third_party/LASSI at the pin 74b4681; LASSI_latest_07-23-2024.ipynb is read at f55bf0c. No upstream text of 40 or more characters is quoted here (OQ-018): notebook code is cited by the first eight characters of its cell id and a line number, and paraphrased.
- The paper's tables are read from the pypdf text of arXiv:2407.01638v2 (PDF sha256 21074bbc12e4460b40a541ec0b028f8de1071a4c2d9fe250007bfa8ea803fd32, as in plans/spikes/p2-lassi-metrics.md; text sha256 d3c970d53f20a4f9711d1d3ac37e4c41d125cb3cb150b573c3133aa303acfb11). Counts read from the paper are paper values, not measurements.
- Baseline of the suites at 1ccf1fb: `PYTHONPATH=<export> $PY -m pytest -q -p no:cacheprovider tests/scoring tests/analysis` printed `1 failed, 423 passed, 79 skipped`. The one failure is environmental: test_provenance_records_the_scoring_pass_its_profiles_and_the_source_run runs `git rev-parse HEAD` in the export, which has no .git.

## Closures

| Question | Closure | Recorded in |
| --- | --- | --- |
| 1 The reviewed run | Standing | Decision Log; plans/PHASE-NOTES.md (two findings) |
| 2 Planning decisions | Standing; one rule added by evidence (the wall limit) | Decision Log; LASSI Paper Metrics |
| 3 In-task readings | OQ-028 (1a, 2a, 3b, 4a) | Reward Function; Decision Log |
| 4 The lassi profile's nulls | The four nulls standing; OQ-029 (1a, 2a); one wording fact | LASSI Score Profile; Run Metrics; Decision Log |
| 5 Run metrics | OQ-030 (1b, 2c, 3b, 4a) | Run Metrics; Decision Log |
| 6 Sim-T | sim_t_tiktoken added; OQ-031 (part 1 (c), part 2 (a)) | LASSI Paper Metrics [OPEN]; LASSI Score Profile; Run Metrics; Decision Log |
| 7 OQ-021 | Applied as answered (option (c)); OQ-032 (a) for the reproduction test; the copied atomicCost row standing | LASSI Paper Metrics; Run Metrics; Acceptance Criteria; Decision Log |
| 8 df-v0 weights | OQ-033 (c) | Reward Function; plans/PHASE-NOTES.md; Decision Log |

## Question 1: is demo-rngd-cpu-1 the right run to review?

Closure: standing. demo-rngd-cpu-1 stays the run the P2 score components were reviewed on. The question reopens when a faithful run that executes programs is scored (P10, or after an owner answer that widens OQ-003's scope); when a scored experimental-arm run (P3) shows a component behaving in a way that run and the tests do not cover; when a component comes to read a record field that run lacks (its records predate P2.1, P2.2, and P4.5: no model requests, no run flags, no device); or when the owner asks. A changed reading of fields it holds needs only a rescore of the same records and does not reopen it.

Reason and evidence:

- OQ-024 offered (c) "Hold for a faithful run" and the owner chose (b) (1ccf1fb:plans/OWNER-QUEUE.md:270, :272). The Applied line and the bible row say P4.15 settles, brings back, or records why a reading stands and decides nothing on the owner's behalf (1ccf1fb:plans/OWNER-QUEUE.md:273; Decision Log 2026-09-24, OQ-024 row). So (b) only declined to keep P2 open; it did not answer this question, and the closure rests on the facts below.
- The reviewed run was a P2 planning decision (1ccf1fb:plans/p2-scoring.md:20) that named no task, and the bible never named it: `git show 1ccf1fb:docs/BIBLE.md | grep -c "rngd-cpu"` printed `0`. A Decision Log row now records it.
- The run's limits: a demo model; not faithful (a correction cap of 5 and the -mp=multicore proxy); CUDA to OpenMP only; n = 1; records older than P2.1 and P2.2 (results/p2-gate/summary.md, Limits of the run).
- Under OQ-003 (Decision Log 2026-09-23) reproduction work uses the compile-only tier plus the proxy until an NVIDIA host exists; P10 is BLOCKED (1ccf1fb:plans/STATUS.md:21) and the gpu (NVIDIA) executor is Blocked (Execution Backends). A faithful recipe may not bind the proxy (Project Recipes; 1ccf1fb:lassi/core/recipe.py:82 and :879, _check_faithful_toolchains). So a faithful run before P10 that stays in scope is compile-only.
- `cd <export> && PYTHONPATH=<export> $PY plans/spikes/p4-p2-review/q1_faithful_load.py <export>` printed:

  ```
  lassi imported from lassi/__init__.py
  A-faithful-cpu-proxy REFUSED <tree>\projects\lassi-demo\A-faithful-cpu-proxy.yaml: toolchain.omp binds Toolchain 'nvcpp-multicore', which declares openmp_multicore (a proxy build, not upstream's); a recipe with faithful: true may not bind it, so bind upstream's toolchain or set faithful: false
  B-faithful-gcc-native LOADED faithful: True executor: {'kind': 'native'} toolchain: {'cuda': 'nvcc-sm80', 'omp': 'gcc-native'} max_corrections: uncapped chain: ('base', 'rngd-compile', 'rngd-cpu', 'B-faithful-gcc-native')
  C-faithful-nvcpp-native LOADED faithful: True executor: {'kind': 'native'} toolchain: {'cuda': 'nvcc-sm80', 'omp': 'nvcpp-cc80'} max_corrections: uncapped chain: ('base', 'rngd-compile', 'rngd-cpu', 'C-faithful-nvcpp-native')
  D-p1-dry-run LOADED faithful: True executor: {'kind': 'none'} toolchain: {'cuda': 'nvcc-sm80', 'omp': 'nvcpp-cc80'} max_corrections: uncapped chain: ('base', 'lassi-repro', 'p1-dry-run')
  E-lassi-repro REFUSED <tree>\projects\lassi-repro\recipe.yaml: Stage 'summarize_context' (stages[1]) requires LLMBackend capability 'chat', but the recipe binds no LLMBackend
  ```

  The loader refuses only the proxy capability: faithful children that bind gcc-native (a host `g++ -fopenmp` build upstream never used), or nvcpp-cc80 with the native executor, load, and `git show 1ccf1fb:lassi/core/runner.py | grep -n faithful` shows no toolchain or executor check tied to faithful (lines 14, 76, 502, 511, 997, 999, 1174, 1584). Both cases fall outside OQ-003's scope, and whether an -mp=gpu binary runs on a host without a GPU is not measured (plans/spikes/p0-toolchains-verify.md: no built binary was executed). This gap goes to plans/PHASE-NOTES.md; it is not a reason to wait.
- A faithful compile-only run already exists and was scored: p1-gate-dry-run (recipe D above). Its score provenance, `git show 1ccf1fb:results/p2-gate/load-check/p1-gate-dry-run.score-provenance.json`, records score commit a10fdd0 (dirty false) and a source run with recipe_chain [base, lassi-repro, p1-dry-run], executor none, device "none (compile only)", commit fe01ab1. It reaches only the stage and warning terms (1ccf1fb:plans/p2-scoring.md:20); by the bible's definitions its correct and correctness rows are null under the compile-stage label (LASSI Score Profile, Run Metrics). results/p2-gate/load-check/ holds only provenance files, so that null is the definition's, not a committed output.
- The branches demo-rngd-cpu-1 never reached are tested synthetically: correct 1, first_try 1, fence_quirk above 0, a null correct, the correct-trial population, and pass@3 (review.md tallies below; results/p2-gate/score/metrics.md shows "not computed: no trial in the population" and pass@3 not computed at n = 1). `PYTHONPATH=<export> $PY -m pytest -q -p no:cacheprovider tests/scoring/test_lassi_profile.py tests/analysis/test_run_metrics.py tests/scoring/test_df_v0.py` printed `253 passed`.
- Tallies of the reviewed packet, `git show 1ccf1fb:results/p2-gate/score/review.md | grep '^| lassi | <component> |' | sort | uniq -c`: correct 0.0 x10; first_try 0.0 x10; fence_quirk 0.0 x10; compiled 0.0 x5 and 1.0 x5; cap_hit 1.0 x7 and 0.0 x3.

What changed: a Decision Log row records the run and its reopen conditions. plans/PHASE-NOTES.md (All Phases) records two findings for later phases: the faithful check's gap, and that the metric tables do not show a run's faithful flag or executor (a label changes no value; hiding paper values for a run that is not faithful would change what the tables show, so it goes to the owner queue first). results/p4-p2-review/summary.md restates the source run's limits and names alpha01's host CPU as its programs' device, since metrics.md prints "Device: not recorded" for records older than P4.5.

## Question 2: the P2 planning decisions

Closure: standing. The six decisions stand as recorded; each has a Decision Log row with its reason, is implemented and tested as recorded, and rests on asset files unchanged since the gate and on functions unchanged but for OracleStage.__call__, which P4.4 extended with binary_io (8c82b82), and BASELINE_ENDS, which gained baseline-disagree. One reading behind decision 3 departs from the notebook (the wall limit that makes a run a hang), and evidence settles that it would have changed none of the paper's correct trials; it is now a rule in LASSI Paper Metrics.

The decisions and where each is recorded and pinned:

1. Scoring writes a separate score tree: Result Record (Scores); Decision Log 2026-09-24, P2.9 row; tests/scoring/test_score_run.py (test_the_source_run_tree_is_unchanged_byte_for_byte, test_scoring_twice_gives_identical_component_tables_and_review, and the refusal tests). `PYTHONPATH=<export> $PY -m pytest -q -p no:cacheprovider tests/scoring/test_score_run.py -k "source_run_tree_is_unchanged or scoring_twice or refuse"` printed `20 passed, 27 deselected`.
2. The output that stands is the last attempt that ran, stale output included: 1ccf1fb:lassi/core/record.py:668 (standing_attempt); Result Record (final.alignment); Decision Log P2.3 row. It follows the pinned notebook: in cell fdb5b82f only an attempt that passes the execution gate and runs replaces the stored program output (lines 127 to 136), a nonzero exit feeds a correction, and the metadata file stores that output (line 178), so it is stale when a later attempt did not run.
3. correct needs a clean run aligned at 1: LASSI Score Profile (correct row); Decision Log P2.7 row; 1ccf1fb:lassi/scoring/lassi_profile.py:165 (correct_value). The paper's criterion names exit status 0 and a person's judgment of stdout (LASSI Paper Metrics); "aligned at 1" is the stdout_mask proxy for that judgment, and "no hang" comes from the Sandbox wall limit, not from the paper or the notebook (see the rule below). P4.6 extends "clean" to no undefined behavior and reads a sim-gap trial's correct as null (its own Decision Log rows).
4. df-v0's W counts compile- and jit-stage warnings at S4 or S5: Reward Function (df-v0's readings of an attempt); Decision Log P2.6 row; 1ccf1fb:lassi/scoring/df_v0.py:61 (WARNING_STAGES). The restriction is a P2 reading of "warnings count only after a successful compile".
5. final.score is the multi-turn value: Reward Function (df-v0's trial score); Decision Log P2.6 row, whose rationale names the departure from the single_turn default. Both components are kept. `git grep -n "final\.score\|final_score" 1ccf1fb -- lassi` finds readers only in the Parquet mirror (1ccf1fb:lassi/core/parquet.py:311) and trial.md (1ccf1fb:lassi/core/trial_md.py:184); lassi.analysis reads no final.score.
6. Weights and paper values live in assets/scoring/: Reward Function ("all weights live in YAML"); Repository Layout; Decision Log P2.6 row (weights) and P2.8 row (paper values).

Unchanged since the gate's commit: `$PY plans/spikes/p4-p2-review/same_since_gate.py a10fdd0 1ccf1fb` (from the repository root; it reads git objects only) printed SAME for standing_attempt, RunLoopStage.__call__, _past_gate, _parsed_attempt, CompileLoopStage._buildable, CompileLoopStage._needs_correction, GenerateStage.__call__, OracleStage.align_runs, correct_value, outcome_components, count_components, warning_count, guard_state, DfV0Profile.score_attempt, and DfV0Profile._trial_score; CHANGED only for OracleStage.__call__ (P4.4 added binary_io, commit 8c82b82). The asset digests at 1ccf1fb, df-v0.yaml 1bd8b1ea88ea4da9b2c655e1c8287fd4f25ee4d7262aa32b200f178748a14ca4, lassi.yaml cebc187cd29f7b9651636dc699a07e3f6be16f8d807b76b984be2c915566462a, and lassi-paper.yaml 414fbbc67d60d559f91b1c9844075109723d5bff998ddf9e1931ed3545a03fa8, equal those in results/p2-gate/score/provenance.json. The only scoring change since a10fdd0 adds baseline-disagree to BASELINE_ENDS (1ccf1fb:lassi/scoring/lassi_profile.py:93).

The wall limit, settled by evidence:

- The notebook waits for every program with no time limit: cell 976745b6 line 53 waits on the process with no timeout argument.
- run_loop stops an attempt run at ten times the reference run's wall time, never under 30 s (Sandbox; 1ccf1fb:lassi/core/stages.py:257, RUN_WALL_FLOOR_S); such a run is a failed run and scores correct 0.
- `$PY plans/spikes/p4-p2-review/paper_wall.py <v2 text>` compares each of the paper's correct trials' Runtime with max(10 x its Table IV reference, 30 s) and printed:

  ```
  OMP -> CUDA: correct trials 32; over the limit 0 []; lowest Ratios: WizardCoder atomicCost Ratio 0.3777 Runtime 116.2879 (reference 43.919, limit 439.19), DeepSeek atomicCost Ratio 0.4715 Runtime 93.1467 (reference 43.919, limit 439.19), Codestral randomAccess Ratio 0.564 Runtime 8.8905 (reference 5.0139, limit 50.139)
  CUDA -> OMP: correct trials 34; over the limit 0 []; lowest Ratios: Codestral bsearch Ratio 0.0498 Runtime 0.2811 (reference 0.014, limit 30), DeepSeek matrix-rotate Ratio 0.1072 Runtime 11.0047 (reference 1.18, limit 30), GPT-4 atomicCost Ratio 0.2055 Runtime 219.5494 (reference 45.1242, limit 451.242)
  ```

  On the paper's own numbers the limit would have cut none of the 66 correct trials; the one trial more than ten times slower than its reference (Codestral's CUDA -> OMP bsearch, which the verifiers named) ran 0.2811 s against the 30 s floor. The harness times whole processes while the paper's timing method is unstated, so P10's plan checks the limit again against the reference runtimes it measures (plans/PHASE-NOTES.md).

A side observation for P4.11 (plans/PHASE-NOTES.md, P4), from demo-rngd-cpu-1's logged records (the rx pull copy described under Method; [MEASURED] wall times of the CPU proxy run, not performance): `$PY -c "import json,sys; t=json.load(open(sys.argv[1])); r=t['reference_run']; print('reference', r['wall_s'], r['exit_code'], r['hang']); [print('attempt', a['index'], a['stage_reached'], (a['run'] or {}).get('wall_s'), (a['run'] or {}).get('exit_code'), (a['run'] or {}).get('hang')) for a in t['attempts']]; print(t['final']['end_reason']['code'])" <pull>/rngd-cpu/furiosa-ai--Llama-3.1-8B-Instruct/lassi-hecbench-10/cuda-omp/randomAccess/run01/trial.json` printed:

```
reference 4.947733998007607 0 False
attempt 0 S1 None None None
attempt 1 S1 None None None
attempt 2 S4 50.07811091799522 137 False
attempt 3 S4 49.97597110600327 137 False
attempt 4 S4 49.96544308800367 137 False
attempt 5 S4 49.97698279999895 137 False
correction-cap
```

Ten times the reference's 4.95 s is 49.48 s, so each of the four runs ended just past the wall limit with exit status 137 and hang false. The sandbox reports a hang when a run ends with status 124, 137, or 143 at its wall limit (lassi/executors/sandbox.py, classify; the margin is in its module docstring), so whether these runs were stopped by the limit, and why the flag is false, is left to P4.11; nothing in this spike depends on it.

What changed: LASSI Paper Metrics, Rules, gains the wall-limit rule [DESIGN]; a Decision Log row records questions 1 and 2 as standing. No reading, weight, or paper value changes.

## Question 3: df-v0's in-task readings

Closure: owner-queue item OQ-028. The four readings were P2.3 and P2.6 choices put to the owner at the gate, and OQ-024's answer named no change, so none has an owner answer; two of them (a violation on an earlier attempt, a reply of only unlisted files) are not stated in the bible at all. Under the owner's 2026-09-26 direction the recommendation 1a, 2a, 3b, 4a is applied and marked for review.

Evidence:

- What df-v0 does at 1ccf1fb: an S5 attempt with a null alignment mean gets an alignment term of 0 and alignment_missing 1; the guard component is 1 when any guard is true, 0 when all are false, null otherwise, and only 1 sets R to guard_violation; the multi-turn value is the last attempt's R minus correction_penalty x corrections (1ccf1fb:lassi/scoring/df_v0.py:144, :218; Reward Function, df-v0's readings of an attempt and trial score).
- The tests that pin each reading pass at 1ccf1fb: `PYTHONPATH=<export> $PY -m pytest -q -p no:cacheprovider tests/scoring/test_df_v0.py tests/core/test_stage_reading.py -k "never_aligned or alignment_missing or none_counts_as_not_checked or violation_on_the_last_attempt or shipped_weights_score_the_probe or only_files_the_target_does_not_list"` printed `10 passed, 71 deselected`.
- The earlier-attempt case: the probe trial (1ccf1fb:tests/scoring/test_df_v0.py:324-333) sets host_compute on attempt 5 of 7 with a clean last attempt, and test_the_shipped_weights_score_the_probe_by_the_bible (:362-369) asserts that attempt's R is -1.0 and the multi-turn value 0.24 = 0.54 - 0.05 x 6. Under 3b it becomes -1.0 - 0.05 x 6 = -1.30 (arithmetic from assets/scoring/df-v0.yaml).
- Three paths leave an S5 attempt unaligned at 1ccf1fb: a target reference stdout cut at the 1 MiB cap (1ccf1fb:lassi/toolchains/_base.py:59, OUTPUT_CAP_BYTES = 1 << 20); an incomplete binary_io reference workdir (1ccf1fb:lassi/core/oracle_stage.py, module docstring); a recipe that lists run_loop without the oracle stage, since _check_oracle returns at once when no Oracle is bound (1ccf1fb:lassi/core/runner.py:1121-1136) and no stage-order rule asks for an aligning stage.
- The unlisted-only reply: the FILE-block parser keeps files the target does not list without a diagnostic and gives one missing-file error per expected path with no block (1ccf1fb:lassi/core/files.py:190-195, :233-237), and the S1 rule reads "at least one file and every FILE-block error is missing-file" (Reward Function). S1 against S0 changes R by 0.2 and nothing else: the same correction follows either way.
- The reviewed run, [MEASURED] as above: `git show 1ccf1fb:results/p2-gate/score/review.md | grep -c '^### Attempt'` printed `47`; `grep -c '| df-v0 | guard | PLACEHOLDER |'` printed `47`; `grep -c '| df-v0 | alignment_missing | 0.0 |'` printed `47`; `grep -c 'missing-file'` printed `0`. No component sets a guard at 1ccf1fb (`git grep -n "Guards(" 1ccf1fb -- lassi` prints nothing: no package code builds a Guards value), and P4.12 is the first task that sets one.

What changed (applied under the direction): the Reward Function names the unlisted-only reply as S1 (part 4) and states that a guard violation on any attempt makes the multi-turn value, the scalar, and final.score -1 - 0.05 x final.corrections while single_turn stays attempt 0's R (part 3), with P4.6's Simulator readings bullet amended to match; a Decision Log row. lassi/scoring/df_v0.py (_trial_score) puts guard_violation in place of R_final when any attempt's guard component is 1.0, a sim-gap trial's null R_final included; tests/scoring/test_df_v0.py's probe now expects -1.30, and new tests cover an earlier violation with a clean last attempt (tests/scoring/test_df_v0.py) and an earlier violation inside a sim-gap trial (tests/scoring/test_sim_gap_scoring.py). Parts 1 and 2 change nothing.

## Question 4: the lassi profile's nulls

Closure: the four nulls stand; two readings beside them go to OQ-029 (applied: 1a and 2a); one wording fact is corrected.

The four nulls stand as the bible's LASSI Score Profile and the P2.7 row record them, for the P2.7 reason: a null keeps an unmeasured value apart from a measured 0, so no rate counts it as a failure. They are pinned at 1ccf1fb by these tests, all of which pass (`PYTHONPATH=<export> $PY -m pytest -q -p no:cacheprovider tests/scoring/test_lassi_profile.py tests/core/test_binary_io_baseline.py tests/analysis -k "baseline or no_attempt or aligned or compile_only or disagree or compile_stage"` printed `35 passed, 262 deselected`; the expression also selects the whole of test_binary_io_baseline.py by its module name, of which only test_baseline_disagree_joins_the_lassi_profiles_baseline_ends and test_the_lassi_profile_leaves_correct_and_compiled_null_at_baseline_disagree bear on the nulls): test_a_trial_that_ended_at_the_baseline_is_not_scored, test_a_trial_with_no_attempt_past_a_clean_baseline_is_not_correct, test_a_clean_run_the_oracle_never_aligned_is_not_scored, test_compile_only_correct_is_labeled_a_compile_stage_reproduction, and the compile-only table tests of tests/analysis/test_run_metrics.py. The reviewed run reached none of the null branches: no trial lacked an attempt (results/p2-gate/score/metrics.md, stage none 0) and all ten scored correct 0.0. The compile-only demo-rngd-1 of the load check is the logged run where the compile_only null and the label fire (results/p2-gate/load-check/demo-rngd-1.score-provenance.json: executor none, device "none (compile only)"); its values are not committed.

The two readings beside them (OQ-029):

- Part 1, the label. At 1ccf1fb a table is a compile-stage reproduction only when every trial's correct is null with the compile-stage note (1ccf1fb:lassi/analysis/metrics.py:256). A compile-only run with one baseline-ended trial therefore loses the label and prints paper correctness values, against the Acceptance Criteria. `q5_populations.py` group (d), the compile-only group plus one trial that ended at the baseline: before, `(d) compile_only False; correct_rate paper values: 34/40 = 0.850 (Wilson 95% 0.709 to 0.929)`; after, `(d) compile_only True; correct_rate paper values: None`. No trial of any run scored so far ended at the baseline (results/p1-gate/summary.md: all 20 baselines compiled; results/demo-rngd-run/summary.md: every trial reached a compiling translation or the cap; results/p2-gate/score/metrics.md), so no value changes.
- Part 2, two branches no committed recipe reaches. correct is 0 for a trial with no attempt that did not end at the baseline while the reference ran (1ccf1fb:lassi/scoring/lassi_profile.py:165-178); generate always appends attempt 0 when it returns (1ccf1fb:lassi/core/stages.py, GenerateStage.__call__), and _check_stages allows a stage list with no generating stage, so only such a recipe makes this trial. And the profile reads "the target reference was not run" as the compile-only tier (_ran, :152), so a running tier's trial with no run is noted a compile-stage reproduction when its recipe lists no baseline stage. Every committed recipe lists generate and baseline (projects/lassi-repro/recipe.yaml, projects/lassi-demo/rngd-compile.yaml, rngd-cpu.yaml). Both readings stand, with their reach recorded in the bible.

The wording fact: the correct row names "no reference stdout" as a way a clean run goes unaligned, but the oracle stage raises when a trial holds run stdout and no reference stdout (1ccf1fb:lassi/core/oracle_stage.py, module docstring, "a trial holding run stdout but no reference stdout raises"). The reachable case is a recipe that lists no oracle stage; the row and lassi/scoring/lassi_profile.py's docstring now say so.

P4.6's simulator readings (commit 0683b2e, after 1ccf1fb) settle what the review's verifiers raised about sim-gap and undefined behavior: the lassi profile's clean run needs no undefined behavior, and a trial with attempts that ended at sim-gap has correct null with a gap note. The lassi profile scores only items whose target has one file (1ccf1fb:lassi/scoring/lassi_profile.py:290), which Tier A's host-plus-kernel targets are not.

What changed: lassi/analysis/metrics.py (is_compile_only) and its test; the LASSI Score Profile's correct row and the paragraph after the table; the Run Metrics label rule; a Decision Log row.

## Question 5: run metrics

Closure: owner-queue item OQ-030. Parts 1 to 3 are [DESIGN] readings and part 4 a placement the bible does not name. Applied under the direction: 1b, 2c, 3b, 4a.

Evidence, from `q5_populations.py` over the SYNTHETIC trials of tests/analysis/test_run_metrics.py (before: `cd <export> && PYTHONPATH=<export> $PY plans/spikes/p4-p2-review/q5_populations.py <export>`; after: the same in `<after>`):

```
before (1ccf1fb)
(a) compile_rate: 7/9 value 0.7778 | compiled = 1, over all trials of the arm and direction; 1 excluded (compiled is None)
(a) run_rate: 6/10 value 0.6000 | last attempt at S5, over all trials of the arm and direction
(a) correct_rate: 3/9 value 0.3333 | correct = 1 (automated oracle), over all trials of the arm and direction; 1 excluded (correct is None)
(a) cap_hit_rate: 2/10 value 0.2000 | cap_hit = 1, over all trials of the arm and direction
(a) fence_quirk_rate: 2/10 value 0.2000 | fence_quirk > 0, over all trials of the arm and direction; 5 fence-quirk hits in total
(b) pass@1: - value None | not computed: 1 scenario(s) have fewer than k = 1 scored trials (smallest n = 0); 2 excluded (correct is None)
(b) pass@3: - value None | not computed: 1 scenario(s) have fewer than k = 3 scored trials (smallest n = 0); 2 excluded (correct is None)
(c) sim_t 0.5996 0.5951 0.595 0.6 0.59 0.5949: sim_t_ge_0.6_rate: 1/6 value 0.1667 | sim_t >= 0.6, over the correct trials, ...
(e) omp-cuda b0_criterion: no such field
after (1ccf1fb with the P4.15 files)
(a) compile_rate: 7/9 ... 1 excluded (compiled is None: the trial ended at the baseline)
(a) run_rate: 6/9 ... 1 excluded (compiled is None: the trial ended at the baseline)
(a) correct_rate: 3/9 ... 1 excluded (correct is None)
(a) cap_hit_rate: 2/9 ... ; fence_quirk_rate: 2/9 ...
(b) pass@1: 0.6/2 value 0.3000 | ... 2 excluded (correct is None); 1 scenario(s) with no scored trial left out: lassi-hecbench-10/entropy
(b) pass@3: 1.5/2 value 0.7500 | ... (the same note)
(c) ... sim_t_ge_0.6_rate: 3/6 value 0.5000 | sim_t (.2f) >= 0.6, ...
(e) omp-cuda b0_criterion: WizardCoder 9/10 = 0.900 (Wilson 95% 0.596 to 0.982)
(e) cuda-omp b0_criterion: WizardCoder 10/10 = 1.000 (Wilson 95% 0.722 to 1.000)
```

(Lines shortened with "..."; the full output is reproducible from the command.)

- Part 1: at 1ccf1fb run_rate reads the last attempt's stage over every trial (1ccf1fb:lassi/analysis/metrics.py:180-191), while compile_rate and correct_rate drop a null component; a baseline-ended trial has compiled and correct null (lassi profile), so the five rates had two denominators (group (a)).
- Part 2: pass@k is null for the arm when any scenario has fewer than k scored trials, zero included (1ccf1fb:lassi/analysis/metrics.py:213-229). A scenario can have no scored trial while its baseline passed, for example through clean runs the oracle never aligned against a cut reference stdout, which does not end the trial (group (b)).
- Part 3: both committed notebooks format the token similarities to two decimals and compare no value with a threshold. `$PY plans/spikes/p4-p2-review/nb_scan.py third_party/LASSI 74b4681:LASSI_pipeline_v0.ipynb f55bf0c:LASSI_latest_07-23-2024.ipynb` printed, for each, `kernel python 3.10.12; code cells 13` (15 for the latest) `; two-decimal formats 6 (fdb5b82f:152, fdb5b82f:155, fdb5b82f:158, fdb5b82f:167, fdb5b82f:169, fdb5b82f:171); threshold-like lines 0 ()`. Lines 152 and 167 are the tokenize similarity, 155 and 169 the tiktoken one, 158 and 171 Sim-L. The reference recount counts CUDA -> OMP Sim-T >= 0.6 as 15/34 with GPT-4's layout printed as 0.60 (plans/spikes/p2-lassi-metrics.md). The formatted comparison is the reading closest to the stored data, not a proven one (group (c)).
- Part 4: b0_interval is defined and tested but shown nowhere at 1ccf1fb (`git grep -n b0_interval 1ccf1fb -- lassi tests assets tools` finds only lassi/analysis/__init__.py, lassi/analysis/paper.py:177, and its tests). Wilson 95% of the paper counts, computed with lassi.analysis.stats.wilson_interval: 9/10 = 0.900 (0.596 to 0.982), 10/10 = 1.000 (0.722 to 0.9999999999999999). These are paper values (group (e)).
- No value in the reviewed P2 table changes under any option: correct was 0/10 with no exclusion, no trial lacked an attempt, pass@1 covered 10 scenarios, and pass@3 was not computed at n = 1 (results/p2-gate/score/metrics.md).

What changed: lassi/analysis/metrics.py (the five trial rates, pass@k, the Sim-T comparison, b0_criterion on each MetricTable) and lassi/analysis/tables.py (the B0 criterion line, the b0_criterion Parquet column, the legend), with tests in tests/analysis/test_run_metrics.py; the Run Metrics block; a Decision Log row.

## Question 6: Sim-T

Closure: sim_t_tiktoken is added under its own name (OQ-022's pending item), and the review OQ-022's answer asked for goes back as OQ-031 with the new evidence. Applied under the direction: part 1 (c), both similarities compared with the recount and neither marked reproduced; part 2 (a), null with its own note where the notebook raises. Asking the paper's authors (part 1 (d)) is left to the owner.

Evidence:

- The notebook's tokenize similarity appends whole TokenInfo tuples and compares the two lists with difflib's SequenceMatcher, reference first, autojunk on (cell a949bc43, lines 34 to 52 and 63 to 67). TokenInfo is (type, string, start, end, line), so a token matches only at the same row and column of an identical physical line.
- The notebook's extraction takes the first fenced block (cell 00ddfed0, lines 18 to 25) and cuts a leading c++ or cpp, then one more leading c (cell fdb5b82f, lines 110 to 113); it scores that block against the reference as read (line 151 for tokenize, 154 for tiktoken). A block after a cpp, c++, or c tag or a bare fence starts with a line feed, and after a cuda tag with "uda". lassi.core.fragments.first_fence is the faithful port (1ccf1fb:lassi/core/fragments.py:207), and demo-rngd-cpu-1's recipe chain has fence_tag off (projects/lassi-demo/rngd-compile.yaml).
- `cd <export> && PYTHONPATH=<export> $PY plans/spikes/p4-p2-review/q6_similarity.py --upstream C:/dev/lassi/third_party/LASSI --run <pulled demo-rngd-cpu-1>` (exploratory, Python 3.10.21) printed:

  ```
  python 3.10.21; 20 references
  perfect copy, fence tag 'cpp', leading line feed 20/20 [20 pairs]: sim_t 0.0004..0.0240; sim_l 0.9901..0.9965; sim_t_c 1.0000..1.0000; (type,string) 0.9994..0.9998
  perfect copy, fence tag 'c++', leading line feed 20/20 [20 pairs]: sim_t 0.0004..0.0240; sim_l 0.9901..0.9965; sim_t_c 1.0000..1.0000; (type,string) 0.9994..0.9998
  perfect copy, fence tag 'c', leading line feed 20/20 [20 pairs]: sim_t 0.0004..0.0240; sim_l 0.9901..0.9965; sim_t_c 1.0000..1.0000; (type,string) 0.9994..0.9998
  perfect copy, fence tag 'cuda', leading line feed 0/20 [20 pairs]: sim_t 0.0004..0.0239; sim_l 0.9901..0.9965; sim_t_c 0.9993..0.9997; (type,string) 0.9987..0.9996
  perfect copy, fence tag '', leading line feed 20/20 [20 pairs]: sim_t 0.0004..0.0240; sim_l 0.9901..0.9965; sim_t_c 1.0000..1.0000; (type,string) 0.9994..0.9998
  blank line added at the top [20 pairs]: sim_t 0.0004..0.0240; sim_l 0.9901..0.9965; sim_t_c 1.0000..1.0000; (type,string) 0.9994..0.9998
  blank line added in the middle [20 pairs]: sim_t 0.4236..0.5561; sim_l 0.9901..0.9965; sim_t_c 1.0000..1.0000; (type,string) 0.9994..0.9998
  other-language original, unfenced [20 pairs]: sim_t 0.0117..0.2686; sim_l 0.5830..0.8505; sim_t_c 0.4368..0.8484; (type,string) 0.3671..0.8598
  other-language original, cpp fence [20 pairs]: sim_t 0.0014..0.0477; sim_l 0.5805..0.8598; sim_t_c 0.4368..0.8484; (type,string) 0.3670..0.8595
  run demo-rngd-cpu-1, last target file, leading line feed 10/10 [10 pairs]: sim_t 0.0006..0.0069; sim_l 0.2747..0.8365; sim_t_c 0.4244..0.8546; (type,string) 0.2902..0.8324
  run demo-rngd-cpu-1, leading line feed removed [10 pairs]: sim_t 0.0006..0.0333; sim_l 0.2711..0.8447; sim_t_c 0.4244..0.8546; (type,string) 0.2903..0.8326
  ```

  A candidate identical to its reference, extracted the notebook's way, scores at most 0.0240 under the faithful sim_t, the same as one blank line added at the top; one blank line in the middle gives 0.4236 to 0.5561. The demo run's values below 0.01 match the gate's [MEASURED] range (results/p2-gate/summary.md: 0.0006 to 0.0070 rounded outward), and every last target file there starts with a line feed; removing it lifts sim_t only to 0.0333, so the model's layout also differs. These are exploratory values; the clean-commit pass on alpha01 (Python 3.10.12, the notebooks' kernel version) records them with sim_t_tiktoken in results/p4-p2-review/summary.md (PLACEHOLDER until it runs). If those values contradict these (for example a fenced perfect copy above 0.05 under sim_t), this closure and the bible text are amended.
- lassi.scoring.similarity.sim_t equals the notebook's token_similarity with the tokenize method: tests/scoring/test_similarity.py checks one ordered pair per app in the fast suite and every ordered pair in a slow test.
- The paper's column, `$PY plans/spikes/p4-p2-review/paper_simt.py <v2 text>`:

  ```
  OMP -> CUDA: correct trials 32; Sim-T 0.29..0.70; Sim-T >= 0.6 8; Sim-T > Sim-L 24; r(Sim-T, Sim-L) 0.400
  CUDA -> OMP: correct trials 34; Sim-T 0.14..0.87; Sim-T >= 0.6 15; Sim-T > Sim-L 20; r(Sim-T, Sim-L) 0.796
  pooled: correct trials 66; Sim-T 0.14..0.87; Sim-T >= 0.6 23; Sim-T > Sim-L 44; r(Sim-T, Sim-L) 0.681
  ```

  The counts at 0.6 match the reference recount (8/32, 15/34). The paper names no tokenizer and describes Sim-T as a Ratcliff-Obershelp comparison that finds matching subsequences (Sec. V-A).
- The notebook's tiktoken path gets the cl100k_base encoding, encodes each text with tiktoken's defaults (a special-token string raises), decodes and asserts the round trip, and compares the id lists as above (cell a949bc43, lines 21 to 32 and 63 to 67).
- tiktoken 0.14.0: `uv add "tiktoken==0.14.0"` pinned it in pyproject.toml and uv.lock (with its dependencies regex, requests, certifi, charset-normalizer, idna, urllib3). Its cache rule (tiktoken/load.py, read_file_cached): the cached file's name is the sha1 hex digest of its source address, under TIKTOKEN_CACHE_DIR; a file whose sha256 differs from the expected one is deleted and fetched again, which lassi's loader never lets happen (it replaces that reader while it loads). The encoding's source address and the plugin module that holds it are never written in this repository.

What changed: lassi/scoring/similarity.py gains sim_t_tiktoken, cl100k_base() (offline, TIKTOKEN_CACHE_DIR), cached_reader, and tiktoken_version; lassi/scoring/__init__.py exports it; pyproject.toml and uv.lock pin tiktoken 0.14.0; tests/scoring/test_similarity_tiktoken.py (SYNTHETIC encodings, no network) and tests/scoring/test_similarity_tiktoken_remote.py (the real file on alpha01, equality with the notebook on 400 ordered pairs and on fenced perfect copies; its result is pending the clean-commit pass); tests/scoring/conftest.py gains the upstream_with_tiktoken fixture; the lassi profile gains the component (lassi/scoring/lassi_profile.py, after sim_t_c in assets/scoring/lassi.yaml, with the null note tiktoken_null) and reads the encoding when it is built, so a missing cache refuses `lassi score` and a `lassi run` that names its metrics before anything is scored; lassi/analysis/metrics.py adds sim_t_tiktoken_ge_0.6_rate beside the paper's one Sim-T recount; tests/conftest.py replaces the loader with a SYNTHETIC character encoding for every test not marked real_tiktoken; the LASSI Paper Metrics [OPEN] bullet, the LASSI Score Profile row, the Run Metrics rows, and two Decision Log rows; plans/PHASE-NOTES.md's OQ-022 note, which also says every such command on alpha01 needs TIKTOKEN_CACHE_DIR set. Remote: caching cl100k_base under the scratch root and the clean-commit scoring pass.

## Question 7: OQ-021

Closure: settled as answered, plus one owner-queue item. OQ-021 is CLOSED with option (c), the counts from Tables VI and VII (1ccf1fb:plans/OWNER-QUEUE.md:238-240; Decision Log 2026-09-24, OQ-021 row, which names P4.15 to apply it). The comparison that marks a headline metric reproduced is a separate choice nobody has made: the Acceptance Criteria state a test only for B0, so it goes to the owner as OQ-032; applied under the direction: (a), the recount's Wilson interval, with the gap always reported. The recounts' use of GPT-4's copied OMP -> CUDA atomicCost row stands.

Evidence:

- The owner's only written line is "Let's go with the percentages used not in the abstract" (1ccf1fb:plans/OWNER-QUEUE.md:238), which the agent settled as option (c) in the attended session and recorded on the next line (:239); 24/32 as the reference follows option (c)'s recompute-from-raw clause and the Reporting Rules (:240; LASSI Paper Metrics). OQ-032 asks the owner to confirm that reading, without blocking anything.
- 24/32 against 23/32: the printed atomicCost Ratio is 0.5854 and the one recomputed from Table IV is 43.9190 / 45.8775 = 0.9573 (`$PY -c "print(43.9190/45.8775, 1/1.1)"` printed `0.9573102283254319 0.9090909090909091`), so the count is 24/32 at either reading of the within-10% rule.
- Wilson 95% of the paper counts (lassi.analysis.stats.wilson_interval): 23/32 = 0.719 (0.546 to 0.844), 24/32 = 0.750 (0.579 to 0.867), 25/32 = 0.781 (0.612 to 0.890).
- No code marks a metric reproduced at 1ccf1fb: `git grep -n -i reproduc 1ccf1fb -- lassi` finds only bible cross-references, the compile-stage label, and upstream quirks.
- The copied row: GPT-4's atomicCost row repeats the layout row above it (LASSI Paper Metrics); no published data recomputes a Sim-T or a Self-corr, since the generated codes are unpublished, and changing the 8/32 or 21/32 recount would change a paper value, so the row stands as printed and the bible now says so.

What changed: assets/scoring/lassi-paper.yaml swaps recount (now 24/32) and recount_alternate (now 23/32) for OMP -> CUDA within 10% or faster, and its header cites the OQ-021 Decision Log row; the Markdown metric tables label the recount "Paper recount (reference)" and the published value "Paper published (for reference only)", the Parquet names and order unchanged; the legend says no table marks a headline metric reproduced (the P10 gate applies OQ-032's test); tests/analysis/test_lassi_paper_values.py and test_run_metrics.py follow the new order and pin it; LASSI Paper Metrics (the copied row; the reproduction test), Run Metrics (Paper values), and Acceptance Criteria; a Decision Log row that cites the 2026-09-24 OQ-021 row.

## Question 8: df-v0 weights

Closure: owner-queue item OQ-033. Applied under the direction: (c), keep every weight and review them at two named points.

Evidence ([MEASURED] rows from the gate's logged pass; arithmetic from the file):

- The file is unchanged since P2.6: `git show 1ccf1fb:assets/scoring/df-v0.yaml | sha256sum` printed 1bd8b1ea88ea4da9b2c655e1c8287fd4f25ee4d7262aa32b200f178748a14ca4, the digest in results/p2-gate/score/provenance.json, and `git log --oneline 1ccf1fb -- assets/scoring/df-v0.yaml` printed only `9948ef8 P2.6: Add the df-v0 score profile`.
- Coverage, `git show 1ccf1fb:results/p2-gate/score/review.md | grep '^| df-v0 | <component> |' | sort | uniq -c`: stage_base -0.8 x37, 0.0 x7, 0.2 x3; warning_count 0.0 x45, 1.0 x2; alignment_term 0.0 x47; guard PLACEHOLDER x47. Of the 22 compile-stage warnings in the packet, the two counted sit on S5 attempts (entropy attempt 2, matrix-rotate attempt 0) and 20 on S1 attempts, which df-v0 does not count (up to 8 on randomAccess attempt 0). The multi-turn values: 0.2 (bsearch), 0.18 (matrix-rotate), 0.08 (entropy), -0.25 x2 (layout, randomAccess), -1.05 x5.
- `git show 1ccf1fb:assets/scoring/df-v0.yaml | $PY plans/spikes/p4-p2-review/q8_weights.py` printed:

  ```
  unguarded attempt R range: {'S0': (-1.0, -1.0), 'S1': (-0.8, -0.8), 'S2': (-0.6, -0.6), 'S3': (-0.4, -0.4), 'S4': (-0.2, 0.0), 'S5': (0.0, 1.0)}
  min S5 - max S4 = 0.0
  warning_weight x warning_cap = 0.2 ; S5 base - S4 base = 0.2
  guard_violation = -1.0 ; S0 base = -1.0 ; S1 base = -0.8
  corrections per stage step (0.2 / correction_penalty) = 4.0
  S5 clean, A = 1, 5 corrections: 0.75 ; S5 clean, A = 0, 5 corrections: -0.05 ; S4 clean, 0 corrections: 0.0
  S2, 5 corrections: -0.85 ; S1, 0 corrections: -0.8
  S5 at W >= 10, A = 0, 1 correction: -0.05
  cap 5: multi_turn range [-1.25, 1.0]
  cap 10: multi_turn range [-1.5, 1.0]
  ```

- S2 and S3 cannot occur on a source-level target (1ccf1fb:lassi/core/stages.py, the stage constants' comment), and a faithful, uncapped recipe can be scored with df-v0 (tests/fixtures/recipes/p1-dry-run.yaml binds score: df-v0), so the multi-turn value has no lower bound there. df-v0 is not yet a reward: the Reward Function requires it to run on real trajectories first, and no phase before P7 trains.

What changed: the Reward Function records the keep and the two review points; plans/PHASE-NOTES.md (All Phases) carries the first review point and the P7 one; a Decision Log row. No weight changes.

## Remote steps (from a clean commit that holds every P4.15 change, P4.6's included)

The commands run from a clean checkout of that commit (the detached worktree C:/dev/lassi-clean, plans/LESSONS.md); their results go to results/p4-p2-review/ with provenance.json and summary.md. Every file they write lies under /mnt/nvme10/joseph_ufl: the slot and its venv, the uv cache ($UV_CACHE_DIR), TMPDIR, the bench sources, the runs root, and TIKTOKEN_CACHE_DIR=$LASSI_SCRATCH/.cache/tiktoken, which the commands set before tiktoken is imported.

1. Host check: `uv run tools/rx.py doctor`.
2. Cache the encoding (tiktoken's own loader, with the network, writes the one cached file into the directory the variable names; the output is the encoding's name and size, the file's name, size, and sha256):

   ```
   uv run tools/rx.py run -- 'export TIKTOKEN_CACHE_DIR="$LASSI_SCRATCH/.cache/tiktoken"; du -sh /mnt/nvme10/joseph_ufl; mkdir -p "$TIKTOKEN_CACHE_DIR" && uv run python -c "import tiktoken; e = tiktoken.get_encoding(\"cl100k_base\"); print(e.name, e.n_vocab)" && ls -l "$TIKTOKEN_CACHE_DIR" && sha256sum "$TIKTOKEN_CACHE_DIR"/*'
   ```

3. The scoring pass, the remote tiktoken tests, and the question 6 check, in one run:

   ```
   uv run tools/rx.py run -- 'export TIKTOKEN_CACHE_DIR="$LASSI_SCRATCH/.cache/tiktoken"; du -sh /mnt/nvme10/joseph_ufl; uv run tools/fetch_upstream.py && LASSI_REQUIRE_TIKTOKEN=1 uv run pytest -q -p no:cacheprovider -m remote tests/scoring/test_similarity_tiktoken_remote.py; t=$?; uv run lassi score $LASSI_RUNS_ROOT/runs/demo-rngd-cpu-1 --profile df-v0 --profile lassi --score-id p4-p2-review; rc=$?; echo "lassi score exit=$rc tiktoken tests exit=$t"; uv run python plans/spikes/p4-p2-review/q6_similarity.py --bench-root "$LASSI_SCRATCH/bench/lassi-hecbench-10@692cba32c5744f6ef024cca59f65e9488edba8bf" --run $LASSI_RUNS_ROOT/runs/demo-rngd-cpu-1 --tiktoken; ls $LASSI_RUNS_ROOT/scores/p4-p2-review; exit $((rc || t))'
   ```

4. `uv run tools/rx.py pull <id of step 3> --into results/p4-p2-review`, then summary.md written from provenance.json and the pulled files: the sim_t_tiktoken values beside sim_t and sim_t_c, the question 6 lines as [MEASURED] with the rx id, the source run's limits (demo model, not faithful, cap 5, CPU proxy, CUDA to OpenMP, n = 1), alpha01's host CPU as its programs' device, and that the paper columns stand beside a run that is not faithful, so no value there marks a metric reproduced.
