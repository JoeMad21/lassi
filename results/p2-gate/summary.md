# P2 Scoring: phase gate

Gate (bible Build Roadmap, P2 row, Gate column): "Score components reviewed by J on one full run".

Date: 2026-09-24 (UTC 2026-09-25 00:46 to 00:47). Branch p2-scoring, commit a10fdd0 (clean tree). The gate is an owner review: the checks below ran and passed, and the review itself is owner-queue item OQ-024. P2 is set GATE-OWNER.

## Checks

| Check | Command | rx id | Outcome | Provenance |
| --- | --- | --- | --- | --- |
| Host | `rx doctor`; `du -sh /mnt/nvme10/joseph_ufl` | doctor just before the gate run; du in the gate run | doctor ok with no jobs running (its output is not kept under results/); du 94G | du: the output log of rx e027 (the command is in results/p2-gate/provenance.json) |
| Gate run | `uv run lassi score $LASSI_RUNS_ROOT/runs/demo-rngd-cpu-1 --profile df-v0 --profile lassi --score-id p2-gate` | 20260924-174659-desktop-8r113ei-p2-scoring-e027 | exit 0; all 10 trials scored | results/p2-gate/provenance.json, results/p2-gate/score/provenance.json |
| Load check (not the review) | `lassi score` over p1-gate-dry-run and demo-rngd-1, both profiles | 20260924-174713-desktop-8r113ei-p2-scoring-1b00 | exit 0 for each; 20 of 20 trials scored in each | results/p2-gate/load-check/provenance.json and the two `*.score-provenance.json` |

Device: the scoring passes ran on alpha01's CPU and use no accelerator. Each score's provenance.json holds the scoring commit (a10fdd0, clean), the interpreter (Python 3.10.12), the sha256 of assets/scoring/df-v0.yaml, lassi.yaml, and lassi-paper.yaml, the bench root (lassi-hecbench-10 at 692cba3), and a copy of the source run's provenance.json.

## The packet

- results/p2-gate/score/review.md: one section per trial, with each attempt's stage, diagnostics by code, severity, and place, the run's exit status and hang flag, alignment, and every component of both profiles with its note. It holds no prompt, source, or model text (OQ-018).
- results/p2-gate/score/metrics.md: the Run Metrics table of the run's one arm and direction, with the paper's published values and recounts alongside.
- The source run: demo-rngd-cpu-1 (results/demo-rngd-cpu-run/summary.md; rx job 20260924-093127-demo-rngd-cpu-1ff6, clean 2f052e6, recipe hash 209990b1d6f8). The model furiosa-ai/Llama-3.1-8B-Instruct at revision v2026.2 was served by furiosa-llm 2026.2.1 on RNGD npu1 (rx job 20260924-082436-rngd-serve-9c91; results/demo-rngd-run/summary.md). Toolchain pins: cuda 12.6.3 and nvhpc 24.11, recorded in both provenance files; the scoring passes use neither. Its programs ran on alpha01's host CPU through the sandboxed native executor, whose provenance records the device as null (PHASE-NOTES P2); metrics.md shows "Device: not recorded" for that reason.

## Limits of the run (planning decision, plans/p2-scoring.md)

- A demo model, not an experimental arm.
- Not faithful: a correction cap of 5, and the -mp=multicore CPU proxy instead of GPU offload. The proxy checks outputs, never runtime, so no runtime value is reported (within_10pct is PLACEHOLDER).
- The correction prompts carry upstream's compiler text (-mp=gpu) while the build uses the proxy (-mp=multicore) (results/demo-rngd-cpu-run/summary.md).
- CUDA to OpenMP only, one trial per app, so pass@3 is not computed (n = 1).
- The records predate P2.1 and P2.2: they hold no model requests and no run flags, and review.md names both as not recorded, with the reference stdout size in bytes.
- P2.5 finding for these items (results/p2-proxy-stability/summary.md): every item's masked OpenMP reference stdout is stable under the proxy. dense-embedding's reference prints FAIL under the proxy (one OpenMP team), so its trial cannot pass the oracle here.

## Results [MEASURED]

Values computed by the scoring pass at a10fdd0 from the logged records of demo-rngd-cpu-1; see metrics.md and review.md for every value and note.

- Stage reached by the last attempt: S1 5, S4 2, S5 3. Compiled 5 of 10; ran clean 3 of 10; correct (automated oracle) 0 of 10, Wilson 95% 0.000 to 0.278. The three clean runs printed stdout that the oracle aligned at 0.0, and two compiled programs ended by signal (exit statuses 134 and 137).
- Corrections: 0 for 2 trials, 2 for 1, 5 for 7; 7 of 10 stopped at the correction cap. No fence-quirk hit.
- df-v0, multi-turn (the trial score): 0.200 bsearch, 0.180 matrix-rotate, 0.080 entropy, -0.250 layout and randomAccess, -1.050 for the five trials that stayed at S1 through the cap.
- lassi Sim-T: the faithful sim_t lies between 0.0006 and 0.0070 for every trial, while the C-aware sim_t_c lies between 0.4244 and 0.8546 and sim_l between 0.2747 and 0.8366 (each bound rounded outward to four decimals).

## Review questions for J

1. The run and its limits above: is demo-rngd-cpu-1 the right run to review, or should the review wait for a faithful run?
2. Planning decisions (plans/p2-scoring.md): scoring writes a separate score tree; the output that stands is the last attempt that ran, stale output included; correct needs a clean run (exit 0, no hang) aligned at 1.0; df-v0's W counts only compile- and jit-stage warnings at S4 or S5; final.score is the multi-turn value; weights and paper values live in assets/scoring/.
3. In-task readings: an S5 attempt never aligned is marked alignment_missing and earns no alignment term; a null guard was not checked; a guard violation on the last attempt sets R_final; a reply giving only files the target does not list is S1 (P2.3).
4. The lassi profile's nulls (P2.7): correct is null, with a note, for a clean run never aligned, a trial ended at the baseline, and a compile-only trial; a trial with no attempt has sim_t, sim_t_c, and sim_l null, and its correct is 0 when the reference ran and the trial did not end at the baseline.
5. Run metrics (P2.8): run_rate counts a trial with no attempt as not run while compile_rate and correct_rate exclude a baseline-ended trial; one scenario with no scored trial makes pass@k null for its arm and direction; Sim-T >= 0.6 compares the unrounded sim_t while the paper prints two decimals; where the B0 criterion's interval (lassi.analysis.paper b0_interval) should be shown.
6. Sim-T (OQ-022, kept both, review later): on this run the faithful sim_t, Python tokenize over positional token tuples, stays below 0.01 on every trial, while the paper's Tables VI and VII show Sim-T values of 0.6 and above for 23 of 66 correct trials (the LASSI Paper Metrics recount, 8/32 and 15/34; the published percentages give 29/66). The faithful measure, as the notebook computes it, may not be what the paper's column reports.
7. OQ-021 (open): which paper values mark a metric reproduced.
8. Any df-v0 weight in assets/scoring/df-v0.yaml to change; a change is a Decision Log entry.

## Verdict

PASS for the gate's checks: every command exited 0 and every trial was scored. The owner review is open as OQ-024, so P2 is GATE-OWNER, and work moves to P4.
