# RNGD sweep, fifth pass (2026-10-23 to 2026-10-24)

Exploratory, one sample per item, direction, and recipe. The records read dirty: the slot held rx snapshot 9099a15
of 36286f3, which adds only the generated files assets/context/tt-metalium-5280a9cf/host_api.tt.txt and
metalium_guide.tt.txt (rx 20261023-201911-exec-a4e0); no recipe of this pass reads them. Nothing here is
[MEASURED] (AGENTS.md, Results: runs from a dirty tree are exploratory).

## Provenance

- Job: rx 20261023-201813-rngd-sweep5-226a, big, on alpha01 (Ubuntu 22.04.5, kernel 6.6.29, 256 CPUs), started
  2026-10-23T20:18:13-07:00 and ended 23:25:58-07:00 (23:18 to 02:25 EDT), rc 0; provenance.json beside this file
  (commit 9099a15, dirty, and the toolchain pins).
- Command: `SWEEP_MAX_SHARE=4800 bash plans/runs/rngd-sweep/run.sh 1792827900 projects/lassi-demo/sweep5
  plans/runs/rngd-sweep/models5.tsv`, then plans/runs/rngd-sweep/tabulate.py into tables/ (trials.tsv, passes.tsv;
  copied from $LASSI_RUNS_ROOT/rngd-sweep/tables5). This pass's rows are those whose pass is 20261023-2018. The
  tabulator reads every sweep run on the host, so the tables also hold the morning window's passes, each from its own
  job: 20261022-2100 (rx 20261022-210057-rngd-sweep-491a), 20261022-2149 (rx 20261022-214907-rngd-sweep2-cd9a),
  20261022-2202 (rx 20261022-220226-rngd-sweep3-6fa1), and 20261022-2234 (rx
  20261022-222829-rngd-sweep4-chain-cf48); this file's provenance is not theirs.
- Models, served one at a time on RNGD cards on 127.0.0.1:8123, without --reasoning-parser (models5.tsv):
  EXAONE-4.0-32B (furiosa-ai/EXAONE-4.0-32B-FP8 at v2026.4, 4 cards) and Qwen3-Coder-30B
  (furiosa-ai/Qwen3-Coder-30B-A3B-Instruct-FP8 at v2026.4, 4 cards) from the venv furiosa-venv-2026.4, and
  Llama-3.1-8B (furiosa-ai/Llama-3.1-8B-Instruct at v2026.2, 1 card) from the venv furiosa-venv. The last read of
  those venvs' versions is rx 20261020-101803-exec-1a45 (furiosa-llm 2026.4.0 and 2026.2.1, firmware 2026.4.0;
  results/tier-a-rngd-20261020/summary.md); they were not read again for this pass.
- Devices: every TT program ran on ttsim v1.3.4 (a simulator, libttsim_wh.so on the host CPU) with tt-metal
  5280a9cf; every C++ program ran on the host CPU. No Wormhole card was used. Wall times are not performance.
- Recipes: projects/lassi-demo/sweep5/<slug>/, Tier A of tt-pairs-v0 (unassigned split), correction cap 15. fz is
  the fourth pass's recipe unchanged (the fuzzer adversary after the oracle, frozen seed 1, budget 32); mmfz adds the
  coarse mismatch report (loop.correct_on_mismatch, OQ-077). The adversary never feeds the loop, so each fz trial's
  oracle outcome is the base combination's and each mmfz trial's the mm combination's.

## Outcome by model and recipe

| Model | Recipe | Trials | S5 (ran clean) | Alignment 1.0 | With a fuzzer divergence |
| --- | --- | --- | --- | --- | --- |
| EXAONE-4.0-32B | fz | 10 | 0 | 0 | 0 |
| EXAONE-4.0-32B | mmfz | 10 | 2 | 1 | 1 |
| Qwen3-Coder-30B | fz | 10 | 2 | 1 | 0 |
| Qwen3-Coder-30B | mmfz | 10 | 3 | 1 | 0 |
| Llama-3.1-8B | fz | 5 | 0 | 0 | 0 |
| Llama-3.1-8B | mmfz | 5 | 0 | 0 | 0 |

From tables/passes.tsv. Llama-3.1-8B's four recipes ended at their time share, rc 124 from timeout at 4800 s
(SWEEP_MAX_SHARE), at 23:25:56-07:00 in the job's log; they finished 5 of their 10 trials each way, all cpp -> tt.
EXAONE's and Qwen's recipes ended with rc 0 after every trial.

## The trials that ran clean

| Model | Recipe | Direction and item | Alignment | Corrections | Attempts with a mismatch | Fuzzer |
| --- | --- | --- | --- | --- | --- | --- |
| EXAONE-4.0-32B | mmfz (pcc) | cpp -> tt matmul_single_core | 1.0 | 0 | 0 | 2 of 32 inputs diverge |
| EXAONE-4.0-32B | mmfz (maxabs) | tt -> cpp eltwise_sfpu | 0.0 | 15 | 14 | not run (no pass) |
| Qwen3-Coder-30B | fz (maxabs) | tt -> cpp loopback | 1.0 | 0 | 0 | 0 of 32 diverge |
| Qwen3-Coder-30B | fz (maxabs) | tt -> cpp eltwise_binary | 0.0 | 0 | 0 | not run (no pass) |
| Qwen3-Coder-30B | mmfz (maxabs) | tt -> cpp loopback | 1.0 | 0 | 0 | 0 of 32 diverge |
| Qwen3-Coder-30B | mmfz (maxabs) | tt -> cpp eltwise_binary | 0.0 | 15 | 15 | not run (no pass) |
| Qwen3-Coder-30B | mmfz (pcc) | tt -> cpp matmul_single_core | 0.0 | 15 | 12 | not run (no pass) |

Alignment, corrections, and the fuzzer columns are from tables/trials.tsv. "Attempts with a mismatch" counts the
attempts that carry an oracle-mismatch diagnostic, read from the trial records on the host (rx
20261023-230609-exec-d160). Each of the three trials ended at the cap with a mismatching last attempt, which asked
no correction, so the reports fed back to the model were one fewer: 13, 14, and 11. Every other trial ended at the
correction cap with its last attempt at S1 (a build error), S4 (a failed run), or, for Llama-3.1-8B's fz cpp -> tt
matmul_multi_core, S0 (no extractable output).

## Reading

- One item passed with no input found that breaks it: Qwen3-Coder-30B's tt -> cpp loopback, in both recipes, on
  the first attempt.
- EXAONE-4.0-32B's cpp -> tt matmul_single_core passed the held-out inputs in the mmfz recipe, on its first attempt,
  so the mismatch report never acted; in the fz recipe, which is the fourth pass's setup unchanged, the same item
  ended at S1. With the fourth pass's pass, this setup has passed it in two of three tries, and the fuzzer again
  found inputs where the passing program is wrong.
- The coarse mismatch report reached three trials that ran clean but missed the tolerance, and none of them was
  brought within it after 11 to 14 reports. On this one sample, the order-of-magnitude report did not help these
  models correct their numbers.
- Every other cpp -> tt trial ended at the cap without a clean run, the barrier the API reference (P17.21) and the
  retrieval over the docs (P17.22) are meant to address; neither was in this pass.
