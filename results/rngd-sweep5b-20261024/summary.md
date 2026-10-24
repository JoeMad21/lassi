# RNGD sweep, fifth pass, second sample of Qwen3-Coder-30B (2026-10-24)

One more sample of the fifth pass's Qwen3-Coder-30B recipes, in the second window, 02:34 to 03:07 EDT. From a clean
commit: exploratory (one sample per item, direction, and recipe), not a gate run.

## Provenance

- Job: rx 20261023-233453-rngd-sweep5b-0a2a, big, slot sweep5b, on alpha01, started 2026-10-23T23:34:53-07:00 and
  ended 2026-10-24T00:07:34-07:00 (02:34 to 03:07 EDT), rc 0; provenance.json beside this file (commit e482997,
  clean, and the toolchain pins).
- Command: `SWEEP_MAX_SHARE=4800 bash plans/runs/rngd-sweep/run.sh 1792828200 projects/lassi-demo/sweep5
  plans/runs/rngd-sweep/models5-qwen.tsv`, then plans/runs/rngd-sweep/tabulate.py into tables/ (copied from
  $LASSI_RUNS_ROOT/rngd-sweep/tables5b). This sample's rows are those whose pass is 20261023-2334; the other rows are
  the earlier passes, each from its own job (results/rngd-sweep5-20261024/summary.md names them).
- Model: furiosa-ai/Qwen3-Coder-30B-A3B-Instruct-FP8 at v2026.4 on 4 RNGD cards, 127.0.0.1:8123, from the venv
  furiosa-venv-2026.4 (last version read: rx 20261020-101803-exec-1a45, furiosa-llm 2026.4.0; not read again).
- Devices: every TT program ran on ttsim v1.3.4 (a simulator) with tt-metal 5280a9cf, every C++ program on the host
  CPU; no Wormhole card. Wall times are not performance.
- Recipes: projects/lassi-demo/sweep5/qwen3-coder-30b/, as in the first sample: fz (the fourth pass's recipe) and
  mmfz (with the coarse mismatch report), cap 15.

## Outcome

| Recipe | Trials | S5 (ran clean) | Alignment 1.0 | With a fuzzer divergence |
| --- | --- | --- | --- | --- |
| fz | 10 | 3 | 1 | 0 |
| mmfz | 10 | 2 | 1 | 0 |

| Recipe | Direction and item | Alignment | Corrections | Fuzzer |
| --- | --- | --- | --- | --- |
| fz (maxabs) | tt -> cpp loopback | 1.0 | 0 | 0 of 32 diverge |
| fz (maxabs) | tt -> cpp eltwise_binary | 0.0 | 2 | not run (no pass) |
| fz (pcc) | tt -> cpp matmul_single_core | 0.0 | 1 | not run (no pass) |
| mmfz (maxabs) | tt -> cpp loopback | 1.0 | 0 | 0 of 32 diverge |
| mmfz (maxabs) | tt -> cpp eltwise_binary | 0.0 | 15, at the cap | not run (no pass) |

From tables/trials.tsv and passes.tsv. Every other trial ended at the correction cap without a clean run; no
cpp -> tt trial ran clean.

## Reading

The first sample ran from rx snapshot 9099a15 (dirty) and this one from e482997 (clean), with the same recipes;
lassi/ changed between them only for context packs and silicon, which fz and mmfz do not use.

- Qwen3-Coder-30B's tt -> cpp loopback passed again in both recipes, on the first attempt, with no failing input
  found: four of four samples tonight.
- The tt -> cpp eltwise_binary and matmul_single_core translations ran clean and missed the tolerance again; with
  the mismatch report, eltwise_binary stayed out of tolerance to the cap, as in the first sample.
- No cpp -> tt trial ran clean, as in the first sample.
