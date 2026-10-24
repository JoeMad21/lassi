# Feature matrix, a short first pass on Qwen3-Coder-30B (2026-10-24)

The first run with the API reference (P17.21) and retrieval over the tt-metal docs (P17.22) in the prompts, started
at 03:17 EDT, 43 minutes before the owner's window ended. From a clean commit; exploratory, one trial per item in
each combination (one or two per recipe), too few to compare combinations.

## Provenance

- Job: rx 20261024-001735-rngd-matrix-short-496a, big, slot rngd-matrix, on alpha01, started
  2026-10-24T00:17:35-07:00 and ended 00:45:07-07:00 (03:17 to 03:45 EDT), rc 0; provenance.json beside this file
  (commit d6d27c8, clean, and the toolchain pins).
- Command: `MATRIX_MODELS=plans/runs/rngd-sweep/models5-qwen.tsv bash plans/runs/rngd-sweep/matrix.sh 1792828500`
  (deadline 03:55 EDT). matrix.sh built the context pack from $LASSI_TOOLCHAINS/tt-metal@5280a9cf (40911 of 40960
  bytes) and the docs index (sha256 610376e052c6...), and each matched its tracked manifest (rx
  20261024-001816-exec-0c95 read the build logs), so no recipe was left out; then run.sh over
  projects/lassi-demo/sweep6/qwen3-coder-30b/ (16 recipes side by side) and the tabulator into tables/ (copied from
  $LASSI_RUNS_ROOT/rngd-sweep/tables6; this pass's rows are those whose pass is 20261024-0017). The tables also
  hold the earlier passes' rows, each from its own job (results/rngd-sweep5-20261024/summary.md and
  results/rngd-sweep5b-20261024/summary.md name them); this file's provenance is not theirs.
- Model: furiosa-ai/Qwen3-Coder-30B-A3B-Instruct-FP8 at v2026.4 on 4 RNGD cards, 127.0.0.1:8123, from the venv
  furiosa-venv-2026.4 (last version read: rx 20261020-101803-exec-1a45, furiosa-llm 2026.4.0; not read again).
- Devices: every TT program ran on ttsim v1.3.4 (a simulator) with tt-metal 5280a9cf, every C++ program on the host
  CPU; no Wormhole card. Wall times are not performance.
- Recipes: cpp -> tt Tier A, cap 15; the API reference (context pack and summarize_context), retrieval (k 4,
  max_chars 6000, generate and correct), and the coarse mismatch report each on or off; the fuzzer in all.

## Outcome

25 trials finished across the 16 recipes, one or two each, before the time share ended them: eltwise_binary,
eltwise_sfpu, and matmul_multi_core in every combination, matmul_single_core only in api-pcc, and loopback in none,
so they do not cover Tier A. Every one ended at the correction cap at S1, a build error. No trial ran clean, so
neither the mismatch report nor the fuzzer acted.

The features did reach the prompts (rx 20261024-005401-exec-d3b6 read the trial records): every trial of a recipe
with the API reference recorded one summarize_context request, and every trial of a recipe with retrieval recorded
16 requests that carried passages, one per prompt; the other recipes recorded neither.

## Reading

On this little data, neither the API reference nor the retrieved passages got a cpp -> tt translation past the
build in 15 corrections. One trial per item and combination cannot rank them; the full matrix
(plans/runs/rngd-sweep/start-matrix.sh) is meant to run every item of every combination for two models in a full
window.
