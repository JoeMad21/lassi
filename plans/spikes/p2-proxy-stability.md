# Spike P2.5: stability of the reference output under the CPU proxy

- Status: done. The run is from a clean commit; its per-item findings are [MEASURED] (results/p2-proxy-stability).
- Question: is each item's masked reference stdout stable enough under the CUDA to OpenMP proxy for stdout_mask to score against it?
- Bible: Oracles (stdout_mask rules), Harness Contract (CUDA to OpenMP proxy), Sandbox (native runs, attempt run limits).

## Run

- Recipe: tests/fixtures/recipes/p2-proxy-mock.yaml. It extends projects/lassi-demo/rngd-cpu.yaml with the mock backend and three trials:
  - CUDA to OpenMP, every item of lassi-hecbench-10;
  - nvcpp-multicore and the native executor;
  - stdout_mask with passfail;
  - n = 3.
- Commit 207e4dd, clean tree. rx job 20260924-133017-p2-proxy-3a3a (slot desktop-8r113ei-p2-scoring), 2026-09-24 20:30 to 20:39 UTC.
- Command: `uv run tools/fetch_upstream.py && uv run tools/extract_lassi_assets.py && uv run tools/fetch_bench.py assets/bench/lassi-hecbench-10.yaml && LASSI_REQUIRE_SANDBOX=1 uv run lassi run tests/fixtures/recipes/p2-proxy-mock.yaml --run-id p2-proxy-stability`
- Result: the log ends `lassi_rc=0`, and all 30 trials reached S5 with no corrections.
- Device: the host CPU (alpha01), through the sandboxed native executor, with the reference and candidate runs limited to 16 CPUs. The run tree's provenance.json records device as null for the native executor; that is a recording gap, noted in PHASE-NOTES.
- The mock answers with the item's reference target, so every candidate is the reference program again. Its passes check the path; they are never model results.

## Method

- The run tree was fetched with `rx pull --path lassi-runs/runs/p2-proxy-stability`.
- Each trial holds its reference run's stdout (Trial.reference_run.stdout_ref) and its candidate run's stdout (the attempt's run.stdout_ref).
- For each item, a script masked each of the six stdouts with the item's masks (lassi.oracles.stdout_mask.mask_stdout, assets/harness/masks/lassi-hecbench-10.yaml): the three reference runs and the three candidate runs. It then compared them, and read each candidate's alignment mean from the trial record.

## Findings

| Item | Masked reference stdouts agree (3 runs) | Masked candidate stdouts agree (3 runs) | All six agree | Alignment means | Reference prints PASS |
| --- | --- | --- | --- | --- | --- |
| atomicCost | yes | yes | yes | 1.0, 1.0, 1.0 | yes |
| bsearch | yes | yes | yes | 1.0, 1.0, 1.0 | not printed |
| colorwheel | yes | yes | yes | 1.0, 1.0, 1.0 | yes |
| dense-embedding | yes | yes | yes | 0.0, 0.0, 0.0 | no (prints FAIL) |
| entropy | yes | yes | yes | 1.0, 1.0, 1.0 | yes |
| jacobi | yes | yes | yes | 1.0, 1.0, 1.0 | yes |
| layout | yes | yes | yes | 1.0, 1.0, 1.0 | yes |
| matrix-rotate | yes | yes | yes | 1.0, 1.0, 1.0 | yes |
| pathfinder | yes | yes | yes | 1.0, 1.0, 1.0 | not printed |
| randomAccess | yes | yes | yes | 1.0, 1.0, 1.0 | not printed |

- No item's masked reference stdout varied across three runs, so no item needs a stability mark in the Oracles section.
- dense-embedding scores 0.0 even against its own reference. Its OpenMP reference prints FAIL under the proxy, because the proxy creates one OpenMP team while the program assigns work by team number (results/demo-multicore-proxy). passfail then needs a PASS that never comes. Every metrics table and packet that reports dense-embedding under the proxy must mark it as not scorable there.
- jacobi's CUDA reference, whose "Error after iteration" line PHASE-NOTES flagged, cannot run here: there is no NVIDIA GPU (OQ-003). It stays not checkable until P10. Its OpenMP reference, which this run scores, was stable.
- Three runs per item bound what this shows. A variation rarer than that would not appear.
