# Window Of 2026-10-26: Kernels-Free C++ -> TT, TT -> C++, And The First ttkernel Trials

Provenance: provenance.json in this folder (written by `rx pull`), rx job 20261026-055605-window-119f on alpha01
(Ubuntu 22.04.5 LTS), commit 3d4e0e988c444b82f153dcf5cff3f749bc8efe30, clean tree, 2026-10-26 05:56:05 to 07:41:17
(alpha01 clock, UTC-7), rc 0. Pins: tt-metal 5280a9cf, ttsim v1.3.4, tt-mlir 0.9.0.dev20260221 (5f396fd6), g++ 12.3.0.
Model: furiosa-ai/gpt-oss-120b at v2026.4 on 4 RNGD cards (plans/runs/suite/models.tsv). Command (provenance.json, cmd):
run-suite.sh over projects/lassi-demo/mlir1 (only ttkernel-maxabs) and projects/lassi-demo/sweep8 (only v1* and tc*).

Devices: every TT program ran on ttsim v1.3.4 with tt-metal 5280a9cf, a simulator; every C++ program on alpha01's
CPU. These are pass counts, never performance. A pass is a trial that reached S5 (built and ran clean) with alignment
1.0 on the held-out inputs. Tables: night/tables/trials.tsv, the job's own rows (night/calls.tsv), tabled by the window
runner's report.py (kept outside the repository; the counts are re-derivable from night/tables/trials.tsv filtered
by night/calls.tsv) into csv/ (by_set.csv, by_item.csv, by_recipe.csv, by_end.csv, trials.csv). night/tables/
trials.tsv also carries earlier passes' rows, each with its own provenance; only this job's rows are counted here.

| Combination (sweep8 or mlir1) | Direction | Trials | Passed | Wilson 95% |
| --- | --- | --- | --- | --- |
| v1allsk (examples, host skeleton, API pack and lookup, docs retrieval, restarts) | cpp -> tt | 15 | 15 | 0.796 to 1.0 |
| v1exsk (examples, host skeleton) | cpp -> tt | 15 | 15 | 0.796 to 1.0 |
| v1ex (examples) | cpp -> tt | 14 | 2 | 0.040 to 0.399 |
| tc0 | tt -> cpp | 15 | 15 | 0.796 to 1.0 |
| tc1 | tt -> cpp | 15 | 15 | 0.796 to 1.0 |
| ttkernel (mlir1, loopback only) | cpp -> ttkernel | 3 | 0 | 0.0 to 0.562 |

In cpp -> tt the model writes the TT host program against the item's upstream kernels; v1ex's two passes are both
loopback (csv/by_item.csv). v1ex has 14 trials where its recipes call for 15: v1ex-maxabs's loopback run03 left
no record (no row in night/tables/trials.tsv or night/passes/20261026-0626/summary.tsv); the pulled files do not
show why. The fuzzer found divergent inputs for 5 of tc0's and 5 of tc1's passes (by_set.csv,
fuzzer_divergent_passes). The ttkernel trials' attempts and diagnostics are analyzed in plans/spikes/mlir-issues.md.
