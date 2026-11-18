# tt-par-v0's References On ttsim (Task P17.76)

STAGES THAT RAN: baseline (each item's C++ program built with the pinned g++ and run on the host CPU; its TT
reference built with the pinned clang++-20 against tt-metal 5280a9cf-tracy and run on ttsim; their agreement
recorded), generate (the reference mock answered with the TT reference itself; no model was asked), compile_loop and
run_loop (that reference built and run on ttsim a second time), and oracle (the second run compared with the
baseline's TT run under the item's tolerance: a repeatability check). STAGES THAT DID NOT RUN: no model, no
correction (max_corrections 0), no silicon: every TT run is on ttsim v1.3.4, a simulator. No figure here is a timing
or performance measurement; run.md's wall_s is pipeline wall time on a simulator (Agent Rule 2).

## Provenance

- rx job 20261117-193129-ttpar-check-0e04 on alpha01, from clean commit afd410c2dcf7f0bcd7cd2a954207f63191d69498,
  started 2026-11-17T19:31:29-08:00, ended 19:36:37-08:00, rc 0: provenance.json here (the gate's record, with
  every toolchain pin).
- Two runs, each with its own run.md and provenance.json in the folder of its run id (pins: tt-metal 5280a9cf-tracy,
  ttsim v1.3.4, gcc 12.3.0; device of the C++ programs: the host CPU, AMD EPYC 9534; device of the TT programs:
  "ttsim v1.3.4 (libttsim_wh.so, a virtual Wormhole) on the host CPU, with tt-metal 5280a9cf-tracy: a simulator, not
  silicon"):
  - ttpar-check-eval-1117: projects/lassi-demo/tt-par-check/eval-cpp2tt.yaml, recipe_hash 6567840d...2149, the five
    held-out items (split eval);
  - ttpar-check-dev-1117: projects/lassi-demo/tt-par-check/dev-cpp2tt.yaml, recipe_hash 21f52d27...86fe, the five
    development items (split dev).
- The two run folders here were pulled from lassi-runs/summaries/ttpar-check-1117 (rx 20261117-194918-exec-02ca
  copied each run's run.md and provenance.json there and wrote its reference_agreement.tsv from its trial records,
  trial.json's reference_agreement and final.stage_reached). The raw run trees, with the trial.md files run.md links
  to, stay on alpha01 and in the workstation's .rx/pulls.
- The fetch before the job: rx 20261117-193106-desktop-8r113ei-detached-afd410c2-76c5 (tt-par-v0 and tt-par-v0-kh,
  13 upstream kernel files each, every sha256 checked).

## Results

[MEASURED, ttsim v1.3.4 on alpha01] Every item reached S5: its C++ program and its TT reference built and ran, agreed under the item's own tolerance, and
the reference's second ttsim run matched its first. The agreement is that of the TT reference's outputs with the C++
program's, from the baseline.

| Item | Split | Tolerance | PCC | max_abs | Stage |
| --- | --- | --- | --- | --- | --- |
| vecadd_multi_core | dev | max_abs 0.3 | 0.99999125 | 0.0625 | S5 |
| vecadd_sharding | dev | max_abs 0.2 | 0.99999123 | 0.0625 | S5 |
| pad_multi_core | dev | exact | 1.0 | 0.0 | S5 |
| matmul_multicore_reuse | dev | PCC 0.99 | 0.99981272 | 0.3125 | S5 |
| matmul_multicore_reuse_mcast | dev | PCC 0.98 | 0.99980934 | 0.3125 | S5 |
| row_sum | eval | max_abs 4.0 | 0.99997614 | 0.43359375 | S5 |
| col_max | eval | exact | 1.0 | 0.0 | S5 |
| transpose_2d | eval | exact | 1.0 | 0.0 | S5 |
| row_softmax | eval | max_abs 0.03 | 0.99998544 | 0.005859375 | S5 |
| col_scale | eval | max_abs 0.015625 | 0.99999594 | 0.0078125 | S5 |

No reference needed a fix, so P17.76 changed no program. row_sum's tolerance of 4.0 is about nine times the
agreement measured here (0.434); P17.76 never changes an item's tolerance (plans/p17-portable.md, P17.76), so it
stays and is noted for the owner.

## Each Reference's Cores On ttsim

[MEASURED, ttsim v1.3.4 on alpha01] rx 20261117-204246-count-cores-1bc8, from clean commit 61c90d9: each reference
above was run once more through the sandboxed ttsim executor with count_cores on (task P17.70;
plans/runs/tt-par/count_cores.py --include-eval over both check runs), its outputs equal to its check run's, and the
Watcher's last complete dump counted the worker cores that ran a kernel. The table is pulled into reference-cores/
(references.tsv, whose first line names the device), beside the job's own provenance.json (commit
61c90d9eb7b461917756de2e80b92114665f7f74, dirty false, 2026-11-17T20:42:46-08:00 to 20:44:18-08:00, rc 0). Counts of
cores, never times (Agent Rule 2).

| Item | Split | Worker cores that ran a kernel, of 80 |
| --- | --- | --- |
| vecadd_multi_core | dev | 72 |
| vecadd_sharding | dev | 56 |
| pad_multi_core | dev | 72 |
| matmul_multicore_reuse | dev | 56 |
| matmul_multicore_reuse_mcast | dev | 56 |
| row_sum | eval | 72 |
| col_max | eval | 72 |
| transpose_2d | eval | 72 |
| row_softmax | eval | 72 |
| col_scale | eval | 72 |

The development counts equal the spike's record run (plans/spikes/ttsim-cores.md), which ran the same builds outside
the sandbox.

## The Leak Test's Built Half

tests/bench/test_tt_par_heldout.py's remote test ran on alpha01 from commit 4d776d8 in three slots, which together
hold every built context pack: suite (d2m-ref-5f396fd6, ttir-ref-5f396fd6, ttkernel-5f396fd6,
ttkernel-defs-5f396fd6, ttkernel-ref-5f396fd6, tt-metalium-5280a9cf-full, tt-metalium-5280a9cf-kh; rx
20261117-193850-suite-6dab), window (tt-metalium-5280a9cf; rx 20261117-193859-window-db9d), and demo-run
(cuda-12.5-ch5, openmp-4.0-card; rx 20261117-193913-demo-run-1652). Each run passed: no held-out file shares three
consecutive non-blank lines with any built pack entry or index chunk. The test prints what is absent only when it
skips, so each rx run first printed the test's own built_texts() result from the same checkout: 3564 built texts in
suite (absent: cuda-12.5-ch5, openmp-4.0-card, tt-metalium-5280a9cf), 3555 in window (absent: every pack but
tt-metalium-5280a9cf), and 3555 in demo-run (absent: every pack but cuda-12.5-ch5 and openmp-4.0-card); no retrieval
index was absent in any of them.

## Held-Out Results

The held-out rows above are evaluation results of the references written here, read for no prompt, skeleton, or
correction work (Agent Rule 5).
