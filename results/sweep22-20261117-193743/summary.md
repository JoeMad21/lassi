# sweep22: The tt-par Development Ladder On Three 30B Models (Task P17.74)

STAGES THAT RAN: baseline (each item's C++ program on the host CPU and its TT reference on ttsim, their agreement
recorded), generate (the served model wrote the TT host program, or in the kh variant the host program and every
kernel, from the item's serial C++ program), compile_loop (the pinned clang++-20 against tt-metal 5280a9cf), run_loop
(the program on ttsim v1.3.4, a simulator, kernels compiled by tt-metal's JIT), and oracle (outputs judged against the
TT reference's by the item's own tolerance), with up to 8 corrections. STAGES THAT DID NOT RUN: the two correction
rungs (P17.73 is not built), the held-out run, any silicon, and three of the six models (below). Every TT program ran
on ttsim, a simulator: no figure here is a timing or performance measurement (Agent Rule 2). Every item is a
development item (tt-par-v0 and tt-par-v0-kh, split dev; OQ-129), so these are development results.

## Provenance

- rx job 20261117-193743-sweep22-8f46 on alpha01, the owner's window of 2026-11-17 (OQ-130 (a);
  plans/runs/2026-11-17-window-plan.md), from clean commit 4d776d8d3cc32b2c3669a34b8d89a30099dca4f6, rc 0:
  provenance.json here (the gate's record). It ran plans/runs/suite/run-passes.sh with SUITE_MARGIN_GB=1 over
  projects/lassi-demo/sweep22: one pass, three run.sh calls (passes 20261117-1937, -2023, -2107), each model served on
  4 RNGD cards at revision v2026.4 from furiosa-venv-2026.4 (night/models-*.tsv), ten recipes side by side.
- The job stopped after its third call: the sweep's STOP file was created at 21:33 alpha01 time with tools/suite.py
  stop (rx 20261117-213313-exec-a223) so that no model would take the shared scratch disk below the gate's 60 GB floor
  (OQ-131; output.log, "STOP file: no more calls"). gemma4-31b, exaone4-32b, and exaone45-33b did not run.
  qwen3-32b's recipes reached its time share (night/sweep-3-qwen3-32b.log, "DONE qwen3-32b ... 597 s left"): nine of
  its ten recipes finished one item, matmul_multicore_reuse, and sk1-host finished none, so it has 9 trials.
- Pulled with tools/suite.py pull into night/ and output.log. Counts below are from night/tables/trials.tsv, the rows
  whose pass begins 20261117- (tabulate.py reads the whole runs root).

## Results

[MEASURED, ttsim v1.3.4 on alpha01] No trial of any model reached S5 with alignment 1.0: no translation matched its
reference. Final stages, per model, over the three items (vecadd_multi_core, matmul_multicore_reuse, pad_multi_core;
qwen3-32b only matmul_multicore_reuse) and both variants (host: cpp -> tt; kh: cpp -> ttkh): S0 the reply held no
usable program; S1 the program did not compile; S4 it compiled and its run failed on ttsim; gap, an S4 trial that
ended at a simulator gap (end reason sim-gap), which is not verifiable on ttsim and is counted apart from the
failures (bible, Reporting Rules; OQ-098); S5 it ran cleanly to the oracle.

| Model | Trials | S0 | S1 | S4, run failed | S4, gap | S5 (alignment 1.0) |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3-coder-30b | 30 | 1 | 13 | 12 | 3 | 1 (0) |
| qwen3-30b-instruct | 30 | 0 | 17 | 13 | 0 | 0 (0) |
| qwen3-32b | 9 | 0 | 5 | 2 | 1 | 1 (0) |

By rung, all three models and both variants together (bare: the old template set, no skeleton; sk0: with the v0
skeleton; p2sk0: the core-grid template set with the v0 skeleton; sk1: the old set with the v1 skeleton; p2sk1: the
core-grid set with the v1 skeleton):

| Rung | Trials | S0 | S1 | S4, run failed | S4, gap | S5 |
| --- | --- | --- | --- | --- | --- | --- |
| bare | 14 | 1 | 12 | 0 | 0 | 1 |
| sk0 | 14 | 0 | 6 | 6 | 2 | 0 |
| p2sk0 | 14 | 0 | 9 | 5 | 0 | 0 |
| sk1 | 13 | 0 | 4 | 6 | 2 | 1 |
| p2sk1 | 14 | 0 | 4 | 10 | 0 | 0 |

The four trials that ended at a gap: qwen3-coder-30b sk0 host (matmul_multicore_reuse and pad_multi_core) and sk1 host
(pad_multi_core), and qwen3-32b sk1 kh (matmul_multicore_reuse). Every attempt that met a gap, 25 in these passes, is
listed in night/tables/not_verifiable.tsv and night/tables/hardware_check.tsv, for a check on hardware.

These are first-look counts, one trial per item and rung, on development items: small samples, read as a direction
rather than a ranking. With no skeleton, 12 of 14 programs did not compile; with a skeleton, 32 of 55 compiled, of
which 27 failed at run time, 4 ended at a simulator gap, and 1 ran cleanly to a mismatch. In qwen3-30b-instruct's host
variant every skeleton rung compiled all three items, and the bare rung none.

## Cores Of The Translations That Ran Cleanly

[MEASURED, ttsim v1.3.4 on alpha01] Each translation whose last attempt reached S5 was run once more through the
sandboxed ttsim executor with count_cores (plans/runs/tt-par/count_cores.py, from clean commit 61c90d9): cores/ holds
the tables, and cores/qwen3-coder-30b-job/ and cores/rest-job/ the jobs' provenance.json (rx
20261117-204129-count-cores-b916 and 20261117-221244-count-cores-fd9f). Both reruns' outputs equal their attempts'.

| Model | Rung, variant, item | Alignment | Worker cores that ran a kernel, of 80 |
| --- | --- | --- | --- |
| qwen3-coder-30b | sk1, host, matmul_multicore_reuse | 0.0 | 72 |
| qwen3-32b | bare, kh, matmul_multicore_reuse | 0.0 | none: no cores note |

qwen3-coder-30b's program with the v1 skeleton ran its kernels on 72 of 80 worker cores, with wrong outputs; the
item's own reference, matmul_multicore_reuse's, runs on 56 (results/tt-par-v0-ttsim), so the translation spread over
more cores than its reference does. For qwen3-32b's kh program the count tool gave "no cores note", which it prints
both for a program that never opened the device and for a Watcher rerun with no complete dump; the read-only probes
(rx 20261117-221300-exec-4b39, 20261117-221312-exec-e75f, and 20261117-221321-exec-7735) found no tt-metal device
call in its source and no tt-metal log or JIT cache from its rerun, so it did not open the device: it computed on the
host, with wrong outputs.
