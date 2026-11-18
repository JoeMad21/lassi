# sweep21: bale-v0-tt On The 30B Models (Task P17.64)

STAGES THAT RAN: baseline (each bale app's serial C++ program built with the pinned g++ and run on the host CPU),
generate (the served model wrote the whole TT-Metalium program, host program and kernel, from the C++ program),
compile_loop (the host program built with the pinned clang++-20 against tt-metal), run_loop (the program run on ttsim
v1.3.4, a simulator, with its kernels compiled by tt-metal's kernel JIT), and oracle (outputs compared exactly with
the serial program's), with up to 8 corrections. STAGES THAT DID NOT RUN: no silicon; no held-out evaluation, since
every bale-v0-tt item is unassigned (Agent Rule 5). Every TT program ran on ttsim, a simulator: nothing here is a
silicon result, and no figure here is a timing or performance measurement (Agent Rule 2).

## Provenance

- rx job 20261111-204428-suite21-02f0 on alpha01, the owner's window of 2026-11-12
  (plans/runs/2026-11-12-window-plan.md), from clean commit ff91d925e1683468ff21cd35323558f37d9818f8, started
  2026-11-11T20:44:28-08:00, ended 2026-11-12T01:18:31-08:00, rc 0, state done: provenance.json here (the gate's
  record; it cuts each pin file at 4000 characters, so the clang++-20 lines are read from toolchains/tt-metal.pin at
  ff91d925). It ran one pass of plans/runs/suite/run-passes.sh: six run.sh calls (passes 20261111-2100 to
  20261111-2343, one per model) over projects/lassi-demo/sweep21, the six models of plans/runs/suite/models-30b.tsv
  served one at a time on 4 RNGD cards, each furiosa-ai's build at revision v2026.4 from furiosa-venv-2026.4
  (night/models-*.tsv); every TT program ran on ttsim v1.3.4 with tt-metal 5280a9cf. The next pass was not started
  (1589 s to the deadline, under its 1860 s floor).
- Pulled on 2026-11-17 (OQ-130 (a), step (1)) with tools/suite.py pull: the job's output directory is night/, and
  its log is output.log. The six model download logs were left out (not plain ASCII; the models were already in the
  cache).
- Counts below are from night/tables/trials.tsv, the rows whose pass begins 20261111- (tabulate.py reads the whole
  runs root, so the table also holds earlier passes).

## Results

[MEASURED, ttsim v1.3.4 on alpha01] Exact: a trial that reached S5 with alignment 1.0, its outputs equal to the serial
program's. One trial per app and arm; a model's eight recipes ran side by side within its share of the window, so a
model whose trials ran long finished fewer of them (each recipe run ended at its model's time cap; three of
qwen3-32b's ended with rc 1 rather than 124 at that cap, output.log).

| Model | Arm | Trials finished | Exact | Apps exact |
| --- | --- | --- | --- | --- |
| qwen3-coder-30b | bale-ex | 6 | 0 | - |
| qwen3-coder-30b | bale-all | 7 | 0 | - |
| gemma4-31b | bale-ex | 2 | 0 | - |
| gemma4-31b | bale-all | 4 | 2 | transpose_matrix, randperm |
| qwen3-32b | bale-ex | 3 | 0 | - |
| qwen3-32b | bale-all | 2 | 0 | - |
| qwen3-30b-instruct | bale-ex | 8 | 0 | - |
| qwen3-30b-instruct | bale-all | 8 | 0 | - |
| exaone4-32b | bale-ex | 8 | 5 | histo, transpose_matrix, triangle, sssp, randperm |
| exaone4-32b | bale-all | 8 | 5 | histo, transpose_matrix, triangle, permute_matrix, randperm |
| exaone45-33b | bale-ex | 8 | 3 | ig, permute_matrix, randperm |
| exaone45-33b | bale-all | 8 | 8 | every app |

In all, 23 of 72 finished trials were exact. By app: randperm 5 of 11, transpose_matrix 4 of 10, histo 3 of 12,
permute_matrix 3 of 9, triangle 3 of 7, ig 2 of 8, sssp 2 of 7, toposort 1 of 8. qwen3-coder-30b's bale-ex ig trial
reached S5 without alignment 1.0, so it is not counted as exact. These are exploratory results on unassigned items
(OQ-108), which may not steer prompt work (Agent Rule 5), one trial each: small counts, read as a first look rather
than a ranking.
