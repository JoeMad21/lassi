# Window Of 2026-11-02: sweep18, The TTIR Ladder And d2m-min For Llama-3.3-70B On ttsim

Stages run, in order:

- baseline: builds and runs both references. The tt reference runs on ttsim, a simulator. The C++ reference is built
  with g++ 12.3.0 and runs on alpha01's CPU.
- verbatim_context: puts the API reference in the prompt. Every recipe but ttir-l0 has this stage.
- generate.
- compile.
- run on ttsim v1.3.4.
- output check against the item's declared tolerance.

Not run: silicon. No program ran on the Wormhole cards. ttsim is a simulator, so nothing here measures performance.

This summary covers the window's ten passes. Passes 1 to 8 came from one job, whose records are in this folder.
Passes 9 and 10 came from a second job, whose records are in results/window-20261102-sweep18b-pass1.

Provenance:

- **First job.** provenance.json in this folder, written by `rx pull`.
  - rx job 20261101-214439-suite18-7433 ran on alpha01 (Ubuntu 22.04.5 LTS) from commit
    b47bd6080945237dac143bf305a8cd1d0a8e48f6, clean tree, rc 0.
  - It started at 2026-11-01 21:44:39 and ended at 2026-11-02 03:58:33 (alpha01 clock, UTC-8).
  - It waited on alpha01, with no card and no server, until its start time of 2026-11-02 00:00:00 (03:00 EST).
  - It then ran passes 1 to 8, its --max-passes cap.
- **Second job.** results/window-20261102-sweep18b-pass1/provenance.json.
  - rx job 20261102-035924-suite18b-75cd ran from the same commit, clean tree, rc 0, from 03:59:24 to 04:56:22.
  - It ran passes 9 and 10.
  - It then stopped with 1118 s left to the deadline, under --min-left's 1860 s.
- **Pins** (provenance.json, pins): tt-mlir 0.9.0.dev20260221 (5f396fd6) and its runtime build, the LASSI driver
  lassi-ttmlir-driver@a90585ca29fa, ttsim v1.3.4, tt-metal 5280a9cf, and g++ 12.3.0.
- **Model.** furiosa-ai/Llama-3.3-70B-Instruct-FP8-dynamic at v2026.4, served by furiosa-llm on 4 RNGD cards (npu0 to
  npu3). Settings: temperature 0.6, top_p 0.9, max_tokens 8192, request timeout 1800 s, no reasoning setting.
- **Commands** (provenance.json, cmd):
  - First job: `bash plans/runs/suite/run-passes.sh --start 1793606400 --deadline 1793625300 -- --table
    plans/runs/suite/models.tsv --models llama33-70b --sweep projects/lassi-demo/sweep18 --skip-download`
  - Second job: the same command with `--start 1793620760`.
- **Deadline.** 1793625300 is 05:15:00 alpha01 time, 08:15 EST. With the suite's 900 s margin, every call ended by
  08:30 EST, the owner's end of testing.

## What Counts As A Pass

These are development results on tt-pairs-v0's unassigned items (OQ-083 (b)). They are pass counts, never
performance.

- A trial passed when it reached S5 with alignment 1.0. "First attempt" means it passed with 0 corrections.
- Compile errors, run errors, output mismatches, and simulator gaps were all returned to the model as corrections, up
  to 8 per trial (loop.max_corrections).
- A trial that ended at a simulator gap counts as not passed.

Sources:

- Each pass's night/passes/<pass>/summary.tsv: one row per trial, in this folder and the other nine pass folders.
- night/tables/passes.tsv: one row per pass and combo, where the ttir row pools the four rungs. It also carries
  earlier windows' rows, each with its own provenance.

## Results

| Recipe (rung) | Feature added | Trials | Passed | Wilson 95% | First attempt | Failed trials end in |
| --- | --- | --- | --- | --- | --- | --- |
| ttir-l0 | none (the bare loop) | 60 | 0 | 0.0 to 0.060 | 0 | correction cap (60) |
| ttir-l1 | the TTIR API reference | 60 | 0 | 0.0 to 0.060 | 0 | correction cap (60) |
| ttir-l2 | retrieval over tt-mlir's TTIR tests | 60 | 56 | 0.841 to 0.974 | 41 | simulator gap (4) |
| ttir-l3 | the cgeist affine source | 60 | 49 | 0.701 to 0.894 | 40 | simulator gap (11) |
| d2m-min | the D2M arm at rung l1's features | 80 | 0 | 0.0 to 0.046 | 0 | correction cap (80) |

Passed trials per item, out of 20 each:

| Recipe | eltwise_binary | eltwise_sfpu | matmul_multi_core | loopback |
| --- | --- | --- | --- | --- |
| ttir-l0 | 0 | 0 | 0 | not in the recipe |
| ttir-l1 | 0 | 0 | 0 | not in the recipe |
| ttir-l2 | 20 | 16 | 20 | not in the recipe |
| ttir-l3 | 20 | 9 | 20 | not in the recipe |
| d2m-min | 0 | 0 | 0 | 0 |

TTIR trials passed in each pass, out of 24: 9, 10, 12, 10, 9, 11, 11, 12, 10, 11. d2m-min passed 0 of 8 in every pass
(night/tables/passes.tsv in each pass folder). The other nine pass folders each hold their own pass's table.

Pass 1 alone (pass 20261102-0000; night/passes/20261102-0000/summary.tsv):

| Recipe | Trials | Passed | First attempt | Ended at a simulator gap |
| --- | --- | --- | --- | --- |
| d2m-min | 8 | 0 | 0 | 0 |
| ttir-l0 | 6 | 0 | 0 | 0 |
| ttir-l1 | 6 | 0 | 0 | 0 |
| ttir-l2 | 6 | 5 | 4 | 1 |
| ttir-l3 | 6 | 4 | 3 | 2 |

## Attempts

The attempt-level counts come from rx 20261102-045707-exec-9f2c, run at 2026-11-02 04:57 alpha01 time.

- It was a read-only pass over the trial records under
  lassi-runs/runs/sweep-<pass>-llama33-70b--<recipe>, for the ten passes above.
- The script was kept outside the repository. It prints each recipe's six most common error diagnostics, so a
  diagnostic outside a recipe's top six is not listed.

Its whole output:

```
slug llama33-70b; passes 20261102-0000 20261102-0030 20261102-0101 20261102-0130 20261102-0159 20261102-0234 20261102-0302 20261102-0332 20261102-0359 20261102-0426

d2m-min: 80 trials, 720 attempts
  attempts: compile-failed 717, no-file-accepted 3
  corrections that resubmitted the previous files unchanged: 491 of 640
  trials that passed after a sim-gap attempt: 0
    690 verify [None] custom op 'func' is unknown (tried 'builtin.func' as well)
     26 verify [None] expected attribute name
      1 verify [None] expected operation name in quotes
      1 compile [duplicate-file] the FILE block at line 15 repeats the file 'eltwise_binary.mlir' from line 2; it was dropped and the
      1 compile [duplicate-file] the FILE block at line 28 repeats the file 'eltwise_binary.mlir' from line 2; it was dropped and the
      1 compile [duplicate-file] the FILE block at line 43 repeats the file 'eltwise_binary.mlir' from line 2; it was dropped and the

ttir-l0: 60 trials, 540 attempts
  attempts: compile-failed 469, no-file-accepted 71
  corrections that resubmitted the previous files unchanged: 374 of 480
  trials that passed after a sim-gap attempt: 0
    469 verify [None] custom op 'func' is unknown (tried 'builtin.func' as well)
      8 compile [duplicate-file] the FILE block at line 25 repeats the file 'eltwise_binary.mlir' from line 2; it was dropped and the
      7 compile [duplicate-file] the FILE block at line 21 repeats the file 'eltwise_sfpu.mlir' from line 2; it was dropped and the f
      6 compile [duplicate-file] the FILE block at line 22 repeats the file 'eltwise_sfpu.mlir' from line 2; it was dropped and the f
      6 compile [duplicate-file] the FILE block at line 24 repeats the file 'eltwise_sfpu.mlir' from line 2; it was dropped and the f
      6 compile [duplicate-file] the FILE block at line 23 repeats the file 'eltwise_sfpu.mlir' from line 2; it was dropped and the f

ttir-l1: 60 trials, 540 attempts
  attempts: compile-failed 540
  corrections that resubmitted the previous files unchanged: 336 of 480
  trials that passed after a sim-gap attempt: 0
    540 verify [None] custom op 'func' is unknown (tried 'builtin.func' as well)

ttir-l2: 60 trials, 112 attempts
  attempts: compile-failed 16, passed 56, sim-gap 40
  corrections that resubmitted the previous files unchanged: 10 of 52
  trials that passed after a sim-gap attempt: 1
     40 run [UnimplementedFunctionality] UnimplementedFunctionality: tensix_execute_sfploadmacro: macro_index=0 sequence=0x1b8400de template[
     14 verify [None] expected attribute name
      2 verify [None] expected '}' in attribute dictionary

ttir-l3: 60 trials, 157 attempts
  attempts: compile-failed 9, passed 49, sim-gap 99
  corrections that resubmitted the previous files unchanged: 15 of 97
  trials that passed after a sim-gap attempt: 0
     99 run [UnimplementedFunctionality] UnimplementedFunctionality: tensix_execute_sfploadmacro: macro_index=0 sequence=0x1b8400de template[
      6 verify [None] expected attribute name
      3 verify [None] expected '}' in attribute dictionary
```

How to read it:

- **"no-file-accepted"**: an answer that stopped before the compiler, because of a compile-stage error in how the
  answer gave its files (lassi/core/files.py).
  - In d2m-min, the listed duplicate-file refusals account for all 3: answers that gave the same file in more than one
    block.
  - In ttir-l0, the listed duplicate-file refusals account for 33 of the 71. The output does not show the cause of the
    other 38, since it lists only six lines per recipe.
- **"sim-gap"**: an attempt whose program ttsim could not run, because the simulator does not implement an
  instruction the program uses.
- **Unchanged resubmissions**: the count excludes each trial's first attempt.

## What The Counts Show

- **Rungs l0 and l1, and d2m-min.** Every TTIR attempt that reached the compiler failed on the same error, "custom op
  'func' is unknown (tried 'builtin.func' as well)": 469 of 469 compile failures in l0 and 540 of 540 in l1. In
  d2m-min, 690 of its 717 compile failures were that error.
  - The model writes the function header as `func @name`, a spelling the pinned MLIR does not parse.
  - Each correction quoted the error with its file, line, and column, along with the file the model had answered.
  - No trial changed the header within 8 corrections. Most corrections returned the previous file unchanged: 374 of
    480 in l0, 336 of 480 in l1, and 491 of 640 in d2m-min.
- **Rung l2.** Its resolved recipe differs from l1's by the retrieval setting alone, so retrieval is the feature that
  moved the model past the header: 56 of 60 passed, 41 of them on the first attempt.
- **Rung l3.** The cgeist affine source gave no gain over l2: 49 of 60, with intervals overlapping l2's. Its extra
  failures are all eltwise_sfpu simulator gaps (11 trials, against l2's 4).
- **Simulator gaps.** Every failed l2 and l3 trial ended at the same ttsim gap,
  `UnimplementedFunctionality: tensix_execute_sfploadmacro`, on eltwise_sfpu. Returning the gap to the model
  recovered one trial in l2 and none in l3. These trials count as not passed in the tables above. The gap is the
  simulator's: ttsim does not implement that instruction, and how those programs behave on silicon is unknown.
- **Wrong outputs.** None. Every program that ran to completion on ttsim matched its item's declared tolerance.
- **Comparison with sweep19.** l1, l3, and d2m-min match sweep19's gpt-oss-120b arms (ttir-min-gap, ttir-full-gap,
  d2m-min-gap) in everything but the model and its sampling and request settings. sweep18 uses Llama's own sampling,
  no reasoning setting, max_tokens 8192, and a 1800 s timeout. The comparison waits on the pull of the 2026-11-01
  window, which is not in this folder.

## Hosts And Space

Cards:

- run.sh claims only cards that read 0.00 GiB (plans/runs/rngd-sweep/run.sh, which polls every 60 s). Each pass's
  START line names npu0 to npu3, and no pass logged a WAIT.
- Each pass's server was READY 25 to 26 s after its START line (output.log in this folder and in
  results/window-20261102-sweep18b-pass1). Its /v1/models reply named
  furiosa-ai/Llama-3.3-70B-Instruct-FP8-dynamic (night/passes/<pass>/llama33-70b.models.json, all ten).
- After the second job, at 04:56:44 alpha01 time, no process of this account's sweep remained, and all eight RNGD
  cards read 0.00 GiB (rx 20261102-045644-exec-bfd4).

Space:

- Free space on the shared scratch disk went from 262.8 GB to 255.1 GB (both provenance files).
- The scratch root went from 151G to 159G (the jobs' logs). It is past the 150G cap under the owner's memory lift
  (OQ-074).
