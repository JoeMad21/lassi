# The Window Of 2026-10-31: Plan

Booked by the owner on 2026-10-29: Saturday 2026-10-31, 14:00 to 21:00 EDT on alpha01 ("Continue working and plan
ahead"). It serves gpt-oss-120b first (item 0), then the 70B and 30B models on RNGD (below), and, like the window of
2026-10-29, runs every TT program on ttsim, a simulator, and the host CPU; no model-generated program runs on a Wormhole card. The job's deadline is 20:45, so it ends
by 21:00 with its 900 s margin.

## Models

The owner, 2026-10-29: "I plan on deleting GPT OSS soon so we can test a score of 70B and 30B models." The window
serves those models one at a time, the sets below for each. Known rows: llama33-70b
(furiosa-ai/Llama-3.3-70B-Instruct-FP8-dynamic, plans/runs/suite/models.tsv) and Qwen3-Coder-30B-A3B
(furiosa-ai/Qwen3-Coder-30B-A3B-Instruct-FP8, the tier-a-rngd-30b recipes); the owner names the rest. Each needs a
models.tsv row with its revision's size, its snapshot in the host's cache (the owner's download, or the job's own
after gpt-oss-120b is gone: about 199 GB freed), and a recipe directory per set (sweepN/<slug>/) with the model's
own settings: no reasoning effort for a model without a reasoning parser, and its own context length. gpt-oss-120b
stays until item 0 has run in this window (about 17:30 on 2026-10-31); agents never delete files on alpha01.

## Inputs

- The window of 2026-10-29: one job, sweep9 (the six kernels-and-host recipes left out on 2026-10-27) and then sweep10
  (bale-v0-tt, cpp and omp -> ttkh; the ttkernel API reference and retrieval recipes), deadline 2026-10-30 01:45. The
  window runner pulls it into results/window-20261029 in the launch tree and writes its tables.
- The owner's standing direction of 2026-10-23: test every combination of the features added to base LASSI (RAG, API
  reference, training, fuzzing, ...).

## Candidate Sets, In Order

The owner's direction of 2026-10-29 limits every set to MLIR lowering and rewrites (plans/LESSONS.md, Owner
directions): no recipe whose model writes C++ runs in this window. Items 0 to 4 below are read under it: item 0 runs
only the arms' mlir-rag recipes (--only), item 2 is not run, and items 1 and 4 keep only their MLIR recipes. Added:
the TTIR and D2M arms (compile-only until the tt-mlir runtime of OQ-111 is built and a driver exists, then with
outputs checked), and bale-v0's apps on the ttkernel path (sweep14) once built.

0. gpt-oss-120b at high reasoning effort in four temperature arms (0.2, 0.5, 0.7, 1.0; top_p 1.0; tasks P17.51 and
   P17.52, the owner's directions of 2026-10-29), with restarts (P17.50), over the sets the window of 2026-10-29
   shows most worth it. It runs first, before the owner deletes gpt-oss-120b; proposed split: gpt-oss-120b from 14:00
   to about 17:30, then the 70B and 30B models that are in the cache, to the 20:45 deadline. The split and each
   arm's size are set on 2026-10-30 from that night's timings.

1. Completion. Any sweep9 or sweep10 recipe the 01:45 deadline cut, read from the job's own log (its left-out lines,
   which `suite.py pull` now lists) and calls.tsv, reruns first, at the same recipe hashes.
2. bale-v0-tt with the feature combinations. sweep10 ran one combination, the worked examples (kh-ex's). sweep11 adds
   the combinations sweep9 has for the same language (ttkh): examples with the host skeleton (kh-exsk), and everything
   (kh-allsk: examples, skeleton, the kernel API pack, and tt-metal-docs retrieval), for cpp and omp. A new template set,
   bale-tt-v1, is bale-tt-v0 with the frames those features need copied from tt-kh-v0, and bale-tt-v0's own wording
   unchanged: rewording bale-tt-v0 after sweep10's results would be prompt tuning on unassigned items (Agent Rule 5).
   Built and checked on 2026-10-30 (task P17.45).
3. The ttkernel path, by tonight's result. If mlir-ref or mlir-rag attempts now build, more trials of the better
   combination (n 5). If none builds, first a read-only pull of the trial records to classify the verify errors, then
   a fix round on tt-pairs-v0-ttkernel, a development set (OQ-083 (b)), where prompt changes are allowed.
4. Per model. The 70B and 30B models run the same sets; run-suite.sh serves one model at a time and shares the time
   between calls, so with several models each set may run on a subset of items or fewer trials (chosen once the
   2026-10-29 timings are known).

Not in this window: training (task P17.38). The owner chose evaluation only for this window (OQ-110 (a)), and on
2026-10-29: "I'm trying to get you AMD access, so no training until then."

## Before The Window (2026-10-30, unattended where possible)

- Read the window of 2026-10-29: tables, left-out lines, and each set's ends; write results/window-20261029/summary.md
  from its provenance.json; commit it, with results/window-20261026 and results/window-20261027.
- Update the results document with the night's numbers.
- Add each 70B and 30B model's models.tsv row and recipe directories (sweep9, sweep10, sweep11 under its slug), with
  tests as test_sweep9_recipes.py has them.
- Build sweep11 (item 2) and, if item 3 needs it, the ttkernel fix; tests, audit, commit, push before 13:30.
- Sync the master bible for P17.38 to P17.44 and P17.50.
- Write plan.json for 14:00 (hard end 21:00, deadline 2026-10-31 20:45, so the job's timeout ends at 21:00) and check it with `suite.py plan`. The owner
  starts the window runner; if no job is running at 14:06, this session launches the same command, the booking being the owner's yes for the window's runs (Agent Rule 17). The pull after the job is a short CPU-only read, under the owner's standing permission of 2026-10-26.
