# Window of 2026-11-12: plan

The owner, at about 23:25 EST on 2026-11-11: "I booked you 12am-5am EST. Go ahead and do more tests with BALE. I want
BALE applications converted from C++ to TT-Metallium. You were able to do three applications, I want more of the BALE
applications ready. Run them on the 30B models and if we have time, we'll do the other models."

Window: 00:00 to 05:00 EST on 2026-11-12, which is 21:00 to 02:00 on alpha01's clock (UTC-8). The sweep's deadline is
04:45 (epoch 1794476700), so the suite's 900 s backstop ends every call by 05:00. Every run is a development result on
bale-v0-tt's unassigned items: the model writes the whole TT-Metalium program (the host program and its kernel) from
bale's serial C++ program, the host program builds against the pinned tt-metal, and it runs on ttsim v1.3.4, a
simulator (nothing measures performance), with outputs checked exactly against the serial program's. No program runs
on silicon (Agent Rule 9 stands as written); the RNGD cards only serve the models.

## What runs

- sweep21 (task P17.64): all eight bale apps (histo, ig, triangle, transpose_matrix, permute_matrix, sssp, toposort,
  randperm), cpp -> ttkh, in two arms per model: bale-ex (sweep10's worked examples) and bale-all (sweep11's bale-all:
  the kernels pack and its lookup, retrieval over the tt-metal docs, bale-tt-v1, and restarts). Each arm's items go in
  four recipes of two, so a model's eight recipes run side by side: a failing trial makes up to 9 attempts, each a
  generation, a host build, and a kernel JIT and run on ttsim under a 300 s limit (PROJECTED: no 30B trial of these
  items has run).
- The six 30B-class models of sweep20, in serving order qwen3-coder-30b, gemma4-31b, qwen3-32b, qwen3-30b-instruct,
  exaone4-32b, exaone45-33b (plans/runs/suite/models-30b.tsv; sweep20's sampling for each). The EXAONE models come
  last, as in sweep20 (plans/runs/suite/models-30b.tsv), where a later pass reaches them.
- One rx job runs plans/runs/suite/run-passes.sh with --deadline 04:45, from a clean worktree of the pushed commit,
  only after the owner's gate step of P17.63 (the gate at /mnt/nvme10/shrec-shared/lassi with its shared_root), so
  that the sandbox hides the shared layer, and after a check that a ttkh program builds and runs on ttsim in the
  sandbox from the new root.

## After the window

Pull every pass, write results/ summaries from provenance.json, and report each model and arm per app; the other
models (Llama-3.3-70B, gpt-oss-120b) follow if the owner books time, as the owner said.
