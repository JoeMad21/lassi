# Window of 2026-11-17: plan

The owner, at about 21:50 EST on 2026-11-17: "I've booked you six hours to run amok starting at 10:30pm EST. Do what
you need to do starting then. Launch the work yourself, try not to include me. Try to run the various models in that
window." That is the owner's yes to OQ-130's batch and to model runs in the window (Agent Rule 17), given in the
working session.

Window: 22:30 on 2026-11-17 to 04:30 on 2026-11-18 EST, which is 19:30 to 01:30 on alpha01's clock (UTC-8). Every
model run is a development result on the development split of tt-par-v0 and tt-par-v0-kh; every TT program runs
on ttsim v1.3.4, a simulator (nothing measures performance, Agent Rule 2). No program runs on the Wormhole cards:
silicon runs are the owner's to start (Agent Rule 9, OQ-127). The RNGD cards only serve the models.

## What Runs, In Order

1. `rx doctor`: free space on the scratch disk, the rngd device class, and no running job; a read-only listing of
   the Hugging Face cache, which says which of the six models can be served without a download.
2. OQ-130 (1), P17.64: sweep21's pull and summaries.
3. OQ-130 (2), P17.76: `tools/fetch_bench.py assets/bench/tt-par-v0.yaml` and
   `tools/fetch_bench.py assets/bench/tt-par-v0-kh.yaml` (the sweep's host-and-kernels recipes read the second,
   and run-suite.sh fetches neither), then the two check recipes of
   projects/lassi-demo/tt-par-check/ as one job (plans/runs/tt-par/README.md). It is one sequential process, so it
   runs as an ordinary job in its own slot beside the sweep, the window's one big job.
4. P17.71's compile of tt-host-skeleton-v1's host.cpp, which tt-kh-skeleton-v1 holds byte for byte
   (`bash plans/runs/night/skeleton_build.sh assets/skeletons/tt-host-skeleton-v1`), before the sweep starts, so
   the sk1 and p2sk1 rungs start from a skeleton that compiles at the pin; their results are read together with
   that compile and P17.71's one-tile run.
5. sweep22 (P17.74, its first five rungs): plans/runs/suite/run-passes.sh over projects/lassi-demo/sweep22 and the six
   30B-class models of plans/runs/suite/models-30b.tsv, from a clean worktree of the pushed commit, with the deadline
   04:15 EST, as one big rx job. Each model is served on 4 RNGD cards in turn, and its ten recipes run side by side:
   five rungs (bare, sk0, p2sk0, sk1, p2sk1) in two variants (host only, and host and kernels), each on
   vecadd_multi_core, matmul_multicore_reuse, and pad_multi_core, one trial each. How long a model takes is PROJECTED:
   no tt-par trial has run.
6. While the sweep runs: OQ-130 (3), the leak test's built half; (4), P17.70's ttsim spike; (5), P17.71's one-tile
   ttsim run. Each is a short `rx run`.
7. Before 04:30: the pulls (provenance into results/, run trees into .rx/pulls only).

## What Does Not Run

- The correction rungs (P17.73 is not built) and the held-out run (its recipes are chosen from these development
  results and committed before it runs; P17.74).
- gpt-oss-120b: its weights were removed from alpha01 after the window of 2026-11-01, and its download (198.6 GB)
  exceeds the scratch disk's free space (65.0 GB after the window of 2026-11-16, results/tracy-20261116; the
  download sizes are plans/runs/suite/README.md's, from the night of 2026-10-25). Llama-3.3-70B has no sweep22
  recipes: six models already share the sweep's time, and its download, if its snapshot is gone, would not fit
  either (73.8 GB).
- Any Wormhole card run (P17.75 is the owner's to start).

## After The Window

results/ summaries from each provenance.json, the run report, and the held-out recipes chosen from the development
results, committed before any held-out run.
