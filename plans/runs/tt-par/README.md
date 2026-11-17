# Checking tt-par-v0's Programs On ttsim (Task P17.76)

tt-par-v0 (assets/bench/tt-par-v0.yaml; OQ-129) holds five development items from tt-metal's multi-core examples and
five held-out items written here. Every program was written on the workstation, which has no compiler, so none has
been built or run. This note gives the commands for steps (2) and (3) of the alpha01 batch OQ-130 asks the owner to
approve: what runs, where, and what it touches. Every step runs through tools/rx.py, uses no device and no model, and
writes on alpha01 only under the scratch root; the pulls write on the workstation.

## What Runs

1. The fetch (OQ-130 step (2)): tools/fetch_bench.py copies the development items' upstream kernels from the pinned
   tt-metal tree into $LASSI_SCRATCH/bench, each checked against its sha256 in the manifest. Seconds.
2. The checks (step (2)), two recipes in projects/lassi-demo/tt-par-check/, one per split, one trial per item. The
   baseline builds each item's C++ program with the pinned g++ and runs it on the host CPU, builds its TT reference
   with the pinned clang++-20 and runs it on ttsim, and records their agreement under the item's own tolerance
   (Trial.reference_agreement, the comparison with the C++ program); a trial whose programs disagree ends there, at
   baseline-disagree. Then the reference mock submits the TT reference as the candidate, which is built and run on
   ttsim a second time, and the oracle compares that run with the baseline's TT run: a repeatability check, not the
   C++ comparison. So each TT reference runs twice on ttsim. How long the job takes is PROJECTED with no run behind it:
   none of these programs has run on ttsim, so the job carries a timeout of three hours.
3. The leak test's built half (step (3)): the remote-marked test in tests/bench/test_tt_par_heldout.py reads the built
   context packs beside each assets/context/*/MANIFEST.yaml in the checkout and the retrieval indexes under
   $LASSI_SCRATCH/rag, and lists any it finds absent; it reads only and writes nothing. Minutes. Build any absent pack
   in the same checkout first (tools/build_tt_context.py; the packs' manifests name their build), so the check covers
   every pack.
4. The pulls: the job's provenance, and the two run trees into the workstation's .rx/pulls (raw run trees never go in
   results/); then results/tt-par-v0-ttsim/ gets the provenance and a summary.md written from each run's run.md and
   records.

A trial that ends at baseline-compile, baseline-run, or baseline-disagree names the program to fix. P17.76 fixes only a
host program, or a held-out item's kernels; a development item whose upstream kernels fail, and any item that cannot
pass, goes to the owner queue with its runs as evidence (OQ-129). A rerun after a fix takes a new --run-id, since the
runner refuses a run directory that exists.

## Commands

From a clean checkout of the commit that holds this note, on the workstation, each line one command (double quotes,
so they work from cmd.exe):

    uv run tools/rx.py doctor
    uv run tools/rx.py run -- "uv run tools/fetch_bench.py assets/bench/tt-par-v0.yaml"
    uv run tools/rx.py job start --big --name ttpar-check --timeout 10800 -- "uv run lassi --graphics off run projects/lassi-demo/tt-par-check/dev-cpp2tt.yaml --run-id ttpar-check-dev; uv run lassi --graphics off run projects/lassi-demo/tt-par-check/eval-cpp2tt.yaml --run-id ttpar-check-eval"
    uv run tools/rx.py job tail <job id> -n 80
    uv run tools/rx.py run -- "uv run pytest -q -m remote tests/bench/test_tt_par_heldout.py"

The job prints each run's directory, a run tree <runs root>/runs/<run id> under $LASSI_RUNS_ROOT. Pull the job's
provenance, and each run tree into .rx/pulls only (no --into), checking the paths against the job's output first.
The gate refuses a pull over 200 MB ("pull summaries, not raw trees"), and a run tree holds its builds; if it
refuses, pull each run's run.md, provenance.json, and trial records by their own paths instead:

    uv run tools/rx.py pull <job id> --into results/tt-par-v0-ttsim
    uv run tools/rx.py pull --path lassi-runs/runs/ttpar-check-dev
    uv run tools/rx.py pull --path lassi-runs/runs/ttpar-check-eval

## Reading The Results

- The comparison with the C++ program is each trial's reference_agreement, from the baseline; a trial that ends at
  baseline-disagree failed it. The oracle's numbers compare the TT reference's second ttsim run with its first.
- Every TT run here is on ttsim, which the provenance names as the device; its wall times are simulator wall time,
  never performance (Agent Rule 2).
- The results summary opens with the stages that ran and those that did not (no model, no correction, no silicon).
- The held-out items' results are evaluation results of their references, read for no prompt or feature work
  (Agent Rule 5).
