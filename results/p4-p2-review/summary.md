# P4.15: the P2 review questions, scoring pass and tiktoken check

[MEASURED] provenance.json (rx run 20260927-204249-desktop-8r113ei-detached-f0c4a852-bcbb):

- commit: f0c4a852 (f0c4a8523559...), clean tree (dirty false, snapshot_of null; the slot's `git status --short | wc -l` printed 0), run from the detached clean worktree;
- host: alpha01, Ubuntu 22.04.5 LTS; start 2026-09-27T20:42:49-07:00, end 20:43:05; rc 0; scratch free 582.0 GB before, 581.9 GB after;
- pins: tiktoken 0.14.0 (pyproject.toml, uv.lock); python 3.10.12;
- device: none for this run, which only scored and compared text. The scored run's programs ran on alpha01's host CPU (the source run's executor is native; its records predate P4.5, so review.md shows its device as -).

Command (provenance.json, cmd): with TIKTOKEN_CACHE_DIR=$LASSI_SCRATCH/.cache/tiktoken and LASSI_GRAPHICS=off, cache cl100k_base with tiktoken's own loader, then `tools/fetch_upstream.py`, the remote tiktoken tests (`LASSI_REQUIRE_TIKTOKEN=1 uv run pytest -q -p no:cacheprovider -m remote tests/scoring/test_similarity_tiktoken_remote.py`), `lassi score` of demo-rngd-cpu-1 with df-v0 and lassi into score id p4-p2-review, and plans/spikes/p4-p2-review/q6_similarity.py with --tiktoken. The output printed "cache exit=0 lassi score exit=0 tiktoken tests exit=0".

## Results

- The encoding: `cl100k_base 100277`, cached as one file of 1681126 bytes, sha256 223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7, under $LASSI_SCRATCH/.cache/tiktoken.
- The remote tiktoken tests: 7 passed (the profile's sim_t_tiktoken against the notebook's own function).
- The score tree is score/ (rx pull of lassi-runs/scores/p4-p2-review): metrics.md, review.md, provenance.json, and parquet/.

The source run, demo-rngd-cpu-1 (rx job 20260924-093127-demo-rngd-cpu-1ff6, clean 2f052e6), has these limits: a demo model (furiosa-ai/Llama-3.1-8B-Instruct served on RNGD), not faithful, a correction cap of 5, the -mp=multicore CPU proxy, CUDA to OpenMP only, n = 1 per scenario. The paper columns in metrics.md stand beside a run that is not faithful and are not compared with it.

Per trial (score/review.md, lassi profile; values rounded to four decimals here):

| Item | sim_t | sim_t_c | sim_t_tiktoken | correct |
| --- | --- | --- | --- | --- |
| atomicCost | 0.0012 | 0.6728 | 0.5331 | 0.0 |
| bsearch | 0.0024 | 0.7201 | 0.6042 | 0.0 |
| colorwheel | 0.0034 | 0.8546 | 0.7098 | 0.0 |
| dense-embedding | 0.0014 | 0.6544 | 0.5502 | 0.0 |
| entropy | 0.0026 | 0.5487 | 0.3337 | 0.0 |
| jacobi | 0.0021 | 0.4244 | 0.3686 | 0.0 |
| layout | 0.0033 | 0.7680 | 0.4762 | 0.0 |
| matrix-rotate | 0.0069 | 0.6165 | 0.4121 | 0.0 |
| pathfinder | 0.0006 | 0.5443 | 0.3903 | 0.0 |
| randomAccess | 0.0059 | 0.8465 | 0.5878 | 0.0 |

Question 6, from q6_similarity.py in the same run (python 3.10.12; tiktoken 0.14.0; 20 references; ranges over the pairs):

- a perfect copy extracted as the notebook extracts it, fence tag cpp, c++, c, or none (a leading line feed in 20/20): sim_t 0.0004 to 0.0240; sim_l 0.9901 to 0.9965; sim_t_c 1.0000; sim_t_tiktoken 0.9994 to 0.9998;
- a perfect copy, fence tag cuda (no leading line feed): sim_t 0.0004 to 0.0239;
- a blank line added at the top: sim_t 0.0004 to 0.0240; in the middle: sim_t 0.4236 to 0.5561;
- the other-language original, unfenced: sim_t 0.0117 to 0.2686, sim_t_tiktoken 0.3892 to 0.8531;
- demo-rngd-cpu-1's last target files (a leading line feed in 10/10): sim_t 0.0006 to 0.0069, sim_t_tiktoken 0.3337 to 0.7098; with the leading line feed removed: sim_t 0.0006 to 0.0333.

These match the workstation values the spike's question 6 closure used (a perfect copy at most 0.0240 under sim_t), so the closure and the bible text stand unamended. Which measure the paper's Sim-T column is stays open (OQ-031).
