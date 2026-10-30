# Window Of 2026-10-27: The ttkernel Path With The P17.41 Fixes, And bale-v0 C++ <-> OpenMP

Provenance: provenance.json in this folder (written by `rx pull`), rx job 20261027-005611-window-7998 on alpha01
(Ubuntu 22.04.5 LTS), commit 7e42091403071489e61590076a55f66f3cd71a84, clean tree, 2026-10-27 00:56:11 to 03:12:21
(alpha01 clock, UTC-7), rc 0. Pins: tt-metal 5280a9cf built with Tracy (tt-metal@5280a9cf-tracy), ttsim v1.3.4, tt-mlir
0.9.0.dev20260221 (5f396fd6), Polygeist 77c04bb2, g++ 12.3.0. Model: furiosa-ai/gpt-oss-120b at v2026.4 on 4 RNGD cards.
Command (provenance.json, cmd): run-suite.sh over projects/lassi-demo/sweep9, mlir1, and bale1.

Devices: every TT program ran on ttsim v1.3.4, a simulator; every C++ and OpenMP program on alpha01's CPU. These are
pass counts, never performance. A pass reached S5 with alignment 1.0 on the held-out inputs. Tables: night/tables,
the job's own rows (night/calls.tsv), tabled by the window runner's report.py (kept outside the repository) into
csv/; night/tables/trials.tsv also carries earlier passes' rows, each with its own provenance.

| Set | Direction | Trials | Passed | Wilson 95% |
| --- | --- | --- | --- | --- |
| mlir1 ttkernel | cpp -> ttkernel | 15 | 0 | 0.0 to 0.204 |
| mlir1 ttkernel-affine | cpp -> ttkernel | 15 | 0 | 0.0 to 0.204 |
| bale1 cpp-to-omp | cpp -> omp | 9 | 9 | 0.701 to 1.0 |
| bale1 omp-to-cpp | omp -> cpp | 9 | 9 | 0.701 to 1.0 |

Every mlir1 trial ended at S1 at the correction cap (csv/by_end.csv), and every one asked generate once and
compile_loop eight times with no run_loop request (night/features-20261027-0056.tsv): no ttkernel attempt built. Every bale1 trial
passed on its first attempt. sweep9 ran nothing: night/calls.tsv has no call for it. The cause, from the job's own log, which
this folder does not hold and the commit message of 45135c4 records: the kernels-and-host examples' manifest
recorded a stale hash of assets/bench/tt-pairs-v0.yaml, so its six recipes were left out at the job's start
(fixed in 45135c4; `suite.py pull` now keeps that log).
