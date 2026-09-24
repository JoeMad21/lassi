# Demo run: the LASSI pipeline on RNGD, compile-only

Provenance (provenance.json, from the job; the run tree's own manifest is under the runs root):
- run: rx job 20260924-083052-demo-rngd-4c8c, run directory demo-rngd-1 on alpha01;
- commit: 7c5cad7 (clean tree);
- recipe: projects/lassi-demo/rngd-compile.yaml, hash b7290b165f7e;
- date: 2026-09-24. The job ran from 15:30:52 to 16:26:21 UTC (provenance.json); the run tree's manifest records 15:30:55 as the run's start.
- exit status: the job log (output.log, pulled with the job) ends with `lassi_rc=0`, the status of `lassi run` itself; provenance.json's rc is the job's.
- toolchain pins: cuda 12.6.3 and nvhpc 24.11;
- device: none (compile only). No generated program ran.
- The model: furiosa-ai/Llama-3.1-8B-Instruct at revision v2026.2, served by furiosa-llm 2026.2.1 on RNGD npu1 with tp 8, with temperature 0.2, top_p 0.9, and max_tokens 4096. The server ran as rx job 20260924-082436-rngd-serve-9c91 (slot rngd-serve), started at 15:24:36 UTC with the watchdog wrapper of plans/spikes/p3-rngd-demo.md. Before it started, rx 20260924-082417-exec-3a7d showed npu1 at 0.00 GiB in furiosa-smi status. rx 20260924-082454-exec-b592 confirmed that /v1/models listed the model id before the first request.

This is a demo model, not an experimental arm (Decision Log, 2026-09-24). The recipe reproduces every upstream quirk the stages implement, with every fix off. It is not faithful, because it caps the correction loop at 5.

[MEASURED] results, 20 trials (10 apps x 2 directions, one run each):

| Direction | Reached a compiling translation | Compiled on the first try | Stopped at the correction cap |
| --- | --- | --- | --- |
| OpenMP to CUDA | 7 of 10 | 4 | 3 (colorwheel, jacobi, pathfinder) |
| CUDA to OpenMP | 7 of 10 | 1 | 3 (colorwheel, layout, pathfinder) |

- Corrections for the trials that compiled were 0, 0, 2, 2, 0, 0, 1 (OpenMP to CUDA) and 3, 1, 2, 1, 1, 1, 0 (CUDA to OpenMP).
- No attempt hit upstream's fence quirk; the parse-stage fence-quirk warning appears nowhere.
- The job log (output.log) lists every trial with its stage and correction count.

Limits:
- This is one sample of one demo model, not a reproduction of the paper's tables, which average several runs and models.
- Compile-only: a compiling translation is not a correct one, and nothing here says whether a translation computes the right output.
- Wall times include model calls and compiles on a shared host. They are not performance numbers, and none is reported here.
