# Tier A on RNGD, one trial per item and direction (2026-10-20 window)

The owner's 2026-10-20 hardware window, in the working session: OQ-045's real-model Tier A run cut to one trial per item and direction, with the correction cap at 5. Exploratory, as the recipes label themselves: one sample of one demo model. No translation compiled, so no generated program ran; only the reference programs ran, the TT ones on ttsim, the Tenstorrent simulator, not on silicon. No wall time here is performance.

Provenance (maxabs/provenance.json, pcc/provenance.json, and serve/provenance.json, from the jobs; each run tree's own manifest is under the runs root on alpha01):
- runs: rx job 20261020-091247-tiera1-maxabs-1b74 (run directory 20261020-tiera1-maxabs) and rx job 20261020-091252-tiera1-pcc-049e (run directory 20261020-tiera1-pcc);
- commit: fa3a979 (clean tree), branch p16-adversarial;
- recipes: projects/lassi-demo/tier-a-rngd-1trial-maxabs.yaml, hash 4fd4650c6829, and projects/lassi-demo/tier-a-rngd-1trial-pcc.yaml, hash 3197dddd88e2;
- date: 2026-10-20. The jobs ran from 16:12:47 to 16:43:18 UTC and from 16:12:52 to 16:44:54 UTC. Each job log records `lassi_rc=0`, the status of `lassi run` itself; provenance.json's rc 0 is the job's;
- toolchain pins: gcc 12.3.0 (host /usr/bin/g++-12) for the C++ targets; tt-metal 5280a9cf (host build with clang++-20) and ttsim v1.3.4 for the TT targets;
- device: ttsim v1.3.4 on tt-metal 5280a9cf for TT programs (a simulator), and the host CPU for C++ programs;
- the model: furiosa-ai/Llama-3.1-8B-Instruct at revision v2026.2, with temperature 0.2 and max_tokens 4096, served on RNGD npu1 by rx job 20261020-091143-rngd-serve-440e (slot rngd-serve) with the watchdog wrapper of plans/spikes/p3-rngd-demo.md. That job's log (pulled with it) shows npu1 at 0.00/47.50 GiB in furiosa-smi status before the server started. rx 20261020-101803-exec-1a45 read furiosa-llm 2026.2.1 in the serving venv and firmware 2026.4.0 on npu1; run.md records no driver or server version.

This is a demo model, not an experimental arm (Decision Log, 2026-09-24). The resolved recipe has all six fixes on (baseline_both, execution_gate, fence_tag, parsed_diagnostics, prompt_newlines, prompt_spaces), so it is not faithful, and it caps the correction loop at 5 (upstream has no cap).

[MEASURED] results, 10 trials (5 items x 2 directions, one run each):

| Direction | Items | Reached a compiling translation | Stopped at the correction cap |
| --- | --- | --- | --- |
| C++ to TT | eltwise_binary, eltwise_sfpu, loopback, matmul_multi_core, matmul_single_core | 0 of 5 | 5 |
| TT to C++ | the same five | 0 of 5 | 5 |

- Every trial ended at stage S0 after 6 attempts (the first translation and 5 corrections), with end reason correction-cap.
- The reference check passed in all 10 trials: the C++ and TT reference programs agree within the oracle's threshold on each item's inputs.
- The first attempt:
  - failed to compile in 6 trials, with 17 to 31 compiler errors each. Examples: "conflicting types for 'main'" and "variable type 'MeshCommandQueue' is an abstract class" (C++ to TT); "tt-metalium/base_types.hpp: No such file or directory" and "tt-metalium/constants.hpp: No such file or directory", TT headers kept in the C++ target (TT to C++);
  - broke the reply format in 4 trials, before any compile: in C++ to TT matmul_multi_core it returned the file in two FILE blocks (the diagnostic is recorded under the compile stage); in TT to C++ loopback, matmul_multi_core, and matmul_single_core the reply carried no FILE block for the expected file.
- Every correction attempt (5 per trial, 50 in all) broke the reply format, before any compile: the reply did not carry the file in a fenced block whose first line is `// FILE: <name>.cpp`. In C++ to TT eltwise_binary, attempt 1 begins with that FILE line but has no opening fence, and attempt 2 puts a FILE line before the fence and names a kernel file instead.
- The job logs (output.log, pulled with each job) list every trial with its stage and correction count. No scores or per-trial CSVs were produced, since no translation compiled.

Reading:
- The FILE-block format is the fence_tag fix's parse, which is on in this recipe; with fence_tag off, the stages read the first fenced block as upstream does. The format failures are failures against that fix's format, not against upstream's.
- The 8B model loses that format once it is asked to correct, so no correction reached a compile; 4 of the 10 trials never reached a first compile either.

Limits:
- One trial per item, one demo model, five items per direction: not a rate, and not comparable with the paper's tables.
- No translation compiled, so nothing here says anything about TT correctness or performance; ttsim is a simulator in any case.
- Wall times include model calls and compiles on a shared host. They are not performance numbers, and none is reported here.
