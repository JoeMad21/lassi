# The night window of 2026-10-25: cpp -> tt on Llama-3.3-70B and gpt-oss-120b (task P17.32)

Provenance: provenance.json is the night job's record, rx 20261025-170035-night-395c (a big job; commit
e43a1e060b53d034f6d0afc1b28bbb1f012a50e2, dirty false; alpha01, Ubuntu 22.04.5 LTS; 2026-10-25 17:00:35 to
21:12:13 on alpha01's clock, 20:00:35 to 00:12:13 EDT; rc 0; scratch free 345.9 GB before, 65.8 GB after). Values
cited from the job's output.log come from the log of rx 20261025-170035-night-395c. night/ is the job's output
directory, lassi-runs/night/20261025-1700, pulled with `rx pull --path`, without two of its files. The pull held 17
files and 439,018 bytes (rx pull counted the directory itself as an 18th entry). The two download logs,
fetch-gpt-oss-120b.log and fetch-Llama-3.3-70B-Instruct-FP8-dynamic.log, are left out because their progress bars
are not plain ASCII; each download's outcome stays in its .done marker and in the job log's "download ended rc=0"
line. What is committed under night/ is 15 files, 433,541 bytes: the downloads' .done markers, the asset builds'
logs, each model's sweep log, the per-trial feature counts, the classifier's report, and the tables (trials.tsv,
passes.tsv) that plans/runs/rngd-sweep/tabulate.py wrote over the whole runs root at the end (381 trials from 24
pass-model pairs; the night's are the passes 20261025-1712 and 20261025-1909).

- Models: furiosa-ai/Llama-3.3-70B-Instruct-FP8-dynamic and furiosa-ai/gpt-oss-120b at v2026.4, each on 4 RNGD
  cards (npu0 to npu3), served at 127.0.0.1:8123 from furiosa-venv-2026.4 (last version read: rx
  20261020-101803-exec-1a45, furiosa-llm 2026.4.0; not read again), one after the other (OQ-087). Each listed
  max_model_len 131072 (llama33-70b.models.json and gpt-oss-120b.models.json under lassi-runs/rngd-sweep/<pass>/):
  Llama's read in rx 20261025-174517-exec-b886, gpt-oss-120b's in rx 20261025-201551-exec-1bd2. gpt-oss-120b ran
  at reasoning effort medium with a 32768-token budget, and its lowall combination at low (OQ-080).
- Device: every TT program ran on ttsim v1.3.4 (a simulator) with tt-metal 5280a9cf; every C++ reference on the
  host CPU. No Wormhole card was used. Wall times are not performance (Agent Rule 2).
- Recipes: projects/lassi-demo/sweep7 (cpp -> tt only, 3 samples per item, correction cap 8, the fuzzer adversary
  in every recipe; OQ-086), combinations base, ex (worked examples), lk (declarations), all (both, retrieval, the
  mismatch report, restarts after 3), and for gpt-oss-120b lowall (all at low effort).
- Labels: tt-pairs-v0 is a development set (OQ-083 (b)); every number below is a development result on its five
  items, exploratory in the sense that the features were chosen from these items' earlier outcomes, never an
  evaluation result.

## Counts [MEASURED, ttsim v1.3.4 on alpha01, 2026-10-25]

From night/trials.tsv and night/passes.tsv (a pass is S5 at alignment 1.0: the program built, ran clean on ttsim,
and every output matched the C++ reference on the held-out inputs within the item's tolerance). Intervals are
Wilson 95% (lassi.analysis.stats wilson_interval).

| Model | Combination | Trials | Reached S5 | Passed |
| --- | --- | ---: | ---: | ---: |
| Llama-3.3-70B | base | 15 | 0 | 0 |
| Llama-3.3-70B | ex | 15 | 3 | 2 |
| Llama-3.3-70B | lk | 15 | 0 | 0 |
| Llama-3.3-70B | all | 15 | 2 | 0 |
| gpt-oss-120b | base | 15 | 6 | 6 |
| gpt-oss-120b | ex | 15 | 3 | 3 |
| gpt-oss-120b | lk | 15 | 2 | 0 |
| gpt-oss-120b | all | 15 | 9 | 7 |
| gpt-oss-120b | lowall | 11 | 0 | 0 |

- Llama-3.3-70B: 2 of 60 passed (3%; 1% to 11%), both loopback; 52 ended at S1, 3 at S4, 5 at S5.
- gpt-oss-120b: 16 of 71 passed (23%; 14% to 34%); 32 ended at S1, 19 at S4, 20 at S5. By item: loopback 9 of 12,
  eltwise_binary 3 of 15, eltwise_sfpu 3 of 14, matmul_multi_core 1 of 15, matmul_single_core 0 of 15. By
  combination: all 7 of 15 (47%; 25% to 70%), base 6 of 15 (40%; 20% to 64%), ex 3 of 15 (20%; 7% to 45%), lk 0 of
  15, lowall 0 of 11 (0% to 26%).
- The fuzzer adversary found no divergence in any passing trial (passes.tsv, with_divergence 0 in each of the
  night's nine rows).
- lowall-maxabs stopped with rc 2 after 5 of its 9 trials (the lowall combination: 11 of its 15). Its eltwise_sfpu
  run03 was cut off at attempt 8, when the sandbox's teardown failed (`umount: ... target is busy`, then
  `lassi-sandbox: the setup stopped after the program ended`), and its three loopback trials never started. The
  attempt and the umount messages are from its log, lassi-runs/rngd-sweep/20261025-1909/gpt-oss-120b--lowall-maxabs.log,
  read in rx 20261025-225039-exec-31b7. A harness failure, not a model outcome.

## Reading (not measured)

Three samples per item and combination are few, and the intervals overlap, so no combination is shown better than
another. With that said: gpt-oss-120b at medium effort passed 16 trials over four of the five items, where the
day's earlier C++ -> TT matrix passed 2 of 82 (results/rngd-matrix-20261025/summary.md: gpt-oss-20b on
eltwise_binary, and EXAONE-4.0-32B on matmul_single_core with 2 fuzzer divergences); the same combination at low
effort passed nothing; and 13 of gpt-oss-120b's 29 failed matmul trials ended after the program built (9 at S4, 4
at S5), against 6 of Llama-3.3-70B's 58 failed trials over all five items (3 at S4, 3 at S5; on the matmuls 2 of
its 24, both at S4); the other 16 and 52 ended at the build (S1). The next sweep (sweep8, task
P17.34) tests the host skeleton, tt-host-v1, and corrections on wrong outputs and simulator gaps on the same models.
