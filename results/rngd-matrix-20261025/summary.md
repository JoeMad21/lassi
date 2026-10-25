# The feature matrix in the owner's window of 2026-10-25 (task P17.23)

Provenance: provenance.json is the record of the last step, rx 20261025-113551-post-92fb (commit de5a1ab, clean;
alpha01, Ubuntu 22.04.5 LTS; 2026-10-25 11:35:51 on alpha01's clock, 14:35:51 EDT). That step read records only:
it ran plans/runs/window/count_features.py over each of the day's four sweep passes, tools/classify_build_errors.py
over the whole runs root, and copied the tables that matrix.sh's tabulate.py wrote at the end of the last job. It
started no model and built or ran no program. The trials themselves came from four big jobs, each from a clean
commit, in slots window, window, window3, and window4 (plans/runs/2026-10-25-window.md, What Ran):

| Pass | Job | Commit | What ran |
| --- | --- | --- | --- |
| 20261025-0503 | rx 20261025-050036-window-0350 | 3b41733 | Qwen3-Coder-30B, 18 recipes, cut after about 18 minutes when the job was stopped to fix the GPT-OSS download |
| 20261025-0527 | rx 20261025-052309-window2-6ecd | 955745f | gpt-oss-20b at the server's default reasoning effort, about 41 minutes, ended when alpha01 went down (OQ-085) |
| 20261025-0824 | rx 20261025-082401-window3-b393 | 955745f | gpt-oss-20b at the default effort (every recipe ended on a reply with no answer), then Qwen3-Coder-30B (complete), then EXAONE-4.0-32B, cut after 51 minutes |
| 20261025-1047 | rx 20261025-104732-window4-5821 | de5a1ab | gpt-oss-20b at reasoning effort low (P17.28), 47 minutes |

- Models: furiosa-ai/gpt-oss-20b at v2026.4 on 1 RNGD card (npu0); furiosa-ai/Qwen3-Coder-30B-A3B-Instruct-FP8 and
  furiosa-ai/EXAONE-4.0-32B-FP8 at v2026.4 on 4 RNGD cards (npu0 to npu3); each served at 127.0.0.1:8123 from the
  venv furiosa-venv-2026.4 (last version read: rx 20261020-101803-exec-1a45, furiosa-llm 2026.4.0; not read again).
  gpt-oss-20b ran at the server's default reasoning effort in passes 0527 and 0824 and at low (llm.sampling.
  reasoning_effort, P17.28) in pass 1047.
- Devices: every TT program ran on ttsim v1.3.4 (a simulator) with tt-metal 5280a9cf, every C++ program on the host
  CPU; no Wormhole card. Wall times and durations are not performance.
- Recipes: projects/lassi-demo/sweep6, cpp -> tt Tier A, the correction cap 15, the fuzzer adversary in every recipe
  (it runs after the oracle and never feeds the loop). Each trial is one sample. Per-run provenance (commit, clean
  state, device, pins, date) is in build-errors.txt's runs section; every one of the day's runs reads clean.

## Files

- trials.tsv and passes.tsv: tabulate.py's tables over every sweep pass in the runs root (250 trials from 22
  pass-model pairs); the day's passes are the rows whose pass begins 20261025.
- features-<pass>.tsv: per trial of each of the day's passes, its outcome, its requests per stage name (restart
  among them), and how many requests carried a looked-up note (P17.26) or a retrieved note (P17.22).
- build-errors.txt: the classifier's report over the whole runs root after the window, the day's passes included
  (results/build-errors-20261025 is the same tool before the window). build-errors.stderr is empty.

## Counts [MEASURED]

Counts of the day's trial records (trials.tsv and features-<pass>.tsv; exploratory, one sample each):

- 82 trials finished: 79 ended at the correction cap at S1 (a build error), one at S4, and two at S5.
- The two at S5, both at alignment 1.0:
  - gpt-oss-20b at the default effort, combination mmrag (retrieval and the mismatch report), eltwise_binary, after
    3 corrections; the fuzzer's 64 runs found no divergence (pass 20261025-0527).
  - EXAONE-4.0-32B, combination mmbase (the mismatch report), matmul_single_core, at correction 0; the fuzzer's 64
    runs found 2 divergences: binary_io failed the candidate on 2 of the 64 fuzzed input sets, on ttsim, after it
    passed the held-out input (pass 20261025-0824).
- The one at S4: Qwen3-Coder-30B, combination mmapiragrl (every feature, restarts and declaration lookup
  included), loopback, after 8 corrections, ended on a simulator gap (sim-gap).
- Qwen3-Coder-30B, pass 20261025-0824: 45 trials, 5 per combination; 44 at the cap at S1 and the S4 above.
- EXAONE-4.0-32B, pass 20261025-0824: 23 trials in 51 minutes; 22 at the cap and the S5 above.
- gpt-oss-20b at the default effort, pass 20261025-0824: no trial finished; every one of its 18 recipes ended on a
  ServingError, an HTTP 200 reply with a null content (plans/runs/2026-10-25-window.md).
- gpt-oss-20b at low effort, pass 20261025-1047: 3 trials finished (base and mmapi on matmul_multi_core, mmrag on
  eltwise_binary), all at the cap at S1; none of its 18 recipes ended on a ServingError, and its server answered
  261 requests, all status 200 (rx 20261025-113728-exec-a71b). Since P17.28 a null content cut at length is the
  empty answer, so that alone does not show answers; build-errors.txt's buckets for these three runs do: compile
  errors in the files the replies returned.
- The new features fired where set: in mmapiragrl, Qwen3-Coder-30B's 5 trials had 13 restarts and 34 requests that
  carried declarations, and EXAONE-4.0-32B's 2 had 6 restarts and 23 such requests (pass 20261025-0824); the
  first pass's one mmapiragrl trial had 3 and 10.

## Reading (not measured)

One sample per item and combination, and passes cut short, so no comparison between combinations is drawn here:
the two S5 trials came in combinations without the API reference, restarts, or lookup, and the furthest Qwen3-Coder
trial came in the combination with all of them; none of that separates a feature from chance. At the server's
default effort, every gpt-oss-20b recipe of pass 0824 ended on a null-content reply, while pass 0527 gave 72
replies with status 200 and its one finished trial passed; whether the 16384-token budget cut the null replies is
not recorded. At low effort gpt-oss-20b answered, but its three finished trials did not build within 15
corrections. The tt-pairs-v0 items are unassigned (Agent Rule 5); Restarts
(P17.25) and declaration lookup (P17.26) were proposed from these items' outcomes, and OQ-083 asks how results like
these should be labeled.
