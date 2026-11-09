# Window of 2026-11-09: plan

The owner, at about 00:25 EST on 2026-11-09: "Starting at 5am, you will have 5 hours to conduct LASSI testing with a 30B
model to compare to the 120B and 70B work. I would like to see the EXAONE models as well as any other 30B models we
discussed be tested. Get ready for that and automate testing. Give me commands to delete any model weights we don't
need."

Window: 05:00 to 10:00 EST, which is 02:00 to 07:00 on alpha01's clock (UTC-8). The sweep's deadline is 09:45 (epoch
1794235500), so the suite's 900 s backstop ends every call by 10:00. Every run is a development result on
tt-pairs-v0's unassigned items (OQ-083 (b)): each compiles, runs on ttsim v1.3.4 (a simulator: nothing measures
performance), and checks outputs against the item's declared tolerance. No program runs on silicon (OQ-078; Agent
Rule 9 stands as written); the RNGD cards only serve the model.

## What runs

- sweep20 (task P17.61): sweep18's five arms (ttir-l0 to ttir-l3, the TTIR ladder, and d2m-min) for each 30B-class
  model, every setting sweep18's but the model and its sampling. Each arm compares with Llama-3.3-70B's sweep18 arm
  (2026-11-02, 10 passes) and, for l1, l3, and d2m-min, with gpt-oss-120b's sweep19 arm (2026-11-01, rx
  20261101-152009-window10-eb9b and 20261101-160743-window11-a936, the ids rx job start printed at their launch on
  2026-11-01), which match sweep18's but for the model.
- The models, in serving order (plans/runs/suite/models-30b.tsv; each a furiosa-ai build at v2026.4 on 4 RNGD cards
  from furiosa-venv-2026.4 with no --reasoning-parser; sizes from the Hub's tree API, read 2026-11-09):

| Slug | Model | GB | temperature, top_p | max_tokens |
| --- | --- | --- | --- | --- |
| qwen3-coder-30b | Qwen3-Coder-30B-A3B-Instruct-FP8 | 33.3 | 0.7, 0.8 | 8192 |
| qwen3-30b-instruct | Qwen3-30B-A3B-Instruct-2507-FP8 | 33.3 | 0.7, 0.8 | 8192 |
| qwen3-32b | Qwen3-32B-FP8 | 36.0 | 0.6, 0.95 | 16384 (thinks by default) |
| gemma4-31b | gemma-4-31B-it-FP8-block | 35.0 | 1.0, 0.95 | 16384 |
| exaone4-32b | EXAONE-4.0-32B-FP8 | 35.0 | 0.2, 0.9 (none in its generation_config; the card advises under 0.6 without reasoning) | 8192 |
| exaone45-33b | EXAONE-4.5-33B-FP8 | 43.9 | 1.0, 0.95 | 16384 |

  Sampling is each model's https://huggingface.co/<model>/raw/v2026.4/generation_config.json, read 2026-11-09, except
  EXAONE-4.0's, which sets none. Sampling has no top_k or penalty field, so Qwen3's and gemma's top_k, Qwen3-Coder's
  repetition_penalty 1.05, and EXAONE-4.5's presence_penalty 1.5 are not sent. EXAONE-3.5-32B-Instruct is left out:
  furiosa-ai publishes it only at v2025.3.0 tags, which furiosa-llm 2026.4 does not load.
- Why the EXAONE models go last: in pass 1 every call gets an equal share of the time left, so a later model never
  gets less than an earlier one, and time an earlier model does not use passes on. A later pass gives each call an
  equal share again, and run.sh serves no model with under 30 minutes, so only the tail of the order runs in it. The
  EXAONE models, the owner's first interest, are at the tail and get any extra trials.
- One rx job, started before the window on the owner's go-ahead, runs plans/runs/suite/run-passes.sh with --start
  05:00 (1794218400) and --deadline 09:45. Until 05:00 it only waits on alpha01, holding no card, server, or
  download. Pass 1 then downloads the models in serving order (the first, 33.3 GB, before its call, the rest in
  the background while earlier models are served), builds the TT assets, and serves each model in turn with an equal
  share of the time left. A pass that ends with 31 minutes or more left is followed by another.

## Before the window: the owner deletes unneeded weights

The six models need 216.4 GB, and each download runs only while 65 GB stay free after its missing bytes. The cache
last held Llama-3.3-70B-Instruct-FP8-dynamic (73.8 GB, sweep18's model, no longer needed) and the two small Qwen3
bases of the training recipes (the read-only reads before gpt-oss-120b's deletion, among them rx
20261101-204121-exec-16c0; not read since). The owner runs a list-then-delete script outside the repository
(agents never delete files on alpha01); it keeps the six models and the two Qwen3 bases, removes every other model
folder and the shared-store blobs only those folders link to, and refuses while a server, sweep, or download runs.

## After the window

Pull every pass and sweep19's two jobs, write results/ summaries from provenance.json, and compare arm for arm with
sweep18 and sweep19; then the run report. The bible asks for a Decision Log entry before a vendor-named model joins
as an experimental arm; EXAONE-4.0-32B, EXAONE-4.5-33B, gemma-4-31B, and Qwen3-30B-A3B-Instruct-2507 have none yet,
so that entry goes through the bible-sync procedure with the results (the owner directed these runs in the session).
