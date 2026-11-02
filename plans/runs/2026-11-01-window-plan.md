# Windows of 2026-11-01 and 2026-11-02: plan

The owner reserved the Tenstorrent and RNGD cards on 2026-11-01 from 18:00 to 21:00 EST and on 2026-11-02 from 03:00 to
09:00 EST ("Keep working. Finish any interesting experiments on the 120B model so that we can move to promising 70B
models."). Testing in the second window ends at 08:30 (changed at 23:15 on 2026-11-01; see below). Every run below is
a development result on tt-pairs-v0's unassigned items (OQ-083 (b)). Each run compiles,
runs on ttsim (a simulator: nothing measures performance), and checks outputs against the item's declared tolerance;
none runs on silicon (the IOMMU read pass-through at its last reading, 2026-10-22, OQ-078, and Agent Rule 9 stands as written).

## 2026-11-01, 18:00 to 21:00: gpt-oss-120b, then TurboQuant

Changed at 17:50 (the owner: "I have some TurboQuant code for you to check and run. I want it to happen during the 6pm to 9pm window after the currently planned sweep. Give me a reminder to launch that once the sweep ends or there is one hour remaining."): every sweep17 pass ends by 20:00, and the owner's TurboQuant code runs from 20:00 (or from the sweep's end, if earlier) to 21:00, after a check. The deadlines below that say 20:50 and 20:45 are superseded by 20:00.

- sweep17 (tasks P17.48, P17.49): ttir-min (the bare loop with the TTIR API reference), ttir-full (plus retrieval and
  the cgeist affine source), d2m-min. The first pass's deadline is 20:50, when run.sh stops the recipes and the server; the rx timeout (the deadline plus 900 s) is only a backstop at 21:05. Later passes use 20:45, so their backstop is 21:00.
- Early finish (the owner: "plan around the sweep ending earlier than anticipated"): a pass that ends with 31 minutes
  or more left is followed at once by another pass (the launcher refuses a deadline under 30 minutes away,
  tools/suite.py CALL_MIN_S) of the same recipes from the same clean commit, as many times as
  the window allows; passes are pooled in the report. A sweep directory named twice in one launch is run once, so the
  passes are separate launches.
- Watched within minutes of each launch; a harness fault stops, fixes, and relaunches (plans/LESSONS.md).

## Between the windows: gpt-oss-120b out, Llama-3.3-70B in

- After 21:00 the owner deletes gpt-oss-120b's weights (a read-only dry run, rx 20261101-144559-exec-e52d, selected 25 blobs of the shared hub/blobs store
  that no other model's snapshot links to, 198,607,837,395 bytes; the model table's 198,607,992,266 bytes also count the
  small files kept in the model's own folder). Agents never delete files on alpha01.
- Then one download job fetches furiosa-ai/Llama-3.3-70B-Instruct-FP8-dynamic at v2026.4 (73,775,947,887 bytes, plans/runs/suite/models.tsv;
  not gated, the Hub's API for that revision read 2026-11-01, sha c3dddd674db6e22cc6a91a9a7e1336bf67a7a8ff) into $LASSI_SCRATCH/.cache/huggingface, the cache run.sh serves from, and the snapshot is confirmed before
  03:00 (the owner: "Plan to download the Llama model once we delete GPT OSS").

## 2026-11-02, 03:00 to 08:30: Llama-3.3-70B

Changed at 23:15 on 2026-11-01 (the owner: "A quick note, we must suspend testing 30 minutes earlier than I reported."):
testing in this window ends at 08:30, not 09:00, so sweep18's deadline is 08:15, which with the suite's 900 s backstop
ends every call by 08:30.

- sweep18 (task P17.61): the TTIR ladder on ttsim, l0 (the bare loop) to l3 (plus the API reference, retrieval, and
  the cgeist affine source), and d2m-min, with Llama's own sampling (temperature 0.6, top_p 0.9, from
  https://huggingface.co/furiosa-ai/Llama-3.3-70B-Instruct-FP8-dynamic/raw/v2026.4/generation_config.json, read
  2026-11-01) and no reasoning setting (the owner: "consider minimum features tests for 70B. May be the case that the model needs more features to
  converge."). l1, l3, and d2m-min match sweep19's gpt-oss-120b arms (ttir-min-gap, ttir-full-gap, d2m-min-gap) but
  for the model, and every recipe feeds simulator gaps back as corrections, as sweep19 does. Deadline 08:15; passes
  chained as above.
- TurboQuant testing (the owner, 2026-11-01: "Set a reminder to also do TurboQuant testing during one of the windows"):
  a candidate for the end of this window or a later one. TurboQuant is P18, deferred and outside the work order
  (OQ-066), so what "testing" runs needs the owner's scope first (to be asked in the attended session); plans/spikes/p5-tier-b.md and results/p5-tier-b
  hold P5.3's evidence.
