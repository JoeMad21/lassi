# sweep18 Pass 4 Of The Window Of 2026-11-02 (Pass 20261102-0130)

The window's summary, with every stage run and skipped and the ten passes pooled: results/window-20261102-sweep18-pass1/summary.md.

Provenance: results/window-20261102-sweep18-pass1/provenance.json, rx job 20261101-214439-suite18-7433 on alpha01, commit b47bd6080945237dac143bf305a8cd1d0a8e48f6, clean
tree, rc 0; this pass ran as 20261101-214439-suite18-7433-p4 (output.log in results/window-20261102-sweep18-pass1). This folder holds that pass's suite output,
lassi-runs/suite/20261101-214439-suite18-7433-p4, as tools/suite.py copies it (plain-ASCII files only).

Stages run: baseline (both references: the tt reference on ttsim, the C++ reference with g++ 12.3.0 on alpha01's CPU),
verbatim_context (every recipe but ttir-l0), generate, compile (tt-mlir's tools for the recorded ttsim descriptor), run
on ttsim v1.3.4 (a simulator, never performance), and the output check against the item's declared tolerance. Not
run: silicon. Model: furiosa-ai/Llama-3.3-70B-Instruct-FP8-dynamic at v2026.4 on 4 RNGD cards. Development results
on tt-pairs-v0's unassigned items; a pass reached S5 with alignment 1.0, and "first attempt" means with 0
corrections (night/passes/20261102-0130/summary.tsv).

| Recipe | Trials | Passed | First attempt | Ended at a simulator gap |
| --- | --- | --- | --- | --- |
| d2m-min | 8 | 0 | 0 | 0 |
| ttir-l0 | 6 | 0 | 0 | 0 |
| ttir-l1 | 6 | 0 | 0 | 0 |
| ttir-l2 | 6 | 5 | 4 | 1 |
| ttir-l3 | 6 | 5 | 5 | 1 |
