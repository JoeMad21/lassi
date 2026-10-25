# Spike P9.8: the LASSI-EE paper's protocol details

- Task: P9.8 in `plans/p9-ee.md`. From arXiv:2505.02184 v3, say how the
  stopping counter ends a trial beside the maximum iteration count and maps
  to temperature; what each stage's prompt asks and what context it gets; how
  a refactor's energy is measured (repetitions, which value is fed back); the
  Stage 4 selection rule and report; the judge's input, output, and timing;
  n = 30; and what energy-reduction@k and the headline mean condition on.
  Every point the paper leaves open is listed as a choice for the stages'
  [DESIGN] entries, with a proposed default.
- Date: 2026-10-25. The three versions were fetched at about 2026-10-25T21:51Z.
- Where: the workstation (Windows, Git Bash, git 2.55.0.windows.3, curl
  8.21.0, poppler pdftotext, pdfimages, and pdfinfo 24.04.0, Python through
  `uv run --no-project python -I`), on branch `p9-ee` at 2697241, with this
  file uncommitted. No remote work, no rx, no ssh, and no device.
- Evidence label: exploratory reading. Every number below was read from the
  paper or recomputed from its published tables. None is a measurement by
  this project, and none carries [MEASURED].
- Quoting rule (the practice of plans/spikes/p2-lassi-metrics.md under
  OQ-018): quoted paper phrases are shorter than 40 characters; the prompt
  tables of v1 (Tables 1 to 8) are paraphrased, never quoted. All three
  versions are CC BY-SA 4.0 (the arXiv HTML license line of each).

## Question

1. How does the stopping counter (start 0.2, plus 0.2 per non-improving
   iteration, reset on improvement, doubling as temperature) end a trial
   beside the maximum iteration count, and what happens for models without
   temperature (o4-mini, GPT-5)?
2. What does each stage's prompt ask (stages 0 to 4), and what context is
   each given?
3. How is a refactor's energy measured: sampling, idle subtraction, clamping,
   repetitions, and which value goes back to the model in what form?
4. What is the Stage 4 selection rule, and what does the report hold?
5. What are the judge's input and output, and when does it run?
6. What is n = 30 a count of?
7. What exactly do energy-reduction@k and the headline mean energy reduction
   condition on?

Why it matters: P9.9 (stages 0 to 2 and the lassi-ee-2026 prompts) and P9.10
(refine_loop, select, compare_report) implement these stages and record
their choices as [DESIGN]; P9.11 places the judge; P9.6 implements the power
protocol; P9.12 computes energy_reduction, energy_reduction_at_k, and the
conditional and unconditional values beside the paper's.

## Classification

Factual: each answer comes from reading the three arXiv versions and
recomputing from their tables. Where the paper is silent, the gap is listed
in the Choices section for the owning task's [DESIGN] entry; one item, the
wording the lassi-ee-2026 prompts may carry in tracked files, is a policy
choice and has a draft owner-queue item (Consequences).

## Sources

- v3: https://arxiv.org/abs/2505.02184v3 (submitted 2026-05-09), HTML
  https://arxiv.org/html/2505.02184v3 and PDF https://arxiv.org/pdf/2505.02184v3.
  Sections II-A to II-F, III-A to III-C, IV-A, V-A, V-B; Figures 1 and 2;
  Tables I to III.
- v2: https://arxiv.org/abs/2505.02184v2 (2025-11-05), HTML and PDF. Sections
  2.1 to 2.6, 3.1, 3.3, 3.4. It cites Appendices A to E (prompts in Appendix A
  Tables 5 to 12, the energy-reduction@k algorithm in Appendix C); neither its
  arXiv HTML nor its PDF contains them.
- v1: https://arxiv.org/abs/2505.02184v1 (2025-05-04), HTML and PDF. Sections
  3.1 to 3.5, 4.1.5, 4.2, 5; Tables 1 to 8 (the prompt texts, CUDA on A100)
  and Table 10.
- GitHub API, read 2026-10-25: https://api.github.com/orgs/SPEAR-UIC/repos
  lists no LASSI-EE repository; SPEAR-UIC/LASSI's newest commit is still
  74b4681 (2024-09-23), whose tree holds only the LASSI notebook,
  prompt_dictionary.py, translated_code, images, LICENSE, and README.

## Commands and outputs (trimmed)

All downloads went to separate new directories under the session scratchpad;
the HTML-to-text script sat in a different directory and ran with `-I`.

```
$ curl -sSL -A "Mozilla/5.0" -o dl-v3/v3.html https://arxiv.org/html/2505.02184v3
$ curl -sSL -A "Mozilla/5.0" -o dl-v3/v3.pdf  https://arxiv.org/pdf/2505.02184v3
  (and the same for v2 and v1)
$ sha256sum dl-v1/* dl-v2/* dl-v3/*
fde0855d1e8258b56eec15c8539b9d8634a0196763a520ac5d82a25f55ca0dec dl-v1/v1.html
b263f2fd41bc4adcdd7cde96bbf418a09f8992362558391a110c13d96c73d787 dl-v1/v1.pdf
dc01131248410c717f6b081667196ddc264d409ed6fe1c3acb12bab3b1074af6 dl-v2/v2.html
db1d42f45bef3273b65cef58aa61a0207185f65578c49fa297aef1e2ebda6207 dl-v2/v2.pdf
35c755563e7e9cb845eff998d2912357265458fd4f00e37d4c0310543def07ec dl-v3/v3.html
1e5880211c74b3f1b644b5c85253a361833a9971422356f975baa62dd7453188 dl-v3/v3.pdf
$ pdfinfo dl-v3/v3.pdf | grep Pages   -> Pages: 12   (v2 12, v1 11)
$ uv run --no-project python -I scripts/html2txt.py dl-vN/vN.html txt/vN.txt
$ pdftotext dl-vN/vN.pdf txt/vN-pdfraw.txt
$ pdfimages -f 3 -l 3 -png dl-v3/v3.pdf img3/fig   (Figure 1, 4812x1920)
  (and v2 page 3, v1 pages 3 and 4; each Figure 1 was viewed)
$ curl -sS https://api.github.com/orgs/SPEAR-UIC/repos?per_page=100
  -> 31 repositories; none is LASSI-EE; LASSI pushed_at 2025-12-22
$ curl -sS https://api.github.com/repos/SPEAR-UIC/LASSI/commits?per_page=10
  -> 74b4681 2024-09-23T23:17:51Z Merge pull request #6 ... (newest)
```

Recounts from the papers' tables (Python, `-I`):

```
v3 Table II energy delta per app, mean over 22 apps:
  MI100 -36.16 (paper -36.2%), A100 -33.87 (paper -33.9%), both -35.02
v3 Table II Best Trial "Iter. EE": max 11 on each device;
  means MI100 5.82, A100 5.41 (the cross-app row prints 5.8 and 5.4)
v3 Table III cross-device = mean of device means:
  (29.6+28.6)/2 = 29.1, (49.5+46.9)/2 = 48.2, (55.2+54.1)/2 = 54.65 (paper 54.6)
v3 Table II delta = ratio of the printed means, e.g. segment-reduce MI100
  (468.0-710.9)/710.9 = -34.17% (paper -34.2), colorwheel MI100 -95.14% (-95.1)
float counter: 0.2, 0.4, 0.6000000000000001, 0.8, 1.0; "> 0.8" first true at 1.0
v1 Table 10 (one exemplar trial per app), Refactor Count minus Code Version
  for the 17 apps with a better code: 4 for 10 apps; 3 (randomAccess, 11
  attempts), 1 (entropy, 11), 3 (lid-driven-cavity, 11); 1 (colorwheel, 5),
  2 (keogh, 6), 5 (floydwarshall, 6), 6 (extrema, 6, Code Version 0);
  layout, dense-embedding, marchingCubes: no better code, 6 attempts each
v1 jacobi (4.2): 162.46 W x 7.90 s = 1283.4 J against 1281.83 J printed;
  1281.83 / 162.46 = 7.89 s, so average power is the net sample mean
```

## Findings

### 1. The stopping counter and temperature

- v3 II-E (Stopping Conditions): Stage 3 ends at "a configurable maximum
  iteration count", or after "a fixed number of consecutive" iterations with
  no energy improvement. The tracker "starts at 0.2", "increases by 0.2 on
  each unsuccessful" iteration, and "resets when an iteration produces" a
  new best. It doubles as the sampling temperature for models that take one ("doubles as the LLM sampling temperature").
  v3 states neither the fixed number nor the maximum.
- The threshold comes only from older versions. v1 3.4: the pipeline stops
  once the increment "goes above 0.8". v2 2.5: a stop "triggers when
  temperature exceeds 0.8"; for models without temperature the same stop
  fires after a "fixed number of unsuccessful iterations"; and v2 2.6 ends
  Stage 3 on that or on the maximum count. Read together: start 0.2, stop when
  the counter exceeds 0.8, so four consecutive non-improving iterations end
  Stage 3 (0.4, 0.6, 0.8, 1.0). Plain float accumulation gives the same
  count (0.8 is not above 0.8; the fourth step reaches 1.0).
- Maximum iteration count: no version gives the value. v3 Table II's
  best-trial "Iter. EE" reaches 11 on both devices (randomAccess A100,
  dense-embedding and pathfinder MI100), and v1 Table 10's Refactor Count
  reaches 11, where three apps stop at 11 attempts with fewer than four
  non-improving attempts after their best. That fits a cap of 11 Stage 3
  attempts in v1, though it is not stated. The bible recipe's max_iters: 10
  is not from the paper, and under it a best at iteration 11 is unreachable.
- What counts as unsuccessful is not defined. v3's wording (iterations that
  produce no energy improvement) covers both an INVALID candidate and a VALID
  one that is not better. v1 Table 10 does not settle it: most exemplar
  trials stop four attempts after their best, but extrema stops six after,
  floydwarshall five, and colorwheel and keogh one and two (Choices C2, C4).
- Reset value: v3 says only that it resets; v1 3.4 gives 0.2 as the default
  temperature, so the reset presumably returns to 0.2 (Choice C3).
- Models without temperature: v3 III-C, "Neither o4-mini nor GPT-5 supports"
  temperature; the iteration tracking ran, but "the temperature value had no
  effect" on inference. So no reported result exercised the temperature
  schedule. v2 3.4 had said it was "employed for GPT-5 during miniApp"
  generation, contradicting its own 2.5 (GPT-5 named among models without
  temperature); v3 removes that. v1 4.1.1 used o3-mini alone, also without
  temperature.
- Self-correction cap: v3 II-E names "A configurable maximum threshold"
  against runaway corrections, value unstated, and says no reported trial
  "terminated due to unrepairable" code errors, implying that exhausting it
  ends the trial (Choice C7).

### 2. What each stage's prompt asks, and its context

v3 prints no prompt; II says the templates "will be released in our
open-source" repository, and none is (Sources, GitHub API). v2 cites its
Appendix A Tables 5 to 12, which the arXiv copy lacks. The only published
prompt texts are v1 Tables 1 to 8, written for CUDA on an A100 with o3-mini.
Paraphrased:

- System prompt (v1 Table 2; v3 II-B: the same system prompt for every
  inference step, an expert C++/CUDA, or HIP, engineer refactoring for energy
  efficiency). It asks the model to refactor for lower GPU and CPU power while
  preserving all functionality and expected output, to use the provided
  background documents and plans, and to put code in triple-backtick fences.
- Stage 0, power profiling (v3 II-A): no prompt. The source is compiled, run,
  and profiled once per trial.
- Stage 1, zero-shot vanilla refactor (v1 Table 1; v3 II-B): one sentence
  asking to refactor the given CUDA code to cut GPU energy while preserving
  functionality. Context: the source code only, "without context or
  self-correction" assistance. v3 II-B: the result is "compiled and executed
  once"; a compile or run failure marks the vanilla attempt unsuccessful. If
  it runs, it is profiled and judged; a VALID code with lower energy seeds
  Stages 2 and 3, otherwise the source passes to Stage 2 (v3 Figure 1, the
  "Use source code" path). v1 3.2.1 and its Figure 1 differ: v1 self-corrected
  the vanilla code.
- Stage 2, context building (v1 Table 5, three standalone calls; v3 II-D, the
  model is self-prompted to "summarize relevant documentation" and to
  describe the code and "identify inefficiencies"):
  - background knowledge: summarize the key points of the given documents
    that apply to energy-efficient refactoring. v1 3.3.1 used Chapter 2 of the
    CUDA C++ Programming Guide and selected content of Chapters 12 to 15 of the
    CUDA C++ Best Practices Guide; v3 names only CUDA or HIP programming guide
    excerpts.
  - system details: describe the given hardware query output (GPU, CPU,
    DRAM, power, clocks, utilization, thermals, PCIe) with attention to
    energy; summary only, no code. v1 3.3.1 queried the host (nvidia-smi among
    others); v3 III-B still runs system-state queries from the pipeline.
  - source code analysis: with the documents, describe the code's whole
    function and name its main types, routines, and data layouts; no code changes.
  v1 3.3.1 calls these summaries a one-time process.
- Stage 2, refactoring plan (v1 Table 6; v3 II-D, "produce a refactoring
  plan"): five steps: read the documents, the summaries, and the system
  description; read the code analysis; find inefficiencies (redundant
  computation, memory use, synchronization, divergence, and others); list
  them; propose a plan that uses the system and the best practices, with no
  code. Context: the documents' summary, the system description, the code
  analysis, the code to refactor, and one sentence giving that code's measured
  total energy in joules and average power in watts. v1 3.3.2: when the
  incoming code is a revision (the vanilla seed), the prompt carries the
  source's analysis and the revision's.
- Stage 3, generation (v1 Table 7; v3 II-E, the model answers with "a
  description of changes" and one code block): pick the most impactful plan
  item not yet in the code, implement that one change to cut runtime, average
  power, or energy, keep all names, signatures, and outputs, make it compile
  with the named compiler, and describe the change before the code. Context:
  the plan, the code to refactor (the current best, v1 3.4), and the same
  energy and power sentence.
- Stage 3, self-correction (v1 Table 3; v3 II-E): fix the stated error while
  keeping names, signatures, kernel logic, and behavior; code only, in triple
  backticks. Context: the failing code, the compile command, and the compiler
  error; for a run error, the same with the run's error output. Not used for
  Stage 1 in v2 and v3.
- Stage 3, after a result (v3 II-E and Figure 1): INVALID, the code is
  archived and Stage 3 "restarts with the same refactoring plan"; VALID but
  not better, it is stored and the next attempt refactors the current best
  with the same plan (v1 3.4); a new best is saved and the model writes "a
  new refactoring plan reflecting" the latest optimizations, from a fresh
  analysis of the new best (Figure 1, the new version feeds "Code Eval +
  Refactoring Plan").
- Stage 4, comparison report (v1 Table 8; v3 II-F): compare the source and
  the refactored code, each with its stdout, runtime, total energy, and
  average power, both run on the same GPU, and highlight changes that may
  explain the energy gain, without repeating the code. v3 II-F: it names the
  transformations, the energy-relevant optimizations, and likely reasons;
  IV-C used it for "explicitly listing the optimizations" (over 4,000
  extracted).

### 3. How a refactor's energy is measured

- Sampling (v3 II-A and III-B): power "sampled every 0.01 seconds", NVML
  through pyNVML on the A100, the rocm-smi tool with its "--showpower flag"
  on the MI100 (v2 3.3: through subprocess calls). During execution power is
  collected "in a separate thread" at the same interval; launch and
  completion timestamps mark the runtime window.
- Idle (v3 III-B): "a brief initial period" before the run estimates idle
  power (length not given); sampling continues 15 s after the run; "post-run
  idle values are averaged" with the pre-run ones to give one idle value.
  How the two windows are weighted is not said (Choice C14).
- Net energy (v3 III-B; v1 4.1.5 and v2 3.3 give the formula): every sample
  has the idle value subtracted, negative samples are "clamped to zero", and
  net energy is the sum of the N samples times "the time interval of 0.01 s",
  a fixed interval, not the measured spacing. Reported power is net: v3
  Figure 2's caption says idle power is subtracted, and v1's jacobi numbers
  (162.46 W, 7.90 s, 1281.83 J) fit average power as the mean net sample.
  Only GPU power is measured (v3 III-B).
- Repetitions: none described. Stage 1's code is "compiled and executed
  once" (v3 II-B); each Stage 3 candidate gets one profiled run (v3 II-E);
  the source is profiled once per trial in Stage 0 and its Table II values
  are "averaged over 30 independent trials" (v3 IV-A). Every comparison
  inside a trial is therefore between single runs.
- Fed back to the model: v3 I, total energy goes into later prompts based on
  whether it improved. The published prompts (v1 Tables 6 and 7) carry one sentence after the code to refactor giving its measured total energy (J) and average power (W) on this host. The values are the
  code to refactor's, which is the current best, so a non-improving or
  INVALID candidate's own numbers are not fed back. Runtime is not in those
  prompts; the judge's and the report's prompts carry metrics (v1 Tables 4
  and 8; the report adds runtime). Number formatting is not stated.

### 4. The Stage 4 selection rule and report

- v3 II-E takes as the best the candidate with the least energy among the source, the vanilla code, and earlier Stage 3 codes that built, ran, and were screened in; a new version replaces it only if it "improves on the best". v3
  II-F: after Stage 3 the best is taken from all candidates; if no generated
  code passes and cuts energy, the source stays the best and no improvement
  is reported. Ties are not addressed; "improves" reads as strictly lower.
- Comparisons use the trial's own single measurements, the source's from
  that trial's Stage 0.
- The report (v3 II-F) runs only when the best cuts energy; "no automated
  decisions are based on" it. Saved with the best (v3 II-F): every
  intermediate version, energy profiles, judge verdicts, plans, iteration
  counts, and the report when there is one.

### 5. The judge

- Input (v3 II-C): the source code, the refactored code, both stdouts from runs "on identical inputs", and both codes' measured energy and runtime, given as context. v1 Table 4 lays them out as source code, its stdout, its metrics,
  then the refactored code, its stdout, its metrics.
- Output (v3 II-C): VALID (passes screening) or INVALID (filtered from the
  energy metrics). v1 3.2.4 and Table 4: reasoning first, the verdict last;
  the pipeline extracts it from the reply.
- Instructions (v3 II-C): tolerate output layout changes and small numeric drift from reordered floating-point work; mark changed algorithms, lost features, or large output gaps INVALID. For the 18
  of 20 HeCBench apps with built-in validators the judge "reads PASS/FAIL
  outcomes" in the stdout. No separate programmatic PASS/FAIL check is
  described: the model reads it.
- When (v3 II-B, II-E, Figure 1): after every candidate that compiled, ran,
  and was profiled: the vanilla code in Stage 1 and each Stage 3 candidate
  after self-correction. Never on a failed build or run, never on the
  source, and not on the Stage 4 report.
- Pass (v3 III-C): a code passes screening if it compiles, runs without
  errors, and gets VALID.
- Model: GPT-4.1 in v2 and v3 (v3 III-C); v1 used o3-mini for every call,
  the judge included, with the shared system prompt.

### 6. n = 30

v3 III-C: n = 30 "independent generation attempts" per application per
device; each run executes a vanilla generation and the whole pipeline,
"producing paired results". So one trial is one full Stage 0 to 4 run, and
its Stage 1 code is that trial's vanilla sample. 22 apps x 2 GPUs x 30 =
1,320 trials (v3 IV-A), about 30 minutes each, 10 to 114 minutes (v3 IV-A,
Pipeline Cost).

### 7. What the metrics condition on

- Headline (v3 abstract, IV-A, Table II): 36.2% (MI100) and 33.9% (A100).
  Per app and device, the source's energy is averaged over all 30 trials; the
  LASSI-EE energy is "averaged over only those trials" that compiled, ran,
  passed the judge, and "consumed less total energy" than the source. The
  delta is the ratio of those two means (recomputed from Table II). The
  headline is the unweighted mean of the 22 per-app deltas, which the
  caption says is "computed over all 22 applications" (recomputed: -36.16 and
  -33.87). Each per-app value conditions on success, and the number of
  trials behind each is not given. Whether "less energy than the source" is
  judged against the trial's own Stage 0 run or the 30-trial mean is not said.
- energy-reduction@k (v3 V-A, Eq. 1): the product of pass@k, the unbiased
  estimator 1 - C(n-c, k) / C(n, k) over n runs with c passing screening, and
  the expected best clamped relative reduction, max over i of
  ((E_source_mean - E_i) / E_source_mean)^+, over k candidates
  "sampled from generated codes" that passed screening. E_source_mean is the
  source's mean energy over all trial executions. So the second factor
  conditions on k passing candidates; v3 calls pass@k "independent of the
  energy reduction" factor. That is an assumption: the expectation of making
  k runs and keeping the best passing one generally differs, since among k
  runs fewer than k may pass. Passing candidates that use more energy count as 0
  (the clamp). v2 3.1 estimated the expectation "estimated by Monte Carlo"
  with an Appendix C algorithm that is not in the arXiv copy; v3 drops the
  Monte Carlo remark. The case c < k is not addressed.
- Which code is a trial's candidate is not stated. For LASSI-EE the natural
  reading is the trial's Stage 4 best, or its lowest-energy VALID generated
  code when none beats the source; for the vanilla LLM, the trial's Stage 1
  code.
- Cross-app values in Table III are unweighted means over the 22 apps per
  device, and the cross-device values are the mean of the two device means:
  29.1%, 48.2%, 54.6% at k = 1, 3, 5 against 10.5%, 26.3%, 34.6% for the
  vanilla LLM (v3 V-B).

## Differences between v1, v2, and v3 that bear on the protocol

| Point | v1 (2025-05-04) | v2 (2025-11-05) | v3 (2026-05-09) |
| --- | --- | --- | --- |
| Scope | 20 HeCBench apps, A100, CUDA | 22 apps, A100 and MI100 | same as v2 |
| Trials | 3 per app; one exemplar reported | 30 per app per device | 30 per app per device |
| Models | o3-mini for every call | o4-mini; GPT-5 generator for the two miniApps; GPT-4.1 judge | same as v2 |
| Stage 1 | self-corrected | compiled and run once, no correction | same as v2 |
| Stop rule | above 0.8 | exceeds 0.8, or a fixed count without temperature, or the max count | max count, or a fixed number of non-improving iterations (no value) |
| Temperature in runs | none (o3-mini) | says GPT-5 used it, contradicting its 2.5 | had no effect for either model |
| Judge input | codes, stdouts, metrics (Table 4) | codes, stdouts, energy metrics | adds PASS/FAIL reading and floating-point tolerance |
| Prompts | Tables 1 to 8 printed | Appendix A cited, absent | none; "will be released" |
| Metrics | 73.3% of trials, 47.36% mean saving | pass@k and energy-reduction@k = pass@k x energy-reduction@k^valid, Monte Carlo | one product formula, no Monte Carlo remark |
| Headline | 47% across 85% of 20 apps | abstract: 29% expected at k = 1 (81% pass), 48% at k = 3 | 36% MI100, 34% A100, conditional |
| MI100 ROCm | not run | driver 6.14.0 | driver 6.4.1 |

The bible's LASSI-EE section matches v3 except where the edit below adds
detail; its Stage 0 to 4 names and the n = 30, 1,320-trial, and k-metric
numbers are right.

## Choices the paper leaves open (for the stages' [DESIGN] entries)

Each choice should be an explicit recipe value with no default that fails
with a clear message, unless marked as fixed behavior of the stage.

Stage 3 control (P9.10):

- C1 Stop threshold. Proposed: a required key refine.counter_stop, 0.8 in the
  lassi-ee recipe, compared strictly greater, with the counter computed
  exactly (start + step x count, in decimal), so four consecutive
  non-improving iterations end Stage 3, as v1 and v2 state.
- C2 What counts as non-improving. Proposed: every Stage 3 iteration that
  yields no new best: INVALID, VALID but not lower, and one whose correction
  budget ran out if C7 keeps the trial going. v3's wording covers all three.
- C3 Reset value. Proposed: back to counter_start.
- C4 max_iters. Proposed: keep it required; set 11 in the lassi-ee recipe,
  in place of 10, since v3's best trials reach iteration 11 and v1's
  attempts stop at 11. Count only fresh Stage 3 generations (not corrections,
  not Stage 1, not re-plans), matching v3 IV-A's "fourth refactoring attempt"
  reading of Iter. EE = 4.
- C5 Counter as temperature. The paper's design applies it when the model
  takes temperature; its reported runs never did. Proposed: a required bool,
  true in the lassi-ee recipes (the RNGD generators take temperature), and
  every result labeled that the schedule was applied while the paper's runs
  had none.
- C6 Temperature of the other calls. Proposed: Stages 1, 2, and 4 and the
  judge use llm.sampling (0.2, top_p 0.9, v1 3.4's 0.2 default); every call
  inside a Stage 3 iteration (generation, its corrections, a re-plan) uses
  the iteration's counter when C5 is true.
- C7 Correction budget exhausted. Proposed: loop.max_corrections bounds
  corrections per candidate; exhausting it ends the trial's Stage 3 (v3 II-E,
  no trial "terminated due to unrepairable" errors), Stage 4 runs on what
  exists, and the reason is recorded.
- C8 Re-plan scope. Proposed: on a new best, a new code analysis of the best
  and a new plan; the document summary and system description stay as made
  once per trial (v1 3.3.1, v3 Figure 1).

Prompts and context (P9.9):

- C9 Prompt wording. v1 Tables 1 to 8 are the only published texts. Proposed:
  write lassi-ee-2026 from their structure and v3 II-C's judge additions, per
  language: HIP with the hipcc command of v3 III-B, and the OpenMP-on-CPU
  variant naming g++ and CPU package energy in place of GPU power. Whether
  tracked templates may carry the papers' wording is the owner item below.
- C10 Context documents. Not pinned by v3. Proposed: the recipe names the pack
  (hip-guide-excerpt for HIP; for the CPU port a named OpenMP pack, the
  existing OpenMP card unless P9.9 builds one), recorded in the run.
- C11 System details. Proposed: a fixed, read-only host description per run
  (lscpu, memory size, the package and powercap domain names; for GPUs a
  read-only query P9.14 names), with no device open, recorded in the trial.
- C12 Feedback format. Proposed: energy in J and average power in W, two
  decimals, in one sentence after the code to refactor, as v1 Tables 6 and 7
  lay out; runtime only in the judge and report prompts.

Power protocol (P9.6):

- C13 Pre-run idle window length. "brief" only. Proposed: its own required
  key beside idle_window_s (the 15 s post-run window), set to 5 s in the
  lassi-ee recipes.
- C14 Idle combination. Proposed: the mean of the two window means, each
  weighted one half, with drift = post mean minus pre mean recorded (the
  bible's drift rule).
- C15 Integration interval. The paper multiplies by a fixed 0.01 s. Proposed:
  record the achieved mean sample spacing on every profile; integrate with
  the fixed interval_ms as written, and offer measured-spacing integration as
  a named, off-by-default option, since a sampler that cannot keep 10 ms
  (a CLI call per sample, for one) would undercount under the fixed interval.
  For a cumulative counter (RAPL), net energy = counter delta over the
  runtime window minus idle power x runtime, floored at 0, and the per-sample
  clamp does not apply; the Profile says which rule was used.
- C16 Repetitions. Proposed: a required profiler key for runs per
  measurement, 1 in the lassi-ee recipes (the paper's single runs), with the
  measurement taken as the mean when it is set higher.
- C17 Runtime. Proposed: wall time from process launch to completion, the
  paper's runtime window, not kernel time.
- C18 Power reported. Proposed: net average power (mean clamped net sample)
  as the paper reports, with gross average power and idle power beside it.

Judge (P9.11):

- C19 Metrics in the judge's input. The paper gives them. Proposed: include
  energy, power, and runtime as v1 Table 4 lays them out; the P9 plan's
  comment stripping is not in the paper, so it is a named option, off for
  the paper-faithful screen.
- C20 Verdict parsing. Proposed: the last whole-word VALID or INVALID in the
  reply (INVALID contains VALID); a reply with neither is INVALID with a note.
- C21 Pass definition. The paper's pass is the judge's VALID alone. Agent Rule
  14 forbids a judge overriding an oracle, so proposed: pass = passfail PASS
  where the app has a validator, and VALID; the paper's judge-only count is
  reported beside it, tagged [JUDGED], with agreement per trial.

Selection and report (P9.10):

- C22 Ties. Proposed: strictly lower energy replaces the best; on a tie the
  earlier code stays.
- C23 Report. Proposed: only when the best is not the source; input as v1
  Table 8 (both codes, stdouts, runtime, energy, average power); ask for a
  list of the optimizations applied (v3 IV-C); text tagged [JUDGED].

Metrics (P9.12):

- C24 Trial candidate. Proposed: for LASSI-EE, a trial passes if any
  generated code (Stage 1 or 3) passed screening; its candidate energy is the
  lowest such code's single measurement. For vanilla, the Stage 1 code.
- C25 Estimator. Proposed: the paper's product computed exactly, without
  Monte Carlo: pass@k by the unbiased estimator, and the expected maximum of
  k draws without replacement from the c passing reductions by order
  statistics, sum over j of r_(j) x C(j-1, k-1) / C(c, k) with r sorted
  ascending; for 0 < c < k, the maximum over all c, with a note; for c = 0,
  0. Report beside it, under its own name, the exact expectation of making k
  runs and keeping the best passing one, which drops v3's independence
  assumption.
- C26 Source reference. Proposed: E_source_mean over the 30 Stage 0 runs per
  app and device; the conditional filter "less energy than the source" uses
  the trial's own Stage 0 run (the pipeline's own decision), with the
  30-trial-mean variant reported as a check.
- C27 Aggregation. Proposed: per-app delta = ratio of means; cross-app =
  unweighted mean over apps; cross-device = mean of device means, as the
  paper's tables compute them; every conditional value carries its trial
  count.

## Finding

The paper leaves the protocol under-specified in v3 alone; v1's prompt
tables and v1 and v2's stop rule fill the main gaps. The counter stops Stage
3 when it exceeds 0.8, four consecutive non-improving iterations, beside an
unstated maximum that the published tables put at 11 or more, and in every
reported run it never acted as temperature. Each candidate and each source
run is measured once; the model sees the current best's energy (J) and
average power (W) in one sentence. The headline 36.2% and 33.9% are
unweighted means over 22 apps of per-app conditional means, and
energy-reduction@k multiplies pass@k by an expected best reduction over k
passing candidates.

Confidence: high for what v3 states and for the recomputations; medium for
the stop threshold (stated in v1 and v2, dropped in v3) and for the reading
that the fed-back values are the current best's; low for the cap of 11,
which is inferred from tables.

## Consequences for the plan

- P9.10: refine needs counter_stop (C1) and the C2 to C8 rules; the lassi-ee
  recipe's max_iters: 10 should become 11 (C4) in the bible's Project
  Recipes with a Decision Log entry; tests should pin the sequence 0.2, 0.4,
  0.6, 0.8, then stop at 1.0, the reset to 0.2, and the strict tie rule.
- P9.9: Stage 1 makes no correction; Stage 2 is three summary calls and one
  plan call; prompts follow C9 to C12, pending the owner item on wording.
- P9.6: add the pre-run window key (C13), the C14 to C18 rules, and record
  achieved sample spacing.
- P9.11: the judge sees metrics (C19); screening = passfail and VALID (C21).
- P9.12: implement C24 to C27; label the paper's energy-reduction@k as its
  product formula; report the conditional trial counts.
- Bible: the Source Papers edit below; the "code and prompts are not
  released" caveat becomes more exact (v1 prints prompts).

Owner-queue item (filed as OQ-094 on 2026-10-25):

- Title: Paper prompt wording in tracked lassi-ee-2026 templates.
- Kind: policy. Blocks: P9.9 (the template files only; the stage code does
  not wait).
- Question: The only published LASSI-EE prompts are v1 Tables 1 to 8 of
  arXiv:2505.02184 (CC BY-SA 4.0). OQ-018 kept upstream LASSI's prompts out
  of the repository. May the lassi-ee-2026 templates carry the paper's
  wording in tracked files?
- Options: (a) keep them local only, generated from the paper by a tool into
  a gitignored tree as with lassi-2024; the repository holds a manifest with
  sha256, and every run records the bytes it used; costs a generator and a
  local step on each host. (b) track templates written in the project's own
  words that follow the tables' structure and inputs, no sentence copied;
  simple, but a paraphrase drifts from the paper and is labeled a port.
  (c) track the paper's wording with CC BY-SA attribution; most faithful,
  but adds share-alike text to an unlicensed public repository, which
  OQ-018's answer argues against.
- Recommendation: (a), since it matches OQ-018's answer and keeps the
  faithful wording; (b) as the fallback if the owner wants no generator.

## Proposed bible edit (Source Papers, LASSI-EE)

Replace the five bullets after the scope line with:

```
- Stage 0: power profiling of the source, once per trial. Stage 1: zero-shot vanilla refactor, compiled and run once with no self-correction (v1 self-corrected it). Stage 2: summaries of documentation excerpts and of host system details, a description of the incoming code, and a refactoring plan. Stage 3: one plan item per iteration, self-corrected for compile and run errors, profiled, and judged; a new best brings a new code analysis and plan, otherwise the next iteration refactors the current best with the same plan. Stage 4: the best is the lowest-energy code among the source, a passing vanilla code, and passing Stage 3 codes (the source when none beats it); an LLM comparison report follows only when the best is not the source, and no decision uses it.
- Prompt texts: v1 prints the CUDA versions (system, zero-shot, correction, judge, context, plan, generation, and comparison prompts; v1 Tables 1 to 8); v2 cites appendix prompts its arXiv copy lacks; v3 prints none. The plan and generation prompts give the code to refactor's measured total energy (J) and average power (W) in one sentence; the comparison prompt adds runtime.
- LLM-as-a-Judge (GPT-4.1) returns VALID or INVALID from source, candidate, both stdouts on identical inputs, and both codes' energy and runtime measurements, reading PASS/FAIL where present (18 of 20 HeCBench apps). It runs after every candidate that compiled and ran (Stage 1 and each Stage 3 iteration). A code passes screening if it compiles, runs, and is VALID; the paper describes no programmatic PASS/FAIL check.
- Stopping counter starts at 0.2, adds 0.2 per non-improving Stage 3 iteration, resets on a new best, and doubles as temperature where the model takes one. v1 and v2 stop when it exceeds 0.8 (four consecutive non-improving iterations); v3 says only a fixed number. A configurable maximum iteration count also ends Stage 3; no version states it, and v3's best trials reach iteration 11. o4-mini and GPT-5 take no temperature, so in every reported run the counter only counted (v3 III-C).
- Power sampled every 10 ms via pynvml or `rocm-smi --showpower`; idle power is the average of a brief pre-run window (length not stated) and a 15 s post-run window (weighting not stated); net energy is 0.01 s times the sum of idle-subtracted samples in the runtime window, negative samples clamped to 0; reported power is net. Each candidate and each trial's source run is executed once; no repetitions are described.
- n = 30 per app per device, each trial a full pipeline run whose Stage 1 code is that trial's vanilla sample: 1,320 trials, about 30 minutes per trial. Generators o4-mini (GPT-5 for miniApps).
- Results: 36.2% (MI100) and 33.9% (A100) mean energy reduction: per app, the mean energy of trials that passed screening and used less energy than the source, against the source's 30-trial mean, then an unweighted mean over the 22 apps. energy-reduction@k is pass@k times the expected best clamped reduction against the source's mean energy over k candidates sampled from the passing codes: 29.1%, 48.2%, 54.6% at k = 1, 3, 5 (the mean of the MI100 and A100 cross-app means), versus 10.5%, 26.3%, 34.6% for the vanilla LLM.
- Caveats: headline averages condition on success; v1 reported 47% across 85% of 20 apps (A100, three trials per app), so cite the version; no LASSI-EE code or prompt set is released (SPEAR-UIC on GitHub, read 2026-10-25). Protocol details and open choices: plans/spikes/p9-ee-protocol.md.
```

Decision Log entry (top of the table; raise the count sentence by one, and
the 2026-10-25 count by one, at the time it is applied):

```
| 2026-10-25 | LASSI-EE protocol details recorded from arXiv:2505.02184 v1 to v3 (task P9.8; Source Papers, LASSI-EE; plans/spikes/p9-ee-protocol.md): the stop threshold (the counter exceeds 0.8, four consecutive non-improving iterations, from v1 and v2), the unstated maximum iteration count (v3's best trials reach iteration 11), temperature unused in every reported run, Stage 1 without self-correction, the judge's input with energy and runtime, single runs per measurement with the current best's energy and power fed back, the selection rule, and what the headline mean and energy-reduction@k condition on. The choices the paper leaves open are listed in the spike for P9.6, P9.9 to P9.12 to record as [DESIGN] | v3 alone leaves the stop threshold, the cap, the prompts, and the metrics' conditioning unstated, and the stages must implement them; the older versions and the published tables settle most of it, and what stays open is now named instead of picked silently ([OPEN] rule; Agent Rules 1 and 4) |
```
