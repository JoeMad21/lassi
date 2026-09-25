# Spike P2.4: the paper's LASSI metric definitions

- Task: P2.4 in `plans/p2-scoring.md`. It must say what counted as correct
  output, give the denominator of each percentage in the bible's LASSI
  results table, say how runtime ratios and first try were counted, and say
  what the notebook stores per trial. Each claim cites arXiv:2407.01638, and
  a pinned notebook cell where one applies.
- Date: 2026-09-24 (EDT). The paper was retrieved at about 2026-09-24T20:32Z.
- Where: the workstation (Windows, Git Bash, git 2.55.0.windows.3, curl
  8.21.0; Python 3.14.2 for the recount; pypdf 6.19.0 under Python 3.10.21
  for text extraction), on branch `p2-scoring` at 207e4dd, with this file
  uncommitted. No remote work, no rx, and no device.
- Evidence label: exploratory reading. Every count below was read from the
  paper's published tables or from upstream source. None is a measurement by
  this project, and none carries [MEASURED].
- OQ-018: this file quotes no upstream prompt or notebook text of 40 or more
  characters. It paraphrases notebook behavior and cites the cell id and the
  line within the cell (lines count from 1 in the cell's joined source).
  Quoted paper phrases are shorter than 40 characters, and the paper's prompt
  tables (Tables I to III) are not quoted.

## Question

The lassi score profile (P2.7) and the metrics tables (P2.8) need these
answers from the LASSI paper and the pinned notebook:

1. What counted as correct output?
2. What is the denominator of each percentage in the bible's LASSI results
   table (correct output, within 10% or faster, first try, Sim-T >= 0.6)?
3. How were runtime ratios and first try counted?
4. What does the notebook store per trial?

Why it matters: P2.7 defines the lassi profile's components as this spike
defines them. P2.8 shows the paper's values next to each metric. The
Acceptance Criteria mark each headline metric reproduced, or report its gap.

## Classification

Factual. The answers come from reading the paper and the pinned notebook and
from counting rows of the paper's tables. Two consequences are choices for
the owner (see Owner-queue drafts).

## Sources

- Paper: arXiv:2407.01638 (LASSI, CLUSTER 2024 LLMxHPC workshop), by
  Dearing, Tao, Wu, Lan, and Taylor. The abstract page https://arxiv.org/abs/2407.01638 lists
  v1 (2024-06-30 19:36:04 UTC) and v2 (2025-05-04 17:21:20 UTC). The PDFs,
  both retrieved 2026-09-24 and both 8 pages:
  - https://arxiv.org/pdf/2407.01638v1, sha256
    e1e64e804a0819f2189944696fab6ab7ea5423e29993c04e6343039ef9b7a460
  - https://arxiv.org/pdf/2407.01638v2, sha256
    21074bbc12e4460b40a541ec0b028f8de1071a4c2d9fe250007bfa8ea803fd32

  Page, section, and table citations below are to the v2 PDF. v1 has the same
  tables and percentages (see Commands).
- Upstream: SPEAR-UIC/LASSI at 74b46812523f2ff79b53b6880a4521690d7478b0
  (`third_party/LASSI`, fetched as `assets/upstream/lassi.yaml` pins it):
  `LASSI_pipeline_v0.ipynb` (nbformat 4.5, 23 cells, no stored outputs) and
  `README.md`. From the same repository's history: `LASSI_latest_07-23-2024.ipynb`,
  added in f55bf0c (2024-07-26) and deleted in e39d123 (2024-09-23), before
  the pin.

## Commands and outputs

The scratch files (`old.ipynb`, `new.ipynb`, `abs.html`, `v1.pdf`, `v2.pdf`,
`v1.txt`, `v2.txt`, `recount.py`) were kept in the workstation's temporary
directory, never under the repository. Outputs are trimmed.

### Upstream

```
$ git -C third_party/LASSI log --oneline -1
74b4681 Merge pull request #6 from SPEAR-UIC/developmtd
$ git -C third_party/LASSI ls-files
LASSI_pipeline_v0.ipynb
LICENSE
README.md
images/LASSI_logo_v0.png
prompt_dictionary.py
translated_code/input_codes/HeCBench/<app>/<app>-<cuda|omp>_main.<cu|cpp>   (20 files)
$ git -C third_party/LASSI show HEAD:LASSI_pipeline_v0.ipynb | python -c '<print id, type, size, and output count of each cell>'
nbformat 4 5
(23 cells, each with 0 outputs; the ones read for this spike:)
cell 11 00ddfed0-af7a-4236-8d82-b4b816784e7a code   reply -> first fenced block
cell 13 e0fe91ba-6894-43b3-a964-654266dcf704 code   compile_code
cell 14 976745b6-f871-4500-9527-950a7f30c91f code   execute_code
cell 15 a949bc43-3e33-4e51-815d-2ef27c350835 code   token counts, similarities
cell 17 fdb5b82f-b135-4226-92ff-056d69e17903 code   the pipeline function
cell 20 62e4eab8-7be4-4423-9e74-c420ef8f51a6 code   experimental_setup
cell 22 75a66833-3bc6-44e3-9138-7d574a821cfd code   the driver
$ git -C third_party/LASSI log --format='%h %ad %s' --date=short --name-status
(trimmed to the notebook commits)
1fa2716 2024-09-23 Initial commit of LASSI pipeline and prompt dictionary   A LASSI_pipeline_v0.ipynb
e39d123 2024-09-23 Delete LASSI_latest_07-23-2024.ipynb                      D LASSI_latest_07-23-2024.ipynb
f55bf0c 2024-07-26 Experiment version of pipeline for running without context.   A LASSI_latest_07-23-2024.ipynb
```

The README at 74b4681 says that the generated codes from the paper are not
yet included, and it defines no metric.

The earlier notebook, compared cell by cell with the pinned one:

```
$ git -C third_party/LASSI show f55bf0c:LASSI_latest_07-23-2024.ipynb > old.ipynb
$ git -C third_party/LASSI show HEAD:LASSI_pipeline_v0.ipynb > new.ipynb
$ python -c '<unified diff of the joined source of each cell id>'
differs: 3d11fe58 (title), efdff123 (a comment moved), 62e4eab8 (setup values:
         app, direction, context toggle, model, num_ctx, label; a private
         gateway branch removed), 75a66833 (one comment)
only in old: e95c7c38 (a private gateway model call), b8128101 (empty)
```

The cells that generate, compile, run, and store (00ddfed0, e0fe91ba,
976745b6, a949bc43, fdb5b82f) are identical in both notebooks. The old
notebook's one stored output shows a baseline run failing for mallocFree. The
pinned setup lists that app with a DNU mark, and the paper does not use it.
The output holds no metric.

### Paper

```
$ curl -sSL -o abs.html https://arxiv.org/abs/2407.01638
$ grep -o -E '<submission lines>' abs.html
Sun, 30 Jun 2024 19:36:04 UTC (209 KB)
Sun, 4 May 2025 17:21:20 UTC (784 KB)
$ curl -sSL -o v1.pdf https://arxiv.org/pdf/2407.01638v1
$ curl -sSL -o v2.pdf https://arxiv.org/pdf/2407.01638v2
$ sha256sum v1.pdf v2.pdf
e1e64e804a0819f2189944696fab6ab7ea5423e29993c04e6343039ef9b7a460 *v1.pdf
21074bbc12e4460b40a541ec0b028f8de1071a4c2d9fe250007bfa8ea803fd32 *v2.pdf
$ uv run --no-project --with pypdf python -c '<write the text of each page to v1.txt and v2.txt>'
v1 pages 8 (CreationDate D:20240703000327Z)
v2 pages 8 (CreationDate D:20250506005407Z)
```

Cross-checks:

- Tables IV, VI, and VII in `v2.txt` agree with PDF pages 5 to 7 rendered as
  images.
- A word-level diff of `v1.txt` and `v2.txt` shows only wording, reference
  numbering, and layout changes. Every table row is identical, and the decimal
  and percent tokens differ only in reference identifiers:

```
$ python -c '<compare decimal and percent tokens, and the table rows, of v1.txt and v2.txt>'
decimal/percent tokens v1 332 v2 338
only v1: {'2401.12554': 1}
only v2: {'10.48550': 4, '2107.03374': 1, '10.1145': 1, '3625549.3658689': 1}
table rows identical v1 vs v2 (in order of appearance): True [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
```

### Recount

`recount.py` (listed below) parses Tables IV, VI, and VII from `v2.txt` and
counts every rate. It runs on v2 only, because v1 captions its panels
differently. The row comparison above covers v1.

```
$ python recount.py v2.txt
== OMP->CUDA: 40 trials, 32 not N/A; per model: {'GPT-4': 7, 'Codestral': 9, 'WizardCoder': 9, 'DeepSeek': 7}
   correct output                           32/40 =  80.0%
   within 10% (printed Ratio >= 0.9)        23/32 =  71.9%
   within 10% (printed Ratio >= 1/1.1)      23/32 =  71.9%
   within 10% (Ratio recomputed, >= 0.9)    24/32 =  75.0%
   first try (Self-corr == 0)               21/32 =  65.6%
   Sim-T >= 0.6                              8/32 =  25.0%
   Sim-T > 0.6                               8/32 =  25.0%
   paper correct     80.0%  fractions with denominator <= 40: [(4, 5), (8, 10), (12, 15), (16, 20), (20, 25), (24, 30), (28, 35), (32, 40)]
   paper within 10%  78.1%  fractions with denominator <= 40: [(25, 32)]
   paper first try   65.6%  fractions with denominator <= 40: [(21, 32)]
   paper Sim-T       40.6%  fractions with denominator <= 40: [(13, 32)]
   printed Ratio != Table IV runtime / Runtime: [('GPT-4', 'atomicCost', 0.5854, 0.9573)]
   Self-corr > 7 among successes: []
   printed-Ratio thresholds (>=, step 0.001) giving the paper's within 10%: (0.636, 0.794)
   Sim-T thresholds (>=, step 0.01) giving the paper's Sim-T share: [0.54]
== CUDA->OMP: 40 trials, 34 not N/A; per model: {'GPT-4': 9, 'Codestral': 8, 'WizardCoder': 10, 'DeepSeek': 7}
   correct output                           34/40 =  85.0%
   within 10% (printed Ratio >= 0.9)        20/34 =  58.8%
   within 10% (printed Ratio >= 1/1.1)      20/34 =  58.8%
   within 10% (Ratio recomputed, >= 0.9)    20/34 =  58.8%
   first try (Self-corr == 0)               18/34 =  52.9%
   Sim-T >= 0.6                             15/34 =  44.1%
   Sim-T > 0.6                              14/34 =  41.2%
   paper correct     85.0%  fractions with denominator <= 40: [(17, 20), (34, 40)]
   paper within 10%  61.8%  fractions with denominator <= 40: [(21, 34)]
   paper first try   55.9%  fractions with denominator <= 40: [(19, 34)]
   paper Sim-T       47.1%  fractions with denominator <= 40: [(8, 17), (16, 34)]
   printed Ratio != Table IV runtime / Runtime: []
   Self-corr > 7 among successes: [('Codestral', 'pathfinder', 34)]
   printed-Ratio thresholds (>=, step 0.001) giving the paper's within 10%: (0.89, 0.89)
   Sim-T thresholds (>=, step 0.01) giving the paper's Sim-T share: [0.59]
```

The trials behind the recounts (G GPT-4, C Codestral, W WizardCoder, D
DeepSeek-Coder-V2), from the same tables:

- N/A, OMP to CUDA (8): G dense-embedding, D dense-embedding, G bsearch, C
  colorwheel, D colorwheel, G randomAccess, W randomAccess, D randomAccess.
- N/A, CUDA to OMP (6): C jacobi, G, C, and D dense-embedding, D pathfinder,
  D randomAccess.
- Printed Ratio below 0.9, OMP to CUDA (9): G layout 0.5854, D layout
  0.6355, G atomicCost 0.5854 (0.9573 recomputed), W atomicCost 0.3777, D
  atomicCost 0.4715, G pathfinder 0.8595, D pathfinder 0.7946, C entropy
  0.6037, C randomAccess 0.5640.
- Printed Ratio below 0.9, CUDA to OMP (14): D matrix-rotate 0.1072, C layout
  0.6369, G atomicCost 0.2055, C atomicCost 0.6260, C bsearch 0.0498, W
  bsearch 0.8861, C entropy 0.8763, W entropy 0.8763, D entropy 0.4394, G
  colorwheel 0.7273, W colorwheel 0.6957, D colorwheel 0.2192, C randomAccess
  0.8907, W randomAccess 0.8896.
- Sim-T >= 0.6, OMP to CUDA (8): G jacobi, G layout, G atomicCost, W
  dense-embedding, G entropy, G colorwheel, W colorwheel, C randomAccess.
- Sim-T >= 0.6, CUDA to OMP (15): G and C matrix-rotate, G layout (0.60), G
  and C atomicCost, G, C, W, and D entropy, G, C, W, and D colorwheel, G and C
  randomAccess.

`recount.py` (sha256
101e10f3b78213a4087d023ba156c46117278461df339b4a811335e8f8f3e880):

```
"""Recount the LASSI paper's headline percentages from its Tables IV, VI, VII (P2.4 spike helper).

Reads the pypdf text of arXiv:2407.01638v2 (argv[1]); prints counts only.
"""
import re, sys
TXT = open(sys.argv[1], encoding="utf-8").read()
APPS = ["matrix-rotate", "jacobi", "layout", "atomicCost", "dense-embedding",
        "pathfinder", "bsearch", "entropy", "colorwheel", "randomAccess"]
MODELS = ["GPT-4", "Codestral", "WizardCoder", "DeepSeek"]
PAPER = {"OMP->CUDA": (80.0, 78.1, 65.6, 40.6), "CUDA->OMP": (85.0, 61.8, 55.9, 47.1)}
T4 = {}
for app in APPS:
    m = re.search(re.escape(app) + r" (?:\[[^\]]*\]|None) ([0-9.]+) ([0-9.]+)", TXT)
    T4[app] = {"CUDA": float(m.group(1)), "OMP": float(m.group(2))}

def panel(start, end):
    seg = TXT[TXT.index(start):TXT.index(end)]
    rows = {}
    for app in APPS:
        toks = re.search(r"^" + re.escape(app) + r" (.*)$", seg, re.M).group(1).split()
        assert len(toks) == 10
        rows[app] = [None if t[0] == "N/A" else dict(zip(("rt", "ratio", "simt", "siml", "sc"),
                     (*map(float, t[:4]), int(t[4])))) for t in (toks[:5], toks[5:])]
    return rows

def trials(a, b, target):
    return [(app, model, target, rec) for app in APPS
            for model, rec in zip(MODELS, a[app] + b[app])]

DIRS = {"OMP->CUDA": trials(panel("(a) Panel A", "(b) Panel B"), panel("(b) Panel B", "represents the number"), "CUDA"),
        "CUDA->OMP": trials(panel("(a) *Panel A", "(b) *Panel B"), panel("(b) *Panel B", "D. Discussion"), "OMP")}

def fits(value, top=40):
    return [(n, d) for d in range(1, top + 1) for n in range(d + 1) if round(100 * n / d, 1) == value]

for d, ts in DIRS.items():
    ok = [t for t in ts if t[3]]
    n = len(ok)
    ratio_rc = lambda t: T4[t[0]][t[2]] / t[3]["rt"]
    rows = [
        ("correct output", sum(1 for t in ts if t[3]), len(ts)),
        ("within 10% (printed Ratio >= 0.9)", sum(1 for t in ok if t[3]["ratio"] >= 0.9), n),
        ("within 10% (printed Ratio >= 1/1.1)", sum(1 for t in ok if t[3]["ratio"] >= 1 / 1.1), n),
        ("within 10% (Ratio recomputed, >= 0.9)", sum(1 for t in ok if ratio_rc(t) >= 0.9), n),
        ("first try (Self-corr == 0)", sum(1 for t in ok if t[3]["sc"] == 0), n),
        ("Sim-T >= 0.6", sum(1 for t in ok if t[3]["simt"] >= 0.6), n),
        ("Sim-T > 0.6", sum(1 for t in ok if t[3]["simt"] > 0.6), n),
    ]
    print(f"== {d}: {len(ts)} trials, {n} not N/A; per model:",
          {m: sum(1 for t in ok if t[1] == m) for m in MODELS})
    for label, num, den in rows:
        print(f"   {label:40s} {num:2d}/{den} = {100 * num / den:5.1f}%")
    for label, v in zip(("correct", "within 10%", "first try", "Sim-T"), PAPER[d]):
        print(f"   paper {label:10s} {v:5.1f}%  fractions with denominator <= 40: {fits(v)}")
    bad = [(t[1], t[0], t[3]["ratio"], round(ratio_rc(t), 4)) for t in ok if abs(ratio_rc(t) - t[3]["ratio"]) > 5e-4]
    print("   printed Ratio != Table IV runtime / Runtime:", bad)
    print("   Self-corr > 7 among successes:", [(t[1], t[0], t[3]["sc"]) for t in ok if t[3]["sc"] > 7])
    w10 = [x / 1000 for x in range(500, 1201) if round(100 * sum(1 for t in ok if t[3]["ratio"] >= x / 1000) / n, 1) == PAPER[d][1]]
    print("   printed-Ratio thresholds (>=, step 0.001) giving the paper's within 10%:", (w10[0], w10[-1]) if w10 else None)
    st = [x / 100 for x in range(0, 101) if round(100 * sum(1 for t in ok if t[3]["simt"] >= x / 100) / n, 1) == PAPER[d][3]]
    print("   Sim-T thresholds (>=, step 0.01) giving the paper's Sim-T share:", st)
```

## Findings

Terms: a paper trial is one pipeline run for one app, direction, and model.
The paper ran 80 scenarios, 10 apps x 4 models x 2 directions (p. 5, Sec. V),
which gives 40 trials per direction. In Tables VI and VII, a row with values is
a success and an N/A row is a failure (captions, pp. 6 and 7). The paper
sometimes calls the HeCBench original in the target language the "source"
code (p. 5, Sec. V-A; p. 6, Sec. V-B, "the source HeCBench code in CUDA" for
OpenMP to CUDA). That means the program the translation is compared with, not
the input program.

### 1. What counted as correct output

Paper:

- The baseline compiles and runs the original target-language code and keeps
  its stdout for later comparison (p. 2, Sec. III-A).
- The generated code's stdout, from a run after a successful compile, was
  stored in a metadata file for manual comparison with that output.
  Automated verification is named as future work (p. 4, end of Sec. III-D;
  p. 7, Sec. VI).
- The target-language outputs were captured "for visual inspection" (p. 6,
  Sec. V-B).
- N/A marks code that could not be compiled, could not be executed, or had
  "significantly different output" (Table VI and VII captions, pp. 6 and 7;
  also Sec. V-B). Sec. V-A (p. 5) describes the output case as stdout that
  "did not match the expected result".
- The worked cases: for Codestral's CUDA to OpenMP bsearch, the stdouts were
  identical "except for reported timings". For DeepSeek-Coder-V2's CUDA to
  OpenMP atomicCost, the outputs were identical (p. 7, Sec. V-D).

Notebook:

- The pipeline counts a run as failed only when its exit status is not 0
  (cell 976745b6, lines 53 to 89). A nonzero exit feeds a correction.
- The loop ends after an attempt that compiles and then either runs with exit
  0 or is left unrun by the execution gate (cell fdb5b82f, lines 80 and 127 to
  136).
- The notebook never compares outputs and records no verdict. It writes the
  reference stdout and the generated stdout into the metadata file (cell
  fdb5b82f, lines 174 to 178).

Answer: a trial counted as correct output when three things held: its final
code compiled, it ran with exit status 0, and a person judged its stdout to
match the target-language original's stdout, reported timings aside. The
sources give no rule beyond that. "Did not match" (p. 5) and "significantly
different" (pp. 6 and 7) are the only criteria stated. No numeric tolerance,
masking rule, or PASS/FAIL reading is given (Open 1). The paper does not say
why each N/A trial failed: no build, a failed run, or wrong output (Open 1).

### 2. Denominators

| Direction | Metric | Published (page, section) | Fraction | Denominator |
| --- | --- | --- | --- | --- |
| OMP -> CUDA | Correct output | 80% (p. 1, abstract; p. 6, V-B) | 32/40 | all trials |
| OMP -> CUDA | Within 10% or faster | 78.1% (p. 6, V-B) | 25/32 | correct trials |
| OMP -> CUDA | First try | 65.6% (p. 6, V-B) | 21/32 | correct trials |
| OMP -> CUDA | Sim-T >= 0.6 | 40.6% (p. 6, V-B) | 13/32 | correct trials |
| CUDA -> OMP | Correct output | 85% (p. 1, abstract; p. 6, V-C) | 34/40 | all trials |
| CUDA -> OMP | Within 10% or faster | 61.8% (p. 6, V-C) | 21/34 | correct trials |
| CUDA -> OMP | First try | 55.9% (p. 6, V-C) | 19/34 | correct trials |
| CUDA -> OMP | Sim-T >= 0.6 | 47.1% (p. 6, V-C) | 16/34 | correct trials |

How the denominators are known:

- Correct output: Tables VI and VII have 32 and 34 rows with values out of 40
  (recount), which give 80% and 85% exactly.
- The other six: for each published value, every fraction with a denominator
  of at most 40 that rounds to it has denominator 32 (OMP to CUDA) or 34 (CUDA
  to OMP). Those are the direction's correct-trial counts. The one exception,
  8/17 for 47.1%, matches no trial count in the paper. The text agrees: "Of
  these successful generations" (p. 6, V-B, within 10%), "of the successful
  codes" (V-B, Sim-T), and "Among these" (p. 6, V-C, all three). The OMP to
  CUDA first-try sentence says "of the trials" (p. 6, V-B). Still, no fraction
  over 40 gives 65.6%, and 21/32 does.
- Every rate pools the four models, and the paper reports no per-model rate.
  Correct counts per model (OMP to CUDA, CUDA to OMP): GPT-4 7 and 9,
  Codestral 9 and 8, WizardCoder 9 and 10, DeepSeek-Coder-V2 7 and 7.
- The abstract and the summary round the within-10% values to 78% and 62%
  (p. 1; p. 7, Sec. VI).
- These eight values are the ones in the bible's LASSI results table (Source
  Papers).

### 3. Runtime ratios

Paper:

- Ratio is the runtime of the original code in the target language divided by
  the runtime of the generated code (p. 5, Sec. V-A). A value above 1 means
  the generated code is faster.
- The reference runtimes are in Table IV (p. 5): each is the mean of three
  runs on an A100, using the same compilers, flags, and run arguments as the
  pipeline (p. 5, Sec. IV).
- The generated runtimes are the Runtime column of Tables VI and VII. The
  text calls them average runtimes (p. 6, V-B) and speaks of the "average
  runtime ... over multiple runs" (p. 7, V-D). It does not give the number of
  runs (Open 3).
- The within-10% wording is "within 10% of or at a faster runtime" than the
  original (p. 1; p. 7, Sec. VI), "within 10% or faster" (p. 6, V-B), and
  "near or below" (p. 6, V-C). The exact inequality is not written. The two
  plain readings are Ratio >= 0.9, and the generated time at most 1.1 x the
  reference (Ratio >= 1/1.1, about 0.909). They count the same on every
  printed Ratio, because none lies between the two thresholds (Open 2).
- The paper does not say how a runtime was taken (process wall clock or the
  program's own timing print), for the references or the generated code
  (Open 3).

Notebook:

- It records no program runtime. Its only time is the wall time of the
  generation loop: from just before the first generation request to the end
  of the loop. That covers model requests, compiles, runs, and Ollama unloads,
  and leaves out the baseline and the two context requests (cell fdb5b82f,
  lines 77, 140, and 165).
- A program run has no time limit (cell 976745b6, line 53).
- So the paper's runtimes were taken outside the pinned notebook.

Recount:

- Table IV's runtime divided by the Runtime column gives every printed Ratio
  to 4 decimals, except GPT-4's OpenMP to CUDA atomicCost (printed 0.5854,
  recomputed 0.9573). The bible already records that error (Source Papers,
  fidelity findings).
- The published within-10% values do not follow from the tables.
  - OMP to CUDA: the printed Ratios give 23/32 (71.9%) at 0.9 or 1/1.1, and
    24/32 (75.0%) with atomicCost recomputed. The paper gives 78.1% (25/32).
    Printed-Ratio thresholds from 0.636 to 0.794 give 25/32.
  - CUDA to OMP: 20/34 (58.8%) at 0.9 or 1/1.1, printed or recomputed. The
    paper gives 61.8% (21/34). Only a threshold of 0.89, between WizardCoder's
    randomAccess (0.8896) and Codestral's (0.8907), gives 21/34.
  - No single threshold gives both published values (Open 4).

### 4. First try

Paper:

- Self-corr is the number of self-correction iterations: each re-prompts the
  model after a compile error or an execution error.
- Self-corr 0 means the generated code compiled and ran on the first try (p. 6,
  Sec. V-A, continued from p. 5).
- First try is the share of correct trials with Self-corr 0 (p. 6, V-B and
  V-C; denominators above).

Notebook:

- `self_correction_counter` starts at -1 and goes up by 1 at the top of each
  pass of the loop. The first generation is therefore 0, and each correction
  request adds 1.
- Compile-error and run-error corrections share the counter (cell fdb5b82f,
  lines 48, 80 to 81, and 88 to 108).
- Its final value is written to the metadata file as the self-correction
  iteration count (line 173).
- Self-corr is therefore the number of correction requests, which is the
  attempt count minus one. That is final.corrections in the Result Record.

Recount: OMP to CUDA has 21/32 correct trials with Self-corr 0, the published
65.6%. CUDA to OMP has 18/34 (52.9%), where the paper gives 55.9% (19/34)
(Open 4).

### 5. Sim-T >= 0.6

Paper: Sim-T tokenizes both codes and compares them with Ratcliff-Obershelp
sequence matching. The result is a ratio in [0, 1], and "values over 0.6"
mean high similarity (p. 5, Sec. V-A). The share counts 0.6 "or higher"
(p. 6, V-B) over the correct trials. It compares the generated code with the
target-language original (p. 5, V-A).

Notebook:

- After the loop, the notebook compares two texts: the target-language
  reference, read in text mode, and the final attempt's code block after fence
  stripping (cell 75a66833, lines 44 to 54; cell fdb5b82f, lines 110 to 113
  and 150 to 158).
- It computes two token similarities, both difflib SequenceMatcher ratios over
  token lists. One uses Python `tokenize` tokens, skipping errors. The other
  uses tiktoken cl100k_base token ids (cell a949bc43, lines 21 to 67).
- It prints both and writes both to the metadata file at two decimals (cell
  fdb5b82f, lines 150 to 155 and 167 to 169). It also writes Sim-L, the count
  of matching stripped lines, each match used once, over the longer line
  count (cell a949bc43, lines 69 to 90; cell fdb5b82f, lines 157 and 171).
- The paper does not say which token similarity its Sim-T column is (Open 5).
  `lassi/scoring/similarity.py` computes `sim_t` from Python tokenize, as the
  bible's quirk table (Sim-T row) says.

Recount:

- OMP to CUDA: 8/32 (25.0%) at >= 0.6. The paper gives 40.6% (13/32).
- CUDA to OMP: 15/34 (44.1%) at >= 0.6 and 14/34 (41.2%) at > 0.6. The paper
  gives 47.1% (16/34).
- The thresholds that give the published shares are 0.54 (OMP to CUDA) and
  0.59 (CUDA to OMP), in steps of 0.01.
- The generated codes are not published (README), so neither similarity can
  be recomputed (Open 4, Open 5).

### 6. The trial with Self-corr 34

Codestral's CUDA to OpenMP pathfinder has Self-corr 34, Runtime 0.2659, and
Ratio 2.7288 (Table VII, p. 7), and it counts as correct.

In the pinned notebook's loop, an attempt runs only while the counter is at
most 7. A compiling attempt past that ends the loop without running (cell
fdb5b82f, lines 80 to 136). A loop that reaches 34 therefore implies three
things:

- Attempts 8 to 33 each failed to compile.
- Attempt 34 compiled and never ran.
- Every attempt that did run failed its run, since a clean run ends the loop.

The metadata file's generated stdout would then come from a failed run.
With no run at all, the notebook raises (bible, quirk table, loop row). So the
paper's verdict and runtime for this trial did not come from the pinned
notebook's stored output. Both committed notebooks are later than the paper's
v1 (f55bf0c on 2024-07-26 and 1fa2716 on 2024-09-23, against 2024-06-30), and
their loop cells are identical (Open 6).

### 7. What the notebook stores per trial

The notebook writes three things per trial:

1. The generated code file, `<directory>/<file_name>.<ext>`. The directory
   comes from the LLM source, the model id with '/' written '--', and the
   direction. The file name comes from the input file name, the target
   language, and the experiment label (cell 75a66833, lines 57 to 59).
   - The file is rewritten on every attempt, so it holds the last attempt's
     code block after fence stripping (cell fdb5b82f, lines 110 to 120).
   - A rerun with the same app, direction, model, and label overwrites it.
2. The executable of the last successful build, beside the code file (cell
   e0fe91ba).
3. One metadata text file per trial,
   `metadata_<file_name>__<MM-DD-YYYY_tHH_MM_SS>.txt`, in the same directory.
   Its timestamp is taken when the loop ends (cell fdb5b82f, lines 142 to 143
   and 181 to 184), and a rerun adds a new file. It holds:
   - Configuration: application, direction key, source and target
     extensions, LLM source, model, prompt index, the code file path template,
     the direction's system prompt, and the translation prompt (cell 75a66833,
     lines 62 to 73).
   - Metrics: the date-time, the loop's wall time in seconds (2 decimals), the
     tokenize and tiktoken token similarities and the line similarity (2
     decimals each), and the self-correction count (cell fdb5b82f, lines 160
     to 173).
   - When execution is on: the reference stdout and the generated stdout
     (cell fdb5b82f, lines 174 to 178). The header of the reference stdout
     names it the source's output, but the notebook ran only the
     target-language reference (lines 27 to 38). The generated stdout comes
     from the last attempt that ran, and it is stale when a later attempt did
     not run (lines 134 and 177 to 178).

The notebook does not store the prompts sent, the context summary or the
source description, the model replies, earlier attempts' code, compiler or
run stderr, exit statuses, token counts, which attempt the stored stdout came
from, whether the final attempt ran, any program runtime, or any verdict on
correctness. The notebook prints or displays some of these; it keeps none.

## Open

The sources leave these open. None is guessed here.

1. What counted as a stdout match beyond "reported timings aside": the
   tolerance and which lines were compared. Also why each N/A trial failed.
2. The exact within-10% inequality. Ratio >= 0.9 and Ratio >= 1/1.1 count the
   same on the printed data.
3. How runtimes were taken (wall clock or program timing lines), and how many
   runs the generated code's mean used.
4. What the non-reproducing percentages were computed from: within 10% (both
   directions), Sim-T >= 0.6 (both), and first try (CUDA to OMP).
5. Which of the notebook's two token similarities the Sim-T column is.
6. Which pipeline version produced the paper's results, and how Codestral's
   CUDA to OpenMP pathfinder (Self-corr 34) got a verdict and a runtime.
7. How a trial that never reached a clean run was stopped. The pinned loop has
   no cap on compile errors, and the paper states no stopping rule.

## Confidence

- High: correct output was a manual stdout judgment of final code that
  compiled and ran cleanly, against the target-language original. Also high:
  the denominators (all trials for correct output, correct trials for the
  other three), the Ratio definition, Self-corr and first try, and what the
  notebook stores. These rest on the paper's text, on arithmetic that allows
  one denominator, and on code read directly.
- Established by recount, cause unknown: five of the eight published
  percentages do not follow from Tables VI and VII. They are within 10% (both
  directions), Sim-T >= 0.6 (both), and CUDA to OMP first try.
- The items under Open cannot be answered from these sources.

## Consequences for the plan

- P2.7:
  - `correct` (automated oracle, planning decision) sits next to the paper
    criterion, which stays None and is labeled manual inspection.
  - `first_try` is final.corrections == 0 on a correct trial.
  - `sim_t` compares the last attempt's target file with the text-mode
    reference, as planned. Its paper comparison carries the Open 5 caveat.
  - `within_10pct` stays None, since no timing profiler exists. Its
    inequality stays open until P10.
- P2.8:
  - A rate shown next to a paper value uses the paper's denominator: all
    trials for correct output, and correct trials for within 10%, first try,
    and Sim-T >= 0.6.
  - `assets/scoring/lassi-paper.yaml` needs the published values with their
    fractions, plus the recounts. Owner-queue draft A covers which of the two
    the Acceptance Criteria compare with.
- P2.G: the review packet can cite the definitions table and the recount.
- P10: the timing rule (Open 2 and 3) must be fixed before `within_10pct` is
  computed.
- A tiktoken Sim-T under its own name is a later-phase note, if the owner
  wants it (owner-queue draft B). P2's scope is fixed.

## Proposed bible edit

Evaluation Protocol: insert this subsection after the metrics table and before
Acceptance Criteria.

```
### LASSI Paper Metrics

The LASSI paper's metrics, from arXiv:2407.01638 (v1 and v2 have the same tables and percentages) and the pinned notebook at 74b4681 (task P2.4; evidence in plans/spikes/p2-lassi-metrics.md). A paper trial is one run of one app, direction, and model. The paper ran one trial per scenario: 10 apps x 4 models, or 40 trials per direction.

| Metric | Paper definition | Denominator |
| --- | --- | --- |
| Correct output | The final code compiled, ran with exit status 0, and a person judged its stdout to match the stdout of the HeCBench original in the target language, reported timings aside (paper pp. 4 to 7, Sec. III-D, V-A, V-B, V-D). The notebook records both stdouts and no verdict | All trials of the direction: 32/40 OMP -> CUDA, 34/40 CUDA -> OMP |
| Within 10% or faster | Ratio = the target-language original's runtime (Table IV, the mean of three A100 runs) / the generated code's runtime (a mean over an unstated number of runs); the share of trials where the generated code is at most 10% slower | Correct trials: 25/32, 21/34 |
| First try | Self-corr = 0. Self-corr is the notebook's final correction count, one counter for compile and run corrections; it equals final.corrections | Correct trials: 21/32, 19/34 |
| Sim-T >= 0.6 | Token similarity of the final code against the target-language original, 0.6 or higher | Correct trials: 13/32, 16/34 |

For correct output, the counts are the rows of Tables VI and VII that have values. For the other three metrics, every fraction with a denominator of at most 40 that gives a published percentage has the direction's correct-trial count as its denominator (47.1% is also 8/17, which matches no trial count), and the paper's text says the same. Every rate pools the four models; the paper reports no per-model rate. Correct counts per model (OMP -> CUDA, CUDA -> OMP): GPT-4 7 and 9, Codestral 9 and 8, WizardCoder 9 and 10, DeepSeek-Coder-V2 7 and 7.

Five of the eight published percentages do not follow from the paper's Tables VI and VII. The recounts below are counts read from those tables, not measurements:

| Direction | Metric | Published | Recount from Tables VI and VII |
| --- | --- | --- | --- |
| OMP -> CUDA | Within 10% or faster | 78.1% (25/32) | 23/32 from the printed Ratios; 24/32 with GPT-4's atomicCost Ratio recomputed |
| OMP -> CUDA | Sim-T >= 0.6 | 40.6% (13/32) | 8/32 |
| CUDA -> OMP | Within 10% or faster | 61.8% (21/34) | 20/34 |
| CUDA -> OMP | First try | 55.9% (19/34) | 18/34 |
| CUDA -> OMP | Sim-T >= 0.6 | 47.1% (16/34) | 15/34 |

Correct output (both directions) and OMP -> CUDA first try recount to their published values. Neither 0.9 nor 1/1.1 gives either published within-10% value, and no single Ratio threshold gives both.

Rules for the reproduction:

- The paper criterion for correct output is a manual judgment, so it is never computed: it is None and labeled. The automated oracle gives correct.
- A rate shown next to a paper value uses the paper's denominator. [DESIGN]
- The notebook keeps, per trial, the last attempt's code, the last successful build, and one metadata file. The file holds the configuration, the direction's system and translation prompts, the generation loop's wall time, both token similarities and Sim-L at two decimals, the correction count, the reference stdout, and the stdout of the last attempt that ran. The notebook records no program runtime, no verdict on correctness, and no per-attempt history, so the paper's runtimes and verdicts were taken outside it.

[OPEN], because the sources do not say:

- What else counted as a stdout match (tolerance, which lines), and why each N/A trial failed.
- The exact within-10% inequality (Ratio >= 0.9 and Ratio >= 1/1.1 count the same on the printed Ratios), how many runs the generated code's mean used, and how runtimes were timed. Settle these before within_10pct is computed (P10).
- Which of the notebook's two token similarities the Sim-T column is: Python tokenize, or tiktoken cl100k_base. The faithful sim_t uses Python tokenize (quirk table, Sim-T row).
- What the percentages that do not recount were computed from.
- Which pipeline version produced the paper's results. Both committed notebooks postdate the paper's v1. Codestral's CUDA -> OMP pathfinder (Self-corr 34, correct, with a runtime) cannot come from the pinned notebook's stored output, because its execution gate leaves attempt 34 unrun.
- How a trial that never reached a clean run was stopped. The pinned loop has no cap, and the paper names no stopping rule.
```

Source Papers, LASSI, fidelity findings (optional): add one bullet.

```
- Five of the eight results-table percentages do not follow from the paper's Tables VI and VII; the recounts and the paper's definitions are in Evaluation Protocol, LASSI Paper Metrics.
```

Decision Log row (newest first), with the count sentence raised by one:

```
| 2026-09-24 | The LASSI paper's metrics join the Evaluation Protocol as LASSI Paper Metrics (task P2.4). Correct output was a manual judgment of stdout against the target-language original's, over all 40 trials of a direction. Within 10% or faster, first try (final.corrections = 0), and Sim-T >= 0.6 are shares of the correct trials (32 OMP -> CUDA, 34 CUDA -> OMP). A rate shown next to a paper value uses the paper's denominator. Five published percentages that differ from recounts of Tables VI and VII are recorded with both values. Six questions the sources leave open are [OPEN] | Spike plans/spikes/p2-lassi-metrics.md read arXiv:2407.01638 v1 and v2 (same tables and percentages) and the pinned notebook at 74b4681. The paper's text ("Of these successful generations", "Among these") and the arithmetic (every fraction with a denominator of at most 40 that gives a published share uses the correct-trial count) agree on the denominators. The notebook records no runtime or verdict, and it computes two token similarities without saying which the paper used, so those questions are named, not guessed. The recounts are read from published tables, not measured |
```

## Independent recount

A second, independent recount read both v1 and v2 of the paper as PDF and as HTML: the PDFs have the same sha256 as above, and all four sources give the same 40 rows for Tables VI and VII. It gave the same values as this spike for all eight percentages, the same N/A lists, and the same per-model counts. It adds one detail: GPT-4's OMP -> CUDA atomicCost row repeats the whole layout row above it (Ratio 0.5854, Sim-T 0.63, Sim-L 0.68, Self-corr 0), so that trial's Sim-T and Self-corr are as doubtful as its Ratio. The bible text as applied (master revision 129) includes this.

## Owner-queue drafts

```
## OQ-021 LASSI Paper Values That Do Not Follow From Its Tables
State: OPEN
Kind: decision
Blocks: none (P2.8 carries both values; the choice applies when a headline metric is marked reproduced or reported with its gap, Acceptance Criteria)
Evidence: plans/spikes/p2-lassi-metrics.md
Question: Five of the eight percentages in the LASSI results table do not follow from the paper's own Tables VI and VII: within 10% or faster in both directions, Sim-T >= 0.6 in both, and CUDA -> OMP first try. Each recount differs from the published value by 1 to 5 trials, and the sources do not say why. When the reproduction marks a headline metric reproduced or reports its gap, which paper value is the reference?
Options: (a) The published percentages only: this matches the paper's claims, but part of a gap may be the paper's own inconsistency, unmarked. (b) Both values side by side, with the gap reported against each and the difference noted: nothing is hidden, but tables get wider and five metrics carry two gaps. (c) The recounts only: consistent with the published rows and the Reporting Rules' recompute-from-raw rule, but departs from the paper's stated headline numbers.
Recommendation: (b). The sources cannot tell which value is right, so neither is picked silently.
Answer:
```

```
## OQ-022 Which Token Similarity Is The Paper's Sim-T
State: OPEN
Kind: decision
Blocks: none
Evidence: plans/spikes/p2-lassi-metrics.md (Findings 5); lassi/scoring/similarity.py
Question: The pinned notebook computes and stores two token similarities, one over Python tokenize tokens and one over tiktoken cl100k_base ids. The paper names one Sim-T and does not say which. The faithful sim_t uses Python tokenize (quirk table). The paper's generated codes are unpublished, so no data can settle this. P2's scope is fixed, so a second measure would come in a later phase.
Options: (a) Keep sim_t as the faithful Sim-T and label every paper comparison of Sim-T "tokenizer not stated by the paper": no new dependency, and the comparison stays caveated. (b) As (a), and add sim_t_tiktoken under its own name in a later phase: both candidates are reported, but this adds the tiktoken dependency and its cl100k_base file, which must be cached under the scratch root before offline use on alpha01. (c) Make the tiktoken similarity the faithful Sim-T: this changes the quirk table's Sim-T row and the P1.9 replay comparison, and no evidence favors it.
Recommendation: (a). A second measure cannot resolve a question that no published data can settle, and (a) keeps the documented design.
Answer:
```
