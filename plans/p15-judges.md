# P15 Judges

Branch `p15-judges`. Base: p16-adversarial at e43a1e0 (P16 not merged; its gate waits on OQ-067 and OQ-069). The owner opened the phase on 2026-10-25 without the DF dialect (OQ-088, option (a); P11 stays deferred, OQ-066). Code references name files as they stand at e43a1e0.

## Scope

P15 builds what the P15 row of the bible's Build Roadmap names: the Judge interface, the equivalence and efficiency judges, their rubrics, and calibration reports (Agents And Judges, the Judge interface, Judge Types, and Judge rules; Component Interfaces, the Judge row; Repository Layout, `lassi/judges/` and `assets/rubrics/`; Evaluation Protocol, the Judges and adversaries row). The equivalence judge is LASSI-EE's LLM-as-a-Judge (Source Papers, LASSI-EE; Oracles, the llm_judge row): P9, planned beside this phase on p9-ee, consumes the interface and that judge from P15.2. Out: the readability and plan-quality judges (Judge Types), adversarial review and judge probing (Adversarial Setups, items 2 and 4), and judge terms in rewards (Training Module, Reward Function; P7 and P8), which a later plan takes; P11.

Gate: the row's Gate column reads "Efficiency judge calibration reported on MI300X measurements", blocked by P9. Under OQ-088 the owner chose "Calibrate on CPU measurements": part (a), run as P15.G, reports the efficiency judge's calibration on host-CPU measurements (runtime, and energy where RAPL is readable) and the equivalence judge's on oracle outcomes; part (b), P15.12, reports the efficiency judge on MI300X measurements and stays BLOCKED. P15.1 records this in the row with a Decision Log entry before any gate evidence exists.

Hosts: P15.1 to P15.5 run on Windows with fakes and replay recordings; nothing is served. The calibration runs need alpha01 in an owner window (Agent Rule 17) with judges served on RNGD through openai_compat (Model Serving, Serving Rules; Agent Rule 8: `furiosa-smi ps` and `furiosa-smi status`, a card at 0.00 GiB, port 8123, `/v1/models`), never during the owner's sweep of 2026-10-25, 20:00 to 02:00. The owner's models are furiosa-ai/Llama-3.3-70B-Instruct-FP8-dynamic and furiosa-ai/gpt-oss-120b at v2026.4 (OQ-087), one served at a time.

Constraints for every task:

- Agent Rule 14 and the Judge rules: a judge never sees or changes an oracle outcome, an alignment, a binary_io statistic, or a profiler measurement in predict mode; every judged value is stored and shown with [JUDGED]; comments are stripped from candidates before judging; a rubric change invalidates its calibration.
- Agent Rule 5: rubrics are written and checked on synthetic replay fixtures and on development items only (tt-pairs-v0 is a development set, OQ-083 (b); P17.31); a calibration set is frozen, with the rubrics' sha256, before its first judge call, and nothing in a frozen set or a rubric changes after it (P16's Freeze decision is the pattern). A change is measured only on a fresh set.
- Agent Rules 1 and 2: calibration measurements come from logged runs with provenance; ttsim and every other simulator never give efficiency measurements; host-CPU numbers name the host CPU, never a GPU. Agent Rule 6: candidates run only through the sandboxed native executor. Agent Rule 12: keys only by environment variable. Agent Rule 13 holds for any RNGD use.
- Agent Rules 3 and 10: judges are generic components; recipes go under projects/judges/; every judge choice (backend, model, rubric, mode, use, sampling, reasoning_effort for gpt-oss, OQ-080) is required with no default and enters the resolved recipe and its hash.
- Branch route: PHASE-NOTES All Phases, Stacked phase branches, and P16's Branch route; owner-queue numbers follow step 6. Tests stay off P17's CI list (no device layer). Practice as in plans/runs/p4-retrospective.md: one review round and one commit audit per task; limits text opens with "known cases, not complete".

Decisions taken at planning (the named task records each in the bible with a Decision Log entry):

- Evidence a judge sees (P15.1, P15.2). Equivalence: the source and the candidate (comments stripped) and both stdouts on identical inputs, with PASS/FAIL where the item lists passfail, as LASSI-EE gives; for output-file items the stdouts alone, never the files or their statistics, so the calibration measures the judge, not the oracle. Efficiency predict mode: the source and the candidate only.
- Model families (P15.1). A tracked table maps each model id to its family; a judge whose family equals the generator's is refused in a run, and in calibration such a trial leaves that judge's set and is counted. Bindings follow the bible's bindings block (equivalence on the Llama model, efficiency on gpt-oss).
- Calibration statistics (P15.4). Equivalence: the four counts against the oracle outcome, agreement, false-VALID rate (VALID among oracle failures), and false-INVALID rate, each with its Wilson 95% interval. Efficiency: sign accuracy with its Wilson interval, and Spearman's rank correlation per target with a Fisher z interval, since a correlation has no Wilson interval (the row's wording is edited).
- Measurement noise (P15.8). Each program of a pair runs n = 10 times, interleaved (PROJECTED cost); a measured change has a sign only when the two interquartile ranges do not overlap, else it is "no measurable change", counted and left out of sign accuracy but kept in Spearman.

Owner queue (filed 2026-10-25): OQ-090, the calibration sets (answered: (a), development and synthetic items only, measured on alpha01's EPYC), blocked P15.7; OQ-091, the reward admission threshold (Risks And Questions, open question 14), blocks P15.11 only; OQ-092, the alpha01 batch (one RAPL probe, the efficiency measurements, two RNGD judge runs). OQ-089 binds the equivalence judge to llama-3.3-70b and the efficiency judge to gpt-oss-120b, with gpt-oss-120b generating.

## Tasks

### P15.1 The Judge interface, rubrics, and judge records
- Bible: Agents And Judges (the Judge interface; Judge rules), Component Interfaces (the Judge row; capabilities), Result Record, Project Recipes (the bindings block's judges lines), Build Roadmap (the P15 row), Readability Standards, Agent Rules 3, 10, 14.
- Accept: interfaces.py states a JudgeInput (candidate files, source files, artifacts as text by name, measurements as finite numbers by name) with no field for an oracle outcome, alignment, or statistic, and a Verdict whose verdict is one of the rubric's declared values, score a finite float in [0, 1], named judged values finite, rationale plain ASCII; anything else is refused naming the field (tests). A rubric loader (lassi/judges/) reads assets/rubrics/<name>/ (versioned prompt and output schema), refuses a missing file or unknown key, and its sha256 enters the resolved recipe and the hash (test: one changed byte changes the hash). The judges section binds a registered Judge per key with its backend settings, sampling, rubric, use (screen or metric), and mode where the judge declares modes, each required; `use: reward` is refused naming judges.<kind>.use until P15.11 records a threshold. The family table loads and an unknown model id is refused when a judge is bound. A JudgeRecord (kind, rubric and sha256, model and family, mode, verdict, score, values, rationale) round-trips in trial.json, an old trial.json loads with none, trial.md marks it [JUDGED], and Parquet gains a judgements table. Bible edits, the P15 row per OQ-088 included, with a Decision Log entry.
- Files: lassi/core/interfaces.py, lassi/core/record.py, lassi/core/recipe.py, lassi/core/trial_md.py, lassi/core/parquet.py, lassi/judges/, assets/rubrics/, assets/models/families.yaml (proposed), tests/core/, tests/judges/, docs/BIBLE.md.
- Host: Windows. Depends: none.

### P15.2 The equivalence judge
- Bible: Judge Types (Equivalence), Source Papers (LASSI-EE, the judge line), Oracles (llm_judge), Component Interfaces (LLMBackend; ContextExceeded), Model Serving (Serving Rules), Agent Rules 6, 12, 14.
- Accept: a registered Judge `equivalence` asks its LLMBackend with rubric equivalence-v1, written here from the paper's description (LASSI-EE's prompts are not released), and returns VALID or INVALID with a score and rationale. A comment stripper (lassi/judges/) handles C, C++, CUDA, and HIP, keeping string, character, and raw-string literals and line splices (fixture tests), and refuses another language by name. A test over the rendered messages shows no oracle outcome, alignment, statistic, held-out input, or reference output file reaches a prompt. The reply is read strictly (one fenced JSON block); a malformed reply is asked again once, then gives no verdict and a counted malformed reply, never a guess; ContextExceeded gives no verdict, recorded. Tests with synthetic replay recordings (reply text only) and fakes.
- Files: lassi/judges/, assets/rubrics/equivalence-v1/, tests/judges/, tests/fixtures/judges/, docs/BIBLE.md.
- Host: Windows. Depends: P15.1.

### P15.3 Judges in runs
- Bible: Project Recipes (Notes; the bindings block), Component Interfaces (Stage; LLMBackend model_check), Model Serving (Serving Rules), Result Record, Agent Rule 14.
- Accept: the runner stops refusing `judges` (refine and agents stay refused; test); before any directory exists it builds each bound judge, runs its backend's model check, refuses a judge whose family equals the generator's, and writes both families into provenance. A registered stage `judge`, listed after oracle, judges each trial's standing attempt that ran for every judge with use metric, records JudgeRecords, and changes no stage, alignment, or score (test with fakes). Screening stages stay P9's (LASSI-EE). Bible edits with a Decision Log entry.
- Files: lassi/core/runner.py, lassi/core/stages.py or a new stage module, tests/core/, docs/BIBLE.md.
- Host: Windows. Depends: P15.1, P15.2.

### P15.4 Calibration records, the calibration tool, and its metrics
- Bible: Agents And Judges (the calibration rule), Evaluation Protocol (the Judges and adversaries row; Run Metrics rules), Readability Standards, Agent Rules 1 and 14.
- Accept: a tool (proposed `lassi calibrate <judge recipe> <set>`) asks a judge for every entry of a frozen set and writes calibration records (rubric sha256, judge model and family, set sha256, provenance) beside the planning decision's statistics, in Markdown and Parquet, names, counts, rates, and notes only; judged values carry [JUDGED] and measured ones [MEASURED] with their provenance. It refuses a set whose freeze record does not match, and reports a record whose rubric sha256 differs from the current rubric as invalid (tests). Known-answer tests: Wilson bounds, Spearman with ties by average rank, the Fisher z interval, exclusions counted.
- Files: lassi/judges/calibration.py, lassi/analysis/, lassi/cli.py, tests/judges/, tests/analysis/, docs/BIBLE.md.
- Host: Windows. Depends: P15.1.

### P15.5 The efficiency judge, explain and predict modes
- Bible: Judge Types (Efficiency; the two modes), Source Papers (LASSI-EE, Stage 4), Agents And Judges (Judge rules), Agent Rule 14.
- Accept: a registered Judge `efficiency` with rubric efficiency-v1 and mode required. Predict mode sees the source and the candidate only (test: no measurement reaches a prompt) and returns judged values for the predicted relative change of energy, power, and runtime. Explain mode gets measurements and writes the diff report, every number quoted from the measurements; its records never enter a calibration (test). Strict reply reading and the stripper as P15.2. Tests with replay recordings.
- Files: lassi/judges/, assets/rubrics/efficiency-v1/, tests/judges/, tests/fixtures/judges/, docs/BIBLE.md.
- Host: Windows. Depends: P15.1, P15.2.

### P15.6 Spike: RAPL energy on the host CPUs
- Bible: Host Facts, Agent Rules 1, 7, 17; OQ-088.
- Accept: plans/spikes/p15-rapl.md records, with commands and outputs, whether this account can read RAPL energy (powercap domains, energy_uj permissions, counter range, update interval) in the LASSI WSL distro and on alpha01's EPYC; the result enters Host Facts tagged [MEASURED] with a Decision Log entry. If P9's plan holds the same probe, one probe serves both.
- Host: the LASSI distro (read-only, local); alpha01, one read-only `rx exec` in an owner window (OQ-092). Depends: none.

### P15.7 The calibration sets and their freeze
- Bible: Benchmark Suites (split rules), Agents And Judges (calibration on held-out measured trials), Agent Rule 5.
- Accept: under OQ-090's answer, manifests name each equivalence entry (run id, trial id, trial.json sha256, oracle outcome, generator family) and each efficiency pair (both programs, inputs, build), disjoint from every trial a rubric was checked on (test); a freeze record holds the sets', rubrics', and judge recipes' sha256, and a fast test fails on any difference. Judge recipes under projects/judges/.
- Files: assets/calibration/ (proposed), projects/judges/, tests/judges/, docs/BIBLE.md.
- Host: Windows; reading run trees on alpha01 needs the owner's window. Depends: P15.2, P15.5. State: OWNER (OQ-090).

### P15.8 Host-CPU measurements of the efficiency pairs
- Bible: Component Interfaces (Profiler), Sandbox, Execution Backends (native), Reporting Rules, Agent Rules 1, 2, 6.
- Accept: each pair runs through the sandboxed native executor with the timing profiler, and with the host-CPU energy source of P9 where P15.6 finds RAPL readable, under the planning decision's noise rule; load average and CPU frequency governor are recorded; results/p15-cpu-measure/ holds provenance.json (rx pull, or the local writer if OQ-090 picks the workstation) and summary.md, each value [MEASURED] on the named host CPU.
- Host: alpha01 in an owner window: `rx doctor`, then a big job (`rx job start --big`, `rx job wait`), `rx pull`. Depends: P15.6, P15.7. State: BLOCKED (owner window on alpha01; OQ-092).

### P15.9 Equivalence calibration on oracle outcomes
- Bible: Model Serving (Serving Rules), Agents And Judges, Evaluation Protocol (the Judges and adversaries row), Agent Rules 8, 13, 14.
- Accept: the frozen equivalence judge on the Llama model over the frozen set, through P15.4's tool, with the serving record; results/p15-calib-equivalence/ holds provenance.json, summary.md, and the tables: verdicts [JUDGED] beside [MEASURED] oracle outcomes; the server stopped and the cards read 0.00 GiB afterwards.
- Host: alpha01 in an owner window with RNGD, outside 20:00 to 02:00 on 2026-10-25: a big job serving the model and running the tool, `rx pull`. Depends: P15.4, P15.7. State: BLOCKED (owner window; RNGD).

### P15.10 Efficiency calibration on host-CPU measurements
- Bible: as P15.9, plus Judge Types (predict mode).
- Accept: the frozen efficiency judge in predict mode on gpt-oss-120b (reasoning_effort set) over the frozen pairs; results/p15-calib-efficiency/ holds sign accuracy and Spearman per target against P15.8's measurements, with provenance and the exclusions counted.
- Host: alpha01 in an owner window with RNGD (one arm at a time, after P15.9 or in another window). Depends: P15.4, P15.7, P15.8. State: BLOCKED (owner window; RNGD).

### P15.11 The reward admission threshold
- Bible: Agents And Judges (Judge rules), Training Module (Reward Function), Risks And Questions (open question 14).
- Accept: the owner's answer to OQ-091 is recorded in the bible with a Decision Log entry; `use: reward` is accepted only for a judge whose calibration record passes it (tests), with a bounded weight.
- Host: Windows. Depends: P15.4. State: OWNER (OQ-091).

### P15.12 Gate part (b): efficiency calibration on MI300X measurements
- Bible: Build Roadmap (the P15 and P9 rows), Agents And Judges, Evaluation Protocol.
- Accept: the frozen efficiency judge's calibration reported on MI300X measurements from P9's rocm_smi runs, per target, beside part (a)'s.
- Host: alpha01 MI300X (devices class enabled by the owner), RNGD. Depends: P15.10. State: BLOCKED (MI300X access, OQ-040; P9's measurements).

### P15.G Phase gate, part (a)
- Bible: Build Roadmap (the P15 row as P15.1 records OQ-088), AGENTS.md Phase Gate.
- Accept: the gate as the row states it after P15.1: results/p15-gate/ cites P15.9's and P15.10's evidence with provenance: the equivalence judge's calibration on oracle outcomes and the efficiency judge's on host-CPU measurements, each judged value [JUDGED]. A pass opens the pull request `P15 Judges` for part (a); the phase row stays ACTIVE with P15.12 BLOCKED, as P17 did (AGENTS.md Work Order). A fail adds fix tasks under the freeze.
- Host: Windows for the packet; the evidence comes from P15.9 and P15.10. Depends: P15.9, P15.10.
