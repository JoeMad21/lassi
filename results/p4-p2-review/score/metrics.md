# Run metrics

One table per arm and direction (bible Evaluation Protocol). Every value comes from the lassi score profile's components and the trial records; nothing is rescored. Intervals are Wilson 95% over each row's own count: compile, run, correct, cap-hit, and fence-quirk rates cover the trials of the arm and direction that reached a model call, a trial that ended at the baseline being excluded and counted in each note; first try and the two Sim-T >= 0.6 rows cover the correct trials, the paper's denominator; pass@k is the mean over scenarios of each scenario's unbiased pass@k, with its interval over the scenario count, and a scenario with no scored trial is left out and named in the note. A trial whose component is None is excluded and counted in the row's note. PLACEHOLDER marks a value not yet measured. The paper columns give the recount read from the paper's tables, which is the reference value, and the paper's published value, shown for reference only (OQ-021), each with the Wilson 95% interval of the paper's own count and denominator; the B0 criterion line gives the interval around the paper's B0 model count. These are paper values, not measurements, a compile-stage reproduction shows none, and no table marks a headline metric reproduced or computes its gap (the P10 gate applies the test of OQ-032). The paper does not state which tokenizer its Sim-T used ([OPEN], OQ-022, OQ-031): one Sim-T row compares the faithful Python-tokenize sim_t and the other the tiktoken cl100k_base sim_t_tiktoken, each formatted to two decimals, as the notebooks store them, while each Score keeps the unrounded value; both stand beside the paper's one Sim-T recount, and neither is marked reproduced.

## furiosa-ai--Llama-3.1-8B-Instruct cuda-omp

Trials: 10 in 10 scenario(s), n = 1 per scenario.
Device: not recorded.
B0 criterion (a paper value, not a measurement; it applies only to arm B0's pass@1, and no verdict is computed here): B0 pass@1 inside WizardCoder 10/10 = 1.000 (Wilson 95% 0.722 to 1.000); bible Evaluation Protocol, Acceptance Criteria.

| Metric | Value | Count | Wilson 95% | Paper published (for reference only) | Paper recount (reference) | Paper cite | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| compile_rate | 0.500 | 5/10 | 0.237 to 0.763 | - | - | - | compiled = 1, over the trials of the arm and direction that reached a model call |
| run_rate | 0.300 | 3/10 | 0.108 to 0.603 | - | - | - | last attempt at S5, over the trials of the arm and direction that reached a model call |
| correct_rate | 0.000 | 0/10 | 0.000 to 0.278 | 34/40 = 0.850 (Wilson 95% 0.709 to 0.929) | 34/40 = 0.850 (Wilson 95% 0.709 to 0.929) | Evaluation Protocol, LASSI Paper Metrics: definitions table (Correct output); recount as published | correct = 1 (automated oracle), over the trials of the arm and direction that reached a model call and whose correct is set |
| cap_hit_rate | 0.700 | 7/10 | 0.397 to 0.892 | - | - | - | cap_hit = 1, over the trials of the arm and direction that reached a model call |
| fence_quirk_rate | 0.000 | 0/10 | 0.000 to 0.278 | - | - | - | fence_quirk > 0, over the trials of the arm and direction that reached a model call; 0 fence-quirk hits in total |
| first_try_rate | - | - | - | 19/34 = 0.559 (Wilson 95% 0.395 to 0.711) | 18/34 = 0.529 (Wilson 95% 0.367 to 0.685) | Evaluation Protocol, LASSI Paper Metrics: definitions table (First try); recount table | not computed: no trial in the population; first_try = 1, over the correct trials, the paper's denominator |
| sim_t_ge_0.6_rate | - | - | - | 16/34 = 0.471 (Wilson 95% 0.315 to 0.633) | 15/34 = 0.441 (Wilson 95% 0.289 to 0.605) | Evaluation Protocol, LASSI Paper Metrics: definitions table (Sim-T >= 0.6); recount table | not computed: no trial in the population; sim_t (.2f) >= 0.6, over the correct trials, the paper's denominator; reads the faithful sim_t (Python tokenize) formatted to two decimals, as the notebooks store it; the Score keeps the unrounded value, and the paper does not state its Sim-T tokenizer ([OPEN], OQ-022, OQ-031) |
| sim_t_tiktoken_ge_0.6_rate | - | - | - | 16/34 = 0.471 (Wilson 95% 0.315 to 0.633) | 15/34 = 0.441 (Wilson 95% 0.289 to 0.605) | Evaluation Protocol, LASSI Paper Metrics: definitions table (Sim-T >= 0.6); recount table | not computed: no trial in the population; sim_t_tiktoken (.2f) >= 0.6, over the correct trials, the paper's denominator; reads sim_t_tiktoken (the notebook's tiktoken cl100k_base similarity) formatted to two decimals, as the notebooks store it; the Score keeps the unrounded value, and the paper does not state its Sim-T tokenizer ([OPEN], OQ-022, OQ-031), so neither Sim-T row is marked reproduced |
| within_10pct_rate | PLACEHOLDER | - | - | 21/34 = 0.618 (Wilson 95% 0.450 to 0.761) | 20/34 = 0.588 (Wilson 95% 0.422 to 0.736) | Evaluation Protocol, LASSI Paper Metrics: definitions table (Within 10% or faster); recount table | PLACEHOLDER: not measured; no timing profiler exists until P10, and the within-10% rule is open (Evaluation Protocol, LASSI Paper Metrics) |
| pass@1 | 0.000 | 0.000/10 | 0.000 to 0.278 | - | - | - | unbiased pass@1 per scenario over its scored trials, mean over 10 scenario(s); Wilson interval over the scenario count |
| pass@3 | - | - | - | - | - | - | not computed: 10 scenario(s) have fewer than k = 3 scored trials (smallest n = 1) |

Stage reached by the last attempt of each trial (none: the trial holds no attempt):

| Stage | Trials |
| --- | --- |
| S0 | 0 |
| S1 | 5 |
| S2 | 0 |
| S3 | 0 |
| S4 | 2 |
| S5 | 3 |
| none | 0 |

Corrections per trial (final.corrections):

| Corrections | Trials |
| --- | --- |
| 0 | 2 |
| 2 | 1 |
| 5 | 7 |
