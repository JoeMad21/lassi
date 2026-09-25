# Run metrics

One table per arm and direction (bible Evaluation Protocol). Every value comes from the lassi score profile's components and the trial records; nothing is rescored. Intervals are Wilson 95% over each row's own count: compile, run, correct, cap-hit, and fence-quirk rates cover every trial of the arm and direction; first try and Sim-T >= 0.6 cover the correct trials, the paper's denominator; pass@k is the mean over scenarios of each scenario's unbiased pass@k, with its interval over the scenario count. A trial whose component is None is excluded and counted in the row's note. PLACEHOLDER marks a value not yet measured. The paper columns give the paper's published value and the recount read from its tables, each with the Wilson 95% interval of the paper's own count and denominator; they are paper values, not measurements, and a compile-stage reproduction shows none. The paper does not state which tokenizer its Sim-T used ([OPEN], OQ-022); the Sim-T row compares the faithful Python-tokenize sim_t, unrounded, while the paper prints Sim-T to two decimals.

## furiosa-ai--Llama-3.1-8B-Instruct cuda-omp

Trials: 10 in 10 scenario(s), n = 1 per scenario.
Device: not recorded.

| Metric | Value | Count | Wilson 95% | Paper published | Paper recount | Paper cite | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| compile_rate | 0.500 | 5/10 | 0.237 to 0.763 | - | - | - | compiled = 1, over all trials of the arm and direction |
| run_rate | 0.300 | 3/10 | 0.108 to 0.603 | - | - | - | last attempt at S5, over all trials of the arm and direction |
| correct_rate | 0.000 | 0/10 | 0.000 to 0.278 | 34/40 = 0.850 (Wilson 95% 0.709 to 0.929) | 34/40 = 0.850 (Wilson 95% 0.709 to 0.929) | Evaluation Protocol, LASSI Paper Metrics: definitions table (Correct output); recount as published | correct = 1 (automated oracle), over all trials of the arm and direction |
| cap_hit_rate | 0.700 | 7/10 | 0.397 to 0.892 | - | - | - | cap_hit = 1, over all trials of the arm and direction |
| fence_quirk_rate | 0.000 | 0/10 | 0.000 to 0.278 | - | - | - | fence_quirk > 0, over all trials of the arm and direction; 0 fence-quirk hits in total |
| first_try_rate | - | - | - | 19/34 = 0.559 (Wilson 95% 0.395 to 0.711) | 18/34 = 0.529 (Wilson 95% 0.367 to 0.685) | Evaluation Protocol, LASSI Paper Metrics: definitions table (First try); recount table | not computed: no trial in the population; first_try = 1, over the correct trials, the paper's denominator |
| sim_t_ge_0.6_rate | - | - | - | 16/34 = 0.471 (Wilson 95% 0.315 to 0.633) | 15/34 = 0.441 (Wilson 95% 0.289 to 0.605) | Evaluation Protocol, LASSI Paper Metrics: definitions table (Sim-T >= 0.6); recount table | not computed: no trial in the population; sim_t >= 0.6, over the correct trials, the paper's denominator; reads the faithful sim_t (Python tokenize), unrounded; the paper does not state its Sim-T tokenizer ([OPEN], OQ-022) and prints Sim-T to two decimals |
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
