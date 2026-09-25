# Score review

## Source run

| Field | Value |
| --- | --- |
| run_id | demo-rngd-cpu-1 |
| status | complete |
| recipe | rngd-cpu |
| recipe_hash | 209990b1d6f8a12c0a8e1426d040ec8ef09570ae65e97065e6168cdd4da53d84 |
| commit | 2f052e6a20436e53f01d18e0fcf42fc33e0e1508 |
| dirty | false |
| executor | native |
| device | - |
| driver | - |

## Profiles

| Profile | File | sha256 |
| --- | --- | --- |
| df-v0 | assets/scoring/df-v0.yaml | 1bd8b1ea88ea4da9b2c655e1c8287fd4f25ee4d7262aa32b200f178748a14ca4 |
| lassi | assets/scoring/lassi.yaml | cebc187cd29f7b9651636dc699a07e3f6be16f8d807b76b984be2c915566462a |

Trials scored: 10. One section per trial, in trial_id order: the trial's facts, one part per attempt, every component of every profile with its note, and the fields the record lacks. Values read as in trial.md: PLACEHOLDER is a value not recorded or not computed (null), flags are true or false, and numbers are shown in full. Diagnostics show stage, severity, code, and place; their messages, the prompts, replies, code, context, and stdout stay in the run tree (OQ-018). The scoring commit, date, and interpreter, and a copy of the run's provenance.json, are in provenance.json beside this page.

## Trial `rngd-cpu/furiosa-ai--Llama-3.1-8B-Instruct/lassi-hecbench-10/cuda-omp/atomicCost/run01`

| Field | Value |
| --- | --- |
| model | openai_compat `furiosa-ai/Llama-3.1-8B-Instruct` |
| final.stage_reached | S1 |
| final.alignment | PLACEHOLDER |
| final.corrections | 5 |
| final.end_reason | correction-cap |
| requests | not recorded |
| reference_run.exit_code | 0 |
| reference_run.hang | false |
| reference_run.stdout_truncated | PLACEHOLDER |
| reference_run.stderr_truncated | PLACEHOLDER |
| reference_run.workdir_incomplete | PLACEHOLDER |

### Attempt 0

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:61:43 |
| compile | error | - | main.cpp:61:46 |
| compile | error | - | main.cpp:61:46 |
| compile | error | - | main.cpp:61:47 |
| compile | error | - | main.cpp:34:43 |
| compile | error | - | main.cpp:34:46 |
| compile | error | - | main.cpp:34:47 |
| compile | error | - | main.cpp:23:43 |
| compile | error | - | main.cpp:23:46 |
| compile | error | - | main.cpp:23:47 |
| compile | error | - | main.cpp:61:43 |
| compile | error | - | main.cpp:61:46 |
| compile | error | - | main.cpp:61:46 |
| compile | error | - | main.cpp:61:47 |
| compile | error | - | main.cpp:34:43 |
| compile | error | - | main.cpp:34:46 |
| compile | error | - | main.cpp:34:47 |
| compile | error | - | main.cpp:23:43 |
| compile | error | - | main.cpp:23:46 |
| compile | error | - | main.cpp:23:47 |
| compile | error | - | main.cpp:61:43 |
| compile | error | - | main.cpp:61:46 |
| compile | error | - | main.cpp:61:46 |
| compile | error | - | main.cpp:61:47 |
| compile | error | - | main.cpp:34:43 |
| compile | error | - | main.cpp:34:46 |
| compile | error | - | main.cpp:34:47 |
| compile | error | - | main.cpp:23:43 |
| compile | error | - | main.cpp:23:46 |
| compile | error | - | main.cpp:23:47 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 1

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:54:54 |
| compile | error | - | main.cpp:54:57 |
| compile | error | - | main.cpp:54:58 |
| compile | error | - | main.cpp:33:46 |
| compile | error | - | main.cpp:33:49 |
| compile | error | - | main.cpp:33:50 |
| compile | error | - | main.cpp:66:54 |
| compile | error | - | main.cpp:66:57 |
| compile | error | - | main.cpp:66:58 |
| compile | error | - | main.cpp:23:46 |
| compile | error | - | main.cpp:23:49 |
| compile | error | - | main.cpp:23:50 |
| compile | error | - | main.cpp:84:54 |
| compile | error | - | main.cpp:84:57 |
| compile | error | - | main.cpp:84:58 |
| compile | error | - | main.cpp:54:54 |
| compile | error | - | main.cpp:54:57 |
| compile | error | - | main.cpp:54:58 |
| compile | error | - | main.cpp:33:46 |
| compile | error | - | main.cpp:33:49 |
| compile | error | - | main.cpp:33:50 |
| compile | error | - | main.cpp:66:54 |
| compile | error | - | main.cpp:66:57 |
| compile | error | - | main.cpp:66:58 |
| compile | error | - | main.cpp:23:46 |
| compile | error | - | main.cpp:23:49 |
| compile | error | - | main.cpp:23:50 |
| compile | error | - | main.cpp:84:54 |
| compile | error | - | main.cpp:84:57 |
| compile | error | - | main.cpp:84:58 |
| compile | error | - | main.cpp:54:54 |
| compile | error | - | main.cpp:54:57 |
| compile | error | - | main.cpp:54:58 |
| compile | error | - | main.cpp:33:46 |
| compile | error | - | main.cpp:33:49 |
| compile | error | - | main.cpp:33:50 |
| compile | error | - | main.cpp:66:54 |
| compile | error | - | main.cpp:66:57 |
| compile | error | - | main.cpp:66:58 |
| compile | error | - | main.cpp:23:46 |
| compile | error | - | main.cpp:23:49 |
| compile | error | - | main.cpp:23:50 |
| compile | error | - | main.cpp:84:54 |
| compile | error | - | main.cpp:84:57 |
| compile | error | - | main.cpp:84:58 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 2

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:54:54 |
| compile | error | - | main.cpp:54:57 |
| compile | error | - | main.cpp:54:58 |
| compile | error | - | main.cpp:33:46 |
| compile | error | - | main.cpp:33:49 |
| compile | error | - | main.cpp:33:50 |
| compile | error | - | main.cpp:66:54 |
| compile | error | - | main.cpp:66:57 |
| compile | error | - | main.cpp:66:58 |
| compile | error | - | main.cpp:23:46 |
| compile | error | - | main.cpp:23:49 |
| compile | error | - | main.cpp:23:50 |
| compile | error | - | main.cpp:84:54 |
| compile | error | - | main.cpp:84:57 |
| compile | error | - | main.cpp:84:58 |
| compile | error | - | main.cpp:54:54 |
| compile | error | - | main.cpp:54:57 |
| compile | error | - | main.cpp:54:58 |
| compile | error | - | main.cpp:33:46 |
| compile | error | - | main.cpp:33:49 |
| compile | error | - | main.cpp:33:50 |
| compile | error | - | main.cpp:66:54 |
| compile | error | - | main.cpp:66:57 |
| compile | error | - | main.cpp:66:58 |
| compile | error | - | main.cpp:23:46 |
| compile | error | - | main.cpp:23:49 |
| compile | error | - | main.cpp:23:50 |
| compile | error | - | main.cpp:84:54 |
| compile | error | - | main.cpp:84:57 |
| compile | error | - | main.cpp:84:58 |
| compile | error | - | main.cpp:54:54 |
| compile | error | - | main.cpp:54:57 |
| compile | error | - | main.cpp:54:58 |
| compile | error | - | main.cpp:33:46 |
| compile | error | - | main.cpp:33:49 |
| compile | error | - | main.cpp:33:50 |
| compile | error | - | main.cpp:66:54 |
| compile | error | - | main.cpp:66:57 |
| compile | error | - | main.cpp:66:58 |
| compile | error | - | main.cpp:23:46 |
| compile | error | - | main.cpp:23:49 |
| compile | error | - | main.cpp:23:50 |
| compile | error | - | main.cpp:84:54 |
| compile | error | - | main.cpp:84:57 |
| compile | error | - | main.cpp:84:58 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 3

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:54:54 |
| compile | error | - | main.cpp:54:57 |
| compile | error | - | main.cpp:54:58 |
| compile | error | - | main.cpp:33:46 |
| compile | error | - | main.cpp:33:49 |
| compile | error | - | main.cpp:33:50 |
| compile | error | - | main.cpp:66:54 |
| compile | error | - | main.cpp:66:57 |
| compile | error | - | main.cpp:66:58 |
| compile | error | - | main.cpp:23:46 |
| compile | error | - | main.cpp:23:49 |
| compile | error | - | main.cpp:23:50 |
| compile | error | - | main.cpp:84:54 |
| compile | error | - | main.cpp:84:57 |
| compile | error | - | main.cpp:84:58 |
| compile | error | - | main.cpp:54:54 |
| compile | error | - | main.cpp:54:57 |
| compile | error | - | main.cpp:54:58 |
| compile | error | - | main.cpp:33:46 |
| compile | error | - | main.cpp:33:49 |
| compile | error | - | main.cpp:33:50 |
| compile | error | - | main.cpp:66:54 |
| compile | error | - | main.cpp:66:57 |
| compile | error | - | main.cpp:66:58 |
| compile | error | - | main.cpp:23:46 |
| compile | error | - | main.cpp:23:49 |
| compile | error | - | main.cpp:23:50 |
| compile | error | - | main.cpp:84:54 |
| compile | error | - | main.cpp:84:57 |
| compile | error | - | main.cpp:84:58 |
| compile | error | - | main.cpp:54:54 |
| compile | error | - | main.cpp:54:57 |
| compile | error | - | main.cpp:54:58 |
| compile | error | - | main.cpp:33:46 |
| compile | error | - | main.cpp:33:49 |
| compile | error | - | main.cpp:33:50 |
| compile | error | - | main.cpp:66:54 |
| compile | error | - | main.cpp:66:57 |
| compile | error | - | main.cpp:66:58 |
| compile | error | - | main.cpp:23:46 |
| compile | error | - | main.cpp:23:49 |
| compile | error | - | main.cpp:23:50 |
| compile | error | - | main.cpp:84:54 |
| compile | error | - | main.cpp:84:57 |
| compile | error | - | main.cpp:84:58 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 4

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:54:54 |
| compile | error | - | main.cpp:54:57 |
| compile | error | - | main.cpp:54:58 |
| compile | error | - | main.cpp:33:46 |
| compile | error | - | main.cpp:33:49 |
| compile | error | - | main.cpp:33:50 |
| compile | error | - | main.cpp:66:54 |
| compile | error | - | main.cpp:66:57 |
| compile | error | - | main.cpp:66:58 |
| compile | error | - | main.cpp:23:46 |
| compile | error | - | main.cpp:23:49 |
| compile | error | - | main.cpp:23:50 |
| compile | error | - | main.cpp:84:54 |
| compile | error | - | main.cpp:84:57 |
| compile | error | - | main.cpp:84:58 |
| compile | error | - | main.cpp:54:54 |
| compile | error | - | main.cpp:54:57 |
| compile | error | - | main.cpp:54:58 |
| compile | error | - | main.cpp:33:46 |
| compile | error | - | main.cpp:33:49 |
| compile | error | - | main.cpp:33:50 |
| compile | error | - | main.cpp:66:54 |
| compile | error | - | main.cpp:66:57 |
| compile | error | - | main.cpp:66:58 |
| compile | error | - | main.cpp:23:46 |
| compile | error | - | main.cpp:23:49 |
| compile | error | - | main.cpp:23:50 |
| compile | error | - | main.cpp:84:54 |
| compile | error | - | main.cpp:84:57 |
| compile | error | - | main.cpp:84:58 |
| compile | error | - | main.cpp:54:54 |
| compile | error | - | main.cpp:54:57 |
| compile | error | - | main.cpp:54:58 |
| compile | error | - | main.cpp:33:46 |
| compile | error | - | main.cpp:33:49 |
| compile | error | - | main.cpp:33:50 |
| compile | error | - | main.cpp:66:54 |
| compile | error | - | main.cpp:66:57 |
| compile | error | - | main.cpp:66:58 |
| compile | error | - | main.cpp:23:46 |
| compile | error | - | main.cpp:23:49 |
| compile | error | - | main.cpp:23:50 |
| compile | error | - | main.cpp:84:54 |
| compile | error | - | main.cpp:84:57 |
| compile | error | - | main.cpp:84:58 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 5

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:54:54 |
| compile | error | - | main.cpp:54:57 |
| compile | error | - | main.cpp:54:58 |
| compile | error | - | main.cpp:33:46 |
| compile | error | - | main.cpp:33:49 |
| compile | error | - | main.cpp:33:50 |
| compile | error | - | main.cpp:66:54 |
| compile | error | - | main.cpp:66:57 |
| compile | error | - | main.cpp:66:58 |
| compile | error | - | main.cpp:23:46 |
| compile | error | - | main.cpp:23:49 |
| compile | error | - | main.cpp:23:50 |
| compile | error | - | main.cpp:84:54 |
| compile | error | - | main.cpp:84:57 |
| compile | error | - | main.cpp:84:58 |
| compile | error | - | main.cpp:54:54 |
| compile | error | - | main.cpp:54:57 |
| compile | error | - | main.cpp:54:58 |
| compile | error | - | main.cpp:33:46 |
| compile | error | - | main.cpp:33:49 |
| compile | error | - | main.cpp:33:50 |
| compile | error | - | main.cpp:66:54 |
| compile | error | - | main.cpp:66:57 |
| compile | error | - | main.cpp:66:58 |
| compile | error | - | main.cpp:23:46 |
| compile | error | - | main.cpp:23:49 |
| compile | error | - | main.cpp:23:50 |
| compile | error | - | main.cpp:84:54 |
| compile | error | - | main.cpp:84:57 |
| compile | error | - | main.cpp:84:58 |
| compile | error | - | main.cpp:54:54 |
| compile | error | - | main.cpp:54:57 |
| compile | error | - | main.cpp:54:58 |
| compile | error | - | main.cpp:33:46 |
| compile | error | - | main.cpp:33:49 |
| compile | error | - | main.cpp:33:50 |
| compile | error | - | main.cpp:66:54 |
| compile | error | - | main.cpp:66:57 |
| compile | error | - | main.cpp:66:58 |
| compile | error | - | main.cpp:23:46 |
| compile | error | - | main.cpp:23:49 |
| compile | error | - | main.cpp:23:50 |
| compile | error | - | main.cpp:84:54 |
| compile | error | - | main.cpp:84:57 |
| compile | error | - | main.cpp:84:58 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Scores

| Profile | Component | Value | Note |
| --- | --- | --- | --- |
| df-v0 | single_turn | -0.8 | - |
| df-v0 | multi_turn | -1.05 | - |
| df-v0 | scalar | -1.05 | - |
| lassi | correct | 0.0 | - |
| lassi | correct_paper | PLACEHOLDER | not computed: the paper's criterion is a manual inspection of stdout against the reference |
| lassi | within_10pct | PLACEHOLDER | not measured: no timing profiler exists yet, and the proxy checks outputs, never runtime |
| lassi | first_try | 0.0 | - |
| lassi | sim_t | 0.0011661807580174927 | faithful Sim-T: Python tokenize tokens, difflib ratio with autojunk on; computed by python 3.10.12 |
| lassi | sim_t_c | 0.6728110599078341 | C-aware Sim-T: 2M/T over C tokens, difflib with autojunk off; computed by python 3.10.12 |
| lassi | sim_l | 0.5166666666666667 | faithful Sim-L: stripped lines matched in any order over the larger line count; computed by python 3.10.12 |
| lassi | self_corr | 5.0 | - |
| lassi | cap_hit | 1.0 | - |
| lassi | fence_quirk | 0.0 | - |
| lassi | compiled | 0.0 | - |
| lassi | compiled_first_try | 0.0 | - |
| lassi | scalar | 0.0 | - |

### Not recorded

- Reference-run flags (stdout_truncated, stderr_truncated, workdir_incomplete): not recorded (null), as in a record written before them.
- Reference stdout size: 8578 bytes (UTF-8, from the text store).
- Model requests: not recorded (null), as in a record written before them.

## Trial `rngd-cpu/furiosa-ai--Llama-3.1-8B-Instruct/lassi-hecbench-10/cuda-omp/bsearch/run01`

| Field | Value |
| --- | --- |
| model | openai_compat `furiosa-ai/Llama-3.1-8B-Instruct` |
| final.stage_reached | S5 |
| final.alignment | PLACEHOLDER |
| final.corrections | 0 |
| final.end_reason | none |
| requests | not recorded |
| reference_run.exit_code | 0 |
| reference_run.hang | false |
| reference_run.stdout_truncated | PLACEHOLDER |
| reference_run.stderr_truncated | PLACEHOLDER |
| reference_run.workdir_incomplete | PLACEHOLDER |

### Attempt 0

| Field | Value |
| --- | --- |
| stage_reached | S5 |
| exit_code | 0 |
| hang | false |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | 0.0 |

#### Diagnostics

None.

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | 0.2 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | 0.2 |

### Scores

| Profile | Component | Value | Note |
| --- | --- | --- | --- |
| df-v0 | single_turn | 0.2 | - |
| df-v0 | multi_turn | 0.2 | - |
| df-v0 | scalar | 0.2 | - |
| lassi | correct | 0.0 | - |
| lassi | correct_paper | PLACEHOLDER | not computed: the paper's criterion is a manual inspection of stdout against the reference |
| lassi | within_10pct | PLACEHOLDER | not measured: no timing profiler exists yet, and the proxy checks outputs, never runtime |
| lassi | first_try | 0.0 | - |
| lassi | sim_t | 0.0024330900243309003 | faithful Sim-T: Python tokenize tokens, difflib ratio with autojunk on; computed by python 3.10.12 |
| lassi | sim_t_c | 0.7200847158489234 | C-aware Sim-T: 2M/T over C tokens, difflib with autojunk off; computed by python 3.10.12 |
| lassi | sim_l | 0.5319148936170213 | faithful Sim-L: stripped lines matched in any order over the larger line count; computed by python 3.10.12 |
| lassi | self_corr | 0.0 | - |
| lassi | cap_hit | 0.0 | - |
| lassi | fence_quirk | 0.0 | - |
| lassi | compiled | 1.0 | - |
| lassi | compiled_first_try | 1.0 | - |
| lassi | scalar | 0.0 | - |

### Not recorded

- Reference-run flags (stdout_truncated, stderr_truncated, workdir_incomplete): not recorded (null), as in a record written before them.
- Reference stdout size: 206 bytes (UTF-8, from the text store).
- Run flags of attempt(s) 0: not recorded (null).
- Model requests: not recorded (null), as in a record written before them.

## Trial `rngd-cpu/furiosa-ai--Llama-3.1-8B-Instruct/lassi-hecbench-10/cuda-omp/colorwheel/run01`

| Field | Value |
| --- | --- |
| model | openai_compat `furiosa-ai/Llama-3.1-8B-Instruct` |
| final.stage_reached | S1 |
| final.alignment | PLACEHOLDER |
| final.corrections | 5 |
| final.end_reason | correction-cap |
| requests | not recorded |
| reference_run.exit_code | 0 |
| reference_run.hang | false |
| reference_run.stdout_truncated | PLACEHOLDER |
| reference_run.stderr_truncated | PLACEHOLDER |
| reference_run.workdir_incomplete | PLACEHOLDER |

### Attempt 0

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:99:2 |
| compile | error | - | main.cpp:100:2 |
| compile | error | - | main.cpp:118:2 |
| compile | error | - | main.cpp:142:34 |
| compile | error | - | main.cpp:142:2 |
| compile | error | - | main.cpp:158:2 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 1

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:9:26 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 2

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:9:26 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 3

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:9:26 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 4

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:9:26 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 5

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:9:30 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Scores

| Profile | Component | Value | Note |
| --- | --- | --- | --- |
| df-v0 | single_turn | -0.8 | - |
| df-v0 | multi_turn | -1.05 | - |
| df-v0 | scalar | -1.05 | - |
| lassi | correct | 0.0 | - |
| lassi | correct_paper | PLACEHOLDER | not computed: the paper's criterion is a manual inspection of stdout against the reference |
| lassi | within_10pct | PLACEHOLDER | not measured: no timing profiler exists yet, and the proxy checks outputs, never runtime |
| lassi | first_try | 0.0 | - |
| lassi | sim_t | 0.0034423407917383822 | faithful Sim-T: Python tokenize tokens, difflib ratio with autojunk on; computed by python 3.10.12 |
| lassi | sim_t_c | 0.8545897644191714 | C-aware Sim-T: 2M/T over C tokens, difflib with autojunk off; computed by python 3.10.12 |
| lassi | sim_l | 0.5 | faithful Sim-L: stripped lines matched in any order over the larger line count; computed by python 3.10.12 |
| lassi | self_corr | 5.0 | - |
| lassi | cap_hit | 1.0 | - |
| lassi | fence_quirk | 0.0 | - |
| lassi | compiled | 0.0 | - |
| lassi | compiled_first_try | 0.0 | - |
| lassi | scalar | 0.0 | - |

### Not recorded

- Reference-run flags (stdout_truncated, stderr_truncated, workdir_incomplete): not recorded (null), as in a record written before them.
- Reference stdout size: 79 bytes (UTF-8, from the text store).
- Model requests: not recorded (null), as in a record written before them.

## Trial `rngd-cpu/furiosa-ai--Llama-3.1-8B-Instruct/lassi-hecbench-10/cuda-omp/dense-embedding/run01`

| Field | Value |
| --- | --- |
| model | openai_compat `furiosa-ai/Llama-3.1-8B-Instruct` |
| final.stage_reached | S1 |
| final.alignment | PLACEHOLDER |
| final.corrections | 5 |
| final.end_reason | correction-cap |
| requests | not recorded |
| reference_run.exit_code | 0 |
| reference_run.hang | false |
| reference_run.stdout_truncated | PLACEHOLDER |
| reference_run.stderr_truncated | PLACEHOLDER |
| reference_run.workdir_incomplete | PLACEHOLDER |

### Attempt 0

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:136:2 |
| compile | error | - | main.cpp:137:47 |
| compile | error | - | main.cpp:137:2 |
| compile | error | - | main.cpp:143:2 |
| compile | error | - | main.cpp:163:2 |
| compile | error | - | main.cpp:185:49 |
| compile | error | - | main.cpp:196:2 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 1

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:112:54 |
| compile | error | - | main.cpp:112:9 |
| compile | error | - | main.cpp:116:9 |
| compile | error | - | main.cpp:133:9 |
| compile | error | - | main.cpp:151:56 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 2

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:112:54 |
| compile | error | - | main.cpp:112:9 |
| compile | error | - | main.cpp:116:9 |
| compile | error | - | main.cpp:133:9 |
| compile | error | - | main.cpp:151:56 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 3

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:112:54 |
| compile | error | - | main.cpp:112:9 |
| compile | error | - | main.cpp:116:9 |
| compile | error | - | main.cpp:133:9 |
| compile | error | - | main.cpp:151:56 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 4

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:112:54 |
| compile | error | - | main.cpp:112:9 |
| compile | error | - | main.cpp:116:9 |
| compile | error | - | main.cpp:133:9 |
| compile | error | - | main.cpp:151:56 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 5

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:112:54 |
| compile | error | - | main.cpp:112:9 |
| compile | error | - | main.cpp:116:9 |
| compile | error | - | main.cpp:133:9 |
| compile | error | - | main.cpp:151:56 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Scores

| Profile | Component | Value | Note |
| --- | --- | --- | --- |
| df-v0 | single_turn | -0.8 | - |
| df-v0 | multi_turn | -1.05 | - |
| df-v0 | scalar | -1.05 | - |
| lassi | correct | 0.0 | - |
| lassi | correct_paper | PLACEHOLDER | not computed: the paper's criterion is a manual inspection of stdout against the reference |
| lassi | within_10pct | PLACEHOLDER | not measured: no timing profiler exists yet, and the proxy checks outputs, never runtime |
| lassi | first_try | 0.0 | - |
| lassi | sim_t | 0.001386001386001386 | faithful Sim-T: Python tokenize tokens, difflib ratio with autojunk on; computed by python 3.10.12 |
| lassi | sim_t_c | 0.6543852623147778 | C-aware Sim-T: 2M/T over C tokens, difflib with autojunk off; computed by python 3.10.12 |
| lassi | sim_l | 0.5449101796407185 | faithful Sim-L: stripped lines matched in any order over the larger line count; computed by python 3.10.12 |
| lassi | self_corr | 5.0 | - |
| lassi | cap_hit | 1.0 | - |
| lassi | fence_quirk | 0.0 | - |
| lassi | compiled | 0.0 | - |
| lassi | compiled_first_try | 0.0 | - |
| lassi | scalar | 0.0 | - |

### Not recorded

- Reference-run flags (stdout_truncated, stderr_truncated, workdir_incomplete): not recorded (null), as in a record written before them.
- Reference stdout size: 1093 bytes (UTF-8, from the text store).
- Model requests: not recorded (null), as in a record written before them.

## Trial `rngd-cpu/furiosa-ai--Llama-3.1-8B-Instruct/lassi-hecbench-10/cuda-omp/entropy/run01`

| Field | Value |
| --- | --- |
| model | openai_compat `furiosa-ai/Llama-3.1-8B-Instruct` |
| final.stage_reached | S5 |
| final.alignment | PLACEHOLDER |
| final.corrections | 2 |
| final.end_reason | none |
| requests | not recorded |
| reference_run.exit_code | 0 |
| reference_run.hang | false |
| reference_run.stdout_truncated | PLACEHOLDER |
| reference_run.stderr_truncated | PLACEHOLDER |
| reference_run.workdir_incomplete | PLACEHOLDER |

### Attempt 0

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:38:26 |
| compile | error | - | main.cpp:113:5 |
| compile | warning | declared_but_not_referenced | main.cpp:31:12 |
| compile | warning | declared_but_not_referenced | main.cpp:32:12 |
| compile | warning | declared_but_not_referenced | main.cpp:33:12 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 1

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:42:30 |
| compile | error | - | main.cpp:72:5 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 2

| Field | Value |
| --- | --- |
| stage_reached | S5 |
| exit_code | 0 |
| hang | false |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | 0.0 |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | warning | set_but_not_used | main.cpp:59:11 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | 0.2 |
| df-v0 | warning_count | 1.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | 0.18000000000000002 |

### Scores

| Profile | Component | Value | Note |
| --- | --- | --- | --- |
| df-v0 | single_turn | -0.8 | - |
| df-v0 | multi_turn | 0.08000000000000002 | - |
| df-v0 | scalar | 0.08000000000000002 | - |
| lassi | correct | 0.0 | - |
| lassi | correct_paper | PLACEHOLDER | not computed: the paper's criterion is a manual inspection of stdout against the reference |
| lassi | within_10pct | PLACEHOLDER | not measured: no timing profiler exists yet, and the proxy checks outputs, never runtime |
| lassi | first_try | 0.0 | - |
| lassi | sim_t | 0.002575107296137339 | faithful Sim-T: Python tokenize tokens, difflib ratio with autojunk on; computed by python 3.10.12 |
| lassi | sim_t_c | 0.5486981677917069 | C-aware Sim-T: 2M/T over C tokens, difflib with autojunk off; computed by python 3.10.12 |
| lassi | sim_l | 0.38414634146341464 | faithful Sim-L: stripped lines matched in any order over the larger line count; computed by python 3.10.12 |
| lassi | self_corr | 2.0 | - |
| lassi | cap_hit | 0.0 | - |
| lassi | fence_quirk | 0.0 | - |
| lassi | compiled | 1.0 | - |
| lassi | compiled_first_try | 0.0 | - |
| lassi | scalar | 0.0 | - |

### Not recorded

- Reference-run flags (stdout_truncated, stderr_truncated, workdir_incomplete): not recorded (null), as in a record written before them.
- Reference stdout size: 114 bytes (UTF-8, from the text store).
- Run flags of attempt(s) 2: not recorded (null).
- Model requests: not recorded (null), as in a record written before them.

## Trial `rngd-cpu/furiosa-ai--Llama-3.1-8B-Instruct/lassi-hecbench-10/cuda-omp/jacobi/run01`

| Field | Value |
| --- | --- |
| model | openai_compat `furiosa-ai/Llama-3.1-8B-Instruct` |
| final.stage_reached | S1 |
| final.alignment | PLACEHOLDER |
| final.corrections | 5 |
| final.end_reason | correction-cap |
| requests | not recorded |
| reference_run.exit_code | 0 |
| reference_run.hang | false |
| reference_run.stdout_truncated | PLACEHOLDER |
| reference_run.stderr_truncated | PLACEHOLDER |
| reference_run.workdir_incomplete | PLACEHOLDER |

### Attempt 0

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:45:13 |
| compile | error | - | main.cpp:67:2 |
| compile | warning | set_but_not_used | main.cpp:39:8 |
| compile | error | - | main.cpp:82:2 |
| compile | error | - | main.cpp:82:9 |
| compile | error | - | main.cpp:89:2 |
| compile | error | - | main.cpp:89:13 |
| compile | error | - | main.cpp:90:6 |
| compile | error | - | main.cpp:91:2 |
| compile | error | - | main.cpp:91:33 |
| compile | error | - | main.cpp:95:2 |
| compile | error | - | main.cpp:100:2 |
| compile | error | - | main.cpp:100:8 |
| compile | error | - | main.cpp:102:9 |
| compile | error | - | main.cpp:105:2 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 1

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:38:24 |
| compile | warning | set_but_not_used | main.cpp:34:19 |
| compile | error | - | main.cpp:82:33 |
| compile | error | - | main.cpp:84:17 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 2

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | S-0155 | main.cpp:59 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 3

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | S-0155 | main.cpp:59 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 4

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | S-0155 | main.cpp:60 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 5

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | S-0155 | main.cpp:60 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Scores

| Profile | Component | Value | Note |
| --- | --- | --- | --- |
| df-v0 | single_turn | -0.8 | - |
| df-v0 | multi_turn | -1.05 | - |
| df-v0 | scalar | -1.05 | - |
| lassi | correct | 0.0 | - |
| lassi | correct_paper | PLACEHOLDER | not computed: the paper's criterion is a manual inspection of stdout against the reference |
| lassi | within_10pct | PLACEHOLDER | not measured: no timing profiler exists yet, and the proxy checks outputs, never runtime |
| lassi | first_try | 0.0 | - |
| lassi | sim_t | 0.0020876826722338203 | faithful Sim-T: Python tokenize tokens, difflib ratio with autojunk on; computed by python 3.10.12 |
| lassi | sim_t_c | 0.4244306418219462 | C-aware Sim-T: 2M/T over C tokens, difflib with autojunk off; computed by python 3.10.12 |
| lassi | sim_l | 0.5942857142857143 | faithful Sim-L: stripped lines matched in any order over the larger line count; computed by python 3.10.12 |
| lassi | self_corr | 5.0 | - |
| lassi | cap_hit | 1.0 | - |
| lassi | fence_quirk | 0.0 | - |
| lassi | compiled | 0.0 | - |
| lassi | compiled_first_try | 0.0 | - |
| lassi | scalar | 0.0 | - |

### Not recorded

- Reference-run flags (stdout_truncated, stderr_truncated, workdir_incomplete): not recorded (null), as in a record written before them.
- Reference stdout size: 122 bytes (UTF-8, from the text store).
- Model requests: not recorded (null), as in a record written before them.

## Trial `rngd-cpu/furiosa-ai--Llama-3.1-8B-Instruct/lassi-hecbench-10/cuda-omp/layout/run01`

| Field | Value |
| --- | --- |
| model | openai_compat `furiosa-ai/Llama-3.1-8B-Instruct` |
| final.stage_reached | S4 |
| final.alignment | PLACEHOLDER |
| final.corrections | 5 |
| final.end_reason | correction-cap |
| requests | not recorded |
| reference_run.exit_code | 0 |
| reference_run.hang | false |
| reference_run.stdout_truncated | PLACEHOLDER |
| reference_run.stderr_truncated | PLACEHOLDER |
| reference_run.workdir_incomplete | PLACEHOLDER |

### Attempt 0

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:66:58 |
| compile | error | - | main.cpp:114:58 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 1

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:68:55 |
| compile | error | - | main.cpp:68:68 |
| compile | error | - | main.cpp:70:29 |
| compile | error | - | main.cpp:111:55 |
| compile | error | - | main.cpp:111:68 |
| compile | error | - | main.cpp:113:29 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 2

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:66:55 |
| compile | error | - | main.cpp:66:68 |
| compile | error | - | main.cpp:112:55 |
| compile | error | - | main.cpp:112:68 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 3

| Field | Value |
| --- | --- |
| stage_reached | S4 |
| exit_code | 139 |
| hang | false |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| run | error | run-error | - |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | 0.0 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | 0.0 |

### Attempt 4

| Field | Value |
| --- | --- |
| stage_reached | S4 |
| exit_code | 134 |
| hang | false |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| run | error | run-error | - |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | 0.0 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | 0.0 |

### Attempt 5

| Field | Value |
| --- | --- |
| stage_reached | S4 |
| exit_code | 134 |
| hang | false |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| run | error | run-error | - |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | 0.0 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | 0.0 |

### Scores

| Profile | Component | Value | Note |
| --- | --- | --- | --- |
| df-v0 | single_turn | -0.8 | - |
| df-v0 | multi_turn | -0.25 | - |
| df-v0 | scalar | -0.25 | - |
| lassi | correct | 0.0 | - |
| lassi | correct_paper | PLACEHOLDER | not computed: the paper's criterion is a manual inspection of stdout against the reference |
| lassi | within_10pct | PLACEHOLDER | not measured: no timing profiler exists yet, and the proxy checks outputs, never runtime |
| lassi | first_try | 0.0 | - |
| lassi | sim_t | 0.0033264033264033266 | faithful Sim-T: Python tokenize tokens, difflib ratio with autojunk on; computed by python 3.10.12 |
| lassi | sim_t_c | 0.7680440771349862 | C-aware Sim-T: 2M/T over C tokens, difflib with autojunk off; computed by python 3.10.12 |
| lassi | sim_l | 0.5027322404371585 | faithful Sim-L: stripped lines matched in any order over the larger line count; computed by python 3.10.12 |
| lassi | self_corr | 5.0 | - |
| lassi | cap_hit | 1.0 | - |
| lassi | fence_quirk | 0.0 | - |
| lassi | compiled | 1.0 | - |
| lassi | compiled_first_try | 0.0 | - |
| lassi | scalar | 0.0 | - |

### Not recorded

- Reference-run flags (stdout_truncated, stderr_truncated, workdir_incomplete): not recorded (null), as in a record written before them.
- Reference stdout size: 110 bytes (UTF-8, from the text store).
- Run flags of attempt(s) 3, 4, 5: not recorded (null).
- Model requests: not recorded (null), as in a record written before them.

## Trial `rngd-cpu/furiosa-ai--Llama-3.1-8B-Instruct/lassi-hecbench-10/cuda-omp/matrix-rotate/run01`

| Field | Value |
| --- | --- |
| model | openai_compat `furiosa-ai/Llama-3.1-8B-Instruct` |
| final.stage_reached | S5 |
| final.alignment | PLACEHOLDER |
| final.corrections | 0 |
| final.end_reason | none |
| requests | not recorded |
| reference_run.exit_code | 0 |
| reference_run.hang | false |
| reference_run.stdout_truncated | PLACEHOLDER |
| reference_run.stderr_truncated | PLACEHOLDER |
| reference_run.workdir_incomplete | PLACEHOLDER |

### Attempt 0

| Field | Value |
| --- | --- |
| stage_reached | S5 |
| exit_code | 0 |
| hang | false |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | 0.0 |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | warning | declared_but_not_referenced | main.cpp:32:6 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | 0.2 |
| df-v0 | warning_count | 1.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | 0.18000000000000002 |

### Scores

| Profile | Component | Value | Note |
| --- | --- | --- | --- |
| df-v0 | single_turn | 0.18000000000000002 | - |
| df-v0 | multi_turn | 0.18000000000000002 | - |
| df-v0 | scalar | 0.18000000000000002 | - |
| lassi | correct | 0.0 | - |
| lassi | correct_paper | PLACEHOLDER | not computed: the paper's criterion is a manual inspection of stdout against the reference |
| lassi | within_10pct | PLACEHOLDER | not measured: no timing profiler exists yet, and the proxy checks outputs, never runtime |
| lassi | first_try | 0.0 | - |
| lassi | sim_t | 0.006920415224913495 | faithful Sim-T: Python tokenize tokens, difflib ratio with autojunk on; computed by python 3.10.12 |
| lassi | sim_t_c | 0.6165311653116531 | C-aware Sim-T: 2M/T over C tokens, difflib with autojunk off; computed by python 3.10.12 |
| lassi | sim_l | 0.8365384615384616 | faithful Sim-L: stripped lines matched in any order over the larger line count; computed by python 3.10.12 |
| lassi | self_corr | 0.0 | - |
| lassi | cap_hit | 0.0 | - |
| lassi | fence_quirk | 0.0 | - |
| lassi | compiled | 1.0 | - |
| lassi | compiled_first_try | 1.0 | - |
| lassi | scalar | 0.0 | - |

### Not recorded

- Reference-run flags (stdout_truncated, stderr_truncated, workdir_incomplete): not recorded (null), as in a record written before them.
- Reference stdout size: 49 bytes (UTF-8, from the text store).
- Run flags of attempt(s) 0: not recorded (null).
- Model requests: not recorded (null), as in a record written before them.

## Trial `rngd-cpu/furiosa-ai--Llama-3.1-8B-Instruct/lassi-hecbench-10/cuda-omp/pathfinder/run01`

| Field | Value |
| --- | --- |
| model | openai_compat `furiosa-ai/Llama-3.1-8B-Instruct` |
| final.stage_reached | S1 |
| final.alignment | PLACEHOLDER |
| final.corrections | 5 |
| final.end_reason | correction-cap |
| requests | not recorded |
| reference_run.exit_code | 0 |
| reference_run.hang | false |
| reference_run.stdout_truncated | PLACEHOLDER |
| reference_run.stderr_truncated | PLACEHOLDER |
| reference_run.workdir_incomplete | PLACEHOLDER |

### Attempt 0

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:40:68 |
| compile | error | - | main.cpp:40:79 |
| compile | error | - | main.cpp:40:93 |
| compile | error | - | main.cpp:40:110 |
| compile | warning | declared_but_not_referenced | main.cpp:141:6 |
| compile | warning | declared_but_not_referenced | main.cpp:179:7 |
| compile | warning | declared_but_not_referenced | main.cpp:180:7 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 1

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | warning | declared_but_not_referenced | main.cpp:103:9 |
| compile | warning | declared_but_not_referenced | main.cpp:131:10 |
| compile | warning | declared_but_not_referenced | main.cpp:132:10 |
| compile | error | S-0000 | main.cpp:57 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 2

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | S-0000 | main.cpp:57 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 3

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | S-0000 | main.cpp:57 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 4

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | S-0000 | main.cpp:56 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 5

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | S-0000 | main.cpp:56 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Scores

| Profile | Component | Value | Note |
| --- | --- | --- | --- |
| df-v0 | single_turn | -0.8 | - |
| df-v0 | multi_turn | -1.05 | - |
| df-v0 | scalar | -1.05 | - |
| lassi | correct | 0.0 | - |
| lassi | correct_paper | PLACEHOLDER | not computed: the paper's criterion is a manual inspection of stdout against the reference |
| lassi | within_10pct | PLACEHOLDER | not measured: no timing profiler exists yet, and the proxy checks outputs, never runtime |
| lassi | first_try | 0.0 | - |
| lassi | sim_t | 0.0006253908692933083 | faithful Sim-T: Python tokenize tokens, difflib ratio with autojunk on; computed by python 3.10.12 |
| lassi | sim_t_c | 0.5443091138177236 | C-aware Sim-T: 2M/T over C tokens, difflib with autojunk off; computed by python 3.10.12 |
| lassi | sim_l | 0.27472527472527475 | faithful Sim-L: stripped lines matched in any order over the larger line count; computed by python 3.10.12 |
| lassi | self_corr | 5.0 | - |
| lassi | cap_hit | 1.0 | - |
| lassi | fence_quirk | 0.0 | - |
| lassi | compiled | 0.0 | - |
| lassi | compiled_first_try | 0.0 | - |
| lassi | scalar | 0.0 | - |

### Not recorded

- Reference-run flags (stdout_truncated, stderr_truncated, workdir_incomplete): not recorded (null), as in a record written before them.
- Reference stdout size: 88 bytes (UTF-8, from the text store).
- Model requests: not recorded (null), as in a record written before them.

## Trial `rngd-cpu/furiosa-ai--Llama-3.1-8B-Instruct/lassi-hecbench-10/cuda-omp/randomAccess/run01`

| Field | Value |
| --- | --- |
| model | openai_compat `furiosa-ai/Llama-3.1-8B-Instruct` |
| final.stage_reached | S4 |
| final.alignment | PLACEHOLDER |
| final.corrections | 5 |
| final.end_reason | correction-cap |
| requests | not recorded |
| reference_run.exit_code | 0 |
| reference_run.hang | false |
| reference_run.stdout_truncated | PLACEHOLDER |
| reference_run.stderr_truncated | PLACEHOLDER |
| reference_run.workdir_incomplete | PLACEHOLDER |

### Attempt 0

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:97:47 |
| compile | warning | hidden_by_old_for_init | main.cpp:144:7 |
| compile | warning | hidden_by_old_for_init | main.cpp:144:12 |
| compile | warning | hidden_by_old_for_init | main.cpp:144:23 |
| compile | warning | hidden_by_old_for_init | main.cpp:150:7 |
| compile | warning | hidden_by_old_for_init | main.cpp:150:12 |
| compile | warning | hidden_by_old_for_init | main.cpp:150:25 |
| compile | warning | hidden_by_old_for_init | main.cpp:151:12 |
| compile | warning | hidden_by_old_for_init | main.cpp:151:17 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 1

| Field | Value |
| --- | --- |
| stage_reached | S1 |
| exit_code | PLACEHOLDER |
| hang | PLACEHOLDER |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| compile | error | - | main.cpp:74:50 |
| compile | warning | declared_but_not_referenced | main.cpp:52:12 |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | -0.8 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | -0.8 |

### Attempt 2

| Field | Value |
| --- | --- |
| stage_reached | S4 |
| exit_code | 137 |
| hang | false |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| run | error | run-error | - |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | 0.0 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | 0.0 |

### Attempt 3

| Field | Value |
| --- | --- |
| stage_reached | S4 |
| exit_code | 137 |
| hang | false |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| run | error | run-error | - |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | 0.0 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | 0.0 |

### Attempt 4

| Field | Value |
| --- | --- |
| stage_reached | S4 |
| exit_code | 137 |
| hang | false |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| run | error | run-error | - |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | 0.0 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | 0.0 |

### Attempt 5

| Field | Value |
| --- | --- |
| stage_reached | S4 |
| exit_code | 137 |
| hang | false |
| stdout_truncated | PLACEHOLDER |
| stderr_truncated | PLACEHOLDER |
| workdir_incomplete | PLACEHOLDER |
| alignment | PLACEHOLDER |

#### Diagnostics

| Stage | Severity | Code | Place |
| --- | --- | --- | --- |
| run | error | run-error | - |

#### Attempt scores

| Profile | Component | Value |
| --- | --- | --- |
| df-v0 | stage_base | 0.0 |
| df-v0 | warning_count | 0.0 |
| df-v0 | alignment_term | 0.0 |
| df-v0 | guard | PLACEHOLDER |
| df-v0 | alignment_missing | 0.0 |
| df-v0 | scalar | 0.0 |

### Scores

| Profile | Component | Value | Note |
| --- | --- | --- | --- |
| df-v0 | single_turn | -0.8 | - |
| df-v0 | multi_turn | -0.25 | - |
| df-v0 | scalar | -0.25 | - |
| lassi | correct | 0.0 | - |
| lassi | correct_paper | PLACEHOLDER | not computed: the paper's criterion is a manual inspection of stdout against the reference |
| lassi | within_10pct | PLACEHOLDER | not measured: no timing profiler exists yet, and the proxy checks outputs, never runtime |
| lassi | first_try | 0.0 | - |
| lassi | sim_t | 0.0058823529411764705 | faithful Sim-T: Python tokenize tokens, difflib ratio with autojunk on; computed by python 3.10.12 |
| lassi | sim_t_c | 0.8464501926252064 | C-aware Sim-T: 2M/T over C tokens, difflib with autojunk off; computed by python 3.10.12 |
| lassi | sim_l | 0.4666666666666667 | faithful Sim-L: stripped lines matched in any order over the larger line count; computed by python 3.10.12 |
| lassi | self_corr | 5.0 | - |
| lassi | cap_hit | 1.0 | - |
| lassi | fence_quirk | 0.0 | - |
| lassi | compiled | 1.0 | - |
| lassi | compiled_first_try | 0.0 | - |
| lassi | scalar | 0.0 | - |

### Not recorded

- Reference-run flags (stdout_truncated, stderr_truncated, workdir_incomplete): not recorded (null), as in a record written before them.
- Reference stdout size: 185 bytes (UTF-8, from the text store).
- Run flags of attempt(s) 2, 3, 4, 5: not recorded (null).
- Model requests: not recorded (null), as in a record written before them.
