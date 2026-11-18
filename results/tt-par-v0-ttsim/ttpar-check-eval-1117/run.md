# Run ttpar-check-eval-1117

| Field | Value |
| --- | --- |
| Recipe | eval-cpp2tt |
| Recipe hash | `6567840d300bc45ef3480ef250798a4c55c20081af36b751898f32ca9d342149` |
| Commit | afd410c2dcf7f0bcd7cd2a954207f63191d69498 |
| Dirty | false |
| Executor | cpp: native; tt: ttsim |
| Device (cpp) | host CPU (native): AMD EPYC 9534 64-Core Processor |
| Device (tt) | ttsim v1.3.4 (libttsim_wh.so, a virtual Wormhole) on the host CPU, with tt-metal 5280a9cf-tracy: a simulator, not silicon |
| Driver | - |
| Started (UTC) | 2026-11-18T03:31:30+00:00 |
| Finished (UTC) | 2026-11-18T03:34:02+00:00 |
| Trials | 5 |

## Resolved recipe

```yaml
# Resolved recipe eval-cpp2tt (chain: base -> eval-cpp2tt)
# recipe_hash: 6567840d300bc45ef3480ef250798a4c55c20081af36b751898f32ca9d342149 is the sha256 of the YAML below these comment lines
bench:
  split: eval
  suite: tt-par-v0
directions:
- source: cpp
  target: tt
executor:
  cpp:
    kind: native
  tt:
    arch: wormhole_b0
    dispatch: slow
    kind: ttsim
faithful: false
fixes:
  baseline_both: true
  execution_gate: true
  fence_tag: true
  parsed_diagnostics: true
  prompt_newlines: true
  prompt_spaces: true
llm:
  sampling:
    max_tokens: 4096
    temperature: 0.2
    top_p: 0.9
loop:
  max_corrections: 0
model:
  backend: mock
  id: mock-reference
oracle:
  kind: binary_io
  metric: from_item
  shape: elements
  threshold: from_item
project: eval-cpp2tt
prompts: tt-host-v1
report:
  parquet: true
  trial_md: true
runs_root: /mnt/nvme10/joseph_ufl/lassi-runs
sandbox:
  mem_gb: 16
  network: false
  wall_s: baseline_x10
stages:
- baseline
- generate
- compile_loop
- run_loop
- oracle
toolchain:
  cpp: gcc-native
  tt: ttmetal-host
trials:
  n: 1
```

## Toolchain pins

| Toolchain | Languages | Pin | Version | Install prefix |
| --- | --- | --- | --- | --- |
| gcc-native | cpp | gcc | 12.3.0 | none, host /usr/bin/g++-12 |
| ttmetal-host | tt | tt-metal | 5280a9cf | tt-metal@5280a9cf-tracy |

## Trials

The tt programs of this run ran on a simulator executor: their run wall times are simulator wall time, not performance, and trial.md labels each one it shows. The wall_s column below is each trial's pipeline wall time, which includes those runs, so it is not performance either.

### Arm mock-reference

| trial | direction | item | stage reached | corrections | wall_s | trial.md |
| --- | --- | --- | --- | --- | --- | --- |
| `eval-cpp2tt/mock-reference/tt-par-v0/cpp-tt/col_max/run01` | cpp-tt | col_max | S5 | 0 | 28.98643103300128 | [trial.md](eval-cpp2tt/mock-reference/tt-par-v0/cpp-tt/col_max/run01/trial.md) |
| `eval-cpp2tt/mock-reference/tt-par-v0/cpp-tt/col_scale/run01` | cpp-tt | col_scale | S5 | 0 | 31.203056246973574 | [trial.md](eval-cpp2tt/mock-reference/tt-par-v0/cpp-tt/col_scale/run01/trial.md) |
| `eval-cpp2tt/mock-reference/tt-par-v0/cpp-tt/row_softmax/run01` | cpp-tt | row_softmax | S5 | 0 | 31.844522099010646 | [trial.md](eval-cpp2tt/mock-reference/tt-par-v0/cpp-tt/row_softmax/run01/trial.md) |
| `eval-cpp2tt/mock-reference/tt-par-v0/cpp-tt/row_sum/run01` | cpp-tt | row_sum | S5 | 0 | 29.12929905205965 | [trial.md](eval-cpp2tt/mock-reference/tt-par-v0/cpp-tt/row_sum/run01/trial.md) |
| `eval-cpp2tt/mock-reference/tt-par-v0/cpp-tt/transpose_2d/run01` | cpp-tt | transpose_2d | S5 | 0 | 30.160338151967153 | [trial.md](eval-cpp2tt/mock-reference/tt-par-v0/cpp-tt/transpose_2d/run01/trial.md) |

## Guard coverage

How much of the run the host-compute guard checked. The unit is the attempt: every attempt of every trial, whatever its stage or the trial's end reason; a trial with no attempt adds nothing. Checked: Attempt.guards.host_compute is true or false. Not checked: it is null, counted under the attempt's host_compute_not_checked code, or under not recorded when the attempt records none (recorded before task P17.14). The not-checked rate is not checked over attempts, - with no attempt, and has no interval. One row for the run, every arm and direction together, then one per target language, the part of the direction's name after its only hyphen; a direction whose name holds more than one hyphen gets its own row.

| Scope | Attempts | Checked | Not checked | Not-checked rate | not-built | no-program | no-guard | guard-not-checked | not recorded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run | 5 | 5 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 |
| target tt | 5 | 5 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 |
