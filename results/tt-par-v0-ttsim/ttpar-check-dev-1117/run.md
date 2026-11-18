# Run ttpar-check-dev-1117

| Field | Value |
| --- | --- |
| Recipe | dev-cpp2tt |
| Recipe hash | `21f52d272efda6cb9e26e9cd8eed3350e425756c25e3162f40456538fb8386fe` |
| Commit | afd410c2dcf7f0bcd7cd2a954207f63191d69498 |
| Dirty | false |
| Executor | cpp: native; tt: ttsim |
| Device (cpp) | host CPU (native): AMD EPYC 9534 64-Core Processor |
| Device (tt) | ttsim v1.3.4 (libttsim_wh.so, a virtual Wormhole) on the host CPU, with tt-metal 5280a9cf-tracy: a simulator, not silicon |
| Driver | - |
| Started (UTC) | 2026-11-18T03:34:03+00:00 |
| Finished (UTC) | 2026-11-18T03:36:37+00:00 |
| Trials | 5 |

## Resolved recipe

```yaml
# Resolved recipe dev-cpp2tt (chain: base -> dev-cpp2tt)
# recipe_hash: 21f52d272efda6cb9e26e9cd8eed3350e425756c25e3162f40456538fb8386fe is the sha256 of the YAML below these comment lines
bench:
  split: dev
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
project: dev-cpp2tt
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
| `dev-cpp2tt/mock-reference/tt-par-v0/cpp-tt/matmul_multicore_reuse/run01` | cpp-tt | matmul_multicore_reuse | S5 | 0 | 33.0254015349783 | [trial.md](dev-cpp2tt/mock-reference/tt-par-v0/cpp-tt/matmul_multicore_reuse/run01/trial.md) |
| `dev-cpp2tt/mock-reference/tt-par-v0/cpp-tt/matmul_multicore_reuse_mcast/run01` | cpp-tt | matmul_multicore_reuse_mcast | S5 | 0 | 33.2876727640396 | [trial.md](dev-cpp2tt/mock-reference/tt-par-v0/cpp-tt/matmul_multicore_reuse_mcast/run01/trial.md) |
| `dev-cpp2tt/mock-reference/tt-par-v0/cpp-tt/pad_multi_core/run01` | cpp-tt | pad_multi_core | S5 | 0 | 25.9529890359845 | [trial.md](dev-cpp2tt/mock-reference/tt-par-v0/cpp-tt/pad_multi_core/run01/trial.md) |
| `dev-cpp2tt/mock-reference/tt-par-v0/cpp-tt/vecadd_multi_core/run01` | cpp-tt | vecadd_multi_core | S5 | 0 | 31.914755019941367 | [trial.md](dev-cpp2tt/mock-reference/tt-par-v0/cpp-tt/vecadd_multi_core/run01/trial.md) |
| `dev-cpp2tt/mock-reference/tt-par-v0/cpp-tt/vecadd_sharding/run01` | cpp-tt | vecadd_sharding | S5 | 0 | 28.42657637502998 | [trial.md](dev-cpp2tt/mock-reference/tt-par-v0/cpp-tt/vecadd_sharding/run01/trial.md) |

## Guard coverage

How much of the run the host-compute guard checked. The unit is the attempt: every attempt of every trial, whatever its stage or the trial's end reason; a trial with no attempt adds nothing. Checked: Attempt.guards.host_compute is true or false. Not checked: it is null, counted under the attempt's host_compute_not_checked code, or under not recorded when the attempt records none (recorded before task P17.14). The not-checked rate is not checked over attempts, - with no attempt, and has no interval. One row for the run, every arm and direction together, then one per target language, the part of the direction's name after its only hyphen; a direction whose name holds more than one hyphen gets its own row.

| Scope | Attempts | Checked | Not checked | Not-checked rate | not-built | no-program | no-guard | guard-not-checked | not recorded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run | 5 | 5 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 |
| target tt | 5 | 5 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0 |
