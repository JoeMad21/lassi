# The seeded set's first adversary runs (task P16.10, 2026-10-23)

Exploratory, as P16.10 labels these runs. This is not the P16.G gate, which runs in the LASSI distro. Each recipe of projects/adversary-seeded/ ran once with `lassi --graphics off run`, through `uv run tools/rx.py run` (rx 20261022-215349-p16-witness-78eb). The runs started after the freeze record was committed (tests/fixtures/adversary/freeze.yaml, commit b1a2b6b), so nothing in the record changes in response to them (the Freeze decision).

- Host: alpha01, CPU only, every program in the sandbox (native executor, gcc-native 12.3.0). No card was used. The owner approved the run in the working session (OQ-073).
- Commit: b1a2b6b6897ed75cacc234fb1615d6e8983890fb, clean (provenance.json: `dirty: false`).
- Date: 2026-10-22T21:53:49-07:00 to 22:03:02-07:00 (00:53 to 01:03 EDT on 2026-10-23).
- Candidates: the seeded faulty candidates are synthetic replay fixtures, never model output. The control's candidates are the references themselves, delivered through the mock.
- Run trees on the build host: $LASSI_RUNS_ROOT/runs/p16-10-<recipe>-20261023.
- Evidence: the per-trial stages and adversary counts below are in adversary-counts.txt. It was written on the host by rx 20261022-222446-exec-2be5, which read each trial.json the same way rx 20261022-220329-exec-8945 first did, and was pulled with `rx pull --path`.
- Scratch space:
  - du of the scratch root was 318180212 KiB before the runs and 318333460 KiB after (+150 MiB), from the run's own output.
  - The gate measured free space on the scratch disk at 120.5 GB before and 120.4 GB after (provenance.json, scratch_free_gb_before and scratch_free_gb_after); `df -h` showed 113G free after.
  - This step ran with the scratch root at about 303 GiB, past the 145G stop line and the 150G cap. The cap had been lifted for 2026-10-21 and 2026-10-22, when the model weights were downloaded, and was not re-checked before this step. The owner decides (OQ-074).
  - On alpha01 (OQ-073), the scratch disk stands in for the C: drive the plan names. Nothing ran on the workstation.
- lassi's own exit status was not captured. The rc lines in the run's output are those of the `tail` each `lassi run` was piped to. Every trial was written and read back (adversary-counts.txt), and each run printed its run directory, which `lassi run` does only after its last trial.

## Results

Every trial reached S5: each seeded candidate passed binary_io on its item's held-out inputs, so every divergence below is the adversary's finding, not the held-out inputs'.

| Recipe | Adversary | Trials | Proposals per trial | Refused | Runs per trial | Trials with a divergence |
| --- | --- | --- | --- | --- | --- | --- |
| seeded-fuzzer.yaml | fuzzer, seed 1, budget 32 | 12 seeded candidates | 32 | 0 | 64 | 12 of 12 |
| control-fuzzer.yaml | fuzzer, seed 1, budget 32 | 6 references as their own candidates | 32 | 0 | 64 | 0 of 6 |
| seeded-llm.yaml | llm on the break-contract replay, budget 8 | 12 seeded candidates | 8 | 8 | 0 | 0 of 12 |

Divergences per seeded-fuzzer trial. Each trial is one seeded candidate, in the order of tests/fixtures/adversary/seeded/index.json:

| Item | Trial run01: candidate, divergences | Trial run02: candidate, divergences |
| --- | --- | --- |
| pair_max | special-gt, 7 | special-fmax, 7 |
| piecewise_gate | boundary-zero, 9 | boundary-one, 7 |
| piecewise_ops | arith-sign, 23 | arith-divide, 25 |
| prefix_sum | precision-float, 20 | precision-blocks, 18 |
| scale_i64 | narrow-wrap, 25 | narrow-float, 24 |
| signed_root | sign-abs, 23 | sign-unary, 23 |

## Reading

- **The fuzzer found every seeded fault.** Within its budget it found an oracle-confirmed divergence on all 12 seeded candidates, and none on the 6 controls. That is the reading P16.G asks for, holding in this exploratory run.
- **Every out-of-contract proposal was refused before any run.** Each of the replay-driven llm adversary's 96 proposals broke the contract and was refused, so 0 runs started, and the invalid-input rejection rate is 96 of 96.
- **Limits:** one seed and one run per recipe, so this is not a rate across seeds. The divergence counts depend on the fuzzer's strategies, which the bible lists as known cases, not complete. Wall times are not reported, and none is performance.
