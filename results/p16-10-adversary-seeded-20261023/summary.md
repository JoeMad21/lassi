# The seeded set's first adversary runs (task P16.10, 2026-10-23)

Exploratory, as P16.10 labels these runs. This is not the P16.G gate, which runs in the LASSI distro. Each recipe of projects/adversary-seeded/ ran once with `lassi --graphics off run`, through `uv run tools/rx.py run` (rx 20261022-215349-p16-witness-78eb). The runs started after the freeze record was committed (tests/fixtures/adversary/freeze.yaml, commit b1a2b6b), so nothing in the record changes in response to them (the Freeze decision).

- Host: alpha01, CPU only, every program in the sandbox (native executor, gcc-native 12.3.0). No card was used. The owner approved the run in the working session (OQ-073).
- Commit: b1a2b6b6897ed75cacc234fb1615d6e8983890fb, clean (provenance.json: `dirty: false`).
- Date: 2026-10-22T21:53:49-07:00 to 22:03:02-07:00 (00:53 to 01:03 EDT on 2026-10-23).
- Candidates: the seeded faulty candidates are synthetic replay fixtures, never model output. The control's candidates are the references themselves, delivered through the mock.
- Run trees on the build host: $LASSI_RUNS_ROOT/runs/p16-10-<recipe>-20261023.
- The per-trial adversary counts below were read from each trial.json's adversary record (rx 20261022-220329-exec-8945).
- Scratch use: du of the scratch root was 318180212 KiB before and 318333460 KiB after (+150 MiB). /mnt/nvme10 had 113G free afterwards.

## Results

Every trial reached S5: each seeded candidate passed binary_io on its item's held-out inputs, so every divergence below is the adversary's finding, not the held-out inputs'.

| Recipe | Adversary | Trials | Proposals per trial | Refused | Runs per trial | Trials with a divergence |
| --- | --- | --- | --- | --- | --- | --- |
| seeded-fuzzer.yaml | fuzzer, seed 1, budget 32 | 12 seeded candidates | 32 | 0 | 64 | 12 of 12 |
| control-fuzzer.yaml | fuzzer, seed 1, budget 32 | 6 references as their own candidates | 32 | 0 | 64 | 0 of 6 |
| seeded-llm.yaml | llm on the break-contract replay, budget 8 | 12 seeded candidates | 8 | 8 | 0 | 0 of 12 |

Divergences per seeded-fuzzer trial:

| Item | run01 | run02 |
| --- | --- | --- |
| pair_max | 7 | 7 |
| piecewise_gate | 9 | 7 |
| piecewise_ops | 23 | 25 |
| prefix_sum | 20 | 18 |
| scale_i64 | 25 | 24 |
| signed_root | 23 | 23 |

## Reading

- **The fuzzer found every seeded fault.** Within its budget it found an oracle-confirmed divergence on all 12 seeded candidates, and none on the 6 controls. This is the P16.G gate reading, but from an exploratory run on alpha01 rather than the gate run in the distro.
- **Every out-of-contract proposal was refused before any run.** Each of the replay-driven llm adversary's 96 proposals broke the contract and was refused, so 0 runs started, and the invalid-input rejection rate is 96 of 96.
- **Limits:** one seed and one run per recipe, so this is not a rate across seeds. The divergence counts depend on the fuzzer's strategies, which the bible lists as known cases, not complete. Wall times are not reported, and none is performance.
