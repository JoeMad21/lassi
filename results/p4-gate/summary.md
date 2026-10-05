# P4 ttsim Execution: phase gate

Gate (bible Build Roadmap, P4 row, Gate column): "`metal_example_add_2_integers_in_riscv` passes on ttsim; Tier A references pass".

Date: 2026-10-05 (alpha01 clock 12:29 to 12:35 -07:00). Branch p4-ttsim, commit ec1c9a2 (ec1c9a2b4708fdb2240ac15707e3c613bb0ecaa4, the commit "P4.G: Start the phase gate"), clean: every run below came from the detached clean worktree at that commit, and each provenance.json records `dirty` false and `snapshot_of` null. The gate names no owner review, so a pass sets P4 DONE (plans/p4-ttsim.md, P4.G).

## Checks

| Check | Command | rx id | Outcome | Provenance |
| --- | --- | --- | --- | --- |
| Host, before | `rx doctor`; `timeout 150 du -sh /mnt/nvme10/joseph_ufl` | doctor just before the gate runs; du exec 20261005-122941-exec-1b5b | doctor Ready, no STOP, no job running, scratch free 427.0 GB, root free 244.3 GB; du 97G, under the owner's 120G cap | doctor output not kept under results/; du in this summary |
| Example on ttsim | `uv run tools/ttsim_smoke.py metal_example_add_2_integers_in_riscv` | 20261005-123001-desktop-8r113ei-detached-ec1c9a2b-2979 | rc 0; the driver printed `add_2_integers_in_riscv: pass (every check passed)` | results/p4-gate/smoke/provenance.json |
| Tier A dry run | `uv run tools/fetch_bench.py assets/bench/tt-pairs-v0.yaml && LASSI_GRAPHICS=off uv run lassi run tests/fixtures/recipes/p4-tier-a-dry-run.yaml --run-id p4-gate-tier-a` (rx job p4-tier-a) | job 20261005-123023-p4-tier-a-c247 | rc 0; the run's provenance.json reads status complete; ten trials, every one at S5 | results/p4-gate/tier-a/provenance.json |
| Host, after | `timeout 150 du -sh /mnt/nvme10/joseph_ufl` | exec 20261005-123454-exec-76c0 | du 98G; the gate run's tree, lassi-runs/runs/p4-gate-tier-a, is 124M | this summary |

The Tier A command is the plan's (`uv run lassi run tests/fixtures/recipes/p4-tier-a-dry-run.yaml --run-id p4-gate-tier-a`) with two additions: tools/fetch_bench.py first, which found $LASSI_SCRATCH/bench/tt-pairs-v0@5280a9cf... already holding the suite with every sha256 matching and changed nothing, and LASSI_GRAPHICS=off, as AGENTS.md asks of unattended runs. The scratch disk's free space fell from 419.7 GB to 395.9 GB during the job (provenance.json), while this account's use went from 97G to 98G; the disk is shared, and the rest of the difference was not examined.

Devices (the run's provenance.json, devices): every TT run used ttsim v1.3.4 (libttsim_wh.so, a virtual Wormhole) on the host CPU with tt-metal 5280a9cf, a simulator, not silicon; every native C++ run used the host CPU, AMD EPYC 9534 64-Core Processor. Each ttsim pass is provisional until a silicon check (ttsim Facts), a human-started run outside P4. No simulator time is shown as performance.

## Results [MEASURED]

Read from the run's trial.json files on alpha01 (read-only rx exec 20261005-123421-exec-e2b1 and 20261005-123430-exec-8656); the run tree stays on the host and nothing from it enters this repository but these values.

Recipe p4-tier-a-dry-run, hash 797b5bf5837a624c6c4a20aad731ed8e44c2fb0a221822846979aa6a2eebee8e, executors {cpp: native, tt: ttsim}. Every trial passed the baseline: no trial has an end reason, so both references of every item built, ran clean (no undefined behavior, no gap, exit 0), and agreed within the item's declared tolerance.

| Item | Tolerance | Reference agreement | CPU -> TT attempt: stage, alignment, host_compute, guard tag | TT -> CPU attempt: stage, alignment | unpack_to_dest |
| --- | --- | --- | --- | --- | --- |
| loopback | max_abs 0 | output: pcc 1.0, max_abs 0.0, passed | S5, 1.0, false, guard-data-movement-only | S5, 1.0 | not taken (no compute kernel) |
| eltwise_binary | max_abs 0.01 | c: pcc 0.999999, max_abs 0.001953125, passed | S5, 1.0, false, none | S5, 1.0 | not taken |
| eltwise_sfpu | max_abs 0.05 | result: pcc 0.99996, max_abs 0.015625, passed | S5, 1.0, false, none | S5, 1.0 | not taken |
| matmul_single_core | pcc 0.97 | c: pcc 0.98054, max_abs 7.0, passed | S5, 1.0, false, none | S5, 1.0 | not taken |
| matmul_multi_core | pcc 0.97 | c: pcc 0.99994, max_abs 0.25, passed | S5, 1.0, false, none | S5, 1.0 | not taken |

- Gate text, part 1: the gate's example passes on ttsim (smoke run above).
- Gate text, part 2, Tier A references pass: each TT reference is within its tolerance of its C++ counterpart, with no undefined behavior and no gap, and the guard is clear: every built CPU -> TT attempt records host_compute false, loopback with only the data-movement-only tag. TT -> CPU attempts record no reading, since gcc-native declares no guard.
- Extra check, beside the gate text (it tests the path, as P1.G's mock did): every mock candidate reached S5 with alignment 1.0 and no correction.
- unpack_to_dest: each item's manifest note records that it does not take ttsim issue #18's path (no fp32_dest_acc_en, no unpack_to_dest_mode, Float16_b circular buffers; tests/bench/test_tt_pairs.py), and no run reported a ttsim finding.
- The agreement values are the references' own (the mock answers with them), from one run at each item's fixed seed; matmul_single_core's 0.98054 is 0.0105 above its bound.

Earlier P4 evidence the gate rests on: results/p4-ttsim-smoke (P4.11, the six pinned examples on ttsim), results/p4-guard-remote (P4.12), and results/p4-tier-a-remote (P4.13, the five pairs' remote tests from clean commit 5e72f4e).

## Verdict

PASS: the example passes on ttsim, and every Tier A reference passes, from one clean commit. P4 is set DONE and the pull request "P4 ttsim Execution" goes from p4-ttsim to main, for the owner to merge.
