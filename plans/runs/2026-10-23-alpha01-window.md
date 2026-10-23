# Run 2026-10-23-alpha01-window

Window: the owner's alpha01 window, 2026-10-23 00:00 to 06:00 EDT (2026-10-22 21:00 to 2026-10-23 03:00 on alpha01's clock, UTC-7). Session: one attended session; the owner approved work on the build host for the window ("You are go to start at 12am EST to 6am EST"), enabled the gate's tt_silicon class, and said "Do whatever you have to do. Go Go!" (OQ-070, OQ-073). End reason: PLACEHOLDER (the report is written as the window runs; the last section is filled at its end).

## Completed

| Task or item | Commit | Result |
| --- | --- | --- |
| RNGD sweep scripts | d6ee127, 8d061da, 27b835e, 2813419 | plans/runs/rngd-sweep/run.sh serves each model of a table in turn on port 8123, polling `furiosa-smi status` until enough cards read 0.00 GiB |
| OQ-071 (npu0) | e5f9440 | Agent Rule 8 claims npu0 like any card at 0.00 GiB (bible master rev 333); tools/demo/lassi-live reads the status table without a State column |
| Wormhole smoke | 982e71b, 37231a5 | The five Tier A TT references ran once each on n300 card 0 and passed binary_io (results/tt-silicon-smoke-20261023) |
| extract.unfenced_file_lines | 450a373 | Opt-in reading of unfenced FILE lines (bible master rev 335); the sweep's second pass |
| tt_silicon executor spike, OQ-072 | 00904f1 | plans/spikes/tt-silicon-executor.md; OQ-072 asks the executor's open choices |
| P16.10 | c5a9a1f, b1a2b6b, b1a4038, bc64df5 | Witness checks 30 passed; fuzzer tests 246 passed on Linux; freeze record; exploratory adversary-seeded runs (results/p16-10-adversary-seeded-20261023); the stage's remote tests 4 passed |
| Bible, Tenstorrent and freeze | 2f952fd | Environment State, Device Targets, Execution Backends, Risks, the freeze paragraph, three Decision Log entries (master rev 336); OQ-073 |
| Sweep third pass | 969c69e | projects/lassi-demo/sweep3, correction cap 15, at the owner's request |

## Build-host runs (Agent Rule 17: approved for the window)

All through `uv run tools/rx.py`, from clean commits unless noted.

- Card and host checks, read-only:
  - rx 20261022-210011-exec-0726: `furiosa-smi ps` (empty), `furiosa-smi status` (npu0 to npu7 at 0.00/47.50 GiB), `lspci -nn` (8 RNGD 1ed2:0001, 8 Wormhole 1e52:401e), df, du.
  - rx 20261022-211015-exec-9a1c and rx 20261022-211052-exec-939f: Wormhole nodes, driver 2.8.0, hugepages, sysfs firmware.
  - The tt_silicon spike's probes rx 20261022-213241-exec-45d6 to rx 20261022-214312-exec-59e3.
- Wormhole runs (the agent issued them on the owner's go, one at a time, card 0, outside the sandbox, tracked references only):
  - rx 20261022-211744-tt-silicon-f12e: loopback on ttsim, PASS.
  - rx 20261022-211828-tt-silicon-abe4: loopback on card 0, PASS, but its record reads dirty, because the command wrote its log inside the slot; not reported.
  - rx 20261022-211857-tt-silicon-869d (loopback), 20261022-211920-tt-silicon-72a5 (eltwise_binary), 20261022-211940-tt-silicon-49a1 (eltwise_sfpu), 20261022-212000-tt-silicon-96c5 (matmul_single_core), and 20261022-212024-tt-silicon-daa2 (matmul_multi_core): all PASS on card 0, clean.
  - By the pinned sources, UMD opened every card's node during each program start to read PCI addresses (spike, Finding 1). Two 1 GiB hugepage files and seven UMD lock files remain (rx 20261022-213300-exec-7aeb; OQ-072 (8) asks whether to remove them).
- RNGD sweeps (rx jobs, big, one model at a time on port 8123):
  - rngd-sweep, 20261022-210057-rngd-sweep-491a: first pass, cap 5, models.tsv.
    - Qwen3-Coder-30B: 10 trials, all S0 at the cap; the replies had no fences.
    - Qwen3-32B: both runs ended rc=1, the server returning content null after reasoning used max_tokens.
    - Qwen3-30B-Instruct: 10 trials, all S1 at the cap.
    - Stopped by the STOP file before EXAONE-4.
  - rngd-sweep2, 20261022-214907-rngd-sweep2-cd9a: second pass, cap 5, extract.unfenced_file_lines on, no reasoning parser (models2.tsv). Qwen3-Coder-30B: its trials reach S1 at the cap. Stopped before Qwen3-32B for the third pass.
  - rngd-sweep3, 20261022-220226-rngd-sweep3-6fa1: third pass, cap 15. PLACEHOLDER (results at the window's end).
- P16.10 (slot p16-witness, CPU only):
  - rx 20261022-212552-p16-witness-6488: tests/bench/test_adv_seeded_remote.py, 30 passed.
  - rx 20261022-214701-p16-witness-b32e: tests/agents/test_fuzzer.py, 246 passed.
  - rx 20261022-215349-p16-witness-78eb: the three adversary-seeded recipes, from b1a2b6b.
  - rx 20261022-220924-p16-witness-d6a5: tests/core/test_adversary_stage_remote.py, 4 passed, at bc64df5.

## Blocked Or Waiting

| Task | State | Reason | Evidence |
| --- | --- | --- | --- |
| Training runs | not run | No train split (every item unassigned, OQ-025, Agent Rule 5); no GPU host; RNGD serves inference only | plans/STATUS.md (P7, P8) |
| P16.11 (llm adversary on a real model) | OWNER | OQ-068; lassi/agents/llm.py is frozen and refuses served backends, so a served adversary needs a fresh seeded set or the owner (Freeze decision) | tests/fixtures/adversary/freeze.yaml |
| P16.12 (adversarial training) | BLOCKED | P8 | plans/p16-adversarial.md |
| Model output on silicon | waiting | The sandboxed tt_silicon executor (OQ-072) | plans/spikes/tt-silicon-executor.md |

## Owner Queue

New:
- OQ-071 (npu0), CLOSED.
- OQ-072 (the tt_silicon executor), OPEN.
- OQ-073 (P16.10's host), CLOSED.
- OQ-074 (the scratch root past the cap), OPEN.

Answered and applied: OQ-070 (the Wormholes), ANSWERED; closed once the executor question is settled.

## Remote State

Jobs running: PLACEHOLDER. Slots added: tt-silicon, p16-witness, rngd-sweep2, rngd-sweep3, rngd-sweep4.

Scratch space:
- du of the scratch root: 318180212 KiB before the P16.10 adversary runs and 318333460 KiB after, about 303.4 to 303.6 GiB (rx 20261022-215349-p16-witness-78eb).
- Free space on the scratch disk: the gate measured 120.5 GB before and 120.4 GB after; `df -h /mnt/nvme10` showed 113G free after. On alpha01 (OQ-073) the scratch disk stands in for the C: drive the P16 plan names; nothing ran on the workstation.
- The window's steps ran with the scratch root past the 145G stop line and the 150G cap. The cap had been lifted for 2026-10-21 and 2026-10-22, when the model weights were downloaded, and was not re-checked at the window's start. The Hugging Face cache holds 215G (rx 20261022-222349-exec-ba36). OQ-074 asks the owner what to do with the weights; agents delete nothing there.

## Next

PLACEHOLDER.
