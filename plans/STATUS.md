# Status

Single source of phase and task state. Edit only with `uv run tools/status.py`; the tables are parsed.
Work order while hardware is blocked: P0, P1, P2, P4, P5, P12, P11 (docs/BIBLE.md, Build Roadmap).
A phase marked BLOCKED stays blocked until the owner records in its Note that the blocker is cleared.

## Phases

| Phase | Branch | State | Note |
| --- | --- | --- | --- |
| P0 Core | p0-core | ACTIVE | base: main; scope frozen by the owner 2026-09-23: finish P0.17, P0.19, then P0.G; no new P0 tasks |
| P1 Faithful LASSI | p1-faithful | NOT-STARTED | owner 2026-09-23: start as soon as P0 finishes (after the P0 retrospective), no pause; P2 right after |
| P2 Scoring | p2-scoring | NOT-STARTED | gate needs owner review |
| P3 RNGD Serving | p3-rngd | BLOCKED | RNGD host answered (OQ-001: alpha01, gate rngd enabled 2026-09-23); the owner records here when the phase may start |
| P4 ttsim Execution | p4-ttsim | NOT-STARTED | - |
| P5 IR Levels | p5-ir | NOT-STARTED | - |
| P6 DF Zero-Shot | p6-zeroshot | BLOCKED | P3 |
| P7 Offline Training | p7-offline | BLOCKED | P3; training GPUs: AMD deferred by the owner (OQ-002, 2026-09-23) |
| P8 Online Training | p8-online | BLOCKED | MI300X access deferred by the owner (OQ-002, 2026-09-23) |
| P9 LASSI-EE | p9-ee | BLOCKED | MI300X access deferred by the owner (OQ-002, 2026-09-23) |
| P10 Full LASSI | p10-full | BLOCKED | no NVIDIA host (OQ-003, 2026-09-23); compile-only tier and -mp=multicore proxy meanwhile |
| P11 Dataflow Dialect | p11-df | NOT-STARTED | full scope needs P5; design notes may start earlier |
| P12 Language Frontends | p12-frontends | NOT-STARTED | - |
| P13 Cerebras Target | p13-cerebras | BLOCKED | Cerebras SDK access incoming on the owner's side (OQ-004, 2026-09-23) |
| P14 Furiosa Target | p14-furiosa | BLOCKED | RNGD host answered (OQ-001); TCL authoring; the owner records here when the phase may start |
| P15 Judges | p15-judges | BLOCKED | P9 for measurements |
| P16 Adversarial | p16-adversarial | BLOCKED | P4, P8 for training |

## Tasks

| ID | State | Title | Depends | Note |
| --- | --- | --- | --- | --- |
| P0.0 | DONE | Plan phase P0 into plans/p0-core.md and add its tasks here | - | plan committed; push pending OQ-005 |
| P0.1 | DONE | Package skeleton and the twelve interfaces | - | 87c0d17; pushed to origin/p0-core |
| P0.2 | DONE | Result record, text store, trial.md, Parquet mirror | P0.1 | record, store, trial.md, Parquet mirror; tests split per module; P0.3 adds pyyaml; OQ-008 provenance |
| P0.3 | DONE | Recipe loader with capability validation | P0.1 | loader, registry, base.yaml; P0.4 relaxes empty-registry test, sets constructor convention; P0.11 imports components, maps arms |
| P0.4 | DONE | LLM backends: mock, openai_compat, ollama | P0.1,P0.2 | mock, openai_compat, ollama; files.py renderer for P0.5; handoffs in PHASE-NOTES |
| P0.5 | DONE | FILE-block parser and nvcc, nvc++ toolchain adapters | P0.2 | FILE-block parser, nvcc-sm80 and nvcpp-cc80 adapters; hand-written stderr fixtures pending P0.7 captures |
| P0.6 | DONE | Spike: nvcc and nvc++ on alpha01 | - | nvcc V12.6.85 at /mnt/nvme10/joseph_ufl/cuda-12.6.3 builds sm_80 without a GPU; no nvc++ found (P0.7) |
| P0.7 | DONE | Pin CUDA and NVHPC under LASSI_TOOLCHAINS | P0.6 | cuda@12.6.3 and nvhpc@24.11 pinned and verified from 4ef46ee; nvc++ needs NVHPC_CUDA_HOME; OQ-010 answered: redistributable reinstall is P0.19 |
| P0.8 | DONE | Bench registry and one pinned HeCBench app | P0.1 | layout at HeCBench 7d2d3c5; eval refused to training; fetch exercised by rx 20260923-050731-desktop-8r113ei-p0-core-cecf (exploratory) |
| P0.9 | DONE | Spike: sandbox isolation on alpha01 | - | unshare -rnmpf in a systemd-run --user scope; network blocked; OQ-011 CPU quota and uid |
| P0.10 | DONE | Sandbox and the none and native executors | P0.1,P0.9 | sandbox and native executor; 23/23 remote tests from f57c90a; gaps to P0.16 |
| P0.11 | DONE | Stage runner and lassi run compile-only path | P0.2,P0.3,P0.4,P0.5,P0.8,P0.10 | stage runner, lassi run compile-only path, p0-smoke recipe; stage ladder per bible (S4 compiles) |
| P0.12 | DONE | Verify text-policy tooling, local half | - | check_setup 15 PASS; canary local blocks both commits; --history clean; tests added for --message, --history, canary local |
| P0.13 | DONE | Verify text-policy tooling, CI half | P0.12 | owner ran push and cleanup 2026-09-23; CI run 35921343647 on p0-canary (a039a53) concluded failure on the canary in the commit message and canary.txt; branch deleted; evidence to results/p0-gate in P0.G |
| P0.G | READY | Phase gate: mock LLM compile-only run and text-policy canary | P0.7,P0.11,P0.12,P0.13 | owner order 2026-09-23: runs last, after P0.20, P0.17, and P0.19; P0.14 and P0.18 done |
| P0.14 | DONE | Fix rx local-transport tests on Windows | - | rx local transport works on Windows; also touches tools/server/gate.py (Windows-only branches, refusals unchanged) and adds tests/tools/test_gate_windows.py; gate tests pass on Windows and on alpha01 (rx 20260923-101115-desktop-8r113ei-p0-core-d7ce) |
| P0.15 | DONE | Replace toolchain stderr fixtures with alpha01 captures | P0.7 | fixtures are alpha01 captures (rx 20260923-112105-desktop-8r113ei-p0-core-ba1a from clean ebe6b07); expected lists hand-derived; nvcc drops GCC columns, nvc++ backend and linker places fixed; results/p0-toolchain-fixtures; unparsed formats to P0.17 |
| P0.16 | DONE | Sandbox hardening before native runs of generated code | P0.10 | 59b5799: 46 remote tests from the clean commit (rx 20260923-173420-desktop-8r113ei-p0-core-192f); three review rounds; bible Sandbox and Decision Log at master revision 81; compile-side items to P0.20 |
| P0.17 | ACTIVE | Parse the compiler error formats the P0.15 probes found unread | P0.15 | nvlink undefined reference, ptxas file-line error, nvc++ driver error lines; seen only in exploratory P0.15 probes (rx ids in plans/p0-core.md P0.17) |
| P0.18 | DONE | Carry run provenance in each Trial | P0.11 | Trial.provenance {commit, dirty, device, sdk, date} copied from provenance.json by the runner; trial.md, run.md, and the Parquet trials table show it; Result Record lines of OQ-008 mirrored here |
| P0.19 | READY | Install CUDA 12.6.3 from the redistributable archives | P0.7 | OQ-010 answered (b), 2026-09-23; check the 120G scratch cap before staging |
| P0.20 | DONE | Harden compiles of generated sources | P0.16 | 1de7db6; its acceptance record landed in f9d1e68 under the hooks subject (OQ-016): 63 remote tests and a fixture recapture (12 of 12 byte-stable identical) from 1de7db6; bible at master revision 83 |
