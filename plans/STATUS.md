# Status

Single source of phase and task state. Edit only with `uv run tools/status.py`; the tables are parsed.
Work order while hardware is blocked: P0, P1, P2, P4, P5, P12, P11 (docs/BIBLE.md, Build Roadmap).
A phase marked BLOCKED stays blocked until the owner records in its Note that the blocker is cleared.

## Phases

| Phase | Branch | State | Note |
| --- | --- | --- | --- |
| P0 Core | p0-core | ACTIVE | base: main |
| P1 Faithful LASSI | p1-faithful | NOT-STARTED | - |
| P2 Scoring | p2-scoring | NOT-STARTED | gate needs owner review |
| P3 RNGD Serving | p3-rngd | BLOCKED | RNGD host (OQ-001) |
| P4 ttsim Execution | p4-ttsim | NOT-STARTED | - |
| P5 IR Levels | p5-ir | NOT-STARTED | - |
| P6 DF Zero-Shot | p6-zeroshot | BLOCKED | P3 |
| P7 Offline Training | p7-offline | BLOCKED | P3, training GPUs (OQ-002) |
| P8 Online Training | p8-online | BLOCKED | MI300X access (OQ-002) |
| P9 LASSI-EE | p9-ee | BLOCKED | MI300X access (OQ-002) |
| P10 Full LASSI | p10-full | BLOCKED | A100 host (OQ-003) |
| P11 Dataflow Dialect | p11-df | NOT-STARTED | full scope needs P5; design notes may start earlier |
| P12 Language Frontends | p12-frontends | NOT-STARTED | - |
| P13 Cerebras Target | p13-cerebras | BLOCKED | Cerebras SDK access (OQ-004) |
| P14 Furiosa Target | p14-furiosa | BLOCKED | RNGD host, TCL authoring (OQ-001) |
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
| P0.7 | DONE | Pin CUDA and NVHPC under LASSI_TOOLCHAINS | P0.6 | cuda@12.6.3 and nvhpc@24.11 pinned and verified from 4ef46ee; nvc++ needs NVHPC_CUDA_HOME; OQ-010 |
| P0.8 | DONE | Bench registry and one pinned HeCBench app | P0.1 | layout at HeCBench 7d2d3c5; eval refused to training; fetch exercised by rx 20260923-050731-desktop-8r113ei-p0-core-cecf (exploratory) |
| P0.9 | READY | Spike: sandbox isolation on alpha01 | - | remote: rx run |
| P0.10 | READY | Sandbox and the none and native executors | P0.1,P0.9 |  |
| P0.11 | READY | Stage runner and lassi run compile-only path | P0.2,P0.3,P0.4,P0.5,P0.8,P0.10 |  |
| P0.12 | READY | Verify text-policy tooling, local half | - |  |
| P0.13 | OWNER | Verify text-policy tooling, CI half | P0.12 | OQ-005 no origin remote; OQ-007 Actions variable and branch protection |
| P0.G | READY | Phase gate: mock LLM compile-only run and text-policy canary | P0.7,P0.11,P0.12,P0.13 | also needs P0.14 (fast suite green) |
| P0.14 | READY | Fix rx local-transport tests on Windows | - | 5 test_rx_gate errors at baseline |
| P0.15 | READY | Replace toolchain stderr fixtures with alpha01 captures | P0.7 | fixtures in tests/toolchains/fixtures are PLACEHOLDER |
