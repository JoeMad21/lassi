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
| P0.0 | READY | Plan phase P0 into plans/p0-core.md and add its tasks here | - | read plans/PHASE-NOTES.md first |
