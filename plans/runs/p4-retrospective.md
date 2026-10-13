# P0 to P4 retrospective

Written 2026-10-05, after the P4 gate passed and pull request 4 merged into main (0dcc261). It covers P0, P1, P2, and P4 (P3 is BLOCKED and the work order skipped it) and extends plans/runs/p0-retrospective.md. Its sources are the commit history on main, plans/, results/, and the session and agent records.

How the numbers were read:

- Wall time runs from a phase's plan commit to its gate commit (commit times, EDT).
- Active time is the part of that span covered by session or agent records with no gap longer than 30 minutes. It is an approximation from the records, not a measurement.
- rx ids carry alpha01's clock (UTC-7), three hours behind the commit times.
- Durations are approximate wherever work overlapped.

## Phases at a glance

| Phase | Plan commit | Gate commit | Wall time | Active time | Tasks (gate included) | Commits, plan to gate | Usage-limit stops |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P0 Core | a9c8c06, 09-22 21:53 | 894be87, 09-24 01:14 | about 27.3 h | about 25 h | 21 (14 planned, 7 added) | 50 | 1 |
| P1 Faithful LASSI | 3d47f1c, 09-24 01:46 | 5d5fd0c, 09-24 13:43 | about 11.9 h | about 6.3 h | 13 | 29 | 1 (weekly) |
| P2 Scoring | 814f9c0, 09-24 13:54 | cc57d0d, 09-24 21:08 | about 7.2 h | about 6.1 h | 11 | 15 | 0 |
| P4 ttsim Execution | 2c5f491, 09-24 21:51 | ca4f8d1, 10-05 15:41 | about 257.8 h (10.7 days) | about 28 h | 16 | 30 | 6 |

Notes:

- P1.12 closed after the P1 gate (ca0bdb6, 09-24 16:28), once the owner answered OQ-018, so its commit is one of P2's 15.
- P1 counted 13 commits labeled "P1:" besides the task commits: the 15:00 demo's recipes, runs, and evidence, two fixes found on the way (7c5cad7, a model id's slash in trial ids; 6566e87, OpenMP threads in native runs), stale recipe comments (21131f7), and a corrected P1.9 note (4d7928e).
- P2's gate is an owner review (OQ-024). Its gate commit records the checks; the review came later.
- P4's 30 commits include the P17 plan (e2cacb0) and three commits that record owner answers, owner directions, or lessons.
- Every P4 task with remote acceptance took two commits: the change, then the evidence from that clean commit (P4.2, P4.5, P4.9, P4.10, P4.11, P4.12, P4.13, P4.15). No P4 task needed a fix commit after its own commit, because every audit ran before the commit.

P4 task spans, from each task's first session or agent record to its last commit (wall time, idle stretches included):

| Task | Span | Commits | Usage-limit stops crossed |
| --- | --- | --- | --- |
| P4.1 joint pin spike | 09-24 21:51 to 22:39, about 0.8 h | 1 | 0 |
| P4.7 graphics setting | 09-24 21:51 to 23:21, about 1.5 h | 1 | 0 |
| P4.3 lassi_io harness | 09-25 18:38 to 20:12, about 1.6 h | 1 | 0 |
| P4.2 tt-metal and ttsim install | 09-25 18:38 to 21:13, about 2.6 h | 2 | 0 |
| P4.4 binary_io oracle | 09-25 19:26 to 21:47, about 2.4 h | 1 | 0 |
| P4.5 native C++ and executors per language | 09-25 20:31 to 23:05, about 2.6 h | 2 | 0 |
| P4.6 simulator readings | 09-25 21:06 to 09-26 16:39, about 19.6 h | 1 | 2 |
| P4.10 tt-metal host toolchain | 09-25 23:06 to 09-26 17:42, about 18.6 h | 2 | 2 |
| P4.9 ttsim runtime spike | 09-25 21:28 to 09-28 00:30, about 51 h | 2 | 4 |
| P4.15 P2 review questions | 09-26 01:33 to 09-27 23:53, about 46.3 h | 2 | 2 |
| P4.8 progress hook and live table | 09-26 17:10 to 09-28 01:28, about 32.3 h | 1 | 2 |
| P4.11 ttsim executor | 09-30 18:10 to 20:30, about 2.3 h | 2 | 0 |
| P4.12 CPU -> TT guard | 10-01 07:56 to 10-05 01:21, about 89.4 h | 2 | 1 |
| P4.13 Tier A suite | 10-05 01:21 to 15:28, about 14.1 h | 2 | 1 |
| P4.14 Tier A splits | 10-05 15:28 to 15:29 | 1 | 0 |
| P4.G gate | 10-05 15:29 to 15:41 | 2 | 0 |

Every P4 task whose span passed 3 hours crossed at least one usage-limit stop. Every task that crossed none finished in under 3 hours.

## Where the time went

### P1 (about 11.9 h wall, about 6.3 h active)

| Work | Time (EDT) | Hours | Why |
| --- | --- | --- | --- |
| Planning, and the RNGD serving spike for the demo | 01:16 to 01:48 | about 0.5 | Planner with a critic. The 15:00 demo needed a served model. |
| A lost first wave | 01:47 to 01:49 | under 0.1 | A question the owner typed during the launch reached the wave's agents, and they answered it instead of their tasks. |
| P1.1 as a submodule, then redone | 01:49 to 02:24 | about 0.6 | The text-policy check refused the staged gitlink (OQ-020). P1.1 was redone in wave 2 as a manifest and a fetch tool. |
| Waves 2 to 4: P1.1 to P1.5, P1.7, P1.8, the P1.9 backend, DEMO.1, DEMO.2 | 02:26 to 05:31 | about 3.1 | Planned work: one review and at most one fix pass per task. |
| Weekly usage limit | 05:43 to 11:22 | about 5.6 | Wave 5 (P1.6, P1.10) and the wave-4 audit failed. The limit reset at 07:00, and work resumed when the owner wrote at 11:22. |
| P1.6 rerun, demo recipes, the RNGD and CPU demo runs, their evidence | 11:22 to 13:20 | about 2.0 | The owner's 15:00 demo (PHASE-NOTES P1). |
| P1.9 harness, P1.10, the gate | 13:08 to 13:43 | about 0.6 | The last wave died when the session restarted at 13:07. It resumed from its run record and finished at 13:23. |

### P2 (about 7.2 h wall, about 6.1 h active)

| Work | Time (EDT) | Hours | Why |
| --- | --- | --- | --- |
| Planning (P2.0) | 13:43 to 13:54 | about 0.2 | Planned. |
| P2.1 tests and code | 13:54 to 14:33 | about 0.7 | Its audit then waited until 16:19. |
| Demo support for the owner | 14:14 to 15:24 | about 1.2 | Commands, a demo script on alpha01, and a live table. The first commands for the owner's SSH session used workstation paths that do not exist on alpha01 (09-24 14:29). |
| Paused | 15:58 to 16:18 | about 0.3 | The owner suspended all work to keep the usage allowance for another task. |
| P2.1 audit through P2.8, with the P2.4 and P2.5 spikes and P1.12 | 16:18 to 18:41 | about 2.4 | One audit per task, except P2.8 (two). |
| P2.9 lassi score | 18:47 to 20:03 | about 1.3 | Five audit passes, four of them FAIL, all on real code paths. |
| Terminal presentation docs, OQ-022 and OQ-023 | 19:37 to 20:22 | about 0.7 | The owner's request at 19:37. Overlaps P2.9 and P2.10. |
| P2.10 and the gate | 20:03 to 21:08 | about 1.1 | Each needed a second audit. |

### P4 (about 257.8 h wall, about 28 h active)

About 230 of P4's 258 wall-clock hours had no session or agent activity.

| Idle stretch (EDT) | Hours | What happened |
| --- | --- | --- |
| 09-25 23:11 to 09-26 01:31 | about 2.3 | Usage limit (reset 00:30). |
| 09-26 02:09 to 16:07 | about 14.0 | Usage limit (reset 06:30). The owner resumed at 16:08. |
| 09-26 17:46 to 09-27 23:25 | about 29.7 | Usage limit (reset 21:00). |
| 09-28 00:00 to 00:17 | about 0.3 | Usage limit (reset 00:10). |
| 10-01 12:19 to 10-04 16:55 | about 76.6 | Usage limit (reset 12:50) during P4.12's build. Work resumed in a new session. |
| 10-05 02:08 to 14:18 | about 12.2 | Usage limit (reset 02:50) during P4.13. A resume at 13:03 failed on a login refresh error, and nobody retried until 14:18. |
| Subtotal, after usage limits | about 135 | Six stops. About 125 h of it came after the limit had reset. |
| 09-24 23:41 to 09-25 18:30 | about 18.8 | The owner interrupted the session and resumed it the next evening. The records do not say why. |
| 09-28 01:36 to 09-30 18:03 | about 64.4 | The owner ordered all work suspended by 02:00. |
| 09-30 20:30 to 10-01 07:55 | about 11.4 | The session ended its turn after reporting P4.11. Nothing started P4.12 until the owner wrote "Keep going." |
| Subtotal, owner stops and an ended turn | about 95 | |

| Active work | Active time | Why |
| --- | --- | --- |
| P4.12 CPU -> TT guard | about 12.8 h (10-01 07:56 to 12:19; 10-04 16:55 to 10-05 01:21) | Nine multi-agent runs, 134 agents in all, about 12.5 h of run time. 13 review rounds before the commit (below). |
| P4.1 to P4.7, P4.10, the P4.9 batch, P4.15 research (09-24 21:51 to 09-26 17:46) | about 8.8 h | Up to five tasks in parallel. Each needed 2 to 7 audit passes, most of them for text. |
| P4.15 and P4.9 close-out, P4.8, P17 planning, a live demo (09-27 23:25 to 09-28 01:36, idle 00:00 to 00:17) | about 1.9 h | P17 planning (owner request) took four audit rounds. The owner asked for a live demo by 03:00. |
| P4.11 ttsim executor | about 2.5 h | A shared design file. One code blocker. |
| P4.13, P4.14, the gate (10-05 01:21 to 02:08; 14:18 to 15:41) | about 2.2 h | Per-item agents in parallel slots, one review round, gate passed on its first run. |

## Review iterations

From P1 on, the rule was one review round per task and one commit audit, with no new audit for a wording or label fix (p0-retrospective.md, Changes from P1 on). The P1, P2, and P4 plans each repeat it in their constraints.

### P1

P1 kept the rule.

- In-wave reviews: each task had one review round and at most one fix pass.
  - P1.2, P1.9, P1.10, and P1.11 passed.
  - Seven reviews found something to fix:
    - P1.1, attribution wording;
    - P1.3, tests checked set membership, not bytes;
    - P1.4, a raw reply stored without the guard;
    - P1.5, attribution wording;
    - P1.6, the faithful error text (major);
    - P1.7, an oracle check removed with nothing in its place (major);
    - P1.8, a stale hint.
- Commit audits ran per wave.
  - Five completed. One failed on bookkeeping (wave 2: a status row in the wrong commit), and that was fixed without a re-audit.
  - The wave-4 audit was cut off by the weekly usage limit and rerun after it.

### P2

| Task | Audit passes | Blocking findings |
| --- | --- | --- |
| P2.1, P2.2, P2.3, P2.6, P2.7 | 1 each | None. The P2.7 auditor warned that P2.8's staged files would ride along in a plain commit. |
| P2.8 | 2 | Text: a Sim-T row lacked its OQ-022 caveat. It was re-audited although the fix was text. |
| P2.9 | 5 (4 FAIL) | Code: a refusal could still happen after the score directory was created. Four audits found it route by route (an edge case, a NaN, three other routes, the output path's encoding). These were real defects. |
| Docs commit for OQ-022 and OQ-023 | 2 | Bookkeeping: OQ-023 was answered but still OPEN. |
| P2.10 | 2 | Code: run.md left out notes that the Parquet file held (Design Principle 7). |
| P2.G | 2 | Text: two Sim-T ranges in summary.md were false. It was re-audited although the fix was text. |

### P4, without P4.12

| Task | Audit passes | Blocking findings |
| --- | --- | --- |
| P4.0 plan | 2 | Text: estimates with no PROJECTED label, and one more. |
| OQ-021 and OQ-025 answers | 2 | Text: bible accuracy. |
| P4.1 | 2 | Text: two ttsim Facts sentences. |
| P4.7 | 3 (one inside its multi-agent run) | Text: three bible statements stronger than the code. One was fixed by tightening the parser. |
| P4.3 | 2 | Code and text: the reader failed with the wrong error on large dims, plus two more code-against-bible gaps. The lessons change had tool-specific text, an instruction that contradicted AGENTS.md, and host facts with no source. |
| P4.2 | 5 | The scripts passed twice. The completion commit failed twice on text: six findings, then one ("the four others the pin's ttsim CI ran" was false). |
| P4.4 | 2 | Code: two identical arrays could score pcc below 1. Text: two claims wider than the code, and a missing Harness Contract edit. |
| P4.5 | 4 (two commits) | Text only: two, then three one-line fixes, each re-audited. |
| P4.6 | 3 | Code and bible: the task made a clean run also need no undefined behavior and no gap, but the lassi profile still read an undefined-behavior run as clean. |
| P4.9 | 7 (four before the launch, three on the results) | Before the launch, shared-host safety in the batch script: a network fallback, a 1024-byte core limit, and strace re-raising signals. On the results, text: three wording blockers, then one. |
| P4.10 | 2 (two commits) | Bookkeeping: a stale STATUS note, fixed without a re-audit. |
| P4.15 | 5 completed, 1 cut off by a usage limit | Text: a stated cause that the spike's own output contradicted, and a Decision Log row claiming unchanged code. Bookkeeping: the bible listing, STATUS, and evidence cited but not in the commit. |
| P4.8 | 3 | Code: a failing observer's guard crashed the run when stderr itself failed; then the exit status still changed (0 to 120). |
| P17 planning (inside P4) | 4 | Text consistency: holding a task for the owner's review made "the phase stalls" untrue in five files. |
| P4.11 | 3 (evidence audit included) | Code: the smoke driver refused the gate's example name. |
| P4.13 | 1 round, two lenses | Text: five wording and comment items, fixed in one pass with no re-review. |
| P4.14 | 0 | A confirmation task. |

In P4, outside P4.12, 28 audit rounds failed (approximate, counted from the auditors' reports):

- 18 failed on text or bookkeeping alone.
- 10 failed on code or shared-host safety. P4.3 and P4.4 were mixed and are counted here.
- Of the tasks whose audits failed on text alone, only P4.10 and P4.13 committed the fix without a re-audit.

### P4.12, the CPU -> TT guard

| Round | When (EDT) | Blocking (confirmed) | Kind |
| --- | --- | --- | --- |
| Build review, four lenses | 10-01, to 12:19 | 20 of 21 | Mostly analyzer behavior (cost, false positives, misses); 4 on docs and rules. The fix pass was cut off by a usage limit. |
| Finish rounds 1 to 3 | 10-04 17:01 to 19:20 | 7, 1, 1 | The limits text claimed more than the code did, or left cases out. Fixed in code and text. |
| Close rounds 1 to 3 | 19:25 to 22:39 | 1, 1, 2 | Rounds 1 and 2 were regressions that the previous fix round had put into the analyzer. Round 3 was text: rule H was false for named lambdas, and a fixture README was wrong. |
| Final audit rounds 1 to 4 | 22:40 to 00:29 | 1, 2, 3, 1 | Text only: the sentences on named lambdas called inside other lambdas, the P4.13 hint, a docstring, a test comment. The fixes changed no code. |
| Last audit | 00:31 to 00:50 | 2 | Text: the P4.13 hint said the limits list the other forms, though the limits say they are not complete. The rewrite had also dropped the self-call exclusion. |
| Hint confirmation | 00:51 to 01:00 | 0 | Pass. |
| Evidence audit | 01:09 to 01:19 | 0 | Pass, after the remote test (10 passed, rx 20261004-220812-desktop-8r113ei-detached-9e8dc504-a6b1). |

The rounds:

- There were 13 review rounds before the code commit (9e8dc50, 10-05 01:07).
- The analysis code last changed in close round 3's fix. Six rounds followed, and they raised nine blocking findings, all of them text.
- Five rounds in a row (close round 3 and final rounds 1 to 4) failed on sentences about lambdas called inside other lambdas. Each round found another sub-case.
- The sentence "These are the known cases, found by review, not a complete account" entered the limits at the start of the close run (10-04, about 19:25 to 20:06). Nine more rounds followed it.
- The rounds stopped when the P4.13 hint became plain instructions that a probe checks, ending in "the guard's limits ... list the known ones, and that list is not complete" (PHASE-NOTES P4).

## Causes

1. Usage limits, with nothing to resume work after them. This was the largest share of wall time.
   - There were eight usage-limit stops in P0 to P4: one in P0, one weekly limit in P1, and six in P4.
   - About 135 of P4's 258 hours followed a stop, and about 125 h of that came after the limit had reset (session records). The 10-01 stop alone was followed by 76.6 h with no activity.
   - Each resume waited for the owner to type. The unattended session driver, which waits through a limit, was last used on 2026-09-22 for P0.0 and P0.1 (plans/runs/20260922-2146.md, 20260922-2223.md).
   - A stop also skips the end-of-run report that AGENTS.md asks for. plans/runs/ holds no run report after 2026-09-23.
   - In P4 there were 18 agent failures on a usage limit, by 15 distinct agents (some failed twice or three times after being resumed). They were working on P4.6, P4.8, P4.9, P4.10, P4.15, the P17 plan, P4.12's fixer, and P4.13's integrator and two reviewers. One of them, P4.15's bible payload builder, left nothing usable.
   - Three of the six P4 stops came at the end of the largest fan-outs:
     - 09-26 02:09, three concurrent multi-agent runs of 32, 23, and 9 agents;
     - 10-01 12:19, a 49-agent build;
     - 10-05 02:06, a 15-agent run.
   - The other three came while three or four single agents ran (09-25 23:08, 09-26 17:45, 09-28 00:00). The records do not settle whether fan-out size caused any stop. The usage allowance is shared with the owner's other work (the owner's request of 09-24 15:58).
2. Owner availability, in a mode that needs the owner to restart work.
   - About 95 h of P4 had no work because the owner stopped it, or because nobody restarted it after an ended turn (the three stretches above).
   - Owner requests also ran inside phases:
     - the 15:00 demo of 09-24, about 2 h in P1 and about 1.2 h in P2;
     - a live demo and the P17 plan inside P4 (09-27 23:46 to 09-28 01:36).
3. Review rounds spent on text, against the P0 rule.
   - Text fixes were re-audited in P2.8, P2.G, the P4.0 plan, the OQ-021 and OQ-025 answers, P4.1, P4.2 (twice), P4.5 (twice), P4.9's results (twice), P4.15 (three times), and the P17 plan (three times).
   - Most text FAILs were bible or docstring text that said more than the code (plans/LESSONS.md, Audits and the bible). Writing the text from the code first did not stop the recheck habit.
4. P4.12 made an open-ended limits list into an audit target.
   - The plan's acceptance asked for a component, four hand-written test programs, and the method and its limits in the Harness Contract (plans/p4-ttsim.md, P4.12).
   - The design run compared three designs (static data flow, null-kernel probe, layered static) and chose the static interprocedural data-flow analyzer.
   - The commit 9e8dc50 changes 31 files (23 of them new) and adds 11,318 lines:
     - lassi/toolchains/_cxx_scan.py, 2,460 lines;
     - _cxx_flow.py, 1,676;
     - ttmetal_guard.py, 489;
     - tests/toolchains/test_ttmetal_guard.py, 4,336.
   - Any audit lens could find a construct that the list did not name. The bible's limits bullet ended as one paragraph of 1,704 words (docs/BIBLE.md:662).
   - Fix rounds that changed the analyzer created the next round's blockers (close rounds 1 and 2).
   - Advisory code changes in close round 3 made an unaudited delta, which needed one more multi-agent audit run.
   - The P4.13 hint in PHASE-NOTES restated the analyzer's behavior. That put each limits sentence in a second place to keep true; final rounds 2 to 4 and the last audit all had blockers on the hint.
   - The claim that every listed reading is pinned by a test let a consistency lens find partial pins in each of the three close rounds.
5. Facts taken from the wrong source. Audits or a later task caught each of these.
   - P4.9 read eltwise_sfpu's tolerance from the comment above the code (2e-2), not from the code (eps 5e-2 at eltwise_sfpu.cpp:147).
     - The value reached the P4.13 brief, assets/bench/tt-pairs-v0.yaml, and tests/bench/test_tt_pairs.py.
     - The P4.13 manifest stage then read the code (rx 20261004-225947-p413-fw-8362). The main session confirmed it with rx 20261005-111942-exec-4372 and corrected it in 5e72f4e.
     - No test could have caught it: the measured max_abs, 0.015625, passes both bounds.
   - Others:
     - P4.7 put alpha01's home "on /" (it is the scratch root);
     - P4.5 took EXPECT_VERSION from another binary's banner;
     - P4.2 claimed "the four others the pin's ttsim CI ran";
     - P4.15 stated a cause that its spike's output contradicted, and wrote "unchanged since the P2 gate" when one function had changed;
     - P2.G gave two Sim-T ranges wrongly;
     - P4.12's spike file misnumbered the ReadShard lines (fixed before its audit, 10-04 19:23).
6. Session and environment friction.
   - A login refresh error ended the first turn of five resumes (09-25 18:30, 09-26 16:08, 09-27 23:26, 09-30 18:03, 10-05 13:03). The last of these cost 75 minutes.
   - The master bible copy was unreachable after two resumes (09-25 18:31, 09-27 23:35). On 09-27 the fix was expected to need a session restart, so the P4.8 implementer was stopped mid-edit (23:36). The copy was back at 23:37 (the records do not show whether a restart happened), and the implementer resumed at 23:47.
   - The 10-04 and 10-05 session started outside the repository, so the project's agent roles were not loaded. Reviewers worked from the rules-auditor checklist by hand.
   - A 15-minute status schedule set at 10-04 17:20 never fired. Its replacement had to be re-armed every 30 minutes.
   - Windows effects:
     - Path.write_text writes CRLF (tools/status.py, PHASE-NOTES All Phases);
     - non-ASCII host output raises a cp1252 error locally and loses the rx id (PHASE-NOTES All Phases).
7. Remote host friction (alpha01 is shared).
   - All eight RNGD cards were held by other tenants on the evening of 09-24, so the owner could not run live inference at 23:41.
   - The shared-host hazards (core dumps on the root filesystem, the tt-metal JIT cache deleting other entries, a network fallback) needed four pre-launch audit passes for P4.9.
   - A new rx slot lacks the generated prompt files, which cost one demo check run (plans/LESSONS.md, alpha01).
   - A run inside a snapshot slot records a dirty snapshot as clean (PHASE-NOTES All Phases).
   - P4.13 left seven rx slots and several run trees on alpha01 (PHASE-NOTES P4).
   - The Actions variable TEXT_POLICY_PATTERNS, due for deletion after the P0 merge, still existed at the 10-05 15:53 check. OQ-013 is CLOSED, so no open item tracks it.

## What worked

1. Per-item agents in their own rx slots (P4.13).
   - All five item drafts were done in about 17 minutes (01:21 to 01:38).
   - Each item agent iterated in its own slot (p413-loopback and the rest). Every pair passed its remote test on its first run (exploratory: rx 20261004-230418-p413-loopback-51fc, 20261004-230443-p413-eltwise_binary-c2d9, 20261004-230426-p413-eltwise_sfpu-189d, 20261004-230432-p413-matmul_single_core-9c86, 20261004-230426-p413-matmul_multi_core-b038).
   - After the integration, there was one review round with two lenses and one fix pass with no re-review.
   - The clean-commit evidence run passed 5 of 5 (rx 20261005-122628-desktop-8r113ei-detached-5e72f4e8-56d0).
   - P4.13 took about 2 h of active time for a 32-file commit.
2. One clean-commit evidence run per task, from the detached clean worktree. The gate passed on its first run: the smoke run and the Tier A job from ec1c9a2 ran from 15:30:01 to 15:33:45 (results/p4-gate), and the task took 12 minutes from its start commit to its evidence commit (15:29 to 15:41).
3. Pre-launch audits and a local dry run of the remote batch (P4.9).
   - Before any host activity, they found four script bugs and three shared-host hazards (plans/LESSONS.md, alpha01).
   - The batch then ran once: 24 of 24 steps in 315 s, rc 0 (rx job 20260926-140947-p49-ttsim-runtime-6f4c).
4. A shared design file before the tests (P4.11). The test-writer raised eight points, and the implementer built them in one pass (plans/LESSONS.md, Agents and briefs). P4.11 took about 2.3 h from start to evidence commit, with one code blocker.
5. Minimal remote builds. P4.2's install job, a 676-step build among its steps, took 3 minutes 44 seconds in all (results/p4-tt-install).
6. P1's waves of tasks, each task getting one review and at most one fix pass. 11 of P1's 12 tasks and the gate were done in about 6.3 h of active time (P1.12 waited for the owner's answer to OQ-018), and the gate took about 3 minutes (13:40 to 13:43).
7. Bible text audited on the mirror, then sent to the master in one edit (from P4.3 on). P4.13 sent one 8-operation edit (revision 243 to 244).
8. The owner's standing directions of 09-26 and 09-27. Unattended sessions took the owner-queue recommendations and flagged them for review, so no P4 task waited on an answer (OQ-027 to OQ-042).
9. Snapshots of the files a fix round would change. Each P4.12 audit after them reviewed only that round's delta.

## Changes for P17 and later

Each change can be checked.

1. Review budget.
   - The rule: one review round (its lenses run in parallel) and one commit audit per task. A second round is only for a confirmed code defect. A text blocker is fixed and committed, and the commit body lists it.
   - Check: at the next retrospective, count review and audit passes per task in the session records. More than 2 needs a code defect named in the task's STATUS note.
2. Limits text for heuristics and static analysis.
   - The rule: the limits open with "known cases, not complete", give one line per case, and pin each case with a test. An audit probes each listed case; a case missing from the list is advisory and goes to PHASE-NOTES, never blocking.
   - Check: grep the limits for the opening sentence, and grep the audit brief for "omissions are advisory".
3. Hints are instructions.
   - The rule: a PHASE-NOTES hint for a later task says what to do, points to the bible section, and is checked by a probe. It never restates analyzer behavior.
   - Check: no limits sentence appears twice (grep).
4. Design size.
   - The rule: a design file maps each component to an acceptance bullet and takes the smallest design that meets the acceptance. A larger option goes to PHASE-NOTES under a later phase.
   - Check: the design file has the mapping table, and no component lacks a bullet.
5. Fix rounds after the first audit.
   - The rule: they change code only for a confirmed code blocker. An advisory that needs code becomes a note for a later task. Every code-changing fix reruns the round-start probe set and reports changed readings.
   - Check: each fix report lists its probe differences, and no fix round carries an advisory code change.
6. Spikes read code, not comments.
   - The rule: every value taken from upstream source cites the line where code sets or compares it, and the spike's audit re-reads each cited line at the pin.
   - Check: a script greps each cited line and finds the value outside a comment.
7. Evidence batching.
   - The rule: one clean-commit remote run per task covers all its remote tests. Runs from dirty snapshots stay exploratory.
   - Check: each results/<name>/summary.md cites one evidence rx id per task.
8. Parallel per-item slots.
   - The rule: a suite or multi-item task gets a shared framework first, then one agent per item, each in its own rx slot (p<task>-<item>), then one integrator.
   - Check: the rx ids carry per-item slot names, and PHASE-NOTES lists the slots for cleanup.
9. Resume after a usage limit.
   - The rule: unattended stretches run under the unattended session driver, which waits for the reset and resumes.
   - Target: the next session record within 30 minutes of a reset.
   - Check: compute reset-to-resume times from the session records, as this retrospective did.
10. Fan-out size.
    - The rule: one multi-agent run at a time, of at most about 15 agents unless the brief says why more are needed.
    - Check: agent counts in the run records.
    - Basis: the stops came at the end of the largest fan-outs. Causation is not settled.
11. The end of a task is not the end of a turn.
    - The rule: in an unattended stretch, after a task's evidence commit the session takes `tools/status.py next`.
    - Check: no gap over 30 minutes between a task's last commit and the next task's first record, unless the owner stopped work.
12. Owner-facing commands and owner actions.
    - The rule: a command for the owner names the machine it runs on and uses that machine's paths. An owner action that outlives a CLOSED item gets an OPEN follow-up item.
    - Check: the TEXT_POLICY_PATTERNS deletion from OQ-013 has its own item.
13. Run reports.
    - The rule: an unattended stretch that ends by a usage limit gets its run report at the next session's start.
    - Check: plans/runs/ gains one report per unattended stretch.
