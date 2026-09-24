# P0 retrospective

Written for the owner after the P0 gate passed (2026-09-24). P0 took about
27 hours of wall time: the plan was committed at 2026-09-22 21:53 and the
gate at 2026-09-24 01:14 (EDT, from commit times). It ended with 21 tasks
done and the pull request "P0 Core" open. Durations below are read from
commit times and are approximate, because some work overlapped.

## Where the time went

| Work | Time | Why |
| --- | --- | --- |
| Core records, recipe loader, backends, parser (P0.1-P0.5) | about 9.5 h | Planned work. A session usage limit stopped P0.2 midway. |
| Toolchains, bench, sandbox v1, runner (P0.6-P0.11) | about 5.5 h | Planned work, plus remote installs and two sandbox probes. |
| Sandbox hardening (P0.16) | about 4.3 h | Added mid-phase. Four review rounds; the first found three real escape paths. |
| Compile hardening (P0.20) | about 2.3 h | Added mid-phase, from the sandbox work. |
| Real compiler fixtures (P0.15, P0.17) | about 3.8 h | Added mid-phase. Each needed two commits: code first, then a capture from that clean commit. |
| Owner answers, GitHub settings, canary, hooks | about 2.5 h | Answers applied to the bible; the secret switch; two canary rounds; the hook mode fix. |
| Windows rx fix, provenance, CUDA reinstall (P0.14, P0.18, P0.19) | about 2 h | Added mid-phase. |

The plan started with 14 tasks and ended with 21. Seven were added during
the phase (P0.14 to P0.20), a 50% increase. Each added task carried the full
cycle: contract, tests, implementation, reviews, audit, commit, bible sync.

## Review iterations

Most commit-audit failures were about labels and records, not code:
[MEASURED] on exploratory results, a stale status note, a Decision Log count
off by one, a claim stronger than its evidence. Each failure cost a full
re-audit for what was usually a one-line fix. The sandbox needed its extra
rounds: its first red team found a disk-cap bypass, a core-limit bypass, and
an environment leak. Later rounds found nothing major.

| Task | Rounds in the workflow | Commit audits |
| --- | --- | --- |
| P0.2-P0.12 | 1 each (P0.2: 3) | about 6 failures in total, on labels |
| P0.14, P0.18, P0.17 | 1 | pass |
| P0.15 | 2 (stage A and B) | pass |
| P0.16 | 4 | 2 failures, on labels, then pass |
| P0.20 | 1 plus a recheck | 1 failure (the mixed commit, below) |

## Causes

1. Scope grew inside the phase. Security gaps and owner answers became new
   P0 tasks rather than notes for later phases. The owner's scope freeze
   stopped this on 2026-09-23.
2. Evidence rules added commits. A result counts only from a clean commit, so
   each task with remote acceptance became two commits, two audits, and
   often two bible syncs.
3. Labels were fixed after audit rather than written right the first time.
4. Friction in the tooling:
   - the guard hook refused commands that only mentioned `.githooks`,
     `results/`, or a skipped-hook flag;
   - subagents may not write results summaries;
   - rx keeps only the last 120 lines of output;
   - Windows sent a bare `bash` to the WSL launcher, and `os.kill(pid, 0)`
     kills the process.
5. Work ran in series longer than it needed to. Local-only work and remote
   jobs were overlapped only at the end (P0.17 stage B ran alongside P0.19).
6. My own mistakes:
   - I gave the owner a commit command while my changes were staged, so one
     pushed commit holds two changes (OQ-016);
   - a status edit made one gate run a dirty snapshot;
   - a bug in a workflow script cost one rerun.

## Changes from P1 on

- Phase scope is fixed at planning. New findings go to plans/PHASE-NOTES.md
  under a later phase, or to the owner queue.
- One review round per task (contract, safety, and rules together) and one
  commit audit. A wording or label fix goes in without a new audit.
  Red-team rounds for security code stop at two, unless the second finds a
  major issue.
- Every contract carries an evidence checklist: [MEASURED] only with a
  clean-commit rx id; everything else is labeled exploratory; counts and
  dates are checked before the audit.
- Before any evidence run: commit every change, check that the tree is clean,
  and record pytest's own status (PIPESTATUS), not a pipe's.
- Remote jobs and local work run in parallel wherever they touch different
  files.
- Before any command meant for the owner, the index is empty, or the command
  names its paths.
- Probe output stays under 120 lines per rx call.

## State at the end of P0

- The gate evidence is in results/p0-gate/.
- The pull request is https://github.com/JoeMad21/lassi/pull/1. It is not
  merged; only the owner merges.
- Open owner items:
  - OQ-017: remove the old CUDA runfile tree.
  - After the merge, delete the Actions variable TEXT_POLICY_PATTERNS
    (OQ-013).
- Scratch use is about 101G of the 120G cap.
- Next: P1 starts at once, then P2 (the owner's order of 2026-09-23).
