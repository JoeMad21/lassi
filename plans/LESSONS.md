# Working Lessons

What slowed work down, and the practice that avoids it next time. Read this before starting a task; append a line when something costs a retry, an extra audit round, or a wait. Newest entries at the end of each section. Plain ASCII.

## Owner directions

- Minimize activity on alpha01 (owner, 2026-09-25). Work on the local copy until a step needs Linux, a compiler, the simulator, or a device. Batch remote tests into one `rx run` per task where possible, and never poll the host more often than the work needs.
- Record inefficiencies and hang-ups here as they happen (owner, 2026-09-25).
- The owner cannot monitor on 2026-09-26: work unattended, push each commit, and take the recommendation of every owner-queue item the owner does not answer personally, recording it on the item as the owner's standing direction and marking it for review later (owner, 2026-09-26).

## Audits and the bible

- Most audit FAILs in P2 and P4 were bible or docstring text that claimed more than the code does (P2.9 refusal paths, P4.7 preset strictness and error format, P4.1 CI coverage). Write the bible text after the code, from the code, and check each claim against a line of code or a spike line before sending the audit.
- Agent reports can state a fact wrongly (P4.7: alpha01's home "on /"; it is the scratch root, plans/spikes/p0-sandbox-hardening.md). Verify every factual claim against a spike or a probe before it enters the bible.
- A "every refusal before any directory exists" contract needs the whole output built in memory first and only I/O after mkdir (P2.9 took four audit rounds to reach that). Design it that way from the start.
- Master bible edits: rewrite a multi-clause sentence by replacing its whole block; chained partial find-and-replace edits on overlapping text produced broken sentences (P4.7, master revision 210).
- Apply one task's master edits, mirror, and commit before starting the next task's master edits; interleaved edits leave the mirror header naming a revision whose content it does not fully hold.
- In an attended session, when the master copy of the bible is unreachable, tell the owner and wait for it to be restored before editing the bible (owner, 2026-09-25). An unattended session follows AGENTS.md, Authority, and appends the edit to plans/BIBLE-SYNC.md.
- Audit the bible text before the master edit (from P4.3 on): apply it to the mirror with a provisional header revision, audit the staged diff, and only then send one clean edit to the master and set the real revision. Editing the master first cost P4.1 and P4.7 extra master revisions for every wording fix.
- Keep each task's bible text in one scratch script that prints the master payload and applies the mirror, so both copies use the same strings.
- Test a threshold's boundary values exactly, never with approx: P4.4's pcc tests used approx and missed that two identical arrays could score 0.9999999999999998 and fail a pcc threshold of 1 (commit audit).
- Before the audit, tick off every bible section the task's acceptance names; P4.4 missed its Harness Contract edit, and scoped rules (which runs, which settings) were written wider than the code (commit audit).
- When a task changes what a shared term means (P4.6 made a clean run also need no UB and no simulator gap), grep every reader of that term before the audit: the lassi profile still read exit 0 and no hang as clean, and four audit lenses found it.

## Agents and briefs

- Give agents task-specific log names in the shared scratchpad (for example p43-full-suite.txt). Two agents wrote the same full-suite.txt, and one read the other's failure as its own (P4.3).
- Brief build scripts for the smallest build that serves the task. The first P4.2 draft built everything with Python bindings on; the owner wants minimal host activity, so name the needed targets in the brief.
- Writing a task's tests and then its implementation took roughly 45 minutes of wall time in P4.3 (run times, not a measurement). Write the next task's tests while the current task is implemented, when their files do not overlap.
- Brief the test-writer to update repository-wide invariant tests that a new kind of component breaks; P4.5's host compiler pin broke a test asserting every toolchain has PIN_BIN and PREFIX_NAME, found only at implementation.
- Never kill processes by image name on the workstation (`taskkill /IM python.exe`): it ends every Python process, including a parallel agent's test run (P4.5 did it once while P4.4 was running). Kill a hung process by its PID.

## Owner queue

- The owner's editor can overwrite items appended while the file is open in it (OQ-025 was lost once). Re-read plans/OWNER-QUEUE.md just before committing, and prefer asking in the session when the owner is present.
- An answer given in the session is recorded on its own line, "Owner's choice (asked in the working session, <date>; recorded by the agent)", never in the owner's Answer field. An ambiguous written answer is settled by asking at once in an attended session, never by reading one option into it; an unattended session records the ambiguity as a Response and leaves the item OPEN.

## Status and planning

- tools/status.py cannot change a task's title or dependencies. Put tasks in their final order, with the gate last and depending on every task it needs, before `status.py add`; if it is wrong before the plan commit, restore STATUS from HEAD and re-add.

## Git on the workstation

- Several agents staging at once can leave `.git/index.lock` for a moment; retry once. Commit a task with `git commit -m ... -- <its paths>`, which commits those paths' working-tree content and leaves other staged work alone, and never edit a path another pending commit holds.

## alpha01

- It is shared and changes under us. It had rebooted about 1h46m before rx 20260925-153149-exec-4e68 (its uptime). The shared disk's free space was 427 GB at the P4.1 spike's rx doctor (plans/spikes/p4-tt-pins.md) and 179 GB at an rx doctor on 2026-09-25 about 15:31 (doctor output, no rx id). All eight RNGD cards were held by other tenants on the evening of 2026-09-24 (rx 20260924-204044-exec-641f, 20260924-204055-exec-6e99). Run `rx doctor` and a `du` check before every big job, and check `furiosa-smi status` early whenever a card is needed.
- `rx pull --path <file>` extracts into .rx/pulls/<basename>, so two single-file pulls overwrite each other and `--into` copies nothing for a file. Pull a directory into its own folder, then pick files.
- The gate refuses any command text that names the Tenstorrent device path or the Tenstorrent SMI tool, even in a read-only probe. Keep those strings out of rx command lines.
- Look for shared state before running another project's tooling on a shared host. tt-metal's JIT deletes other entries in its cache root at every program start, and the default root sits under HOME, which the TurboQuant project shares; always set TT_METAL_CACHE and TT_METAL_RUNTIME_ROOT (PHASE-NOTES P4). Read an upstream tool's startup and cleanup code before its first run.
- Verify an assumed host fact with one read-only `rx exec` before the audit, not after: P4.5's EXPECT_VERSION was taken from a different binary's banner, and a mismatch found by the remote test would have cost a second commit round (rx 20260925-192127-exec-fcd6 confirmed the banner text before the audit).
- `rx pull --path` reaches only lassi-gate/runs, lassi-runs, and lassi-wt; a file under the toolchains root (an install record, a fetched-sources list) is read with one `rx exec cat` and quoted in the task's summary.md.
- The P4.2 job, building only the needed targets (676 Ninja steps) at 32 jobs, took 3 minutes 44 seconds in all: fetch, configure, build, checks, and the ttsim install (results/p4-tt-install). Name only the targets a task needs; match the `rx job wait --interval` to the expected length (300 seconds for a long job, the default for a build of this size) so a short job is not left waiting on a poll.
- A results/<name>/ folder holds what `rx pull --into` wrote plus summary.md (AGENTS.md, Results). Evidence read by `rx exec` goes inside summary.md with the exec id, not in extra files; P4.2 wrote two extra files and had to fold them back.
- Count each grep section's worst case before a read-only `rx exec`: two P4.9 probes printed 173 and 123 lines, over the 120-line practice.
- Dry-run a remote batch locally first, with stand-ins in a local Linux user namespace (a fake scratch root bind-mounted at /mnt/nvme10/joseph_ufl). It found four P4.9 script bugs with no host activity: the helper's import path, the stdout budget, the adaptive limits after a fallback, and a tool looked up by name after the wrapper narrowed PATH.
- Audit a remote batch for shared-host safety before its first launch: the P4.9 review found a network fallback that would have re-enabled the host network and a locale that would have put non-ASCII quotes in the evidence, both before any host activity.
- A 1-byte core limit, the only one that stops alpha01's piped systemd-coredump (plans/spikes/p0-sandbox-hardening.md), needs `prlimit --core=1`; bash's `ulimit -c 1` sets 1024 bytes, and a local check with `ulimit -c` hides it. Check with getrlimit(RLIMIT_CORE) == (1, 1). The P4.9 batch audit caught this before any host activity.
- Any wrapper that re-raises a child's fatal signal can undo a 1-byte core limit: strace sets its own RLIMIT_CORE to 0 and re-raises, and 0 does not stop alpha01's piped systemd-coredump. Put a shell that exits normally between such a wrapper and the program (P4.9 recheck, 2026-09-26).
- A clean-commit remote run needs a clean checkout. When the main tree holds other staged work, commit the task, then run rx from the detached clean worktree C:/dev/lassi-clean checked out at that commit, as the P1 and P2 evidence runs did.
