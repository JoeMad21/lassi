# Working Lessons

What slowed work down, and the practice that avoids it next time. Read this before starting a task; append a line when something costs a retry, an extra audit round, or a wait. Newest entries at the end of each section. Plain ASCII.

## Owner directions

- Minimize activity on alpha01 (owner, 2026-09-25). Work on the local copy until a step needs Linux, a compiler, the simulator, or a device. Batch remote tests into one `rx run` per task where possible, and never poll the host more often than the work needs.
- Record inefficiencies and hang-ups here as they happen (owner, 2026-09-25).

## Audits and the bible

- Most audit FAILs in P2 and P4 were bible or docstring text that claimed more than the code does (P2.9 refusal paths, P4.7 preset strictness and error format, P4.1 CI coverage). Write the bible text after the code, from the code, and check each claim against a line of code or a spike line before sending the audit.
- Agent reports can state a fact wrongly (P4.7: alpha01's home "on /"; it is the scratch root, plans/spikes/p0-sandbox-hardening.md). Verify every factual claim against a spike or a probe before it enters the bible.
- A "every refusal before any directory exists" contract needs the whole output built in memory first and only I/O after mkdir (P2.9 took four audit rounds to reach that). Design it that way from the start.
- Master bible edits: rewrite a multi-clause sentence by replacing its whole block; chained partial find-and-replace edits on overlapping text produced broken sentences (P4.7, master revision 210).
- Apply one task's master edits, mirror, and commit before starting the next task's master edits; interleaved edits leave the mirror header naming a revision whose content it does not fully hold.
- In an attended session, when the master copy of the bible is unreachable, tell the owner and wait for it to be restored before editing the bible (owner, 2026-09-25). An unattended session follows AGENTS.md, Authority, and appends the edit to plans/BIBLE-SYNC.md.
- Audit the bible text before the master edit (from P4.3 on): apply it to the mirror with a provisional header revision, audit the staged diff, and only then send one clean edit to the master and set the real revision. Editing the master first cost P4.1 and P4.7 extra master revisions for every wording fix.
- Keep each task's bible text in one scratch script that prints the master payload and applies the mirror, so both copies use the same strings.

## Agents and briefs

- Give agents task-specific log names in the shared scratchpad (for example p43-full-suite.txt). Two agents wrote the same full-suite.txt, and one read the other's failure as its own (P4.3).
- Brief build scripts for the smallest build that serves the task. The first P4.2 draft built everything with Python bindings on; the owner wants minimal host activity, so name the needed targets in the brief.
- Writing a task's tests and then its implementation took roughly 45 minutes of wall time in P4.3 (run times, not a measurement). Write the next task's tests while the current task is implemented, when their files do not overlap.

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
- A clean-commit remote run needs a clean checkout. When the main tree holds other staged work, commit the task, then run rx from the detached clean worktree C:/dev/lassi-clean checked out at that commit, as the P1 and P2 evidence runs did.
