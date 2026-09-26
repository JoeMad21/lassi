# AGENTS.md

Operating manual for coding agents in this repository. Operational instructions only.

## Authority

- `docs/BIBLE.md` is the project bible. It wins over code, prompts, plans, and this file. Read its Agent Rules (restated below) before acting, then only the bible sections that own your task.
- The owner (J) keeps the master copy of the bible outside the repository; `docs/BIBLE.md` mirrors it minus the Local Tooling section. To change the bible: edit `docs/BIBLE.md`, add a dated Decision Log entry (newest first, update the count sentence), and mirror the same edit to the master copy with the local tooling procedure. If the master copy is unreachable, append the exact edit to `plans/BIBLE-SYNC.md` for the owner.
- Changing an Agent Rule, the Attribution Policy, a project's scope, or a pinned toolchain needs an owner decision first (see Owner Queue).

## Agent Rules

These restate the bible's Agent Rules verbatim.

Status tags used throughout this document and the repo:

| Tag | Meaning |
| --- | --- |
| [MEASURED] | From a logged run, with provenance |
| [JUDGED] | From an LLM judge; never presented as a measurement |
| [DESIGN] | Decided here; change only by editing this document and logging why |
| [OPEN] | Unresolved; never pick an answer silently; resolve with a spike and record the result |
| PLACEHOLDER, PROJECTED | Required inside any artifact that shows a value not yet measured |

Hard rules:

1. Never fabricate, relabel, or extrapolate a measurement. Every number carries commit, toolchain pins, device, SDK or driver version, and date.
2. Simulator results name the simulator as the device (ttsim, Cerebras fabric simulator). Never report a simulator result as silicon, and never report simulator timing as performance.
3. Projects are recipes. No project-specific code enters the `lassi/` package; new behavior is a component behind an existing interface.
4. Faithful mode reproduces upstream behavior, quirks included. Fixes are separate named toggles, off in faithful recipes.
5. Evaluation splits are untouchable: no training, prompt tuning, or corpus harvest on eval items, under any method.
6. Model-generated code runs only through the sandbox executor, never directly in a shell.
7. Nothing on the alpha01 root filesystem. Builds, venvs, HF caches, JIT caches, SDK containers, and results live under `/mnt/nvme10/joseph_ufl`; check free space before large runs.
8. Before claiming NPUs, run `furiosa-smi ps` and `furiosa-smi status`, and claim only a card whose memory reads 0.00 GiB in `furiosa-smi status` (`furiosa-smi ps` lists only the caller's own processes). Never claim npu0 (another tenant). Serve on port 8123 and confirm `/v1/models` returns the expected id. Wait for full process exit before relaunching (EBUSY).
9. Tenstorrent silicon: one placement at a time, never looped device opens (they have taken alpha01's network down twice). RL rewards never execute on silicon; silicon runs are evaluation runs a human starts.
10. Every run records its resolved recipe and toolchain pins. Never upgrade a pinned toolchain mid-run; changing a pin is a Decision Log entry.
11. IR is committed in custom assembly format only, never generic form. Corpus IR is never truncated to fit a limit; split it by function or drop it.
12. No credentials in the repo. API keys come from environment variables.
13. RNGD cards that execute Furiosa target candidates are never the cards serving a model arm.
14. Measured beats judged. A judge never overrides an oracle outcome or a profiler measurement, and judged values carry [JUDGED].
15. Every repository surface follows the Attribution Policy: no reference to any AI vendor, assistant, or coding tool, or to AI assistance in design or implementation, in commits, branch names, docs, comments, PRs, issues, or releases.
16. The repository keeps a vendor-neutral `AGENTS.md` at the root, with nested ones where useful. Vendor-named agent files and tool config directories stay local and uncommitted.

## Operating Mode

Work is unattended by default. The owner reviews asynchronously through `plans/OWNER-QUEUE.md`, run reports in `plans/runs/`, pushed branches, and pull requests. Never wait for an answer and never ask questions in an unattended session; record the question in the owner queue and continue with other work.

Once terminal graphics exist (bible, Readability Standards, Terminal Presentation), run `lassi` commands in unattended sessions with `--graphics off` or `LASSI_GRAPHICS=off`, so no command can wait on the graphics prompt, and never save a graphics preset for the owner.

## Session Loop

1. Orient: `git status`, current branch, `uv run tools/status.py summary`. Read `plans/LESSONS.md`. Read `plans/OWNER-QUEUE.md`; apply every item in state ANSWERED first (record the decision where it belongs, unblock tasks, set the item CLOSED).
2. Pick: `uv run tools/status.py next`. It returns a task (resume an ACTIVE one first), an advance (plan the next phase), or none.
3. Do that one item with the Task Lifecycle or Phase Planning below.
4. Leave clean: working tree clean, `plans/STATUS.md` true, branch pushed. End with one line: `SUMMARY: <item> <state> <result>`.

A dirty tree at session start is unfinished work from an interrupted session: inspect it, then finish or revert it before anything else.

## Task Lifecycle

1. `uv run tools/status.py set <ID> ACTIVE`.
2. Read the task in `plans/p<N>-<topic>.md` and the bible sections it cites.
3. Tests first (test-writer role): encode the acceptance criteria as tests; confirm they fail for the right reason.
4. Implement the smallest change that passes. Follow the bible's Readability Standards: docstrings on public interfaces, type hints, functions under about 60 lines, plain ASCII.
5. Check: `uv run pytest -q` (the fast suite; mark long tests `slow`, remote tests `remote`) and `uv run ruff check .`.
6. Audit (rules-auditor role) over `git diff --cached`; it must end with `VERDICT: PASS`. On FAIL, fix and re-audit.
7. Commit once per task: subject `P<N>.<k>: <imperative summary>` (72 characters or fewer), optional body describing the change. The message describes the change and nothing else: no trailers, footers, session links, or tool names.
8. `uv run tools/status.py set <ID> DONE --note "<commit or one-line result>"`, include `plans/STATUS.md` in the task commit, and push the phase branch to origin.

After three genuinely different approaches fail the same test or gate: set the task BLOCKED with a note pointing to the evidence (`plans/spikes/<id>.md`), add an owner-queue item if a decision would unblock it, and move on.

## Phase Planning

For an advance item on phase P<N>:

1. Branch `p<N>-<topic>` (names in the STATUS phase table) from `main` if the previous phase in the work order is merged, else from the previous phase branch tip. Record the base in the phase Note.
2. Planner role: write `plans/p<N>-<topic>.md` from the phase's Build Roadmap row. 5 to 15 tasks, each with ID, title, bible sections, testable acceptance criteria, files, remote needs, and dependencies. Hardware-dependent tasks start BLOCKED with the blocker named. The last task is `P<N>.G`, the phase gate.
3. Point to bible sections; never restate bible lists (restated lists go stale).
4. `uv run tools/status.py phase P<N> ACTIVE`, add every task with `uv run tools/status.py add`, commit `P<N>.0: plan phase`, push.
5. Read `plans/PHASE-NOTES.md` for repository-specific hints first.

## Phase Gate

- The gate is the Gate column of the phase's Build Roadmap row. Run it exactly; record evidence under `results/<gate-id>/` with provenance (see Results).
- Pass: set the phase DONE, or GATE-OWNER when the gate names an owner review (prepare the review packet and add an owner-queue review item). Open a pull request from the phase branch to `main` titled `P<N> <phase name>` whose body summarizes the gate evidence. Never merge it.
- Fail: add fix tasks. After three failed gate attempts, set the gate task BLOCKED and add an owner-queue item.

## Work Order

P0, P1, P2, P4, P5, then P12 and P11, as in the bible's Build Roadmap. A phase may start when the previous phase in this order is DONE, GATE-OWNER, or stalled (only OWNER or BLOCKED tasks left). P11 beyond design notes needs P5. Phases marked BLOCKED in `plans/STATUS.md` stay blocked until the owner records in the phase Note that the blocker is cleared.

## Owner Queue

`plans/OWNER-QUEUE.md` is the only channel to the owner in an unattended session, and the record of every owner decision; in an attended session a question may be asked directly, and its answer is recorded there. Item format:

```
## OQ-<nnn> <Title>
State: OPEN | ANSWERED | CLOSED
Kind: decision | review | access | policy
Blocks: <task IDs or none>
Evidence: <paths>
Question: <one paragraph>
Options: <each with consequences>
Recommendation: <option and why>
Answer: <owner writes here>
```

Rules for [OPEN] items in the bible:

- Factual (answerable by reading source or docs, or by running something): resolve it with a spike (spike-investigator role). Write `plans/spikes/<id>.md` with commands, outputs, and sources; record the result in the owning bible section with a Decision Log entry; tag measured values [MEASURED].
- Choice (preference, policy, cost, access, scope, or anything that depends on the owner): add an owner-queue item with evidence and a recommendation, set dependent tasks OWNER, and continue with unblocked work. Build behind the interface with the choice as an explicit configuration value that has no default and fails with a clear message, so nothing picks silently.
- Never edit an owner's Answer. Never mark an item CLOSED before its answer is applied.

## Stop Conditions

End the session cleanly (tree clean, STATUS true, summary line) when any of these holds:

- The item is done.
- A stop file exists: `.rip-stop` at the repository root, or the gate reports owner STOP (remote work only; local work may continue).
- Proceeding would break an Agent Rule or the Attribution Policy.
- `uv run tools/status.py next` returns none.

## Branches, Commits, Pushes

- Commit under the configured git identity. Never change git identity or global git configuration.
- Branch names: `main` or `p<N>-<topic>`. Never commit to `main`, never merge into `main`, never merge pull requests.
- Never force push, never rewrite pushed history, never pass `--no-verify`, never change `core.hooksPath`. If a pushed commit violates policy, add an owner-queue item; the owner remediates.
- Never weaken `.githooks/`, `tools/check_text_policy.py`, or `.github/workflows/text-policy.yml` to get a commit through. A false positive is an owner-queue item.

## Remote Execution

The remote build host is alpha01 on the I/ONX cluster. Reach it only through `uv run tools/rx.py` (`--help` lists commands). Raw ssh, scp, rsync, and sftp are forbidden.

- `rx run -- '<cmd>'` syncs the current commit (a dirty tree goes as a snapshot commit) into a worktree slot and runs there. Pass complex commands as one quoted string.
- Anything that may take longer than about 20 minutes (LLVM, Polygeist, tt-mlir, tt-metal, CUDA or NVHPC installs, large runs) is a job: `rx job start --big --name <n> -- '<cmd>'`, then `rx job wait <id> --timeout 600` or `rx job status`, doing local work between polls.
- `rx exec -- '<cmd>'` inspects the host from the scratch root without a checkout.
- `rx doctor` before big work; the gate enforces free-space floors, one big job at a time, and a cap on running jobs. Remove stale slots with `rx slot-rm`.
- Keep `du -sh /mnt/nvme10/joseph_ufl` under 120G as far as you can (owner decision, bible Decision Log). Check it before installs and big jobs. Never delete files there. To save space, back files up to the workstation (zipped); a large folder moves off the host only after the owner approves its backup, and removing any host copy after a backup needs the owner's approval. Recommend deletions through the owner queue.
- The gate sets a scratch-only environment: TMPDIR, caches, CARGO_HOME, RUSTUP_HOME, HF_HOME under `/mnt/nvme10/joseph_ufl`, plus `LASSI_SCRATCH`, `LASSI_RUNS_ROOT`, `LASSI_TOOLCHAINS`, and `LASSI_JOBS` (use it for `-j`).
- Toolchains install under `$LASSI_TOOLCHAINS/<name>@<pin>` through scripts in `toolchains/`, never inside a slot. Each has `toolchains/<name>.pin` recording commit or version, build flags, and install path (bible, Toolchain Pins).
- Device classes (RNGD, Tenstorrent silicon, AMD and NVIDIA GPUs) are enabled or disabled in the gate by the owner only; check `rx doctor` (devices_enabled) before device work. A device refusal is final; record the need in the owner queue. `rx devcheck` is the read-only inventory for re-verifying the bible's Environment State.
- Raw run trees go under `$LASSI_RUNS_ROOT`; only summaries and provenance come back into `results/`.

## Results

- Every number in `results/`, docs, reports, or pull requests comes from a logged run with a provenance manifest: commit, clean or dirty tree, toolchain pins, device, SDK or driver version, date. `rx pull <id> --into results/<name>` writes `provenance.json`; add a `summary.md` that cites it.
- Runs from a dirty tree are exploratory and are never reported as [MEASURED].
- Values not yet measured carry PLACEHOLDER or PROJECTED inside the artifact.

## Repository Checks

- `git config core.hooksPath .githooks` in every clone. The commit-msg, pre-commit, and pre-push hooks run `tools/check_text_policy.py`: owner pattern list (kept outside git), plain ASCII, branch names, commit identities.
- CI runs the same checker on every push and pull request.
- Before ending a session: `uv run pytest -q` passes and `uv run tools/status.py check` passes.

## Roles

When a step names a role, use a dedicated sub-agent for it if your environment provides one; otherwise do the step yourself as described.

- planner: turns a Build Roadmap row into `plans/p<N>-<topic>.md` and STATUS tasks.
- test-writer: writes failing tests from acceptance criteria before implementation.
- rules-auditor: reviews a diff against the Agent Rules, the Attribution Policy, Readability Standards, and this file; ends with `VERDICT: PASS` or `VERDICT: FAIL` plus reasons.
- spike-investigator: gathers evidence for an [OPEN] item and writes `plans/spikes/<id>.md`.

## Files

- `plans/STATUS.md`: phase and task state; change it only with `tools/status.py`.
- `plans/OWNER-QUEUE.md`: owner decisions, reviews, access requests.
- `plans/PHASE-NOTES.md`: repository-specific hints per phase.
- `plans/LESSONS.md`: working lessons; read it before a task, and append what cost a retry, an extra audit round, or a wait.
- `plans/p<N>-<topic>.md`: phase plans. `plans/spikes/`: evidence. `plans/runs/`: run reports.
- `tools/`: `check_text_policy.py`, `policy_canary.py`, `status.py`, `rx.py`, `check_setup.py`; `tools/server/` is the gate the owner installs on the build host.

## Run Reports

At the end of an unattended run, write `plans/runs/<run-id>.md`: tasks done with commit hashes, tasks blocked and why, owner-queue changes, remote jobs still running, scratch free space, and the next item. Commit it on the current phase branch and push.
