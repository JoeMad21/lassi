# Owner Queue

The only channel from agents to the owner. Format and rules: AGENTS.md, Owner Queue.
The owner answers by filling Answer and setting State to ANSWERED; the next session applies it.

## OQ-001 RNGD Host
State: CLOSED
Kind: access
Blocks: P3, P6, P7, P14
Evidence: docs/BIBLE.md, Environment State (2026-09-22 check)
Question: Which I/ONX host holds the Furiosa RNGD cards now, and can they return to alpha01 or be reached from it?
Options: (a) cards return to alpha01; (b) agents get gate access on the host that holds them; (c) stay blocked.
Recommendation: (a); it keeps one gate and one scratch root.
Answer: Alpha01 can directly interface with the Furiosa cards. You can run furiosa-smi on the host and get status.
Applied: 2026-09-23. Verified read-only (plans/spikes/oq-001-rngd-host.md); the owner enabled the gate's rngd class the same day (rx doctor shows rngd true). Recorded in the bible (Environment State, Decision Log); P3 and P14 Notes say the host is answered, and the owner records there when each phase may start.

## OQ-002 MI300X Render Group
State: CLOSED
Kind: access
Blocks: P7 (training GPUs), P8, P9
Evidence: docs/BIBLE.md, Environment State; /dev/kfd permission denied for joseph_ufl
Question: Ask an administrator to run usermod -aG render joseph_ufl on alpha01. When granted, set rocm_gpu enabled in the gate config and record it here.
Options: (a) grant; (b) AMD Developer Cloud credit for single-GPU work.
Recommendation: (a).
Answer: Developer access on the AMD GPUs isn't happening yet. Don't worry about AMD right now.
Applied: 2026-09-23. AMD deferred, not dropped; recorded in the bible (Decision Log); P7, P8, P9 stay BLOCKED with the deferral in their Notes.

## OQ-003 NVIDIA A100 Host
State: CLOSED
Kind: access
Blocks: P10
Evidence: docs/BIBLE.md, Risks And Questions, question 2
Question: Which A100 host runs the full LASSI reproduction?
Options: (a) a UF or partner cluster; (b) cloud rental; (c) compile-only tier plus the -mp=multicore proxy only.
Recommendation: (a) when available; (c) meanwhile.
Answer: We currently do not have an Nvidia host, work without it.
Applied: 2026-09-23. Option (c): compile-only tier plus the -mp=multicore proxy; recorded in the bible (Risks And Questions, Decision Log); P10 stays BLOCKED.

## OQ-004 Cerebras SDK Access
State: CLOSED
Kind: access
Blocks: P13
Evidence: docs/BIBLE.md, Device Targets and question 12
Question: Request the Cerebras SDK directly, or through the Sandia collaboration?
Options: (a) direct request; (b) through the collaboration contact.
Recommendation: (b) if the collaboration starts this term, else (a).
Answer: Cerebras access is incoming. Don't worry about it right now.
Applied: 2026-09-23. Recorded in the bible (Decision Log); P13 stays BLOCKED until access arrives.

## OQ-005 Monorepo Path And Remote
State: CLOSED
Kind: decision
Blocks: P0.13, P0.G, every branch push (no origin remote was configured when this was asked; origin is set now)
Evidence: docs/BIBLE.md, Agent Rules workflow and question 3
Question: Local repository path and remote name for the monorepo.
Options: (a) C:\dev\lassi on Windows, ~/dev/lassi on Linux, origin = private github.com/JoeMad21/lassi; (b) other.
Recommendation: (a). Record the answer in the bible and close question 3.
Answer: Stick with the repository on Windows. Push to the Github regularly to ensure alignment.
Applied: 2026-09-23. Repository C:\dev\lassi, origin github.com/JoeMad21/lassi, public during development and private once possible (owner, in the session); recorded in the bible (question 3, Decision Log); plans/p0-core.md no longer defers pushes.

## OQ-006 Kit Decisions Review
State: CLOSED
Kind: review
Blocks: none
Evidence: docs/BIBLE.md, Decision Log, top three entries dated 2026-09-22
Question: Ratify the three decisions added with the agent kit: vendor-free wording of the attribution rule plus the checker canary; remote access through tools/rx.py and the project gate; unattended work order with stacked phase branches and this queue.
Options: (a) ratify; (b) amend.
Recommendation: (a).
Answer: Ratify.
Applied: 2026-09-23. Ratification recorded as a bible Decision Log row.

## OQ-007 GitHub Enforcement Settings
State: CLOSED
Kind: access
Blocks: P0.13, P0.G (CI half of the text-policy gate)
Evidence: docs/BIBLE.md, Attribution Policy, Enforcement items 2 and 3
Question: Set the repository Actions variable TEXT_POLICY_PATTERNS to the pattern list, and protect main with the text-policy check required. Agents cannot change repository settings.
Options: (a) set both; (b) variable only (branch protection on private repositories needs a paid or education plan).
Recommendation: (a).
Answer: Why would you need to change the repository settings? You're able to push and commit aren't you?
Applied: 2026-09-23. In the session the owner approved protecting main and set it (agents do not change repository settings); gh api confirms TEXT_POLICY_PATTERNS and the required text-policy check on main. Recorded in the bible (Attribution Policy, Enforcement; Decision Log); P0.13 is READY.

## OQ-008 Provenance In The Result Record
State: CLOSED
Kind: decision
Blocks: none yet. P0.11 (done) writes run-level provenance only (provenance.json and run.md beside every trial); per-trial provenance in trial.json, trial.md, and Parquet awaits this answer, and P0.G is the first run that writes records
Evidence: docs/BIBLE.md (Agent Rules 1 and 2, Result Record); lassi/core/record.py; lassi/core/trial_md.py
Question: Agent Rules 1 and 2 require every number to carry commit, toolchain pins, device, SDK or driver version, and date, and require a simulator result to name the simulator as the device. The bible's Result Record gives each Trial recipe_hash and toolchain_pins, but no commit, device, SDK or driver version, or date, so trial.json, trial.md, and the Parquet rows cannot show them on their own. Where should they live?
Options: (a) run level only: run.md and the run's provenance manifest (P0.11) carry them for every trial in the run, and trial.md links to run.md; no record change, but a trial file read outside its run tree has no provenance. (b) add `provenance: {commit, dirty, device, sdk, date}` to Trial in the bible's Result Record (Decision Log entry), filled by the runner and shown in trial.md and the trials table; a small schema change now, before any real run writes records. (c) both: the run manifest stays authoritative and each Trial carries a copy.
Recommendation: (c); every artifact that shows a number then carries its provenance (Rule 1) while the run manifest stays the source (AGENTS.md, Results), and the change is cheapest before real runs exist.
Answer: (c) Both. (Given by the owner in the working session on 2026-09-23.)
Applied: 2026-09-23. Recorded in the bible (Decision Log; the Result Record lines land with P0.18, together with the record change); implementation is task P0.18, which P0.G now needs.

## OQ-009 Stale Storage Row In Environment State
State: CLOSED
Kind: review
Blocks: none
Evidence: plans/spikes/p0-nvcc.md (finding 4); docs/BIBLE.md Environment State, Storage row
Question: The bible's Environment State (dated 2026-09-22) says "Root filesystem full", but the P0.6 spike measured 300G available on `/` (82% used; rx doctor root_free_gb 321.3) on 2026-09-23. Should the row be updated with the measured value?
Options: (a) an agent updates the Storage row with the 2026-09-23 measurement and a Decision Log entry at the next bible sync; Agent Rule 7 stays as it is either way. (b) keep the row as the 2026-09-22 record and add a dated note beside it. (c) leave it.
Recommendation: (a); the Environment State should match the latest measurement, and Rule 7 keeps everything off the root filesystem regardless of free space.
Answer: Go with the recommendation.
Applied: 2026-09-23. Storage row updated with rx 20260923-120027-exec-208c (bible Environment State, Decision Log).

## OQ-010 CUDA Runfile Installer Log On The Root Filesystem
State: CLOSED
Kind: policy
Blocks: none (P0.7 proceeds; the next CUDA reinstall follows the answer)
Evidence: toolchains/cuda.sh; install job 20260923-043820-toolchains-p07-dc38; its log, moved to $LASSI_TOOLCHAINS/cuda@12.6.3/cuda-installer.log
Question: The CUDA runfile installer cannot write /var/log as a user, so it writes /tmp/cuda-installer.log (18,820 bytes in this job) on the alpha01 root filesystem while it runs. Job 20260923-043820-toolchains-p07-dc38 did this once; the script moved the log into the install prefix and no file remains in /tmp or /var/tmp. Agent Rule 7 allows nothing on the root filesystem, and only you can grant an exception. cuda.sh now clears the log with an EXIT trap on every exit path except SIGKILL, which the gate sends 10 s after SIGTERM when it kills a job.
Options: (a) grant a narrow exception for this transient installer log, cleared by the trap; (b) switch cuda.sh to NVIDIA's per-component redistributable archives (published sha256, no installer, nothing outside the scratch disk) and reinstall cuda@12.6.3 that way; (c) keep the current install and forbid further runfile installs until (b) exists.
Recommendation: (b); it removes the root-filesystem write entirely and gives a published checksum per component, at the cost of one reinstall (download size PROJECTED; the runfile is 4,446,722,669 bytes per rx 20260923-043558-exec-a651, and the installed tree measured 7.0G in plans/spikes/p0-nvcc.md).
Answer: Go with recommendation.
Applied: 2026-09-23. Recorded in the bible (Toolchain Pins, Decision Log); the reinstall is task P0.19. The owner removed the runfile and NVHPC tarball downloads the same day.

## OQ-011 Sandbox CPU Quota And Separate Uid
State: CLOSED
Kind: decision
Blocks: none (P0.10 builds on the measured mechanism)
Evidence: plans/spikes/p0-sandbox.md; docs/BIBLE.md Sandbox and Decision Log (2026-09-23)
Question: The bible's Sandbox asks for a separate uid or container and cgroup limits on CPU, memory, and wall time. On alpha01, unprivileged, the P0.9 spike found that the user systemd manager delegates only the memory and pids controllers, so CPU can be capped only as CPU time (prlimit --cpu), not as a cgroup quota; and the sandbox is a user-namespace container whose processes keep the host uid. Do these meet the bible's intent?
Options: (a) accept both: CPU time cap via RLIMIT_CPU and a user-namespace container count as meeting the Sandbox rules; (b) grant access: a root drop-in delegating the cpu controller to user@1025.service (Delegate=cpu cpuset io memory pids) so the scope enforces CPUQuota; (c) require a separate host uid, which needs a newuidmap-based helper that was not tested.
Recommendation: (a) for P0, with (b) later if CPU contention between parallel trials shows up; network isolation, memory, wall time, and the read-only harness already hold. Please also confirm that sandboxed runs may use systemd-run --user through the gate, which refuses the systemctl command pattern.
Answer: Go with A, I cannot be clear enough about this, you will not get root access.
Applied: 2026-09-23. Recorded in the bible (Sandbox, Decision Log). No option that needs root will be proposed again.

## OQ-012 Scratch Cleanup Candidates
State: CLOSED
Kind: decision
Blocks: none (usage is 108G of the 120G cap, rx 20260923-121133-exec-e28f)
Evidence: rx 20260923-120802-exec-c07d and rx 20260923-120813-exec-70cc (du of /mnt/nvme10/joseph_ufl on 2026-09-23); bible Decision Log (scratch cap)
Question: Agents may not delete files under /mnt/nvme10/joseph_ufl. These are the remaining candidates for deletion or for a backup to the workstation. The removed installer downloads are done. Which should go?
Options: (a) `.cache/pip` (7.8G), pip's download cache: `python3 -m pip cache purge`; the only cost is slower installs. (b) `cuda-12.6.3/` (7.0G, dated 2026-07-08), an older CUDA 12.6.3 tree; LASSI uses `toolchains/cuda@12.6.3` and does not need it. Remove it only if nothing outside LASSI uses it. (c) `amd/` (18G: extracted_rocm 15G, debs 3.9G), with AMD deferred (OQ-002): an agent backs it up to the workstation as a tar.gz, but only after you approve that and say whether the host copy then goes. (d) keep everything for now.
Recommendation: (a) now, (b) if unused elsewhere, and (c) when space is next needed. Keep `.cache/huggingface` (23G) and `furiosa-venv` (8G), which RNGD serving may need.
Answer:Give me a command to delete the pip cache.
Applied: 2026-09-23. The owner removed `.cache/pip` (pip cache purge, then the older entries it left); usage is 100G (rx 20260923-140158-exec-791b, 2026-09-23T14:01:58-07:00). Options (b) and (c) were not chosen, so `cuda-12.6.3/` and `amd/` stay.

## OQ-013 Pattern List Visible In Public Actions Logs
State: CLOSED
Kind: policy
Blocks: none
Evidence: CI run https://github.com/JoeMad21/lassi/actions/runs/35921343647 (the P0.13 canary, 2026-09-23): the log of the step "Check commits, files, branch, and pull request text" prints the step environment, including the full TEXT_POLICY_PATTERNS value; .github/workflows/text-policy.yml passes it as `vars.TEXT_POLICY_PATTERNS`, and Actions variables are not masked. The repository is public (OQ-005), so every run log shows the list.
Question: The text-policy pattern list is kept out of git by design, but the CI logs of this public repository publish it on every run. Should it move to an Actions secret, which Actions masks in logs?
Options: (a) move it: create an Actions secret TEXT_POLICY_PATTERNS with the same value, change the workflow's env line to `secrets.TEXT_POLICY_PATTERNS`, delete the variable, and optionally delete old run logs; an agent can make the one-line workflow change (it does not weaken the check) once you create the secret, and a canary run proves it still fails. (b) keep the variable; the list is not a credential and is also derivable from what the check refuses. (c) move it and also make the repository private sooner.
Recommendation: (a); it keeps the list out of public view at no cost to enforcement. Secrets need you to set them (repository settings are owner actions).
Answer: (a). (Given by the owner in the working session on 2026-09-23: "I agree with the fix." The owner set the Actions secret the same day.)
Applied: 2026-09-23. The owner set the Actions secret (last from the variable through a bash pipe, updated 21:28:26Z); the workflow reads it since ba90a9c. Canary run https://github.com/JoeMad21/lassi/actions/runs/35922699182 (commit 42ea4b6) failed on the canary in the commit message and in canary.txt, and its log shows TEXT_POLICY_PATTERNS as ***. Left for the owner: delete the variable after the P0 pull request merges (main's workflow reads it until then), and optionally delete the older run logs that show the list. Not verified: whether Actions masks each line of the multi-line secret; a real violation prints the matching pattern, so such a line could appear in a failure log.

## OQ-014 Core Files On The alpha01 Root Filesystem
State: CLOSED
Kind: access
Blocks: none
Evidence: plans/spikes/p0-sandbox-hardening.md (probes F2 and G1, inline rx exec probes on 2026-09-23); plans/p0-core.md P0.17 (nvcpfe TERMINATED by signal 11 in dirty-tree rx 20260923-104313-desktop-8r113ei-p0-core-2b22 and ...-173d)
Question: Agent probes on 2026-09-23 caused systemd-coredump to store four core files under /var/lib/systemd/coredump on the alpha01 root filesystem, against Agent Rule 7: /usr/bin/unshare (about 20K) and /usr/bin/dash (about 21K) from the sandbox probes, and two nvcpfe cores (about 500K each) from P0.15 compiler probes. They are root-owned, so this account cannot remove them, and agents may not delete files anyway. The P0.16 sandbox now keeps crashes away from that handler (prlimit --core=1 and a seccomp filter; no core since, per coredumpctl). Compiles still run at the host's core limit of 0, which systemd-coredump ignores; task P0.20 runs them under --core=1 too. How should the four files be handled?
Options: (a) leave them: systemd-coredump's default cleanup (systemd-tmpfiles, about 3 days) removes them. (b) ask an administrator to delete them now. (c) record only.
Recommendation: (a); they are small and expire on their own. For awareness, outside LASSI: the spike found /var/lib/amd-metrics-exporter/amdgpu_device_metrics_exporter_grpc.socket at mode 777, so any user on alpha01 can reach that GPU metrics exporter; the sandbox hides it, and the host setting is an administrator's matter.
Answer: Let's go with option A.
Applied: 2026-09-23. Nothing to do: the four core files are left for systemd-coredump's own cleanup, and no agent touches them. Since P0.16 and P0.20, programs and compiles run under prlimit --core=1.

## OQ-015 Git Hooks Not Executable On Linux
State: CLOSED
Kind: policy
Blocks: none (CI still enforces the text policy on every push)
Evidence: at 1de7db6, `git ls-tree -r HEAD` showed .githooks/commit-msg, .githooks/pre-commit, and .githooks/pre-push stored as mode 100644. The fast suite from the clean commit 1de7db6 on alpha01 (rx 20260923-195751-desktop-8r113ei-p0-core-7f98) failed only tests/tools/test_text_policy_modes.py::test_canary_local_blocks_both_seeded_commits; a rerun (rx 20260923-200253-desktop-8r113ei-p0-core-0d3e) showed the seeded canary commit going through. results/p0-compile-hardening/summary.md.
Question: The hooks were committed from Windows, where Git Bash runs a hook whatever its mode. Git on Linux and macOS ignores a hook without the executable bit, so on alpha01 or any Linux clone the local text-policy hooks never run. Agents leave changes to .githooks/ to the owner. Should the three hooks be marked executable?
Options: (a) mark them executable. In Git Bash at C:\dev\lassi on branch p0-core, run `git update-index --chmod=+x .githooks/commit-msg .githooks/pre-commit .githooks/pre-push`, then `git commit -m "P0: Mark the git hooks executable"` and `git push`. Only the file modes change, and lib.sh is sourced, so it needs no bit. (b) leave them; CI alone enforces on Linux clones, and the canary test keeps failing there.
Recommendation: (a); it restores local enforcement on every platform. After it, an agent reruns the fast suite on alpha01 to show the canary test passes.
Answer: (a), applied by the owner in the working session on 2026-09-23 as commit f9d1e68, which stores the three hooks as mode 100755.
Applied: 2026-09-23. On alpha01 the hooks now run: rx 20260923-201529-desktop-8r113ei-p0-core-e098 showed both canary commits refused, with a configuration error, since the owner's pattern list is not installed there (fail closed). Commit 60197f4 lets the canary test accept that refusal where no list is installed while still requiring real findings where one is. The full fast suite from the clean commit 60197f4 on alpha01 passed: 2442 passed, 13 skipped, pytest status 0 (rx 20260923-201618-desktop-8r113ei-p0-core-b46a).

## OQ-016 Commit f9d1e68 Holds The P0.20 Acceptance Record
State: CLOSED
Kind: review
Blocks: none
Evidence: `git show --stat f9d1e68` (pushed to origin/p0-core; CI text-policy run 35950138033 passed on it). Besides the three hook mode changes, it holds the whole staged P0.20 acceptance change: results/p0-compile-hardening/ (both provenance files and summary.md), the bible Sandbox, Toolchain Pins, and Decision Log edits (master revision 83), plans/STATUS.md (P0.20 DONE), plans/PHASE-NOTES.md, and OQ-015. An audit found every fact in it correct.
Question: The OQ-015 instructions an agent gave told the owner to commit while the agent's P0.20 change was still staged, so one commit carries both under the subject "P0: Mark the git hooks executable". AGENTS.md wants one commit per task with a message that describes the change, and forbids rewriting pushed history. How should this stand?
Options: (a) leave it as is: the content is correct and audited, plans/STATUS.md points the P0.20 note at f9d1e68, and this item records the mismatch. (b) rewrite the branch history (only the owner may; it changes pushed commits).
Recommendation: (a). Agents now empty their own index, or use `git commit -- <paths>`, before giving the owner any commit command.
Answer: Let's go with option A.
Applied: 2026-09-23. f9d1e68 stays as pushed; plans/STATUS.md's P0.20 note points to it, and agents empty their index before giving the owner a commit command.

## OQ-017 Remove The Old CUDA Runfile Tree
State: CLOSED
Kind: decision
Blocks: none (usage is about 101G of the 120G cap)
Evidence: results/p0-cuda-redist/summary.md; plans/spikes/p0-cuda-redist.md; bible Decision Log (P0.19 CUDA pin)
Question: P0.19 replaced the runfile CUDA install with the redistributable archives. The installer moved the old tree aside rather than deleting it: /mnt/nvme10/joseph_ufl/toolchains/cuda@12.6.3.runfile-20260923-214325 (about 7.0G, including the runfile's cuda-installer.log). Nothing uses it; the pinned prefix cuda@12.6.3 is the new install, and it passed every check from the clean commit 7d8d3d5. Agents may not delete files there. Remove it?
Options: (a) remove it: on alpha01, `rm -rf /mnt/nvme10/joseph_ufl/toolchains/cuda@12.6.3.runfile-20260923-214325`; this frees about 7G. (b) keep it as a fallback for now.
Recommendation: (a); the new install is verified, and the runfile tree only takes space under the cap.
Answer: Went with option A. Deleted the files.
Applied: 2026-09-23. Verified read-only: only toolchains/cuda@12.6.3 remains, and `du -sh /mnt/nvme10/joseph_ufl` is 94G (rx 20260923-222319-exec-07d8).

## OQ-018 Upstream LASSI Text In The Public Repository
State: CLOSED
Kind: policy
Blocks: P1.12 (the P1 gate and the demo do not wait on it)
Evidence: plans/p1-faithful.md (P1.3, P1.12). Upstream SPEAR-UIC/LASSI at 74b4681 is GPL-3.0 (bible Source Papers, LASSI). Its prompt_dictionary.py holds the lassi-2024 system and translation prompts and both context packs: about 28.6 KB of OpenMP reference card text and 18.2 KB of CUDA C++ Programming Guide chapter 5 text. The repository is public (OQ-005) and has no LICENSE file. An exploratory local check on 2026-09-24 ran prompt_dictionary.py, the notebook, and the 20 *_main sources through tools/check_text_policy.py --text-stdin: no pattern hits; the only failures were a non-ASCII copyright sign in the two layout sources, which are never committed. Each upstream system prompt opens with a sentence that casts the model in its role; the Attribution Policy covers this repository's own text, so publishing upstream wording word for word is worth a deliberate choice.
Question: The bible puts the lassi-2024 prompts under assets/prompts/ and the context packs under assets/context/. That text is upstream's, word for word, so committing it publishes GPL-3.0 text and third-party documentation excerpts in an unlicensed public repository. May agents commit it?
Options: (a) commit all generated text with a NOTICE naming the upstream commit and its GPL-3.0 license: the repository reads as the bible describes, but carries GPL-3.0 material and the two documentation excerpts. (b) commit only the manifests (upstream key and sha256) and generate the text from the pinned upstream checkout before each run: nothing third-party is published, readers see the prompts in each trial.md, and the bible's Repository Layout note is updated. (c) commit the prompts with a NOTICE and generate only the two context packs: the short prompts stay readable in the repository and the documentation excerpts stay out.
Recommendation: (c); it keeps the prompt templates readable (Readability Standards, Prompts row) and keeps copyrighted documentation text out of a public repository. Until you answer, P1.3 keeps all generated upstream text gitignored, so nothing is published first. The gate and the demo do not wait, because the text is byte-identical either way. Note from P1.3: its leak-guard test fails when any tracked file outside the upstream checkout holds a generated fragment of 40 or more characters. It exempts only the experimental_setup compiler and flag literals that docs/BIBLE.md already records (today the CUDA flag text); every other literal, upstream's absolute nvc++ path included, stays guarded.
Answer: I did not want the LASSI prompts in the repository. Remove them and keep the prompts local only.
Applied: 2026-09-24. Nothing had been committed to remove: no upstream prompt or documentation text was ever tracked; only the three MANIFEST.yaml files (keys, source cells, sha256) are, and the loader needs them to verify the local files. The prompts and context packs stay generated locally by tools/extract_lassi_assets.py into gitignored trees, and the tests/prompts leak guard keeps any upstream text out of tracked files. Recorded in the bible (Repository Layout, Decision Log); P1.12 is DONE.

## OQ-019 Agent Rule 8: furiosa-smi ps Misses Other Tenants
State: CLOSED
Kind: policy
Blocks: none (serving checks furiosa-smi status too in the meantime)
Evidence: plans/spikes/p3-rngd-demo.md, section 2 (exploratory, 2026-09-24)
Question: Agent Rule 8 requires `furiosa-smi ps` before claiming NPUs. On alpha01 it lists only the caller's own processes: it showed no rows while another tenant's server held npu4-npu7 (45.93 of 47.50 GiB each in `furiosa-smi status`). Should the rule also name `furiosa-smi status`?
Options: (a) amend Rule 8 to "run furiosa-smi ps and furiosa-smi status; claim only a card whose memory reads 0.00 GiB", changing the bible and AGENTS.md together. (b) keep the rule as written and record the extra check only in Host Facts and the Serving Rules; the rule alone would still allow claiming an occupied card. (c) also have the gate refuse a furiosa-llm command whose --devices names a card with nonzero memory; mechanical, but a gate change you reinstall.
Recommendation: (a) now, (c) later. Agents already check the Memory column before every claim; that is stricter than Rule 8, not looser.
Answer: Go ahead with option A with a note to review this question again later.
Applied: 2026-09-24. Agent Rule 8 amended in the bible and AGENTS.md (furiosa-smi ps and furiosa-smi status; claim only a card at 0.00 GiB), with a Decision Log entry. The review note is in plans/PHASE-NOTES.md (All Phases).

## OQ-020 Text-Policy Checker Refuses Staged Gitlinks
State: CLOSED
Kind: policy
Blocks: none (P1.1 pins upstream with a manifest and a fetch tool instead)
Evidence: plans/spikes/p1-hecbench-pin.md (Upstream pin); tools/check_text_policy.py, check_staged (it reads every staged path with `git show :<path>`)
Question: A gitlink (mode 160000) names a commit in another repository, so `git show :<path>` fails on it and the pre-commit check refuses the commit. No submodule can be committed while the checker reads gitlinks this way. P1.1 pinned upstream LASSI with assets/upstream/lassi.yaml and tools/fetch_upstream.py into a gitignored checkout, which works and is tested. Should the checker skip gitlinks?
Options: (a) leave the checker as it is and pin third-party code with manifests and fetch tools: no checker change; submodules stay impossible. (b) skip mode-160000 entries in check_staged (or check the gitlink's commit id as text): submodules become possible; a gitlink holds no text of this repository to scan. Either way the owner makes the change, since agents may not edit the checker.
Recommendation: (a) for P1, since the manifest pin works; decide (b) before any later phase that needs a submodule.
Answer: Go ahead with option A for now. Make a notice to review this question again later.
Applied: 2026-09-24. The checker stays as it is; third-party code is pinned with manifests and fetch tools. Recorded in the bible (Decision Log); the review note is in plans/PHASE-NOTES.md (All Phases).
