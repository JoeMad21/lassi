# Training across hosts: the plan (draft)

Status: draft for the owner, 2026-11-01. Written from a read-only planning pass over commit 40bd4a8; nothing ran.
The owner's answers so far (2026-11-01, working session): "This server may at some point provide AMD access, but we
cannot count on it. Just focus on the training logisitics [sic] for now, and we'll add training data soon." The owner
drafts the training rule (Agent Rule 9's change) during the window of 2026-11-01. The owner-queue items below are
filed as OQ-116 to OQ-123 (2026-11-01). File:line references are at commit 40bd4a8; later commits moved some lines.


I read this from C:/dev/lassi-check at 40bd4a8 (branch p16-adversarial, clean). I ran nothing. Cited figures carry their source; every unmeasured latency, throughput, or size figure is PLACEHOLDER or PROJECTED. Anything about the new server is open. "Inferred" marks my reasoning, as opposed to text I read.

## 0. Blockers, stated first (plans/LESSONS.md:24, "never blindside")

1. **The rules and the code still forbid silicon in RL rewards.**
   - The rule text says "RL rewards never execute on silicon" in AGENTS.md:35 and docs/BIBLE.md:53.
   - lassi train refuses a silicon reward.executor (lassi/train/run.py:270-283). The runner refuses silicon beside a stage that declares computes_reward (lassi/core/runner.py:11-27).
   - The owner said to remove that rule (plans/OWNER-QUEUE.md:1028). The removal is task P17.53, which is READY but not done (plans/STATUS.md:205). The owner's own edit of Rule 9 is still pending (plans/OWNER-QUEUE.md:1184).
2. **No model-generated program goes on the cards while the IOMMU is in pass-through.** This is an agent practice the owner accepted; no code enforces it (plans/OWNER-QUEUE.md:1028; plans/p17-portable.md:492).
3. **No reward that runs code exists yet.**
   - The train recipe's `reward` key is in NOT_CARRIED_OUT (lassi/train/run.py:99).
   - The only reward is the text-only fixture reward (lassi/train/trl_trainer.py:61-67; lassi/train/fixture_reward.py:1-10).
   - The trl trainer refuses any device but the CPU until P17.13 (trl_trainer.py:26, :142-147). vLLM rollouts are "P8's" (trl_trainer.py:349).
4. **No Tenstorrent item is in a train split.**
   - tt-pairs-v0 is a development set (plans/OWNER-QUEUE.md:1094).
   - bale-v0 and bale-v0-tt are unassigned (docs/BIBLE.md:1326).
   - An RL reward that runs code needs train items that have references and held-out inputs.
5. **There is no channel from a second server to alpha01.**
   - alpha01 is reached only through rx from the workstation (AGENTS.md:153).
   - Uploads go only as git pushes to refs/wip (tools/rx.py:194-207; tools/server/pre-receive:2-8).
   - Pulls are capped at 200 MB (tools/server/gate.py:63, :832-833).
   - Commands on the host may not run ssh, scp, rsync or sftp (tools/server/config.default.json:21).
   - "No further SSH keys can be added" (docs/BIBLE.md:65).
6. **Training waits for the owner's word.** The owner said on 2026-10-29: "no training until AMD access" (plans/LESSONS.md:31). Whether the new server is that access is open.

## 1. What runs where

| Work | Host | Basis |
| --- | --- | --- |
| Policy training (sft, dpo, grpo, gspo, ppo; full or lora) | Training server GPUs | Compute targets GPUs and the layer is device-neutral (docs/BIBLE.md:1012, :1015); P17.13's GPU half is unblocked by a GPU host (plans/STATUS.md:113-114) |
| Online rollouts (vLLM) | Training server GPUs | docs/BIBLE.md:983-986, :1012; not built yet (trl_trainer.py:349) |
| Offline sampling (rft, dpo) | The bible says RNGD (docs/BIBLE.md:981-982, :1013); I recommend the training server's GPUs (OQ-123) | Each RNGD round would need a checkpoint transfer and an RNGD export (section 4) |
| Compile rewards: tt-mlir ttkernel lowering, ttmetal-host clang++-20, g++-12, cgeist | Training server CPU, compile sandbox | CPU-only toolchains (docs/BIBLE.md:569-571); compiles run in the sandbox (docs/BIBLE.md:831) |
| ttsim run rewards | Training server CPU, sandbox | ttsim opens no device (lassi/executors/ttsim.py:10-11) and needs only glibc 2.14 (docs/BIBLE.md:746) |
| binary_io oracle and df-v0 score | Training server, pure Python | docs/BIBLE.md:816-826, :947-962 |
| Silicon final test, and silicon reward terms once Rule 9 is edited | alpha01 only | The cards and their lock are host-local: $LASSI_SCRATCH/locks/tt-silicon.lock (lassi/executors/tt_silicon.py:248-251, :659) |
| SFT on a Wormhole card (tt-xla) | alpha01 only, unchanged | docs/BIBLE.md:1018 |
| RNGD serving of a trained model; eval-split runs | alpha01 only | docs/BIBLE.md:845; Agent Rules 8 and 13 (AGENTS.md:34, :39) |
| Checkpoint selection during training | Training server, on train items only | Agent Rule 5 (AGENTS.md:31) |
| Data | Rebuilt on each host from pinned upstream and checked by hash | Corpus records pinned by sha256 (docs/BIBLE.md:1006); held-out inputs give "the same bytes on every host" (docs/BIBLE.md:786) |
| Checkpoints | Written on the training server; moved to alpha01 only for RNGD | Content digest in provenance (docs/BIBLE.md:1029); the channel is OQ-119 |
| Result summaries | Workstation results/ | AGENTS.md:165-170 |

**Can the compile and ttsim rewards run locally on the training server?** Yes, on three conditions. Otherwise they fall back to a remote reward service on alpha01's CPU (OQ-118 (c)).

1. **The host is x86_64 and Debian-family.** The pinned sfpi is sfpi_7.25.0_x86_64_debian.txz (toolchains/tt-metal.pin:27), and tt-metal.sh refuses a host whose sfpi lookup names another archive (docs/BIBLE.md:562).
2. **The host tools come from Ubuntu 22.04 (jammy).**
   - g++-12 must print EXPECT_VERSION "(Ubuntu 12.3.0-1ubuntu1~22.04.3) 12.3.0" (toolchains/gcc.pin:10, :16).
   - clang++-20 must be the exact apt.llvm.org build (toolchains/tt-metal.pin:77-80). Noble's clang-20 "is a different build" (docs/BIBLE.md:569).
   - tt-mlir and Polygeist need clang-17 and lld-17 17.0.6, ninja 1.10.1, Python 3.10.12 and CMake 3.31.6 (docs/BIBLE.md:563).
3. **The sandbox works there.** It needs a systemd user manager (systemd-run --user --scope) and unprivileged user namespaces (lassi/executors/sandbox.py:16-22). It needs util-linux 2.37.2, or the -n fix for 2.39 (docs/BIBLE.md:832). Root will not be granted (docs/BIBLE.md:830). Whether a cluster node provides a user manager is open (inferred risk: batch schedulers often do not).

## 2. The interface between the servers

### 2.1 Two reward paths

- **Synchronous and local, on the training server, every step.**
  - What: compile, then ttsim or native, then binary_io, then df-v0. This is the stage path lassi run already uses, through the bound sandboxed executor (docs/BIBLE.md:840).
  - It is the reward.executor of the train recipe. It stays limited to ttsim and native (lassi/train/run.py:270-283) even after Rule 9 is edited. So no host's synchronous path can name silicon.
  - Null rewards are skipped, never counted as 0 (docs/BIBLE.md:1030).
  - Results are cached by the hash of normalized code, inputs and pins (docs/BIBLE.md:1027).
- **Asynchronous and remote, to alpha01, in batches: the silicon final test.**
  - What enters: only candidates of train items that reached S5 and passed the oracle on ttsim. That matches "they have to pass the simulator and compiler first" (plans/OWNER-QUEUE.md:1028). It also meets OQ-084's recommendation (c), so no kernel JIT failure reaches a card (plans/OWNER-QUEUE.md:1105-1106).
  - Candidates go to alpha01 as text files, never binaries. alpha01 rebuilds them for the cards, as P17.55 specifies (plans/p17-portable.md:506).
  - Proposed recipe shape: a separate section such as `reward.final: {queue: <name>}` (name suggested). This keeps the existing refusal intact.

### 2.2 Messages and per-reward provenance (design)

Both message types are JSONL plus content-addressed files, carried by the channel of OQ-119.

**RewardRequest** carries:
- a request id, train id, step and completion index;
- suite, item, split (must be train) and direction;
- each model file as path, sha256 and bytes;
- the training server's ttsim reward record and its digest;
- the commit and dirty flag;
- a pins digest: the sha256 of every toolchains/*.pin text, plus each install's lassi-install.txt first line and its fingerprint (section 3.2);
- the reward profile's weights sha256 (assets/scoring/df-v0.yaml);
- the requested tt_silicon settings: hugepages, reset_filter, device_profiler (docs/BIBLE.md:738).

**RewardResult** carries:
- the request id and a status: done, refused-host-mismatch, refused-split, parity-mismatch, cards-busy, quarantined, or expired;
- the silicon run's exit status, hang and kill flags;
- binary_io statistics per output, the alignment mean, df-v0 components and R;
- device timing when the profiler is on (docs/BIBLE.md:738);
- the device record: each card's serial, firmware bundle and ARC firmware (tt_silicon.py:284), the driver version, and the IOMMU domain type, which P17.53 records instead of refusing (plans/p17-portable.md:492);
- the cards held, the lock wait, and wall time;
- alpha01's commit and pins, the host, the date, and the rx job id. A run's clean or dirty state comes from the rx record (plans/PHASE-NOTES.md:28).

**Per-reward provenance on the training server.** ttsim's device() names only "the host CPU" (lassi/executors/ttsim.py:599-612). Each reward record therefore adds:
- the host name, as lassi train already records (lassi/train/run.py:494), and the CPU model (lassi/executors/devices.py:139-146);
- the pins digest;
- the id of the parity run that covers this host, commit and pins (section 3.4).

All of this follows Agent Rules 1 and 10 (AGENTS.md:27, :36).

### 2.3 Channel

The channel is open (OQ-119). What exists today:
- workstation to alpha01 through rx;
- an owner-run transfer with the owner's own ssh, the precedent of OQ-087's backup scripts (plans/OWNER-QUEUE.md:1145).

Nothing reaches alpha01 from a third host.

### 2.4 Latency and batching

- **Nothing is measured.** The bible says: "Measure reward latency on ttsim first, then size group size and batch to the reward workers" (docs/BIBLE.md:1028).
- **Known bounds:**
  - A ttsim run limit is 10x the reference wall time, at least 30 s (docs/BIBLE.md:791).
  - A run uses 16 CPUs (docs/BIBLE.md:840). Tier A peaked at 114 MiB (docs/BIBLE.md:757).
  - Silicon runs are serial under one lock, each preceded by a poll of up to 300 s (tt_silicon.py:244-245).
  - An rx job lasts at most 48 h, with at most 3 jobs and 1 big job at a time (tools/server/config.default.json:4-5, :8).
  - alpha01 work happens only in owner windows (plans/LESSONS.md:21).
- **Consequence (inferred):** silicon latency is hours to a window, not seconds. So silicon cannot sit inside a GRPO step.
- **Batching (design):**
  - One request file per range of steps, deduplicated by the reward cache key. Identical candidates within a GRPO group are common (docs/BIBLE.md:1027).
  - A per-batch cap the owner sets.
  - The worker on alpha01 drains a batch as one rx job inside a window.

### 2.5 Failure and quarantine

- **Refusals before any program starts:** a commit or pins mismatch, an item that is not train (the worker re-reads it through the bench registry for purpose train, docs/BIBLE.md:1023), or cards busy after the poll. Each returns its status. These may be re-queued in a later batch, because no program ran.
- **Per-request parity:** before silicon, the worker builds and runs the candidate on alpha01's ttsim and compares the result with the request's ttsim record. A disagreement returns parity-mismatch, sends nothing to silicon, and becomes a parity finding.
- **A program that ran is never rerun** (docs/BIBLE.md:738).
- **Hang, kill, crash, or status outside 0 to 127:** the quarantine marker stays (docs/BIBLE.md:738; OQ-075 (3a)). The worker stops, and every pending request gets status quarantined. A human clears the marker.
- **On the training server:** a silicon result that is pending, quarantined or expired is null and skipped. Training continues on local rewards. A result that arrives after its step closed goes into a ledger and is never back-filled (Agent Rule 1).

### 2.6 Agent Rule 9 across hosts

There is one card host, so the rule's three parts hold as follows.

- **One placement at a time.** Every silicon open on alpha01 takes the one flock lock, LOCK_NB, refused while another process holds it (tt_silicon.py:659). The tt-xla trainer's CardGuard shares that lock and marker (lassi/train/silicon.py:20-45). The training server has no card: the tt_silicon executor refuses a host with no numbered node (docs/BIBLE.md:738). Its recipes name no silicon executor. So the only path from it to a card is the queue, and the queue's only consumer is the one worker on alpha01.
- **No looped device opens.** One program per device open, never retried. The worker pauses between programs, as in the pacing OQ-099 (a) proposed (plans/OWNER-QUEUE.md:1179). This needs the owner's rule wording: P17.53 proposes "never looped device opens within a program" (plans/p17-portable.md:492). Today's AGENTS.md:35 text forbids looped opens outright.
- **Quarantine after a hang.** The worker refuses while any marker exists (tt_silicon.py:341-356) and stops at the first one. A human clears it.

## 3. Toolchain parity

### 3.1 Identical by construction (checked by the existing scripts)

- **Source commits:**
  - tt-mlir 5f396fd6 with LLVM 4efe170d (toolchains/tt-mlir.pin);
  - tt-metal 5280a9cf-tracy with its submodules (toolchains/tt-metal.pin);
  - Polygeist 77c04bb2 with LLVM 26eb4285 (toolchains/polygeist.pin).
- **Downloads checked by sha256:**
  - ttsim's library (toolchains/ttsim.pin:10-11), which the executor re-checks before every run (ttsim.py:24-35);
  - the sfpi txz and CPM.cmake (docs/BIBLE.md:561);
  - the CPM package list, compared line by line against the tracked copy (docs/BIBLE.md:569).
- **Host compilers:** identical by their --version EXPECT_VERSION check in the sandbox (docs/BIBLE.md:573).
- **The scripts:** they already take LASSI_ALLOWED_ROOT and LASSI_STOP_GIB with no default, so they work on a new host (plans/PHASE-NOTES.md:14).
- **Parity gap:** toolchains/polygeist.sh is not in the tree at 40bd4a8. Only the pin is tracked, and its header says it is "shell-sourced by toolchains/polygeist.sh" (toolchains/polygeist.pin:2). P5.5's script sits in a workstation stash (plans/STATUS.md:122). It must be committed before "the same scripts" holds.

### 3.2 Equal only if shown

The built trees (libtt_metal, tt-mlir tools, Polygeist, the tt-mlir runtime of docs/BIBLE.md:566) come from the same inputs. They are not guaranteed byte-identical (inferred: embedded absolute paths, build ids). The plan:

- Record a per-install fingerprint on both hosts: a sha256 listing of every file, path-relative, in the manner of lassi.core.digest tree_sha256 (lassi/core/digest.py:60).
- Capture dpkg versions and md5sums of the host tools.
- Diff the listings and explain each difference.
- Optionally use alpha01's absolute toolchains path (/mnt/nvme10/joseph_ufl/toolchains, tools/server/gate.py:55) on the new host to remove path differences (OQ-118 (d)).
- Agent Rule 10 across hosts: one training run uses one commit and one pins digest on both hosts, and the worker refuses a mismatch.

### 3.3 Re-qualification on the training server

From a clean commit, with provenance naming the host:
- **Sandbox:** the remote tests tests/executors/test_sandbox_remote.py, test_sandbox_compile_remote.py and test_native_threads_remote.py.
- **Core dumps:** the 1-byte core limit actually stops that host's core handler (AGENTS.md:145).
- **util-linux:** P5.16's fix if the version is 2.39 (plans/STATUS.md:133).
- **ttsim:**
  - tests/executors/test_ttsim_remote.py;
  - tools/ttsim_smoke.py on the six P4.9 examples (docs/BIBLE.md:776);
  - the finding, JIT-abort and Watcher behaviour against tests/executors/fixtures/ttsim (docs/BIBLE.md:758-760);
  - Tier A references passing, i.e. P4.G reproduced (docs/BIBLE.md:1272).
- **Toolchain fixtures:** the ttmetal-host and gcc fixtures, the tt-mlir smoke, and the cgeist lowering probe.
- **Determinism checks:** generate_inputs' pinned sha256 (docs/BIBLE.md:786) and the fuzzer's known-answer digest (docs/BIBLE.md:937).
- **GPU:** P17.12's device measurements and P17.13's GPU runs (plans/p17-portable.md:385-400). The GPU driver must support torch 2.14.1+cu130 or +rocm7.2 (docs/BIBLE.md:575); otherwise it is a pin change.

### 3.4 The cross-host reward check

- **Fixed set:**
  - the tracked references of tt-pairs-v0 (dev) and bale-v0-tt;
  - adv-seeded-v0's 12 seeded faulty candidates and 6 controls (plans/STATUS.md:160);
  - seeded TT faults: undefined behavior, a simulator gap, a JIT error, a hang, in the P4.9 capture forms (docs/BIBLE.md:758-759);
  - replay recordings of earlier model replies through the replay backend (model.recording, plans/STATUS.md:166).
  The same replay recipes run on both hosts, so the model replies are identical.
- **Procedure:** each candidate runs twice per host, which separates within-host nondeterminism (for example OpenMP reductions; inferred) from cross-host differences.
- **Equal means:** the same attempt stage and end reason; the same set of diagnostics, normalized by (stage, code, workdir-relative file, line); the same W, per-output pcc, max_abs and max_ulp, output-file sha256s, guards, hang flag, simulator finding classes, and df-v0 R.
- **Not compared:** wall times (simulator wall time is for sizing only, docs/BIBLE.md:755) and absolute paths.
- **When it runs:** at each pin or host change, and as a canary subset at the start of every training run, compared against stored alpha01 digests so no alpha01 access is needed each time. Section 2.5's per-request check then adds continuous parity on every silicon request.
- **A failure** stops training-server rewards from being used until it is resolved.

## 4. Models: weights from the training server served on RNGD

**Known:**
- furiosa-llm 2026.3 compiles the Qwen 2.5, Qwen 3 and Qwen 3 MoE architectures, among others (docs/BIBLE.md:847).
- The training bases are Qwen3ForCausalLM, chosen so that full or scratch models can later compile for RNGD (docs/BIBLE.md:1017).
- furiosa-llm's documentation names no adapter input and takes only text or token ids. So LoRA must be merged into the base, and prompt tuning is served by hf_local only (plans/spikes/own-model.md:80-85; docs/BIBLE.md:1000).
- ArtifactBuilder accepts a local path (plans/spikes/own-model.md:82).
- The bible's export path: merge, then `fxb check` against the base's bundle, then `fxb build` only if the fingerprints differ. Fingerprint reuse is "experimental" (docs/BIBLE.md:876).
- The export recipe keys exist (lassi/core/recipe.py:991) but are not carried out (lassi/train/run.py:99). `registry:<id>` is not built (docs/BIBLE.md:875).
- Checkpoints are safetensors with a content digest. hf_local serves them on the CPU as a cross-check (docs/BIBLE.md:870, :1029).
- Size: the Qwen3-1.7B base's two safetensors files are 4,063,515,592 bytes (plans/spikes/own-model.md:75), about 20 times rx's 200 MB pull cap.

**Open:**
- the FP8 path for fine-tuned BF16 weights, and fingerprint reuse for LoRA-merged checkpoints (docs/BIBLE.md:878, :1321);
- which furiosa-llm SDK is current: the bible says 2026.3.x (docs/BIBLE.md:861) while the models are at v2026.4 (plans/OWNER-QUEUE.md:1146);
- whether the revision-tag pinning of docs/BIBLE.md:1254 matters for a local directory;
- whether the scratch qwen3-77m compiles;
- numeric differences between GPU training and RNGD serving. Results must name the device (Agent Rule 1).

**Recommendation:** a full fine-tune or merged LoRA of a Qwen3 base. Run the export as its own spike (P19.11) before anyone relies on RNGD evaluation, as OQ-076's recommendation proposed (plans/OWNER-QUEUE.md:1001).

## 5. Rules and safety

- **Rule 6 (sandbox):**
  - On every host, model files are compiled and run only through lassi.executors.sandbox. The runner refuses an unsandboxed executor for model-code stages (docs/BIBLE.md:840).
  - The sandbox's env -i strips credentials (lassi/executors/sandbox.py:28-31).
  - Candidates crossing hosts are data. They are unpacked only into a trial workdir under the runs root.
- **Rule 5 (eval splits):**
  - The training server fetches and rewards train items only. Registry and corpus checks refuse held items (docs/BIBLE.md:1023-1024). The alpha01 worker re-checks each request.
  - Eval-split runs happen only on alpha01.
  - Counterexamples go only into synthetic fixtures (AGENTS.md:96).
  - Splits need the owner's assignment before any training run reads a suite (plans/PHASE-NOTES.md:19).
- **Network (OQ-107 counterpart, OQ-121):**
  - OQ-107 (a) is about the tt-xla wheel's Tracy client on alpha01 (plans/OWNER-QUEUE.md:1289-1297). The training server never loads that wheel, because it has no card.
  - The pinned tt-metal is built with no broadcast and listens on loopback only. Its runs happen in the sandbox's empty network namespace (docs/BIBLE.md:562).
  - Multi-node collectives need a network, so "no network" cannot apply literally to training processes (inferred).
- **Credentials (Rule 12):**
  - Any Hub token (for OQ-119 (d)) lives in an environment variable named by a recipe key, as with model.api_key_env (docs/BIBLE.md:868). It is never in the repo or in provenance.
  - The gate copies its own environment into the commands it runs (tools/server/gate.py:417). So a token is set only for the step that needs it.
- **Approvals:** Rule 17 names alpha01 only (AGENTS.md:43). Proposed counterpart for OQ-116:
  > "No access to the training server without the owner's approval in the working session: no command of any kind through its gate (run, job, exec, doctor, pull, sync), read-only probes, installs, and uploads included, and no training or model inference there, except under a standing permission the owner records in plans/LESSONS.md with its bounds (devices, duration, windows). Ask once per batch, with what runs, for how long, on which devices, and what it touches."

  Silicon-worker jobs remain alpha01 device work, which needs the owner's yes and an owner window (plans/LESSONS.md:21, :28).
- **Disk and scratch (Rule 7 counterpart, OQ-117):**
  > "Nothing on the training server's root filesystem or in its /tmp, /var/tmp, /opt, or a HOME outside the scratch root; builds, venvs, caches, checkpoints, and run trees live under the scratch root, with the owner's cap and stop line (LASSI_ALLOWED_ROOT, LASSI_STOP_GIB); the sandbox's runs root on node-local disk; free space checked before large runs; agents never delete files there."

  Other points:
  - The sandbox refuses a workdir under /tmp (AGENTS.md:176).
  - The 1-byte core limit applies (AGENTS.md:145).
  - Whether the overlay or userxattr works on a shared filesystem is open (inferred risk).
  - alpha01's scratch already sits on a disk at 98% use (plans/OWNER-QUEUE.md:1352), so checkpoints moved there need space approved first.
- **Rules 1 and 2:** ttsim rewards name ttsim. Silicon terms name the card. A ttsim reward never stands in for a silicon one.

## 6. Tasks, owner decisions, risks

### 6.1 Tasks: a new phase, P19 Training Host (branch p19-train-host, suggested)

A new Build Roadmap row needs the owner (AGENTS.md:9). P19.2 and P19.6 also carry P17.12 and P17.13 (plans/STATUS.md:113-114). P19.7, P19.9 and P19.10 overlap P8, which they would feed.

| ID | Title | Host | Depends | Acceptance |
| --- | --- | --- | --- | --- |
| P19.1 | Host policy text for the training server | Windows | OQ-116, OQ-117, OQ-121 answered | An AGENTS.md "Training Host Execution" section with the counterparts to Rules 7 and 17 and the network rule; draft Host Facts and Environment State rows, PLACEHOLDER until the spike; Decision Log entry; text-policy and status checks pass |
| P19.2 | Spike: training server inventory (plus P17.12) | Training server, read-only, approved | P19.1 | plans/spikes/train-host.md records: OS, CPU architecture, CPU count, RAM, GPUs and driver, scheduler, systemd user manager and delegation, user namespaces, util-linux, filesystems, egress, and install route; each gap filed as an owner-queue item |
| P19.3 | rx and gate for a second host | Windows; owner runs rx bootstrap | P19.1 | rx selects a host profile (it reads one profile from RX_* or .rx.json today, tools/rx.py:22-25, :55-70); a gate config with GPU classes disabled by default (tools/server/config.default.json:27-43); local-transport tests pass (tests/tools/test_rx_gate.py); rx doctor answers |
| P19.4 | Pinned host-tool install batch | Training server, owner-run | P19.2, P19.3, OQ-118 | A script with exact versions and sha256 values (the AGENTS.md:143 pattern); every EXPECT_VERSION reads back; dpkg versions equal alpha01's |
| P19.5 | Toolchains at the pins, plus install fingerprints | Training server long jobs; alpha01 fingerprint read under 10 min, CPU only (plans/LESSONS.md:28) | P19.4 | polygeist.sh committed first; tt-metal, ttsim, tt-mlir, tt-mlir runtime and Polygeist installed with install records; CPM list equal to the tracked copy; fingerprint diff explained in results/ |
| P19.6 | Qualify sandbox, ttsim and GPU (plus P17.13) | Training server | P19.5; P5.16 if util-linux is 2.39 | Every item of section 3.3 passes from a clean commit; hf_local and the TRL smoke run on the GPU |
| P19.7 | The code-running reward behind lassi train | Windows (fakes), then the training server | P19.6, OQ-122; owner reviews of df-v0 weights (docs/BIBLE.md:974), the guard (plans/PHASE-NOTES.md:167), and printed gaps (plans/PHASE-NOTES.md:166) | `reward` carried out (run.py:99); single-attempt trials through the stages; cache; nulls skipped; per-reward provenance as in section 2.2; eval and unassigned items refused |
| P19.8 | Cross-host reward parity tool and first run | Both hosts | P19.6, P19.7 | Section 3.4 passes on the fixed set; a seeded pin mismatch is caught by a test; results/reward-parity-<date>/ holds both provenance files |
| P19.9 | Silicon reward queue and alpha01 worker | Windows (tt_silicon_fakes.py); alpha01 in a window | P17.53, P17.54, P17.55, the owner's Rule 9 edit, OQ-119, OQ-120, P19.7; DMA IOMMU before any agent-launched model code | Schema, writer, channel adapter and worker as in sections 2.2 to 2.6; refusal, quarantine and expiry tests; one batch round trip with provenance |
| P19.10 | Silicon outcomes in training | Windows, then the training server | P19.9 | Per OQ-120: a labelled late component in steps.jsonl's reward record, never back-filled; rft keep-sets and dpo pairs built from silicon labels |
| P19.11 | Checkpoint export to RNGD (spike) | Training server, then alpha01 (RNGD window) | P19.6, OQ-119, OQ-123 | Merge and digest; transfer; fxb or ArtifactBuilder; serve on port 8123 with /v1/models checked (Rule 8); hf_local CPU cross-check; Model Registry entry; open items resolved or carried |
| P19.G | Gate | All hosts | P19.8, P19.9 (or recorded BLOCKED), P19.11 | Parity passes at the gate commit; a short grpo run on the training server's GPU with local compile and ttsim rewards on train items, each reward naming host, pins, device and commit; one silicon round trip, or BLOCKED naming OQ-078 (a) or the Rule 9 edit; one evaluation run on alpha01 with the checkpoint on RNGD |

### 6.2 Owner-queue drafts

Numbers should be checked against every open branch (checked when filed, 2026-11-01) (plans/PHASE-NOTES.md:42). Each item is State OPEN and uses the AGENTS.md:104-114 format.

**OQ-116 The training server: phase, access, and approvals** (access/policy; blocks P19.*)
- (a) A new phase P19, a Rule 17 counterpart (text in section 5), and a second gate reached through rx.
- (b) Fold the work into P17 and P8, with the same rule.
- (c) Unattended access within caps.
- Recommendation: (a), plus a standing permission like plans/LESSONS.md:28 once P19.6 passes. Please also say whether this server is the "AMD access" of plans/LESSONS.md:31.

**OQ-117 Scratch root and caps on the training server** (blocks P19.1, P19.5)
- (a) alpha01's 150G cap and 145G stop line.
- (b) An owner-named cap sized for checkpoints, with the runs root on node-local disk.
- (c) A free-space floor only.
- Recommendation: (b). Toolchains alone are PROJECTED at 28 to 46 GiB plus 20 to 29 GiB for the runtime (docs/BIBLE.md:563, :566).

**OQ-118 How the training server gets the pinned toolchains**
- (a) An Ubuntu 22.04 host or partition, owner-run install batches, the same scripts, and the parity check.
- (b) An Ubuntu 22.04 container. The sandbox inside it is unqualified, and systemd-run --user is likely missing there (inferred).
- (c) No TT toolchains there: compile and ttsim rewards on alpha01's CPU as a remote service, each batch an alpha01 job under Rule 17.
- (d) Option (a) with alpha01's absolute toolchains path.
- Recommendation: (a), plus (d) if feasible; (c) only if the host is not x86_64 or cannot get the jammy packages.

**OQ-119 The channel between the servers**
- (a) A workstation relay: a size-capped inbox and outbox verb pair with sha256 manifests on both gates, installed by the owner with rx bootstrap (tools/rx.py:17).
- (b) Shared storage the cluster admins provide.
- (c) A direct link to alpha01, which docs/BIBLE.md:65 rules out for SSH keys.
- (d) Checkpoints only, through a private Hugging Face repository at pinned revisions, with a read token from an environment variable.
- Recommendation: (a) for requests and (d) for checkpoints, or (b) if offered.

**OQ-120 Silicon in RL rewards across hosts** (prerequisite: the owner's Rule 9 edit, wording in plans/p17-portable.md:492)
- (a) An asynchronous final test as in section 2.1, with results used as a late labelled component and in offline rounds.
- (b) A synchronous step that waits for silicon.
- (c) Silicon only at checkpoint evaluation.
- Recommendation: (a). Option (b) cannot run at hours-per-window latency (inferred).

**OQ-121 Network containment on the training server** (OQ-107 counterpart)
- (a) Sandboxed reward runs with no network, as now; trainers with no egress outside the cluster; the hub offline; downloads only in named prepare or install steps (docs/BIBLE.md:1016); collectives only on the cluster's internal fabric; vLLM colocated or on loopback.
- (b) OQ-107 (a) as is: an empty network namespace per training process, workable for single-node runs only (inferred).
- (c) None.
- Recommendation: (a), using (b) for single-node runs.

**OQ-122 Train items for code-running rewards**
- (a) Assign train splits inside the existing suites. An eval label can never be undone (plans/PHASE-NOTES.md:19).
- (b) A new train-only suite built from non-test tt-metal examples, with references and held-out inputs, audited with the tt-train-v0 rules (docs/BIBLE.md:1007).
- (c) Synthetic seeded train items for smoke runs only.
- Recommendation: (b), with (c) for smoke runs.

**OQ-123 Where offline sampling and evaluation run**
- (a) As the bible says: rft and dpo sample on RNGD (docs/BIBLE.md:981-982, :1013), which means a checkpoint transfer and RNGD export every round.
- (b) Sampling on the training server's GPUs with vLLM. RNGD is used only for final evaluation of exported checkpoints. Checkpoint selection uses train-only validation items.
- Recommendation: (b), with a Decision Log entry.

### 6.3 Risks

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| The host is not x86_64 or jammy-compatible | No local TT rewards (toolchains/tt-metal.pin:27; docs/BIBLE.md:569) | OQ-118 (c), at a latency cost |
| No systemd user manager or user namespaces on cluster nodes | Model code cannot run there (Rule 6) | Spike first (P19.2); fall back to OQ-118 (c) |
| Built trees differ between hosts | Rewards cannot be compared | Fingerprints, the same path (OQ-118 (d)), the parity gate |
| A policy learns to print a fake simulator gap | Its reward becomes null and is skipped, so it escapes penalty (docs/BIBLE.md:971, :1306) | Decide on OQ-036 (b) or an alternative before RL (plans/PHASE-NOTES.md:166) |
| CPU to TT guard false positives and misses | Wrong R = -1 or rewarded host compute (docs/BIBLE.md:792-794) | Owner review before use as a reward (plans/PHASE-NOTES.md:167) |
| Gaps between simulator and silicon; quarantines | The silicon term stalls until a human clears the marker | Asynchronous term; nulls skipped; ttsim passes only |
| IOMMU in pass-through | No agent-launched model code on the cards (plans/OWNER-QUEUE.md:1028) | OQ-078 (a) host change; P17.54 |
| OQ-084 and OQ-085 still open | JIT quarantine; alpha01's stability under training load (plans/OWNER-QUEUE.md:1096-1121) | Queue only ttsim passes; move training off alpha01 |
| RNGD export is unproven | No RNGD evaluation of trained weights (docs/BIBLE.md:878) | P19.11 spike; hf_local fallback, labelled |
| The workstation relay is a single point of failure, and approvals are per batch | Slow loop | OQ-119 (b); standing permissions |
| Multi-turn GRPO with vLLM and LoRA diverges | Unstable training (docs/BIBLE.md:988) | single_turn by default |
| alpha01 disk at 98% use | Checkpoints and RNGD bundles may not fit (plans/OWNER-QUEUE.md:1352) | Ask for space before P19.11 |

### Critical Files for Implementation
- lassi/train/run.py
- lassi/executors/tt_silicon.py
- lassi/train/trl_trainer.py
- tools/server/gate.py (with tools/rx.py)
- docs/BIBLE.md (Agent Rules, Training Module, Toolchain Pins, Host Facts)