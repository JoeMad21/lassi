# Owner Queue

The only channel from agents to the owner. Format and rules: AGENTS.md, Owner Queue.
The owner answers by filling Answer and setting State to ANSWERED; the next session applies it.

## OQ-001 RNGD Host
State: OPEN
Kind: access
Blocks: P3, P6, P7, P14
Evidence: docs/BIBLE.md, Environment State (2026-09-22 check)
Question: Which I/ONX host holds the Furiosa RNGD cards now, and can they return to alpha01 or be reached from it?
Options: (a) cards return to alpha01; (b) agents get gate access on the host that holds them; (c) stay blocked.
Recommendation: (a); it keeps one gate and one scratch root.
Answer:

## OQ-002 MI300X Render Group
State: OPEN
Kind: access
Blocks: P7 (training GPUs), P8, P9
Evidence: docs/BIBLE.md, Environment State; /dev/kfd permission denied for joseph_ufl
Question: Ask an administrator to run usermod -aG render joseph_ufl on alpha01. When granted, set rocm_gpu enabled in the gate config and record it here.
Options: (a) grant; (b) AMD Developer Cloud credit for single-GPU work.
Recommendation: (a).
Answer:

## OQ-003 NVIDIA A100 Host
State: OPEN
Kind: access
Blocks: P10
Evidence: docs/BIBLE.md, Risks And Questions, question 2
Question: Which A100 host runs the full LASSI reproduction?
Options: (a) a UF or partner cluster; (b) cloud rental; (c) compile-only tier plus the -mp=multicore proxy only.
Recommendation: (a) when available; (c) meanwhile.
Answer:

## OQ-004 Cerebras SDK Access
State: OPEN
Kind: access
Blocks: P13
Evidence: docs/BIBLE.md, Device Targets and question 12
Question: Request the Cerebras SDK directly, or through the Sandia collaboration?
Options: (a) direct request; (b) through the collaboration contact.
Recommendation: (b) if the collaboration starts this term, else (a).
Answer:

## OQ-005 Monorepo Path And Remote
State: OPEN
Kind: decision
Blocks: P0.13, P0.G, every branch push (no origin remote is configured)
Evidence: docs/BIBLE.md, Agent Rules workflow and question 3
Question: Local repository path and remote name for the monorepo.
Options: (a) C:\dev\lassi on Windows, ~/dev/lassi on Linux, origin = private github.com/JoeMad21/lassi; (b) other.
Recommendation: (a). Record the answer in the bible and close question 3.
Answer:

## OQ-006 Kit Decisions Review
State: OPEN
Kind: review
Blocks: none
Evidence: docs/BIBLE.md, Decision Log, top three entries dated 2026-09-22
Question: Ratify the three decisions added with the agent kit: vendor-free wording of the attribution rule plus the checker canary; remote access through tools/rx.py and the project gate; unattended work order with stacked phase branches and this queue.
Options: (a) ratify; (b) amend.
Recommendation: (a).
Answer:

## OQ-007 GitHub Enforcement Settings
State: OPEN
Kind: access
Blocks: P0.13, P0.G (CI half of the text-policy gate)
Evidence: docs/BIBLE.md, Attribution Policy, Enforcement items 2 and 3
Question: Set the repository Actions variable TEXT_POLICY_PATTERNS to the pattern list, and protect main with the text-policy check required. Agents cannot change repository settings.
Options: (a) set both; (b) variable only (branch protection on private repositories needs a paid or education plan).
Recommendation: (a).
Answer:

## OQ-008 Provenance In The Result Record
State: OPEN
Kind: decision
Blocks: none yet; needed before P0.11 writes records from real runs
Evidence: docs/BIBLE.md (Agent Rules 1 and 2, Result Record); lassi/core/record.py; lassi/core/trial_md.py
Question: Agent Rules 1 and 2 require every number to carry commit, toolchain pins, device, SDK or driver version, and date, and require a simulator result to name the simulator as the device. The bible's Result Record gives each Trial recipe_hash and toolchain_pins, but no commit, device, SDK or driver version, or date, so trial.json, trial.md, and the Parquet rows cannot show them on their own. Where should they live?
Options: (a) run level only: run.md and the run's provenance manifest (P0.11) carry them for every trial in the run, and trial.md links to run.md; no record change, but a trial file read outside its run tree has no provenance. (b) add `provenance: {commit, dirty, device, sdk, date}` to Trial in the bible's Result Record (Decision Log entry), filled by the runner and shown in trial.md and the trials table; a small schema change now, before any real run writes records. (c) both: the run manifest stays authoritative and each Trial carries a copy.
Recommendation: (c); every artifact that shows a number then carries its provenance (Rule 1) while the run manifest stays the source (AGENTS.md, Results), and the change is cheapest before real runs exist.
Answer:
