# Bible edits awaiting the master copy

AGENTS.md, Authority: when the master copy is unreachable, each mirror edit is appended here, exact, for the owner. Apply them to the master in order, then clear an entry once it is mirrored.

## 2026-10-30: mirror rows awaiting the master copy (deferred during the window of 2026-10-29)

The master copy was reachable; these mirror edits were deferred to one batch after the window. Apply each to the master as it stands in docs/BIBLE.md, then clear this entry.

- Decision Log rows, newest first: 2026-10-30, route A's compile side (P17.57); 2026-10-29, gather and scatter for bale-v0 above ttkh (OQ-112); 2026-10-29, compile-only TTIR and D2M arms (P17.48 and P17.49); and the count sentence as the mirror gives it.
- IR Strategy, Level Rules: the bullet "Gather and scatter [DESIGN]" (OQ-112).

## 2026-11-01: more mirror edits awaiting the master copy

- Decision Log rows dated 2026-10-30: the driver and the Executor ttmlir-ttsim (P17.48); the TTIR and D2M arms on ttsim with outputs checked (P17.49); and the count sentence as the mirror gives it.
- Toolchain Pins: the tt-mlir runtime bullet's last sentence (built, with its rx id), and the new tt-mlir driver bullet.
- Execution Backends: the ttmlir-ttsim row.
- Oracles: the binary_io row's shape key and from_item.
- Benchmark Suites, Item tolerance paragraph: its last sentence (from_item).
- Sandbox: the HOME rule after the ttsim program runs bullet.

## 2026-11-01: silicon as an executor like ttsim (P17.53 to P17.56; OQ-078 and OQ-099)

Mirror edits of 2026-11-01, not yet in the master. Apply each to the master as it stands in docs/BIBLE.md, then clear this entry. Agent Rule 9 is not among them: its text is the owner's to write (plans/rule9-proposed.md), and every sentence below that names it says that its current text still reads as it does and that the owner's revision is pending (OQ-099).

- Decision Log rows dated 2026-11-01, newest first: the silicon launch kit (P17.56); silicon as the correction loop's final test (P17.55); silicon in RL rewards after the compiler and ttsim (P17.53 and P17.55); the IOMMU group view (P17.54); no gate on silicon use in LASSI's software (P17.53); and the count sentence as the mirror gives it (two hundred and four; five on 2026-11-01).
- Component Interfaces, the Executor bullet: the tt_silicon device() line with each card's IOMMU group; the `silicon` capability's sentences (no refusal and no start since P17.53, Agent Rule 9's text pending its revision, and the final_executor section's check and build).
- Result Record, the yaml block: Attempt.final_test and the FinalTest block (P17.55).
- Result Record, the Simulator gaps fed back paragraph: its sentence on the later hardware check.
- Result Record: the new Final tests paragraph after Simulator wall times.
- Result Record, Storage: the final_executor key of provenance.json and run.md's Final executor and Final device rows.
- Project Recipes, Notes: the new Final executor bullet after the Loop opt-ins bullet; the train recipes bullet (no `--silicon`; a tenstorrent trainer.device refused only without trainer.network none).
- Device Targets, the Tenstorrent row's status cell.
- Execution Backends: the opening paragraph; the tt_silicon row (its Key settings cell: no start and no refusal, the training sentence, the exposure, the IOMMU group view, the construction refusals, the lock's wait, the poll and hugepage directory for pass-through cards only, the identity-domain limit and OQ-078's answer, the PROJECTED memory note; its Status cell: the tests and what a launch waits on).
- ttsim Facts: the simulator gaps bullet's sentence on the later hardware check.
- Sandbox: the tt_silicon program runs bullet (the IOMMU group view, the covers and links, the four setup scripts, the remote tests not run since P17.54); the Attempt run limits bullet (the final test's 600 s floor).
- Reward Function, Scoring decisions: the new Final tests bullet after Simulator readings.
- Training Module, Compute: the Wormhole bullet (sft as the ttxla Trainer's one method, no start flag, the lock waited for, run.sh without `--silicon`).
- Training Module, Safeguards: the rewards bullet.
- LASSI Score Profile, the correct row: a final test that did not pass reads 0.
- Evaluation Protocol, Reporting Rules: the simulator gaps bullet (the hardware check list, final_tests.tsv).
- Environment State, the Tenstorrent Wormhole row's last cell (the final test, the pending revision, the identity domains of 2026-10-29).
- Risks And Questions: the simulator-to-silicon gap row.

## 2026-11-16: the tt_silicon visible_card setting (P17.66)

Mirror edits of 2026-11-16, deferred during the owner's window of that night. The master's tt_silicon row still predates the entries above (it lacks P17.42's device_profiler sentence), so apply this after them, as each edit stands in docs/BIBLE.md, then clear this entry.

- Decision Log row dated 2026-11-16 (P17.66), and the count sentence as the mirror gives it (two hundred and six; one on 2026-11-16).
- Execution Backends, the tt_silicon row: the visible_card clause after the settings sentence ("and visible_card (task P17.66), null unless it names one card's number"), and the sentence on TT_VISIBLE_DEVICES (unset unless visible_card names a card; the abort of rx 20261116-192634-tracy-07f1; unverified on the cards until the next run).
- Execution Backends, the tt_silicon row's Status cell: the config keys now end "harness, and visible_card (task P17.66)".
- Sandbox, the tt_silicon program runs bullet: the environment sentence (the profiler's names with device_profiler kernel, TT_VISIBLE_DEVICES with visible_card, the allowlist's TT_PROFILER_NAMES and TT_VISIBILITY_NAMES, each only with a Tenstorrent exposure and never in a compile), replacing "so the allowlist gains no name".

## 2026-11-17: the dev split (P17.31)

Mirror edits in docs/BIBLE.md, to apply to the master copy after the pending batches above:

- Old: A manifest item's split is train, eval, or unassigned [DESIGN] (task P4.13; OQ-025, option (d)): unassigned marks an item the owner has not yet assigned and decides nothing. The bench registry (lassi.bench.registry, Suite.item) refuses an eval or unassigned item to training, prompt tuning, and corpus harvest, the purposes Agent Rule 5 forbids, with EvalSplitError naming its split, and gives it for evaluation; a train item serves every purpose. A recipe selects the items of its split by the split's exact name, and the runner refuses a recipe that asks for an item outside its split or for a split with no items, so until the owner assigns Tier A's splits (P7) a run reads them with `split: unassigned`.
  New: A manifest item's split is train, eval, unassigned, or dev [DESIGN] (task P4.13; OQ-025, option (d); dev, task P17.31): unassigned marks an item the owner has not yet assigned and decides nothing, and dev marks a development item. The bench registry (lassi.bench.registry, Suite.item, REFUSED_PURPOSES) refuses an eval or unassigned item to training, prompt tuning, and corpus harvest, the purposes Agent Rule 5 forbids, and a dev item to training and corpus harvest, with EvalSplitError naming its split; it gives each for evaluation, and a dev item for prompt tuning as well; a train item serves every purpose. The held splits (HELD_SPLITS: eval, unassigned, and dev) are those kept from training and corpus harvest, so a train recipe, a corpus, and tt-train-v0's build refuse a dev item as they refuse an eval one. A recipe selects the items of its split by the split's exact name, and the runner refuses a recipe that asks for an item outside its split or for a split with no items.
- Old: the manifest's splits and the registry (lassi.bench.registry SPLITS) are unchanged for now, its items still unassigned, and a later task, P17.31, adds the split value.
  New: its five items and those of its four siblings (tt-pairs-v0-kh, tt-pairs-v0-ttkernel, tt-pairs-v0-ttir, tt-pairs-v0-d2m) are in the dev split, and every recipe that reads them names `split: dev` (task P17.31), which changed those recipes' hashes; the held-out C++ -> TT set evaluation needs is to be tt-par-v0's held-out items (task P17.68; OQ-129).
- Old: | Split by item; every item unassigned until P7 (OQ-025) |
  New: | Split by item; every item dev, a development set (OQ-083 (b); task P17.31) |
- Old: | As tt-pairs-v0: every item unassigned (OQ-025), a development set (OQ-083 (b)) |
  New: | As tt-pairs-v0: every item dev, a development set (OQ-083 (b); task P17.31) |
- Old: | As tt-pairs-v0: every item unassigned (OQ-025), a development set (OQ-083 (b)) |
  New: | As tt-pairs-v0: every item dev, a development set (OQ-083 (b); task P17.31) |
- Old: (1) every eval and unassigned item of tt-pairs-v0 has its example directory
  New: (1) every held item (eval, unassigned, or dev) of tt-pairs-v0 has its example directory
- Old: Tier A (tt-pairs-v0): every item stays unassigned, refused to training as eval items are, until training exists; the owner assigns its splits before any training run reads the suite, in P7 (OQ-025, owner: option (d)).
  New: Tier A (tt-pairs-v0): every item dev, a development set (OQ-083 (b); task P17.31), refused to training and corpus harvest, so its items can never be train or eval items (before it, unassigned under OQ-025, owner: option (d)).
- Decision Log, first data row: | 2026-11-17 | The dev split (task P17.31; Benchmark Suites, the split rules; OQ-083 (b)): lassi.bench.registry SPLITS gains dev, a development item that the registry gives for evaluation and prompt tuning and refuses to training and corpus harvest (REFUSED_PURPOSES), and HELD_SPLITS, the splits kept from training and corpus harvest, holds eval, unassigned, and dev; tt-pairs-v0's five items and those of tt-pairs-v0-kh, -ttkernel, -ttir, and -d2m are dev, every recipe that reads them names `split: dev`, which changes those recipes' hashes, and the three manifests that record tt-pairs-v0's hash record its new one | OQ-083 (b) made tt-pairs-v0 a development set on 2026-10-25 and left the split value to this task: prompt and feature work steered by its outcomes has been allowed since, while the registry still refused its unassigned items to prompt tuning. A split of its own says what the items are, keeps them from training and corpus harvest as before, and lets tt-par-v0 (OQ-129) hold development and eval items side by side. A run's provenance names its commit, so an earlier run's recipe hash still reproduces at its own commit |
- Count sentence: "Two hundred and six" -> "Two hundred and seven"; "and one on 2026-11-16" -> "one on 2026-11-16, and one on 2026-11-17".
- Also in this batch (same section or the sections named), each Old -> New:
  - Old: Split by item; every item unassigned until P7 (OQ-025). C++ -> TT:
    New: Split by item; every item dev, a development set (OQ-083 (b); task P17.31). C++ -> TT:
  - Old: and the eval and unassigned names with their splits only, in read-only mappings, so no eval or unassigned item and no Suite is reachable from TrainData (Agent Rule 5).
    New: and the eval, unassigned, and dev names with their splits only, in read-only mappings, so no held item and no Suite is reachable from TrainData (Agent Rule 5).
  - Old: which reads the view: an eval or unassigned item raises EvalSplitError
    New: which reads the view: an eval, unassigned, or dev item raises EvalSplitError
  - Old: bench.split must be train: eval and unassigned are refused at load, naming bench.split (Agent Rule 5), and so is any other split.
    New: bench.split must be train: eval, unassigned, and dev are refused at load, naming bench.split (Agent Rule 5), and so is any other split.
  - Old: so an unknown, eval, or unassigned item is refused, naming bench.items;
    New: so an unknown, eval, unassigned, or dev item is refused, naming bench.items;
  - Old: verification/ holds the eval and unassigned records of round trips read for purpose eval,
    New: verification/ holds the eval, unassigned, and dev records of round trips read for purpose eval,
  - Old: The writers of the corpus and of a pair set refuse an eval or unassigned record with EvalSplitError
    New: The writers of the corpus and of a pair set refuse an eval, unassigned, or dev record with EvalSplitError
  - Old: never the held-out inputs of an eval or unassigned item (owner, 2026-10-15, OQ-055, decision 12; Agent Rule 5).
    New: never the held-out inputs of an eval, unassigned, or dev item (owner, 2026-10-15, OQ-055, decision 12; Agent Rule 5).
  - Old: and every eval and unassigned item of that suite, read through the bench registry, must be among them,
    New: and every eval, unassigned, and dev item of that suite, read through the bench registry, must be among them,
  - Old: `lassi train` refuses a bench split of eval or unassigned at load and fetches every bench item through the bench registry for purpose train, which refuses an eval or unassigned item
    New: `lassi train` refuses a bench split of eval, unassigned, or dev at load and fetches every bench item through the bench registry for purpose train, which refuses an eval, unassigned, or dev item
  - Old: - A corpus is built without every eval and unassigned item (Agent Rule 5) [DESIGN]
    New: - A corpus is built without every eval, unassigned, and dev item (Agent Rule 5) [DESIGN]

## 2026-11-17: tt-par-v0's development items (P17.67)

Mirror edits in docs/BIBLE.md, to apply to the master copy after the batches above:

- Benchmark Suites table, a new row after tt-pairs-v0-kh: | tt-par-v0 | Multi-core C++ -> TT items (tasks P17.67 and P17.68; OQ-129): development items from five of tt-metal's multi-core programming examples at the pin (vecadd_multi_core, vecadd_sharding, pad_multi_core, matmul_multicore_reuse, matmul_multicore_reuse_mcast; plans/spikes/tt-par-upstream.md), at sizes whose outputs span more tiles than the grid has cores, each a C++ program and a TT host program written here that runs the upstream kernels unmodified, named by path and sha256 only; held-out items written here (P17.68): row_sum, col_max, transpose_2d, row_softmax, and col_scale, each a C++ program and a TT host program and kernels, all tracked. No reference has been built or run yet (P17.76) | Hand-written (cpp -> tt) | Per item: the development items dev (OQ-129), the held-out items eval |
- Decision Log, first data row: | 2026-11-17 | tt-par-v0, the multi-core C++ -> TT suite, and tt-par-v0-kh, its items with a ttkh language as tt-pairs-v0-kh is to tt-pairs-v0 (tasks P17.67, P17.68, P17.69, and P17.76; Benchmark Suites; OQ-129, both variants): its development items are five of tt-metal's multi-core examples, four of them resized, so each output spans more tiles than an 8x9 grid has cores, with C++ and TT host programs written here and the upstream kernels named by path and sha256; its held-out items are programs written here (P17.68); its programs are written on the workstation and built and checked on ttsim by P17.76, with the owner's yes (OQ-130) | In the window of 2026-11-16 every gpt-oss-120b translation of tt-pairs-v0 matmul_multi_core ran on one core where the reference ran on 56 (rx 20261116-212311-tracy-0bc3), and tt-pairs-v0 holds one multi-core item, so prompt and skeleton work toward the core grid had one item to steer by and none to measure on. The owner chose items split per item, both variants, and items from tt-metal and written here (OQ-129); tt-metal's examples are close to Tier A and partly in the retrieval corpus, so they are the development items. The builds and ttsim runs moved to P17.76 because the workstation has no compiler |
- Count sentence: "Two hundred and seven" -> "Two hundred and eight"; "and one on 2026-11-17" -> "and two on 2026-11-17".

## 2026-11-17: tt-par-v0's held-out items (P17.68)

The tt-par-v0 row queued above (P17.67) now names the held-out items; apply it in the form above, which this task's edit of the mirror already carries.

## 2026-11-17: tt-par-v0-kh (P17.69)

Mirror edit in docs/BIBLE.md: Benchmark Suites table, a new row after tt-par-v0: | tt-par-v0-kh | tt-par-v0's ten items unchanged, plus a ttkh language: the whole TT program a translation writes, the host program and every kernel it launches, at the paths the item's tracked host program gives them, with lassi_io.h its only support and the tt program its reference (task P17.69; OQ-129, both variants) | Hand-written (cpp -> ttkh) | As tt-par-v0: the development items dev, the held-out items eval (OQ-129) |
- The 2026-11-17 tt-par-v0 Decision Log row queued under P17.67 above now also names tt-par-v0-kh and P17.69, as the mirror carries it.

## 2026-11-17: the core-grid prompt sets (P17.72)

Mirror edits in docs/BIBLE.md:

- Repository Layout, the prompt sets list: Old: tt-kh-v0/ (cpp -> ttkh templates: the host program and every kernel, P17.39)
  New: tt-kh-v0/ (cpp -> ttkh templates: the host program and every kernel, P17.39), tt-host-v2/ and tt-kh-v1/ (tt-host-v1 and tt-kh-v0 with the goal of spreading the work over the compute grid and the multi-core calls, P17.72)
- Decision Log, first data row: | 2026-11-17 | The core-grid prompt sets tt-host-v2 and tt-kh-v1 (task P17.72; Repository Layout; OQ-129): each is tt-host-v1 or tt-kh-v0 with exactly the edits tests/prompts/test_tt_parallel_prompts.py pins, two kinds in tt-host-v2 and a third in tt-kh-v1: the goal, to spread the work over the device's whole compute grid, after the rule that the device computes every output, in the generation and both correction templates; in generate.txt, the single-core core line replaced by the multi-core calls at the pin (compute_with_storage_grid_size, split_work_to_cores from work_split.hpp, corerange_to_cores, runtime args per core); and, for tt-kh-v1, kernels that read their core's first tile and tile count from their runtime args. Opt-in: a recipe names the set, and the v1 and v0 sets are unchanged | In the window of 2026-11-16 the matmul_multi_core translations ran on one core (rx 20261116-212311-tracy-0bc3), and tt-host-v1 showed only the single-core core line. The edits name no item and come from the pinned API, after tt-par-v0's held-out items were committed (P17.68), so only development items can steer them (Agent Rule 5) |
- Count sentence: "Two hundred and eight" -> "Two hundred and nine"; "and two on 2026-11-17" -> "and three on 2026-11-17".
