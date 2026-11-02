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
