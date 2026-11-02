# Agent Rule 9: The Proposed Replacement Text

For the owner's decision (AGENTS.md, Authority: changing an Agent Rule needs the owner's decision first). The owner's
answers of 2026-10-29: to OQ-078, "Don't gate the DMA IOMMU feature, remove all gates. I'm okay with you not allowing
the runs, but you cannot enforce gates on my software."; to OQ-099, "It should be like TT-SIM. SIlicon [sic] is there
as a final test gate, no human's required."; and the follow-up that evening, which kept one placement at a time with
no looped device opens, the quarantine after a hang (OQ-075 (3a)), and training with no network (OQ-107 (a)): "First
of all, you should be open to use as many Tenstorrent cards as is necessary alongside the Furiosas. ... Second, OF
COURSE silicon runs should be included in RL rewards. Remove that rule that says RL rewards must never come from
silicon. It's fine, they have to pass the simulator and compiler first." The software side is tasks P17.53 to P17.56.

## The Current Text

The same line in both files: docs/BIBLE.md, Agent Rules, item 9 (line 53 at 40bd4a8), and AGENTS.md, Agent Rules,
item 9 (line 35 at 40bd4a8), which restates the bible's rules verbatim.

```
9. Tenstorrent silicon: one placement at a time, never looped device opens (they have taken alpha01's network down twice). RL rewards never execute on silicon; silicon runs are evaluation runs a human starts.
```

## The Proposed Text

Replace that line, in both files, with exactly this line:

```
9. Tenstorrent silicon is an executor like ttsim, with no human start: a candidate that passed the compiler and ttsim runs on the cards as the correction loop's final test, and such silicon runs may feed RL rewards. One placement at a time, holding as many cards as a program needs beside the RNGD serving, and never looped device opens within a program (they have taken alpha01's network down twice); a run that hangs quarantines the cards until a person clears them. Training runs on the cards take the same lock and quarantine and run with no network.
```

Optional, the owner's choice: the agents' practice the owner accepted on 2026-10-29 ("I'm okay with you not allowing
the runs"), as a last sentence of the same line. It binds agents only; no code enforces it:

```
 Agents launch no model-generated program on the cards while their IOMMU domain is identity (pass-through).
```

How it differs from the wording plans/p17-portable.md, P17.53, proposed on 2026-10-29:

- "with no human start" moves to the first clause, and "such silicon runs" ties the RL rewards to candidates that
  passed the compiler and ttsim, the owner's condition. The loop's final test and df-v0's reading of it are built so
  (lassi.core.stages, the final test; lassi.scoring.df_v0 scored_run).
- "beside the RNGD serving" carries "alongside the Furiosas".
- The draft's last sentence, "Training runs of tracked code on the cards follow OQ-100 (a).", is replaced: OQ-100 (a)
  required a human start (`lassi train --silicon`), which P17.53 removed for training as for evaluation, and its
  sft-only reading came from the reward rule now removed. What stays for training is the lock, the quarantine, and
  no network (OQ-107 (a)), so the sentence names those.

## Applying It

1. The owner records the decision in plans/OWNER-QUEUE.md, OQ-099 (and the wording chosen, with or without the
   optional sentence).
2. docs/BIBLE.md, line 53, and AGENTS.md, line 35: the line above, the same bytes in both.
3. A dated Decision Log entry, newest first, and the count sentence updated (2026-11-01 or the day it lands).
4. The same edit to the master copy (the bible-sync procedure), or plans/BIBLE-SYNC.md if it is unreachable.
5. OQ-099 closes; OQ-078 closes when P17.53 to P17.56 are committed.

## Other Bible Text The Same Edit Makes Stale

These sentences describe gates P17.53 removed or the rule as it reads today. They are design text, not Agent Rules,
so they can land in the same bible edit (each needs the master mirror too):

- Execution Backends, the opening paragraph: "silicon runs are only evaluation runs a human starts, one placement at
  a time (Agent Rule 9)", "not before OQ-078 is answered, since until then the runner refuses that executor beside any
  stage that runs model-generated code or an adversary's proposals". Now: tt_silicon is an executor like ttsim, a
  recipe's final_executor section makes it the loop's final test (task P17.55), and no stage, reward, flag, or IOMMU
  domain refuses it.
- Execution Backends, the tt_silicon row: "human-started evaluation only, through a command-line flag a recipe cannot
  set and an unattended session never passes (OQ-075 (6a)); never a reward (Agent Rule 9)"; "whose iommu_group/type
  starts with DMA ... So under OQ-078's option (a) ... every card is refused until the sandbox exposes the card's IOMMU
  group view, a follow-up not built"; "with LOCK_NB, and refuses at once when another process holds it" (now waited
  for up to LOCK_WAIT_S, trying every 5 s); "Until OQ-078 is answered the runner refuses the executor, with or without
  `--silicon`, ..."; "[OPEN] whether model output runs on silicon before then (OQ-078)" (answered). New facts there:
  the IOMMU group view (task P17.54: each card's device directory bound at its own path, the bus path a link with the
  host's text, /sys/kernel in view), no hugepage directory and no hugepage poll when every card is in a DMA domain,
  the device() line naming each card's IOMMU group and type, and the PROJECTED memory note (UMD's anonymous host
  memory under DMA counts toward sandbox.mem_gb).
- Component Interfaces, the Executor bullet (line 236 at 40bd4a8): its sentences on the silicon capability's
  refusals by the runner and the train layer.
- Execution Backends, Sandbox, the tt_silicon program runs bullet (line 847 at 40bd4a8): "only a run the tt_silicon
  executor ..." and the identity-domain limit, which now holds only under pass-through.
- Project Recipes, Notes: the train recipes bullet (line 532 at 40bd4a8) on `lassi train --silicon`, and a new bullet
  for the final_executor section (task P17.55; lassi.core.recipe FINAL_EXECUTOR).
- Training Module, Compute, the Wormhole bullet (line 1028 at 40bd4a8): "trains sft only (no reward, so RL never
  touches silicon; Agent Rule 9)" (the ttxla trainer declares sft alone, but nothing else limits a card to sft now)
  and "`lassi train --silicon` ... is the human start of such a run".
- Training Module, Safeguards (line 1036 at 40bd4a8): "Rewards execute on ttsim or native only, inside the sandbox;
  never on Tenstorrent silicon: lassi train refuses a reward.executor that names an Executor declaring `silicon`, and
  the runner refuses such an executor beside a stage that declares `computes_reward`, even with `--silicon`" and "A
  training run on a card trains sft only". Now: rewards execute inside the sandbox; a silicon run feeds a reward only
  as a final test after the compiler and ttsim (df-v0 scores an attempt with a final test by that run).
- Training Module, Reward Function, the scoring decisions: df-v0's reading of Attempt.final_test.
- Result Record: the yaml block gained Attempt.final_test and its FinalTest block in this change (the record test
  reads it); the prose after the block can say what run_loop writes there.
- Decision Log rows that record the old rule stay as history.

## For the owner, before applying it

- "never looped device opens within a program" narrows the current, unqualified clause, whose rationale of 2026-09-22 is that repeated device opens took alpha01's network down twice; OQ-099's question flagged this.
- No pacing between placements is built (OQ-099 (a) listed it as a prerequisite): with these changes the final test and an adversary stage beside tt_silicon open the cards once per run, back to back, with only the lock and the poll between those runs.
