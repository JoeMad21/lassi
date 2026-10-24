# Run 2026-10-23-evening-window

Window: the owner's alpha01 window, 2026-10-23 20:00 to 24:00 EDT (17:00 to 21:00 on alpha01's clock, UTC-7). One
attended session. This file was written before the window as its runbook, and its plan stands for the next
window; What Ran says what happened.

## The Plan

Every step on alpha01 goes through `uv run tools/rx.py`, from clean commits, after the owner's yes for the batch
(Agent Rule 17). The job runs on the host on its own, so it continues if the session stops.

1. Card poll, read-only: `furiosa-smi ps` and `furiosa-smi status` (Agent Rule 8: claim only cards at 0.00 GiB,
   npu0 included). run.sh polls again every 60 s before each model and waits until enough cards are free, logging
   each poll in card-polls.txt, since the previous user sometimes takes a few minutes to release the cards.
2. Pull the fourth pass's tables ($LASSI_RUNS_ROOT/rngd-sweep/tables: trials.tsv, passes.tsv) into the
   results tree, with provenance, to see how close each model came and at which attempt.
3. The fifth pass, one big job, from a clean commit:

   ```
   PYTHONIOENCODING=utf-8 uv run tools/rx.py job start --big --name rngd-sweep5 --timeout 14400 -- \
     'SWEEP_MAX_SHARE=4200 bash plans/runs/rngd-sweep/run.sh <deadline> projects/lassi-demo/sweep5 \
        plans/runs/rngd-sweep/models5.tsv; python3 plans/runs/rngd-sweep/tabulate.py "$LASSI_RUNS_ROOT" \
        "$LASSI_RUNS_ROOT/rngd-sweep/tables5"'
   ```

   - The deadline is 15 minutes before the window's end, as epoch seconds (`date -d '<end> 15 minutes ago' +%s`;
     23:45 EDT on 2026-10-23 was 1792813500). run.sh starts no model with under 30 minutes left and shares the time left
     among the models still to run, at most 70 minutes (SWEEP_MAX_SHARE=4200) each; a recipe cut by its limit
     keeps every trial it finished. It always stops its server on exit and waits for the cards to read 0.00 GiB.
   - Models, in order (plans/runs/rngd-sweep/models5.tsv): EXAONE-4.0-32B (4 cards), Qwen3-Coder-30B (4 cards),
     Llama-3.1-8B (1 card), one at a time on port 8123.
   - Per model, four recipes side by side (projects/lassi-demo/sweep5/<slug>/): fz and mmfz, each at the two
     tolerance halves (pcc, maxabs); Tier A, 10 trials per recipe pair over both halves, correction cap 15, every
     TT program on ttsim, a simulator. The fuzzer adversary runs after the oracle and never feeds the loop, so each
     trial's oracle outcome is the combination without the fuzzer, and its adversary record the one with it; the
     two recipes give all four combinations:

     | Combination | Mismatch report (loop.correct_on_mismatch, OQ-077) | Fuzzer adversary (P16) | Read from |
     | --- | --- | --- | --- |
     | base | off | off | fz, the oracle outcome |
     | fz | off | on | fz, the adversary record |
     | mm | on | off | mmfz, the oracle outcome |
     | mmfz | on | on | mmfz, the adversary record |

   - Stop early: create $LASSI_RUNS_ROOT/rngd-sweep/STOP with one `rx exec`; the sweep stops before the next model
     and at its next check. Remove the file before the next sweep.
4. After the job: pull $LASSI_RUNS_ROOT/rngd-sweep/tables5 and the pass's summary.tsv into results/ with
   provenance, and fill the Results section.

The correction cap stays 15. The owner asked to consider raising it ("Make a best guess based on time and how
close the systems got last time") and first approved 20; after the earlier passes' tables were pulled
(lassi-runs/rngd-sweep/tables, written by rx 20261022-222829-rngd-sweep4-chain-cf48), the owner chose 15 in the
working session. Those tables show every trial that reached alignment 1.0 did so at correction 0, except Llama-3.1-8B's
cpp -> tt eltwise_binary at correction 4; Qwen3-Coder-30B's cpp -> tt trials in the third pass stayed at S1 through
all 15 corrections (rx 20261022-220226-rngd-sweep3-6fa1); and in the fourth pass each model finished only 1 to 8
trials within its time share, so a higher cap would finish fewer trials. Free space before the job: 111G free on
/mnt/nvme10, the scratch root at 305G under the extended lift (OQ-074; rx 20261023-165250-exec-c546).

## Not In Tonight's Runs

| Feature | Why | Where it stands |
| --- | --- | --- |
| API reference (the tt-metalium pack in the prompts) | P17.21 is not built: the template prompt set cannot carry a context pack yet | Design in plans/p17-portable.md; the usage limit stopped its work at about 15:50 EDT |
| Retrieval over the tt-metal docs (RAG) | P17.22 is not built: no stage calls the retriever yet | The retriever and its corpus spec exist, uncommitted; same stop |
| Training (full, LoRA, prompt tuning, from scratch) | P17.18 and P17.19 are unfinished, so no trained checkpoint exists | Uncommitted work in lassi/train; same stop |
| Model output on the Wormhole cards | OQ-078: under iommu=pt a program holding a card node can reach host memory; the tt_silicon executor (P17.16) is not built | plans/spikes/tt-card-dma.md |

## What Ran

The window was lost: the fifth pass never started. After the owner approved the batch, the session held the job
for a confirmation question on the correction cap and saw the clock at 23:14 EDT; with under 46 minutes left the
owner chose to skip the rest of the window (plans/LESSONS.md, Owner directions, 2026-10-23, "Commit to testing in a
window"). The recipes and scripts above are committed for the next window.

- rx 20261023-165250-exec-c546 (19:52 EDT), read-only: `furiosa-smi ps` empty; `furiosa-smi status` npu0 to npu7
  at 0.00/47.50 GiB; /mnt/nvme10 111G free (97% used); the scratch root 305G; no STOP file.
- `rx pull --path lassi-runs/rngd-sweep/tables`: trials.tsv and passes.tsv of the first to fourth passes, written by
  rx 20261022-222829-rngd-sweep4-chain-cf48, copied to the workstation for the cap decision only. They are not in
  results/ yet: a pull by that job's id, which writes provenance.json, is the next window's first step.

## Results

None: no run of tonight's recipes took place.
