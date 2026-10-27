# Training on a Wormhole card: the owner's guide (task P17.38)

`uv run tools/train_tt.py` starts the window's training on alpha01's Tenstorrent cards as one rx job, which needs
no session once started, and follows, stops, and pulls it. Every command has `--help` with examples.

What it runs, and why it may: OQ-100 (a) with (s2) (the owner, 2026-10-26: "Yes, approved on both fronts."):
training runs of tracked code on the cards, started by the owner; one start runs a sequence of runs, each its own
process, spaced, stopping at the first failure or hang. Every run is sft through the tt-xla wheel (the ttxla
trainer; plans/spikes/tt-training.md, Finding 3): no reward, so RL never touches silicon, and nothing
model-generated runs on a card. Each run holds the cards' lock for its whole length, polls the cards first, and
leaves the tt_silicon quarantine marker after a hang, a kill, or a crash; nothing is retried. Each worker runs in
its own empty network namespace (`trainer.network: none` in every recipe; `unshare --user --map-root-user --net`),
since the wheel's Tracy client listens on TCP 8086 to 8105 and broadcasts on UDP from library load
(plans/spikes/tt-perf.md); a host where that namespace cannot be made refuses the run before any directory exists.

## Before the window

1. The owner enables the gate's `tt_silicon` device class for the window (alpha01's gate config); launch refuses
   while `rx doctor` lists it disabled. The install does not need it.
2. Optionally, install the venv ahead of time, CPU only (downloads about 0.45 GB, PROJECTED, and makes a venv of
   2.5 to 4 GB under `$LASSI_TOOLCHAINS/tt-xla@0.9.0.dev20260223001148`):

   ```
   uv run tools/train_tt.py install
   uv run tools/train_tt.py wait <id>
   ```

   It needs the tt-metal install at the pin (`$LASSI_TOOLCHAINS/tt-metal@5280a9cf`), whose sfpi 7.25.0 it links into
   the wheel. Launch runs the same script first anyway: it keeps a checked install and installs a missing one.
3. From a clean, pushed commit (a clean worktree of it keeps other work out of the job), look at the plan:

   ```
   uv run tools/train_tt.py plan --until 12:00
   ```

## In the window

```
uv run tools/train_tt.py launch --until 12:00
uv run tools/train_tt.py status <id>
uv run tools/train_tt.py wait <id>
```

The sequence, by default (projects/lassi-demo/train-tt/, each recipe's comments holding its PROJECTED time and
memory): the smoke on the synthetic fixture; LoRA on Qwen2.5-0.5B, Qwen3-0.6B, and Qwen3-1.7B; full training of
the 77M scratch model; prompt tuning on Qwen3-1.7B. Choose a subset or another order with `--recipes a,b,...`.
Before any card is used the job builds tt-train-v0's records and fetches each base (CPU only). A recipe starts only
while the time left covers its `trainer.timeout_s`; the first failure or hang ends the sequence.

To stop before the next recipe:

```
uv run tools/train_tt.py stop
```

A run already going ends at its `trainer.timeout_s` or when it finishes: killing a run that holds the card would
quarantine it. `uv run tools/train_tt.py stop --clear` removes the STOP file, without which a new launch refuses.

## A quarantine

After a hang, a kill, or a crash, `$LASSI_SCRATCH/locks/tt-silicon.quarantine` names the ending and the card. Every
later silicon run, `lassi run --silicon` or `lassi train --silicon`, is refused until a person checks the card (a
reset, if one is needed, is the owner's: agents have no SMI tool and the gate refuses its name) and removes the
file. A failure of the compile or a refused record ends cleanly and leaves no marker.

## After the window

```
uv run tools/train_tt.py pull <id>
```

copies the job's provenance and its output directory (summary.tsv, one log per step, and each run's
provenance.json, checkpoint.json, steps.jsonl, and the worker's result.json) into `results/<name>/`. Write
`results/<name>/summary.md` from them. The checkpoints stay on alpha01 under `lassi-runs/train/`. A run is
[MEASURED] only from a clean commit, and its numbers are cited only from the provenance pull writes (Agent Rules 1
and 10).
