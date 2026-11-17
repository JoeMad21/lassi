# Device Profiles Of The Saved TT Translations (Task P17.65)

The owner asked on 2026-11-16: "I want to utilize Tracy to understand if our program translations on Tenstorrent
are promising." This kit runs every saved translation that passed exactly on ttsim once on the Wormhole cards, with
tt-metal's device profiler on, and records whether it still passes there and how long its kernels ran on the device.

The device profiler is part of tt-metal's Tracy build: the pin builds tt-metal 5280a9cf with ENABLE_TRACY at
tt-metal@5280a9cf-tracy (task P17.42), and only that build compiles it. The kit reads the profiler's files, not a
live Tracy capture: a sandboxed run has an empty network namespace, so the Tracy client inside cannot reach a
capture server. The raw device log of every card run is kept, so a timeline can be drawn from it afterwards.

## Who Starts It

The owner (OQ-127). Every card is in an identity IOMMU domain (rx 20261116-161608-exec-0f2e), in which a program
on a card can reach host memory through the card's DMA (OQ-078), so no agent launches a model-generated program on
the cards (lassi.executors.tt_silicon, Known limits). Agent Rule 9 holds: one placement at a time, each program one
run, a pause between runs.

## What Runs, In Order

1. The qualification: tt-pairs-v0 loopback's C++ program on the host CPU, then its tracked TT reference on the
   cards through the sandboxed tt_silicon executor with device_profiler kernel and visible_card 0 (task P17.66:
   every card's node is bound, and TT_VISIBLE_DEVICES=0 should keep tt-metal on card 0, as in the 2026-10-23
   smoke, unverified on the cards until this run;
   with all eight cards visible, tt-metal's fabric setup aborted this reference, rx 20261116-192634-tracy-07f1).
   It is the first program this executor runs on the cards. Unless both programs build, run cleanly, and pass,
   and the reference gives a device timing, the run stops (exit 4) before any model-written program is read or
   run.
2. The 41 translations of candidates.tsv, which make_candidates.py wrote from the read-only inventory
   (inventory.py, the text run as rx 20261116-161608-exec-0f2e with one paragraph added and one line wrapped; its output is
   inventory-rx-20261116-161608-exec-0f2e.txt; the file's header counts the rows each rule dropped), in its
   order: Gemma-4-31B's bale-v0-tt transpose_matrix and randperm (the model wrote the host program and the
   kernels), gpt-oss-120b's tt-pairs-v0-kh programs (host program and
   kernels), then gpt-oss-120b's tt-pairs-v0 host programs (the item's tracked kernels). Before an item's first
   translation, its C++ program runs once on the host CPU, and its tracked TT reference, when it has one, runs once
   on the cards. bale-v0-tt has no TT reference; its C++ program is the reference.

Each program is built again from its saved files with the pinned toolchains, on the item's held-out inputs, and
judged by binary_io against the C++ outputs under the item's declared tolerance; a translation is also compared
with its item's TT reference on the cards when that ran.

## Stops

- No step starts once fewer than --budget seconds of --minutes are left (default 420: a build, the pause, and a run
  up to its 300-second wall limit): exit 2. A step that has started can still end later than that: the executor
  waits up to 300 seconds for busy cards, and up to an hour for another process holding the cards' lock.
- Any error from a card run, such as cards still busy after the executor's 300-second poll, or another process
  holding the cards' lock for an hour: exit 3.
- A quarantine marker after a card run (a hang, a crash, a kill, or a kernel compile failure there): exit 3. Every
  later run is refused until a person checks the cards and removes the marker from $LASSI_SCRATCH/locks.
- The qualification failed (its C++ program or its reference did not build, run cleanly, and pass, or the
  reference gave no device timing): exit 4. A translation of the same item that fails is recorded like any other.

## Commands

From a clean checkout of the kit's commit on the workstation, each line one command.

Optional, before the window: the dry run reads and checks the 41 trial files and prints the plan. It builds
nothing and opens no device:

    uv run tools/rx.py run -- 'uv run python plans/runs/tracy/profile.py --dry-run'

Optional, before the window: who holds the 1 GiB hugepages. The executor needs 2 free pages per card, 16 in all, and
waits while another account's file sits directly in /dev/hugepages-1G:

    uv run tools/rx.py exec -- 'cat /sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages; ls -la /dev/hugepages-1G /dev/shm'

In the window, first check that the gate's tt_silicon class is enabled (devices_enabled) and that no big job runs:

    uv run tools/rx.py doctor

The run, with the window's length in minutes (here 120). --timeout is the job's hard stop, in seconds: the minutes
times 60 plus 1800 (here 9000), so the script's own deadline ends it first. A kill during a card run leaves a
quarantine marker that a person must clear:

    uv run tools/rx.py job start --big --name tracy --timeout 9000 -- 'uv run python plans/runs/tracy/profile.py --minutes 120'
    uv run tools/rx.py job tail <job id> -n 80

The run prints its report folder, $LASSI_RUNS_ROOT/tracy/<stamp>/report. Afterwards, pull the job's provenance and
the report (records, summary, profiler files; the builds stay on alpha01):

    uv run tools/rx.py pull <job id> --into results/tracy-YYYYMMDD
    uv run tools/rx.py pull --path lassi-runs/tracy/<stamp>/report --into results/tracy-YYYYMMDD/report

## The Report

- provenance.json: the commit and its clean state, the tt-metal and gcc pins, the cards' device line (each card's
  serial, firmware, and IOMMU group and type; the driver; tt-metal 5280a9cf-tracy), the host CPU, the date, and the
  candidates file's sha256.
- records.jsonl: one record per step, written as the step ends: its status (ran, skipped, build-failed, error),
  whether it passed, exit status, hang, wall time, per-output oracle statistics (and against the TT reference), the
  device timing, the profiler files kept, and the streams' tails. The last record of a stopped run says why.
- summary.tsv: one row per record.
- profiler/<step>/: the run's profiler files, the device log profile_log_device.csv among them.

## Reading The Numbers

- Device time is tt-metal's DEVICE KERNEL DURATION per program: from the first kernel start to the last kernel end
  over every core and RISC, at the AICLK the cards reported (kernel_ns; kernel_ns_total sums the run's programs).
- A translation's device time is self-reported (self_reported true): its own process writes the profiler files, so
  a model-written program could forge them. A tracked reference's is not.
- Wall time is the whole sandboxed command, with device open, the kernel JIT, and setup; it is never the device
  time. The C++ program's wall time is the host CPU's whole process, inputs and outputs included.
- Nothing from ttsim is a time (Agent Rule 2); every number here comes from the cards or the host CPU, named in
  provenance.json.
