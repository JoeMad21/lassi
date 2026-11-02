# Silicon Launch Kit: The Correction Loop With The Wormhole Cards As Its Final Test

Task P17.56 (the owner, 2026-10-29, about 22:00 EDT: "Get everything to the point that the IOMMU thing is the final
issue. I want to make sure that we can launch efficiently once the issue is resolved."). The software side is built
and tested with fakes: no gate on silicon use (P17.53), the sandbox under a translating IOMMU (P17.54), and silicon as
the loop's final test (P17.55). What is left is the owner's and the host's: the steps marked OWNER below. Nothing in
this file has run on the cards; every step that touches alpha01 waits for the owner's approval in the working session
(Agent Rule 17).

## What Runs

- Recipes: projects/lassi-demo/sweep10-si/gpt-oss-120b/, four recipes, the MLIR path only (the owner's MLIR-only
  direction of 2026-10-29): cpp -> ttkernel on tt-pairs-v0-ttkernel, combos ref and rag, halves maxabs and pcc, each
  sweep10's recipe plus `final_executor: {kind: tt_silicon, hugepages: private-per-run, reset_filter: true,
  device_profiler: kernel}` and `sandbox.mem_gb: 32` (PROJECTED; the recipes' comments say why). TTIR and D2M join
  once a silicon executor runs their flatbuffer (below, Later).
- Per attempt: build, run on ttsim (a simulator), judge by binary_io; a pass is built once more and run once on the
  cards through the sandboxed tt_silicon executor on the same held-out inputs, judged by the same oracle. A pass there
  passes the trial; a failure or mismatch there is a correction within the cap of 8 (lassi.core.stages, the final
  test). Each trial records both runs (Attempt.run and Attempt.final_test), the silicon run's device time from
  tt-metal's device profiler (RunInfo.device_timing), and the cards' device line, IOMMU groups and types included.
- Safety that stays (the owner's follow-up of 2026-10-29): one placement at a time under the cards' lock
  ($LASSI_SCRATCH/locks/tt-silicon.lock, waited for, up to LOCK_WAIT_S), never a retry or a Watcher rerun, the
  quarantine marker after a hang, a kill, or a crash (every later silicon run refused until a person clears it), and
  the RNGD cards serving the model beside it (Agent Rules 8 and 13).

## Before The Window, In Order

1. OWNER: Agent Rule 9 in the owner's words, in docs/BIBLE.md, AGENTS.md, and the master copy, with a Decision Log
   entry (plans/rule9-proposed.md holds the proposed text). OQ-099 closes then. Until it lands no agent launches a
   silicon run.
2. OWNER or root: every card in a translating (DMA) IOMMU domain (OQ-078 (a); docs/tt-silicon-iommu.md): either boot
   alpha01 without iommu=pt, so every group defaults to a DMA domain, or, per card, unbind the tenstorrent driver,
   write DMA to /sys/kernel/iommu_groups/<n>/type (groups 135 to 138 and 150 to 153 on 2026-10-29), and bind it
   again. Agents never make this change.
3. The read-only check, with the owner's yes (one CPU-only exec, under a minute, files and links only, no card
   opened). It prints each card and exits 0 only when there are eight cards and every type starts with DMA (DMA or
   DMA-FQ, the types UMD reads as an IOMMU, umd pcie/pci_device.cpp:105):

   ```
   uv run tools/rx.py exec -- 'n=0; bad=0; for d in /sys/class/tenstorrent/tenstorrent!*; do n=$((n+1)); a=$(basename "$(readlink "$d/device")"); g=$(basename "$(readlink "/sys/bus/pci/devices/$a/iommu_group")"); t=$(cat "/sys/bus/pci/devices/$a/iommu_group/type" 2>&1); echo "card=${d##*!} pci=$a group=$g type=$t"; case "$t" in DMA*) ;; *) bad=1 ;; esac; done; echo "cards=$n not_dma=$bad"; [ "$n" -eq 8 ] && [ "$bad" -eq 0 ]'
   ```

   On 2026-10-29 every type read identity (plans/spikes/tt-silicon-executor.md, Follow-up (task P17.54)), so this
   check fails today. Record its output in that spike when it passes.
4. OWNER: the gate's tt_silicon device class enabled in the gate's config.json on alpha01 (AGENTS.md, Remote
   Execution). `uv run tools/rx.py doctor` shows it under devices_enabled.
5. Read-only, in the same exec batch as step 3 if the owner agrees: no quarantine marker and enough 1 GiB pages (UMD
   reads nr_hugepages in either IOMMU mode for its channel count; two per card):

   ```
   uv run tools/rx.py exec -- 'ls -la "$LASSI_SCRATCH/locks" 2>&1; cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages /sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages'
   ```

   A file whose name holds quarantine means a person checks the cards and removes it first.
6. The install: device_profiler kernel needs the Tracy-enabled tt-metal install of the pin (task P17.42; the executor
   refuses at construction without it). `rx doctor` and the install record under $LASSI_TOOLCHAINS show it.
7. Recommended, the owner's choice: the qualification first, a tracked reference through tt_silicon under the DMA
   domain before any model output (P17.17; tests/executors/test_tt_silicon_remote.py, the loopback reference), as one
   short job the owner starts. Its result says whether tt-metal runs end to end under the DMA path at the pin, which
   no one has measured.
8. The dry run on the workstation, which contacts nothing (done on 2026-11-01 from the working tree; repeat from the
   launch commit):

   ```
   uv run tools/suite.py plan --models gpt-oss-120b --sweep projects/lassi-demo/sweep10-si --hours 6 --skip-download --name silicon
   ```

   Its devices line names the four recipes that bind tt_silicon.

## The Window Runner's plan.json

The owner's launcher (the window runner, outside the repository) reads a plan.json. For a silicon window it is the
usual suite launch with the step 3 check as its smoke and no fallback, so nothing launches unless every card reads
DMA (an agent's practice the owner accepted on 2026-10-29; the owner may set smoke to null). Fill in the date, the
times, the branch, and the deadline, and copy the worktree, git, uv, and path_prepend values and the report.py
path from the plan.json of the window of 2026-10-29.

```json
{
  "note": "Silicon window: sweep10-si (cpp -> ttkernel, the ttkernel path), gpt-oss-120b on RNGD, every attempt on ttsim first and the passes on the Wormhole cards as the final test.",
  "date": "YYYY-MM-DD",
  "start_at": "HH:MM",
  "hard_end": "HH:MM",
  "worktree": "<the launch worktree>",
  "branch": "<the launch branch>",
  "git": "<as in the window of 2026-10-29>",
  "uv": "<as in the window of 2026-10-29>",
  "path_prepend": ["<as in the window of 2026-10-29>"],
  "smoke": ["uv", "run", "tools/rx.py", "exec", "--", "n=0; bad=0; for d in /sys/class/tenstorrent/tenstorrent!*; do n=$((n+1)); a=$(basename \"$(readlink \"$d/device\")\"); t=$(cat \"/sys/bus/pci/devices/$a/iommu_group/type\" 2>&1); echo \"card=${d##*!} pci=$a type=$t\"; case \"$t\" in DMA*) ;; *) bad=1 ;; esac; done; [ \"$n\" -eq 8 ] && [ \"$bad\" -eq 0 ]"],
  "smoke_timeout": 600,
  "launch_ok": ["uv", "run", "tools/suite.py", "launch", "--models", "gpt-oss-120b", "--sweep", "projects/lassi-demo/sweep10-si", "--until", "YYYY-MM-DD HH:MM", "--skip-download", "--name", "silicon", "--yes"],
  "launch_fallback": null,
  "after": [
    ["uv", "run", "tools/suite.py", "pull", "{id}", "--name", "silicon-YYYYMMDD"],
    ["uv", "run", "--no-project", "python", "<the window runner folder>/report.py", "table", "results/silicon-YYYYMMDD"]
  ]
}
```

## The Pull And The Report

- `uv run tools/suite.py pull <job id> --name silicon-YYYYMMDD` copies the job's provenance, log, and output
  directory into results/silicon-YYYYMMDD (a short CPU-only read; Agent Rule 17 applies).
- The job's tables (plans/runs/rngd-sweep/tabulate.py, run by run-suite.sh) include final_tests.tsv: one row per
  final test with the executor, passed, the stage, the device kernel time in ns summed over the run's programs, the
  AICLK, and the device line. lassi.analysis.tt_rates final_test_rates derives the rates from each record, labelled
  with the final test's device; no rate ever comes from ttsim (Agent Rule 2).
- A summary.md in results/silicon-YYYYMMDD cites provenance.json for every number (commit, pins, the cards' device
  line with each IOMMU type, driver, date). A run from a dirty tree is exploratory, never [MEASURED].
- If a quarantine marker was written, the report names it, and no later silicon run starts until a person checks the
  cards and removes it.

## Later

- TTIR and D2M (the ttmlir-ttsim route, which runs a TTIR flatbuffer): the same final_executor section applies once a
  silicon executor runs that artifact on the cards. The final test reads no artifact kind: it rebuilds the attempt's
  files with the target's toolchain and hands the artifact to the final executor, so the work is that executor and
  its sandbox exposure, not the loop.
- bale-v0's apps on the ttkernel path: their recipe set joins sweep10-si's pattern once the suite's ttkernel recipes
  exist.

Note: the gate's tt_silicon class is not a stop for a suite launch. The gate refuses only commands that match its patterns (tools/server/gate.py; config.default.json lists tt-smi, /dev/tenstorrent, tt-flash, tt-topology), and none matches a suite launch (plans/runs/suite/README.md), so nothing in the gate stops sweep10-si; the stop is the agents' practice of launching no model-generated program on the cards while any card's IOMMU domain is identity, and Agent Rule 9's current text.
