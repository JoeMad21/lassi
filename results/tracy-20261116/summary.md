# Device Profiles Of The Saved TT Translations, 2026-11-16 (Task P17.65)

STAGES THAT RAN: build with the pinned toolchains, host CPU run of each item's C++ program, a run on card 0 of a
Wormhole n300 (silicon) of each TT reference and translation under tt-metal's device profiler, output check
(binary_io) against the C++ outputs. NOT IN THIS RUN: ttsim (every translation here had passed exactly on ttsim in its
own sweep before; plans/runs/tracy/candidates.tsv), any correction or generation step. Each program ran once.

## Provenance

- Job rx 20261116-212311-tracy-0bc3 on alpha01, `uv run python plans/runs/tracy/profile.py --minutes 90`, started by the
  owner (OQ-127), 2026-11-16 21:23:11 to 21:47:43 PST (2026-11-17 05:23 to 05:47 UTC), exit 0, all 57 steps reached:
  53 ran, and 4 (steps 7, 8, 19, and 20) were recorded as the same program as an earlier step.
- Commit 5d75218b6d948b13860f4adb9adce5033423198e, clean (provenance.json: `"dirty": false`).
- Device, from report/provenance.json: Tenstorrent Wormhole n300 x 8 (silicon), the run on card 0 alone
  (TT_VISIBLE_DEVICES=0), card 0 serial 01000146118320F8, firmware bundle 19.5.0.0, ARC firmware 2.37.0.0, IOMMU group
  135 identity; tenstorrent driver 2.8.0; tt-metal 5280a9cf-tracy (the pin built with ENABLE_TRACY, task P17.42);
  AICLK 1000 MHz as each device log reports it.
- Host CPU for the C++ programs: AMD EPYC 9534 64-Core Processor, g++ 12.3.0 with -O3 -fopenmp.
- Candidates: plans/runs/tracy/candidates.tsv, sha256 7211a06906a7161d6e586e3bd82d99390bf9bdfdeb73eaa40a073e8bf312b5f1.
- Files: provenance.json (the job), report/provenance.json (the run), report/records.jsonl and report/summary.tsv
  (one record per step), report/profiler/<step>/ (each card run's profiler files, profile_log_device.csv among them).

## Results [MEASURED]

Device kernel time is tt-metal's DEVICE KERNEL DURATION of the run's one program, first kernel start to last kernel
end over every core and RISC (kernel_ns in records.jsonl). Cores is the program's core count in its device timing
record, which agrees with the distinct (core_x, core_y) pairs of its profile_log_device.csv. A translation's time is
self-reported: its own process wrote the profiler files. Every reference and translation below passed its output
check on the card.

| Suite | Item | Reference: kernel ns, cores | Translations (model) | Passed on card 0 | Kernel ns | Cores | Time / reference |
|---|---|---|---|---|---|---|---|
| tt-pairs-v0 | loopback | 44,799, 1 | 8 (gpt-oss-120b) | 8 of 8 | 44,213 to 44,941 | 1 | 0.987 to 1.003 |
| tt-pairs-v0 | eltwise_sfpu | 74,272, 1 (step 11) | 6 (gpt-oss-120b) | 6 of 6 | 74,262 to 74,453 | 1 | 1.000 to 1.002 |
| tt-pairs-v0 | eltwise_binary | 41,680, 1 | 6 (gpt-oss-120b) | 6 of 6 | 40,987 to 1,292,572 | 1 | 0.983 to 31.0 |
| tt-pairs-v0 | matmul_single_core | 7,868,523, 1 | 6 (gpt-oss-120b) | 6 of 6 | 7,860,707 to 7,869,015 | 1 | 0.999 to 1.000 |
| tt-pairs-v0 | matmul_multi_core | 544,279, 56 | 6 (gpt-oss-120b) | 6 of 6 | 7,813,901 to 7,901,294 | 1 | 14.4 to 14.5 |
| tt-pairs-v0-kh | loopback | 44,799, 1 (step 2) | 4 (gpt-oss-120b) | 4 of 4 | 47,114 to 47,553 | 1 | 1.052 to 1.061 |
| tt-pairs-v0-kh | eltwise_sfpu | 74,272, 1 | 3 (gpt-oss-120b) | 3 of 3 | 74,248 to 74,269 | 1 | 1.000 |
| bale-v0-tt | transpose_matrix | none (no TT reference) | 1 (Gemma-4-31B-it-FP8-block) | 1 of 1 | 865,967 | 1 | - |
| bale-v0-tt | randperm | none (no TT reference) | 1 (Gemma-4-31B-it-FP8-block) | 1 of 1 | 609,153 | 1 | - |

In tt-pairs-v0 the model wrote the host program and the item's tracked kernels were supplied; in tt-pairs-v0-kh and
bale-v0-tt the model wrote the host program and every kernel.

Findings:

- All 41 translations passed on the card, and every reference passed.
- matmul_multi_core: the reference's device log lists 56 cores of card 0 (8 wide by 7 tall); each of the six
  translations lists one core and takes 14.4 to 14.5 times the reference's kernel time, about the single-core
  reference's. The kernels are the item's tracked multi-core kernels, so the host programs put the work on one core.
  37 of the 39 gpt-oss-120b tt-pairs-v0 and -kh translations, all six matmul_multi_core ones among them, come from
  recipes with a skeleton (tt-host-skeleton-v0 or tt-kh-skeleton-v0), whose cores blank holds one core (host.cpp,
  lines 134 to 136 of each); tasks P17.67 to P17.75 follow from this (OQ-129).
- eltwise_binary step 53 (candidates.tsv order 37): one core like the reference, 31.0 times its kernel time, and every
  RISC's kernel zone about 1.29 ms long (the other five translations: kernel times of 40,987 to 41,802 ns). Its cause is not determined
  here: the trial's host program is on alpha01.
- The model-written kernels of tt-pairs-v0-kh: loopback 5 to 6 percent slower than the reference, eltwise_sfpu equal.
- The two bale-v0-tt translations have no TT reference to compare with. bale-v0-tt is unassigned (OQ-108), so its
  results may not steer prompt work (Agent Rule 5).
- tt-pairs-v0 is a development set (OQ-083 (b)): every result on it is a development result, never a held-out one.

How to read the numbers:

- One run per program: the spread between translations of the same item (for loopback 44,213 to 44,941 ns) is the only
  view of run-to-run variation, and it is not a variance estimate.
- Wall times in summary.tsv (2.8 to 3.3 seconds per card run) hold device open, the kernel JIT, and setup; they
  are never the device time.

## The night's earlier attempts

These come from rx job status and the owner's pasted command output, read in the session; only the last run was
pulled.

- rx 20261116-183955-tracy-e78e and 20261116-184005-tracy-699a: exit 127 at once; the command's single quotes reached
  the remote shell from cmd.exe. Nothing ran.
- rx 20261116-184803-tracy-621d: exit 3 after the executor's 300-second busy poll; another account's 16 UMD hugepage
  files held 16 of the 32 1 GiB pages (OQ-128). Nothing ran on the cards. The owner removed the files
  (rx 20261116-192617-exec-0242).
- rx 20261116-192634-tracy-07f1 (commit f34c0da): exit 3; step 2, the tracked loopback reference, ended with exit
  status 134 and a quarantine marker. With every card visible, tt-metal 5280a9cf-tracy's fabric control plane fell back
  to a 2-chip mesh and MeshDevice::create_unit_mesh(0) raised TT_FATAL "Physical chip id 0 not found in control plane
  chip mapping" (control_plane.cpp:1249) (plans/p17-portable.md, P17.66). Task P17.66 added visible_card; the owner
  checked the cards (rx 20261116-211458-exec-1b79 and 20261116-211506-exec-d003: all eight heartbeats rising, AICLK
  1000, 32 hugepages free), and the marker was removed before this run (the executor refuses every run while it
  exists).
