# ttsim run fixtures

Each folder here is one run of a seeded copy of the gate example's kernel (add_2_integers_in_riscv) on ttsim,
captured on alpha01 by the P4.9 batch (plans/spikes/p4-ttsim-runtime.md) and copied with one mechanical rule
(below). `stdout.txt` and `stderr.txt` are the run's streams, `status.json` is the batch's record of the step,
and `watcher-hang/watcher.log` is tt-metal's Watcher log of that run. `captures.json` records, per file, the
sha256 of the pulled capture and of the tracked copy.

Device: ttsim v1.3.4 (libttsim_wh.so, a virtual Wormhole) on the host CPU, with tt-metal 5280a9cf. This is a
simulator, not silicon (Agent Rule 2). Every time or rate inside these files, the log timestamps, ttsim's own
`[<n>] <s> seconds (<f> KHz)` lines, and the Watcher dump times, is simulator wall time or the simulator's own
simulated clock rate: exploratory, sizing only, never performance. No value here is a measurement of a program.

## Provenance

- rx job `20260926-140947-p49-ttsim-runtime-6f4c` on alpha01, 2026-09-26, 14:09:48 to 14:15:03 (UTC-07:00),
  exit status 0; the job record is `results/p4-ttsim-runtime/provenance.json`.
- Commit 755c739 (`755c739e3d44a5a94c5f942f8f3872cc1fbb5fd3`) with a clean tree: status.json has `dirty`
  false and `snapshot_of` null, and the batch refuses an rx snapshot launch.
- Pins: toolchains/tt-metal.pin (tt-metal 5280a9cfb00998fd49667a29523d03aee905c129, built at
  $LASSI_TOOLCHAINS/tt-metal@5280a9cf, results/p4-tt-install) and toolchains/ttsim.pin (ttsim v1.3.4,
  libttsim_wh.so sha256 7b10aa05a5297c4a28f274e39526bfe6c69373661d1b4d4e47c4257e44b79507).
- Environment of each run (env -i): PATH=/usr/sbin:/usr/bin:/sbin:/bin, `LANG=C` and `LC_ALL=C` (so compiler
  quotes are plain ASCII), TMPDIR=/tmp (private), TT_METAL_RUNTIME_ROOT on the pinned tree,
  TT_METAL_SIMULATOR on libttsim_wh.so, TT_METAL_SLOW_DISPATCH_MODE=1, TT_METAL_DISABLE_SFPLOADMACRO=1,
  TT_METAL_INSPECTOR_RPC=0, TT_METAL_CACHE and TT_METAL_LOGS_PATH inside the step, HOME inside the step,
  TT_METAL_KERNEL_PATH on the step's working directory, TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1, and
  stdout and stderr line-buffered (stdbuf -oL -eL); watcher-hang adds TT_METAL_WATCHER=1.
- Each run was one of P4.9's measuring runs, outside the sandbox as plans/p4-ttsim.md allows, under
  `unshare -rmnipf` (no network, private /tmp, /var/tmp, /dev/shm, and /run, the pinned trees read-only)
  and prlimit with a 1-byte core limit. The seeded lines are hand-written reference edits, never model code,
  and each ran only on ttsim's simulated cores under the unmodified upstream host program.
- Each seeded line is in status.json (`seed_line`); it was inserted as its own line right after the line that
  opens the kernel's entry function, so a compiler diagnostic quotes only that line. status.json's
  `seeded_copy_in_cache` true means the step's own JIT cache held the seeded copy.

## Stored text

The batch replaced, by code, every compiler source-context line other than the seeded one, every kernel
compile-command line, and every line naming TT_FATAL, TT_ASSERT, TT_THROW, or backtrace with the one line
`[P4.9: upstream-derived line removed]` (status.json `upstream_lines_removed` counts them). The copy into
this folder applied one more rule, by code: a backtrace frame line (one that starts with ` --- `) becomes the
same marker line, since frames carry host addresses and demangled tt-metal symbols and no parser needs them
(`frames_replaced` in captures.json). Nothing else was changed, so the files hold no upstream source text.
They do hold absolute host paths (the run directory, the pinned tree) and tt-metal's own log lines, which vary
from run to run in their timestamps and addresses. `watcher-hang/watcher.log` was capped by the batch at
64 KiB: its first and last 32 KiB around one line that says how many bytes were cut.

## Fixtures

| Folder | Pulled step | ttsim class | Exit status | What it shows |
| --- | --- | --- | --- | --- |
| `non-contractual-behavior/` | seed-ub-unaligned | NonContractualBehavior | 1 | `[4500] ERROR: NonContractualBehavior: rv32_mem_rd: unaligned addr=0x16dfe2 size=4` for a 4-byte load from an address 2 mod 4: a fourth class, beside the three the bible names (status.json `sim_classes` is empty because the batch's pattern knew only three classes; `sim_lines` holds the line) |
| `undefined-behavior/` | seed-gap-zicsr | UndefinedBehavior | 1 | `[4489] ERROR: UndefinedBehavior: decode_and_execute_csrrs: Wormhole does not support Zicsr` for `csrrs a0, mcycle, zero` |
| `unsupported-functionality/` | seed-ub-fence | UnsupportedFunctionality | 1 | `[4489] ERROR: UnsupportedFunctionality: decode_and_execute_fence: fence instructions do not enforce memory ordering on Wormhole and should not be used` for `fence iorw, iorw` |
| `jit-error/` | seed-jit-error | none (a kernel JIT error) | 134 | the JIT's `brisc build failed. Log:` with the compiler's error at the seeded line (`reader_writer_add_in_riscv.cpp:6:5`), then std::terminate and SIGABRT: a kernel JIT error aborts the host program |
| `hang/` | seed-hang | none | 137 | a kernel waiting on an L1 word nothing writes, stopped at its 120 s limit (SIGTERM, then SIGKILL after 10 s); meanwhile ttsim printed only its progress lines, `[10000000]` to `[50000000]` |
| `watcher-hang/` | watcher-hang | none | 137 | the same hang with TT_METAL_WATCHER=1, stopped at the same limit: Watcher dumped about once a second (128 dumps; this copy keeps dumps 1, 2, 127, and 128 whole and parts of 3 and 126); worker core (0,0) shows BRISC at `R` (run) in the first dump and at `NSW` (noc semaphore wait) in dumps 2, 3, 127, and 128, with k_id 1, the seeded kernel's path |

No UnimplementedFunctionality capture exists: none of the seven seeds produced that class
(plans/spikes/p4-ttsim-runtime.md, Results).
