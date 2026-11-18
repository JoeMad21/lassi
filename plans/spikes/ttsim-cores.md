# Spike: Counting The Cores A Program Uses Under ttsim (Task P17.70)

Question (plans/p17-portable.md, P17.70): how can a ttsim run count the cores a TT program's kernels run on, as a
count and never a time (Agent Rule 2), and how many cores does ttsim's grid have? Run in the owner's window of
2026-11-17 (OQ-130 (4)).

## Method

plans/spikes/ttsim-cores/run.sh reruns built tt-par-v0 references (tracked host programs and upstream kernels, no
model-generated code, so outside the sandbox; Agent Rule 6) from the check run ttpar-check-dev-1117
(results/tt-par-v0-ttsim) on ttsim v1.3.4 with tt-metal 5280a9cf-tracy, in a fresh copy under
$LASSI_SCRATCH/spike-ttsim-cores/<stamp>, three ways: plain; with tt-metal's device profiler (the settings
lassi.executors.tt_profiler.profiler_environment sets on the cards); and with the Watcher (TT_METAL_WATCHER=1). Its
environment is the ttsim executor's (lassi.executors.ttsim program_environment) but for two values: TMPDIR is the
copy's own tmp directory (the executor's is /tmp inside the sandbox), and TT_METAL_THREADCOUNT is 8 (the executor's is
the run's CPU limit).

    uv run tools/rx.py run --slot spike-cores -- 'bash plans/spikes/ttsim-cores/run.sh'

The record run: rx 20261117-195815-spike-cores-5c02 on alpha01, from clean commit
def005a8d20216ee1887fe5b036efd5bd4cbfc09, 2026-11-17 19:58 alpha01 time (22:58 EST), over the five development
references (`bash plans/spikes/ttsim-cores/run.sh vecadd_multi_core vecadd_sharding pad_multi_core
matmul_multicore_reuse matmul_multicore_reuse_mcast`), its files under
$LASSI_SCRATCH/spike-ttsim-cores/20261117-195815; the profiler's error line was read from that run's stdout.txt with
rx 20261117-195934-exec-4b84. Exploratory runs from snapshot commits before it, not reportable: rx
20261117-194514-spike-cores-53f7, 20261117-194621-spike-cores-72ef, 20261117-194704-spike-cores-a70a.

## Ways Compared

1. The device profiler, which gives each program's CORE COUNT on the cards (P17.42): under ttsim each of the five
   programs exits 1 after "Profiler started on device 0" and writes only new_zone_src_locations.log; its outputs are
   not written. vecadd_multi_core's output, read with rx 20261117-195934-exec-4b84 (and in the first exploratory
   run), holds "ERROR: UnimplementedFunctionality: noc_regs_rd32: offset=0x1c"; the other four's error lines were not
   read. ttsim v1.3.4 does not
   implement the register read the profiler needs, so this way does not work under ttsim.
2. The kernel JIT's outputs (@ttsim/cache/<cache>/<hash>/kernels/<kernel>/<hash>/): one build per kernel and its
   compile-time arguments, shared by every core the kernel runs on, so they name the kernels, not the cores.
3. tt-metal's Inspector logs, which every ttsim run already writes (@ttsim/logs/generated/inspector/): kernels.yaml
   and programs_log.yaml name each kernel, its source, and its program, with no core range, so they cannot count
   cores.
4. The core sets the host program passes to CreateKernel: reading them needs a change to the host program, which a
   model writes, so it is not tried.
5. The Watcher (TT_METAL_WATCHER=1): the program's outputs equal the check run's, and the last complete dump of
   @ttsim/logs/generated/watcher/watcher.log has one line per core, each with the kernel ids of the last launch
   message the core received (k_ids, five fields). A worker core whose k_ids hold a nonzero id ran a kernel.

## Counts

[MEASURED, ttsim v1.3.4 on alpha01, the record run rx 20261117-195815-spike-cores-5c02, commit def005a] Each
reference's plain and Watcher runs exited 0 with outputs equal to the check run's. The exploratory runs gave the same
counts for the three items they ran.

| Reference | Watcher worker core lines | Workers with a nonzero kernel id | Their k_ids (workers) |
| --- | --- | --- | --- |
| vecadd_multi_core | 80 | 72 | 1/2/3/3/3 (72) |
| vecadd_sharding | 80 | 56 | 0/0/1/1/1 (56) |
| pad_multi_core | 80 | 72 | 1/2/0/0/0 (72) |
| matmul_multicore_reuse | 80 | 56 | 2/1/3/3/3 (56) |
| matmul_multicore_reuse_mcast | 80 | 56 | 5/3/7/7/7 (7), 5/4/7/7/7 (42), 6/1/7/7/7 (1), 6/2/7/7/7 (6) |

vecadd_sharding's 56 is the core count its manifest sets (448 tiles on 56 cores), and matmul_multicore_reuse_mcast's
four kernel sets are its sender and receiver roles.

Each dump also lists 16 idle ethernet cores ("idleth"). ttsim's descriptor is tt-metal's
wormhole_b0_80_arch.yaml (toolchains/ttsim.pin), and the dumps list 80 worker cores. vecadd_multi_core and
pad_multi_core split their work with the grid compute_with_storage_grid_size returns and reach 72 workers: [PROJECTED:
inferred from these counts, not printed by a program] a compute grid of 8 x 9 under slow dispatch.
matmul_multicore_reuse's 56 agrees with the arithmetic of plans/spikes/tt-par-upstream.md, which projected 56 for it
under ttsim if ttsim's grid is 8 x 9.

## Choice

The Watcher's last complete dump, counting the worker cores whose k_ids hold a nonzero id, is the way P17.70 builds
behind the ttsim executor, with these limits:

- It reads each core's last launch message, so a run of several programs counts, for each core, the program that
  launched on it last; the tt-par references and the skeletons launch one program.
- The Watcher polls the device while the program runs, so the count comes from a separate run with it on, as the
  executor's hang rerun already does (lassi.executors.ttsim, watcher_note), only for a run the stages read as
  clean (built task P17.70: under the ttsim executor's count_cores setting).
- The count is a count of cores; the Watcher's timestamps and ttsim's wall time are never read as a time.
