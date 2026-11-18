# The v1 Skeletons At The Pin On ttsim (Task P17.71)

STAGES THAT RAN: the compile of tt-kh-skeleton-v1's host.cpp (byte for byte tt-host-skeleton-v1's, sha256
22a3bd75...d617) with the pinned clang++-20 against tt-metal 5280a9cf-tracy, as the ttmetal-host toolchain compiles a
host program (plans/runs/night/skeleton_build.sh); then tt-kh-skeleton-v1's placeholder program, its host program with
its reader, compute, and writer kernels (a tile copy), run on ttsim v1.3.4 for bf16 inputs of 1 and 200 tiles, plain
and with the Watcher (plans/runs/tt-par/skeleton_run.sh). STAGES THAT DID NOT RUN: no model, no recipe, no silicon.
ttsim is a simulator: no figure here is a timing or performance measurement (Agent Rule 2). Run in the owner's window
of 2026-11-17 (OQ-130 (a); plans/runs/2026-11-17-window-plan.md).

## Provenance

rx 20261117-200159-spike-cores-7739 on alpha01 (slot spike-cores), from clean commit
62c99aa736436df71474e1ae7a037ffdc93d058f, 2026-11-17T20:01:59-08:00 to 20:02:16-08:00, rc 0: provenance.json here
(the gate's record). Its output.log, pulled to the workstation's .rx/pulls, is quoted below; the build directory on
alpha01 is $LASSI_SCRATCH/skeleton-build.PCjjCB. Two earlier runs, not pulled and not reportable on their own: an
exploratory run from a snapshot commit, rx 20261117-195721-spike-cores-a063, printed the same result lines, and the
compile alone from clean commit 96739bc, rx 20261117-193147-desktop-8r113ei-detached-96739bcf-a764, printed exit 0
with no error or warning line.

    core limit (soft hard): 1 1
    commit: 62c99aa736436df71474e1ae7a037ffdc93d058f (0 changed paths)
    host.cpp sha256: 22a3bd75a8bde876ed3bbfcb7d5abb021377b83ad815c1c76d29521bea53d617
    compiler: /usr/bin/clang++-20 (Ubuntu clang version 20.1.8 (...)), the rest of the version line left out here
    compile exit status: 0
    artifact: main, 66768 bytes
    error lines: 0; warning lines: 0
    == 1 tile(s), plain: exit 0; output equals input: yes
    == 1 tile(s), watcher: exit 0; output equals input: yes
    watcher: dump #2: 1 worker core(s) with a nonzero kernel id of 80
    == 200 tile(s), plain: exit 0; output equals input: yes
    == 200 tile(s), watcher: exit 0; output equals input: yes
    watcher: dump #1: 72 worker core(s) with a nonzero kernel id of 80

## Results

- [MEASURED, alpha01's CPU, clang++-20 20.1.8] The compile: exit status 0, artifact main of 66768 bytes, 0 error lines
  and 0 warning lines.
- [MEASURED, ttsim v1.3.4 on alpha01] 1 tile, plain and with the Watcher: exit 0, and the output equals the input
  (dtype, shape, and every byte); the Watcher's last complete dump shows 1 worker core of 80 with a nonzero kernel id.
  With fewer tiles than cores, split_work_to_cores gives one core the one tile and the other cores no kernel, as
  P17.71 expects.
- [MEASURED, ttsim v1.3.4 on alpha01] 200 tiles, plain and with the Watcher: exit 0, and the output equals the input;
  72 worker cores of 80 ran a kernel, the count plans/spikes/ttsim-cores.md infers as the compute grid [PROJECTED: the
  grid is not printed by a program]. Inferred from these lines, not printed per core: 200 tiles cannot split evenly
  over 72 cores, every one of the 72 ran a kernel, and the output is right, so the split with two tile counts
  (cores_1 and cores_2) works.

So both v1 skeletons' host program compiles at the pin, and a program built from it runs on ttsim with one tile and
with more tiles than the grid has cores. The host-only skeleton's kernels are an item's own, so its run is the kh
skeleton's host program with those placeholder kernels.
