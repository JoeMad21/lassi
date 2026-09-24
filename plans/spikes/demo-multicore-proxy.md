# Spike DEMO.1: the nvc++ multicore proxy on the ten OpenMP references

- Task: DEMO.1, demo preparation the owner asked for (not a P1 plan task).
  Bible: Execution Backends, Harness Contract (the bullet "CUDA -> OMP proxy
  without a GPU: build a second binary with `nvc++ -mp=multicore` so target
  regions run on the host. It checks outputs, never runtime."); Risks And
  Questions and the Decision Log, OQ-003 (the compile-only tier plus the
  -mp=multicore proxy).
- Date: 2026-09-24 (EDT). The rx ids embed the build host's clock
  (UTC-07:00), so they read 2026-09-24 00:38 to 00:43.
- Where: branch `p1-faithful` at 35bc5dd with this task's changes (and the
  uncommitted work of the tasks running beside it) sent by rx as a dirty
  snapshot; `rx run` on alpha01 (256 CPUs as `nproc` prints, no GPU used).
  Toolchain: the pinned nvc++ (toolchains/nvhpc.pin, NVHPC 24.11), built by
  build_toolchain in the compile sandbox. Sources: lassi-hecbench-10 at
  HeCBench 692cba3, fetched by tools/fetch_bench.py under $LASSI_SCRATCH/bench.
- Evidence label: exploratory. Every rx run below went from a dirty tree (a
  snapshot commit), so nothing here is [MEASURED]; the rerun from a clean
  commit is the evidence. Wall seconds are exploratory and are never
  performance numbers: the proxy checks outputs, never runtime. Kernel
  times the programs print are left out on purpose.

## Question

Do the 10 OpenMP reference mains compile with Toolchain "nvcpp-multicore"
(`nvc++ -Wall -O3 -Minfo -mp=multicore -o main <sources>`), and do the
built mains run with the manifest's run_args through the registered
"native" executor, in the sandbox, on the build host's CPU?

## Classification

Factual: answered by running the remote test and two probes.

## Method

1. `uv run tools/rx.py doctor` (ok, scratch free 427.6 GB, no running jobs)
   and `uv run tools/rx.py exec -- 'du -sh /mnt/nvme10/joseph_ufl'`
   (94G, rx 20260924-003801-exec-7c72), before any run.
2. The remote test, three times as the test changed (the last is the table
   below):
   `uv run tools/rx.py run --timeout 3000 --tail 100 -- 'LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -s -m remote tests/bench/test_multicore_proxy.py; echo "pytest_status=$?"'`
   - rx 20260924-003829-desktop-8r113ei-p1-faithful-5617: memory 8192 MB,
     no stderr tail in the report.
   - rx 20260924-003943-desktop-8r113ei-p1-faithful-d008: the report adds
     a failed run's stderr tail.
   - rx 20260924-004241-desktop-8r113ei-p1-faithful-4af9: memory 32768 MB
     (atomicCost's main allocates two arrays of 922521600 doubles, about
     14.8 GB, read from its source). pytest status 0, `1 passed`.
3. Two probes, each a Python script sent inline through `rx run` (base64,
   written to $TMPDIR, removed after). Each compiles an app with
   build_toolchain("nvcpp-multicore") and runs a two-line shell wrapper
   (`OMP_NUM_THREADS=<n> exec ./main "$@"`, or similar) in the build dir
   through the registered "native" executor, so every run stays in the
   sandbox (Agent Rule 6):
   - rx 20260924-004125-desktop-8r113ei-p1-faithful-44bd: layout plain,
     with NVCOMPILER_TERM=trace, with OMP_NUM_THREADS 4 and 64, and the
     sandbox's view (ulimit, nproc, the CPU list); atomicCost plain and
     with 4 threads (memory 8192 MB, wall 60 s).
   - rx 20260924-004319-desktop-8r113ei-p1-faithful-fff8: layout at
     OMP_NUM_THREADS 200, 240, 248, 252, 256; atomicCost at 16 threads
     (memory 32768 MB, wall 180 s).

## Results

The remote test's report (rx 20260924-004241-desktop-8r113ei-p1-faithful-4af9;
wall 180 s, memory 32768 MB, cpus 16; wall_s exploratory):

| App | Compile | Exit | wall_s | Timeout | stdout bytes | PASS seen |
| --- | --- | --- | --- | --- | --- | --- |
| atomicCost | ok | 134 | 0.87 | False | 0 | no |
| bsearch | ok | 134 | 0.37 | False | 0 | n/a |
| colorwheel | ok | 134 | 0.32 | False | 0 | no |
| dense-embedding | ok | 134 | 0.42 | False | 0 | no |
| entropy | ok | 134 | 0.47 | False | 0 | no |
| jacobi | ok | 134 | 0.42 | False | 0 | no |
| layout | ok | 134 | 0.42 | False | 0 | no |
| matrix-rotate | ok | 134 | 0.97 | False | 0 | no |
| pathfinder | ok | 0 | 1.29 | False | 88 | n/a |
| randomAccess | ok | 134 | 0.42 | False | 0 | n/a |

- 10/10 compile. Every run went through the sandbox command (prlimit,
  systemd-run, unshare, then the artifact and its run_args); the test
  asserts it.
- 9/10 runs end with exit 134 (SIGABRT) and empty stdout and stderr; only
  pathfinder exits 0. The two earlier runs showed the same pattern.

Probes (exploratory):

- Inside the sandbox `nproc` prints 256 and the allowed CPU list is 0-255.
  With no OMP_NUM_THREADS the OpenMP runtime starts, by its default, one
  thread per CPU; the default run and the 256-thread run below fail the
  same way.
- layout: default threads, exit 134; with NVCOMPILER_TERM=trace the runtime
  prints "Error: abort" and a register dump (exit 127). With
  OMP_NUM_THREADS 4, 64, 200, 240, 248, and 252 it exits 0 with 2 PASS
  lines each; with 256 it exits 134.
- atomicCost: at 4 threads and 8192 MB it ends with exit 137 (SIGKILL,
  consistent with the memory limit); at 16 threads and 32768 MB it exits 0
  with 48 PASS lines and no FAIL, in 66.7 s of wall time (exploratory).

## Conclusion

The proxy preset works: all 10 OpenMP references compile with
nvcpp-multicore. The runs fail for a reason outside the preset. The sandbox
caps the run's cgroup at TasksMax=256 (SandboxSpec.tasks_max), and on this
256-CPU host the OpenMP runtime starts 256 threads by default; with the
sandbox's own processes in the same scope, a thread creation fails and the
runtime aborts. At 252 threads or fewer, layout passes. pathfinder may start
fewer threads (its target region sets num_teams and thread_limit); this was
not probed.

The native executor gives no way to set the thread count: the program's
environment is the sandbox default, and SandboxSpec.environment admits only
PATH, the locale, TMPDIR, and pin variables. A fix belongs to the executor or
the sandbox, not to the preset or the test. Options, for the orchestrator or
the owner:

1. The sandbox restricts the program's CPU affinity to Limits.cpus CPUs,
   so the runtime starts that many threads and the CPU-time budget
   (wall x cpus, OQ-011) matches what the program can use.
2. The native executor sets OMP_NUM_THREADS to Limits.cpus, which needs
   the name on the sandbox's environment list.
3. The native executor passes a larger tasks_max. This keeps 256 threads,
   which spend the CPU-time budget (2880 s at 180 s x 16) in about 11 s.

Option 1 or 2 with cpus=16 matches the atomicCost probe above.
