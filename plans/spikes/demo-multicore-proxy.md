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

## DEMO.2 rerun with the thread bound (2026-09-24)

- Task: DEMO.2, demo preparation the owner asked for (not a P1 plan task):
  the native executor bounds a program's OpenMP threads by Limits.cpus
  (option 2 above). Every native run now gets exactly PATH=SANDBOX_PATH,
  LANG=C.UTF-8, TMPDIR=/tmp, and OMP_NUM_THREADS=<Limits.cpus> through
  SandboxSpec.environment; OMP_NUM_THREADS joins the sandbox's
  ENVIRONMENT_NAMES. The one variable that changes: a native run has no
  HOME (the sandbox default set HOME=<workdir>), since
  SandboxSpec.environment never holds HOME. TasksMax stays 256.
- Date: 2026-09-24 (EDT). The rx ids embed the build host's clock
  (UTC-07:00), so they read 2026-09-24 01:36 to 01:42.
- Where: branch `p1-faithful` at f01237b with this task's changes (and the
  uncommitted work of the tasks running beside it) sent by rx as dirty
  snapshots; `rx run` on alpha01, no GPU used. Toolchain and sources as
  above (NVHPC 24.11 per toolchains/nvhpc.pin; lassi-hecbench-10 at
  HeCBench 692cba3).
- Evidence label: exploratory. Every rx run below went from a dirty tree, so
  nothing here is [MEASURED]; the rerun from a clean commit is the
  evidence. Wall seconds are exploratory and never performance numbers;
  kernel times the programs print are left out.

### Method

1. `uv run tools/rx.py doctor` (ok, no running jobs) and
   `uv run tools/rx.py exec -- 'du -sh /mnt/nvme10/joseph_ufl'` (94G,
   rx 20260924-013655-exec-3f3c).
2. The remote executor tests, which cover the program environment (the
   sandbox default through Sandbox.run, and the native executor's bounded
   environment through tests/executors/test_native_threads_remote.py):
   `uv run tools/rx.py run --timeout 1800 --tail 60 -- 'LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -rs -m remote tests/executors/; echo "pytest_status=$?"'`
   (rx 20260924-013723-desktop-8r113ei-p1-faithful-8628, snapshot
   d065e7a23062): `62 passed, 471 deselected`, pytest status 0. That
   includes test_the_program_sees_only_the_allowed_environment
   (Sandbox.run with no environment: HOME, LANG, PATH, TMPDIR, unchanged),
   test_native_executor_runs_a_shell_script_artifact, and
   test_a_native_run_sees_exactly_the_default_environment_plus_the_thread_bound
   (a copy of the system env as the artifact, cpus 3, with a secret and
   OMP_NUM_THREADS=999 in the caller: it saw exactly PATH, LANG, TMPDIR,
   and OMP_NUM_THREADS=3).
3. The proxy test:
   `uv run tools/rx.py run --timeout 3000 --tail 100 -- 'LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -s -m remote tests/bench/test_multicore_proxy.py; echo "pytest_status=$?"'`
   (rx 20260924-013910-desktop-8r113ei-p1-faithful-d1ab, snapshot
   819e58ffe27c): `1 passed`, pytest status 0. The test now also asserts
   that each sandboxed command carries OMP_NUM_THREADS=16 before the
   artifact.
4. A probe for dense-embedding, a Python script sent inline through
   `rx run` (base64, written to $TMPDIR, removed after; rx
   20260924-014208-desktop-8r113ei-p1-faithful-3ba7, snapshot
   edabc5d49acc). It compiles, with build_toolchain("nvcpp-multicore"), a
   short C++ program of its own that runs one
   `target teams num_teams(<n>) thread_limit(256)` region with a nested
   `parallel` and prints omp_get_num_teams(), the team ids seen, and the
   threads in team 0; and it compiles the dense-embedding OpenMP reference
   and prints its stdout without the timing lines. Both run through the
   registered "native" executor (wall 60 s, memory 8192 MB, cpus 16), so
   every run stays in the sandbox (Agent Rule 6). The script, the command,
   and the verbatim output are under Probe record below.

### Results

The proxy test's report (rx 20260924-013910-desktop-8r113ei-p1-faithful-d1ab;
wall 180 s, memory 32768 MB, cpus 16, so OMP_NUM_THREADS=16; wall_s
exploratory):

| App | Compile | Exit | wall_s | Timeout | stdout bytes | PASS seen |
| --- | --- | --- | --- | --- | --- | --- |
| atomicCost | ok | 0 | 66.66 | False | 8578 | yes |
| bsearch | ok | 0 | 0.37 | False | 204 | n/a |
| colorwheel | ok | 0 | 0.37 | False | 79 | yes |
| dense-embedding | ok | 0 | 3.14 | False | 1094 | no |
| entropy | ok | 0 | 2.38 | False | 114 | yes |
| jacobi | ok | 0 | 0.42 | False | 121 | yes |
| layout | ok | 0 | 0.52 | False | 110 | yes |
| matrix-rotate | ok | 0 | 1.07 | False | 49 | yes |
| pathfinder | ok | 0 | 0.52 | False | 88 | n/a |
| randomAccess | ok | 0 | 5.20 | False | 185 | n/a |

- 10/10 compile and 10/10 runs exit 0 (DEMO.1: 1/10). No run hit its wall
  limit.
- Of the 7 apps whose manifest lists passfail for omp, 6 print PASS
  (atomicCost, colorwheel, entropy, jacobi, layout, matrix-rotate).
  "PASS seen" is the test's check: PASS appears somewhere in stdout.
  dense-embedding exits 0 but prints no PASS.

The probe (rx 20260924-014208-desktop-8r113ei-p1-faithful-3ba7; its
verbatim output is under Probe record below):

- The teams program, asked for 1, 8, and 64 teams, reports each time
  omp_get_num_teams()=1, one team id seen (0), and 16 threads in team 0.
  So nvc++ -mp=multicore runs a `target teams` region as a single team of
  OMP_NUM_THREADS threads, whatever num_teams asks for.
- dense-embedding (run_args 10000 8 1) prints the table size, batch size
  8, and for each embedding dimension 64, 128, 256, 512, 1024, and 2048 a
  FAIL line; exit 0.

### Diagnosis

dense-embedding's OpenMP reference (main.cpp at HeCBench 692cba3) runs
`target teams num_teams(batch_size)` and has each team handle the batch
whose index is omp_get_team_num(); the threads of a team split the
embedding columns by omp_get_num_threads(). Under the proxy only team 0
exists, so only batch 0 of the output is written, the rest stays 0, and the
check against the host reference fails for every dimension. OpenMP lets an
implementation create fewer teams than num_teams requests, so the program
depends on a GPU-style league, not on the thread count: the thread bound is
not the cause (the thread split adapts to any count), and TasksMax is not
involved (the run exits 0). This is a property of the multicore proxy for
this one reference, found inside the sandbox; nothing ran outside it. It
was not changed here: the reference is pinned benchmark source, and the
proxy preset only builds it.

### Conclusion

With OpenMP threads bounded by Limits.cpus, all 10 OpenMP references
compile with nvcpp-multicore and exit 0 in the sandbox on alpha01, and 6
of the 7 that print PASS or FAIL print PASS. dense-embedding prints FAIL
under the proxy because the proxy runs its `target teams` region as one
team. For the demo, dense-embedding's proxy result is not a correctness
signal: a translation that keeps the team-indexed batching prints FAIL
under the proxy as the reference does.

### Probe record

The probe's script, command, and output, recovered on 2026-09-24 from the
gate's record of rx 20260924-014208-desktop-8r113ei-p1-faithful-3ba7:
`uv run tools/rx.py pull 20260924-014208-desktop-8r113ei-p1-faithful-3ba7`
fetches it, meta.json holds the command, and output.log the output. The
record says: commit edabc5d49acc (a snapshot of f01237b), dirty true, rc 0,
start 2026-09-24T01:42:08-07:00, end 2026-09-24T01:42:15-07:00, timeout_s
900. Exploratory, as above.

The command, as meta.json records it, with its base64 text elided (it
decodes to the script below, sha256
157a2a8071f439caa153143469622668ace153671d79d29d438335a354caca2d):

```sh
echo <base64 of the script below> | base64 -d > "$TMPDIR/demo2_probe.py" && uv run python "$TMPDIR/demo2_probe.py"; st=$?; rm -f "$TMPDIR/demo2_probe.py"; echo probe_status=$st
```

To re-run it: base64-encode the script below into that command and pass it
to `uv run tools/rx.py run --timeout 900 -- '<command>'`; a run from a
dirty tree stays exploratory.

The script (demo2_probe.py), verbatim; the C++ teams program is its TEAMS
string:

```python
"""DEMO.2 probe: how many teams nvc++ -mp=multicore creates, and what dense-embedding's main prints.

Every program runs through the registered native executor, in the sandbox. Exploratory.
"""

import os
import re
import shutil
import tempfile
from pathlib import Path

from lassi.bench import Direction, load_suite, sources_dir
from lassi.core.interfaces import Limits
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.core.runner import build_toolchain

TEAMS = r"""
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  int want = argc > 1 ? atoi(argv[1]) : 8;
  int seen[1024] = {0};
  int nteams = -1, nthreads = -1, maxteam = -1;
  #pragma omp target teams num_teams(want) thread_limit(256) map(tofrom: seen[0:1024], nteams, nthreads, maxteam)
  {
    #pragma omp parallel
    {
      int t = omp_get_team_num();
      if (omp_get_thread_num() == 0 && t < 1024) seen[t] = 1;
      if (t == 0 && omp_get_thread_num() == 0) { nteams = omp_get_num_teams(); nthreads = omp_get_num_threads(); }
    }
  }
  int count = 0;
  for (int i = 0; i < 1024; i++) { if (seen[i]) { count++; maxteam = i; } }
  printf("requested=%d omp_get_num_teams=%d team_ids_seen=%d max_team_id=%d threads_in_team0=%d\n",
         want, nteams, count, maxteam, nthreads);
  return 0;
}
"""

LIMITS = Limits(wall_s=60.0, memory_mb=8192, cpus=16)
MANIFEST = Path("assets/bench/lassi-hecbench-10.yaml")
TIMING = re.compile(r"^Average execution time of dense embedding kernel")


def main() -> None:
    built = build_toolchain("nvcpp-multicore", Path(os.environ["LASSI_TOOLCHAINS"]))
    executor = DEFAULT_REGISTRY.get("Executor", "native").factory()
    base = Path(tempfile.mkdtemp(prefix="lassi-demo2-probe.", dir=os.environ["LASSI_RUNS_ROOT"]))
    try:
        work = base / "teams" / "build"
        work.mkdir(parents=True)
        result = built.toolchain.build({"main.cpp": TEAMS}, work)
        assert result.artifact is not None, result.diagnostics
        for want in ("1", "8", "64"):
            run = executor.run(result.artifact, [want], LIMITS)
            print(f"teams probe want={want} exit={run.exit_code} stdout={run.stdout.strip()!r}")
        suite = load_suite(MANIFEST)
        root = sources_dir(Path(os.environ["LASSI_SCRATCH"]), suite)
        direction = Direction(source="cuda", target="omp")
        app = "dense-embedding"
        work = base / "dense" / "build"
        work.mkdir(parents=True)
        files = suite.reference_target(app, direction, root, purpose="eval")
        harness = suite.support_files(app, root, purpose="eval")
        result = built.toolchain.build(files, work, harness=harness)
        assert result.artifact is not None, result.diagnostics
        run = executor.run(result.artifact, list(suite.items[app].run_args), LIMITS)
        kept = [line for line in run.stdout.splitlines() if not TIMING.match(line)]
        print(f"dense-embedding exit={run.exit_code} (timing lines left out)")
        for line in kept:
            print(f"  | {line}")
    finally:
        shutil.rmtree(base)


if __name__ == "__main__":
    main()
```

output.log, verbatim (the script itself leaves out dense-embedding's timing
lines, the ones its TIMING pattern matches, and prints the rest):

```text
teams probe want=1 exit=0 stdout='requested=1 omp_get_num_teams=1 team_ids_seen=1 max_team_id=0 threads_in_team0=16'
teams probe want=8 exit=0 stdout='requested=8 omp_get_num_teams=1 team_ids_seen=1 max_team_id=0 threads_in_team0=16'
teams probe want=64 exit=0 stdout='requested=64 omp_get_num_teams=1 team_ids_seen=1 max_team_id=0 threads_in_team0=16'
dense-embedding exit=0 (timing lines left out)
  | Number of rows in the embedding table: 10000
  | Batch size: 8
  | 
  | Embedding dimension: 64
  | FAIL
  | 
  | Embedding dimension: 128
  | FAIL
  | 
  | Embedding dimension: 256
  | FAIL
  | 
  | Embedding dimension: 512
  | FAIL
  | 
  | Embedding dimension: 1024
  | FAIL
  | 
  | Embedding dimension: 2048
  | FAIL
probe_status=0
```
