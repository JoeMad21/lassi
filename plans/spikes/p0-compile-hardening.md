# Spike P0.20: what a sandboxed compile needs, and the compile hardening on alpha01

- Task: P0.20 (plans/p0-core.md). Bible: Sandbox, Toolchain Pins; Agent Rules 6, 7, 10, 12. Earlier evidence: plans/spikes/p0-sandbox-hardening.md (the P0.16 mechanisms and its compile handoff) and the Known limits in the lassi/executors/sandbox.py docstring.
- Date: 2026-09-23, between 18:35 and 19:44 (alpha01 clock, UTC-07:00): the first runs 18:35 to 18:42, the review-fix runs 19:29 to 19:44. Each rx id below carries its own start time.
- Base: local commit 20eb012 (`20eb0125fccc5bf27db00207282fa05755e77bf3`, branch p0-core) with the uncommitted P0.20 tree. Every run below went through `uv run tools/rx.py run`, which sent the dirty tree as a snapshot commit, so every figure here is exploratory and not [MEASURED].
- Host: alpha01. Kernel 6.6.29+main+3.0.0r1-amd64-gio-epilmore-dev+, Ubuntu 22.04.5 LTS, Python 3.10.12, 256 CPUs, about 3.1 TB of memory (rx doctor). Toolchains: toolchains/cuda.pin 12.6.3 (nvcc V12.6.85) and toolchains/nvhpc.pin 24.11 (nvc++ 24.11-0) under $LASSI_TOOLCHAINS. Device: host CPU only; no built program ran and no accelerator node was opened.
- Environment the gate set: HOME, LASSI_SCRATCH = /mnt/nvme10/joseph_ufl (HOME is the scratch root), TMPDIR = /mnt/nvme10/joseph_ufl/tmp, LASSI_RUNS_ROOT = /mnt/nvme10/joseph_ufl/lassi-runs, LASSI_TOOLCHAINS = /mnt/nvme10/joseph_ufl/toolchains (rx 20260923-183739-exec-7786).
- Sources compiled: the 14 hand-written fixture scenarios (tests/toolchains/fixtures/sources/) and the reference targets of the HeCBench layout app of lassi-hecbench-10 (cuda with nvcc-sm80, omp with nvc++ nvcpp-cc80), as tests/executors/test_sandbox_compile_remote.py compiles them. Compiling a reference to size the harness is no training, tuning, or harvest on the item (Agent Rule 5).

## Question

Which memory, CPU-time, wall, and disk limits fit a compile of HeCBench-sized sources in the P0.20 compile sandbox (lassi.executors.sandbox.SandboxedCompileRunner), with margin, and does the sandboxed compile change the compiler's stderr? Classification: factual.

## Method

The helper below (an untracked file at the repository root while it ran, removed afterwards) compiled each scenario twice in fresh workdirs under $LASSI_RUNS_ROOT/p020-measure/<rx id>/:

1. outside the sandbox, with the compile environment build_toolchain gives (PATH, LANG=C, LC_ALL=C, NVHPC_CUDA_HOME for nvc++) and a private TMPDIR under the workdir, polling the compile's session every 10 ms for its summed RSS and the TMPDIR's bytes and entries; getrusage(RUSAGE_CHILDREN) deltas give the CPU time, and its ru_maxrss the largest single-process RSS (a running maximum over the helper's life, so an upper bound for each later row);
2. through the toolchain's own build() with the sandboxed compile runner, for its wall time, outcome, and stderr, compared byte for byte with the first compile's stderr.

Command (rx 20260923-183814-desktop-8r113ei-p0-core-b8ff, snapshot 74c1a20b5628 of 20eb012):

```
uv run tools/rx.py run --timeout 2400 -- 'uv sync --quiet && LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -rfEs tests/executors/test_sandbox_compile_remote.py::test_a1_a_generated_include_of_a_hidden_absolute_path_fails; uv run python p020_measure.py'
```

The helper:

```python
"""Exploratory P0.20 measurement: what one compile needs (wall, CPU time, memory, temp space); untracked helper.

For each capture scenario and the HeCBench layout app (cuda with nvcc, omp
with nvc++) it compiles twice in fresh workdirs under
$LASSI_RUNS_ROOT/p020-measure/<run id>/:

1. outside the sandbox, with the compile environment and a private TMPDIR
   under the workdir, polling the process session every 10 ms for the sum
   of its RSS and the TMPDIR's bytes and entries; getrusage(RUSAGE_CHILDREN)
   gives the CPU time and the largest single RSS;
2. through the toolchain's build() with the sandboxed compile runner, for
   its wall time and outcome.

It prints one JSON line per compile. Hand-written fixtures and pinned
reference sources only; no built program runs.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from lassi.bench import Direction, load_suite, sources_dir  # noqa: E402
from lassi.core.runner import build_toolchain  # noqa: E402
from lassi.executors.sandbox import COMPILE_TMPDIR  # noqa: E402
from tools.capture_toolchain_fixtures import load_scenarios, scenario_registry  # noqa: E402


def tree_rss_kib(session: int) -> int:
    """Return the summed RSS in KiB of every process in `session`."""
    total = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_text()
            fields = stat[stat.rindex(")") + 2 :].split()
            if int(fields[3]) != session:
                continue
            pages = int(Path(f"/proc/{entry}/statm").read_text().split()[1])
            total += pages * (os.sysconf("SC_PAGE_SIZE") // 1024)
        except (OSError, ValueError, IndexError):
            continue
    return total


def dir_usage(path: Path) -> tuple[int, int]:
    """Return the bytes and entries under `path`."""
    size, count = 0, 0
    for root, dirs, files in os.walk(path):
        count += len(dirs) + len(files)
        for name in files:
            try:
                size += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return size, count


def measure(argv: list[str], env: dict[str, str], workdir: Path, tmpdir: Path) -> dict[str, Any]:
    """Run one compile outside the sandbox and return its resource figures."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    peaks = {"rss_kib": 0, "tmp_bytes": 0, "tmp_entries": 0}
    start = time.monotonic()
    process = subprocess.Popen(argv, cwd=workdir, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    done = threading.Event()

    def poll() -> None:
        while not done.is_set():
            peaks["rss_kib"] = max(peaks["rss_kib"], tree_rss_kib(process.pid))
            size, count = dir_usage(tmpdir)
            peaks["tmp_bytes"] = max(peaks["tmp_bytes"], size)
            peaks["tmp_entries"] = max(peaks["tmp_entries"], count)
            time.sleep(0.01)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    _out, err = process.communicate(timeout=900)
    done.set()
    thread.join()
    wall = time.monotonic() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    out_bytes, out_entries = dir_usage(workdir)
    return {
        "status": process.returncode,
        "wall_s": round(wall, 3),
        "cpu_s": round((after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime), 3),
        "max_single_rss_mib": round(after.ru_maxrss / 1024, 1),
        "peak_tree_rss_mib": round(peaks["rss_kib"] / 1024, 1),
        "peak_tmp_mib": round(peaks["tmp_bytes"] / 1048576, 2),
        "peak_tmp_entries": peaks["tmp_entries"],
        "workdir_after_mib": round(out_bytes / 1048576, 2),
        "workdir_after_entries": out_entries,
        "stderr_bytes": len(err),
        "stderr": err,
    }


def one(label: str, name: str, registry: object, files: dict[str, str], base: Path) -> None:
    """Measure one compile outside the sandbox, then build it in the sandbox; print one JSON line."""
    built = build_toolchain(name, Path(os.environ["LASSI_TOOLCHAINS"]), registry)
    toolchain = built.toolchain
    plain = base / "plain" / label
    plain.mkdir(parents=True)
    for path, text in files.items():
        (plain / path).parent.mkdir(parents=True, exist_ok=True)
        (plain / path).write_text(text, encoding="utf-8")
    suffixes = type(toolchain).SOURCE_SUFFIXES
    argv = toolchain.command(sorted(p for p in files if Path(p).suffix in suffixes))
    tmpdir = plain / COMPILE_TMPDIR
    tmpdir.mkdir()
    env = {**(built.environment or {}), "TMPDIR": str(tmpdir)}
    figures = measure(argv, env, plain, tmpdir)
    boxed = base / "sandboxed" / label
    boxed.mkdir(parents=True)
    start = time.monotonic()
    try:
        result = toolchain.build(files, boxed)
    except Exception as exc:  # noqa: BLE001
        figures.pop("stderr")
        figures.update({"label": label, "sandbox_error": f"{type(exc).__name__}: {exc}"[-3000:]})
        print(json.dumps(figures, sort_keys=True), flush=True)
        return
    sandbox_wall = time.monotonic() - start
    plain_err = figures.pop("stderr")
    boxed_err = (boxed / "compile.stderr").read_bytes()
    figures.update({
        "label": label,
        "toolchain": name,
        "sandboxed_wall_s": round(sandbox_wall, 3),
        "sandboxed_artifact": result.artifact is not None,
        "sandboxed_codes": sorted({d.code for d in result.diagnostics}),
        "sandboxed_stderr_bytes": len(boxed_err),
        "stderr_same_as_plain": boxed_err == plain_err,
        "tmpdir_left": (boxed / COMPILE_TMPDIR).exists(),
    })
    print(json.dumps(figures, sort_keys=True), flush=True)


def main() -> int:
    """Measure every scenario and the layout app both ways."""
    run_id = os.environ.get("LASSI_RX_RUN_ID") or time.strftime("%Y%m%d-%H%M%S")
    base = Path(os.environ["LASSI_RUNS_ROOT"]) / "p020-measure" / run_id
    base.mkdir(parents=True)
    print(f"base {base}", flush=True)
    for scenario in load_scenarios().values():
        one(scenario.name, scenario.toolchain, scenario_registry(scenario), dict(scenario.files), base)
    suite = load_suite(REPO / "assets" / "bench" / "lassi-hecbench-10.yaml")
    sources = sources_dir(Path(os.environ["LASSI_SCRATCH"]), suite)
    from lassi.core.registry import DEFAULT_REGISTRY

    for name, target in (("nvcc-sm80", "cuda"), ("nvcpp-cc80", "omp")):
        direction = Direction(source="omp" if target == "cuda" else "cuda", target=target)
        files = suite.reference_target("layout", direction, sources, purpose="eval")
        one(f"layout-{target}", name, DEFAULT_REGISTRY, files, base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## Output

```
[rx] dirty tree sent as snapshot 74c1a20b5628 (exploratory; not reportable)
6 passed in 3.10s
base /mnt/nvme10/joseph_ufl/lassi-runs/p020-measure/20260923-183814-desktop-8r113ei-p0-core-b8ff
{"cpu_s": 0.638, "label": "nvcc_clean", "max_single_rss_mib": 64.0, "peak_tmp_entries": 14, "peak_tmp_mib": 2.85, "peak_tree_rss_mib": 2.0, "sandboxed_artifact": true, "sandboxed_codes": [], "sandboxed_stderr_bytes": 0, "sandboxed_wall_s": 0.973, "status": 0, "stderr_bytes": 0, "stderr_same_as_plain": true, "tmpdir_left": false, "toolchain": "nvcc-sm80", "wall_s": 0.68, "workdir_after_entries": 3, "workdir_after_mib": 0.96}
{"cpu_s": 0.003, "label": "nvcc_fatal", "max_single_rss_mib": 64.0, "peak_tmp_entries": 0, "peak_tmp_mib": 0.0, "peak_tree_rss_mib": 0.0, "sandboxed_artifact": false, "sandboxed_codes": [null], "sandboxed_stderr_bytes": 74, "sandboxed_wall_s": 0.372, "status": 1, "stderr_bytes": 74, "stderr_same_as_plain": true, "tmpdir_left": false, "toolchain": "nvcc-sm80", "wall_s": 0.166, "workdir_after_entries": 2, "workdir_after_mib": 0.0}
{"cpu_s": 0.895, "label": "nvcc_host_gcc_warning", "max_single_rss_mib": 184.0, "peak_tmp_entries": 18, "peak_tmp_mib": 3.71, "peak_tree_rss_mib": 186.0, "sandboxed_artifact": true, "sandboxed_codes": ["-Wsign-compare"], "sandboxed_stderr_bytes": 295, "sandboxed_wall_s": 1.225, "status": 0, "stderr_bytes": 295, "stderr_same_as_plain": true, "tmpdir_left": false, "toolchain": "nvcc-sm80", "wall_s": 1.009, "workdir_after_entries": 3, "workdir_after_mib": 0.96}
{"cpu_s": 0.611, "label": "nvcc_linker_error", "max_single_rss_mib": 184.0, "peak_tmp_entries": 14, "peak_tmp_mib": 2.84, "peak_tree_rss_mib": 2.0, "sandboxed_artifact": false, "sandboxed_codes": [null], "sandboxed_stderr_bytes": 354, "sandboxed_wall_s": 1.024, "status": 1, "stderr_bytes": 350, "stderr_same_as_plain": false, "tmpdir_left": false, "toolchain": "nvcc-sm80", "wall_s": 0.659, "workdir_after_entries": 2, "workdir_after_mib": 0.0}
{"cpu_s": 0.41, "label": "nvcc_ptxas_error", "max_single_rss_mib": 184.0, "peak_tmp_entries": 8, "peak_tmp_mib": 2.83, "peak_tree_rss_mib": 2.0, "sandboxed_artifact": false, "sandboxed_codes": [null], "sandboxed_stderr_bytes": 104, "sandboxed_wall_s": 0.775, "status": 255, "stderr_bytes": 104, "stderr_same_as_plain": true, "tmpdir_left": false, "toolchain": "nvcc-sm80", "wall_s": 0.489, "workdir_after_entries": 2, "workdir_after_mib": 0.0}
{"cpu_s": 0.174, "label": "nvcc_undefined_identifier", "max_single_rss_mib": 184.0, "peak_tmp_entries": 4, "peak_tmp_mib": 0.93, "peak_tree_rss_mib": 2.0, "sandboxed_artifact": false, "sandboxed_codes": [null], "sandboxed_stderr_bytes": 183, "sandboxed_wall_s": 0.522, "status": 2, "stderr_bytes": 183, "stderr_same_as_plain": true, "tmpdir_left": false, "toolchain": "nvcc-sm80", "wall_s": 0.327, "workdir_after_entries": 2, "workdir_after_mib": 0.0}
{"cpu_s": 0.652, "label": "nvcc_warning_177", "max_single_rss_mib": 184.0, "peak_tmp_entries": 14, "peak_tmp_mib": 2.85, "peak_tree_rss_mib": 2.0, "sandboxed_artifact": true, "sandboxed_codes": ["177-D"], "sandboxed_stderr_bytes": 211, "sandboxed_wall_s": 0.924, "status": 0, "stderr_bytes": 211, "stderr_same_as_plain": true, "tmpdir_left": false, "toolchain": "nvcc-sm80", "wall_s": 0.685, "workdir_after_entries": 5, "workdir_after_mib": 0.96}
{"cpu_s": 0.087, "label": "nvcpp_backend_error", "max_single_rss_mib": 184.0, "peak_tmp_entries": 1, "peak_tmp_mib": 0.0, "peak_tree_rss_mib": 0.0, "sandboxed_artifact": false, "sandboxed_codes": ["S-1101"], "sandboxed_stderr_bytes": 710, "sandboxed_wall_s": 0.421, "status": 2, "stderr_bytes": 710, "stderr_same_as_plain": true, "tmpdir_left": false, "toolchain": "nvcpp-cc80", "wall_s": 0.168, "workdir_after_entries": 3, "workdir_after_mib": 0.0}
{"cpu_s": 0.064, "label": "nvcpp_edg_error", "max_single_rss_mib": 184.0, "peak_tmp_entries": 0, "peak_tmp_mib": 0.0, "peak_tree_rss_mib": 0.0, "sandboxed_artifact": false, "sandboxed_codes": [null], "sandboxed_stderr_bytes": 192, "sandboxed_wall_s": 0.472, "status": 2, "stderr_bytes": 192, "stderr_same_as_plain": true, "tmpdir_left": false, "toolchain": "nvcpp-cc80", "wall_s": 0.173, "workdir_after_entries": 2, "workdir_after_mib": 0.0}
{"cpu_s": 0.317, "label": "nvcpp_edg_warning", "max_single_rss_mib": 184.0, "peak_tmp_entries": 6, "peak_tmp_mib": 0.07, "peak_tree_rss_mib": 6.0, "sandboxed_artifact": true, "sandboxed_codes": ["declared_but_not_referenced"], "sandboxed_stderr_bytes": 717, "sandboxed_wall_s": 0.723, "status": 0, "stderr_bytes": 717, "stderr_same_as_plain": true, "tmpdir_left": false, "toolchain": "nvcpp-cc80", "wall_s": 0.332, "workdir_after_entries": 4, "workdir_after_mib": 0.04}
{"cpu_s": 0.06, "label": "nvcpp_fatal_abort", "max_single_rss_mib": 184.0, "peak_tmp_entries": 0, "peak_tmp_mib": 0.0, "peak_tree_rss_mib": 0.0, "sandboxed_artifact": false, "sandboxed_codes": ["F-0000"], "sandboxed_stderr_bytes": 177, "sandboxed_wall_s": 0.428, "status": 2, "stderr_bytes": 177, "stderr_same_as_plain": true, "tmpdir_left": false, "toolchain": "nvcpp-cc80", "wall_s": 0.166, "workdir_after_entries": 2, "workdir_after_mib": 0.0}
{"cpu_s": 0.307, "label": "nvcpp_linker_error", "max_single_rss_mib": 184.0, "peak_tmp_entries": 6, "peak_tmp_mib": 0.07, "peak_tree_rss_mib": 6.0, "sandboxed_artifact": false, "sandboxed_codes": [null], "sandboxed_stderr_bytes": 899, "sandboxed_wall_s": 0.674, "status": 2, "stderr_bytes": 894, "stderr_same_as_plain": false, "tmpdir_left": false, "toolchain": "nvcpp-cc80", "wall_s": 0.327, "workdir_after_entries": 3, "workdir_after_mib": 0.0}
{"cpu_s": 0.307, "label": "nvcpp_minfo_clean", "max_single_rss_mib": 184.0, "peak_tmp_entries": 6, "peak_tmp_mib": 0.07, "peak_tree_rss_mib": 6.0, "sandboxed_artifact": true, "sandboxed_codes": [], "sandboxed_stderr_bytes": 485, "sandboxed_wall_s": 0.672, "status": 0, "stderr_bytes": 485, "stderr_same_as_plain": true, "tmpdir_left": false, "toolchain": "nvcpp-cc80", "wall_s": 0.33, "workdir_after_entries": 4, "workdir_after_mib": 0.04}
{"cpu_s": 0.067, "label": "nvcpp_missing_include", "max_single_rss_mib": 184.0, "peak_tmp_entries": 0, "peak_tmp_mib": 0.0, "peak_tree_rss_mib": 0.0, "sandboxed_artifact": false, "sandboxed_codes": [null], "sandboxed_stderr_bytes": 230, "sandboxed_wall_s": 0.472, "status": 2, "stderr_bytes": 230, "stderr_same_as_plain": true, "tmpdir_left": false, "toolchain": "nvcpp-cc80", "wall_s": 0.177, "workdir_after_entries": 2, "workdir_after_mib": 0.0}
{"cpu_s": 1.404, "label": "layout-cuda", "max_single_rss_mib": 214.0, "peak_tmp_entries": 18, "peak_tmp_mib": 5.45, "peak_tree_rss_mib": 188.0, "sandboxed_artifact": true, "sandboxed_codes": [], "sandboxed_stderr_bytes": 0, "sandboxed_wall_s": 1.73, "status": 0, "stderr_bytes": 0, "stderr_same_as_plain": true, "tmpdir_left": false, "toolchain": "nvcc-sm80", "wall_s": 1.499, "workdir_after_entries": 3, "workdir_after_mib": 0.97}
{"cpu_s": 0.844, "label": "layout-omp", "max_single_rss_mib": 216.0, "peak_tmp_entries": 10, "peak_tmp_mib": 0.4, "peak_tree_rss_mib": 90.0, "sandboxed_artifact": true, "sandboxed_codes": [], "sandboxed_stderr_bytes": 3210, "sandboxed_wall_s": 1.124, "status": 0, "stderr_bytes": 3210, "stderr_same_as_plain": true, "tmpdir_left": false, "toolchain": "nvcpp-cc80", "wall_s": 0.848, "workdir_after_entries": 4, "workdir_after_mib": 0.06}
[rx] id=20260923-183814-desktop-8r113ei-p0-core-b8ff rc=0 state=done
```

## Findings (exploratory)

- Every compile needed at most 1.404 s of CPU time (layout-cuda), 1.499 s of wall time outside the sandbox and 1.73 s inside it, 216 MiB of RSS in its largest process, 188 MiB of sampled RSS for its whole process tree, 5.45 MiB and 18 entries in its TMPDIR, and left under 1 MiB of outputs in its workdir. The 10 ms sampling misses short-lived processes, so the tree figure is a lower bound; the cgroup's own memory.peak (tmpfs and page cache included) was not read.
- The sandbox added about 0.2 to 0.4 s per compile (the sandboxed wall minus the plain wall).
- Stderr inside the sandbox equals stderr outside it byte for byte for every scenario except nvcc_linker_error and nvcpp_linker_error, whose ld lines name the per-compile TMPDIR, as the fixtures README says they would.
- The private TMPDIR was gone after every sandboxed compile (tmpdir_left false).

## Decision (recorded in lassi/executors/sandbox.py)

- Wall: the toolchain's timeout (600 s by default), about 350 times the slowest compile here.
- Memory: COMPILE_MEMORY_MB = 8192 MiB, about 38 times the largest process, and at least twice the disk cap (here four times), so workdir_cap_bytes never halves it; a full disk cap still leaves 6 GiB for the compiler itself.
- Disk: COMPILE_DISK_MB = 2048 MiB for the build dir's overlay (TMPDIR included) and the file size limit, over 300 times the largest TMPDIR here.
- CPU time: COMPILE_CPUS = 2, so each process may use 1200 s of CPU at the default timeout: room for a compiler that runs two threads for the whole wall limit, where every compile here used under 1.5 s.
- Tasks: the SandboxSpec default of 256.

## Remote tests and capture (same tree, exploratory)

- rx 20260923-183540-desktop-8r113ei-p0-core-00bc (snapshot 6c422989167a): `LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote -rfEs`: 58 passed, 2 failed. Both failures were the `home` case of test_a1_a_generated_include_of_a_hidden_absolute_path_fails, which refused to run because HOME contains the scratch root; on alpha01 they are one directory. The test now accepts that layout, since a file directly under HOME then lies under both hidden roots.
- rx 20260923-183814-desktop-8r113ei-p0-core-b8ff (above): the six include tests, home included: 6 passed.
- rx 20260923-184109-desktop-8r113ei-p0-core-8573 (snapshot 0f57d2bf3f5d of 20eb012: the P0.20 tree without the helper; after it only comments in lassi/executors/sandbox.py and this file changed): `LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote -rfEs`: 60 passed, none skipped.
- rx 20260923-183922-desktop-8r113ei-p0-core-372b (snapshot 7dd1044442aa of 20eb012): `uv run python tools/capture_toolchain_fixtures.py`, pulled with `rx pull --path lassi-runs/fixture-captures/<rx id>` and compared locally with tests/toolchains/fixtures/: the 12 byte-stable scenarios (all but the two linker errors) are byte-identical to their fixtures and to the stderr_sha256 in captures.json; every scenario kept its recorded exit status and diagnostic count. The manifest records the compile environment names LANG, LC_ALL, PATH (and NVHPC_CUDA_HOME for nvc++), no HOME, and the --version banners holding each pin's EXPECT_VERSION. The two linker errors now name `<workdir>/@lassi-tmp/...` where the fixtures name /mnt/nvme10/joseph_ufl/tmp.

## Review fixes (same day, exploratory)

A review of the P0.20 tree found that compiler output was the one channel past every sandbox limit: the host runner (outside the run's MemoryMax) held all of a compile's stdout and stderr, build() wrote all of stderr to compile.stderr, and a generated source can make a compiler print a diagnostic per macro expansion until the wall limit. It also found that the toolchains root was exposed resolved while argv[0] and NVHPC_CUDA_HOME kept the unresolved path, that the run's own runs root was hidden only through $LASSI_RUNS_ROOT or $LASSI_SCRATCH, and that a compile layout the sandbox refuses surfaced as a ValueError at the first compile. The fixes, in lassi/executors/sandbox.py, lassi/toolchains/_base.py, lassi/core/runner.py, and tools/capture_toolchain_fixtures.py:

- Compile output cap [DESIGN]: COMPILE_OUTPUT_CAP_BYTES = 64 MiB per stream (lassi.toolchains CappedRunner, the compile runner's default), over 20000 times the largest compiler stderr above (3210 bytes, layout-omp). A stream past it keeps its head and its last OUTPUT_TAIL_BYTES bytes, and a "lassi-sandbox:" line at the end of stderr says it was cut; so do a compile killed before its wall limit (memory or CPU-time limit) and one whose whole sandbox was killed. The cap departs from the earlier rule that compilers keep all of their output; the bible's Sandbox text and a Decision Log entry record the change with P0.20 (no owner decision is needed: it is a design value, not an Agent Rule).
- The toolchains root is resolved once; the executable, each linked prefix, and the sandbox's toolchains root come from it, and an executable or prefix that resolves outside it is refused.
- The --version check runs through the toolchain's own SandboxedCompileRunner (the builds' executable, environment, and view, under prlimit --core=1) in a fresh directory under TMPDIR, so the compiler is proven reachable in the compile view before the first build; BuiltToolchain.version_status carries the observed status, which the capture manifest records.
- The run's runs root is always a hidden root, and the compile layout is checked before the run directory exists (RunError).
- The capture tool checks each compiler's --version once per toolchain and once per scenario with an override (3 checks for the 14 scenarios, not 30).

Runs (tree of 20eb012 plus the uncommitted P0.20 fixes):

- rx 20260923-192933-desktop-8r113ei-p0-core-f332 (snapshot 2cfd8b62d458): `LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote -rfEs`: 63 passed (the 60 above plus the version check in the sandbox, a toolchains root behind a symbolic link, and a 72 MiB stderr flood cut at the cap). Then `uv run pytest -q -m "not remote"` over tests/core/test_runner.py, tests/executors/test_sandbox_compile.py, tests/executors/test_sandbox.py, tests/tools/test_capture_toolchain_fixtures.py, and tests/toolchains/test_runner_caps.py on alpha01: 680 passed, 1 failed. The failure is test_copy_back_adds_owner_read_write_and_removes_group_and_other_write (P0.16; this task fixes the test): it runs COPY_BACK_PROGRAM as the plain host user, who cannot read the test's mode-000 file, where the sandbox runs it as uid 0 of its user namespace. /mnt/nvme10 had 392G free.
- rx 20260923-193217-desktop-8r113ei-p0-core-e449 (snapshot 2b0b1abc38dd): `uv run python tools/capture_toolchain_fixtures.py`, compared on the host with tests/toolchains/fixtures/ and captures.json: the 12 byte-stable scenarios are byte-identical to their fixtures and to their recorded stderr_sha256; all 14 kept their recorded exit status, diagnostic count, and argv; both toolchains record version_exit_status 0 and the environment names LANG, LC_ALL, PATH (and NVHPC_CUDA_HOME for nvc++).

- rx 20260923-194357-desktop-8r113ei-p0-core-e935 (dirty snapshot 13bc9b91f394): `uv run pytest -q tests/executors/test_sandbox.py -k copy_back -rs`: 28 passed, none skipped. test_copy_back_adds_owner_read_write_and_removes_group_and_other_write had failed on alpha01 because the unprivileged test process could not read the mode-0 file it made; it now runs the copy-back under unshare -r, as root of a user namespace, which is how the sandbox runs it.

## Not measured

- The cgroup's peak memory (tmpfs pages and page cache included) and the CPU time of the sandbox setup itself.
- A compile larger than the layout app; the margins above are wide, but no HeCBench app beyond layout is fetched yet.
- A clean-commit rerun: the orchestrator reruns the capture from the task's commit (A6).
