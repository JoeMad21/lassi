"""Remote checks of bale-v0's references: each builds, runs clean, and the two languages agree exactly (task P17.40).

Bible: Benchmark Suites (bale-v0), Harness Contract (lassi_io.h; held-out
inputs), Execution Backends (the native row), Sandbox (native program runs),
Oracles (binary_io), Toolchain Pins (gcc), Agent Rules 5, 6, and 7;
AGENTS.md Local Execution. Plan: plans/p17-portable.md, P17.40.

For each item of assets/bench/bale-v0.yaml, read for eval only (Agent Rule
5), the cpp program is built with gcc-native and the omp program with
gcc-openmp, as a run builds them (lassi.core.runner build_toolchain: the
pinned host g++, checked by --version in the compile sandbox), each with its
language's tracked support file, lassi_io.h. bale's upstream C files, the
other support files, are never compiled, so they are left out and nothing is
fetched. Each program runs twice on the item's held-out inputs (staged with
lassi.bench.registry stage_inputs) through the native executor, in the
sandbox (Agent Rule 6), with an attempt run's limits when no reference wall
time is known (the 30 s floor, 16 GB, RUN_CPUS); every run must be clean and
write exactly the item's outputs. Then:

1. the cpp and omp runs' output files are byte-identical, run to run and
   language to language, and binary_io passes each under the item's
   tolerance (max_abs 0), as the baseline's agreement check would;
2. each output equals what an independent Python model of the item's
   computation gives on the same inputs (histo's counts, ig's gather,
   triangle's count by row-set intersection).

The tests are marked `remote` and skip unless the host can run them (Linux,
the sandbox tools on PATH, a reachable user systemd manager,
$LASSI_SCRATCH, $LASSI_RUNS_ROOT, and $LASSI_TOOLCHAINS set, TMPDIR inside
$LASSI_SCRATCH, and the pinned host g++ of toolchains/gcc.pin present). With
LASSI_REQUIRE_SANDBOX=1 such a host fails them instead. Run them from a
clean commit in the LASSI distro's clone with the host profile sourced
(AGENTS.md, Local Execution), or on the build host only with the owner's
approval in the working session (Agent Rule 17):

    LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote tests/bench/test_bale_remote.py

Every build and run lies in a temp directory under $LASSI_RUNS_ROOT, removed
afterwards (Agent Rule 7). No value here is a measurement.
"""

from __future__ import annotations

import collections
import os
import shutil
import struct
import sys
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from lassi.bench import Suite, load_suite, registry
from lassi.core.interfaces import Limits, RunResult
from lassi.core.runner import BuiltToolchain, build_toolchain
from lassi.core.stages import PURPOSE, RUN_CPUS, RUN_WALL_FLOOR_S
from lassi.executors.native import NativeExecutor
from lassi.harness import read_array
from lassi.oracles.binary_io import BinaryIOOracle
from lassi.toolchains import pins as pins_module

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "assets" / "bench" / "bale-v0.yaml"
ITEMS = ("histo", "ig", "triangle")
TOOLCHAINS = {"cpp": "gcc-native", "omp": "gcc-openmp"}
TOOLS = ("unshare", "systemd-run", "nice", "setpriv", "prlimit", "timeout", "python3", "awk", "env", "sh")
REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"
MEMORY_MB = 16 * 1024  # projects/base.yaml sandbox.mem_gb, the recipes' memory limit


def host_problem() -> str:
    """Return why this host cannot run the tests, or "" when it can."""
    if not sys.platform.startswith("linux"):
        return f"the sandbox needs Linux, not {sys.platform}; run it in the LASSI distro or through tools/rx.py"
    missing = [tool for tool in TOOLS if shutil.which(tool) is None]
    if missing:
        return f"not on PATH: {', '.join(missing)}"
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    if not runtime or not (Path(runtime) / "bus").exists():
        return "no user systemd manager: $XDG_RUNTIME_DIR/bus does not exist"
    for name in ("LASSI_SCRATCH", "LASSI_RUNS_ROOT", "LASSI_TOOLCHAINS", "TMPDIR"):
        if not os.environ.get(name):
            return f"{name} is not set"
    scratch, tmpdir = Path(os.environ["LASSI_SCRATCH"]).resolve(), Path(os.environ["TMPDIR"]).resolve()
    if scratch not in tmpdir.parents:
        return "TMPDIR does not lie inside $LASSI_SCRATCH"
    executable = pins_module.read_pin("gcc").get("EXECUTABLE", "")
    if not executable or not Path(executable).is_file():
        return f"the pinned host compiler {executable!r} (toolchains/gcc.pin) is not on this host"
    return ""


PROBLEM = host_problem()
pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(bool(PROBLEM) and not REQUIRE, reason=PROBLEM or "the host can run native programs"),
]


@pytest.fixture(autouse=True)
def host_ready() -> None:
    """Fail, rather than skip, when LASSI_REQUIRE_SANDBOX=1 asks for a real sandbox and the host cannot run one."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")


@pytest.fixture(scope="module")
def built() -> dict[str, BuiltToolchain]:
    """Build both toolchains as a run builds them: the pinned host g++, checked by --version in the compile sandbox."""
    root = Path(os.environ["LASSI_TOOLCHAINS"])
    return {language: build_toolchain(name, root) for language, name in TOOLCHAINS.items()}


@pytest.fixture
def base() -> Iterator[Path]:
    """Create a temp directory under $LASSI_RUNS_ROOT for the builds and runs; remove it afterwards."""
    path = Path(tempfile.mkdtemp(prefix="lassi-bale.", dir=os.environ["LASSI_RUNS_ROOT"]))
    try:
        yield path
    finally:
        shutil.rmtree(path)


def suite() -> Suite:
    """Load the bale-v0 manifest."""
    return load_suite(MANIFEST)


def tracked_support(item: str, language: str) -> dict[str, str]:
    """Return the language's tracked support files (lassi_io.h); the upstream C files are never compiled."""
    spec = suite().items[item].languages[language]
    return {
        name: (registry.TRACKED_ROOT / entry.path).read_bytes().decode("utf-8")
        for name, entry in spec.support.items()
        if entry.tracked
    }


def build_and_run(built: BuiltToolchain, item: str, language: str, base: Path, runs: int = 2) -> list[RunResult]:
    """Build the item's program in `language`, then run it `runs` times on the held-out inputs, each in a fresh copy."""
    spec = suite().items[item].languages[language]
    files = {name: (registry.TRACKED_ROOT / spec.dir / name).read_bytes().decode("ascii") for name in spec.files}
    support = tracked_support(item, language)
    workdir = base / f"{language}-build"
    workdir.mkdir(parents=True)
    result = built.toolchain.build(files, workdir, harness=support)
    errors = [entry for entry in result.diagnostics if entry.severity == "error"]
    assert result.artifact is not None and not errors, f"{item} {language} did not build: {errors}"
    found = suite().item(item, purpose=PURPOSE)
    limits = Limits(wall_s=RUN_WALL_FLOOR_S, memory_mb=MEMORY_MB, cpus=RUN_CPUS)
    results = []
    for count in range(1, runs + 1):
        place = base / f"{language}-run{count}"
        shutil.copytree(workdir, place)
        artifact = place / Path(result.artifact).relative_to(workdir)
        args = registry.stage_inputs(found, place)
        run = NativeExecutor().run(artifact, args, limits)
        tail = f"{run.stdout[-2000:]}\n{run.stderr[-2000:]}"
        assert (run.exit_code, run.hang, run.workdir_incomplete) == (0, False, False), f"{item} {language}: {tail}"
        names = sorted(read_array(path).name for path in run.output_files.values())
        assert names == sorted(found.outputs), f"{item} {language}: outputs {sorted(run.output_files)}"
        results.append(run)
    return results


def output_bytes(run: RunResult) -> dict[str, bytes]:
    """Return a run's output files as relative path -> bytes."""
    return {relative: Path(path).read_bytes() for relative, path in run.output_files.items()}


def i64(path: Path) -> list[int]:
    """Return the i64 values of one lassi_io file."""
    data = bytes(read_array(path).data)
    return list(struct.unpack(f"<{len(data) // 8}q", data))


def expected(item: str, inputs: Mapping[str, Path]) -> dict[str, list[int]]:
    """Return an independent Python model's outputs of `item` on its staged inputs, by output name."""
    if item == "histo":
        counts = [0] * i64(inputs["table_size"])[0]
        for index in i64(inputs["index"]):
            counts[index] += 1
        return {"counts": counts}
    if item == "ig":
        table = i64(inputs["table"])
        return {"tgt": [table[index] for index in i64(inputs["index"])]}
    rows: dict[int, set[int]] = collections.defaultdict(set)
    for u, v in zip(i64(inputs["edge_u"]), i64(inputs["edge_v"]), strict=True):
        if u != v:
            rows[max(u, v)].add(min(u, v))
    empty: set[int] = set()
    return {"triangles": [sum(len(rows[a] & rows.get(b, empty)) for a in rows for b in rows[a])]}


@pytest.mark.parametrize("item", ITEMS)
def test_both_references_run_clean_agree_exactly_and_match_a_python_model(
    item: str, built: dict[str, BuiltToolchain], base: Path
) -> None:
    runs = {language: build_and_run(built[language], item, language, base / language) for language in TOOLCHAINS}
    found = suite().item(item, purpose=PURPOSE)
    first = output_bytes(runs["cpp"][0])
    for language, results in runs.items():
        for count, run in enumerate(results, start=1):
            assert output_bytes(run) == first, f"{item}: {language} run {count} differs from cpp run 1"
    assert found.tolerance is not None
    oracle = BinaryIOOracle(metric=found.tolerance.metric, threshold=found.tolerance.threshold)
    stats = oracle.compare(runs["cpp"][0].output_files, runs["omp"][0].output_files, sides=("cpp", "omp"))
    assert stats and all(entry.passed for entry in stats), stats
    model_dir = base / "model"
    model_dir.mkdir()
    registry.stage_inputs(found, model_dir)
    inputs = {entry["name"]: model_dir / registry.INPUTS_DIR / f"{entry['name']}.lassiio" for entry in found.inputs}
    model = expected(item, inputs)
    for path in runs["cpp"][0].output_files.values():
        array = read_array(path)
        assert i64(path) == model[array.name], f"{item}: {array.name} differs from the Python model"
