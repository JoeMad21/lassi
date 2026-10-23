"""Remote witness checks of the seeded set adv-seeded-v0, which run no adversary (task P16.10).

Bible: Sandbox (native program runs; attempt run limits), Execution Backends
(the native row), Harness Contract (lassi_io.h; held-out inputs), Oracles
(binary_io), Toolchain Pins (gcc), Adversarial Setups (item 1), Agent Rules
5, 6, and 7; AGENTS.md Local Execution. Plan: plans/p16-adversarial.md,
P16.10, the first acceptance bullet (the witness checks), and its planning
decision "Freeze (P16.10)": a candidate is fixed only on these checks, which
run no adversary, and is re-hashed before the freeze record is written.

The suite adv-seeded-v0 (assets/bench/adv-seeded-v0.yaml) is tracked, so
nothing is fetched: its six C++ references and the twelve seeded candidates
(tests/fixtures/adversary/seeded/index.json) are read from this repository,
the items with the stages' purpose (eval), as unassigned items allow; nothing
here trains or tunes on them (Agent Rule 5). Every program is built with
gcc-native as a run builds it (lassi.core.runner build_toolchain: the pinned
host g++, checked by --version in the compile sandbox), with the item's cpp
support files (lassi_io.h) as the harness, once per program. Each run gets a
fresh workdir holding the program's files and a copy of the built program at
its place in its build directory, as the adversary stage lays one out, so the
native executor lists only that run's output files. Inputs are staged as the
stages stage them: an item's held-out inputs with lassi.bench.registry
stage_inputs (drawn with its seed), and a witness as the adversary stage
takes a proposal (read with lassi.agents.llm parse_reply from the index's
adversary-v0 input form, as tests/bench/test_adv_seeded.py reads it, staged
and checked with check_inputs against the item's contract, then written with
write_inputs under @inputs); the arguments are program_args. Every run has an
attempt run's limits under the gate recipe
projects/adversary-seeded/seeded-fuzzer.yaml when no reference wall time is
known (lassi.core.stages attempt_limits): the wall floor, the recipe's
sandbox.mem_gb, and RUN_CPUS.

Each test takes one item or one candidate id (parametrized, so `-k <id>`
runs one):

1. test_a_reference_runs_clean_and_repeats_its_outputs (per item): the
   reference builds; on its held-out inputs and on the witness of every
   candidate of its item it runs clean (exit status 0, no hang, a complete
   workdir) and writes exactly the item's declared outputs, which binary_io
   can serve as a reference; two runs on each input set write byte-identical
   output files.
2. test_a_seeded_candidate_passes_held_out_and_fails_its_witness (per
   candidate): the candidate builds; on the item's held-out inputs it runs
   clean and every output passes binary_io against the reference's under the
   item's declared tolerance (so the adversary stage takes its trial); on its
   own witness at least one output fails binary_io against the reference's
   outputs on that witness.
3. test_a_seeded_candidates_file_hashes_to_its_index_entry (per candidate):
   the sha256 the index records for the candidate, which is that of its
   completion (the README: the candidate file as a FILE block,
   lassi.core.files render_file_blocks), matches the file this module builds.

Every program runs only through the native executor, in the sandbox (Agent
Rule 6); no adversary runs and no model is called or loaded. The tests are
marked `remote` and skip unless the host can run them (Linux, the sandbox
tools on PATH, a reachable user systemd manager, $LASSI_SCRATCH,
$LASSI_RUNS_ROOT, and $LASSI_TOOLCHAINS set, TMPDIR inside $LASSI_SCRATCH,
and the pinned host g++ of toolchains/gcc.pin present). With
LASSI_REQUIRE_SANDBOX=1 such a host fails them instead, so a silent skip
never passes for evidence. They load the manifest and the index as they are,
so a problem in either fails them, never skips them. Run one candidate from
the task's clean commit in the LASSI distro's clone with the host profile
sourced (AGENTS.md, Local Execution), or on the build host, only with the
owner's approval in the working session (Agent Rule 17), with
`uv run tools/rx.py run -- '<command>'`, the command being

    LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote tests/bench/test_adv_seeded_remote.py -k special-gt

and all of them by leaving out `-k special-gt`. Every build and run lies in a
temp directory under $LASSI_RUNS_ROOT, removed afterwards (Agent Rule 7).
No value here is a measurement.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from lassi import agents  # noqa: F401  (registers the Agents the gate recipe binds)
from lassi.agents.llm import parse_reply
from lassi.bench import Direction, Suite, load_suite, registry
from lassi.core.adversary_stage import staged_inputs
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import Limits, RunResult
from lassi.core.recipe import load_recipe
from lassi.core.runner import BuiltToolchain, build_toolchain
from lassi.core.stages import BASELINE_X10, PURPOSE, RUN_CPUS, RUN_WALL_FLOOR_S
from lassi.executors.native import NativeExecutor
from lassi.harness import InputContract, InputContractError, LassiArray, check_inputs, read_array, write_inputs
from lassi.oracles.binary_io import BinaryIOOracle
from lassi.toolchains import pins as pins_module

REPO = Path(__file__).resolve().parents[2]
SUITE = "adv-seeded-v0"
MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
SEEDED = REPO / "tests" / "fixtures" / "adversary" / "seeded"
RECIPE = REPO / "projects" / "adversary-seeded" / "seeded-fuzzer.yaml"
ITEMS = ("pair_max", "piecewise_gate", "piecewise_ops", "prefix_sum", "scale_i64", "signed_root")
TOOLS = ("unshare", "systemd-run", "nice", "setpriv", "prlimit", "timeout", "python3", "awk", "env", "sh")
REQUIRE = os.environ.get("LASSI_REQUIRE_SANDBOX") == "1"
CPP = Direction("cpp", "cpp")
HELD_OUT = "held-out"


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


def index_entries() -> list[dict[str, Any]]:
    """Return the seeded set's candidate entries (index.json), in recording order."""
    return json.loads((SEEDED / "index.json").read_bytes().decode("ascii"))["candidates"]


PROBLEM = host_problem()
CANDIDATES = {entry["id"]: entry for entry in index_entries()}
pytestmark = [
    pytest.mark.remote,
    pytest.mark.skipif(bool(PROBLEM) and not REQUIRE, reason=PROBLEM or "the host can run native programs"),
]


@dataclass(frozen=True)
class Program:
    """One built program: its role, its artifact, its build directory, and the files that directory held."""

    role: str
    artifact: Path
    base: Path
    files: Mapping[str, str]


@pytest.fixture(autouse=True)
def host_ready() -> None:
    """Fail, rather than skip, when LASSI_REQUIRE_SANDBOX=1 asks for a real sandbox and the host cannot run one."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")


@pytest.fixture(scope="module")
def built() -> BuiltToolchain:
    """Build gcc-native as a run builds it: the pinned host g++, checked by --version in the compile sandbox."""
    if PROBLEM:
        pytest.fail(f"LASSI_REQUIRE_SANDBOX=1, but {PROBLEM}")
    return build_toolchain("gcc-native", Path(os.environ["LASSI_TOOLCHAINS"]))


@pytest.fixture(scope="module")
def limits() -> Limits:
    """Return an attempt run's limits under the gate recipe with no reference wall time (attempt_limits)."""
    sandbox = load_recipe(RECIPE).data["sandbox"]
    rule = sandbox["wall_s"]
    wall_s = RUN_WALL_FLOOR_S if rule == BASELINE_X10 else float(rule)
    return Limits(wall_s=wall_s, memory_mb=round(float(sandbox["mem_gb"]) * 1024), cpus=RUN_CPUS)


@pytest.fixture
def base() -> Iterator[Path]:
    """Create a temp directory under $LASSI_RUNS_ROOT for the builds and runs; remove it afterwards."""
    path = Path(tempfile.mkdtemp(prefix="lassi-adv-seeded.", dir=os.environ["LASSI_RUNS_ROOT"]))
    try:
        yield path
    finally:
        shutil.rmtree(path)


def suite() -> Suite:
    """Load the adv-seeded-v0 manifest."""
    return load_suite(MANIFEST)


def candidate_files(entry: Mapping[str, Any]) -> dict[str, str]:
    """Return a candidate as the files its completion carries: {<item>.cpp: the text of the entry's file}."""
    file = entry["file"]
    path = REPO / file if file.startswith("tests/") else SEEDED / file
    return {f"{entry['item']}.cpp": path.read_bytes().decode("ascii")}


def build(built: BuiltToolchain, item: str, role: str, files: Mapping[str, str], base: Path) -> Program:
    """Build `files` with the item's cpp support files in <base>/<role>-build, as a run builds a cpp program."""
    support = suite().support_files(item, registry.TRACKED_ROOT, purpose=PURPOSE, language="cpp")
    workdir = base / f"{role}-build"
    workdir.mkdir()
    result = built.toolchain.build(files, workdir, harness=support)
    stderr_path = workdir / "compile.stderr"
    stderr = stderr_path.read_bytes().decode("utf-8", "replace") if stderr_path.is_file() else ""
    errors = [entry for entry in result.diagnostics if entry.severity == "error"]
    assert result.artifact is not None and not errors, f"{item} {role} did not build:\n{stderr[-4000:]}"
    return Program(role, Path(result.artifact), workdir, {**support, **files})


def placed(program: Program, workdir: Path) -> Path:
    """Write the program's files under the new `workdir`, copy its artifact to its place there, and return the copy."""
    workdir.mkdir(parents=True)
    for relative, text in program.files.items():
        target = workdir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8"))
    try:
        relative = program.artifact.relative_to(program.base)
    except ValueError:
        relative = Path(program.artifact.name)
    target = workdir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(program.artifact, target)
    return target


def witness_arrays(entry: Mapping[str, Any], contract: InputContract | None, folder: Path) -> Sequence[LassiArray]:
    """Return an entry's witness as the adversary stage takes a proposal: parsed, staged under `folder`, checked."""
    (proposal,) = parse_reply(f"```json\n{json.dumps({'proposals': [entry['witness']]})}\n```")
    assert contract is not None, f"{entry['item']} declares no input contract"
    try:
        check_inputs(contract, staged_inputs(proposal.arrays, folder))
    except InputContractError as error:
        pytest.fail(f"{entry['id']}: the witness is refused by the item's contract ({error.reason}): {error}")
    return proposal.arrays


def run_program(
    program: Program, item: registry.SuiteItem, arrays: Sequence[LassiArray] | None, workdir: Path, limits: Limits
) -> RunResult:
    """Run `program` in a fresh `workdir` on the held-out inputs (arrays None) or on `arrays`, through the executor."""
    artifact = placed(program, workdir)
    if arrays is None:
        args = registry.stage_inputs(item, artifact.parent)
    else:
        inputs = artifact.parent / registry.INPUTS_DIR
        inputs.mkdir()
        write_inputs(arrays, inputs)
        args = registry.program_args(item)
    return NativeExecutor().run(artifact, args, limits)


def clean(run: RunResult, side: str) -> None:
    """Assert a clean run: exit status 0, no hang, and a complete workdir."""
    tail = f"{run.stdout[-3000:]}\n{run.stderr[-3000:]}"
    found = (run.exit_code, run.hang, run.workdir_incomplete)
    assert found == (0, False, False), f"{side}: exit {run.exit_code}, hang {run.hang}, incomplete {found[2]}\n{tail}"


def output_names(run: RunResult) -> list[str]:
    """Return the array names of a run's output files, sorted."""
    return sorted(read_array(path).name for path in run.output_files.values())


def output_bytes(run: RunResult) -> dict[str, bytes]:
    """Return a run's output files as relative path -> bytes."""
    return {relative: Path(path).read_bytes() for relative, path in run.output_files.items()}


def oracle_of(item: registry.SuiteItem) -> BinaryIOOracle:
    """Return binary_io bound to the item's declared tolerance."""
    assert item.tolerance is not None, f"{item.name} declares no tolerance"
    return BinaryIOOracle(metric=item.tolerance.metric, threshold=item.tolerance.threshold)


@pytest.mark.parametrize("item", ITEMS)
def test_a_reference_runs_clean_and_repeats_its_outputs(
    item: str, built: BuiltToolchain, limits: Limits, base: Path
) -> None:
    spec = suite().item(item, purpose=PURPOSE)
    files = suite().reference_target(item, CPP, registry.TRACKED_ROOT, purpose=PURPOSE)
    reference = build(built, item, "reference", files, base)
    sets: dict[str, Sequence[LassiArray] | None] = {HELD_OUT: None}
    for entry in CANDIDATES.values():
        if entry["item"] == item:
            sets[f"witness-{entry['id']}"] = witness_arrays(entry, spec.contract, base / f"proposed-{entry['id']}")
    assert len(sets) > 1, f"no candidate of {item} has a witness"
    oracle = oracle_of(spec)
    for name, arrays in sets.items():
        runs = [run_program(reference, spec, arrays, base / name / f"run{count}", limits) for count in (1, 2)]
        for count, run in enumerate(runs, start=1):
            clean(run, f"{item} reference on {name}, run {count}")
            assert output_names(run) == sorted(spec.outputs), f"{item} on {name}: {sorted(run.output_files)}"
            assert oracle.reference_problem(run.output_files) is None, f"{item} on {name}, run {count}"
        first, second = (output_bytes(run) for run in runs)
        assert first == second, f"{item} on {name}: the two runs' output files differ"


@pytest.mark.parametrize("candidate", list(CANDIDATES))
def test_a_seeded_candidate_passes_held_out_and_fails_its_witness(
    candidate: str, built: BuiltToolchain, limits: Limits, base: Path
) -> None:
    entry = CANDIDATES[candidate]
    item = entry["item"]
    spec = suite().item(item, purpose=PURPOSE)
    oracle = oracle_of(spec)
    reference_files = suite().reference_target(item, CPP, registry.TRACKED_ROOT, purpose=PURPOSE)
    reference = build(built, item, "reference", reference_files, base)
    program = build(built, item, "candidate", candidate_files(entry), base)
    sides = (f"{item} reference", f"{candidate} candidate")

    held = [run_program(each, spec, None, base / HELD_OUT / each.role, limits) for each in (reference, program)]
    clean(held[0], f"{item} reference on its held-out inputs")
    clean(held[1], f"{candidate} on {item}'s held-out inputs")
    assert output_names(held[0]) == sorted(spec.outputs)
    stats = oracle.compare(held[0].output_files, held[1].output_files, sides=sides)
    assert stats and all(found.passed for found in stats), f"{candidate} on the held-out inputs: {stats}"

    arrays = witness_arrays(entry, spec.contract, base / "proposed")
    on_witness = run_program(reference, spec, arrays, base / "witness" / reference.role, limits)
    clean(on_witness, f"{item} reference on {candidate}'s witness")
    assert oracle.reference_problem(on_witness.output_files) is None
    attempt = run_program(program, spec, arrays, base / "witness" / program.role, limits)
    stats = oracle.compare(on_witness.output_files, attempt.output_files, sides=sides)
    assert stats and not all(found.passed for found in stats), f"{candidate} passes binary_io on its witness: {stats}"


@pytest.mark.parametrize("candidate", list(CANDIDATES))
def test_a_seeded_candidates_file_hashes_to_its_index_entry(candidate: str) -> None:
    entry = CANDIDATES[candidate]
    completion = render_file_blocks(candidate_files(entry))
    assert hashlib.sha256(completion.encode("ascii")).hexdigest() == entry["sha256"], candidate
