"""Tests for tools/export_counterexamples.py, the export of confirmed counterexamples (task P16.7).

Bible: Adversarial Setups (item 1: confirmed counterexamples join only
synthetic fixtures outside every suite, never the held-out inputs of an
eval or unassigned item), Benchmark Suites (split rules), Project Recipes
(Notes, train recipes: data.synthetic), Result Record (Trial.adversary,
Divergence, the binary store; Storage), Agent Rules 1 and 5; AGENTS.md
Results. Plan: plans/p16-adversarial.md, P16.7, and its Constraints (Agent
Rule 5 and OQ-055, decision 12 (b)).

The interface these tests fix:

- tools/export_counterexamples.py defines COUNTEREXAMPLES_DIR, the one
  destination it writes to, <repository>/tests/fixtures/adversary/counterexamples.
  It reads that module attribute when it runs, so a test points it at a
  directory under its own tmp_path (monkeypatch), laid out as the
  repository lays it out; every other destination is refused.
- export_counterexamples(run_dir, dest=None) -> list[Path] reads a run tree
  (<runs root>/runs/<run id>: provenance.json, the text store texts/, the
  binary store blobs/, and one trial.json per trial at trial depth) and
  writes one fixture per confirmed divergence (each entry of
  Trial.adversary.divergences, in every trial) directly in the destination,
  returning the paths it wrote. dest None means COUNTEREXAMPLES_DIR; a dest
  that does not resolve to it is refused with a ValueError before anything
  is written. A run tree without provenance.json, and a divergence whose
  input file is not in the binary store, are refused the same way, before
  anything is written. So are a COUNTEREXAMPLES_DIR that is missing, that
  overlaps one of the tool's DATA_ROOTS, or that a link at or above it sends
  elsewhere; a manifest key the fixture copies that is missing or of the
  wrong type; a trial with a confirmed divergence whose provenance or
  recipe_hash differs from provenance.json's, whose standing attempt the
  adversary stage would not take, or whose target device the manifest's
  devices cannot name; two fixtures whose names differ at most in letter
  case; and a fixture file already there with other bytes. An identical
  re-export is accepted and changes no byte.
- A fixture is one plain ASCII JSON object with LF line ends, in a file
  named <name>.json, holding at least these keys:
  - provenance: {commit, dirty, device, date}, the run's (provenance.json):
    commit and dirty as recorded; device the run's device, or, with
    executors per language, the device of the trial's target language; and
    date the run's started_utc, the date every trial's provenance copies;
    with run_id, recipe, recipe_hash, status, host, platform, sdk (the
    manifest's driver), and pins copied as the manifest holds them;
  - exploratory: true for a run from a dirty tree, false for a run from a
    clean one (Agent Rule 1); a run whose tree state was not recorded is
    refused or exported as exploratory, never as clean;
  - bench_item: {suite, item, split}, the trial's;
  - candidate_sha256: the sha256 of the candidate's reply, the standing
    attempt's (lassi.core.record standing_attempt) response text as the
    text store keeps it, the reply whose program the adversary stage ran;
  - inputs: {name: {sha256, base64}} for each input of the divergence: the
    sha256 that Divergence.inputs records, and the file's bytes from the
    binary store in standard base64.
  Contract findings, agreed and refused proposals, and trials the stage did
  not take give no fixture.
- main(argv) -> int runs `export_counterexamples.py <run dir> [--dest <dir>]`:
  0 after an export, and 2 with a message on stderr for a refusal; after an
  export of a run whose status is not "complete" it names that status.

What the tests show beyond the format: the export writes only in the
destination (every other file under the test's directory, the run tree
included, and every file under the repository's assets/ and tests/fixtures/
keep their bytes; the repository guard only checks those, and removes
nothing but fixtures of this module's SYNTHETIC run id added in the
repository's counterexamples directory); every file under assets/bench and the bytes
generate_inputs writes for every item stay as they were (sha256 before and
after); and no registry reader (load_suite, Suite.item, Suite.train_view)
and no train data source (data.synthetic through load_train_recipe and
load_synthetic; bench through load_bench) reaches a fixture: each refuses it
or never lists it.

The run tree is built by hand: SYNTHETIC trials of tt-pairs-v0 items
(direction tt-cpp, split unassigned) whose adversary records hold confirmed
divergences, with their proposed inputs, constant arrays inside each item's
contract, in the run's binary store. Items are read through Suite.item for
purpose eval only, apart from the refusals these tests expect; nothing
trains, tunes a prompt, or harvests (Agent Rule 5). Every commit, device,
date, digest, and wall time here is a SYNTHETIC or PLACEHOLDER fixture
value, never a measurement.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from math import prod
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from lassi.bench import EvalSplitError, load_suite
from lassi.core import runner
from lassi.core.interfaces import Sampling
from lassi.core.recipe import RecipeError, load_train_recipe
from lassi.core.record import (
    AdversaryRecord,
    Alignment,
    Attempt,
    BenchItem,
    ContractFinding,
    Diagnostic,
    Divergence,
    Final,
    ModelInfo,
    OutputStats,
    Provenance,
    RunInfo,
    Trial,
    json_text,
    make_trial_id,
    parse_trial_id,
)
from lassi.core.runner import RunError
from lassi.core.store import TRIAL_JSON, BlobStore, TextStore, read_trial, trial_dir, write_trial
from lassi.harness.lassi_io import generate_inputs, write_array
from lassi.train import data as train_data

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "export_counterexamples.py"
BENCH_DIR = REPO / "assets" / "bench"
DEFAULT_DEST = REPO / "tests" / "fixtures" / "adversary" / "counterexamples"
# The repository's directories an export must never change.
WATCHED = (REPO / "assets", REPO / "tests" / "fixtures")

SUITE = "tt-pairs-v0"
SPLIT = "unassigned"
SOURCE, TARGET = "tt", "cpp"
DIRECTION = f"{SOURCE}-{TARGET}"
PROJECT = "adv-export"
ARM = "replay-fixture"
RUN_ID = "20261019-010203"
# A SYNTHETIC commit id and recipe hash: neither names a commit or a recipe of this repository.
SYNTHETIC_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SYNTHETIC_RECIPE_HASH = "fe" * 32
# PLACEHOLDER provenance values: no host was probed and nothing was measured.
PLACEHOLDER_DEVICE = "PLACEHOLDER host CPU (native)"
PLACEHOLDER_CPP_DEVICE = "PLACEHOLDER host CPU (native, cpp only)"
PLACEHOLDER_STARTED = "2026-10-19T01:02:03+00:00"
PLACEHOLDER_FINISHED = "2026-10-19T01:09:03+00:00"
PLACEHOLDER_WALL_S = 5.0
README = "README.md"
README_TEXT = b"SYNTHETIC stand-in for the counterexamples directory's README.\n"
TRAIN_FIXTURE = "sft-tiny.jsonl"
TRAIN_FIXTURE_TEXT = b'{"prompt": "SYNTHETIC prompt", "completion": "SYNTHETIC completion"}\n'
REQUIRED_KEYS = ("provenance", "exploratory", "bench_item", "candidate_sha256", "inputs")
PROVENANCE_KEYS = ("commit", "dirty", "device", "date")
HELD_PURPOSES = ("train", "prompt-tuning", "harvest")
CANDIDATE_FAILURE = "SYNTHETIC: the cpp candidate run ended with exit status 139"


@dataclass(frozen=True)
class Plan:
    """One SYNTHETIC trial: its item and run, which attempts ran, and what its adversary record holds.

    `ran` says, per attempt, whether it ran (its run holds stdout); the
    standing attempt is the last that ran. `divergences` holds (proposal,
    input values, candidate_failure) and `findings` (proposal, input
    values), each input a constant bf16 array of the item's held-out shape.
    `refused` proposals were refused; the rest of `proposals` agreed. A plan
    with `taken` False has no adversary record.
    """

    item: str
    run: int
    ran: tuple[bool, ...]
    proposals: int = 0
    refused: int = 0
    divergences: tuple[tuple[int, Mapping[str, float], str | None], ...] = ()
    findings: tuple[tuple[int, Mapping[str, float]], ...] = ()
    taken: bool = True


# Five confirmed divergences in four trials: a trial whose standing attempt is its last, one whose standing attempt
# is not its last, two trials of one item with the same proposed bytes (two fixtures), and a two-input divergence;
# then a trial with a contract finding only and a trial the stage did not take, neither of which gives a fixture.
PLANS = (
    Plan(
        "loopback",
        1,
        ran=(False, True),
        proposals=5,
        refused=1,
        divergences=((1, {"input": 1.0}, None), (3, {"input": 2.0}, None)),
        findings=((2, {"input": 3.0}),),
    ),
    Plan("eltwise_sfpu", 1, ran=(True, False), proposals=2, divergences=((0, {"src0": 0.5}, CANDIDATE_FAILURE),)),
    Plan("eltwise_sfpu", 2, ran=(True,), proposals=1, divergences=((0, {"src0": 0.5}, None),)),
    Plan("eltwise_binary", 1, ran=(True,), proposals=3, divergences=((0, {"a": 0.5, "b": -0.5}, None),)),
    Plan("loopback", 2, ran=(True,), proposals=2, findings=((0, {"input": 4.0}),)),
    Plan("eltwise_binary", 2, ran=(True,), taken=False),
)


@dataclass(frozen=True)
class Expected:
    """What the fixture of one confirmed divergence must hold: its item, candidate reply digest, and input bytes.

    `run` and `proposal` name the divergence: the trial's run number and the proposal's index.
    """

    item: str
    run: int
    proposal: int
    candidate_sha256: str
    inputs: Mapping[str, bytes]

    def key(self) -> tuple[str, frozenset[tuple[str, str]]]:
        """Return the candidate digest and the set of (input name, input digest), which tells fixtures apart."""
        return self.candidate_sha256, frozenset((name, sha256(data)) for name, data in self.inputs.items())


@dataclass(frozen=True)
class BuiltRun:
    """A hand-built run tree: its directory, each confirmed divergence's expected fixture, and the findings' inputs."""

    run_dir: Path
    expected: tuple[Expected, ...]
    finding_digests: frozenset[str]


@dataclass(frozen=True)
class Layout:
    """The test's stand-in for the repository: the redirected destination and a synthetic train directory beside it."""

    repo: Path
    dest: Path
    train: Path


# ---------------------------------------------------------------------------
# Helpers


def sha256(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=1)
def tool() -> ModuleType:
    """Load tools/export_counterexamples.py as a module, failing with the task's name while it does not exist."""
    assert TOOL.is_file(), f"{TOOL.relative_to(REPO).as_posix()} does not exist yet (task P16.7)"
    spec = importlib.util.spec_from_file_location("export_counterexamples_under_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # registered before it runs, as an import would be
    spec.loader.exec_module(module)
    return module


def redirect(monkeypatch: pytest.MonkeyPatch, layout: Layout) -> ModuleType:
    """Return the tool with COUNTEREXAMPLES_DIR pointed at the layout's destination."""
    module = tool()
    assert hasattr(module, "COUNTEREXAMPLES_DIR"), "the tool defines no COUNTEREXAMPLES_DIR"
    monkeypatch.setattr(module, "COUNTEREXAMPLES_DIR", layout.dest)
    return module


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    """Lay out <tmp>/repo/tests/fixtures/{adversary/counterexamples,train} with a README and a train fixture."""
    repo = tmp_path / "repo"
    dest = repo / "tests" / "fixtures" / "adversary" / "counterexamples"
    dest.mkdir(parents=True)
    (dest / README).write_bytes(README_TEXT)
    train = repo / "tests" / "fixtures" / "train"
    train.mkdir(parents=True)
    (train / TRAIN_FIXTURE).write_bytes(TRAIN_FIXTURE_TEXT)
    return Layout(repo=repo, dest=dest, train=train)


def snapshot(root: Path) -> dict[str, str]:
    """Return every file under `root` as its POSIX path relative to `root` -> the sha256 of its bytes."""
    if not root.exists():
        return {}
    return {path.relative_to(root).as_posix(): sha256(path.read_bytes()) for path in root.rglob("*") if path.is_file()}


def synthetic_run_fixtures(directory: Path) -> set[Path]:
    """Return the files directly in `directory` named as a fixture of this module's SYNTHETIC run (RUN_ID)."""
    return {path for path in directory.glob(f"{RUN_ID}--*.json") if path.is_file()}


@contextmanager
def repo_guard() -> Iterator[Callable[[], dict[str, list[str]]]]:
    """Keep the digest of every watched repository file; yield a function that returns what changed so far.

    The function returns, per watched directory, the files added, changed,
    or removed since the guard began, and a test asserts that it is empty.
    The guard restores nothing, so it never undoes another process's edit:
    on exit it removes only fixtures of this module's SYNTHETIC run id
    (RUN_ID) that appeared directly in the repository's counterexamples
    directory, so a wrong export of a test run leaves no fixture there.
    """
    kept = {root: snapshot(root) for root in WATCHED}
    present = synthetic_run_fixtures(DEFAULT_DEST)

    def changes() -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for root, files in kept.items():
            now = snapshot(root)
            differ = sorted(name for name in set(files) | set(now) if files.get(name) != now.get(name))
            if differ:
                found[root.name] = differ
        return found

    try:
        yield changes
    finally:
        for path in synthetic_run_fixtures(DEFAULT_DEST) - present:
            path.unlink()


def written(dest: Path) -> list[Path]:
    """Return every file under `dest` but the README, sorted."""
    return sorted(path for path in dest.rglob("*") if path.is_file() and path != dest / README)


def load_fixture(path: Path) -> dict[str, Any]:
    """Read one fixture as ASCII JSON and check that it holds every required key."""
    data = json.loads(path.read_bytes().decode("ascii"))
    assert isinstance(data, dict), f"{path.name} is not a JSON object"
    missing = [key for key in REQUIRED_KEYS if key not in data]
    assert missing == [], f"{path.name} lacks {missing}"
    return data


def fixture_key(data: Mapping[str, Any]) -> tuple[str, frozenset[tuple[str, str]]]:
    """Return a fixture's candidate digest and its set of (input name, input digest)."""
    return data["candidate_sha256"], frozenset((name, entry["sha256"]) for name, entry in data["inputs"].items())


def exported(dest: Path) -> dict[tuple[str, frozenset[tuple[str, str]]], dict[str, Any]]:
    """Return every fixture in `dest` by its key; two fixtures with one key fail."""
    fixtures = [load_fixture(path) for path in written(dest)]
    found = {fixture_key(data): data for data in fixtures}
    assert len(found) == len(fixtures), "two fixtures hold one candidate and one input set"
    return found


def relative_name(path: Path, start: Path) -> str:
    """Return `path` relative to `start` in POSIX form, or `path` itself when no relative path joins them."""
    try:
        return Path(os.path.relpath(path, start)).as_posix()
    except ValueError:  # another drive on Windows
        return str(path)


# ---------------------------------------------------------------------------
# The hand-built run tree


def bf16_array(path: Path, name: str, shape: Sequence[int], value: float) -> bytes:
    """Write a constant SYNTHETIC bf16 lassi_io array of `value` at `path` and return its bytes."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0] >> 16
    write_array(path, name, "bf16", list(shape), struct.pack("<H", bits) * prod(shape))
    return path.read_bytes()


@lru_cache(maxsize=None)
def held_out_shapes(item: str) -> dict[str, tuple[int, ...]]:
    """Return the held-out input names and shapes of a tt-pairs-v0 item, read through Suite.item for eval."""
    spec = load_suite(BENCH_DIR / f"{SUITE}.yaml").item(item, purpose="eval")
    return {entry["name"]: tuple(entry["shape"]) for entry in spec.inputs}


def store_inputs(
    blobs: BlobStore, scratch: Path, item: str, values: Mapping[str, float]
) -> tuple[dict[str, str], dict[str, bytes]]:
    """Put a proposal's arrays, at the item's held-out names and shapes, in the binary store; return digests, bytes."""
    shapes = held_out_shapes(item)
    assert set(values) == set(shapes), f"the plan's inputs for {item} must be its held-out inputs {sorted(shapes)}"
    scratch.mkdir(parents=True, exist_ok=True)
    raw = {
        name: bf16_array(scratch / f"{item}-{name}-{value}.lassiio", name, shapes[name], value)
        for name, value in values.items()
    }
    return {name: blobs.put(data) for name, data in raw.items()}, raw


def make_attempt(index: int, reply: str, ran: bool, store: TextStore, output_sha: str) -> Attempt:
    """Return a SYNTHETIC attempt: a clean run that passed binary_io, or a compile error that never ran."""
    files = {"main.cpp": f"// SYNTHETIC candidate, attempt {index}\n"}
    if not ran:
        error = Diagnostic(stage="compile", severity="error", message="SYNTHETIC compile error")
        return Attempt(index=index, response_text=reply, files=files, stage_reached="S1", diagnostics=[error])
    run = RunInfo(
        exit_code=0,
        hang=False,
        wall_s=PLACEHOLDER_WALL_S,
        stdout_ref=store.put("SYNTHETIC stdout\n"),
        stdout_truncated=False,
        stderr_truncated=False,
        workdir_incomplete=False,
        outputs={"output.lassiio": output_sha},
    )
    passed = OutputStats(name="output", pcc=1.0, max_abs=0.0, max_ulp=0, passed=True, note=None)
    alignment = Alignment(per_input=[1.0], mean=1.0, outputs=[passed])
    return Attempt(index=index, response_text=reply, files=files, stage_reached="S5", run=run, alignment=alignment)


def adversary_record(plan: Plan, blobs: BlobStore, scratch: Path) -> tuple[AdversaryRecord, list[Expected], set[str]]:
    """Return the plan's adversary record, with its divergences' expected inputs and its findings' input digests."""
    failed = OutputStats(name="output", pcc=0.5, max_abs=3.0, max_ulp=100, passed=False, note="SYNTHETIC: differs")
    divergences: list[Divergence] = []
    expected: list[Expected] = []
    for proposal, values, failure in plan.divergences:
        digests, raw = store_inputs(blobs, scratch, plan.item, values)
        divergences.append(Divergence(proposal=proposal, inputs=digests, stats=[failed], candidate_failure=failure))
        expected.append(Expected(item=plan.item, run=plan.run, proposal=proposal, candidate_sha256="", inputs=raw))
    findings: list[ContractFinding] = []
    finding_digests: set[str] = set()
    for proposal, values in plan.findings:
        digests, _ = store_inputs(blobs, scratch, plan.item, values)
        findings.append(ContractFinding(proposal=proposal, inputs=digests, message="SYNTHETIC: reference exit 2"))
        finding_digests.update(digests.values())
    agreed = plan.proposals - plan.refused - len(divergences) - len(findings)
    record = AdversaryRecord(
        agent="fuzzer",
        config={"budget": {"inputs": max(plan.proposals, 1)}, "seed": 7},
        proposals=plan.proposals,
        refusals={"out-of-range": plan.refused} if plan.refused else {},
        runs=2 * (len(divergences) + agreed) + len(findings),
        wall_s=PLACEHOLDER_WALL_S,
        divergences=divergences,
        contract_findings=findings,
    )
    return record, expected, finding_digests


def provenance_json(
    commit: str | None, dirty: bool | None, per_language: bool, status: str = "complete"
) -> dict[str, Any]:
    """Return a run manifest shaped as lassi.core.runner writes it, with SYNTHETIC and PLACEHOLDER values."""
    if per_language:
        executors: dict[str, Any] = {
            "executor": {SOURCE: "none", TARGET: "native"},
            "devices": {SOURCE: "none (compile only)", TARGET: PLACEHOLDER_CPP_DEVICE},
        }
    else:
        executors = {"executor": "native", "device": PLACEHOLDER_DEVICE}
    return {
        "run_id": RUN_ID,
        "status": status,
        "recipe": PROJECT,
        "recipe_path": f"projects/{PROJECT}/recipe.yaml",
        "recipe_chain": [f"projects/{PROJECT}/recipe.yaml"],
        "recipe_hash": SYNTHETIC_RECIPE_HASH,
        "commit": commit,
        "dirty": dirty,
        "host": "PLACEHOLDER-HOST",
        "platform": "PLACEHOLDER-PLATFORM",
        "python": "PLACEHOLDER-PYTHON",
        **executors,
        "driver": None,
        "device_records": [],
        "started_utc": PLACEHOLDER_STARTED,
        "finished_utc": PLACEHOLDER_FINISHED,
        "pins": {"gcc": "PLACEHOLDER"},
    }


def build_trial(
    plan: Plan, provenance: Provenance, store: TextStore, blobs: BlobStore, scratch: Path
) -> tuple[Trial, list[Expected], set[str]]:
    """Build one plan's trial; return it, its divergences' Expected entries, and its findings' input digests."""
    output_sha = blobs.put(f"SYNTHETIC output of {plan.item} run{plan.run:02d}\n".encode("ascii"))
    replies = [f"SYNTHETIC reply: {plan.item} run{plan.run:02d} attempt {index}\n" for index in range(len(plan.ran))]
    attempts = [
        make_attempt(index, reply, ran, store, output_sha)
        for index, (reply, ran) in enumerate(zip(replies, plan.ran, strict=True))
    ]
    standing = max(index for index, ran in enumerate(plan.ran) if ran)
    record: AdversaryRecord | None = None
    expected: list[Expected] = []
    finding_digests: set[str] = set()
    if plan.taken:
        record, expected, finding_digests = adversary_record(plan, blobs, scratch)
    candidate = sha256(replies[standing].encode("utf-8"))
    expected = [dataclasses.replace(entry, candidate_sha256=candidate) for entry in expected]
    trial = Trial(
        trial_id=make_trial_id(PROJECT, ARM, SUITE, DIRECTION, plan.item, plan.run),
        recipe_hash=SYNTHETIC_RECIPE_HASH,
        provenance=provenance,
        bench_item=BenchItem(suite=SUITE, item=plan.item, split=SPLIT, direction=DIRECTION),
        model=ModelInfo(backend="replay", id=ARM, sampling=Sampling(0.2, 0.9, 64)),
        attempts=attempts,
        final=Final(stage_reached="S5", alignment=1.0, corrections=len(attempts) - 1, wall_s=PLACEHOLDER_WALL_S),
        requests=[],
        adversary=record,
    )
    return trial, expected, finding_digests


def build_run(
    root: Path,
    *,
    commit: str | None = SYNTHETIC_COMMIT,
    dirty: bool | None = False,
    per_language: bool = False,
    status: str = "complete",
) -> BuiltRun:
    """Write the SYNTHETIC run tree <root>/runs-root/runs/<RUN_ID> of PLANS and return what its export must give."""
    run_dir = root / "runs-root" / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    manifest = provenance_json(commit, dirty, per_language, status)
    (run_dir / runner.PROVENANCE_JSON).write_bytes(json_text(manifest).encode("ascii"))
    device = PLACEHOLDER_CPP_DEVICE if per_language else PLACEHOLDER_DEVICE
    provenance = Provenance(
        commit=commit, dirty=dirty, device=device, sdk=None, date=PLACEHOLDER_STARTED, device_records=[]
    )
    store, blobs = TextStore(run_dir), BlobStore(run_dir)
    expected: list[Expected] = []
    finding_digests: set[str] = set()
    for plan in PLANS:
        trial, found, findings = build_trial(plan, provenance, store, blobs, root / "scratch-arrays")
        write_trial(trial, run_dir, store)
        expected.extend(found)
        finding_digests.update(findings)
    assert len(expected) == 5
    return BuiltRun(run_dir=run_dir, expected=tuple(expected), finding_digests=frozenset(finding_digests))


def held_out_digests(out: Path) -> dict[tuple[str, str, str], str]:
    """Write every manifest item's held-out inputs under `out`; return (suite, item, file) -> sha256 of its bytes."""
    digests: dict[tuple[str, str, str], str] = {}
    for manifest in sorted(BENCH_DIR.glob("*.yaml")):
        suite = load_suite(manifest)
        for name in sorted(suite.items):
            item = suite.item(name, purpose="eval")
            if item.inputs:
                files = generate_inputs(item.inputs, item.seed, _fresh(out / suite.name / name))
                digests.update({(suite.name, name, path.name): sha256(path.read_bytes()) for path in files.values()})
    return digests


def _fresh(path: Path) -> Path:
    """Create the empty directory `path` and return it."""
    path.mkdir(parents=True)
    return path


def manifest_views() -> dict[str, str]:
    """Return each manifest under assets/bench, by file name, as the repr of the Suite load_suite gives."""
    return {path.name: repr(load_suite(path)) for path in sorted(BENCH_DIR.glob("*.yaml"))}


def rewrite_manifest(run_dir: Path, changes: Mapping[str, Any], removed: Sequence[str] = ()) -> None:
    """Rewrite the run's provenance.json with `changes` applied and the keys in `removed` left out."""
    path = run_dir / runner.PROVENANCE_JSON
    data = json.loads(path.read_text(encoding="ascii"))
    data.update(changes)
    for key in removed:
        del data[key]
    path.write_bytes(json_text(data).encode("ascii"))


def the_trial(run_dir: Path, item: str, run: int) -> tuple[Trial, TextStore]:
    """Return the run tree's trial of `item` and `run`, read from its trial.json, with the run's text store."""
    store = TextStore(run_dir)
    trial_id = make_trial_id(PROJECT, ARM, SUITE, DIRECTION, item, run)
    return read_trial(trial_dir(run_dir, trial_id) / TRIAL_JSON, store), store


def copy_trial(run_dir: Path, item: str, run: int, *, arm: str = ARM, new_item: str | None = None) -> None:
    """Write a copy of the trial of `item` and `run` under another arm or item name in the same run tree."""
    trial, store = the_trial(run_dir, item, run)
    parts = parse_trial_id(trial.trial_id)
    name = item if new_item is None else new_item
    trial_id = make_trial_id(parts.project, arm, parts.bench, parts.direction, name, parts.run)
    bench_item = dataclasses.replace(trial.bench_item, item=name)
    write_trial(dataclasses.replace(trial, trial_id=trial_id, bench_item=bench_item), run_dir, store)


def the_expected(run: BuiltRun, item: str, run_number: int, proposal: int) -> Expected:
    """Return the expected fixture of the divergence of `item`, `run_number`, and `proposal`."""
    return next(
        entry for entry in run.expected if (entry.item, entry.run, entry.proposal) == (item, run_number, proposal)
    )


def by_name(paths: Sequence[Path]) -> list[Path]:
    """Return `paths` sorted by file name; the last is the export's last fixture (trials in path order)."""
    return sorted(paths, key=lambda path: path.name)


def make_dir_link(link: Path, target: Path) -> None:
    """Make `link` a directory link to `target`: a symlink, or on Windows a junction; skip when neither is allowed."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except OSError as error:
        if sys.platform != "win32":
            pytest.skip(f"cannot make a directory symlink here: {error}")
    import _winapi  # Windows only: a junction needs no privilege

    try:
        _winapi.CreateJunction(str(target), str(link))
    except OSError as error:
        pytest.skip(f"cannot make a junction here: {error}")


def case_sensitive(directory: Path) -> bool:
    """Return whether names that differ only in letter case are two files in `directory`."""
    probe = directory / "Case-Probe"
    probe.mkdir()
    try:
        return not (directory / "case-probe").exists()
    finally:
        probe.rmdir()


# ---------------------------------------------------------------------------
# The destination


def test_the_destination_is_the_counterexamples_fixture_directory() -> None:
    assert Path(tool().COUNTEREXAMPLES_DIR).resolve() == DEFAULT_DEST.resolve()


def test_the_destination_lies_outside_every_suite_and_train_data_root() -> None:
    dest = Path(tool().COUNTEREXAMPLES_DIR).resolve()
    for root in (REPO / "assets", runner.BENCH_DIR, train_data.SYNTHETIC_DIR):
        resolved = root.resolve()
        assert resolved != dest, root
        assert resolved not in dest.parents, f"{dest} lies inside {root}"
        assert dest not in resolved.parents, f"{root} lies inside {dest}"


@pytest.mark.parametrize(
    "where",
    [
        "a new directory elsewhere",
        "an existing directory elsewhere",
        "the destination's parent",
        "a directory inside the destination",
        "the run tree",
        "the train fixture directory beside the destination",
        "the repository's train fixture directory",
        "assets/bench",
        "a suite's tracked sources",
    ],
)
def test_any_other_destination_is_refused_and_nothing_is_written(
    where: str, layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    module = redirect(monkeypatch, layout)
    (tmp_path / "existing").mkdir()
    dests = {
        "a new directory elsewhere": tmp_path / "elsewhere",
        "an existing directory elsewhere": tmp_path / "existing",
        "the destination's parent": layout.dest.parent,
        "a directory inside the destination": layout.dest / "nested",
        "the run tree": run.run_dir,
        "the train fixture directory beside the destination": layout.train,
        "the repository's train fixture directory": train_data.SYNTHETIC_DIR,
        "assets/bench": BENCH_DIR,
        "a suite's tracked sources": BENCH_DIR / SUITE / "loopback" / "cpp",
    }
    dest = dests[where]
    existed = dest.exists()
    before = snapshot(tmp_path)
    with repo_guard() as changes:
        with pytest.raises(ValueError):
            module.export_counterexamples(run.run_dir, dest)
        assert changes() == {}, "the refused export changed the repository"
    assert snapshot(tmp_path) == before, "the refused export wrote or changed a file"
    assert dest.exists() == existed


def test_the_destination_given_explicitly_is_accepted(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    module = redirect(monkeypatch, layout)
    module.export_counterexamples(run.run_dir, layout.dest)
    assert len(written(layout.dest)) == len(run.expected)


def test_the_export_writes_only_in_the_destination(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    module = redirect(monkeypatch, layout)
    before = snapshot(tmp_path)
    dest_prefix = layout.dest.relative_to(tmp_path).as_posix() + "/"
    with repo_guard() as changes:
        paths = module.export_counterexamples(run.run_dir)
        assert changes() == {}, "the export changed the repository"
    after = snapshot(tmp_path)
    assert {name: digest for name, digest in after.items() if name in before} == before
    added = sorted(set(after) - set(before))
    assert added, "the export wrote nothing"
    assert [name for name in added if not name.startswith(dest_prefix)] == []
    assert all(Path(path).resolve().parent == layout.dest.resolve() for path in paths)


def test_a_destination_that_does_not_exist_is_refused(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    module = tool()
    missing = layout.dest.parent / "missing-counterexamples"
    monkeypatch.setattr(module, "COUNTEREXAMPLES_DIR", missing)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="not a directory"):
        module.export_counterexamples(run.run_dir)
    assert not missing.exists()
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("root", ["the destination", "its parent", "a directory inside it", "the stand-in repository"])
def test_a_destination_that_overlaps_a_data_root_is_refused(
    root: str, layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    module = redirect(monkeypatch, layout)
    roots = {
        "the destination": layout.dest,
        "its parent": layout.dest.parent,
        "a directory inside it": layout.dest / "inner",
        "the stand-in repository": layout.repo,
    }
    monkeypatch.setattr(module, "DATA_ROOTS", (roots[root],))
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="data root"):
        module.export_counterexamples(run.run_dir)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("linked", ["the destination", "the destination's parent"])
def test_a_destination_reached_through_a_link_is_refused(
    linked: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    real = tmp_path / "real" / "adversary" / "counterexamples"
    real.mkdir(parents=True)
    (real / README).write_bytes(README_TEXT)
    links = tmp_path / "linked"
    links.mkdir()
    if linked == "the destination":
        link, target, dest = links / "counterexamples", real, links / "counterexamples"
    else:
        link, target, dest = links / "adversary", real.parent, links / "adversary" / "counterexamples"
    make_dir_link(link, target)
    try:
        assert dest.resolve() == real.resolve(), "the link does not lead to the real directory"
        module = tool()
        monkeypatch.setattr(module, "COUNTEREXAMPLES_DIR", dest)
        before = snapshot(real)
        for given in (None, dest, real):
            with pytest.raises(ValueError, match="link"):
                module.export_counterexamples(run.run_dir, given)
        assert snapshot(real) == before, "an export wrote through the link"
    finally:
        os.unlink(link)
    assert real.is_dir()


# ---------------------------------------------------------------------------
# The fixtures


def test_the_export_writes_one_fixture_per_confirmed_divergence(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    module = redirect(monkeypatch, layout)
    paths = module.export_counterexamples(run.run_dir)
    files = written(layout.dest)
    assert sorted(Path(path).resolve() for path in paths) == [path.resolve() for path in files]
    assert len(files) == len(run.expected)
    for path in files:
        assert path.parent == layout.dest, f"{path} is not directly in the destination"
        assert path.suffix == ".json", path.name
    assert sorted(exported(layout.dest)) == sorted(entry.key() for entry in run.expected)


def test_a_fixture_holds_the_runs_provenance_its_bench_item_its_candidate_and_its_inputs(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    redirect(monkeypatch, layout).export_counterexamples(run.run_dir)
    fixtures = exported(layout.dest)
    provenance = {"commit": SYNTHETIC_COMMIT, "dirty": False, "device": PLACEHOLDER_DEVICE, "date": PLACEHOLDER_STARTED}
    for expected in run.expected:
        data = fixtures[expected.key()]
        assert {key: data["provenance"].get(key) for key in PROVENANCE_KEYS} == provenance
        assert {key: data["bench_item"].get(key) for key in ("suite", "item", "split")} == {
            "suite": SUITE,
            "item": expected.item,
            "split": SPLIT,
        }
        assert data["candidate_sha256"] == expected.candidate_sha256
        assert sorted(data["inputs"]) == sorted(expected.inputs)
        for name, entry in data["inputs"].items():
            copied = base64.b64decode(entry["base64"], validate=True)
            assert copied == expected.inputs[name], f"{expected.item} input {name} is not the proposed file's bytes"
            assert entry["sha256"] == sha256(copied)


def test_a_fixture_carries_the_runs_host_platform_and_the_rest_of_its_provenance(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    redirect(monkeypatch, layout).export_counterexamples(run.run_dir)
    fixtures = exported(layout.dest)
    manifest = provenance_json(SYNTHETIC_COMMIT, False, per_language=False)
    keys = ("run_id", "recipe", "recipe_hash", "status", "host", "platform", "pins")
    expected = {**{key: manifest[key] for key in keys}, "sdk": manifest["driver"]}
    assert fixtures, "the export wrote nothing"
    for data in fixtures.values():
        assert {key: data["provenance"].get(key) for key in expected} == expected


def test_contract_findings_and_untaken_trials_give_no_fixture(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    redirect(monkeypatch, layout).export_counterexamples(run.run_dir)
    fixtures = exported(layout.dest)
    digests = {entry["sha256"] for data in fixtures.values() for entry in data["inputs"].values()}
    assert digests & run.finding_digests == set()
    assert len(fixtures) == len(run.expected)


def test_fixtures_are_plain_ascii_json_with_lf_line_ends(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    redirect(monkeypatch, layout).export_counterexamples(run.run_dir)
    files = written(layout.dest)
    assert files, "the export wrote nothing"
    for path in files:
        raw = path.read_bytes()
        assert raw.isascii(), f"{path.name} is not plain ASCII"
        assert b"\r" not in raw, f"{path.name} has a CR"
        assert isinstance(json.loads(raw.decode("ascii")), dict)


@pytest.mark.parametrize("dirty", [False, True])
def test_a_dirty_tree_runs_export_is_marked_exploratory(
    dirty: bool, layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path, dirty=dirty)
    redirect(monkeypatch, layout).export_counterexamples(run.run_dir)
    fixtures = exported(layout.dest)
    assert len(fixtures) == len(run.expected)
    for data in fixtures.values():
        assert data["provenance"]["dirty"] is dirty
        assert data["exploratory"] is dirty


def test_a_run_whose_tree_state_was_not_recorded_is_never_exported_as_clean(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path, commit=None, dirty=None)
    module = redirect(monkeypatch, layout)
    try:
        module.export_counterexamples(run.run_dir)
    except ValueError:
        assert written(layout.dest) == [], "a refused export wrote a fixture"
        return
    fixtures = exported(layout.dest)
    assert len(fixtures) == len(run.expected)
    for data in fixtures.values():
        assert data["provenance"]["commit"] is None and data["provenance"]["dirty"] is None
        assert data["exploratory"] is True


@pytest.mark.parametrize(("commit", "dirty"), [(None, False), (None, None), (SYNTHETIC_COMMIT, None)])
def test_a_run_without_a_known_commit_and_a_clean_tree_is_exported_as_exploratory(
    commit: str | None, dirty: bool | None, layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path, commit=commit, dirty=dirty)
    redirect(monkeypatch, layout).export_counterexamples(run.run_dir)
    fixtures = exported(layout.dest)
    assert len(fixtures) == len(run.expected)
    for data in fixtures.values():
        assert (data["provenance"]["commit"], data["provenance"]["dirty"]) == (commit, dirty)
        assert data["exploratory"] is True


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("commit", "89abcdef" * 5),
        ("dirty", True),
        ("device", "PLACEHOLDER another host CPU"),
        ("driver", "PLACEHOLDER driver"),
        ("started_utc", "2026-10-19T01:02:04+00:00"),
        ("recipe_hash", "ab" * 32),
    ],
)
def test_a_trial_whose_provenance_differs_from_the_runs_is_refused(
    key: str, value: Any, layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    rewrite_manifest(run.run_dir, {key: value})
    module = redirect(monkeypatch, layout)
    with pytest.raises(ValueError, match="provenance.json"):
        module.export_counterexamples(run.run_dir)
    assert written(layout.dest) == []


@pytest.mark.parametrize(
    ("key", "value"),
    [("recipe", 7), ("recipe_hash", None), ("status", None), ("host", None), ("platform", 3), ("pins", [])],
)
def test_a_manifest_value_of_the_wrong_type_is_refused(
    key: str, value: Any, layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    rewrite_manifest(run.run_dir, {key: value})
    module = redirect(monkeypatch, layout)
    with pytest.raises(ValueError, match=key):
        module.export_counterexamples(run.run_dir)
    assert written(layout.dest) == []


@pytest.mark.parametrize("key", ["recipe_hash", "status", "host", "platform", "pins", "started_utc"])
def test_a_manifest_without_a_key_the_fixture_copies_is_refused(
    key: str, layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    rewrite_manifest(run.run_dir, {}, removed=[key])
    module = redirect(monkeypatch, layout)
    with pytest.raises(ValueError, match=key):
        module.export_counterexamples(run.run_dir)
    assert written(layout.dest) == []


def test_an_adversary_record_on_a_trial_the_stage_would_not_take_is_refused(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    trial, store = the_trial(run.run_dir, "loopback", 1)
    assert trial.adversary is not None and trial.adversary.divergences
    failed = OutputStats(name="output", pcc=0.5, max_abs=3.0, max_ulp=100, passed=False, note="SYNTHETIC: differs")
    alignment = Alignment(per_input=[0.5], mean=0.5, outputs=[failed])
    attempts = [*trial.attempts[:-1], dataclasses.replace(trial.attempts[-1], alignment=alignment)]
    final = dataclasses.replace(trial.final, alignment=0.5)
    write_trial(dataclasses.replace(trial, attempts=attempts, final=final), run.run_dir, store)
    module = redirect(monkeypatch, layout)
    with pytest.raises(ValueError, match="adversary stage"):
        module.export_counterexamples(run.run_dir)
    assert written(layout.dest) == []


def test_a_devices_map_that_names_no_target_is_refused(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path, per_language=True)
    rewrite_manifest(run.run_dir, {"devices": {SOURCE: "none (compile only)", "rust": PLACEHOLDER_CPP_DEVICE}})
    module = redirect(monkeypatch, layout)
    with pytest.raises(ValueError, match="devices"):
        module.export_counterexamples(run.run_dir)
    assert written(layout.dest) == []


def test_with_executors_per_language_the_device_is_the_target_languages(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path, per_language=True)
    redirect(monkeypatch, layout).export_counterexamples(run.run_dir)
    fixtures = exported(layout.dest)
    assert len(fixtures) == len(run.expected)
    assert {data["provenance"]["device"] for data in fixtures.values()} == {PLACEHOLDER_CPP_DEVICE}


def test_a_run_tree_without_provenance_is_refused(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    (run.run_dir / runner.PROVENANCE_JSON).unlink()
    module = redirect(monkeypatch, layout)
    with pytest.raises(ValueError):
        module.export_counterexamples(run.run_dir)
    assert written(layout.dest) == []


@pytest.mark.parametrize(
    ("item", "run_number", "proposal", "name"),
    [
        ("eltwise_binary", 1, 0, "b"),  # the first trial in path order
        ("loopback", 1, 1, "input"),
        ("loopback", 1, 3, "input"),  # the last divergence of the last trial in path order
    ],
)
def test_a_divergence_whose_inputs_are_not_in_the_binary_store_is_refused(
    item: str,
    run_number: int,
    proposal: int,
    name: str,
    layout: Layout,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = build_run(tmp_path)
    digest = sha256(the_expected(run, item, run_number, proposal).inputs[name])
    (run.run_dir / "blobs" / digest[:2] / digest).unlink()
    module = redirect(monkeypatch, layout)
    with pytest.raises(ValueError, match="binary store"):
        module.export_counterexamples(run.run_dir)
    assert written(layout.dest) == [], "a refused export wrote a fixture"


def test_an_identical_re_export_is_accepted_and_changes_no_byte(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    module = redirect(monkeypatch, layout)
    first = module.export_counterexamples(run.run_dir)
    before = snapshot(layout.dest)
    second = module.export_counterexamples(run.run_dir)
    assert by_name(second) == by_name(first)
    assert snapshot(layout.dest) == before


def test_a_fixture_already_there_with_other_bytes_is_refused_and_kept(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    module = redirect(monkeypatch, layout)
    *others, changed = by_name(module.export_counterexamples(run.run_dir))
    changed.write_bytes(b'{"SYNTHETIC": "another fixture"}\n')
    for path in others:
        path.unlink()
    before = snapshot(layout.dest)
    with pytest.raises(ValueError, match="overwrite"):
        module.export_counterexamples(run.run_dir)
    assert snapshot(layout.dest) == before, "the refused export wrote, or overwrote, a fixture"


def test_a_directory_at_a_fixtures_name_is_refused(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    module = redirect(monkeypatch, layout)
    paths = by_name(module.export_counterexamples(run.run_dir))
    for path in paths:
        path.unlink()
    paths[-1].mkdir()
    with pytest.raises(ValueError, match="overwrite"):
        module.export_counterexamples(run.run_dir)
    assert paths[-1].is_dir()
    assert written(layout.dest) == []


def test_two_fixtures_with_one_name_are_refused(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    copy_trial(run.run_dir, "eltwise_binary", 1, arm="replay-other")  # the arm is not part of a fixture's name
    module = redirect(monkeypatch, layout)
    with pytest.raises(ValueError, match="fixture name"):
        module.export_counterexamples(run.run_dir)
    assert written(layout.dest) == []


def test_two_fixture_names_that_differ_only_in_letter_case_are_refused(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    if not case_sensitive(run.run_dir):
        pytest.skip("this filesystem keeps one directory for item names that differ only in letter case")
    copy_trial(run.run_dir, "loopback", 1, new_item="LOOPBACK")
    module = redirect(monkeypatch, layout)
    with pytest.raises(ValueError, match="fixture name"):
        module.export_counterexamples(run.run_dir)
    assert written(layout.dest) == []


# ---------------------------------------------------------------------------
# The command line


def test_main_exports_to_the_destination_and_returns_0(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    module = redirect(monkeypatch, layout)
    assert module.main([str(run.run_dir)]) == 0
    assert sorted(exported(layout.dest)) == sorted(entry.key() for entry in run.expected)


@pytest.mark.parametrize("status", ["running", "failed", "complete"])
def test_main_names_the_status_of_a_run_that_did_not_complete(
    status: str, layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = build_run(tmp_path, status=status)
    module = redirect(monkeypatch, layout)
    assert module.main([str(run.run_dir)]) == 0
    out = capsys.readouterr().out
    assert ("not 'complete'" in out) is (status != "complete"), out
    if status != "complete":
        assert f"status is {status!r}" in out, out
    fixtures = exported(layout.dest)
    assert {data["provenance"]["status"] for data in fixtures.values()} == {status}


def test_main_refuses_another_destination_with_status_2(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = build_run(tmp_path)
    module = redirect(monkeypatch, layout)
    elsewhere = tmp_path / "elsewhere"
    assert module.main([str(run.run_dir), "--dest", str(elsewhere)]) == 2
    assert capsys.readouterr().err.strip(), "the refusal printed no message"
    assert not elsewhere.exists()
    assert written(layout.dest) == []


def test_the_script_refuses_another_destination_when_run_alone(layout: Layout, tmp_path: Path) -> None:
    tool()  # the script exists
    run = build_run(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    result = subprocess.run(
        [sys.executable, str(TOOL), str(run.run_dir), "--dest", str(elsewhere)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 2, result.stderr
    assert result.stderr.strip()
    assert "Traceback" not in result.stderr and "usage:" not in result.stderr, result.stderr
    assert not elsewhere.exists()


# ---------------------------------------------------------------------------
# Manifests and held-out inputs


def test_an_export_changes_no_manifest_and_no_held_out_input(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    module = redirect(monkeypatch, layout)
    bench_before = snapshot(BENCH_DIR)
    held_before = held_out_digests(tmp_path / "held-out-before")
    assert held_before, "no manifest declares held-out inputs"
    with repo_guard() as changes:
        module.export_counterexamples(run.run_dir)
        for dest in (BENCH_DIR, BENCH_DIR / SUITE):
            with pytest.raises(ValueError):
                module.export_counterexamples(run.run_dir, dest)
        assert changes() == {}, "an export changed the repository"
    assert written(layout.dest), "the export wrote nothing"
    assert snapshot(BENCH_DIR) == bench_before
    assert held_out_digests(tmp_path / "held-out-after") == held_before


# ---------------------------------------------------------------------------
# No registry reader reaches a fixture


def test_load_suite_refuses_every_fixture(layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = build_run(tmp_path)
    redirect(monkeypatch, layout).export_counterexamples(run.run_dir)
    files = written(layout.dest)
    assert files, "the export wrote nothing"
    for path in files:
        with pytest.raises(ValueError):
            load_suite(path)


def test_no_manifest_lists_a_fixture(layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = build_run(tmp_path)
    before = manifest_views()
    redirect(monkeypatch, layout).export_counterexamples(run.run_dir)
    files = written(layout.dest)
    assert files, "the export wrote nothing"
    after = manifest_views()
    assert after == before
    bench_files = {path.name for path in BENCH_DIR.rglob("*")}
    for path in files:
        assert path.name not in bench_files
        for view in after.values():
            assert path.name not in view
    for view in after.values():
        assert "counterexamples" not in view


def test_suite_item_keeps_a_fixtures_item_held_and_unchanged(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = build_run(tmp_path)
    suite_before = load_suite(BENCH_DIR / f"{SUITE}.yaml")
    redirect(monkeypatch, layout).export_counterexamples(run.run_dir)
    fixtures = exported(layout.dest)
    assert fixtures, "the export wrote nothing"
    for data in fixtures.values():
        suite = load_suite(BENCH_DIR / f"{data['bench_item']['suite']}.yaml")
        name = data["bench_item"]["item"]
        for purpose in HELD_PURPOSES:
            with pytest.raises(EvalSplitError):
                suite.item(name, purpose=purpose)
        item = suite.item(name, purpose="eval")
        assert repr(item) == repr(suite_before.item(name, purpose="eval"))
        assert "counterexamples" not in repr(item)


def test_train_view_never_lists_a_fixture(layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = build_run(tmp_path)

    def views() -> dict[str, tuple[list[str], dict[str, str]]]:
        found = {}
        for path in sorted(BENCH_DIR.glob("*.yaml")):
            view = load_suite(path).train_view()
            found[path.name] = (sorted(view.train_items), dict(view.held))
        return found

    before = views()
    redirect(monkeypatch, layout).export_counterexamples(run.run_dir)
    fixtures = exported(layout.dest)
    assert fixtures, "the export wrote nothing"
    assert views() == before
    view = load_suite(BENCH_DIR / f"{SUITE}.yaml").train_view()
    for path in written(layout.dest):
        assert path.stem not in view.train_items and "counterexamples" not in repr(view)
    for data in fixtures.values():
        with pytest.raises(EvalSplitError):
            view.item(data["bench_item"]["item"])


# ---------------------------------------------------------------------------
# No train data source reaches a fixture


def test_load_synthetic_refuses_every_fixture(layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = build_run(tmp_path)
    redirect(monkeypatch, layout).export_counterexamples(run.run_dir)
    files = written(layout.dest)
    assert files, "the export wrote nothing"
    assert train_data.load_synthetic(TRAIN_FIXTURE, layout.train).records  # the train directory itself reads
    for path in files:
        for name in (relative_name(path, layout.train), str(path)):
            with pytest.raises(RunError):
                train_data.load_synthetic(name, layout.train)


def test_a_train_recipe_cannot_name_a_fixture(layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = build_run(tmp_path)
    redirect(monkeypatch, layout).export_counterexamples(run.run_dir)
    files = written(layout.dest)
    assert files, "the export wrote nothing"
    recipes = _fresh(tmp_path / "recipes")
    for path in files:
        for name in (f"../adversary/counterexamples/{path.name}", path.name, str(path)):
            recipe = recipes / "train.yaml"
            recipe.write_bytes(
                (
                    "base_model: fixture:tiny-causal-lm\nmethod: sft\nweights: full\n"
                    "trainer: {kind: trl, device: {kind: cpu}}\n"
                    f"data: {{synthetic: {json.dumps(name)}}}\n"
                ).encode("ascii")
            )
            with pytest.raises(RecipeError, match="data.synthetic"):
                load_train_recipe(recipe, roots=[recipes])


def test_load_bench_never_reaches_a_fixture(layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = build_run(tmp_path)
    redirect(monkeypatch, layout).export_counterexamples(run.run_dir)
    fixtures = exported(layout.dest)
    assert fixtures, "the export wrote nothing"
    recipe_path = tmp_path / "train.yaml"
    for path in written(layout.dest):
        relative = relative_name(path.with_suffix(""), BENCH_DIR)
        recipe = SimpleNamespace(path=recipe_path, data={"bench": {"suite": relative, "split": "train"}})
        with pytest.raises(RunError):
            train_data.load_bench(recipe, BENCH_DIR)
    for data in fixtures.values():
        bench = {"suite": data["bench_item"]["suite"], "split": "train", "items": [data["bench_item"]["item"]]}
        recipe = SimpleNamespace(path=recipe_path, data={"bench": bench})
        with pytest.raises(RunError, match="training may not read"):
            train_data.load_bench(recipe, BENCH_DIR)


# ---------------------------------------------------------------------------
# The hand-built world


def test_the_hand_built_run_tree_loads_and_holds_every_divergences_inputs(tmp_path: Path) -> None:
    """A check of the fixture world, not of the tool: every trial loads, and every recorded input is in the store."""
    run = build_run(tmp_path)
    store, blobs = TextStore(run.run_dir), BlobStore(run.run_dir)
    paths = sorted(run.run_dir.glob("/".join(["*"] * 6 + [TRIAL_JSON])))
    assert len(paths) == len(PLANS)
    records = [trial.adversary for trial in (read_trial(path, store) for path in paths) if trial.adversary]
    divergences = [divergence for record in records for divergence in record.divergences]
    assert len(divergences) == len(run.expected)
    recorded = {digest for record in records for entry in record.divergences for digest in entry.inputs.values()}
    assert [digest for digest in sorted(recorded | run.finding_digests) if digest not in blobs] == []
    assert {entry.key()[1] for entry in run.expected} == {
        frozenset(divergence.inputs.items()) for divergence in divergences
    }
