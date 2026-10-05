"""Tests for the baseline's reference agreement and binary_io runs end to end (task P4.4).

Bible: Oracles (binary_io row: thresholds per kernel from reference runs),
Result Record (Trial.reference_run, end_reason, Storage), Harness Contract
(lassi_io output files), Benchmark Suites (item manifest), Evaluation
Protocol > LASSI Score Profile (the baseline ends), Component Interfaces
(Oracle; the capabilities rule). Plan: plans/p4-ttsim.md, P4.4, the planning
decision "binary_io (P4.4)", and the P4 constraint that a new end reason
ending a trial at the baseline joins the lassi profile's baseline-end set.

The contract these tests fix:

- A suite manifest item may declare `tolerance: {metric: <pcc | max_abs |
  ulp>, threshold: <number>}`, the tolerance its two references must meet;
  load_suite refuses a tolerance with another metric, a missing key, or a
  threshold that is not a number, with a ValueError naming `tolerance`.
- When the recipe binds an Oracle that declares ALIGNS_OUTPUT_FILES and the
  executor runs programs, the baseline keeps the target reference's output
  files in the binary store (Trial.reference_run.outputs: file -> sha256 in
  BlobStore(<run dir>)). With fixes.baseline_both on, it also runs the
  source reference and records the references' agreement,
  Trial.reference_agreement: one OutputStats per output of the target
  reference against the source reference's, with pcc, max_abs, and max_ulp,
  and passed judged against the item's declared tolerance under that
  tolerance's own metric (which may differ from the recipe's).
- A pair whose agreement misses the item's declared tolerance ends the trial
  at the baseline with final.end_reason `baseline-disagree` (the message
  names the tolerance's metric): no model is asked, no attempt is made, and
  the agreement stays in the record.
- run_loop keeps each attempt run's output files the same way
  (Attempt.run.outputs), and the oracle stage aligns them (see
  tests/oracles/test_binary_io_stage.py); `threshold: from_baseline` takes
  each output's threshold from the recorded agreement.
- trial.json, trial.md, and the Parquet mirror in the run tree carry the
  agreement and each attempt's statistics.
- lassi.scoring.lassi_profile.BASELINE_ENDS holds `baseline-disagree`, so a
  trial that ended there scores correct, first_try, compiled,
  compiled_first_try, and the similarity values null, each noted with the
  profile's `baseline` note and the end code.

The suite manifest, sources, replies, and output arrays are SYNTHETIC and
written for these tests; fake toolchains compile nothing and the scripted
executor runs nothing. Each expected statistic is computed by hand in the
comment beside it; SYNTHETIC_WALL_S is a fixture value. No value here is a
measurement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.bench import load_suite
from lassi.core import record as record_module
from lassi.core import runner as runner_module
from lassi.core import store as store_module
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Limits, Message, RunResult, Sampling
from lassi.core.parquet import read_run_parquet
from lassi.core.record import (
    BenchItem,
    EndReason,
    Final,
    ModelInfo,
    Provenance,
    RunInfo,
    Trial,
    make_trial_id,
)
from lassi.core.registry import DEFAULT_REGISTRY, Registry, RegistryError
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.harness.lassi_io import LassiArray, write_array

REPO = Path(__file__).resolve().parents[2]
PROFILE_FILE = REPO / "assets" / "scoring" / "lassi.yaml"
SUITE = "binio-fixture"
ITEM = "vadd"  # declares tolerance max_abs 0.01, which the references meet
STRICT = "vadd-strict"  # declares tolerance max_abs 0.001, which the references miss
MODEL_ID = "scripted-fixture"
SOURCE, TARGET = "cpu", "tt"
DIRECTION = f"{SOURCE}-{TARGET}"
TARGET_FILE = "host.cpp"
TOOLCHAINS = {SOURCE: "fake-cpu", TARGET: "fake-tt"}
STAGES = ["baseline", "generate", "compile_loop", "run_loop", "oracle"]
DISAGREE = "baseline-disagree"
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
# A commit id for the synthetic manifest and provenance; not a commit of this repository or of any source.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SYNTHETIC_WALL_S = 1.25

# SYNTHETIC sources and reply; the fakes never compile or run them.
SOURCES = {
    SOURCE: "// SYNTHETIC cpu reference of a made-up vector add\nint main() { return 0; }\n",
    TARGET: "// SYNTHETIC tt reference host program of a made-up vector add\nint main() { return 0; }\n",
}
REPLY = render_file_blocks({TARGET_FILE: "// SYNTHETIC candidate host program\nint main() { return 0; }\n"})


def f32(*values: float, name: str = "c") -> LassiArray:
    """Return a SYNTHETIC f32 array of `values` (each exactly representable in f32)."""
    return LassiArray(name, "f32", (len(values),), struct.pack(f"<{len(values)}f", *values))


# The references' outputs: tt differs from cpu by 2**-7 = 0.0078125 on c[2], which is 2**15 = 32768 f32 units of
# 2**-22 in [2, 4). The item vadd allows 0.01, so the references agree; vadd-strict allows 0.001, so they do not.
REFERENCE_OUTPUTS = {SOURCE: [f32(1.0, 2.0, 3.0)], TARGET: [f32(1.0, 2.0, 3.0078125)]}
AGREEMENT_MAX_ABS = 0.0078125
AGREEMENT_MAX_ULP = 32768
# A candidate 2**-8 = 0.00390625 (16384 units) from the target reference, within the agreement.
NEAR = [f32(1.0, 2.0, 3.00390625)]
NEAR_MAX_ABS = 0.00390625
NEAR_MAX_ULP = 16384
# A candidate 0.4921875 from the target reference (3.5 - 3.0078125), past the agreement.
FAR = [f32(1.0, 2.0, 3.5)]
FAR_MAX_ABS = 0.4921875


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate and compile variables a test could inherit, and point TMPDIR at a test directory."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CPATH", raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))


# ---------------------------------------------------------------------------
# Names this task adds, looked up so a missing one fails its test with a clear message


def registered(interface: str, name: str) -> type:
    """Return the class registered as `name` for `interface`; fail the test clearly while it is missing."""
    try:
        return DEFAULT_REGISTRY.get(interface, name).factory
    except RegistryError as error:
        pytest.fail(f"no {interface} is registered as {name!r} ({error}); task P4.4 adds it")


def blob_store(root: Path) -> Any:
    """Return lassi.core.store.BlobStore(root); fail the test clearly while the class is missing."""
    cls = getattr(store_module, "BlobStore", None)
    if cls is None:
        pytest.fail("lassi.core.store has no BlobStore; task P4.4 adds the binary store beside the text store")
    return cls(root)


def require_field(cls: type, name: str) -> None:
    """Fail the test clearly while the record class `cls` has no field `name`."""
    if name not in {spec.name for spec in dataclasses.fields(cls)}:
        pytest.fail(f"{cls.__name__} has no {name} field; task P4.4 adds it to the Result Record")


def require_disagree_code() -> None:
    """Fail the test clearly while END_REASONS lacks baseline-disagree."""
    if DISAGREE not in record_module.END_REASONS:
        pytest.fail(f"lassi.core.record.END_REASONS has no {DISAGREE!r}; task P4.4 adds it")


# ---------------------------------------------------------------------------
# The SYNTHETIC suite


def manifest_data(**tolerances: Any) -> dict[str, Any]:
    """Return the SYNTHETIC binio-fixture manifest; `tolerances` replace an item's tolerance (None drops it)."""
    languages = {
        SOURCE: {"dir": "src/vadd-cpu", "files": ["main.cpp"]},
        TARGET: {"dir": "src/vadd-tt", "files": [TARGET_FILE]},
    }
    items: dict[str, Any] = {
        ITEM: {"split": "eval", "languages": languages, "tolerance": {"metric": "max_abs", "threshold": 0.01}},
        STRICT: {"split": "eval", "languages": languages, "tolerance": {"metric": "max_abs", "threshold": 0.001}},
    }
    for item, tolerance in tolerances.items():
        item = item.replace("_", "-")
        if tolerance is None:
            items[item].pop("tolerance")
        else:
            items[item]["tolerance"] = tolerance
    return {"suite": SUITE, "repo": "https://example.invalid/binio-fixture", "commit": FAKE_COMMIT, "items": items}


def write_manifest(directory: Path, data: Mapping[str, Any]) -> Path:
    """Write `data` as `<directory>/binio-fixture.yaml` and return the file."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{SUITE}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


def write_bench(root: Path) -> Path:
    """Write the SYNTHETIC sources where the manifest lays them out, and return `root`."""
    for language, layout in manifest_data()["items"][ITEM]["languages"].items():
        path = root / layout["dir"] / layout["files"][0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(SOURCES[language].encode("ascii"))
    return root


def test_a_suite_item_may_declare_the_tolerance_its_references_must_meet(tmp_path: Path) -> None:
    suite = load_suite(write_manifest(tmp_path, manifest_data()))
    assert set(suite.items) == {ITEM, STRICT}


@pytest.mark.parametrize(
    "tolerance",
    [
        {"metric": "rmse", "threshold": 0.01},
        {"metric": "max_abs"},
        {"threshold": 0.01},
        {"metric": "max_abs", "threshold": "loose"},
        {"metric": "max_abs", "threshold": 0.01, "extra": 1},
    ],
    ids=["unknown-metric", "no-threshold", "no-metric", "word-threshold", "unknown-key"],
)
def test_a_bad_tolerance_is_refused_naming_it(tmp_path: Path, tolerance: dict[str, Any]) -> None:
    load_suite(write_manifest(tmp_path / "good", manifest_data()))  # a valid tolerance loads, so the key is known
    with pytest.raises(ValueError, match="tolerance"):
        load_suite(write_manifest(tmp_path / "bad", manifest_data(vadd=tolerance)))


# ---------------------------------------------------------------------------
# Fake components


@dataclass
class Script:
    """What the fakes answer and what they saw: replies, output arrays per run, and events in order."""

    replies: list[str] = field(default_factory=list)
    attempt_outputs: list[list[LassiArray]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


def fake_toolchain(registered_as: str, script: Script) -> type:
    """Return a Toolchain class without PIN that writes the files and a PLACEHOLDER artifact; it compiles nothing."""

    class FakeToolchain:
        """Writes the files under the fresh workdir and reports a built artifact."""

        name = registered_as
        capabilities = frozenset({"diagnostics"})

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Record the build, write `files` and a placeholder artifact, and return it with no diagnostics."""
            script.events.append(f"build {registered_as}")
            for path, text in [*files.items(), *(harness or {}).items()]:
                target = Path(workdir) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            artifact = Path(workdir) / "main"
            artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=[])

    return FakeToolchain


def scripted_executor(script: Script) -> type:
    """Return an Executor class that runs programs in name only: it writes the scripted output files.

    A baseline run (its workdir is <trial>/baseline-<language>/build) writes
    that language's REFERENCE_OUTPUTS; an attempt run writes the next entry
    of script.attempt_outputs. Each array goes to `<name>.lassiio` in the
    workdir, and RunResult.output_files names them. It starts no process.
    """

    class ScriptedExecutor:
        """Writes the scripted lassi_io output files and returns a clean SYNTHETIC RunResult."""

        name = "scripted"
        capabilities = frozenset({"runs_code", "sandboxed"})

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Write this run's output files beside the artifact and return them in a clean RunResult."""
            workdir = Path(artifact).parent
            kind = workdir.parent.name
            if kind.startswith("baseline-"):
                language = kind[len("baseline-"):]
                script.events.append(f"run:reference:{language}")
                arrays = REFERENCE_OUTPUTS[language]
            else:
                script.events.append("run:attempt")
                assert script.attempt_outputs, f"the executor was asked to run {artifact} past the script"
                arrays = script.attempt_outputs.pop(0)
            files = {}
            for array in arrays:
                path = workdir / f"{array.name}.lassiio"
                write_array(path, array.name, array.dtype, array.shape, array.data)
                files[path.name] = path
            return RunResult(
                exit_code=0, hang=False, stdout="SYNTHETIC run stdout\n", stderr="", output_files=files,
                wall_s=SYNTHETIC_WALL_S,
            )

    return ScriptedExecutor


def scripted_backend(script: Script) -> type:
    """Return an LLMBackend class, registered as "scripted", that answers from script.replies in order."""

    class ScriptedBackend:
        """Records each request and answers with the next scripted reply."""

        name = "scripted"
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id, as every backend does."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Record the request and return the next scripted reply."""
            script.events.append("complete")
            assert script.replies, "the backend was asked for more replies than the script holds"
            return Completion(text=script.replies.pop(0), prompt_tokens=0, completion_tokens=0)

    return ScriptedBackend


def make_registry(script: Script) -> Registry:
    """Return a test Registry: the scripted backend and executor, fake toolchains, the real stages and binary_io."""
    registry = Registry()
    registry.register("LLMBackend", "scripted", scripted_backend(script))
    registry.register("Executor", "scripted", scripted_executor(script))
    for name in TOOLCHAINS.values():
        registry.register("Toolchain", name, fake_toolchain(name, script))
    registry.register("Oracle", "binary_io", registered("Oracle", "binary_io"))
    for name in STAGES:
        registry.register("Stage", name, registered("Stage", name))
    return registry


# ---------------------------------------------------------------------------
# Runs


@dataclass
class Outcome:
    """One finished run: its directory, its one trial read back from the run tree, and the script."""

    run_dir: Path
    trial_id: str
    trial: Trial
    script: Script


def recipe_data(item: str, metric: str, threshold: Any) -> dict[str, Any]:
    """Return a one-trial recipe for `item` from cpu to tt with binary_io bound and baseline_both on."""
    return {
        "extends": "base",
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [item]},
        "directions": [{"source": SOURCE, "target": TARGET}],
        "prompts": "p0-smoke",
        "toolchain": dict(TOOLCHAINS),
        "stages": list(STAGES),
        "executor": {"kind": "scripted"},
        "oracle": {"kind": "binary_io", "metric": metric, "threshold": threshold},
        "fixes": {"baseline_both": True},
        "trials": {"n": 1},
    }


def run_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    item: str,
    candidate: Sequence[LassiArray],
    metric: str = "max_abs",
    threshold: Any = "from_baseline",
) -> Outcome:
    """Run one trial of `item` whose attempt writes `candidate`, on the SYNTHETIC suite; return the outcome."""
    monkeypatch.setattr(runner_module, "BENCH_DIR", write_manifest(tmp_path / "manifests", manifest_data()).parent)
    script = Script(replies=[REPLY], attempt_outputs=[list(candidate)])
    options = RunOptions(
        runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=write_bench(tmp_path / "bench"),
        registry=make_registry(script),
    )
    recipe = tmp_path / "binio-run.yaml"
    recipe.write_bytes(yaml.safe_dump(recipe_data(item, metric, threshold), sort_keys=False).encode("ascii"))
    run_dir = run_recipe(recipe, options)
    trial_id = make_trial_id("binio-run", MODEL_ID, SUITE, DIRECTION, item, 1)
    trial = read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir))
    return Outcome(run_dir, trial_id, trial, script)


def lassiio_bytes(tmp_path: Path, array: LassiArray) -> bytes:
    """Return the bytes of `array` as a lassi_io file."""
    path = tmp_path / "expected" / f"{array.name}.lassiio"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_array(path, array.name, array.dtype, array.shape, array.data)
    return path.read_bytes()


def sha(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def by_name(stats: Sequence[Any] | None, where: str) -> dict[str, Any]:
    """Return a list of OutputStats keyed by name; fail the test when it is None."""
    assert stats is not None, f"{where} holds no per-output statistics"
    return {entry.name: entry for entry in stats}


def agreement_of(trial: Trial) -> dict[str, Any]:
    """Return the trial's recorded reference agreement by output name."""
    require_field(Trial, "reference_agreement")
    return by_name(trial.reference_agreement, "Trial.reference_agreement")


# ---------------------------------------------------------------------------
# The baseline and the run, end to end


@pytest.mark.parametrize(
    ("metric", "threshold"),
    [("max_abs", "from_baseline"), ("max_abs", 0.01), ("pcc", 0.99)],
    ids=["from-baseline", "numeric-max-abs", "numeric-pcc"],
)
def test_the_baseline_records_the_references_agreement_and_keeps_the_target_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metric: str, threshold: Any
) -> None:
    outcome = run_one(tmp_path, monkeypatch, ITEM, NEAR, metric, threshold)
    trial = outcome.trial
    assert trial.final.end_reason is None
    first_request = outcome.script.events.index("complete")
    assert {"run:reference:tt", "run:reference:cpu"} <= set(outcome.script.events[:first_request])
    require_field(RunInfo, "outputs")
    target_bytes = lassiio_bytes(tmp_path, REFERENCE_OUTPUTS[TARGET][0])
    assert trial.reference_run.outputs == {"c.lassiio": sha(target_bytes)}
    assert blob_store(outcome.run_dir).get(sha(target_bytes)) == target_bytes
    agreement = agreement_of(trial)
    assert set(agreement) == {"c"}
    assert agreement["c"].max_abs == AGREEMENT_MAX_ABS
    assert agreement["c"].max_ulp == AGREEMENT_MAX_ULP
    assert agreement["c"].passed is True, "the item's tolerance of max_abs 0.01 holds the references' 0.0078125"


@pytest.mark.parametrize(
    ("metric", "threshold"),
    [("max_abs", "from_baseline"), ("max_abs", 0.01), ("pcc", 0.99)],
    ids=["from-baseline", "numeric-max-abs", "numeric-pcc"],
)
def test_a_candidate_within_its_threshold_aligns_at_one_and_its_outputs_are_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metric: str, threshold: Any
) -> None:
    outcome = run_one(tmp_path, monkeypatch, ITEM, NEAR, metric, threshold)
    attempt = outcome.trial.attempts[-1]
    assert attempt.stage_reached == "S5"
    require_field(RunInfo, "outputs")
    candidate_bytes = lassiio_bytes(tmp_path, NEAR[0])
    assert attempt.run.outputs == {"c.lassiio": sha(candidate_bytes)}
    assert blob_store(outcome.run_dir).get(sha(candidate_bytes)) == candidate_bytes
    assert (attempt.alignment.per_input, attempt.alignment.mean) == ([1.0], 1.0)
    stats = by_name(attempt.alignment.outputs, "Attempt.alignment.outputs")
    assert stats["c"].passed is True
    assert (stats["c"].max_abs, stats["c"].max_ulp) == (NEAR_MAX_ABS, NEAR_MAX_ULP)
    assert outcome.trial.final.alignment == 1.0


def test_a_candidate_past_the_references_agreement_aligns_at_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = run_one(tmp_path, monkeypatch, ITEM, FAR)
    attempt = outcome.trial.attempts[-1]
    assert attempt.stage_reached == "S5", "an oracle mismatch is not a run error"
    assert (attempt.alignment.per_input, attempt.alignment.mean) == ([0.0], 0.0)
    stats = by_name(attempt.alignment.outputs, "Attempt.alignment.outputs")
    assert stats["c"].passed is False and stats["c"].max_abs == FAR_MAX_ABS
    assert outcome.trial.final.alignment == 0.0


def test_a_pair_outside_its_tolerance_ends_at_the_baseline_before_any_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    require_disagree_code()
    outcome = run_one(tmp_path, monkeypatch, STRICT, NEAR)
    trial = outcome.trial
    assert trial.final.end_reason is not None and trial.final.end_reason.code == DISAGREE
    assert "max_abs" in trial.final.end_reason.message, "the message names the tolerance's metric"
    assert "complete" not in outcome.script.events, "no model is asked after the baseline ends the trial"
    assert trial.attempts == [] and trial.requests == []
    assert trial.final.stage_reached is None
    agreement = agreement_of(trial)
    assert agreement["c"].passed is False
    assert agreement["c"].max_abs == AGREEMENT_MAX_ABS
    require_field(RunInfo, "outputs")
    assert trial.reference_run.outputs == {"c.lassiio": sha(lassiio_bytes(tmp_path, REFERENCE_OUTPUTS[TARGET][0]))}


def test_trial_md_and_the_parquet_mirror_in_the_run_tree_show_the_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = run_one(tmp_path, monkeypatch, ITEM, NEAR)
    page = (trial_dir(outcome.run_dir, outcome.trial_id) / "trial.md").read_text(encoding="ascii")
    first_attempt = page.index("## Attempt 0")
    agreement_rows = [line for line in page[:first_attempt].splitlines() if _row_has(line, "c", "0.0078125", "32768")]
    attempt_rows = [line for line in page[first_attempt:].splitlines() if _row_has(line, "c", "0.00390625", "16384")]
    assert agreement_rows, "trial.md shows no row of the reference agreement before the attempts"
    assert attempt_rows, "trial.md shows no row of attempt 0's statistics"
    rows = read_run_parquet(outcome.run_dir / "parquet").get("output_stats")
    assert rows is not None, "the run's Parquet mirror has no output_stats table"
    picked = sorted(
        ((row["attempt_index"], row["name"], row["max_abs"], row["max_ulp"], row["passed"]) for row in rows),
        key=lambda row: (row[0] is not None, row[0] or 0),
    )
    expected = [(None, "c", AGREEMENT_MAX_ABS, AGREEMENT_MAX_ULP, True), (0, "c", NEAR_MAX_ABS, NEAR_MAX_ULP, True)]
    assert picked == expected


def _row_has(line: str, *cells: str) -> bool:
    """Return True when `line` is a Markdown table row whose cells include each of `cells`."""
    if not line.startswith("|"):
        return False
    found = {cell.strip() for cell in line.strip().strip("|").split("|")}
    return set(cells) <= found


# ---------------------------------------------------------------------------
# The lassi profile at the new baseline end


def disagree_trial(tmp_path: Path) -> Trial:
    """Return a SYNTHETIC trial that ended at baseline-disagree after a clean target reference run."""
    require_disagree_code()
    store = TextStore(tmp_path / "store")
    reference = RunInfo(
        exit_code=0, hang=False, wall_s=SYNTHETIC_WALL_S, stdout_ref=store.put("SYNTHETIC run stdout\n"),
        stdout_truncated=False, stderr_truncated=False, workdir_incomplete=False,
    )
    reason = EndReason(code=DISAGREE, message="SYNTHETIC: the references disagree on c past max_abs 0.001")
    return Trial(
        trial_id=make_trial_id("binio-run", MODEL_ID, SUITE, DIRECTION, STRICT, 1),
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(
            commit=FAKE_COMMIT, dirty=False, device="SYNTHETIC device", sdk=None, date="2026-09-25T00:00:00+00:00"
        ),
        bench_item=BenchItem(suite=SUITE, item=STRICT, split="eval", direction=DIRECTION),
        model=ModelInfo(backend="scripted", id=MODEL_ID, sampling=SAMPLING),
        reference_run=reference,
        requests=[],
        final=Final(corrections=0, end_reason=reason),
    )


def test_baseline_disagree_joins_the_lassi_profiles_baseline_ends() -> None:
    from lassi.scoring.lassi_profile import BASELINE_ENDS

    assert DISAGREE in BASELINE_ENDS


def test_the_lassi_profile_leaves_correct_and_compiled_null_at_baseline_disagree(tmp_path: Path) -> None:
    from lassi.scoring.lassi_profile import LassiProfile

    trial = disagree_trial(tmp_path)
    (tmp_path / "bench").mkdir()
    score = LassiProfile(bench_root=tmp_path / "bench").score(trial)
    unscored = ("correct", "first_try", "compiled", "compiled_first_try", "sim_t", "sim_t_c", "sim_l")
    assert {name: score.components[name] for name in unscored} == dict.fromkeys(unscored)
    assert score.scalar is None
    baseline_note = yaml.safe_load(PROFILE_FILE.read_text(encoding="ascii"))["notes"]["baseline"]
    for name in unscored:
        assert score.notes[name].startswith(baseline_note) and DISAGREE in score.notes[name], name
