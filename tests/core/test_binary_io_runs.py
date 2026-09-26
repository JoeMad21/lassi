"""Tests for binary_io runs in the cases the P4.4 tests leave open: refusals, reference problems, views.

Bible: Oracles (binary_io rules), Result Record (RunInfo.outputs, the
baseline end reasons, Storage), Component Interfaces (the Oracle
capabilities). Plan: plans/p4-ttsim.md, P4.4. The decisions these tests fix
(task P4.4):

- The runner refuses, before any directory is created, threshold
  from_baseline for an item that declares no tolerance (the message names
  the item and from_baseline) or when no listed stage builds the references,
  and an Oracle that declares neither capability the oracle stage accepts.
- baseline and run_loop keep output files (RunInfo.outputs, the binary
  store) only when the recipe's Oracle declares aligns_output_files; a
  stdout_mask recipe keeps none and writes no blobs/ directory.
- A target reference output file that is not a readable lassi_io file, or a
  reference run that wrote no output file, ends the trial at baseline-run
  before any model call, the message naming the file.
- Without a declared tolerance and with a numeric threshold, the baseline
  records no agreement and the trial goes on. Only the target reference's
  output files enter the binary store; the source reference's are compared
  but never stored.
- The Parquet mirror carries RunInfo.outputs as JSON text with sorted keys
  in reference_run_outputs and run_outputs; trial.md shows the file count
  and a table of files and hashes. output_stats rows come in the order
  written (the agreement's, attempt_index null, first; then by attempt and
  ordinal), max_ulp is an unsigned 64-bit column, and a max_ulp of 2**64 or
  more is refused with a ValueError before anything is written.
- A reference run whose workdir came back incomplete (as P2.2 treats a
  truncated reference stdout): its output files are not checked, so the cut
  alone ends nothing. When it is the target's, the oracle stage aligns
  nothing against it: each attempt whose run recorded output files gets one
  run-stage warning, code reference-workdir-incomplete, and its alignment
  stays unset, with from_baseline too. When either reference's workdir is
  incomplete, the baseline records no agreement and notes it in
  Trial.baseline_diagnostics (shown in trial.md and in the Parquet
  diagnostics table with attempt_index null); from_baseline then raises at
  the oracle stage, naming the note, as it does whenever no agreement was
  recorded.

The manifest, sources, replies, and arrays are SYNTHETIC; fake toolchains
compile nothing and the scripted executor runs nothing. No value here is a
measurement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from lassi.core import runner as runner_module
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Limits, Message, RunResult, Sampling
from lassi.core.parquet import read_run_parquet, trial_rows, write_run_parquet
from lassi.core.record import (
    Alignment,
    Attempt,
    BenchItem,
    Diagnostic,
    ModelInfo,
    OutputStats,
    Provenance,
    RunInfo,
    Trial,
    from_dict,
    from_json,
    make_trial_id,
    to_json,
)
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.stages import RunContext
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.core.trial_md import render_trial_md
from lassi.harness.lassi_io import LassiArray, write_array

SUITE = "binio-runs"
SOURCE, TARGET = "cpu", "tt"
TOOLCHAINS = {SOURCE: "fake-cpu", TARGET: "fake-tt"}
STAGES = ["baseline", "generate", "compile_loop", "run_loop", "oracle"]
MODEL_ID = "scripted-fixture"
# A commit id for the synthetic manifest; not a commit of this repository or of any source.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
REPLY = render_file_blocks({"host.cpp": "// SYNTHETIC candidate\nint main() { return 0; }\n"})
JUNK = b"SYNTHETIC: not a lassi_io file\n"


def f32(*values: float, name: str = "c") -> LassiArray:
    """Return a SYNTHETIC f32 array of `values`."""
    return LassiArray(name, "f32", (len(values),), struct.pack(f"<{len(values)}f", *values))


GOOD = [f32(1.0, 2.0)]


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate and compile variables a test could inherit, and point TMPDIR at a test directory."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CPATH", raising=False)
    (tmp_path / "compile-tmp").mkdir()
    monkeypatch.setenv("TMPDIR", str(tmp_path / "compile-tmp"))


# ---------------------------------------------------------------------------
# The SYNTHETIC suite and fakes


def write_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tolerance: Any) -> Path:
    """Write the one-item manifest (item vadd, with `tolerance` unless None) and its sources; return the bench root."""
    languages = {SOURCE: {"dir": "src/cpu", "files": ["main.cpp"]}, TARGET: {"dir": "src/tt", "files": ["host.cpp"]}}
    item: dict[str, Any] = {"split": "eval", "languages": languages}
    if tolerance is not None:
        item["tolerance"] = tolerance
    repo = "https://example.invalid/binio-runs"
    data = {"suite": SUITE, "repo": repo, "commit": FAKE_COMMIT, "items": {"vadd": item}}
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / f"{SUITE}.yaml").write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    monkeypatch.setattr(runner_module, "BENCH_DIR", manifests)
    root = tmp_path / "bench"
    for language in languages.values():
        path = root / language["dir"] / language["files"][0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"// SYNTHETIC source\nint main() { return 0; }\n")
    return root


@dataclass
class Script:
    """What the fakes write and answer: reference outputs per language (arrays or raw files), attempt outputs."""

    references: dict[str, Mapping[str, bytes]]
    attempts: list[Mapping[str, bytes]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    cut: frozenset[str] = frozenset()  # run kinds (baseline-<language>) whose RunResult reports workdir_incomplete


def sha(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def lassiio(array: LassiArray, scratch: Path) -> bytes:
    """Return the bytes of `array` as a lassi_io file."""
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / f"{array.name}.lassiio"
    write_array(path, array.name, array.dtype, array.shape, array.data)
    return path.read_bytes()


class FakeToolchain:
    """Writes the files and a PLACEHOLDER artifact under the fresh workdir; compiles nothing."""

    name = "fake"
    capabilities = frozenset({"diagnostics"})

    def build(
        self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
    ) -> BuildResult:
        """Write `files` and a placeholder artifact and return it with no diagnostics."""
        for path, text in files.items():
            (Path(workdir) / path).parent.mkdir(parents=True, exist_ok=True)
            (Path(workdir) / path).write_bytes(text.encode("utf-8"))
        artifact = Path(workdir) / "main"
        artifact.write_bytes(b"PLACEHOLDER artifact\n")
        return BuildResult(artifact=artifact, diagnostics=[])


def scripted_executor(script: Script) -> type:
    """Return an Executor class that writes the scripted files beside the artifact; it runs nothing."""

    class ScriptedExecutor:
        """Writes the scripted output files and returns a clean SYNTHETIC RunResult."""

        name = "scripted"
        capabilities = frozenset({"runs_code", "sandboxed"})

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Write this run's scripted output files (a reference's by language, else the next attempt's)."""
            workdir = Path(artifact).parent
            kind = workdir.parent.name
            if kind.startswith("baseline-"):
                written = script.references[kind.removeprefix("baseline-")]
            else:
                written = script.attempts.pop(0)
            script.events.append(f"run {kind}")
            files = {}
            for file, data in written.items():
                (workdir / file).write_bytes(data)
                files[file] = workdir / file
            return RunResult(
                exit_code=0, hang=False, stdout="SYNTHETIC\n", stderr="", output_files=files, wall_s=1.0,
                workdir_incomplete=kind in script.cut,
            )

    return ScriptedExecutor


def scripted_backend(script: Script) -> type:
    """Return an LLMBackend class that answers every request with REPLY and records it."""

    class ScriptedBackend:
        """Answers every request with REPLY."""

        name = "scripted"
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Record the request and return REPLY."""
            script.events.append("complete")
            return Completion(text=REPLY, prompt_tokens=0, completion_tokens=0)

    return ScriptedBackend


def make_registry(
    script: Script, oracle_stage: type | None = None, oracle: tuple[str, type] | None = None
) -> Registry:
    """Return a test Registry: scripted backend and executor, fake toolchains, the real stages, and an Oracle."""
    registry = Registry()
    registry.register("LLMBackend", "scripted", scripted_backend(script))
    registry.register("Executor", "scripted", scripted_executor(script))
    for name in TOOLCHAINS.values():
        registry.register("Toolchain", name, FakeToolchain)
    oracle_name, oracle_class = oracle or ("binary_io", DEFAULT_REGISTRY.get("Oracle", "binary_io").factory)
    registry.register("Oracle", oracle_name, oracle_class)
    for name in STAGES:
        factory = oracle_stage if name == "oracle" and oracle_stage is not None else None
        registry.register("Stage", name, factory or DEFAULT_REGISTRY.get("Stage", name).factory)
    return registry


def recipe(tmp_path: Path, oracle: Mapping[str, Any], stages: Sequence[str] = STAGES) -> Path:
    """Write a one-trial recipe for vadd from cpu to tt with `oracle` and baseline_both on, and return its path."""
    data = {
        "extends": "base",
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": ["vadd"]},
        "directions": [{"source": SOURCE, "target": TARGET}],
        "prompts": "p0-smoke",
        "toolchain": dict(TOOLCHAINS),
        "stages": list(stages),
        "executor": {"kind": "scripted"},
        "oracle": dict(oracle),
        "fixes": {"baseline_both": True},
        "trials": {"n": 1},
    }
    path = tmp_path / "binio-runs.yaml"
    path.write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    return path


def run(tmp_path: Path, bench: Path, registry: Registry, oracle: Mapping[str, Any]) -> tuple[Path, Trial]:
    """Run the recipe and return the run directory and its one trial."""
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=bench, registry=registry)
    run_dir = run_recipe(recipe(tmp_path, oracle), options)
    trial_id = make_trial_id("binio-runs", MODEL_ID, SUITE, f"{SOURCE}-{TARGET}", "vadd", 1)
    return run_dir, read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir))


def refused(tmp_path: Path, bench: Path, registry: Registry, oracle: Mapping[str, Any], stages: Sequence[str]) -> str:
    """Run the recipe, expect a refusal before any directory exists, and return its message."""
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=bench, registry=registry)
    with pytest.raises(RunError) as caught:
        run_recipe(recipe(tmp_path, oracle, stages), options)
    assert not (tmp_path / "runs-root").exists(), "a refused run creates no directory"
    return str(caught.value)


# ---------------------------------------------------------------------------
# Refusals before any directory


def test_from_baseline_is_refused_for_an_item_without_a_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = write_suite(tmp_path, monkeypatch, None)
    oracle = {"kind": "binary_io", "metric": "pcc", "threshold": "from_baseline"}
    message = refused(tmp_path, bench, make_registry(Script({})), oracle, STAGES)
    assert "from_baseline" in message and "vadd" in message and "tolerance" in message


def test_from_baseline_is_refused_when_no_stage_builds_the_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = write_suite(tmp_path, monkeypatch, {"metric": "max_abs", "threshold": 0.01})
    oracle = {"kind": "binary_io", "metric": "max_abs", "threshold": "from_baseline"}
    message = refused(tmp_path, bench, make_registry(Script({})), oracle, STAGES[1:])
    assert "from_baseline" in message and "baseline stage" in message


def test_an_oracle_that_declares_neither_capability_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class NeitherOracle:
        """Declares no capability the oracle stage accepts."""

        name = "neither"
        capabilities = frozenset({"aligns"})

    bench = write_suite(tmp_path, monkeypatch, None)
    registry = make_registry(Script({}), oracle=("neither", NeitherOracle))
    message = refused(tmp_path, bench, registry, {"kind": "neither"}, STAGES)
    assert "masks_stdout" in message and "aligns_output_files" in message


# ---------------------------------------------------------------------------
# What the baseline and run_loop keep


def test_a_stdout_mask_recipe_keeps_no_output_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class StandInStage:
        """Stands in for the oracle stage, which would need masks for the synthetic suite; changes nothing."""

        name = "oracle"
        capabilities = frozenset({"aligns"})
        requires: dict[str, frozenset[str]] = {"Oracle": frozenset()}

        def __init__(self, *, context: RunContext) -> None:
            """Keep nothing."""

        def __call__(self, trial: Trial) -> Trial:
            """Return the trial unchanged."""
            return trial

        def describe(self) -> str:
            """Return a one-line description."""
            return "oracle: stand-in"

    bench = write_suite(tmp_path, monkeypatch, None)
    scratch = tmp_path / "arrays"
    good = {"c.lassiio": lassiio(GOOD[0], scratch)}
    script = Script({SOURCE: good, TARGET: good}, attempts=[good])
    stdout_mask = DEFAULT_REGISTRY.get("Oracle", "stdout_mask").factory
    registry = make_registry(script, oracle_stage=StandInStage, oracle=("stdout_mask", stdout_mask))
    run_dir, trial = run(tmp_path, bench, registry, {"kind": "stdout_mask", "passfail": False})
    assert trial.reference_run.outputs is None and trial.reference_agreement is None
    assert trial.attempts[-1].run.outputs is None and trial.attempts[-1].stage_reached == "S5"
    assert not (run_dir / "blobs").exists()


def test_without_a_tolerance_no_agreement_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = write_suite(tmp_path, monkeypatch, None)
    scratch = tmp_path / "arrays"
    good = {"c.lassiio": lassiio(GOOD[0], scratch)}
    script = Script({SOURCE: {"c.lassiio": lassiio(f32(9.0, 9.0), scratch / "far")}, TARGET: good}, attempts=[good])
    oracle = {"kind": "binary_io", "metric": "max_abs", "threshold": 0}
    run_dir, trial = run(tmp_path, bench, make_registry(script), oracle)
    assert trial.final.end_reason is None and trial.reference_agreement is None
    assert trial.reference_run.outputs == trial.attempts[-1].run.outputs == {"c.lassiio": sha(good["c.lassiio"])}
    assert trial.attempts[-1].alignment.mean == 1.0
    stored = {path.name for path in (run_dir / "blobs").rglob("*") if path.is_file()}
    assert stored == {sha(good["c.lassiio"])}, "the source reference's files are compared but never stored"


@pytest.mark.parametrize(
    ("files", "named"),
    [({"c.lassiio": JUNK}, "c.lassiio"), ({}, "no output file")],
    ids=["unreadable-file", "no-file"],
)
def test_a_target_reference_whose_files_cannot_serve_ends_at_baseline_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes], named: str
) -> None:
    bench = write_suite(tmp_path, monkeypatch, {"metric": "max_abs", "threshold": 0.01})
    good = {"c.lassiio": lassiio(GOOD[0], tmp_path / "arrays")}
    script = Script({SOURCE: good, TARGET: files})
    oracle = {"kind": "binary_io", "metric": "max_abs", "threshold": 0.01}
    _, trial = run(tmp_path, bench, make_registry(script), oracle)
    reason = trial.final.end_reason
    assert reason is not None and reason.code == "baseline-run" and named in reason.message
    assert "complete" not in script.events and trial.attempts == []
    assert trial.reference_run.outputs is not None and set(trial.reference_run.outputs) == set(files)


# ---------------------------------------------------------------------------
# Views of RunInfo.outputs


def outputs_trial(tmp_path: Path) -> tuple[Trial, TextStore]:
    """Return a SYNTHETIC trial whose reference run and one attempt run recorded output files."""
    store = TextStore(tmp_path / "store")
    reference = RunInfo(exit_code=0, hang=False, outputs={"z.lassiio": "b" * 64, "a/c.lassiio": "a" * 64})
    attempt = Attempt(index=0, stage_reached="S5", run=RunInfo(exit_code=0, hang=False, outputs={}))
    trial = Trial(
        trial_id=make_trial_id("binio-runs", MODEL_ID, SUITE, "cpu-tt", "vadd", 1),
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(commit=None, dirty=None, device=None, sdk=None, date="2026-09-25T00:00:00+00:00"),
        bench_item=BenchItem(suite=SUITE, item="vadd", split="eval", direction="cpu-tt"),
        model=ModelInfo(backend="scripted", id=MODEL_ID, sampling=Sampling(temperature=0.2, top_p=0.9, max_tokens=8)),
        reference_run=reference,
        attempts=[attempt],
    )
    return trial, store


def test_the_parquet_mirror_holds_outputs_as_sorted_json_text(tmp_path: Path) -> None:
    trial, _ = outputs_trial(tmp_path)
    rows = trial_rows([trial])
    expected = json.dumps({"a/c.lassiio": "a" * 64, "z.lassiio": "b" * 64}, sort_keys=True, ensure_ascii=True)
    assert rows["trials"][0]["reference_run_outputs"] == expected
    assert rows["attempts"][0]["run_outputs"] == "{}"
    write_run_parquet([trial], tmp_path / "parquet")
    back = read_run_parquet(tmp_path / "parquet")
    assert back["trials"][0]["reference_run_outputs"] == expected
    assert back["attempts"][0]["run_outputs"] == "{}"


def test_trial_md_shows_the_file_count_and_each_file_with_its_hash(tmp_path: Path) -> None:
    trial, store = outputs_trial(tmp_path)
    page = render_trial_md(trial, store)
    reference = page[page.index("## Reference run") : page.index("## Context")]
    assert "| outputs | 2 file(s) |" in reference
    assert f"| `a/c.lassiio` | `{'a' * 64}` |" in reference
    assert "| outputs | 0 file(s) |" in page[page.index("## Attempt 0") :]


# ---------------------------------------------------------------------------
# A reference run whose workdir came back incomplete


def cut_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cut: str, target_files: dict[str, bytes], oracle: dict[str, Any]
) -> tuple[Path, Trial]:
    """Run vadd (tolerance max_abs 0.01) with the `cut` reference run's workdir reported incomplete."""
    bench = write_suite(tmp_path, monkeypatch, {"metric": "max_abs", "threshold": 0.01})
    good = {"c.lassiio": lassiio(GOOD[0], tmp_path / "arrays")}
    script = Script({SOURCE: good, TARGET: target_files}, attempts=[good], cut=frozenset({f"baseline-{cut}"}))
    return run(tmp_path, bench, make_registry(script), oracle)


@pytest.mark.parametrize(
    ("threshold", "files"),
    [(0.01, "good"), (0.01, "none"), ("from_baseline", "good")],
    ids=["numeric", "files-cut-away", "from-baseline"],
)
def test_the_oracle_stage_never_aligns_against_a_cut_target_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, threshold: Any, files: str
) -> None:
    good = {"c.lassiio": lassiio(GOOD[0], tmp_path / "target")}
    target_files = good if files == "good" else {}
    oracle = {"kind": "binary_io", "metric": "max_abs", "threshold": threshold}
    _, trial = cut_run(tmp_path, monkeypatch, TARGET, target_files, oracle)
    assert trial.final.end_reason is None, "a cut reference output alone ends nothing"
    assert trial.reference_run.workdir_incomplete is True and trial.reference_agreement is None
    (note,) = trial.baseline_diagnostics
    assert (note.stage, note.severity, note.code) == ("run", "warning", "reference-workdir-incomplete")
    assert "tt reference" in note.message
    attempt = trial.attempts[-1]
    assert attempt.stage_reached == "S5" and attempt.run.outputs is not None
    assert attempt.alignment.per_input == [] and attempt.alignment.mean is None and attempt.alignment.outputs is None
    warnings = [entry for entry in attempt.diagnostics if entry.code == "reference-workdir-incomplete"]
    assert len(warnings) == 1 and (warnings[0].stage, warnings[0].severity) == ("run", "warning")
    assert trial.final.alignment is None


def test_a_cut_source_reference_leaves_no_agreement_and_a_baseline_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = {"c.lassiio": lassiio(GOOD[0], tmp_path / "target")}
    oracle = {"kind": "binary_io", "metric": "max_abs", "threshold": 0.01}
    run_dir, trial = cut_run(tmp_path, monkeypatch, SOURCE, good, oracle)
    assert trial.final.end_reason is None and trial.reference_agreement is None
    (note,) = trial.baseline_diagnostics
    assert note.code == "reference-workdir-incomplete" and "cpu reference" in note.message
    attempt = trial.attempts[-1]
    assert attempt.alignment.mean == 1.0, "a complete target reference is aligned against as before"
    assert not [entry for entry in attempt.diagnostics if entry.code == "reference-workdir-incomplete"]
    page = (trial_dir(run_dir, trial.trial_id) / "trial.md").read_text(encoding="ascii")
    notes = page[page.index("## Baseline diagnostics") : page.index("## Context")]
    assert "reference-workdir-incomplete" in notes
    rows = read_run_parquet(run_dir / "parquet")["diagnostics"]
    baseline_rows = [row for row in rows if row["attempt_index"] is None]
    assert [(row["ordinal"], row["code"]) for row in baseline_rows] == [(0, "reference-workdir-incomplete")]


def test_from_baseline_raises_at_the_oracle_stage_when_a_cut_source_left_no_agreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = {"c.lassiio": lassiio(GOOD[0], tmp_path / "target")}
    oracle = {"kind": "binary_io", "metric": "max_abs", "threshold": "from_baseline"}
    with pytest.raises(ValueError) as caught:
        cut_run(tmp_path, monkeypatch, SOURCE, good, oracle)
    message = str(caught.value)
    assert "from_baseline" in message and "cpu reference" in message and "incomplete" in message
    (run_dir,) = (tmp_path / "runs-root" / "runs").iterdir()
    assert json.loads((run_dir / "provenance.json").read_text(encoding="ascii"))["status"] == "failed"


def test_baseline_diagnostics_round_trip_and_default_to_none_noted(tmp_path: Path) -> None:
    trial, _ = outputs_trial(tmp_path)
    assert trial.baseline_diagnostics == []
    note = Diagnostic(stage="run", severity="warning", code="reference-workdir-incomplete", message="SYNTHETIC note")
    noted = dataclasses.replace(trial, baseline_diagnostics=[note])
    assert from_json(Trial, to_json(noted)) == noted
    data = json.loads(to_json(trial))
    del data["baseline_diagnostics"]
    assert from_dict(Trial, data).baseline_diagnostics == [], "a trial.json written before the field loads with []"


def stats_trial(tmp_path: Path, max_ulp: int) -> Trial:
    """Return outputs_trial with an agreement of one output and two statistics on its attempt."""
    trial, _ = outputs_trial(tmp_path)

    def stat(name: str, units: int | None) -> OutputStats:
        return OutputStats(name=name, pcc=1.0, max_abs=0.0, max_ulp=units, passed=True, note=None)

    attempt = dataclasses.replace(
        trial.attempts[0], alignment=Alignment(per_input=[1.0], mean=1.0, outputs=[stat("z", max_ulp), stat("a", 3)])
    )
    return dataclasses.replace(trial, reference_agreement=[stat("m", 7)], attempts=[attempt])


def test_output_stats_rows_keep_their_written_order_and_max_ulp_is_uint64(tmp_path: Path) -> None:
    big = 2**64 - 1  # the largest max_ulp the unsigned 64-bit column holds
    trial = stats_trial(tmp_path, big)
    expected = [(None, 0, "m", 7), (0, 0, "z", big), (0, 1, "a", 3)]
    rows = trial_rows([trial])["output_stats"]
    assert [(row["attempt_index"], row["ordinal"], row["name"], row["max_ulp"]) for row in rows] == expected
    write_run_parquet([trial], tmp_path / "parquet")
    back = read_run_parquet(tmp_path / "parquet")["output_stats"]
    assert [(row["attempt_index"], row["ordinal"], row["name"], row["max_ulp"]) for row in back] == expected
    (path,) = (tmp_path / "parquet" / "output_stats").rglob("*.parquet")
    assert pq.read_schema(path).field("max_ulp").type == pa.uint64()


def test_a_max_ulp_past_uint64_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    out = tmp_path / "parquet"
    with pytest.raises(ValueError, match="output_stats.max_ulp of trial .* must fit in a 64-bit unsigned int"):
        write_run_parquet([stats_trial(tmp_path, 2**64)], out)
    assert not out.exists()
