"""Tests for the oracle stage over output files and the binary_io recipe section (task P4.4).

Bible: Oracles (binary_io row), Component Interfaces (Oracle and Stage rows;
the capabilities rule), Result Record (RunInfo, Alignment, the reference
run, Storage), Project Recipes (the lassi-df oracle line). Plan:
plans/p4-ttsim.md, P4.4 and the planning decision "binary_io (P4.4)".

The contract these tests fix (see tests/oracles/test_binary_io.py for the
oracle's own rules, and tests/core/test_output_files_record.py for the
record and the binary store):

- A recipe that binds `oracle: {kind: binary_io, metric: ..., threshold:
  ...}` and lists the oracle stage loads with the default registry, the
  lassi-df line `{kind: binary_io, metric: pcc, threshold: from_baseline}`
  included.
- The oracle stage, built as factory(context=<RunContext>), aligns output
  files when the recipe's Oracle declares ALIGNS_OUTPUT_FILES. It aligns
  every attempt whose run recorded its output files (RunInfo.outputs is not
  None) against the target reference's (Trial.reference_run.outputs), both
  read from the binary store beside the context's text store
  (BlobStore(context.store.root)). Each such attempt gets
  Alignment(per_input=[v], mean=v, outputs=<one OutputStats per output>),
  v being 1.0 when every output passes and 0.0 otherwise; every other field
  and every attempt that never ran stays as it was.
- With threshold from_baseline, the thresholds come from the agreement the
  baseline recorded (Trial.reference_agreement); a trial with outputs to
  align but no recorded agreement raises ValueError naming from_baseline. A
  trial whose attempts recorded outputs but whose reference recorded none
  raises ValueError naming the reference.
- The runner refuses, before any directory is created, a binary_io section
  that leaves metric or threshold unset (the message names the key), and a
  from_baseline threshold while fixes.baseline_both is off (the message
  names from_baseline and baseline_both), since then no agreement is ever
  measured.

stdout_mask keeps its behavior; its tests in tests/oracles are unchanged.

Every array, stdout, reply, and source here is SYNTHETIC and written for
these tests; nothing is executed, and no value is a measurement.
"""

from __future__ import annotations

import copy
import dataclasses
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.bench import Direction, load_suite
from lassi.core import record as record_module
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core import store as store_module
from lassi.core.interfaces import BuildResult, Sampling
from lassi.core.recipe import RecipeError, load_recipe
from lassi.core.record import Alignment, Attempt, Diagnostic, ModelInfo, Provenance, RunInfo, Trial, make_trial_id
from lassi.core.registry import DEFAULT_REGISTRY, Registry, RegistryError
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.stages import BaselineStage, CompileLoopStage, GenerateStage, RunContext
from lassi.core.store import TextStore
from lassi.executors import NoneExecutor
from lassi.harness.lassi_io import LassiArray, write_array
from lassi.llm import MockBackend

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "tests" / "fixtures" / "recipes" / "p0-smoke.yaml"
SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
ITEM = "layout"
OMP_TO_CUDA = Direction("omp", "cuda")
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
STAGES = ["baseline", "generate", "compile_loop", "oracle"]
LASSI_DF_ORACLE = {"kind": "binary_io", "metric": "pcc", "threshold": "from_baseline"}
# A commit id for synthetic provenance; not a commit of this repository.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SYNTHETIC_STDOUT = "SYNTHETIC run stdout\n"
SYNTHETIC_WALL_S = 1.25  # a fixture value, not a measurement


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


def binary_io_class() -> type:
    """Return the Oracle class registered as binary_io; fail the test clearly while it is not registered."""
    try:
        return DEFAULT_REGISTRY.get("Oracle", "binary_io").factory
    except RegistryError as error:
        pytest.fail(f"no Oracle is registered as 'binary_io' ({error}); task P4.4 registers it")


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


def output_stats(**fields: Any) -> Any:
    """Return lassi.core.record.OutputStats(**fields); fail the test clearly while the class is missing."""
    cls = getattr(record_module, "OutputStats", None)
    if cls is None:
        pytest.fail("lassi.core.record has no OutputStats; task P4.4 adds the per-output statistics record")
    return cls(**fields)


# ---------------------------------------------------------------------------
# Recipes, contexts, outputs, and trials


def recipe_data(**changes: Any) -> dict[str, Any]:
    """Return the p0-smoke recipe for layout with top-level keys replaced by `changes`; None drops a key."""
    data = copy.deepcopy(yaml.safe_load(SMOKE.read_text(encoding="utf-8")))
    data["bench"]["items"] = [ITEM]
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    return data


def write_recipe(directory: Path, name: str, data: Mapping[str, Any]) -> Path:
    """Write `data` as `<directory>/<name>.yaml` and return its path."""
    path = directory / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


def binio_stage(tmp_path: Path, oracle: Mapping[str, Any]) -> tuple[Any, RunContext]:
    """Return the registered oracle stage built on a RunContext whose recipe binds `oracle`, and the context."""
    binary_io_class()
    recipe = load_recipe(write_recipe(tmp_path, "binio-stage", recipe_data(oracle=dict(oracle), stages=STAGES)))
    context = RunContext(
        recipe=recipe,
        backend=MockBackend("mock-reference"),
        sampling=SAMPLING,
        toolchains={},
        executor=NoneExecutor(),
        store=TextStore(tmp_path / "store"),
        suite=load_suite(SUITE_MANIFEST),
        sources_root=tmp_path / "bench",
        item=ITEM,
        direction=OMP_TO_CUDA,
        build_root=tmp_path / "builds",
        prompts="p0-smoke",
        max_corrections=10,
    )
    return DEFAULT_REGISTRY.get("Stage", "oracle").factory(context=context), context


def f32(*values: float, name: str = "c") -> LassiArray:
    """Return a SYNTHETIC f32 array of `values` (each exactly representable in f32)."""
    return LassiArray(name, "f32", (len(values),), struct.pack(f"<{len(values)}f", *values))


def stored_outputs(context: RunContext, scratch: Path, arrays: Sequence[LassiArray]) -> dict[str, str]:
    """Write each array as `<name>.lassiio`, keep its bytes in the binary store, and return file -> sha256."""
    scratch.mkdir(parents=True, exist_ok=True)
    blobs = blob_store(context.store.root)
    outputs = {}
    for array in arrays:
        path = scratch / f"{array.name}.lassiio"
        write_array(path, array.name, array.dtype, array.shape, array.data)
        outputs[path.name] = blobs.put(path.read_bytes())
    return outputs


def ran(context: RunContext, outputs: Mapping[str, str]) -> RunInfo:
    """Return a SYNTHETIC clean RunInfo that recorded `outputs` (file -> sha256)."""
    require_field(RunInfo, "outputs")
    stdout_ref = context.store.put(SYNTHETIC_STDOUT)
    return RunInfo(
        exit_code=0,
        hang=False,
        wall_s=SYNTHETIC_WALL_S,
        stdout_ref=stdout_ref,
        outputs=dict(outputs),
        stdout_truncated=False,
        stderr_truncated=False,
        workdir_incomplete=False,
    )


def compiled(index: int, run: RunInfo | None = None) -> Attempt:
    """Return a SYNTHETIC attempt that compiled (S4) or ran clean (S5) with `run`."""
    files = {"main.cu": "int main() { return 0; }\n"}
    stage = "S4" if run is None else "S5"
    return Attempt(index=index, response_text="SYNTHETIC reply", files=files, stage_reached=stage, run=run or RunInfo())


def compile_failed(index: int) -> Attempt:
    """Return a SYNTHETIC attempt that parsed (S1) but did not compile."""
    error = Diagnostic(stage="compile", severity="error", code="fake-error", message="SYNTHETIC compile error")
    files = {"main.cu": "#error SYNTHETIC\n"}
    return Attempt(index=index, response_text="SYNTHETIC reply", files=files, stage_reached="S1", diagnostics=[error])


def trial_of(context: RunContext, attempts: Sequence[Attempt], reference: RunInfo | None = None) -> Trial:
    """Return a SYNTHETIC trial of layout (omp to cuda) holding `attempts` and the reference run `reference`."""
    trial_id = make_trial_id("binio-stage", "mock-reference", SUITE, OMP_TO_CUDA.name, ITEM, 1)
    return Trial(
        trial_id=trial_id,
        recipe_hash=context.recipe.recipe_hash,
        provenance=Provenance(
            commit=FAKE_COMMIT, dirty=False, device="SYNTHETIC device", sdk=None, date="2026-09-25T00:00:00+00:00"
        ),
        bench_item=context.suite.bench_item(ITEM, OMP_TO_CUDA),
        model=ModelInfo(backend="mock", id="mock-reference", sampling=SAMPLING),
        reference_run=reference or RunInfo(),
        attempts=list(attempts),
    )


def with_agreement(trial: Trial, agreement: Sequence[Any]) -> Trial:
    """Return `trial` with the baseline's recorded agreement; fail clearly while Trial has no such field."""
    require_field(Trial, "reference_agreement")
    return dataclasses.replace(trial, reference_agreement=list(agreement))


def outputs_of(attempt: Attempt) -> dict[str, Any]:
    """Return an attempt's per-output statistics keyed by output name; fail clearly while Alignment has none."""
    require_field(Alignment, "outputs")
    assert attempt.alignment.outputs is not None, f"attempt {attempt.index} holds no per-output statistics"
    return {entry.name: entry for entry in attempt.alignment.outputs}


def assert_only_alignment_changed(before: Trial, after: Trial) -> None:
    """Assert that `after` differs from `before` in attempt alignments at most."""
    assert dataclasses.replace(after, attempts=[]) == dataclasses.replace(before, attempts=[])
    assert len(after.attempts) == len(before.attempts)
    for old, new in zip(before.attempts, after.attempts, strict=True):
        assert dataclasses.replace(new, alignment=Alignment()) == dataclasses.replace(old, alignment=Alignment())


# ---------------------------------------------------------------------------
# Recipes


@pytest.mark.parametrize(
    "oracle",
    [LASSI_DF_ORACLE, {"kind": "binary_io", "metric": "max_abs", "threshold": 0.001}],
    ids=["lassi-df-line", "numeric-threshold"],
)
def test_a_recipe_with_a_binary_io_section_and_the_oracle_stage_loads(tmp_path: Path, oracle: dict[str, Any]) -> None:
    binary_io_class()
    recipe = load_recipe(write_recipe(tmp_path, "binio-recipe", recipe_data(oracle=oracle, stages=STAGES)))
    bound = {(binding.interface, binding.name): dict(binding.config) for binding in recipe.bindings}
    expected = {key: value for key, value in oracle.items() if key != "kind"}
    assert bound[("Oracle", "binary_io")] == expected
    assert ("Stage", "oracle") in bound


# ---------------------------------------------------------------------------
# The stage on trials


def test_the_stage_aligns_each_run_by_its_output_files(tmp_path: Path) -> None:
    stage, context = binio_stage(tmp_path, {"kind": "binary_io", "metric": "max_abs", "threshold": 0.25})
    reference = ran(context, stored_outputs(context, tmp_path / "ref", [f32(1.0, 2.0), f32(3.0, name="d")]))
    same = ran(context, stored_outputs(context, tmp_path / "same", [f32(1.0, 2.0), f32(3.0, name="d")]))
    # d is 0.5 off: past the threshold of 0.25.
    off = ran(context, stored_outputs(context, tmp_path / "off", [f32(1.0, 2.0), f32(3.5, name="d")]))
    trial = trial_of(context, [compile_failed(0), compiled(1, same), compiled(2, off)], reference)
    result = stage(trial)
    assert result.attempts[0].alignment == Alignment()
    assert (result.attempts[1].alignment.per_input, result.attempts[1].alignment.mean) == ([1.0], 1.0)
    assert (result.attempts[2].alignment.per_input, result.attempts[2].alignment.mean) == ([0.0], 0.0)
    passing = outputs_of(result.attempts[1])
    assert set(passing) == {"c", "d"} and all(entry.passed for entry in passing.values())
    assert passing["c"].max_abs == 0.0 and passing["c"].max_ulp == 0
    failing = outputs_of(result.attempts[2])
    assert (failing["c"].passed, failing["d"].passed) == (True, False)
    assert failing["d"].max_abs == 0.5
    assert_only_alignment_changed(trial, result)


def test_attempts_that_never_ran_keep_their_alignment_unset(tmp_path: Path) -> None:
    stage, context = binio_stage(tmp_path, {"kind": "binary_io", "metric": "max_abs", "threshold": 0.25})
    reference = ran(context, stored_outputs(context, tmp_path / "ref", [f32(1.0, 2.0)]))
    trial = trial_of(context, [compile_failed(0), compiled(1)], reference)
    assert stage(trial) == trial


def test_from_baseline_reads_the_agreement_the_baseline_recorded(tmp_path: Path) -> None:
    stage, context = binio_stage(tmp_path, {"kind": "binary_io", "metric": "max_abs", "threshold": "from_baseline"})
    reference = ran(context, stored_outputs(context, tmp_path / "ref", [f32(1.0, 2.0, 3.0)]))
    # The references agreed within 2**-7 = 0.0078125 on c. 2**-8 off passes; 2**-6 off does not.
    within = ran(context, stored_outputs(context, tmp_path / "within", [f32(1.0, 2.0, 3.0 + 2.0**-8)]))
    past = ran(context, stored_outputs(context, tmp_path / "past", [f32(1.0, 2.0, 3.0 + 2.0**-6)]))
    agreement = [output_stats(name="c", pcc=1.0, max_abs=0.0078125, max_ulp=32768, passed=True, note=None)]
    trial = with_agreement(trial_of(context, [compiled(0, within), compiled(1, past)], reference), agreement)
    result = stage(trial)
    assert result.attempts[0].alignment.per_input == [1.0]
    assert result.attempts[1].alignment.per_input == [0.0]
    assert outputs_of(result.attempts[0])["c"].max_abs == 0.00390625
    assert outputs_of(result.attempts[1])["c"].max_abs == 0.015625
    assert_only_alignment_changed(trial, result)


def test_from_baseline_without_a_recorded_agreement_raises(tmp_path: Path) -> None:
    stage, context = binio_stage(tmp_path, LASSI_DF_ORACLE)
    reference = ran(context, stored_outputs(context, tmp_path / "ref", [f32(1.0, 2.0)]))
    candidate = ran(context, stored_outputs(context, tmp_path / "cand", [f32(1.0, 2.0)]))
    require_field(Trial, "reference_agreement")
    trial = trial_of(context, [compiled(0, candidate)], reference)
    assert trial.reference_agreement is None
    with pytest.raises(ValueError, match="from_baseline"):
        stage(trial)


def test_run_outputs_without_reference_outputs_raise(tmp_path: Path) -> None:
    stage, context = binio_stage(tmp_path, {"kind": "binary_io", "metric": "max_abs", "threshold": 0.25})
    candidate = ran(context, stored_outputs(context, tmp_path / "cand", [f32(1.0, 2.0)]))
    trial = trial_of(context, [compiled(0, candidate)])
    with pytest.raises(ValueError, match="reference"):
        stage(trial)


def test_the_stage_describes_output_file_alignment_in_one_line(tmp_path: Path) -> None:
    stage, _ = binio_stage(tmp_path, LASSI_DF_ORACLE)
    text = stage.describe()
    assert isinstance(text, str) and text and "\n" not in text
    assert "binary_io" in text


# ---------------------------------------------------------------------------
# The runner's refusals


def write_bench(root: Path) -> Path:
    """Write SYNTHETIC layout sources under `root` as the manifest lays them out, and return `root`."""
    spec = load_suite(SUITE_MANIFEST).items[ITEM]
    texts = {"omp": "int main() { return 0; }\n", "cuda": "__global__ void k() { }\nint main() { return 0; }\n"}
    for language, text in texts.items():
        layout = spec.languages[language]
        path = root / layout.dir / layout.files[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def fake_toolchain(registered_as: str) -> type:
    """Return a Toolchain class without PIN that writes the files and a PLACEHOLDER artifact; it runs nothing."""

    class FakeToolchain:
        """Writes the files under the fresh workdir and reports a built artifact; compiles nothing."""

        name = registered_as
        capabilities = frozenset({"diagnostics"})

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Write `files` and a placeholder artifact under `workdir` and return it with no diagnostics."""
            for path, text in files.items():
                target = Path(workdir) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            artifact = Path(workdir) / "main"
            artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=[])

    return FakeToolchain


def runner_registry() -> Registry:
    """Return a Registry with the mock backend, the real stages and binary_io, the none executor, fake toolchains."""
    registry = Registry()
    registry.register("LLMBackend", "mock", MockBackend)
    registry.register("Stage", "baseline", BaselineStage)
    registry.register("Stage", "generate", GenerateStage)
    registry.register("Stage", "compile_loop", CompileLoopStage)
    registry.register("Stage", "oracle", DEFAULT_REGISTRY.get("Stage", "oracle").factory)
    registry.register("Oracle", "binary_io", binary_io_class())
    registry.register("Executor", "none", NoneExecutor)
    registry.register("Toolchain", "nvcc-sm80", fake_toolchain("nvcc-sm80"))
    registry.register("Toolchain", "nvcpp-cc80", fake_toolchain("nvcpp-cc80"))
    return registry


def refusal(tmp_path: Path, data: Mapping[str, Any]) -> str:
    """Run `data` and return the refusal's message; fail the test when it runs or leaves a directory behind."""
    registry = runner_registry()
    options = RunOptions(
        runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=write_bench(tmp_path / "bench"),
        registry=registry,
    )
    with pytest.raises((RecipeError, RunError)) as caught:
        run_recipe(write_recipe(tmp_path, "binio-run", data), options)
    assert not (tmp_path / "runs-root").exists(), "a refused run creates no directory"
    return str(caught.value)


@pytest.mark.parametrize(
    ("oracle", "key"),
    [
        ({"kind": "binary_io", "metric": "pcc"}, "oracle.threshold"),
        ({"kind": "binary_io", "threshold": 0.99}, "oracle.metric"),
    ],
    ids=["no-threshold", "no-metric"],
)
def test_the_runner_refuses_a_binary_io_section_without_a_required_key(
    tmp_path: Path, oracle: dict[str, Any], key: str
) -> None:
    message = refusal(tmp_path, recipe_data(oracle=oracle, stages=["generate", "compile_loop", "oracle"]))
    assert key in message


def test_the_runner_refuses_from_baseline_while_baseline_both_is_off(tmp_path: Path) -> None:
    data = recipe_data(oracle=LASSI_DF_ORACLE, stages=STAGES, fixes={"baseline_both": False})
    message = refusal(tmp_path, data)
    assert "from_baseline" in message and "baseline_both" in message
