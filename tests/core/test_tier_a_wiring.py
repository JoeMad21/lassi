"""The Tier A dry run's path end to end, with the mock backend and stand-in components (task P4.13).

Plan: plans/p4-ttsim.md, P4.13: the gate runs
tests/fixtures/recipes/p4-tier-a-dry-run.yaml (the mock, both directions
between C++ and TT, fixes.baseline_both on, executors per language,
binary_io with threshold from_baseline), and every trial must pass the
baseline with every mock candidate at S5 and alignment 1.0. This file runs
that recipe's shape through the real runner, stages, mock backend, and
binary_io oracle on a SYNTHETIC suite, with stand-in toolchains and
executors (they compile nothing and start no process), and fixes the
wiring P4.13 adds:

- A tracked language's program is the reference target (and so the mock's
  reply) and the source, read from the repository (TRACKED_ROOT), not the
  fetched sources.
- Every build gets its own language's support files: the TT reference, and
  each TT attempt, gets the upstream kernel (from the fetched sources) and
  lassi_io.h (tracked); every C++ build gets lassi_io.h only. The CPU -> TT
  guard reads the TT attempt with the TT support files.
- Before every run, reference or attempt, the item's held-out inputs are
  written under <workdir>/@inputs with the item's seed (the same bytes
  generate_inputs gives), and the program's arguments are the input files
  in input order, then the output files (lassi.bench.registry
  program_args). The inputs are not output files.
- Both references run, their agreement is recorded within the item's
  tolerance, and each mock candidate (the target reference again) reaches
  S5 with alignment 1.0 under from_baseline.

The suite, programs, kernel, header, and arrays are SYNTHETIC; no value
here is a measurement.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.bench import registry as bench_registry
from lassi.core import runner as runner_module
from lassi.core.capabilities import HOST_COMPUTE_GUARD, HostComputeReading
from lassi.core.interfaces import BuildResult, Limits, RunResult
from lassi.core.record import Trial, make_trial_id
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.harness.lassi_io import generate_inputs, read_array, write_array

SUITE = "tier-a-wiring"
ITEM = "vadd"
COMMIT = "d" * 40
MODEL_ID = "mock-reference"
SEED = 11
CPP, TT = "cpp", "tt"
STAGES = ["baseline", "generate", "compile_loop", "run_loop", "oracle"]
HEADER = "// SYNTHETIC lassi_io.h stand-in\n"
KERNEL = "// SYNTHETIC compute kernel\n"
KERNEL_NAME = "vadd/kernels/compute/add.cpp"
KERNEL_UPSTREAM = "examples/vadd/kernels/compute/add.cpp"
PROGRAMS = {CPP: "// SYNTHETIC C++ counterpart of a vector add\n", TT: "// SYNTHETIC TT host program of a vector add\n"}
INPUTS = [
    {"name": "a", "dtype": "f32", "shape": [8], "dist": "uniform", "lo": -1.0, "hi": 1.0},
    {"name": "b", "dtype": "f32", "shape": [8], "dist": "uniform", "lo": -1.0, "hi": 1.0},
]
ARGS = ["@inputs/a.lassiio", "@inputs/b.lassiio", "c.lassiio"]
# The stand-in TT run adds 2**-8 to c[0], inside the item's max_abs 0.01, so the references agree without being equal.
TT_OFFSET = 2.0**-8
SYNTHETIC_WALL_S = 1.5


def digest(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def manifest_data() -> dict[str, Any]:
    """Return the SYNTHETIC suite: one item with tracked programs, per-language support, and held-out inputs."""
    header = {"tracked": "assets/wiring/lassi_io.h"}
    return {
        "suite": SUITE,
        "repo": "https://example.invalid/kernels.git",
        "commit": COMMIT,
        "items": {
            ITEM: {
                "split": "unassigned",
                "seed": SEED,
                "inputs": INPUTS,
                "outputs": ["c"],
                "tolerance": {"metric": "max_abs", "threshold": 0.01},
                "unpack_to_dest": "SYNTHETIC note",
                "languages": {
                    CPP: {
                        "dir": "assets/wiring/vadd/cpp", "tracked": True, "files": ["vadd.cpp"],
                        "support": {"lassi_io.h": dict(header)},
                    },
                    TT: {
                        "dir": "assets/wiring/vadd/tt", "tracked": True, "files": ["vadd.cpp"],
                        "support": {
                            "lassi_io.h": dict(header),
                            KERNEL_NAME: {"upstream": KERNEL_UPSTREAM, "sha256": digest(KERNEL.encode("ascii"))},
                        },
                    },
                },
            }
        },
    }


@dataclass
class Seen:
    """What the stand-ins saw: each build's files and harness, each guard harness, and each run."""

    builds: list[tuple[str, str, dict[str, str], dict[str, str]]] = field(default_factory=list)
    guard_harness: list[dict[str, str]] = field(default_factory=list)
    runs: list[dict[str, Any]] = field(default_factory=list)


def stand_in_toolchain(language: str, seen: Seen) -> type:
    """Return a Toolchain class that writes the files and a PLACEHOLDER artifact; the TT one declares the guard."""

    class StandInToolchain:
        """Writes `files` and `harness` under the workdir and reports a built artifact; it compiles nothing."""

        name = f"stand-in-{language}"
        capabilities = frozenset({"diagnostics", *([HOST_COMPUTE_GUARD] if language == TT else [])})

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Record the build and write its files; the artifact is a placeholder."""
            seen.builds.append((language, Path(workdir).parent.name, dict(files), dict(harness or {})))
            for path, text in [*files.items(), *(harness or {}).items()]:
                target = Path(workdir) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            artifact = Path(workdir) / "main"
            artifact.write_bytes(b"PLACEHOLDER artifact of the stand-in toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=[])

        def host_compute_guard(self, files: Mapping[str, str], harness: Mapping[str, str]) -> HostComputeReading:
            """Record the harness the guard was given and read the program as no host compute."""
            seen.guard_harness.append(dict(harness))
            return HostComputeReading(host_compute=False)

    return StandInToolchain


def stand_in_executor(language: str, seen: Seen) -> type:
    """Return an Executor class that 'runs' a program by computing c = a + b from the argument files."""

    class StandInExecutor:
        """Reads the input files the arguments name, writes the output file, and reports the new files."""

        name = f"stand-in-{language}"
        capabilities = frozenset({"runs_code", "sandboxed"})

        def device(self) -> str:
            """Name a SYNTHETIC device."""
            return f"SYNTHETIC {language} device"

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Compute the output from the inputs named in `inputs`, relative to the artifact's directory."""
            workdir = Path(artifact).parent
            before = {path.relative_to(workdir).as_posix() for path in workdir.rglob("*") if path.is_file()}
            inputs_dir = workdir / "@inputs"
            staged = sorted(path.name for path in inputs_dir.iterdir()) if inputs_dir.is_dir() else []
            seen.runs.append({
                "language": language, "kind": workdir.parent.name, "args": list(inputs), "staged": staged,
                "bytes": {name: (inputs_dir / name).read_bytes() for name in staged},
            })
            a, b = (read_array(workdir / name) for name in inputs[:2])
            values = [x + y for x, y in zip(struct.unpack("<8f", a.data), struct.unpack("<8f", b.data), strict=True)]
            if language == TT:
                values[0] += TT_OFFSET
            write_array(workdir / inputs[2], "c", "f32", (8,), struct.pack("<8f", *values))
            after = {path.relative_to(workdir).as_posix(): path for path in workdir.rglob("*") if path.is_file()}
            outputs = {name: path for name, path in sorted(after.items()) if name not in before}
            return RunResult(
                exit_code=0, hang=False, stdout="SYNTHETIC run\n", stderr="", output_files=outputs,
                wall_s=SYNTHETIC_WALL_S, sim_ub=False if language == TT else None,
            )

    return StandInExecutor


def make_registry(seen: Seen) -> Registry:
    """Return a Registry with the real mock backend, stages, and binary_io, and the stand-in components."""
    registry = Registry()
    registry.register("LLMBackend", "mock", DEFAULT_REGISTRY.get("LLMBackend", "mock").factory)
    registry.register("Oracle", "binary_io", DEFAULT_REGISTRY.get("Oracle", "binary_io").factory)
    for name in STAGES:
        registry.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    for language in (CPP, TT):
        registry.register("Toolchain", f"stand-in-{language}", stand_in_toolchain(language, seen))
        registry.register("Executor", f"stand-in-{language}", stand_in_executor(language, seen))
    return registry


def recipe_data() -> dict[str, Any]:
    """Return the dry run's shape on the SYNTHETIC suite (tests/fixtures/recipes/p4-tier-a-dry-run.yaml)."""
    return {
        "extends": "base",
        "model": {"backend": "mock", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "unassigned", "items": [ITEM]},
        "directions": [{"source": CPP, "target": TT}, {"source": TT, "target": CPP}],
        "prompts": "p0-smoke",
        "toolchain": {CPP: f"stand-in-{CPP}", TT: f"stand-in-{TT}"},
        "fixes": {"baseline_both": True},
        "stages": list(STAGES),
        "executor": {CPP: {"kind": f"stand-in-{CPP}"}, TT: {"kind": f"stand-in-{TT}"}},
        "oracle": {"kind": "binary_io", "metric": "pcc", "threshold": "from_baseline"},
        "trials": {"n": 1},
    }


@dataclass
class Outcome:
    """The finished run: its trials by direction name and what the stand-ins saw."""

    trials: dict[str, Trial]
    seen: Seen


@pytest.fixture
def outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Outcome:
    """Run the dry run's shape once, both directions, and read both trials back from the run tree."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS"):
        monkeypatch.delenv(name, raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    repo = tmp_path / "repo"
    for path, text in {
        "assets/wiring/lassi_io.h": HEADER,
        "assets/wiring/vadd/cpp/vadd.cpp": PROGRAMS[CPP],
        "assets/wiring/vadd/tt/vadd.cpp": PROGRAMS[TT],
    }.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_bytes(text.encode("ascii"))
    monkeypatch.setattr(bench_registry, "TRACKED_ROOT", repo)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / f"{SUITE}.yaml").write_bytes(yaml.safe_dump(manifest_data(), sort_keys=False).encode("ascii"))
    monkeypatch.setattr(runner_module, "BENCH_DIR", manifests)
    sources = tmp_path / "sources"
    (sources / KERNEL_UPSTREAM).parent.mkdir(parents=True)
    (sources / KERNEL_UPSTREAM).write_bytes(KERNEL.encode("ascii"))
    seen = Seen()
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="tier-a-wiring", bench_root=sources,
                         registry=make_registry(seen))
    recipe = tmp_path / "tier-a-wiring.yaml"
    recipe.write_bytes(yaml.safe_dump(recipe_data(), sort_keys=False).encode("ascii"))
    run_dir = run_recipe(recipe, options)
    trials = {}
    for direction in ("cpp-tt", "tt-cpp"):
        trial_id = make_trial_id("tier-a-wiring", MODEL_ID, SUITE, direction, ITEM, 1)
        trials[direction] = read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir))
    return Outcome(trials=trials, seen=seen)


def test_every_build_gets_its_own_languages_support_files(outcome: Outcome) -> None:
    tt_harness = {"lassi_io.h": HEADER, KERNEL_NAME: KERNEL}
    cpp_harness = {"lassi_io.h": HEADER}
    builds = outcome.seen.builds
    assert len(builds) == 6, "two references and one attempt per direction"
    for language, _, files, harness in builds:
        assert files == {"vadd.cpp": PROGRAMS[language]}, "references and mock candidates are the tracked programs"
        assert harness == (tt_harness if language == TT else cpp_harness)
    assert outcome.seen.guard_harness == [tt_harness], "the guard reads the TT attempt with the TT support files"
    assert outcome.trials["cpp-tt"].attempts[0].guards.host_compute is False


def test_every_run_gets_the_seeded_inputs_and_the_arguments_in_order(outcome: Outcome, tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    expected.mkdir()
    generate_inputs(INPUTS, SEED, expected)
    want = {path.name: path.read_bytes() for path in expected.iterdir()}
    runs = outcome.seen.runs
    assert [(run["language"], run["kind"]) for run in runs] == [
        (TT, "baseline-tt"), (CPP, "baseline-cpp"), (TT, "attempt00"),
        (CPP, "baseline-cpp"), (TT, "baseline-tt"), (CPP, "attempt00"),
    ]
    for run in runs:
        assert run["args"] == ARGS
        assert run["staged"] == ["a.lassiio", "b.lassiio"] and run["bytes"] == want


def test_both_references_agree_and_each_mock_candidate_passes_from_baseline(outcome: Outcome) -> None:
    for direction, trial in outcome.trials.items():
        assert trial.final.end_reason is None, f"{direction}: {trial.final.end_reason}"
        agreement = {entry.name: entry for entry in trial.reference_agreement or []}
        assert set(agreement) == {"c"} and agreement["c"].passed
        assert agreement["c"].max_abs == pytest.approx(TT_OFFSET, abs=1e-6)
        assert set(trial.reference_run.outputs or {}) == {"c.lassiio"}, "the staged inputs are not outputs"
        (attempt,) = trial.attempts
        assert attempt.stage_reached == "S5"
        assert set(attempt.run.outputs or {}) == {"c.lassiio"}
        assert (attempt.alignment.per_input, attempt.alignment.mean) == ([1.0], 1.0)
        assert trial.final.alignment == 1.0
