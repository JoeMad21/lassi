"""Tests for how a model-written kernel is built and how its kernel JIT error comes back to the model (task P17.39).

Bible: Harness Contract (a TT host build never compiles a kernel source:
ttmetal-host writes it uncompiled at its path, and the kernel JIT compiles
it at the program's first launch; a jit-stage error leaves the attempt at
S1, corrected as a compile error), ttsim Facts (the ttsim executor reads
each `<path>:<line>:<column>: <severity>: <message>` line of a
"<processor> build failed. Log:" block as a jit-stage Diagnostic, the file
relative to the workdir), Benchmark Suites (tt-pairs-v0-kh), Agent Rules 2
and 6.

What they fix, for C++ -> TT where the model writes the host program and
every kernel (the ttkh language of tt-pairs-v0-kh, the tt-kh-v0 set):

- ttmetal-host builds the model's ttkh files as written: it compiles the
  host program alone and writes each model kernel uncompiled at its path,
  with lassi_io.h, the one support file, beside them; no model kernel is
  refused, since the harness provides none.
- A kernel JIT error in captured style (the P4.9 capture's form, with a
  SYNTHETIC kernel path and message) names the model's kernel file relative
  to the run's directory, and a failed kernel build with no compiler error
  line is still one jit-stage error (kernel-build-failed).
- Through the runner with fakes: the run that reported the JIT error leaves
  the attempt at S1 with no run-error; compile_loop asks for the correction
  under the cap, and its tt-kh-v0 prompt carries the parsed diagnostic line
  `error <kernel path>:<line>:<column> <message>` beside the model's own
  kernel file, never as an opaque run failure.

Every log here is SYNTHETIC, written in the form of the P4.9 capture
tests/executors/fixtures/ttsim/jit-error (README.md there), not captured.
The fake executor runs nothing and opens no device; the fake toolchains and
the fake compile runner compile nothing. No value here is a measurement.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

from lassi.bench import load_suite
from lassi.core import runner as runner_module
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import RunResult
from lassi.core.record import make_trial_id
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.stages import PARSED, run_stage
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.executors import ttsim
from lassi.toolchains import CommandResult
from lassi.toolchains.ttmetal_build import TtMetalHost

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_sim_readings import (  # noqa: E402
    MODEL_ID,
    SIM_EXECUTOR,
    Outcome,
    Script,
    clean_sim_reference,
    clean_sim_run,
    errors_of,
    make_registry,
    sim_run,
    stages_of,
)

KH_SUITE = load_suite(REPO / "assets" / "bench" / "tt-pairs-v0-kh.yaml")
HARNESS = "assets/harness/c/lassi_io.h"
# The SYNTHETIC suite of the runner tests: one item in cpp, tt (a host program and one kernel), and ttkh.
SUITE, ITEM = "khfix", "scale"
SOURCE, TARGET = "cpp", "ttkh"
HOST, KERNEL = "scale.cpp", "kernels/compute/k.cpp"
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SOURCES = {
    "cpp": "// SYNTHETIC cpp program of a made-up scale\nint main() { return 0; }\n",
    "tt": "// SYNTHETIC tt host program of a made-up scale\nint main() { return 0; }\n",
    KERNEL: "// SYNTHETIC upstream kernel of the tt program; no prompt may show it\nvoid kernel_main() {}\n",
}
MODEL_FILES = {
    HOST: "// SYNTHETIC model host program\nint main() { return 0; }\n",
    KERNEL: "// SYNTHETIC model kernel\nvoid kernel_main() {\n    add_tiles_bcast_x(0, 1, 0, 0, 0);\n}\n",
}
FIXED_FILES = {**MODEL_FILES, KERNEL: "// SYNTHETIC corrected model kernel\nvoid kernel_main() {}\n"}
# The POSIX directory the SYNTHETIC log names as the run's workdir; read_jit_diagnostics is given the same.
WORKDIR = PurePosixPath("/SYNTHETIC/runs/test-run/attempt-0/build")
MESSAGE = "'add_tiles_bcast_x' was not declared in this scope"


def jit_log(kernel: str, line: int = 3, column: int = 5) -> str:
    """Return a SYNTHETIC kernel JIT failure in the P4.9 capture's form, naming `kernel` inside WORKDIR."""
    path = f"{WORKDIR}/{kernel}"
    return (
        "SYNTHETIC log in the form of tests/executors/fixtures/ttsim/jit-error; not captured\n"
        "trisc1 build failed. Log: In file included from ../kernel_includes.hpp:1,\n"
        "                 from /SYNTHETIC/toolchains/tt-metal@5280a9cf/tt_metal/hw/firmware/src/trisck.cc:21:\n"
        f"{path}: In function 'void kernel_main()':\n"
        f"{path}:{line}:{column}: error: {MESSAGE}\n"
        f"    {line} |     add_tiles_bcast_x(0, 1, 0, 0, 0);\n"
        "      |     ^~~~~~~~~~~~~~~~~\n"
        "\n"
        "terminate called after throwing an instance of 'std::runtime_error'\n"
    )


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
# ttmetal-host builds a model-written program with its kernels


@dataclass
class FakeRunner:
    """A CommandRunner that records each call and writes a PLACEHOLDER artifact `main`; it runs nothing."""

    calls: list[list[str]] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Record the call and report a successful build."""
        self.calls.append(list(argv))
        (Path(cwd) / "main").write_bytes(b"PLACEHOLDER artifact of a fake compile\n")
        return CommandResult(0, "", "")


@pytest.mark.parametrize("item", sorted(KH_SUITE.items))
def test_ttmetal_host_compiles_the_host_program_and_places_every_model_kernel(item: str, tmp_path: Path) -> None:
    files = {path: f"// SYNTHETIC model file {path}\n" for path in KH_SUITE.items[item].languages["ttkh"].files}
    harness = KH_SUITE.support_files(item, tmp_path / "sources", purpose="eval", language="ttkh")
    runner = FakeRunner()
    workdir = tmp_path / "build"
    workdir.mkdir()
    result = TtMetalHost(tree=Path("/SYNTHETIC/tree"), runner=runner).build(files, workdir, harness=harness)
    host, *kernels = files
    assert result.artifact == workdir / "main" and result.diagnostics == []
    ((argv),) = runner.calls
    assert host in argv and not any(kernel in argv for kernel in kernels), "a kernel is never a host source"
    for path, text in files.items():
        assert (workdir / path).read_text(encoding="utf-8") == text, f"{path} lies at its path, as written"
    assert (workdir / "lassi_io.h").is_file()


# ---------------------------------------------------------------------------
# The executor's reading of a kernel JIT error in a model kernel


def test_a_jit_error_names_the_model_kernel_by_its_path_in_the_run() -> None:
    found = ttsim.read_jit_diagnostics(jit_log(KERNEL), "", WORKDIR)
    errors = [item for item in found if item.severity == "error"]
    assert [(item.stage, item.file, item.line, item.column, item.message) for item in errors] == [
        ("jit", KERNEL, 3, 5, MESSAGE)
    ]
    run = RunResult(exit_code=134, hang=False, stdout=jit_log(KERNEL), stderr="", diagnostics=found)
    assert run_stage(run) == PARSED


# ---------------------------------------------------------------------------
# Through the runner: the correction loop feeds the parsed diagnostic back


def write_manifest(directory: Path, bench: Path) -> Path:
    """Write the SYNTHETIC khfix manifest under `directory` and its fetched sources under `bench`."""
    harness_sha = hashlib.sha256((REPO / HARNESS).read_bytes()).hexdigest()
    kernel_sha = hashlib.sha256(SOURCES[KERNEL].encode("ascii")).hexdigest()
    languages = {
        "cpp": {"dir": "src/scale-cpp", "files": ["main.cpp"]},
        "tt": {"dir": "src/scale-tt", "files": [HOST],
               "support": {KERNEL: {"upstream": f"src/scale-tt/{KERNEL}", "sha256": kernel_sha}}},
        "ttkh": {"dir": "assets/bench/khfix/ttkh", "tracked": True, "files": [HOST, KERNEL], "reference": "tt",
                 "support": {"lassi_io.h": {"tracked": HARNESS, "sha256": harness_sha}}},
    }
    data = {"suite": SUITE, "repo": "https://example.invalid/khfix", "commit": FAKE_COMMIT,
            "items": {ITEM: {"split": "eval", "languages": languages}}}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{SUITE}.yaml").write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    written = {"src/scale-cpp/main.cpp": SOURCES["cpp"], f"src/scale-tt/{HOST}": SOURCES["tt"],
               f"src/scale-tt/{KERNEL}": SOURCES[KERNEL]}
    for path, text in written.items():
        (bench / path).parent.mkdir(parents=True, exist_ok=True)
        (bench / path).write_bytes(text.encode("ascii"))
    return directory


def recipe_data(**changes: Any) -> dict[str, Any]:
    """Return a one-trial cpp -> ttkh recipe on tt-kh-v0 whose ttkh reference is the item's tt program."""
    data: dict[str, Any] = {
        "extends": "base",
        "faithful": False,
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": SOURCE, "target": TARGET}],
        "prompts": "tt-kh-v0",
        "toolchain": {"cpp": "fake-cpu", "tt": "fake-tt", "ttkh": "fake-tt"},
        "stages": ["baseline", "generate", "compile_loop", "run_loop"],
        "executor": {"kind": SIM_EXECUTOR},
        "fixes": {"baseline_both": False},
        "trials": {"n": 1},
    }
    data.update(changes)
    return data


def run_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: Script, data: Mapping[str, Any]) -> Outcome:
    """Run the one-trial recipe on the SYNTHETIC suite with the fakes following `script`; return the outcome."""
    bench = tmp_path / "bench"
    monkeypatch.setattr(runner_module, "BENCH_DIR", write_manifest(tmp_path / "manifests", bench))
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=bench,
                         registry=make_registry(script))
    path = tmp_path / "kh-run.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    run_dir = run_recipe(path, options)
    trial_id = make_trial_id("kh-run", MODEL_ID, SUITE, f"{SOURCE}-{TARGET}", ITEM, 1)
    return Outcome(run_dir, read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir)), script)


def jit_failed(log: str) -> RunResult:
    """Return the SYNTHETIC run of an attempt whose kernel build failed, read as the ttsim executor reads it."""
    return sim_run(exit_code=134, stdout=log, stderr="", sim_ub=False,
                   diagnostics=ttsim.read_jit_diagnostics(log, "", WORKDIR))


def test_a_kernel_jit_error_is_corrected_with_its_parsed_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Script(
        replies=[render_file_blocks(MODEL_FILES), render_file_blocks(FIXED_FILES)],
        references={"tt": clean_sim_reference()},
        attempt_runs=[jit_failed(jit_log(KERNEL)), clean_sim_run()],
    )
    outcome = run_one(tmp_path, monkeypatch, script, recipe_data())
    trial = outcome.trial
    assert stages_of(trial) == ["S1", "S5"], "the JIT failure leaves attempt 0 at S1; the correction runs clean"
    first = trial.attempts[0]
    assert [(item.file, item.line, item.message) for item in errors_of(first, "jit")] == [(KERNEL, 3, MESSAGE)]
    assert not [item for item in errors_of(first, "run") if item.code == "run-error"], "no opaque run failure"
    assert trial.requests is not None and [request.stage for request in trial.requests] == ["generate", "compile_loop"]
    correction = outcome.prompt(1)
    assert f"error {KERNEL}:3:5 {MESSAGE}" in correction, "the parsed diagnostic: severity, file:line:column, message"
    assert MODEL_FILES[KERNEL] in correction, "the model's own kernel, shown back with its path"
    assert "first launches" in " ".join(correction.split())
    for index in (0, 1):
        assert SOURCES[KERNEL] not in outcome.prompt(index), "the upstream kernel never enters a prompt"


def test_a_kernel_build_failure_without_a_compiler_line_is_corrected_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = (
        "trisc2 build failed. Log: SYNTHETIC kernels/compute/k.cpp:(.text+0x10): undefined reference to `helper()'\n"
        "SYNTHETIC collect2: error: ld returned 1 exit status\n\n"
    )
    script = Script(
        replies=[render_file_blocks(MODEL_FILES), render_file_blocks(FIXED_FILES)],
        references={"tt": clean_sim_reference()},
        attempt_runs=[jit_failed(log), clean_sim_run()],
    )
    outcome = run_one(tmp_path, monkeypatch, script, recipe_data())
    assert stages_of(outcome.trial) == ["S1", "S5"]
    correction = outcome.prompt(1)
    assert f"error [{ttsim.JIT_BUILD_FAILED}] trisc2 kernel build failed" in correction
    assert "undefined reference to `helper()'" in correction
