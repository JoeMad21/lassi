"""Tests for how trial.md and run.md label the wall times of simulator runs (task P4.6).

Bible: Agent Rule 2 (never report simulator timing as performance), ttsim
Facts (timing and cycle counters are not meaningful; no performance numbers
come from ttsim), Readability Standards (Trial and Run rows), Result Record
(RunInfo.wall_s, final.wall_s), Component Interfaces (the capability rule).
Plan: plans/p4-ttsim.md, P4.6, second acceptance line.

The contract these tests fix:

- When the run's executor declares the capability `simulator`, trial.md
  labels each run's wall time as simulator wall time, not performance: in
  the Reference run table and in each attempt's Run table, the wall_s row
  (its first cell starts with `wall_s`) says "simulator wall time" and "not
  performance" (any letter case) beside the value, which it still shows.
- With that capability, run.md's Trials section, whose table shows each
  trial's wall_s, says "simulator wall time" and "not performance" too.
- Without the capability both pages print as they do today: the rows read
  `| wall_s | <value> |` and run.md's trials table keeps its header row,
  with no simulator label on either page.
- A simulator run fills no performance field: Attempt.profile.runtime_s
  stays null (no profiler exists, and a simulator wall time is never one).

The fake executors run nothing and return SYNTHETIC RunResults built without
any field this task adds, so these tests read the capability alone. The
suite manifest, sources, and replies are SYNTHETIC; fake toolchains compile
nothing. Wall times are PLACEHOLDER fixture values, never measurements.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.core import runner as runner_module
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Limits, Message, RunResult, Sampling
from lassi.core.record import Trial, make_trial_id
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.store import TextStore, read_trial, trial_dir

SUITE = "simfix"
ITEM = "vadd"
MODEL_ID = "scripted-fixture"
SOURCE, TARGET = "cpu", "tt"
DIRECTION = f"{SOURCE}-{TARGET}"
TARGET_FILE = "host.cpp"
TOOLCHAINS = {SOURCE: "fake-cpu", TARGET: "fake-tt"}
STAGES = ["baseline", "generate", "compile_loop", "run_loop"]
SIMULATOR = "simulator"
EXECUTORS = {
    "fake-sim": frozenset({"runs_code", "sandboxed", SIMULATOR}),
    "scripted": frozenset({"runs_code", "sandboxed"}),
}
# A commit id for the synthetic manifest; not a commit of this repository or of any source.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"

SIM_LABEL = re.compile(r"simulator wall time", re.IGNORECASE)
NOT_PERFORMANCE = re.compile(r"not performance", re.IGNORECASE)
# run.md's trials table header row as it prints today.
TRIALS_HEADER = "| trial | direction | item | stage reached | corrections | wall_s | trial.md |"

SOURCES = {
    SOURCE: "// SYNTHETIC cpu reference of a made-up vector add\nint main() { return 0; }\n",
    TARGET: "// SYNTHETIC tt reference host program of a made-up vector add\nint main() { return 0; }\n",
}
REPLY = render_file_blocks({TARGET_FILE: "// SYNTHETIC candidate host program\nint main() { return 0; }\n"})
PLACEHOLDER_REFERENCE_WALL_S = 1.25
PLACEHOLDER_ATTEMPT_WALL_S = 0.5


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
# Fake components


@dataclass
class Script:
    """The replies the backend gives, in order, and how many runs the executor made."""

    replies: list[str] = field(default_factory=list)
    runs: int = 0


def fake_toolchain(registered_as: str) -> type:
    """Return a Toolchain class without PIN that writes the files and a PLACEHOLDER artifact; it compiles nothing."""

    class FakeToolchain:
        """Writes the files under the fresh workdir and reports a built artifact."""

        name = registered_as
        capabilities = frozenset({"diagnostics"})

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Write `files` and a placeholder artifact, and return it with no diagnostics."""
            for path, text in [*files.items(), *(harness or {}).items()]:
                target = Path(workdir) / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            artifact = Path(workdir) / "main"
            artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=[])

    return FakeToolchain


def executor_class(registered_as: str, declared: frozenset[str], script: Script) -> type:
    """Return an Executor class whose every run is a clean SYNTHETIC RunResult; it starts no process."""

    class FakeExecutor:
        """Returns a clean run: the reference's PLACEHOLDER wall time for a baseline run, the attempt's otherwise."""

        name = registered_as
        capabilities = declared

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Return a clean SYNTHETIC RunResult, built without any field task P4.6 adds."""
            script.runs += 1
            reference = Path(artifact).parent.parent.name.startswith("baseline-")
            wall_s = PLACEHOLDER_REFERENCE_WALL_S if reference else PLACEHOLDER_ATTEMPT_WALL_S
            return RunResult(exit_code=0, hang=False, stdout="SYNTHETIC stdout\n", stderr="", wall_s=wall_s)

        def device(self) -> str:
            """Return a SYNTHETIC device name."""
            return f"SYNTHETIC {registered_as} device"

    return FakeExecutor


def backend_class(script: Script) -> type:
    """Return an LLMBackend class, registered as "scripted", that answers from script.replies in order."""

    class ScriptedBackend:
        """Answers with the next scripted reply."""

        name = "scripted"
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id, as every backend does."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Return the next scripted reply."""
            assert script.replies, "the backend was asked for more replies than the script holds"
            return Completion(text=script.replies.pop(0), prompt_tokens=0, completion_tokens=0)

    return ScriptedBackend


def make_registry(script: Script) -> Registry:
    """Return a test Registry: the scripted backend, both fake executors, fake toolchains, and the real stages."""
    registry = Registry()
    registry.register("LLMBackend", "scripted", backend_class(script))
    for name, declared in EXECUTORS.items():
        registry.register("Executor", name, executor_class(name, declared, script))
    for name in TOOLCHAINS.values():
        registry.register("Toolchain", name, fake_toolchain(name))
    for name in STAGES:
        registry.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    return registry


# ---------------------------------------------------------------------------
# One run and its pages


@dataclass
class Pages:
    """One finished run: its one trial, that trial's trial.md, and run.md."""

    trial: Trial
    trial_md: str
    run_md: str


def write_suite(root: Path) -> Path:
    """Write the SYNTHETIC simfix manifest and sources under `root`; return the manifests directory."""
    languages = {
        SOURCE: {"dir": "src/vadd-cpu", "files": ["main.cpp"]},
        TARGET: {"dir": "src/vadd-tt", "files": [TARGET_FILE]},
    }
    data = {
        "suite": SUITE, "repo": "https://example.invalid/simfix", "commit": FAKE_COMMIT,
        "items": {ITEM: {"split": "eval", "languages": languages}},
    }
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    (manifests / f"{SUITE}.yaml").write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    for language, layout in languages.items():
        path = root / "bench" / layout["dir"] / layout["files"][0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(SOURCES[language].encode("ascii"))
    return manifests


def recipe_data(executor: str) -> dict[str, Any]:
    """Return a one-trial template-set recipe for the item, cpu to tt, run on `executor`."""
    return {
        "extends": "base",
        "faithful": False,
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": SOURCE, "target": TARGET}],
        "prompts": "p0-smoke",
        "toolchain": dict(TOOLCHAINS),
        "stages": list(STAGES),
        "executor": {"kind": executor},
        "fixes": {"baseline_both": False},
        "trials": {"n": 1},
    }


def run_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, executor: str) -> Pages:
    """Run one clean trial on `executor` and return the trial and both pages as the run tree holds them."""
    monkeypatch.setattr(runner_module, "BENCH_DIR", write_suite(tmp_path))
    script = Script(replies=[REPLY])
    options = RunOptions(
        runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=tmp_path / "bench",
        registry=make_registry(script),
    )
    path = tmp_path / "sim-labels.yaml"
    path.write_bytes(yaml.safe_dump(recipe_data(executor), sort_keys=False).encode("ascii"))
    run_dir = run_recipe(path, options)
    folder = trial_dir(run_dir, make_trial_id("sim-labels", MODEL_ID, SUITE, DIRECTION, ITEM, 1))
    trial = read_trial(folder, TextStore(run_dir))
    assert script.runs == 2 and [attempt.stage_reached for attempt in trial.attempts] == ["S5"], "one clean trial"
    trial_md = (folder / "trial.md").read_text(encoding="ascii")
    return Pages(trial, trial_md, (run_dir / "run.md").read_text(encoding="ascii"))


def cells(line: str) -> list[str]:
    """Return the stripped cells of one Markdown table row."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def wall_row(page: str, heading: str, after: str | None = None) -> list[str]:
    """Return the cells of the row whose first cell starts with wall_s, in the table under `heading`.

    The table is the first after the line `after` when given. The test fails
    when the table has no such row.
    """
    lines = page.split("\n")
    start = lines.index(heading, lines.index(after) if after is not None else 0)
    for line in lines[start + 1:]:
        if line.startswith("#"):
            break
        if line.startswith("| ") and cells(line)[0].startswith("wall_s"):
            return cells(line)
    pytest.fail(f"the table under {heading!r} has no wall_s row")


def trials_section(run_md: str) -> str:
    """Return run.md's Trials section: from its heading to the next second-level heading or the end."""
    start = run_md.index("## Trials\n")
    end = run_md.find("\n## ", start + 1)
    return run_md[start:] if end == -1 else run_md[start:end]


# ---------------------------------------------------------------------------
# With the simulator capability


def test_trial_md_labels_each_run_wall_time_as_simulator_wall_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = run_pages(tmp_path, monkeypatch, "fake-sim")
    rows = {
        "reference run": (wall_row(pages.trial_md, "## Reference run"), repr(PLACEHOLDER_REFERENCE_WALL_S)),
        "attempt 0 run": (wall_row(pages.trial_md, "### Run", after="## Attempt 0"), repr(PLACEHOLDER_ATTEMPT_WALL_S)),
    }
    for where, (row, value) in rows.items():
        text = " | ".join(row)
        assert SIM_LABEL.search(text), f"trial.md's {where} wall time is not labeled simulator wall time: {row}"
        assert NOT_PERFORMANCE.search(text), f"trial.md's {where} wall time does not say it is not performance: {row}"
        assert value in text, f"trial.md's {where} row still shows the wall time {value}: {row}"


def test_run_md_labels_the_trials_wall_times_as_simulator_wall_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    section = trials_section(run_pages(tmp_path, monkeypatch, "fake-sim").run_md)
    assert SIM_LABEL.search(section), f"run.md's Trials section does not label its wall times:\n{section}"
    assert NOT_PERFORMANCE.search(section), f"run.md's Trials section does not say not performance:\n{section}"


def test_a_simulator_run_fills_no_performance_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trial = run_pages(tmp_path, monkeypatch, "fake-sim").trial
    assert [attempt.profile.runtime_s for attempt in trial.attempts] == [None], "a simulator wall time is no runtime"


# ---------------------------------------------------------------------------
# Without it, both pages print as today


def test_without_the_simulator_capability_trial_md_prints_wall_times_as_today(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = run_pages(tmp_path, monkeypatch, "scripted").trial_md
    assert wall_row(page, "## Reference run") == ["wall_s", repr(PLACEHOLDER_REFERENCE_WALL_S)]
    assert wall_row(page, "### Run", after="## Attempt 0") == ["wall_s", repr(PLACEHOLDER_ATTEMPT_WALL_S)]
    assert not SIM_LABEL.search(page), "no simulator label without the capability"


def test_without_the_simulator_capability_run_md_prints_wall_times_as_today(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_md = run_pages(tmp_path, monkeypatch, "scripted").run_md
    assert TRIALS_HEADER in run_md.split("\n"), "run.md's trials table keeps today's header row"
    assert not SIM_LABEL.search(run_md), "no simulator label without the capability"


# ---------------------------------------------------------------------------
# Executors per language: a simulator for tt, a plain executor for cpu (the P4.6 commit audit)

CPU_REPLY = render_file_blocks({"main.cpp": "// SYNTHETIC candidate cpu program\nint main() { return 0; }\n"})
TO_TT = {"source": SOURCE, "target": TARGET}
TO_CPU = {"source": TARGET, "target": SOURCE}


def run_mixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directions: list[dict[str, str]], baseline_both: bool
) -> Path:
    """Run one trial per direction with tt on fake-sim and cpu on scripted; return the run directory."""
    monkeypatch.setattr(runner_module, "BENCH_DIR", write_suite(tmp_path))
    replies = [REPLY if direction["target"] == TARGET else CPU_REPLY for direction in directions]
    options = RunOptions(
        runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=tmp_path / "bench",
        registry=make_registry(Script(replies=replies)),
    )
    data = recipe_data("fake-sim")
    data.update(directions=directions, executor={SOURCE: "scripted", TARGET: "fake-sim"})
    data["fixes"] = {"baseline_both": baseline_both}
    path = tmp_path / "sim-labels.yaml"
    path.write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    return run_recipe(path, options)


def trial_page(run_dir: Path, direction: str) -> str:
    """Return the trial.md of the one trial of `direction` in the run."""
    folder = trial_dir(run_dir, make_trial_id("sim-labels", MODEL_ID, SUITE, direction, ITEM, 1))
    return (folder / "trial.md").read_text(encoding="ascii")


def simulator_notes(run_dir: Path) -> list[str]:
    """Return the lines of run.md's Trials section that carry the simulator label."""
    section = trials_section((run_dir / "run.md").read_text(encoding="ascii"))
    return [line for line in section.split("\n") if SIM_LABEL.search(line)]


def test_with_executors_per_language_only_the_simulator_languages_runs_are_labeled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = run_mixed(tmp_path, monkeypatch, [TO_TT, TO_CPU], baseline_both=False)
    to_tt, to_cpu = trial_page(run_dir, "cpu-tt"), trial_page(run_dir, "tt-cpu")
    for row in (wall_row(to_tt, "## Reference run"), wall_row(to_tt, "### Run", after="## Attempt 0")):
        assert SIM_LABEL.search(" | ".join(row)) and NOT_PERFORMANCE.search(" | ".join(row)), row
    assert wall_row(to_cpu, "## Reference run") == ["wall_s", repr(PLACEHOLDER_REFERENCE_WALL_S)]
    assert wall_row(to_cpu, "### Run", after="## Attempt 0") == ["wall_s", repr(PLACEHOLDER_ATTEMPT_WALL_S)]
    assert not SIM_LABEL.search(to_cpu), "a cpu-target trial ran nothing on the simulator"
    notes = simulator_notes(run_dir)
    assert len(notes) == 1 and notes[0].startswith("The tt programs "), f"run.md's note names tt alone: {notes}"


@pytest.mark.parametrize("baseline_both", [True, False], ids=["source-reference-runs", "no-source-reference"])
def test_run_md_names_a_simulator_source_language_only_when_the_baseline_runs_its_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, baseline_both: bool
) -> None:
    run_dir = run_mixed(tmp_path, monkeypatch, [TO_CPU], baseline_both=baseline_both)
    assert not SIM_LABEL.search(trial_page(run_dir, "tt-cpu")), "trial.md records no run of the source reference"
    notes = simulator_notes(run_dir)
    if baseline_both:
        assert len(notes) == 1 and notes[0].startswith("The tt programs "), notes
    else:
        assert notes == [], "no program of this run ran on the simulator"
