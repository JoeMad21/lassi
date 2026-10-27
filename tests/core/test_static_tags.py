"""Tests for compile_loop's static tags: a toolchain's text reading of each built program (task P17.40).

Bible: Harness Contract (the OpenMP-construct tag: tagged, never failed),
Component Interfaces (capability rule), Result Record (Attempt.diagnostics),
Agent Rule 3. Plan: plans/p17-portable.md, P17.40.

The contract these tests fix (lassi.core.stages compile_loop, _build_last):

- After every build that gave a program, and after the host-compute
  reading, compile_loop asks the target language's toolchain for its static
  tags by capability (lassi.core.capabilities static_tags, never by name),
  with the attempt's files and the item's support files for the target
  language, and appends them to the attempt's diagnostics. A build with no
  program, and a toolchain without the capability, get none.
- A tag changes no stage, asks for no correction, and leaves the attempt's
  guards as they were; its code starts with "guard-", so it never enters a
  correction prompt (prompt_diagnostics).
- describe() adds " (static_tags after each build)" after the builder's
  name only when the target toolchain declares the capability.

The fake toolchains compile nothing; the manifest, sources, and diagnostics
are SYNTHETIC. No value here is a measurement.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from lassi.bench import Direction, load_suite
from lassi.core.interfaces import BuildResult, Sampling
from lassi.core.record import (
    NO_GUARD,
    Attempt,
    BenchItem,
    Diagnostic,
    Final,
    Guards,
    ModelInfo,
    Provenance,
    Trial,
    make_trial_id,
)
from lassi.core.stages import CompileLoopStage, RunContext, prompt_diagnostics
from lassi.core.store import TextStore

SUITE, ITEM, MODEL_ID = "tagfix", "sum", "scripted-fixture"
SOURCE, TARGET = "cpp", "omp"
TARGET_FILE = "sum.cpp"
SUPPORT, SUPPORT_PATH, SUPPORT_TEXT = "lassi_io.h", "src/harness/lassi_io.h", "/* SYNTHETIC header */\n"
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
FILES = {TARGET_FILE: "// SYNTHETIC candidate\nint main() { return 0; }\n"}
TAG = Diagnostic(
    stage="parse", severity="warning", code="guard-openmp-none", file=TARGET_FILE,
    message="SYNTHETIC tag: no OpenMP construct",
)
COMPILE_WARNING = Diagnostic(
    stage="compile", severity="warning", code="-Wunused-variable", file=TARGET_FILE, line=1, column=5,
    message="SYNTHETIC compile warning",
)
COMPILE_ERROR = Diagnostic(
    stage="compile", severity="error", code=None, file=TARGET_FILE, line=1, column=5, message="SYNTHETIC error",
)
PARSE_WARNING = Diagnostic(stage="parse", severity="warning", code="invalid-text", message="SYNTHETIC parse warning")


def toolchain(declared: frozenset[str], builds: list[str], calls: list[Any]) -> Any:
    """Return a fake toolchain: each build "ok" (a program and a warning) or "fail"; tags are scripted."""

    class Fake:
        name = "fake-omp"
        capabilities = declared

        def build(self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None) -> Any:
            """Return the next scripted build."""
            if builds.pop(0) == "fail":
                return BuildResult(artifact=None, diagnostics=[COMPILE_ERROR])
            artifact = Path(workdir) / "main"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"PLACEHOLDER artifact\n")
            return BuildResult(artifact=artifact, diagnostics=[COMPILE_WARNING])

        def static_tags(self, files: Mapping[str, str], harness: Mapping[str, str]) -> tuple[Diagnostic, ...]:
            """Record the call and return the one tag."""
            calls.append((dict(files), dict(harness)))
            return (TAG,)

    return Fake()


def context(tmp_path: Path, tool: Any) -> RunContext:
    """Return a RunContext for cpp -> omp of the SYNTHETIC item with the fake target toolchain."""
    languages = {
        SOURCE: {"dir": "src/sum-cpp", "files": ["sum.cpp"]},
        TARGET: {"dir": "src/sum-omp", "files": [TARGET_FILE]},
    }
    item = {"split": "eval", "languages": languages, "support": {SUPPORT: SUPPORT_PATH}}
    data = {"suite": SUITE, "repo": "https://example.invalid/tagfix", "commit": FAKE_COMMIT, "items": {ITEM: item}}
    manifests, bench = tmp_path / "manifests", tmp_path / "bench"
    manifests.mkdir(parents=True)
    (manifests / f"{SUITE}.yaml").write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    (bench / SUPPORT_PATH).parent.mkdir(parents=True)
    (bench / SUPPORT_PATH).write_bytes(SUPPORT_TEXT.encode("ascii"))
    return RunContext(
        recipe=SimpleNamespace(data={"fixes": {}}, bindings=()),  # type: ignore[arg-type]
        backend=SimpleNamespace(),  # type: ignore[arg-type]
        sampling=SAMPLING,
        toolchains={TARGET: tool},
        executor=SimpleNamespace(capabilities=frozenset()),  # type: ignore[arg-type]
        store=TextStore(tmp_path / "store"),
        suite=load_suite(manifests / f"{SUITE}.yaml"),
        sources_root=bench,
        item=ITEM,
        direction=Direction(SOURCE, TARGET),
        build_root=tmp_path / "builds",
        prompts="p0-smoke",
        max_corrections=0,
    )


def trial_of(run: RunContext) -> Trial:
    """Return a SYNTHETIC trial holding one parsed attempt of FILES."""
    attempt = Attempt(
        index=0, prompt_ref=run.store.put("SYNTHETIC prompt\n"), response_text="SYNTHETIC reply", files=dict(FILES),
        stage_reached="S1", diagnostics=[PARSE_WARNING],
    )
    return Trial(
        trial_id=make_trial_id("tag-test", MODEL_ID, SUITE, f"{SOURCE}-{TARGET}", ITEM, 1),
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(
            commit=FAKE_COMMIT, dirty=False, device="SYNTHETIC", sdk=None, date="2026-10-26T00:00:00+00:00"
        ),
        bench_item=BenchItem(suite=SUITE, item=ITEM, split="eval", direction=f"{SOURCE}-{TARGET}"),
        model=ModelInfo(backend="scripted", id=MODEL_ID, sampling=SAMPLING),
        requests=[],
        attempts=[attempt],
        final=Final(stage_reached="S1"),
    )


def test_tags_follow_the_build_diagnostics_and_change_no_stage(tmp_path: Path) -> None:
    calls: list[Any] = []
    run = context(tmp_path, toolchain(frozenset({"diagnostics", "static_tags"}), ["ok"], calls))
    (built,) = CompileLoopStage(context=run)(trial_of(run)).attempts
    assert built.diagnostics == [PARSE_WARNING, COMPILE_WARNING, TAG], "parse, then build, then the tags"
    assert built.stage_reached == "S4"
    assert built.guards == Guards(host_compute_not_checked=NO_GUARD), "the tag leaves the guards as they were"
    assert calls == [(FILES, {SUPPORT: SUPPORT_TEXT})]


def test_a_build_with_no_program_is_not_tagged(tmp_path: Path) -> None:
    calls: list[Any] = []
    run = context(tmp_path, toolchain(frozenset({"diagnostics", "static_tags"}), ["fail"], calls))
    trial = CompileLoopStage(context=run)(trial_of(run))
    assert TAG not in trial.attempts[0].diagnostics and calls == []


def test_a_toolchain_without_the_capability_is_not_asked(tmp_path: Path) -> None:
    calls: list[Any] = []
    run = context(tmp_path, toolchain(frozenset({"diagnostics"}), ["ok"], calls))
    (built,) = CompileLoopStage(context=run)(trial_of(run)).attempts
    assert built.diagnostics == [PARSE_WARNING, COMPILE_WARNING] and calls == []


def test_a_tag_never_enters_a_prompt() -> None:
    assert prompt_diagnostics([COMPILE_WARNING, TAG]) == [COMPILE_WARNING]


def test_describe_names_the_tags_only_when_declared(tmp_path: Path) -> None:
    tagged = context(tmp_path / "a", toolchain(frozenset({"diagnostics", "static_tags"}), [], []))
    plain = context(tmp_path / "b", toolchain(frozenset({"diagnostics"}), [], []))
    assert "fake-omp (static_tags after each build), then correct" in CompileLoopStage(context=tagged).describe()
    assert "fake-omp, then correct" in CompileLoopStage(context=plain).describe()
