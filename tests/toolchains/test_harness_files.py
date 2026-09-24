"""Tests for harness files in a compiler toolchain's build (task P1.2).

Bible: Harness Contract (the harness owns inputs and comparison), Source
Papers (LASSI quirk table, entropy row: `reference.h` comes from pinned
HeCBench), Component Interfaces (Toolchain row).

A bench item's support files, such as entropy's `reference.h`, are harness
files: CompilerToolchain.build(files, workdir, harness=...) writes them
(relative path -> text) into the build directory beside the model's files,
and a model file with the same path cannot replace them. Both compiler
presets are checked. A recording runner stands in for the compiler: it keeps
a copy of what the build directory holds when the command would run and
writes a placeholder artifact; no process starts. The header texts are
synthetic, not HeCBench's. No value here is a measurement.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from lassi.toolchains import CommandResult, NvccSm80, NvcppCc80

PRESETS = [(NvccSm80, "main.cu"), (NvcppCc80, "main.cpp")]
HARNESS = "reference.h"
HARNESS_TEXT = "#pragma once\n// PLACEHOLDER synthetic harness header\nstatic int lassi_harness_value = 1;\n"
MODEL_TEXT = "#pragma once\n// a model's own header of the same name\nstatic int lassi_harness_value = 2;\n"
SOURCE = '#include "reference.h"\nint main() { return lassi_harness_value - 1; }\n'


class RecordingRunner:
    """A CommandRunner that records the build directory at compile time and writes a placeholder artifact."""

    def __init__(self) -> None:
        """Start with no recorded compile."""
        self.snapshots: list[dict[str, bytes]] = []

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Record every file under `cwd` (relative path -> bytes), write the artifact `main`, and succeed."""
        cwd = Path(cwd)
        self.snapshots.append({p.relative_to(cwd).as_posix(): p.read_bytes() for p in cwd.rglob("*") if p.is_file()})
        (cwd / "main").write_bytes(b"PLACEHOLDER artifact of the recording runner\n")
        return CommandResult(returncode=0, stdout="", stderr="")


def files_under(workdir: Path) -> dict[str, bytes]:
    """Return every file under `workdir` as relative path -> bytes."""
    return {p.relative_to(workdir).as_posix(): p.read_bytes() for p in workdir.rglob("*") if p.is_file()}


@pytest.mark.parametrize(("preset", "source"), PRESETS)
def test_harness_files_land_in_the_build_directory(preset: type, source: str, tmp_path: Path) -> None:
    runner = RecordingRunner()
    workdir = tmp_path / "build"
    workdir.mkdir()
    result = preset(runner=runner).build({source: SOURCE}, workdir, harness={HARNESS: HARNESS_TEXT})
    assert result.artifact is not None, result.diagnostics
    assert len(runner.snapshots) == 1
    assert runner.snapshots[0][HARNESS] == HARNESS_TEXT.encode("utf-8")
    assert runner.snapshots[0][source] == SOURCE.encode("utf-8")
    assert files_under(workdir)[HARNESS] == HARNESS_TEXT.encode("utf-8")


@pytest.mark.parametrize(("preset", "source"), PRESETS)
def test_a_model_file_cannot_replace_a_harness_file(preset: type, source: str, tmp_path: Path) -> None:
    runner = RecordingRunner()
    workdir = tmp_path / "build"
    workdir.mkdir()
    files = {source: SOURCE, HARNESS: MODEL_TEXT}
    preset(runner=runner).build(files, workdir, harness={HARNESS: HARNESS_TEXT})
    # Whether the build refuses the model's file or keeps the harness file over it, the compiler never sees the
    # model's text under the harness file's name, and the build directory never holds it.
    for snapshot in runner.snapshots:
        assert snapshot.get(HARNESS) == HARNESS_TEXT.encode("utf-8")
    on_disk = files_under(workdir)
    assert on_disk.get(HARNESS, HARNESS_TEXT.encode("utf-8")) == HARNESS_TEXT.encode("utf-8")
    assert MODEL_TEXT.encode("utf-8") not in on_disk.values()
