"""Tests for the P4.7 follow-ups that task P4.8 takes (plans/PHASE-NOTES.md, P4, "P4.7 follow-ups").

Bible: Readability Standards, Terminal Presentation (the graphics setting:
a bad `--graphics` value is the argument parser's usage error, exit 2 with
the value named; the preset is a YAML mapping with the one key graphics,
and any other key is refused; a preset that cannot be read, checked, or
saved is a usage refusal, exit 2 with `lassi <command>: <message>` before
the command starts; on a POSIX host no preset is read or saved on the
filesystem that holds /, and a prompted answer then applies to that
command only and the prompt names LASSI_CONFIG_DIR).

The follow-ups that are behaviour, one group of tests each:

- a bad --graphics value is the parser's usage error (a CLI test the P4.7
  audit asked for; the behaviour exists since P4.7);
- a YAML merge key (<<) in the preset is refused as any other key is, by
  lassi.present.settings.load_preset and so by every command that reads the
  preset (new behaviour: the loader accepted it at 755c739);
- the exit-2 cases lassi/cli.py's docstrings must list exactly: a failed
  save at the prompt exits 2 before the command starts, while a
  root-filesystem refusal exits 2 only for `lassi settings graphics on|off`;
  `lassi run` then asks, applies the answer once, and runs (both
  behaviours exist since P4.7; these tests keep them while the docstrings
  are corrected).

The other follow-ups (the Terminal Presentation opening sentence, the
Decision Log rationale, and the "first-run prompt" wording) are text, not
behaviour, and have no test here.

`lassi run` uses the real mock backend, stages, and none executor, with a
fake toolchain registered as nvcc-sm80 in place of the default registry,
as tests/present/test_present_cli.py does. The device check is patched so
that "/" alone, or a directory a test names, counts as the root
filesystem. No value here is a measurement.
"""

from __future__ import annotations

import importlib
import io
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from lassi import cli
from lassi.core import runner as runner_module
from lassi.core.interfaces import BuildResult
from lassi.core.registry import Registry
from lassi.core.stages import CompileLoopStage, GenerateStage
from lassi.executors import NoneExecutor
from lassi.llm import MockBackend

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "tests" / "fixtures" / "recipes" / "p0-smoke.yaml"
ROOT_DEVICE = 1
OTHER_DEVICE = 2
FAKE_OMP_SOURCE = "#include <cstdio>\nint main() {\n#pragma omp target\n  { }\n  return 0;\n}\n"
FAKE_CUDA_SOURCE = "__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n"

# Presets that hold a YAML merge key; each read as a valid preset at 755c739.
MERGE_PRESETS = {
    "merge-only": "<<: {graphics: on}\n",
    "empty-merge": "graphics: on\n<<: {}\n",
    "merge-list": "<<: [{graphics: off}]\n",
}


def settings() -> ModuleType:
    """Import and return lassi.present.settings."""
    return importlib.import_module("lassi.present.settings")


# ---------------------------------------------------------------------------
# Environment and fixtures


@pytest.fixture(autouse=True)
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the inherited variables, put the preset under <tmp>/config and TMPDIR under <tmp>; "/" alone is root."""
    for name in ("LASSI_GRAPHICS", "LASSI_CONFIG_DIR", "LASSI_SCRATCH", "LASSI_RUNS_ROOT", "LASSI_TOOLCHAINS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LASSI_CONFIG_DIR", str(tmp_path / "config"))
    (tmp_path / "compile-tmp").mkdir()
    monkeypatch.setenv("TMPDIR", str(tmp_path / "compile-tmp"))
    root_side(monkeypatch)


def root_side(monkeypatch: pytest.MonkeyPatch, *directories: Path) -> ModuleType:
    """Patch the settings module's device check: "/" and `directories` on the root filesystem, the rest off it."""
    module = settings()
    sides = [Path(side).resolve() for side in directories]

    def device(path: Path) -> int:
        """Return the fake device number of `path`."""
        path = Path(path)
        if path.parent == path:
            return ROOT_DEVICE
        path = path.resolve()
        return ROOT_DEVICE if any(path == side or side in path.parents for side in sides) else OTHER_DEVICE

    monkeypatch.setattr(module, "path_device", device)
    return module


class FakeNvcc:
    """A Toolchain without PIN registered as nvcc-sm80: it writes the files and a placeholder artifact."""

    name = "nvcc-sm80"
    capabilities = frozenset({"diagnostics"})

    def build(self, files: Mapping[str, str], workdir: Path) -> BuildResult:
        """Write `files` under `workdir` and return a placeholder artifact; nothing is compiled."""
        workdir = Path(workdir)
        for relative, text in files.items():
            target = workdir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode("utf-8"))
        artifact = workdir / "main"
        artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
        return BuildResult(artifact=artifact, diagnostics=[])


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a registry with the mock backend and the fake toolchain in place of the default; return the bench root."""
    registry = Registry()
    registry.register("LLMBackend", "mock", MockBackend)
    registry.register("Stage", "generate", GenerateStage)
    registry.register("Stage", "compile_loop", CompileLoopStage)
    registry.register("Executor", "none", NoneExecutor)
    registry.register("Toolchain", "nvcc-sm80", FakeNvcc)
    monkeypatch.setattr(runner_module, "DEFAULT_REGISTRY", registry)
    root = tmp_path / "bench"
    for relative, text in (("src/layout-omp/main.cpp", FAKE_OMP_SOURCE), ("src/layout-cuda/main.cu", FAKE_CUDA_SOURCE)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def run_argv(bench: Path, runs_root: Path) -> list[str]:
    """Return the `lassi run` arguments for the smoke recipe with the fake bench."""
    return ["run", str(SMOKE), "--runs-root", str(runs_root), "--run-id", "p48-followup", "--bench-root", str(bench)]


def write_preset(tmp_path: Path, text: str) -> Path:
    """Write `text` as the preset under the test's LASSI_CONFIG_DIR and return its path."""
    path = tmp_path / "config" / "settings.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("ascii"))
    return path


def names_path(text: str, path: Path) -> bool:
    """Return True when `text` names `path` in native or forward-slash spelling."""
    return str(path) in text or path.as_posix() in text


# ---------------------------------------------------------------------------
# Running the command line


class FakeStream(io.StringIO):
    """A text stream with a chosen isatty()."""

    def __init__(self, tty: bool, text: str = "") -> None:
        """Start with `text` to read, reporting `tty` from isatty()."""
        super().__init__(text)
        self.tty = tty

    def isatty(self) -> bool:
        """Return the chosen answer."""
        return self.tty


@dataclass
class Session:
    """One `lassi` invocation: its exit status (the SystemExit code when argparse exited), stdout, and stderr."""

    code: int | str | None
    out: str
    err: str
    stdin_read: int
    parser_exit: bool


def invoke(argv: Sequence[str], *, stdin_text: str = "", tty: bool = False) -> Session:
    """Run lassi.cli.main(argv) with fake streams (stderr never a terminal); an argparse exit is kept, not failed."""
    stdin, stdout, stderr = FakeStream(tty, stdin_text), FakeStream(tty), FakeStream(False)
    saved = sys.stdin, sys.stdout, sys.stderr
    sys.stdin, sys.stdout, sys.stderr = stdin, stdout, stderr
    parser_exit = False
    try:
        code: int | str | None = cli.main(list(argv))
    except SystemExit as exited:
        code, parser_exit = exited.code, True
    finally:
        sys.stdin, sys.stdout, sys.stderr = saved
    return Session(code=code, out=stdout.getvalue(), err=stderr.getvalue(), stdin_read=stdin.tell(),
                   parser_exit=parser_exit)


# ---------------------------------------------------------------------------
# A bad --graphics value is the parser's usage error


@pytest.mark.parametrize("command", ["run", "score", "settings"])
def test_a_bad_graphics_value_is_the_parsers_usage_error(
    command: str, bench: Path, tmp_path: Path
) -> None:
    argv = {
        "run": run_argv(bench, tmp_path / "runs-root"),
        "score": ["score", str(tmp_path / "absent-run"), "--profile", "df-v0"],
        "settings": ["settings", "graphics"],
    }[command]
    session = invoke(["--graphics", "maybe", *argv], stdin_text="y\ny\n", tty=True)
    assert session.parser_exit and session.code == 2, "argparse refuses the value with its usage error, exit 2"
    assert session.err.startswith("usage: lassi"), session.err
    assert "--graphics" in session.err and "'maybe'" in session.err, session.err
    assert session.out == "" and session.stdin_read == 0
    assert not (tmp_path / "runs-root").exists(), "nothing ran"
    assert not (tmp_path / "config").exists(), "nothing was saved"


# ---------------------------------------------------------------------------
# A YAML merge key is another key, refused


@pytest.mark.parametrize("text", list(MERGE_PRESETS.values()), ids=list(MERGE_PRESETS))
def test_a_yaml_merge_key_in_the_preset_is_refused(text: str, tmp_path: Path) -> None:
    module = settings()
    path = write_preset(tmp_path, text)
    with pytest.raises(module.SettingsError) as refused:
        module.load_preset(path)
    message = str(refused.value)
    assert names_path(message, path), message
    assert "<<" in message, f"the refusal names the merge key: {message}"


@pytest.mark.parametrize("text", list(MERGE_PRESETS.values()), ids=list(MERGE_PRESETS))
def test_lassi_settings_graphics_reports_a_preset_with_a_merge_key(text: str, tmp_path: Path) -> None:
    path = write_preset(tmp_path, text)
    session = invoke(["settings", "graphics"])
    assert session.code == 2
    assert session.err.startswith("lassi settings: ") and names_path(session.err, path), session.err


def test_lassi_run_on_a_terminal_refuses_a_preset_with_a_merge_key_before_it_starts(
    bench: Path, tmp_path: Path
) -> None:
    path = write_preset(tmp_path, MERGE_PRESETS["merge-only"])
    session = invoke(run_argv(bench, tmp_path / "runs-root"), stdin_text="y\ny\n", tty=True)
    assert session.code == 2
    assert session.err.startswith("lassi run: ") and names_path(session.err, path), session.err
    assert session.stdin_read == 0, "a preset that cannot be checked is refused, never asked over"
    assert not (tmp_path / "runs-root").exists(), "nothing ran"
    assert path.read_bytes() == MERGE_PRESETS["merge-only"].encode("ascii"), "the preset is left as it was"


# ---------------------------------------------------------------------------
# The exit-2 cases the CLI docstrings list


def test_a_failed_save_at_the_prompt_exits_2_before_the_command_starts(
    bench: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = settings()
    target = tmp_path / "config" / "settings.yaml"

    def refuse(path: Path, on: bool) -> None:
        raise module.SettingsError(f"cannot save the graphics preset {path}: SYNTHETIC save failure")

    monkeypatch.setattr(module, "save_preset", refuse)
    session = invoke(run_argv(bench, tmp_path / "runs-root"), stdin_text="y\ny\n", tty=True)
    assert session.code == 2
    assert session.stdin_read > 0, "the prompt asked"
    assert session.err.startswith("lassi run: cannot save the graphics preset"), session.err
    assert names_path(session.err, target), session.err
    assert not (tmp_path / "runs-root").exists(), "the command did not start"


def test_on_the_root_filesystem_lassi_run_asks_applies_the_answer_once_and_runs(
    bench: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    module = root_side(monkeypatch, config)
    monkeypatch.setattr(module, "POSIX", True)
    session = invoke(run_argv(bench, tmp_path / "runs-root"), stdin_text="n\n", tty=True)
    assert session.code == 0, session.err
    assert session.stdin_read > 0, "the prompt asked"
    assert "LASSI_CONFIG_DIR" in session.out, "the prompt names LASSI_CONFIG_DIR"
    assert (tmp_path / "runs-root" / "runs" / "p48-followup").is_dir(), "the run ran"
    assert sorted(tmp_path.rglob("settings.yaml")) == [], "nothing is saved on the root filesystem"


def test_on_the_root_filesystem_settings_graphics_shows_the_refusal_and_exits_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    module = root_side(monkeypatch, config)
    monkeypatch.setattr(module, "POSIX", True)
    session = invoke(["settings", "graphics"])
    assert session.code == 0, session.err
    assert session.out.splitlines()[0] == "graphics: off"
    assert "LASSI_CONFIG_DIR" in session.out, session.out
