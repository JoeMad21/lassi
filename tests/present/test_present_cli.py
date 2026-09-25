"""Tests for the graphics setting on the `lassi` command line (task P4.7).

Bible: Readability Standards, Terminal Presentation (the graphics setting,
the banner, and the Rules: with graphics off a command prints what it
prints today; the tools under tools/ keep plain output); Repository Layout
(`present/`, `cli.py`); Agent Rule 7; plans/p4-ttsim.md, the planning
decision "Preset (P4.7)"; AGENTS.md, Operating Mode (unattended sessions
pass --graphics off or LASSI_GRAPHICS=off; without a terminal nothing is
asked).

The contract these tests fix, beside tests/present/test_present_settings.py
and tests/present/test_present_banner.py:

- `lassi [--graphics on|off] <command> ...`: the global option goes before
  the subcommand. Every command decides graphics once, at start, as
  lassi.present.settings.choose_graphics does, with the option,
  os.environ, sys.stdin and sys.stdout (terminals when their isatty() says
  so; the questions are asked through them), and Path.home(). With graphics
  on it first prints lassi.present.banner.BANNER once to sys.stderr, before
  anything else it prints; stderr here is never a terminal, so the banner
  is exact. A SettingsError (such as a bad LASSI_GRAPHICS) exits 2 with
  `lassi <command>: <message>` on stderr before the command starts.
- `lassi settings graphics on|off` saves the preset at
  lassi.present.settings.preset_path(os.environ, home=Path.home()) and
  prints where; with no value, `lassi settings graphics` prints the setting
  that applies on a terminal, never asking, as a first line
  `graphics: on` or `graphics: off` followed by its source (the preset and
  its path, LASSI_GRAPHICS, or --graphics). Exit 0, or 2 with
  `lassi settings: <message>` on stderr (a preset refused on the root
  filesystem names LASSI_CONFIG_DIR; a corrupt preset names its file).
- With graphics off, `lassi run` of the mock smoke recipe and `lassi score`
  print exactly what they printed before P4.7, apart from times and the
  run and score ids; tools/status.py prints no banner, even with
  LASSI_GRAPHICS=on.

The CLI runs with fake stdin, stdout, and stderr whose isatty() the test
chooses; stderr always reports False. Every test sets LASSI_CONFIG_DIR to
a test directory unless it tests the other places, and the device check is
patched so that no test directory counts as the root filesystem (off a
POSIX host it is never called). `lassi run` uses the real mock backend,
stages, and none executor, with a fake toolchain registered as nvcc-sm80
in place of the default registry, as tests/core/test_runner.py does; the
bench sources are small synthetic files. TODAY_TRIAL_LINE is the line
`lassi run` printed for this fixture at commit 2c5f491, before P4.7: output
text, not a measurement. No value here is a measurement.
"""

from __future__ import annotations

import importlib
import io
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
STATUS_TOOL = REPO / "tools" / "status.py"
STATUS_FILE = REPO / "plans" / "STATUS.md"
PROFILE = "df-v0"
# The trial line `lassi run` printed for the smoke recipe with this fake toolchain at commit 2c5f491, wall_s
# normalized to <T>; the run directory line follows it.
SMOKE_TRIAL = "p0-smoke/mock-reference/lassi-hecbench-10/omp-cuda/layout/run01"
TODAY_TRIAL_LINE = f"{SMOKE_TRIAL}  stage S4  corrections 0  wall_s <T>"
ESCAPE = "\x1b"
ROOT_DEVICE = 1
OTHER_DEVICE = 2
ANSWERS = "y\ny\n"

FAKE_OMP_SOURCE = "#include <cstdio>\nint main() {\n#pragma omp target\n  { }\n  return 0;\n}\n"
FAKE_CUDA_SOURCE = "__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n"


# ---------------------------------------------------------------------------
# Modules under test, imported per test so each test fails on its own


def settings() -> ModuleType:
    """Import and return lassi.present.settings."""
    return importlib.import_module("lassi.present.settings")


def banner() -> str:
    """Return lassi.present.banner.BANNER without leading and trailing newlines."""
    return importlib.import_module("lassi.present.banner").BANNER.strip("\n")


# ---------------------------------------------------------------------------
# Environment and fixtures


@pytest.fixture(autouse=True)
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the variables a test could inherit, put the preset under <tmp>/config, and TMPDIR under <tmp>."""
    for name in ("LASSI_GRAPHICS", "LASSI_CONFIG_DIR", "LASSI_SCRATCH", "LASSI_RUNS_ROOT", "LASSI_TOOLCHAINS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LASSI_CONFIG_DIR", str(tmp_path / "config"))
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))


def off_root(monkeypatch: pytest.MonkeyPatch, *root_side: Path) -> ModuleType:
    """Patch the settings module's device check: "/" and `root_side` on the root filesystem, the rest off it."""
    module = settings()
    sides = [Path(side).resolve() for side in root_side]

    def device(path: Path) -> int:
        """Return the fake device number of `path`."""
        path = Path(path)
        if path.parent == path:
            return ROOT_DEVICE
        path = path.resolve()
        return ROOT_DEVICE if any(path == side or side in path.parents for side in sides) else OTHER_DEVICE

    monkeypatch.setattr(module, "path_device", device)
    return module


def config_preset(tmp_path: Path) -> Path:
    """Return the preset file under the test's LASSI_CONFIG_DIR."""
    return tmp_path / "config" / "settings.yaml"


def write_preset(tmp_path: Path, value: str) -> Path:
    """Write the hand-written preset `graphics: <value>` under the test's LASSI_CONFIG_DIR; return its path."""
    path = config_preset(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"graphics: {value}\n".encode("ascii"))
    return path


def all_presets(root: Path) -> list[Path]:
    """Return every settings.yaml under `root`."""
    return sorted(root.rglob("settings.yaml"))


def names_path(text: str, path: Path) -> bool:
    """Return True when `text` names `path` in native or forward-slash spelling."""
    return str(path) in text or path.as_posix() in text


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


def run_argv(bench: Path, runs_root: Path, run_id: str) -> list[str]:
    """Return the `lassi run` arguments for the smoke recipe with the fake bench."""
    return ["run", str(SMOKE), "--runs-root", str(runs_root), "--run-id", run_id, "--bench-root", str(bench)]


def score_argv(run_dir: Path, score_id: str | None = None) -> list[str]:
    """Return the `lassi score` arguments for `run_dir` with the df-v0 profile."""
    return ["score", str(run_dir), "--profile", PROFILE, *(["--score-id", score_id] if score_id else [])]


def today_run_stdout(run_dir: Path) -> str:
    """Return what `lassi run` of the smoke recipe printed before P4.7, wall_s normalized."""
    return f"{TODAY_TRIAL_LINE}\nrun directory: {run_dir}\n"


def normalized(text: str) -> str:
    """Return `lassi run` output with every wall_s value written <T>."""
    return re.sub(r"wall_s \S+", "wall_s <T>", text)


# ---------------------------------------------------------------------------
# Running the command line with fake streams


class FakeStream(io.StringIO):
    """A text stream with a chosen isatty() that logs each write, with its stream's label, in a shared list."""

    def __init__(self, label: str, writes: list[tuple[str, str]], tty: bool, text: str = "") -> None:
        """Start with `text` to read, reporting `tty` from isatty()."""
        super().__init__(text)
        self.label = label
        self.writes = writes
        self.tty = tty

    def isatty(self) -> bool:
        """Return the chosen answer."""
        return self.tty

    def write(self, text: str) -> int:
        """Log the write and keep the text."""
        self.writes.append((self.label, text))
        return super().write(text)


@dataclass
class Session:
    """One `lassi` invocation: its exit status, stdout, stderr, how much of stdin it read, and its writes in order."""

    code: int
    out: str
    err: str
    stdin_read: int
    writes: list[tuple[str, str]] = field(default_factory=list)

    def leading_stderr(self) -> str:
        """Return what went to stderr before anything was written to stdout."""
        parts: list[str] = []
        for label, text in self.writes:
            if label == "stderr":
                parts.append(text)
            elif text:
                break
        return "".join(parts)


def invoke(argv: Sequence[str], *, stdin_text: str = "", stdin_tty: bool = False, stdout_tty: bool = False) -> Session:
    """Run lassi.cli.main(argv) with fake streams and return the session; fail the test on an argparse exit."""
    writes: list[tuple[str, str]] = []
    stdin = FakeStream("stdin", writes, stdin_tty, stdin_text)
    stdout = FakeStream("stdout", writes, stdout_tty)
    stderr = FakeStream("stderr", writes, False)
    saved = sys.stdin, sys.stdout, sys.stderr
    sys.stdin, sys.stdout, sys.stderr = stdin, stdout, stderr
    exited: SystemExit | None = None
    code = -1
    try:
        code = cli.main(list(argv))
    except SystemExit as exit_:
        exited = exit_
    finally:
        sys.stdin, sys.stdout, sys.stderr = saved
    if exited is not None:
        pytest.fail(f"argparse exited ({exited.code}) on {list(argv)}: {stderr.getvalue().strip()}")
    return Session(code=code, out=stdout.getvalue(), err=stderr.getvalue(), stdin_read=stdin.tell(), writes=writes)


def assert_no_banner(session: Session) -> None:
    """Fail when the session printed the banner or an escape sequence anywhere."""
    text = banner()
    assert text not in session.out and text not in session.err, "the banner was printed with graphics off"
    assert ESCAPE not in session.out + session.err


def assert_banner_first_once(session: Session) -> None:
    """Fail unless stderr starts with the banner, before any stdout, and holds it exactly once."""
    text = banner()
    leading = session.leading_stderr().lstrip("\n")
    assert leading.startswith(text), f"stderr does not start with the banner: {session.err!r}"
    assert session.err.count(text) == 1, "the banner is printed once"
    assert text not in session.out, "the banner goes to stderr"
    assert ESCAPE not in session.err, "stderr is not a terminal here, so no escape sequence"


# ---------------------------------------------------------------------------
# Graphics off: every command prints what it printed before P4.7


@dataclass(frozen=True)
class Off:
    """One way graphics end up off: global options, environment, saved preset, and which streams are terminals."""

    prefix: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    preset: str | None = None
    tty: bool = False


OFF_MODES = {
    "no-terminal-preset-on": Off(preset="on"),
    "option-off-no-terminal": Off(prefix=("--graphics", "off")),
    "option-off-terminal-preset-on": Off(prefix=("--graphics", "off"), preset="on", tty=True),
    "variable-off-terminal-preset-on": Off(env=(("LASSI_GRAPHICS", "off"),), preset="on", tty=True),
    "preset-off-terminal": Off(preset="off", tty=True),
}


def apply_off(mode: Off, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> bytes | None:
    """Set up `mode`'s environment and preset; return the preset file's bytes, or None when none is saved."""
    for name, value in mode.env:
        monkeypatch.setenv(name, value)
    return None if mode.preset is None else write_preset(tmp_path, mode.preset).read_bytes()


def invoke_off(mode: Off, argv: Sequence[str]) -> Session:
    """Run `argv` after `mode`'s global options, with ANSWERS on stdin and terminals as `mode` says."""
    return invoke([*mode.prefix, *argv], stdin_text=ANSWERS, stdin_tty=mode.tty, stdout_tty=mode.tty)


def assert_nothing_asked_or_saved(session: Session, tmp_path: Path, preset: bytes | None) -> None:
    """Fail when the session read stdin, printed the banner, or changed or added a preset."""
    assert session.stdin_read == 0, "nothing is asked"
    assert_no_banner(session)
    if preset is None:
        assert all_presets(tmp_path) == []
    else:
        assert all_presets(tmp_path) == [config_preset(tmp_path)]
        assert config_preset(tmp_path).read_bytes() == preset


@pytest.mark.parametrize("mode", list(OFF_MODES.values()), ids=list(OFF_MODES))
def test_lassi_run_with_graphics_off_prints_what_it_printed_before(
    mode: Off, bench: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    off_root(monkeypatch)
    runs_root = tmp_path / "runs-root"
    plain = invoke(run_argv(bench, runs_root, "plain"))
    assert plain.code == 0, plain.err
    assert normalized(plain.out) == today_run_stdout(runs_root / "runs" / "plain")
    assert plain.err == ""
    preset = apply_off(mode, tmp_path, monkeypatch)
    session = invoke_off(mode, run_argv(bench, runs_root, "off"))
    assert session.code == 0, session.err
    assert normalized(session.out) == today_run_stdout(runs_root / "runs" / "off")
    assert session.err == ""
    assert_nothing_asked_or_saved(session, tmp_path, preset)


SCORE_OFF_MODES = {name: OFF_MODES[name] for name in (
    "no-terminal-preset-on", "option-off-terminal-preset-on", "variable-off-terminal-preset-on"
)}


@pytest.mark.parametrize("mode", list(SCORE_OFF_MODES.values()), ids=list(SCORE_OFF_MODES))
def test_lassi_score_with_graphics_off_prints_what_it_printed_before(
    mode: Off, bench: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    off_root(monkeypatch)
    runs_root = tmp_path / "runs-root"
    run_dir = runs_root / "runs" / "plain"
    assert invoke(run_argv(bench, runs_root, "plain")).code == 0
    plain = invoke(score_argv(run_dir, "plain"))
    assert (plain.code, plain.out, plain.err) == (0, f"{runs_root / 'scores' / 'plain'}\n", "")
    preset = apply_off(mode, tmp_path, monkeypatch)
    session = invoke_off(mode, score_argv(run_dir, "off"))
    assert (session.code, session.out, session.err) == (0, f"{runs_root / 'scores' / 'off'}\n", "")
    assert_nothing_asked_or_saved(session, tmp_path, preset)


@pytest.mark.parametrize("mode", list(SCORE_OFF_MODES.values()), ids=list(SCORE_OFF_MODES))
def test_a_refused_lassi_score_with_graphics_off_prints_what_it_printed_before(
    mode: Off, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    off_root(monkeypatch)
    argv = score_argv(tmp_path / "absent-run")
    plain = invoke(argv)
    assert plain.code == 2 and plain.out == "" and plain.err.startswith("lassi score: "), plain.err
    preset = apply_off(mode, tmp_path, monkeypatch)
    session = invoke_off(mode, argv)
    assert (session.code, session.out, session.err) == (plain.code, plain.out, plain.err)
    assert_nothing_asked_or_saved(session, tmp_path, preset)


@pytest.mark.parametrize(("stdin_tty", "stdout_tty"), [(False, False), (True, False), (False, True)],
                         ids=["pipes", "stdin-terminal-only", "stdout-terminal-only"])
def test_without_an_interactive_terminal_nothing_is_asked_and_no_banner_prints(
    stdin_tty: bool, stdout_tty: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    off_root(monkeypatch)
    argv = score_argv(tmp_path / "absent-run")
    plain = invoke(argv)
    session = invoke(argv, stdin_text=ANSWERS, stdin_tty=stdin_tty, stdout_tty=stdout_tty)
    assert (session.code, session.out, session.err) == (plain.code, plain.out, plain.err)
    assert_nothing_asked_or_saved(session, tmp_path, None)


def test_a_bad_lassi_graphics_value_is_refused_before_the_run_starts(
    bench: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LASSI_GRAPHICS", "maybe")
    runs_root = tmp_path / "runs-root"
    session = invoke(run_argv(bench, runs_root, "refused"))
    assert session.code == 2
    assert session.out == ""
    assert session.err.startswith("lassi") and "LASSI_GRAPHICS" in session.err and "maybe" in session.err, session.err
    assert not (runs_root / "runs").exists(), "nothing ran"


# ---------------------------------------------------------------------------
# Graphics on: the banner, once, first, on stderr


@pytest.mark.parametrize("command", ["run", "score", "settings"])
def test_graphics_on_prints_the_banner_once_to_stderr_before_anything_else(
    command: str, bench: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    off_root(monkeypatch)
    argv = {
        "run": run_argv(bench, tmp_path / "runs-root", "on"),
        "score": score_argv(tmp_path / "absent-run"),
        "settings": ["settings", "graphics"],
    }[command]
    session = invoke(["--graphics", "on", *argv])
    assert session.code == (2 if command == "score" else 0), session.err
    assert_banner_first_once(session)
    if command == "score":
        assert "lassi score: " in session.err.split(banner(), 1)[1], "the refusal follows the banner"


def test_lassi_graphics_on_prints_the_banner_without_the_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    off_root(monkeypatch)
    monkeypatch.setenv("LASSI_GRAPHICS", "on")
    session = invoke(score_argv(tmp_path / "absent-run"))
    assert session.code == 2
    assert_banner_first_once(session)
    assert session.stdin_read == 0


def test_a_saved_preset_of_on_prints_the_banner_on_a_terminal_without_asking(
    bench: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    off_root(monkeypatch)
    write_preset(tmp_path, "on")
    session = invoke(run_argv(bench, tmp_path / "runs-root", "preset"), stdin_text=ANSWERS, stdin_tty=True,
                     stdout_tty=True)
    assert session.code == 0, session.err
    assert session.stdin_read == 0, "a saved preset is never asked again"
    assert_banner_first_once(session)


def test_the_first_command_on_a_terminal_asks_saves_the_preset_and_prints_the_banner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = off_root(monkeypatch)
    session = invoke(score_argv(tmp_path / "absent-run"), stdin_text=ANSWERS, stdin_tty=True, stdout_tty=True)
    assert session.code == 2
    assert session.stdin_read > 0, "the questions were asked"
    assert "graphics" in (session.out + session.err).lower()
    assert module.load_preset(config_preset(tmp_path)) is True
    text = banner()
    assert session.err.count(text) == 1
    assert session.err.index(text) < session.err.index("lassi score: ")


# ---------------------------------------------------------------------------
# lassi settings graphics


@pytest.mark.parametrize("value", ["on", "off"])
def test_settings_graphics_saves_the_preset_prints_where_and_shows_it(
    value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = off_root(monkeypatch)
    saved = invoke(["settings", "graphics", value])
    path = config_preset(tmp_path)
    assert (saved.code, saved.err) == (0, ""), saved.err
    assert names_path(saved.out, path), saved.out
    assert module.load_preset(path) is (value == "on")
    shown = invoke(["settings", "graphics"])
    assert shown.code == 0, shown.err
    assert re.match(rf"graphics: {value}\b", shown.out.splitlines()[0]), shown.out
    assert "preset" in shown.out and names_path(shown.out, path), shown.out


@pytest.mark.parametrize("place", ["scratch", "home"])
def test_settings_graphics_saves_under_lassi_scratch_then_the_home_config_directory(
    place: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = off_root(monkeypatch)
    monkeypatch.delenv("LASSI_CONFIG_DIR")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    if place == "scratch":
        monkeypatch.setenv("LASSI_SCRATCH", str(tmp_path / "scratch"))
        path = tmp_path / "scratch" / "config" / "lassi" / "settings.yaml"
    else:
        path = home / ".config" / "lassi" / "settings.yaml"
    session = invoke(["settings", "graphics", "on"])
    assert session.code == 0, session.err
    assert names_path(session.out, path), session.out
    assert all_presets(tmp_path) == [path]
    assert module.load_preset(path) is True


SHOW_CASES = {
    "nothing-saved": ((), (), None, "off", ()),
    "preset-on": ((), (), "on", "on", ("preset",)),
    "variable-off-over-preset-on": ((), (("LASSI_GRAPHICS", "off"),), "on", "off", ("LASSI_GRAPHICS",)),
    "option-on-over-preset-off": (("--graphics", "on"), (), "off", "on", ("--graphics",)),
}


@pytest.mark.parametrize(("prefix", "env", "preset", "shown", "words"), list(SHOW_CASES.values()), ids=list(SHOW_CASES))
def test_settings_graphics_with_no_value_shows_the_setting_and_its_source(
    prefix: tuple[str, ...],
    env: tuple[tuple[str, str], ...],
    preset: str | None,
    shown: str,
    words: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    off_root(monkeypatch)
    for name, value in env:
        monkeypatch.setenv(name, value)
    if preset is not None:
        write_preset(tmp_path, preset)
    session = invoke([*prefix, "settings", "graphics"])
    assert session.code == 0, session.err
    assert re.match(rf"graphics: {shown}\b", session.out.splitlines()[0]), session.out
    missing = [word for word in words if word not in session.out]
    assert not missing, f"the source is not named ({missing}): {session.out!r}"
    if preset is not None and "preset" in words:
        assert names_path(session.out, config_preset(tmp_path)), session.out


def test_settings_graphics_refuses_a_preset_on_the_root_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config"
    config.mkdir()
    module = off_root(monkeypatch, config)
    monkeypatch.setattr(module, "POSIX", True)
    session = invoke(["settings", "graphics", "on"])
    assert session.code == 2
    assert session.err.startswith("lassi settings: ") and "LASSI_CONFIG_DIR" in session.err, session.err
    assert all_presets(tmp_path) == [], "nothing is saved on the root filesystem"


@pytest.mark.parametrize("tty", [False, True], ids=["no-terminal", "terminal"])
def test_settings_graphics_reports_a_corrupt_preset(
    tty: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    off_root(monkeypatch)
    path = write_preset(tmp_path, "maybe")
    session = invoke(["settings", "graphics"], stdin_text=ANSWERS, stdin_tty=tty, stdout_tty=tty)
    assert session.code == 2
    assert session.err.startswith("lassi settings: ") and names_path(session.err, path), session.err
    assert session.stdin_read == 0, "the settings command never asks"


@pytest.mark.parametrize("corrupt", ["maybe", "true", "yes"])
@pytest.mark.parametrize("value", ["on", "off"])
def test_settings_graphics_on_a_terminal_replaces_a_corrupt_preset(
    value: str, corrupt: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = off_root(monkeypatch)
    path = write_preset(tmp_path, corrupt)
    session = invoke(["settings", "graphics", value], stdin_text=ANSWERS, stdin_tty=True, stdout_tty=True)
    assert (session.code, session.err) == (0, ""), session.err
    assert names_path(session.out, path), session.out
    assert session.stdin_read == 0, "the settings command never asks"
    assert module.load_preset(path) is (value == "on")


def test_settings_graphics_on_still_refuses_a_bad_lassi_graphics_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    off_root(monkeypatch)
    monkeypatch.setenv("LASSI_GRAPHICS", "maybe")
    path = write_preset(tmp_path, "maybe")
    session = invoke(["settings", "graphics", "on"], stdin_text=ANSWERS, stdin_tty=True, stdout_tty=True)
    assert session.code == 2
    assert session.err.startswith("lassi settings: ") and "LASSI_GRAPHICS" in session.err, session.err
    assert path.read_bytes() == b"graphics: maybe\n", "nothing is saved"


# ---------------------------------------------------------------------------
# The repository tools keep plain output


def test_tools_status_summary_prints_no_banner_even_with_graphics_on(tmp_path: Path) -> None:
    text = banner()
    status_copy = tmp_path / "STATUS.md"
    status_copy.write_bytes(STATUS_FILE.read_bytes())
    command = [sys.executable, str(STATUS_TOOL), "--file", str(status_copy), "summary"]
    plain_env = {name: value for name, value in os.environ.items() if name != "LASSI_GRAPHICS"}
    plain = subprocess.run(command, cwd=REPO, env=plain_env, capture_output=True, text=True, timeout=120)
    write_preset(tmp_path, "on")
    on = subprocess.run(
        command, cwd=REPO, env={**plain_env, "LASSI_GRAPHICS": "on"}, capture_output=True, text=True, timeout=120
    )
    assert plain.returncode == 0 and on.returncode == 0, on.stderr
    assert (on.stdout, on.stderr) == (plain.stdout, plain.stderr)
    assert text not in on.stdout + on.stderr
