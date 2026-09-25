"""Tests for the graphics setting and its saved preset (task P4.7).

Bible: Readability Standards, Terminal Presentation (the graphics setting,
its order, and the preset's place); Agent Rule 7; plans/p4-ttsim.md, the
planning decision "Preset (P4.7)"; AGENTS.md, Operating Mode (unattended
sessions pass --graphics off or LASSI_GRAPHICS=off).

The contract these tests fix, in lassi/present/settings.py:

- SettingsError: the clear error for a bad LASSI_GRAPHICS value, a bad
  option value, or a corrupt preset file; its message names the variable or
  the file.
- resolve_graphics(*, option, env, stdin_isatty, stdout_isatty, preset,
  ask=None) -> (on, source). The order is the --graphics option ("on" or
  "off"), then env["LASSI_GRAPHICS"] ("on" or "off"; anything else is a
  SettingsError), then, only when stdin and stdout are both terminals, the
  saved preset (True, False, or None for none), then ask() (the prompt),
  then off. Sources: "--graphics", "LASSI_GRAPHICS", "preset", "prompt",
  "no terminal" (off because stdin or stdout is not a terminal and neither
  the option nor the variable decided; the preset is not read and ask is
  never called), and "default" (a terminal, no preset, and no ask).
- preset_path(env, *, home, root_device=None, posix=None) -> Path | None:
  $LASSI_CONFIG_DIR/settings.yaml, else
  $LASSI_SCRATCH/config/lassi/settings.yaml, else
  <home>/.config/lassi/settings.yaml. On a POSIX host (posix None means the
  module's POSIX, os.name == "posix") a preset whose directory would lie on
  the filesystem that holds / is refused: None. The check compares
  root_device(<the settings directory or its nearest existing parent>) with
  root_device(Path("/")); root_device None means the module's path_device,
  os.stat(path).st_dev. Both module values are read at call time, so a test
  can patch them. Off POSIX no root-filesystem refusal applies.
- load_preset(path) -> bool | None: None when the file does not exist; the
  value of the YAML file `graphics: on|off`, a mapping with exactly one key,
  graphics, whose value is on or off, plain or quoted; SettingsError, naming
  the file, for anything else, including another key and a repeated key
  (never a silent default or a last-wins).
- save_preset(path, on) -> None writes `graphics: on` or `graphics: off`,
  creating the directory.
- choose_graphics(*, option, env, stdin, out, stdin_isatty, stdout_isatty,
  home, root_device=None, posix=None) -> (on, source): one invocation's
  decision. It finds the preset with preset_path, reads it with load_preset,
  and resolves as above; the prompt writes its question to `out` and reads
  a line from `stdin` (y or yes means yes; Enter, end of input, n, or no
  means no), then asks a second question, whether to save the answer as the
  preset; a yes saves it with save_preset and writes where to `out`. When
  preset_path refuses the root filesystem, nothing is saved, the answer
  applies to this invocation only, and `out` names LASSI_CONFIG_DIR.
  Nothing is read from `stdin` or written to `out` unless the prompt asks.

The fake device check puts "/" (any path that is its own parent) and every
path under the directories a test names on device 1, and every other path
on device 2, so the tests run the POSIX rule on any host. No value here is
a measurement.
"""

from __future__ import annotations

import importlib
import io
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT_DEVICE = 1
OTHER_DEVICE = 2
SOURCES = ("--graphics", "LASSI_GRAPHICS", "preset", "prompt", "no terminal", "default")


def settings() -> ModuleType:
    """Import and return lassi.present.settings."""
    return importlib.import_module("lassi.present.settings")


def devices(*root_side: Path) -> Callable[[Path], int]:
    """Return a device check: "/" and every path under `root_side` on ROOT_DEVICE, the rest on OTHER_DEVICE."""
    sides = [Path(side).resolve() for side in root_side]

    def device(path: Path) -> int:
        """Return the fake device number of `path`."""
        path = Path(path)
        if path.parent == path:
            return ROOT_DEVICE
        path = path.resolve()
        return ROOT_DEVICE if any(path == side or side in path.parents for side in sides) else OTHER_DEVICE

    return device


def never_asked() -> bool:
    """Fail the test: the prompt must not be asked here."""
    raise AssertionError("the prompt was asked")


class Asker:
    """A prompt stand-in that answers `answer` and counts its calls."""

    def __init__(self, answer: bool) -> None:
        """Keep the answer to give."""
        self.answer = answer
        self.calls = 0

    def __call__(self) -> bool:
        """Count the call and give the answer."""
        self.calls += 1
        return self.answer


def resolve(
    *,
    option: str | None = None,
    env: Mapping[str, str] | None = None,
    tty: tuple[bool, bool] = (True, True),
    preset: bool | None = None,
    ask: Callable[[], bool] | None = never_asked,
) -> tuple[bool, str]:
    """Call resolve_graphics with keyword arguments, a terminal on both streams by default."""
    return settings().resolve_graphics(
        option=option, env=dict(env or {}), stdin_isatty=tty[0], stdout_isatty=tty[1], preset=preset, ask=ask
    )


def preset_lines(path: Path) -> list[str]:
    """Return the preset file's lines without comments and blank lines."""
    lines = (line.split("#", 1)[0].rstrip() for line in path.read_text(encoding="ascii").splitlines())
    return [line for line in lines if line.strip()]


def names_path(text: str, path: Path) -> bool:
    """Return True when `text` names `path` in native or forward-slash spelling."""
    return str(path) in text or path.as_posix() in text


def all_presets(root: Path) -> list[Path]:
    """Return every settings.yaml under `root`."""
    return sorted(root.rglob("settings.yaml"))


# ---------------------------------------------------------------------------
# The order: option, LASSI_GRAPHICS, preset, prompt, off


@pytest.mark.parametrize(("option", "env_value", "preset"), [("on", "off", False), ("off", "on", True)])
def test_the_option_decides_before_everything_else(option: str, env_value: str, preset: bool) -> None:
    assert resolve(option=option, env={"LASSI_GRAPHICS": env_value}, preset=preset) == (option == "on", "--graphics")


@pytest.mark.parametrize(("env_value", "preset"), [("on", False), ("off", True)])
def test_lassi_graphics_decides_when_there_is_no_option(env_value: str, preset: bool) -> None:
    assert resolve(env={"LASSI_GRAPHICS": env_value}, preset=preset) == (env_value == "on", "LASSI_GRAPHICS")


def test_lassi_graphics_off_wins_over_a_saved_preset_of_on() -> None:
    assert resolve(env={"LASSI_GRAPHICS": "off"}, preset=True) == (False, "LASSI_GRAPHICS")


@pytest.mark.parametrize("value", ["yes", "1", "true", "maybe"])
def test_any_other_lassi_graphics_value_is_a_clear_error(value: str) -> None:
    module = settings()
    with pytest.raises(module.SettingsError) as refused:
        resolve(env={"LASSI_GRAPHICS": value})
    message = str(refused.value)
    assert "LASSI_GRAPHICS" in message and value in message, message


def test_an_option_value_other_than_on_or_off_is_a_clear_error() -> None:
    module = settings()
    with pytest.raises(module.SettingsError, match="maybe"):
        resolve(option="maybe")


@pytest.mark.parametrize("preset", [True, False])
def test_the_saved_preset_decides_on_a_terminal_when_nothing_else_does(preset: bool) -> None:
    assert resolve(preset=preset) == (preset, "preset")


@pytest.mark.parametrize("answer", [True, False])
def test_the_prompt_decides_on_a_terminal_with_no_preset(answer: bool) -> None:
    asker = Asker(answer)
    assert resolve(ask=asker) == (answer, "prompt")
    assert asker.calls == 1


def test_on_a_terminal_with_no_preset_and_no_prompt_graphics_are_off() -> None:
    assert resolve(ask=None) == (False, "default")


@pytest.mark.parametrize("tty", [(False, False), (True, False), (False, True)], ids=["pipes", "stdout", "stdin"])
@pytest.mark.parametrize("preset", [None, True])
def test_without_a_terminal_nothing_is_asked_and_graphics_are_off(tty: tuple[bool, bool], preset: bool | None) -> None:
    assert resolve(tty=tty, preset=preset) == (False, "no terminal")


@pytest.mark.parametrize(
    ("option", "env", "expected"),
    [("on", {}, (True, "--graphics")), (None, {"LASSI_GRAPHICS": "on"}, (True, "LASSI_GRAPHICS")),
     (None, {"LASSI_GRAPHICS": "off"}, (False, "LASSI_GRAPHICS"))],
    ids=["option-on", "variable-on", "variable-off"],
)
def test_without_a_terminal_only_the_option_or_the_variable_decides(
    option: str | None, env: dict[str, str], expected: tuple[bool, str]
) -> None:
    assert resolve(option=option, env=env, tty=(False, False), preset=True) == expected


def test_every_source_is_one_of_the_named_sources() -> None:
    results = [
        resolve(option="on"),
        resolve(env={"LASSI_GRAPHICS": "on"}),
        resolve(preset=True),
        resolve(ask=Asker(True)),
        resolve(tty=(False, False)),
        resolve(ask=None),
    ]
    assert [source for _, source in results] == list(SOURCES)


# ---------------------------------------------------------------------------
# Where the preset lives


def test_the_preset_is_in_lassi_config_dir_first(tmp_path: Path) -> None:
    env = {"LASSI_CONFIG_DIR": str(tmp_path / "config"), "LASSI_SCRATCH": str(tmp_path / "scratch")}
    found = settings().preset_path(env, home=tmp_path / "home", root_device=devices(), posix=True)
    assert found == tmp_path / "config" / "settings.yaml"


def test_the_preset_is_under_lassi_scratch_without_lassi_config_dir(tmp_path: Path) -> None:
    env = {"LASSI_SCRATCH": str(tmp_path / "scratch")}
    found = settings().preset_path(env, home=tmp_path / "home", root_device=devices(), posix=True)
    assert found == tmp_path / "scratch" / "config" / "lassi" / "settings.yaml"


def test_the_preset_is_under_the_home_config_directory_otherwise(tmp_path: Path) -> None:
    found = settings().preset_path({}, home=tmp_path / "home", root_device=devices(), posix=True)
    assert found == tmp_path / "home" / ".config" / "lassi" / "settings.yaml"


def test_a_posix_home_on_the_root_filesystem_gives_no_preset_path(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    assert settings().preset_path({}, home=home, root_device=devices(home), posix=True) is None


def test_a_posix_lassi_config_dir_on_the_root_filesystem_gives_no_preset_path(tmp_path: Path) -> None:
    config = tmp_path / "config"
    config.mkdir()
    env = {"LASSI_CONFIG_DIR": str(config)}
    assert settings().preset_path(env, home=tmp_path / "home", root_device=devices(config), posix=True) is None


def test_a_posix_scratch_off_the_root_filesystem_is_used_when_home_is_on_it(tmp_path: Path) -> None:
    home, scratch = tmp_path / "home", tmp_path / "scratch"
    home.mkdir()
    scratch.mkdir()
    found = settings().preset_path({"LASSI_SCRATCH": str(scratch)}, home=home, root_device=devices(home), posix=True)
    assert found == scratch / "config" / "lassi" / "settings.yaml"


def test_off_posix_no_root_filesystem_refusal_applies(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    found = settings().preset_path({}, home=home, root_device=devices(tmp_path), posix=False)
    assert found == home / ".config" / "lassi" / "settings.yaml"


def test_the_default_checks_are_the_host_platform_and_st_dev(tmp_path: Path) -> None:
    module = settings()
    assert module.POSIX == (os.name == "posix")
    assert module.path_device(tmp_path) == os.stat(tmp_path).st_dev


def test_preset_path_reads_the_module_defaults_at_call_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = settings()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(module, "POSIX", True)
    monkeypatch.setattr(module, "path_device", devices(home))
    assert module.preset_path({}, home=home) is None
    monkeypatch.setattr(module, "POSIX", False)
    assert module.preset_path({}, home=home) == home / ".config" / "lassi" / "settings.yaml"


@pytest.mark.skipif(os.name != "posix", reason="the real device check runs on a POSIX host only")
def test_on_a_posix_host_the_real_check_refuses_the_root_directory(tmp_path: Path) -> None:
    assert settings().preset_path({"LASSI_CONFIG_DIR": "/"}, home=tmp_path, posix=True) is None


# ---------------------------------------------------------------------------
# Reading and writing the preset


@pytest.mark.parametrize("on", [True, False])
def test_a_saved_preset_reads_back_and_is_the_yaml_graphics_line(tmp_path: Path, on: bool) -> None:
    module = settings()
    path = tmp_path / "a" / "b" / "settings.yaml"
    module.save_preset(path, on)
    assert module.load_preset(path) is on
    assert preset_lines(path) == [f"graphics: {'on' if on else 'off'}"]
    assert yaml.safe_load(path.read_text(encoding="ascii")) == {"graphics": on}


def test_saving_again_replaces_the_preset(tmp_path: Path) -> None:
    module = settings()
    path = tmp_path / "settings.yaml"
    module.save_preset(path, True)
    module.save_preset(path, False)
    assert module.load_preset(path) is False


def test_a_missing_preset_file_is_no_preset(tmp_path: Path) -> None:
    assert settings().load_preset(tmp_path / "absent" / "settings.yaml") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [("graphics: on\n", True), ("graphics: off\n", False), ("# my choice\ngraphics: on  # kept\n", True)],
    ids=["on", "off", "commented"],
)
def test_a_hand_written_preset_reads(tmp_path: Path, text: str, expected: bool) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(text, encoding="ascii")
    assert settings().load_preset(path) is expected


@pytest.mark.parametrize(
    "text",
    ["graphics: [\n", "graphics: maybe\n", "graphics: 1\n", "graphics:\n", "- on\n", "just text\n", ""],
    ids=["not-yaml", "maybe", "number", "null", "list", "scalar", "empty"],
)
def test_a_corrupt_preset_is_a_clear_error_never_a_default(tmp_path: Path, text: str) -> None:
    module = settings()
    path = tmp_path / "settings.yaml"
    path.write_text(text, encoding="ascii")
    with pytest.raises(module.SettingsError) as refused:
        module.load_preset(path)
    assert names_path(str(refused.value), path), str(refused.value)


@pytest.mark.parametrize(
    ("text", "expected"),
    [('graphics: "on"\n', True), ("graphics: 'on'\n", True), ('graphics: "off"\n', False)],
    ids=["double-quoted-on", "single-quoted-on", "double-quoted-off"],
)
def test_a_quoted_on_or_off_reads(tmp_path: Path, text: str, expected: bool) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(text, encoding="ascii")
    assert settings().load_preset(path) is expected


@pytest.mark.parametrize(
    "text",
    ["graphics: on\ntheme: dark\n", "theme: dark\ngraphics: off\n", "graphics: on\nGraphics: on\n"],
    ids=["after", "before", "other-case"],
)
def test_a_preset_with_an_extra_key_is_refused(tmp_path: Path, text: str) -> None:
    module = settings()
    path = tmp_path / "settings.yaml"
    path.write_text(text, encoding="ascii")
    with pytest.raises(module.SettingsError) as refused:
        module.load_preset(path)
    message = str(refused.value)
    assert names_path(message, path), message
    assert "exactly one key" in message, message


@pytest.mark.parametrize(
    "text", ["graphics: off\ngraphics: on\n", "graphics: on\ngraphics: on\n"], ids=["off-then-on", "same-value"]
)
def test_a_repeated_graphics_key_is_refused_never_last_wins(tmp_path: Path, text: str) -> None:
    module = settings()
    path = tmp_path / "settings.yaml"
    path.write_text(text, encoding="ascii")
    with pytest.raises(module.SettingsError) as refused:
        module.load_preset(path)
    message = str(refused.value)
    assert names_path(message, path), message
    assert "repeated key 'graphics'" in message, message


# ---------------------------------------------------------------------------
# One invocation: the prompt, the second question, and the refusal


def choose(
    tmp_path: Path,
    answers: str,
    *,
    env: Mapping[str, str] | None = None,
    tty: tuple[bool, bool] = (True, True),
    option: str | None = None,
    root_device: Callable[[Path], int] | None = None,
    posix: bool = False,
) -> tuple[tuple[bool, str], io.StringIO, io.StringIO]:
    """Run choose_graphics with `answers` on stdin; return its result, stdin, and out.

    The environment defaults to LASSI_CONFIG_DIR=<tmp>/config, and the home
    to <tmp>/home.
    """
    stdin, out = io.StringIO(answers), io.StringIO()
    chosen = settings().choose_graphics(
        option=option,
        env=dict(env if env is not None else {"LASSI_CONFIG_DIR": str(tmp_path / "config")}),
        stdin=stdin,
        out=out,
        stdin_isatty=tty[0],
        stdout_isatty=tty[1],
        home=tmp_path / "home",
        root_device=root_device or devices(),
        posix=posix,
    )
    return chosen, stdin, out


def test_the_first_run_on_a_terminal_asks_then_offers_to_save_the_preset(tmp_path: Path) -> None:
    chosen, _, out = choose(tmp_path, "y\ny\n")
    path = tmp_path / "config" / "settings.yaml"
    assert chosen == (True, "prompt")
    assert settings().load_preset(path) is True
    text = out.getvalue()
    assert "graphics" in text.lower() and "preset" in text.lower(), text
    assert names_path(text, path), f"the saved preset's place is not shown: {text!r}"


@pytest.mark.parametrize(
    ("first", "on"), [("y", True), ("yes", True), ("n", False), ("no", False), ("", False)],
    ids=["y", "yes", "n", "no", "enter"],
)
def test_the_first_answer_decides_and_enter_means_no(tmp_path: Path, first: str, on: bool) -> None:
    chosen, _, _ = choose(tmp_path, f"{first}\n\n")
    assert chosen == (on, "prompt")
    assert all_presets(tmp_path) == [], "Enter at the second question saves nothing"


def test_the_end_of_input_at_the_prompt_means_no(tmp_path: Path) -> None:
    chosen, _, _ = choose(tmp_path, "")
    assert chosen == (False, "prompt")
    assert all_presets(tmp_path) == []


def test_an_answer_not_saved_applies_once_and_the_next_run_asks_again(tmp_path: Path) -> None:
    first, _, _ = choose(tmp_path, "y\n\n")
    second, stdin, out = choose(tmp_path, "n\n\n")
    assert (first, second) == ((True, "prompt"), (False, "prompt"))
    assert stdin.tell() > 0 and out.getvalue(), "the second run asks again"


def test_a_saved_answer_is_the_preset_and_nothing_is_asked_again(tmp_path: Path) -> None:
    first, _, _ = choose(tmp_path, "n\ny\n")
    assert first == (False, "prompt")
    assert preset_lines(tmp_path / "config" / "settings.yaml") == ["graphics: off"]
    second, stdin, out = choose(tmp_path, "y\ny\n")
    assert second == (False, "preset")
    assert stdin.tell() == 0 and out.getvalue() == ""


def test_on_the_root_filesystem_the_answer_applies_once_and_names_lassi_config_dir(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    chosen, _, out = choose(tmp_path, "y\ny\n", env={}, root_device=devices(home), posix=True)
    assert chosen == (True, "prompt")
    assert all_presets(tmp_path) == [], "nothing is saved on the root filesystem"
    assert "LASSI_CONFIG_DIR" in out.getvalue(), out.getvalue()


@pytest.mark.parametrize("tty", [(False, False), (True, False), (False, True)], ids=["pipes", "stdout", "stdin"])
def test_without_a_terminal_nothing_is_asked_read_or_saved(tmp_path: Path, tty: tuple[bool, bool]) -> None:
    chosen, stdin, out = choose(tmp_path, "y\ny\n", tty=tty)
    assert chosen == (False, "no terminal")
    assert stdin.tell() == 0 and out.getvalue() == ""
    assert all_presets(tmp_path) == []


def test_without_a_terminal_a_saved_preset_of_on_does_not_turn_graphics_on(tmp_path: Path) -> None:
    settings().save_preset(tmp_path / "config" / "settings.yaml", True)
    chosen, stdin, out = choose(tmp_path, "y\n", tty=(False, False))
    assert chosen == (False, "no terminal")
    assert stdin.tell() == 0 and out.getvalue() == ""


def test_lassi_graphics_off_beats_a_saved_preset_of_on_on_a_terminal(tmp_path: Path) -> None:
    path = tmp_path / "config" / "settings.yaml"
    settings().save_preset(path, True)
    before = path.read_bytes()
    env = {"LASSI_CONFIG_DIR": str(tmp_path / "config"), "LASSI_GRAPHICS": "off"}
    chosen, stdin, out = choose(tmp_path, "y\ny\n", env=env)
    assert chosen == (False, "LASSI_GRAPHICS")
    assert stdin.tell() == 0 and out.getvalue() == ""
    assert path.read_bytes() == before


def test_the_option_asks_nothing_and_saves_nothing(tmp_path: Path) -> None:
    chosen, stdin, out = choose(tmp_path, "n\ny\n", option="on")
    assert chosen == (True, "--graphics")
    assert stdin.tell() == 0 and out.getvalue() == ""
    assert all_presets(tmp_path) == []


def test_a_corrupt_preset_on_a_terminal_is_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "config" / "settings.yaml"
    path.parent.mkdir()
    path.write_text("graphics: maybe\n", encoding="ascii")
    module = settings()
    with pytest.raises(module.SettingsError) as refused:
        choose(tmp_path, "y\ny\n")
    assert names_path(str(refused.value), path), str(refused.value)
