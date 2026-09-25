"""The graphics setting of the `lassi` command line and its saved preset (bible, Terminal Presentation).

Graphics are on or off for one invocation, decided by the first of these
sources that applies (resolve_graphics):

1. the global option `--graphics on|off`;
2. the environment variable LASSI_GRAPHICS, `on` or `off` (any other value
   is a SettingsError);
3. off when stdin or stdout is not an interactive terminal (a pipe, a file,
   an rx job, CI, an unattended session): nothing is asked and the preset is
   not read;
4. the saved preset;
5. the prompt, which asks whether to show graphics and then whether to save
   the answer as the preset;
6. off.

The preset is `settings.yaml`, holding `graphics: on` or `graphics: off` and
nothing else (load_preset states the exact format), in
$LASSI_CONFIG_DIR, else in $LASSI_SCRATCH/config/lassi, else in
<home>/.config/lassi (preset_path). On a POSIX host a preset on the
filesystem that holds / is refused (Agent Rule 7): none is read or saved
there, a prompted answer applies to that invocation only, and the prompt
names LASSI_CONFIG_DIR. So on a Linux machine whose home is on /, the preset
saves only with LASSI_CONFIG_DIR or LASSI_SCRATCH set.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TextIO

import yaml

POSIX: bool = os.name == "posix"
"""True on a POSIX host, where preset_path refuses the root filesystem; read at call time."""

GRAPHICS_VARIABLE = "LASSI_GRAPHICS"
CONFIG_VARIABLE = "LASSI_CONFIG_DIR"
SCRATCH_VARIABLE = "LASSI_SCRATCH"
PRESET_NAME = "settings.yaml"
VALUES = {"on": True, "off": False}
PRESET_FORMAT = (
    "a YAML mapping with exactly one key, graphics, whose value is on or off (`graphics: on` or `graphics: off`)"
)
ROOT_REFUSAL = (
    "a graphics preset is never saved on the filesystem that holds /; set "
    f"{CONFIG_VARIABLE} (or {SCRATCH_VARIABLE}) to a directory on another filesystem to save one"
)


class SettingsError(ValueError):
    """A bad --graphics or LASSI_GRAPHICS value, or a preset that cannot be read, checked, or saved."""


class _PresetLoader(yaml.SafeLoader):
    """A safe YAML loader that keeps each boolean's spelling and refuses a repeated mapping key.

    Keeping the spelling means only `on` and `off` read as a setting; the
    repeated-key refusal means `graphics: off` then `graphics: on` is an error,
    never a silent last-wins.
    """

    def construct_mapping(self, node: yaml.Node, deep: bool = False) -> dict[object, object]:
        """Construct a mapping as SafeLoader does; raise ConstructorError when any key appears twice."""
        mapping = super().construct_mapping(node, deep=deep)
        seen: set[object] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark, f"found a repeated key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return mapping


_PresetLoader.add_constructor("tag:yaml.org,2002:bool", yaml.SafeLoader.construct_scalar)


def path_device(path: Path) -> int:
    """Return the device number of the filesystem that holds `path` (os.stat(path).st_dev)."""
    return os.stat(path).st_dev


def _value(text: str, name: str) -> bool:
    """Return True for "on" and False for "off"; raise SettingsError naming `name` and `text` otherwise."""
    if text not in VALUES:
        raise SettingsError(f"{name} must be on or off, not {text!r}")
    return VALUES[text]


def resolve_graphics(
    *,
    option: str | None,
    env: Mapping[str, str],
    stdin_isatty: bool,
    stdout_isatty: bool,
    preset: bool | None,
    ask: Callable[[], bool] | None = None,
) -> tuple[bool, str]:
    """Return (on, source) for one invocation, in the module docstring's order.

    The source is "--graphics", "LASSI_GRAPHICS", "no terminal" (stdin or
    stdout is not a terminal; `preset` is ignored and `ask` is not called),
    "preset", "prompt" (ask() answered), or "default" (a terminal, no preset,
    and no `ask`). Raises SettingsError for an option or LASSI_GRAPHICS value
    other than on or off.
    """
    if option is not None:
        return _value(option, "--graphics"), "--graphics"
    if GRAPHICS_VARIABLE in env:
        return _value(env[GRAPHICS_VARIABLE], GRAPHICS_VARIABLE), GRAPHICS_VARIABLE
    if not (stdin_isatty and stdout_isatty):
        return False, "no terminal"
    if preset is not None:
        return preset, "preset"
    if ask is not None:
        return bool(ask()), "prompt"
    return False, "default"


def _on_root_filesystem(directory: Path, device: Callable[[Path], int]) -> bool:
    """Return True when `directory`, or its nearest existing parent, is on the filesystem that holds /."""
    try:
        existing = next((path for path in (directory, *directory.parents) if path.exists()), directory)
        return device(existing) == device(Path("/"))
    except OSError as error:
        raise SettingsError(f"cannot check the filesystem of the preset directory {directory}: {error}") from None


def preset_path(
    env: Mapping[str, str],
    *,
    home: Path,
    root_device: Callable[[Path], int] | None = None,
    posix: bool | None = None,
) -> Path | None:
    """Return where the graphics preset lives, or None when a POSIX host would put it on the root filesystem.

    The place is $LASSI_CONFIG_DIR/settings.yaml, else
    $LASSI_SCRATCH/config/lassi/settings.yaml, else
    <home>/.config/lassi/settings.yaml. `posix` None means the module's
    POSIX, and `root_device` None the module's path_device, both read at call
    time; off POSIX no root-filesystem refusal applies.
    """
    if env.get(CONFIG_VARIABLE):
        directory = Path(env[CONFIG_VARIABLE])
    elif env.get(SCRATCH_VARIABLE):
        directory = Path(env[SCRATCH_VARIABLE]) / "config" / "lassi"
    else:
        directory = Path(home) / ".config" / "lassi"
    if (POSIX if posix is None else posix) and _on_root_filesystem(directory, root_device or path_device):
        return None
    return directory / PRESET_NAME


def load_preset(path: Path) -> bool | None:
    """Return the saved preset at `path`: True for `graphics: on`, False for `graphics: off`, None with no file.

    The file must be a YAML mapping with exactly one key, `graphics`, whose
    value is `on` or `off`, plain or quoted ("on", 'off'); comments are
    allowed. Any other file is a SettingsError naming the file: one that
    cannot be read or is not valid YAML, an empty file, a non-mapping, a
    mapping with any other key or without `graphics`, a repeated key (never a
    silent last-wins), or any other value. A preset is never guessed.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as error:
        raise SettingsError(f"cannot read the graphics preset {path}: {error}") from None
    try:
        data = yaml.load(text, Loader=_PresetLoader)
    except yaml.YAMLError as error:
        reason = " ".join(str(error).split())
        raise SettingsError(f"the graphics preset {path} is not valid YAML: {reason}") from None
    problem = _preset_problem(data)
    if problem is not None:
        raise SettingsError(f"the graphics preset {path} must be {PRESET_FORMAT}; {problem}")
    return VALUES[data["graphics"]]


def _preset_problem(data: object) -> str | None:
    """Return what keeps loaded preset `data` from being PRESET_FORMAT, or None when it is."""
    if data is None:
        return "it is empty"
    if not isinstance(data, dict):
        return "it is not a YAML mapping"
    others = [repr(key) for key in data if key != "graphics"]
    if others:
        return f"it has another key: {', '.join(others)}"
    if "graphics" not in data:
        return "it has no graphics key"
    value = data["graphics"]
    if not isinstance(value, str) or value not in VALUES:
        return "graphics has no value" if value is None else f"graphics is {value!r}"
    return None


def save_preset(path: Path, on: bool) -> None:
    """Write `graphics: on` or `graphics: off` to `path`, creating its directory; SettingsError when it fails."""
    path = Path(path)
    text = f"graphics: {'on' if on else 'off'}  # lassi terminal graphics; lassi settings graphics on|off changes it\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    except OSError as error:
        raise SettingsError(f"cannot save the graphics preset {path}: {error}") from None


def _yes(stdin: TextIO, out: TextIO, question: str) -> bool:
    """Write `question` to `out` and read one line from `stdin`; True only for y or yes."""
    out.write(question)
    out.flush()
    answer = stdin.readline()
    if not answer.endswith("\n"):
        out.write("\n")
    return answer.strip().lower() in ("y", "yes")


def _ask(stdin: TextIO, out: TextIO, path: Path | None) -> bool:
    """Ask whether to show graphics, then offer to save the answer as the preset at `path`; return the answer.

    With `path` None (the root filesystem) nothing is saved and the answer
    applies to this invocation only.
    """
    on = _yes(stdin, out, "Show terminal graphics (a banner and live tables) for lassi commands? [y/N] ")
    if path is None:
        out.write(f"graphics {'on' if on else 'off'} for this command only: {ROOT_REFUSAL}\n")
        return on
    if _yes(stdin, out, f"Save this answer as your graphics preset in {path}? [y/N] "):
        save_preset(path, on)
        out.write(f"graphics preset saved: {path} (lassi settings graphics on|off changes it)\n")
    return on


def choose_graphics(
    *,
    option: str | None,
    env: Mapping[str, str],
    stdin: TextIO,
    out: TextIO,
    stdin_isatty: bool,
    stdout_isatty: bool,
    home: Path,
    root_device: Callable[[Path], int] | None = None,
    posix: bool | None = None,
    prompt: bool = True,
) -> tuple[bool, str]:
    """Decide graphics for one invocation and return (on, source), as resolve_graphics names the source.

    The preset is found with preset_path and read with load_preset only when
    the option, LASSI_GRAPHICS, and the terminal check leave the choice open.
    Then, with no preset and `prompt` true, the questions are written to
    `out` and answered from `stdin`, and a yes to the second saves the
    preset; with `prompt` false nothing is asked and the source is "default".
    Nothing is read from `stdin` or written to `out` unless the prompt asks.
    """
    decided = resolve_graphics(
        option=option, env=env, stdin_isatty=stdin_isatty, stdout_isatty=stdout_isatty, preset=None
    )
    if decided[1] != "default":
        return decided
    path = preset_path(env, home=home, root_device=root_device, posix=posix)
    preset = None if path is None else load_preset(path)
    ask = (lambda: _ask(stdin, out, path)) if prompt else None
    return resolve_graphics(option=option, env=env, stdin_isatty=True, stdout_isatty=True, preset=preset, ask=ask)
