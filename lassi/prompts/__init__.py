"""Prompt templates: one plain-text file per prompt under assets/prompts/<set>/ (bible Design Principle 3).

Prompts are versioned data files, never string literals in code. A prompt set
is a directory; each prompt in it is `<prompt name>.txt`, plain ASCII text
with string.Template placeholders written `$name` (or `${name}`), and `$$`
for a literal dollar sign. A recipe picks its set with the key `prompts`, and
each stage names the prompt it renders (bible Readability Standards, Prompts
row).

render fills a template with Template.substitute: every placeholder must get
a value, and a field value is inserted as is, so a `$` inside it is never
read as a placeholder. Every problem raises ValueError naming the set or the
file.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from string import Template

# The file suffix of every prompt template.
SUFFIX = ".txt"

# A set or prompt name: one plain path segment, so a name never reaches outside its root.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def default_roots() -> tuple[Path, ...]:
    """Return the directories searched for a prompt set: the repository's assets/prompts/."""
    return (Path(__file__).resolve().parents[2] / "assets" / "prompts",)


def prompt_dir(set_name: str, roots: Sequence[Path] | None = None) -> Path:
    """Return the directory of prompt set `set_name`: the first <root>/<set_name> that is a directory.

    `roots` (default: default_roots()) are searched in order. Raises
    ValueError naming the set and the places tried when no root holds it, or
    when the name is not one plain path segment.
    """
    _check_name(set_name, "prompt set")
    search = [Path(root) for root in (default_roots() if roots is None else roots)]
    for root in search:
        candidate = root / set_name
        if candidate.is_dir():
            return candidate
    tried = ", ".join(str(root / set_name) for root in search) or "nowhere (no roots to search)"
    raise ValueError(f"no prompt set {set_name!r}; tried {tried}")


def render(set_name: str, prompt_name: str, fields: Mapping[str, str]) -> str:
    """Return the prompt `<set_name>/<prompt_name>.txt` with every `$name` placeholder filled from `fields`.

    The set is found by prompt_dir with the default roots. Fields that the
    template does not use are ignored. Raises ValueError naming the file when
    the file is missing or not UTF-8, when a placeholder has no field, or
    when the template holds a `$` that is not a placeholder.
    """
    _check_name(prompt_name, "prompt")
    path = prompt_dir(set_name) / f"{prompt_name}{SUFFIX}"
    try:
        text = path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        raise ValueError(f"no prompt file {path}") from None
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read the prompt file {path}: {error}") from error
    try:
        return Template(text).substitute(fields)
    except KeyError as error:
        raise ValueError(f"{path}: the placeholder ${error.args[0]} has no field value") from None
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None


def _check_name(name: str, what: str) -> None:
    """Raise ValueError unless `name` is one plain path segment (letters, digits, '.', '_', '-')."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"a {what} name must match {_NAME.pattern}, got {name!r}")
