"""Rubrics: a judge's versioned prompt and output schema under assets/rubrics/<name>/ (task P15.1).

Bible: Agents And Judges (the Judge interface: a rubric is a versioned
prompt plus an output schema under assets/rubrics/; Judge rules: changing a
rubric invalidates its calibration), Repository Layout (assets/rubrics/),
Readability Standards (plain ASCII).

A rubric's name is versioned, <base>-v<N> (RUBRIC_NAME): lowercase letters
and digits in hyphen-joined words, then -v and a version from 1 with no
leading zero, such as equivalence-v1. A changed rubric is a new version,
and the sha256 below tells any two apart. Its directory, RUBRIC_ROOT/<name>/,
holds exactly two kinds of file, each plain ASCII (printable characters,
line feeds, and tabs; lassi.core.interfaces is_plain_ascii):

- schema.yaml, the output schema: a mapping of exactly the keys of
  SCHEMA_KEYS, each required. rubric is the directory's name; prompts a
  non-empty list of distinct plain file names ending in .txt, the rubric's
  prompt files in the order its judge uses them; verdicts a non-empty list
  of distinct verdict tokens a judge may return (VERDICT_TOKEN, such as
  VALID); values a list of distinct tokens naming the judged values a
  judge may return ([] for none). No key may repeat.
- each prompt file schema.yaml lists, not empty.

load_rubric refuses, with RubricError (a ValueError) naming the rubric and
the file or key: an unversioned name, a missing directory, a missing
schema.yaml or listed prompt, a file or directory there that schema.yaml
does not list, an unknown or missing key, a value of the wrong form, and a
file that is not plain ASCII. A loaded Rubric's sha256 (rubric_sha256) is
the sha256 of, for each of its files in sorted name order, the file's name
in ASCII, a NUL byte, its length in bytes as decimal ASCII, a NUL byte, and
its bytes: it names no path, so a copy has the same sha256, and one changed
byte in any file changes it. The recipe loader enters it in the resolved
recipe as judges.<kind>.rubric_sha256 (lassi.core.recipe), so the recipe
hash follows the rubric and a calibration record can name the rubric it
measured. Rubric.check holds a judge's Verdict to the rubric's verdicts and
value names (lassi.core.interfaces checked_verdict).

The rubric root is read from this module at each call, so a caller (a
test) may point RUBRIC_ROOT elsewhere.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from lassi.core.interfaces import VERDICT_TOKEN, Verdict, checked_verdict, is_plain_ascii
from lassi.judges._yaml import read_yaml

REPO = Path(__file__).resolve().parents[2]
# Where rubrics live (bible Repository Layout); load_rubric reads it at each call.
RUBRIC_ROOT = REPO / "assets" / "rubrics"
SCHEMA_FILE = "schema.yaml"
SCHEMA_KEYS = ("rubric", "prompts", "verdicts", "values")
RUBRIC_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*-v[1-9][0-9]*")
# A prompt file's name: one plain name ending in .txt.
_PROMPT_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\.txt")
_NAME_RULE = (
    "a versioned rubric name <base>-v<N>: lowercase letters and digits in hyphen-joined words, then -v and a version "
    "from 1 with no leading zero, such as equivalence-v1"
)


class RubricError(ValueError):
    """A rubric that cannot load; the message names the rubric and the file or key."""


@dataclass(frozen=True)
class Rubric:
    """A loaded rubric: its name, prompts (file name -> text, in listed order), verdicts, values, and sha256.

    `verdicts` and `values` are the output schema's declared verdict tokens
    and judged value names; `sha256` is rubric_sha256 over its files (see
    the module docstring).
    """

    name: str
    prompts: Mapping[str, str]
    verdicts: tuple[str, ...]
    values: tuple[str, ...]
    sha256: str

    def check(self, verdict: object) -> Verdict:
        """Return `verdict` when its verdict and value names are this rubric's; else refuse it (checked_verdict)."""
        return checked_verdict(verdict, self.verdicts, self.values)


def load_rubric(name: str, root: Path | None = None) -> Rubric:
    """Load the rubric `name` from root/<name>/, root defaulting to RUBRIC_ROOT; refuse it with RubricError.

    See the module docstring for what a rubric holds and what is refused.
    """
    where = f"rubric {name!r}"
    if not isinstance(name, str) or RUBRIC_NAME.fullmatch(name) is None:
        raise RubricError(f"{where} is not {_NAME_RULE}")
    directory = Path(RUBRIC_ROOT if root is None else root) / name
    if not directory.is_dir():
        raise RubricError(f"{where} has no directory {directory}")
    document, schema_bytes = read_yaml(directory / SCHEMA_FILE, RubricError)
    schema = _checked_schema(document, name, where)
    _check_listing(directory, schema["prompts"], where)
    contents = {SCHEMA_FILE: schema_bytes}
    prompts: dict[str, str] = {}
    for file in schema["prompts"]:
        contents[file] = (directory / file).read_bytes()
        prompts[file] = _prompt_text(contents[file], file, where)
    return Rubric(
        name=name,
        prompts=MappingProxyType(prompts),
        verdicts=tuple(schema["verdicts"]),
        values=tuple(schema["values"]),
        sha256=rubric_sha256(contents),
    )


def rubric_sha256(contents: Mapping[str, bytes]) -> str:
    """Return the sha256 of a rubric's files, each file name mapped to its bytes (see the module docstring)."""
    digest = hashlib.sha256()
    for file in sorted(contents):
        data = contents[file]
        digest.update(file.encode("ascii") + b"\0" + str(len(data)).encode("ascii") + b"\0" + data)
    return digest.hexdigest()


def _checked_schema(document: Any, name: str, where: str) -> dict[str, Any]:
    """Return the schema mapping when it holds exactly SCHEMA_KEYS in their forms; raise RubricError naming the key."""
    if not isinstance(document, dict):
        raise RubricError(f"{where}: {SCHEMA_FILE} must be a mapping of {', '.join(SCHEMA_KEYS)}")
    unknown = sorted(str(key) for key in document if key not in SCHEMA_KEYS)
    if unknown:
        keys = ", ".join(SCHEMA_KEYS)
        raise RubricError(f"{where}: {SCHEMA_FILE} has the unknown key {unknown[0]!r}; its keys: {keys}")
    missing = [key for key in SCHEMA_KEYS if key not in document]
    if missing:
        raise RubricError(f"{where}: {SCHEMA_FILE} lacks the key {missing[0]!r}, a required choice with no default")
    if document["rubric"] != name:
        found = document["rubric"]
        raise RubricError(f"{where}: {SCHEMA_FILE}'s rubric is {found!r}, not the directory's name {name!r}")
    _check_names(document["prompts"], "prompts", "plain file names ending in .txt", _PROMPT_FILE, where, False)
    _check_names(document["verdicts"], "verdicts", "verdict tokens", VERDICT_TOKEN, where, False)
    _check_names(document["values"], "values", "judged value tokens", VERDICT_TOKEN, where, True)
    return document


def _check_names(value: Any, key: str, noun: str, pattern: re.Pattern[str], where: str, empty: bool) -> None:
    """Refuse `value` unless it is a list of distinct strs each matching `pattern` (empty only when `empty`)."""
    fits = isinstance(value, list) and (empty or bool(value))
    fits = fits and all(isinstance(item, str) and pattern.fullmatch(item) for item in value)
    if not fits or len(set(value)) != len(value):
        size = "a list" if empty else "a non-empty list"
        raise RubricError(f"{where}: {SCHEMA_FILE}'s {key} must be {size} of distinct {noun}, not {value!r}")


def _check_listing(directory: Path, prompts: Sequence[str], where: str) -> None:
    """Refuse a listed prompt that is not a file there, and any entry of the directory the schema does not list."""
    listed = {SCHEMA_FILE, *prompts}
    present = {entry.name: entry for entry in directory.iterdir()}
    for file in prompts:
        if file not in present or not present[file].is_file():
            raise RubricError(f"{where}: {SCHEMA_FILE} lists the prompt {file}, which is not a file in {directory}")
    extra = sorted(entry for entry in present if entry not in listed)
    if extra:
        raise RubricError(
            f"{where}: {directory} holds {extra[0]}, which {SCHEMA_FILE} does not list; a rubric holds only its "
            "schema and the prompts it lists, so its sha256 covers every file it has"
        )


def _prompt_text(data: bytes, file: str, where: str) -> str:
    """Return a prompt file's text when it is plain ASCII and not empty; raise RubricError naming the file."""
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError:
        text = None
    if text is None or not is_plain_ascii(text):
        raise RubricError(f"{where}: the prompt {file} is not plain ASCII (printable characters, line feeds, and tabs)")
    if not text:
        raise RubricError(f"{where}: the prompt {file} is empty")
    return text
