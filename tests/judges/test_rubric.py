"""Tests for the rubric loader, lassi.judges.rubric (task P15.1).

Bible: Agents And Judges (the Judge interface: a rubric is a versioned
prompt plus an output schema under assets/rubrics/; Judge rules: changing a
rubric invalidates its calibration), Repository Layout (assets/rubrics/),
Readability Standards (plain ASCII). Plan: plans/p15-judges.md, P15.1.

The contract these tests fix:

- RUBRIC_ROOT is the repository's assets/rubrics/. load_rubric(name, root)
  reads root/<name>/, root defaulting to RUBRIC_ROOT as the module holds it
  at the call, and returns a Rubric: name, prompts (each listed prompt
  file's name mapped to its text, in listed order), verdicts and values
  (tuples, as schema.yaml lists them), and sha256.
- A rubric's name is versioned, <base>-v<N>: lowercase letters and digits
  in hyphen-joined words, then -v and a version from 1, with no leading
  zero. Its directory holds exactly schema.yaml and the prompt files
  schema.yaml lists. schema.yaml is a mapping of exactly the keys rubric
  (the directory's name), prompts (a non-empty list of distinct plain
  names ending in .txt), verdicts (a non-empty list of distinct tokens: a
  letter, then letters, digits, '_' or '-', at most 64 characters), and
  values (a list of distinct such tokens, [] for none), with no repeated
  key. Every file is plain ASCII (printable characters, line feeds, and
  tabs) and no prompt is empty.
- Anything else is refused with RubricError, a ValueError, naming the
  rubric and the file or key: an unversioned name, a missing directory, a
  missing schema.yaml or listed prompt, a file or directory the schema does
  not list, an unknown or missing key, a value of the wrong form, and text
  that is not plain ASCII.
- sha256 is the sha256 of, for each of the rubric's files in sorted name
  order, the file's name in ASCII, a NUL byte, its length in bytes as
  decimal ASCII, a NUL byte, and its bytes. It names no path, so a copy
  has the same sha256, and one changed byte in any file changes it.
- Rubric.check(verdict) is checked_verdict (lassi.core.interfaces) against
  the rubric's verdicts and values.

The rubric here is the SYNTHETIC fixture tests/fixtures/judges/rubrics/
synthetic-v1/; no judge uses it. Nothing here is a measurement.
"""

from __future__ import annotations

import hashlib
import importlib
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO / "tests" / "fixtures" / "judges" / "rubrics"
NAME = "synthetic-v1"
SCHEMA = (
    "# SYNTHETIC schema written by the test.\n"
    "rubric: {name}\n"
    "prompts: [prompt.txt]\n"
    "verdicts: [AGREE, DISAGREE]\n"
    "values: [delta]\n"
)
PROMPT = "SYNTHETIC prompt written by the test.\n"


@pytest.fixture(scope="module")
def rubric_module() -> ModuleType:
    """Import lassi.judges.rubric, failing with the task's name when it does not exist yet."""
    try:
        return importlib.import_module("lassi.judges.rubric")
    except ModuleNotFoundError as error:  # pragma: no cover - only before the task's module exists
        pytest.fail(f"lassi.judges.rubric does not exist (task P15.1): {error}")


def write_rubric(root: Path, name: str = NAME, schema: str | None = None, files: dict[str, str] | None = None) -> Path:
    """Write a rubric directory root/<name>/ with schema.yaml and prompt.txt (or `files`); return the directory."""
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "schema.yaml").write_bytes((SCHEMA.format(name=name) if schema is None else schema).encode("utf-8"))
    for file, text in ({"prompt.txt": PROMPT} if files is None else files).items():
        (directory / file).write_bytes(text.encode("utf-8"))
    return directory


def spec_sha256(directory: Path) -> str:
    """Return the rubric sha256 as the module docstring defines it, computed here from the files."""
    digest = hashlib.sha256()
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        data = path.read_bytes()
        digest.update(path.name.encode("ascii") + b"\0" + str(len(data)).encode("ascii") + b"\0" + data)
    return digest.hexdigest()


def load_error(rubric_module: ModuleType, name: str, root: Path) -> str:
    """Expect load_rubric to refuse `name` under `root` with RubricError, a ValueError; return the message."""
    with pytest.raises(rubric_module.RubricError) as caught:
        rubric_module.load_rubric(name, root)
    assert isinstance(caught.value, ValueError)
    return str(caught.value)


# ---------------------------------------------------------------------------
# Loading


def test_the_rubric_root_is_assets_rubrics(rubric_module: ModuleType) -> None:
    assert Path(rubric_module.RUBRIC_ROOT) == REPO / "assets" / "rubrics"


def test_the_fixture_rubric_loads_with_its_prompts_verdicts_values_and_sha256(rubric_module: ModuleType) -> None:
    found = rubric_module.load_rubric(NAME, FIXTURE_ROOT)
    assert found.name == NAME
    assert dict(found.prompts) == {"prompt.txt": (FIXTURE_ROOT / NAME / "prompt.txt").read_text(encoding="ascii")}
    assert found.verdicts == ("AGREE", "DISAGREE")
    assert found.values == ("delta",)
    assert found.sha256 == spec_sha256(FIXTURE_ROOT / NAME)


def test_load_rubric_reads_the_module_root_at_each_call(
    rubric_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rubric_module, "RUBRIC_ROOT", FIXTURE_ROOT)
    assert rubric_module.load_rubric(NAME).sha256 == spec_sha256(FIXTURE_ROOT / NAME)


def test_prompts_keep_the_listed_order(rubric_module: ModuleType, tmp_path: Path) -> None:
    schema = f"rubric: {NAME}\nprompts: [user.txt, malformed.txt, system.txt]\nverdicts: [AGREE]\nvalues: []\n"
    files = {"system.txt": "S\n", "user.txt": "U\n", "malformed.txt": "M\n"}
    write_rubric(tmp_path, schema=schema, files=files)
    found = rubric_module.load_rubric(NAME, tmp_path)
    assert list(found.prompts) == ["user.txt", "malformed.txt", "system.txt"]
    assert found.values == ()


# ---------------------------------------------------------------------------
# The sha256


def test_a_copy_has_the_same_sha256_and_one_changed_byte_changes_it(rubric_module: ModuleType, tmp_path: Path) -> None:
    original = rubric_module.load_rubric(NAME, FIXTURE_ROOT).sha256
    for file in ("prompt.txt", "schema.yaml"):
        root = tmp_path / file
        copy = root / NAME
        shutil.copytree(FIXTURE_ROOT / NAME, copy)
        assert rubric_module.load_rubric(NAME, root).sha256 == original, "the sha256 names no path"
        data = bytearray((copy / file).read_bytes())
        assert chr(data[2]).isalpha(), "the third byte of each fixture file is a letter, in a comment or the text"
        data[2] ^= 0x20  # flip one letter's case: one byte, still plain ASCII, still a valid file
        (copy / file).write_bytes(bytes(data))
        assert rubric_module.load_rubric(NAME, root).sha256 != original, f"one byte of {file} changed"


# ---------------------------------------------------------------------------
# Refusals


@pytest.mark.parametrize(
    "name",
    ["synthetic", "Synthetic-v1", "synthetic-v0", "synthetic-v01", "synthetic_v1", "../synthetic-v1", "a/b-v1", ""],
)
def test_an_unversioned_or_unsafe_name_is_refused(rubric_module: ModuleType, tmp_path: Path, name: str) -> None:
    message = load_error(rubric_module, name, tmp_path)
    assert "-v" in message, message


def test_a_missing_directory_is_refused_naming_the_rubric(rubric_module: ModuleType, tmp_path: Path) -> None:
    assert "absent-v1" in load_error(rubric_module, "absent-v1", tmp_path)


def test_a_missing_schema_is_refused_naming_it(rubric_module: ModuleType, tmp_path: Path) -> None:
    directory = write_rubric(tmp_path)
    (directory / "schema.yaml").unlink()
    message = load_error(rubric_module, NAME, tmp_path)
    assert "schema.yaml" in message and NAME in message, message


def test_a_missing_listed_prompt_is_refused_naming_it(rubric_module: ModuleType, tmp_path: Path) -> None:
    schema = f"rubric: {NAME}\nprompts: [prompt.txt, malformed.txt]\nverdicts: [AGREE]\nvalues: []\n"
    write_rubric(tmp_path, schema=schema)
    assert "malformed.txt" in load_error(rubric_module, NAME, tmp_path)


@pytest.mark.parametrize("extra", ["notes.txt", "README.md"])
def test_a_file_the_schema_does_not_list_is_refused_naming_it(
    rubric_module: ModuleType, tmp_path: Path, extra: str
) -> None:
    directory = write_rubric(tmp_path)
    (directory / extra).write_bytes(b"SYNTHETIC\n")
    assert extra in load_error(rubric_module, NAME, tmp_path)


def test_a_directory_inside_the_rubric_is_refused_naming_it(rubric_module: ModuleType, tmp_path: Path) -> None:
    directory = write_rubric(tmp_path)
    (directory / "nested").mkdir()
    assert "nested" in load_error(rubric_module, NAME, tmp_path)


@pytest.mark.parametrize(
    ("schema", "key"),
    [
        (f"rubric: {NAME}\nprompts: [prompt.txt]\nverdicts: [AGREE]\nvalues: []\nscore: x\n", "score"),
        ("prompts: [prompt.txt]\nverdicts: [AGREE]\nvalues: []\n", "rubric"),
        (f"rubric: {NAME}\nverdicts: [AGREE]\nvalues: []\n", "prompts"),
        (f"rubric: {NAME}\nprompts: [prompt.txt]\nvalues: []\n", "verdicts"),
        (f"rubric: {NAME}\nprompts: [prompt.txt]\nverdicts: [AGREE]\n", "values"),
        ("rubric: other-v1\nprompts: [prompt.txt]\nverdicts: [AGREE]\nvalues: []\n", "rubric"),
        (f"rubric: {NAME}\nprompts: []\nverdicts: [AGREE]\nvalues: []\n", "prompts"),
        (f"rubric: {NAME}\nprompts: prompt.txt\nverdicts: [AGREE]\nvalues: []\n", "prompts"),
        (f"rubric: {NAME}\nprompts: [prompt.txt, prompt.txt]\nverdicts: [AGREE]\nvalues: []\n", "prompts"),
        (f"rubric: {NAME}\nprompts: [prompt.md]\nverdicts: [AGREE]\nvalues: []\n", "prompts"),
        (f"rubric: {NAME}\nprompts: [../prompt.txt]\nverdicts: [AGREE]\nvalues: []\n", "prompts"),
        (f"rubric: {NAME}\nprompts: [prompt.txt]\nverdicts: []\nvalues: []\n", "verdicts"),
        (f"rubric: {NAME}\nprompts: [prompt.txt]\nverdicts: [AGREE, AGREE]\nvalues: []\n", "verdicts"),
        (f"rubric: {NAME}\nprompts: [prompt.txt]\nverdicts: [not a token]\nvalues: []\n", "verdicts"),
        (f"rubric: {NAME}\nprompts: [prompt.txt]\nverdicts: [1]\nvalues: []\n", "verdicts"),
        (f"rubric: {NAME}\nprompts: [prompt.txt]\nverdicts: [AGREE]\nvalues: delta\n", "values"),
        (f"rubric: {NAME}\nprompts: [prompt.txt]\nverdicts: [AGREE]\nvalues: [delta, delta]\n", "values"),
        (f"rubric: {NAME}\nprompts: [prompt.txt]\nverdicts: [AGREE]\nvalues: [two words]\n", "values"),
        (f"rubric: {NAME}\nrubric: {NAME}\nprompts: [prompt.txt]\nverdicts: [AGREE]\nvalues: []\n", "rubric"),
        ("- a list, not a mapping\n", "schema.yaml"),
        (f"rubric: {NAME}\nprompts: [prompt.txt\n", "schema.yaml"),
    ],
)
def test_a_bad_schema_is_refused_naming_the_key(
    rubric_module: ModuleType, tmp_path: Path, schema: str, key: str
) -> None:
    write_rubric(tmp_path, schema=schema)
    message = load_error(rubric_module, NAME, tmp_path)
    assert key in message and NAME in message, message


@pytest.mark.parametrize(
    ("file", "text"),
    [
        ("prompt.txt", "caf\N{LATIN SMALL LETTER E WITH ACUTE}\n"),
        ("prompt.txt", "line one\r\nline two\r\n"),
        ("prompt.txt", ""),
        ("schema.yaml", "# caf\N{LATIN SMALL LETTER E WITH ACUTE}\n" + SCHEMA.format(name=NAME)),
    ],
)
def test_text_that_is_not_plain_ascii_or_an_empty_prompt_is_refused(
    rubric_module: ModuleType, tmp_path: Path, file: str, text: str
) -> None:
    directory = write_rubric(tmp_path)
    (directory / file).write_bytes(text.encode("utf-8"))
    message = load_error(rubric_module, NAME, tmp_path)
    assert file in message, message


# ---------------------------------------------------------------------------
# Checking a verdict against the rubric


def test_rubric_check_takes_a_declared_verdict_and_refuses_another(rubric_module: ModuleType) -> None:
    from lassi.core.interfaces import Verdict

    found = rubric_module.load_rubric(NAME, FIXTURE_ROOT)
    good = Verdict(verdict="DISAGREE", score=0.25, values={"delta": 3.0}, rationale="SYNTHETIC.")
    assert found.check(good) is good
    for bad, field in (
        (Verdict(verdict="VALID", score=0.25, values={}, rationale=""), "Verdict.verdict"),
        (Verdict(verdict="AGREE", score=0.25, values={"power": 1.0}, rationale=""), "Verdict.values"),
    ):
        with pytest.raises(ValueError) as caught:
            found.check(bad)
        assert field in str(caught.value)


def test_the_module_is_documented_typed_and_names_no_project(rubric_module: ModuleType) -> None:
    source = Path(rubric_module.__file__).read_bytes().decode("ascii")
    assert rubric_module.__doc__
    lowered = source.lower()
    assert not [word for word in ("lassi-ee", "lassi-df", "qwen", "gpt-oss", "llama", "projects/") if word in lowered]
    public: list[Any] = [rubric_module.load_rubric, rubric_module.Rubric, rubric_module.RubricError]
    assert all(item.__doc__ for item in public)
