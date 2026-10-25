"""Worked examples packs: complete programs a prompt may carry, checked against their manifest (OQ-083 (b)).

A pack is a directory `<assets root>/EXAMPLES_DIR/<pack>/` holding
MANIFEST.yaml and one file per example file, at `<example>/<path>`, where
`<example>` is the example's directory and `<path>` the file's path in it.
tools/build_tt_examples.py (GENERATOR) writes the pack from the pinned tree:
the files are third-party text, gitignored, so a fresh checkout holds only
the manifest (and the spec) until the tool runs.

The manifest holds MANIFEST_KEYS: the generator, the pack's name, the source
repository and commit, `examples` (in order, each with its `name` and its
`files`, each file's pack path, sha256, and size in bytes, the host program
first), and `sha256`, the pack's digest (pack_sha256): the sha256 of the
lines "<sha256>  <path>\\n" of every file in order, so a changed, added,
dropped, or reordered entry changes it. check_pack names every difference
between a pack's files and its manifest; load_pack returns the examples
only when there is none. Every message names GENERATOR, which rebuilds the
pack.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from lassi.core.files import render_file_blocks

MANIFEST = "MANIFEST.yaml"
# The directory under the assets root that holds the packs, and the tool that writes them.
EXAMPLES_DIR = "examples"
GENERATOR = "tools/build_tt_examples.py"
# The keys every manifest holds; it may hold more (the spec, the checks), which no reader needs.
MANIFEST_KEYS = ("generator", "pack", "repo", "commit", "examples", "sha256")
_FILE_KEYS = {"path", "sha256", "bytes"}


@dataclass(frozen=True)
class Example:
    """One worked example: its directory's name and its files, (path in the directory, text), host program first."""

    name: str
    files: tuple[tuple[str, str], ...]

    @property
    def text(self) -> str:
        """Return the example's files as FILE blocks in their order (file_blocks)."""
        return file_blocks(self.files)


def file_blocks(files: Sequence[tuple[str, str]]) -> str:
    """Return (path, text) pairs as FILE blocks in the order given, one blank line apart.

    Each block is lassi.core.files render_file_blocks's for that one file,
    so the text reads as render_file_blocks's would but keeps the order
    given (the host program first) rather than sorting the paths.
    """
    return "\n".join(render_file_blocks({path: text}) for path, text in files)


def pack_sha256(entries: Sequence[tuple[str, str]]) -> str:
    """Return the pack's sha256 from its (pack path, sha256) entries in order: that of "<sha256>  <path>\\n" lines."""
    lines = "".join(f"{digest}  {path}\n" for path, digest in entries)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def read_manifest(path: Path) -> dict[str, Any]:
    """Return the manifest at `path`; raise ValueError naming it, and GENERATOR, when it is missing or malformed.

    Each example must have a plain `name` and a non-empty `files` list, each
    file a `path` under `<name>/` (relative, no empty, `.`, or `..` part), a
    sha256, and a size in bytes.
    """
    rebuild = f"rebuild the pack with uv run {GENERATOR}"
    try:
        document = yaml.safe_load(Path(path).read_bytes().decode("utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{path} does not exist; {rebuild}") from None
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read {path} ({error}); {rebuild}") from error
    if not isinstance(document, dict) or not set(MANIFEST_KEYS) <= set(document):
        raise ValueError(f"{path} must hold {', '.join(MANIFEST_KEYS)}; {rebuild}")
    examples = document["examples"]
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"{path} lists no example; {rebuild}")
    for example in examples:
        _check_example(path, example, rebuild)
    return document


def _check_example(path: Path, example: Any, rebuild: str) -> None:
    """Raise ValueError naming the manifest when one example entry is malformed (read_manifest)."""
    name = example.get("name") if isinstance(example, dict) else None
    files = example.get("files") if isinstance(example, dict) else None
    if not isinstance(name, str) or not _plain(name) or not isinstance(files, list) or not files:
        raise ValueError(f"{path}: an example lacks a plain name or its files: {example!r}; {rebuild}")
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != _FILE_KEYS:
            raise ValueError(f"{path}: a file of {name} must hold {sorted(_FILE_KEYS)}: {entry!r}; {rebuild}")
        where = entry["path"]
        if not (isinstance(where, str) and where.startswith(f"{name}/") and _plain(where)):
            raise ValueError(f"{path}: {where!r} is not a plain path under {name}/; {rebuild}")


def _plain(path: str) -> bool:
    """Return True for a relative POSIX path with no empty, `.`, or `..` part and no backslash."""
    parts = path.split("/")
    return "\\" not in path and not PurePosixPath(path).is_absolute() and all(
        part not in ("", ".", "..") for part in parts)


def check_pack(tree: Path) -> list[str]:
    """Return one message per difference between the pack at `tree` and its MANIFEST.yaml; [] when it matches.

    A file the manifest lists that is missing, of another size, or of
    another sha256 is named; so is a manifest whose `sha256` is not
    pack_sha256 of its entries, or whose `pack` is not the directory's name.
    A manifest that does not read gives its one message.
    """
    try:
        document = read_manifest(Path(tree) / MANIFEST)
    except ValueError as error:
        return [str(error)]
    problems: list[str] = []
    if document["pack"] != Path(tree).name:
        problems.append(f"{tree / MANIFEST} names the pack {document['pack']!r}, not {Path(tree).name!r}")
    entries: list[tuple[str, str]] = []
    for example in document["examples"]:
        for entry in example["files"]:
            entries.append((entry["path"], entry["sha256"]))
            problem = _file_problem(Path(tree), entry)
            if problem:
                problems.append(problem)
    if pack_sha256(entries) != document["sha256"]:
        problems.append(f"{tree / MANIFEST}: its sha256 {document['sha256']} is not that of its entries, "
                        f"{pack_sha256(entries)}")
    return problems


def _file_problem(tree: Path, entry: Mapping[str, Any]) -> str:
    """Return what is wrong with one listed file of the pack ("" when it is the manifest's bytes)."""
    path = tree / entry["path"]
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return f"{path} is missing"
    except OSError as error:
        return f"cannot read {path} ({error})"
    if len(data) != entry["bytes"]:
        return f"{path} holds {len(data)} bytes, not the manifest's {entry['bytes']}"
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        return f"{path} differs from its sha256 in the manifest"
    return ""


def load_pack(tree: Path) -> tuple[Example, ...]:
    """Return the examples of the pack at `tree` in manifest order, each file's path within its directory.

    Raises ValueError naming every difference check_pack finds, and
    GENERATOR, so a pack that does not match its manifest is never read.
    """
    problems = check_pack(tree)
    if problems:
        raise ValueError(f"{'; '.join(problems)}; rebuild the pack with uv run {GENERATOR}")
    document = read_manifest(Path(tree) / MANIFEST)
    examples: list[Example] = []
    for example in document["examples"]:
        name = example["name"]
        files = []
        for entry in example["files"]:
            text = (Path(tree) / entry["path"]).read_bytes().decode("utf-8")
            files.append((entry["path"][len(name) + 1:], text))
        examples.append(Example(name=name, files=tuple(files)))
    return tuple(examples)
