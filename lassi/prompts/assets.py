"""Prompt fragments and context packs checked against their manifests (bible Design Principle 3).

A manifest tree is a directory holding MANIFEST.yaml and one `<key>.txt` per
manifest entry. The manifest maps `entries` to a list of {key, sha256, and
the entry's provenance (source, cell, sources)}; sha256 is the hex digest of
the file's bytes. Generated trees (the upstream text of OQ-018, and other
third-party text such as the tt-metalium pack) are gitignored except their
manifests, so a fresh checkout holds only the manifests until their
generator runs. A manifest names that tool in `generator`; one without the
field was written by GENERATOR, and a `generator` that is not a tools/
script path is an error. Every error about a tree names its tool.

A recipe names a prompt set with `prompts` (`<root>/prompts/<set>/`) and its
context packs with `context` (each `<root>/context/<name>/`). A pack's text
is its entries' texts in manifest order joined with PACK_JOINER (a pack of
one entry is that entry's text), and every entry key of a pack ends in the
language the pack serves (lassi.core.fragments.pack_language).
load_recipe_assets returns the text exactly as stored: UTF-8, no newline
translation, and no template placeholders filled. Template sets without a
manifest are rendered with lassi.prompts.render instead.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

MANIFEST = "MANIFEST.yaml"
# The tool that regenerates a generated tree whose manifest names no `generator`; every missing or changed file
# of such a tree names it.
GENERATOR = "tools/extract_lassi_assets.py"
# What joins the entries of a pack of several entries, in manifest order; a joiner that is only whitespace lives
# in code, never in a fragment.
PACK_JOINER = "\n"

# A set, pack, or entry key: one plain path segment, so a name never reaches outside its root.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
# A manifest's `generator`: a script under tools/, so an error message never names anything else.
_GENERATOR = re.compile(r"tools/[A-Za-z0-9_]+\.py")


@dataclass(frozen=True)
class RecipeAssets:
    """The checked text a recipe names: `fragments` {key: text} of its prompt set, `packs` {name: text}."""

    fragments: Mapping[str, str]
    packs: Mapping[str, str]


def default_root() -> Path:
    """Return the repository's assets/ directory."""
    return Path(__file__).resolve().parents[2] / "assets"


def load_recipe_assets(recipe: Mapping[str, Any], root: Path | str | None = None) -> RecipeAssets:
    """Return the fragments of the recipe's prompt set and the packs its `context` names, and only those.

    `root` (default: default_root()) holds `prompts/` and `context/`. Raises
    ValueError naming the tree's generator when a file is missing or its
    sha256 differs from the manifest, naming the pack when a pack the recipe
    names has no tree or its entry keys end in different languages, and
    naming the key when the recipe's `prompts` or `context` is not a plain
    name or list of names.
    """
    base = default_root() if root is None else Path(root)
    set_name = _checked_name(recipe.get("prompts"), "prompts")
    set_dir = base / "prompts" / set_name
    if not set_dir.is_dir():
        raise ValueError(f"no prompt set {set_name!r}: {set_dir} is not a directory")
    fragments = load_tree(set_dir)
    names = recipe.get("context") or []
    if not isinstance(names, (list, tuple)):
        raise ValueError(f"the recipe's context must be a list of pack names, got {names!r}")
    packs: dict[str, str] = {}
    for name in names:
        pack_dir = base / "context" / _checked_name(name, "context")
        if not pack_dir.is_dir():
            raise ValueError(f"unknown context pack {name!r}: {pack_dir} is not a directory")
        packs[name] = pack_text(name, load_tree(pack_dir))
    return RecipeAssets(MappingProxyType(fragments), MappingProxyType(packs))


def pack_text(name: str, texts: Mapping[str, str]) -> str:
    """Return the text of pack `name` from its entries {key: text}: the texts in order, joined with PACK_JOINER.

    Raises ValueError naming the pack when it has no entry or its entry keys
    end in different languages (the last dot-separated part of a key, as
    lassi.core.fragments.pack_language reads it).
    """
    if not texts:
        raise ValueError(f"context pack {name!r} has no entry")
    languages = sorted({key.rsplit(".", 1)[-1] for key in texts})
    if len(languages) != 1:
        raise ValueError(f"context pack {name!r} serves several languages ({', '.join(languages)}); "
                         "every entry key of a pack ends in the one language it serves")
    return PACK_JOINER.join(texts.values())


def load_tree(tree: Path) -> dict[str, str]:
    """Return {key: text} for every entry of the manifest tree `tree`, each file checked against its sha256.

    Raises ValueError naming the tree's generator (the manifest's
    `generator`, else GENERATOR) when the manifest or a file is missing,
    unreadable, or not what the manifest records, and naming the manifest
    when its `generator` is not a tools/ script path.
    """
    entries, generator = _manifest_entries(tree)
    regenerate = f"regenerate it with uv run {generator}"
    texts: dict[str, str] = {}
    for key, digest in entries:
        path = tree / f"{key}.txt"
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise ValueError(f"{path} is missing; {regenerate}") from None
        except OSError as error:
            raise ValueError(f"cannot read {path} ({error}); {regenerate}") from error
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"{path} differs from its sha256 in {tree / MANIFEST}; {regenerate}")
        try:
            texts[key] = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{path} is not UTF-8 ({error}); {regenerate}") from error
    return texts


def _manifest_entries(tree: Path) -> tuple[list[tuple[str, str]], str]:
    """Return ((key, sha256) per entry, generator) of `tree`/MANIFEST.yaml; raise ValueError naming it if bad.

    The generator is the manifest's `generator`, else GENERATOR when the
    manifest has no such field; a `generator` that is not a plain tools/
    path is a ValueError naming the manifest and the value, never a silent
    fallback.
    """
    path = tree / MANIFEST
    try:
        loaded = yaml.safe_load(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read the manifest {path} ({error}); restore the tree's committed manifest, "
                         f"which uv run {_unread_generator(path)} regenerates with the tree") from error
    generator = _generator(path, loaded)
    entries = loaded.get("entries") if isinstance(loaded, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"{path} has no entries list; regenerate it with uv run {generator}")
    pairs: list[tuple[str, str]] = []
    for entry in entries:
        key = entry.get("key") if isinstance(entry, dict) else None
        digest = entry.get("sha256") if isinstance(entry, dict) else None
        if not (isinstance(key, str) and _NAME.fullmatch(key) and isinstance(digest, str)
                and _SHA256.fullmatch(digest)):
            raise ValueError(f"{path}: an entry lacks a plain key or a sha256: {entry!r}; regenerate it with "
                             f"uv run {generator}")
        pairs.append((key, digest))
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError(f"{path} repeats a key; regenerate it with uv run {generator}")
    return pairs, generator


def _unread_generator(path: Path) -> str:
    """Return the tool that generates the tree of a manifest that does not load: its `generator:` line, else GENERATOR.

    The line is read from the raw bytes, so a manifest that is not YAML or
    not UTF-8 still names its tool; a missing file, or a line that is not
    a plain tools/ script path (_GENERATOR), gives GENERATOR.
    """
    try:
        raw = path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return GENERATOR
    found = re.search(r"^generator:[ \t]*(\S+)[ \t]*$", raw, flags=re.M)
    return found.group(1) if found and _GENERATOR.fullmatch(found.group(1)) else GENERATOR


def _generator(path: Path, loaded: object) -> str:
    """Return the tool a loaded manifest names in `generator`, or GENERATOR when it names none.

    Raises ValueError naming the manifest `path` when `generator` is present
    but is not a plain tools/ script path (_GENERATOR).
    """
    if not isinstance(loaded, dict) or "generator" not in loaded:
        return GENERATOR
    named = loaded["generator"]
    if not (isinstance(named, str) and _GENERATOR.fullmatch(named)):
        raise ValueError(f"{path}: its generator {named!r} is not a script path matching {_GENERATOR.pattern}; "
                         "restore the committed manifest")
    return named


def _checked_name(value: object, recipe_key: str) -> str:
    """Return `value` when it is one plain path segment; else raise ValueError naming the recipe key."""
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"the recipe's {recipe_key} must name a set or pack matching {_NAME.pattern}, got {value!r}")
    return value
