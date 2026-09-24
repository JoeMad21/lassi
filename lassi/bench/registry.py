"""The bench registry: suite manifests, splits, and reference targets (bible Benchmark Suites).

A suite manifest under assets/bench/ pins its source repository by commit and
lists each item with its split and, per language, the directory and files
that hold that language's version. The registry loads a manifest, hands out
items for a purpose, and refuses an eval item to training (Agent Rule 5:
evaluation splits are untouchable). A direction's reference target is the
item's files in the target language, read from the pinned sources that
tools/fetch_bench.py materializes under $LASSI_SCRATCH, never from git.

Optional keys (P1.2), left out where a manifest does not need them:

- per language, `sha256: {<file>: <hex>}`, the digest of every listed file;
  tools/fetch_bench.py refuses a fetch whose files differ, and the registry's
  reads do not check it;
- per item, `run_args: [<arg>, ...]`, the program's command-line arguments;
- per item, `passfail: [<language>, ...]`, the languages whose program
  prints PASS or FAIL;
- per item, `support: {<name>: <path>}`, harness files: the file at `path`
  in the pinned sources goes into every build directory of the item as
  `name`, beside the model's files (Suite.support_files reads them).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from lassi.core.record import BenchItem

SPLITS = ("train", "eval")
PURPOSES = ("train", "eval")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_REPO = Path(__file__).resolve().parents[2]


class EvalSplitError(ValueError):
    """An eval item was requested for training (Agent Rule 5)."""


@dataclass(frozen=True)
class Direction:
    """A translation direction: source language to target language, named '<source>-<target>' in trial ids."""

    source: str
    target: str

    @property
    def name(self) -> str:
        """Return the direction's name as trial ids spell it, for example 'omp-cuda'."""
        return f"{self.source}-{self.target}"


@dataclass(frozen=True)
class LanguageSources:
    """One language's version of an item: its directory in the source repository, its files, and their sha256.

    `sha256` maps each file name to its sha256 hex digest, or is empty when
    the manifest records none.
    """

    dir: str
    files: tuple[str, ...]
    sha256: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SuiteItem:
    """One bench item: its name, split, sources per language, run arguments, PASS/FAIL languages, and support files.

    `run_args` are the program's command-line arguments, `passfail` the
    languages whose program prints PASS or FAIL, and `support` maps a file
    name in the build directory to its path in the pinned sources.
    """

    name: str
    split: str
    languages: Mapping[str, LanguageSources]
    run_args: tuple[str, ...] = ()
    passfail: frozenset[str] = frozenset()
    support: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Suite:
    """A loaded suite manifest: the pinned repository and its items."""

    name: str
    repo: str
    commit: str
    items: Mapping[str, SuiteItem]

    def item(self, name: str, *, purpose: str) -> SuiteItem:
        """Return item `name` for `purpose` ('train' or 'eval'); an eval item for training raises EvalSplitError."""
        if purpose not in PURPOSES:
            raise ValueError(f"purpose must be one of {', '.join(PURPOSES)}, got {purpose!r}")
        found = self.items.get(name)
        if found is None:
            raise ValueError(f"suite {self.name} has no item {name!r}; items: {', '.join(sorted(self.items))}")
        if purpose == "train" and found.split == "eval":
            raise EvalSplitError(f"{self.name}/{name} is in the eval split and may never be used for training")
        return found

    def reference_target(self, name: str, direction: Direction, root: Path, *, purpose: str) -> dict[str, str]:
        """Return the item's files in the direction's target language (file name -> text), read under `root`."""
        return self._files(name, direction.target, root, purpose)

    def source_files(self, name: str, direction: Direction, root: Path, *, purpose: str) -> dict[str, str]:
        """Return the item's files in the direction's source language (file name -> text), read under `root`."""
        return self._files(name, direction.source, root, purpose)

    def support_files(self, name: str, root: Path, *, purpose: str) -> dict[str, str]:
        """Return the item's support files (build-directory name -> text), read under `root`; {} when it has none.

        They are harness files for every build directory of the item. An eval
        item for training raises EvalSplitError.
        """
        found = self.item(name, purpose=purpose)
        return {file: (Path(root) / path).read_bytes().decode("utf-8") for file, path in found.support.items()}

    def bench_item(self, name: str, direction: Direction) -> BenchItem:
        """Return the Result Record's bench_item for this item and direction."""
        found = self.items.get(name)
        if found is None:
            raise ValueError(f"suite {self.name} has no item {name!r}")
        return BenchItem(suite=self.name, item=name, split=found.split, direction=direction.name)

    def _files(self, name: str, language: str, root: Path, purpose: str) -> dict[str, str]:
        """Read one language's files of an item from the pinned sources under `root`."""
        found = self.item(name, purpose=purpose)
        spec = found.languages.get(language)
        if spec is None:
            known = ", ".join(found.languages)
            raise ValueError(f"{self.name}/{name} has no {language!r} sources; languages: {known}")
        base = Path(root) / spec.dir
        return {file: (base / file).read_bytes().decode("utf-8") for file in spec.files}


def sources_dir(scratch: Path, suite: Suite) -> Path:
    """Return where the pinned sources of `suite` live under the scratch root; never inside the repository."""
    dest = Path(scratch) / "bench" / f"{suite.name}@{suite.commit}"
    resolved = dest.resolve()
    if resolved == _REPO or _REPO in resolved.parents:
        raise ValueError(f"bench sources must live outside the repository, not under {_REPO}")
    return dest


def load_suite(path: Path) -> Suite:
    """Load and check a suite manifest; any problem raises ValueError naming the file and the key."""
    path = Path(path)
    data = yaml.safe_load(path.read_bytes().decode("utf-8"))
    where = str(path)
    _keys(data, {"suite", "repo", "commit", "items"}, where)
    if data["suite"] != path.stem:
        raise ValueError(f"{where}: suite {data['suite']!r} must match the file name {path.stem!r}")
    if not isinstance(data["commit"], str) or not _COMMIT.fullmatch(data["commit"]):
        raise ValueError(f"{where}: commit must be a full 40-hex commit id, got {data['commit']!r}")
    if not isinstance(data["repo"], str) or not data["repo"].startswith("https://"):
        raise ValueError(f"{where}: repo must be an https URL, got {data['repo']!r}")
    if not isinstance(data["items"], dict) or not data["items"]:
        raise ValueError(f"{where}: items must be a non-empty mapping")
    items = {name: _item(name, spec, f"{where}: items.{name}") for name, spec in data["items"].items()}
    return Suite(name=data["suite"], repo=data["repo"], commit=data["commit"], items=items)


def _item(name: Any, spec: Any, where: str) -> SuiteItem:
    """Check one item entry and return it."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"{where}: item name must match {_NAME.pattern}")
    _keys(spec, {"split", "languages"}, where, optional={"run_args", "passfail", "support"})
    if spec["split"] not in SPLITS:
        raise ValueError(f"{where}.split must be one of {', '.join(SPLITS)}, got {spec['split']!r}")
    if not isinstance(spec["languages"], dict) or not spec["languages"]:
        raise ValueError(f"{where}.languages must be a non-empty mapping")
    languages = {lang: _language(value, f"{where}.languages.{lang}") for lang, value in spec["languages"].items()}
    run_args = spec.get("run_args", [])
    if not isinstance(run_args, list) or not all(isinstance(arg, str) for arg in run_args):
        raise ValueError(f"{where}.run_args must be a list of strings, got {run_args!r}")
    passfail = spec.get("passfail", [])
    distinct = isinstance(passfail, list) and all(isinstance(language, str) for language in passfail)
    if not distinct or len(set(passfail)) != len(passfail):
        raise ValueError(f"{where}.passfail must be a list of distinct languages, got {passfail!r}")
    unknown = [language for language in passfail if language not in languages]
    if unknown:
        raise ValueError(f"{where}.passfail names {unknown[0]!r}, not a language of the item ({', '.join(languages)})")
    support = spec.get("support", {})
    if not isinstance(support, dict) or not all(_relative(key) and _relative(path) for key, path in support.items()):
        raise ValueError(f"{where}.support must map relative file names to relative paths, got {support!r}")
    clash = sorted(key for key in support if any(key in sources.files for sources in languages.values()))
    if clash:
        raise ValueError(f"{where}.support names {clash[0]!r}, which is one of the item's own source files")
    return SuiteItem(
        name=name,
        split=spec["split"],
        languages=languages,
        run_args=tuple(run_args),
        passfail=frozenset(passfail),
        support=dict(support),
    )


def _language(value: Any, where: str) -> LanguageSources:
    """Check one language entry: a relative directory, relative file names inside it, and optionally their sha256."""
    _keys(value, {"dir", "files"}, where, optional={"sha256"})
    if not _relative(value["dir"]):
        raise ValueError(f"{where}.dir must be a relative path without '..', got {value['dir']!r}")
    files = value["files"]
    if not isinstance(files, list) or not files or not all(_relative(file) for file in files):
        raise ValueError(f"{where}.files must be a non-empty list of relative paths, got {files!r}")
    digests = value.get("sha256", {})
    valid = isinstance(digests, dict) and set(digests) == set(files)
    valid = valid and all(isinstance(hexdigest, str) and _SHA256.fullmatch(hexdigest) for hexdigest in digests.values())
    if "sha256" in value and not valid:
        raise ValueError(f"{where}.sha256 must map each listed file to a 64-hex sha256 digest, got {digests!r}")
    return LanguageSources(dir=value["dir"], files=tuple(files), sha256=dict(digests))


def _relative(text: Any) -> bool:
    """Return True for a non-empty relative POSIX path with no '..' segment."""
    if not isinstance(text, str) or not text or "\\" in text:
        return False
    path = PurePosixPath(text)
    return not path.is_absolute() and ".." not in path.parts


def _keys(value: Any, allowed: set[str], where: str, optional: set[str] | None = None) -> None:
    """Require a mapping holding every `allowed` key, any of the `optional` keys, and no other key."""
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected a mapping")
    extra = sorted(str(key) for key in value if key not in allowed and key not in (optional or set()))
    missing = sorted(allowed - set(value))
    if extra:
        raise ValueError(f"{where}: unknown key {extra[0]!r}")
    if missing:
        raise ValueError(f"{where}: missing key {missing[0]!r}")
