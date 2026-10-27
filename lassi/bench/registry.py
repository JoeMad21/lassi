"""The bench registry: suite manifests, splits, and reference targets (bible Benchmark Suites).

A suite manifest under assets/bench/ pins its source repository by commit and
lists each item with its split and, per language, the directory and files
that hold that language's version. The registry loads a manifest, hands out
items for a purpose, and refuses an eval or unassigned item to every purpose
Agent Rule 5 forbids (evaluation splits are untouchable). A direction's
reference target is the item's files in the target language, read from the
pinned sources that tools/fetch_bench.py materializes under $LASSI_SCRATCH,
never from git, unless the language is tracked (below).

Splits and purposes (task P4.13; OQ-025, option (d)): SPLITS are train, eval,
and unassigned, a split that decides nothing until the owner assigns it.
PURPOSES are train (training), eval (evaluation), prompt-tuning, and harvest
(corpus harvest). Suite.item refuses an eval or unassigned item to train,
prompt-tuning, and harvest with EvalSplitError; a train item serves every
purpose. Suite.train_view() returns a TrainView, the train split alone, which
is the only bench view a Trainer gets (task P17.9).

Optional keys (P1.2), left out where a manifest does not need them:

- per language, `sha256: {<file>: <hex>}`, the digest of every listed file;
  tools/fetch_bench.py refuses a fetch whose files differ, and the registry's
  reads do not check it;
- per item, `run_args: [<arg>, ...]`, the program's command-line arguments;
- per item, `passfail: [<language>, ...]`, the languages whose program
  prints PASS or FAIL;
- per item, `support: {<name>: <path>}`, harness files: the file at `path`
  in the pinned sources goes into every build directory of the item as
  `name`, beside the model's files (Suite.support_files reads them);
- per item, `tolerance: {metric: <pcc | max_abs | ulp>, threshold: <number>}`
  (P4.4), the tolerance the item's two references must meet under an oracle
  that compares output files (lassi.core.tolerance; exact match is
  `{metric: max_abs, threshold: 0}`). load_suite refuses another metric, a
  missing or unknown key, or a threshold lassi.core.tolerance refuses, with
  a ValueError naming `tolerance`.

Optional keys for hand-written pairs (task P4.13, tt-pairs-v0):

- per suite, `pin: <name>`: the toolchain pin whose installed tree holds the
  source repository at the commit (toolchains/<name>.pin); its COMMIT and
  URL must be the manifest's commit and repo. tools/fetch_bench.py copies
  from that tree when it is installed.
- per language, `tracked: true`: `dir` and its files are paths in this
  repository, under assets/ (TRACKED_PREFIX), read from TRACKED_ROOT rather
  than from the fetched sources; such a language lists no sha256, since git
  holds the bytes, and tools/fetch_bench.py never fetches it.
- per language, `support: {<build-dir name>: <entry>}`, that language's own
  harness files, each entry exactly `{tracked: <path under assets/>}`,
  `{tracked: <path under assets/>, sha256: <64 hex>}`, or
  `{upstream: <path in the pinned sources>, sha256: <64 hex>}`. A name may
  not be one of the language's own files or an item-level support name.
  Suite.support_files(..., language=<language>) adds them to the item-level
  ones and refuses a tracked one whose bytes differ from its sha256 (task
  P17.36: a trusted host program a build gets is the manifest's);
  tools/fetch_bench.py fetches and checks each upstream one.
- per language, `reference: <language>` (task P17.36): another language of
  the item, with no reference of its own, whose program is this language's
  reference program: Suite.reference_target gives that language's files and
  Suite.reference_language names it, so the baseline builds and runs it
  with that language's toolchain, support files, and executor. Such a
  language must be `tracked: true`: its own files, the files a translation
  into it writes, are never read, and nothing is fetched for it.
- per item, held-out inputs (bible Harness Contract): `inputs`, a non-empty
  list of generate_inputs entries (name, dtype, shape, dist, lo, hi), each
  of at most MAX_INPUT_ELEMENTS elements (lassi.harness.lassi_io
  element_count: a zero dim gives 0); `outputs`, the output array names
  in the order the program takes them; and `seed`, an int in [0, 2**64).
  The three go together and never with run_args: the program's arguments
  are then program_args, and stage_inputs writes the inputs before a run.
- per item with held-out inputs, `contract` (task P16.1): its input
  contract, a list of exactly one entry per input, each with the keys
  name, dtype (the input's), lo, and hi, and the optional allow_nan and
  allow_inf (true or false), its range holding every dtype value of the
  input's own [lo, hi), so the item's held-out inputs pass it;
  lassi.harness.contract input_contract states the bound rules and the
  check. SuiteItem.contract is the item's InputContract: the key
  read over its inputs, or, without the key, its inputs read as one, and
  None for an item without held-out inputs. load_suite refuses a malformed
  key, and the key on an item without held-out inputs, with a ValueError
  naming the item's `contract`. The key never changes the held-out inputs:
  generate_inputs and stage_inputs read only inputs and seed.
- per item, `unpack_to_dest`: a non-empty note on the item's unpack_to_dest
  check (ttsim Facts; plans/spikes/p4-ttsim-runtime.md).
- per item, `work` (task P17.42; plans/spikes/tt-perf.md, (C); the owner's
  answer of 2026-10-26 to OQ-106, option (a)): the item's
  analytic useful work, the same for every candidate that solves it and
  never a count read from a device, exactly `{flop: <int or null>, ops:
  {<class>: <int>}, bytes: <int>, basis: <note>}`. flop counts the
  floating-point operations of the computation from its shapes (2MNK for a
  matmul, one per element for an elementwise add) and is null for work
  that is no floating-point arithmetic (a copy) or only transcendental
  (exp), whose count goes under ops; bytes is the fewest bytes the device
  must move, each input read once and each output written once in the
  device's format; basis says how they follow from the shapes. Every count
  is an int of at least 0 (a bool is not one). SuiteItem.work is a Work, or
  None for an item without the key; load_suite refuses another key, a
  missing key, a bad count, and an empty basis, naming `work`.
  lassi.analysis.tt_rates divides it by a measured silicon time only.

A tracked suite (task P16.8): a manifest may leave out `repo` and
`commit` together, for a suite written wholly in this repository, such as
adv-seeded-v0. Every language of every item is then `tracked: true`, every
language support entry is `{tracked: ...}`, no item declares item-level
`support` (which reads the fetched sources), and `pin` is refused; each
refusal names the key. Suite.repo and Suite.commit are None and
Suite.tracked is True; sources_dir refuses such a suite, since it has no
fetched sources, and tools/fetch_bench.py has nothing to fetch for it.
One of repo and commit without the other is refused naming the other.

A string value anywhere in a manifest that starts with PLACEHOLDER is
refused, naming its key, so an unfinished manifest never loads.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from lassi.core.record import BenchItem
from lassi.core.tolerance import Tolerance
from lassi.harness.contract import MAX_INPUT_ELEMENTS, InputContract, input_contract
from lassi.harness.lassi_io import SUFFIX, LassiIOError, check_input_spec, element_count, generate_inputs

SPLITS = ("train", "eval", "unassigned")
PURPOSES = ("train", "eval", "prompt-tuning", "harvest")
# The splits Agent Rule 5 keeps from training, prompt tuning, and corpus harvest, and those purposes, with their words.
HELD_SPLITS = ("eval", "unassigned")
FORBIDDEN_PURPOSES = {"train": "training", "prompt-tuning": "prompt tuning", "harvest": "corpus harvest"}
# MAX_INPUT_ELEMENTS (2**20), the most elements one held-out input may hold, is defined in lassi.harness.contract,
# whose check_inputs refuses a proposed input past it too, and imported here, where load_suite caps each input at it.
# Where stage_inputs writes the held-out inputs inside a run's workdir. No model file path may start with "@"
# (lassi.core.files), so none can take its place, as for the ttsim executor's "@ttsim".
INPUTS_DIR = "@inputs"
# The mark of an unfinished manifest value, refused wherever it appears.
PLACEHOLDER = "PLACEHOLDER"
# Tracked files live in this repository, under assets/.
TRACKED_PREFIX = "assets/"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
# An output array name: a lassi_io name whose file, <name>.lassiio, fits a 255-byte file name.
_OUTPUT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,246}")
_REPO = Path(__file__).resolve().parents[2]
# The root tracked paths are read under: this repository (tests point it elsewhere).
TRACKED_ROOT = _REPO
_ITEM_KEYS = {
    "run_args", "passfail", "support", "tolerance", "inputs", "outputs", "seed", "unpack_to_dest", "contract", "work",
}
# The keys of an item's `work` (task P17.42), each required.
_WORK_KEYS = {"flop", "ops", "bytes", "basis"}
_HELD_OUT = ("inputs", "outputs", "seed")


class EvalSplitError(ValueError):
    """An eval or unassigned item was requested for a purpose Agent Rule 5 forbids: training, prompt tuning, harvest."""


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
class SupportFile:
    """One language's support file: its path, whether it is tracked (in this repository), and its sha256.

    An upstream file's path is in the pinned sources and its sha256 is the
    manifest's; a tracked file's path is under TRACKED_ROOT, with the sha256 its manifest
    entry may give (Suite.support_files checks it), else none.
    """

    path: str
    tracked: bool
    sha256: str | None = None


@dataclass(frozen=True)
class LanguageSources:
    """One language's version of an item: its directory, its files, their sha256, and its own support files.

    `sha256` maps each file name to its sha256 hex digest, or is empty when
    the manifest records none. A `tracked` language lives in this
    repository (see base). `support` maps a build-directory name to a
    SupportFile. `reference` names the language whose program is this
    language's reference program (None: its own; task P17.36).
    """

    dir: str
    files: tuple[str, ...]
    sha256: Mapping[str, str] = field(default_factory=dict)
    tracked: bool = False
    support: Mapping[str, SupportFile] = field(default_factory=dict)
    reference: str | None = None

    def base(self, root: Path) -> Path:
        """Return the directory the files are read from: TRACKED_ROOT/dir when tracked, else <root>/dir."""
        return (TRACKED_ROOT if self.tracked else Path(root)) / self.dir


@dataclass(frozen=True)
class Work:
    """An item's analytic useful work (task P17.42): its FLOP count, named operation counts, bytes, and their basis.

    `flop` is None for work that is no floating-point arithmetic or only
    transcendental; `ops` maps an operation class (add, mac, exp) to its
    count; `bytes` is the fewest bytes the device moves; `basis` says how the
    counts follow from the item's shapes. A property of the item, never a
    measurement (see the module docstring).
    """

    flop: int | None
    ops: Mapping[str, int]
    bytes: int
    basis: str


@dataclass(frozen=True)
class SuiteItem:
    """One bench item: its name, split, sources per language, run arguments, and its optional keys.

    `run_args` are the program's command-line arguments, `passfail` the
    languages whose program prints PASS or FAIL, `support` maps a file name
    in the build directory to its path in the pinned sources, and
    `tolerance` is the tolerance its two references must meet (None when the
    manifest declares none). `inputs`, `outputs`, and `seed` are its
    held-out inputs (empty and None when it declares none),
    `unpack_to_dest` its unpack_to_dest note, `contract` its input
    contract (lassi.harness InputContract; None without held-out inputs),
    and `work` its analytic work (Work; None when the manifest declares none).
    """

    name: str
    split: str
    languages: Mapping[str, LanguageSources]
    run_args: tuple[str, ...] = ()
    passfail: frozenset[str] = frozenset()
    support: Mapping[str, str] = field(default_factory=dict)
    tolerance: Tolerance | None = None
    inputs: tuple[Mapping[str, Any], ...] = ()
    outputs: tuple[str, ...] = ()
    seed: int | None = None
    unpack_to_dest: str | None = None
    contract: InputContract | None = None
    work: Work | None = None


def _held_message(suite: str, name: str, split: str, purpose: str) -> str:
    """Return the EvalSplitError text for held item `name` of `split` asked for a FORBIDDEN_PURPOSES purpose."""
    return (
        f"{suite}/{name} is in the {split} split and may never be used for "
        f"{FORBIDDEN_PURPOSES[purpose]} (Agent Rule 5)"
    )


@dataclass(frozen=True)
class TrainView:
    """A suite's train split: the only bench view a Trainer gets (Agent Rule 5; task P17.9).

    `train_items` maps each train-split item's name to its SuiteItem, each
    fetched with Suite.item(purpose="train"); `held` maps each eval and
    unassigned name to its split, kept only so item() can refuse it with
    Suite.item's words. Both mappings are read-only, and the view holds no
    eval or unassigned SuiteItem and no reference to the Suite.
    """

    name: str
    commit: str | None
    train_items: Mapping[str, SuiteItem]
    held: Mapping[str, str]

    def item(self, name: str) -> SuiteItem:
        """Return train item `name`; an eval or unassigned one raises EvalSplitError, an unknown one ValueError."""
        if name in self.held:
            raise EvalSplitError(_held_message(self.name, name, self.held[name], "train"))
        found = self.train_items.get(name)
        if found is None:
            raise ValueError(
                f"suite {self.name} has no train item {name!r}; train items: {', '.join(sorted(self.train_items))}"
            )
        return found


@dataclass(frozen=True)
class Suite:
    """A loaded suite manifest: the pinned repository, its items, and the pin of its installed copy, if any.

    `repo` and `commit` are None for a tracked suite (the module docstring),
    whose every language is in this repository.
    """

    name: str
    repo: str | None
    commit: str | None
    items: Mapping[str, SuiteItem]
    pin: str | None = None

    @property
    def tracked(self) -> bool:
        """Return True for a tracked suite: no upstream repository, every language in this repository."""
        return self.commit is None

    def item(self, name: str, *, purpose: str) -> SuiteItem:
        """Return item `name` for `purpose`, one of PURPOSES (Agent Rule 5).

        An eval or unassigned item asked for training, prompt tuning, or
        corpus harvest raises EvalSplitError naming its split; an unknown
        purpose or item raises ValueError.
        """
        if purpose not in PURPOSES:
            raise ValueError(f"purpose must be one of {', '.join(PURPOSES)}, got {purpose!r}")
        found = self.items.get(name)
        if found is None:
            raise ValueError(f"suite {self.name} has no item {name!r}; items: {', '.join(sorted(self.items))}")
        if purpose in FORBIDDEN_PURPOSES and found.split in HELD_SPLITS:
            raise EvalSplitError(_held_message(self.name, name, found.split, purpose))
        return found

    def train_view(self) -> TrainView:
        """Return the train-only view of this suite (TrainView; task P17.9), each train item fetched for train."""
        train = {name: self.item(name, purpose="train") for name, spec in self.items.items() if spec.split == "train"}
        held = {name: spec.split for name, spec in self.items.items() if spec.split in HELD_SPLITS}
        return TrainView(self.name, self.commit, MappingProxyType(train), MappingProxyType(held))

    def reference_target(self, name: str, direction: Direction, root: Path, *, purpose: str) -> dict[str, str]:
        """Return the item's reference program for the direction's target (file name -> text), read under `root`.

        It is the target language's files, or, for a target that names a
        `reference` language, that language's (reference_language).
        """
        return self._files(name, self.reference_language(name, direction.target), root, purpose)

    def reference_language(self, name: str, language: str) -> str:
        """Return the language whose program is item `name`'s reference in `language`: its `reference`, else itself.

        An unknown item or a language the item lacks raises ValueError.
        """
        found = self.items.get(name)
        if found is None:
            raise ValueError(f"suite {self.name} has no item {name!r}; items: {', '.join(sorted(self.items))}")
        return self._language(found, language).reference or language

    def source_files(self, name: str, direction: Direction, root: Path, *, purpose: str) -> dict[str, str]:
        """Return the item's files in the direction's source language (file name -> text), read under `root`."""
        return self._files(name, direction.source, root, purpose)

    def support_files(
        self, name: str, root: Path, *, purpose: str, language: str | None = None
    ) -> dict[str, str]:
        """Return the item's support files (build-directory name -> text); {} when it has none.

        They are harness files for the item's build directories: the
        item-level ones, read under `root`, and, with `language`, that
        language's own (an upstream one under `root`, a tracked one under
        TRACKED_ROOT). A language the item lacks raises ValueError, and so
        does a tracked file whose bytes differ from the sha256 its entry
        records; an item refused to `purpose` raises EvalSplitError.
        """
        found = self.item(name, purpose=purpose)
        files = {file: (Path(root) / path).read_bytes().decode("utf-8") for file, path in found.support.items()}
        if language is not None:
            for file, entry in self._language(found, language).support.items():
                base = TRACKED_ROOT if entry.tracked else Path(root)
                data = (base / entry.path).read_bytes()
                if entry.tracked and entry.sha256 and hashlib.sha256(data).hexdigest() != entry.sha256:
                    raise ValueError(
                        f"{self.name}/{name}: the {language} support file {entry.path} differs from the sha256 its "
                        f"manifest records ({entry.sha256}); the manifest pins the trusted file's bytes"
                    )
                files[file] = data.decode("utf-8")
        return files

    def bench_item(self, name: str, direction: Direction) -> BenchItem:
        """Return the Result Record's bench_item for this item and direction."""
        found = self.items.get(name)
        if found is None:
            raise ValueError(f"suite {self.name} has no item {name!r}")
        return BenchItem(suite=self.name, item=name, split=found.split, direction=direction.name)

    def _language(self, found: SuiteItem, language: str) -> LanguageSources:
        """Return one language's sources of an item; ValueError naming the item's languages when it lacks it."""
        spec = found.languages.get(language)
        if spec is None:
            known = ", ".join(found.languages)
            raise ValueError(f"{self.name}/{found.name} has no {language!r} sources; languages: {known}")
        return spec

    def _files(self, name: str, language: str, root: Path, purpose: str) -> dict[str, str]:
        """Read one language's files of an item: from the pinned sources under `root`, or tracked ones."""
        spec = self._language(self.item(name, purpose=purpose), language)
        base = spec.base(root)
        return {file: (base / file).read_bytes().decode("utf-8") for file in spec.files}


def program_args(item: SuiteItem) -> list[str]:
    """Return the program's arguments: its run_args, or, with held-out inputs, the input then output files.

    With inputs: `@inputs/<name>.lassiio` (INPUTS_DIR) per input, in the
    manifest's order, then `<name>.lassiio` per output, in its order; both
    relative to the run's workdir, where the program runs.
    """
    if not item.inputs:
        return list(item.run_args)
    inputs = [f"{INPUTS_DIR}/{entry['name']}{SUFFIX}" for entry in item.inputs]
    return [*inputs, *(f"{name}{SUFFIX}" for name in item.outputs)]


def stage_inputs(item: SuiteItem, workdir: Path) -> list[str]:
    """Write the item's held-out inputs into <workdir>/@inputs and return program_args(item).

    An item without inputs writes nothing and gets its run_args. Otherwise
    whatever is at <workdir>/@inputs is removed first, a link itself and
    never what it points to, and a fresh directory gets one
    generate_inputs file per input, drawn with the item's seed, so every
    run of an item (each reference and each attempt) reads the same bytes.
    """
    if not item.inputs:
        return list(item.run_args)
    target = Path(workdir) / INPUTS_DIR
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        try:
            os.unlink(target)
        except (IsADirectoryError, PermissionError):
            os.rmdir(target)  # a directory link on Windows
    elif target.exists():
        shutil.rmtree(target)
    target.mkdir()
    if item.seed is None:
        raise ValueError(f"{item.name} declares inputs and no seed; load_suite refuses such an item")
    generate_inputs(item.inputs, item.seed, target)
    return program_args(item)


def sources_dir(scratch: Path, suite: Suite) -> Path:
    """Return where the pinned sources of `suite` live under the scratch root; never inside the repository.

    A tracked suite has no fetched sources, so it raises ValueError.
    """
    if suite.tracked:
        raise ValueError(f"{suite.name} is a tracked suite: it has no fetched sources; every language is tracked")
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
    _refuse_placeholders(data, where)
    _keys(data, {"suite", "items"}, where, optional={"repo", "commit", "pin"})
    if data["suite"] != path.stem:
        raise ValueError(f"{where}: suite {data['suite']!r} must match the file name {path.stem!r}")
    tracked = "repo" not in data and "commit" not in data
    if not tracked:
        _check_upstream(data, where)
    if not isinstance(data["items"], dict) or not data["items"]:
        raise ValueError(f"{where}: items must be a non-empty mapping")
    items = {name: _item(name, spec, f"{where}: items.{name}") for name, spec in data["items"].items()}
    if tracked:
        _check_tracked(data, items, where)
    return Suite(
        name=data["suite"], repo=data.get("repo"), commit=data.get("commit"), items=items, pin=data.get("pin")
    )


def _check_upstream(data: Mapping[str, Any], where: str) -> None:
    """Refuse a manifest's repo, commit, and pin unless repo is an https URL and commit a full commit id."""
    for key, other in (("repo", "commit"), ("commit", "repo")):
        if key not in data:
            raise ValueError(
                f"{where}: {other} is set without {key}; set both, or neither for a tracked suite whose every "
                "language is in this repository"
            )
    if not isinstance(data["commit"], str) or not _COMMIT.fullmatch(data["commit"]):
        raise ValueError(f"{where}: commit must be a full 40-hex commit id, got {data['commit']!r}")
    if not isinstance(data["repo"], str) or not data["repo"].startswith("https://"):
        raise ValueError(f"{where}: repo must be an https URL, got {data['repo']!r}")
    if "pin" in data:
        _check_pin(data["pin"], data["repo"], data["commit"], where)


def _check_tracked(data: Mapping[str, Any], items: Mapping[str, SuiteItem], where: str) -> None:
    """Refuse, in a tracked suite (no repo or commit), a pin and anything read from fetched sources (task P16.8)."""
    if "pin" in data:
        raise ValueError(f"{where}: pin needs a repo and a commit; a tracked suite has no installed copy to pin")
    for name, item in items.items():
        if item.support:
            raise ValueError(
                f"{where}: items.{name}.support reads the fetched sources, which a tracked suite has none of; give "
                "each language its own tracked support files instead"
            )
        for language, spec in item.languages.items():
            if not spec.tracked:
                raise ValueError(
                    f"{where}: items.{name}.languages.{language} is not tracked, but a suite with no repo and commit "
                    "is a tracked suite, whose every language is tracked: true"
                )
            upstream = sorted(file for file, entry in spec.support.items() if not entry.tracked)
            if upstream:
                raise ValueError(
                    f"{where}: items.{name}.languages.{language}.support.{upstream[0]} is an upstream file, which a "
                    "tracked suite has no sources for"
                )


def _refuse_placeholders(value: Any, where: str, key: str = "") -> None:
    """Refuse the first string value starting with PLACEHOLDER, naming its dotted key."""
    if isinstance(value, str) and value.startswith(PLACEHOLDER):
        raise ValueError(f"{where}: {key or 'the manifest'} is a {PLACEHOLDER} ({value!r}); the manifest is unfinished")
    children = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else ()
    for name, child in children:
        _refuse_placeholders(child, where, f"{key}.{name}" if key else str(name))


def _check_pin(name: Any, repo: str, commit: str, where: str) -> None:
    """Require toolchains/<name>.pin to exist and to name the manifest's commit (COMMIT) and repo (URL)."""
    from lassi.toolchains import pins  # imported here: lassi.toolchains registers every toolchain on import

    try:
        pin = pins.read_pin(name)
    except (OSError, ValueError) as error:
        raise ValueError(f"{where}: pin {name!r} names no readable toolchain pin ({error})") from None
    if pin.get("COMMIT") != commit:
        raise ValueError(f"{where}: pin {name}: toolchains/{name}.pin has COMMIT {pin.get('COMMIT')!r}, not {commit}")
    if pin.get("URL") != repo:
        raise ValueError(f"{where}: pin {name}: toolchains/{name}.pin has URL {pin.get('URL')!r}, not {repo}")


def _item(name: Any, spec: Any, where: str) -> SuiteItem:
    """Check one item entry and return it."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"{where}: item name must match {_NAME.pattern}")
    _keys(spec, {"split", "languages"}, where, optional=_ITEM_KEYS)
    if spec["split"] not in SPLITS:
        raise ValueError(f"{where}.split must be one of {', '.join(SPLITS)}, got {spec['split']!r}")
    if not isinstance(spec["languages"], dict) or not spec["languages"]:
        raise ValueError(f"{where}.languages must be a non-empty mapping")
    languages = {lang: _language(value, f"{where}.languages.{lang}") for lang, value in spec["languages"].items()}
    _check_references(languages, f"{where}.languages")
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
    support = _item_support(spec.get("support", {}), languages, where)
    inputs, outputs, seed = _held_out(spec, where)
    contract = _contract(spec, inputs, where)
    note = spec.get("unpack_to_dest")
    if "unpack_to_dest" in spec and not (isinstance(note, str) and note.strip()):
        raise ValueError(f"{where}.unpack_to_dest must be a non-empty note, got {note!r}")
    return SuiteItem(
        name=name,
        split=spec["split"],
        languages=languages,
        run_args=tuple(run_args),
        passfail=frozenset(passfail),
        support=support,
        tolerance=_tolerance(spec["tolerance"], f"{where}.tolerance") if "tolerance" in spec else None,
        inputs=inputs,
        outputs=outputs,
        seed=seed,
        unpack_to_dest=note,
        contract=contract,
        work=_work(spec["work"], f"{where}.work") if "work" in spec else None,
    )


def _work_count(value: Any, where: str) -> int:
    """Return a work count: an int of at least 0 (a bool is not one); ValueError naming `where` otherwise."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{where} must be an int of at least 0, got {value!r}")
    return value


def _work(value: Any, where: str) -> Work:
    """Check an item's `work` (exactly flop, ops, bytes, and basis) and return it as a Work."""
    _keys(value, _WORK_KEYS, where)
    flop = None if value["flop"] is None else _work_count(value["flop"], f"{where}.flop")
    ops = value["ops"]
    if not isinstance(ops, dict) or not all(isinstance(name, str) and _NAME.fullmatch(name) for name in ops):
        raise ValueError(f"{where}.ops must map operation names to counts, got {ops!r}")
    counts = {name: _work_count(count, f"{where}.ops.{name}") for name, count in ops.items()}
    basis = value["basis"]
    if not isinstance(basis, str) or not basis.strip():
        raise ValueError(f"{where}.basis must be a non-empty note on how the counts follow from the shapes")
    return Work(flop=flop, ops=MappingProxyType(counts), bytes=_work_count(value["bytes"], f"{where}.bytes"),
                basis=basis.strip())


def _item_support(support: Any, languages: Mapping[str, LanguageSources], where: str) -> dict[str, str]:
    """Check the item-level support files: relative names and paths, none an item file or a language support name."""
    if not isinstance(support, dict) or not all(_relative(key) and _relative(path) for key, path in support.items()):
        raise ValueError(f"{where}.support must map relative file names to relative paths, got {support!r}")
    clash = sorted(key for key in support if any(key in sources.files for sources in languages.values()))
    if clash:
        raise ValueError(f"{where}.support names {clash[0]!r}, which is one of the item's own source files")
    shared = sorted(key for key in support if any(key in sources.support for sources in languages.values()))
    if shared:
        raise ValueError(f"{where}.support names {shared[0]!r}, which a language of the item also lists as support")
    return dict(support)


def _held_out(spec: Mapping[str, Any], where: str) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...], int | None]:
    """Check an item's held-out inputs, outputs, and seed (all three or none, never with run_args); return them."""
    present = [key for key in _HELD_OUT if key in spec]
    if not present:
        return (), (), None
    missing = [key for key in _HELD_OUT if key not in spec]
    if missing:
        raise ValueError(f"{where}: inputs, outputs, and seed go together; {missing[0]} is missing")
    if "run_args" in spec:
        raise ValueError(f"{where}.run_args: an item with inputs takes its arguments from inputs and outputs")
    inputs = spec["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError(f"{where}.inputs must be a non-empty list of input specs, got {inputs!r}")
    try:
        check_input_spec(inputs, f"{where}.inputs")
    except LassiIOError as error:
        raise ValueError(str(error)) from None
    for entry in inputs:
        if element_count(entry["shape"], MAX_INPUT_ELEMENTS) is None:
            raise ValueError(
                f"{where}.inputs: {entry['name']!r} holds more than {MAX_INPUT_ELEMENTS} elements, the cap per input"
            )
    outputs = spec["outputs"]
    names = isinstance(outputs, list) and all(isinstance(name, str) and _OUTPUT.fullmatch(name) for name in outputs)
    if not names or not outputs or len(set(outputs)) != len(outputs):
        raise ValueError(f"{where}.outputs must be a non-empty list of distinct output array names, got {outputs!r}")
    seed = spec["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 1 << 64:
        raise ValueError(f"{where}.seed must be an int in [0, 2**64)")
    return tuple(dict(entry) for entry in inputs), tuple(outputs), seed


def _contract(spec: Mapping[str, Any], inputs: tuple[dict[str, Any], ...], where: str) -> InputContract | None:
    """Return the item's input contract: its `contract` key read over its held-out inputs, or the inputs read as one.

    None for an item without held-out inputs, which may not declare the key.
    A malformed key raises ValueError naming `<where>.contract`.
    """
    at = f"{where}.contract"
    if not inputs:
        if "contract" in spec:
            raise ValueError(f"{at}: only an item with held-out inputs (inputs, outputs, and seed) declares a contract")
        return None
    declared = spec.get("contract")
    if "contract" in spec and declared is None:
        raise ValueError(f"{at} must be a list of one entry per held-out input, got null")
    return input_contract(inputs, declared=declared, where=at)


def _tolerance(value: Any, where: str) -> Tolerance:
    """Check an item's tolerance: exactly the keys metric and threshold, a known metric, and a valid threshold."""
    _keys(value, {"metric", "threshold"}, where)
    try:
        return Tolerance(metric=value["metric"], threshold=value["threshold"])
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from None


def _check_references(languages: Mapping[str, LanguageSources], where: str) -> None:
    """Refuse a `reference` that names no other language of the item, or one with a reference of its own."""
    for language, sources in languages.items():
        named = sources.reference
        if named is None:
            continue
        if named == language or named not in languages:
            others = ", ".join(sorted(set(languages) - {language})) or "none"
            raise ValueError(f"{where}.{language}.reference must name another language of the item ({others}), "
                             f"got {named!r}")
        if languages[named].reference is not None:
            raise ValueError(f"{where}.{language}.reference names {named!r}, which names a reference of its own; "
                             "a reference program is a language's own")


def _language(value: Any, where: str) -> LanguageSources:
    """Check one language entry: its directory and files, their sha256, whether it is tracked, and its support."""
    _keys(value, {"dir", "files"}, where, optional={"sha256", "tracked", "support", "reference"})
    tracked = value.get("tracked", False)
    if not isinstance(tracked, bool):
        raise ValueError(f"{where}.tracked must be true or false, got {tracked!r}")
    reference = value.get("reference")
    if "reference" in value and not (isinstance(reference, str) and _NAME.fullmatch(reference)):
        raise ValueError(f"{where}.reference must name a language of the item, got {reference!r}")
    if reference is not None and not tracked:
        raise ValueError(f"{where}: a language with a reference must be tracked: true, since its own files are "
                         "never read and nothing is fetched for it")
    if not _relative(value["dir"]):
        raise ValueError(f"{where}.dir must be a relative path without '..', got {value['dir']!r}")
    if tracked and not value["dir"].startswith(TRACKED_PREFIX):
        raise ValueError(f"{where}.dir: a tracked language lives under {TRACKED_PREFIX} in this repository")
    files = value["files"]
    if not isinstance(files, list) or not files or not all(_relative(file) for file in files):
        raise ValueError(f"{where}.files must be a non-empty list of relative paths, got {files!r}")
    if tracked and "sha256" in value:
        raise ValueError(f"{where}.sha256: a tracked language's files are in this repository, whose history holds them")
    digests = value.get("sha256", {})
    valid = isinstance(digests, dict) and set(digests) == set(files)
    valid = valid and all(isinstance(hexdigest, str) and _SHA256.fullmatch(hexdigest) for hexdigest in digests.values())
    if "sha256" in value and not valid:
        raise ValueError(f"{where}.sha256 must map each listed file to a 64-hex sha256 digest, got {digests!r}")
    support = _language_support(value.get("support", {}), files, f"{where}.support")
    return LanguageSources(
        dir=value["dir"], files=tuple(files), sha256=dict(digests), tracked=tracked, support=support,
        reference=reference,
    )


def _language_support(value: Any, files: list[str], where: str) -> dict[str, SupportFile]:
    """Check a language's support entries: {tracked: <path under assets/>} or {upstream: <path>, sha256: <hex>}."""
    if not isinstance(value, dict) or not all(_relative(name) for name in value):
        raise ValueError(f"{where} must map relative build-directory names to entries, got {value!r}")
    support = {}
    for name, entry in value.items():
        at = f"{where}.{name}"
        if name in files:
            raise ValueError(f"{at}: {name!r} is one of the language's own files")
        if isinstance(entry, dict) and set(entry) in ({"tracked"}, {"tracked", "sha256"}):
            path = entry["tracked"]
            if not (_relative(path) and path.startswith(TRACKED_PREFIX)):
                raise ValueError(f"{at}: tracked must be a relative path under {TRACKED_PREFIX} without '..'")
            digest = entry.get("sha256")
            if "sha256" in entry and not (isinstance(digest, str) and _SHA256.fullmatch(digest)):
                raise ValueError(f"{at}: sha256 must be a 64-hex sha256 digest, got {digest!r}")
            support[name] = SupportFile(path=path, tracked=True, sha256=digest)
        elif isinstance(entry, dict) and set(entry) == {"upstream", "sha256"}:
            if not _relative(entry["upstream"]):
                raise ValueError(f"{at}: upstream must be a relative path without '..', got {entry['upstream']!r}")
            if not (isinstance(entry["sha256"], str) and _SHA256.fullmatch(entry["sha256"])):
                raise ValueError(f"{at}: sha256 must be a 64-hex sha256 digest, got {entry['sha256']!r}")
            support[name] = SupportFile(path=entry["upstream"], tracked=False, sha256=entry["sha256"])
        else:
            raise ValueError(
                f"{at}: an entry is exactly {{tracked: <path under {TRACKED_PREFIX}>}} or "
                f"{{upstream: <path>, sha256: <64 hex>}}, got {entry!r}"
            )
    return support


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
