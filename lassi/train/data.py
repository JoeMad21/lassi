"""The train layer's data sources and the data split hash (task P17.8).

A train recipe names exactly one data source (lassi.core.recipe,
load_train_recipe):

- data.synthetic, a JSON Lines fixture in the synthetic fixture directory
  (SYNTHETIC_DIR, tests/fixtures/train/, outside every bench suite;
  PHASE-NOTES P17, Agent Rule 5). load_synthetic reads it as plain ASCII
  split on LF only: one JSON object per line, in file order. The last line's
  LF is optional; a CR, a blank line anywhere, a repeated key, NaN, Infinity,
  a number too large for a float (it would read as Infinity), a line that is
  not an object, and a file with no record are refused, and so is a name
  that resolves outside the fixture directory.
- bench, a suite manifest's train split. load_bench loads the manifest and
  fetches every selected item (bench.items, else every item of the train
  split) with Suite.item(name, purpose="train"), so an eval or unassigned
  item raises EvalSplitError, reported as a RunError (Agent Rule 5). The
  TrainData holds the suite's train-only view (Suite.train_view(); task
  P17.9), never the suite. It reads no source file.
- data.corpus, a corpus a tool built from upstream text (OQ-076 (1a)): a
  tracked manifest, <manifests>/<name>/manifest.json (CORPUS_DIR,
  assets/train/, by default), and a generated records file the manifest
  names, <records root>/<records.file>, outside the repository (the records
  root is TrainOptions.corpus_root, else $LASSI_SCRATCH/train-data;
  corpus_root). load_corpus checks that the manifest names the corpus, that
  the records file's sha256 and record count are the manifest's, so only
  the file the tool checked is trained on, and Agent Rule 5: held_out names
  the suite the corpus was built without, and every eval and unassigned
  item of that suite, read through the bench registry, must be among its
  items. The records follow data.synthetic's line rules.

Each returns a TrainData whose split_hash is the sha256 of the canonical JSON
of the data's identity (split_hash): for synthetic data the records in file
order, a content hash that the file's name and whitespace do not change; for
bench data the suite, its pinned commit (null for a tracked suite), the split, the sorted item names,
and the manifest's sha256; for a corpus its name, the records file's sha256,
and the manifest's sha256. Every refusal is a RunError (lassi.core.runner)
naming the recipe key, before any directory exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lassi.bench import EvalSplitError, Suite, load_suite
from lassi.bench.registry import HELD_SPLITS
from lassi.core.interfaces import TrainData
from lassi.core.recipe import TRAIN_SPLIT, Recipe
from lassi.core.runner import RunError

REPO = Path(__file__).resolve().parents[2]
# Where data.synthetic names a fixture when TrainOptions.synthetic_dir is None.
SYNTHETIC_DIR = REPO / "tests" / "fixtures" / "train"
# Where data.corpus names <name>/CORPUS_MANIFEST when TrainOptions.corpus_manifests is None, and the directory under
# $LASSI_SCRATCH that holds the generated records when TrainOptions.corpus_root is None.
CORPUS_DIR = REPO / "assets" / "train"
CORPUS_MANIFEST = "manifest.json"
CORPUS_RECORDS = "train-data"
_RECORDS_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.jsonl")
_SHA256 = re.compile(r"[0-9a-f]{64}")
# A suite name: one plain path segment, as the runner requires of bench.suite.
_SUITE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
# The purpose every bench item is fetched for (lassi.bench.registry PURPOSES).
_PURPOSE = "train"


class _LineError(ValueError):
    """A fixture line that is not one JSON object with distinct keys and no NaN or Infinity."""


def split_hash(identity: Mapping[str, Any]) -> str:
    """Return the sha256 hex digest of `identity` as canonical JSON: sorted keys, "," and ":", ASCII, no NaN.

    A NaN or an infinity raises ValueError.
    """
    text = json.dumps(dict(identity), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def load_synthetic(name: str, root: Path) -> TrainData:
    """Read the JSON Lines fixture <root>/<name> as synthetic TrainData; RunError naming data.synthetic.

    The file must be a regular file whose resolved parent is the resolved
    `root`, plain ASCII, split on LF only (the last line's LF optional), with
    one JSON object per line and at least one line; each refusal names the
    file and, for a bad line, its number.
    """
    root = Path(root)
    path = root / name
    where = f"data.synthetic {name!r} ({path})"
    if not path.is_file():
        raise RunError(f"{where} is not a file in the synthetic fixture directory {root}")
    try:
        inside = path.resolve().parent == root.resolve()
        raw = path.read_bytes()
    except (OSError, RuntimeError) as error:
        raise RunError(f"{where} cannot be read: {error}") from None
    if not inside:
        raise RunError(f"{where} resolves outside the synthetic fixture directory {root}; a fixture is a file in it")
    if not raw.isascii():
        line = raw[: next(i for i, byte in enumerate(raw) if byte > 127)].count(b"\n") + 1
        raise RunError(f"{where}, line {line}: the fixture is not plain ASCII")
    records = _records(raw.decode("ascii"), where)
    identity = {"source": "synthetic", "records": list(records)}
    return TrainData(source="synthetic", split_hash=split_hash(identity), records=records, suite=None, items=())


def _records(text: str, where: str) -> tuple[dict[str, Any], ...]:
    """Return the JSON object of each LF-separated line of `text`; RunError naming `where` and the line."""
    if "\r" in text:
        line = text.count("\n", 0, text.index("\r")) + 1
        raise RunError(f"{where}, line {line}: a CR; a fixture's lines end in LF only")
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()  # the empty remainder after the final LF is not a line
    if not lines:
        raise RunError(f"{where} holds no record; a fixture holds one JSON object per line")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            raise RunError(f"{where}, line {number}: a blank line; every line holds one JSON object")
        try:
            record = json.loads(
                line, object_pairs_hook=_distinct_keys, parse_constant=_no_constant, parse_float=_finite_float
            )
        except (_LineError, ValueError) as error:
            raise RunError(f"{where}, line {number}: not one JSON object: {error}") from None
        if not isinstance(record, dict):
            raise RunError(f"{where}, line {number}: a JSON {type(record).__name__}, not an object")
        records.append(record)
    return tuple(records)


def _distinct_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Return the pairs as a dict; raise _LineError for a key named twice, which JSON would silently overwrite."""
    found: dict[str, Any] = {}
    for key, value in pairs:
        if key in found:
            raise _LineError(f"the key {key!r} appears twice")
        found[key] = value
    return found


def _no_constant(name: str) -> Any:
    """Refuse NaN, Infinity, and -Infinity, which are not JSON numbers."""
    raise _LineError(f"{name} is not a JSON number")


def _finite_float(text: str) -> float:
    """Return the JSON number `text` as a float; refuse one too large for a float, which would read as Infinity."""
    value = float(text)
    if not math.isfinite(value):
        raise _LineError(f"{text} is too large for a float")
    return value


def load_bench(recipe: Recipe, suites_dir: Path) -> tuple[TrainData, dict[str, Any]]:
    """Load bench.suite from `suites_dir` and fetch the selected items for purpose train; return data and identity.

    The items are bench.items, else every item of the train split, each
    fetched with Suite.item(name, purpose="train") and sorted. A suite name
    that is not one plain segment, a missing or refused manifest, a split
    with no items, an item named twice, an unknown item, and an eval or
    unassigned item (EvalSplitError, Agent Rule 5) are RunErrors naming the
    key. The TrainData holds the suite's train-only view (Suite.train_view()).
    The identity is what the split hash is taken over.
    """
    bench = recipe.data["bench"]
    name = bench["suite"]
    manifest = Path(suites_dir) / f"{name}.yaml"
    if not isinstance(name, str) or not _SUITE_NAME.fullmatch(name) or not manifest.is_file():
        raise RunError(f"{recipe.path}: bench.suite {name!r} names no suite manifest; expected {manifest}")
    try:
        raw = manifest.read_bytes()
        suite = load_suite(manifest)
    except (OSError, ValueError) as error:
        raise RunError(
            f"{recipe.path}: bench.suite {name!r}: cannot load the suite manifest {manifest}: {error}"
        ) from error
    items = _selected(recipe, suite)
    identity: dict[str, Any] = {
        "source": "bench",
        "suite": suite.name,
        "commit": suite.commit,
        "split": TRAIN_SPLIT,
        "items": list(items),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
    }
    view = suite.train_view()
    data = TrainData(source="bench", split_hash=split_hash(identity), records=(), suite=view, items=items)
    return data, identity


def _selected(recipe: Recipe, suite: Suite) -> tuple[str, ...]:
    """Return the selected item names, sorted, each fetched through Suite.item for purpose train."""
    selected = recipe.data["bench"].get("items")
    key = "bench.items" if selected is not None else "bench.split"
    if selected is None:
        selected = [item for item, spec in suite.items.items() if spec.split == TRAIN_SPLIT]
        if not selected:
            raise RunError(f"{recipe.path}: bench.suite {suite.name!r} has no items in the split {TRAIN_SPLIT!r}")
    for index, item in enumerate(selected):
        if item in selected[:index]:
            raise RunError(f"{recipe.path}: bench.items names {item!r} twice")
    names = tuple(sorted(selected))
    for item in names:
        try:
            suite.item(item, purpose=_PURPOSE)
        except EvalSplitError as error:
            raise RunError(f"{recipe.path}: {key} names {item!r}, which training may not read: {error}") from error
        except ValueError as error:
            raise RunError(f"{recipe.path}: {key} names {item!r}: {error}") from error
    return names


def corpus_root(option: Path | None) -> Path:
    """Return where corpus records live: `option`, else $LASSI_SCRATCH/train-data; RunError naming data.corpus."""
    if option is not None:
        return Path(option)
    scratch = os.environ.get("LASSI_SCRATCH", "")
    if not scratch:
        raise RunError(
            f"data.corpus: no records root: set LASSI_SCRATCH, whose {CORPUS_RECORDS}/ holds the generated corpus "
            "records outside the repository, or pass the corpus_root option"
        )
    return Path(scratch) / CORPUS_RECORDS


def load_corpus(name: str, manifests: Path, records_root: Path, suites_dir: Path) -> tuple[TrainData, dict[str, Any]]:
    """Read corpus `name` (its manifest under `manifests`, its records under `records_root`); return data, identity.

    The manifest must name the corpus and its records file (records.file,
    records.sha256, records.count), and its held_out must exclude every eval
    and unassigned item of the suite it names (Agent Rule 5). The records file
    must have that sha256 and count and follow data.synthetic's line rules.
    Every refusal is a RunError naming data.corpus.
    """
    where = f"data.corpus {name!r}"
    raw, manifest = _corpus_manifest(Path(manifests) / name / CORPUS_MANIFEST, name, where)
    meta = _records_meta(manifest.get("records"), where)
    _check_held_out(manifest.get("held_out"), Path(suites_dir), where)
    text = _records_text(Path(records_root), meta, where)
    records = _records(text, f"{where} ({meta['file']})")
    if len(records) != meta["count"]:
        raise RunError(f"{where}: {meta['file']} holds {len(records)} records; the manifest's records.count is "
                       f"{meta['count']}")
    identity = {
        "source": "corpus",
        "corpus": name,
        "records_sha256": meta["sha256"],
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
    }
    data = TrainData(source="corpus", split_hash=split_hash(identity), records=records, suite=None, items=())
    return data, identity


def _corpus_manifest(path: Path, name: str, where: str) -> tuple[bytes, dict[str, Any]]:
    """Return the manifest's bytes and its JSON object; RunError when it is missing, not an object, or not `name`'s."""
    if not path.is_file():
        raise RunError(f"{where}: no corpus manifest at {path}; build the corpus with its tool first")
    raw = path.read_bytes()
    try:
        manifest = json.loads(raw.decode("ascii"), object_pairs_hook=_distinct_keys, parse_constant=_no_constant)
    except (UnicodeDecodeError, _LineError, ValueError) as error:
        raise RunError(f"{where}: {path} is not a plain ASCII JSON manifest: {error}") from None
    if not isinstance(manifest, dict):
        raise RunError(f"{where}: {path} holds a JSON {type(manifest).__name__}, not an object")
    if manifest.get("corpus") != name:
        raise RunError(f"{where}: {path} names the corpus {manifest.get('corpus')!r}, not {name!r}")
    return raw, manifest


def _records_meta(meta: Any, where: str) -> dict[str, Any]:
    """Return the manifest's records section after checking file, sha256, and count; RunError naming the key."""
    if not isinstance(meta, dict):
        raise RunError(f"{where}: the manifest's records must be a mapping {{file, sha256, count}}, not {meta!r}")
    file, sha, count = meta.get("file"), meta.get("sha256"), meta.get("count")
    if not isinstance(file, str) or not _RECORDS_FILE.fullmatch(file):
        raise RunError(f"{where}: the manifest's records.file must be one plain name ending in .jsonl, not {file!r}")
    if not isinstance(sha, str) or not _SHA256.fullmatch(sha):
        raise RunError(f"{where}: the manifest's records.sha256 must be 64 lowercase hex digits, not {sha!r}")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise RunError(f"{where}: the manifest's records.count must be an integer of at least 1, not {count!r}")
    return {"file": file, "sha256": sha, "count": count}


def _check_held_out(held: Any, suites_dir: Path, where: str) -> None:
    """Refuse a corpus whose held_out does not exclude every eval and unassigned item of its suite (Agent Rule 5)."""
    items = held.get("items") if isinstance(held, dict) else None
    suite_name = held.get("suite") if isinstance(held, dict) else None
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        raise RunError(f"{where}: the manifest's held_out must name a suite and the list of items the corpus was "
                       f"built without (Agent Rule 5), not {held!r}")
    manifest = suites_dir / f"{suite_name}.yaml"
    if not isinstance(suite_name, str) or not _SUITE_NAME.fullmatch(suite_name) or not manifest.is_file():
        raise RunError(f"{where}: held_out.suite {suite_name!r} names no suite manifest; expected {manifest}")
    try:
        suite = load_suite(manifest)
    except (OSError, ValueError) as error:
        raise RunError(f"{where}: held_out.suite {suite_name!r}: cannot load {manifest}: {error}") from error
    for item, spec in sorted(suite.items.items()):
        if spec.split in HELD_SPLITS and item not in items:
            raise RunError(
                f"{where}: the corpus manifest does not record that it was built without the {spec.split} item "
                f"{item!r} of suite {suite.name!r}; a corpus excludes every eval and unassigned item (Agent Rule 5)"
            )


def _records_text(root: Path, meta: Mapping[str, Any], where: str) -> str:
    """Return the records file's text after the location, sha256, and ASCII checks; RunError naming data.corpus."""
    path = root / meta["file"]
    if not path.is_file():
        raise RunError(f"{where}: no records file at {path}; build the corpus with its tool into {root}")
    if path.resolve().parent != root.resolve():
        raise RunError(f"{where}: {path} resolves outside the records root {root}")
    raw = path.read_bytes()
    found = hashlib.sha256(raw).hexdigest()
    if found != meta["sha256"]:
        raise RunError(f"{where}: {path} has sha256 {found}, not the manifest's records.sha256 {meta['sha256']}; "
                       "only the records the corpus tool checked are trained on, so rebuild them")
    if not raw.isascii():
        raise RunError(f"{where}: {path} is not plain ASCII")
    return raw.decode("ascii")
