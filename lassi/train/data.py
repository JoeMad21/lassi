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

Each returns a TrainData whose split_hash is the sha256 of the canonical JSON
of the data's identity (split_hash): for synthetic data the records in file
order, a content hash that the file's name and whitespace do not change; for
bench data the suite, its pinned commit, the split, the sorted item names,
and the manifest's sha256. Every refusal is a RunError (lassi.core.runner)
naming the recipe key, before any directory exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lassi.bench import EvalSplitError, Suite, load_suite
from lassi.core.interfaces import TrainData
from lassi.core.recipe import TRAIN_SPLIT, Recipe
from lassi.core.runner import RunError

REPO = Path(__file__).resolve().parents[2]
# Where data.synthetic names a fixture when TrainOptions.synthetic_dir is None.
SYNTHETIC_DIR = REPO / "tests" / "fixtures" / "train"
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
