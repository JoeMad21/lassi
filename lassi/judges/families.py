"""The model family table, assets/models/families.yaml, and its reader (task P15.1).

Bible: Agents And Judges (recipes bind roles to models; a judge uses a
different model family from the generator in the same trial, and both are
recorded), Model Serving (Serving Rules, the reasoning effort; OQ-080).

The table maps each model id to its family, read from the id itself (the
P15 plan's planning decision "Model families"). The file is plain ASCII
YAML holding exactly the keys of TABLE_KEYS, with no repeated key, so a
model id written twice is refused rather than read last-wins:

- families: a non-empty list of distinct family names (FAMILY_NAME: a
  lowercase letter, then lowercase letters, digits, and hyphens);
- reasoning_effort: a list of distinct listed families whose models take a
  reasoning effort, so a judge on one sets judges.<kind>.sampling.
  reasoning_effort and a judge on any other may not (lassi.core.recipe);
- models: a non-empty mapping of each model id (a non-empty str, exactly
  as recipes write it) to a listed family.

load_families refuses anything else with FamilyError (a ValueError) naming
the file and the key. FamilyTable.family matches an id exactly as written,
with no case folding and no prefix stripped, and gives None for an id the
table does not list; the recipe loader refuses such an id when a recipe
binds a judge. The table's file is read from this module at each call, so
a caller (a test) may point FAMILIES_FILE elsewhere.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from lassi.judges._yaml import read_yaml

REPO = Path(__file__).resolve().parents[2]
# The committed table; load_families reads it at each call.
FAMILIES_FILE = REPO / "assets" / "models" / "families.yaml"
TABLE_KEYS = ("families", "reasoning_effort", "models")
FAMILY_NAME = re.compile(r"[a-z][a-z0-9-]*")


class FamilyError(ValueError):
    """A family table that cannot load; the message names the file and the key."""


@dataclass(frozen=True)
class FamilyTable:
    """A loaded family table: the families as listed, those that take a reasoning effort, and each model's family."""

    path: Path
    families: tuple[str, ...]
    reasoning_effort: frozenset[str]
    models: Mapping[str, str]

    def family(self, model_id: str) -> str | None:
        """Return the family of `model_id` exactly as written, or None when the table does not list it."""
        return self.models.get(model_id)


def load_families(path: Path | None = None) -> FamilyTable:
    """Load the family table at `path`, defaulting to FAMILIES_FILE; refuse it with FamilyError naming the key."""
    path = Path(FAMILIES_FILE if path is None else path)
    document, _ = read_yaml(path, FamilyError)
    if not isinstance(document, dict):
        raise FamilyError(f"{path} must be a mapping of {', '.join(TABLE_KEYS)}")
    unknown = sorted(str(key) for key in document if key not in TABLE_KEYS)
    if unknown:
        raise FamilyError(f"{path} has the unknown key {unknown[0]!r}; its keys: {', '.join(TABLE_KEYS)}")
    missing = [key for key in TABLE_KEYS if key not in document]
    if missing:
        raise FamilyError(f"{path} lacks the key {missing[0]!r}, a required choice with no default")
    families = _names(document["families"], path, "families", allow_empty=False)
    reasoning = _names(document["reasoning_effort"], path, "reasoning_effort", allow_empty=True)
    strangers = [name for name in reasoning if name not in families]
    if strangers:
        raise FamilyError(f"{path}: reasoning_effort names {strangers[0]!r}, which families does not list")
    return FamilyTable(
        path=path,
        families=families,
        reasoning_effort=frozenset(reasoning),
        models=_models(document["models"], path, frozenset(families)),
    )


def _names(value: Any, path: Path, key: str, *, allow_empty: bool) -> tuple[str, ...]:
    """Return `value` as a tuple when it is a list of distinct family names; raise FamilyError naming `key`."""
    fits = isinstance(value, list) and (allow_empty or bool(value))
    fits = fits and all(isinstance(item, str) and FAMILY_NAME.fullmatch(item) for item in value)
    if not fits or len(set(value)) != len(value):
        size = "a list" if allow_empty else "a non-empty list"
        raise FamilyError(
            f"{path}: {key} must be {size} of distinct family names (a lowercase letter, then lowercase letters, "
            f"digits, and hyphens), not {value!r}"
        )
    return tuple(value)


def _models(value: Any, path: Path, families: frozenset[str]) -> Mapping[str, str]:
    """Return the models mapping, read-only, when each non-empty str id maps to a listed family; else FamilyError."""
    if not isinstance(value, dict) or not value:
        raise FamilyError(f"{path}: models must be a non-empty mapping of model id to family, not {value!r}")
    for model_id, family in value.items():
        if not isinstance(model_id, str) or not model_id:
            raise FamilyError(f"{path}: models has the id {model_id!r}; each model id is a non-empty string")
        if not isinstance(family, str) or family not in families:
            raise FamilyError(f"{path}: models maps {model_id!r} to {family!r}, which families does not list")
    return MappingProxyType(dict(value))
