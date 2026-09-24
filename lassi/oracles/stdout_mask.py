"""The stdout_mask and passfail oracles (bible Oracles; Harness Contract; Component Interfaces, Oracle rules).

Rules [DESIGN, task P1.7]:

- stdout_mask: both stdouts are split into lines on "\\n" only. A line that
  one of the item's masks matches (a regular expression searched in the
  line) is a timing line and is replaced by one fixed marker line, never
  deleted, so a missing or an extra timing line is still a difference.
  per_input is 1.0 when the masked candidate equals the masked reference
  exactly (no whitespace or case normalization), else 0.0; mean is the
  arithmetic mean of per_input.
- passfail: PASS and FAIL are whole-word tokens anywhere in stdout. They are
  read only when the suite manifest lists the target language under the
  item's `passfail`. Where they are read, a candidate scores 1.0 when its
  stdout holds PASS, holds no FAIL, and stdout_mask scores 1.0; a missing
  PASS is a fail. Where they are not read, the value is stdout_mask's alone.
  A program's own PASS is never the only signal (Oracle rules).

Masks live in `assets/harness/masks/<suite>.yaml`, one list per item,
derived from the print statements of the item's pinned sources in both
languages:

    suite: <suite name>
    items:
      <item>: [<regex>, ...]

`load_masks(suite, root=None)` reads `<root>/<suite>.yaml` (root defaults to
assets/harness/masks) and fails loudly: a missing file raises ValueError
naming the suite, and a regex that does not compile raises ValueError naming
its item.

The Oracle component is registered as `stdout_mask` and built from its
recipe section as `StdoutMaskOracle(passfail=<bool>)`; `passfail` has no
default. `for_item` binds an item's masks and whether its target language
prints PASS/FAIL, and `align(reference, candidate)` then compares the two
RunResults' stdout under those rules.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from lassi.core.interfaces import RunResult
from lassi.core.record import Alignment
from lassi.core.registry import register

# Where the suites' mask files live: assets/harness/masks/<suite>.yaml.
MASKS_ROOT = Path(__file__).resolve().parents[2] / "assets" / "harness" / "masks"
# The line mask_stdout shows in place of each timing line; the scores compare masked lines structurally.
MASK_MARKER = "<timing line masked>"
# The self-reported verdict tokens: whole words, anywhere in a line, case as printed.
PASS_TOKEN = re.compile(r"\bPASS\b")
FAIL_TOKEN = re.compile(r"\bFAIL\b")
# A suite name: one plain path segment.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")


def load_masks(suite: str, root: Path | None = None) -> Mapping[str, tuple[str, ...]]:
    """Return the suite's masks, item -> regexes, read from `<root>/<suite>.yaml` and checked.

    Raises ValueError naming the suite for a bad name, a missing file, or a
    malformed file, and naming the item for a mask that is not a non-empty
    string or does not compile.
    """
    if not isinstance(suite, str) or not _NAME.fullmatch(suite):
        raise ValueError(f"mask suite name must be one plain path segment, got {suite!r}")
    path = Path(MASKS_ROOT if root is None else root) / f"{suite}.yaml"
    if not path.is_file():
        raise ValueError(f"suite {suite!r} has no mask file at {path}; add one derived from its sources' prints")
    data = yaml.safe_load(path.read_bytes().decode("utf-8"))
    where = f"{path} (suite {suite!r})"
    if not isinstance(data, dict) or set(data) != {"suite", "items"}:
        raise ValueError(f"{where}: the mask file must hold exactly the keys suite and items")
    if data["suite"] != suite:
        raise ValueError(f"{where}: suite {data['suite']!r} must match the file name")
    if not isinstance(data["items"], dict) or not data["items"]:
        raise ValueError(f"{where}: items must be a non-empty mapping of item -> list of regexes")
    return MappingProxyType({str(item): _item_masks(item, masks, where) for item, masks in data["items"].items()})


def _item_masks(item: Any, masks: Any, where: str) -> tuple[str, ...]:
    """Return one item's masks after checking each is a non-empty string that compiles."""
    if not isinstance(masks, list) or not masks:
        raise ValueError(f"{where}: item {item!r} must list at least one mask")
    for mask in masks:
        if not isinstance(mask, str) or not mask:
            raise ValueError(f"{where}: item {item!r} has a mask that is not a non-empty string: {mask!r}")
        try:
            re.compile(mask)
        except re.error as error:
            raise ValueError(f"{where}: item {item!r} has a mask that does not compile: {mask!r} ({error})") from None
    return tuple(masks)


def _masked_lines(text: str, masks: Sequence[str]) -> list[str | None]:
    """Return the lines of `text` split on "\\n", with None in place of each timing line."""
    patterns = [re.compile(mask) for mask in masks]
    return [None if any(pattern.search(line) for pattern in patterns) else line for line in text.split("\n")]


def mask_stdout(text: str, masks: Sequence[str]) -> str:
    """Return `text` with each line that a mask matches replaced by MASK_MARKER; no line is added or removed."""
    return "\n".join(MASK_MARKER if line is None else line for line in _masked_lines(text, masks))


def stdout_mask_score(reference: str, candidate: str, masks: Sequence[str]) -> float:
    """Return 1.0 when the masked candidate stdout equals the masked reference stdout exactly, else 0.0."""
    return 1.0 if _masked_lines(candidate, masks) == _masked_lines(reference, masks) else 0.0


def passfail_score(reference: str, candidate: str, masks: Sequence[str], *, reads_passfail: bool) -> float:
    """Return passfail's value: stdout_mask's alone, or, where PASS/FAIL is read, also needing PASS and no FAIL.

    `reads_passfail` is True only when the suite manifest lists the target
    language under the item's `passfail`. A missing PASS is a fail there.
    """
    matched = stdout_mask_score(reference, candidate, masks)
    if not reads_passfail:
        return matched
    verdict = PASS_TOKEN.search(candidate) is not None and FAIL_TOKEN.search(candidate) is None
    return matched if verdict else 0.0


def alignment(per_input: Sequence[float]) -> Alignment:
    """Return the Result Record Alignment of per-input values: the values and their arithmetic mean."""
    values = [float(value) for value in per_input]
    if not values:
        raise ValueError("an alignment needs at least one per-input value")
    return Alignment(per_input=values, mean=math.fsum(values) / len(values))


@register("Oracle", "stdout_mask")
class StdoutMaskOracle:
    """Aligns a run's stdout with the reference's under the item's timing masks, reading PASS/FAIL when asked.

    Built from the recipe section `oracle: {kind: stdout_mask, passfail: <bool>}`.
    `passfail: true` reads PASS/FAIL wherever the item's target language
    prints it; `passfail: false` scores stdout_mask alone everywhere.
    """

    name = "stdout_mask"
    capabilities = frozenset({"masks_stdout"})
    config_keys = frozenset({"passfail"})

    def __init__(self, *, passfail: bool | None = None) -> None:
        """Keep the recipe's passfail choice, which has no default."""
        if not isinstance(passfail, bool):
            raise ValueError(f"oracle.passfail is a required choice: set it to true or false, got {passfail!r}")
        self.passfail = passfail
        self.masks: tuple[str, ...] | None = None
        self.prints_passfail = False

    def for_item(self, masks: Sequence[str], *, prints_passfail: bool) -> StdoutMaskOracle:
        """Return a copy bound to one item: its masks and whether its target language prints PASS/FAIL."""
        bound = StdoutMaskOracle(passfail=self.passfail)
        bound.masks = tuple(masks)
        bound.prints_passfail = bool(prints_passfail)
        return bound

    @property
    def reads_passfail(self) -> bool:
        """Return True when PASS/FAIL is read: the recipe asks for it and the target language prints it."""
        return self.passfail and self.prints_passfail

    def align(self, reference: RunResult, candidate: RunResult) -> float:
        """Return the candidate's value in [0, 1] against the reference: 1.0 or 0.0 (see the module rules)."""
        if self.masks is None:
            raise ValueError("the stdout_mask oracle has no item masks; bind them with for_item before aligning")
        return passfail_score(reference.stdout, candidate.stdout, self.masks, reads_passfail=self.reads_passfail)
