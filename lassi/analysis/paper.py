"""A paper's reference values for the run metrics, read from a YAML file (task P2.8).

Bible: Evaluation Protocol (LASSI Paper Metrics; Acceptance Criteria, the
B0 criterion), Source Papers (LASSI results table), Repository Layout
(`assets/scoring/`, `analysis/`).

The file, by default assets/scoring/lassi-paper.yaml, holds numbers and
citations only; this module holds no paper value. Its layout:

- `b0_model`: the model whose correct counts bound the B0 acceptance
  criterion, and `b0_cite`, the bible section that names it.
- `directions`: per direction key (as trial ids spell it, such as
  `omp-cuda`), `metrics`, keyed by the metric row names of
  lassi.analysis.metrics, each with `published`, `recount`, an optional
  `recount_alternate`, and `cite`; and `correct_by_model`, each model's
  correct count. Every count is written `<count>/<denominator>`.

A metric the file does not list for a direction is one the paper does not
report: PaperValues.metric gives None for it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from lassi.analysis.stats import wilson_interval

# The paper values file metric_tables reads when it is given none.
PAPER_FILE = Path(__file__).resolve().parents[2] / "assets" / "scoring" / "lassi-paper.yaml"

FILE_KEYS = ("b0_model", "b0_cite", "directions")
DIRECTION_KEYS = ("metrics", "correct_by_model")
METRIC_KEYS = ("published", "recount", "recount_alternate", "cite")
REQUIRED_METRIC_KEYS = ("published", "recount", "cite")
_FRACTION = re.compile(r"([0-9]+)/([0-9]+)")


@dataclass(frozen=True)
class PaperCount:
    """A count the paper gives or implies: `count` of `denominator`."""

    count: int
    denominator: int

    def fraction(self) -> str:
        """Return the count as `<count>/<denominator>`."""
        return f"{self.count}/{self.denominator}"


@dataclass(frozen=True)
class PaperMetric:
    """One metric's paper values: published, recounted from the paper's tables, an alternate recount, and a cite."""

    published: PaperCount
    recount: PaperCount
    recount_alternate: PaperCount | None
    cite: str


@dataclass(frozen=True)
class PaperValues:
    """The paper values of every direction, as load_paper reads them."""

    metrics: Mapping[str, Mapping[str, PaperMetric]]
    models: Mapping[str, Mapping[str, PaperCount]]
    b0_model: str
    b0_cite: str

    def metric(self, direction: str, name: str) -> PaperMetric | None:
        """Return the paper's values of metric row `name` in `direction`, or None when the paper reports none."""
        return self.metrics.get(direction, {}).get(name)

    def correct_by_model(self, direction: str) -> Mapping[str, PaperCount]:
        """Return each model's correct count in `direction`; raise ValueError for a direction the file lacks."""
        if direction not in self.models:
            known = ", ".join(sorted(self.models))
            raise ValueError(f"the paper values hold no direction {direction!r}; known: {known}")
        return self.models[direction]


def _text(value: Any, where: str) -> str:
    """Return `value` when it is a non-empty plain ASCII string; raise ValueError naming `where` otherwise."""
    if not isinstance(value, str) or not value.strip() or not value.isascii():
        raise ValueError(f"{where}: expected a non-empty plain ASCII text, got {value!r}")
    return value


def _count(value: Any, where: str) -> PaperCount:
    """Return the PaperCount written `<count>/<denominator>`; raise ValueError naming `where` otherwise."""
    found = _FRACTION.fullmatch(value) if isinstance(value, str) else None
    if found is None:
        raise ValueError(f"{where}: expected <count>/<denominator>, got {value!r}")
    count, denominator = int(found.group(1)), int(found.group(2))
    if denominator < 1 or count > denominator:
        raise ValueError(f"{where}: expected 0 <= count <= denominator and denominator >= 1, got {value!r}")
    return PaperCount(count=count, denominator=denominator)


def _mapping(value: Any, keys: tuple[str, ...] | None, required: tuple[str, ...], where: str) -> dict[str, Any]:
    """Return `value` as a dict; raise ValueError naming `where` when it is not one or its keys break the rules.

    `keys` lists the allowed keys (None allows any string key) and
    `required` the keys that must be present.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected a mapping, got {value!r}")
    missing = [key for key in required if key not in value]
    unknown = sorted(str(key) for key in value if not isinstance(key, str) or (keys is not None and key not in keys))
    if missing or unknown:
        raise ValueError(f"{where}: missing key(s) {missing or 'none'}, unknown key(s) {unknown or 'none'}")
    return value


def _metric(value: Any, where: str) -> PaperMetric:
    """Return one metric's PaperMetric from its mapping in the file."""
    data = _mapping(value, METRIC_KEYS, REQUIRED_METRIC_KEYS, where)
    alternate = data.get("recount_alternate")
    return PaperMetric(
        published=_count(data["published"], f"{where}.published"),
        recount=_count(data["recount"], f"{where}.recount"),
        recount_alternate=None if alternate is None else _count(alternate, f"{where}.recount_alternate"),
        cite=_text(data["cite"], f"{where}.cite"),
    )


def load_paper(path: str | Path = PAPER_FILE) -> PaperValues:
    """Return the paper values in the YAML file at `path`, by default PAPER_FILE.

    Raises ValueError naming the file and the key for a missing file, a
    file that is not plain ASCII, a missing or unknown key, a count not
    written `<count>/<denominator>` with 0 <= count <= denominator, a cite
    that is not plain ASCII text, or a b0_model missing from a direction's
    correct_by_model.
    """
    where = Path(path).as_posix()
    if not Path(path).is_file():
        raise ValueError(f"the paper values file {where} does not exist")
    raw = Path(path).read_bytes()
    if not raw.isascii():
        raise ValueError(f"{where}: the paper values file must be plain ASCII")
    data = _mapping(yaml.safe_load(raw.decode("ascii")), FILE_KEYS, FILE_KEYS, where)
    b0_model = _text(data["b0_model"], f"{where} b0_model")
    directions = _mapping(data["directions"], None, (), f"{where} directions")
    metrics: dict[str, Mapping[str, PaperMetric]] = {}
    models: dict[str, Mapping[str, PaperCount]] = {}
    for direction, value in directions.items():
        at = f"{where} directions.{direction}"
        entry = _mapping(value, DIRECTION_KEYS, DIRECTION_KEYS, at)
        rows = _mapping(entry["metrics"], None, (), f"{at}.metrics")
        metrics[direction] = MappingProxyType(
            {name: _metric(row, f"{at}.metrics.{name}") for name, row in rows.items()}
        )
        counts = _mapping(entry["correct_by_model"], None, (b0_model,), f"{at}.correct_by_model")
        models[direction] = MappingProxyType(
            {model: _count(count, f"{at}.correct_by_model.{model}") for model, count in counts.items()}
        )
    return PaperValues(
        metrics=MappingProxyType(metrics),
        models=MappingProxyType(models),
        b0_model=b0_model,
        b0_cite=_text(data["b0_cite"], f"{where} b0_cite"),
    )


def paper_interval(count: PaperCount) -> tuple[float, float]:
    """Return the Wilson 95% interval of a paper value on the paper's own count and denominator."""
    return wilson_interval(count.count, count.denominator)


def b0_interval(paper: PaperValues, direction: str) -> tuple[float, float]:
    """Return the B0 criterion's interval in `direction`: Wilson 95% around the b0_model's correct count."""
    return paper_interval(paper.correct_by_model(direction)[paper.b0_model])
