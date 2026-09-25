"""The recipe's `score` and `metrics` keys in `lassi run` (task P2.10).

Bible: Project Recipes (Notes), Component Interfaces (ScoreProfile; the
capability rule), Design Principles 1, 5, and 7, Evaluation Protocol (Run
Metrics), Readability Standards (Run row).

The runner (lassi.core.runner) calls plan_scoring before the run directory
exists, score_trial after each trial's stages, and compute_metrics,
metrics_section, and write_metrics after the trials. Nothing here names a
profile: which profile gets a bench root follows its declared capability
(lassi.scoring.profiles.build_profile), which profile gives a metric follows
the trial components it declares, and the run metric tables read the
profile that lassi.analysis.metrics names (SCORE_PROFILE).

`score: <name>` binds a registered ScoreProfile, built once for the run.
After a trial's stages, the scalar of its trial Score becomes final.score,
and a profile that declares scores_attempts also sets each Attempt.score to
that attempt's Score (components and scalar), so trial.json, trial.md, and
the Parquet mirror show them. Every Score is checked as `lassi score` checks
it (lassi.scoring.score_run.score_trials).

`metrics: [...]` names metrics, each offered by exactly one provider:

- a trial component that a registered ScoreProfile declares, once built, in
  its TRIAL_COMPONENTS attribute (the components of its trial Score, in
  order); a profile that declares none offers none;
- a row of the run metric tables (lassi.analysis.metrics.METRIC_NAMES),
  which read the Scores of SCORE_PROFILE, so they are offered only when
  that profile is registered.

Checking the names builds every registered ScoreProfile, with the run's
bench root for a profile that reads bench sources. A name no provider
offers, a name two providers offer, and a name listed twice are refused,
all before the run directory exists. After the trials, every profile that
gives a named metric scores every trial. These metric-only Scores never
touch Attempt.score or final.score. When SCORE_PROFILE gives a named metric,
the P2.8 table of every arm and direction follows
(lassi.analysis.metrics.metric_tables).

run.md's Metrics section (metrics_section) lists each named metric and its
source, then the per-trial values of the named components (PLACEHOLDER for
null) and the notes on them (one row per trial and component whose Score
carries a note), then the P2.8 tables as
lassi.analysis.tables.table_markdown renders them, each table's heading one
level down. write_metrics writes the tables' Parquet (write_metrics_parquet)
and parquet/metric_values/part-0.parquet, one row per trial and named
component with its profile, value, and note. Both read the same rows
(_value_rows), so the values and notes tables show every metric_values row
(Design Principle 7). The section holds names, values, and notes only,
never prompt, source, context, or model text (OQ-018).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from lassi.analysis.metrics import METRIC_NAMES, SCORE_PROFILE, MetricTable, metric_tables
from lassi.analysis.tables import LEGEND, table_markdown, write_metrics_parquet
from lassi.core.interfaces import Score, ScoreProfile
from lassi.core.record import ScoreBreakdown, Trial
from lassi.core.registry import Registry
from lassi.core.trial_md import fmt
from lassi.scoring.profiles import INTERFACE, build_profile
from lassi.scoring.score_run import TRIAL_SCHEMA, ScoredTrial, note_of, score_trials

# The attribute through which a built ScoreProfile declares the components of its trial Score, in order.
TRIAL_COMPONENTS = "trial_components"
# The heading of run.md's Metrics section, and the Parquet table of the named components' per-trial values.
METRICS_HEADING = "## Metrics"
VALUES_TABLE = "metric_values"
# The headings and legends of the named components' per-trial values and of the notes on them, in run.md.
VALUES_HEADING = "### Named components per trial"
NOTES_HEADING = "### Notes on the named components"
VALUES_LEGEND = (
    "One row per trial and one column per named component, each value from the trial Score of the ScoreProfile "
    "that the component's source names. PLACEHOLDER marks a null component, a value not measured or not "
    "computed; the notes below give the profile's reason where it records one.\n"
)
NOTES_LEGEND = (
    "The profile's note on each value that carries one, one row per trial and component: why a null value is "
    "null, or how a value was computed and by which interpreter. With the table above, it shows every row of "
    f"parquet/{VALUES_TABLE}, which mirrors the two.\n"
)
NO_NOTES = "No value of a named component carries a note.\n"
NOTES_HEADER = ("Trial", "Profile", "Component", "Note")


@dataclass(frozen=True)
class MetricSource:
    """Where one named metric comes from: the trial Scores of `profile`, as a component or as a run table row."""

    metric: str
    profile: str
    table_row: bool

    def describe(self) -> str:
        """Return the source as run.md names it."""
        if self.table_row:
            return f"row of the run metric tables (lassi.analysis), over the ScoreProfile {self.profile}'s scores"
        return f"trial component of the ScoreProfile {self.profile}"


@dataclass(frozen=True)
class ScoringPlan:
    """A run's scoring, every profile built: the bound profile and the source of each named metric.

    `score` and `bound` are None when the recipe binds no score. `profiles`
    holds each profile a named metric needs, by name, in the order the
    metrics first name it; it is empty when the recipe names no metric.
    """

    score: str | None = None
    bound: ScoreProfile | None = None
    metrics: tuple[MetricSource, ...] = ()
    profiles: Mapping[str, ScoreProfile] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class RunMetrics:
    """The named metrics of a finished run: their sources, every trial's metric-only Scores, and the P2.8 tables.

    `scored` is in trial_id order. `tables` is None unless SCORE_PROFILE
    gives a named metric.
    """

    sources: tuple[MetricSource, ...]
    scored: tuple[ScoredTrial, ...]
    tables: tuple[MetricTable, ...] | None


# ---------------------------------------------------------------------------
# Before the run: build the profiles and check the metrics names


def plan_scoring(data: Mapping[str, Any], registry: Registry, bench_root: Path | None) -> ScoringPlan:
    """Build the profiles the resolved recipe `data` needs and check its metrics names (see the module docstring).

    Raises ValueError (RegistryError among them) naming the problem: a
    profile that cannot be built, a declaration that is not a list of
    distinct names, and a metric that no provider or more than one offers,
    or that is named twice.
    """
    name = data.get("score")
    bound = None if name is None else _build(name, bench_root, registry, "")
    names = list(data.get("metrics") or [])
    if not names:
        return ScoringPlan(score=name, bound=bound)
    built: dict[str, ScoreProfile] = {}
    for other in registry.names(INTERFACE):
        if other == name and bound is not None:
            built[other] = bound
        else:
            built[other] = _build(other, bench_root, registry, " to read the metrics it offers")
    offered = offered_metrics(built)
    sources = tuple(_source(metric, offered, names[:index]) for index, metric in enumerate(names))
    profiles = {source.profile: built[source.profile] for source in sources}
    return ScoringPlan(score=name, bound=bound, metrics=sources, profiles=MappingProxyType(profiles))


def _build(name: str, bench_root: Path | None, registry: Registry, why: str) -> ScoreProfile:
    """Return the registered ScoreProfile `name` built by build_profile; ValueError says why it cannot be."""
    try:
        return build_profile(name, bench_root=bench_root, registry=registry)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot build the ScoreProfile {name!r}{why}: {error}") from error


def trial_components(profile: object, name: str) -> tuple[str, ...]:
    """Return the trial components the built ScoreProfile `name` declares in TRIAL_COMPONENTS; () when none.

    Raises ValueError when the declaration is not a list or tuple of
    distinct non-empty names.
    """
    declared = getattr(profile, TRIAL_COMPONENTS, ())
    if (
        not isinstance(declared, (list, tuple))
        or not all(isinstance(item, str) and item for item in declared)
        or len(set(declared)) != len(declared)
    ):
        raise ValueError(
            f"the ScoreProfile {name!r} declares {TRIAL_COMPONENTS} as {declared!r}; "
            "expected a list of distinct component names"
        )
    return tuple(declared)


def offered_metrics(profiles: Mapping[str, ScoreProfile]) -> dict[str, list[MetricSource]]:
    """Return each metric the built `profiles` offer, with every source that offers it.

    A profile offers the trial components it declares; the run metric
    tables offer their rows when SCORE_PROFILE is among `profiles`.
    """
    offered: dict[str, list[MetricSource]] = {}
    for name, profile in profiles.items():
        for component in trial_components(profile, name):
            offered.setdefault(component, []).append(MetricSource(component, name, table_row=False))
    if SCORE_PROFILE in profiles:
        for row in METRIC_NAMES:
            offered.setdefault(row, []).append(MetricSource(row, SCORE_PROFILE, table_row=True))
    return offered


def _source(metric: str, offered: Mapping[str, Sequence[MetricSource]], earlier: Sequence[str]) -> MetricSource:
    """Return the one source of `metric`; ValueError when it was named before, or when none or several offer it."""
    if metric in earlier:
        raise ValueError(f"metrics names {metric!r} twice; name each metric once")
    found = offered.get(metric, ())
    if not found:
        known = ", ".join(sorted(offered)) or "none"
        raise ValueError(
            f"metrics names {metric!r}, which no registered provider offers: it is neither a trial component that "
            f"a registered ScoreProfile declares nor a row of the run metric tables; known metrics: {known}"
        )
    if len(found) > 1:
        sources = "; ".join(source.describe() for source in found)
        raise ValueError(f"metrics names {metric!r}, which more than one provider offers ({sources}); none is picked")
    return found[0]


# ---------------------------------------------------------------------------
# During the run: the bound score of each trial


def _breakdown(score: Score) -> ScoreBreakdown:
    """Return a Score as the Result Record's ScoreBreakdown."""
    return ScoreBreakdown(components=dict(score.components), scalar=score.scalar)


def score_trial(plan: ScoringPlan, trial: Trial) -> Trial:
    """Return `trial` with the bound profile's scores: final.score, and Attempt.score for a scores_attempts profile.

    Without a bound profile the trial comes back unchanged. Raises
    ScoreError (lassi.scoring.score_run) when the profile cannot score it.
    """
    if plan.score is None or plan.bound is None:
        return trial
    (scored,) = score_trials([trial], {plan.score: plan.bound})
    attempts = trial.attempts
    if plan.score in scored.attempts:
        pairs = zip(trial.attempts, scored.attempts[plan.score], strict=True)
        attempts = [dataclasses.replace(attempt, score=_breakdown(score)) for attempt, score in pairs]
    final = dataclasses.replace(trial.final, score=scored.scores[plan.score].scalar)
    return dataclasses.replace(trial, attempts=attempts, final=final)


# ---------------------------------------------------------------------------
# After the run: the named metrics


def compute_metrics(plan: ScoringPlan, trials: Sequence[Trial]) -> RunMetrics | None:
    """Score every trial with each profile a named metric needs, and build the P2.8 tables; None without metrics.

    The Scores are the metrics' own and never enter the trials. Raises
    ScoreError when a profile cannot score a trial, and ValueError when a
    Score lacks a named component or the tables cannot be built.
    """
    if not plan.metrics:
        return None
    scored = tuple(score_trials(trials, plan.profiles))
    for item in scored:
        for source in plan.metrics:
            if not source.table_row and source.metric not in item.scores[source.profile].components:
                raise ValueError(
                    f"the ScoreProfile {source.profile!r} declares the component {source.metric!r}, but its Score "
                    f"of {item.trial.trial_id} lacks it"
                )
    tables = None
    if SCORE_PROFILE in plan.profiles:
        tables = tuple(metric_tables([(item.trial, item.scores[SCORE_PROFILE]) for item in scored]))
    return RunMetrics(sources=plan.metrics, scored=scored, tables=tables)


def _cell(text: str) -> str:
    """Return one Markdown table cell: pipes escaped and line breaks turned into spaces."""
    return " ".join(text.replace("|", "\\|").splitlines())


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Return a Markdown table block with escaped cells."""
    lines = [header, ["---"] * len(header), *rows]
    return "".join("| " + " | ".join(_cell(cell) for cell in line) + " |\n" for line in lines)


def _components(metrics: RunMetrics) -> list[MetricSource]:
    """Return the sources of the named metrics that are trial components, in the recipe's order."""
    return [source for source in metrics.sources if not source.table_row]


def _value_rows(metrics: RunMetrics) -> list[tuple[str, str, str, float | None, str | None]]:
    """Return one (trial_id, profile, component, value, note) row per trial and named component.

    The rows follow `scored` (trial_id order), each trial's components in
    the recipe's order. The note is the Score's note on the component, None
    when it has none or an empty one (note_of). parquet/metric_values holds
    these rows, and run.md's values and notes tables show every one.
    """
    rows: list[tuple[str, str, str, float | None, str | None]] = []
    components = _components(metrics)
    for item in metrics.scored:
        for source in components:
            score = item.scores[source.profile]
            note = note_of(score, source.metric)
            rows.append((item.trial.trial_id, source.profile, source.metric, score.components[source.metric], note))
    return rows


def _values_blocks(metrics: RunMetrics) -> list[str]:
    """Return the named components' per-trial values and the notes on them, each under a heading; none without any.

    The values table has one row per trial and one column per named
    component; the notes table has one row per value that carries a note,
    or NO_NOTES stands in its place when none does.
    """
    components = _components(metrics)
    if not components:
        return []
    header = ("Trial", *(source.metric for source in components))
    rows = [
        (f"`{item.trial.trial_id}`", *(fmt(item.scores[s.profile].components[s.metric]) for s in components))
        for item in metrics.scored
    ]
    notes = [
        (f"`{trial_id}`", profile, component, note)
        for trial_id, profile, component, _, note in _value_rows(metrics)
        if note is not None
    ]
    blocks = [f"{VALUES_HEADING}\n", VALUES_LEGEND, _table(header, rows), f"{NOTES_HEADING}\n"]
    return blocks + ([NOTES_LEGEND, _table(NOTES_HEADER, notes)] if notes else [NO_NOTES])


def _demoted(text: str) -> str:
    """Return a table_markdown text with its heading one level down; every other line stays as rendered."""
    head, _, rest = text.partition("\n")
    return f"#{head}\n{rest}"


def metrics_section(metrics: RunMetrics, score: str | None) -> str:
    """Return run.md's Metrics section: each named metric's source, the per-trial values and notes, the P2.8 tables.

    `score` is the ScoreProfile the recipe binds, None when it binds none.
    The text is plain ASCII with LF newlines (see the module docstring).
    """
    held = (
        f"hold only the scores of the ScoreProfile {score}, which the recipe binds"
        if score is not None
        else "stay empty, since the recipe binds no score"
    )
    intro = (
        f"The recipe's metrics line names {len(metrics.sources)} metric(s), each from the trial Scores of the "
        "ScoreProfile its source names. Those Scores are computed for the metrics only: Attempt.score and "
        f"final.score in trial.json, trial.md, and the Parquet trials tables {held}.\n"
    )
    sources = [(source.metric, source.describe()) for source in metrics.sources]
    blocks = [f"{METRICS_HEADING}\n", intro, _table(("Metric", "Source"), sources), *_values_blocks(metrics)]
    if metrics.tables is not None:
        blocks += [LEGEND + "\n", *(_demoted(table_markdown(table)) for table in metrics.tables)]
    return "\n".join(blocks).encode("ascii", "backslashreplace").decode("ascii")


def write_metrics(metrics: RunMetrics, parquet_dir: Path) -> None:
    """Write the named metrics' Parquet under `parquet_dir`: the metric_values table and the P2.8 tables.

    metric_values/part-0.parquet holds one row per trial and named
    component (_value_rows, as lassi.scoring.score_run.TRIAL_SCHEMA:
    trial_id, profile, component, value, note), written only when a named
    metric is a component; run.md's Metrics section shows the same rows.
    The P2.8 tables go through write_metrics_parquet.
    """
    rows = [dict(zip(TRIAL_SCHEMA.names, values, strict=True)) for values in _value_rows(metrics)]
    if rows:
        table_dir = Path(parquet_dir) / VALUES_TABLE
        table_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=TRIAL_SCHEMA), str(table_dir / "part-0.parquet"))
    if metrics.tables is not None:
        write_metrics_parquet(metrics.tables, Path(parquet_dir))
