"""`lassi score`: score a finished run tree and write its review packet (task P2.9).

Bible: Design Principles 2 (one result record), 5 (reproducible runs), and 7
(human-readable artifacts first; Parquet mirrors them); Result Record
(Storage); Readability Standards (Trial and Run rows); Agent Rules 1 and 10.

score_run(run_dir, profiles) reads the run tree the runner wrote at
`run_dir` and writes a new tree, `<runs root>/scores/<score id>/`. It never
writes under the run tree it reads: the records, the text store, and the
run's Parquet mirror stay byte for byte as the run left them, and scoring one
run twice gives the same component tables and the same review.md.

Reading the run:

- every trial.json one level per trial_id segment below `run_dir` (build
  directories are never searched), loaded with the strict loader
  lassi.core.store.read_trial, records written by earlier commits included;
  each record's trial_id must name its own directory;
- provenance.json, the run manifest, copied whole into the score's
  provenance (Agent Rule 1).

The runs root is `runs_root`, else $LASSI_RUNS_ROOT, else the run dir's
grandparent when the run dir's parent is named `runs` (the runner's layout,
<runs root>/runs/<run id>). As for a run tree, it must resolve outside the
repository and, when $LASSI_SCRATCH is set, inside it (Agent Rule 7), and
the score directory may not lie inside the run tree. The score id is
`score_id`, else the UTC time as YYYYMMDD-HHMMSS, as `lassi run` names runs;
an existing score directory is never overwritten.

Each profile name is a registered ScoreProfile, built by
lassi.scoring.profiles.build_profile. A profile that declares
reads_bench_sources gets the bench root: `bench_root`, else the runner's
default for the one suite the trials name,
lassi.bench.sources_dir($LASSI_SCRATCH, suite). A profile that declares
scores_attempts also scores each attempt (score_attempts). Every refusal
raises ScoreError before any directory exists: every trial is scored and
every output built in memory first, the Markdown and JSON as ASCII bytes
and the Parquet as Arrow tables, so a value no output can hold (a number
strict JSON cannot write, nesting too deep to encode, a lone surrogate, a
count beyond int64) is refused there. Writing the score directory is then
I/O only.

The score directory holds:

- parquet/trial_components/part-0.parquet: trial_id, profile, component,
  value (float64, null for None), and note (the Score's note on the
  component, else null); one row per trial, profile, and component in the
  Score's order, then a row `scalar` holding Score.scalar.
- parquet/attempt_components/part-0.parquet: trial_id, attempt_index,
  profile, component, and value, likewise, for each profile that declares
  scores_attempts; no file when none does.
- metrics.md and parquet/metrics, stage_reached, and corrections: when the
  profile lassi.analysis.metrics reads (SCORE_PROFILE) is among the
  profiles, the P2.8 metric tables of its Scores (metrics_markdown, and
  metrics_arrow written by write_metrics_arrow, as write_metrics_parquet
  writes them); otherwise metrics.md is one line saying no metrics were
  computed and why.
- review.md (review_markdown): the page a person reads to review the scores.
- provenance.json, written last, so a score directory without it is
  incomplete: the score id, the scoring commit and dirty flag (git, as the
  runner records them), the UTC date, the interpreter version, each
  profile's file assets/scoring/<name>.yaml with its sha256, the bench root,
  the paper values file the metrics read with its sha256, the trial count,
  and the source run: its path, a copy of its provenance.json, and its
  recipe hash.

No output holds prompt, context, source, stdout, diagnostic message, or
model text (OQ-018): review.md shows diagnostics by stage, severity, code,
and place only, and reads the text store only for the size of a reference
stdout.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from lassi.analysis.metrics import SCORE_PROFILE, MetricTable, metric_tables
from lassi.analysis.paper import PAPER_FILE
from lassi.analysis.tables import metrics_arrow, metrics_markdown, write_metrics_arrow
from lassi.bench import load_suite, sources_dir
from lassi.core.capabilities import SCORES_ATTEMPTS, declares
from lassi.core.interfaces import Score, ScoreProfile
from lassi.core.record import RUN_FLAG_NAMES, Attempt, Diagnostic, RunInfo, Trial, json_text
from lassi.core.registry import DEFAULT_REGISTRY, RegistryError
from lassi.core.store import TRIAL_JSON, TextNotFoundError, TextStore, read_trial
from lassi.core.trial_md import fmt, fmt_provenance
from lassi.executors import workdir
from lassi.scoring.profiles import INTERFACE, build_profile, reads_bench_sources
from lassi.toolchains import EnvRunner

REPO = Path(__file__).resolve().parents[2]
# Where suite manifests and score profile files live.
BENCH_DIR = REPO / "assets" / "bench"
SCORING_DIR = REPO / "assets" / "scoring"

RUNS_DIR, SCORES_DIR = "runs", "scores"
PROVENANCE_JSON, REVIEW_MD, METRICS_MD, PARQUET_DIR = "provenance.json", "review.md", "metrics.md", "parquet"
TRIAL_TABLE, ATTEMPT_TABLE = "trial_components", "attempt_components"
# The component name of the row that holds a Score's scalar.
SCALAR = "scalar"
NO_METRICS = (
    f"No metrics were computed because the {SCORE_PROFILE} profile was not requested; "
    "the run metrics read its scores.\n"
)

# A score id or suite name: one plain path segment, as the runner requires of a run id.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
# A trial directory lies one level per trial_id segment below the run directory.
_TRIAL_DEPTH = 6
# How long git may take to report the commit or the dirty flag, in seconds.
_GIT_TIMEOUT_S = 60.0
# The run manifest fields review.md shows; the whole manifest is in provenance.json.
_SOURCE_FIELDS = ("run_id", "status", "recipe", "recipe_hash", "commit", "dirty", "executor", "device", "driver")

_STRING, _DOUBLE = pa.string(), pa.float64()
TRIAL_SCHEMA = pa.schema(
    [("trial_id", _STRING), ("profile", _STRING), ("component", _STRING), ("value", _DOUBLE), ("note", _STRING)]
)
ATTEMPT_SCHEMA = pa.schema(
    [("trial_id", _STRING), ("attempt_index", pa.int64()), ("profile", _STRING), ("component", _STRING),
     ("value", _DOUBLE)]
)
# What building an output in memory raises for a value the output cannot hold.
_BUILD_ERRORS = (OSError, ValueError, OverflowError, RecursionError, UnicodeError, pa.ArrowException)


class ScoreError(RuntimeError):
    """A scoring pass that cannot run; the message names the problem. Raised before any directory is created."""


@dataclass(frozen=True)
class ScoredTrial:
    """One trial with each profile's Score of it, and each attempt's Scores for the profiles that score attempts."""

    trial: Trial
    scores: Mapping[str, Score]
    attempts: Mapping[str, Sequence[Score]]


@dataclass(frozen=True)
class _Packet:
    """Everything a scoring pass writes, built before the score directory is created: bytes and Arrow tables."""

    review: bytes
    metrics: bytes
    provenance: bytes
    components: dict[str, pa.Table]
    metric_arrow: dict[str, pa.Table] | None


# ---- the scoring pass


def score_run(
    run_dir: str | Path,
    profiles: Sequence[str],
    *,
    score_id: str | None = None,
    bench_root: str | Path | None = None,
    runs_root: str | Path | None = None,
) -> Path:
    """Score the finished run tree at `run_dir` with each named profile and return the score directory.

    The directory is `<runs root>/scores/<score id>/` (see the module
    docstring for the defaults and the files). Raises ScoreError, naming
    the problem, for an empty or unknown profile list, a run dir that does
    not exist or holds no trial.json, a record the strict loader refuses, a
    missing run manifest, a runs root it cannot find or may not use, an
    existing score directory, a bench root a profile needs but cannot get,
    a profile that cannot be built or cannot score a trial, and a value an
    output cannot hold; each comes before any directory is created.
    """
    started = datetime.now(timezone.utc).replace(microsecond=0)
    names = profile_names(profiles)
    source = _source_dir(run_dir)
    trials = load_trials(source)
    manifest = read_manifest(source)
    out = _score_dir(source, runs_root, score_id, started)
    root = find_bench_root(names, trials, bench_root)
    built = _build(names, root)
    scored = score_trials(trials, built)
    packet = _packet(scored, names, manifest, source, root, out.name, started)
    _write_packet(out, packet)
    return out


def profile_names(profiles: Sequence[str]) -> tuple[str, ...]:
    """Return the profile names in the order given; ScoreError for none, a repeat, or a name not registered."""
    if isinstance(profiles, str):
        raise ScoreError(f"profiles must be a list of names, not the string {profiles!r}")
    names = tuple(profiles)
    registered = ", ".join(DEFAULT_REGISTRY.names(INTERFACE))
    if not names:
        raise ScoreError(f"name at least one ScoreProfile (--profile); registered: {registered}")
    for position, name in enumerate(names):
        if name in names[:position]:
            raise ScoreError(f"the ScoreProfile {name!r} is named twice")
        try:
            DEFAULT_REGISTRY.get(INTERFACE, name)
        except RegistryError as error:
            raise ScoreError(str(error)) from error
    return names


def _source_dir(run_dir: str | Path) -> Path:
    """Return the run directory, resolved; ScoreError when it does not exist or is not a directory."""
    path = Path(run_dir)
    if not path.exists():
        raise ScoreError(f"the run directory {path.as_posix()} does not exist")
    if not path.is_dir():
        raise ScoreError(f"the run directory {path.as_posix()} is not a directory")
    return path.resolve()


def load_trials(run_dir: Path) -> list[Trial]:
    """Return every trial of the run tree at `run_dir`, loaded strictly and sorted by trial_id.

    Only trial.json files one level per trial_id segment below `run_dir`
    are read, so no file in a build directory is taken for a record.
    Raises ScoreError when there is none, when the strict loader refuses
    one, and when a record's trial_id does not name its own directory.
    """
    store = TextStore(run_dir)
    paths = sorted(run_dir.glob("/".join(["*"] * _TRIAL_DEPTH + [TRIAL_JSON])))
    if not paths:
        raise ScoreError(f"the run directory {run_dir.as_posix()} holds no {TRIAL_JSON} at a trial's depth")
    trials: list[Trial] = []
    for path in paths:
        where = path.as_posix()
        try:
            trial = read_trial(path, store)
        except TextNotFoundError as error:
            raise ScoreError(f"cannot load {where}: the text store holds no text {error}") from error
        except (OSError, ValueError, RecursionError) as error:
            raise ScoreError(f"cannot load {where}: {error}") from error
        expected = path.parent.relative_to(run_dir).as_posix()
        if trial.trial_id != expected:
            raise ScoreError(f"{where} holds the trial {trial.trial_id!r}, not {expected!r}, which its place names")
        trials.append(trial)
    return sorted(trials, key=lambda trial: trial.trial_id)


def read_manifest(run_dir: Path) -> dict[str, Any]:
    """Return the run's provenance.json as a dict; ScoreError when it is missing or not a JSON object.

    provenance.json of the score copies the manifest and is written as
    strict JSON (lassi.core.record json_text), so NaN, Infinity, a number
    that overflows a float (1e999), and nesting too deep to read are refused
    here. The in-memory build of the packet is the guard for anything else
    the copy cannot hold.
    """
    path = run_dir / PROVENANCE_JSON
    if not path.is_file():
        raise ScoreError(f"{path.as_posix()} is missing; every score carries its run's provenance (Agent Rule 1)")
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text, parse_constant=_refuse_constant, parse_float=_finite_float)
    except (OSError, ValueError, RecursionError) as error:
        raise ScoreError(f"cannot read {path.as_posix()}: {error}") from error
    if not isinstance(data, dict):
        raise ScoreError(f"{path.as_posix()} is not a JSON object")
    return data


def _refuse_constant(name: str) -> Any:
    """Refuse a JSON constant (NaN, Infinity, -Infinity), which strict JSON does not allow."""
    raise ValueError(f"{name} is not a strict JSON value")


def _finite_float(text: str) -> float:
    """Return a JSON number as a float; refuse one that overflows a float, which strict JSON cannot write back."""
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"the number {text} overflows a float, and strict JSON has no infinity")
    return value


# ---- where the score goes


def _score_dir(source: Path, runs_root: str | Path | None, score_id: str | None, started: datetime) -> Path:
    """Return the score directory `<runs root>/scores/<score id>`; ScoreError for a bad id or an existing one.

    A path that does not encode as UTF-8 (a lone surrogate, or on POSIX a
    byte that is not UTF-8) is refused here, since the Parquet writer
    encodes its paths as UTF-8 and would fail after the directory exists.
    """
    name = started.strftime("%Y%m%d-%H%M%S") if score_id is None else score_id
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ScoreError(f"a score id must match {_NAME.pattern}, got {name!r}")
    out = find_runs_root(source, runs_root) / SCORES_DIR / name
    try:
        str(out).encode("utf-8")
    except UnicodeEncodeError as error:
        raise ScoreError(f"the score directory {out!r} is not a UTF-8 path, which the Parquet writer needs") from error
    if out.exists():
        raise ScoreError(_exists(out))
    if _within(out, source):
        raise ScoreError(f"the score directory {out.as_posix()} lies inside the run tree it reads; choose another")
    return out


def _exists(out: Path) -> str:
    """Return the refusal for a score directory that already exists."""
    return f"the score directory {out.as_posix()} already exists; a score is never overwritten"


def find_runs_root(source: Path, runs_root: str | Path | None) -> Path:
    """Return the resolved runs root: `runs_root`, else $LASSI_RUNS_ROOT, else the run dir's grandparent.

    The grandparent counts only when the run dir's parent is named `runs`.
    Raises ScoreError when none applies, when $LASSI_RUNS_ROOT is not an
    absolute path, and when the root resolves inside the repository or,
    with $LASSI_SCRATCH set, outside it (Agent Rule 7).
    """
    if runs_root is not None:
        root, origin = Path(runs_root), "the runs root option"
    elif os.environ.get("LASSI_RUNS_ROOT"):
        try:
            root = workdir.runs_root()
        except ValueError as error:
            raise ScoreError(str(error)) from error
        origin = "$LASSI_RUNS_ROOT"
    elif source.parent.name == RUNS_DIR:
        root, origin = source.parent.parent, "the run directory's place"
    else:
        raise ScoreError(
            f"no runs root for {source.as_posix()}: its parent is not named {RUNS_DIR!r}; "
            "pass a runs root (--runs-root) or set $LASSI_RUNS_ROOT"
        )
    resolved = root.resolve()
    what = f"the runs root {root.as_posix()} (from {origin})"
    if _within(resolved, REPO):
        raise ScoreError(f"{what} is inside the repository {REPO.as_posix()}; score trees stay out of git")
    scratch = os.environ.get("LASSI_SCRATCH", "")
    if scratch and not _within(resolved, Path(scratch).resolve()):
        raise ScoreError(f"{what} is outside $LASSI_SCRATCH ({scratch}); on the build host scores stay on scratch")
    return resolved


def _within(path: Path, root: Path) -> bool:
    """Return True when the resolved `path` is `root` or lies under it."""
    path, root = path.resolve(), root.resolve()
    return path == root or root in path.parents


# ---- profiles and the bench root


def find_bench_root(names: Sequence[str], trials: Sequence[Trial], bench_root: str | Path | None) -> Path | None:
    """Return the bench root the profiles get: None when no named profile reads bench sources.

    Otherwise `bench_root`, else lassi.bench.sources_dir($LASSI_SCRATCH,
    suite) for the one suite the trials name, as the runner finds it.
    Raises ScoreError when the trials name more than one suite, when
    $LASSI_SCRATCH is not set, or when the suite has no manifest.
    """
    readers = [name for name in names if reads_bench_sources(name)]
    if not readers:
        return None
    if bench_root is not None:
        return Path(bench_root).resolve()
    who = f"the ScoreProfile(s) {', '.join(readers)} read the suite's bench sources"
    suites = sorted({trial.bench_item.suite for trial in trials})
    if len(suites) != 1:
        raise ScoreError(f"{who}, and the trials name the suites {', '.join(suites)}; pass a bench root (--bench-root)")
    suite = suites[0]
    scratch = os.environ.get("LASSI_SCRATCH", "")
    if not scratch:
        raise ScoreError(f"{who}: no bench root was given and LASSI_SCRATCH is not set; pass one (--bench-root)")
    manifest = BENCH_DIR / f"{suite}.yaml"
    if not _NAME.fullmatch(suite) or not manifest.is_file():
        raise ScoreError(f"{who}, but the suite {suite!r} has no manifest; expected {manifest.as_posix()}")
    try:
        return sources_dir(Path(scratch), load_suite(manifest)).resolve()
    except (OSError, ValueError) as error:
        raise ScoreError(f"cannot find the bench sources of {suite}: {error}") from error


def _build(names: Sequence[str], bench_root: Path | None) -> dict[str, ScoreProfile]:
    """Return each named profile built by build_profile, in order; ScoreError when one cannot be built."""
    built: dict[str, ScoreProfile] = {}
    for name in names:
        try:
            built[name] = build_profile(name, bench_root=bench_root)
        except (OSError, ValueError) as error:
            raise ScoreError(f"cannot build the ScoreProfile {name!r}: {error}") from error
    return built


# ---- scoring


def score_trials(trials: Sequence[Trial], profiles: Mapping[str, ScoreProfile]) -> list[ScoredTrial]:
    """Score every trial with every profile, in trial_id order; attempts too for profiles that score them.

    Every value becomes a float or None. Raises ScoreError when a profile
    raises OSError, ValueError, or OverflowError (a count no float holds),
    gives a value that is not a finite number or None, names a component
    `scalar`, or gives attempt Scores that do not match the trial's
    attempts one for one.
    """
    scored: list[ScoredTrial] = []
    for trial in sorted(trials, key=lambda item: item.trial_id):
        scores: dict[str, Score] = {}
        attempts: dict[str, Sequence[Score]] = {}
        for name, profile in profiles.items():
            where = f"the ScoreProfile {name!r} on {trial.trial_id}"
            try:
                scores[name] = _checked(profile.score(trial), where)
                if declares(profile, SCORES_ATTEMPTS):
                    attempts[name] = _attempt_scores(profile, trial, where)
            except (OSError, ValueError, OverflowError) as error:
                raise ScoreError(f"{where}: {error}") from error
        scored.append(ScoredTrial(trial, MappingProxyType(scores), MappingProxyType(attempts)))
    return scored


def _attempt_scores(profile: ScoreProfile, trial: Trial, where: str) -> tuple[Score, ...]:
    """Return the profile's score_attempts(trial), each Score checked; one per attempt of the trial."""
    method = getattr(profile, "score_attempts", None)
    if not callable(method):
        raise ScoreError(f"{where}: it declares {SCORES_ATTEMPTS} but has no score_attempts()")
    found = tuple(_checked(score, f"{where} attempt {index}") for index, score in enumerate(method(trial)))
    if len(found) != len(trial.attempts):
        raise ScoreError(f"{where}: {len(found)} attempt Score(s) for {len(trial.attempts)} attempt(s)")
    return found


def _checked(score: Any, where: str) -> Score:
    """Return a copy of `score` with every value a float or None; ScoreError for anything else."""
    if not isinstance(score, Score):
        raise ScoreError(f"{where}: expected a Score, got {type(score).__name__}")
    if SCALAR in score.components:
        raise ScoreError(f"{where}: a component may not be named {SCALAR!r}, the name of the scalar's row")
    components = {name: _value(value, f"{where}, {name}") for name, value in score.components.items()}
    return Score(components=components, scalar=_value(score.scalar, f"{where}, {SCALAR}"), notes=dict(score.notes))


def _value(value: Any, where: str) -> float | None:
    """Return a finite number as a float and None as None; ScoreError for a bool, NaN, infinity, or a non-number."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ScoreError(f"{where}: expected a finite number or None, got {value!r}")
    return float(value)


def entries(score: Score) -> Iterator[tuple[str, float | None]]:
    """Yield a Score's (component, value) pairs in its order, then (`scalar`, its scalar)."""
    yield from score.components.items()
    yield SCALAR, score.scalar


def note_of(score: Score, component: str) -> str | None:
    """Return the Score's note on `component`, or None when it has none (the scalar row never has one)."""
    return (score.notes.get(component) or None) if component != SCALAR else None


# ---- the packet


def _packet(
    scored: Sequence[ScoredTrial], names: Sequence[str], manifest: Mapping[str, Any], source: Path,
    bench_root: Path | None, score_id: str, started: datetime,
) -> _Packet:
    """Build every output of the score directory in memory; ScoreError, naming it, when one cannot be built.

    The Markdown and provenance.json become ASCII bytes and the Parquet
    becomes Arrow tables here, before any directory exists, so every check
    and every encoding that can fail on a value runs first: the metric
    tables' partition check, a count beyond int64 or a lone surrogate in an
    Arrow column, and a non-finite number or nesting too deep for strict
    JSON. _write_packet then does I/O only.
    """
    tables: list[MetricTable] | None = None
    metric_arrow: dict[str, pa.Table] | None = None
    if SCORE_PROFILE in names:
        with _building("the metric tables"):
            tables = metric_tables([(item.trial, item.scores[SCORE_PROFILE]) for item in scored])
            metric_arrow = metrics_arrow(tables)
    with _building("the component tables"):
        components = component_tables(scored, names)
    with _building(PROVENANCE_JSON):
        files = [profile_file(name) for name in names]
        provenance = _provenance(files, manifest, source, bench_root, score_id, started, len(scored), tables)
    with _building(REVIEW_MD):
        review = review_markdown(scored, names, manifest, files, TextStore(source)).encode("ascii")
    with _building(METRICS_MD):
        metrics = (NO_METRICS if tables is None else metrics_markdown(tables)).encode("ascii")
    return _Packet(review=review, metrics=metrics, provenance=provenance, components=components,
                   metric_arrow=metric_arrow)


@contextmanager
def _building(what: str) -> Iterator[None]:
    """Turn an error building `what` in memory (_BUILD_ERRORS) into a ScoreError naming it and the error."""
    try:
        yield
    except _BUILD_ERRORS as error:
        raise ScoreError(f"cannot build {what}: {type(error).__name__}: {error}") from error


def _provenance(
    files: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], source: Path, bench_root: Path | None,
    score_id: str, started: datetime, count: int, tables: Sequence[MetricTable] | None,
) -> bytes:
    """Return provenance.json of the score as strict ASCII JSON bytes (see the module docstring for its fields)."""
    data = {
        "score_id": score_id,
        **_git_state(),
        "date": started.isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "profiles": list(files),
        "bench_root": None if bench_root is None else bench_root.as_posix(),
        "metrics": None if tables is None else {"paper": _file_entry(PAPER_FILE)},
        "trials": count,
        "source_run": {"path": source.as_posix(), "provenance": dict(manifest),
                       "recipe_hash": manifest.get("recipe_hash")},
    }
    return json_text(data).encode("ascii")


def profile_file(name: str) -> dict[str, Any]:
    """Return the profile's name with its file, assets/scoring/<name>.yaml, and that file's sha256 (None without)."""
    return {"name": name, **_file_entry(SCORING_DIR / f"{name}.yaml")}


def _file_entry(path: Path) -> dict[str, str | None]:
    """Return a file's repository path and the sha256 of its bytes, both None when the file does not exist."""
    if not path.is_file():
        return {"file": None, "sha256": None}
    return {"file": path.relative_to(REPO).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _git_state() -> dict[str, Any]:
    """Return the repository's commit and whether `git status --porcelain` lists anything; None when git fails."""
    head, status = _git("rev-parse", "HEAD"), _git("status", "--porcelain")
    return {
        "commit": (head.strip() or None) if head is not None else None,
        "dirty": bool(status.strip()) if status is not None else None,
    }


def _git(*args: str) -> str | None:
    """Return the stdout of `git <args>` run in the repository through the audited runner, or None on failure."""
    try:
        result = EnvRunner(os.environ)(["git", *args], REPO, _GIT_TIMEOUT_S)
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def component_tables(scored: Sequence[ScoredTrial], names: Sequence[str]) -> dict[str, pa.Table]:
    """Return the trial component table and, when any profile scored attempts, the attempt component table."""
    trial_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    for item in scored:
        trial_id = item.trial.trial_id
        for name in names:
            score = item.scores[name]
            for component, value in entries(score):
                row = (trial_id, name, component, value, note_of(score, component))
                trial_rows.append(dict(zip(TRIAL_SCHEMA.names, row, strict=True)))
            if name not in item.attempts:
                continue
            for attempt, found in zip(item.trial.attempts, item.attempts[name], strict=True):
                for component, value in entries(found):
                    row = (trial_id, attempt.index, name, component, value)
                    attempt_rows.append(dict(zip(ATTEMPT_SCHEMA.names, row, strict=True)))
    tables = {TRIAL_TABLE: pa.Table.from_pylist(trial_rows, schema=TRIAL_SCHEMA)}
    if attempt_rows:
        tables[ATTEMPT_TABLE] = pa.Table.from_pylist(attempt_rows, schema=ATTEMPT_SCHEMA)
    return tables


def _write_packet(out: Path, packet: _Packet) -> None:
    """Create the score directory and write the prebuilt packet into it, provenance.json last; I/O only.

    Nothing is built or checked here (see _packet); an existing directory
    is still refused with ScoreError, for a race with another pass.
    """
    try:
        out.mkdir(parents=True)
    except FileExistsError as error:
        raise ScoreError(_exists(out)) from error
    parquet = out / PARQUET_DIR
    for name, table in packet.components.items():
        (parquet / name).mkdir(parents=True)
        pq.write_table(table, str(parquet / name / "part-0.parquet"))
    if packet.metric_arrow is not None:
        write_metrics_arrow(packet.metric_arrow, parquet)
    (out / METRICS_MD).write_bytes(packet.metrics)
    (out / REVIEW_MD).write_bytes(packet.review)
    (out / PROVENANCE_JSON).write_bytes(packet.provenance)


# ---- review.md


def review_markdown(
    scored: Sequence[ScoredTrial], names: Sequence[str], manifest: Mapping[str, Any],
    files: Sequence[Mapping[str, Any]], store: TextStore,
) -> str:
    """Return review.md: the source run, the profiles, then one section per trial in trial_id order.

    A trial's section (a level-2 heading with its trial_id) gives its facts,
    one part per attempt (`### Attempt <index>`: stage, run, alignment,
    diagnostics by stage, severity, code, and place, and the attempt Scores
    of each profile that scores attempts), every component of every profile
    with its note, and the fields the record lacks. Values read as in
    trial.md (lassi.core.trial_md.fmt). The page is plain ASCII with LF
    newlines and holds no score id, no date, and no prompt, context, source,
    stdout, diagnostic message, or model text; the store is read only for
    a reference stdout's size.
    """
    blocks = _head_blocks(len(scored), manifest, files)
    for item in scored:
        blocks += _trial_blocks(item, names, store)
    page = "\n".join(blocks)
    return page.encode("ascii", "backslashreplace").decode("ascii")


def _cell(text: str) -> str:
    """Return one Markdown table cell: pipes escaped and line breaks turned into spaces."""
    return text.replace("|", "\\|").replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Return a Markdown table block with escaped cells."""
    lines = [header, ["---"] * len(header), *rows]
    return "".join("| " + " | ".join(_cell(cell) for cell in line) + " |\n" for line in lines)


def _head_blocks(count: int, manifest: Mapping[str, Any], files: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return the page title, the source run's manifest fields, the profiles, and how to read the page."""
    source = [(key, fmt_provenance(manifest.get(key))) for key in _SOURCE_FIELDS]
    profiles = [(entry["name"], entry["file"] or "none", fmt_provenance(entry["sha256"])) for entry in files]
    legend = (
        f"Trials scored: {count}. One section per trial, in trial_id order: the trial's facts, one part per "
        "attempt, every component of every profile with its note, and the fields the record lacks. Values read "
        "as in trial.md: PLACEHOLDER is a value not recorded or not computed (null), flags are true or false, "
        "and numbers are shown in full. Diagnostics show stage, severity, code, and place; their messages, the "
        "prompts, replies, code, context, and stdout stay in the run tree (OQ-018). The scoring commit, date, "
        "and interpreter, and a copy of the run's provenance.json, are in provenance.json beside this page.\n"
    )
    return [
        "# Score review\n",
        "## Source run\n",
        _table(("Field", "Value"), source),
        "## Profiles\n",
        _table(("Profile", "File", "sha256"), profiles),
        legend,
    ]


def _trial_blocks(item: ScoredTrial, names: Sequence[str], store: TextStore) -> list[str]:
    """Return one trial's section: facts, request diagnostics, attempts, scores, and what the record lacks."""
    trial = item.trial
    blocks = [f"## Trial `{trial.trial_id}`\n", _table(("Field", "Value"), _trial_facts(trial))]
    for request in trial.requests or []:
        if request.diagnostics:
            blocks += _diagnostic_blocks(request.diagnostics, f"Request {request.index} ({request.stage})")
    for position, attempt in enumerate(trial.attempts):
        scores = {name: found[position] for name, found in item.attempts.items()}
        blocks += _attempt_blocks(attempt, scores)
    rows = []
    for name in names:
        score = item.scores[name]
        for component, value in entries(score):
            rows.append((name, component, fmt(value), note_of(score, component) or "-"))
    blocks += ["### Scores\n", _table(("Profile", "Component", "Value", "Note"), rows)]
    return [*blocks, "### Not recorded\n", *missing_lines(trial, store)]


def _trial_facts(trial: Trial) -> list[tuple[str, str]]:
    """Return the trial's fact rows: model, final block, request count, and the reference run."""
    final, reference = trial.final, trial.reference_run
    rows = [
        ("model", f"{trial.model.backend} `{trial.model.id}`"),
        ("final.stage_reached", fmt(final.stage_reached)),
        ("final.alignment", fmt(final.alignment)),
        ("final.corrections", fmt(final.corrections)),
        ("final.end_reason", "none" if final.end_reason is None else final.end_reason.code),
        ("requests", "not recorded" if trial.requests is None else str(len(trial.requests))),
    ]
    return rows + [(f"reference_run.{name}", fmt(value)) for name, value in _run_facts(reference)]


def _run_facts(run: RunInfo) -> list[tuple[str, Any]]:
    """Return a run's exit status, hang flag, and run flags as (field, value) pairs."""
    return [("exit_code", run.exit_code), ("hang", run.hang), *((name, getattr(run, name)) for name in RUN_FLAG_NAMES)]


def _attempt_blocks(attempt: Attempt, scores: Mapping[str, Score]) -> list[str]:
    """Return one attempt's part: stage, run, and alignment rows, its diagnostics, and its Scores."""
    facts = [("stage_reached", attempt.stage_reached), *((name, fmt(value)) for name, value in _run_facts(attempt.run))]
    facts.append(("alignment", fmt(attempt.alignment.mean)))
    blocks = [f"### Attempt {attempt.index}\n", _table(("Field", "Value"), facts)]
    blocks += _diagnostic_blocks(attempt.diagnostics, "Diagnostics")
    if scores:
        rows = [(name, component, fmt(value)) for name, score in scores.items() for component, value in entries(score)]
        blocks += ["#### Attempt scores\n", _table(("Profile", "Component", "Value"), rows)]
    return blocks


def _diagnostic_blocks(diagnostics: Sequence[Diagnostic], title: str) -> list[str]:
    """Return a level-4 heading and a table of stage, severity, code, and place per diagnostic; never the message."""
    if not diagnostics:
        return [f"#### {title}\n", "None.\n"]
    rows = [(item.stage, item.severity, item.code or "-", place(item)) for item in diagnostics]
    return [f"#### {title}\n", _table(("Stage", "Severity", "Code", "Place"), rows)]


def place(diagnostic: Diagnostic) -> str:
    """Return a diagnostic's place as trial.md shows it: file, file:line, or file:line:column; '-' without a file."""
    if diagnostic.file is None:
        return "-"
    text = diagnostic.file
    if diagnostic.line is not None:
        text += f":{diagnostic.line}"
        if diagnostic.column is not None:
            text += f":{diagnostic.column}"
    return text


def _ran(run: RunInfo) -> bool:
    """Return True when the record shows that the run happened: an exit status, a hang, or kept stdout."""
    return run.exit_code is not None or run.hang is True or run.stdout_ref is not None


def _unflagged(run: RunInfo) -> bool:
    """Return True when a run that happened lacks a run flag (null), as a record written before them does."""
    return _ran(run) and any(getattr(run, name) is None for name in RUN_FLAG_NAMES)


def missing_lines(trial: Trial, store: TextStore) -> list[str]:
    """Return the lines naming the fields a record lacks, or one line saying it lacks none this page reads.

    A reference run that happened without its run flags gets a line saying
    so and a line with its stdout's size in UTF-8 bytes from the text store;
    an attempt run likewise gets a line; requests of None get a line.
    """
    lines: list[str] = []
    if _unflagged(trial.reference_run):
        flags = ", ".join(RUN_FLAG_NAMES)
        lines.append(f"- Reference-run flags ({flags}): not recorded (null), as in a record written before them.\n")
        lines.append(f"- Reference stdout size: {_stdout_size(trial.reference_run, store)}.\n")
    unflagged = [str(attempt.index) for attempt in trial.attempts if _unflagged(attempt.run)]
    if unflagged:
        lines.append(f"- Run flags of attempt(s) {', '.join(unflagged)}: not recorded (null).\n")
    if trial.requests is None:
        lines.append("- Model requests: not recorded (null), as in a record written before them.\n")
    return ["".join(lines)] if lines else ["None: the record holds every field this page reads.\n"]


def _stdout_size(run: RunInfo, store: TextStore) -> str:
    """Return the size of a run's kept stdout as `<n> bytes (UTF-8, from the text store)`, or why there is none."""
    if run.stdout_ref is None:
        return "no stdout was kept"
    try:
        text = store.get(run.stdout_ref)
    except (OSError, KeyError, ValueError):
        return "the text store does not hold it"
    return f"{len(text.encode('utf-8'))} bytes (UTF-8, from the text store)"
