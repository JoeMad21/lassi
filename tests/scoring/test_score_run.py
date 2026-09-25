"""Tests for `lassi score`: score a finished run and write the review packet (task P2.9).

Bible: Design Principles 2 (one result record), 5 (reproducible runs), and 7
(human-readable artifacts first; Parquet mirrors them); Result Record
(Storage); Readability Standards (Trial and Run rows); Agent Rules 1 and 10;
AGENTS.md, Results. OQ-018: score outputs and the review packet carry no
prompt, context, source, or model text.

The contract these tests fix (plans/p2-scoring.md, P2.9, and the task's
contract decisions):

- lassi.scoring.score_run.score_run(run_dir, profiles, *, score_id=None,
  bench_root=None, runs_root=None) -> Path loads every trial.json of the
  finished run tree at `run_dir` with the strict loader
  (lassi.core.store.read_trial, records from P1 commits included), scores
  each trial with every ScoreProfile named in `profiles` (registry names, in
  the order given), and returns the score directory
  `<runs root>/scores/<score id>/`. It raises
  lassi.scoring.score_run.ScoreError, naming the problem, for an unknown
  profile, a run dir that does not exist or holds no trial.json, a
  trial.json the strict loader refuses, an existing score directory (never
  overwritten), a runs root it cannot find, a bench root the lassi
  profile needs but cannot get, and a loadable value no output can hold (a
  manifest number that overflows a float or nesting too deep, a lone
  surrogate, a count beyond int64 or a float). Every refusal comes before
  any directory is created.
- `lassi score <run dir> --profile <name> [--profile <name> ...]
  [--score-id ID] [--bench-root P] [--runs-root P]` (lassi.cli.main) calls
  it and returns 0, or returns 2 with `lassi score: <message>` on stderr
  for each refusal (never an argparse exit).
- The runs root is --runs-root, then $LASSI_RUNS_ROOT, then the run dir's
  grandparent when the run dir's parent is named `runs`; otherwise the pass
  is refused. The default score id is the UTC time as YYYYMMDD-HHMMSS, as
  `lassi run` names runs.
- The bench root is --bench-root, else the runner's default for the suite
  the trials name: lassi.bench.sources_dir($LASSI_SCRATCH, suite). Only a
  profile whose factory takes `bench_root` gets it (lassi); df-v0 needs
  none, so a df-v0 pass runs without one.
- The score directory holds:
  - provenance.json (plain ASCII JSON): `commit` and `dirty` as the runner
    records them (git in the repository), `date` (ISO 8601, UTC, seconds),
    `python` (platform.python_version()), `profiles` (one object per
    profile, in the order given, with `name` and `sha256`, the sha256 of
    assets/scoring/<name>.yaml), and `source_run` with `path` (the run
    dir), `provenance` (a copy of the run's provenance.json), and
    `recipe_hash` (as that provenance.json records it).
  - parquet/trial_components/: a Parquet dataset (any file layout below it;
    Hive partition directories allowed) with the columns trial_id, profile,
    component, value (float64, null for None), and note (the Score's note
    on that component, else null): one row per trial, profile, and
    component, and one more per trial and profile whose component is
    `scalar`, holding Score.scalar.
  - parquet/attempt_components/: for each profile that declares the
    capability scores_attempts (df-v0), the columns trial_id,
    attempt_index, profile, component, and value, one row per attempt and
    component of score_attempts(trial), and a `scalar` row holding the
    attempt's R.
  - metrics.md and the task P2.8 metrics Parquet, written with
    lassi.analysis.tables.write_metrics_parquet(tables, <score dir>/parquet),
    when `lassi` is among the profiles: metrics.md holds
    metrics_markdown(metric_tables(pairs)) over the (trial, lassi Score)
    pairs. Otherwise metrics.md is one line saying that no metrics were
    computed because the lassi profile was not requested, and no metrics
    Parquet is written.
  - review.md, below.
- review.md is plain ASCII and holds no score id and no date, so scoring one
  run twice gives identical bytes.
  - One level-2 heading (`## ...`) per trial holds its trial_id, in
    trial_id order; the trial's section runs to the next level-2 heading.
  - One level-3 heading `### Attempt <index>` per attempt; the attempt's
    part runs to the next heading of level 3 or above. The rest of the
    trial's section is its trial part.
  - Facts are Markdown table rows. A check asks for one row that holds the
    named cells, in any order (a cell's surrounding backticks are ignored).
    Values are formatted as trial.md formats them
    (lassi.core.trial_md.fmt: None as PLACEHOLDER, bools as true or false,
    floats by repr).
  - Each attempt's part has the rows `stage_reached | <stage>`,
    `exit_code | <v>`, `hang | <v>`, and `alignment | <mean>`; per
    diagnostic, a row with its severity, its code (`-` when None), and its
    place (file, file:line, or file:line:column as trial.md shows a place;
    `-` without a file), never its message; per component of each profile
    that scores attempts, `<profile> | <component> | <value>`, and
    `<profile> | scalar | <R>`.
  - The trial part has, per profile and component,
    `<profile> | <component> | <value>` with the note as one more cell
    when the Score has one, and `<profile> | scalar | <scalar>`.
  - The trial part names what the source record lacks. For a record from
    before P2.2 (reference_run flags null) it has a line that says the
    flags are not recorded (it holds "flag" and "not recorded") and a line
    giving the reference stdout's size as "<n> bytes" (the UTF-8 bytes of
    its text-store text); for a record from before P2.1 (requests null), a
    line holding "request" and "not recorded". A current record's trial
    part has neither line.
- The source run tree is unchanged byte for byte, and two passes under two
  score ids give identical component tables and identical review.md.

The fixture run tree is hand-built with the record writers
(lassi.core.store write_trial and TextStore); no runner, model, compiler,
or program runs. It holds four SYNTHETIC trials: a current record that
corrects twice (a compile error, then a hang, then a clean run), a record
as a P1 commit wrote it (no requests, no run flags) that ends at the
correction cap after a crash, a trial that ends at the baseline with no
attempt, and a clean first try of a second arm. Every text-store text,
reply, file, context text, and diagnostic message holds a sentinel, and so
do the reference programs of the tiny bench root. Wall times are
PLACEHOLDER fixture values, and the expected scores are those the
registered profiles compute on the loaded records. No value in this module
is a measurement, and no upstream text is copied into it (OQ-018).
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import re
import subprocess
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from lassi.bench import Direction, load_suite, sources_dir
from lassi.core.interfaces import Sampling, Score
from lassi.core.parquet import write_run_parquet
from lassi.core.record import (
    Alignment,
    Attempt,
    Context,
    Diagnostic,
    EndReason,
    Final,
    ModelInfo,
    Provenance,
    Request,
    RequestMessage,
    RunInfo,
    TextRef,
    Trial,
    arm_segment,
    json_text,
    make_trial_id,
    unified_diff,
)
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.core.store import TextStore, read_trial, trial_dir, write_trial
from lassi.core.trial_md import fmt

REPO = Path(__file__).resolve().parents[2]
SCORE_MODULE = "lassi.scoring.score_run"
SCORING_ASSETS = REPO / "assets" / "scoring"
SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
PROJECT = "lassi-repro"
RUN_ID = "fixture-run"
SCORE_ID = "fixture-score"
PROFILES = ("df-v0", "lassi")
CUDA_TO_OMP = Direction("cuda", "omp")
OMP_TO_CUDA = Direction("omp", "cuda")
MODEL_A = "fixture-org/fixture-model"
# Its arm sorts before MODEL_A's by code point but after it without letter case, as a directory walk may list it.
MODEL_B = "Fixture-Upper/model-b"
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
# A commit id for synthetic provenance; not a commit of this repository.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
DEVICE = "hand-built fixture (SYNTHETIC)"
STARTED = "2026-09-24T00:00:00+00:00"
FINISHED = "2026-09-24T00:10:00+00:00"
PLACEHOLDER_WALL_S = 1.5
SIGNAL_EXIT = 139  # a death by signal 11 in the shell's form (128 + 11)
FLAGS = ("stdout_truncated", "stderr_truncated", "workdir_incomplete")
TRIAL_COLUMNS = ("trial_id", "profile", "component", "value", "note")
ATTEMPT_COLUMNS = ("trial_id", "attempt_index", "profile", "component", "value")
METRICS_TABLES = ("metrics", "stage_reached", "corrections")

# A marker for text that must never reach a score output (OQ-018); it holds non-ASCII on purpose.
SENTINEL_CORE = "SENTINEL-p29-b7d4"
SENTINEL = SENTINEL_CORE + "-" + chr(0xE9) + chr(0xFC) + "-model-source-text"

RECIPE_TEXT = (
    "# SYNTHETIC resolved recipe of a hand-built run tree (task P2.9 tests)\n"
    f"project: {PROJECT}\n"
    "bench:\n"
    f"  suite: {SUITE}\n"
    "  split: eval\n"
)
RECIPE_HASH = hashlib.sha256(RECIPE_TEXT.encode("ascii")).hexdigest()
# The reference stdout of the P1 record: its UTF-8 size in bytes differs from its length in characters.
P1_REFERENCE_STDOUT = f"SYNTHETIC reference stdout of the P1 record {SENTINEL}\n" + "fixture line\n" * 7


# ---------------------------------------------------------------------------
# The module under test and the pieces the expected values come from


def score_module() -> ModuleType:
    """Import lassi.scoring.score_run (inside each test, so a missing module fails that test)."""
    return importlib.import_module(SCORE_MODULE)


def analysis(name: str) -> ModuleType:
    """Import a lassi.analysis module; fail the test clearly while task P2.8 has not added it."""
    try:
        return importlib.import_module(f"lassi.analysis.{name}")
    except ModuleNotFoundError as error:
        pytest.fail(f"task P2.8 adds lassi.analysis.{name}, which lassi score uses for metrics: {error}")


def built_profiles(names: Sequence[str], bench_root: Path | None) -> dict[str, Any]:
    """Return each registered profile built as a scoring pass builds it: bench_root only where the factory takes it."""
    importlib.import_module("lassi.scoring")
    profiles: dict[str, Any] = {}
    for name in names:
        factory = DEFAULT_REGISTRY.get("ScoreProfile", name).factory
        takes_root = "bench_root" in inspect.signature(factory).parameters
        profiles[name] = factory(bench_root=bench_root) if takes_root else factory()
    return profiles


def git(*args: str) -> str:
    """Run git with `args` in the repository and return its stripped stdout (git only reads here)."""
    done = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True)
    return done.stdout.strip()


# ---------------------------------------------------------------------------
# The SYNTHETIC run tree


SUITE_SPEC = load_suite(SUITE_MANIFEST)


def target_name(item: str, direction: Direction) -> str:
    """Return the item's one file name in the direction's target language."""
    return SUITE_SPEC.items[item].languages[direction.target].files[0]


def write_bench(root: Path) -> Path:
    """Write SYNTHETIC reference programs where the manifest lays out bsearch and layout; return `root`."""
    for item in ("bsearch", "layout"):
        for language, spec in SUITE_SPEC.items[item].languages.items():
            path = root / spec.dir / spec.files[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            body = f"// SYNTHETIC {language} reference of the {item} fixture {SENTINEL}\n"
            body += f"#include <cstdio>\nint main() {{ puts(\"fixture {item}\"); return 0; }}\n"
            path.write_bytes(body.encode("utf-8"))
    return root


def stored(store: TextStore, label: str) -> TextRef:
    """Store a SYNTHETIC text named by `label` and holding the sentinel; return its reference."""
    return store.put(f"SYNTHETIC {label} {SENTINEL}\n")


def candidate(label: str, value: int) -> str:
    """Return a SYNTHETIC candidate program holding the sentinel."""
    return f"// SYNTHETIC candidate {label} {SENTINEL}\nint main() {{ return {value}; }}\n"


def diag(stage: str, severity: str, code: str | None, file: str | None = None, line: int | None = None,
         column: int | None = None) -> Diagnostic:
    """Return a SYNTHETIC Diagnostic whose message echoes source text (the sentinel)."""
    return Diagnostic(stage=stage, severity=severity, code=code, file=file, line=line, column=column,
                      message=f"SYNTHETIC {severity} echoing a source line: {SENTINEL}")


def ran(store: TextStore, label: str, exit_code: int | None, hang: bool, *, flags: bool | None = False) -> RunInfo:
    """Return a SYNTHETIC run that kept stdout; `flags` None leaves the run flags unrecorded, as P1 did."""
    return RunInfo(exit_code=exit_code, hang=hang, wall_s=PLACEHOLDER_WALL_S, stdout_ref=stored(store, label),
                   **dict.fromkeys(FLAGS, flags))


def attempt(store: TextStore, index: int, stage: str, files: dict[str, str], previous: dict[str, str],
            diagnostics: Sequence[Diagnostic], run: RunInfo | None = None, mean: float | None = None) -> Attempt:
    """Return a SYNTHETIC attempt: its prompt and reply hold the sentinel, and so do its files."""
    return Attempt(
        index=index, prompt_ref=stored(store, f"user prompt of attempt {index}"),
        response_text=f"SYNTHETIC reply of attempt {index} {SENTINEL}", files=files,
        diff_from_previous=unified_diff(previous, files), stage_reached=stage, diagnostics=list(diagnostics),
        run=RunInfo() if run is None else run,
        alignment=Alignment() if mean is None else Alignment(per_input=[mean], mean=mean),
    )


def request(store: TextStore, index: int, stage: str, attempt_index: int | None, user: TextRef,
            reply: str) -> Request:
    """Return a SYNTHETIC model request: a system and a user message, then its reply, all in the store."""
    messages = [RequestMessage(role="system", ref=stored(store, f"system prompt of {stage}")),
                RequestMessage(role="user", ref=user)]
    return Request(index=index, stage=stage, attempt_index=attempt_index, messages=messages,
                   reply_ref=store.put(reply))


def reference(store: TextStore, exit_code: int, text: str | None = None, *, flags: bool | None = False) -> RunInfo:
    """Return the target reference's SYNTHETIC baseline run."""
    ref = store.put(text) if text is not None else stored(store, "reference stdout")
    return RunInfo(exit_code=exit_code, hang=False, wall_s=PLACEHOLDER_WALL_S, stdout_ref=ref,
                   **dict.fromkeys(FLAGS, flags))


def sentinel_context() -> Context:
    """Return a Trial.context whose texts hold the sentinel."""
    return Context(knowledge_summary=f"SYNTHETIC summary {SENTINEL}",
                   source_description=f"SYNTHETIC description {SENTINEL}")


def make_trial(model_id: str, direction: Direction, item: str, attempts: list[Attempt], *, ref_run: RunInfo,
               requests: list[Request] | None, context: Context, end: EndReason | None,
               alignment: float | None) -> Trial:
    """Return a SYNTHETIC trial with the final block the runner writes (alignment as given)."""
    return Trial(
        trial_id=make_trial_id(PROJECT, arm_segment(model_id), SUITE, direction.name, item, 1),
        recipe_hash=RECIPE_HASH,
        provenance=Provenance(commit=FAKE_COMMIT, dirty=False, device=DEVICE, sdk=None, date=STARTED),
        bench_item=SUITE_SPEC.bench_item(item, direction),
        model=ModelInfo(backend="mock", id=model_id, sampling=SAMPLING),
        reference_run=ref_run, context=context, requests=requests, attempts=attempts,
        final=Final(stage_reached=attempts[-1].stage_reached if attempts else None, alignment=alignment,
                    corrections=max(len(attempts) - 1, 0), wall_s=PLACEHOLDER_WALL_S, end_reason=end),
    )


def corrected_trial(store: TextStore) -> Trial:
    """A current record: a compile error, a hang, then a clean run the oracle aligned at 1.0, with every request."""
    name = target_name("bsearch", CUDA_TO_OMP)
    files = [{name: candidate(f"attempt {index}", index)} for index in range(3)]
    first = attempt(store, 0, "S1", files[0], {}, [
        diag("parse", "warning", "fence-quirk"), diag("compile", "error", "fixture-error", name, 12, 5)])
    hung = attempt(store, 1, "S4", files[1], files[0], [
        diag("compile", "warning", "fixture-warning", name, 7), diag("run", "error", "run-error")],
        run=ran(store, "stdout of the hung run", None, True), mean=0.0)
    clean = attempt(store, 2, "S5", files[2], files[1], [
        diag("compile", "warning", "fixture-warning", name, 7, 3), diag("run", "note", None)],
        run=ran(store, "stdout of the clean run", 0, False), mean=1.0)
    context = sentinel_context()
    requests = [
        request(store, 0, "summarize_context", None, stored(store, "context prompt 0"), context.knowledge_summary),
        request(store, 1, "describe_source", None, stored(store, "context prompt 1"), context.source_description),
    ]
    stages = ("generate", "compile_loop", "run_loop")
    for index, (stage, made) in enumerate(zip(stages, (first, hung, clean), strict=True)):
        requests.append(request(store, 2 + index, stage, index, made.prompt_ref, made.response_text))
    return make_trial(MODEL_A, CUDA_TO_OMP, "bsearch", [first, hung, clean], ref_run=reference(store, 0),
                      requests=requests, context=context, end=None, alignment=1.0)


def p1_trial(store: TextStore) -> Trial:
    """A record as a P1 commit wrote it: no requests, no run flags, no final alignment; a crash at the cap."""
    name = target_name("layout", CUDA_TO_OMP)
    files = [{name: candidate(f"P1 attempt {index}", index)} for index in range(2)]
    first = attempt(store, 0, "S1", files[0], {}, [diag("compile", "error", "fixture-error", name, 3, 1)])
    crashed = attempt(store, 1, "S4", files[1], files[0], [diag("run", "error", "run-error")],
                      run=ran(store, "stdout of the crashed run", SIGNAL_EXIT, False, flags=None))
    end = EndReason(code="correction-cap", message="SYNTHETIC: a run error remained at the correction cap")
    return make_trial(MODEL_A, CUDA_TO_OMP, "layout", [first, crashed],
                      ref_run=reference(store, 0, P1_REFERENCE_STDOUT, flags=None), requests=None,
                      context=sentinel_context(), end=end, alignment=None)


def baseline_trial(store: TextStore) -> Trial:
    """A current record that ended at the baseline: the reference run exited 1, so no model call and no attempt."""
    end = EndReason(code="baseline-run", message="SYNTHETIC: the target reference run exited 1")
    return make_trial(MODEL_A, OMP_TO_CUDA, "layout", [], ref_run=reference(store, 1), requests=[],
                      context=Context(), end=end, alignment=None)


def first_try_trial(store: TextStore) -> Trial:
    """A current record of the second arm: a clean run on the first try, aligned at 1.0."""
    name = target_name("bsearch", CUDA_TO_OMP)
    only = attempt(store, 0, "S5", {name: candidate("of arm b", 0)}, {}, [],
                   run=ran(store, "stdout of arm b", 0, False), mean=1.0)
    requests = [request(store, 0, "generate", 0, only.prompt_ref, only.response_text)]
    return make_trial(MODEL_B, CUDA_TO_OMP, "bsearch", [only], ref_run=reference(store, 0), requests=requests,
                      context=Context(), end=None, alignment=1.0)


@dataclass(frozen=True)
class RunTree:
    """A hand-built run tree: where it lies, its bench root, and its trial ids (sorted)."""

    runs_root: Path
    run_dir: Path
    bench_root: Path
    trial_ids: tuple[str, ...]
    p1_trial: str
    current_trials: tuple[str, ...]


def run_manifest() -> dict[str, Any]:
    """Return the run's provenance.json as the runner writes it at the end of a run."""
    return {
        "run_id": RUN_ID, "status": "complete", "recipe": "fixture", "recipe_path": "fixture/recipe.yaml",
        "recipe_chain": ["fixture/recipe.yaml"], "recipe_hash": RECIPE_HASH, "commit": FAKE_COMMIT,
        "dirty": False, "host": "fixture-host", "platform": "SYNTHETIC", "python": "3.10.0",
        "executor": "native", "device": DEVICE, "driver": None, "started_utc": STARTED,
        "finished_utc": FINISHED, "pins": {},
    }


def strip_to_p1(json_path: Path) -> None:
    """Rewrite a trial.json as a P1 commit wrote it: no requests key and no run flag keys (P2.1, P2.2)."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    del data["requests"]
    for run in [data["reference_run"], *(item["run"] for item in data["attempts"])]:
        for flag in FLAGS:
            del run[flag]
    json_path.write_bytes(json_text(data).encode("ascii"))


def build_run(run_dir: Path, bench_root: Path) -> RunTree:
    """Write the SYNTHETIC run tree at `run_dir` and the bench root; return where they lie."""
    store = TextStore(run_dir)
    trials = [corrected_trial(store), p1_trial(store), baseline_trial(store), first_try_trial(store)]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "recipe.resolved.yaml").write_bytes(RECIPE_TEXT.encode("ascii"))
    (run_dir / "toolchains.json").write_bytes(b"{}\n")
    (run_dir / "provenance.json").write_bytes(json_text(run_manifest()).encode("ascii"))
    (run_dir / "run.md").write_bytes(f"# Run {RUN_ID}\n\nSYNTHETIC hand-built run tree.\n".encode("ascii"))
    for trial in trials:
        write_trial(trial, run_dir, store)
    strip_to_p1(trial_dir(run_dir, trials[1].trial_id) / "trial.json")
    write_run_parquet(trials, run_dir / "parquet")
    build = trial_dir(run_dir, trials[0].trial_id) / "attempt01" / "build"
    build.mkdir(parents=True)
    (build / "main.cpp").write_bytes(candidate("left in a build dir", 1).encode("utf-8"))
    ids = [trial.trial_id for trial in trials]
    return RunTree(runs_root=run_dir.parent.parent, run_dir=run_dir, bench_root=write_bench(bench_root),
                   trial_ids=tuple(sorted(ids)), p1_trial=ids[1], current_trials=(ids[0], ids[2], ids[3]))


@pytest.fixture(autouse=True)
def no_run_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate's runs and scratch roots, so each test sets what it needs."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def tree(tmp_path: Path) -> RunTree:
    """Return the SYNTHETIC run tree at <tmp>/runs-root/runs/fixture-run with its bench root."""
    return build_run(tmp_path / "runs-root" / "runs" / RUN_ID, tmp_path / "bench")


def loaded(run_dir: Path) -> list[Trial]:
    """Return every trial of the run tree, loaded strictly and sorted by trial_id."""
    store = TextStore(run_dir)
    return sorted((read_trial(path, store) for path in run_dir.rglob("trial.json")), key=lambda t: t.trial_id)


def score(tree: RunTree, profiles: Sequence[str] = PROFILES, *, score_id: str | None = SCORE_ID,
          with_bench: bool = True, runs_root: Path | None = None) -> Path:
    """Score the fixture run with score_run and return the score directory it names."""
    bench_root = tree.bench_root if with_bench else None
    options = {"score_id": score_id, "bench_root": bench_root, "runs_root": runs_root}
    return Path(score_module().score_run(tree.run_dir, list(profiles), **options))


def snapshot(root: Path) -> dict[str, str]:
    """Return every path under `root` with the sha256 of its bytes ('dir' for a directory)."""
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "dir"
        for path in sorted(root.rglob("*"))
    }


# ---------------------------------------------------------------------------
# Reading the Parquet tables


def read_dataset(path: Path, columns: Sequence[str]) -> list[dict[str, Any]]:
    """Return the named columns of every row of the Parquet dataset at `path`; no rows when it holds no file."""
    if not path.is_dir() or not any(path.rglob("*.parquet")):
        return []
    table = ds.dataset(str(path), format="parquet", partitioning="hive").to_table()
    missing = [name for name in columns if name not in table.column_names]
    assert not missing, f"{path.name} lacks the columns {missing}; it has {table.column_names}"
    return [{name: row[name] for name in columns} for row in table.to_pylist()]


def trial_rows(score_dir: Path) -> dict[tuple[str, str, str], tuple[float | None, str | None]]:
    """Return the trial component table keyed by (trial_id, profile, component), each once."""
    keyed: dict[tuple[str, str, str], tuple[float | None, str | None]] = {}
    for row in read_dataset(score_dir / "parquet" / "trial_components", TRIAL_COLUMNS):
        key = (row["trial_id"], row["profile"], row["component"])
        assert key not in keyed, f"one row per trial, profile, and component; {key} repeats"
        keyed[key] = (row["value"], row["note"] or None)
    return keyed


def attempt_rows(score_dir: Path) -> dict[tuple[str, int, str, str], float | None]:
    """Return the attempt component table keyed by (trial_id, attempt_index, profile, component), each once."""
    keyed: dict[tuple[str, int, str, str], float | None] = {}
    for row in read_dataset(score_dir / "parquet" / "attempt_components", ATTEMPT_COLUMNS):
        key = (row["trial_id"], int(row["attempt_index"]), row["profile"], row["component"])
        assert key not in keyed, f"one row per trial, attempt, profile, and component; {key} repeats"
        keyed[key] = row["value"]
    return keyed


def expected_trial_rows(trials: Sequence[Trial], profiles: dict[str, Any]) -> dict[tuple[str, str, str], Any]:
    """Return what the trial table must hold: every component with its note, and the scalar, per trial and profile."""
    rows: dict[tuple[str, str, str], Any] = {}
    for trial in trials:
        for name, profile in profiles.items():
            found = profile.score(trial)
            for component, value in found.components.items():
                rows[(trial.trial_id, name, component)] = (value, found.notes.get(component) or None)
            rows[(trial.trial_id, name, "scalar")] = (found.scalar, None)
    return rows


def attempt_scores(trial: Trial, profiles: dict[str, Any]) -> dict[str, list[Score]]:
    """Return score_attempts(trial) of each profile that declares scores_attempts."""
    return {name: list(profile.score_attempts(trial)) for name, profile in profiles.items()
            if "scores_attempts" in profile.capabilities}


def expected_attempt_rows(trials: Sequence[Trial], profiles: dict[str, Any]) -> dict[tuple[str, int, str, str], Any]:
    """Return what the attempt table must hold: every attempt component and the attempt's scalar (R)."""
    rows: dict[tuple[str, int, str, str], Any] = {}
    for trial in trials:
        for name, scores in attempt_scores(trial, profiles).items():
            for index, found in enumerate(scores):
                for component, value in found.components.items():
                    rows[(trial.trial_id, index, name, component)] = value
                rows[(trial.trial_id, index, name, "scalar")] = found.scalar
    return rows


# ---------------------------------------------------------------------------
# Reading review.md


def table_cells(text: str) -> list[list[str]]:
    """Return the cells of every Markdown table line of `text`: split at unescaped pipes, stripped, backticks off."""
    rows = []
    for line in text.splitlines():
        body = line.strip()
        if len(body) > 1 and body.startswith("|") and body.endswith("|"):
            rows.append([cell.strip().strip("`").strip() for cell in re.split(r"(?<!\\)\|", body[1:-1])])
    return rows


def has_row(text: str, *cells: str) -> bool:
    """Return True when one Markdown table row of `text` holds every cell given (any order, repeats counted)."""
    need = Counter(cells)
    return any(not need - Counter(row) for row in table_cells(text))


def require_row(text: str, where: str, *cells: str) -> None:
    """Assert that one table row of `text` holds `cells`, naming `where` when none does."""
    assert has_row(text, *cells), f"{where}: no review.md table row holds the cells {list(cells)}"


def trial_sections(review: str, trial_ids: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    """Return the trial ids in the order of their level-2 headings, and each trial's section text."""
    lines = review.splitlines()
    starts: list[tuple[int, str | None]] = []
    for number, line in enumerate(lines):
        if re.match(r"##\s", line):
            held = [trial_id for trial_id in trial_ids if trial_id in line]
            starts.append((number, held[0] if len(held) == 1 else None))
    order = [trial_id for _, trial_id in starts if trial_id is not None]
    sections: dict[str, str] = {}
    for position, (start, trial_id) in enumerate(starts):
        if trial_id is None:
            continue
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        sections[trial_id] = "\n".join(lines[start:end])
    return order, sections


def split_attempts(section: str) -> tuple[dict[int, str], str]:
    """Return each attempt's part of a trial section, and the trial part (the rest of the section).

    An attempt's part runs from its `### Attempt <n>` heading to the next
    heading of level 3 or above.
    """
    parts: dict[int, list[str]] = {}
    rest: list[str] = []
    current: int | None = None
    for line in section.splitlines():
        found = re.match(r"###\s+Attempt\s+(\d+)\b", line)
        if found:
            current = int(found.group(1))
            assert current not in parts, f"attempt {current} has two headings"
            parts[current] = [line]
            continue
        if re.match(r"#{1,3}\s", line):
            current = None
        (rest if current is None else parts[current]).append(line)
    return {index: "\n".join(lines) for index, lines in parts.items()}, "\n".join(rest)


def place(item: Diagnostic) -> str:
    """Return a diagnostic's place as trial.md shows it: file, file:line, or file:line:column; '-' without a file."""
    if item.file is None:
        return "-"
    text = item.file
    if item.line is not None:
        text += f":{item.line}"
        if item.column is not None:
            text += f":{item.column}"
    return text


def review_of(score_dir: Path) -> str:
    """Return review.md, checked to be plain ASCII."""
    raw = (score_dir / "review.md").read_bytes()
    assert raw.isascii(), "review.md is plain ASCII"
    return raw.decode("ascii")


def sections_of(tree: RunTree, score_dir: Path) -> dict[str, str]:
    """Return each trial's review.md section, checking there is exactly one per trial."""
    order, sections = trial_sections(review_of(score_dir), tree.trial_ids)
    assert sorted(order) == sorted(tree.trial_ids) and len(order) == len(set(order)), (
        f"one level-2 heading per trial holds its trial_id; found {order}")
    return sections


# ---------------------------------------------------------------------------
# The fixture itself (guards the tests below)


def test_the_fixture_run_tree_carries_the_sentinel_and_a_p1_record(tree: RunTree) -> None:
    blobs = sorted((tree.run_dir / "texts").rglob("*.txt"))
    assert blobs and all(SENTINEL in path.read_bytes().decode("utf-8") for path in blobs)
    trials = {trial.trial_id: trial for trial in loaded(tree.run_dir)}
    assert sorted(trials) == list(tree.trial_ids), "every record loads with the strict loader"
    for trial in trials.values():
        for made in trial.attempts:
            assert SENTINEL in made.response_text and SENTINEL in made.diff_from_previous
            assert all(SENTINEL in text for text in made.files.values())
            assert all(SENTINEL in item.message for item in made.diagnostics)
    p1 = json.loads((trial_dir(tree.run_dir, tree.p1_trial) / "trial.json").read_text(encoding="utf-8"))
    assert "requests" not in p1 and not set(FLAGS) & set(p1["reference_run"])
    assert trials[tree.p1_trial].requests is None and trials[tree.p1_trial].reference_run.stdout_truncated is None
    for trial_id in tree.current_trials:
        assert trials[trial_id].requests is not None and trials[trial_id].reference_run.stdout_truncated is False
    assert len(P1_REFERENCE_STDOUT.encode("utf-8")) != len(P1_REFERENCE_STDOUT) and not SENTINEL.isascii()
    walked = [path.parent.relative_to(tree.run_dir).as_posix() for path in tree.run_dir.rglob("trial.json")]
    assert sorted(set(walked)) == sorted(tree.trial_ids)


# ---------------------------------------------------------------------------
# The score directory


def test_score_run_writes_the_score_directory_under_the_runs_root(tree: RunTree) -> None:
    out = score(tree)
    assert out.resolve() == (tree.runs_root / "scores" / SCORE_ID).resolve()
    for name in ("provenance.json", "review.md", "metrics.md"):
        assert (out / name).is_file(), f"the score directory holds {name}"
    for name in ("trial_components", "attempt_components", *METRICS_TABLES):
        assert any((out / "parquet" / name).rglob("*.parquet")), f"the score directory holds parquet/{name}/"


def test_provenance_records_the_scoring_pass_its_profiles_and_the_source_run(tree: RunTree) -> None:
    before = datetime.now(timezone.utc).replace(microsecond=0)
    out = score(tree)
    after = datetime.now(timezone.utc)
    raw = (out / "provenance.json").read_bytes()
    assert raw.isascii(), "provenance.json is plain ASCII"
    data = json.loads(raw)
    assert data["commit"] == git("rev-parse", "HEAD"), "the scoring commit"
    assert data["dirty"] is bool(git("status", "--porcelain")), "a pass from a dirty tree is exploratory"
    moment = datetime.fromisoformat(data["date"])
    assert moment.utcoffset() == timedelta(0) and before <= moment <= after, data["date"]
    assert data["python"] == platform.python_version()
    assert [entry["name"] for entry in data["profiles"]] == list(PROFILES), "profiles in the order given"
    for entry in data["profiles"]:
        asset = SCORING_ASSETS / f"{entry['name']}.yaml"
        assert entry["sha256"] == hashlib.sha256(asset.read_bytes()).hexdigest(), entry["name"]
    source = data["source_run"]
    assert Path(source["path"]).resolve() == tree.run_dir.resolve()
    manifest = json.loads((tree.run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert source["provenance"] == manifest, "a copy of the source run's provenance.json"
    assert source["recipe_hash"] == manifest["recipe_hash"] == RECIPE_HASH


def test_the_trial_table_holds_every_component_of_every_profile(tree: RunTree) -> None:
    out = score(tree)
    trials = loaded(tree.run_dir)
    want = expected_trial_rows(trials, built_profiles(PROFILES, tree.bench_root))
    got = trial_rows(out)
    assert sorted(got) == sorted(want), "one row per trial, profile, and component, plus each scalar"
    wrong = {key: (got[key], value) for key, value in want.items() if got[key] != value}
    assert not wrong, f"(got, expected) values and notes: {wrong}"


def test_the_attempt_table_holds_the_attempt_scores_of_profiles_that_score_attempts(tree: RunTree) -> None:
    out = score(tree)
    trials = loaded(tree.run_dir)
    profiles = built_profiles(PROFILES, tree.bench_root)
    assert "scores_attempts" in profiles["df-v0"].capabilities
    assert "scores_attempts" not in profiles["lassi"].capabilities
    want = expected_attempt_rows(trials, profiles)
    got = attempt_rows(out)
    assert {key[2] for key in got} == {"df-v0"}, "only df-v0 scores attempts"
    assert sorted(got) == sorted(want)
    wrong = {key: (got[key], value) for key, value in want.items() if got[key] != value}
    assert not wrong, f"(got, expected): {wrong}"


def test_null_values_stay_null_and_every_value_is_a_float(tree: RunTree) -> None:
    out = score(tree)
    baseline = next(trial for trial in loaded(tree.run_dir) if not trial.attempts)
    got = trial_rows(out)
    assert got[(baseline.trial_id, "df-v0", "multi_turn")] == (None, None), "a trial with no attempt has no R"
    value, note = got[(tree.trial_ids[0], "lassi", "correct_paper")]
    assert value is None and note, "the paper criterion is null with its note"
    values = [value for value, _ in got.values()] + list(attempt_rows(out).values())
    assert all(value is None or (type(value) is float and math.isfinite(value)) for value in values)
    for name in ("trial_components", "attempt_components"):
        dataset = ds.dataset(str(out / "parquet" / name), format="parquet", partitioning="hive")
        assert dataset.schema.field("value").type == pa.float64(), f"{name}.value is a float64 column"


def test_metrics_are_the_p2_8_tables_of_the_lassi_scores(tree: RunTree, tmp_path: Path) -> None:
    out = score(tree)
    metrics, tables = analysis("metrics"), analysis("tables")
    lassi = built_profiles(["lassi"], tree.bench_root)["lassi"]
    expected = metrics.metric_tables([(trial, lassi.score(trial)) for trial in loaded(tree.run_dir)])
    text = (out / "metrics.md").read_bytes().decode("ascii")
    assert tables.metrics_markdown(expected) in text, "metrics.md holds the P2.8 Markdown of the lassi scores"
    tables.write_metrics_parquet(expected, tmp_path / "expected")
    assert tables.read_metrics_parquet(out / "parquet") == tables.read_metrics_parquet(tmp_path / "expected")


def test_without_the_lassi_profile_no_bench_root_is_needed_and_metrics_md_says_why(tree: RunTree) -> None:
    out = score(tree, ("df-v0",), with_bench=False)
    lines = [line for line in (out / "metrics.md").read_bytes().decode("ascii").splitlines() if line.strip()]
    assert len(lines) == 1, f"metrics.md is one line, got {lines}"
    line = lines[0].lower()
    assert "no metrics" in line and "lassi" in line and "not requested" in line, lines[0]
    for name in METRICS_TABLES:
        assert not (out / "parquet" / name).exists(), f"no metrics Parquet ({name}) without the lassi profile"
    assert {key[1] for key in trial_rows(out)} == {"df-v0"}


def test_the_default_score_id_is_the_utc_time(tree: RunTree) -> None:
    before = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = score(tree, ("df-v0",), score_id=None, with_bench=False)
    after = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    assert re.fullmatch(r"\d{8}-\d{6}", out.name) and before <= out.name <= after, out.name
    assert out.parent.resolve() == (tree.runs_root / "scores").resolve()


# ---------------------------------------------------------------------------
# Where the score directory goes and which bench root the lassi profile reads


RUNS_ROOT_CASES = ("option", "environment", "grandparent")


@pytest.mark.parametrize("case", RUNS_ROOT_CASES)
def test_the_runs_root_is_the_option_then_the_environment_then_the_grandparent(
    case: str, tree: RunTree, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = {"option": tmp_path / "option-root", "environment": tmp_path / "env-root",
             "grandparent": tree.runs_root}
    if case != "grandparent":
        monkeypatch.setenv("LASSI_RUNS_ROOT", str(roots["environment"]))
    option = roots["option"] if case == "option" else None
    out = score(tree, ("df-v0",), score_id="where", with_bench=False, runs_root=option)
    assert out.resolve() == (roots[case] / "scores" / "where").resolve()
    assert (out / "review.md").is_file()
    for other, root in roots.items():
        if other != case:
            assert not (root / "scores").exists(), f"nothing is written under the {other} root"


def test_a_run_dir_outside_a_runs_directory_needs_a_runs_root(tree: RunTree, tmp_path: Path) -> None:
    loose = build_run(tmp_path / "loose" / RUN_ID, tmp_path / "loose-bench")
    before = snapshot(tmp_path)
    with pytest.raises(score_module().ScoreError, match=r"(?i)runs[ _-]root"):
        score(loose, ("df-v0",), with_bench=False)
    assert snapshot(tmp_path) == before, "a refusal creates nothing"


def test_the_default_bench_root_is_the_runners_for_the_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = tmp_path / "scratch"
    bench = sources_dir(scratch, SUITE_SPEC)
    tree = build_run(scratch / "runs-root" / "runs" / RUN_ID, bench)
    monkeypatch.setenv("LASSI_SCRATCH", str(scratch))
    out = score(tree, ("lassi",), with_bench=False)
    want = expected_trial_rows(loaded(tree.run_dir), built_profiles(["lassi"], bench))
    assert trial_rows(out) == want


def test_score_error_names_an_unknown_profile(tree: RunTree) -> None:
    with pytest.raises(score_module().ScoreError, match="no-such-profile"):
        score(tree, ("df-v0", "no-such-profile"))
    assert not (tree.runs_root / "scores").exists()


# ---------------------------------------------------------------------------
# review.md


def test_review_has_one_section_per_trial_in_trial_id_order(tree: RunTree) -> None:
    out = score(tree)
    order, _ = trial_sections(review_of(out), tree.trial_ids)
    assert order == sorted(tree.trial_ids), f"one level-2 heading per trial, in trial_id order; found {order}"
    sections = sections_of(tree, out)
    for trial in loaded(tree.run_dir):
        parts, _ = split_attempts(sections[trial.trial_id])
        assert sorted(parts) == list(range(len(trial.attempts))), f"{trial.trial_id}: one part per attempt"


def test_review_shows_each_attempts_stage_run_and_alignment(tree: RunTree) -> None:
    sections = sections_of(tree, score(tree))
    for trial in loaded(tree.run_dir):
        parts, _ = split_attempts(sections[trial.trial_id])
        for made in trial.attempts:
            where = f"{trial.trial_id} attempt {made.index}"
            part = parts[made.index]
            require_row(part, where, "stage_reached", made.stage_reached)
            require_row(part, where, "exit_code", fmt(made.run.exit_code))
            require_row(part, where, "hang", fmt(made.run.hang))
            require_row(part, where, "alignment", fmt(made.alignment.mean))


def test_review_shows_each_diagnostic_by_code_severity_and_place(tree: RunTree) -> None:
    sections = sections_of(tree, score(tree))
    shown = 0
    for trial in loaded(tree.run_dir):
        parts, _ = split_attempts(sections[trial.trial_id])
        for made in trial.attempts:
            for item in made.diagnostics:
                where = f"{trial.trial_id} attempt {made.index} diagnostic {item.code}"
                require_row(parts[made.index], where, item.severity, item.code or "-", place(item))
                shown += 1
    assert shown == 8, "the fixture shows full places, a place without a column, no file, and no code"


def test_review_shows_every_attempt_component_of_profiles_that_score_attempts(tree: RunTree) -> None:
    sections = sections_of(tree, score(tree))
    profiles = built_profiles(PROFILES, tree.bench_root)
    for trial in loaded(tree.run_dir):
        parts, _ = split_attempts(sections[trial.trial_id])
        for name, scores in attempt_scores(trial, profiles).items():
            for index, found in enumerate(scores):
                where = f"{trial.trial_id} attempt {index} {name}"
                for component, value in found.components.items():
                    require_row(parts[index], where, name, component, fmt(value))
                require_row(parts[index], where, name, "scalar", fmt(found.scalar))


def test_review_shows_every_trial_component_of_every_profile_with_its_notes(tree: RunTree) -> None:
    sections = sections_of(tree, score(tree))
    profiles = built_profiles(PROFILES, tree.bench_root)
    noted = 0
    for trial in loaded(tree.run_dir):
        _, trial_part = split_attempts(sections[trial.trial_id])
        for name, profile in profiles.items():
            found = profile.score(trial)
            for component, value in found.components.items():
                note = found.notes.get(component)
                cells = (name, component, fmt(value), *([note] if note else []))
                require_row(trial_part, f"{trial.trial_id} {name}", *cells)
                noted += bool(note)
            require_row(trial_part, f"{trial.trial_id} {name}", name, "scalar", fmt(found.scalar))
    assert noted, "the fixture's lassi scores carry notes"


def lines_with(text: str, *words: str) -> list[str]:
    """Return the lines of `text` that hold every word given, ignoring letter case."""
    return [line for line in text.splitlines() if all(word in line.lower() for word in words)]


def test_review_names_the_fields_a_p1_record_lacks(tree: RunTree) -> None:
    sections = sections_of(tree, score(tree))
    _, trial_part = split_attempts(sections[tree.p1_trial])
    assert lines_with(trial_part, "flag", "not recorded"), "a record from before P2.2 has no reference-run flags"
    size = len(P1_REFERENCE_STDOUT.encode("utf-8"))
    assert re.search(rf"\b{size} bytes\b", trial_part), f"the reference stdout's size, {size} bytes, is given"
    assert lines_with(trial_part, "request", "not recorded"), "a record from before P2.1 has no requests"


def test_review_names_nothing_missing_for_a_current_record(tree: RunTree) -> None:
    sections = sections_of(tree, score(tree))
    for trial_id in tree.current_trials:
        _, trial_part = split_attempts(sections[trial_id])
        assert not lines_with(trial_part, "flag", "not recorded"), f"{trial_id} records its reference-run flags"
        assert not lines_with(trial_part, "request", "not recorded"), f"{trial_id} records its requests"


def test_review_is_plain_ascii_and_holds_no_model_source_or_context_text(tree: RunTree) -> None:
    out = score(tree)
    review = review_of(out)
    assert SENTINEL_CORE not in review, "prompt, reply, file, context, stdout, or diagnostic text reached review.md"
    assert SCORE_ID not in review, "review.md names no score id, so two passes give the same bytes"


def test_no_score_output_holds_model_source_or_context_text(tree: RunTree) -> None:
    out = score(tree)
    files = [path for path in out.rglob("*") if path.is_file()]
    assert files
    for path in files:
        if path.suffix == ".parquet":
            text = repr(ds.dataset(str(path), format="parquet").to_table().to_pylist())
        else:
            text = path.read_bytes().decode("utf-8", "backslashreplace")
        assert SENTINEL_CORE not in text, f"{path.relative_to(out).as_posix()} holds model, source, or context text"
    for name in ("review.md", "metrics.md", "provenance.json"):
        assert (out / name).read_bytes().isascii(), f"{name} is plain ASCII"


# ---------------------------------------------------------------------------
# The source run is read, never changed; scoring is repeatable


def test_the_source_run_tree_is_unchanged_byte_for_byte(tree: RunTree) -> None:
    before = snapshot(tree.run_dir)
    score(tree)
    assert snapshot(tree.run_dir) == before, "scoring changes no byte of the run tree and adds nothing to it"


def test_scoring_twice_gives_identical_component_tables_and_review(tree: RunTree) -> None:
    first = score(tree, score_id="first")
    second = score(tree, score_id="second")
    assert first.resolve() != second.resolve()
    assert trial_rows(first) == trial_rows(second)
    assert attempt_rows(first) == attempt_rows(second)
    assert (first / "review.md").read_bytes() == (second / "review.md").read_bytes()


# ---------------------------------------------------------------------------
# The command line


def run_cli(argv: Sequence[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    """Run lassi.cli.main(argv) and return its status, stdout, and stderr; fail on an argparse exit."""
    from lassi import cli

    try:
        status = cli.main(list(argv))
    except SystemExit as stop:
        err = capsys.readouterr().err.strip()
        pytest.fail(f"`lassi {argv[0]}` exited through argparse ({stop.code}); P2.9 adds `lassi score`, which "
                    f"returns its status: {err[-400:]}")
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def score_argv(run_dir: Path, *profiles: str, score_id: str = SCORE_ID, bench_root: Path | None = None) -> list[str]:
    """Return the argv of `lassi score` for the run, each profile, the score id, and a bench root if given."""
    argv = ["score", str(run_dir)]
    for name in profiles:
        argv += ["--profile", name]
    argv += ["--score-id", score_id]
    return argv + (["--bench-root", str(bench_root)] if bench_root is not None else [])


def test_lassi_score_scores_a_fixture_run_end_to_end(tree: RunTree, capsys: pytest.CaptureFixture[str]) -> None:
    status, _, err = run_cli(score_argv(tree.run_dir, *PROFILES, score_id="e2e", bench_root=tree.bench_root), capsys)
    assert status == 0, err
    out = tree.runs_root / "scores" / "e2e"
    for name in ("provenance.json", "review.md", "metrics.md"):
        assert (out / name).is_file(), name
    trials = loaded(tree.run_dir)
    profiles = built_profiles(PROFILES, tree.bench_root)
    assert trial_rows(out) == expected_trial_rows(trials, profiles)
    assert attempt_rows(out) == expected_attempt_rows(trials, profiles)
    assert sorted(sections_of(tree, out)) == sorted(tree.trial_ids)


def test_lassi_score_takes_a_runs_root(tree: RunTree, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    elsewhere = tmp_path / "elsewhere"
    status, _, err = run_cli([*score_argv(tree.run_dir, "df-v0"), "--runs-root", str(elsewhere)], capsys)
    assert status == 0, err
    assert (elsewhere / "scores" / SCORE_ID / "review.md").is_file()
    assert not (tree.runs_root / "scores").exists()


def refuse_unknown_profile(tree: RunTree, tmp_path: Path) -> tuple[list[str], list[str]]:
    """An unknown profile name."""
    return score_argv(tree.run_dir, "df-v0", "no-such-profile", bench_root=tree.bench_root), ["no-such-profile"]


def refuse_missing_run_dir(tree: RunTree, tmp_path: Path) -> tuple[list[str], list[str]]:
    """A run dir that does not exist."""
    return score_argv(tree.runs_root / "runs" / "no-such-run", "df-v0"), ["no-such-run"]


def refuse_no_trial_json(tree: RunTree, tmp_path: Path) -> tuple[list[str], list[str]]:
    """A run dir that holds no trial.json."""
    empty = tree.runs_root / "runs" / "empty-run"
    empty.mkdir()
    (empty / "provenance.json").write_bytes((tree.run_dir / "provenance.json").read_bytes())
    return score_argv(empty, "df-v0"), [r"trial\.json"]


def refuse_not_strict(tree: RunTree, tmp_path: Path) -> tuple[list[str], list[str]]:
    """A trial.json the strict loader refuses: a key the Result Record does not have."""
    path = trial_dir(tree.run_dir, tree.current_trials[0]) / "trial.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["surprise_field"] = 1
    path.write_bytes(json_text(data).encode("ascii"))
    return score_argv(tree.run_dir, "df-v0"), ["surprise_field"]


def refuse_non_finite_manifest(tree: RunTree, tmp_path: Path) -> tuple[list[str], list[str]]:
    """A run provenance.json holding NaN, which the strict provenance.json of the score cannot copy."""
    path = tree.run_dir / "provenance.json"
    path.write_bytes(path.read_bytes().replace(b'"dirty": false', b'"dirty": false, "probe": NaN', 1))
    assert b"NaN" in path.read_bytes()
    return score_argv(tree.run_dir, "df-v0"), ["NaN"]


def add_to_manifest(tree: RunTree, probe: str) -> None:
    """Add the raw JSON text `probe` to the run's provenance.json as the value of a key named probe."""
    path = tree.run_dir / "provenance.json"
    path.write_bytes(path.read_bytes().replace(b'"dirty": false', f'"dirty": false, "probe": {probe}'.encode(), 1))
    assert b'"probe"' in path.read_bytes()


def refuse_overflowing_manifest(tree: RunTree, tmp_path: Path) -> tuple[list[str], list[str]]:
    """A run provenance.json whose number overflows a float (1e999), which the strict JSON of the score cannot hold."""
    add_to_manifest(tree, "1e999")
    return score_argv(tree.run_dir, "df-v0"), [r"provenance\.json"]


def refuse_deep_manifest(tree: RunTree, tmp_path: Path) -> tuple[list[str], list[str]]:
    """A run provenance.json nested deeper than the JSON reader and writer recurse."""
    add_to_manifest(tree, "[" * 5000 + "]" * 5000)
    return score_argv(tree.run_dir, "df-v0"), [r"provenance\.json", "recursion"]


def edit_trial(tree: RunTree, edit: Callable[[dict[str, Any]], object]) -> None:
    """Apply `edit` to the data of the first current trial.json and write it back; the strict loader still loads it."""
    path = trial_dir(tree.run_dir, tree.current_trials[0]) / "trial.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_bytes(json_text(data).encode("ascii"))
    read_trial(path, TextStore(tree.run_dir))


def refuse_surrogate_device(tree: RunTree, tmp_path: Path) -> tuple[list[str], list[str]]:
    """A trial whose provenance.device is a lone surrogate, which no Parquet string column holds."""
    edit_trial(tree, lambda data: data["provenance"].update(device=chr(0xD800)))
    return score_argv(tree.run_dir, *PROFILES, bench_root=tree.bench_root), ["surrogate"]


def refuse_corrections_past_int64(tree: RunTree, tmp_path: Path) -> tuple[list[str], list[str]]:
    """A trial whose final.corrections (2**63) the int64 corrections column of the metrics Parquet cannot hold."""
    edit_trial(tree, lambda data: data["final"].update(corrections=2**63))
    return score_argv(tree.run_dir, *PROFILES, bench_root=tree.bench_root), ["metric"]


def refuse_corrections_past_float(tree: RunTree, tmp_path: Path) -> tuple[list[str], list[str]]:
    """A trial whose final.corrections (10**400) no float holds, so a profile cannot score it."""
    edit_trial(tree, lambda data: data["final"].update(corrections=10**400))
    return score_argv(tree.run_dir, "df-v0"), ["df-v0"]


def refuse_deep_trial_json(tree: RunTree, tmp_path: Path) -> tuple[list[str], list[str]]:
    """A trial.json nested deeper than the JSON reader recurses."""
    path = trial_dir(tree.run_dir, tree.current_trials[0]) / "trial.json"
    path.write_bytes(("[" * 5000 + "]" * 5000).encode("ascii"))
    return score_argv(tree.run_dir, "df-v0"), [r"trial\.json", "recursion"]


def refuse_existing_score_dir(tree: RunTree, tmp_path: Path) -> tuple[list[str], list[str]]:
    """A score directory that already exists: it is never overwritten."""
    taken = tree.runs_root / "scores" / "taken"
    taken.mkdir(parents=True)
    (taken / "marker.txt").write_bytes(b"SYNTHETIC earlier score\n")
    return score_argv(tree.run_dir, "df-v0", score_id="taken"), ["taken", "exist"]


def refuse_no_bench_root(tree: RunTree, tmp_path: Path) -> tuple[list[str], list[str]]:
    """The lassi profile with no --bench-root and no $LASSI_SCRATCH to find the suite's sources under."""
    return score_argv(tree.run_dir, *PROFILES), ["bench"]


def refuse_no_runs_root(tree: RunTree, tmp_path: Path) -> tuple[list[str], list[str]]:
    """A run dir whose parent is not named runs, with no --runs-root and no $LASSI_RUNS_ROOT."""
    loose = build_run(tmp_path / "loose" / RUN_ID, tmp_path / "loose-bench")
    return score_argv(loose.run_dir, "df-v0"), [r"runs[ _-]root"]


def refuse_non_utf8_runs_root(tree: RunTree, tmp_path: Path) -> tuple[list[str], list[str]]:
    """A runs root whose name is not UTF-8, which the Parquet writer cannot encode (a lone surrogate)."""
    odd = "odd" + (chr(0xDC80) if os.name == "nt" else os.fsdecode(bytes([0xFF]))) + "root"
    return [*score_argv(tree.run_dir, "df-v0"), "--runs-root", str(tmp_path / odd)], ["UTF-8"]


REFUSALS: tuple[Callable[[RunTree, Path], tuple[list[str], list[str]]], ...] = (
    refuse_unknown_profile, refuse_missing_run_dir, refuse_no_trial_json, refuse_not_strict,
    refuse_existing_score_dir, refuse_no_bench_root, refuse_no_runs_root, refuse_non_finite_manifest,
    refuse_overflowing_manifest, refuse_deep_manifest, refuse_surrogate_device, refuse_corrections_past_int64,
    refuse_corrections_past_float, refuse_deep_trial_json, refuse_non_utf8_runs_root,
)


@pytest.mark.parametrize("refusal", REFUSALS, ids=[case.__name__.removeprefix("refuse_") for case in REFUSALS])
def test_lassi_score_refuses_with_status_2_before_creating_anything(
    refusal: Callable[[RunTree, Path], tuple[list[str], list[str]]], tree: RunTree, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv, named = refusal(tree, tmp_path)
    before = snapshot(tmp_path)
    status, _, err = run_cli(argv, capsys)
    assert status == 2, f"expected status 2, got {status}; stderr: {err}"
    assert err.startswith("lassi score:"), f"the message goes to stderr as `lassi score: <message>`: {err!r}"
    for pattern in named:
        assert re.search(pattern, err, re.IGNORECASE), f"the message names {pattern!r}: {err!r}"
    assert snapshot(tmp_path) == before, "a refusal creates, changes, and overwrites nothing"


def test_metric_tables_the_parquet_writer_would_refuse_are_refused_before_creating_anything(
    tree: RunTree, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arms that differ only by letter case are refused by the metrics Parquet check before any directory exists.

    A real tree with such arms cannot be built on a file system that ignores
    case, so the tables are given one arm in both cases.
    """
    module = score_module()
    real = module.metric_tables

    def case_variant_tables(pairs: Any) -> list[Any]:
        tables = real(pairs)
        return [*tables, dataclasses.replace(tables[0], arm=tables[0].arm.swapcase())]

    monkeypatch.setattr(module, "metric_tables", case_variant_tables)
    before = snapshot(tmp_path)
    with pytest.raises(module.ScoreError, match="letter case"):
        score(tree)
    assert snapshot(tmp_path) == before, "the refusal creates nothing"


@pytest.mark.slow
def test_metric_tables_past_the_default_partition_cap_are_written(
    tree: RunTree, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """More arms and directions than pyarrow writes by default (1024 partitions) still reach the metrics Parquet."""
    module = score_module()
    real = module.metric_tables

    def many_arm_tables(pairs: Any) -> list[Any]:
        first = real(pairs)[0]
        return [dataclasses.replace(first, arm=f"arm{index:04d}") for index in range(1025)]

    monkeypatch.setattr(module, "metric_tables", many_arm_tables)
    out = score(tree)
    rows = analysis("tables").read_metrics_parquet(out / "parquet")
    assert len({row["arm"] for row in rows["corrections"]}) == 1025
    assert (out / "provenance.json").is_file()


def nested_list(depth: int) -> list[Any]:
    """Return a list nested `depth` levels deep, built without recursion."""
    value: list[Any] = []
    for _ in range(depth):
        value = [value]
    return value


@pytest.mark.parametrize("probe", ["infinity", "deep"])
def test_a_manifest_value_provenance_json_cannot_hold_is_refused_before_creating_anything(
    probe: str, tree: RunTree, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A manifest value that passes the reader but not the strict JSON writer is refused while the packet is built.

    The reader is patched so the value reaches the in-memory build, the
    guard that holds whatever the reader lets through.
    """
    module = score_module()
    real = module.read_manifest
    value = math.inf if probe == "infinity" else nested_list(5000)
    monkeypatch.setattr(module, "read_manifest", lambda run_dir: {**real(run_dir), "probe": value})
    before = snapshot(tmp_path)
    with pytest.raises(module.ScoreError, match=r"provenance\.json"):
        score(tree, ("df-v0",), with_bench=False)
    assert snapshot(tmp_path) == before, "the refusal creates nothing"
