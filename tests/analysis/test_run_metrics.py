"""Tests for the run metrics per arm and direction (task P2.8).

Bible: Evaluation Protocol (preamble: per arm and direction, the paper's
values alongside, n trials per scenario, pass@k with Wilson 95% intervals;
the LASSI reproduction row; LASSI Paper Metrics, the rule that a rate shown
next to a paper value uses the paper's denominator; Acceptance Criteria, the
compile-only label), Reporting Rules (unmeasured values carry PLACEHOLDER),
Readability Standards (Run row: per-arm tables), Repository Layout
(`analysis/`), OQ-018 (no prompt, source, context, or model text in score
outputs), OQ-021 (published and recounted paper values both carried).

The contract these tests fix:

- `lassi.analysis.metrics.metric_tables(pairs, paper=None) -> list[MetricTable]`
  takes a sequence of (Trial, Score) pairs, the Score being the lassi
  profile's (task P2.7: components correct, correct_paper, within_10pct,
  first_try, sim_t, sim_t_c, sim_l, self_corr, cap_hit, fence_quirk,
  compiled, compiled_first_try, each a float or None, and `notes` keyed by
  component name). It groups the pairs by the trial_id's arm and direction
  segments and gives one MetricTable per group. `paper` is a
  lassi.analysis.paper PaperValues; None loads assets/scoring/lassi-paper.yaml.
  It never rescores: every value comes from the Score's components and from
  the trial's fields (attempts[-1].stage_reached, final.corrections,
  final.end_reason). It takes no bench root, so it cannot recompute a
  similarity.
- A MetricTable has `arm` (the trial_id arm segment), `direction` (the
  trial_id direction segment, such as `omp-cuda`), `compile_only` (bool),
  `rows` (MetricRow, one per metric name, each name once; the names of
  METRIC_NAMES below are all present), `stage_reached` (a mapping with the
  keys S0 to S5 and `none`, in that order, to the count of trials whose last
  attempt reached that stage, `none` for a trial with no attempts, zero
  counts included), and `corrections` (a mapping from each final.corrections
  value that occurs, ascending, to its count of trials).
- A MetricRow has `name`, `value` (float or None), `numerator`,
  `denominator`, `wilson_low`, `wilson_high`, `paper_published`,
  `paper_recount`, `paper_cite`, and `note` (plain ASCII str).
- Populations:
  - Trial rates over every trial of the arm and direction: compile_rate
    (compiled == 1.0), run_rate (last attempt S5), correct_rate (correct ==
    1.0), cap_hit_rate (cap_hit == 1.0), fence_quirk_rate (fence_quirk > 0;
    the note gives the total quirk count).
  - Paper-denominator rates over the correct trials: first_try_rate
    (first_try == 1.0) and sim_t_ge_0.6_rate (sim_t >= 0.6).
  - within_10pct_rate is always None; its note says no timing profiler
    exists and names P10; the Markdown shows PLACEHOLDER for it.
  - pass@1 and pass@3: pass_at_k per scenario (item) over its trials whose
    correct is not None, then the mean over scenarios. numerator is the sum
    of the per-scenario values, denominator the number of scenarios, and the
    Wilson interval is wilson_interval(numerator, denominator): the scenario
    is the unit pass@k averages over. When any scenario has fewer than k
    scored trials, the value is None and the note names n and k ("n = 2",
    "k = 3", spacing free).
  - Every other interval is wilson_interval(numerator, denominator) of the
    rate itself. A None value has no interval.
  - A trial whose needed component is None is left out of the numerator and
    the denominator, and the note gives how many were left out (the count
    and the word "excluded"). A value that cannot be computed is None, with
    a note.
- Paper columns: rows correct_rate, within_10pct_rate, first_try_rate, and
  sim_t_ge_0.6_rate show the paper's values for the table's direction.
  paper_published holds the published count as "<count>/<denominator>";
  paper_recount holds the recount the same way, then the alternate reading
  where the bible gives one (OMP -> CUDA within 10%: 23/32, then 24/32).
  paper_cite names "Evaluation Protocol, LASSI Paper Metrics". Rows the
  paper does not report (compile, run, cap-hit, fence-quirk) have None in
  paper_published and paper_recount.
- Compile-only: when every trial's correct is None with a note naming a
  compile-stage reproduction (P2.7's compile-only note), the table is
  labeled "compile-stage reproduction". correct_rate, first_try_rate,
  sim_t_ge_0.6_rate, pass@1, and pass@3 are None with that label in their
  notes and empty paper columns (None), and the Markdown shows none of the
  paper's correctness values.
- `lassi.analysis.tables`: `table_markdown(table) -> str` renders one
  table; `metrics_markdown(tables) -> str` is a document holding each
  table_markdown section. Each metric row is one Markdown table line whose
  first cell is the row name; a rate line shows "<numerator>/<denominator>".
  The stage-reached distribution is one line per key (first cells: key,
  count) and the corrections distribution one line per value (first cells:
  value, count); the text names "stage reached" and "corrections".
  `write_metrics_parquet(tables, out_dir)` writes Parquet under
  `<out_dir>/<table>/` as lassi.core.parquet does, and
  `read_metrics_parquet(out_dir)` returns {"metrics", "stage_reached",
  "corrections"} rows: metrics rows hold arm, direction, and every
  MetricRow field; stage_reached rows hold arm, direction, stage, and
  count; corrections rows hold arm, direction, corrections, and count.
- Output is plain ASCII and holds no prompt, source, context, or model text
  (OQ-018): a sentinel placed in replies, files, context, and diagnostics
  never reaches it.

Every trial and score here is SYNTHETIC and hand-built from the Result
Record and Score classes; no model is called, no program is run, and the
lassi profile is not used. Every expected value is counted by hand from the
specs below, as the comments show. No value in this module is a measurement.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow.parquet as pq
import pytest

from lassi.core.interfaces import Sampling, Score
from lassi.core.record import (
    Attempt,
    BenchItem,
    Context,
    Diagnostic,
    EndReason,
    Final,
    ModelInfo,
    Provenance,
    RunInfo,
    TextRef,
    Trial,
    arm_segment,
    make_trial_id,
)

REPO = Path(__file__).resolve().parents[2]
METRICS = "lassi.analysis.metrics"
TABLES = "lassi.analysis.tables"
STATS = "lassi.analysis.stats"
PAPER_FILE = REPO / "assets" / "scoring" / "lassi-paper.yaml"
PAPER_CITE = "Evaluation Protocol, LASSI Paper Metrics"
COMPILE_STAGE = "compile-stage reproduction"
EXACT = 1e-12

PROJECT, SUITE = "lassi-repro", "lassi-hecbench-10"
OMP_TO_CUDA, CUDA_TO_OMP = "omp-cuda", "cuda-omp"
TARGET_FILE = {OMP_TO_CUDA: "main.cu", CUDA_TO_OMP: "main.cpp"}
MODEL_A, MODEL_B, MODEL_C = "fixture-org/arm-a", "fixture-org/arm-b", "fixture-org/arm-c"
ARM_A, ARM_B, ARM_C = arm_segment(MODEL_A), arm_segment(MODEL_B), arm_segment(MODEL_C)
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
# A commit id for synthetic provenance; not a commit of this repository.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"

COMPONENTS = (
    "correct", "correct_paper", "within_10pct", "first_try", "sim_t", "sim_t_c", "sim_l",
    "self_corr", "cap_hit", "fence_quirk", "compiled", "compiled_first_try",
)
METRIC_NAMES = (
    "compile_rate", "run_rate", "correct_rate", "cap_hit_rate", "fence_quirk_rate",
    "first_try_rate", "sim_t_ge_0.6_rate", "within_10pct_rate", "pass@1", "pass@3",
)
ROW_FIELDS = (
    "name", "value", "numerator", "denominator", "wilson_low", "wilson_high",
    "paper_published", "paper_recount", "paper_cite", "note",
)
STAGE_KEYS = ("S0", "S1", "S2", "S3", "S4", "S5", "none")
NO_PAPER_VALUE = ("compile_rate", "run_rate", "cap_hit_rate", "fence_quirk_rate")
CORRECTNESS_ROWS = ("correct_rate", "first_try_rate", "sim_t_ge_0.6_rate", "pass@1", "pass@3")
# The bible's LASSI Paper Metrics values per direction and row: (published, recount readings in order).
PAPER_COLUMNS = {
    OMP_TO_CUDA: {
        "correct_rate": (["32/40"], ["32/40"]),
        "within_10pct_rate": (["25/32"], ["23/32", "24/32"]),
        "first_try_rate": (["21/32"], ["21/32"]),
        "sim_t_ge_0.6_rate": (["13/32"], ["8/32"]),
    },
    CUDA_TO_OMP: {
        "correct_rate": (["34/40"], ["34/40"]),
        "within_10pct_rate": (["21/34"], ["20/34"]),
        "first_try_rate": (["19/34"], ["18/34"]),
        "sim_t_ge_0.6_rate": (["16/34"], ["15/34"]),
    },
}

# A marker for text that must never reach a metrics output (OQ-018); it holds non-ASCII on purpose.
SENTINEL_CORE = "SENTINEL-7f3a"
SENTINEL = SENTINEL_CORE + "-" + chr(0xE9) + chr(0xE8) + "-reply-source-context"
COMPILE_ONLY_NOTE = "SYNTHETIC note: compile-only trial, a compile-stage reproduction; correct is not computed"
NOT_SCORED_NOTE = "SYNTHETIC note: correct not scored"


# ---------------------------------------------------------------------------
# Modules under test, looked up so that a missing one fails each test with a clear message


def module(name: str) -> ModuleType:
    """Import `name` (a lassi.analysis module); fail the test clearly while task P2.8 has not added it."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name not in ("lassi.analysis", name):
            raise
        pytest.fail(f"task P2.8 adds {name}: {error}")


def wilson(successes: float, n: int) -> tuple[float, float]:
    """Return lassi.analysis.stats.wilson_interval(successes, n) (tested against Newcombe 1998 on its own)."""
    return module(STATS).wilson_interval(successes, n)


# ---------------------------------------------------------------------------
# SYNTHETIC trials and scores


@dataclass(frozen=True)
class Spec:
    """One SYNTHETIC trial: its last stage (None: no attempt), corrections, and the lassi components it scored."""

    item: str
    run: int
    stage: str | None
    corrections: int
    correct: float | None
    first_try: float | None
    sim_t: float | None
    compiled: float
    cap_hit: float
    fence_quirk: float
    correct_note: str = NOT_SCORED_NOTE


# The main group: two scenarios (layout, bsearch), n = 5 trials each, 10 trials.
MAIN = (
    Spec("layout", 1, "S5", 0, 1.0, 1.0, 0.75, 1.0, 0.0, 0.0),  # clean pass on the first try
    Spec("layout", 2, "S5", 2, 0.0, 0.0, 0.40, 1.0, 0.0, 2.0),  # clean run, output differs; 2 fence-quirk hits
    Spec("layout", 3, "S1", 5, 0.0, 0.0, 0.30, 0.0, 1.0, 0.0),  # cap hit at a compile error
    Spec("layout", 4, None, 0, 0.0, 0.0, 0.00, 0.0, 0.0, 0.0),  # no attempt: the reference run failed
    Spec("layout", 5, "S4", 8, 0.0, 0.0, 0.20, 1.0, 0.0, 0.0),  # compiled past the gate, stale output
    Spec("bsearch", 1, "S5", 1, 1.0, 0.0, 0.60, 1.0, 0.0, 0.0),  # pass after one correction; Sim-T exactly 0.6
    Spec("bsearch", 2, "S5", 0, 1.0, 1.0, 0.59, 1.0, 0.0, 0.0),  # pass on the first try; Sim-T just below 0.6
    Spec("bsearch", 3, "S5", 3, 0.0, 0.0, 0.65, 1.0, 0.0, 3.0),  # clean run, output differs; 3 fence-quirk hits
    Spec("bsearch", 4, "S0", 5, 0.0, 0.0, 0.10, 0.0, 1.0, 0.0),  # cap hit with no output
    Spec("bsearch", 5, "S5", 0, 0.0, 0.0, 0.55, 1.0, 0.0, 0.0),  # clean run on the first try, output differs
)
# Hand counts over MAIN's 10 trials.
MAIN_TRIAL_RATES = {
    "compile_rate": (7, 10),  # compiled 1.0: layout 1, 2, 5; bsearch 1, 2, 3, 5
    "run_rate": (6, 10),  # last attempt S5: layout 1, 2; bsearch 1, 2, 3, 5
    "correct_rate": (3, 10),  # layout 1; bsearch 1, 2
    "cap_hit_rate": (2, 10),  # layout 3; bsearch 4
    "fence_quirk_rate": (2, 10),  # layout 2 (2 hits); bsearch 3 (3 hits): 5 hits in total
}
MAIN_FENCE_QUIRK_TOTAL = 5
MAIN_CORRECT_RATES = {
    "first_try_rate": (2, 3),  # of the 3 correct trials, layout 1 and bsearch 2 have first_try 1.0
    "sim_t_ge_0.6_rate": (2, 3),  # of the 3 correct trials, layout 1 (0.75) and bsearch 1 (0.60); not 0.59
}
# pass@k per scenario (n = 5): layout c = 1, bsearch c = 2.
# pass@1: 1/5 and 2/5, mean 0.3, sum 0.6 over 2 scenarios.
# pass@3: 1 - C(4,3)/C(5,3) = 0.6 and 1 - C(3,3)/C(5,3) = 0.9, mean 0.75, sum 1.5 over 2 scenarios.
MAIN_PASS_AT_K = {"pass@1": (0.3, 0.6, 2), "pass@3": (0.75, 1.5, 2)}
MAIN_STAGES = {"S0": 1, "S1": 1, "S2": 0, "S3": 0, "S4": 1, "S5": 6, "none": 1}
MAIN_CORRECTIONS = {0: 4, 1: 1, 2: 1, 3: 1, 5: 2, 8: 1}

# MAIN with correct unset on layout 4 and bsearch 5, and sim_t unset on bsearch 1 (a correct trial).
EXCLUDED = tuple(
    replace(spec, correct=None, first_try=None) if (spec.item, spec.run) in {("layout", 4), ("bsearch", 5)}
    else replace(spec, sim_t=None) if (spec.item, spec.run) == ("bsearch", 1)
    else spec
    for spec in MAIN
)
# correct_rate: 3 of 8 scored, 2 excluded. sim_t_ge_0.6_rate: of correct layout 1 and bsearch 2 (bsearch 1 excluded),
# only layout 1 is >= 0.6: 1/2. pass@k over scored trials, layout n = 4 c = 1, bsearch n = 4 c = 2:
# pass@1 1/4 and 2/4, mean 0.375, sum 0.75; pass@3 1 - C(3,3)/C(4,3) = 0.75 and 1 - C(2,3)/C(4,3) = 1.0,
# mean 0.875, sum 1.75.

# layout's five trials and bsearch 1 and 2: bsearch has 2 trials, fewer than k = 3.
SHORT = MAIN[:7]
# pass@1: layout 1/5 = 0.2, bsearch 2/2 = 1.0; mean 0.6, sum 1.2 over 2 scenarios (the 7 trials pooled give 3/7).

# layout 2, 3, and 5: no correct trial, so the paper-denominator rates cannot be computed.
NO_CORRECT = (MAIN[1], MAIN[2], MAIN[4])

# A compile-only group: correct and first_try None with the compile-only note on every trial.
COMPILE_ONLY = tuple(
    Spec(item, run, stage, corrections, None, None, sim_t, compiled, cap_hit, fence, COMPILE_ONLY_NOTE)
    for item, run, stage, corrections, sim_t, compiled, cap_hit, fence in (
        ("layout", 1, "S4", 0, 0.70, 1.0, 0.0, 0.0),
        ("layout", 2, "S4", 1, 0.50, 1.0, 0.0, 0.0),
        ("layout", 3, "S1", 5, 0.20, 0.0, 1.0, 2.0),
        ("bsearch", 1, "S4", 0, 0.80, 1.0, 0.0, 0.0),
        ("bsearch", 2, "S4", 2, 0.65, 1.0, 0.0, 0.0),
        ("bsearch", 3, "S0", 5, 0.00, 0.0, 1.0, 0.0),
    )
)
COMPILE_ONLY_TRIAL_RATES = {
    "compile_rate": (4, 6),  # layout 1, 2; bsearch 1, 2
    "run_rate": (0, 6),
    "cap_hit_rate": (2, 6),  # layout 3; bsearch 3
    "fence_quirk_rate": (1, 6),  # layout 3
}


def text_ref(text: str) -> TextRef:
    """Return a text-store reference for `text` (the store itself is not needed here)."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return TextRef(sha256=digest, path=f"text/{digest[:2]}/{digest}")


def compile_error() -> Diagnostic:
    """Return a SYNTHETIC compile error whose message echoes the sentinel, as a compiler echoes source."""
    return Diagnostic(stage="compile", severity="error", code="synthetic", message=f"SYNTHETIC error {SENTINEL}")


def fence_quirk() -> Diagnostic:
    """Return a SYNTHETIC fence-quirk warning."""
    return Diagnostic(stage="parse", severity="warning", code="fence-quirk", message=f"SYNTHETIC quirk {SENTINEL}")


def attempts_of(spec: Spec, target: str) -> list[Attempt]:
    """Return the attempts of `spec`: failed compiles (S1), then the last attempt at the spec's stage."""
    if spec.stage is None:
        return []
    assert spec.fence_quirk <= spec.corrections, "each fence-quirk hit sits on an earlier failed attempt"
    reply, code = f"SYNTHETIC reply {SENTINEL}", f"// SYNTHETIC code {SENTINEL}\nint main() {{ return 0; }}\n"
    earlier = [
        Attempt(index=index, response_text=reply, files={target: code}, stage_reached="S1",
                diagnostics=[*([fence_quirk()] if index < spec.fence_quirk else []), compile_error()])
        for index in range(spec.corrections)
    ]
    ran = spec.stage == "S5"
    last = Attempt(
        index=spec.corrections, response_text=reply, files={} if spec.stage == "S0" else {target: code},
        stage_reached=spec.stage, diagnostics=[] if spec.stage in ("S4", "S5") else [compile_error()],
        run=RunInfo(exit_code=0, hang=False, wall_s=0.5, stdout_ref=text_ref(f"SYNTHETIC stdout {spec.run}\n"),
                    stdout_truncated=False, stderr_truncated=False, workdir_incomplete=False) if ran else RunInfo(),
    )
    return [*earlier, last]


def end_reason(spec: Spec) -> EndReason | None:
    """Return the end reason the runner would record for `spec`."""
    if spec.stage is None:
        return EndReason(code="baseline-run", message="SYNTHETIC: the reference run failed")
    if spec.cap_hit == 1.0:
        return EndReason(code="correction-cap", message="SYNTHETIC: an error remained at the cap")
    return None


def make_trial(spec: Spec, model_id: str, direction: str, *, compile_only: bool = False) -> Trial:
    """Return the SYNTHETIC trial of `spec` for the arm of `model_id` in `direction`."""
    attempts = attempts_of(spec, TARGET_FILE[direction])
    reference = RunInfo()
    if not compile_only:
        reference = RunInfo(exit_code=0 if attempts else 1, hang=False, wall_s=1.0,
                            stdout_ref=text_ref("SYNTHETIC reference stdout\n"), stdout_truncated=False,
                            stderr_truncated=False, workdir_incomplete=False)
    return Trial(
        trial_id=make_trial_id(PROJECT, arm_segment(model_id), SUITE, direction, spec.item, spec.run),
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(commit=FAKE_COMMIT, dirty=False, device="hand-built (SYNTHETIC)", sdk=None,
                              date="2026-09-24T00:00:00+00:00"),
        bench_item=BenchItem(suite=SUITE, item=spec.item, split="eval", direction=direction),
        model=ModelInfo(backend="mock", id=model_id, sampling=SAMPLING),
        reference_run=reference,
        context=Context(knowledge_summary=f"SYNTHETIC summary {SENTINEL}",
                        source_description=f"SYNTHETIC description {SENTINEL}"),
        requests=[],
        attempts=attempts,
        final=Final(stage_reached=attempts[-1].stage_reached if attempts else None, corrections=spec.corrections,
                    wall_s=1.0, end_reason=end_reason(spec)),
    )


def make_score(spec: Spec) -> Score:
    """Return the lassi Score of `spec`, built by hand with the P2.7 component names and notes."""
    components: dict[str, float | None] = {
        "correct": spec.correct, "correct_paper": None, "within_10pct": None, "first_try": spec.first_try,
        "sim_t": spec.sim_t, "sim_t_c": spec.sim_t, "sim_l": spec.sim_t, "self_corr": float(spec.corrections),
        "cap_hit": spec.cap_hit, "fence_quirk": spec.fence_quirk, "compiled": spec.compiled,
        "compiled_first_try": 1.0 if spec.compiled == 1.0 and spec.corrections == 0 else 0.0,
    }
    assert tuple(components) == COMPONENTS
    notes = {
        "correct_paper": "SYNTHETIC note: the paper criterion is manual inspection and is never computed",
        "within_10pct": "SYNTHETIC note: no timing profiler exists",
        **{name: "SYNTHETIC note: computed by python 3.10" for name in ("sim_t", "sim_t_c", "sim_l")},
    }
    if spec.correct is None:
        notes["correct"] = notes["first_try"] = spec.correct_note
    if spec.sim_t is None:
        notes["sim_t"] = "SYNTHETIC note: sim_t not computed"
    try:
        return Score(components=components, scalar=spec.correct, notes=notes)
    except TypeError as error:
        pytest.fail(f"task P2.7 gives lassi.core.interfaces.Score a `notes` field: {error}")


def pairs_of(specs: Iterable[Spec], model_id: str = MODEL_A, direction: str = OMP_TO_CUDA, *,
             compile_only: bool = False) -> list[tuple[Trial, Score]]:
    """Return (Trial, Score) pairs for `specs` under one arm and direction."""
    return [(make_trial(spec, model_id, direction, compile_only=compile_only), make_score(spec)) for spec in specs]


# The two-arms-by-two-directions set: each group holds a different share of MAIN.
GROUPS = {
    (MODEL_A, OMP_TO_CUDA): MAIN,  # 10 trials, correct 3/10
    (MODEL_A, CUDA_TO_OMP): MAIN[:5],  # layout only: 5 trials, correct 1/5
    (MODEL_B, OMP_TO_CUDA): MAIN[5:],  # bsearch only: 5 trials, correct 2/5
    (MODEL_B, CUDA_TO_OMP): MAIN[1:],  # all but layout 1: 9 trials, correct 2/9
}
GROUP_CORRECT = {
    (ARM_A, OMP_TO_CUDA): (3, 10), (ARM_A, CUDA_TO_OMP): (1, 5),
    (ARM_B, OMP_TO_CUDA): (2, 5), (ARM_B, CUDA_TO_OMP): (2, 9),
}


def all_group_pairs() -> list[tuple[Trial, Score]]:
    """Return the pairs of every group of GROUPS."""
    return [pair for (model_id, direction), specs in GROUPS.items() for pair in pairs_of(specs, model_id, direction)]


# ---------------------------------------------------------------------------
# Running the metrics


def require_paper_file() -> None:
    """Fail the test clearly while the paper values file does not exist."""
    if not PAPER_FILE.is_file():
        pytest.fail(f"task P2.8 adds {PAPER_FILE.relative_to(REPO).as_posix()}")


def tables_of(pairs: Sequence[tuple[Trial, Score]]) -> dict[tuple[str, str], Any]:
    """Return the metric tables of `pairs` keyed by (arm, direction), checking there is one per key."""
    metrics = module(METRICS)
    require_paper_file()
    tables = list(metrics.metric_tables(list(pairs)))
    keyed = {(table.arm, table.direction): table for table in tables}
    assert len(keyed) == len(tables), f"one table per arm and direction, got {[key for key in keyed]}"
    return keyed


def only_table(pairs: Sequence[tuple[Trial, Score]]) -> Any:
    """Return the one metric table of `pairs`."""
    tables = tables_of(pairs)
    assert len(tables) == 1, f"one arm and direction gives one table, got {list(tables)}"
    return next(iter(tables.values()))


def rows_of(table: Any) -> dict[str, Any]:
    """Return the table's rows keyed by name, checking each name appears once and every METRIC_NAMES entry is there."""
    names = [row.name for row in table.rows]
    assert len(names) == len(set(names)), f"each metric appears once, got {names}"
    missing = [name for name in METRIC_NAMES if name not in names]
    assert not missing, f"missing metric rows {missing}"
    return {row.name: row for row in table.rows}


def fractions(text: str | None) -> list[str]:
    """Return every '<count>/<denominator>' in `text`, in order."""
    return re.findall(r"\d+/\d+", text or "")


def assert_rate(row: Any, numerator: int, denominator: int) -> None:
    """Assert a rate row's value, counts, and Wilson interval from its own counts."""
    assert row.numerator == numerator and row.denominator == denominator, (
        f"{row.name}: expected {numerator}/{denominator}, got {row.numerator}/{row.denominator}")
    assert type(row.value) is float and row.value == pytest.approx(numerator / denominator, abs=EXACT)
    assert (row.wilson_low, row.wilson_high) == pytest.approx(wilson(numerator, denominator), abs=EXACT)


def assert_pass_at_k(row: Any, value: float, numerator: float, scenarios: int) -> None:
    """Assert a pass@k row: the mean over scenarios, the sum and scenario count, and Wilson over scenarios."""
    assert type(row.value) is float and row.value == pytest.approx(value, abs=EXACT), f"{row.name}: {row.value}"
    assert row.numerator == pytest.approx(numerator, abs=EXACT) and row.denominator == scenarios
    assert (row.wilson_low, row.wilson_high) == pytest.approx(wilson(numerator, scenarios), abs=EXACT)


def assert_none(row: Any) -> None:
    """Assert a row that cannot be computed: no value, no interval, and a note."""
    assert row.value is None and row.wilson_low is None and row.wilson_high is None, f"{row.name}: {row!r}"
    assert isinstance(row.note, str) and row.note.strip(), f"{row.name}: a None value carries a note"


def cell_rows(markdown: str) -> list[list[str]]:
    """Return the cells of every Markdown table line, stripped."""
    rows = []
    for line in markdown.splitlines():
        text = line.strip()
        if text.startswith("|") and text.endswith("|") and len(text) > 1:
            rows.append([cell.strip() for cell in text[1:-1].split("|")])
    return rows


def metric_line(markdown: str, name: str) -> str:
    """Return the one Markdown table line whose first cell is `name`, joined."""
    found = [row for row in cell_rows(markdown) if row and row[0] == name]
    assert len(found) == 1, f"one Markdown line for {name}, got {len(found)}"
    return " | ".join(found[0])


def markdown_of(table: Any) -> str:
    """Return tables.table_markdown(table)."""
    return module(TABLES).table_markdown(table)


# ---------------------------------------------------------------------------
# One table per arm and direction


def test_one_table_per_arm_and_direction() -> None:
    tables = tables_of(all_group_pairs())
    assert set(tables) == set(GROUP_CORRECT), "two arms by two directions give four tables"
    for key, (correct, trials) in GROUP_CORRECT.items():
        table = tables[key]
        assert sum(table.stage_reached.values()) == trials, f"{key}: every trial of the group, none of another"
        rows = rows_of(table)
        assert_rate(rows["correct_rate"], correct, trials)
        assert rows["compile_rate"].denominator == trials


def test_the_arm_is_the_trial_id_segment_not_the_model_id() -> None:
    tables = tables_of(pairs_of(MAIN, MODEL_A, OMP_TO_CUDA))
    assert list(tables) == [(ARM_A, OMP_TO_CUDA)]
    assert ARM_A == "fixture-org--arm-a" and ARM_A != MODEL_A


def test_paper_values_follow_each_table_direction() -> None:
    tables = tables_of(all_group_pairs())
    for (arm, direction), table in tables.items():
        row = rows_of(table)["correct_rate"]
        published, recount = PAPER_COLUMNS[direction]["correct_rate"]
        assert fractions(row.paper_published) == published, f"{arm} {direction}"
        assert fractions(row.paper_recount) == recount, f"{arm} {direction}"


# ---------------------------------------------------------------------------
# Row fields


def test_every_row_carries_its_fields() -> None:
    for row in only_table(pairs_of(MAIN)).rows:
        for name in ROW_FIELDS:
            assert hasattr(row, name), f"metric row {row.name!r} lacks {name}"
        assert row.value is None or type(row.value) is float, f"{row.name}: value is a float or None"
        for bound in (row.wilson_low, row.wilson_high):
            assert bound is None or type(bound) is float, f"{row.name}: Wilson bounds are floats or None"
        assert isinstance(row.note, str) and row.note.isascii(), f"{row.name}: the note is plain ASCII"


# ---------------------------------------------------------------------------
# Populations


@pytest.mark.parametrize("name", list(MAIN_TRIAL_RATES))
def test_trial_rates_cover_every_trial_of_the_arm_and_direction(name: str) -> None:
    row = rows_of(only_table(pairs_of(MAIN)))[name]
    assert_rate(row, *MAIN_TRIAL_RATES[name])


def test_the_fence_quirk_note_gives_the_total_quirk_count() -> None:
    row = rows_of(only_table(pairs_of(MAIN)))["fence_quirk_rate"]
    assert re.search(rf"\b{MAIN_FENCE_QUIRK_TOTAL}\b", row.note), f"total quirk count missing from {row.note!r}"


@pytest.mark.parametrize("name", list(MAIN_CORRECT_RATES))
def test_paper_denominator_rates_cover_the_correct_trials(name: str) -> None:
    """first_try and Sim-T >= 0.6 are shares of the correct trials; Sim-T exactly 0.6 counts, 0.59 does not."""
    row = rows_of(only_table(pairs_of(MAIN)))[name]
    assert_rate(row, *MAIN_CORRECT_RATES[name])


def test_the_sim_t_row_names_the_open_tokenizer_question() -> None:
    """The paper names no Sim-T tokenizer (OQ-022), so the row and the legend say which sim_t they read."""
    table = only_table(pairs_of(MAIN))
    note = rows_of(table)["sim_t_ge_0.6_rate"].note
    assert "OQ-022" in note and "tokenize" in note and "unrounded" in note
    assert "OQ-022" in metric_line(markdown_of(table), "sim_t_ge_0.6_rate")
    assert "OQ-022" in module(TABLES).metrics_markdown([table]).split("## ")[0]


@pytest.mark.parametrize("name", list(MAIN_PASS_AT_K))
def test_pass_at_k_is_the_mean_over_scenarios_with_wilson_over_scenarios(name: str) -> None:
    row = rows_of(only_table(pairs_of(MAIN)))[name]
    value, numerator, scenarios = MAIN_PASS_AT_K[name]
    assert_pass_at_k(row, value, numerator, scenarios)
    assert (row.wilson_low, row.wilson_high) != pytest.approx(wilson(value * 10, 10), abs=1e-6), (
        "the scenario, not the trial, is the unit of the pass@k interval")


def test_pass_at_k_weights_each_scenario_equally() -> None:
    rows = rows_of(only_table(pairs_of(SHORT)))
    assert_pass_at_k(rows["pass@1"], 0.6, 1.2, 2)


def test_pass_at_k_is_none_naming_n_and_k_when_a_scenario_has_fewer_than_k_trials() -> None:
    row = rows_of(only_table(pairs_of(SHORT)))["pass@3"]
    assert_none(row)
    assert re.search(r"\bn\s*=\s*2\b", row.note), f"the note names n: {row.note!r}"
    assert re.search(r"\bk\s*=\s*3\b", row.note), f"the note names k: {row.note!r}"


def test_within_10pct_is_none_until_a_timing_profiler_exists() -> None:
    table = only_table(pairs_of(MAIN))
    row = rows_of(table)["within_10pct_rate"]
    assert_none(row)
    assert "profiler" in row.note.lower() and "P10" in row.note, f"the note names the missing profiler: {row.note!r}"
    assert "PLACEHOLDER" in metric_line(markdown_of(table), "within_10pct_rate")


def test_a_trial_with_a_none_component_is_excluded_and_counted_in_the_note() -> None:
    rows = rows_of(only_table(pairs_of(EXCLUDED)))
    correct = rows["correct_rate"]
    assert_rate(correct, 3, 8)
    assert "excluded" in correct.note.lower() and re.search(r"\b2\b", correct.note), correct.note
    sim_t = rows["sim_t_ge_0.6_rate"]
    assert_rate(sim_t, 1, 2)
    assert "excluded" in sim_t.note.lower() and re.search(r"\b1\b", sim_t.note), sim_t.note
    assert_rate(rows["first_try_rate"], 2, 3)
    assert_rate(rows["compile_rate"], 7, 10)
    assert_pass_at_k(rows["pass@1"], 0.375, 0.75, 2)
    assert_pass_at_k(rows["pass@3"], 0.875, 1.75, 2)


def test_a_value_that_cannot_be_computed_is_none_with_a_note() -> None:
    rows = rows_of(only_table(pairs_of(NO_CORRECT)))
    assert_none(rows["first_try_rate"])
    assert_none(rows["sim_t_ge_0.6_rate"])
    assert_rate(rows["correct_rate"], 0, 3)


def test_metrics_read_the_score_and_never_rescore() -> None:
    # The record alone says this trial never compiled or ran; its Score says it compiled and is correct.
    spec = Spec("layout", 1, "S1", 0, 1.0, 1.0, 0.9, 1.0, 0.0, 0.0)
    rows = rows_of(only_table(pairs_of([spec])))
    assert_rate(rows["correct_rate"], 1, 1)
    assert_rate(rows["compile_rate"], 1, 1)
    assert_rate(rows["first_try_rate"], 1, 1)
    assert_rate(rows["sim_t_ge_0.6_rate"], 1, 1)
    assert_rate(rows["run_rate"], 0, 1)


# ---------------------------------------------------------------------------
# Distributions


def test_the_stage_reached_distribution_counts_each_last_stage_and_none() -> None:
    table = only_table(pairs_of(MAIN))
    assert list(table.stage_reached) == list(STAGE_KEYS)
    assert dict(table.stage_reached) == MAIN_STAGES


def test_the_correction_distribution_counts_each_final_corrections_value() -> None:
    table = only_table(pairs_of(MAIN))
    assert list(table.corrections) == sorted(MAIN_CORRECTIONS)
    assert dict(table.corrections) == MAIN_CORRECTIONS


# ---------------------------------------------------------------------------
# Paper columns


@pytest.mark.parametrize("direction", [OMP_TO_CUDA, CUDA_TO_OMP])
def test_rows_show_the_paper_values_of_their_direction(direction: str) -> None:
    rows = rows_of(only_table(pairs_of(MAIN, MODEL_A, direction)))
    for name, (published, recount) in PAPER_COLUMNS[direction].items():
        row = rows[name]
        assert fractions(row.paper_published) == published, f"{name}: {row.paper_published!r}"
        assert fractions(row.paper_recount) == recount, f"{name}: {row.paper_recount!r}"
        assert row.paper_cite and PAPER_CITE in row.paper_cite, f"{name}: {row.paper_cite!r}"
    for name in NO_PAPER_VALUE:
        assert rows[name].paper_published is None and rows[name].paper_recount is None, name


def test_the_markdown_shows_the_paper_values_next_to_the_rate() -> None:
    markdown = markdown_of(only_table(pairs_of(MAIN)))
    correct = metric_line(markdown, "correct_rate")
    assert "3/10" in correct and "32/40" in correct, correct
    within = metric_line(markdown, "within_10pct_rate")
    assert all(fraction in within for fraction in ("25/32", "23/32", "24/32")), within


# ---------------------------------------------------------------------------
# Compile-only runs


def compile_only_table() -> Any:
    """Return the table of the compile-only group (arm C, CUDA to OpenMP)."""
    return only_table(pairs_of(COMPILE_ONLY, MODEL_C, CUDA_TO_OMP, compile_only=True))


def test_a_compile_only_run_is_labeled_a_compile_stage_reproduction() -> None:
    table = compile_only_table()
    assert table.compile_only is True
    assert COMPILE_STAGE in markdown_of(table).lower()
    assert only_table(pairs_of(MAIN)).compile_only is False
    assert only_table(pairs_of(EXCLUDED)).compile_only is False, "a few unscored trials do not make a run compile-only"


@pytest.mark.parametrize("name", CORRECTNESS_ROWS)
def test_compile_only_correctness_rows_are_none_labeled_and_without_paper_values(name: str) -> None:
    row = rows_of(compile_only_table())[name]
    assert_none(row)
    assert COMPILE_STAGE in row.note.lower(), f"{name}: {row.note!r}"
    assert row.paper_published is None and row.paper_recount is None and row.paper_cite is None, (
        f"{name}: a compile-stage reproduction is never compared with the paper's correctness values")


@pytest.mark.parametrize("name", list(COMPILE_ONLY_TRIAL_RATES))
def test_compile_only_runs_still_report_their_compile_stage_rates(name: str) -> None:
    assert_rate(rows_of(compile_only_table())[name], *COMPILE_ONLY_TRIAL_RATES[name])


def test_compile_only_markdown_shows_no_paper_correctness_value() -> None:
    markdown = markdown_of(compile_only_table())
    shown = [fraction for name in ("correct_rate", "first_try_rate", "sim_t_ge_0.6_rate")
             for readings in PAPER_COLUMNS[CUDA_TO_OMP][name] for fraction in readings if fraction in markdown]
    assert not shown, f"a compile-stage reproduction shows paper correctness values {shown}"


# ---------------------------------------------------------------------------
# Markdown


def test_each_metric_is_one_markdown_line_named_by_its_first_cell() -> None:
    markdown = markdown_of(only_table(pairs_of(MAIN)))
    for name in METRIC_NAMES:
        metric_line(markdown, name)


def test_the_markdown_shows_both_distributions() -> None:
    markdown = markdown_of(only_table(pairs_of(MAIN)))
    lines = [row[:2] for row in cell_rows(markdown)]
    assert "stage reached" in markdown.lower() and "corrections" in markdown.lower()
    for stage, count in MAIN_STAGES.items():
        assert [stage, str(count)] in lines, f"stage-reached line for {stage} with {count}"
    for value, count in MAIN_CORRECTIONS.items():
        assert [str(value), str(count)] in lines, f"corrections line for {value} with {count}"


def test_a_trial_whose_provenance_names_no_device_reads_not_recorded() -> None:
    """The runner records device None for the native executor; the table names that, never raising."""
    pairs = [(replace(trial, provenance=replace(trial.provenance, device=None)), score)
             for trial, score in pairs_of(MAIN)]
    pairs[0] = (replace(pairs[0][0], provenance=replace(pairs[0][0].provenance, device="SYNTHETIC device")),
                pairs[0][1])
    table = only_table(pairs)
    assert table.devices == ("SYNTHETIC device", "not recorded")
    assert "Device: SYNTHETIC device, not recorded." in markdown_of(table)


def test_the_metrics_document_holds_every_table() -> None:
    tables = list(tables_of(all_group_pairs()).values())
    document = module(TABLES).metrics_markdown(tables)
    for table in tables:
        assert markdown_of(table) in document, f"{table.arm} {table.direction} is missing from the document"


def test_the_output_does_not_depend_on_the_input_order() -> None:
    pairs = all_group_pairs() + pairs_of(COMPILE_ONLY, MODEL_C, CUDA_TO_OMP, compile_only=True)
    forward = module(TABLES).metrics_markdown(list(tables_of(pairs).values()))
    backward = module(TABLES).metrics_markdown(list(module(METRICS).metric_tables(list(reversed(pairs)))))
    assert forward == backward


# ---------------------------------------------------------------------------
# Parquet


def written(tmp_path: Path, tables: Sequence[Any]) -> dict[str, list[dict[str, Any]]]:
    """Write `tables` as Parquet under `tmp_path` and read them back."""
    out_dir = tmp_path / "metrics"
    module(TABLES).write_metrics_parquet(tables, out_dir)
    return module(TABLES).read_metrics_parquet(out_dir)


def mixed_tables() -> list[Any]:
    """Return the four group tables and the compile-only table."""
    pairs = all_group_pairs() + pairs_of(COMPILE_ONLY, MODEL_C, CUDA_TO_OMP, compile_only=True)
    return list(tables_of(pairs).values())


def test_parquet_holds_every_metric_row(tmp_path: Path) -> None:
    tables = mixed_tables()
    read = written(tmp_path, tables)
    keyed = {(row["arm"], row["direction"], row["name"]): row for row in read["metrics"]}
    expected = [(table, row) for table in tables for row in table.rows]
    assert len(read["metrics"]) == len(expected) == len(keyed)
    for table, row in expected:
        stored = keyed[(table.arm, table.direction, row.name)]
        for name in ROW_FIELDS:
            want, got = getattr(row, name), stored[name]
            if want is None:
                assert got is None, f"{table.arm} {table.direction} {row.name}.{name}: expected null, got {got!r}"
            else:
                assert got == want, f"{table.arm} {table.direction} {row.name}.{name}: expected {want!r}, got {got!r}"


def test_parquet_holds_both_distributions(tmp_path: Path) -> None:
    tables = mixed_tables()
    read = written(tmp_path, tables)
    for table in tables:
        key = (table.arm, table.direction)
        stages = {row["stage"]: row["count"] for row in read["stage_reached"]
                  if (row["arm"], row["direction"]) == key}
        corrections = {row["corrections"]: row["count"] for row in read["corrections"]
                       if (row["arm"], row["direction"]) == key}
        assert stages == dict(table.stage_reached), key
        assert corrections == dict(table.corrections), key


def test_parquet_holds_the_scenario_counts_the_markdown_shows(tmp_path: Path) -> None:
    tables = mixed_tables()
    read = written(tmp_path, tables)
    for table in tables:
        sizes = list(table.trials_per_scenario.values())
        stored = [row for row in read["metrics"] if (row["arm"], row["direction"]) == (table.arm, table.direction)]
        assert stored
        for row in stored:
            assert row["trials"] == table.trials
            assert row["scenarios"] == len(sizes)
            assert (row["trials_per_scenario_min"], row["trials_per_scenario_max"]) == (min(sizes), max(sizes))


def test_the_metrics_files_are_parquet(tmp_path: Path) -> None:
    tables = mixed_tables()
    out_dir = tmp_path / "metrics"
    module(TABLES).write_metrics_parquet(tables, out_dir)
    for name in ("metrics", "stage_reached", "corrections"):
        files = sorted((out_dir / name).rglob("*.parquet"))
        assert files, f"no Parquet file under {name}/"
        assert sum(pq.read_table(path).num_rows for path in files) > 0


# ---------------------------------------------------------------------------
# No upstream, source, or model text; plain ASCII (OQ-018)


def test_the_markdown_is_plain_ascii_and_holds_no_reply_source_or_context_text() -> None:
    tables = mixed_tables()
    documents = [module(TABLES).metrics_markdown(tables), *(markdown_of(table) for table in tables)]
    for text in documents:
        assert text.isascii(), "the metrics Markdown is plain ASCII"
        assert SENTINEL_CORE not in text, "reply, source, context, or diagnostic text reached the metrics Markdown"


def test_the_parquet_holds_no_reply_source_or_context_text(tmp_path: Path) -> None:
    read = written(tmp_path, mixed_tables())
    for table, rows in read.items():
        text = repr(rows)
        assert SENTINEL_CORE not in text, f"reply, source, context, or diagnostic text reached the {table} Parquet"


def test_every_note_is_plain_ascii_and_holds_no_reply_text() -> None:
    for table in mixed_tables():
        for row in table.rows:
            assert isinstance(row.note, str) and row.note.isascii(), f"{table.arm} {row.name}: {row.note!r}"
            assert SENTINEL_CORE not in row.note


def test_the_fixture_does_carry_the_sentinel() -> None:
    """Guards the two tests above: the sentinel is in each trial's reply, files, context, and diagnostics."""
    trial, _ = pairs_of(MAIN)[1]
    assert SENTINEL in trial.attempts[0].response_text
    assert all(SENTINEL in text for text in trial.attempts[0].files.values())
    assert SENTINEL in trial.context.knowledge_summary and SENTINEL in trial.context.source_description
    assert any(SENTINEL in diagnostic.message for diagnostic in trial.attempts[0].diagnostics)
    assert not SENTINEL.isascii()
