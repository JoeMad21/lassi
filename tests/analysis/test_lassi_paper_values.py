"""Tests for the LASSI paper's reference values and their loader (task P2.8).

Bible: Evaluation Protocol (LASSI Paper Metrics: the definitions table, the
denominators, the per-model correct counts, and the recount table;
Acceptance Criteria, the B0 criterion), Source Papers (LASSI results table),
Repository Layout (`assets/scoring/`, `analysis/`).

The contract these tests fix:

- `assets/scoring/lassi-paper.yaml` is plain ASCII YAML holding a mapping.
  lassi.analysis.paper reads it; the code holds no paper value. The file's
  own layout is the implementer's; these tests read it through the loader.
- `lassi.analysis.paper.PAPER_FILE` is that path, and
  `load_paper(path: Path = PAPER_FILE) -> PaperValues` loads it.
- `PaperValues.metric(direction, name) -> PaperMetric | None` gives the
  paper's values for one metric row of the P2.8 tables, keyed by the repo's
  direction keys (`omp-cuda`, `cuda-omp`, as trial ids spell them) and the
  metric row names `correct_rate`, `within_10pct_rate`, `first_try_rate`,
  and `sim_t_ge_0.6_rate`. A metric the paper does not report, such as
  `compile_rate`, gives None.
- A PaperMetric has `published` and `recount` (each a PaperCount with int
  `count` and `denominator`), `recount_alternate` (a PaperCount or None),
  and `cite`, which names "Evaluation Protocol, LASSI Paper Metrics". Both
  the published and the recounted values are carried, since OQ-021 is open
  and its recommendation carries both. Where the bible says a published
  value recounts to itself (correct output in both directions, OMP -> CUDA
  first try), the recount equals the published count.
- `PaperValues.correct_by_model(direction) -> Mapping[str, PaperCount]` gives
  the per-model correct counts over the 10 apps, keyed by the model names
  the bible spells.
- `paper_interval(count: PaperCount) -> tuple[float, float]` is the Wilson
  95% interval of the paper's own count and denominator, and
  `b0_interval(paper, direction) -> tuple[float, float]` is the interval the
  B0 acceptance criterion names: around the paper's WizardCoder correct
  count (9/10 OMP -> CUDA, 10/10 CUDA -> OMP).

Every expected count is copied from the bible's LASSI Paper Metrics tables.
They are published values and recounts read from the paper's tables, not
measurements by this project. The interval values are Wilson arithmetic on
those counts, recomputed for this module.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
MODULE = "lassi.analysis.paper"
STATS = "lassi.analysis.stats"
PAPER_FILE = REPO / "assets" / "scoring" / "lassi-paper.yaml"
PAPER_CITE = "Evaluation Protocol, LASSI Paper Metrics"
OMP_TO_CUDA, CUDA_TO_OMP = "omp-cuda", "cuda-omp"
DIRECTIONS = (OMP_TO_CUDA, CUDA_TO_OMP)
Z95 = 1.959963984540054
FOUR_DECIMALS = 5e-5
EXACT = 1e-12

# The bible's LASSI Paper Metrics tables: (published, recount, alternate recount or None), each as (count, denominator).
PAPER = {
    OMP_TO_CUDA: {
        "correct_rate": ((32, 40), (32, 40), None),
        "within_10pct_rate": ((25, 32), (23, 32), (24, 32)),
        "first_try_rate": ((21, 32), (21, 32), None),
        "sim_t_ge_0.6_rate": ((13, 32), (8, 32), None),
    },
    CUDA_TO_OMP: {
        "correct_rate": ((34, 40), (34, 40), None),
        "within_10pct_rate": ((21, 34), (20, 34), None),
        "first_try_rate": ((19, 34), (18, 34), None),
        "sim_t_ge_0.6_rate": ((16, 34), (15, 34), None),
    },
}
# The percentages of the bible's LASSI results table (Source Papers), as printed.
PRINTED_PERCENT = {
    OMP_TO_CUDA: {"correct_rate": 80.0, "within_10pct_rate": 78.1, "first_try_rate": 65.6, "sim_t_ge_0.6_rate": 40.6},
    CUDA_TO_OMP: {"correct_rate": 85.0, "within_10pct_rate": 61.8, "first_try_rate": 55.9, "sim_t_ge_0.6_rate": 47.1},
}
# Correct counts per model over the 10 apps (LASSI Paper Metrics).
CORRECT_BY_MODEL = {
    OMP_TO_CUDA: {"GPT-4": 7, "Codestral": 9, "WizardCoder": 9, "DeepSeek-Coder-V2": 7},
    CUDA_TO_OMP: {"GPT-4": 9, "Codestral": 8, "WizardCoder": 10, "DeepSeek-Coder-V2": 7},
}
APPS = 10
PAPER_METRICS = tuple(PAPER[OMP_TO_CUDA])
NO_PAPER_VALUE = ("compile_rate", "run_rate", "cap_hit_rate", "fence_quirk_rate")


def module(name: str) -> ModuleType:
    """Import `name` (a lassi.analysis module); fail the test clearly while task P2.8 has not added it."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name not in ("lassi.analysis", name):
            raise
        pytest.fail(f"task P2.8 adds {name}: {error}")


def shipped_paper() -> Any:
    """Return the paper values loaded from the shipped file; fail clearly while the file does not exist."""
    paper = module(MODULE)
    if not PAPER_FILE.is_file():
        pytest.fail(f"task P2.8 adds {PAPER_FILE.relative_to(REPO).as_posix()}")
    return paper.load_paper()


def pair(count: Any) -> tuple[int, int]:
    """Return a PaperCount as (count, denominator), checking both are ints."""
    assert type(count.count) is int and type(count.denominator) is int, f"counts are ints, got {count!r}"
    return count.count, count.denominator


# ---------------------------------------------------------------------------
# The file


def test_the_paper_file_is_plain_ascii_yaml_at_its_path() -> None:
    paper = module(MODULE)
    assert Path(paper.PAPER_FILE).resolve() == PAPER_FILE.resolve()
    if not PAPER_FILE.is_file():
        pytest.fail(f"task P2.8 adds {PAPER_FILE.relative_to(REPO).as_posix()}")
    raw = PAPER_FILE.read_bytes()
    assert raw.isascii(), "the paper file is plain ASCII"
    assert isinstance(yaml.safe_load(raw.decode("ascii")), dict), "the paper file holds a mapping"


def test_load_paper_reads_the_given_path(tmp_path: Path) -> None:
    shipped_paper()
    copy = tmp_path / "copy.yaml"
    copy.write_bytes(PAPER_FILE.read_bytes())
    loaded = module(MODULE).load_paper(copy)
    assert pair(loaded.metric(OMP_TO_CUDA, "correct_rate").published) == (32, 40)


# ---------------------------------------------------------------------------
# Published values and recounts


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("name", PAPER_METRICS)
def test_published_and_recount_values_match_the_bible(direction: str, name: str) -> None:
    metric = shipped_paper().metric(direction, name)
    assert metric is not None, f"the paper reports {name} for {direction}"
    published, recount, alternate = PAPER[direction][name]
    assert pair(metric.published) == published
    assert pair(metric.recount) == recount
    if alternate is None:
        assert metric.recount_alternate is None
    else:
        assert pair(metric.recount_alternate) == alternate


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("name", PAPER_METRICS)
def test_each_published_fraction_gives_the_printed_percentage(direction: str, name: str) -> None:
    count, denominator = pair(shipped_paper().metric(direction, name).published)
    assert round(100 * count / denominator, 1) == PRINTED_PERCENT[direction][name]


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_denominators_are_all_trials_for_correct_and_correct_trials_otherwise(direction: str) -> None:
    paper = shipped_paper()
    correct = paper.metric(direction, "correct_rate").published
    assert correct.denominator == 40, "40 trials per direction: 10 apps x 4 models"
    for name in PAPER_METRICS[1:]:
        metric = paper.metric(direction, name)
        assert metric.published.denominator == correct.count, f"{name} is a share of the correct trials"
        assert metric.recount.denominator == correct.count


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("name", PAPER_METRICS)
def test_each_paper_value_cites_its_bible_table(direction: str, name: str) -> None:
    cite = shipped_paper().metric(direction, name).cite
    assert isinstance(cite, str) and cite.isascii()
    assert PAPER_CITE in cite


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("name", NO_PAPER_VALUE)
def test_a_metric_the_paper_does_not_report_has_no_value(direction: str, name: str) -> None:
    assert shipped_paper().metric(direction, name) is None


# ---------------------------------------------------------------------------
# Per-model correct counts


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_per_model_correct_counts_match_the_bible(direction: str) -> None:
    by_model = shipped_paper().correct_by_model(direction)
    assert {model: pair(count) for model, count in by_model.items()} == {
        model: (count, APPS) for model, count in CORRECT_BY_MODEL[direction].items()
    }


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_per_model_counts_sum_to_the_direction_correct_count(direction: str) -> None:
    paper = shipped_paper()
    total = sum(count.count for count in paper.correct_by_model(direction).values())
    assert total == paper.metric(direction, "correct_rate").published.count


# ---------------------------------------------------------------------------
# Intervals around paper values


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("name", PAPER_METRICS)
def test_a_paper_value_interval_uses_the_paper_count_and_denominator(direction: str, name: str) -> None:
    published = shipped_paper().metric(direction, name).published
    interval = module(MODULE).paper_interval(published)
    expected = module(STATS).wilson_interval(published.count, published.denominator)
    assert interval == pytest.approx(expected, abs=EXACT)


def test_b0_interval_omp_to_cuda_is_wilson_around_wizardcoder_9_of_10() -> None:
    paper = shipped_paper()
    low, high = module(MODULE).b0_interval(paper, OMP_TO_CUDA)
    # Wilson 95% of 9/10, recomputed: (0.5958, 0.9821).
    assert (low, high) == pytest.approx((0.5958, 0.9821), abs=FOUR_DECIMALS)
    assert (low, high) == pytest.approx(module(STATS).wilson_interval(9, 10), abs=EXACT)


def test_b0_interval_cuda_to_omp_is_wilson_around_wizardcoder_10_of_10() -> None:
    paper = shipped_paper()
    low, high = module(MODULE).b0_interval(paper, CUDA_TO_OMP)
    # Wilson 95% of 10/10: lower = 10 / (10 + z^2) = 0.7225, upper = 1.
    assert low == pytest.approx(10 / (10 + Z95**2), abs=EXACT)
    assert high == pytest.approx(1.0, abs=EXACT)
    assert (low, high) == pytest.approx((0.7225, 1.0), abs=FOUR_DECIMALS)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_b0_interval_reads_the_wizardcoder_count_from_the_file(direction: str) -> None:
    paper = shipped_paper()
    wizardcoder = paper.correct_by_model(direction)["WizardCoder"]
    expected = module(MODULE).paper_interval(wizardcoder)
    assert module(MODULE).b0_interval(paper, direction) == pytest.approx(expected, abs=EXACT)
