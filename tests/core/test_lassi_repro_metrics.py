"""Tests for the lassi-repro metrics line and the scored P1 dry run (task P2.10).

Bible: Project Recipes (the lassi-repro block and Notes), Component
Interfaces (ScoreProfile), Design Principles 1 and 5, Readability Standards
(Run row), Evaluation Protocol (Acceptance Criteria, the compile-stage label).

The contract these tests fix, from the P2.10 acceptance criteria
(plans/p2-scoring.md) and the task's contract decisions:

- projects/lassi-repro/recipe.yaml carries the bible block's metrics line,
  `metrics: [correct, within_10pct, first_try, sim_t, sim_l, self_corr]`,
  binds no score (the block binds none), and drops its comment line
  "metrics: left out until P2". The recipe hash changes with it; no golden
  pins a hash of lassi-repro, so the tests pin the metrics key of the
  resolved recipe and its canonical YAML instead.
- tests/fixtures/recipes/p1-dry-run.yaml inherits that metrics line from
  lassi-repro and binds `score: df-v0`.
- A local mock dry run of p1-dry-run writes the scores and the metrics
  tables: every trial.json holds final.score and a score on each attempt
  (df-v0's), run.md has a `## Metrics` section with the P2.8 table of each
  arm and direction, each labeled a compile-stage reproduction since the
  dry run is compile only, and `<run dir>/parquet/` holds the metrics
  tables.

The dry run follows tests/core/test_lassi_repro.py: the lassi-2024 prompt
set from a fresh extraction of the pinned upstream checkout into a temporary
directory (tools/extract_lassi_assets.py), skipping, naming that tool, when
third_party/LASSI is absent or not at the pin; SYNTHETIC bench programs
written here, not HeCBench sources; fake toolchains that compile nothing;
the none executor, which runs nothing. Its registry adds both registered
ScoreProfiles. Expected scores and tables are what the registered profiles
and lassi.analysis give for the trials read back from the run tree. No
upstream text is copied into this file (OQ-018), and no value in this
module is a measurement.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

import lassi.prompts as prompts_module
import lassi.prompts.assets as prompt_assets
from lassi.analysis.metrics import COMPILE_STAGE, MetricTable, metric_tables
from lassi.analysis.tables import TABLES as METRICS_TABLES
from lassi.analysis.tables import metrics_rows, read_metrics_parquet, table_markdown
from lassi.bench import load_suite
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.interfaces import BuildResult, Score
from lassi.core.recipe import load_recipe, resolved_yaml
from lassi.core.record import ScoreBreakdown, Trial, arm_segment
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.executors.none import NoneExecutor
from lassi.llm import MockBackend
from lassi.scoring.df_v0 import DfV0Profile
from lassi.scoring.lassi_profile import LassiProfile

REPO = Path(__file__).resolve().parents[2]
BIBLE = REPO / "docs" / "BIBLE.md"
PROJECTS = REPO / "projects"
FIXTURES = REPO / "tests" / "fixtures" / "recipes"
LASSI_REPRO = PROJECTS / "lassi-repro" / "recipe.yaml"
DRY_RUN = FIXTURES / "p1-dry-run.yaml"
BIBLE_BLOCK = "projects/lassi-repro/recipe.yaml"
PROJECT = "lassi-repro"
# The bible block's metrics line, as the bible writes it (Project Recipes, lassi-repro block).
REPRO_METRICS = ["correct", "within_10pct", "first_try", "sim_t", "sim_l", "self_corr"]
# How the canonical YAML of a resolved recipe (lassi.core.recipe) writes that line.
METRICS_YAML = "metrics:\n" + "".join(f"- {name}\n" for name in REPRO_METRICS)
DRY_RUN_SCORE = "df-v0"
SCORE_PROFILES = ("df-v0", "lassi")

UPSTREAM_DIR = REPO / "third_party" / "LASSI"
UPSTREAM_PIN = "74b46812523f2ff79b53b6880a4521690d7478b0"
EXTRACTOR = REPO / "tools" / "extract_lassi_assets.py"
SKIP_REASON = (
    "needs the upstream checkout at third_party/LASSI on commit 74b4681 so that tools/extract_lassi_assets.py "
    "can generate the lassi-2024 fragments; run uv run tools/fetch_upstream.py first"
)

SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
STAGES = ("baseline", "summarize_context", "describe_source", "generate", "compile_loop", "run_loop", "oracle")
TOOLCHAIN_NAMES = ("nvcc-sm80", "nvcpp-cc80")
DIRECTIONS = ("cuda-omp", "omp-cuda")
TRIAL_COUNT = 20

# The heading of the Metrics section in run.md, and the sections run.md held before it.
METRICS_HEADING = "## Metrics"
RUN_MD_SECTIONS = ("## Resolved recipe", "## Toolchain pins", "## Trials")


# ---------------------------------------------------------------------------
# Reading the recipe files and the bible


def read_ascii(path: Path) -> str:
    """Return a file's text, which must be plain ASCII."""
    return path.read_bytes().decode("ascii")


def own_keys(path: Path) -> dict[str, Any]:
    """Return the mapping one recipe file sets itself (before extends)."""
    data = yaml.safe_load(read_ascii(path))
    assert isinstance(data, dict), path
    return data


def bible_block(key: str) -> dict[str, Any]:
    """Return the bible's Project Recipes yaml block whose first line is the comment `# <key>`, parsed."""
    text = BIBLE.read_bytes().decode("utf-8")
    for block in re.findall(r"```yaml\n(.*?)```", text, re.DOTALL):
        if block.split("\n", 1)[0].strip() == f"# {key}":
            data = yaml.safe_load(block)
            assert isinstance(data, dict), key
            return data
    raise AssertionError(f"the bible has no yaml block headed # {key}")


def model_only_child(directory: Path) -> Path:
    """Write a recipe that extends lassi-repro and names only the model (lassi-repro binds none); return it."""
    child = directory / "model-only.yaml"
    child.write_bytes(b"extends: lassi-repro\nmodel: {backend: mock, id: mock-reference}\n")
    return child


# ---------------------------------------------------------------------------
# projects/lassi-repro/recipe.yaml and tests/fixtures/recipes/p1-dry-run.yaml


def test_the_bible_block_lists_these_metrics_and_binds_no_score() -> None:
    block = bible_block(BIBLE_BLOCK)
    assert block["metrics"] == REPRO_METRICS
    assert "score" not in block


def test_lassi_repro_carries_the_bible_blocks_metrics_line() -> None:
    own = own_keys(LASSI_REPRO)
    assert own.get("metrics") == bible_block(BIBLE_BLOCK)["metrics"]
    assert "score" not in own, "the bible block binds no score profile"


def test_lassi_repro_drops_its_comment_that_leaves_metrics_out_until_p2() -> None:
    lines = read_ascii(LASSI_REPRO).split("\n")
    comments = [line.split("#", 1)[1].strip() for line in lines if "#" in line]
    assert not [text for text in comments if re.match(r"metrics\s*:", text)], "a comment still stands for metrics"
    assert not [text for text in comments if "left out until P2" in text]


def test_the_resolved_lassi_repro_recipe_pins_the_metrics_key(tmp_path: Path) -> None:
    recipe = load_recipe(model_only_child(tmp_path))
    assert recipe.data["metrics"] == REPRO_METRICS
    assert "score" not in recipe.data
    assert METRICS_YAML in recipe.canonical_yaml
    assert METRICS_YAML in resolved_yaml(recipe), "the resolved recipe a run saves holds the metrics line"


def test_the_dry_run_recipe_binds_df_v0_and_inherits_the_metrics() -> None:
    own = own_keys(DRY_RUN)
    assert own.get("score") == DRY_RUN_SCORE
    assert "metrics" not in own, "p1-dry-run inherits the metrics line from lassi-repro"
    recipe = load_recipe(DRY_RUN)
    assert recipe.chain == ("base", PROJECT, "p1-dry-run")
    assert recipe.data["score"] == DRY_RUN_SCORE
    assert recipe.data["metrics"] == REPRO_METRICS
    bound = [(binding.interface, binding.name) for binding in recipe.bindings if binding.interface == "ScoreProfile"]
    assert bound == [("ScoreProfile", DRY_RUN_SCORE)]


# ---------------------------------------------------------------------------
# Upstream extraction, bench sources, fakes, and the dry run


def checkout_at_pin() -> bool:
    """Return True when third_party/LASSI is its own git checkout whose HEAD is the upstream pin."""
    if not (UPSTREAM_DIR / ".git").exists():
        return False
    done = subprocess.run(
        ["git", "-C", str(UPSTREAM_DIR), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return done.returncode == 0 and done.stdout.strip() == UPSTREAM_PIN


def load_extractor() -> ModuleType:
    """Load tools/extract_lassi_assets.py as a module."""
    spec = importlib.util.spec_from_file_location("extract_lassi_assets_for_p210", EXTRACTOR)
    assert spec and spec.loader, f"cannot load {EXTRACTOR}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def assets_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a temporary assets root holding a fresh extraction of the pinned checkout, or skip naming the tool."""
    if not checkout_at_pin():
        pytest.skip(SKIP_REASON)
    out = tmp_path_factory.mktemp("lassi-assets")
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        status = load_extractor().main(["--upstream", str(UPSTREAM_DIR), "--out", str(out)])
    assert status == 0, f"tools/extract_lassi_assets.py exited {status}: {sink.getvalue()[-2000:]}"
    return out


def synthetic_source(item: str, language: str) -> str:
    """Return a SYNTHETIC program standing in for one bench file."""
    kernel = "__global__ void k() { }\n" if language == "cuda" else ""
    launch = "  k<<<1, 1>>>();\n" if language == "cuda" else ""
    return (
        f"// SYNTHETIC {language} stand-in for {item}, written by tests/core/test_lassi_repro_metrics.py\n"
        f"#include <cstdio>\n{kernel}int main() {{\n{launch}"
        f'  std::printf("{item} {language}\\n");\n  return 0;\n}}\n'
    )


def write_bench(root: Path) -> Path:
    """Write every item's SYNTHETIC sources and support files where the suite manifest lays them out; return `root`."""
    suite = load_suite(SUITE_MANIFEST)
    for name, item in suite.items.items():
        for language, layout in item.languages.items():
            for file in layout.files:
                path = root / layout.dir / file
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(synthetic_source(name, language).encode("ascii"))
        for file, relative in item.support.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"// SYNTHETIC support file {file} for {name}\n".encode("ascii"))
    return root


def fake_toolchain(name: str) -> type:
    """Return a Toolchain class without PIN that compiles nothing and reports a PLACEHOLDER artifact."""

    class FakeToolchain:
        """Compiles nothing: every build reports a PLACEHOLDER artifact path, and nothing is written."""

        capabilities = frozenset({"diagnostics", "emits_warnings"})

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Report a PLACEHOLDER artifact."""
            return BuildResult(artifact=Path(workdir) / "PLACEHOLDER-artifact", diagnostics=[])

    FakeToolchain.__name__ = f"FakeToolchain_{name.replace('-', '_')}"
    return FakeToolchain


def make_registry() -> Registry:
    """Return a test Registry: the mock, stages, none executor, oracle, both ScoreProfiles, and fake toolchains."""
    registry = Registry()
    registry.register("LLMBackend", "mock", MockBackend)
    registry.register("Executor", "none", NoneExecutor)
    registry.register("Oracle", "stdout_mask", DEFAULT_REGISTRY.get("Oracle", "stdout_mask").factory)
    for name in TOOLCHAIN_NAMES:
        registry.register("Toolchain", name, fake_toolchain(name))
    for name in STAGES:
        registry.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    for name in SCORE_PROFILES:
        registry.register("ScoreProfile", name, DEFAULT_REGISTRY.get("ScoreProfile", name).factory)
    return registry


@dataclass
class DryRun:
    """One finished run (or the error that stopped it): its directory, bench root, and trials sorted by id."""

    run_dir: Path | None
    bench_root: Path
    trials: list[Trial] = field(default_factory=list)
    error: BaseException | None = None

    def checked(self) -> DryRun:
        """Return self, failing the test with the run's error when it did not complete."""
        if self.error is not None:
            pytest.fail(f"the dry run did not complete: {type(self.error).__name__}: {self.error}")
        assert self.run_dir is not None
        assert len(self.trials) == TRIAL_COUNT, f"precondition: {len(self.trials)} trials, expected {TRIAL_COUNT}"
        return self


@pytest.fixture(scope="module")
def dry_run(assets_root: Path, tmp_path_factory: pytest.TempPathFactory) -> DryRun:
    """Run tests/fixtures/recipes/p1-dry-run.yaml once for this module; each test reports a run that failed."""
    root = tmp_path_factory.mktemp("p1-dry-run-scored")
    outcome = DryRun(run_dir=None, bench_root=write_bench(root / "bench"))
    options = RunOptions(
        runs_root=root / "runs", run_id="dry-run", bench_root=outcome.bench_root, registry=make_registry(),
        roots=(FIXTURES, PROJECTS),
    )
    with pytest.MonkeyPatch.context() as patch:
        for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
            patch.delenv(name, raising=False)
        tmpdir = root / "compile-tmp"
        tmpdir.mkdir()
        patch.setenv("TMPDIR", str(tmpdir))
        patch.setattr(prompt_assets, "default_root", lambda: assets_root)
        patch.setattr(prompts_module, "default_roots", lambda: (assets_root / "prompts", REPO / "assets" / "prompts"))
        try:
            outcome.run_dir = run_recipe(DRY_RUN, options)
        except Exception as error:  # the tests report it through DryRun.checked
            outcome.error = error
            return outcome
    store = TextStore(outcome.run_dir)
    trials = (read_trial(path.parent, store) for path in outcome.run_dir.rglob("trial.json"))
    outcome.trials = sorted(trials, key=lambda trial: trial.trial_id)
    return outcome


# ---------------------------------------------------------------------------
# Oracles


def breakdown(score: Score) -> ScoreBreakdown:
    """Return a Score as the Result Record's ScoreBreakdown."""
    return ScoreBreakdown(components=dict(score.components), scalar=score.scalar)


def expected_tables(run: DryRun) -> list[MetricTable]:
    """Return the P2.8 tables of the run: lassi.analysis over the lassi profile's Score of every trial."""
    profile = LassiProfile(bench_root=run.bench_root)
    return metric_tables([(trial, profile.score(trial)) for trial in run.trials])


def metrics_section(run_md: str) -> str | None:
    """Return the Metrics section of run.md, from its heading to the next earlier-run.md section; None when absent."""
    lines = run_md.split("\n")
    if METRICS_HEADING not in lines:
        return None
    start = lines.index(METRICS_HEADING)
    end = next((index for index in range(start + 1, len(lines)) if lines[index] in RUN_MD_SECTIONS), len(lines))
    return "\n".join(lines[start:end])


def assert_tables_shown(section: str, tables: Sequence[MetricTable]) -> None:
    """Assert that `section` shows each table under a heading naming its arm and direction, line by line in order."""
    lines = section.split("\n")
    heading_at: dict[str, int] = {}
    for table in tables:
        title = f"{table.arm} {table.direction}"
        found = [index for index, line in enumerate(lines) if line.startswith("#") and line.rstrip().endswith(title)]
        assert len(found) == 1, f"the Metrics section has {len(found)} headings naming {title!r}; expected one"
        heading_at[title] = found[0]
    starts = sorted(heading_at.values())
    for table in tables:
        title = f"{table.arm} {table.direction}"
        start = heading_at[title]
        region = lines[start + 1 : next((index for index in starts if index > start), len(lines))]
        position = 0
        for line in table_markdown(table).split("\n"):
            if not line.strip() or line.startswith("#"):
                continue
            assert line in region[position:], f"{title}: the Metrics section lacks, or misorders, the line {line!r}"
            position = region.index(line, position) + 1


# ---------------------------------------------------------------------------
# The scored dry run


def test_the_dry_run_saves_its_score_and_metrics_in_the_resolved_recipe(dry_run: DryRun) -> None:
    run = dry_run.checked()
    resolved = yaml.safe_load(read_ascii(run.run_dir / "recipe.resolved.yaml"))
    assert resolved["score"] == DRY_RUN_SCORE
    assert resolved["metrics"] == REPRO_METRICS


def test_every_dry_run_trial_has_a_final_score_and_a_score_on_each_attempt(dry_run: DryRun) -> None:
    run = dry_run.checked()
    profile = DfV0Profile()
    for trial in run.trials:
        raw = json.loads(read_ascii(trial_dir(run.run_dir, trial.trial_id) / "trial.json"))
        assert raw["final"]["score"] is not None, f"{trial.trial_id}: trial.json holds no final.score"
        assert trial.final.score == profile.score(trial).scalar, f"{trial.trial_id}: final.score is not df-v0's"
        assert trial.attempts, f"{trial.trial_id}: precondition, the trial holds an attempt"
        for attempt in trial.attempts:
            assert attempt.score.scalar is not None, f"{trial.trial_id} attempt {attempt.index}: no score"
            assert attempt.score == breakdown(profile.score_attempt(attempt)), f"{trial.trial_id} {attempt.index}"


def test_the_dry_run_md_has_a_metrics_section_labeled_a_compile_stage_reproduction(dry_run: DryRun) -> None:
    run = dry_run.checked()
    tables = expected_tables(run)
    arm = arm_segment(load_recipe(DRY_RUN, registry=make_registry()).data["model"]["id"])
    assert [(table.arm, table.direction) for table in tables] == [(arm, direction) for direction in DIRECTIONS]
    assert all(table.compile_only for table in tables), "precondition: the compile-only dry run gives no correctness"
    section = metrics_section(read_ascii(run.run_dir / "run.md"))
    assert section is not None, f"run.md has no line {METRICS_HEADING!r}"
    assert_tables_shown(section, tables)
    assert section.count(COMPILE_STAGE) >= len(tables), "each table carries the compile-stage label"


def test_the_dry_run_writes_the_metrics_tables_to_its_parquet_mirror(dry_run: DryRun) -> None:
    run = dry_run.checked()
    parquet = run.run_dir / "parquet"
    for name in METRICS_TABLES:
        assert (parquet / name).is_dir(), f"parquet/{name} was not written"
    rows = read_metrics_parquet(parquet)
    assert rows == metrics_rows(expected_tables(run))
    assert rows["metrics"] and all(row["compile_only"] is True for row in rows["metrics"])
