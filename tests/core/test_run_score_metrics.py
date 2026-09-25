"""Tests for `score` and `metrics` in `lassi run` (task P2.10).

Bible: Project Recipes (the lassi-repro and lassi-df blocks, Notes),
Component Interfaces (ScoreProfile; the capability rule), Design Principles
1 and 5, Readability Standards (Run row).

The contract these tests fix, from the P2.10 acceptance criteria
(plans/p2-scoring.md) and the task's contract decisions:

- The runner no longer refuses `score` or `metrics`.
- `score: <name>` builds that registered ScoreProfile, passing the run's
  bench root to a profile that needs one (the lassi profile). A profile
  that declares `scores_attempts` (df-v0) fills every Attempt.score, and
  every profile sets final.score to the scalar of its trial Score.
  trial.json, trial.md, and the Parquet mirror show both through the
  existing record writers.
- Each `metrics` name is checked by the runner before any directory is
  created or any model is asked, against the names the registered
  providers offer: the components each registered ScoreProfile declares in
  trial_components and the row names of lassi.analysis.metrics.METRIC_NAMES.
  An unknown name and a name listed twice are refused with a RecipeError or
  RunError that names it (tests/fixtures/recipes/unknown-metric.yaml), and
  so is a registered ScoreProfile whose trial_components is not a list of
  distinct names.
- With `metrics` set, the runner computes the metrics after the trials. It
  builds each ScoreProfile that provides a named metric (the lassi profile
  for the lassi-repro names, whatever `score` binds); these metric-only
  scores never write Attempt.score or final.score. run.md gains a section
  headed `## Metrics` that holds the P2.8 table of every arm and direction
  (lassi.analysis.tables), and `<run dir>/parquet/` gains the metrics
  tables (lassi.analysis.tables.write_metrics_parquet). Without `metrics`,
  run.md has no Metrics section and no metrics table is written.
- When a named metric is a trial component, `<run dir>/parquet/
  metric_values/` holds one row per trial and named component (trial_id,
  profile, component, value, note), and the Metrics section shows every
  one of those rows (Design Principle 7): a table of each trial's value of
  each named component, and a notes table (Trial, Profile, Component, Note)
  with one row per non-empty note. Neither holds source or model text
  (OQ-018).
- Importing lassi.core.runner registers both ScoreProfiles, so `lassi run`
  binds them from the command line.

Oracle style: the expected scores and tables are what the registered
profiles and lassi.analysis give for the trials read back from the run
tree; no score or rate is written into this module.

Two tests run programs through a scripted executor. The runner records
device None for every executor that runs programs, so the tables of those
runs list a None device, which their Markdown must still render. Every
other metrics test uses a compile-only run.

The runs use the p0-smoke template prompt set, a scripted backend, fake
toolchains that compile nothing, and a scripted executor that runs nothing.
Model replies, program output, wall times, and bench sources are SYNTHETIC.
No value in this module is a measurement.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import yaml

from lassi.analysis.metrics import COMPILE_STAGE, METRIC_NAMES, MetricTable, metric_tables
from lassi.analysis.tables import TABLES as METRICS_TABLES
from lassi.analysis.tables import metrics_rows, read_metrics_parquet, table_markdown
from lassi.bench import Direction, load_suite
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Limits, Message, RunResult, Sampling, Score
from lassi.core.parquet import read_run_parquet
from lassi.core.recipe import RecipeError
from lassi.core.record import Diagnostic, ScoreBreakdown, Trial
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.core.trial_md import fmt
from lassi.executors.none import NoneExecutor
from lassi.scoring.df_v0 import MULTI_TURN, SINGLE_TURN, DfV0Profile
from lassi.scoring.lassi_profile import PROFILE_FILE, LassiProfile, load_profile

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "recipes"
UNKNOWN_METRIC = FIXTURES / "unknown-metric.yaml"

SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
ITEM = "layout"
MODEL_ID = "scripted-fixture"
TEMPLATE_SET = "p0-smoke"
CUDA_TO_OMP = Direction("cuda", "omp")
OMP_TO_CUDA = Direction("omp", "cuda")
TOOLCHAINS = {"cuda": "nvcc-sm80", "omp": "nvcpp-cc80"}
ORACLE = {"kind": "stdout_mask", "passfail": True}
COMPILE_STAGES = ("generate", "compile_loop")
RUN_STAGES = ("baseline", "generate", "compile_loop", "run_loop", "oracle")
SCORE_PROFILES = ("df-v0", "lassi")
# The bible's lassi-repro metrics line (Project Recipes); tests/core/test_lassi_repro_metrics.py pins it to the bible.
REPRO_METRICS = ["correct", "within_10pct", "first_try", "sim_t", "sim_l", "self_corr"]
NO_SUCH_METRIC = "no_such_metric"
# The registry name of the SYNTHETIC ScoreProfile that declares a bad trial_components.
FAKE_PROFILE = "declaring-fake"
# Named trial components of both profiles, each with the profile that offers it: the lassi-repro metrics (the
# lassi profile notes some of them) and df-v0's multi-turn value, which carries no note.
COMPONENT_PROFILES = {**dict.fromkeys(REPRO_METRICS, "lassi"), MULTI_TURN: "df-v0"}
COMPONENT_METRICS = list(COMPONENT_PROFILES)

# The heading of the Metrics section in run.md, and the sections run.md held before it; the Metrics section ends
# at the next of those, or at the end of run.md.
METRICS_HEADING = "## Metrics"
RUN_MD_SECTIONS = ("## Resolved recipe", "## Toolchain pins", "## Trials")
# The Metrics section's tables of the named components: the value of each per trial, and the notes on them.
VALUES_HEADING = "### Named components per trial"
NOTES_HEADING = "### Notes on the named components"
NOTES_HEADER = ["Trial", "Profile", "Component", "Note"]
# parquet/metric_values: its directory under the run tree and its columns, in order.
METRIC_VALUES = Path("parquet") / "metric_values"
VALUE_COLUMNS = ["trial_id", "profile", "component", "value", "note"]
# SYNTHETIC markers placed in a bench source and in a model reply; neither may reach the metrics (OQ-018).
SENTINEL_SOURCE = "SENTINEL-bench-source-5e1c"
SENTINEL_REPLY = "SENTINEL-model-reply-a94d"

# SYNTHETIC bench sources for the layout item, one file per language.
OMP_SOURCE = '#include <cstdio>\nint main() {\n  std::printf("SYNTHETIC omp\\n");\n  return 0;\n}\n'
CUDA_SOURCE = "#include <cstdio>\n__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n"
SOURCES = {"omp": OMP_SOURCE, "cuda": CUDA_SOURCE}
# SYNTHETIC model code: the fake toolchains refuse a file holding an #error line and build anything else.
BAD_CODE = "#error SYNTHETIC not translated yet\nint main() {\n    return 1;\n}\n"
GOOD_CODE = "int main() {\n    return 0;\n}\n"
# SYNTHETIC program output in layout's print format (values invented) and SYNTHETIC wall times in seconds. The
# attempt's stdout differs from the reference's only in its timing lines.
LAYOUT = "Average kernel execution time (AoS): 1.5 (us)\nPASS\nAverage kernel execution time (SoA): 2.5 (us)\nPASS\n"
LAYOUT_OTHER_TIMES = LAYOUT.replace("1.5 (us)", "3.75 (us)").replace("2.5 (us)", "0.5 (us)")
REFERENCE_WALL_S = 1.25
ATTEMPT_WALL_S = 0.5


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate and compile variables a test could inherit, and point TMPDIR at a test directory."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CPATH", raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))


# ---------------------------------------------------------------------------
# Fake components


@dataclass
class Log:
    """The script the fakes follow and what they saw: model replies and attempt runs in order, and the requests."""

    replies: list[str] = field(default_factory=list)
    attempt_runs: list[RunResult] = field(default_factory=list)
    requests: list[list[Message]] = field(default_factory=list)


def fake_toolchain(registered_as: str) -> type:
    """Return a Toolchain class without PIN: a file holding `#error` fails, anything else builds a PLACEHOLDER."""

    class FakeToolchain:
        """Writes the files and reports a build or one SYNTHETIC error; compiles nothing."""

        name = registered_as
        capabilities = frozenset({"diagnostics"})

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Write every file under `workdir` and return the SYNTHETIC build result."""
            workdir = Path(workdir)
            for path, text in [*files.items(), *(harness or {}).items()]:
                target = workdir / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            if any("#error" in text for text in files.values()):
                error = Diagnostic(stage="compile", severity="error", code="synthetic", message="SYNTHETIC error")
                return BuildResult(artifact=None, diagnostics=[error])
            artifact = workdir / "main"
            artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=[])

    return FakeToolchain


def scripted_executor(log: Log) -> type:
    """Return an Executor class that runs programs: reference runs print LAYOUT, attempts follow the script."""

    class ScriptedExecutor:
        """Returns a SYNTHETIC RunResult for each run; runs nothing."""

        name = "scripted"
        capabilities = frozenset({"runs_code", "sandboxed"})

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Return the reference's run for a baseline build, else the next scripted attempt run."""
            if Path(artifact).parent.parent.name.startswith("baseline-"):
                return RunResult(exit_code=0, hang=False, stdout=LAYOUT, stderr="", wall_s=REFERENCE_WALL_S)
            assert log.attempt_runs, f"the executor was asked to run {artifact}, but the script holds no more runs"
            return log.attempt_runs.pop(0)

    return ScriptedExecutor


def scripted_backend(log: Log) -> type:
    """Return an LLMBackend class that answers from `log.replies` in order and records every request."""

    class ScriptedBackend:
        """Records each request and answers with the next scripted reply."""

        name = "scripted"
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id, as every backend does."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Record the request and return the next scripted reply."""
            log.requests.append(list(messages))
            assert log.replies, "the backend was asked for more replies than the script holds"
            return Completion(text=log.replies.pop(0), prompt_tokens=0, completion_tokens=0)

    return ScriptedBackend


def make_registry(log: Log) -> Registry:
    """Return a test Registry: scripted fakes, the real stages, oracle, none executor, and both ScoreProfiles.

    The scripted backend is registered as "scripted" and as "mock", so the
    unknown-metric fixture, which extends p0-smoke (the mock), asks it.
    """
    registry = Registry()
    backend = scripted_backend(log)
    registry.register("LLMBackend", "scripted", backend)
    registry.register("LLMBackend", "mock", backend)
    registry.register("Executor", "none", NoneExecutor)
    registry.register("Executor", "scripted", scripted_executor(log))
    for name in TOOLCHAINS.values():
        registry.register("Toolchain", name, fake_toolchain(name))
    registry.register("Oracle", ORACLE["kind"], DEFAULT_REGISTRY.get("Oracle", ORACLE["kind"]).factory)
    for name in RUN_STAGES:
        registry.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    for name in SCORE_PROFILES:
        registry.register("ScoreProfile", name, DEFAULT_REGISTRY.get("ScoreProfile", name).factory)
    return registry


# ---------------------------------------------------------------------------
# Recipes, bench sources, scripts, and runs


def write_bench(root: Path, sources: Mapping[str, str] = SOURCES) -> Path:
    """Write the item's SYNTHETIC source per language where the suite manifest lays it out; return `root`."""
    spec = load_suite(SUITE_MANIFEST).items[ITEM]
    for language, text in sources.items():
        layout = spec.languages[language]
        path = root / layout.dir / layout.files[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def blocks(code: str, direction: Direction) -> str:
    """Return a SYNTHETIC reply holding `code` as the direction's one target FILE block."""
    target = load_suite(SUITE_MANIFEST).items[ITEM].languages[direction.target].files[0]
    return render_file_blocks({target: code})


def clean_result() -> RunResult:
    """Return a SYNTHETIC attempt run that exited 0 and printed LAYOUT_OTHER_TIMES."""
    return RunResult(exit_code=0, hang=False, stdout=LAYOUT_OTHER_TIMES, stderr="", wall_s=ATTEMPT_WALL_S)


def recipe_data(
    stages: Sequence[str], directions: Sequence[Direction], executor: str, **changes: Any
) -> dict[str, Any]:
    """Return a p0-smoke template recipe for layout, fixes on, one trial per direction; `changes` add keys."""
    data: dict[str, Any] = {
        "extends": "base",
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": d.source, "target": d.target} for d in directions],
        "prompts": TEMPLATE_SET,
        "toolchain": dict(TOOLCHAINS),
        "stages": list(stages),
        "executor": {"kind": executor},
        "trials": {"n": 1},
    }
    if "oracle" in stages:
        data["oracle"] = dict(ORACLE)
    data.update(copy.deepcopy(changes))
    return data


def compile_only(**changes: Any) -> tuple[dict[str, Any], Log]:
    """Return a compile-only recipe over both directions and its script: each attempt 0 compiles (S4)."""
    directions = (CUDA_TO_OMP, OMP_TO_CUDA)
    data = recipe_data(COMPILE_STAGES, directions, "none", **changes)
    return data, Log(replies=[blocks(GOOD_CODE, direction) for direction in directions])


def clean_run(**changes: Any) -> tuple[dict[str, Any], Log]:
    """Return a recipe that runs programs (CUDA to OpenMP) and a script whose attempt 0 runs clean (S5)."""
    data = recipe_data(RUN_STAGES, (CUDA_TO_OMP,), "scripted", **changes)
    return data, Log(replies=[blocks(GOOD_CODE, CUDA_TO_OMP)], attempt_runs=[clean_result()])


def corrected_run(**changes: Any) -> tuple[dict[str, Any], Log]:
    """Return clean_run's recipe and a script whose attempt 0 fails to compile (S1) and attempt 1 runs clean (S5)."""
    data = recipe_data(RUN_STAGES, (CUDA_TO_OMP,), "scripted", **changes)
    replies = [blocks(BAD_CODE, CUDA_TO_OMP), blocks(GOOD_CODE, CUDA_TO_OMP)]
    return data, Log(replies=replies, attempt_runs=[clean_result()])


def write_recipe(directory: Path, name: str, data: Mapping[str, Any]) -> Path:
    """Write `data` as the recipe `<directory>/<name>.yaml` and return its path."""
    path = directory / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


@dataclass
class Outcome:
    """One finished run: its directory, the bench root it read, the script log, and its trials by trial id."""

    run_dir: Path
    bench_root: Path
    log: Log
    trials: list[Trial]

    def run_md(self) -> str:
        """Return run.md as ASCII text."""
        return (self.run_dir / "run.md").read_bytes().decode("ascii")


def run_scenario(
    tmp_path: Path, name: str, data: Mapping[str, Any], log: Log, sources: Mapping[str, str] = SOURCES
) -> Outcome:
    """Run the recipe `data`, named `name`, over the bench `sources` with the test registry; read the tree back."""
    bench = write_bench(tmp_path / "bench", sources)
    options = RunOptions(
        runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=bench, registry=make_registry(log)
    )
    run_dir = run_recipe(write_recipe(tmp_path, name, data), options)
    store = TextStore(run_dir)
    trials = sorted((read_trial(path.parent, store) for path in run_dir.rglob("trial.json")), key=lambda t: t.trial_id)
    assert log.replies == [], "precondition: the script was used up"
    return Outcome(run_dir=run_dir, bench_root=bench, log=log, trials=trials)


# ---------------------------------------------------------------------------
# Oracles: what the registered profiles and lassi.analysis give for the trials read back


def breakdown(score: Score) -> ScoreBreakdown:
    """Return a Score as the Result Record's ScoreBreakdown."""
    return ScoreBreakdown(components=dict(score.components), scalar=score.scalar)


def expected_tables(outcome: Outcome) -> list[MetricTable]:
    """Return the P2.8 tables of the run: lassi.analysis over the lassi profile's Score of every trial."""
    profile = LassiProfile(bench_root=outcome.bench_root)
    return metric_tables([(trial, profile.score(trial)) for trial in outcome.trials])


def metrics_section(run_md: str) -> str | None:
    """Return the Metrics section of run.md, from its heading to the next earlier-run.md section; None when absent."""
    lines = run_md.split("\n")
    if METRICS_HEADING not in lines:
        return None
    start = lines.index(METRICS_HEADING)
    end = next((index for index in range(start + 1, len(lines)) if lines[index] in RUN_MD_SECTIONS), len(lines))
    return "\n".join(lines[start:end])


def assert_tables_shown(section: str, tables: Sequence[MetricTable]) -> None:
    """Assert that `section` shows each table under a heading naming its arm and direction, line by line in order.

    Every non-blank line of table_markdown(table) that is not a heading
    must follow that heading, in order, before the next table's heading;
    the heading level is left to the runner.
    """
    lines = section.split("\n")
    titles = [f"{table.arm} {table.direction}" for table in tables]
    heading_at: dict[str, int] = {}
    for title in titles:
        found = [index for index, line in enumerate(lines) if line.startswith("#") and line.rstrip().endswith(title)]
        assert len(found) == 1, f"the Metrics section has {len(found)} headings naming {title!r}; expected one"
        heading_at[title] = found[0]
    starts = sorted(heading_at.values())
    for table, title in zip(tables, titles, strict=True):
        start = heading_at[title]
        end = next((index for index in starts if index > start), len(lines))
        region = lines[start + 1 : end]
        position = 0
        for line in table_markdown(table).split("\n"):
            if not line.strip() or line.startswith("#"):
                continue
            assert line in region[position:], f"{title}: the Metrics section lacks, or misorders, the line {line!r}"
            position = region.index(line, position) + 1


def attempt_sections(trial_md: str) -> list[str]:
    """Return the text of each attempt's section of trial.md, in order."""
    return trial_md.split("\n## Attempt ")[1:]


def md_table(section: str, heading: str) -> list[list[str]]:
    """Return the first Markdown table under the line `heading` in `section`: its rows as cells, header first.

    The separator row is dropped. The table ends at its first line that is
    not a table row; [] when no table comes before the next heading. Cells
    split at " | ", which an escaped pipe never forms.
    """
    lines = section.split("\n")
    assert heading in lines, f"the Metrics section has no line {heading!r}"
    rows: list[list[str]] = []
    for line in lines[lines.index(heading) + 1 :]:
        if line.startswith("|"):
            rows.append(line[2:-2].split(" | "))
        elif rows or line.startswith("#"):
            break
    return [row for row in rows if set(row) != {"---"}]


def metric_values(run_dir: Path) -> list[dict[str, Any]]:
    """Return the rows of the run's parquet/metric_values in order, after checking its columns."""
    table = pq.read_table(run_dir / METRIC_VALUES)
    assert table.column_names == VALUE_COLUMNS, f"{METRIC_VALUES} has the columns {table.column_names}"
    return table.to_pylist()


def component_scores(outcome: Outcome) -> list[tuple[Trial, dict[str, Score]]]:
    """Return each trial read back, in trial_id order, with its Score by each profile COMPONENT_PROFILES names."""
    profiles = {"lassi": LassiProfile(bench_root=outcome.bench_root), "df-v0": DfV0Profile()}
    return [(trial, {name: profile.score(trial) for name, profile in profiles.items()}) for trial in outcome.trials]


def as_stored(value: float | None) -> float | None:
    """Return a Score value as the runner checks it: a float, or None (lassi.scoring.score_run.score_trials)."""
    return None if value is None else float(value)


# ---------------------------------------------------------------------------
# score: the bound ScoreProfile fills Attempt.score and final.score


def test_score_df_v0_fills_each_attempt_score_and_the_final_score(tmp_path: Path) -> None:
    data, log = corrected_run(score="df-v0")
    outcome = run_scenario(tmp_path, "score-df-v0", data, log)
    (trial,) = outcome.trials
    assert [attempt.stage_reached for attempt in trial.attempts] == ["S1", "S5"], "precondition: one correction"
    profile = DfV0Profile()
    for attempt in trial.attempts:
        assert attempt.score == breakdown(profile.score_attempt(attempt)), f"attempt {attempt.index}"
        assert attempt.score.components and attempt.score.scalar is not None, f"attempt {attempt.index} is unscored"
    expected = profile.score(trial)
    assert expected.scalar is not None
    assert trial.final.score == expected.scalar, "final.score is df-v0's trial scalar (the multi-turn value)"


def test_score_lassi_sets_the_final_score_to_its_scalar_and_leaves_attempt_scores_unset(tmp_path: Path) -> None:
    data, log = clean_run(score="lassi")
    outcome = run_scenario(tmp_path, "score-lassi", data, log)
    (trial,) = outcome.trials
    expected = LassiProfile(bench_root=outcome.bench_root).score(trial)
    assert expected.scalar is not None, "precondition: a clean run the oracle aligned, so correct is set"
    assert trial.final.score == expected.scalar, "final.score is the lassi profile's scalar (correct)"
    assert all(attempt.score == ScoreBreakdown() for attempt in trial.attempts), (
        "the lassi profile scores trials, not attempts, so every Attempt.score stays unset"
    )


def test_trial_json_trial_md_and_the_parquet_mirror_show_the_scores(tmp_path: Path) -> None:
    data, log = corrected_run(score="df-v0")
    outcome = run_scenario(tmp_path, "score-shown", data, log)
    (trial,) = outcome.trials
    assert trial.final.score is not None and all(a.score.scalar is not None for a in trial.attempts)
    home = trial_dir(outcome.run_dir, trial.trial_id)
    raw = json.loads((home / "trial.json").read_bytes().decode("ascii"))
    assert raw["final"]["score"] == trial.final.score
    for attempt, stored in zip(trial.attempts, raw["attempts"], strict=True):
        assert stored["score"] == {"components": attempt.score.components, "scalar": attempt.score.scalar}
    text = (home / "trial.md").read_bytes().decode("ascii")
    assert f"| Score | {fmt(trial.final.score)} |\n" in text
    sections = attempt_sections(text)
    assert len(sections) == len(trial.attempts)
    for attempt, section in zip(trial.attempts, sections, strict=True):
        assert f"| scalar | {fmt(attempt.score.scalar)} |\n" in section, f"attempt {attempt.index}"
        for name, value in attempt.score.components.items():
            assert f"| {name} | {fmt(value)} |\n" in section, f"attempt {attempt.index}: {name}"
    tables = read_run_parquet(outcome.run_dir / "parquet")
    (row,) = tables["trials"]
    assert row["final_score"] == trial.final.score
    rows = sorted(tables["attempts"], key=lambda item: item["index"])
    assert [item["score_scalar"] for item in rows] == [attempt.score.scalar for attempt in trial.attempts]
    assert [json.loads(item["score_components"]) for item in rows] == [a.score.components for a in trial.attempts]


def test_importing_the_runner_registers_both_score_profiles() -> None:
    # A fresh interpreter, so this module's own imports of lassi.scoring cannot register them.
    code = (
        "import lassi.core.runner\n"
        "from lassi.core.registry import DEFAULT_REGISTRY\n"
        "names = DEFAULT_REGISTRY.names('ScoreProfile')\n"
        "missing = [name for name in ('df-v0', 'lassi') if name not in names]\n"
        "assert not missing, f'importing lassi.core.runner leaves unregistered: {missing}; registered: {names}'\n"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr[-2000:]


# ---------------------------------------------------------------------------
# metrics: the runner checks every name before any directory exists


def assert_refused(tmp_path: Path, recipe: Path, log: Log, registry: Registry, match: str) -> None:
    """Assert that running `recipe` raises RecipeError or RunError matching `match`, creating nothing, asking none."""
    runs_root = tmp_path / "runs-root"
    options = RunOptions(
        runs_root=runs_root, run_id="refused", bench_root=write_bench(tmp_path / "bench"), registry=registry
    )
    with pytest.raises((RecipeError, RunError), match=match):
        run_recipe(recipe, options)
    assert not runs_root.exists(), "a refused recipe creates no directory"
    assert log.requests == [], "a refused recipe asks no model"


def test_the_unknown_metric_fixture_names_one_metric() -> None:
    own = yaml.safe_load(UNKNOWN_METRIC.read_bytes().decode("ascii"))
    assert own["metrics"] == [NO_SUCH_METRIC]


def test_an_unknown_metric_is_refused_by_name_before_any_directory_is_created(tmp_path: Path) -> None:
    log = Log()
    assert_refused(tmp_path, UNKNOWN_METRIC, log, make_registry(log), NO_SUCH_METRIC)


def test_a_metric_named_twice_is_refused_by_name_before_any_directory_is_created(tmp_path: Path) -> None:
    data, log = compile_only(metrics=["sim_t", "correct", "sim_t"])
    recipe = write_recipe(tmp_path, "named-twice", data)
    assert_refused(tmp_path, recipe, log, make_registry(log), r"'sim_t' twice")


def declaring_profile(declared: object) -> type:
    """Return a ScoreProfile class that declares `declared` as its trial_components and never scores a trial."""

    class DeclaringProfile:
        """A SYNTHETIC ScoreProfile, built only to read its trial_components."""

        name = FAKE_PROFILE
        capabilities: frozenset[str] = frozenset()
        trial_components = declared

        def score(self, trial: Trial) -> Score:
            """Fail: the runner refuses this profile before any trial runs."""
            raise AssertionError(f"{FAKE_PROFILE} scored {trial.trial_id}, but it should have been refused")

    return DeclaringProfile


@pytest.mark.parametrize(
    "declared",
    ["correct", ["correct", 3], ["correct", ""], ["sim_t", "sim_t"], {"correct": "lassi"}, None],
    ids=["a-bare-string", "a-number", "an-empty-name", "a-name-twice", "a-mapping", "none"],
)
def test_a_score_profile_whose_trial_components_are_not_distinct_names_is_refused(
    tmp_path: Path, declared: object
) -> None:
    data, log = compile_only(metrics=["correct"])
    registry = make_registry(log)
    registry.register("ScoreProfile", FAKE_PROFILE, declaring_profile(declared))
    recipe = write_recipe(tmp_path, "bad-declaration", data)
    assert_refused(tmp_path, recipe, log, registry, rf"'{FAKE_PROFILE}' declares trial_components")


def provider_names() -> dict[str, list[str]]:
    """Return the metric names each provider kind offers: lassi and df-v0 trial components, the P2.8 rows."""
    groups = {
        "lassi-components": list(load_profile(PROFILE_FILE).components),
        "df-v0-components": [SINGLE_TURN, MULTI_TURN],
        "analysis-rows": list(METRIC_NAMES),
    }
    groups["all"] = [name for names in groups.values() for name in names]
    return groups


@pytest.mark.parametrize("group", ["lassi-components", "df-v0-components", "analysis-rows", "all"])
def test_every_metric_a_registered_provider_offers_is_accepted(tmp_path: Path, group: str) -> None:
    names = provider_names()[group]
    data, log = compile_only(metrics=names)
    outcome = run_scenario(tmp_path, "accepted", data, log)
    assert len(outcome.trials) == 2
    resolved = yaml.safe_load((outcome.run_dir / "recipe.resolved.yaml").read_bytes().decode("ascii"))
    assert resolved["metrics"] == names, "the resolved recipe keeps the metrics line (Design Principle 5)"


# ---------------------------------------------------------------------------
# metrics: metric-only scores never write Attempt.score or final.score


def test_metric_only_df_v0_scores_never_write_attempt_scores_or_the_final_score(tmp_path: Path) -> None:
    data, log = compile_only(metrics=[MULTI_TURN])
    outcome = run_scenario(tmp_path, "metric-df-v0", data, log)
    profile = DfV0Profile()
    for trial in outcome.trials:
        assert profile.score(trial).scalar is not None, "precondition: df-v0 gives this trial a scalar"
        assert trial.final.score is None, f"{trial.trial_id}: a metric-only score wrote final.score"
        assert all(attempt.score == ScoreBreakdown() for attempt in trial.attempts), (
            f"{trial.trial_id}: a metric-only score wrote Attempt.score"
        )


def test_score_df_v0_with_the_lassi_repro_metrics_keeps_the_df_v0_scores(tmp_path: Path) -> None:
    data, log = compile_only(score="df-v0", metrics=REPRO_METRICS)
    outcome = run_scenario(tmp_path, "score-and-metrics", data, log)
    df_v0, lassi = DfV0Profile(), LassiProfile(bench_root=outcome.bench_root)
    for trial in outcome.trials:
        expected = df_v0.score(trial)
        assert expected.scalar != lassi.score(trial).scalar, "precondition: the two profiles' scalars differ here"
        assert trial.final.score == expected.scalar, f"{trial.trial_id}: final.score is not df-v0's"
        for attempt in trial.attempts:
            assert attempt.score == breakdown(df_v0.score_attempt(attempt)), f"{trial.trial_id} {attempt.index}"


def test_metric_only_lassi_scores_never_write_the_final_score_of_a_run_that_ran_programs(tmp_path: Path) -> None:
    # Programs run here, so the lassi scalar (correct) is set; the scripted executor names no device.
    data, log = clean_run(metrics=REPRO_METRICS)
    outcome = run_scenario(tmp_path, "metric-lassi", data, log)
    (trial,) = outcome.trials
    assert LassiProfile(bench_root=outcome.bench_root).score(trial).scalar is not None, "precondition"
    assert trial.final.score is None, "a metric-only lassi score wrote final.score"
    assert all(attempt.score == ScoreBreakdown() for attempt in trial.attempts)


# ---------------------------------------------------------------------------
# metrics: run.md and the Parquet mirror


def test_run_md_gains_a_metrics_section_with_the_table_of_each_arm_and_direction(tmp_path: Path) -> None:
    data, log = compile_only(metrics=REPRO_METRICS)
    outcome = run_scenario(tmp_path, "metrics-md", data, log)
    tables = expected_tables(outcome)
    assert [(table.arm, table.direction) for table in tables] == [(MODEL_ID, "cuda-omp"), (MODEL_ID, "omp-cuda")]
    assert all(table.compile_only for table in tables), "precondition: compile-only trials give compile-stage tables"
    section = metrics_section(outcome.run_md())
    assert section is not None, f"run.md has no line {METRICS_HEADING!r}"
    assert_tables_shown(section, tables)
    assert section.count(COMPILE_STAGE) >= len(tables), "each compile-only table carries the compile-stage label"


def test_the_metrics_tables_join_the_parquet_mirror(tmp_path: Path) -> None:
    data, log = compile_only(metrics=REPRO_METRICS)
    outcome = run_scenario(tmp_path, "metrics-parquet", data, log)
    tables = expected_tables(outcome)
    parquet = outcome.run_dir / "parquet"
    for name in METRICS_TABLES:
        assert (parquet / name).is_dir(), f"parquet/{name} was not written"
    assert read_metrics_parquet(parquet) == metrics_rows(tables)
    assert len(read_run_parquet(parquet)["trials"]) == len(outcome.trials), "the trial tables stay beside them"


@pytest.mark.parametrize("score", [None, "df-v0"], ids=["no-score", "score-df-v0"])
def test_without_metrics_run_md_has_no_metrics_section_and_no_metrics_parquet(
    tmp_path: Path, score: str | None
) -> None:
    changes = {} if score is None else {"score": score}
    data, log = compile_only(**changes)
    outcome = run_scenario(tmp_path, "no-metrics", data, log)
    text = outcome.run_md()
    assert METRICS_HEADING not in text.split("\n")
    assert COMPILE_STAGE not in text
    for name in METRICS_TABLES:
        assert not (outcome.run_dir / "parquet" / name).exists(), f"parquet/{name} written without metrics"


def test_the_metrics_of_a_run_whose_executor_names_no_device(tmp_path: Path) -> None:
    # The runner records device None for every executor that runs programs, as for the native executor.
    data, log = clean_run(metrics=REPRO_METRICS)
    outcome = run_scenario(tmp_path, "metrics-no-device", data, log)
    (trial,) = outcome.trials
    assert trial.provenance.device is None, "precondition: the executor names no device"
    tables = expected_tables(outcome)
    assert [(table.arm, table.direction, table.compile_only) for table in tables] == [(MODEL_ID, "cuda-omp", False)]
    section = metrics_section(outcome.run_md())
    assert section is not None, f"run.md has no line {METRICS_HEADING!r}"
    assert_tables_shown(section, tables)
    assert read_metrics_parquet(outcome.run_dir / "parquet") == metrics_rows(tables)


# ---------------------------------------------------------------------------
# metrics: the named components' values and notes, in run.md and parquet/metric_values


def test_the_metric_values_table_holds_each_named_component_of_each_trial(tmp_path: Path) -> None:
    data, log = compile_only(metrics=COMPONENT_METRICS)
    outcome = run_scenario(tmp_path, "values-parquet", data, log)
    assert (outcome.run_dir / METRIC_VALUES).is_dir(), f"{METRIC_VALUES} was not written"
    expected = [
        {
            "trial_id": trial.trial_id,
            "profile": profile,
            "component": metric,
            "value": as_stored(scores[profile].components[metric]),
            "note": scores[profile].notes.get(metric) or None,
        }
        for trial, scores in component_scores(outcome)
        for metric, profile in COMPONENT_PROFILES.items()
    ]
    assert metric_values(outcome.run_dir) == expected


def test_the_per_trial_table_lists_each_named_component_of_each_trial(tmp_path: Path) -> None:
    data, log = compile_only(metrics=COMPONENT_METRICS)
    outcome = run_scenario(tmp_path, "values-md", data, log)
    section = metrics_section(outcome.run_md())
    assert section is not None, f"run.md has no line {METRICS_HEADING!r}"
    expected = [["Trial", *COMPONENT_METRICS]]
    for trial, scores in component_scores(outcome):
        values = (fmt(as_stored(scores[profile].components[metric])) for metric, profile in COMPONENT_PROFILES.items())
        expected.append([f"`{trial.trial_id}`", *values])
    assert md_table(section, VALUES_HEADING) == expected


@pytest.mark.parametrize("scenario", [compile_only, clean_run], ids=["compile-only", "clean-run"])
def test_every_metric_values_row_appears_in_the_metrics_section(
    tmp_path: Path, scenario: Callable[..., tuple[dict[str, Any], Log]]
) -> None:
    data, log = scenario(metrics=COMPONENT_METRICS)
    outcome = run_scenario(tmp_path, "values-shown", data, log)
    rows = metric_values(outcome.run_dir)
    assert any(row["note"] is not None and row["value"] is None for row in rows), "precondition: a noted null"
    assert any(row["note"] is not None and row["value"] is not None for row in rows), "precondition: a noted value"
    assert any(row["note"] is None for row in rows), "precondition: a value without a note"
    section = metrics_section(outcome.run_md())
    assert section is not None, f"run.md has no line {METRICS_HEADING!r}"
    sources = {metric: source for metric, source in md_table(section, METRICS_HEADING)[1:]}
    header, *trial_rows = md_table(section, VALUES_HEADING)
    shown = {row[0]: dict(zip(header, row, strict=True)) for row in trial_rows}
    expected_notes = []
    for row in rows:
        trial, where = f"`{row['trial_id']}`", f"{row['trial_id']} {row['component']}"
        assert f"ScoreProfile {row['profile']}" in sources[row["component"]], f"{where}: the source names no profile"
        assert shown[trial][row["component"]] == fmt(row["value"]), f"{where}: run.md shows another value"
        if row["note"] is not None:
            expected_notes.append([trial, row["profile"], row["component"], row["note"]])
    assert md_table(section, NOTES_HEADING) == [NOTES_HEADER, *expected_notes], "the notes table differs"


def test_a_run_whose_named_components_carry_no_note_shows_an_empty_notes_part(tmp_path: Path) -> None:
    data, log = compile_only(metrics=[MULTI_TURN])
    outcome = run_scenario(tmp_path, "no-notes", data, log)
    rows = metric_values(outcome.run_dir)
    assert len(rows) == len(outcome.trials) and all(row["note"] is None for row in rows), "precondition"
    section = metrics_section(outcome.run_md())
    assert section is not None, f"run.md has no line {METRICS_HEADING!r}"
    assert md_table(section, NOTES_HEADING) == [], "a notes table without notes"


def test_a_run_that_names_no_trial_component_writes_no_metric_values(tmp_path: Path) -> None:
    data, log = compile_only(metrics=list(METRIC_NAMES))
    outcome = run_scenario(tmp_path, "rows-only", data, log)
    assert not (outcome.run_dir / METRIC_VALUES).exists(), f"{METRIC_VALUES} written without a named component"
    section = metrics_section(outcome.run_md())
    assert section is not None, f"run.md has no line {METRICS_HEADING!r}"
    lines = section.split("\n")
    assert VALUES_HEADING not in lines and NOTES_HEADING not in lines


def test_no_source_or_reply_text_reaches_the_metrics_section_or_metric_values(tmp_path: Path) -> None:
    directions = (CUDA_TO_OMP, OMP_TO_CUDA)
    sources = {language: f"/* {SENTINEL_SOURCE} */\n{text}" for language, text in SOURCES.items()}
    data = recipe_data(COMPILE_STAGES, directions, "none", metrics=COMPONENT_METRICS)
    log = Log(replies=[blocks(f"/* {SENTINEL_REPLY} */\n{GOOD_CODE}", direction) for direction in directions])
    outcome = run_scenario(tmp_path, "no-text", data, log, sources)
    stored = [path.read_bytes().decode("utf-8") for path in (outcome.run_dir / "texts").rglob("*.txt")]
    section = metrics_section(outcome.run_md())
    assert section is not None, f"run.md has no line {METRICS_HEADING!r}"
    cells = [str(value) for row in metric_values(outcome.run_dir) for value in row.values()]
    for sentinel in (SENTINEL_SOURCE, SENTINEL_REPLY):
        assert any(sentinel in text for text in stored), f"precondition: the text store holds {sentinel}"
        assert sentinel not in section, f"run.md's Metrics section holds {sentinel}"
        assert not any(sentinel in cell for cell in cells), f"{METRIC_VALUES} holds {sentinel}"
