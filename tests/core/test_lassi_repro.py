"""Tests for the lassi-repro recipe, the P1 dry-run fixture, and the mock dry run (task P1.10).

Bible: Project Recipes (the lassi-repro block and Notes), Design Principles
1, 4, and 5, Readability Standards (Naming, Config), Evaluation Protocol
(Acceptance Criteria), Risks And Questions (mock LLM).

The contract these tests fix, from the P1.10 acceptance criteria
(plans/p1-faithful.md), its planning decisions, and the P1 phase notes
("For P1.10 (from P1.5)", "Run report fragments", "The mock tags"):

- projects/lassi-repro/recipe.yaml sets `project: lassi-repro` and matches
  the bible's lassi-repro block, except the keys P1 cannot carry out, which
  it leaves out with a comment naming their phase: `arms` (P3) and the
  `gpu` executor (P10), in whose place it keeps the block's tier-1
  executor `{kind: none}`. Its `metrics` line, left out until P2, is the
  block's since task P2.10 (tests/core/test_lassi_repro_metrics.py). It
  sets no other key. It leaves llm.sampling.max_tokens unset with a comment
  naming P3 (upstream sets none). Its stages are the block's, baseline
  first and run_loop listed (every faithful: true recipe must list it). It
  binds no model (arms are P3), so it loads once a child names the model
  and nothing else.
- tests/fixtures/recipes/p1-dry-run.yaml extends it with the mock backend,
  a max_tokens value for the mock, executor `none`, all 10 items, both
  directions, and n = 1. Its resolved project is lassi-repro, so its trial
  ids start with `lassi-repro/`. Since task P2.10 it also binds
  `score: df-v0`, so the test registry registers both ScoreProfiles.
- Under faithful extraction (fixes.fence_tag off) the mock answers in one
  untagged fence, the form upstream's system prompts ask for, so its
  attempt 0 holds the reference target with no fence-quirk or no-fence
  Diagnostic.
- A local dry run with a fake toolchain reaches S4 at attempt 0 for all 20
  trials (10 apps x 2 directions), after the baseline built each trial's
  target reference; a reference that fails to compile ends its trial with
  `baseline-compile` before any model call.
- Both recipe files are plain ASCII with a comment on every value
  (Readability Standards, Config row).

The dry runs use the lassi-2024 prompt set from a fresh extraction of the
pinned upstream checkout into a temporary directory
(tools/extract_lassi_assets.py); they skip, naming that tool, when
third_party/LASSI is absent or not at the pin. No upstream text is copied
into this file (OQ-018). Bench sources are SYNTHETIC programs written here,
not HeCBench sources; the fake toolchains compile nothing and report a
PLACEHOLDER artifact; the none executor runs nothing. No value in this
module is a measurement.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

import lassi.prompts as prompts_module
import lassi.prompts.assets as prompt_assets
from lassi.bench import load_suite
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.interfaces import BuildResult
from lassi.core.recipe import FIXES, UNCAPPED, load_recipe
from lassi.core.record import Diagnostic, Trial, arm_segment
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.executors.none import NoneExecutor
from lassi.llm import MockBackend

REPO = Path(__file__).resolve().parents[2]
BIBLE = REPO / "docs" / "BIBLE.md"
PROJECTS = REPO / "projects"
FIXTURES = REPO / "tests" / "fixtures" / "recipes"
LASSI_REPRO = PROJECTS / "lassi-repro" / "recipe.yaml"
DRY_RUN = FIXTURES / "p1-dry-run.yaml"
BIBLE_BLOCK = "projects/lassi-repro/recipe.yaml"
PROJECT = "lassi-repro"

UPSTREAM_DIR = REPO / "third_party" / "LASSI"
UPSTREAM_PIN = "74b46812523f2ff79b53b6880a4521690d7478b0"
EXTRACTOR = REPO / "tools" / "extract_lassi_assets.py"
SKIP_REASON = (
    "needs the upstream checkout at third_party/LASSI on commit 74b4681 so that tools/extract_lassi_assets.py "
    "can generate the lassi-2024 fragments; run uv run tools/fetch_upstream.py first"
)

SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
DIRECTIONS = ({"source": "omp", "target": "cuda"}, {"source": "cuda", "target": "omp"})
# The bible block's stages, which P1 carries out in full: baseline first, run_loop listed.
STAGES = ("baseline", "summarize_context", "describe_source", "generate", "compile_loop", "run_loop", "oracle")
# The block's tier-1 executor (its comment on the executor line): compile only.
TIER_1_EXECUTOR = {"kind": "none"}
# Keys of the bible block the recipe leaves out, and the phase each comment names (metrics joined in P2.10).
LEFT_OUT_KEYS = {"arms": "P3"}
# Every left-out choice and the phase its comment names: the key, the gpu executor, and max_tokens.
LEFT_OUT_COMMENTS = (("arms", "P3"), ("gpu", "P10"), ("max_tokens", "P3"))
# The ScoreProfiles p1-dry-run.yaml binds (score: df-v0) or its metrics line needs (lassi), registered for its runs.
SCORE_PROFILES = ("df-v0", "lassi")
# The fake toolchains are registered under the names lassi-repro binds, one per target language.
TOOLCHAIN_OF = {"cuda": "nvcc-sm80", "omp": "nvcpp-cc80"}
COMPILED = "S4"
BASELINE_COMPILE = "baseline-compile"
FENCE_DIAGNOSTICS = ("fence-quirk", "no-fence")
# A line holding this marker makes the fake toolchain report a failed build.
BROKEN = "SYNTHETIC-BROKEN-REFERENCE"
FENCE = "```"


# ---------------------------------------------------------------------------
# Reading the recipe files and the bible


def read_ascii(path: Path) -> str:
    """Return a file's text; fail clearly while task P1.10 has not written it."""
    if not path.is_file():
        pytest.fail(f"{path.relative_to(REPO).as_posix()} does not exist; task P1.10 adds it")
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


def comments(text: str) -> list[str]:
    """Return the comment part of every line that has one."""
    return [line.split("#", 1)[1] for line in text.split("\n") if "#" in line]


def suite_items() -> list[str]:
    """Return every item of the suite's eval split, sorted."""
    suite = load_suite(SUITE_MANIFEST)
    return sorted(name for name, item in suite.items.items() if item.split == "eval")


# ---------------------------------------------------------------------------
# projects/lassi-repro/recipe.yaml


def test_lassi_repro_sets_its_project_key() -> None:
    assert own_keys(LASSI_REPRO).get("project") == PROJECT


def test_lassi_repro_matches_the_bible_block_except_what_p1_cannot_carry_out() -> None:
    own, block = own_keys(LASSI_REPRO), bible_block(BIBLE_BLOCK)
    for key, value in block.items():
        if key in LEFT_OUT_KEYS or key == "executor":
            continue
        assert own.get(key) == value, f"{key}: the recipe has {own.get(key)!r}, the bible block {value!r}"
    for key in LEFT_OUT_KEYS:
        assert key not in own, f"{key} is a later phase's key ({LEFT_OUT_KEYS[key]}); leave it out"
    assert own.get("executor") == TIER_1_EXECUTOR, "the gpu executor is P10; keep the block's tier-1 executor"
    extra = sorted(set(own) - set(block) - {"project"})
    assert extra == [], f"keys the bible block does not set: {extra}"


def test_lassi_repro_lists_baseline_first_and_every_stage_p1_carries_out() -> None:
    own = own_keys(LASSI_REPRO)
    assert own.get("faithful") is True
    assert tuple(own.get("stages", ())) == STAGES


@pytest.mark.parametrize(("word", "phase"), LEFT_OUT_COMMENTS, ids=[word for word, _ in LEFT_OUT_COMMENTS])
def test_each_choice_left_out_has_a_comment_naming_its_phase(word: str, phase: str) -> None:
    found = [text for text in comments(read_ascii(LASSI_REPRO))
             if re.search(rf"\b{re.escape(word)}\b", text) and re.search(rf"\b{phase}\b", text)]
    assert found, f"no comment in lassi-repro/recipe.yaml names {word} with its phase {phase}"


def test_lassi_repro_loads_once_a_child_names_the_model(tmp_path: Path) -> None:
    read_ascii(LASSI_REPRO)
    child = tmp_path / "model-only.yaml"
    child.write_bytes(b"extends: lassi-repro\nmodel: {backend: mock, id: mock-reference}\n")
    recipe = load_recipe(child)
    data = recipe.data
    assert recipe.chain == ("base", PROJECT, "model-only")
    assert data["project"] == PROJECT, "the nearest file that sets project wins over the loaded file's name"
    assert data["faithful"] is True
    assert data["loop"]["max_corrections"] == UNCAPPED
    assert data["fixes"] == dict.fromkeys(FIXES, False)
    assert "max_tokens" not in data["llm"]["sampling"], "lassi-repro leaves max_tokens unset (P3)"
    assert data["executor"] == TIER_1_EXECUTOR
    assert tuple(data["stages"]) == STAGES


@pytest.mark.parametrize("path", [LASSI_REPRO, DRY_RUN], ids=["lassi-repro", "p1-dry-run"])
def test_the_recipes_comment_every_value_in_plain_ascii(path: Path) -> None:
    text = read_ascii(path)
    assert "\r" not in text and text.endswith("\n")
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.endswith(":"):
            continue
        assert "#" in stripped, f"{path.name}: no comment on {line!r}"


# ---------------------------------------------------------------------------
# tests/fixtures/recipes/p1-dry-run.yaml


def test_the_dry_run_recipe_extends_lassi_repro_with_the_mock() -> None:
    own = own_keys(DRY_RUN)
    assert own.get("extends") == PROJECT or str(own.get("extends", "")).endswith("lassi-repro/recipe.yaml")
    recipe = load_recipe(DRY_RUN)
    data = recipe.data
    assert recipe.chain == ("base", PROJECT, "p1-dry-run")
    assert data["project"] == PROJECT
    assert data["model"]["backend"] == "mock"
    max_tokens = data["llm"]["sampling"].get("max_tokens")
    assert isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens >= 1
    assert data["executor"] == {"kind": "none"}
    assert data["trials"] == {"n": 1}
    assert [dict(entry) for entry in data["directions"]] == list(DIRECTIONS)
    assert sorted(data["bench"].get("items", suite_items())) == suite_items() and len(suite_items()) == 10
    assert data["faithful"] is True and tuple(data["stages"]) == STAGES


# ---------------------------------------------------------------------------
# Upstream extraction, bench sources, fakes, and runs


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
    spec = importlib.util.spec_from_file_location("extract_lassi_assets_for_p110", EXTRACTOR)
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


def synthetic_source(item: str, language: str, broken: bool = False) -> str:
    """Return a SYNTHETIC program standing in for one bench file; `broken` adds the line the fake build fails on."""
    kernel = "__global__ void k() { }\n" if language == "cuda" else ""
    launch = "  k<<<1, 1>>>();\n" if language == "cuda" else ""
    marker = f"// {BROKEN}\n" if broken else ""
    return (
        f"// SYNTHETIC {language} stand-in for {item}, written by tests/core/test_lassi_repro.py\n"
        f"{marker}#include <cstdio>\n{kernel}int main() {{\n{launch}"
        f'  std::printf("{item} {language}\\n");\n  return 0;\n}}\n'
    )


def write_bench(root: Path, broken: frozenset[tuple[str, str]] = frozenset()) -> Path:
    """Write every item's SYNTHETIC sources and support files where the suite manifest lays them out.

    `broken` holds the (item, language) files that carry the BROKEN marker. Returns `root`.
    """
    suite = load_suite(SUITE_MANIFEST)
    for name, item in suite.items.items():
        for language, layout in item.languages.items():
            for file in layout.files:
                path = root / layout.dir / file
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(synthetic_source(name, language, (name, language) in broken).encode("ascii"))
        for file, relative in item.support.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"// SYNTHETIC support file {file} for {name}\n".encode("ascii"))
    return root


@dataclass(frozen=True)
class Build:
    """One build a fake toolchain was asked for: its toolchain, build directory, files, and harness files."""

    toolchain: str
    workdir: Path
    files: dict[str, str]
    harness: dict[str, str]


def fake_toolchain(name: str, builds: list[Build]) -> type:
    """Return a Toolchain class without PIN that logs each build into `builds` and compiles nothing.

    A build whose files hold the BROKEN marker fails with one SYNTHETIC error
    Diagnostic; any other gets a PLACEHOLDER artifact path, and nothing is written.
    """

    class FakeToolchain:
        """Compiles nothing: logs the build and reports a PLACEHOLDER artifact, or a SYNTHETIC failure."""

        capabilities = frozenset({"diagnostics", "emits_warnings"})

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Log the build; fail it when a file holds the BROKEN marker, else report a PLACEHOLDER artifact."""
            builds.append(Build(name, Path(workdir), dict(files), dict(harness or {})))
            broken = sorted(path for path, text in files.items() if BROKEN in text)
            if broken:
                error = Diagnostic(
                    stage="compile", severity="error", file=broken[0], line=2,
                    message="SYNTHETIC error from the fake toolchain",
                )
                return BuildResult(artifact=None, diagnostics=[error])
            return BuildResult(artifact=Path(workdir) / "PLACEHOLDER-artifact", diagnostics=[])

    FakeToolchain.__name__ = f"FakeToolchain_{name.replace('-', '_')}"
    return FakeToolchain


def make_registry(builds: list[Build]) -> Registry:
    """Return a test Registry: the real mock, stages, none executor, oracle, and ScoreProfiles; fake toolchains."""
    registry = Registry()
    registry.register("LLMBackend", "mock", MockBackend)
    registry.register("Executor", "none", NoneExecutor)
    registry.register("Oracle", "stdout_mask", DEFAULT_REGISTRY.get("Oracle", "stdout_mask").factory)
    for name in TOOLCHAIN_OF.values():
        registry.register("Toolchain", name, fake_toolchain(name, builds))
    for name in STAGES:
        registry.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    for name in SCORE_PROFILES:
        registry.register("ScoreProfile", name, DEFAULT_REGISTRY.get("ScoreProfile", name).factory)
    return registry


@dataclass
class DryRun:
    """One finished run (or the error that stopped it): its directory, trials by id, builds, and bench root."""

    run_dir: Path | None
    bench_root: Path
    builds: list[Build]
    trials: dict[str, Trial] = field(default_factory=dict)
    error: BaseException | None = None

    def checked(self) -> DryRun:
        """Return self, failing the test with the run's error when it did not complete."""
        if self.error is not None:
            pytest.fail(f"the dry run did not complete: {type(self.error).__name__}: {self.error}")
        return self


def isolate(patch: pytest.MonkeyPatch, root: Path, assets: Path) -> None:
    """Unset the gate variables, point TMPDIR under `root`, and point the prompt assets at the fresh extraction."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        patch.delenv(name, raising=False)
    tmpdir = root / "compile-tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    patch.setenv("TMPDIR", str(tmpdir))
    patch.setattr(prompt_assets, "default_root", lambda: assets)
    patch.setattr(prompts_module, "default_roots", lambda: (assets / "prompts", REPO / "assets" / "prompts"))


def dry_run(recipe: Path, root: Path, assets: Path, broken: frozenset[tuple[str, str]] = frozenset()) -> DryRun:
    """Run `recipe` with the test registry over SYNTHETIC sources under `root`; keep any error it raised."""
    builds: list[Build] = []
    outcome = DryRun(run_dir=None, bench_root=write_bench(root / "bench", broken), builds=builds)
    registry = make_registry(builds)
    options = RunOptions(
        runs_root=root / "runs", run_id="dry-run", bench_root=outcome.bench_root, registry=registry,
        roots=(FIXTURES, PROJECTS),
    )
    with pytest.MonkeyPatch.context() as patch:
        isolate(patch, root, assets)
        try:
            outcome.run_dir = run_recipe(recipe, options)
        except Exception as error:  # the tests report it through DryRun.checked
            outcome.error = error
            return outcome
    store = TextStore(outcome.run_dir)
    for path in sorted(outcome.run_dir.rglob("trial.json")):
        trial = read_trial(path.parent, store)
        outcome.trials[trial.trial_id] = trial
    return outcome


@pytest.fixture(scope="module")
def full_run(assets_root: Path, tmp_path_factory: pytest.TempPathFactory) -> DryRun:
    """Run tests/fixtures/recipes/p1-dry-run.yaml once for this module; each test reports a run that failed."""
    return dry_run(DRY_RUN, tmp_path_factory.mktemp("p1-dry-run"), assets_root)


def mock_arm() -> str:
    """Return the arm segment of the dry run's mock model id."""
    return arm_segment(load_recipe(DRY_RUN).data["model"]["id"])


def expected_ids(items: list[str]) -> list[str]:
    """Return the trial ids a run of `items` in both directions writes, sorted."""
    arm = mock_arm()
    return sorted(
        f"{PROJECT}/{arm}/{SUITE}/{d['source']}-{d['target']}/{item}/run01" for d in DIRECTIONS for item in items
    )


def reference_target(bench_root: Path, trial: Trial) -> dict[str, str]:
    """Return the trial's target reference files as the bench holds them."""
    suite = load_suite(SUITE_MANIFEST)
    item = suite.item(trial.bench_item.item, purpose="eval")
    target = trial.trial_id.split("/")[3].split("-")[1]
    layout = item.languages[target]
    return {file: (bench_root / layout.dir / file).read_bytes().decode("ascii") for file in layout.files}


# ---------------------------------------------------------------------------
# The dry run


def test_the_dry_run_writes_20_trials_whose_ids_start_with_lassi_repro(full_run: DryRun) -> None:
    run = full_run.checked()
    assert sorted(run.trials) == expected_ids(suite_items())
    assert len(run.trials) == 20


def test_every_dry_run_trial_reaches_s4_at_attempt_0(full_run: DryRun) -> None:
    run = full_run.checked()
    assert run.trials, "the dry run wrote no trials"
    for trial_id, trial in run.trials.items():
        assert trial.final.end_reason is None, f"{trial_id}: ended {trial.final.end_reason}"
        assert len(trial.attempts) == 1, f"{trial_id}: {len(trial.attempts)} attempts"
        assert trial.attempts[0].stage_reached == COMPILED, f"{trial_id}: {trial.attempts[0].stage_reached}"
        assert (trial.final.stage_reached, trial.final.corrections) == (COMPILED, 0), trial_id


def test_the_baseline_builds_each_trials_target_reference(full_run: DryRun) -> None:
    run = full_run.checked()
    for trial_id, trial in run.trials.items():
        home = trial_dir(run.run_dir, trial_id)
        reference = reference_target(run.bench_root, trial)
        built = [b for b in run.builds if b.files == reference and home in b.workdir.parents]
        assert len(built) == 1, f"{trial_id}: {len(built)} builds of its target reference"


def test_the_mock_answers_in_one_untagged_fence_under_faithful_extraction(full_run: DryRun) -> None:
    run = full_run.checked()
    for trial_id, trial in run.trials.items():
        attempt = trial.attempts[0]
        reply = attempt.response_text
        assert reply.count(FENCE) == 2, f"{trial_id}: the reply holds {reply.count(FENCE)} runs of three backticks"
        tag = reply.split(FENCE, 1)[1].split("\n", 1)[0]
        assert tag.strip() == "", f"{trial_id}: the fence carries the tag {tag!r}"
        codes = [d.code for d in attempt.diagnostics if d.code in FENCE_DIAGNOSTICS]
        assert codes == [], f"{trial_id}: {codes}"
        (reference,) = reference_target(run.bench_root, trial).items()
        assert list(attempt.files) == [reference[0]], trial_id
        assert attempt.files[reference[0]].strip("\n") == reference[1].strip("\n"), trial_id


def test_a_reference_that_fails_to_compile_ends_its_trial_with_baseline_compile(
    assets_root: Path, tmp_path: Path
) -> None:
    read_ascii(DRY_RUN)
    broken_item, other_item = "layout", "bsearch"
    child = tmp_path / "recipes" / "broken-reference.yaml"
    child.parent.mkdir()
    child.write_bytes(
        f"extends: p1-dry-run\nbench: {{items: [{broken_item}, {other_item}]}}\n".encode("ascii")
    )
    run = dry_run(child, tmp_path / "work", assets_root, frozenset({(broken_item, "cuda")})).checked()
    arm = mock_arm()
    ended = f"{PROJECT}/{arm}/{SUITE}/omp-cuda/{broken_item}/run01"
    assert sorted(run.trials) == expected_ids([broken_item, other_item])
    trial = run.trials[ended]
    assert trial.final.end_reason is not None and trial.final.end_reason.code == BASELINE_COMPILE
    assert trial.attempts == [], "the trial ends before any model call"
    assert (trial.context.knowledge_summary, trial.context.source_description) == ("", "")
    for trial_id, other in run.trials.items():
        if trial_id != ended:
            assert other.final.end_reason is None, f"{trial_id}: ended {other.final.end_reason}"
            assert [a.stage_reached for a in other.attempts] == [COMPILED], trial_id
