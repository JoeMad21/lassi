"""The stage runner behind `lassi run <recipe>`: load a recipe, run its stages over every trial, write the run tree.

run_recipe (bible Project Recipes; Component Interfaces; Result Record):

1. loads the recipe with the registry (every component registers when this
   module is imported, since it imports lassi.llm, lassi.toolchains,
   lassi.executors, lassi.core.stages, lassi.core.oracle_stage, which
   imports lassi.oracles, and lassi.scoring, which registers every
   ScoreProfile);
2. refuses what it cannot run, before any directory is created or any model
   is asked: arms without a model (the model registry that maps arms to
   backends comes later) and arms beside a model; a recipe with no
   llm.sampling.max_tokens or no prompt set, since it never picks a value;
   a fix toggle turned off (faithful: true turns them all off) when no
   listed stage names that fix in `reproduces`, a report toggle turned off,
   or a section this runner does not carry out (judges and the like),
   since it never ignores a choice; a stage list whose order cannot
   work (one generating stage at most, every compiling stage after it, and
   every stage that builds the reference programs, `builds_references`,
   before any stage that asks the model);
   a stage that builds the source reference under a fix that is on
   (`source_build_fix`, as baseline declares for baseline_both) when no
   toolchain is bound for a direction's source language;
   a template prompt set that lacks a template a stage renders or uses a
   placeholder the stage does not fill; context packs with a prompt set
   that is not a fragment set; a fragment prompt set or context packs that
   do not load (lassi.prompts.load_recipe_assets, which checks every file
   against its manifest), two packs that serve one language, a fragment a
   stage needs for a direction, a context pack a stage needs for a target
   language, and a Trial.context field a stage joins into its prompt that
   no earlier stage fills; a project (the recipe's `project`, the first
   trial id segment) that is not one plain segment or that names an entry
   of the run tree; trial ids that repeat; missing bench sources; an
   item with more than one source file under a fragment set, or more than
   one target file with fixes.fence_tag off, since each reads one file; and
   an oracle section that no listed stage uses (none names Oracle in
   `requires`), whose Oracle declares none of the capabilities a listed
   stage accepts (`oracle_capabilities`), or that its Oracle refuses
   (factory(**config) raises ValueError, as for an unset oracle.passfail or
   oracle.threshold); an Oracle that reads the references' agreement
   (`agreement_setting`, binary_io's threshold from_baseline) while
   fixes.baseline_both is off, no listed stage builds the references, or an
   item of the run declares no tolerance in its suite manifest; a backend that declares
   `unload_before_run` but has no unload(), or `needs_reference` but has no
   with_reference() (or, with fixes.fence_tag off, no with_untagged_fence());
   a `score` profile or, when the recipe names metrics, any registered
   ScoreProfile that cannot be built, and a `metrics` name that no
   registered provider offers, that two offer, or that is named twice
   (lassi.scoring.run_scoring.plan_scoring);
   and, when any bound executor runs programs, a sandbox.wall_s that is
   neither a number of seconds above 0 nor `baseline_x10`, or a
   sandbox.mem_gb that is not above 0 (the limits of every run,
   lassi.core.stages attempt_limits and reference_limits), and each bound
   executor that runs programs without `sandboxed` when a listed stage
   declares `runs_model_code` (Agent Rule 6); with executors bound per
   language (lassi.core.recipe, task P4.5), a direction's target language
   with no executor, and its source language with none when a listed stage
   builds the source reference under its `source_build_fix` (baseline under
   baseline_both), each message naming executor.<language>; and a device()
   that returns neither None nor one non-empty line of printable ASCII with
   no leading or trailing blank.
   A stage it does not implement already fails at load, unregistered;
3. builds the components: the backend as factory(model.id), each toolchain
   with its pinned compiler and a clean environment (below), each executor
   as factory(**config) (one, or one per language), asking each for its
   device(), and the ScoreProfiles `score` and `metrics` need
   (lassi.scoring.profiles.build_profile, with the bench root for a profile
   that declares reads_bench_sources);
4. loads the suite manifest assets/bench/<bench.suite>.yaml and finds the
   fetched sources (tools/fetch_bench.py puts them under $LASSI_SCRATCH);
5. runs every trial, direction by direction in recipe order, item by item in
   sorted order (the items bench.items selects, else every item of the
   recipe's split; an unknown item, one outside the split, or a repeat is a
   RunError before anything is written), then run 1 to trials.n: a backend
   that declares `needs_reference` gets the item's reference target, with
   with_untagged_fence() when fixes.fence_tag is off (the mock's faithful
   reply form), a backend that declares `unload_before_run` is asked to
   unload (upstream's setup unload, lassi.core.capabilities), then the
   stages run in recipe order on a fresh RunContext (which carries the
   prompt set's fragments, the context packs by language, the target
   language's executor, which runs every attempt, and, with executors per
   language, every language's executor, so the baseline runs each reference
   on its own language's), then the
   trial's final block, whose alignment is that of the attempt whose output
   stands (_final). A stage that sets final.end_reason ends the trial:
   no later stage runs, and the final block keeps the end reason. With
   `score`, the bound profile then sets final.score, and each Attempt.score
   when it declares scores_attempts, before the trial is written;
6. with `metrics`, scores every trial again with each profile that gives a
   named metric (these Scores stay out of the trials) and builds the run
   metric tables (lassi.scoring.run_scoring.compute_metrics);
7. writes the run tree and prints one line per trial and the run directory.

The run tree is <runs root>/runs/<run_id>, where the runs root is the
runs_root option, else $LASSI_RUNS_ROOT, else the recipe's runs_root. The
runs root and the run directory must be absolute, resolve outside the
repository, and, when $LASSI_SCRATCH is set (the gate sets it on the build
host), resolve inside it (Agent Rule 7). The tree holds:

- recipe.resolved.yaml: the resolved recipe, which reruns the run alone
  (Design Principle 5);
- toolchains.json: per toolchain, its languages, pinned executable, compile
  environment, and pin files;
- provenance.json: the commit, the dirty flag, host, Python, the executor
  and its device, the start and end times (UTC), recipe hash, pin versions,
  and a status. The single executor form records `executor` (its name) and
  `device` (its device(), null for an executor that names none); executors
  per language record `executor` and `devices`, each a mapping from language
  to that language's executor name and device. It is
  written before the first trial with status "running", and again at the
  end with "complete", or with "failed" when a trial or a write raised, so
  every trial.json in the tree has a commit and a date beside it. Each
  trial's provenance copies the commit, dirty flag, device (the device of
  its target language with executors per language), driver (as sdk),
  and started_utc (as date) of the manifest written first, and the final
  manifest is that same manifest with only its status and finish time
  changed, so a trial and its manifest never disagree;
- run.md: the page a person reads (Readability Standards, Run row). Its
  summary shows the manifest's provenance, with an unknown value (null) as
  "-" as trial.md shows it, since provenance is not a measurement; with
  executors per language, one Device row per language. When a direction's
  target language, or its source language when the baseline runs the
  source reference too, runs on an executor that declares `simulator`
  (lassi.core.capabilities SIMULATOR), its Trials section opens with a
  note naming those languages: their run wall times are simulator wall
  time, not performance, and the wall_s column is each trial's pipeline
  wall time (task P4.6). With
  `metrics` it ends in a Metrics section
  (lassi.scoring.run_scoring.metrics_section);
- one directory per trial (one level per trial_id segment) with trial.json,
  trial.md (each run's wall_s row labeled as simulator wall time when the
  trial's target-language executor declares `simulator`), and
  attempt<NN>/build, each attempt's fresh build directory
  (the run directory is the stages' build root, so a rerun of the recipe
  never meets an earlier run's builds);
- texts/, the text store of prompts and replies; parquet/, the mirror, which
  with `metrics` also holds the metric tables and the named components'
  values (lassi.scoring.run_scoring.write_metrics).

Pinned toolchains (Agent Rule 10): a toolchain class that declares PIN and
PIN_BIN is built as factory(executable=<toolchains root>/<PREFIX_NAME>/
<PIN_BIN>, runner=SandboxedCompileRunner(...)); one that declares PIN and no
PIN_BIN uses a host compiler the project does not install, and its
executable is the pin's EXECUTABLE, an absolute path taken as given
(_host_executable; toolchains/gcc.pin). Such a class that also defines
check_tree builds against the pin's installed tree, <resolved toolchains
root>/<PREFIX_NAME>, which must be a directory that resolves inside the
root and which check_tree(tree, pin) must accept (its ValueError becomes a
RunError); it is built as factory(executable=..., runner=..., tree=<that
tree>) (_checked_tree; toolchains/tt-metal.pin). The toolchains root is the
toolchains_root option, else $LASSI_TOOLCHAINS, resolved once (links and
`..` segments followed), and the pin is read from toolchains/<PIN>.pin. The
executable, the tree, each linked prefix, and the sandbox's read-only
toolchains root all come from that resolved root, which is what a compile
sees, and an executable, tree, or prefix that resolves outside it is
refused. The compile
environment is the parent's PATH, LANG=C and LC_ALL=C (ASCII diagnostics),
and each variable the pin names (NVHPC_CUDA_HOME for toolchains/nvhpc.pin);
nothing else, so a variable such as NVCC_PREPEND_FLAGS never reaches a
compile, and HOME stays out on purpose (P0.20). Every compile of generated
sources runs in the sandbox (lassi.executors.sandbox.SandboxedCompileRunner,
P0.20): its build dir is the one writable directory, the toolchains root is
read-only, $HOME, $LASSI_SCRATCH, and $LASSI_RUNS_ROOT (those set) and the
run's own runs root are hidden, it runs under prlimit --core=1, and it gets
a private TMPDIR under its build dir. TMPDIR itself is still required, and
when $LASSI_SCRATCH is set it must resolve inside it (Agent Rule 7). The
scratch root is hidden, and TMPDIR checked against it, only when
$LASSI_SCRATCH is set, as the gate sets it on the build host; without it a
compile still hides $HOME and the runs root. Before a toolchain is used, the
compile layout is checked as the sandbox checks it, and `<executable>
--version` runs once through the toolchain's own compile runner (so in the
sandbox, with the builds' environment and view, in a fresh directory under
TMPDIR) and must print the pin's EXPECT_VERSION; that also proves the
compiler reachable in the compile's view. Every refusal comes before the
run directory exists, and so does a SandboxUnavailableError from the
--version check. A class without PIN (a test fake) is built as factory().
build_toolchain is that construction for one registry name, public so that
a tool compiles exactly as a run does; the runner builds every bound
toolchain with it. A trial's toolchain_pins records the pins of the
toolchain that builds its target language; provenance.json records every
bound toolchain's pins. git, for the commit and dirty flag, runs through
lassi.toolchains.EnvRunner, the audited command runner, and every compiler
command through the sandbox's runner, so this module starts no process
itself.
"""

from __future__ import annotations

import dataclasses
import os
import platform
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lassi.bench import Direction, Suite, load_suite, sources_dir
from lassi.core import oracle_stage  # noqa: F401  (registers Stage "oracle" and, through lassi.oracles, the oracles)
from lassi.core.capabilities import SIMULATOR, UNLOAD_BEFORE_RUN, declares, unload_before_run
from lassi.core.fragments import fragment_key, pack_language
from lassi.core.interfaces import Executor, Sampling, Toolchain
from lassi.core.parquet import write_run_parquet
from lassi.core.recipe import (
    UNCAPPED,
    Recipe,
    RecipeError,
    executor_languages,
    load_recipe,
    resolved_data,
    resolved_yaml,
)
from lassi.core.record import (
    TOOLCHAIN_PIN_NAMES,
    Final,
    Provenance,
    ToolchainPins,
    Trial,
    arm_segment,
    json_text,
    make_trial_id,
    standing_attempt,
)
from lassi.core.registry import DEFAULT_REGISTRY, Binding, Registry
from lassi.core.stages import BASELINE_X10, PURPOSE, RUNS_CODE, SANDBOXED, RunContext
from lassi.core.store import TextStore, write_trial
from lassi.core.trial_md import fenced, fmt, fmt_provenance
from lassi.executors import workdir
from lassi.executors.sandbox import SandboxedCompileRunner
from lassi.llm import model_info
from lassi.prompts import assets as prompt_assets
from lassi.prompts import load_recipe_assets, render
from lassi.scoring.run_scoring import (
    RunMetrics,
    ScoringPlan,
    compute_metrics,
    metrics_section,
    plan_scoring,
    score_trial,
    write_metrics,
)
from lassi.scoring.score_run import ScoreError
from lassi.toolchains import EnvRunner
from lassi.toolchains.pins import linked_prefixes, prefix_pin_name, read_pin

REPO = Path(__file__).resolve().parents[2]
# Where suite manifests live: bench.suite names <BENCH_DIR>/<suite>.yaml.
BENCH_DIR = REPO / "assets" / "bench"

RUNS_DIR = "runs"
RESOLVED_RECIPE = "recipe.resolved.yaml"
TOOLCHAINS_JSON = "toolchains.json"
PROVENANCE_JSON = "provenance.json"
RUN_MD = "run.md"
PARQUET_DIR = "parquet"

# The status provenance.json records: while trials run, after a trial or a write raised, and at the end.
RUNNING, FAILED, COMPLETE = "running", "failed", "complete"

# A run id or suite name: one plain path segment.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
# Names the run tree uses for itself: a recipe (the first trial_id segment) with one of these names would
# put its trial directories among the text store or the Parquet tables.
_RUN_TREE_NAMES = frozenset({"texts", PARQUET_DIR, RESOLVED_RECIPE, TOOLCHAINS_JSON, PROVENANCE_JSON, RUN_MD})
# Recipe sections this runner does not carry out; a recipe that sets one is refused, never run without it.
_NOT_CARRIED_OUT = ("profiler", "adversary", "refine", "agents", "judges")
# How long git may take to report the commit or the dirty flag, in seconds.
_GIT_TIMEOUT_S = 60.0
# How long a pinned compiler's --version may take, in seconds, and the name prefix of the fresh directory under
# TMPDIR it runs in.
_VERSION_TIMEOUT_S = 120.0
_VERSION_PROBE_PREFIX = "lassi-version-check."
# The build dir, under the runs root, against which the compile layout is checked before a run (never created).
_LAYOUT_PROBE = "lassi-layout-check"
# The environment variables that name the roots every compile hides (P0.20), in order.
_COMPILE_HIDDEN_VARIABLES = ("HOME", "LASSI_SCRATCH", "LASSI_RUNS_ROOT")


class RunError(RuntimeError):
    """A run that cannot start or a recipe choice the runner cannot carry out; the message says what to do."""


@dataclass(frozen=True)
class RunOptions:
    """How to run a recipe; every None falls back as each field says.

    - runs_root: the runs root; None means $LASSI_RUNS_ROOT, then the recipe's
      runs_root. The result must be absolute, outside the repository, and
      inside $LASSI_SCRATCH when that is set.
    - run_id: the run directory's name; None means the UTC start time as
      YYYYMMDD-HHMMSS.
    - bench_root: the fetched sources of the suite; None means
      lassi.bench.sources_dir($LASSI_SCRATCH, suite).
    - toolchains_root: where pinned toolchains are installed; None means
      $LASSI_TOOLCHAINS.
    - registry: the components a recipe may bind; None means DEFAULT_REGISTRY.
    - roots: the directories searched for a recipe named in `extends`; None
      means lassi.core.recipe.default_roots().
    """

    runs_root: Path | None = None
    run_id: str | None = None
    bench_root: Path | None = None
    toolchains_root: Path | None = None
    registry: Registry | None = None
    roots: Sequence[Path] | None = None


# The options run_recipe uses when none are given: every field falls back as RunOptions says.
_DEFAULT_OPTIONS = RunOptions()


@dataclass(frozen=True)
class _Settings:
    """The run choices read from a loaded recipe.

    `fragments` holds a fragment prompt set's fragments (empty for a template
    set) and `packs` the context packs by the language each serves.
    """

    backend: str
    model_id: str
    sampling: Sampling
    max_corrections: int | None
    prompts: str
    directions: tuple[Direction, ...]
    trials: int
    split: str
    fragments: Mapping[str, str] = dataclasses.field(default_factory=dict)
    packs: Mapping[str, str] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class _Bench:
    """The suite, the root of its fetched sources, and the items the run covers, sorted."""

    suite: Suite
    root: Path
    items: tuple[str, ...]


@dataclass(frozen=True)
class BuiltToolchain:
    """One built toolchain: its registry name, the languages a recipe binds it for, the object, and its pins.

    build_toolchain returns one with no languages; the runner fills them in
    for each toolchain a recipe binds. `executable` and `environment` are
    None for a toolchain without a pin; `pins` maps each pin it uses to the
    pin file's pairs. `version` holds the non-blank lines the pinned
    compiler's `--version` printed when build_toolchain checked it against
    the pin's EXPECT_VERSION, and `version_status` the exit status it
    observed (empty and None without a pin).
    """

    name: str
    languages: tuple[str, ...]
    toolchain: Toolchain
    executable: str | None
    environment: dict[str, str] | None
    pins: dict[str, dict[str, str]]
    version: tuple[str, ...] = ()
    version_status: int | None = None


@dataclass(frozen=True)
class _Executors:
    """The run's executors and what provenance.json records of them.

    The single form binds one executor, `single`, for every language; the
    per-language form binds one per language in `by_language`, and `single`
    is None. `record` holds the manifest's executor keys: `executor` and
    `device` for the single form, `executor` and `devices` (each by
    language) for the per-language form.
    """

    single: Executor | None
    by_language: Mapping[str, Executor]
    record: Mapping[str, Any]

    def for_language(self, language: str) -> Executor:
        """Return the executor that runs programs in `language`."""
        return self.single if self.single is not None else self.by_language[language]


@dataclass(frozen=True)
class _Run:
    """Everything the trial loop and the run files read: the recipe, its components, the bench, and provenance.

    `pins` holds every bound toolchain's pin versions and `target_pins` those
    of the toolchain that builds each target language. `scoring` holds the
    built ScoreProfiles of `score` and `metrics` (lassi.scoring.run_scoring).
    """

    recipe: Recipe
    registry: Registry
    settings: _Settings
    bench: _Bench
    backend: Any
    executors: _Executors
    toolchains: tuple[BuiltToolchain, ...]
    pins: ToolchainPins
    target_pins: Mapping[str, ToolchainPins]
    run_dir: Path
    store: TextStore
    started: datetime
    commit: str | None
    dirty: bool | None
    scoring: ScoringPlan


def run_recipe(path: Path, options: RunOptions = _DEFAULT_OPTIONS) -> Path:
    """Run the recipe at `path` and return its run directory, `<runs root>/runs/<run_id>`.

    Raises RecipeError when the recipe does not load and RunError when the
    run cannot start (see the module docstring); both come before any
    directory is created or any model is asked. A component's own errors,
    such as SandboxUnavailableError from an executor or a sandboxed compile,
    propagate; one raised during the trials leaves provenance.json with
    status "failed", and so does a RunError from a ScoreProfile that cannot
    score a trial or from metric tables that cannot be built. The
    toolchains' --version checks run in the compile sandbox before any
    directory is created, so a sandbox that cannot run them raises
    SandboxUnavailableError then.
    """
    run = _prepare(Path(path), options, datetime.now(timezone.utc))
    run_dir = run.run_dir
    _write(run_dir / RESOLVED_RECIPE, resolved_yaml(run.recipe))
    _write(run_dir / TOOLCHAINS_JSON, json_text(_toolchains_record(run.toolchains)))
    manifest = _provenance(run)
    _write(run_dir / PROVENANCE_JSON, json_text(manifest))
    try:
        trials = _run_trials(run, manifest)
        metrics = _metrics(run, trials)
        provenance = _final_provenance(manifest, COMPLETE)
        _write(run_dir / RUN_MD, _run_md(run, provenance, trials, metrics))
        write_run_parquet(trials, run_dir / PARQUET_DIR)
        if metrics is not None:
            write_metrics(metrics, run_dir / PARQUET_DIR)
    except BaseException:
        _write(run_dir / PROVENANCE_JSON, json_text(_final_provenance(manifest, FAILED)))
        raise
    _write(run_dir / PROVENANCE_JSON, json_text(provenance))
    print(f"run directory: {run_dir}", flush=True)
    return run_dir


def _prepare(path: Path, options: RunOptions, started: datetime) -> _Run:
    """Load and check the recipe, build its components, and create the run directory; RunError before any mkdir."""
    registry = DEFAULT_REGISTRY if options.registry is None else options.registry
    recipe = _load(path, options, registry)
    settings = _settings(recipe)
    runs_root = _runs_root(options, recipe)
    run_dir = runs_root / RUNS_DIR / _run_id(options, started)
    _check_location(run_dir, f"the run directory {run_dir}")
    if run_dir.exists():
        raise RunError(f"the run directory {run_dir} already exists; choose another run id")
    bench = _bench(recipe, settings, options)
    _check_plan(recipe, registry, settings, bench)
    _check_oracle(recipe, registry, bench)
    scoring = _scoring(recipe, registry, bench)
    backend = registry.get("LLMBackend", settings.backend).factory(settings.model_id)
    _check_backend(recipe, settings, backend)
    toolchains = _toolchains(recipe, registry, _toolchains_root(options), runs_root)
    pins = _trial_pins(toolchains)
    target_pins = {
        direction.target: _trial_pins(built for built in toolchains if direction.target in built.languages)
        for direction in settings.directions
    }
    executors = _executors(recipe, registry, settings)
    commit, dirty = _git_state()
    try:
        run_dir.mkdir(parents=True)
    except FileExistsError:
        raise RunError(f"the run directory {run_dir} already exists; choose another run id") from None
    return _Run(
        recipe=recipe,
        registry=registry,
        settings=settings,
        bench=bench,
        backend=backend,
        executors=executors,
        toolchains=toolchains,
        pins=pins,
        target_pins=target_pins,
        run_dir=run_dir,
        store=TextStore(run_dir),
        started=started,
        commit=commit,
        dirty=dirty,
        scoring=scoring,
    )


def _scoring(recipe: Recipe, registry: Registry, bench: _Bench) -> ScoringPlan:
    """Build the ScoreProfiles `score` and `metrics` need and check every metrics name; RunError says why not.

    lassi.scoring.run_scoring.plan_scoring decides, with the run's bench
    root for a profile that reads bench sources, before any directory is
    created or any model is asked.
    """
    try:
        return plan_scoring(recipe.data, registry, bench.root)
    except (OSError, ValueError) as error:
        raise RunError(f"{recipe.path}: {error}") from error


def _check_backend(recipe: Recipe, settings: _Settings, backend: Any) -> None:
    """Refuse a backend that lacks a method the run will call on it.

    A backend that declares `needs_reference` gets each item's reference
    target through with_reference(), and, with fixes.fence_tag off, is asked
    for its faithful reply form, one untagged fence, through
    with_untagged_fence(). One that declares `unload_before_run` needs unload().
    """
    if "needs_reference" in backend.capabilities:
        if not hasattr(backend, "with_reference"):
            raise RunError(f"LLMBackend {settings.backend!r} needs the reference target but has no with_reference()")
        if not recipe.data["fixes"]["fence_tag"] and not hasattr(backend, "with_untagged_fence"):
            raise RunError(
                f"LLMBackend {settings.backend!r} needs the reference target, but fixes.fence_tag is off and it has "
                "no with_untagged_fence() to answer in the one untagged fence that faithful extraction reads"
            )
    if declares(backend, UNLOAD_BEFORE_RUN) and not callable(getattr(backend, "unload", None)):
        raise RunError(f"LLMBackend {settings.backend!r} declares {UNLOAD_BEFORE_RUN!r} but has no unload()")


# ---------------------------------------------------------------------------
# Reading the recipe and the options


def _load(path: Path, options: RunOptions, registry: Registry) -> Recipe:
    """Load the recipe; when the loader refuses one whose model section cannot run, raise the RunError that says why.

    A recipe with arms but no model fails the loader's binding check (a
    generating stage needs an LLMBackend, and only model binds one), so the
    resolved mapping is read again to explain the refusal.
    """
    try:
        return load_recipe(path, roots=options.roots, registry=registry)
    except RecipeError:
        problem = _model_problem(path, resolved_data(path, roots=options.roots))
        if problem is None:
            raise
        raise RunError(problem) from None


def _model_problem(path: Path, data: Mapping[str, Any]) -> str | None:
    """Return why the recipe's model section cannot run, or None when it names a backend and a model id."""
    model = data.get("model")
    if model is None:
        if isinstance(data.get("arms"), list) and data["arms"]:
            return (
                f"{path}: the recipe lists arms but no model; the model registry that maps arms to backends "
                "comes later, so set model: {backend: <name>, id: <model id>} to run one model"
            )
        return f"{path}: the recipe names no model; set model.backend and model.id"
    if not isinstance(model, dict):
        return None
    for key in ("backend", "id"):
        if key not in model:
            return f"{path}: model.{key} has no value; set it in the recipe"
    return None


def _settings(recipe: Recipe) -> _Settings:
    """Return the run choices of a loaded recipe; raise RunError for a choice the runner cannot carry out."""
    data = recipe.data
    problem = _model_problem(recipe.path, data)
    if problem is not None:
        raise RunError(problem)
    if data.get("arms"):
        raise RunError(
            f"{recipe.path}: the recipe sets both model and arms; the runner runs model alone and never drops arms "
            "silently, so remove arms (the model registry that maps arms to backends comes later)"
        )
    sampling = data["llm"]["sampling"]
    if "max_tokens" not in sampling:
        raise RunError(
            f"{recipe.path}: llm.sampling.max_tokens has no value; set it in the recipe (the runner never picks one)"
        )
    if sampling["max_tokens"] < 1:
        raise RunError(f"{recipe.path}: llm.sampling.max_tokens must be at least 1, not {sampling['max_tokens']}")
    if "prompts" not in data:
        raise RunError(f"{recipe.path}: prompts has no value; name the prompt set under assets/prompts/")
    _check_carried_out(recipe)
    fragments, packs = _recipe_assets(recipe)
    cap = data["loop"]["max_corrections"]
    return _Settings(
        fragments=fragments,
        packs=packs,
        backend=data["model"]["backend"],
        model_id=data["model"]["id"],
        sampling=Sampling(
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            max_tokens=sampling["max_tokens"],
        ),
        max_corrections=None if cap == UNCAPPED else cap,
        prompts=data["prompts"],
        directions=tuple(Direction(entry["source"], entry["target"]) for entry in data["directions"]),
        trials=data["trials"]["n"],
        split=data["bench"]["split"],
    )


def _check_carried_out(recipe: Recipe) -> None:
    """Refuse a recipe choice the stages cannot honor, so the saved recipe never claims behavior the run lacked.

    A fix turned off is checked against the listed stages in _check_stages.
    """
    data = recipe.data
    for name, on in sorted(data["report"].items()):
        if not on:
            raise RunError(
                f"{recipe.path}: report.{name} is false, but the runner always writes trial.md and the Parquet "
                "mirror; set it to true"
            )
    unused = [key for key in _NOT_CARRIED_OUT if key in data]
    if unused:
        raise RunError(
            f"{recipe.path}: the recipe sets {', '.join(unused)}, which this runner does not carry out yet; "
            "remove them rather than have the run ignore them"
        )


def _recipe_assets(recipe: Recipe) -> tuple[dict[str, str], dict[str, str]]:
    """Return the fragments of the recipe's prompt set and its context packs keyed by the language each serves.

    A template set (no MANIFEST.yaml under the assets root) with no context
    gives two empty mappings, and with context it is a RunError, since only
    a fragment set joins a pack into its prompts. Otherwise
    lassi.prompts.load_recipe_assets loads the set and the packs from the
    default assets root, checking every file against its manifest. A pack
    serves the language its one entry key ends in
    (lassi.core.fragments.pack_language). Raises RunError, with the loader's
    message, for a set or pack that does not load, and when two packs serve
    one language.
    """
    data = recipe.data
    root = prompt_assets.default_root()
    names = data.get("context", [])
    manifest = root / "prompts" / str(data["prompts"]) / prompt_assets.MANIFEST
    if not manifest.is_file():
        if not names:
            return {}, {}
        raise RunError(
            f"{recipe.path}: context names packs, but the prompt set {data['prompts']!r} is not a fragment set "
            f"(there is no {manifest}); only a fragment prompt set uses context packs, so drop context or name one"
        )
    try:
        assets = load_recipe_assets(data, root=root)
        packs: dict[str, str] = {}
        by_language: dict[str, str] = {}
        for name in names:
            language = pack_language(next(iter(prompt_assets.load_tree(root / "context" / name))))
            if language in packs:
                raise RunError(
                    f"{recipe.path}: the context packs {by_language[language]!r} and {name!r} both serve "
                    f"{language!r}; name one pack per language"
                )
            packs[language], by_language[language] = assets.packs[name], name
    except ValueError as error:
        raise RunError(f"{recipe.path}: {error}") from error
    return dict(assets.fragments), packs


def _runs_root(options: RunOptions, recipe: Recipe) -> Path:
    """Return the runs root: the option, else $LASSI_RUNS_ROOT, else the recipe's; absolute and outside the repo."""
    if options.runs_root is not None:
        root, source = Path(options.runs_root), "the runs_root option"
    elif os.environ.get("LASSI_RUNS_ROOT"):
        try:
            root = workdir.runs_root()
        except ValueError as error:
            raise RunError(str(error)) from error
        source = "$LASSI_RUNS_ROOT"
    else:
        root, source = Path(recipe.data["runs_root"]), f"runs_root in {recipe.path}"
    if not root.is_absolute():
        raise RunError(f"the runs root {root} (from {source}) must be an absolute path")
    _check_location(root, f"the runs root {root} (from {source})")
    return root


def _within(path: Path, root: Path) -> bool:
    """Return True when the resolved `path` is `root` or lies under it."""
    return path == root or root in path.parents


def _check_location(path: Path, what: str) -> None:
    """Refuse a run path that resolves inside the repository, or outside $LASSI_SCRATCH when that is set.

    The path is resolved, so a link into the repository is refused too;
    `what` names the path in the message (Agent Rule 7).
    """
    resolved = path.resolve()
    if _within(resolved, REPO):
        raise RunError(f"{what} is inside the repository {REPO}; run trees stay out of git (Agent Rule 7)")
    scratch = os.environ.get("LASSI_SCRATCH", "")
    if scratch and not _within(resolved, Path(scratch).resolve()):
        raise RunError(
            f"{what} is outside $LASSI_SCRATCH ({scratch}); on the build host run trees stay on the scratch disk, "
            "never on the root filesystem (Agent Rule 7)"
        )


def _run_id(options: RunOptions, started: datetime) -> str:
    """Return the run id: the option, else the UTC start time as YYYYMMDD-HHMMSS; it must be one plain name."""
    run_id = started.strftime("%Y%m%d-%H%M%S") if options.run_id is None else options.run_id
    if not isinstance(run_id, str) or not _NAME.fullmatch(run_id):
        raise RunError(f"a run id must match {_NAME.pattern}, got {run_id!r}")
    return run_id


def _toolchains_root(options: RunOptions) -> Path | None:
    """Return the toolchains root: the option, else $LASSI_TOOLCHAINS, else None."""
    if options.toolchains_root is not None:
        return Path(options.toolchains_root)
    value = os.environ.get("LASSI_TOOLCHAINS", "")
    return Path(value) if value else None


# ---------------------------------------------------------------------------
# The bench and the plan


def _bench(recipe: Recipe, settings: _Settings, options: RunOptions) -> _Bench:
    """Load the recipe's suite and find its fetched sources; the items are those `_items` selects, sorted."""
    name = recipe.data["bench"]["suite"]
    manifest = BENCH_DIR / f"{name}.yaml"
    if not _NAME.fullmatch(name) or not manifest.is_file():
        raise RunError(f"{recipe.path}: bench.suite {name!r} names no suite manifest; expected {manifest}")
    try:
        suite = load_suite(manifest)
    except (OSError, ValueError) as error:
        raise RunError(f"cannot load the suite manifest {manifest}: {error}") from error
    items = _items(recipe, suite, settings.split)
    return _Bench(suite=suite, root=_sources_root(suite, options), items=items)


def _items(recipe: Recipe, suite: Suite, split: str) -> tuple[str, ...]:
    """Return the items the run covers, sorted: those bench.items names, else every item of the recipe's split.

    A selected item the suite lacks, one outside the split, or one named
    twice is a RunError naming it.
    """
    in_split = sorted(item for item, spec in suite.items.items() if spec.split == split)
    selected = recipe.data["bench"].get("items")
    if selected is None:
        if not in_split:
            raise RunError(f"{recipe.path}: suite {suite.name!r} has no items in the split {split!r}")
        return tuple(in_split)
    for index, item in enumerate(selected):
        if item not in suite.items:
            raise RunError(
                f"{recipe.path}: bench.items names {item!r}, which suite {suite.name!r} does not have; "
                f"items: {', '.join(sorted(suite.items))}"
            )
        if item not in in_split:
            raise RunError(f"{recipe.path}: bench.items names {item!r}, which is not in the split {split!r}")
        if item in selected[:index]:
            raise RunError(f"{recipe.path}: bench.items names {item!r} twice")
    return tuple(sorted(selected))


def _fetch_hint(suite: Suite) -> str:
    """Return how to fetch a suite's pinned sources on the build host."""
    return f"fetch them on the build host with `uv run tools/fetch_bench.py assets/bench/{suite.name}.yaml`"


def _sources_root(suite: Suite, options: RunOptions) -> Path:
    """Return the root of the suite's fetched sources: the bench_root option, else under $LASSI_SCRATCH."""
    if options.bench_root is not None:
        root = Path(options.bench_root)
    else:
        scratch = os.environ.get("LASSI_SCRATCH", "")
        if not scratch:
            raise RunError(
                f"no bench sources for {suite.name}: LASSI_SCRATCH is not set and no bench root was given; "
                + _fetch_hint(suite)
            )
        try:
            root = sources_dir(Path(scratch), suite)
        except ValueError as error:
            raise RunError(f"{error}; {_fetch_hint(suite)}") from error
    if not root.is_dir():
        raise RunError(f"the sources of {suite.name} are not at {root}; {_fetch_hint(suite)}")
    return root


def _check_sandbox(recipe: Recipe) -> None:
    """Refuse run limits the stages cannot apply: sandbox.wall_s and sandbox.mem_gb, read for every run of a program.

    wall_s is a number of seconds above 0 or BASELINE_X10 (ten times the
    reference run's wall time, lassi.core.stages attempt_limits), and mem_gb
    a number of GB above 0.
    """
    sandbox = recipe.data.get("sandbox", {})
    wall = sandbox.get("wall_s")
    if wall != BASELINE_X10 and not (isinstance(wall, (int, float)) and not isinstance(wall, bool) and wall > 0):
        raise RunError(
            f"{recipe.path}: sandbox.wall_s is {wall!r}, but an executor that runs programs needs a number of seconds "
            f"above 0 or {BASELINE_X10!r}; set it in the recipe"
        )
    memory = sandbox.get("mem_gb")
    if not (isinstance(memory, (int, float)) and not isinstance(memory, bool) and memory > 0):
        raise RunError(
            f"{recipe.path}: sandbox.mem_gb is {memory!r}, but an executor that runs programs needs a number of GB "
            "above 0; set it in the recipe"
        )


def _check_sandboxed(recipe: Recipe, registry: Registry, binding: Binding, capabilities: frozenset[str]) -> None:
    """Refuse an executor that runs programs outside the sandbox when a listed stage runs model-generated code.

    Such a stage declares `runs_model_code` (run_loop); the executor must
    then declare SANDBOXED (Agent Rule 6). baseline runs only bench
    references, so it needs no such declaration. With executors per
    language, every bound executor that runs programs is checked, and the
    message names its recipe path, executor.<language>.
    """
    if SANDBOXED in capabilities:
        return
    for name in recipe.data["stages"]:
        if getattr(registry.get("Stage", name).factory, "runs_model_code", False):
            raise RunError(
                f"{recipe.path}: stage {name!r} runs model-generated code, but Executor {binding.name!r} "
                f"({binding.where}) runs programs without declaring {SANDBOXED!r}; model-generated code runs only "
                "in the sandbox (Agent Rule 6)"
            )


def _executors(recipe: Recipe, registry: Registry, settings: _Settings) -> _Executors:
    """Check and build the recipe's executors, one or one per language, and name each one's device.

    Everything is checked before any executor is built: the languages the
    per-language form must bind (_check_executor_languages), the sandbox
    limits when any bound executor runs programs, and the sandboxed check for
    each one that does. Each executor is then built as factory(**config) and
    asked for its device() once (_device, _checked_device); RunError
    before any directory exists.
    """
    bindings = [binding for binding in recipe.bindings if binding.interface == "Executor"]
    languages = executor_languages(recipe.data)
    if languages:
        _check_executor_languages(recipe, registry, settings, set(languages))
    entries = [registry.get("Executor", binding.name) for binding in bindings]
    if any(RUNS_CODE in entry.capabilities for entry in entries):
        _check_sandbox(recipe)
    for binding, entry in zip(bindings, entries, strict=True):
        if RUNS_CODE in entry.capabilities:
            _check_sandboxed(recipe, registry, binding, entry.capabilities)
    built = [entry.factory(**binding.config) for binding, entry in zip(bindings, entries, strict=True)]
    devices = [_checked_device(_device(executor), binding) for executor, binding in zip(built, bindings, strict=True)]
    if not languages:
        record = {"executor": bindings[0].name, "device": devices[0]}
        return _Executors(single=built[0], by_language={}, record=record)
    record = {
        "executor": {language: binding.name for language, binding in zip(languages, bindings, strict=True)},
        "devices": dict(zip(languages, devices, strict=True)),
    }
    return _Executors(single=None, by_language=dict(zip(languages, built, strict=True)), record=record)


def _check_executor_languages(recipe: Recipe, registry: Registry, settings: _Settings, bound: set[str]) -> None:
    """Refuse executors per language that leave a language the run needs without an executor.

    Every direction's target language needs one, since the attempts run
    there. A stage that builds the source reference under a fix that is on
    (`source_build_fix`, baseline under baseline_both) runs it on the source
    language's executor, so that language needs one too. Each message names
    the key to set, executor.<language>.
    """
    for direction in settings.directions:
        if direction.target not in bound:
            raise RunError(
                f"{recipe.path}: no executor is bound for the target language {direction.target!r}; "
                f"set executor.{direction.target}"
            )
    for name in recipe.data["stages"]:
        fix = getattr(registry.get("Stage", name).factory, "source_build_fix", None)
        if fix is None or recipe.data["fixes"].get(fix, True) is False:
            continue
        for direction in settings.directions:
            if direction.source not in bound:
                raise RunError(
                    f"{recipe.path}: stage {name!r} builds the source reference too while fixes.{fix} is on, and "
                    f"runs it on the source language's executor, but no executor is bound for {direction.source!r}; "
                    f"set executor.{direction.source}, or turn fixes.{fix} off"
                )


def _checked_device(device: Any, binding: Binding) -> str | None:
    """Return an executor's device after checking it: None, or one non-empty line of printable ASCII.

    The line has no leading or trailing blank (whitespace). A device enters
    provenance.json, every trial's provenance, and run.md, so anything else
    (a tab or another character that is not printable ASCII, a line break,
    a leading or trailing blank) is a RunError naming the binding, before
    any directory exists.
    """
    if device is None:
        return None
    if not isinstance(device, str) or not device.isascii() or not device.isprintable() or device.strip() != device:
        raise RunError(
            f"Executor {binding.name!r} ({binding.where}): device() must return one non-empty line of printable "
            f"ASCII with no leading or trailing blank, got {device!r}"
        )
    if not device:
        raise RunError(f"Executor {binding.name!r} ({binding.where}): device() returned an empty string")
    return device


def _check_plan(recipe: Recipe, registry: Registry, settings: _Settings, bench: _Bench) -> None:
    """Refuse, before anything is built, a plan whose project, stages, prompts, trial ids, or sources cannot work.

    The project (the recipe's `project` key, else its name) is the first
    trial id segment, so it must be one plain segment that names no entry of
    the run tree.
    """
    project = recipe.data["project"]
    if not _NAME.fullmatch(project):
        raise RunError(
            f"{recipe.path}: the project {project!r} is the first trial id segment, so it must match {_NAME.pattern}; "
            "set the project key"
        )
    if project in _RUN_TREE_NAMES:
        raise RunError(f"{recipe.path}: the project may not be {project!r}, which the run tree uses itself")
    _check_stages(recipe, registry, settings)
    _check_trials(recipe, settings, bench)


def _check_stages(recipe: Recipe, registry: Registry, settings: _Settings) -> None:
    """Refuse a stage order that cannot run, a target language with no toolchain, and a prompt a stage cannot render.

    A fix turned off needs a listed stage that reproduces its quirk
    (_check_fixes). A trial gets one first attempt, so at most one stage
    declares `generates`, and a stage that declares `compiles` builds that
    attempt, so it comes after it. A stage that declares `builds_references`
    runs before any model call, so no stage that asks the model
    (_asks_model) comes before it. The prompts are checked by _check_prompts.
    """
    entries = [registry.get("Stage", name) for name in recipe.data["stages"]]
    _check_fixes(recipe, entries)
    generated = False
    asked: str | None = None
    for index, entry in enumerate(entries):
        where = f"{recipe.path}: stages[{index}] {entry.name!r}"
        if "builds_references" in entry.capabilities and asked is not None:
            raise RunError(
                f"{where} builds the reference programs before any model call, but {asked} asks the model; "
                "list it before every stage that asks the model"
            )
        if _asks_model(entry) and asked is None:
            asked = f"stages[{index}] {entry.name!r}"
        if "compiles" in entry.capabilities and not generated:
            raise RunError(f"{where} builds the attempt a generating stage appends; list such a stage before it")
        if "generates" in entry.capabilities:
            if generated:
                raise RunError(f"{where} is a second generating stage; a trial has one first attempt, so list one")
            generated = True
    bound = recipe.data.get("toolchain", {})
    if any("Toolchain" in entry.requires for entry in entries):
        for direction in settings.directions:
            if direction.target not in bound:
                raise RunError(
                    f"{recipe.path}: no toolchain is bound for the target language {direction.target!r}; "
                    f"set toolchain.{direction.target}"
                )
    _check_source_toolchains(recipe, entries, settings)
    _check_prompts(recipe, entries, settings)


def _asks_model(entry: Any) -> bool:
    """Return True for a stage that asks the model: it requires an LLMBackend, or it generates or corrects attempts."""
    return "LLMBackend" in entry.requires or bool({"generates", "compiles"} & entry.capabilities)


def _check_source_toolchains(recipe: Recipe, entries: Sequence[Any], settings: _Settings) -> None:
    """Refuse a stage that builds the source reference too, under a fix that is on, when no toolchain builds it.

    A stage class names that fix in `source_build_fix` (baseline names
    baseline_both); with the fix on, every direction's source language needs
    a bound toolchain.
    """
    bound = recipe.data.get("toolchain", {})
    for entry in entries:
        fix = getattr(entry.factory, "source_build_fix", None)
        if fix is None or recipe.data["fixes"].get(fix, True) is False:
            continue
        for direction in settings.directions:
            if direction.source not in bound:
                raise RunError(
                    f"{recipe.path}: stage {entry.name!r} builds the source reference too while fixes.{fix} is on, "
                    f"but no toolchain is bound for the source language {direction.source!r}; "
                    f"set toolchain.{direction.source}, or turn fixes.{fix} off"
                )


def _check_fixes(recipe: Recipe, entries: Sequence[Any]) -> None:
    """Refuse a fix turned off when no listed stage names it in `reproduces`, so no quirk is asked for in vain."""
    data = recipe.data
    reproduced = {name for entry in entries for name in getattr(entry.factory, "reproduces", ())}
    for name, on in sorted(data["fixes"].items()):
        if not on and name not in reproduced:
            raise RunError(
                f"{recipe.path}: fixes.{name} is off (faithful: {fmt(data['faithful'])}), which asks for an upstream "
                "quirk that no listed stage reproduces; list a stage that reproduces it, or turn the fix on and "
                "leave faithful off"
            )


def _check_prompts(recipe: Recipe, entries: Sequence[Any], settings: _Settings) -> None:
    """Refuse a prompt a stage cannot build from the recipe's prompt set and context packs.

    With a template set, each stage's `prompt_fields` templates are rendered
    once with empty fields, which finds a missing set or file and a
    placeholder the stage does not fill; a stage that reads only fragments
    is refused. With a fragment set, a stage that renders templates but
    declares no `fragment_keys` is refused, every key a stage declares must
    be in the set for every direction, a stage that declares
    `needs_context` needs a pack for every target language, and each
    Trial.context field a stage names in `joins_context` must be named in
    `fills_context` by an earlier stage whenever a pack serves a target
    (_check_context_order).
    """
    filled: set[str] = set()
    for entry in entries:
        templates = getattr(entry.factory, "prompt_fields", {})
        keys = getattr(entry.factory, "fragment_keys", None)
        where = f"{recipe.path}: stage {entry.name!r}"
        if not settings.fragments:
            if keys and not templates:
                raise RunError(f"{where} reads prompt fragments, but {settings.prompts!r} is a template prompt set")
            for prompt, fields in templates.items():
                try:
                    render(settings.prompts, prompt, dict.fromkeys(fields, ""))
                except ValueError as error:
                    raise RunError(f"{where} cannot use the prompt set: {error}") from error
            continue
        _check_context_order(where, entry, settings, filled)
        filled.update(getattr(entry.factory, "fills_context", ()))
        if keys is None:
            if templates:
                raise RunError(f"{where} renders templates, but {settings.prompts!r} is a fragment prompt set")
            continue
        for direction in settings.directions:
            missing = [key for key in (fragment_key(template, direction) for template in keys)
                       if key not in settings.fragments]
            if missing:
                raise RunError(f"{where}: the prompt set {settings.prompts!r} has no fragment {', '.join(missing)}")
            if getattr(entry.factory, "needs_context", False) and direction.target not in settings.packs:
                served = ", ".join(sorted(settings.packs)) or "none"
                raise RunError(
                    f"{where} needs a context pack for the target language {direction.target!r}; "
                    f"the recipe's context serves: {served}"
                )


def _check_context_order(where: str, entry: Any, settings: _Settings, filled: set[str]) -> None:
    """Refuse a stage that joins a Trial.context field into its prompt when no earlier stage fills that field.

    A fragment set's generation prompt joins the summary and the description
    whenever a pack serves the target; with a field left empty the model
    would get empty markers, a prompt upstream never sends.
    """
    missing = [name for name in getattr(entry.factory, "joins_context", ()) if name not in filled]
    served = [direction.target for direction in settings.directions if direction.target in settings.packs]
    if missing and served:
        raise RunError(
            f"{where} joins the Trial.context field(s) {', '.join(missing)} into its prompt when a context pack "
            f"serves {served[0]!r}, but no earlier listed stage fills them; list the stages that fill them first"
        )


def _check_trials(recipe: Recipe, settings: _Settings, bench: _Bench) -> None:
    """Refuse trial ids that are invalid or repeat (in any letter case), and bench sources that cannot be read.

    Each item's file counts are checked against the prompt set and the
    fence_tag fix by _check_one_file.
    """
    seen: dict[str, str] = {}
    for direction in settings.directions:
        for item in bench.items:
            try:
                arm = arm_segment(settings.model_id)
                trial_id = make_trial_id(recipe.data["project"], arm, bench.suite.name, direction.name, item, 1)
            except ValueError as error:
                raise RunError(f"{recipe.path}: {error}") from error
            key = trial_id.casefold()
            if key in seen:
                raise RunError(
                    f"{recipe.path}: the trials {seen[key]} and {trial_id} would share one trial directory; "
                    "list each direction once"
                )
            seen[key] = trial_id
            try:
                sources = bench.suite.source_files(item, direction, bench.root, purpose=PURPOSE)
                bench.suite.reference_target(item, direction, bench.root, purpose=PURPOSE)
                bench.suite.support_files(item, bench.root, purpose=PURPOSE)
            except (OSError, ValueError) as error:
                raise RunError(
                    f"cannot read {bench.suite.name}/{item} ({direction.name}) under {bench.root}: {error}; "
                    + _fetch_hint(bench.suite)
                ) from error
            _check_one_file(recipe, settings, bench, item, direction, len(sources))


def _check_one_file(
    recipe: Recipe, settings: _Settings, bench: _Bench, item: str, direction: Direction, source_count: int
) -> None:
    """Refuse an item a fragment set or the fence_tag quirk cannot take, before any model is asked.

    A fragment set joins one source text into its prompts
    (lassi.core.stages.source_as_read), and with fixes.fence_tag off the
    first fenced block of a reply is the one target file.
    """
    where = f"{recipe.path}: {bench.suite.name}/{item} ({direction.name})"
    if settings.fragments and source_count != 1:
        raise RunError(
            f"{where} has {source_count} {direction.source!r} source files; a fragment prompt set joins one source"
        )
    targets = bench.suite.item(item, purpose=PURPOSE).languages[direction.target].files
    if recipe.data["fixes"].get("fence_tag", True) is False and len(targets) != 1:
        raise RunError(
            f"{where} has {len(targets)} {direction.target!r} files; with fixes.fence_tag off the first fenced "
            "block of a reply is the one target file"
        )


def _check_oracle(recipe: Recipe, registry: Registry, bench: _Bench) -> None:
    """Refuse an oracle section no listed stage uses or accepts, one its Oracle refuses, or one the run cannot serve.

    A listed stage uses an Oracle when its registry entry names Oracle in
    `requires` (the load already refuses such a stage when no oracle is
    bound). A stage class that names `oracle_capabilities` (the oracle stage)
    needs its Oracle to declare one of them. Each bound Oracle is built once
    as factory(**config) and dropped, so a choice its section leaves unset,
    such as oracle.passfail or oracle.threshold, fails here, before any
    directory is created, and not when the first trial builds its stages.
    An Oracle whose `agreement_setting` is set (binary_io's threshold
    from_baseline) is checked by _check_agreement.
    """
    bindings = [binding for binding in recipe.bindings if binding.interface == "Oracle"]
    if not bindings:
        return
    stages = [registry.get("Stage", name) for name in recipe.data["stages"]]
    if not any("Oracle" in entry.requires for entry in stages):
        raise RunError(
            f"{recipe.path}: sets oracle, but no listed stage uses an Oracle; "
            "list the oracle stage or remove the section"
        )
    for binding in bindings:
        entry = registry.get("Oracle", binding.name)
        for stage in stages:
            accepted = tuple(getattr(stage.factory, "oracle_capabilities", ()))
            if accepted and not set(accepted) & entry.capabilities:
                declared = ", ".join(sorted(entry.capabilities)) or "nothing"
                raise RunError(
                    f"{recipe.path}: {binding.where}: stage {stage.name!r} needs an Oracle that declares one of "
                    f"{', '.join(accepted)}, but Oracle {binding.name!r} declares: {declared}"
                )
        try:
            oracle = entry.factory(**binding.config)
        except ValueError as error:
            raise RunError(f"{recipe.path}: {binding.where}: {error}") from error
        setting = getattr(oracle, "agreement_setting", None)
        if setting is not None:
            _check_agreement(recipe, stages, bench, setting)


def _check_agreement(recipe: Recipe, stages: Sequence[Any], bench: _Bench, setting: str) -> None:
    """Refuse an oracle `setting` that reads the references' agreement when the run could never measure it.

    The baseline stage measures the source reference's agreement with the
    target reference only with fixes.baseline_both on and only for an item
    whose suite manifest declares a tolerance, so the fix must be on, a
    listed stage must build the references (`builds_references`), and every
    item the run covers must declare a tolerance.
    """
    reads = f"{recipe.path}: {setting} reads the references' agreement"
    if recipe.data.get("fixes", {}).get("baseline_both", True) is False:
        raise RunError(
            f"{reads}, which the baseline measures only with fixes.baseline_both on (faithful: true turns it off); "
            "turn fixes.baseline_both on, or set a number"
        )
    if not any("builds_references" in stage.capabilities for stage in stages):
        raise RunError(f"{reads}, but no listed stage builds the reference programs; list the baseline stage")
    missing = [item for item in bench.items if bench.suite.items[item].tolerance is None]
    if missing:
        raise RunError(
            f"{reads}, which the baseline measures only for an item whose suite manifest declares a tolerance; "
            f"{bench.suite.name} item(s) {', '.join(missing)} declare none"
        )


# ---------------------------------------------------------------------------
# Toolchains and pins


def _toolchains(
    recipe: Recipe, registry: Registry, root: Path | None, build_root: Path | None = None
) -> tuple[BuiltToolchain, ...]:
    """Build each toolchain the recipe binds, once per registry name, with build_toolchain, and set its languages.

    `build_root` is the run's runs root, which every compile hides (see
    build_toolchain).
    """
    languages: dict[str, list[str]] = {}
    for language, name in sorted(recipe.data.get("toolchain", {}).items()):
        languages.setdefault(name, []).append(language)
    return tuple(
        dataclasses.replace(build_toolchain(name, root, registry, build_root=build_root), languages=tuple(bound))
        for name, bound in languages.items()
    )


def build_toolchain(
    name: str, root: Path | None, registry: Registry = DEFAULT_REGISTRY, *, build_root: Path | None = None
) -> BuiltToolchain:
    """Build the toolchain registered as `name` exactly as the stage runner builds it, with no languages.

    A class that declares PIN gets its pinned executable under the
    toolchains root `root` (None means none is set, which is refused), or,
    without PIN_BIN, the pin's EXECUTABLE (a host compiler, which must be
    an absolute path to an existing file; the root is still required, since
    the compile sandbox exposes it; with check_tree, also the installed tree
    it builds against, checked before any process starts), a
    clean compile environment, and the linked prefixes its pin names, run
    through SandboxedCompileRunner (see the module docstring), after its
    `--version` output was checked against the pin's EXPECT_VERSION through
    that same runner; a class without PIN is built as factory().
    `build_root`, when given, is the directory the build dirs lie under (a
    run passes its runs root): every compile hides it, apart from its own
    build dir, and the compile layout is checked against a build dir under
    it first. Raises RunError, saying what to install or set, when the pin
    file, the root, the executable (or it resolves outside the root), the
    installed tree (or check_tree refuses it), a
    linked prefix, TMPDIR (unset, or outside $LASSI_SCRATCH), or every
    hidden root is missing, when a hidden root is not absolute, when the
    sandbox refuses the compile layout, or when the compiler is not the
    pinned version; SandboxUnavailableError when the sandbox cannot run the
    --version check. The runner builds every bound toolchain through this,
    so a tool that calls it compiles with the same command, executable,
    environment, and sandbox as a run.
    """
    factory = registry.get("Toolchain", name).factory
    if getattr(factory, "PIN", None) is None:
        return BuiltToolchain(name, (), factory(), None, None, {})
    return _pinned_toolchain(name, factory, root, build_root)


def _pinned_toolchain(name: str, factory: type, root: Path | None, build_root: Path | None) -> BuiltToolchain:
    """Build a toolchain with its pinned executable and the sandboxed compile runner, checked by --version.

    A class with PIN_BIN finds its compiler under the toolchains root
    (_pinned_executable); a class with PIN and no PIN_BIN uses a host
    compiler the project does not install, the pin's EXECUTABLE
    (_host_executable), and one that also defines check_tree builds
    against the pin's installed tree (_checked_tree). Every refusal is a
    RunError that says what is missing, raised before any process starts:
    the pin file, the root, the executable, the tree (or check_tree refuses
    it), TMPDIR (unset, or outside $LASSI_SCRATCH), a hidden root, a linked
    prefix, or a compile layout the sandbox refuses. Then
    `<executable> --version` must print the pin's EXPECT_VERSION in the
    compile sandbox.
    """
    pin_name = factory.PIN
    host = getattr(factory, "PIN_BIN", None) is None
    check_tree = getattr(factory, "check_tree", None) if host else None
    pin = _pin(pin_name, ("VERSION",) if host and check_tree is None else ("VERSION", "PREFIX_NAME"))
    if host:
        resolved, executable = _host_executable(name, pin_name, pin, root)
    else:
        resolved, executable = _pinned_executable(name, factory, pin, root)
    keywords = {} if check_tree is None else {"tree": _checked_tree(name, pin_name, pin, resolved, check_tree)}
    tmpdir = _checked_tmpdir()
    hidden_roots = _compile_hidden_roots(build_root)
    environment = _compile_environment()
    pins = {pin_name: pin, **_linked_pins(name, pin_name, pin, resolved, environment)}
    runner = _compile_runner(name, environment, resolved, hidden_roots, build_root)
    version, status = _checked_version(name, executable, runner, Path(tmpdir), pin_name, pin)
    toolchain = factory(executable=str(executable), runner=runner, **keywords)
    return BuiltToolchain(name, (), toolchain, str(executable), environment, pins, version, status)


def _host_executable(name: str, pin_name: str, pin: Mapping[str, str], root: Path | None) -> tuple[Path, str]:
    """Return the resolved toolchains root and a host pin's EXECUTABLE, as the pin gives it.

    A host compiler (toolchains/gcc.pin) is not installed under the
    toolchains root, so its path is the pin's EXECUTABLE: it must be set, an
    absolute path on this host (Path.is_absolute, so the host's own path
    rules apply), and an existing file, or a RunError names it before any
    process starts. The compile sandbox still exposes the toolchains root
    read-only, as for every compile, so a root must be set. The --version
    check then proves the compiler is the pinned one and reachable in the
    compile's view.
    """
    if root is None:
        raise RunError(
            f"toolchain {name!r} uses the pinned host {pin_name} {pin['VERSION']} in the compile sandbox, which "
            "exposes the toolchains root read-only, but no toolchains root is set; set LASSI_TOOLCHAINS (the gate "
            "sets it on the build host)"
        )
    if not root.is_absolute():
        raise RunError(f"the toolchains root {root} must be an absolute path (LASSI_TOOLCHAINS or the option)")
    executable = pin.get("EXECUTABLE", "")
    if not executable:
        raise RunError(
            f"toolchains/{pin_name}.pin has no EXECUTABLE, the absolute path of the host compiler toolchain {name!r} "
            "runs; add it"
        )
    if not Path(executable).is_absolute():
        raise RunError(
            f"toolchain {name!r}: EXECUTABLE {executable!r} in toolchains/{pin_name}.pin is not an absolute path; "
            "a pinned compiler is never looked up on PATH (Agent Rule 10)"
        )
    if not Path(executable).is_file():
        raise RunError(
            f"toolchain {name!r}: the pinned host compiler {executable} (EXECUTABLE in toolchains/{pin_name}.pin) "
            "does not exist on this host; install the package it names, or run on the build host"
        )
    return root.resolve(), executable


def _checked_tree(
    name: str, pin_name: str, pin: Mapping[str, str], root: Path, check_tree: Callable[[Path, Mapping[str, str]], None]
) -> Path:
    """Return the installed tree a host compiler builds against, <root>/<PREFIX_NAME>, once check_tree accepts it.

    `root` is the resolved toolchains root. The tree must be a directory
    that resolves inside it, since a compile sees only that root, and
    `check_tree(tree, pin)`, the toolchain class's own check of its install
    (lassi.toolchains.ttmetal_build), must raise no ValueError. Every
    refusal is a RunError, raised before any process starts.
    """
    tree = root / pin["PREFIX_NAME"]
    if not tree.is_dir() or not _within(tree.resolve(), root):
        raise RunError(
            f"toolchain {name!r} builds against the installed {pin_name} tree {tree}, which is not a directory "
            f"inside the toolchains root; install it on the build host with toolchains/{pin_name}.sh"
        )
    try:
        check_tree(tree, pin)
    except (OSError, ValueError) as error:
        raise RunError(f"toolchain {name!r}: {error}") from error
    return tree


def _pinned_executable(name: str, factory: type, pin: Mapping[str, str], root: Path | None) -> tuple[Path, Path]:
    """Return the resolved toolchains root and the pinned executable under it; RunError says what is missing.

    The root is resolved once (symbolic links and `..` segments followed),
    and the executable, every linked prefix, and the sandbox's read-only
    toolchains root all come from that resolved root, which is what a
    compile sees; an unresolved path would not exist in the compile's view.
    The executable must also resolve inside the root, since nothing outside
    it is exposed to a compile.
    """
    pin_name = factory.PIN
    if root is None:
        raise RunError(
            f"toolchain {name!r} uses the pinned {pin_name} {pin['VERSION']}, but no toolchains root is set; "
            f"set LASSI_TOOLCHAINS (the gate sets it on the build host), where toolchains/{pin_name}.sh installs it"
        )
    if not root.is_absolute():
        raise RunError(f"the toolchains root {root} must be an absolute path (LASSI_TOOLCHAINS or the option)")
    try:
        relative = factory.PIN_BIN.format_map(pin)
    except (KeyError, ValueError) as error:
        message = f"toolchain {name!r}: PIN_BIN {factory.PIN_BIN!r} does not fit toolchains/{pin_name}.pin"
        raise RunError(f"{message} ({error!r})") from None
    resolved = root.resolve()
    executable = resolved / pin["PREFIX_NAME"] / relative
    if not executable.is_file():
        raise RunError(
            f"toolchain {name!r}: the pinned compiler {executable} does not exist; "
            f"install it on the build host with toolchains/{pin_name}.sh"
        )
    target = executable.resolve()
    if not _within(target, resolved):
        raise RunError(
            f"toolchain {name!r}: the pinned compiler {executable} resolves to {target}, outside the toolchains root "
            f"{resolved}, and a compile sees only that root; install it there with toolchains/{pin_name}.sh"
        )
    return resolved, executable


def _linked_pins(
    name: str, pin_name: str, pin: Mapping[str, str], root: Path, environment: dict[str, str]
) -> dict[str, dict[str, str]]:
    """Set each linked prefix the pin names in `environment` and return the linked pins, by pin name.

    `root` is the resolved toolchains root. Raises RunError when a linked
    prefix is not the installed prefix of its own pin under it, or resolves
    outside it, where a compile could not see it.
    """
    linked_pins: dict[str, dict[str, str]] = {}
    for variable, prefix in linked_prefixes(pin).items():
        try:
            linked_name = prefix_pin_name(prefix)
        except ValueError as error:
            raise RunError(f"{pin_name}.pin: {error}") from error
        linked = _pin(linked_name)
        home = root / prefix
        if linked["PREFIX_NAME"] != prefix or not home.is_dir() or not _within(home.resolve(), root):
            raise RunError(
                f"toolchain {name!r} needs {home} ({pin_name}.pin), which is not the installed {linked_name} pin "
                f"inside the toolchains root; install it on the build host with toolchains/{linked_name}.sh"
            )
        environment[variable] = str(home)
        linked_pins[linked_name] = linked
    return linked_pins


def _pin(name: str, keys: Sequence[str] = ("VERSION", "PREFIX_NAME")) -> dict[str, str]:
    """Return toolchains/<name>.pin as read_pin reads it; RunError when it is missing or lacks one of `keys`."""
    try:
        pin = read_pin(name)
    except (OSError, ValueError) as error:
        raise RunError(f"cannot read the pin file toolchains/{name}.pin: {error}") from error
    for key in keys:
        if not pin.get(key):
            raise RunError(f"toolchains/{name}.pin has no {key}")
    return pin


def _checked_tmpdir() -> str:
    """Return $TMPDIR after checking that it is set and, when $LASSI_SCRATCH is set, resolves inside it.

    Each compile gets a private TMPDIR under its own build dir
    (SandboxedCompileRunner); the parent's TMPDIR holds the --version
    check's directory, and on the build host it must stay on the scratch
    disk (Agent Rule 7). Both paths are resolved, so a `..` segment or a
    link cannot lead outside, and a sibling that only shares a name prefix
    is refused. Without $LASSI_SCRATCH (the gate sets it on the build host)
    there is no scratch root to check against.
    """
    tmpdir = os.environ.get("TMPDIR", "")
    if not tmpdir:
        raise RunError(
            "TMPDIR is not set, so a compiler would write its temporary files to /tmp on the root filesystem "
            "(Agent Rule 7); set TMPDIR to a directory on the scratch disk (the gate sets it on the build host)"
        )
    scratch = os.environ.get("LASSI_SCRATCH", "")
    if scratch and not _within(Path(tmpdir).resolve(), Path(scratch).resolve()):
        raise RunError(
            f"TMPDIR ({tmpdir}) is outside $LASSI_SCRATCH ({scratch}); on the build host temporary files stay on "
            "the scratch disk, never on the root filesystem (Agent Rule 7); set TMPDIR inside $LASSI_SCRATCH"
        )
    return tmpdir


def _compile_hidden_roots(build_root: Path | None) -> tuple[Path, ...]:
    """Return the roots every compile hides: $HOME, $LASSI_SCRATCH, and $LASSI_RUNS_ROOT (those set), then `build_root`.

    Raises RunError when one of them is not an absolute path, so a bad value
    never changes what a compile sees without notice, and when there is
    none, since a sandbox needs at least one hidden root. Only the roots
    set are hidden: without the gate's $LASSI_SCRATCH the scratch root is
    not known, so a compile then hides $HOME and the build root (for a run,
    its runs root) alone.
    """
    candidates = [(f"${variable}", os.environ.get(variable, "")) for variable in _COMPILE_HIDDEN_VARIABLES]
    if build_root is not None:
        candidates.append(("the build root", str(build_root)))
    roots: list[Path] = []
    for what, value in candidates:
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            raise RunError(f"{what} must be an absolute path to be hidden from compiles, got {value!r}")
        if path not in roots:
            roots.append(path)
    if not roots:
        names = ", ".join(f"${variable}" for variable in _COMPILE_HIDDEN_VARIABLES)
        raise RunError(f"none of {names} is set, so a compile would have no directory to hide; set them")
    return tuple(roots)


def _compile_environment() -> dict[str, str]:
    """Return the clean compile environment: the parent's PATH, LANG=C, and LC_ALL=C.

    HOME stays out on purpose (P0.20), and so does TMPDIR: each compile gets
    its own, a private directory under its build dir (SandboxedCompileRunner).
    """
    return {"PATH": os.environ.get("PATH", os.defpath), "LANG": "C", "LC_ALL": "C"}


def _compile_runner(
    name: str, environment: Mapping[str, str], root: Path, hidden_roots: tuple[Path, ...], build_root: Path | None
) -> SandboxedCompileRunner:
    """Return the toolchain's sandboxed compile runner after checking its layout; RunError names what is refused.

    The sandbox checks its layout (the hidden roots, the toolchains root
    `root`, and the build dir) when a compile builds its spec. Checking it
    here, against a build dir under `build_root`, makes a layout it refuses
    (a hidden root that is a filesystem root, a runs root inside the
    toolchains root, a build dir under /tmp) stop the run before the run
    directory exists, never at its first compile. Nothing is created.
    """
    runner = SandboxedCompileRunner(environment=environment, toolchains=root, hidden_roots=hidden_roots)
    if build_root is not None:
        try:
            runner.spec(build_root / _LAYOUT_PROBE)
        except ValueError as error:
            raise RunError(
                f"toolchain {name!r}: the compile sandbox refuses a build dir under {build_root} with the toolchains "
                f"root {root} and the hidden roots {', '.join(map(str, hidden_roots))}: {error}"
            ) from error
    return runner


def _checked_version(
    name: str,
    executable: Path | str,
    runner: SandboxedCompileRunner,
    tmpdir: Path,
    pin_name: str,
    pin: Mapping[str, str],
) -> tuple[tuple[str, ...], int]:
    """Run `<executable> --version` once through `runner`; return its non-blank lines and its status.

    It runs through the toolchain's own compile runner, so it runs the same
    executable with the same environment in the same view as every build,
    under prlimit --core=1, and proves the compiler reachable there before
    the first build. Its build dir is a fresh directory under the parent's
    TMPDIR, removed afterwards. It must exit 0 and print the pin's
    EXPECT_VERSION, so a run is never labeled with a pin its compiler is not
    (Agent Rule 10); a pin without EXPECT_VERSION is refused before anything
    runs. RunError also covers a directory TMPDIR cannot hold and a layout
    the sandbox refuses there; SandboxUnavailableError propagates.
    """
    expected = pin.get("EXPECT_VERSION", "")
    if not expected:
        raise RunError(
            f"toolchains/{pin_name}.pin has no EXPECT_VERSION, so toolchain {name!r} cannot check that "
            f"{executable} is the pinned compiler (Agent Rule 10); add the --version text it must print"
        )
    try:
        probe = Path(tempfile.mkdtemp(prefix=_VERSION_PROBE_PREFIX, dir=tmpdir))
    except OSError as error:
        raise RunError(f"toolchain {name!r}: cannot make the --version check's directory in TMPDIR: {error}") from error
    try:
        result = runner([str(executable), "--version"], probe, _VERSION_TIMEOUT_S)
    except ValueError as error:
        raise RunError(
            f"toolchain {name!r}: the compile sandbox refuses the --version check's build dir {probe} under TMPDIR "
            f"({tmpdir}): {error}"
        ) from error
    finally:
        shutil.rmtree(probe, ignore_errors=True)
    output = result.stdout + result.stderr
    if result.returncode != 0 or expected not in output:
        raise RunError(
            f"toolchain {name!r}: {executable} --version exited {result.returncode} and must print the "
            f"EXPECT_VERSION of toolchains/{pin_name}.pin ({expected!r}); the compiler is not the pinned one, "
            "so the run would be labeled with a pin it does not use (Agent Rule 10)"
        )
    return tuple(line.rstrip() for line in output.splitlines() if line.strip()), result.returncode


def _trial_pins(toolchains: Iterable[BuiltToolchain]) -> ToolchainPins:
    """Return the VERSION of every pin the given toolchains use, under the matching ToolchainPins field."""
    versions: dict[str, str] = {}
    for built in toolchains:
        for pin_name, pin in built.pins.items():
            field = pin_name.replace("-", "_")
            if field not in TOOLCHAIN_PIN_NAMES:
                raise RunError(f"the pin {pin_name!r} has no Trial toolchain_pins field; fields: {TOOLCHAIN_PIN_NAMES}")
            versions[field] = pin["VERSION"]
    return ToolchainPins(**versions)


def _toolchains_record(toolchains: Sequence[BuiltToolchain]) -> dict[str, Any]:
    """Return toolchains.json: per toolchain name, its languages, executable, environment, and pin files."""
    return {
        built.name: {
            "languages": list(built.languages),
            "executable": built.executable,
            "environment": built.environment,
            "pins": built.pins,
        }
        for built in toolchains
    }


# ---------------------------------------------------------------------------
# Trials


def _run_trials(run: _Run, manifest: Mapping[str, Any]) -> list[Trial]:
    """Run and write every trial: directions in recipe order, items sorted, runs 1 to n.

    Each trial carries the provenance the first-written `manifest` gives its
    direction's target language (_trial_provenance). The recipe's `score`
    profile, when it binds one, scores each trial before it is written
    (_scored). Its trial.md labels its run wall times as simulator wall time
    when its target language's executor declares SIMULATOR (task P4.6).
    """
    trials: list[Trial] = []
    for direction in run.settings.directions:
        provenance = _trial_provenance(manifest, direction.target)
        simulator = declares(run.executors.for_language(direction.target), SIMULATOR)
        for item in run.bench.items:
            for number in range(1, run.settings.trials + 1):
                trial = _scored(run, _run_trial(run, provenance, direction, item, number))
                write_trial(trial, run.run_dir, run.store, simulator=simulator)
                trials.append(trial)
                final = trial.final
                ended = "" if final.end_reason is None else f"  end {final.end_reason.code}"
                print(
                    f"{trial.trial_id}  stage {fmt(final.stage_reached)}  corrections {final.corrections}  "
                    f"wall_s {fmt(final.wall_s)}{ended}",
                    flush=True,
                )
    return trials


def _run_trial(run: _Run, provenance: Provenance, direction: Direction, item: str, number: int) -> Trial:
    """Run the recipe's stages on one new trial, each built on a fresh RunContext, and set its final block.

    The trial id's first segment is the recipe's project. A backend that
    declares `needs_reference` gets the item's reference target, as one
    untagged fence when fixes.fence_tag is off (faithful extraction reads
    that form). A backend that declares `unload_before_run` is asked to
    unload first, as upstream's setup unloads the model before anything
    runs. A stage that sets final.end_reason ends the trial: no later stage
    runs, and the final block keeps the end reason. Every trial starts with
    an empty Trial.requests, so its model requests are recorded and a trial
    that ends before any model call records none ([], never None).
    """
    settings, suite = run.settings, run.bench.suite
    backend = run.backend
    if "needs_reference" in backend.capabilities:
        backend = backend.with_reference(suite.reference_target(item, direction, run.bench.root, purpose=PURPOSE))
        if not run.recipe.data["fixes"]["fence_tag"]:
            backend = backend.with_untagged_fence()
    trial = Trial(
        trial_id=make_trial_id(
            run.recipe.data["project"], arm_segment(settings.model_id), suite.name, direction.name, item, number
        ),
        recipe_hash=run.recipe.recipe_hash,
        toolchain_pins=run.target_pins[direction.target],
        provenance=provenance,
        bench_item=suite.bench_item(item, direction),
        model=model_info(backend, settings.sampling),
        requests=[],
    )
    context = RunContext(
        recipe=run.recipe,
        backend=backend,
        sampling=settings.sampling,
        toolchains={language: built.toolchain for built in run.toolchains for language in built.languages},
        executor=run.executors.for_language(direction.target),
        executors=run.executors.by_language,
        store=run.store,
        suite=suite,
        sources_root=run.bench.root,
        item=item,
        direction=direction,
        build_root=run.run_dir,
        prompts=settings.prompts,
        max_corrections=settings.max_corrections,
        fragments=settings.fragments,
        packs=settings.packs,
    )
    started = time.monotonic()
    unload_before_run(backend)
    stages = [run.registry.get("Stage", name).factory(context=context) for name in run.recipe.data["stages"]]
    for stage in stages:
        trial = stage(trial)
        if trial.final.end_reason is not None:
            break
    return dataclasses.replace(trial, final=_final(trial, time.monotonic() - started))


def _scored(run: _Run, trial: Trial) -> Trial:
    """Return the trial with the bound ScoreProfile's scores (lassi.scoring.run_scoring.score_trial).

    A profile that cannot score the trial raises RunError, which leaves
    provenance.json with status "failed".
    """
    try:
        return score_trial(run.scoring, trial)
    except ScoreError as error:
        raise RunError(f"{run.recipe.path}: score {run.scoring.score!r}: {error}") from error


def _metrics(run: _Run, trials: Sequence[Trial]) -> RunMetrics | None:
    """Return the run's named metrics (lassi.scoring.run_scoring.compute_metrics); None when it names none.

    A profile that cannot score a trial, or tables that cannot be built,
    raise RunError, which leaves provenance.json with status "failed".
    """
    try:
        return compute_metrics(run.scoring, trials)
    except (ScoreError, ValueError) as error:
        raise RunError(f"{run.recipe.path}: metrics: {error}") from error


def _final(trial: Trial, wall_s: float) -> Final:
    """Return the final block of a trial whose stages have run, taking `wall_s` as its wall time.

    It holds the last attempt's stage, the correction count, the alignment
    mean of the attempt whose output stands (standing_attempt; None when no
    attempt ran), and the end reason a stage set. A stage that sets the end
    reason ends the trial, so the oracle stage never aligns a trial that
    ended at the correction cap, and its alignment stays None.
    """
    last = trial.attempts[-1].stage_reached if trial.attempts else None
    named = standing_attempt(trial)
    return Final(
        stage_reached=last,
        alignment=None if named is None else named.alignment.mean,
        corrections=max(len(trial.attempts) - 1, 0),
        wall_s=wall_s,
        end_reason=trial.final.end_reason,
    )


# ---------------------------------------------------------------------------
# Provenance and run.md


def _git_state() -> tuple[str | None, bool | None]:
    """Return the repository's commit and whether `git status --porcelain` lists anything; None when git fails."""
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    commit = (head.strip() or None) if head is not None else None
    dirty = bool(status.strip()) if status is not None else None
    return commit, dirty


def _git(*args: str) -> str | None:
    """Return the stdout of `git <args>` run in the repository, or None when git is missing or fails."""
    try:
        result = EnvRunner(os.environ)(["git", *args], REPO, _GIT_TIMEOUT_S)
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def _utc(moment: datetime) -> str:
    """Return a UTC time as ISO 8601 with seconds, such as 2026-09-23T12:34:56+00:00."""
    return moment.isoformat(timespec="seconds")


def _device(executor: Executor) -> str | None:
    """Return the device an executor's programs run on: its device() (task P4.5; Agent Rule 1).

    An executor without device() (a test fake) names "none (compile only)"
    when it declares compile_only and no device (None, null) otherwise.
    """
    named = getattr(executor, "device", None)
    if callable(named):
        return named()
    return "none (compile only)" if "compile_only" in getattr(executor, "capabilities", ()) else None


def _provenance(run: _Run) -> dict[str, Any]:
    """Return provenance.json as first written: where the run came from, when it started, what it ran on, the pins.

    Its status is "running" and `finished_utc` is null until
    _final_provenance closes it. The executor keys are the run's
    (_Executors.record): executor and device, or, with executors per
    language, executor and devices by language. `driver` (an SDK or driver
    version) stays null until an executor that runs programs reports one.
    """
    pins = dataclasses.asdict(run.pins)
    return {
        "run_id": run.run_dir.name,
        "status": RUNNING,
        "recipe": run.recipe.name,
        "recipe_path": Path(run.recipe.path).as_posix(),
        "recipe_chain": list(run.recipe.chain),
        "recipe_hash": run.recipe.recipe_hash,
        "commit": run.commit,
        "dirty": run.dirty,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        **run.executors.record,
        "driver": None,
        "started_utc": _utc(run.started),
        "finished_utc": None,
        "pins": {name: version for name, version in pins.items() if version is not None},
    }


def _final_provenance(manifest: Mapping[str, Any], status: str) -> dict[str, Any]:
    """Return the first-written manifest with its final `status` and the finish time; every other value stays.

    The trials copied their provenance from that first manifest, so building
    the final one from it (not from the run again) keeps every trial's copy
    equal to provenance.json.
    """
    return {**manifest, "status": status, "finished_utc": _utc(datetime.now(timezone.utc))}


def _trial_provenance(manifest: Mapping[str, Any], language: str) -> Provenance:
    """Return the Trial provenance a run manifest gives a trial whose target is `language`.

    Every trial carries a copy of provenance.json: commit and dirty keep
    their manifest keys; device is the manifest's "device", or, with
    executors per language, its "devices" entry for `language`; sdk is the
    manifest's "driver" and date its "started_utc". A key the manifest lacks
    raises KeyError, and a value of the wrong type raises ValueError: the
    copy never fills in a value the manifest does not hold.
    """
    device = manifest["devices"][language] if "devices" in manifest else manifest["device"]
    return Provenance(
        commit=manifest["commit"],
        dirty=manifest["dirty"],
        device=device,
        sdk=manifest["driver"],
        date=manifest["started_utc"],
    )


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Return a Markdown table block; a pipe in a cell is escaped and a line break becomes a space."""

    def cell(text: str) -> str:
        return " ".join(text.replace("|", "\\|").splitlines())

    lines = [header, ["---"] * len(header), *rows]
    return "".join("| " + " | ".join(cell(text) for text in line) + " |\n" for line in lines)


def _pin_rows(toolchains: Sequence[BuiltToolchain]) -> list[tuple[str, ...]]:
    """Return one run.md row per pin each toolchain uses, or one 'not pinned' row for a toolchain without one."""
    rows: list[tuple[str, ...]] = []
    for built in toolchains:
        languages = ", ".join(built.languages)
        if not built.pins:
            rows.append((built.name, languages, "-", "not pinned", "-"))
        for pin_name, pin in built.pins.items():
            prefix = pin.get("PREFIX_NAME") or f"none, host {pin.get('EXECUTABLE', '-')}"
            rows.append((built.name, languages, pin_name, pin["VERSION"], prefix))
    return rows


def _trial_blocks(trials: Sequence[Trial]) -> list[str]:
    """Return one heading and table per arm (model id), in order of first appearance, linking each trial.md."""
    arms: dict[str, list[Trial]] = {}
    for trial in trials:
        arms.setdefault(trial.model.id, []).append(trial)
    header = ("trial", "direction", "item", "stage reached", "corrections", "wall_s", "trial.md")
    blocks: list[str] = []
    for arm, arm_trials in arms.items():
        rows = [
            (
                f"`{trial.trial_id}`",
                trial.bench_item.direction,
                trial.bench_item.item,
                fmt(trial.final.stage_reached),
                fmt(trial.final.corrections),
                fmt(trial.final.wall_s),
                f"[trial.md]({trial.trial_id}/trial.md)",
            )
            for trial in arm_trials
        ]
        blocks += [f"### Arm {arm}\n", _md_table(header, rows)]
    return blocks or ["No trials.\n"]


def _simulator_note(run: _Run, trials: Sequence[Trial]) -> list[str]:
    """Return the Trials section's note on simulator wall times, or nothing (task P4.6; Agent Rule 2).

    The note is written when the run has trials and _simulated_languages
    names any language; it names them. Otherwise run.md is as before.
    """
    languages = _simulated_languages(run)
    if not trials or not languages:
        return []
    return [
        f"The {', '.join(languages)} programs of this run ran on a simulator executor: their run wall times are "
        "simulator wall time, not performance, and trial.md labels each one it shows. The wall_s column below is "
        "each trial's pipeline wall time, which includes those runs, so it is not performance either.\n"
    ]


def _simulated_languages(run: _Run) -> list[str]:
    """Return, sorted, each language whose programs the run's trials run on an executor that declares SIMULATOR.

    A direction's target language counts when its executor declares it, and
    so does its source language when a listed stage builds the source
    reference under a fix that is on (`source_build_fix`, baseline under
    baseline_both), since that reference then runs on the source language's
    executor and its run time is part of the trial's pipeline wall time.
    """
    stages = [run.registry.get("Stage", name).factory for name in run.recipe.data["stages"]]
    fixes = [getattr(stage, "source_build_fix", None) for stage in stages]
    sources = any(fix is not None and run.recipe.data["fixes"].get(fix, True) is not False for fix in fixes)
    found: set[str] = set()
    for direction in run.settings.directions:
        languages = [direction.target]
        if sources and direction.source != direction.target:
            languages.append(direction.source)
        found.update(language for language in languages if declares(run.executors.for_language(language), SIMULATOR))
    return sorted(found)


def _executor_rows(provenance: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return run.md's executor rows: Executor and Device, or, per language, the executors and one Device row each."""
    if "devices" not in provenance:
        return [("Executor", provenance["executor"]), ("Device", fmt_provenance(provenance["device"]))]
    names = "; ".join(f"{language}: {name}" for language, name in sorted(provenance["executor"].items()))
    devices = sorted(provenance["devices"].items())
    return [("Executor", names), *((f"Device ({language})", fmt_provenance(device)) for language, device in devices)]


def _run_md(
    run: _Run, provenance: Mapping[str, Any], trials: Sequence[Trial], metrics: RunMetrics | None = None
) -> str:
    """Return run.md: the run summary, the resolved recipe, the toolchain pins, and one trials table per arm.

    The Trials section opens with _simulator_note when it applies. With
    `metrics` (the recipe names metrics), a Metrics section follows
    the trials (lassi.scoring.run_scoring.metrics_section). It is
    deterministic for given records, plain ASCII (non-ASCII becomes
    backslash escapes) with LF newlines, and names no absolute run path, so
    it reads the same wherever the run tree is copied.
    """
    summary = [
        ("Recipe", run.recipe.name),
        ("Recipe hash", f"`{run.recipe.recipe_hash}`"),
        ("Commit", fmt_provenance(provenance["commit"])),
        ("Dirty", fmt_provenance(provenance["dirty"])),
        *_executor_rows(provenance),
        ("Driver", fmt_provenance(provenance["driver"])),
        ("Started (UTC)", provenance["started_utc"]),
        ("Finished (UTC)", fmt(provenance["finished_utc"])),
        ("Trials", str(len(trials))),
    ]
    blocks = [f"# Run {run.run_dir.name}\n", _md_table(("Field", "Value"), summary)]
    blocks += ["## Resolved recipe\n", fenced(resolved_yaml(run.recipe), "yaml")]
    pin_header = ("Toolchain", "Languages", "Pin", "Version", "Install prefix")
    blocks += ["## Toolchain pins\n", _md_table(pin_header, _pin_rows(run.toolchains))]
    blocks += ["## Trials\n", *_simulator_note(run, trials), *_trial_blocks(trials)]
    if metrics is not None:
        blocks.append(metrics_section(metrics, run.scoring.score))
    return "\n".join(blocks).encode("ascii", "backslashreplace").decode("ascii")


def _write(path: Path, text: str) -> None:
    """Write `text` to `path` as ASCII bytes, so newlines stay LF on every OS."""
    path.write_bytes(text.encode("ascii"))
