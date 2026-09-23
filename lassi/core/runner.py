"""The stage runner behind `lassi run <recipe>`: load a recipe, run its stages over every trial, write the run tree.

run_recipe (bible Project Recipes; Component Interfaces; Result Record):

1. loads the recipe with the registry (every component registers when this
   module is imported, since it imports lassi.llm, lassi.toolchains,
   lassi.executors, and lassi.core.stages);
2. refuses what it cannot run, before any directory is created or any model
   is asked: arms without a model (the model registry that maps arms to
   backends comes later) and arms beside a model; a recipe with no
   llm.sampling.max_tokens or no prompt set, since it never picks a value;
   a fix toggle turned off (faithful: true turns them all off), a report
   toggle turned off, or a section this runner does not carry out (context,
   oracle, judges, and the like), since it never ignores a choice; a stage
   list whose order cannot work (one generating stage at most, and every
   compiling stage after it); a prompt set that lacks a template a stage
   renders or uses a placeholder the stage does not fill; trial ids that
   repeat; and missing bench sources. A stage it does not implement already
   fails at load, unregistered;
3. builds the components: the backend as factory(model.id), each toolchain
   with its pinned compiler and a clean environment (below), and the
   executor as factory(**config);
4. loads the suite manifest assets/bench/<bench.suite>.yaml and finds the
   fetched sources (tools/fetch_bench.py puts them under $LASSI_SCRATCH);
5. runs every trial, direction by direction in recipe order, item by item in
   sorted order (the items of the recipe's split), then run 1 to trials.n: the
   stages in recipe order on a fresh RunContext, then the trial's final block;
6. writes the run tree and prints one line per trial and the run directory.

The run tree is <runs root>/runs/<run_id>, where the runs root is the
runs_root option, else $LASSI_RUNS_ROOT, else the recipe's runs_root. The
runs root and the run directory must be absolute, resolve outside the
repository, and, when $LASSI_SCRATCH is set (the gate sets it on the build
host), resolve inside it (Agent Rule 7). The tree holds:

- recipe.resolved.yaml: the resolved recipe, which reruns the run alone
  (Design Principle 5);
- toolchains.json: per toolchain, its languages, pinned executable, compile
  environment, and pin files;
- provenance.json: the commit, the dirty flag, host, Python, the device, the
  start and end times (UTC), recipe hash, pin versions, and a status. It is
  written before the first trial with status "running", and again at the
  end with "complete", or with "failed" when a trial or a write raised, so
  every trial.json in the tree has a commit and a date beside it;
- run.md: the page a person reads (Readability Standards, Run row);
- one directory per trial (one level per trial_id segment) with trial.json,
  trial.md, and attempt<NN>/build, each attempt's fresh build directory
  (the run directory is the stages' build root, so a rerun of the recipe
  never meets an earlier run's builds);
- texts/, the text store of prompts and replies; parquet/, the mirror.

Pinned toolchains (Agent Rule 10): a toolchain class that declares PIN and
PIN_BIN is built as factory(executable=<toolchains root>/<PREFIX_NAME>/
<PIN_BIN>, runner=EnvRunner(env)), where the toolchains root is the
toolchains_root option, else $LASSI_TOOLCHAINS, and the pin is read from
toolchains/<PIN>.pin. The compile environment is the parent's PATH, LANG=C
and LC_ALL=C (ASCII diagnostics), HOME when set, TMPDIR (required, so no
compiler writes temporary files to /tmp on the root filesystem), and each
variable the pin names (NVHPC_CUDA_HOME for toolchains/nvhpc.pin); nothing
else, so a variable such as NVCC_PREPEND_FLAGS never reaches a compile. A
class without PIN (a test fake) is built as factory(). A trial's
toolchain_pins records the pins of the toolchain that builds its target
language; provenance.json records every bound toolchain's pins. git, for the
commit and dirty flag, runs through lassi.toolchains.EnvRunner, the audited
command runner, so this module starts no process itself.
"""

from __future__ import annotations

import dataclasses
import os
import platform
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lassi.bench import Direction, Suite, load_suite, sources_dir
from lassi.core.interfaces import Executor, Sampling, Toolchain
from lassi.core.parquet import write_run_parquet
from lassi.core.recipe import UNCAPPED, Recipe, RecipeError, load_recipe, resolved_data, resolved_yaml
from lassi.core.record import TOOLCHAIN_PIN_NAMES, Final, ToolchainPins, Trial, json_text, make_trial_id
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.stages import PURPOSE, RunContext
from lassi.core.store import TextStore, write_trial
from lassi.core.trial_md import fenced, fmt
from lassi.executors import workdir
from lassi.llm import model_info
from lassi.prompts import render
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
_NOT_CARRIED_OUT = ("context", "oracle", "profiler", "adversary", "metrics", "refine", "score", "agents", "judges")
# How long git may take to report the commit or the dirty flag, in seconds.
_GIT_TIMEOUT_S = 60.0


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
    """The run choices read from a loaded recipe."""

    backend: str
    model_id: str
    sampling: Sampling
    max_corrections: int | None
    prompts: str
    directions: tuple[Direction, ...]
    trials: int
    split: str


@dataclass(frozen=True)
class _Bench:
    """The suite, the root of its fetched sources, and the items the run covers, sorted."""

    suite: Suite
    root: Path
    items: tuple[str, ...]


@dataclass(frozen=True)
class _BuiltToolchain:
    """One toolchain a recipe binds: its registry name, languages, the built object, and how it was pinned.

    `executable` and `environment` are None for a toolchain without a pin;
    `pins` maps each pin it uses to the pin file's pairs.
    """

    name: str
    languages: tuple[str, ...]
    toolchain: Toolchain
    executable: str | None
    environment: dict[str, str] | None
    pins: dict[str, dict[str, str]]


@dataclass(frozen=True)
class _Run:
    """Everything the trial loop and the run files read: the recipe, its components, the bench, and provenance.

    `pins` holds every bound toolchain's pin versions and `target_pins` those
    of the toolchain that builds each target language.
    """

    recipe: Recipe
    registry: Registry
    settings: _Settings
    bench: _Bench
    backend: Any
    executor: Executor
    toolchains: tuple[_BuiltToolchain, ...]
    pins: ToolchainPins
    target_pins: Mapping[str, ToolchainPins]
    run_dir: Path
    store: TextStore
    started: datetime
    commit: str | None
    dirty: bool | None


def run_recipe(path: Path, options: RunOptions = _DEFAULT_OPTIONS) -> Path:
    """Run the recipe at `path` and return its run directory, `<runs root>/runs/<run_id>`.

    Raises RecipeError when the recipe does not load and RunError when the
    run cannot start (see the module docstring); both come before any
    directory is created or any model is asked. A component's own errors,
    such as SandboxUnavailableError from an executor, propagate; one raised
    during the trials leaves provenance.json with status "failed".
    """
    run = _prepare(Path(path), options, datetime.now(timezone.utc))
    run_dir = run.run_dir
    _write(run_dir / RESOLVED_RECIPE, resolved_yaml(run.recipe))
    _write(run_dir / TOOLCHAINS_JSON, json_text(_toolchains_record(run.toolchains)))
    _write(run_dir / PROVENANCE_JSON, json_text(_provenance(run, RUNNING, None)))
    try:
        trials = _run_trials(run)
        provenance = _provenance(run, COMPLETE, datetime.now(timezone.utc))
        _write(run_dir / RUN_MD, _run_md(run, provenance, trials))
        write_run_parquet(trials, run_dir / PARQUET_DIR)
    except BaseException:
        _write(run_dir / PROVENANCE_JSON, json_text(_provenance(run, FAILED, datetime.now(timezone.utc))))
        raise
    _write(run_dir / PROVENANCE_JSON, json_text(provenance))
    print(f"run directory: {run_dir}", flush=True)
    return run_dir


def _prepare(path: Path, options: RunOptions, started: datetime) -> _Run:
    """Load and check the recipe, build its components, and create the run directory; RunError before any mkdir."""
    registry = DEFAULT_REGISTRY if options.registry is None else options.registry
    recipe = _load(path, options, registry)
    settings = _settings(recipe)
    run_dir = _runs_root(options, recipe) / RUNS_DIR / _run_id(options, started)
    _check_location(run_dir, f"the run directory {run_dir}")
    if run_dir.exists():
        raise RunError(f"the run directory {run_dir} already exists; choose another run id")
    bench = _bench(recipe, settings, options)
    _check_plan(recipe, registry, settings, bench)
    backend = registry.get("LLMBackend", settings.backend).factory(settings.model_id)
    if "needs_reference" in backend.capabilities and not hasattr(backend, "with_reference"):
        raise RunError(f"LLMBackend {settings.backend!r} needs the reference target but has no with_reference()")
    toolchains = _toolchains(recipe, registry, _toolchains_root(options))
    pins = _trial_pins(toolchains)
    target_pins = {
        direction.target: _trial_pins(built for built in toolchains if direction.target in built.languages)
        for direction in settings.directions
    }
    executor_binding = next(binding for binding in recipe.bindings if binding.interface == "Executor")
    executor = registry.get("Executor", executor_binding.name).factory(**executor_binding.config)
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
        executor=executor,
        toolchains=toolchains,
        pins=pins,
        target_pins=target_pins,
        run_dir=run_dir,
        store=TextStore(run_dir),
        started=started,
        commit=commit,
        dirty=dirty,
    )


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
    cap = data["loop"]["max_corrections"]
    return _Settings(
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
    """Refuse a recipe choice the stages cannot honor, so the saved recipe never claims behavior the run lacked."""
    data = recipe.data
    for name, on in sorted(data["fixes"].items()):
        if not on:
            raise RunError(
                f"{recipe.path}: fixes.{name} is off (faithful: {fmt(data['faithful'])}), which asks for an upstream "
                "quirk that no stage here reproduces yet; turn the fix on and leave faithful off"
            )
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
    """Load the recipe's suite and find its fetched sources; the items are those of the recipe's split, sorted."""
    name = recipe.data["bench"]["suite"]
    manifest = BENCH_DIR / f"{name}.yaml"
    if not _NAME.fullmatch(name) or not manifest.is_file():
        raise RunError(f"{recipe.path}: bench.suite {name!r} names no suite manifest; expected {manifest}")
    try:
        suite = load_suite(manifest)
    except (OSError, ValueError) as error:
        raise RunError(f"cannot load the suite manifest {manifest}: {error}") from error
    items = tuple(sorted(item for item, spec in suite.items.items() if spec.split == settings.split))
    if not items:
        raise RunError(f"{recipe.path}: suite {name!r} has no items in the split {settings.split!r}")
    return _Bench(suite=suite, root=_sources_root(suite, options), items=items)


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


def _check_plan(recipe: Recipe, registry: Registry, settings: _Settings, bench: _Bench) -> None:
    """Refuse, before anything is built, a plan whose stages, prompts, trial ids, or sources cannot work."""
    if recipe.name in _RUN_TREE_NAMES:
        raise RunError(f"{recipe.path}: a recipe may not be named {recipe.name!r}, which the run tree uses itself")
    _check_stages(recipe, registry, settings)
    _check_trials(recipe, settings, bench)


def _check_stages(recipe: Recipe, registry: Registry, settings: _Settings) -> None:
    """Refuse a stage order that cannot run, a target language with no toolchain, and a prompt a stage cannot render.

    A trial gets one first attempt, so at most one stage declares
    `generates`, and a stage that declares `compiles` builds that attempt,
    so it comes after it. Each stage's `prompt_fields` templates are rendered
    once with empty fields, which finds a missing set or file and a
    placeholder the stage does not fill.
    """
    entries = [registry.get("Stage", name) for name in recipe.data["stages"]]
    generated = False
    for index, entry in enumerate(entries):
        where = f"{recipe.path}: stages[{index}] {entry.name!r}"
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
    for entry in entries:
        for prompt, fields in getattr(entry.factory, "prompt_fields", {}).items():
            try:
                render(settings.prompts, prompt, dict.fromkeys(fields, ""))
            except ValueError as error:
                raise RunError(f"{recipe.path}: stage {entry.name!r} cannot use the prompt set: {error}") from error


def _check_trials(recipe: Recipe, settings: _Settings, bench: _Bench) -> None:
    """Refuse trial ids that are invalid or repeat (in any letter case), and bench sources that cannot be read."""
    seen: dict[str, str] = {}
    for direction in settings.directions:
        for item in bench.items:
            try:
                trial_id = make_trial_id(recipe.name, settings.model_id, bench.suite.name, direction.name, item, 1)
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
                bench.suite.source_files(item, direction, bench.root, purpose=PURPOSE)
                bench.suite.reference_target(item, direction, bench.root, purpose=PURPOSE)
            except (OSError, ValueError) as error:
                raise RunError(
                    f"cannot read {bench.suite.name}/{item} ({direction.name}) under {bench.root}: {error}; "
                    + _fetch_hint(bench.suite)
                ) from error


# ---------------------------------------------------------------------------
# Toolchains and pins


def _toolchains(recipe: Recipe, registry: Registry, root: Path | None) -> tuple[_BuiltToolchain, ...]:
    """Build each toolchain the recipe binds, once per registry name, pinned when its class declares a pin."""
    languages: dict[str, list[str]] = {}
    for language, name in sorted(recipe.data.get("toolchain", {}).items()):
        languages.setdefault(name, []).append(language)
    built: list[_BuiltToolchain] = []
    for name, bound in languages.items():
        factory = registry.get("Toolchain", name).factory
        if getattr(factory, "PIN", None) is None:
            built.append(_BuiltToolchain(name, tuple(bound), factory(), None, None, {}))
        else:
            built.append(_pinned_toolchain(name, tuple(bound), factory, root))
    return tuple(built)


def _pinned_toolchain(name: str, languages: tuple[str, ...], factory: type, root: Path | None) -> _BuiltToolchain:
    """Build a toolchain with its pinned executable and a clean compile environment; RunError says what is missing."""
    pin_name = factory.PIN
    pin = _pin(pin_name)
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
    executable = root / pin["PREFIX_NAME"] / relative
    if not executable.is_file():
        raise RunError(
            f"toolchain {name!r}: the pinned compiler {executable} does not exist; "
            f"install it on the build host with toolchains/{pin_name}.sh"
        )
    environment = _compile_environment()
    pins = {pin_name: pin}
    for variable, prefix in linked_prefixes(pin).items():
        try:
            linked_name = prefix_pin_name(prefix)
        except ValueError as error:
            raise RunError(f"{pin_name}.pin: {error}") from error
        linked = _pin(linked_name)
        home = root / prefix
        if linked["PREFIX_NAME"] != prefix or not home.is_dir():
            raise RunError(
                f"toolchain {name!r} needs {home} ({pin_name}.pin), which is not the installed {linked_name} pin; "
                f"install it on the build host with toolchains/{linked_name}.sh"
            )
        environment[variable] = str(home)
        pins[linked_name] = linked
    toolchain = factory(executable=str(executable), runner=EnvRunner(environment))
    return _BuiltToolchain(name, languages, toolchain, str(executable), environment, pins)


def _pin(name: str) -> dict[str, str]:
    """Return toolchains/<name>.pin as read_pin reads it; RunError when it is missing or lacks a needed key."""
    try:
        pin = read_pin(name)
    except (OSError, ValueError) as error:
        raise RunError(f"cannot read the pin file toolchains/{name}.pin: {error}") from error
    for key in ("VERSION", "PREFIX_NAME"):
        if not pin.get(key):
            raise RunError(f"toolchains/{name}.pin has no {key}")
    return pin


def _compile_environment() -> dict[str, str]:
    """Return the clean compile environment: the parent's PATH, LANG=C, LC_ALL=C, HOME when set, and TMPDIR.

    TMPDIR is required: without it a compiler writes its temporary files to
    /tmp, which on the build host is the root filesystem (Agent Rule 7).
    """
    tmpdir = os.environ.get("TMPDIR", "")
    if not tmpdir:
        raise RunError(
            "TMPDIR is not set, so a compiler would write its temporary files to /tmp on the root filesystem "
            "(Agent Rule 7); set TMPDIR to a directory on the scratch disk (the gate sets it on the build host)"
        )
    environment = {"PATH": os.environ.get("PATH", os.defpath), "LANG": "C", "LC_ALL": "C"}
    home = os.environ.get("HOME", "")
    if home:
        environment["HOME"] = home
    environment["TMPDIR"] = tmpdir
    return environment


def _trial_pins(toolchains: Iterable[_BuiltToolchain]) -> ToolchainPins:
    """Return the VERSION of every pin the given toolchains use, under the matching ToolchainPins field."""
    versions: dict[str, str] = {}
    for built in toolchains:
        for pin_name, pin in built.pins.items():
            field = pin_name.replace("-", "_")
            if field not in TOOLCHAIN_PIN_NAMES:
                raise RunError(f"the pin {pin_name!r} has no Trial toolchain_pins field; fields: {TOOLCHAIN_PIN_NAMES}")
            versions[field] = pin["VERSION"]
    return ToolchainPins(**versions)


def _toolchains_record(toolchains: Sequence[_BuiltToolchain]) -> dict[str, Any]:
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


def _run_trials(run: _Run) -> list[Trial]:
    """Run and write every trial: directions in recipe order, items sorted, then runs 1 to trials.n."""
    trials: list[Trial] = []
    for direction in run.settings.directions:
        for item in run.bench.items:
            for number in range(1, run.settings.trials + 1):
                trial = _run_trial(run, direction, item, number)
                write_trial(trial, run.run_dir, run.store)
                trials.append(trial)
                final = trial.final
                print(
                    f"{trial.trial_id}  stage {fmt(final.stage_reached)}  corrections {final.corrections}  "
                    f"wall_s {fmt(final.wall_s)}",
                    flush=True,
                )
    return trials


def _run_trial(run: _Run, direction: Direction, item: str, number: int) -> Trial:
    """Run the recipe's stages on one new trial, each built on a fresh RunContext, and set its final block."""
    settings, suite = run.settings, run.bench.suite
    backend = run.backend
    if "needs_reference" in backend.capabilities:
        backend = backend.with_reference(suite.reference_target(item, direction, run.bench.root, purpose=PURPOSE))
    trial = Trial(
        trial_id=make_trial_id(run.recipe.name, settings.model_id, suite.name, direction.name, item, number),
        recipe_hash=run.recipe.recipe_hash,
        toolchain_pins=run.target_pins[direction.target],
        bench_item=suite.bench_item(item, direction),
        model=model_info(backend, settings.sampling),
    )
    context = RunContext(
        recipe=run.recipe,
        backend=backend,
        sampling=settings.sampling,
        toolchains={language: built.toolchain for built in run.toolchains for language in built.languages},
        executor=run.executor,
        store=run.store,
        suite=suite,
        sources_root=run.bench.root,
        item=item,
        direction=direction,
        build_root=run.run_dir,
        prompts=settings.prompts,
        max_corrections=settings.max_corrections,
    )
    started = time.monotonic()
    stages = [run.registry.get("Stage", name).factory(context=context) for name in run.recipe.data["stages"]]
    for stage in stages:
        trial = stage(trial)
    wall_s = time.monotonic() - started
    last = trial.attempts[-1].stage_reached if trial.attempts else None
    final = Final(stage_reached=last, corrections=max(len(trial.attempts) - 1, 0), wall_s=wall_s)
    return dataclasses.replace(trial, final=final)


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
    """Return the device the run's programs ran on: "none (compile only)" for a compile-only executor, else None.

    None (null) means the executor reports no device yet; an executor that
    runs programs fills this field when it can name its device.
    """
    return "none (compile only)" if "compile_only" in getattr(executor, "capabilities", ()) else None


def _provenance(run: _Run, status: str, finished: datetime | None) -> dict[str, Any]:
    """Return provenance.json: where the run came from, when it ran, what it ran on, the pins, and its status.

    `driver` (an SDK or driver version) stays null until an executor that
    runs programs reports one; `finished_utc` is null while the run runs.
    """
    pins = dataclasses.asdict(run.pins)
    return {
        "run_id": run.run_dir.name,
        "status": status,
        "recipe": run.recipe.name,
        "recipe_path": Path(run.recipe.path).as_posix(),
        "recipe_chain": list(run.recipe.chain),
        "recipe_hash": run.recipe.recipe_hash,
        "commit": run.commit,
        "dirty": run.dirty,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executor": run.recipe.data["executor"]["kind"],
        "device": _device(run.executor),
        "driver": None,
        "started_utc": _utc(run.started),
        "finished_utc": None if finished is None else _utc(finished),
        "pins": {name: version for name, version in pins.items() if version is not None},
    }


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Return a Markdown table block; a pipe in a cell is escaped and a line break becomes a space."""

    def cell(text: str) -> str:
        return " ".join(text.replace("|", "\\|").splitlines())

    lines = [header, ["---"] * len(header), *rows]
    return "".join("| " + " | ".join(cell(text) for text in line) + " |\n" for line in lines)


def _pin_rows(toolchains: Sequence[_BuiltToolchain]) -> list[tuple[str, ...]]:
    """Return one run.md row per pin each toolchain uses, or one 'not pinned' row for a toolchain without one."""
    rows: list[tuple[str, ...]] = []
    for built in toolchains:
        languages = ", ".join(built.languages)
        if not built.pins:
            rows.append((built.name, languages, "-", "not pinned", "-"))
        for pin_name, pin in built.pins.items():
            rows.append((built.name, languages, pin_name, pin["VERSION"], pin["PREFIX_NAME"]))
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


def _run_md(run: _Run, provenance: Mapping[str, Any], trials: Sequence[Trial]) -> str:
    """Return run.md: the run summary, the resolved recipe, the toolchain pins, and one trials table per arm.

    It is deterministic for given records, plain ASCII (non-ASCII becomes
    backslash escapes) with LF newlines, and names no absolute run path, so
    it reads the same wherever the run tree is copied.
    """
    summary = [
        ("Recipe", run.recipe.name),
        ("Recipe hash", f"`{run.recipe.recipe_hash}`"),
        ("Commit", fmt(provenance["commit"])),
        ("Dirty", fmt(provenance["dirty"])),
        ("Executor", provenance["executor"]),
        ("Started (UTC)", provenance["started_utc"]),
        ("Finished (UTC)", fmt(provenance["finished_utc"])),
        ("Trials", str(len(trials))),
    ]
    blocks = [f"# Run {run.run_dir.name}\n", _md_table(("Field", "Value"), summary)]
    blocks += ["## Resolved recipe\n", fenced(resolved_yaml(run.recipe), "yaml")]
    pin_header = ("Toolchain", "Languages", "Pin", "Version", "Install prefix")
    blocks += ["## Toolchain pins\n", _md_table(pin_header, _pin_rows(run.toolchains))]
    blocks += ["## Trials\n", *_trial_blocks(trials)]
    return "\n".join(blocks).encode("ascii", "backslashreplace").decode("ascii")


def _write(path: Path, text: str) -> None:
    """Write `text` to `path` as ASCII bytes, so newlines stay LF on every OS."""
    path.write_bytes(text.encode("ascii"))
