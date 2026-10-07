"""The train layer behind `lassi train`: every refusal first, then the records, then the steps (task P17.8).

run_training(path, options) loads a train recipe (lassi.core.recipe
load_train_recipe) and, before any directory exists, refuses:

- a key this layer does not carry out yet (NOT_CARRIED_OUT), naming every
  one the recipe sets, in that order;
- a runs root or train directory outside the run tree's location rules
  (lassi.core.runner resolve_runs_root, run_id_for, check_location; Agent
  Rule 7) and a train directory that exists;
- data that does not load (lassi.train.data; a bench item is read only
  through Suite.item for purpose train, Agent Rule 5);
- a device the host lacks (lassi.core.runner probe_devices, the same probe
  path as lassi run; never a fallback);
- a Trainer whose framework() is missing, raises, or returns anything but a
  FrameworkBuild, or whose packages declaration is not a collection of
  distribution names (Agent Rule 10);
- a Trainer that refuses its settings when built as factory(**config);
- a Trainer without a callable check(), or whose check(recipe, data),
  called with the resolved recipe mapping and the TrainData, raises
  ValueError for a recipe key, value, or record it does not carry out
  (task P17.9).

It then creates <runs root>/train/<train id>/ exclusively and writes
recipe.resolved.yaml and provenance.json (status running) as ASCII with LF,
creates the empty output/, and only then calls trainer.train(job) once, so
the resolved train recipe, the device record, the framework pins, and the
data split hash are on disk before any step (Agent Rule 10). provenance.json
is rewritten with status complete, the steps, and the checkpoints, or with
status failed when train() raises or returns a result outside the contract.
It prints `train directory: <dir>`; the live training table is P7's.

The layer writes only the train directory, its two files, and output/; the
Trainer writes only under output/ (lassi.core.interfaces Trainer). Its
modules import only the standard library and lassi at module level, so
importing lassi.train registers every trainer without a framework.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.metadata
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from lassi.core.devices import DEFAULT_PROBES, DeviceProbe, FrameworkBuild, device_driver
from lassi.core.interfaces import TrainData, TrainJob, TrainResult
from lassi.core.recipe import Recipe, load_train_recipe, resolved_yaml, train_data_source
from lassi.core.record import DeviceRecord, json_text
from lassi.core.registry import DEFAULT_REGISTRY, Binding, Entry, Registry
from lassi.core.runner import (
    BENCH_DIR,
    RunError,
    check_location,
    framework_build,
    git_state,
    probe_devices,
    resolve_runs_root,
    run_id_for,
)
from lassi.train import data as train_data

# Train recipe keys no Trainer carries out yet; a recipe that sets one is refused, never trained without it: the
# episode and the executing reward (P7, P8), the adversary (P16), and export (P7). lora and rollout pass to the
# Trainer's check(), which refuses what it does not carry out (task P17.9).
NOT_CARRIED_OUT: tuple[str, ...] = ("episode", "reward", "adversary", "export")
# The train tree: <runs root>/TRAIN_DIR/<train id>/ holds RESOLVED_RECIPE, PROVENANCE_JSON, and OUTPUT_DIR.
TRAIN_DIR = "train"
RESOLVED_RECIPE = "recipe.resolved.yaml"
PROVENANCE_JSON = "provenance.json"
OUTPUT_DIR = "output"
# The status provenance.json records: while the steps run, after train() raised or broke the contract, and at the end.
RUNNING, FAILED, COMPLETE = "running", "failed", "complete"


@dataclass(frozen=True)
class TrainOptions:
    """How to run a train recipe; every None falls back as each field says.

    - runs_root: the runs root; None means $LASSI_RUNS_ROOT, then the recipe's
      runs_root (lassi.core.runner resolve_runs_root).
    - train_id: the train directory's name; None means the UTC start time as
      YYYYMMDD-HHMMSS.
    - registry: the trainers a recipe may bind; None means DEFAULT_REGISTRY.
    - roots: the directories searched for a recipe named in `extends`; None
      means lassi.core.recipe.default_roots().
    - probes: the device probes by kind; None means
      lassi.core.devices.DEFAULT_PROBES.
    - suites_dir: where bench.suite's manifest is; None means
      lassi.core.runner.BENCH_DIR.
    - synthetic_dir: where data.synthetic's fixture is; None means
      lassi.train.data.SYNTHETIC_DIR.
    """

    runs_root: Path | None = None
    train_id: str | None = None
    registry: Registry | None = None
    roots: Sequence[Path] | None = None
    probes: Mapping[str, DeviceProbe] | None = None
    suites_dir: Path | None = None
    synthetic_dir: Path | None = None


# The options run_training uses when none are given: every field falls back as TrainOptions says.
_DEFAULT_OPTIONS = TrainOptions()


@dataclass(frozen=True)
class _Training:
    """A checked train run: the recipe, the built trainer, the data, the device record, the pins, and where."""

    recipe: Recipe
    trainer: Any
    data: TrainData
    data_record: Mapping[str, Any]
    device: DeviceRecord
    framework_pins: Mapping[str, Any]
    train_dir: Path
    started: datetime
    commit: str | None
    dirty: bool | None


def run_training(path: Path, options: TrainOptions = _DEFAULT_OPTIONS) -> Path:
    """Run the train recipe at `path` and return its train directory, `<runs root>/train/<train id>`.

    Raises RecipeError when the recipe does not load and RunError when the
    run cannot start, both before any directory exists. A Trainer's own
    error from train() propagates, and a result outside the contract is a
    RunError; either leaves provenance.json with status failed.
    """
    training = _prepare(Path(path), options, datetime.now(timezone.utc))
    train_dir = training.train_dir
    recipe_yaml = resolved_yaml(training.recipe)
    manifest = _provenance(training)
    first = json_text(manifest)  # built before the directory exists, so a text that cannot be built leaves none
    _make_train_dir(train_dir)
    _write(train_dir / RESOLVED_RECIPE, recipe_yaml)
    _write(train_dir / PROVENANCE_JSON, first)
    out_dir = train_dir / OUTPUT_DIR
    try:
        out_dir.mkdir()
        job = TrainJob(
            recipe=copy.deepcopy(training.recipe.data),
            recipe_yaml=recipe_yaml,
            data=training.data,
            device=training.device,
            provenance=copy.deepcopy(manifest),
            out_dir=out_dir,
        )
        steps, checkpoints = _checked_result(training.recipe, training.trainer.train(job), out_dir)
    except BaseException:
        _write(train_dir / PROVENANCE_JSON, json_text(_final(manifest, FAILED)))
        raise
    final = {**_final(manifest, COMPLETE), "steps": steps, "checkpoints": list(checkpoints)}
    _write(train_dir / PROVENANCE_JSON, json_text(final))
    print(f"train directory: {train_dir}", flush=True)
    return train_dir


def _prepare(path: Path, options: TrainOptions, started: datetime) -> _Training:
    """Load and check the recipe, read the data, probe the device, read the pins, and build the trainer.

    DEFAULT_REGISTRY is read here, when the run starts, so a caller may put
    another registry in its place.
    """
    registry = DEFAULT_REGISTRY if options.registry is None else options.registry
    recipe = load_train_recipe(path, roots=options.roots, registry=registry)
    _check_carried_out(recipe)
    runs_root = resolve_runs_root(options.runs_root, recipe)
    train_dir = runs_root / TRAIN_DIR / run_id_for(options.train_id, started)
    check_location(train_dir, f"the train directory {train_dir}")
    if train_dir.exists():
        raise RunError(f"the train directory {train_dir} already exists; choose another train id")
    data, data_record = _data(recipe, options)
    (device,) = probe_devices(recipe, registry, DEFAULT_PROBES if options.probes is None else options.probes)
    binding = recipe.bindings[0]
    entry = registry.get(binding.interface, binding.name)
    pins = {"build": dataclasses.asdict(_framework(recipe, binding, entry)), "packages": _packages(recipe, entry)}
    trainer = _build(recipe, binding, entry)
    _check_trainer(recipe, entry, trainer, data)
    commit, dirty = git_state()
    return _Training(
        recipe=recipe,
        trainer=trainer,
        data=data,
        data_record=data_record,
        device=device,
        framework_pins=pins,
        train_dir=train_dir,
        started=started,
        commit=commit,
        dirty=dirty,
    )


def _make_train_dir(train_dir: Path) -> None:
    """Create the train directory exclusively; one that appeared since the check is a RunError."""
    try:
        train_dir.mkdir(parents=True)
    except FileExistsError:
        raise RunError(f"the train directory {train_dir} already exists; choose another train id") from None


def _check_carried_out(recipe: Recipe) -> None:
    """Refuse a recipe that sets any NOT_CARRIED_OUT key, naming every one it sets, in that order."""
    unused = [key for key in NOT_CARRIED_OUT if key in recipe.data]
    if unused:
        raise RunError(
            f"{recipe.path}: the recipe sets {', '.join(unused)}, which lassi train does not carry out yet; "
            "remove them rather than have the run ignore them"
        )


def _data(recipe: Recipe, options: TrainOptions) -> tuple[TrainData, dict[str, Any]]:
    """Return the recipe's data and its provenance record: the identity or the fixture's name and record count."""
    if train_data_source(recipe.data) == "bench":
        suites_dir = BENCH_DIR if options.suites_dir is None else options.suites_dir
        data, identity = train_data.load_bench(recipe, suites_dir)
        return data, {**identity, "split_hash": data.split_hash}
    name = recipe.data["data"]["synthetic"]
    root = train_data.SYNTHETIC_DIR if options.synthetic_dir is None else options.synthetic_dir
    data = train_data.load_synthetic(name, root)
    return data, {"source": "synthetic", "name": name, "record_count": len(data.records), "split_hash": data.split_hash}


def _who(entry: Entry) -> str:
    """Name a trainer for a message: its interface, registry name, and recipe key."""
    return f"{entry.interface} {entry.name!r} (trainer.kind)"


def _framework(recipe: Recipe, binding: Binding, entry: Entry) -> FrameworkBuild:
    """Return the Trainer's FrameworkBuild; a missing framework() or a None result is a RunError (Agent Rule 10).

    framework_build (lassi.core.runner) calls it on the class and refuses an
    error it raises or a result that is neither a FrameworkBuild nor None.
    """
    if not callable(getattr(entry.factory, "framework", None)):
        raise RunError(
            f"{recipe.path}: {_who(entry)} defines no framework(); a Trainer names the framework build it trains on, "
            "which every train run records (Agent Rule 10)"
        )
    build = framework_build(recipe, binding, entry)
    if build is None:
        raise RunError(
            f"{recipe.path}: {_who(entry)} framework() returned None; a Trainer must return the FrameworkBuild it "
            "trains on (Agent Rule 10)"
        )
    return build


def _packages(recipe: Recipe, entry: Entry) -> dict[str, str | None]:
    """Return each declared package's installed version, None when it is not installed; RunError when malformed.

    The versions come from the installed distributions' metadata
    (importlib.metadata), so no package is imported.
    """
    declared = getattr(entry.factory, "packages", None)
    is_collection = isinstance(declared, (list, tuple))
    if not is_collection or not all(isinstance(name, str) and name for name in declared):
        raise RunError(
            f"{recipe.path}: {_who(entry)} declares packages as {declared!r}; it must be a list or tuple of "
            "distribution names, whose installed versions are its framework pins (Agent Rule 10)"
        )
    versions: dict[str, str | None] = {}
    for name in declared:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _build(recipe: Recipe, binding: Binding, entry: Entry) -> Any:
    """Build the Trainer as factory(**config) from its trainer section; a ValueError is a RunError naming it."""
    try:
        return entry.factory(**copy.deepcopy(dict(binding.config)))
    except ValueError as error:
        raise RunError(f"{recipe.path}: {_who(entry)} refused its settings: {error}") from None


def _check_trainer(recipe: Recipe, entry: Entry, trainer: Any, data: TrainData) -> None:
    """Call trainer.check(recipe, data) on a copy of the recipe; a missing check() or its ValueError is a RunError."""
    if not callable(getattr(trainer, "check", None)):
        raise RunError(
            f"{recipe.path}: {_who(entry)} defines no check(); a Trainer refuses there every recipe key, value, and "
            "record it does not carry out"
        )
    try:
        trainer.check(copy.deepcopy(recipe.data), data)
    except ValueError as error:
        raise RunError(f"{recipe.path}: {_who(entry)} does not carry out the recipe: {error}") from None


def _checked_result(recipe: Recipe, result: Any, out_dir: Path) -> tuple[int, tuple[str, ...]]:
    """Return a TrainResult's steps and checkpoints, or raise RunError for one outside the Trainer contract.

    steps is an int of at least 0 (not a bool); each checkpoint is a relative
    path written with "/" (no backslash, drive, "..", or "."), normalized,
    naming an existing directory strictly inside out_dir once resolved.
    """
    if not isinstance(result, TrainResult):
        raise RunError(f"{recipe.path}: the trainer returned {result!r}, not a TrainResult")
    steps = result.steps
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise RunError(f"{recipe.path}: the trainer reported steps {steps!r}; steps is an integer of at least 0")
    if not isinstance(result.checkpoints, (tuple, list)):
        raise RunError(f"{recipe.path}: the trainer reported checkpoints {result.checkpoints!r}, not a tuple of paths")
    base = out_dir.resolve()
    for checkpoint in result.checkpoints:
        if not _is_plain_relative(checkpoint):
            raise RunError(
                f"{recipe.path}: the trainer reported the checkpoint {checkpoint!r}; a checkpoint is a relative "
                f"path with / separators, without '..' or '.', inside {out_dir}"
            )
        found = (out_dir / checkpoint).resolve()
        if base not in found.parents or not found.is_dir():
            raise RunError(
                f"{recipe.path}: the trainer reported the checkpoint {checkpoint!r}, which is not a directory "
                f"inside {out_dir}"
            )
    return steps, tuple(result.checkpoints)


def _is_plain_relative(checkpoint: Any) -> bool:
    """Return True for a normalized relative POSIX path with no backslash, drive, "..", or "." part."""
    if not isinstance(checkpoint, str) or not checkpoint or "\\" in checkpoint or PureWindowsPath(checkpoint).drive:
        return False
    posix = PurePosixPath(checkpoint)
    parts = posix.parts
    return not posix.is_absolute() and bool(parts) and ".." not in parts and posix.as_posix() == checkpoint


def _provenance(training: _Training) -> dict[str, Any]:
    """Return provenance.json as first written: the recipe, the commit, the host, the device, the pins, and the data.

    Its status is "running", and finished_utc, steps, and checkpoints are
    null until _final closes it. device_records holds the one DeviceRecord of
    trainer.device and driver its driver (lassi.core.devices device_driver).
    framework_pins holds the FrameworkBuild and each declared package's
    version (null when not installed); data holds the split hash.
    """
    recipe = training.recipe
    return {
        "train_id": training.train_dir.name,
        "status": RUNNING,
        "recipe": recipe.name,
        "recipe_path": Path(recipe.path).as_posix(),
        "recipe_chain": list(recipe.chain),
        "recipe_hash": recipe.recipe_hash,
        "commit": training.commit,
        "dirty": training.dirty,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "trainer": recipe.bindings[0].name,
        "device_records": [dataclasses.asdict(training.device)],
        "driver": device_driver([training.device]),
        "framework_pins": copy.deepcopy(dict(training.framework_pins)),
        "data": dict(training.data_record),
        "started_utc": _utc(training.started),
        "finished_utc": None,
        "steps": None,
        "checkpoints": None,
    }


def _final(manifest: Mapping[str, Any], status: str) -> dict[str, Any]:
    """Return the first-written manifest with its final `status` and the finish time; every other value stays."""
    return {**manifest, "status": status, "finished_utc": _utc(datetime.now(timezone.utc))}


def _utc(moment: datetime) -> str:
    """Return a UTC time as ISO 8601 with seconds, as the runner writes it."""
    return moment.isoformat(timespec="seconds")


def _write(path: Path, text: str) -> None:
    """Write `text` to `path` as ASCII bytes, so newlines stay LF on every OS."""
    path.write_bytes(text.encode("ascii"))
