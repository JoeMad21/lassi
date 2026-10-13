"""Tests for the train layer of `lassi train` (tasks P17.8, P17.9): the refusals, then the records, then the steps.

Bible: Training Module (Compute: a train recipe names its device with no
default, and every checkpoint records the device, the framework pins, and
the resolved train recipe; Safeguards: eval splits are refused by the
trainer, and checkpoints log data split hashes), Project Recipes (train.yaml;
the Notes on device sections and train recipes), Component Interfaces
(Trainer), Result Record (provenance), Agent Rules 5, 7, and 10;
plans/p17-portable.md, tasks P17.8 and P17.9.

The contract these tests fix (lassi.train.run):

- run_training(path, options=TrainOptions()) -> Path loads the train recipe
  (lassi.core.recipe.load_train_recipe), then refuses, before any directory
  exists: a key the layer does not carry out (NOT_CARRIED_OUT: episode,
  reward, adversary, export; task P17.9 passes lora and rollout to the
  Trainer's check), naming every one the recipe sets, in that order; a runs
  root that is not absolute, that resolves
  inside the repository, or outside $LASSI_SCRATCH when that is set, or
  that no source gives (the option, else $LASSI_RUNS_ROOT, else the
  recipe's runs_root); a train directory that resolves inside the
  repository although its runs root does not (check_location); a train id
  that is not one plain segment and a train directory that exists; data
  that does not load (a synthetic fixture that is missing, not plain ASCII,
  not LF-only JSON Lines of objects, holds a repeated key, NaN, Infinity,
  or a number too large for a float, is empty, or links out of the fixture
  directory; a bench suite that does not load; a bench item that is
  unknown, repeated, or eval or unassigned, read through Suite.item for
  purpose train, so EvalSplitError becomes a RunError; a split with no
  items); a device the host lacks (the probes of lassi.core.devices, never
  a fallback); a framework() that is missing, raises, returns None, or
  returns something other than a FrameworkBuild; a packages declaration
  that is missing or not a collection of distribution names; a trainer
  that refuses its settings when built as factory(**config); and, task
  P17.9, a trainer without a callable check(), or whose check(recipe.data,
  data), called right after the build with the resolved recipe mapping and
  the TrainData, raises ValueError, which becomes a RunError naming the
  trainer.
- The train directory is <runs root>/train/<train id> (the id defaults to
  the UTC start time, YYYYMMDD-HHMMSS). The layer creates it exclusively,
  writes recipe.resolved.yaml and provenance.json (status running) as
  ASCII with LF, creates an empty output/, and only then calls
  trainer.train(job) once: so the resolved recipe, the device record, the
  framework pins, and the data split hash are on disk before any step
  (Agent Rule 10). provenance.json then holds status complete with the
  steps and checkpoints, or failed when train() raised or returned a result
  outside the contract (steps an int of at least 0, not a bool; each
  checkpoint a relative POSIX path, without "..", to an existing directory
  strictly under output/, never output/ itself). It prints
  `train directory: <dir>`.
- provenance.json records the train id, the recipe and its hash, the commit
  and dirty flag, the host, the trainer's registry name, device_records
  (one record, key trainer.device), driver, framework_pins (the
  FrameworkBuild and each declared package's installed version, null when
  it is not installed), and data (the source, the split hash, and for
  synthetic data the fixture name and record_count; for bench data the
  suite, commit, split, items, and manifest sha256).
- The runner's location, runs-root, id, git, device-probe, and framework
  helpers are public (lassi.core.runner check_location, resolve_runs_root,
  run_id_for, git_state, probe_devices, framework_build), so a train tree
  and a run tree follow one set of rules.

Every component is the fake trainer of tests/train/train_fakes.py in a test
Registry, every probe is a fake with SYNTHETIC facts, git answers a
SYNTHETIC commit (tests/train/conftest.py), and the suite and fixtures are
SYNTHETIC files written by the test, except the refusals that read the
committed lassi-hecbench-10 (eval) and tt-pairs-v0 (unassigned) manifests.
No value in this module is a measurement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml
from train_fakes import (
    COMMITTED_FIXTURE,
    CPU_BUILD,
    CPU_NAME,
    FAILURE,
    FAKE_COMMIT,
    FIXTURE,
    HIP_BUILD,
    LAYER_KEYS,
    MISSING,
    NO_EXTRA,
    NO_KFD,
    OUTPUT_DIR,
    PACKAGES,
    PROVENANCE_JSON,
    RECORDS,
    REFUSAL,
    REJECTION,
    REPO,
    RESOLVED_RECIPE,
    ROCM_DRIVER,
    SUITE,
    SUITE_COMMIT,
    TRAIN,
    TRAIN_ID,
    TRAINER,
    TRAINER_KEYS,
    FakeProbe,
    Log,
    Setup,
    bible_train_block,
    canonical_json,
    core_name,
    fake_probes,
    fake_registry,
    fake_trainer,
    jsonl,
    on_bench,
    provenance,
    recipe_data,
    train_data,
    train_run,
    write_recipe,
)

from lassi.bench import Suite
from lassi.core import runner as runner_module
from lassi.core.recipe import RecipeError, resolved_yaml
from lassi.core.record import DeviceRecord
from lassi.core.runner import RunError

# The keys provenance.json holds from its first write (other keys may follow).
PROVENANCE_KEYS = frozenset(
    {
        "train_id", "status", "recipe", "recipe_path", "recipe_chain", "recipe_hash", "commit", "dirty", "host",
        "platform", "python", "trainer", "device_records", "driver", "framework_pins", "data", "started_utc",
        "finished_utc", "steps", "checkpoints",
    }
)
# The keys the final write changes; every other key keeps its first value.
FINAL_KEYS = ("status", "finished_utc", "steps", "checkpoints")
UTC_ID = re.compile(r"\d{8}-\d{6}")


@pytest.fixture
def setup(tmp_path: Path) -> Setup:
    """Return the test's directories and log."""
    return Setup(tmp_path)


def synthetic_hash(records: tuple[dict[str, Any], ...] = RECORDS) -> str:
    """Return the split hash of synthetic records: sha256 of the canonical JSON of their identity."""
    identity = {"source": "synthetic", "records": [dict(record) for record in records]}
    return hashlib.sha256(canonical_json(identity).encode("ascii")).hexdigest()


def loaded_recipe(setup: Setup, path: Path) -> Any:
    """Load the train recipe at `path` as the layer does, with a fake registry of the same declarations."""
    loader = core_name("lassi.core.recipe", "load_train_recipe")
    return loader(path, roots=[setup.recipes], registry=fake_registry(Log()))


# ---------------------------------------------------------------------------
# Options and the keys carried out


def test_train_options_default_every_field_to_none() -> None:
    options = train_run().TrainOptions
    assert dataclasses.is_dataclass(options) and options.__dataclass_params__.frozen
    names = [item.name for item in dataclasses.fields(options)]
    assert names == ["runs_root", "train_id", "registry", "roots", "probes", "suites_dir", "synthetic_dir"]
    assert all(value is None for value in dataclasses.astuple(options()))


def test_not_carried_out_names_the_keys_later_phases_own() -> None:
    # P17.9 passes lora and rollout to the Trainer's check(); episode and reward are P7's and P8's, adversary is
    # P16's, and export is P7's.
    assert train_run().NOT_CARRIED_OUT == tuple(LAYER_KEYS) == ("episode", "reward", "adversary", "export")


@pytest.mark.parametrize("key", list(LAYER_KEYS))
def test_a_key_not_carried_out_is_refused_naming_it(key: str, setup: Setup) -> None:
    with pytest.raises(RunError) as refused:
        setup.run(recipe_data(**{key: LAYER_KEYS[key]}))
    assert key in str(refused.value), str(refused.value)
    assert not setup.runs_root.exists()
    assert "build" not in setup.log.events and "train" not in setup.log.events


def test_every_key_not_carried_out_is_named_in_order(setup: Setup) -> None:
    with pytest.raises(RunError) as refused:
        setup.run(recipe_data(reward=LAYER_KEYS["reward"], episode=LAYER_KEYS["episode"]))
    message = str(refused.value)
    assert "episode" in message and "reward" in message, message
    assert message.index("episode") < message.index("reward"), message
    assert not setup.runs_root.exists()


def test_lora_and_rollout_pass_the_layer_to_the_trainers_check(setup: Setup) -> None:
    train_dir = setup.run(recipe_data(**TRAINER_KEYS))
    assert provenance(train_dir)["status"] == "complete"
    [(checked, _)] = setup.log.checks
    assert {key: checked[key] for key in TRAINER_KEYS} == TRAINER_KEYS


# ---------------------------------------------------------------------------
# The Trainer's check(recipe, data): after the build, before any directory (task P17.9)


def test_check_gets_the_resolved_recipe_and_the_data_after_the_build(setup: Setup) -> None:
    setup.run()
    events = setup.log.events
    assert events.count("check") == 1
    assert events.index("build") < events.index("check") < events.index("train"), events
    [(recipe, data)] = setup.log.checks
    assert recipe == loaded_recipe(setup, setup.recipes / "train.yaml").data
    job = setup.log.jobs[0]
    assert isinstance(data, core_name("lassi.core.interfaces", "TrainData"))
    assert (data.source, data.split_hash, tuple(data.records)) == (
        job.data.source, job.data.split_hash, tuple(job.data.records)
    )


def test_a_check_refusal_is_a_run_error_naming_the_trainer_before_any_directory(setup: Setup) -> None:
    with pytest.raises(RunError) as refused:
        setup.run(recipe_data(trainer={**TRAIN["trainer"], "reject": True}))
    message = str(refused.value)
    assert REJECTION in message and TRAINER in message, message
    assert not setup.runs_root.exists(), "check() runs before any directory exists"
    assert "check" in setup.log.events and "train" not in setup.log.events


def test_the_bible_train_block_is_refused_naming_every_key_not_carried_out(setup: Setup) -> None:
    block = bible_train_block()
    every = fake_trainer(
        setup.log,
        name="trl",
        methods=frozenset(core_name("lassi.core.recipe", "METHODS")),
        weight_modes=frozenset(core_name("lassi.core.recipe", "WEIGHT_MODES")),
    )
    path = write_recipe(setup.recipes, block)
    with pytest.raises(RunError) as refused:
        train_run().run_training(path, setup.options(registry=fake_registry(setup.log, ("trl", every))))
    message = str(refused.value)
    named = [key for key in train_run().NOT_CARRIED_OUT if key in yaml.safe_load(block)]
    assert named, "the bible block sets keys the layer does not carry out yet"
    positions = [message.index(key) for key in named]
    assert positions == sorted(positions), message
    assert not setup.runs_root.exists()


# ---------------------------------------------------------------------------
# The records come before any step (Agent Rule 10)


def test_records_are_written_before_any_step(setup: Setup) -> None:
    train_dir = setup.run()
    path = setup.recipes / "train.yaml"
    assert setup.log.events.count("train") == 1
    seen = setup.log.seen[0]
    assert seen["parent_entries"] == sorted([OUTPUT_DIR, PROVENANCE_JSON, RESOLVED_RECIPE])
    assert seen["out_dir_entries"] == [], "output/ is empty when the first step runs"
    loaded = loaded_recipe(setup, path)
    assert seen["resolved"] == resolved_yaml(loaded)
    first = seen["provenance"]
    assert PROVENANCE_KEYS <= set(first), sorted(PROVENANCE_KEYS - set(first))
    assert (first["status"], first["finished_utc"], first["steps"], first["checkpoints"]) == (
        "running", None, None, None
    )
    assert first["train_id"] == TRAIN_ID == train_dir.name
    assert (first["recipe"], first["recipe_chain"]) == ("train", ["train"])
    assert first["recipe_path"] == path.as_posix()
    assert first["recipe_hash"] == loaded.recipe_hash
    assert (first["commit"], first["dirty"]) == (FAKE_COMMIT, False)
    assert (first["host"], first["platform"], first["python"]) == (
        platform.node(), platform.platform(), platform.python_version()
    )
    assert first["trainer"] == TRAINER
    assert first["started_utc"].endswith("+00:00")


def test_provenance_records_the_device_the_pins_and_the_split_hash(setup: Setup) -> None:
    setup.run()
    first = setup.log.seen[0]["provenance"]
    assert first["device_records"] == [
        {
            "key": "trainer.device", "kind": "cpu", "indices": [], "name": CPU_NAME, "count": 8,
            "memory_bytes": 4096, "driver": None, "runtime": None, "framework": CPU_BUILD[0],
            "framework_version": CPU_BUILD[1],
        }
    ]
    assert first["driver"] is None
    name, version, cuda, hip = CPU_BUILD
    assert first["framework_pins"] == {
        "build": {"name": name, "version": version, "cuda": cuda, "hip": hip},
        "packages": {PACKAGES[0]: importlib.metadata.version(PACKAGES[0]), PACKAGES[1]: None},
    }
    assert first["data"] == {"source": "synthetic", "name": FIXTURE, "record_count": len(RECORDS),
                             "split_hash": synthetic_hash()}


def test_the_job_carries_the_recipe_data_device_provenance_and_output_dir(setup: Setup) -> None:
    train_dir = setup.run()
    job = setup.log.jobs[0]
    seen = setup.log.seen[0]
    assert isinstance(job, core_name("lassi.core.interfaces", "TrainJob"))
    assert dict(job.recipe) == loaded_recipe(setup, setup.recipes / "train.yaml").data
    assert job.recipe_yaml == seen["resolved"]
    assert isinstance(job.device, DeviceRecord) and job.device.key == "trainer.device"
    assert dataclasses.asdict(job.device) == seen["provenance"]["device_records"][0]
    assert yaml.safe_load(yaml.safe_dump(dict(job.provenance))) == seen["provenance"]
    assert Path(job.out_dir) == train_dir / OUTPUT_DIR
    assert isinstance(job.data, core_name("lassi.core.interfaces", "TrainData"))
    assert job.data.source == "synthetic"
    assert [dict(record) for record in job.data.records] == list(RECORDS)
    assert job.data.split_hash == synthetic_hash() == seen["provenance"]["data"]["split_hash"]


def test_the_trainer_is_built_with_its_section_as_given_after_the_probe(setup: Setup) -> None:
    setup.run()
    assert setup.log.configs == [{"device": {"kind": "cpu"}, "steps": 2}]
    events = setup.log.events
    assert events.index("framework") < events.index("probe cpu") < events.index("build") < events.index("check")
    assert events.index("check") < events.index("train")


def test_train_tree_lies_under_the_runs_root_train_directory(
    setup: Setup, capsys: pytest.CaptureFixture[str]
) -> None:
    train_dir = setup.run()
    assert train_dir == setup.train_dir()
    assert sorted(item.name for item in train_dir.iterdir()) == sorted([OUTPUT_DIR, PROVENANCE_JSON, RESOLVED_RECIPE])
    assert (train_dir / OUTPUT_DIR / "checkpoint-1" / "marker").is_file()
    assert sorted(item.name for item in setup.runs_root.iterdir()) == ["train"]
    for name in (PROVENANCE_JSON, RESOLVED_RECIPE):
        raw = (train_dir / name).read_bytes()
        assert raw.isascii() and b"\r" not in raw, name
    first = setup.log.seen[0]["provenance"]
    final = provenance(train_dir)
    assert (final["status"], final["steps"], final["checkpoints"]) == ("complete", 2, ["checkpoint-1"])
    assert final["finished_utc"] and final["finished_utc"].endswith("+00:00")
    unchanged = {key: value for key, value in final.items() if key not in FINAL_KEYS}
    assert unchanged == {key: value for key, value in first.items() if key not in FINAL_KEYS}
    assert capsys.readouterr().out == f"train directory: {train_dir}\n"


def test_the_default_train_id_is_the_utc_start_time(setup: Setup) -> None:
    train_dir = setup.run(train_id=None)
    assert UTC_ID.fullmatch(train_dir.name), train_dir.name
    assert train_dir.parent == setup.runs_root / "train"


def test_runs_root_comes_from_the_option_then_the_environment_then_the_recipe(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    from_recipe = setup.root / "recipe-root"
    from_env = setup.root / "env-root"
    data = recipe_data(runs_root=str(from_recipe))
    assert setup.run(data, runs_root=None, train_id="recipe") == from_recipe / "train" / "recipe"
    monkeypatch.setenv("LASSI_RUNS_ROOT", str(from_env))
    assert setup.run(data, runs_root=None, train_id="env") == from_env / "train" / "env"
    assert setup.run(data, train_id="option") == setup.runs_root / "train" / "option"


def test_a_rocm_device_is_probed_and_recorded_with_no_fallback(setup: Setup) -> None:
    data = recipe_data(trainer={"kind": TRAINER, "device": {"kind": "rocm", "indices": [1]}})
    setup.run(data, registry=fake_registry(setup.log, framework=HIP_BUILD))
    first = setup.log.seen[0]["provenance"]
    record = first["device_records"][0]
    assert (record["key"], record["kind"], record["indices"]) == ("trainer.device", "rocm", [1])
    assert (record["driver"], record["runtime"]) == (ROCM_DRIVER, HIP_BUILD[3])
    assert first["driver"] == ROCM_DRIVER
    assert "probe cpu" not in setup.log.events, "nothing falls back to the CPU"


# ---------------------------------------------------------------------------
# Every refusal comes before any directory


@dataclass(frozen=True)
class Refusal:
    """One refused train run: what to change, the error class, and what its message must name."""

    error: type
    needles: tuple[str, ...]
    data: Callable[[], dict[str, Any]] = lambda: recipe_data()
    content: bytes | None = None
    no_fixture: bool = False
    declared: Mapping[str, Any] = field(default_factory=dict)
    probes: Mapping[str, Any] = field(default_factory=dict)
    options: Callable[[Setup], dict[str, Any]] = lambda setup: {}
    prepare: Callable[[Setup, pytest.MonkeyPatch], None] = lambda setup, monkeypatch: None


def _rocm(indices: list[int]) -> Callable[[], dict[str, Any]]:
    """Return a recipe factory whose trainer names rocm with `indices`."""
    return lambda: recipe_data(trainer={"kind": TRAINER, "device": {"kind": "rocm", "indices": indices}})


def _committed_suites(setup: Setup) -> dict[str, Any]:
    """Return the option that reads the committed suite manifests (assets/bench)."""
    return {"suites_dir": None}


def _scratch_elsewhere(setup: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
    """Set $LASSI_SCRATCH to a directory the runs root is not under."""
    scratch = setup.root / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("LASSI_SCRATCH", str(scratch))


def _fixture_directory(setup: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
    """Put a directory where the fixture file should be."""
    (setup.fixtures / FIXTURE).mkdir(parents=True)


def _raise_no_extra() -> Any:
    """Raise as a framework() does when its extra is not installed."""
    raise ModuleNotFoundError(NO_EXTRA)


LINE_ONE = jsonl(RECORDS[:1])
LINE_TWO = jsonl(RECORDS[1:])

REFUSALS: dict[str, Refusal] = {
    # Splits and bench items (Agent Rule 5).
    "eval-split": Refusal(RecipeError, ("bench.split", "Agent Rule 5"),
                          data=lambda: on_bench(suite="lassi-hecbench-10", split="eval"), options=_committed_suites),
    "unassigned-split": Refusal(RecipeError, ("bench.split", "Agent Rule 5"),
                                data=lambda: on_bench(suite="tt-pairs-v0", split="unassigned"),
                                options=_committed_suites),
    "eval-item": Refusal(RunError, ("bench.items", "layout", "Agent Rule 5"),
                         data=lambda: on_bench(suite="lassi-hecbench-10", items=["layout"]),
                         options=_committed_suites),
    "unassigned-item": Refusal(RunError, ("bench.items", "eltwise_binary", "Agent Rule 5"),
                               data=lambda: on_bench(suite="tt-pairs-v0", items=["eltwise_binary"]),
                               options=_committed_suites),
    "suite-without-train-items": Refusal(RunError, ("lassi-hecbench-10", "train"),
                                         data=lambda: on_bench(suite="lassi-hecbench-10"), options=_committed_suites),
    "eval-item-of-a-train-suite": Refusal(RunError, ("bench.items", "held", "Agent Rule 5"),
                                          data=lambda: on_bench(items=["alpha", "held"])),
    "unknown-item": Refusal(RunError, ("bench.items", "nosuch"), data=lambda: on_bench(items=["nosuch"])),
    "repeated-item": Refusal(RunError, ("bench.items", "alpha"), data=lambda: on_bench(items=["alpha", "alpha"])),
    "unknown-suite": Refusal(RunError, ("bench.suite", "nosuch-suite"), data=lambda: on_bench(suite="nosuch-suite")),
    "suite-not-a-segment": Refusal(RunError, ("bench.suite",), data=lambda: on_bench(suite="../synthetic-train")),
    # Devices: the probes, never a fallback.
    "cuda-without-probe": Refusal(RunError, ("trainer.device", "cuda"), probes={"cuda": None},
                                  data=lambda: recipe_data(
                                      trainer={"kind": TRAINER, "device": {"kind": "cuda", "indices": [0]}})),
    "rocm-probe-refuses": Refusal(RunError, ("trainer.device", NO_KFD), data=_rocm([0]),
                                  declared={"framework": HIP_BUILD}, probes={"rocm": {"refuse": NO_KFD}}),
    "index-past-the-count": Refusal(RunError, ("trainer.device", "index 5"), data=_rocm([5]),
                                    declared={"framework": HIP_BUILD}),
    "cpu-build-on-rocm": Refusal(RunError, ("trainer.device", "HIP"), data=_rocm([0])),
    # The framework build and the pins.
    "framework-raises": Refusal(RunError, ("trainer.device", NO_EXTRA), declared={"framework": _raise_no_extra}),
    "framework-returns-none": Refusal(RunError, ("framework", TRAINER), declared={"framework": lambda: None}),
    "framework-missing": Refusal(RunError, ("framework", TRAINER), declared={"framework": MISSING}),
    "framework-wrong-type": Refusal(RunError, ("FrameworkBuild",), declared={"framework": lambda: "fakefw 1.0"}),
    "packages-missing": Refusal(RunError, ("packages", TRAINER), declared={"packages": MISSING}),
    "packages-bare-string": Refusal(RunError, ("packages", TRAINER), declared={"packages": "pytest"}),
    "packages-non-string": Refusal(RunError, ("packages", TRAINER), declared={"packages": ("pytest", 3)}),
    # Building the trainer.
    "trainer-refuses": Refusal(RunError, (REFUSAL, TRAINER),
                               data=lambda: recipe_data(trainer={**TRAIN["trainer"], "refuse": True})),
    # The Trainer's check (task P17.9).
    "check-refuses": Refusal(RunError, (REJECTION, TRAINER),
                             data=lambda: recipe_data(trainer={**TRAIN["trainer"], "reject": True})),
    "check-missing": Refusal(RunError, ("check", TRAINER), declared={"check": MISSING}),
    # The runs root and the train id (Agent Rule 7).
    "relative-runs-root": Refusal(RunError, ("absolute",), options=lambda setup: {"runs_root": Path("rel-runs")}),
    "runs-root-outside-scratch": Refusal(RunError, ("LASSI_SCRATCH",), prepare=_scratch_elsewhere),
    "no-runs-root": Refusal(RunError, ("--runs-root", "LASSI_RUNS_ROOT", "runs_root"),
                            options=lambda setup: {"runs_root": None}),
    "train-id-not-a-segment": Refusal(RunError, ("../t1",), options=lambda setup: {"train_id": "../t1"}),
    # Synthetic fixtures: JSON Lines of objects, plain ASCII, LF only, not empty.
    "missing-fixture": Refusal(RunError, ("data.synthetic", FIXTURE), no_fixture=True),
    "fixture-is-a-directory": Refusal(RunError, ("data.synthetic", FIXTURE), no_fixture=True,
                                      prepare=_fixture_directory),
    "non-ascii": Refusal(RunError, ("data.synthetic",), content=b'{"prompt": "caf\xc3\xa9"}\n'),
    "crlf": Refusal(RunError, ("data.synthetic",), content=jsonl(RECORDS).replace(b"\n", b"\r\n")),
    "lone-cr": Refusal(RunError, ("data.synthetic",), content=LINE_ONE[:-1] + b"\r" + LINE_TWO),
    "blank-line": Refusal(RunError, ("data.synthetic",), content=LINE_ONE + b"\n" + LINE_TWO),
    "repeated-key": Refusal(RunError, ("data.synthetic",), content=b'{"prompt": "a", "prompt": "b"}\n'),
    "nan": Refusal(RunError, ("data.synthetic",), content=b'{"prompt": NaN}\n'),
    "infinity": Refusal(RunError, ("data.synthetic",), content=b'{"prompt": -Infinity}\n'),
    "float-overflow": Refusal(RunError, ("data.synthetic", "line 1", "1e400"), content=b'{"prompt": 1e400}\n'),
    "not-an-object": Refusal(RunError, ("data.synthetic",), content=b'["prompt", "completion"]\n'),
    "not-json": Refusal(RunError, ("data.synthetic",), content=LINE_ONE + b'{"prompt": \n'),
    "empty": Refusal(RunError, ("data.synthetic",), content=b""),
    "only-a-newline": Refusal(RunError, ("data.synthetic",), content=b"\n"),
}


def attempt(setup: Setup, refusal: Refusal, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run the refused case and return its message; it must raise the case's error class."""
    log = setup.log
    probes = {kind: None if change is None else FakeProbe(kind, log, **{"count": 2, **change})
              for kind, change in refusal.probes.items()}
    setup.write(refusal.data(), content=refusal.content)
    if refusal.no_fixture:
        (setup.fixtures / FIXTURE).unlink()
    refusal.prepare(setup, monkeypatch)
    options = setup.options(
        registry=fake_registry(log, **refusal.declared), probes=fake_probes(log, **probes), **refusal.options(setup)
    )
    with pytest.raises(refusal.error) as refused:
        train_run().run_training(setup.recipes / "train.yaml", options)
    return str(refused.value)


@pytest.mark.parametrize("case", list(REFUSALS.values()), ids=list(REFUSALS))
def test_every_refusal_comes_before_any_directory(
    case: Refusal, setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    message = attempt(setup, case, monkeypatch)
    missing = [needle for needle in case.needles if needle not in message]
    assert not missing, f"the message does not name {missing}: {message}"
    assert not setup.runs_root.exists(), "a refusal comes before any directory exists"
    assert not Path("rel-runs").exists()
    assert "train" not in setup.log.events, "no step ran"


def test_a_runs_root_inside_the_repository_is_refused_before_any_directory(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    inside = REPO / "p17-8-refused-runs-root"
    refusal = Refusal(RunError, ("inside the repository",), options=lambda setup: {"runs_root": inside})
    try:
        message = attempt(setup, refusal, monkeypatch)
        assert "inside the repository" in message and "Agent Rule 7" in message, message
        assert not inside.exists(), "a refused runs root is never created"
    finally:
        if inside.exists():
            shutil.rmtree(inside)


def test_a_train_directory_inside_the_repository_is_refused_before_any_directory(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A stand-in repository named "train": its parent passes as a runs root, but <root>/train/<id> lies inside it.
    stand_in = setup.root / "outer" / "train"
    stand_in.mkdir(parents=True)
    monkeypatch.setattr(runner_module, "REPO", stand_in)
    refusal = Refusal(RunError, (), options=lambda setup: {"runs_root": setup.root / "outer"})
    message = attempt(setup, refusal, monkeypatch)
    assert re.search(r"the train directory .* inside the repository", message), message
    assert list(stand_in.iterdir()) == [], "nothing is written into the repository"
    assert "train" not in setup.log.events


def test_an_existing_train_directory_is_refused_and_left_alone(setup: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = setup.train_dir()
    existing.mkdir(parents=True)
    (existing / "keep").write_bytes(b"SYNTHETIC earlier run\n")
    message = attempt(setup, Refusal(RunError, ()), monkeypatch)
    assert "already exists" in message, message
    assert sorted(item.name for item in existing.iterdir()) == ["keep"]
    assert "train" not in setup.log.events


def test_a_fixture_refusal_names_the_file_and_the_line(setup: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
    message = attempt(setup, Refusal(RunError, (), content=LINE_ONE + b"[1, 2]\n"), monkeypatch)
    assert "data.synthetic" in message and FIXTURE in message and "line 2" in message, message


def test_a_fixture_that_links_out_of_the_fixture_directory_is_refused(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = setup.root / "elsewhere" / FIXTURE
    outside.parent.mkdir(parents=True)
    outside.write_bytes(jsonl(RECORDS))
    setup.fixtures.mkdir(parents=True)
    try:
        os.symlink(outside, setup.fixtures / "linked.jsonl")
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"this host cannot make a symbolic link here: {error}")
    data = recipe_data(data={"synthetic": "linked.jsonl"})
    message = attempt(setup, Refusal(RunError, (), data=lambda: data), monkeypatch)
    assert "data.synthetic" in message and "linked.jsonl" in message, message
    assert not setup.runs_root.exists()


def test_a_missing_package_is_recorded_as_null_not_refused(setup: Setup) -> None:
    setup.run(registry=fake_registry(setup.log, packages=("lassi-no-such-dist",)))
    assert setup.log.seen[0]["provenance"]["framework_pins"]["packages"] == {"lassi-no-such-dist": None}


# ---------------------------------------------------------------------------
# Bench data goes through the registry for purpose train (Agent Rule 5)


def spy_on_suite_item(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Record every Suite.item call as (item, purpose), passing it through."""
    calls: list[tuple[str, str]] = []
    original = Suite.item

    def spy(self: Suite, name: str, *, purpose: str) -> Any:
        calls.append((name, purpose))
        return original(self, name, purpose=purpose)

    monkeypatch.setattr(Suite, "item", spy)
    return calls


@pytest.mark.parametrize(
    ("items", "expected"),
    [(None, ["alpha", "beta"]), (["beta"], ["beta"]), (["beta", "alpha"], ["alpha", "beta"])],
    ids=["whole-train-split", "one-item", "named-items-sorted"],
)
def test_bench_items_are_read_through_the_registry_for_train(
    items: list[str] | None, expected: list[str], setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = spy_on_suite_item(monkeypatch)
    data = on_bench() if items is None else on_bench(items=items)
    data["trainer"] = {**TRAIN["trainer"], "forbidden": "held"}
    setup.run(data)
    assert calls, "no item was read through Suite.item"
    assert {purpose for _, purpose in calls} == {"train"}, calls
    assert set(expected) <= {name for name, _ in calls}
    job = setup.log.jobs[0]
    assert job.data.source == "bench" and list(job.data.items) == expected and tuple(job.data.records) == ()
    assert [event for event in setup.log.events if event.startswith("bench_item")] == [
        f"bench_item {name}" for name in expected
    ]
    assert len(setup.log.refusals) == 1 and "Agent Rule 5" in setup.log.refusals[0], setup.log.refusals
    manifest = (setup.suites / f"{SUITE}.yaml").read_bytes()
    identity = {
        "source": "bench", "suite": SUITE, "commit": SUITE_COMMIT, "split": "train", "items": expected,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    split_hash = hashlib.sha256(canonical_json(identity).encode("ascii")).hexdigest()
    assert setup.log.seen[0]["provenance"]["data"] == {**identity, "split_hash": split_hash}
    assert job.data.split_hash == split_hash


# ---------------------------------------------------------------------------
# After the records: a failing trainer and a result outside the contract


def test_a_failing_trainer_leaves_provenance_failed(setup: Setup) -> None:
    with pytest.raises(RuntimeError, match=FAILURE):
        setup.run(recipe_data(trainer={**TRAIN["trainer"], "fail": True}))
    final = provenance(setup.train_dir())
    assert final["status"] == "failed" and final["finished_utc"], final
    assert (final["steps"], final["checkpoints"]) == (None, None)
    assert (setup.train_dir() / RESOLVED_RECIPE).is_file()


@pytest.mark.parametrize(
    "result",
    [
        {"checkpoint": "../escape"},
        {"checkpoint": "missing", "create": False},
        {"checkpoint": "/absolute/checkpoint", "create": False},
        {"checkpoint": "sub\\checkpoint", "create": False},
        {"checkpoint": ".", "create": False},
        {"steps": True},
        {"steps": -1},
        {"steps": "2"},
    ],
    ids=["parent", "missing", "absolute", "backslash", "output-itself", "bool-steps", "negative-steps", "string-steps"],
)
def test_a_result_outside_the_contract_is_refused_and_leaves_provenance_failed(
    result: dict[str, Any], setup: Setup
) -> None:
    with pytest.raises(RunError):
        setup.run(recipe_data(trainer={**TRAIN["trainer"], **result}))
    final = provenance(setup.train_dir())
    assert final["status"] == "failed", final


def test_a_nested_checkpoint_path_is_kept(setup: Setup) -> None:
    train_dir = setup.run(recipe_data(trainer={**TRAIN["trainer"], "checkpoint": "steps/checkpoint-2"}))
    assert provenance(train_dir)["checkpoints"] == ["steps/checkpoint-2"]


# ---------------------------------------------------------------------------
# The default fixture directory


def test_default_synthetic_dir_is_tests_fixtures_train_outside_assets_bench() -> None:
    directory = Path(train_data().SYNTHETIC_DIR)
    assert directory == REPO / "tests" / "fixtures" / "train"
    assert REPO / "assets" / "bench" not in [directory, *directory.parents]


def test_the_default_synthetic_dir_is_read_when_no_option_names_one(setup: Setup) -> None:
    committed = Path(train_data().SYNTHETIC_DIR) / COMMITTED_FIXTURE
    raw = committed.read_bytes()
    assert raw.isascii() and b"\r" not in raw
    records = tuple(json.loads(line) for line in raw.decode("ascii").splitlines())
    setup.run(recipe_data(data={"synthetic": COMMITTED_FIXTURE}), synthetic_dir=None)
    data = setup.log.seen[0]["provenance"]["data"]
    assert (data["name"], data["record_count"]) == (COMMITTED_FIXTURE, len(records))
    assert data["split_hash"] == synthetic_hash(records)


# ---------------------------------------------------------------------------
# The runner helpers the layer reuses, public with their behavior unchanged


RUNNER_HELPERS = ("check_location", "resolve_runs_root", "run_id_for", "git_state", "probe_devices", "framework_build")


@pytest.mark.parametrize("name", RUNNER_HELPERS)
def test_the_runner_helpers_the_layer_reuses_are_public(name: str) -> None:
    helper = core_name("lassi.core.runner", name)
    assert callable(helper) and (helper.__doc__ or "").strip(), name


def test_the_public_runner_helpers_keep_the_run_tree_rules(setup: Setup) -> None:
    check_location = core_name("lassi.core.runner", "check_location")
    with pytest.raises(RunError, match="inside the repository"):
        check_location(REPO / "anything", "the train directory")
    check_location(setup.root / "outside", "the train directory")
    run_id_for = core_name("lassi.core.runner", "run_id_for")
    started = datetime(2026, 10, 6, 1, 2, 3, tzinfo=timezone.utc)
    assert run_id_for(None, started) == "20261006-010203"
    assert run_id_for("given", started) == "given"
    with pytest.raises(RunError):
        run_id_for("../escape", started)
    assert core_name("lassi.core.runner", "git_state")() == (FAKE_COMMIT, False)


def test_resolve_runs_root_names_every_source_when_none_gives_one(setup: Setup) -> None:
    path = setup.write()
    recipe = loaded_recipe(setup, path)
    resolve = core_name("lassi.core.runner", "resolve_runs_root")
    assert resolve(setup.runs_root, recipe) == setup.runs_root
    with pytest.raises(RunError) as refused:
        resolve(None, recipe)
    message = str(refused.value)
    assert "--runs-root" in message and "LASSI_RUNS_ROOT" in message and "runs_root" in message, message


def test_no_step_runs_twice_and_the_trainer_is_built_once(setup: Setup) -> None:
    setup.run()
    assert setup.log.events.count("build") == 1 and setup.log.events.count("train") == 1


def test_a_trainer_declaring_methods_and_sources_as_lists_loads(setup: Setup) -> None:
    registry = fake_registry(setup.log, methods=["sft"], weight_modes=("full",), data_sources=["synthetic"])
    assert setup.run(registry=registry) == setup.train_dir()


def test_the_fake_is_never_in_the_default_registry() -> None:
    from lassi.core.registry import DEFAULT_REGISTRY, INTERFACES

    if "Trainer" not in INTERFACES:
        pytest.fail("the registry has no Trainer interface yet (task P17.8)")
    modules = [DEFAULT_REGISTRY.get("Trainer", name).factory.__module__ for name in DEFAULT_REGISTRY.names("Trainer")]
    assert all(module.startswith("lassi.") for module in modules), modules


def test_the_layer_never_falls_back_when_the_named_gpu_kind_has_no_probe(setup: Setup) -> None:
    data = recipe_data(trainer={"kind": TRAINER, "device": {"kind": "cuda", "indices": [0]}})
    with pytest.raises(RunError):
        setup.run(data, probes=fake_probes(setup.log, cuda=None))
    assert "probe cpu" not in setup.log.events and "build" not in setup.log.events
    assert not setup.runs_root.exists()
