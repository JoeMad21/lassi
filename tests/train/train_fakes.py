"""Fakes and file helpers for the `lassi train` tests (task P17.8).

A fake Trainer registered in a test Registry stands in for a real one, so
the layer (lassi.train.run), the data module (lassi.train.data), and the
command line are tested with no framework. The fake declares what a Trainer
declares (bible Component Interfaces, the Trainer contract rule): the
capability takes_device, methods, weight_modes, data_sources, packages, a
staticmethod framework() that returns a SYNTHETIC FrameworkBuild, and its
config keys. When train(job) runs it records what was on disk at that
moment (the resolved recipe and provenance.json beside job.out_dir, and
whether job.out_dir was empty), so a test can show that the records came
before any step, reads every bench item through job.data.bench_item, and
writes one checkpoint directory under job.out_dir. Config keys turn on its
failure modes: refuse (a ValueError when built), fail (an error during the
steps), checkpoint and create (the checkpoint it reports and whether it
makes it), steps (the step count it reports), and forbidden (an item it
asks for through bench_item, expecting EvalSplitError).

The probes answer with SYNTHETIC host facts and every framework build is
SYNTHETIC. The fake is never registered in lassi's DEFAULT_REGISTRY. No
value in this module is a measurement.
"""

from __future__ import annotations

import copy
import importlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from lassi.bench import EvalSplitError
from lassi.core.registry import Registry

REPO = Path(__file__).resolve().parents[2]
BIBLE = REPO / "docs" / "BIBLE.md"
TAKES_DEVICE = "takes_device"
# Stands for an attribute or key a test leaves out.
MISSING = object()

TRAINER = "fake"
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
RESOLVED_RECIPE = "recipe.resolved.yaml"
PROVENANCE_JSON = "provenance.json"
OUTPUT_DIR = "output"
TRAIN_ID = "t1"

# SYNTHETIC framework builds (name, version, cuda, hip).
CPU_BUILD = ("fakefw", "1.0", None, None)
HIP_BUILD = ("fakefw", "1.0+synthetic-rocm", None, "9.9.99999-synthetic")
# SYNTHETIC host facts and refusals.
CPU_NAME = "SYNTHETIC CPU model 9000"
ROCM_DRIVER = "SYNTHETIC-amdgpu-6.8.5"
NO_KFD = "SYNTHETIC: this host has no kfd node"
NO_EXTRA = "SYNTHETIC: no framework extra is installed"
REFUSAL = "SYNTHETIC: the fake trainer refuses its settings"
FAILURE = "SYNTHETIC: the fake trainer failed during its steps"
# The distributions the fake names as its framework pins: one installed in every test environment, one never.
PACKAGES = ("pytest", "lassi-no-such-dist")

# The synthetic fixture every layer test reads, and its records in file order.
FIXTURE = "sft-tiny.jsonl"
RECORDS: tuple[dict[str, Any], ...] = (
    {"prompt": "SYNTHETIC prompt one", "completion": "SYNTHETIC completion one"},
    {"prompt": "SYNTHETIC prompt two", "completion": "SYNTHETIC completion two"},
)
# The committed fixture under tests/fixtures/train/ that the command-line tests read (lassi.train.data SYNTHETIC_DIR).
COMMITTED_FIXTURE = "layer-smoke.jsonl"

# A complete train recipe on synthetic data with the fake trainer on the CPU.
TRAIN: dict[str, Any] = {
    "base_model": "SYNTHETIC/tiny-model",
    "method": "sft",
    "weights": "full",
    "trainer": {"kind": TRAINER, "device": {"kind": "cpu"}, "steps": 2},
    "data": {"synthetic": FIXTURE},
}

# A SYNTHETIC suite manifest: two train items, one eval item, and one unassigned item.
SUITE = "synthetic-train"
SUITE_TEXT = """\
suite: synthetic-train
repo: https://example.invalid/synthetic-train
commit: 0123456789abcdef0123456789abcdef01234567
items:
  alpha:
    split: train
    languages:
      omp: {dir: src/alpha-omp, files: [main.cpp]}
  beta:
    split: train
    languages:
      omp: {dir: src/beta-omp, files: [main.cpp]}
  held:
    split: eval
    languages:
      omp: {dir: src/held-omp, files: [main.cpp]}
  open:
    split: unassigned
    languages:
      omp: {dir: src/open-omp, files: [main.cpp]}
"""
SUITE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
TRAIN_ITEMS = ("alpha", "beta")

# Every train key the layer does not carry out yet, with a value the schema accepts (lassi.train.run NOT_CARRIED_OUT).
LATER_KEYS: dict[str, Any] = {
    "episode": "single_turn",
    "lora": {"r": 8, "targets": "all-linear"},
    "reward": {"profile": "df-v0", "executor": "ttsim", "cache": True},
    "rollout": {"engine": "vllm", "group_size": 8},
    "adversary": {"kind": "llm", "model": "SYNTHETIC-model", "trained": False},
    "export": {"merge": True, "fxb": "check_then_build", "register_as": "SYNTHETIC-registered"},
}


# ---------------------------------------------------------------------------
# Modules under test, imported per test so each test fails on its own


def module(name: str) -> ModuleType:
    """Import `name`, failing the test clearly while it does not exist (task P17.8)."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        pytest.fail(f"{name} does not exist yet (task P17.8): {error}")


def train_run() -> ModuleType:
    """Import lassi.train.run."""
    return module("lassi.train.run")


def train_data() -> ModuleType:
    """Import lassi.train.data."""
    return module("lassi.train.data")


def core_name(module_name: str, name: str) -> Any:
    """Return `name` from a lassi.core module, failing the test clearly while it does not exist (task P17.8)."""
    found = importlib.import_module(module_name)
    if not hasattr(found, name):
        pytest.fail(f"{module_name} has no {name} yet (task P17.8)")
    return getattr(found, name)


def framework_build(values: tuple[str, str, str | None, str | None]) -> Any:
    """Return a FrameworkBuild from (name, version, cuda, hip)."""
    name, version, cuda, hip = values
    return core_name("lassi.core.devices", "FrameworkBuild")(name=name, version=version, cuda=cuda, hip=hip)


# ---------------------------------------------------------------------------
# The fake trainer


@dataclass
class Log:
    """What the fakes saw, in order: framework() calls, builds, probes, steps, and what train() found on disk."""

    events: list[str] = field(default_factory=list)
    configs: list[dict[str, Any]] = field(default_factory=list)
    jobs: list[Any] = field(default_factory=list)
    seen: list[dict[str, Any]] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)


def _on_disk(out_dir: Path) -> dict[str, Any]:
    """Return what lies beside and in `out_dir` when a step is about to run."""
    parent = out_dir.parent
    resolved = parent / RESOLVED_RECIPE
    provenance = parent / PROVENANCE_JSON
    return {
        "parent_entries": sorted(item.name for item in parent.iterdir()) if parent.is_dir() else None,
        "out_dir_entries": sorted(item.name for item in out_dir.iterdir()) if out_dir.is_dir() else None,
        "resolved": resolved.read_bytes().decode("ascii") if resolved.is_file() else None,
        "provenance": json.loads(provenance.read_bytes().decode("ascii")) if provenance.is_file() else None,
    }


def fake_trainer(
    log: Log,
    *,
    name: str = TRAINER,
    framework: Any = CPU_BUILD,
    packages: Any = PACKAGES,
    methods: Any = frozenset({"sft"}),
    weight_modes: Any = frozenset({"full"}),
    data_sources: Any = frozenset({"synthetic", "bench"}),
    takes_device: bool = True,
) -> type:
    """Return a fake Trainer class; `framework` is a build tuple, a callable, or MISSING, and MISSING drops a key."""

    def report() -> Any:
        log.events.append("framework")
        if callable(framework):
            return framework()
        return framework_build(framework)

    def build(self: Any, **config: Any) -> None:
        log.events.append("build")
        log.configs.append(copy.deepcopy(config))
        if config.get("refuse"):
            raise ValueError(REFUSAL)
        self.config = config

    def train(self: Any, job: Any) -> Any:
        log.events.append("train")
        log.jobs.append(job)
        out_dir = Path(job.out_dir)
        log.seen.append(_on_disk(out_dir))
        _read_bench(log, job, self.config.get("forbidden"))
        if self.config.get("fail"):
            raise RuntimeError(FAILURE)
        checkpoint = self.config.get("checkpoint", "checkpoint-1")
        if self.config.get("create", True):
            (out_dir / checkpoint).mkdir(parents=True)
            (out_dir / checkpoint / "marker").write_bytes(b"SYNTHETIC checkpoint marker\n")
        result = core_name("lassi.core.interfaces", "TrainResult")
        return result(steps=self.config.get("steps", 1), checkpoints=(checkpoint,))

    namespace: dict[str, Any] = {
        "__doc__": f"Fake Trainer {name} for the lassi train tests.",
        "name": name,
        "capabilities": frozenset({TAKES_DEVICE} if takes_device else set()),
        "config_keys": frozenset({"steps", "refuse", "fail", "checkpoint", "create", "forbidden"}),
        "methods": methods,
        "weight_modes": weight_modes,
        "data_sources": data_sources,
        "packages": packages,
        "framework": MISSING if framework is MISSING else staticmethod(report),
        "__init__": build,
        "train": train,
    }
    return type(f"Fake_{name}", (), {key: value for key, value in namespace.items() if value is not MISSING})


def _read_bench(log: Log, job: Any, forbidden: str | None) -> None:
    """Read every bench item of the job through bench_item, then ask for `forbidden`, expecting EvalSplitError."""
    if job.data.source != "bench":
        return
    for item in job.data.items:
        log.events.append(f"bench_item {item}")
        job.data.bench_item(item)
    if forbidden is not None:
        try:
            job.data.bench_item(forbidden)
        except EvalSplitError as error:
            log.refusals.append(str(error))


def fake_registry(log: Log, *extra: tuple[str, type], **declared: Any) -> Registry:
    """Return a test Registry with the fake trainer (its declarations changed by `declared`) and `extra` trainers."""
    registry = Registry()
    registry.register("Trainer", TRAINER, fake_trainer(log, **declared))
    for name, cls in extra:
        registry.register("Trainer", name, cls)
    return registry


# ---------------------------------------------------------------------------
# Fake probes


@dataclass
class FakeProbe:
    """A probe of one kind with SYNTHETIC facts, or a refusal; it records each call in the Log."""

    kind: str
    log: Log
    count: int = 2
    name: str | None = None
    memory_bytes: int | None = None
    driver: str | None = None
    runtime: str | None = None
    refuse: str | None = None

    def probe(self) -> Any:
        """Record the call; raise DeviceUnavailable with `refuse`, or return the facts."""
        self.log.events.append(f"probe {self.kind}")
        if self.refuse is not None:
            raise core_name("lassi.core.devices", "DeviceUnavailable")(self.refuse)
        facts = core_name("lassi.core.devices", "HostFacts")
        return facts(
            count=self.count, name=self.name, memory_bytes=self.memory_bytes, driver=self.driver, runtime=self.runtime
        )


def fake_probes(log: Log, **changes: Any) -> dict[str, FakeProbe]:
    """Return SYNTHETIC cpu, rocm, and cuda probes; `changes` maps a kind to a replacement probe or None (absent)."""
    probes: dict[str, FakeProbe | None] = {
        "cpu": FakeProbe("cpu", log, count=8, name=CPU_NAME, memory_bytes=4096),
        "rocm": FakeProbe("rocm", log, count=2, driver=ROCM_DRIVER),
        "cuda": FakeProbe("cuda", log, count=2),
    }
    probes.update(changes)
    return {kind: probe for kind, probe in probes.items() if probe is not None}


# ---------------------------------------------------------------------------
# Files


def recipe_data(**changes: Any) -> dict[str, Any]:
    """Return a copy of TRAIN with top-level keys replaced by `changes`; MISSING removes a key."""
    data = copy.deepcopy(TRAIN)
    for key, value in changes.items():
        if value is MISSING:
            data.pop(key, None)
        else:
            data[key] = copy.deepcopy(value)
    return data


def on_bench(**bench: Any) -> dict[str, Any]:
    """Return TRAIN reading the SYNTHETIC suite's train split in place of data.synthetic; `bench` changes keys."""
    return recipe_data(data=MISSING, bench={"suite": SUITE, "split": "train", **bench})


def write_recipe(directory: Path, data: Mapping[str, Any] | str, name: str = "train") -> Path:
    """Write `data` (a mapping, or YAML text) as <directory>/<name>.yaml in ASCII with LF; return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    text = data if isinstance(data, str) else yaml.safe_dump(dict(data), sort_keys=False)
    path.write_bytes(text.encode("ascii"))
    return path


def jsonl(records: Iterable[Mapping[str, Any]]) -> bytes:
    """Return records as JSON Lines bytes: one compact object per line, ASCII, each line ending in LF."""
    return "".join(json.dumps(dict(record)) + "\n" for record in records).encode("ascii")


def write_fixture(directory: Path, content: bytes, name: str = FIXTURE) -> Path:
    """Write the fixture bytes as <directory>/<name>; return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(content)
    return path


def write_suite(directory: Path, text: str = SUITE_TEXT, name: str = SUITE) -> Path:
    """Write a suite manifest as <directory>/<name>.yaml; return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_bytes(text.encode("ascii"))
    return path


def canonical_json(identity: Mapping[str, Any]) -> str:
    """Return the canonical JSON text a split hash is taken over: sorted keys, compact, ASCII, no NaN."""
    return json.dumps(dict(identity), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def bible_train_block() -> str:
    """Return the text of the bible's projects/lassi-df/train.yaml block in Project Recipes."""
    lines = BIBLE.read_text(encoding="utf-8").splitlines()
    start = lines.index("## Project Recipes")
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("## "))
    cursor = start
    while True:
        begin = lines.index("```yaml", cursor, end) + 1
        stop = lines.index("```", begin)
        body = lines[begin:stop]
        if body[0].lstrip("#").strip() == "projects/lassi-df/train.yaml":
            return "\n".join(body) + "\n"
        cursor = stop + 1


# ---------------------------------------------------------------------------
# One layer run in a test directory


@dataclass
class Setup:
    """The test's directories: recipes, the synthetic fixtures, the suite manifests, and the runs root."""

    root: Path
    log: Log = field(default_factory=Log)

    @property
    def recipes(self) -> Path:
        """Return the directory recipes are written to and extends searches."""
        return self.root / "recipes"

    @property
    def fixtures(self) -> Path:
        """Return the synthetic fixture directory the options name."""
        return self.root / "fixtures"

    @property
    def suites(self) -> Path:
        """Return the suite manifest directory the options name."""
        return self.root / "suites"

    @property
    def runs_root(self) -> Path:
        """Return the runs root the options name; nothing creates it before the train directory."""
        return self.root / "runs-root"

    def train_dir(self, train_id: str = TRAIN_ID) -> Path:
        """Return where the layer puts the train directory of `train_id`."""
        return self.runs_root / "train" / train_id

    def options(self, registry: Registry | None = None, probes: Mapping[str, Any] | None = None, **changes: Any) -> Any:
        """Return TrainOptions naming this setup's directories, the fake registry, and the fake probes."""
        values: dict[str, Any] = {
            "runs_root": self.runs_root,
            "train_id": TRAIN_ID,
            "registry": fake_registry(self.log) if registry is None else registry,
            "roots": [self.recipes],
            "probes": fake_probes(self.log) if probes is None else probes,
            "suites_dir": self.suites,
            "synthetic_dir": self.fixtures,
        }
        values.update(changes)
        return train_run().TrainOptions(**values)

    def write(self, data: Mapping[str, Any] | str = TRAIN, content: bytes | None = None) -> Path:
        """Write the recipe, the fixture (RECORDS unless `content` is given), and the SYNTHETIC suite."""
        write_fixture(self.fixtures, jsonl(RECORDS) if content is None else content)
        write_suite(self.suites)
        return write_recipe(self.recipes, data)

    def run(self, data: Mapping[str, Any] | str = TRAIN, **changes: Any) -> Path:
        """Write the files and run run_training with this setup's options changed by `changes`."""
        path = self.write(data)
        return train_run().run_training(path, self.options(**changes))


def provenance(directory: Path) -> dict[str, Any]:
    """Return the provenance.json of a train directory."""
    return json.loads((directory / PROVENANCE_JSON).read_bytes().decode("ascii"))
