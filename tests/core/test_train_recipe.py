"""Tests for the train recipe loader (task P17.8): one schema beside the run recipe's, the same machinery.

Bible: Project Recipes (the projects/lassi-df/train.yaml block and the Notes:
device sections, train recipes), Training Module (Algorithms, Weight Modes,
Compute: a train recipe names its device with no default), Component
Interfaces (Trainer; recipes are validated against capabilities at load),
Agent Rules 5 and 10; plans/p17-portable.md, task P17.8 and the planning
decision "Trainer (P17.8)".

The contract these tests fix (lassi.core.recipe):

- load_train_recipe(path, *, roots=None, registry=None) -> Recipe reads each
  file of the extends chain strictly, refuses any key TRAIN_SCHEMA does not
  list (so a train recipe that extends a run recipe is refused by the run
  recipe's key), merges as load_recipe does (a child's device replaces the
  inherited one whole; a trainer section naming another kind replaces the
  inherited one), materializes no default, and hashes the canonical YAML
  exactly as load_recipe does. Run recipes never see the train keys: SCHEMA
  still refuses trainer and data.
- TRAIN_SCHEMA holds every key of the bible's train.yaml block, plus
  extends, runs_root (optional), trainer (a kind section that binds the
  Trainer at trainer.kind, its device and config keys beside kind), and
  data.synthetic (a JSON Lines fixture name). TRAIN_REQUIRED is base_model,
  method, weights, and trainer.kind; the loader picks none of them.
- METHODS, WEIGHT_MODES, and EPISODES are the bible block's names (its
  comments), and DATA_SOURCES is synthetic and bench.
- A train recipe names exactly one data source, bench or data.synthetic.
  bench.split must be train: eval and unassigned are refused at load,
  naming bench.split and Agent Rule 5.
- The Trainer must declare takes_device, and its device section is checked
  as every device section is (lassi.core.devices.parse_device), each
  refusal naming trainer.device. Its declared methods, weight_modes, and
  data_sources must be collections of strings, and the recipe's method,
  weights, and data source must be among them, each refusal naming the key.
  A config key the Trainer does not accept is refused at trainer.<key>.

Every component is a fake that fails when constructed or asked for its
framework, since loading builds nothing. Run recipes keep their resolved
recipes and hashes: tests/core/test_recipe.py (CHILD_HASH and the golden
resolved file) and tests/core/test_single_executor_golden.py pin them
unchanged. No value in this module is a measurement.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from lassi.core.recipe import RecipeError, load_recipe, resolved_yaml
from lassi.core.registry import Registry

REPO = Path(__file__).resolve().parents[2]
BIBLE = REPO / "docs" / "BIBLE.md"
BIBLE_TRAIN_BLOCK = "projects/lassi-df/train.yaml"
TAKES_DEVICE = "takes_device"
MISSING = object()

# A complete train recipe on synthetic data with the fake trainer.
TRAIN: dict[str, Any] = {
    "base_model": "SYNTHETIC/tiny-model",
    "method": "sft",
    "weights": "full",
    "trainer": {"kind": "fake", "device": {"kind": "cpu"}, "steps": 2},
    "data": {"synthetic": "sft-tiny.jsonl"},
}
# The bench section of a train recipe, in place of data.
BENCH_TRAIN = {"suite": "synthetic-train", "split": "train"}
# Every train key the layer does not carry out, with a value the schema accepts (lassi.train.run NOT_CARRIED_OUT;
# task P17.9), and the keys it passes to the Trainer's check(recipe, data); the loader accepts both.
LAYER_KEYS: dict[str, Any] = {
    "episode": "single_turn",
    "reward": {"profile": "df-v0", "executor": "ttsim", "cache": True},
    "adversary": {"kind": "llm", "model": "SYNTHETIC-model", "trained": False},
    "export": {"merge": True, "fxb": "check_then_build", "register_as": "SYNTHETIC-registered"},
}
TRAINER_KEYS: dict[str, Any] = {
    "lora": {"r": 8, "targets": "all-linear"},
    "rollout": {"engine": "vllm", "group_size": 8},
}


def recipe_module() -> ModuleType:
    """Import lassi.core.recipe."""
    return importlib.import_module("lassi.core.recipe")


def named(name: str) -> Any:
    """Return lassi.core.recipe.<name>, failing the test clearly while it does not exist (task P17.8)."""
    module = recipe_module()
    if not hasattr(module, name):
        pytest.fail(f"lassi.core.recipe has no {name} yet (task P17.8)")
    return getattr(module, name)


# ---------------------------------------------------------------------------
# Fake trainers, never built


class NeverBuilt:
    """Base for every fake: constructing one fails the test, since loading constructs nothing."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError(f"component {type(self).__name__} was constructed while loading a recipe")


def trainer_class(
    name: str,
    *,
    takes_device: bool = True,
    config_keys: Iterable[str] = ("steps",),
    methods: Any = frozenset({"sft", "dpo"}),
    weight_modes: Any = frozenset({"full"}),
    data_sources: Any = frozenset({"synthetic", "bench"}),
) -> type:
    """Return a fake Trainer class with the declarations the loader reads; MISSING leaves an attribute out."""

    def framework() -> Any:
        raise AssertionError(f"the loader asked trainer {name} for its framework build")

    namespace: dict[str, Any] = {
        "__doc__": f"Fake trainer {name} for the train recipe tests; never constructed.",
        "name": name,
        "capabilities": frozenset({TAKES_DEVICE} if takes_device else set()),
        "config_keys": frozenset(config_keys),
        "methods": methods,
        "weight_modes": weight_modes,
        "data_sources": data_sources,
        "packages": ("pytest",),
        "framework": staticmethod(framework),
    }
    namespace = {key: value for key, value in namespace.items() if value is not MISSING}
    return type(f"Fake_{name}", (NeverBuilt,), namespace)


def registry(*extra: tuple[str, type]) -> Registry:
    """Return a Registry with the fake trainers fake, other (no config keys), and nodev (no device), plus `extra`."""
    found = Registry()
    found.register("Trainer", "fake", trainer_class("fake"))
    found.register("Trainer", "other", trainer_class("other", config_keys=()))
    found.register("Trainer", "nodev", trainer_class("nodev", takes_device=False))
    for name, cls in extra:
        found.register("Trainer", name, cls)
    return found


# ---------------------------------------------------------------------------
# Writing and loading


def write(directory: Path, name: str, data: Mapping[str, Any] | str) -> Path:
    """Write `data` (a mapping, or YAML text) as <directory>/<name>.yaml in ASCII with LF; return its path."""
    path = directory / f"{name}.yaml"
    text = data if isinstance(data, str) else yaml.safe_dump(dict(data), sort_keys=False)
    path.write_bytes(text.encode("ascii"))
    return path


def train(**changes: Any) -> dict[str, Any]:
    """Return a copy of TRAIN with top-level keys replaced by `changes`; MISSING removes a key."""
    data = copy.deepcopy(TRAIN)
    for key, value in changes.items():
        if value is MISSING:
            data.pop(key, None)
        else:
            data[key] = copy.deepcopy(value)
    return data


def on_bench(**bench: Any) -> dict[str, Any]:
    """Return TRAIN reading the bench in place of data.synthetic; `bench` replaces BENCH_TRAIN's keys."""
    return train(data=MISSING, bench={**BENCH_TRAIN, **bench})


def load(
    tmp_path: Path, data: Mapping[str, Any] | str, name: str = "train", found: Registry | None = None
) -> Any:
    """Write and load `data` as a train recipe with tmp_path as the only root."""
    loader: Callable[..., Any] = named("load_train_recipe")
    return loader(write(tmp_path, name, data), roots=[tmp_path], registry=registry() if found is None else found)


def load_error(
    tmp_path: Path, data: Mapping[str, Any] | str, name: str = "train", found: Registry | None = None
) -> str:
    """Return the RecipeError message for `data`; the message must name the recipe file."""
    named("load_train_recipe")
    with pytest.raises(RecipeError) as info:
        load(tmp_path, data, name, found)
    message = str(info.value)
    assert f"{name}.yaml" in message, message
    return message


def canonical(data: Mapping[str, Any]) -> str:
    """Return the canonical YAML of a resolved mapping, as load_recipe writes it."""
    return yaml.safe_dump(dict(data), sort_keys=True, default_flow_style=False, allow_unicode=False, width=4096)


def sha256(text: str) -> str:
    """Return the sha256 hex digest of ASCII text."""
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def bindings(loaded: Any) -> list[tuple[str, str, str, dict[str, Any]]]:
    """Return (interface, name, where, config) for each binding of a loaded recipe."""
    return [(item.interface, item.name, item.where, dict(item.config)) for item in loaded.bindings]


# ---------------------------------------------------------------------------
# The bible's train.yaml block


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
        if body[0].lstrip("#").strip() == BIBLE_TRAIN_BLOCK:
            return "\n".join(body) + "\n"
        cursor = stop + 1


def bible_choices(key: str) -> tuple[str, ...]:
    """Return the 'a | b | c' names the bible block's comment lists for the top-level `key`."""
    for line in bible_train_block().splitlines():
        if line.startswith(f"{key}:") and "#" in line:
            return tuple(part.strip() for part in line.split("#", 1)[1].split("|"))
    raise AssertionError(f"the bible's train.yaml block lists no choices for {key}")


def test_the_bible_block_parser_finds_the_train_block() -> None:
    block = yaml.safe_load(bible_train_block())
    assert {"base_model", "method", "weights", "bench"} <= set(block), block


def test_the_loader_names_are_the_bible_blocks_names() -> None:
    assert tuple(named("METHODS")) == bible_choices("method")
    assert tuple(named("WEIGHT_MODES")) == bible_choices("weights")
    assert tuple(named("EPISODES")) == bible_choices("episode")
    assert tuple(named("DATA_SOURCES")) == ("synthetic", "bench")


def test_the_train_schema_holds_the_bible_block_keys_plus_its_own() -> None:
    # The bible block gains its trainer line with task P17.8 (Project Recipes, train.yaml).
    block = yaml.safe_load(bible_train_block())
    assert "trainer" in block, "the bible's train.yaml block names no trainer section yet (task P17.8)"
    schema = named("TRAIN_SCHEMA")
    assert set(schema.fields) == set(block) | {"extends", "runs_root", "trainer", "data"}


def test_the_required_train_choices() -> None:
    assert tuple(named("TRAIN_REQUIRED")) == ("base_model", "method", "weights", "trainer.kind")


def test_bible_train_block_loads_with_a_trainer_named_trl(tmp_path: Path) -> None:
    every = trainer_class(
        "trl",
        methods=frozenset(bible_choices("method")),
        weight_modes=frozenset(bible_choices("weights")),
        data_sources=frozenset({"synthetic", "bench"}),
    )
    loaded = load(tmp_path, bible_train_block(), name="train", found=registry(("trl", every)))
    block = yaml.safe_load(bible_train_block())
    assert loaded.data == block
    assert loaded.data["bench"]["split"] == "train"
    assert bindings(loaded) == [
        ("Trainer", "trl", "trainer.kind", {key: value for key, value in block["trainer"].items() if key != "kind"})
    ]


# ---------------------------------------------------------------------------
# Loading, resolving, and the hash


def test_train_recipe_loads_resolves_and_hashes(tmp_path: Path) -> None:
    loaded = load(tmp_path, TRAIN)
    assert loaded.data == TRAIN, "a train recipe materializes no default (no project, faithful, or fixes)"
    assert (loaded.name, loaded.chain) == ("train", ("train",))
    assert loaded.canonical_yaml == canonical(TRAIN)
    assert loaded.recipe_hash == sha256(canonical(TRAIN))
    assert bindings(loaded) == [("Trainer", "fake", "trainer.kind", {"device": {"kind": "cpu"}, "steps": 2})]
    text = resolved_yaml(loaded)
    assert text.startswith(f"# Resolved recipe train (chain: train)\n# recipe_hash: {loaded.recipe_hash} ")
    assert text.endswith(canonical(TRAIN))


def test_train_recipe_hash_is_stable_and_follows_the_data(tmp_path: Path) -> None:
    first = load(tmp_path, TRAIN, name="first")
    reordered = "\n".join(
        [
            "data: {synthetic: sft-tiny.jsonl}",
            "trainer:",
            "  steps: 2",
            "  device: {kind: cpu}",
            "  kind: fake",
            "weights: full",
            "method: sft",
            "base_model: SYNTHETIC/tiny-model",
            "",
        ]
    )
    again = load(tmp_path, reordered, name="again")
    assert again.recipe_hash == first.recipe_hash
    other_steps = load(tmp_path, train(trainer={**TRAIN["trainer"], "steps": 3}), name="steps")
    assert other_steps.recipe_hash != first.recipe_hash
    other_model = load(tmp_path, train(base_model="SYNTHETIC/other-model"), name="model")
    assert other_model.recipe_hash != first.recipe_hash


@pytest.mark.parametrize("extends", ["parent", "parent.yaml"], ids=["by-name", "by-path"])
def test_train_child_extends_a_train_parent_and_replaces_the_device_whole(extends: str, tmp_path: Path) -> None:
    rocm = {"kind": "rocm", "indices": [0, 1]}
    write(tmp_path, "parent", train(trainer={"kind": "fake", "device": rocm, "steps": 2}))
    child = {"extends": extends, "trainer": {"kind": "fake", "device": {"kind": "cpu"}}}
    loaded = load(tmp_path, child, name="child")
    assert loaded.data["trainer"] == {"kind": "fake", "device": {"kind": "cpu"}, "steps": 2}
    assert "extends" not in loaded.data
    assert loaded.chain == ("parent", "child")
    assert loaded.data == train()


def test_a_child_naming_another_trainer_kind_replaces_the_inherited_section(tmp_path: Path) -> None:
    write(tmp_path, "parent", TRAIN)
    child = {"extends": "parent", "trainer": {"kind": "other", "device": {"kind": "cpu"}}}
    loaded = load(tmp_path, child, name="child")
    assert loaded.data["trainer"] == {"kind": "other", "device": {"kind": "cpu"}}


def test_train_recipe_that_extends_a_run_recipe_is_refused_naming_the_key(tmp_path: Path) -> None:
    run_parent = {
        "llm": {"sampling": {"temperature": 0.2, "top_p": 0.9}},
        "loop": {"max_corrections": 3},
        "trials": {"n": 1},
    }
    write(tmp_path, "runparent", run_parent)
    named("load_train_recipe")
    with pytest.raises(RecipeError) as info:
        load(tmp_path, train(extends="runparent"), name="child")
    message = str(info.value)
    assert "runparent.yaml" in message and "unknown key llm" in message, message


@pytest.mark.parametrize("key", ["llm", "stages", "model", "executor", "directions", "project", "faithful"])
def test_run_recipe_keys_are_unknown_in_a_train_recipe(key: str, tmp_path: Path) -> None:
    values = {
        "llm": {"sampling": {"temperature": 0.2}},
        "stages": ["generate"],
        "model": {"backend": "mock", "id": "x"},
        "executor": {"kind": "none"},
        "directions": [{"source": "omp", "target": "cuda"}],
        "project": "SYNTHETIC-project",
        "faithful": True,
    }
    message = load_error(tmp_path, train(**{key: values[key]}))
    assert f"unknown key {key}" in message, message


@pytest.mark.parametrize("key", ["trainer", "data"])
def test_train_keys_are_unknown_in_a_run_recipe(key: str, tmp_path: Path) -> None:
    path = write(tmp_path, "runrecipe", {key: TRAIN[key]})
    with pytest.raises(RecipeError) as info:
        load_recipe(path, roots=[tmp_path], registry=Registry())
    assert f"unknown key {key}" in str(info.value), str(info.value)


@pytest.mark.parametrize(
    ("changes", "path"),
    [
        pytest.param({"epochs": 3}, "epochs", id="top-level"),
        pytest.param({"lora": {"rank": 8}}, "lora.rank", id="lora"),
        pytest.param({"data": {"synthetic": "sft-tiny.jsonl", "records": 3}}, "data.records", id="data"),
        pytest.param({"reward": {"weights": "x"}}, "reward.weights", id="reward"),
        pytest.param({"rollout": {"engines": "x"}}, "rollout.engines", id="rollout"),
        pytest.param({"export": {"format": "x"}}, "export.format", id="export"),
    ],
)
def test_unknown_train_key_is_refused_naming_it(changes: dict[str, Any], path: str, tmp_path: Path) -> None:
    message = load_error(tmp_path, train(**changes))
    assert f"unknown key {path}" in message, message


def test_bench_takes_its_run_recipe_keys_and_nothing_else(tmp_path: Path) -> None:
    message = load_error(tmp_path, on_bench(splits="train"))
    assert "unknown key bench.splits" in message, message


# ---------------------------------------------------------------------------
# Required choices and values


@pytest.mark.parametrize(
    ("changes", "path"),
    [
        pytest.param({"base_model": MISSING}, "base_model", id="base_model"),
        pytest.param({"method": MISSING}, "method", id="method"),
        pytest.param({"weights": MISSING}, "weights", id="weights"),
        pytest.param({"trainer": MISSING}, "trainer.kind", id="no-trainer"),
        pytest.param({"trainer": {"device": {"kind": "cpu"}}}, "trainer.kind", id="trainer-without-kind"),
        pytest.param({"method": None}, "method", id="null-method"),
        pytest.param({"data": {}}, "data.synthetic", id="empty-data"),
    ],
)
def test_each_required_train_choice_is_refused_when_missing(
    changes: dict[str, Any], path: str, tmp_path: Path
) -> None:
    message = load_error(tmp_path, train(**changes))
    assert f"{path} is a required choice" in message, message


@pytest.mark.parametrize(
    ("bench", "path"),
    [
        pytest.param({"suite": "synthetic-train"}, "bench.split", id="no-split"),
        pytest.param({"split": "train"}, "bench.suite", id="no-suite"),
    ],
)
def test_bench_needs_its_suite_and_split(bench: dict[str, Any], path: str, tmp_path: Path) -> None:
    message = load_error(tmp_path, train(data=MISSING, bench=bench))
    assert f"{path} is a required choice" in message, message


@pytest.mark.parametrize(
    ("changes", "key"),
    [
        pytest.param({"method": "sftt"}, "method", id="method"),
        pytest.param({"method": 3}, "method", id="method-not-a-string"),
        pytest.param({"weights": "half"}, "weights", id="weights"),
        pytest.param({"episode": "three_turn"}, "episode", id="episode"),
        pytest.param({"base_model": 7}, "base_model", id="base_model"),
        pytest.param({"runs_root": 5}, "runs_root", id="runs_root"),
        pytest.param({"lora": {"r": 0, "targets": "all-linear"}}, "lora.r", id="lora-r"),
        pytest.param({"rollout": {"engine": "vllm", "group_size": 0}}, "rollout.group_size", id="group-size"),
    ],
)
def test_values_outside_the_schema_are_refused_naming_the_key(
    changes: dict[str, Any], key: str, tmp_path: Path
) -> None:
    message = load_error(tmp_path, train(**changes))
    assert key in message, message


def test_every_bible_method_weight_mode_and_episode_loads(tmp_path: Path) -> None:
    every = trainer_class(
        "every", methods=frozenset(named("METHODS")), weight_modes=frozenset(named("WEIGHT_MODES"))
    )
    found = registry(("every", every))
    for method in named("METHODS"):
        for weights in named("WEIGHT_MODES"):
            data = train(method=method, weights=weights, trainer={"kind": "every", "device": {"kind": "cpu"}})
            assert load(tmp_path, data, name=f"{method}-{weights}", found=found).data["method"] == method
    for episode in named("EPISODES"):
        assert load(tmp_path, train(episode=episode), name=episode).data["episode"] == episode


def test_the_loader_accepts_every_later_key_and_leaves_refusal_to_lassi_train(tmp_path: Path) -> None:
    # lassi.train.run refuses LAYER_KEYS and the Trainer's check() what it does not carry out (task P17.9).
    later = {**LAYER_KEYS, **TRAINER_KEYS}
    loaded = load(tmp_path, train(**later))
    assert {key: loaded.data[key] for key in later} == later


def test_runs_root_is_an_optional_string(tmp_path: Path) -> None:
    root = str(tmp_path / "runs-root")
    assert load(tmp_path, train(runs_root=root)).data["runs_root"] == root
    assert "runs_root" not in load(tmp_path, TRAIN, name="without").data


# ---------------------------------------------------------------------------
# Data sources and splits (Agent Rule 5)


@pytest.mark.parametrize("split", ["eval", "unassigned"])
def test_eval_and_unassigned_bench_splits_are_refused_at_load(split: str, tmp_path: Path) -> None:
    message = load_error(tmp_path, on_bench(suite="lassi-hecbench-10", split=split))
    assert "bench.split" in message and split in message, message
    assert "Agent Rule 5" in message, message


@pytest.mark.parametrize("split", ["test", "Train", "dev"])
def test_a_split_other_than_train_is_refused(split: str, tmp_path: Path) -> None:
    message = load_error(tmp_path, on_bench(split=split))
    assert "bench.split" in message and "train" in message, message


def test_a_train_split_loads(tmp_path: Path) -> None:
    loaded = load(tmp_path, on_bench(items=["alpha"]))
    assert loaded.data["bench"] == {"suite": "synthetic-train", "split": "train", "items": ["alpha"]}
    assert "data" not in loaded.data


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(train(bench=BENCH_TRAIN), id="both"),
        pytest.param(train(data=MISSING), id="neither"),
    ],
)
def test_a_train_recipe_names_exactly_one_data_source(data: dict[str, Any], tmp_path: Path) -> None:
    message = load_error(tmp_path, data)
    assert "bench" in message and "data.synthetic" in message, message


@pytest.mark.parametrize(
    "name",
    [
        "../escape.jsonl", "sub/sft.jsonl", "sub\\sft.jsonl", "sft.json", ".hidden.jsonl", "sft.jsonl ", "",
        "C:sft.jsonl",
    ],
    ids=["parent", "subdir", "backslash", "json", "hidden", "trailing-blank", "empty", "drive"],
)
def test_data_synthetic_names_one_jsonl_file(name: str, tmp_path: Path) -> None:
    message = load_error(tmp_path, train(data={"synthetic": name}))
    assert "data.synthetic" in message, message


@pytest.mark.parametrize("name", ["sft-tiny.jsonl", "a.b_c-1.jsonl", "DPO2.jsonl"])
def test_a_plain_jsonl_name_loads(name: str, tmp_path: Path) -> None:
    assert load(tmp_path, train(data={"synthetic": name})).data["data"] == {"synthetic": name}


# ---------------------------------------------------------------------------
# The trainer: binding, device, and declarations


def test_an_unknown_trainer_is_refused_naming_trainer_kind(tmp_path: Path) -> None:
    message = load_error(tmp_path, train(trainer={"kind": "nosuch", "device": {"kind": "cpu"}}))
    assert "trainer.kind" in message and "nosuch" in message, message


def test_trainer_config_key_it_does_not_accept_is_refused_by_dotted_path(tmp_path: Path) -> None:
    message = load_error(tmp_path, train(trainer={**TRAIN["trainer"], "max_stepz": 4}))
    assert "trainer.max_stepz" in message, message


@pytest.mark.parametrize(
    ("device", "path"),
    [
        pytest.param(MISSING, "trainer.device", id="missing"),
        pytest.param(None, "trainer.device", id="null"),
        pytest.param("cpu", "trainer.device", id="not-a-mapping"),
        pytest.param({}, "trainer.device.kind", id="no-kind"),
        pytest.param({"kind": "gpu"}, "trainer.device.kind", id="unknown-kind"),
        pytest.param({"kind": "CPU"}, "trainer.device.kind", id="kind-case"),
        pytest.param({"kind": "cuda"}, "trainer.device.indices", id="no-indices"),
        pytest.param({"kind": "rocm", "indices": []}, "trainer.device.indices", id="empty-indices"),
        pytest.param({"kind": "rocm", "indices": [0, 0]}, "trainer.device.indices", id="repeated-index"),
        pytest.param({"kind": "cuda", "indices": [True]}, "trainer.device.indices", id="bool-index"),
        pytest.param({"kind": "cpu", "indices": [0]}, "trainer.device.indices", id="cpu-indices"),
        pytest.param({"kind": "cpu", "vendor": "amd"}, "trainer.device.vendor", id="unknown-key"),
    ],
)
def test_trainer_device_is_required_and_its_shape_is_checked(device: Any, path: str, tmp_path: Path) -> None:
    section = {"kind": "fake", "steps": 2}
    if device is not MISSING:
        section["device"] = device
    message = load_error(tmp_path, train(trainer=section))
    assert path in message, message


@pytest.mark.parametrize(
    "device",
    [{"kind": "cpu"}, {"kind": "cuda", "indices": [1]}, {"kind": "rocm", "indices": [3, 0, 2]}],
    ids=["cpu", "cuda", "rocm"],
)
def test_a_device_section_enters_the_resolved_recipe_as_written(device: dict[str, Any], tmp_path: Path) -> None:
    loaded = load(tmp_path, train(trainer={"kind": "fake", "device": device}))
    assert loaded.data["trainer"]["device"] == device
    assert loaded.recipe_hash == sha256(canonical(train(trainer={"kind": "fake", "device": device})))


def test_a_trainer_that_does_not_take_a_device_is_refused(tmp_path: Path) -> None:
    without = load_error(tmp_path, train(trainer={"kind": "nodev"}), name="without")
    assert "takes_device" in without and "nodev" in without, without
    with_device = load_error(tmp_path, train(trainer={"kind": "nodev", "device": {"kind": "cpu"}}), name="with")
    assert "trainer.device" in with_device, with_device


@pytest.mark.parametrize(
    ("changes", "key", "value"),
    [
        pytest.param({"method": "gspo"}, "method", "gspo", id="method"),
        pytest.param({"weights": "lora"}, "weights", "lora", id="weights"),
    ],
)
def test_method_or_weights_the_trainer_does_not_declare_is_refused_naming_the_key(
    changes: dict[str, Any], key: str, value: str, tmp_path: Path
) -> None:
    message = load_error(tmp_path, train(**changes))
    assert key in message and value in message and "fake" in message, message


def test_a_data_source_the_trainer_does_not_declare_is_refused(tmp_path: Path) -> None:
    synthetic_only = trainer_class("synthonly", data_sources=frozenset({"synthetic"}))
    found = registry(("synthonly", synthetic_only))
    data = on_bench()
    data["trainer"] = {"kind": "synthonly", "device": {"kind": "cpu"}}
    message = load_error(tmp_path, data, found=found)
    assert "bench" in message and "synthonly" in message, message
    assert load(tmp_path, train(trainer={"kind": "synthonly", "device": {"kind": "cpu"}}), "ok", found)


@pytest.mark.parametrize("attribute", ["methods", "weight_modes", "data_sources"])
@pytest.mark.parametrize(
    "declared",
    [MISSING, "sft", ["sft", 3], None],
    ids=["missing", "bare-string", "non-string-item", "none"],
)
def test_malformed_trainer_declarations_are_refused_naming_the_attribute(
    attribute: str, declared: Any, tmp_path: Path
) -> None:
    broken = trainer_class("broken", **{attribute: declared})
    found = registry(("broken", broken))
    message = load_error(tmp_path, train(trainer={"kind": "broken", "device": {"kind": "cpu"}}), found=found)
    assert attribute in message, message


def test_loading_builds_nothing_and_asks_no_framework(tmp_path: Path) -> None:
    # Every fake raises when built or asked for its framework build, so a load that did either fails here.
    assert load(tmp_path, TRAIN).bindings
    assert load(tmp_path, on_bench(), name="bench").bindings
