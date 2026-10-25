"""Tests for the judges section of a run recipe: binding, required choices, rubrics, and families (task P15.1).

Bible: Project Recipes (the agent and judge bindings block, its judges
lines), Agents And Judges (the Judge interface; a judge uses a different
model family from the generator; Judge rules: a judge enters a reward only
after its calibration passes a recorded threshold), Component Interfaces
(the Judge row; capabilities checked at load), Model Serving (Serving
Rules, the reasoning effort; OQ-080), Agent Rules 3, 10, and 14. Plan:
plans/p15-judges.md, P15.1; OQ-091 (the reward admission threshold, open).

The contract these tests fix:

- judges.<kind> binds the registered Judge <kind> at judges.<kind> (config
  {}) and its LLMBackend at judges.<kind>.backend, whose config holds the
  section's backend settings (base_url, timeout_s, api_key_env, recording)
  checked against that backend's config_keys and named judges.<kind>.<key>
  when refused. model (the judge's model id), rubric, use, backend, and
  sampling (temperature, a finite number of at least 0; top_p, in (0, 1];
  max_tokens, an int of at least 1) are required choices with no default,
  each refused when missing or of the wrong form, naming the dotted key.
- use is screen or metric. reward is refused naming judges.<kind>.use and
  OQ-091 until task P15.11 records a threshold; any other value is
  refused listing screen and metric.
- mode is required when the Judge declares modes (a class attribute, a
  non-empty tuple or list of names) and must be one of them; it is refused
  for a Judge that declares none, and a malformed declaration is refused
  naming the Judge.
- The rubric loads through lassi.judges.rubric (RUBRIC_ROOT as the module
  holds it at the load), its refusals named judges.<kind>.rubric; its
  sha256 enters the resolved recipe as judges.<kind>.rubric_sha256, so the
  recipe hash follows the rubric's bytes: one changed byte changes it. A
  recipe may state rubric_sha256, which must then match.
- When a judge is bound, the family table (lassi.judges.families,
  FAMILIES_FILE at the load) must list each judge's model, the generator's
  model.id, and every arm; an unknown id is refused naming its key. A
  recipe without judges reads no table.
- A judge whose model's family takes a reasoning effort must set
  judges.<kind>.sampling.reasoning_effort (one of low, medium, high), and
  one whose family does not may not; a set value needs a backend that
  declares the capability reasoning_effort.
- judges.<kind>.recording is checked as model.recording is (a file under
  tests/fixtures/), its sha256 entered at judges.<kind>.recording_sha256,
  and the judge's backend binding carries the checked file; the model's
  own recording never reaches a judge's backend.
- A backend that declares takes_device is refused for a judge, naming
  judges.<kind>.backend: a judge's section takes no device section yet.

Every component here is a SYNTHETIC fake that is never built; the rubric
and family table are the SYNTHETIC fixtures under tests/fixtures/judges/.
"""

from __future__ import annotations

import copy
import hashlib
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.core import recipe as recipe_module
from lassi.core.recipe import RecipeError, load_recipe
from lassi.core.registry import Registry
from lassi.judges import families as families_module
from lassi.judges import rubric as rubric_module

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "judges"
RUBRICS = FIXTURES / "rubrics"
RUBRIC = "synthetic-v1"
FAMILIES = FIXTURES / "families.yaml"
# Two SYNTHETIC recordings: any file under tests/fixtures/ serves, as the loader reads only its bytes.
RECORDING = "tests/fixtures/judges/families.yaml"
OTHER_RECORDING = "tests/fixtures/judges/rubrics/synthetic-v1/prompt.txt"

GENERATOR = "SYNTHETIC-generator"
JUDGE_MODEL = "SYNTHETIC-judge"
REASONER = "SYNTHETIC-reasoner"
SAMPLING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64}

BASE: dict[str, Any] = {
    "llm": {"sampling": {"temperature": 0.2, "top_p": 0.9}},
    "loop": {"max_corrections": 1},
    "trials": {"n": 1},
    "runs_root": "fixture-runs",
    "sandbox": {"network": False, "wall_s": 60, "mem_gb": 4},
    "report": {"trial_md": True, "parquet": True},
    "bench": {"suite": "fixture-suite", "split": "eval"},
    "directions": [{"source": "omp", "target": "cuda"}],
    "stages": ["generate"],
    "executor": {"kind": "native"},
    "model": {"backend": "chatty", "id": GENERATOR},
}

CONSTRUCTED: list[str] = []


class NeverBuilt:
    """Base of every fake: constructing one records it and fails the test."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        CONSTRUCTED.append(type(self).__name__)
        raise AssertionError(f"{type(self).__name__} was constructed while loading a recipe")


def fake(name: str, capabilities: set[str], config_keys: tuple[str, ...] | set[str] = (), **attributes: Any) -> type:
    """Return a SYNTHETIC component class with the attributes the registry and the loader read."""
    namespace = {"__doc__": f"SYNTHETIC {name}; never built.", "name": name, "capabilities": frozenset(capabilities)}
    namespace.update(config_keys=frozenset(config_keys), **attributes)
    return type(f"Fake_{name.replace('-', '_')}", (NeverBuilt,), namespace)


@pytest.fixture
def registry() -> Registry:
    """Return a fresh registry of SYNTHETIC fakes: backends, judges, an executor, and a stage."""
    found = Registry()
    found.register("LLMBackend", "chatty", fake("chatty", {"chat"}, {"base_url", "timeout_s", "api_key_env"}))
    found.register("LLMBackend", "thinker", fake("thinker", {"chat", "reasoning_effort"}))
    found.register("LLMBackend", "replayer", fake("replayer", {"chat", "reasoning_effort"}, {"recording"}))
    found.register("LLMBackend", "inproc", fake("inproc", {"chat", "takes_device"}))
    found.register("Judge", "plain", fake("plain", {"verdict"}))
    found.register("Judge", "moded", fake("moded", {"verdict"}, modes=("explain", "predict")))
    found.register("Judge", "bare-modes", fake("bare-modes", {"verdict"}, modes="predict"))
    found.register("Judge", "empty-modes", fake("empty-modes", {"verdict"}, modes=()))
    found.register("Executor", "native", fake("native", {"runs_code"}))
    found.register("Stage", "generate", fake("generate", set(), requires={"LLMBackend": {"chat"}}))
    return found


@pytest.fixture(autouse=True)
def fixture_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the rubric root and the family table at the SYNTHETIC fixtures, and start with nothing built."""
    monkeypatch.setattr(rubric_module, "RUBRIC_ROOT", RUBRICS)
    monkeypatch.setattr(families_module, "FAMILIES_FILE", FAMILIES)
    CONSTRUCTED.clear()


def judge(**changes: Any) -> dict[str, Any]:
    """Return a SYNTHETIC judge section with `changes` applied; a None value removes the key."""
    section: dict[str, Any] = {
        "backend": "chatty",
        "model": JUDGE_MODEL,
        "sampling": dict(SAMPLING),
        "rubric": RUBRIC,
        "use": "metric",
    }
    section.update(changes)
    return {key: value for key, value in section.items() if value is not None}


def write(tmp_path: Path, judges: dict[str, Any] | None, **changes: Any) -> Path:
    """Write BASE with `judges` and the top-level `changes` (None removes a key) as recipe.yaml; return its path."""
    data = copy.deepcopy(BASE)
    if judges is not None:
        data["judges"] = judges
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="ascii")
    return path


def load(tmp_path: Path, registry: Registry, judges: dict[str, Any] | None, **changes: Any) -> Any:
    """Load the written recipe with the fake registry; nothing may be built."""
    found = load_recipe(write(tmp_path, judges, **changes), roots=[tmp_path], registry=registry)
    assert CONSTRUCTED == []
    return found


def refused(tmp_path: Path, registry: Registry, judges: dict[str, Any] | None, *keys: str, **changes: Any) -> str:
    """Expect the written recipe to be refused with a RecipeError naming every one of `keys`; return the message."""
    with pytest.raises(RecipeError) as caught:
        load(tmp_path, registry, judges, **changes)
    message = str(caught.value)
    assert CONSTRUCTED == []
    for key in keys:
        assert key in message, message
    return message


def tuples(found: Any) -> list[tuple[str, str, str, dict[str, Any]]]:
    """Return each binding as (interface, name, where, config)."""
    return [(item.interface, item.name, item.where, dict(item.config)) for item in found.bindings]


# ---------------------------------------------------------------------------
# Binding


def test_a_judge_binds_its_judge_and_its_backend_with_the_backend_settings(tmp_path: Path, registry: Registry) -> None:
    settings = {"base_url": "http://127.0.0.1:8123/v1", "timeout_s": 30, "api_key_env": "SYNTHETIC_KEY"}
    found = load(tmp_path, registry, {"plain": judge(**settings)})
    assert tuples(found)[-2:] == [
        ("Judge", "plain", "judges.plain", {}),
        ("LLMBackend", "chatty", "judges.plain.backend", settings),
    ]
    assert tuples(found)[0] == ("LLMBackend", "chatty", "model.backend", {}), "the model's backend binds first"
    section = found.data["judges"]["plain"]
    assert section["rubric_sha256"] == rubric_module.load_rubric(RUBRIC).sha256
    assert {key: value for key, value in section.items() if key != "rubric_sha256"} == judge(**settings)


def test_two_judges_bind_in_sorted_order_each_with_its_own_backend(tmp_path: Path, registry: Registry) -> None:
    judges = {"plain": judge(), "moded": judge(backend="thinker", mode="predict", use="screen")}
    found = load(tmp_path, registry, judges)
    assert [(item.interface, item.where) for item in found.bindings][-4:] == [
        ("Judge", "judges.moded"),
        ("LLMBackend", "judges.moded.backend"),
        ("Judge", "judges.plain"),
        ("LLMBackend", "judges.plain.backend"),
    ]


def test_an_unregistered_judge_or_backend_is_refused_naming_its_key(tmp_path: Path, registry: Registry) -> None:
    refused(tmp_path, registry, {"absent": judge()}, "judges.absent", "absent")
    refused(tmp_path, registry, {"plain": judge(backend="absent")}, "judges.plain.backend", "absent")


def test_a_backend_that_takes_a_device_section_is_refused_for_a_judge(tmp_path: Path, registry: Registry) -> None:
    refused(tmp_path, registry, {"plain": judge(backend="inproc")}, "judges.plain.backend", "inproc", "device")


@pytest.mark.parametrize(
    ("changes", "key"),
    [
        ({"recording": RECORDING}, "judges.plain.recording"),
        ({"temperature": 0.5}, "judges.plain.temperature"),
        ({"device": {"kind": "cpu"}}, "judges.plain.device"),
    ],
)
def test_a_setting_the_backend_does_not_take_is_refused_naming_it(
    tmp_path: Path, registry: Registry, changes: dict[str, Any], key: str
) -> None:
    refused(tmp_path, registry, {"plain": judge(**changes)}, key)


# ---------------------------------------------------------------------------
# Required choices


@pytest.mark.parametrize("key", ["model", "rubric", "use", "backend", "sampling"])
def test_each_required_key_is_a_required_choice(tmp_path: Path, registry: Registry, key: str) -> None:
    message = refused(tmp_path, registry, {"plain": judge(**{key: None})}, f"judges.plain.{key}")
    assert "required" in message, message


@pytest.mark.parametrize("key", ["temperature", "top_p", "max_tokens"])
def test_each_sampling_value_is_a_required_choice(tmp_path: Path, registry: Registry, key: str) -> None:
    sampling = {name: value for name, value in SAMPLING.items() if name != key}
    message = refused(tmp_path, registry, {"plain": judge(sampling=sampling)}, f"judges.plain.sampling.{key}")
    assert "required" in message, message


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("temperature", -0.1),
        ("temperature", "0.2"),
        ("temperature", True),
        ("top_p", 0),
        ("top_p", 1.5),
        ("top_p", False),
        ("max_tokens", 0),
        ("max_tokens", 1.5),
        ("max_tokens", True),
    ],
)
def test_a_bad_sampling_value_is_refused_naming_it(tmp_path: Path, registry: Registry, key: str, value: Any) -> None:
    refused(tmp_path, registry, {"plain": judge(sampling={**SAMPLING, key: value})}, f"judges.plain.sampling.{key}")


def test_an_unknown_sampling_key_is_refused(tmp_path: Path, registry: Registry) -> None:
    refused(tmp_path, registry, {"plain": judge(sampling={**SAMPLING, "seed": 1})}, "judges.plain.sampling.seed")


# ---------------------------------------------------------------------------
# use


@pytest.mark.parametrize("use", ["screen", "metric"])
def test_use_screen_and_metric_load(tmp_path: Path, registry: Registry, use: str) -> None:
    assert load(tmp_path, registry, {"plain": judge(use=use)}).data["judges"]["plain"]["use"] == use


def test_use_reward_is_refused_until_a_threshold_is_recorded(tmp_path: Path, registry: Registry) -> None:
    message = refused(tmp_path, registry, {"plain": judge(use="reward")}, "judges.plain.use", "OQ-091")
    assert "threshold" in message, message


@pytest.mark.parametrize("use", ["rank", "Screen", 3])
def test_another_use_is_refused_naming_the_key(tmp_path: Path, registry: Registry, use: Any) -> None:
    refused(tmp_path, registry, {"plain": judge(use=use)}, "judges.plain.use")


# ---------------------------------------------------------------------------
# mode


def test_a_judge_that_declares_modes_needs_one_of_them(tmp_path: Path, registry: Registry) -> None:
    message = refused(tmp_path, registry, {"moded": judge()}, "judges.moded.mode", "explain", "predict")
    assert "required" in message, message
    refused(tmp_path, registry, {"moded": judge(mode="train")}, "judges.moded.mode", "explain", "predict")
    found = load(tmp_path, registry, {"moded": judge(mode="predict")})
    assert found.data["judges"]["moded"]["mode"] == "predict"


@pytest.mark.parametrize("kind", ["plain", "empty-modes"])
def test_a_mode_is_refused_for_a_judge_that_declares_none(tmp_path: Path, registry: Registry, kind: str) -> None:
    refused(tmp_path, registry, {kind: judge(mode="predict")}, f"judges.{kind}.mode")
    load(tmp_path, registry, {kind: judge()})


def test_a_malformed_modes_declaration_is_refused_naming_the_judge(tmp_path: Path, registry: Registry) -> None:
    refused(tmp_path, registry, {"bare-modes": judge(mode="predict")}, "bare-modes", "modes")


# ---------------------------------------------------------------------------
# The rubric and its sha256


def test_an_unknown_or_unversioned_rubric_is_refused_naming_the_key(tmp_path: Path, registry: Registry) -> None:
    refused(tmp_path, registry, {"plain": judge(rubric="absent-v1")}, "judges.plain.rubric", "absent-v1")
    refused(tmp_path, registry, {"plain": judge(rubric="unversioned")}, "judges.plain.rubric", "unversioned")


def test_a_stated_rubric_sha256_must_match(tmp_path: Path, registry: Registry) -> None:
    digest = rubric_module.load_rubric(RUBRIC).sha256
    assert load(tmp_path, registry, {"plain": judge(rubric_sha256=digest)}).data["judges"]["plain"]["rubric_sha256"]
    message = refused(tmp_path, registry, {"plain": judge(rubric_sha256="0" * 64)}, "judges.plain.rubric_sha256")
    assert digest in message, message


def test_one_changed_rubric_byte_changes_the_recipe_hash(
    tmp_path: Path, registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "rubrics"
    shutil.copytree(RUBRICS / RUBRIC, root / RUBRIC)
    monkeypatch.setattr(rubric_module, "RUBRIC_ROOT", root)
    before = load(tmp_path, registry, {"plain": judge()})
    prompt = root / RUBRIC / "prompt.txt"
    data = bytearray(prompt.read_bytes())
    data[2] ^= 0x20  # one letter's case: one byte, still plain ASCII
    prompt.write_bytes(bytes(data))
    after = load(tmp_path, registry, {"plain": judge()})
    assert after.recipe_hash != before.recipe_hash
    assert after.recipe_hash == hashlib.sha256(after.canonical_yaml.encode("ascii")).hexdigest()
    sha = "rubric_sha256"
    assert after.data["judges"]["plain"][sha] == rubric_module.load_rubric(RUBRIC).sha256
    assert after.data["judges"]["plain"][sha] != before.data["judges"]["plain"][sha]
    strip = {key: value for key, value in after.data["judges"]["plain"].items() if key != sha}
    assert strip == {key: value for key, value in before.data["judges"]["plain"].items() if key != sha}


def test_the_rubric_sha256_is_in_the_resolved_yaml(tmp_path: Path, registry: Registry) -> None:
    found = load(tmp_path, registry, {"plain": judge()})
    reloaded = yaml.safe_load(recipe_module.resolved_yaml(found))
    assert reloaded["judges"]["plain"]["rubric_sha256"] == rubric_module.load_rubric(RUBRIC).sha256


# ---------------------------------------------------------------------------
# Model families


def test_an_unknown_judge_model_is_refused_naming_the_key_and_the_table(tmp_path: Path, registry: Registry) -> None:
    refused(tmp_path, registry, {"plain": judge(model="SYNTHETIC-unknown")}, "judges.plain.model", "families.yaml")


def test_an_unknown_generator_model_is_refused_when_a_judge_is_bound(tmp_path: Path, registry: Registry) -> None:
    model = {"backend": "chatty", "id": "SYNTHETIC-unknown"}
    refused(tmp_path, registry, {"plain": judge()}, "model.id", "SYNTHETIC-unknown", model=model)
    refused(tmp_path, registry, {"plain": judge()}, "arms[1]", "SYNTHETIC-arm", arms=[GENERATOR, "SYNTHETIC-arm"])
    load(tmp_path, registry, {"plain": judge()}, arms=[GENERATOR, REASONER])


def test_a_recipe_without_judges_reads_no_family_table(
    tmp_path: Path, registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(families_module, "FAMILIES_FILE", tmp_path / "absent.yaml")
    load(tmp_path, registry, None, model={"backend": "chatty", "id": "SYNTHETIC-unknown"})
    refused(tmp_path, registry, {"plain": judge()}, "absent.yaml")


def test_the_family_table_is_read_at_each_load(
    tmp_path: Path, registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = tmp_path / "families.yaml"
    table.write_bytes(b"families: [solo]\nreasoning_effort: []\nmodels: {SYNTHETIC-generator: solo}\n")
    monkeypatch.setattr(families_module, "FAMILIES_FILE", table)
    refused(tmp_path, registry, {"plain": judge()}, "judges.plain.model", JUDGE_MODEL)


# ---------------------------------------------------------------------------
# The reasoning effort


def test_a_judge_on_a_reasoning_family_must_set_the_effort(tmp_path: Path, registry: Registry) -> None:
    section = judge(model=REASONER, backend="thinker")
    message = refused(tmp_path, registry, {"plain": section}, "judges.plain.sampling.reasoning_effort")
    assert "synthetic-reasoning" in message and "required" in message, message
    sampling = {**SAMPLING, "reasoning_effort": "medium"}
    found = load(tmp_path, registry, {"plain": judge(model=REASONER, backend="thinker", sampling=sampling)})
    assert found.data["judges"]["plain"]["sampling"]["reasoning_effort"] == "medium"


def test_the_effort_is_refused_for_another_family(tmp_path: Path, registry: Registry) -> None:
    section = judge(backend="thinker", sampling={**SAMPLING, "reasoning_effort": "low"})
    refused(tmp_path, registry, {"plain": section}, "judges.plain.sampling.reasoning_effort", "synthetic-b")


def test_the_effort_needs_a_backend_that_declares_it(tmp_path: Path, registry: Registry) -> None:
    section = judge(model=REASONER, sampling={**SAMPLING, "reasoning_effort": "low"})
    key = "judges.plain.sampling.reasoning_effort"
    refused(tmp_path, registry, {"plain": section}, key, "chatty", "capability")


@pytest.mark.parametrize("value", ["LOW", "max", 1, True])
def test_a_bad_effort_is_refused_naming_it(tmp_path: Path, registry: Registry, value: Any) -> None:
    section = judge(model=REASONER, backend="thinker", sampling={**SAMPLING, "reasoning_effort": value})
    refused(tmp_path, registry, {"plain": section}, "judges.plain.sampling.reasoning_effort")


# ---------------------------------------------------------------------------
# A judge's replay recording


def test_a_judge_recording_is_checked_hashed_and_kept_apart_from_the_models(tmp_path: Path, registry: Registry) -> None:
    model = {"backend": "replayer", "id": GENERATOR, "recording": OTHER_RECORDING}
    section = judge(backend="replayer", recording=RECORDING)
    found = load(tmp_path, registry, {"plain": section}, model=model)
    digest = hashlib.sha256((REPO / RECORDING).read_bytes()).hexdigest()
    assert found.data["judges"]["plain"]["recording_sha256"] == digest
    assert found.data["judges"]["plain"]["recording"] == RECORDING, "the path is kept as written"
    configs = {item.where: dict(item.config) for item in found.bindings if item.interface == "LLMBackend"}
    assert Path(configs["judges.plain.backend"]["recording"]) == (REPO / RECORDING).resolve()
    assert Path(configs["model.backend"]["recording"]) == (REPO / OTHER_RECORDING).resolve()
    assert "recording_sha256" not in configs["judges.plain.backend"]


def test_a_judge_recording_outside_the_fixtures_or_with_a_stale_sha256_is_refused(
    tmp_path: Path, registry: Registry
) -> None:
    outside = judge(backend="replayer", recording="docs/BIBLE.md")
    refused(tmp_path, registry, {"plain": outside}, "judges.plain.recording")
    stale = judge(backend="replayer", recording=RECORDING, recording_sha256="0" * 64)
    refused(tmp_path, registry, {"plain": stale}, "judges.plain.recording_sha256")
