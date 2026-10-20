"""Tests for adversary.recording, the llm adversary's replay recording bound from a recipe (task P16.4, design 2).

Bible: Project Recipes (the agent and judge bindings block; Notes, server
settings), Component Interfaces (LLMBackend), Result Record (the adversary
record copies its binding's config), Agent Rules 10 and 12. Plan:
plans/p16-adversarial.md, P16.4 ("a replay backend takes its recording
under P16.13's rule, path under tests/fixtures/ and sha256 in the resolved
recipe") and P16.13.

The contract these tests fix:

- lassi.core.recipe.recording_file(value, where) -> Path is public: the
  rule model.recording follows (task P16.13), its messages opening with
  `where`, the prefix naming the key, and holding the same words ("is
  empty", "is an absolute path", "must use / separators", "empty, '.', or
  '..' segment", "is not under tests/fixtures/", "names no file ...
  including case", "outside tests/fixtures/"). A value that is not a str
  is refused naming the key ("must be a string"). It raises RecipeError, a
  ValueError, and returns the resolved file. It reads REPO at each call.
- At load, an adversary section that holds `recording` is checked by that
  rule (where "<recipe path>: adversary.recording"); the sha256 of the
  file's bytes enters the resolved recipe as adversary.recording_sha256,
  so the recipe hash changes with the bytes; a stated sha256 must match
  them, and one stated without adversary.recording is refused, each naming
  adversary.recording_sha256. The adversary binding's config holds the
  digest as recording_sha256 and the recording AS WRITTEN (the trial
  record copies the config, so no host path enters it). The model
  section keeps P16.13's behavior beside it: its binding gets the checked
  absolute file.
- An adversary section without `recording`, the fuzzer's, is unchanged:
  its resolved section is the section as written and its binding config
  the same keys. A kind whose Agent does not list `recording` (the fuzzer)
  is refused by the registry's config check, naming the key.

The seam the tests use: as tests/core/test_recipe_recording.py does, the
loader resolves the path against lassi.core.recipe.REPO, read at each load;
the tests that need recordings of their own point it at a stand-in
repository under tmp_path, the rest read the committed SYNTHETIC recordings
under tests/fixtures/adversary/recordings/. Loading constructs no component:
the adversary Agent is a fake that fails when constructed, beside the real
Fuzzer class. Every recording written here is SYNTHETIC, and no value in
this module is a measurement.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.agents.fuzzer import Fuzzer
from lassi.core import recipe as recipe_module
from lassi.core.recipe import RecipeError, load_recipe, resolved_yaml
from lassi.core.registry import Binding, Registry
from lassi.llm import ReplayBackend

REPO = Path(__file__).resolve().parents[2]
# Committed SYNTHETIC recordings, named relative to the repository root (tests/fixtures/adversary/recordings/README.md).
RECORDING = "tests/fixtures/adversary/recordings/in-contract.json"
OTHER_RECORDING = "tests/fixtures/adversary/recordings/malformed.json"
MODEL_RECORDING = "tests/fixtures/replay/recordings/first-try-success.json"
STAND_IN_RECORDING = "tests/fixtures/adversary/stand-in.json"
AGENT = "fake-llm"
SECTION: dict[str, Any] = {
    "kind": AGENT,
    "budget": {"inputs": 4, "tokens": 20000, "steps": 6},
    "backend": "replay",
    "model": "adversary-replay",
    "sampling": {"temperature": 0.2, "top_p": 0.9, "max_tokens": 2048},
    "prompts": "adversary-v0",
}
FUZZER_SECTION: dict[str, Any] = {"kind": "fuzzer", "budget": {"inputs": 4}, "seed": 3}
STANDALONE: dict[str, Any] = {
    "llm": {"sampling": {"temperature": 0.2, "top_p": 0.9}},
    "loop": {"max_corrections": 3},
    "trials": {"n": 1},
    "runs_root": "fixture-runs",
    "sandbox": {"network": False, "wall_s": 60, "mem_gb": 4},
    "report": {"trial_md": True, "parquet": True},
    "bench": {"suite": "fixture-suite", "split": "eval"},
    "directions": [{"source": "omp", "target": "cpp"}],
    "stages": ["generate"],
    "executor": {"kind": "plainexec"},
    "model": {"backend": "replay", "id": "replay-fixture"},
}


class NeverBuilt:
    """Base for every fake component: constructing one fails the test, since loading constructs nothing."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError(f"component {type(self).__name__} was constructed while loading a recipe")


def fake(name: str, caps: Iterable[str] = (), keys: Iterable[str] = ()) -> type:
    """Return a fake component class with the class attributes the registry reads."""
    namespace: dict[str, Any] = {
        "__doc__": f"Fake component {name} for the adversary recording tests; never constructed.",
        "name": name,
        "capabilities": frozenset(caps),
        "config_keys": frozenset(keys),
    }
    return type(f"Fake_{name}", (NeverBuilt,), namespace)


def registry() -> Registry:
    """Return a Registry of the real replay backend, the real Fuzzer class, a fake llm-like Agent, and fakes."""
    found = Registry()
    found.register("LLMBackend", "replay", ReplayBackend)
    found.register("Executor", "plainexec", fake("plainexec", {"runs_code"}))
    found.register("Stage", "generate", fake("generate"))
    keys = {"budget", "backend", "model", "sampling", "prompts", "recording", "recording_sha256"}
    found.register("Agent", AGENT, fake(AGENT, {"proposes_inputs", "reports_log"}, keys))
    found.register("Agent", "fuzzer", Fuzzer)
    return found


def recipe(adversary: Mapping[str, Any] | None = None, **model_keys: Any) -> dict[str, Any]:
    """Return a copy of STANDALONE with the adversary section `adversary` and the extra model keys."""
    data = copy.deepcopy(STANDALONE)
    data["model"].update(copy.deepcopy(model_keys))
    if adversary is not None:
        data["adversary"] = copy.deepcopy(dict(adversary))
    return data


def section(**keys: Any) -> dict[str, Any]:
    """Return SECTION with `keys` set."""
    return {**copy.deepcopy(SECTION), **keys}


def write(directory: Path, name: str, data: Mapping[str, Any]) -> Path:
    """Write `data` as the recipe <directory>/<name>.yaml (ASCII, LF) and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


def load(tmp_path: Path, data: Mapping[str, Any], name: str = "adversary-recording") -> Any:
    """Write and load `data` with the test registry, from tmp_path/recipes."""
    directory = tmp_path / "recipes"
    return load_recipe(write(directory, name, data), roots=[directory], registry=registry())


def load_error(tmp_path: Path, data: Mapping[str, Any], name: str = "adversary-recording") -> str:
    """Return the message of the RecipeError `data` raises at load; it must name the recipe file."""
    with pytest.raises(RecipeError) as info:
        load(tmp_path, data, name)
    message = str(info.value)
    assert f"{name}.yaml" in message, message
    return message


def names_key(message: str, key: str) -> bool:
    """Return True when `message` names `key` itself, not a longer key such as key_sha256."""
    return re.search(re.escape(key) + r"(?!\w)", message) is not None


def assert_path_refusal(message: str) -> None:
    """Fail unless `message` is adversary.recording's own path refusal, not an unknown key or a registry refusal."""
    assert names_key(message, "adversary.recording"), message
    assert "unknown key" not in message, message
    assert "does not accept" not in message, f"the Agent lists recording: {message}"


def adversary_binding(loaded: Any) -> Binding:
    """Return the loaded recipe's adversary binding (the Agent at adversary.kind)."""
    (found,) = [item for item in loaded.bindings if item.interface == "Agent" and item.where == "adversary.kind"]
    return found


def model_binding(loaded: Any) -> Binding:
    """Return the loaded recipe's LLMBackend binding."""
    (found,) = [item for item in loaded.bindings if item.interface == "LLMBackend"]
    return found


def sha256_of(path: Path) -> str:
    """Return the sha256 hex digest of the file at `path`."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_recording(text: str) -> bytes:
    """Return a SYNTHETIC replay recording of one completion, as ASCII JSON with LF."""
    data = {"synthetic": True, "completions": [{"text": text}]}
    return (json.dumps(data, indent=2) + "\n").encode("ascii")


def recording_file() -> Any:
    """Return lassi.core.recipe.recording_file, failing with the task's name when it is not public yet."""
    assert hasattr(recipe_module, "recording_file"), "lassi.core.recipe has no public recording_file (task P16.4)"
    return recipe_module.recording_file


@pytest.fixture(autouse=True)
def away_from_the_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Work from tmp_path, so a path read against the working directory, not the repository root, finds no file."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def stand_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a stand-in repository root under tmp_path, with tests/fixtures/adversary/, as lassi.core.recipe.REPO."""
    root = tmp_path / "stand-in-repo"
    (root / "tests" / "fixtures" / "adversary").mkdir(parents=True)
    monkeypatch.setattr(recipe_module, "REPO", root)
    return root


def make_dir_link(link: Path, target: Path) -> None:
    """Make `link` a directory link to `target`: a symlink, or on Windows a junction; skip when neither is allowed."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except OSError as error:
        if sys.platform != "win32":
            pytest.skip(f"cannot make a directory symlink here: {error}")
    import _winapi  # Windows only: a junction needs no privilege

    try:
        _winapi.CreateJunction(str(target), str(link))
    except OSError as error:
        pytest.skip(f"cannot make a junction here: {error}")


# ---------------------------------------------------------------------------
# recording_file, the public rule


def test_the_committed_recordings_exist() -> None:
    for relative in (RECORDING, OTHER_RECORDING, MODEL_RECORDING):
        assert (REPO / relative).is_file(), relative


def test_recording_file_returns_the_resolved_file() -> None:
    assert recording_file()(RECORDING, "adversary.recording") == (REPO / RECORDING).resolve()
    assert recording_file()(MODEL_RECORDING, "recipe.yaml: model.recording") == (REPO / MODEL_RECORDING).resolve()


WHERE = ["adversary.recording", "some/recipe.yaml: model.recording"]

REFUSALS = [
    pytest.param("", "is empty", id="empty"),
    pytest.param(str(REPO / RECORDING), "is an absolute path", id="host-absolute"),
    pytest.param("/" + RECORDING, "is an absolute path", id="leading-slash"),
    pytest.param("C:/" + RECORDING, "is an absolute path", id="drive"),
    pytest.param(RECORDING.replace("/", "\\"), "must use / separators", id="backslashes"),
    pytest.param("./" + RECORDING, "empty, '.', or '..' segment", id="leading-dot"),
    pytest.param("tests/fixtures/adversary/../adversary/recordings/in-contract.json", "'..' segment", id="dot-dot"),
    pytest.param("tests//fixtures/adversary/recordings/in-contract.json", "'..' segment", id="empty-segment"),
    pytest.param("pyproject.toml", "is not under tests/fixtures/", id="outside"),
    pytest.param("tests/fixtures/adversary/recordings/In-contract.json", "including case", id="wrong-case"),
    pytest.param("tests/fixtures/adversary/recordings/no-such.json", "names no file", id="missing"),
    pytest.param("tests/fixtures/adversary/recordings", "names no file", id="directory"),
    pytest.param(5, "must be a string", id="integer"),
    pytest.param(True, "must be a string", id="boolean"),
    pytest.param([RECORDING], "must be a string", id="list"),
    pytest.param(None, "must be a string", id="null"),
]


@pytest.mark.parametrize("where", WHERE)
@pytest.mark.parametrize(("value", "words"), REFUSALS)
def test_recording_file_refuses_with_the_rules_words_after_its_prefix(value: Any, words: str, where: str) -> None:
    with pytest.raises(RecipeError) as info:
        recording_file()(value, where)
    assert isinstance(info.value, ValueError)
    message = str(info.value)
    assert message.startswith(where), f"the message does not open with {where!r}: {message}"
    assert words in message, f"refused by another check than the one expected ({words!r}): {message}"


def test_recording_file_refuses_a_link_out_of_tests_fixtures(tmp_path: Path, stand_in: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "linked.json").write_bytes(synthetic_recording("SYNTHETIC reply outside"))
    make_dir_link(stand_in / "tests" / "fixtures" / "linked", outside)
    with pytest.raises(RecipeError) as info:
        recording_file()("tests/fixtures/linked/linked.json", "adversary.recording")
    assert str(info.value).startswith("adversary.recording") and "outside tests/fixtures/" in str(info.value)


def test_recording_file_reads_the_repository_root_at_each_call(stand_in: Path) -> None:
    (stand_in / STAND_IN_RECORDING).write_bytes(synthetic_recording("SYNTHETIC reply"))
    assert recording_file()(STAND_IN_RECORDING, "adversary.recording") == (stand_in / STAND_IN_RECORDING).resolve()
    with pytest.raises(RecipeError):
        recording_file()(RECORDING, "adversary.recording")


# ---------------------------------------------------------------------------
# The adversary section's recording at load


def test_the_sha256_enters_the_resolved_adversary_section(tmp_path: Path) -> None:
    loaded = load(tmp_path, recipe(section(recording=RECORDING)))
    adversary = loaded.data["adversary"]
    assert adversary["recording"] == RECORDING, "the path stays as written"
    assert adversary["recording_sha256"] == sha256_of(REPO / RECORDING)
    assert yaml.safe_load(loaded.canonical_yaml) == loaded.data


def test_the_binding_keeps_the_path_as_written_and_carries_the_sha256(tmp_path: Path) -> None:
    found = adversary_binding(load(tmp_path, recipe(section(recording=RECORDING))))
    assert found.name == AGENT
    assert found.config["recording"] == RECORDING, "the record copies this config, so the path stays as written"
    assert found.config["recording_sha256"] == sha256_of(REPO / RECORDING)
    assert set(found.config) == set(SECTION) - {"kind"} | {"recording", "recording_sha256"}


def test_no_host_path_enters_the_resolved_recipe_or_the_adversary_config(tmp_path: Path) -> None:
    loaded = load(tmp_path, recipe(section(recording=RECORDING)))
    config = json.dumps(dict(adversary_binding(loaded).config), sort_keys=True)
    for host in (str(REPO), REPO.as_posix(), str(tmp_path)):
        assert host not in loaded.canonical_yaml and host not in config, host


def test_the_resolved_recipe_reloads_to_the_same_hash(tmp_path: Path) -> None:
    loaded = load(tmp_path, recipe(section(recording=RECORDING)))
    saved = tmp_path / "saved" / "recipe.resolved.yaml"
    saved.parent.mkdir()
    saved.write_bytes(resolved_yaml(loaded).encode("ascii"))
    again = load_recipe(saved, roots=[tmp_path / "recipes"], registry=registry())
    assert again.recipe_hash == loaded.recipe_hash and again.data == loaded.data


def test_two_recordings_hash_apart(tmp_path: Path) -> None:
    first = load(tmp_path, recipe(section(recording=RECORDING)))
    second = load(tmp_path, recipe(section(recording=OTHER_RECORDING)))
    assert first.recipe_hash != second.recipe_hash
    assert second.data["adversary"]["recording_sha256"] == sha256_of(REPO / OTHER_RECORDING)


def test_the_recipe_hash_changes_with_the_recording_bytes(tmp_path: Path, stand_in: Path) -> None:
    path = stand_in / STAND_IN_RECORDING
    path.write_bytes(synthetic_recording("SYNTHETIC reply A"))
    first = load(tmp_path, recipe(section(recording=STAND_IN_RECORDING)))
    path.write_bytes(synthetic_recording("SYNTHETIC reply B"))
    second = load(tmp_path, recipe(section(recording=STAND_IN_RECORDING)))
    assert second.data["adversary"]["recording_sha256"] == sha256_of(path)
    assert second.recipe_hash != first.recipe_hash, "the same path with other bytes must hash apart"
    path.write_bytes(synthetic_recording("SYNTHETIC reply A"))
    assert load(tmp_path, recipe(section(recording=STAND_IN_RECORDING))).recipe_hash == first.recipe_hash


def test_a_child_recording_replaces_the_parent_one(tmp_path: Path) -> None:
    directory = tmp_path / "recipes"
    write(directory, "parent", recipe(section(recording=RECORDING)))
    child = write(directory, "child", {"extends": "parent", "adversary": {"kind": AGENT, "recording": OTHER_RECORDING}})
    loaded = load_recipe(child, roots=[directory], registry=registry())
    assert loaded.data["adversary"]["recording"] == OTHER_RECORDING
    assert loaded.data["adversary"]["recording_sha256"] == sha256_of(REPO / OTHER_RECORDING)
    assert sha256_of(REPO / RECORDING) not in loaded.canonical_yaml


# ---------------------------------------------------------------------------
# adversary.recording_sha256: entered by the loader, checked when a recipe states it


def test_a_stated_sha256_that_matches_loads_to_the_same_hash(tmp_path: Path) -> None:
    plain = load(tmp_path, recipe(section(recording=RECORDING)))
    stated = load(tmp_path, recipe(section(recording=RECORDING, recording_sha256=sha256_of(REPO / RECORDING))))
    assert stated.recipe_hash == plain.recipe_hash and stated.data == plain.data


@pytest.mark.parametrize("stated", [sha256_of(REPO / OTHER_RECORDING), "0" * 64, "ABC"])
def test_a_stated_sha256_that_differs_is_refused(tmp_path: Path, stated: str) -> None:
    message = load_error(tmp_path, recipe(section(recording=RECORDING, recording_sha256=stated)))
    assert "adversary.recording_sha256" in message, message


def test_a_resolved_recipe_refuses_a_recording_whose_bytes_changed(tmp_path: Path, stand_in: Path) -> None:
    path = stand_in / STAND_IN_RECORDING
    path.write_bytes(synthetic_recording("SYNTHETIC reply A"))
    saved = tmp_path / "saved" / "recipe.resolved.yaml"
    saved.parent.mkdir()
    saved.write_bytes(resolved_yaml(load(tmp_path, recipe(section(recording=STAND_IN_RECORDING)))).encode("ascii"))
    path.write_bytes(synthetic_recording("SYNTHETIC reply B"))
    with pytest.raises(RecipeError) as info:
        load_recipe(saved, roots=[tmp_path / "recipes"], registry=registry())
    assert "recipe.resolved.yaml" in str(info.value) and "adversary.recording_sha256" in str(info.value)


def test_a_sha256_without_a_recording_is_refused(tmp_path: Path) -> None:
    message = load_error(tmp_path, recipe(section(recording_sha256=sha256_of(REPO / RECORDING))))
    assert "adversary.recording_sha256" in message and "without adversary.recording" in message, message


# ---------------------------------------------------------------------------
# Refused at load, naming adversary.recording


LOAD_REFUSALS = [
    pytest.param(str(REPO / RECORDING), id="host-absolute"),
    pytest.param("/" + RECORDING, id="leading-slash"),
    pytest.param("tests/fixtures/adversary/../../../pyproject.toml", id="dot-dot-out"),
    pytest.param("./" + RECORDING, id="leading-dot"),
    pytest.param(RECORDING.replace("/", "\\"), id="backslashes"),
    pytest.param("pyproject.toml", id="outside"),
    pytest.param("tests/fixtures/adversary/recordings/no-such.json", id="missing"),
    pytest.param("tests/fixtures/adversary/recordings/In-contract.json", id="wrong-case"),
    pytest.param("", id="empty"),
    pytest.param(5, id="integer"),
    pytest.param(True, id="boolean"),
    pytest.param([RECORDING], id="list"),
]


@pytest.mark.parametrize("value", LOAD_REFUSALS)
def test_a_recording_outside_the_rule_is_refused_naming_adversary_recording(tmp_path: Path, value: Any) -> None:
    assert_path_refusal(load_error(tmp_path, recipe(section(recording=value))))


def test_a_recording_that_is_not_a_string_says_so(tmp_path: Path) -> None:
    message = load_error(tmp_path, recipe(section(recording=5)))
    assert names_key(message, "adversary.recording") and "must be a string" in message, message


# ---------------------------------------------------------------------------
# The fuzzer's section, and the model section beside the adversary's


@pytest.mark.parametrize("key", ["recording", "recording_sha256"])
def test_the_registry_refuses_a_recording_for_an_agent_that_does_not_list_it(tmp_path: Path, key: str) -> None:
    value = RECORDING if key == "recording" else sha256_of(REPO / RECORDING)
    message = load_error(tmp_path, recipe({**FUZZER_SECTION, key: value}))
    assert names_key(message, f"adversary.{key}"), message
    assert f"does not accept the key '{key}'" in message, message


def test_a_fuzzer_section_is_unchanged(tmp_path: Path) -> None:
    # The hash is the sha256 of the canonical YAML, which reads back as the resolved mapping; so a resolved fuzzer
    # section equal to the section as written, with no recording key anywhere, leaves the hash as it was.
    loaded = load(tmp_path, recipe(FUZZER_SECTION))
    assert loaded.data["adversary"] == FUZZER_SECTION
    assert "recording_sha256" not in loaded.canonical_yaml, "no sha256 is entered for a section with no recording"
    assert yaml.safe_load(loaded.canonical_yaml) == loaded.data
    assert loaded.recipe_hash == hashlib.sha256(loaded.canonical_yaml.encode("ascii")).hexdigest()
    found = adversary_binding(loaded)
    assert dict(found.config) == {"budget": {"inputs": 4}, "seed": 3}


def test_the_model_recording_keeps_its_rule_beside_the_adversarys(tmp_path: Path) -> None:
    loaded = load(tmp_path, recipe(section(recording=RECORDING), recording=MODEL_RECORDING))
    assert Path(model_binding(loaded).config["recording"]) == (REPO / MODEL_RECORDING).resolve(), (
        "the model binding carries the checked absolute file (task P16.13)"
    )
    assert adversary_binding(loaded).config["recording"] == RECORDING, "the adversary binding keeps the path as written"
    assert loaded.data["model"]["recording_sha256"] == sha256_of(REPO / MODEL_RECORDING)
    assert loaded.data["adversary"]["recording_sha256"] == sha256_of(REPO / RECORDING)
