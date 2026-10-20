"""Tests for model.recording, the replay backend's recording bound from a recipe (task P16.13).

Bible: Project Recipes (the model section; Notes, server settings and
in-process model settings), Component Interfaces (LLMBackend), Agent Rules 10
and 12; plans/p16-adversarial.md, task P16.13.

The contract these tests fix:

- The model section accepts the key `recording` for a backend that lists it in
  its config_keys; ReplayBackend lists exactly {"recording"}. The registry's
  config check refuses it for any other backend (the mock and openai_compat
  here), naming model.recording.
- The value is a path relative to the repository root (never the working
  directory; every test here works from tmp_path), written with "/"
  separators, that must name a file under tests/fixtures/. Anything else is
  refused at load (load_recipe raises RecipeError naming the recipe file and
  model.recording): an absolute path (a drive, drive-relative included, or a
  leading "/"), a path with a "..", ".", or empty segment (even one that
  comes back inside tests/fixtures/, so a path has no redundant segment), a
  backslash, a relative path outside tests/fixtures/, a path that differs
  from the disk in case alone, a link under tests/fixtures/ that leads out
  of it, a missing file or a directory, and a value that is not a string. The refusal
  is the path rule's, never the schema's "unknown key" or the registry's
  "does not accept".
- The recording's path, as written, and the sha256 of its bytes enter the
  resolved recipe's model section, so the recipe hash changes when the
  recording's bytes change at the same path, and comes back when they do. The
  resolved recipe reloads to the same hash (Design Principle 5).
- The model section merges key by key, so a child's recording replaces the
  parent's, sha256 included.
- The loader enters the sha256 as model.recording_sha256, which never reaches
  the backend's config. A recipe may state it (a resolved recipe does), and a
  stated one must match the bytes, so a rerun from a resolved recipe whose
  recording changed is refused; one stated without model.recording, or of
  the wrong shape, is refused too. A backslash in the path is refused, as
  extends refuses one. The binding carries the checked file, resolved.

The seam the tests use: the loader resolves model.recording against
lassi.core.recipe.REPO, the repository root, read at each load (as
lassi.core.runner.REPO and lassi.train.data.REPO name theirs). The tests that
need recordings of their own point it at a stand-in repository under tmp_path;
the rest read the committed SYNTHETIC recordings under
tests/fixtures/replay/recordings/ in this repository. Loading constructs no
component: the backends are the real classes registered in a test Registry,
and every other component is a fake that fails when constructed. Every
recording written here is SYNTHETIC, and no value in this module is a
measurement.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.core import recipe as recipe_module
from lassi.core.recipe import RecipeError, load_recipe, resolved_yaml
from lassi.core.registry import Binding, Registry
from lassi.llm import MockBackend, OpenAICompatBackend, ReplayBackend

REPO = Path(__file__).resolve().parents[2]
# Committed SYNTHETIC recordings (tests/fixtures/replay/README.md), named relative to the repository root.
FIXTURE = "tests/fixtures/replay/recordings/first-try-success.json"
OTHER_FIXTURE = "tests/fixtures/replay/recordings/fence-cpp.json"
MODEL_ID = "replay-fixture"
# The stand-in repository's recording, relative to its root.
STAND_IN_RECORDING = "tests/fixtures/replay/stand-in.json"

STANDALONE: dict[str, Any] = {
    "llm": {"sampling": {"temperature": 0.2, "top_p": 0.9}},
    "loop": {"max_corrections": 3},
    "trials": {"n": 1},
    "runs_root": "fixture-runs",
    "sandbox": {"network": False, "wall_s": 60, "mem_gb": 4},
    "report": {"trial_md": True, "parquet": True},
    "bench": {"suite": "fixture-suite", "split": "eval"},
    "directions": [{"source": "omp", "target": "cuda"}],
    "stages": ["generate"],
    "executor": {"kind": "plainexec"},
    "model": {"backend": "replay", "id": MODEL_ID},
}


class NeverBuilt:
    """Base for every fake component: constructing one fails the test, since loading constructs nothing."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError(f"component {type(self).__name__} was constructed while loading a recipe")


def fake(name: str, caps: Iterable[str] = ()) -> type:
    """Return a fake component class with the class attributes the registry reads."""
    namespace: dict[str, Any] = {
        "__doc__": f"Fake component {name} for the recording tests; never constructed.",
        "name": name,
        "capabilities": frozenset(caps),
        "config_keys": frozenset(),
    }
    return type(f"Fake_{name}", (NeverBuilt,), namespace)


def registry() -> Registry:
    """Return a Registry of the real replay, mock, and openai_compat backends and fake others."""
    found = Registry()
    found.register("LLMBackend", "replay", ReplayBackend)
    found.register("LLMBackend", "mock", MockBackend)
    found.register("LLMBackend", "openai_compat", OpenAICompatBackend)
    found.register("Executor", "plainexec", fake("plainexec", {"runs_code"}))
    found.register("Stage", "generate", fake("generate"))
    return found


def write(directory: Path, name: str, data: Mapping[str, Any]) -> Path:
    """Write `data` as the recipe <directory>/<name>.yaml (ASCII, LF) and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


def recipe(backend: str = "replay", **model_keys: Any) -> dict[str, Any]:
    """Return a copy of STANDALONE whose model section names `backend` with the extra `model_keys`."""
    data = copy.deepcopy(STANDALONE)
    data["model"] = {"backend": backend, "id": MODEL_ID, **copy.deepcopy(model_keys)}
    return data


def load(tmp_path: Path, data: Mapping[str, Any], name: str = "recording") -> Any:
    """Write and load `data` with the test registry, from tmp_path/recipes."""
    directory = tmp_path / "recipes"
    return load_recipe(write(directory, name, data), roots=[directory], registry=registry())


def load_error(tmp_path: Path, data: Mapping[str, Any], name: str = "recording") -> str:
    """Return the message of the RecipeError `data` raises at load; it must name the recipe file."""
    with pytest.raises(RecipeError) as info:
        load(tmp_path, data, name)
    message = str(info.value)
    assert f"{name}.yaml" in message, message
    return message


def assert_path_refusal(message: str) -> None:
    """Fail unless `message` is model.recording's own path refusal, not an unknown key or a backend's refusal."""
    assert "model.recording" in message, message
    assert "unknown key" not in message, f"model.recording must be a known key: {message}"
    assert "does not accept" not in message, f"the replay backend must accept recording: {message}"


def backend_binding(loaded: Any) -> Binding:
    """Return the loaded recipe's one LLMBackend binding."""
    (found,) = [item for item in loaded.bindings if item.interface == "LLMBackend"]
    return found


def scalars(value: Any) -> Iterator[Any]:
    """Yield every scalar inside nested mappings and lists, mapping values only."""
    if isinstance(value, dict):
        for item in value.values():
            yield from scalars(item)
    elif isinstance(value, list):
        for item in value:
            yield from scalars(item)
    else:
        yield value


def sha256_of(path: Path) -> str:
    """Return the sha256 hex digest of the file at `path`."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_holds_recording(model: Mapping[str, Any], relative: str, digest: str) -> None:
    """Fail unless the model section holds the recording's path as written and its sha256."""
    values = list(scalars(dict(model)))
    assert relative in values, f"the path as written is not in the model section: {model}"
    assert any(isinstance(value, str) and digest in value for value in values), (
        f"the recording's sha256 {digest} is not in the model section: {model}"
    )


def synthetic_recording(text: str) -> bytes:
    """Return a SYNTHETIC replay recording of one completion, as ASCII JSON with LF."""
    data = {"synthetic": True, "completions": [{"text": text}]}
    return (json.dumps(data, indent=2) + "\n").encode("ascii")


@pytest.fixture(autouse=True)
def away_from_the_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Work from tmp_path, so a path read against the working directory, not the repository root, finds no file."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def stand_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a stand-in repository root under tmp_path, with tests/fixtures/replay/, as lassi.core.recipe.REPO.

    raising=False: before task P16.13 the loader has no REPO, and these tests
    must then fail on the missing key, not on the seam.
    """
    root = tmp_path / "stand-in-repo"
    (root / "tests" / "fixtures" / "replay").mkdir(parents=True)
    monkeypatch.setattr(recipe_module, "REPO", root, raising=False)
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
# Accepted: a file under tests/fixtures/, its path and sha256 in the resolved recipe


def test_the_committed_fixtures_exist() -> None:
    for relative in (FIXTURE, OTHER_FIXTURE):
        assert (REPO / relative).is_file(), relative


def test_replay_accepts_a_recording_under_tests_fixtures(tmp_path: Path) -> None:
    loaded = load(tmp_path, recipe(recording=FIXTURE))
    found = backend_binding(loaded)
    assert (found.name, found.where) == ("replay", "model.backend")
    assert "recording" in found.config, "the runner builds the backend as factory(model.id, recording=...)"


def test_the_path_and_sha256_enter_the_resolved_recipe(tmp_path: Path) -> None:
    loaded = load(tmp_path, recipe(recording=FIXTURE))
    assert loaded.data["model"]["backend"] == "replay" and loaded.data["model"]["id"] == MODEL_ID
    assert_holds_recording(loaded.data["model"], FIXTURE, sha256_of(REPO / FIXTURE))
    assert yaml.safe_load(loaded.canonical_yaml) == loaded.data, "the canonical YAML holds them"


def test_the_resolved_recipe_reloads_to_the_same_hash(tmp_path: Path) -> None:
    loaded = load(tmp_path, recipe(recording=FIXTURE))
    saved = tmp_path / "saved" / "recipe.resolved.yaml"
    saved.parent.mkdir()
    saved.write_bytes(resolved_yaml(loaded).encode("ascii"))
    again = load_recipe(saved, roots=[tmp_path / "recipes"], registry=registry())
    assert again.recipe_hash == loaded.recipe_hash
    assert again.data == loaded.data


def test_two_recordings_hash_apart(tmp_path: Path) -> None:
    first = load(tmp_path, recipe(recording=FIXTURE))
    second = load(tmp_path, recipe(recording=OTHER_FIXTURE))
    assert first.recipe_hash != second.recipe_hash
    assert_holds_recording(second.data["model"], OTHER_FIXTURE, sha256_of(REPO / OTHER_FIXTURE))


def test_the_recipe_hash_changes_with_the_recording_bytes(tmp_path: Path, stand_in: Path) -> None:
    path = stand_in / STAND_IN_RECORDING
    path.write_bytes(synthetic_recording("SYNTHETIC reply A"))
    first = load(tmp_path, recipe(recording=STAND_IN_RECORDING))
    assert_holds_recording(first.data["model"], STAND_IN_RECORDING, sha256_of(path))
    path.write_bytes(synthetic_recording("SYNTHETIC reply B"))
    second = load(tmp_path, recipe(recording=STAND_IN_RECORDING))
    assert_holds_recording(second.data["model"], STAND_IN_RECORDING, sha256_of(path))
    assert second.recipe_hash != first.recipe_hash, "the same path with other bytes must hash apart"
    path.write_bytes(synthetic_recording("SYNTHETIC reply A"))
    third = load(tmp_path, recipe(recording=STAND_IN_RECORDING))
    assert third.recipe_hash == first.recipe_hash, "the same bytes at the same path hash the same"


def test_a_child_recording_replaces_the_parent_one(tmp_path: Path) -> None:
    directory = tmp_path / "recipes"
    write(directory, "parent", recipe(recording=FIXTURE))
    child = write(directory, "child", {"extends": "parent", "model": {"recording": OTHER_FIXTURE}})
    loaded = load_recipe(child, roots=[directory], registry=registry())
    assert_holds_recording(loaded.data["model"], OTHER_FIXTURE, sha256_of(REPO / OTHER_FIXTURE))
    assert FIXTURE not in list(scalars(loaded.data["model"]))
    assert sha256_of(REPO / FIXTURE) not in loaded.canonical_yaml


# ---------------------------------------------------------------------------
# Backends that do not declare recording


@pytest.mark.parametrize("backend", ["mock", "openai_compat"])
def test_a_backend_without_the_key_refuses_a_recording_naming_it(tmp_path: Path, backend: str) -> None:
    message = load_error(tmp_path, recipe(backend, recording=FIXTURE))
    assert "model.recording" in message and f"LLMBackend '{backend}'" in message, message
    assert "does not accept the key 'recording'" in message, message


# ---------------------------------------------------------------------------
# Refused at load, naming model.recording


ABSOLUTE = [
    pytest.param(str(REPO / FIXTURE), id="host-absolute"),
    pytest.param((REPO / FIXTURE).as_posix(), id="posix-form-absolute"),
    pytest.param("/" + FIXTURE, id="leading-slash"),
    pytest.param("C:/" + FIXTURE, id="drive"),
    pytest.param("C:" + FIXTURE, id="drive-relative"),
]

# Each but the trailing slash names the committed fixture on Windows and Linux alike, so only the redundant segment
# refuses it; the trailing slash names its directory, and the segment check refuses it before the file check runs.
REDUNDANT = [
    pytest.param("./" + FIXTURE, id="leading-dot"),
    pytest.param("tests/./fixtures/replay/recordings/first-try-success.json", id="inner-dot"),
    pytest.param("tests//fixtures/replay/recordings/first-try-success.json", id="empty-segment"),
    pytest.param("tests/fixtures/replay/recordings/", id="trailing-slash"),
]

# Each differs from a committed path in case alone, which Windows would match and Linux would not.
WRONG_CASE = [
    pytest.param("tests/fixtures/Replay/recordings/first-try-success.json", id="directory"),
    pytest.param("tests/fixtures/replay/recordings/First-try-success.json", id="file-name"),
    pytest.param("tests/fixtures/replay/recordings/first-try-success.JSON", id="extension"),
]

DOT_DOT = [
    pytest.param("tests/fixtures/../../pyproject.toml", id="out-of-the-repository-files"),
    pytest.param(f"../{REPO.name}/{FIXTURE}", id="out-and-back-in"),
    pytest.param("tests/fixtures/replay/../replay/recordings/first-try-success.json", id="inside-tests-fixtures"),
    pytest.param("tests/fixtures/..", id="bare"),
]

OUTSIDE = [
    pytest.param("pyproject.toml", id="repository-root"),
    pytest.param("assets/bench/lassi-hecbench-10.yaml", id="assets"),
    pytest.param("tests/llm/test_replay.py", id="tests-not-fixtures"),
]

MISSING = [
    pytest.param("tests/fixtures/replay/recordings/no-such-recording.json", id="missing-file"),
    pytest.param("tests/fixtures/replay/recordings", id="directory"),
    pytest.param("tests/fixtures", id="the-fixture-root"),
    pytest.param("", id="empty"),
]

NOT_A_STRING = [
    pytest.param(5, id="integer"),
    pytest.param(True, id="boolean"),
    pytest.param([FIXTURE], id="list"),
    pytest.param(None, id="null"),
]


@pytest.mark.parametrize("value", ABSOLUTE)
def test_an_absolute_path_is_refused(tmp_path: Path, value: str) -> None:
    message = load_error(tmp_path, recipe(recording=value))
    assert_path_refusal(message)
    assert "is an absolute path" in message, f"refused by another check than the absolute one: {message}"


@pytest.mark.parametrize("value", DOT_DOT)
def test_a_dot_dot_path_is_refused(tmp_path: Path, value: str) -> None:
    assert_path_refusal(load_error(tmp_path, recipe(recording=value)))


@pytest.mark.parametrize("value", REDUNDANT)
def test_a_dot_or_empty_segment_is_refused(tmp_path: Path, value: str) -> None:
    message = load_error(tmp_path, recipe(recording=value))
    assert_path_refusal(message)
    assert "empty, '.', or '..' segment" in message, f"refused by another check than the segment one: {message}"


@pytest.mark.parametrize("value", WRONG_CASE)
def test_a_path_that_differs_in_case_is_refused(tmp_path: Path, value: str) -> None:
    assert value.lower() == FIXTURE.lower() != value
    message = load_error(tmp_path, recipe(recording=value))
    assert_path_refusal(message)
    assert "including case" in message, f"refused by another check than the as-spelled one: {message}"


@pytest.mark.parametrize("value", OUTSIDE)
def test_a_path_outside_tests_fixtures_is_refused(tmp_path: Path, value: str) -> None:
    assert (REPO / value).is_file(), "each case names a real file, so only its place refuses it"
    assert_path_refusal(load_error(tmp_path, recipe(recording=value)))


@pytest.mark.parametrize("value", MISSING)
def test_a_missing_file_is_refused(tmp_path: Path, value: str) -> None:
    assert_path_refusal(load_error(tmp_path, recipe(recording=value)))


@pytest.mark.parametrize("value", NOT_A_STRING)
def test_a_value_that_is_not_a_string_is_refused(tmp_path: Path, value: Any) -> None:
    assert_path_refusal(load_error(tmp_path, recipe(recording=value)))


def test_a_directory_link_out_of_tests_fixtures_is_refused(tmp_path: Path, stand_in: Path) -> None:
    inside = stand_in / STAND_IN_RECORDING
    inside.write_bytes(synthetic_recording("SYNTHETIC reply inside"))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "linked.json").write_bytes(synthetic_recording("SYNTHETIC reply outside"))
    make_dir_link(stand_in / "tests" / "fixtures" / "linked", outside)
    load(tmp_path, recipe(recording=STAND_IN_RECORDING), "control")  # the stand-in itself loads
    message = load_error(tmp_path, recipe(recording="tests/fixtures/linked/linked.json"))
    assert_path_refusal(message)
    assert "outside tests/fixtures/" in message, f"refused by another check than the link one: {message}"


def test_a_file_link_out_of_tests_fixtures_is_refused(tmp_path: Path, stand_in: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_bytes(synthetic_recording("SYNTHETIC reply outside"))
    link = stand_in / "tests" / "fixtures" / "replay" / "linked.json"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"this host cannot make a file symlink here: {error}")
    message = load_error(tmp_path, recipe(recording="tests/fixtures/replay/linked.json"))
    assert_path_refusal(message)
    assert "outside tests/fixtures/" in message, f"refused by another check than the link one: {message}"


def test_the_stand_in_root_is_the_only_root(tmp_path: Path, stand_in: Path) -> None:
    # With the stand-in as the repository root, the committed fixture of the real repository is a missing file.
    assert_path_refusal(load_error(tmp_path, recipe(recording=FIXTURE)))


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(FIXTURE.replace("/", "\\"), id="all-backslashes"),
        pytest.param("tests/fixtures/replay\\recordings/first-try-success.json", id="one-backslash"),
    ],
)
def test_a_backslash_path_is_refused(tmp_path: Path, value: str) -> None:
    # A backslash would name the file on Windows only, so the path is refused on every OS, as extends refuses one.
    assert_path_refusal(load_error(tmp_path, recipe(recording=value)))


# ---------------------------------------------------------------------------
# model.recording_sha256: entered by the loader, checked when a recipe states it


def test_a_stated_sha256_that_matches_loads_to_the_same_hash(tmp_path: Path) -> None:
    plain = load(tmp_path, recipe(recording=FIXTURE), "plain")
    stated = load(tmp_path, recipe(recording=FIXTURE, recording_sha256=sha256_of(REPO / FIXTURE)), "plain")
    assert stated.recipe_hash == plain.recipe_hash and stated.data == plain.data


def test_a_stated_sha256_that_differs_is_refused(tmp_path: Path) -> None:
    message = load_error(tmp_path, recipe(recording=FIXTURE, recording_sha256=sha256_of(REPO / OTHER_FIXTURE)))
    assert "model.recording_sha256" in message and sha256_of(REPO / FIXTURE) in message, message


def test_a_resolved_recipe_refuses_a_recording_whose_bytes_changed(tmp_path: Path, stand_in: Path) -> None:
    # A rerun from recipe.resolved.yaml reads the bytes the run read, or nothing (Design Principle 5).
    path = stand_in / STAND_IN_RECORDING
    path.write_bytes(synthetic_recording("SYNTHETIC reply A"))
    saved = tmp_path / "saved" / "recipe.resolved.yaml"
    saved.parent.mkdir()
    saved.write_bytes(resolved_yaml(load(tmp_path, recipe(recording=STAND_IN_RECORDING))).encode("ascii"))
    path.write_bytes(synthetic_recording("SYNTHETIC reply B"))
    with pytest.raises(RecipeError) as info:
        load_recipe(saved, roots=[tmp_path / "recipes"], registry=registry())
    assert "recipe.resolved.yaml" in str(info.value) and "model.recording_sha256" in str(info.value)


def test_a_sha256_without_a_recording_is_refused(tmp_path: Path) -> None:
    message = load_error(tmp_path, recipe(recording_sha256=sha256_of(REPO / FIXTURE)))
    assert "model.recording_sha256" in message and "without model.recording" in message, message


@pytest.mark.parametrize("value", ["ABC", "0" * 63, "0" * 64 + "0", "G" * 64])
def test_a_sha256_of_the_wrong_shape_is_refused(tmp_path: Path, value: str) -> None:
    message = load_error(tmp_path, recipe(recording=FIXTURE, recording_sha256=value))
    assert "model.recording_sha256 must be a sha256" in message, message


def test_the_sha256_never_reaches_the_backend(tmp_path: Path) -> None:
    # recording_sha256 is the loader's record, not a setting: ReplayBackend accepts only recording.
    found = backend_binding(load(tmp_path, recipe(recording=FIXTURE)))
    assert set(found.config) == {"recording"}
    assert Path(found.config["recording"]) == (REPO / FIXTURE).resolve(), "the binding carries the checked file"
