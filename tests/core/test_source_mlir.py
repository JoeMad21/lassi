"""Tests for task P17.41's source_mlir: the item's compute unit lowered to MLIR in the generation prompt.

Bible: IR Strategy (the cpu-mlir row: cgeist raises C to affine, scf, and
memref), Language Frontends (C through cgeist), Project Recipes (Notes: the
opt-in sections), Agent Rules 4 and 10. The owner (2026-10-26): lower the
C++ code to affine, linalg, or similar dialects, then have the model write
ttkernel MLIR that tt-mlir emits for the device. plans/spikes/mlir-issues.md,
Half 2: cgeist aborted on the items' whole sources, so each item has a
compute-only C unit (its manifest's `compute`), and the pinned Polygeist has
no affine-to-linalg raising, so the level is affine.

The contract these tests fix:

1. source_mlir is an optional recipe section (toolchain and max_chars
   required), never materialized, refused under faithful: true. The runner
   refuses, before any directory exists, a toolchain that does not declare
   lowers_source, a fragment prompt set, a template set lacking
   source_mlir.txt or whose generate.txt never shows $source_mlir, and an
   item without a compute unit; it builds the toolchain once per run and
   records it in toolchains.json.
2. With it, generate runs the toolchain on the item's compute unit, by its
   file, text, and function, in a fresh directory under the trial's, and
   fills $source_mlir with source_mlir.txt rendered with the unit's path,
   the function, the level, and the module; its Request gets one
   source-mlir note naming the unit, the size, the sha256, and the dialects.
3. A lowering that fails, or a module past max_chars (never cut), leaves
   $source_mlir empty and the trial goes on; the Request gets a source-mlir
   warning and the toolchain's diagnostics.
4. Without the section the field is empty, as before.

The suite, unit, modules, and errors are SYNTHETIC; the fake toolchain runs
nothing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_template_context import (
    CPP,
    ITEM,
    MODEL_ID,
    SUITE,
    TT,
    Script,
    doubled,
    load,
    make_assets,
    make_registry,
    program,
    recipe_data,
    runs_root,
    template_set,
    user_prompts,
    write_recipe,
    write_suite,
)

from lassi.bench import registry as bench_registry
from lassi.core import runner as runner_module
from lassi.core.capabilities import LOWERS_SOURCE
from lassi.core.interfaces import BuildResult, Lowering
from lassi.core.recipe import RecipeError
from lassi.core.record import Diagnostic, Trial, make_trial_id
from lassi.core.registry import Registry
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.stages import SOURCE_MLIR_CODE, SOURCE_MLIR_TEMPLATES
from lassi.core.store import TextStore, read_trial, trial_dir

SECTION = "source_mlir"
SET = "syn-mlir"
UNIT_PATH = "assets/compute/scale.c"
UNIT_TEXT = "/* SYNTHETIC compute unit */\nvoid lassi_scale(const float *x, float *y, long n) {}\n"
FUNCTION = "lassi_scale"
MODULE = "module {\n  func.func @lassi_scale() {\n    affine.for %i = 0 to 4 {\n    }\n    return\n  }\n}\n"
GENERATE = (
    "SYNTHETIC: translate the $source_language program to $target_language as $target_files.\n"
    "$source_files\n$source_mlir\nEnd of the SYNTHETIC generation prompt.\n"
)
SOURCE_TEMPLATE = "SYNTHETIC MLIR of $path ($function, $level):\n$mlir"


@dataclass
class Seen:
    """What the fake lowering saw and what it answers: a module or None, and diagnostics."""

    module: str | None = MODULE
    diagnostics: list[Diagnostic] = field(default_factory=list)
    calls: list[tuple[str, str, str, Path]] = field(default_factory=list)


SEEN = Seen()


class FakeLowering:
    """A SYNTHETIC lowering toolchain: records its call and answers SEEN's module; runs nothing."""

    name = "fake-lower"
    capabilities = frozenset({LOWERS_SOURCE})
    LEVEL = "affine"

    def lower(self, file: str, text: str, function: str, workdir: Path) -> Lowering:
        """Record the call and return SEEN's answer."""
        SEEN.calls.append((file, text, function, Path(workdir)))
        return Lowering(module=SEEN.module, level=self.LEVEL, diagnostics=list(SEEN.diagnostics))

    def build(self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None) -> BuildResult:
        """Build nothing."""
        return BuildResult(artifact=None, diagnostics=[])


class NotLowering(FakeLowering):
    """A toolchain that does not declare lowers_source."""

    name = "fake-not-lower"
    capabilities = frozenset({"diagnostics"})


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate variables, point TMPDIR at a test directory, and reset the fake's answer."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "scratch" / "tmp").mkdir(parents=True)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch" / "tmp"))
    SEEN.module, SEEN.diagnostics, SEEN.calls = MODULE, [], []


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """Return the directory every run of a test writes under."""
    return tmp_path / "scratch"


@pytest.fixture
def assets(base: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return the temporary assets root: tt-host-v0 and SET (it with GENERATE and SOURCE_TEMPLATE)."""
    root = make_assets(base, monkeypatch)
    template_set(root, SET, texts={"generate": GENERATE, "source_mlir": SOURCE_TEMPLATE})
    return root


def add_compute(base: Path, monkeypatch: pytest.MonkeyPatch, unit: bool = True) -> None:
    """Give the SYNTHETIC suite's item a compute unit under a temporary tracked root (or none with `unit` False)."""
    manifest = base / "manifests" / f"{SUITE}.yaml"
    data = yaml.safe_load(manifest.read_bytes())
    if unit:
        data["items"][ITEM]["compute"] = {"tracked": UNIT_PATH, "function": FUNCTION,
                                          "sha256": hashlib.sha256(UNIT_TEXT.encode("ascii")).hexdigest()}
    manifest.write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    (base / UNIT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (base / UNIT_PATH).write_bytes(UNIT_TEXT.encode("ascii"))
    monkeypatch.setattr(bench_registry, "TRACKED_ROOT", base)


@pytest.fixture
def bench(base: Path, monkeypatch: pytest.MonkeyPatch, assets: Path) -> Path:
    """Return the bench root of the SYNTHETIC suite, its item with a compute unit."""
    root = write_suite(base, monkeypatch)
    add_compute(base, monkeypatch)
    return root


def registry(script: Script) -> Registry:
    """Return test_template_context's registry plus the fake lowering toolchains."""
    found = make_registry(script)
    found.register("Toolchain", FakeLowering.name, FakeLowering)
    found.register("Toolchain", NotLowering.name, NotLowering)
    return found


def mlir_recipe(**changes: Any) -> dict[str, Any]:
    """Return test_template_context's cpp -> tt recipe on SET with a source_mlir section, and `changes`."""
    data = recipe_data(prompts=SET, loop={}, source_mlir={"toolchain": FakeLowering.name, "max_chars": 4000})
    data.update(changes)
    return data


def run(base: Path, bench: Path, script: Script, data: Mapping[str, Any], name: str) -> tuple[Path, Trial]:
    """Run recipe `data` as <base>/<name>.yaml on the test registry; return the run directory and trial 1."""
    options = RunOptions(runs_root=runs_root(base), run_id=name, bench_root=bench, registry=registry(script))
    run_dir = run_recipe(write_recipe(base, data, name), options)
    trial_id = make_trial_id(name, MODEL_ID, SUITE, f"{CPP}-{TT}", ITEM, 1)
    return run_dir, read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir))


def refused(base: Path, bench: Path, data: Mapping[str, Any], name: str = "refused") -> str:
    """Run `data`, expect a RunError before any directory exists, and return its message."""
    options = RunOptions(runs_root=runs_root(base), run_id=name, bench_root=bench,
                         registry=registry(Script(replies=[], attempts=[])))
    with pytest.raises(RunError) as caught:
        run_recipe(write_recipe(base, data, name), options)
    assert not (runs_root(base) / "runs" / name).exists()
    return str(caught.value)


def notes(trial: Trial) -> list[Diagnostic]:
    """Return the generation request's diagnostics."""
    return list((trial.requests or [])[0].diagnostics)


# ---------------------------------------------------------------------------
# The recipe section and the runner's checks (contract point 1)


def test_absent_the_section_adds_nothing(base: Path) -> None:
    recipe = load(base, recipe_data())
    assert SECTION not in recipe.canonical_yaml


@pytest.mark.parametrize(
    "section",
    [{"toolchain": "fake-lower"}, {"max_chars": 4000}, {"toolchain": "fake-lower", "max_chars": 10},
     {"toolchain": "fake-lower", "max_chars": 4000, "level": "linalg"}],
    ids=["no-max", "no-toolchain", "small", "unknown-key"],
)
def test_a_bad_section_is_refused(base: Path, section: Mapping[str, Any]) -> None:
    with pytest.raises(RecipeError, match=SECTION):
        load(base, mlir_recipe(source_mlir=dict(section)))


def test_the_section_is_refused_under_faithful(base: Path) -> None:
    with pytest.raises(RecipeError) as caught:
        load(base, mlir_recipe(faithful=True))
    assert SECTION in str(caught.value) and "faithful: true" in str(caught.value)


def test_the_runner_refuses_a_toolchain_that_does_not_lower_source(base: Path, bench: Path) -> None:
    message = refused(base, bench, mlir_recipe(source_mlir={"toolchain": NotLowering.name, "max_chars": 4000}))
    assert NotLowering.name in message and LOWERS_SOURCE in message, message
    message = refused(base, bench, mlir_recipe(source_mlir={"toolchain": "no-such-toolchain", "max_chars": 4000}))
    assert "no-such-toolchain" in message, message


def test_the_runner_refuses_a_set_that_never_shows_the_module(base: Path, bench: Path, assets: Path) -> None:
    template_set(assets, "syn-hidden", texts={"generate": GENERATE.replace("$source_mlir\n", ""),
                                              "source_mlir": SOURCE_TEMPLATE})
    message = refused(base, bench, mlir_recipe(prompts="syn-hidden"))
    assert "$source_mlir" in message and "generate.txt" in message, message
    template_set(assets, "syn-no-template", texts={"generate": GENERATE}, drop=["source_mlir"])
    message = refused(base, bench, mlir_recipe(prompts="syn-no-template"), "refused-2")
    assert "source_mlir.txt" in message, message


def test_the_runner_refuses_an_item_without_a_compute_unit(
    base: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = base / "manifests" / f"{SUITE}.yaml"
    data = yaml.safe_load(manifest.read_bytes())
    data["items"][ITEM].pop("compute")
    manifest.write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    message = refused(base, bench, mlir_recipe())
    assert ITEM in message and "compute" in message, message


def test_the_runner_refuses_the_section_with_a_fragment_set(base: Path, bench: Path, assets: Path) -> None:
    from test_correct_on_sim_gap import FRAGMENT_SET, FRAGMENTS
    from test_template_context import write_tree

    write_tree(assets / "prompts" / FRAGMENT_SET, FRAGMENTS)
    data = mlir_recipe(prompts=FRAGMENT_SET, stages=["generate", "compile_loop"])
    data.pop("oracle")
    message = refused(base, bench, data)
    assert SECTION in message and "fragment" in message, message


# ---------------------------------------------------------------------------
# The prompt (contract points 2 to 4)


def test_generate_shows_the_lowered_unit_and_notes_it(base: Path, bench: Path) -> None:
    script = Script(replies=[program(0)], attempts=[doubled])
    run_dir, trial = run(base, bench, script, mlir_recipe(), "on")
    assert len(SEEN.calls) == 1
    file, text, function, workdir = SEEN.calls[0]
    assert (file, text, function) == ("scale.c", UNIT_TEXT, FUNCTION)
    assert run_dir in workdir.parents and workdir.name == "build" and workdir.parent.name == "frontend"
    prompt = user_prompts(script)[0]
    assert f"SYNTHETIC MLIR of {UNIT_PATH} ({FUNCTION}, affine):\n{MODULE}" in prompt
    noted = [item for item in notes(trial) if item.code == SOURCE_MLIR_CODE]
    assert len(noted) == 1 and noted[0].severity == "note" and noted[0].stage == "parse"
    digest = hashlib.sha256(MODULE.encode("utf-8")).hexdigest()
    for words in (UNIT_PATH, FUNCTION, FakeLowering.name, str(len(MODULE)), digest, "affine (1)", "func (1)"):
        assert words in noted[0].message, (words, noted[0].message)
    record = json.loads((run_dir / "toolchains.json").read_text(encoding="utf-8"))
    assert record[FakeLowering.name]["languages"] == []


@pytest.mark.parametrize("label", ["failed", "too-long"])
def test_a_lowering_that_fails_or_is_too_long_leaves_the_field_empty(base: Path, bench: Path, label: str) -> None:
    error = Diagnostic(stage="lower", severity="error", code="exit-status", file="scale.c",
                       message="SYNTHETIC cgeist exited with status 134")
    if label == "failed":
        SEEN.module, SEEN.diagnostics = None, [error]
    else:
        SEEN.module = MODULE + "// " + "x" * 5000 + "\n"
    script = Script(replies=[program(0)], attempts=[doubled])
    _, trial = run(base, bench, script, mlir_recipe(), label)
    prompt = user_prompts(script)[0]
    assert "SYNTHETIC MLIR of" not in prompt and "End of the SYNTHETIC generation prompt" in prompt
    warned = [item for item in notes(trial) if item.code == SOURCE_MLIR_CODE]
    assert len(warned) == 1 and warned[0].severity == "warning", warned
    if label == "failed":
        assert error in notes(trial) and "status 134" in warned[0].message
    else:
        assert "max_chars" in warned[0].message and "never cut" in warned[0].message
    assert trial.final.end_reason is None and trial.attempts[-1].stage_reached == "S5", "the trial goes on"


def test_without_the_section_the_field_is_empty(base: Path, bench: Path) -> None:
    data = mlir_recipe()
    data.pop(SECTION)
    script = Script(replies=[program(0)], attempts=[doubled])
    _, trial = run(base, bench, script, data, "off")
    assert SEEN.calls == [] and "SYNTHETIC MLIR of" not in user_prompts(script)[0]
    assert [item for item in notes(trial) if item.code == SOURCE_MLIR_CODE] == []


def test_the_tt_mlir_v0_set_holds_the_source_mlir_template() -> None:
    from lassi.prompts import render

    for name, fields in SOURCE_MLIR_TEMPLATES.items():
        text = render("tt-mlir-v0", name, dict.fromkeys(fields, "SYNTHETIC\n"))
        assert text.isascii() and "cgeist" in text and "SYNTHETIC" in text, name


def test_the_runner_builds_the_lowering_toolchain_once_per_run(
    base: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[str] = []
    original = runner_module.build_toolchain

    def counting(name: str, *args: Any, **kwargs: Any) -> Any:
        """Count each build of a toolchain."""
        built.append(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(runner_module, "build_toolchain", counting)
    script = Script(replies=[program(0), program(1)], attempts=[doubled, doubled])
    run(base, bench, script, mlir_recipe(trials={"n": 2}), "twice")
    assert built.count(FakeLowering.name) == 1 and len(SEEN.calls) == 2


def test_the_lowering_toolchains_pin_joins_every_trials_toolchain_pins() -> None:
    # Acceptance item 5 (Agent Rule 10): a toolchain bound for no language, the source_mlir lowering, serves every
    # trial, so its pin joins each target language's toolchain_pins and the run's beside the target's own.
    from types import SimpleNamespace

    lowering = runner_module.BuiltToolchain(
        "cgeist-affine", (), object(), "/x", {},
        {"polygeist": {"VERSION": "77c04bb2", "PREFIX_NAME": "polygeist@77c04bb2"}})
    target = runner_module.BuiltToolchain(
        "ttkernel-ttmetal", ("ttkernel",), object(), "/y", {},
        {"tt-mlir": {"VERSION": "0.9.0.dev20260221", "PREFIX_NAME": "tt-mlir@0.9.0.dev20260221"}})
    executors = SimpleNamespace(declared=[], pins_for=lambda language: [])
    settings = SimpleNamespace(directions=[SimpleNamespace(source="cpp", target="ttkernel")])
    run_pins, target_pins = runner_module._run_pins(SimpleNamespace(path="r.yaml"), settings, [lowering, target],
                                                    executors)
    assert target_pins["ttkernel"].polygeist == "77c04bb2" and target_pins["ttkernel"].tt_mlir == "0.9.0.dev20260221"
    assert run_pins.polygeist == "77c04bb2"
