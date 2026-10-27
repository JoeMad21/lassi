"""Tests for task P17.41's dialect_lookup: the pinned dialect's definitions in a verify error's correction.

Bible: Project Recipes (Notes: the opt-in sections), IR Strategy (the
ttkernel row; ttmlir-opt's parse and verify step), Design Principle 4 and
Agent Rule 4. plans/spikes/mlir-issues.md, Half 1, finding 1: the model
did not know the ttkernel dialect's syntax at the pin; it borrowed names from
TT-Metalium's C++ API and missed ops' assembly formats, and one parser
message was not enough feedback.

The contract these tests fix:

1. dialect_lookup is an optional recipe section (defs and max_chars
   required, nearest optional), never materialized, refused under
   faithful: true; the runner refuses it with a fragment prompt set, with a
   template set lacking one of its templates, and with a definitions tree
   that does not load, before any directory exists.
2. With it, a compile correction whose listed errors include verify-stage
   errors carries, before correct.txt, the definitions block: the exact
   definition of each op or type an error names or the line it points at
   names, then for each name the dialect lacks its nearest names, then the
   definitions of those nearest names, each definition skipped whole past
   max_chars; its Request gets one dialect-lookup note.
3. A compile-stage error, a run-error correction, and a recipe without the
   section carry no block, and the prompts are what they were.

The suite, replies, definitions, and errors are SYNTHETIC; the fake
toolchain compiles nothing and the scripted executor runs nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_template_context import (
    CONTEXT_STAGES,
    CPP,
    ITEM,
    MODEL_ID,
    SUITE,
    TT,
    Script,
    crashed,
    doubled,
    load,
    make_assets,
    program,
    recipe_data,
    runs_root,
    scripted_backend,
    scripted_executor,
    template_set,
    user_prompts,
    write_recipe,
    write_suite,
    write_tree,
)

from lassi.core.interfaces import BuildResult
from lassi.core.recipe import RecipeError
from lassi.core.record import Attempt, Diagnostic, Trial, make_trial_id
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.stages import DIALECT_CODE, DIALECT_TEMPLATES, dialect_feedback
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.prompts.dialect_defs import DialectDefs, parse_defs

SECTION = "dialect_lookup"
DEFS = "syn-ttkernel-defs"
SET = "syn-dialect"
VERIFY, COMPILE = "// verify-error: ", "// compile-error: "
DEFS_TEXT = (
    "// ==== ttkernel.TensorAccessorArgs ====\n"
    'def TTKernel_TensorAccessorArgsOp : TTKernel_Op<"TensorAccessorArgs"> {\n'
    "  let summary = \"SYNTHETIC accessor arguments\";\n"
    "  let hasCustomAssemblyFormat = 1;\n"
    "}\n"
    "// SYNTHETIC custom format: TensorAccessorArgs(%cta, %crta)\n"
    "// ==== ttkernel.get_compile_time_arg_val ====\n"
    'def TTKernel_GetCompileArgValOp : TTKernel_Op<"get_compile_time_arg_val"> {\n'
    "  let assemblyFormat = [{ `(` $arg_index `)` attr-dict `:` functional-type(operands, results) }];\n"
    "}\n"
    "// ==== !ttkernel.TensorAccessorArgs ====\n"
    'def TTKernel_TensorAccessorArgs : TTKernel_Type<"TensorAccessorArgs", "TensorAccessorArgs"> {\n}\n'
    "// ==== !ttkernel.interleaved_addr_gen_fast ====\n"
    'def TTKernel_InterleavedAddrGenFast : TTKernel_Type<"InterleavedAddrGenFast", "interleaved_addr_gen_fast"> {\n'
    "}\n"
    "// ==== ttkernel.get_interleaved_addr_gen_fast ====\n"
    'def TTKernel_GetInterleavedAddrGenFastOp : TTKernel_Op<"get_interleaved_addr_gen_fast"> {\n}\n'
)
TEMPLATES = {
    "dialect_lookup": "SYNTHETIC: the pinned dialect's definitions follow.\n\n$definitions\n",
    "dialect_definition": "SYNTHETIC definition of $name:\n$text\n",
    "dialect_unknown": "SYNTHETIC: no $name in the dialect; nearest: $candidates\n",
}
# A line of a SYNTHETIC module that ttmlir-opt refuses: an op whose custom form takes no trailing type, and a type
# the dialect does not have (the spike's run01 attempts 2 and 8).
BAD_LINE = "%a = ttkernel.TensorAccessorArgs(%c0, %c0) : (i32, i32) -> !ttkernel.tensor_accessor_args"
UNKNOWN_OP = "%g = ttkernel.InterleavedAddrGenFast(%a, %b)"


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate variables a test could inherit, and point TMPDIR at a test directory."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "scratch" / "tmp").mkdir(parents=True)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch" / "tmp"))


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """Return the directory every run of a test writes under."""
    return tmp_path / "scratch"


@pytest.fixture
def assets(base: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return the temporary assets root: tt-host-v0, SET (it plus TEMPLATES), and the SYNTHETIC definitions DEFS."""
    root = make_assets(base, monkeypatch)
    template_set(root, SET, texts=TEMPLATES)
    write_tree(root / "context" / DEFS, {"syn_defs.ttkernel": DEFS_TEXT})
    return root


@pytest.fixture
def bench(base: Path, monkeypatch: pytest.MonkeyPatch, assets: Path) -> Path:
    """Return the bench root of the SYNTHETIC suite, with the assets root in place."""
    return write_suite(base, monkeypatch)


class VerifyToolchain:
    """Writes the files; a line ending in VERIFY or COMPILE and a message is an error at that line. Compiles nothing."""

    name = "fake-verify"
    capabilities = frozenset({"diagnostics"})

    def build(self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None) -> BuildResult:
        """Write `files`; return each marked line's error, and a PLACEHOLDER artifact when there is none."""
        found: list[Diagnostic] = []
        for path, text in sorted(files.items()):
            (Path(workdir) / path).parent.mkdir(parents=True, exist_ok=True)
            (Path(workdir) / path).write_bytes(text.encode("utf-8"))
            for number, line in enumerate(text.splitlines(), start=1):
                for marker, stage in ((VERIFY, "verify"), (COMPILE, "compile")):
                    if marker in line:
                        message = line.split(marker, 1)[1]
                        found.append(Diagnostic(stage=stage, severity="error", file=path, line=number, column=3,
                                                message=message))
        if found:
            return BuildResult(artifact=None, diagnostics=found)
        artifact = Path(workdir) / "main"
        artifact.write_bytes(b"PLACEHOLDER artifact\n")
        return BuildResult(artifact=artifact, diagnostics=[])


def registry(script: Script) -> Registry:
    """Return a test Registry: test_template_context's, with VerifyToolchain building tt."""
    found = Registry()
    found.register("LLMBackend", "scripted", scripted_backend(script))
    found.register("Executor", "scripted", scripted_executor(script))
    from test_template_context import FakeToolchain

    found.register("Toolchain", "fake-cpp", FakeToolchain)
    found.register("Toolchain", "fake-tt", VerifyToolchain)
    for name in ("binary_io", "stdout_mask"):
        found.register("Oracle", name, DEFAULT_REGISTRY.get("Oracle", name).factory)
    for name in CONTEXT_STAGES:
        found.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    return found


def lookup(**changes: Any) -> dict[str, Any]:
    """Return a dialect_lookup section on DEFS with `changes`."""
    return {"defs": DEFS, "max_chars": 4000, **changes}


def dialect_recipe(**changes: Any) -> dict[str, Any]:
    """Return test_template_context's cpp -> tt recipe on SET, with `changes`."""
    data = recipe_data(prompts=SET, loop={})
    data.update(changes)
    return data


def run(base: Path, bench: Path, script: Script, data: Mapping[str, Any], name: str) -> Trial:
    """Run recipe `data` as <base>/<name>.yaml on the test registry; return trial 1."""
    options = RunOptions(runs_root=runs_root(base), run_id=name, bench_root=bench, registry=registry(script))
    run_dir = run_recipe(write_recipe(base, data, name), options)
    trial_id = make_trial_id(name, MODEL_ID, SUITE, f"{CPP}-{TT}", ITEM, 1)
    return read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir))


def refused(base: Path, bench: Path, data: Mapping[str, Any], name: str = "refused") -> str:
    """Run `data`, expect a RunError before any directory exists, and return its message."""
    options = RunOptions(runs_root=runs_root(base), run_id=name, bench_root=bench,
                         registry=registry(Script(replies=[], attempts=[])))
    with pytest.raises(RunError) as caught:
        run_recipe(write_recipe(base, data, name), options)
    assert not (runs_root(base) / "runs" / name).exists()
    return str(caught.value)


def notes(trial: Trial, index: int) -> list[Diagnostic]:
    """Return the dialect-lookup notes of request `index`."""
    return [item for item in (trial.requests or [])[index].diagnostics if item.code == DIALECT_CODE]


# ---------------------------------------------------------------------------
# The recipe section (contract point 1)


def test_absent_the_section_adds_nothing(base: Path) -> None:
    recipe = load(base, dialect_recipe())
    assert SECTION not in recipe.data and SECTION not in recipe.canonical_yaml


def test_the_section_loads_as_written(base: Path) -> None:
    recipe = load(base, dialect_recipe(dialect_lookup=lookup(nearest=3)))
    assert recipe.data[SECTION] == {"defs": DEFS, "max_chars": 4000, "nearest": 3}


@pytest.mark.parametrize(
    "section",
    [{"defs": DEFS}, {"max_chars": 4000}, lookup(max_chars=10), lookup(nearest=0), lookup(defs="../x"),
     lookup(extra=1)],
    ids=["no-max", "no-defs", "small", "nearest-0", "path", "unknown-key"],
)
def test_a_bad_section_is_refused(base: Path, section: Mapping[str, Any]) -> None:
    with pytest.raises(RecipeError, match=SECTION):
        load(base, dialect_recipe(dialect_lookup=dict(section)))


def test_the_section_is_refused_under_faithful(base: Path) -> None:
    with pytest.raises(RecipeError) as caught:
        load(base, dialect_recipe(dialect_lookup=lookup(), faithful=True))
    assert SECTION in str(caught.value) and "faithful: true" in str(caught.value)


def test_the_runner_refuses_a_set_without_the_templates(base: Path, bench: Path, assets: Path) -> None:
    template_set(assets, "syn-without", texts={k: v for k, v in TEMPLATES.items() if k != "dialect_unknown"},
                 drop=["dialect_unknown"])
    message = refused(base, bench, dialect_recipe(prompts="syn-without", dialect_lookup=lookup()))
    assert SECTION in message and "dialect_unknown.txt" in message, message


def test_the_runner_refuses_a_missing_definitions_tree(base: Path, bench: Path) -> None:
    message = refused(base, bench, dialect_recipe(dialect_lookup=lookup(defs="syn-missing")))
    assert "syn-missing" in message and SECTION in message, message


def test_the_runner_refuses_the_section_with_a_fragment_set(base: Path, bench: Path, assets: Path) -> None:
    from test_correct_on_sim_gap import FRAGMENT_SET, FRAGMENTS

    write_tree(assets / "prompts" / FRAGMENT_SET, FRAGMENTS)
    data = dialect_recipe(prompts=FRAGMENT_SET, stages=["generate", "compile_loop"], dialect_lookup=lookup())
    data.pop("oracle")
    message = refused(base, bench, data)
    assert SECTION in message and "fragment" in message, message


# ---------------------------------------------------------------------------
# The block (contract points 2 and 3)


def test_a_verify_error_is_corrected_with_the_definitions_it_names(base: Path, bench: Path) -> None:
    script = Script(replies=[program(0, f"{BAD_LINE} {VERIFY}expected operation name in quotes"), program(1)],
                    attempts=[doubled])
    trial = run(base, bench, script, dialect_recipe(dialect_lookup=lookup()), "on")
    assert [request.stage for request in trial.requests or []] == ["generate", "compile_loop"]
    prompt = user_prompts(script)[1]
    assert prompt.startswith("SYNTHETIC: the pinned dialect's definitions follow.")
    defs = parse_defs(DEFS_TEXT)
    assert f"SYNTHETIC definition of ttkernel.TensorAccessorArgs:\n{defs['ttkernel.TensorAccessorArgs']}" in prompt
    unknown = "SYNTHETIC: no !ttkernel.tensor_accessor_args in the dialect; nearest: !ttkernel.TensorAccessorArgs"
    assert unknown in prompt
    assert prompt.index(unknown) < prompt.index("SYNTHETIC definition of !ttkernel.TensorAccessorArgs")
    assert prompt.index("pinned dialect's definitions") < prompt.index("Your tt program failed")
    noted = notes(trial, 1)
    assert len(noted) == 1 and "ttkernel.TensorAccessorArgs" in noted[0].message
    assert "!ttkernel.tensor_accessor_args" in noted[0].message and noted[0].severity == "note"
    assert notes(trial, 0) == []


def test_an_unknown_op_in_the_message_gets_its_nearest_names(base: Path, bench: Path) -> None:
    error = f"{UNKNOWN_OP} {VERIFY}custom op 'ttkernel.InterleavedAddrGenFast' is unknown"
    script = Script(replies=[program(0, error), program(1)], attempts=[doubled])
    trial = run(base, bench, script, dialect_recipe(dialect_lookup=lookup(nearest=2)), "unknown")
    prompt = user_prompts(script)[1]
    assert ("SYNTHETIC: no ttkernel.InterleavedAddrGenFast in the dialect; nearest: "
            "!ttkernel.interleaved_addr_gen_fast, ttkernel.get_interleaved_addr_gen_fast") in prompt
    assert len(notes(trial, 1)) == 1


@pytest.mark.parametrize("label", ["absent", "compile-stage"])
def test_no_block_without_the_section_or_a_verify_error(base: Path, bench: Path, label: str) -> None:
    marker = VERIFY if label == "absent" else COMPILE
    script = Script(replies=[program(0, f"{BAD_LINE} {marker}expected operation name in quotes"), program(1)],
                    attempts=[doubled])
    data = dialect_recipe() if label == "absent" else dialect_recipe(dialect_lookup=lookup())
    trial = run(base, bench, script, data, label)
    prompt = user_prompts(script)[1]
    assert prompt.startswith("Your tt program failed"), prompt[:200]
    assert notes(trial, 1) == []


def test_a_run_error_correction_carries_no_block(base: Path, bench: Path) -> None:
    script = Script(replies=[program(0), program(1)], attempts=[crashed, doubled])
    trial = run(base, bench, script, dialect_recipe(dialect_lookup=lookup()), "run-error")
    assert [request.stage for request in trial.requests or []] == ["generate", "run_loop"]
    assert user_prompts(script)[1].startswith("Your tt program failed") and notes(trial, 1) == []


def feedback_context(max_chars: int, nearest: int | None = None) -> Any:
    """Return a stand-in RunContext with the SYNTHETIC definitions and a dialect_lookup section."""
    section: dict[str, Any] = {"defs": DEFS, "max_chars": max_chars}
    if nearest is not None:
        section["nearest"] = nearest
    defs = DialectDefs(name=DEFS, dialect="ttkernel", definitions=parse_defs(DEFS_TEXT))
    return SimpleNamespace(recipe=SimpleNamespace(data={SECTION: section}), dialect=defs)


def attempt_with(lines: Sequence[str], errors: Sequence[Diagnostic]) -> Attempt:
    """Return a SYNTHETIC attempt whose one file holds `lines` and whose diagnostics are `errors`."""
    return Attempt(index=0, stage_reached="S1", files={"k.mlir": "\n".join(lines) + "\n"}, diagnostics=list(errors))


def test_dialect_feedback_reads_the_line_and_skips_a_definition_past_the_budget() -> None:
    error = Diagnostic(stage="verify", severity="error", file="k.mlir", line=2, column=5, message="expected ')'")
    previous = attempt_with(["// SYNTHETIC", BAD_LINE], [error, error])
    full = dialect_feedback(feedback_context(4000), previous)
    assert [name for name, _ in full.named] == ["ttkernel.TensorAccessorArgs"]
    assert full.unknown[0][0] == "!ttkernel.tensor_accessor_args"
    assert "!ttkernel.TensorAccessorArgs" in [name for name, _ in full.nearest]
    assert "ttkernel.TensorAccessorArgs" not in [name for name, _ in full.nearest], "a definition is carried once"
    small = dialect_feedback(feedback_context(120), previous)
    assert small.named == () and small.unknown == full.unknown, "a definition past the budget is skipped whole"
    assert all(len(text) <= 120 for _, text in small.nearest)
    warning = Diagnostic(stage="verify", severity="warning", file="k.mlir", line=2, message="SYNTHETIC")
    assert dialect_feedback(feedback_context(4000), attempt_with(["", BAD_LINE], [warning])).empty


def test_the_tt_mlir_v0_set_holds_the_dialect_templates() -> None:
    from lassi.prompts import render

    for name, fields in DIALECT_TEMPLATES.items():
        text = render("tt-mlir-v0", name, dict.fromkeys(fields, "SYNTHETIC"))
        assert text.isascii() and "SYNTHETIC" in text, name
