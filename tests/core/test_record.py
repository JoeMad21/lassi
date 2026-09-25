"""Tests for the Result record, trial naming, JSON, and the text store (P0.2), Trial provenance (P0.18), and requests.

The expected field names come from the Result Record yaml block in
docs/BIBLE.md, so drift between the bible and lassi/core/record.py fails in
either direction. The tests also cover validation, the trial_id naming rule
(bible Readability Standards, Naming), strict JSON round trips, unified_diff,
the sha256 text store in lassi/core/store.py, and write_trial and read_trial.
One test checks that the four P0.2 core modules have docstrings and type hints
on every public class and function and plain ASCII source (Readability
Standards). Every fixture value is synthetic and fixed; none is a measurement.

Trial.provenance (P0.18, the OQ-008 decision) is a copy of the run manifest:
a Provenance record of commit, dirty, device, sdk, and date, typed as
provenance.json holds them (commit, dirty, device, and sdk may be null there;
date may not). It is required: a Trial built without it, and a trial.json that
lacks it or one of its keys, are refused with a message naming the field.

Trial.requests (P2.1) records every model call: a list of Request records
(index, stage, attempt_index, messages, reply_ref, diagnostics), each message
a RequestMessage (role, ref). The bible block defines Request and
RequestMessage after Trial and before Attempt, and requests sits between
context and attempts. None means not recorded: a trial.json without the key
loads with None. A request's index is its position, and its attempt_index,
when set, names an attempt of the trial.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import json
import os
import shutil
import threading
import types
import typing
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from lassi.core import interfaces, record, store, trial_md

REPO = Path(__file__).resolve().parents[2]
BIBLE = REPO / "docs" / "BIBLE.md"

EXAMPLE_ID = "lassi-repro/qwen3-coder-30b-a3b-fp8/lassi-hecbench-10/omp-cuda/entropy/run01"
RECIPE_HASH = "0123456789abcdef" * 4
SHA_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SHA_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
VALID_SHA = "a" * 64

# Synthetic provenance values: FIXTURE_COMMIT is not a commit of this repository, and FIXTURE_DATE is in the
# format the runner writes started_utc (ISO 8601, seconds, UTC offset).
FIXTURE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
FIXTURE_DATE = "2026-09-23T12:34:56+00:00"
PROVENANCE_FIELDS = ("commit", "dirty", "device", "sdk", "date")

# Bible mappings whose keys are data (paths, component names), not record fields.
DICT_VALUED = frozenset({"Attempt.files", "Attempt.score.components"})

# Record paths the bible block defines as nested records; guards the parser against passing vacuously.
NESTED_RECORDS = {
    "Trial": [
        "Trial",
        "Trial.toolchain_pins",
        "Trial.provenance",
        "Trial.bench_item",
        "Trial.model",
        "Trial.model.sampling",
        "Trial.reference_run",
        "Trial.context",
        "Trial.final",
        "Trial.final.end_reason",
    ],
    "Attempt": [
        "Attempt",
        "Attempt.prompt_ref",
        "Attempt.run",
        "Attempt.alignment",
        "Attempt.profile",
        "Attempt.guards",
        "Attempt.score",
    ],
    "Request": ["Request", "Request.reply_ref"],
    "RequestMessage": ["RequestMessage", "RequestMessage.ref"],
    "Diagnostic": ["Diagnostic"],
}
# The top-level blocks of the bible's Result Record yaml block, in order, and the record class of each.
BIBLE_BLOCKS = ("Trial", "Request", "RequestMessage", "Attempt", "Diagnostic")

CORE_MODULES = ("lassi.core.record", "lassi.core.store", "lassi.core.trial_md", "lassi.core.parquet")
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)

RECORD_CLASSES: tuple[type, ...] = (
    record.TextRef,
    record.Diagnostic,
    record.RunInfo,
    record.Alignment,
    record.Profile,
    record.Guards,
    record.ScoreBreakdown,
    record.Attempt,
    record.RequestMessage,
    record.Request,
    record.ToolchainPins,
    record.Provenance,
    record.BenchItem,
    record.ModelInfo,
    record.Context,
    record.Final,
    record.Trial,
)

E_ACUTE = "\N{LATIN SMALL LETTER E WITH ACUTE}"
I_DIAERESIS = "\N{LATIN SMALL LETTER I WITH DIAERESIS}"
LEFT_QUOTE = "\N{LEFT DOUBLE QUOTATION MARK}"
RIGHT_QUOTE = "\N{RIGHT DOUBLE QUOTATION MARK}"
APOSTROPHE = "\N{RIGHT SINGLE QUOTATION MARK}"
EMOJI = "\N{GRINNING FACE}"

PROMPTS = (
    "Translate entropy.cpp from OpenMP to CUDA.\nReturn every file in a // FILE: block.\n",
    f"Fix the error in main.cu: identifier {LEFT_QUOTE}blockDimx{RIGHT_QUOTE} is undefined.",
)
RESPONSES = (
    f'// FILE: main.cu\r\nconst char* s = "caf{E_ACUTE}"; int distinctive_marker_0;\r\n',
    "",
)
STDOUT = "PASS\n"
# Synthetic request texts: the two system prompts and the summary request.
SYSTEM_GENERAL = "Synthetic general system prompt.\n"
SYSTEM_DIRECTION = "Synthetic direction system prompt.\n"
SUMMARY_REQUEST = "Summarize the synthetic notes.\n"
KNOWLEDGE = f"Target offload maps to a grid; na{I_DIAERESIS}ve copy.\n"
CODE_0 = "int x;\n"
CODE_1 = f"int y; // na{I_DIAERESIS}ve\n"


# ---------------------------------------------------------------------------
# Bible parsing helpers


def bible_record_block() -> list[str]:
    """Return the lines of the yaml block in the Result Record section of the bible."""
    lines = BIBLE.read_text(encoding="utf-8").splitlines()
    start = lines.index("## Result Record")
    begin = lines.index("```yaml", start) + 1
    end = lines.index("```", begin)
    return lines[begin:end]


def split_top_level(text: str) -> list[str]:
    """Split text on commas that are not nested inside braces or brackets."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def inline_keys(raw: str) -> dict[str, str] | None:
    """Return the ordered keys of an inline mapping like '{a, b: {c}}' with their raw values, else None."""
    raw = raw.strip()
    if not (raw.startswith("{") and raw.endswith("}")):
        return None
    keys: dict[str, str] = {}
    for entry in split_top_level(raw[1:-1]):
        name, _, value = entry.partition(":")
        keys[name.strip()] = value.strip()
    return keys


def bible_record_fields() -> dict[str, dict[str, str]]:
    """Map each top-level name in the bible block (Trial, Attempt, Diagnostic) to its ordered fields and raw values."""
    blocks: dict[str, dict[str, str]] = {}
    current: dict[str, str] = {}
    for raw in bible_record_block():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            current = blocks.setdefault(line.strip().rstrip(":"), {})
            continue
        stripped = line.strip()
        entries = [stripped] if ":" in stripped else split_top_level(stripped)
        for entry in entries:
            name, _, value = entry.partition(":")
            current[name.strip()] = value.strip()
    return blocks


def bible_choices(block: str, name: str) -> tuple[str, ...]:
    """Return the 'a | b | c' choices the bible lists for one field."""
    return tuple(part.strip() for part in bible_record_fields()[block][name].split("|"))


def unwrap_optional(hint: Any) -> Any:
    """Return X for a hint of X | None, else the hint itself."""
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        args = [arg for arg in typing.get_args(hint) if arg is not type(None)]
        assert len(args) == 1, hint
        return args[0]
    return hint


def check_record_fields(cls: type, fields: dict[str, str], where: str) -> list[str]:
    """Assert dataclass field names equal the bible fields, recursing into nested records; return visited paths."""
    names = [f.name for f in dataclasses.fields(cls)]
    assert names == list(fields), f"{where}: {names} != {list(fields)}"
    hints = typing.get_type_hints(cls)
    visited = [where]
    for name, raw in fields.items():
        path = f"{where}.{name}"
        sub = inline_keys(raw)
        if sub is None or path in DICT_VALUED:
            continue
        nested = unwrap_optional(hints[name])
        assert dataclasses.is_dataclass(nested), f"{path} is not a record: {nested}"
        visited += check_record_fields(nested, sub, path)
    return visited


def check_dict_keys(data: Any, fields: dict[str, str], where: str) -> list[str]:
    """Assert JSON dict keys equal the bible fields in order, recursing into nested records; return visited paths."""
    assert isinstance(data, dict), f"{where} is not a dict: {data!r}"
    assert list(data) == list(fields), f"{where}: {list(data)} != {list(fields)}"
    visited = [where]
    for name, raw in fields.items():
        path = f"{where}.{name}"
        sub = inline_keys(raw)
        if sub is None or path in DICT_VALUED:
            continue
        visited += check_dict_keys(data[name], sub, path)
    return visited


def assert_plain_json(value: Any, where: str = "root") -> None:
    """Assert value is built only from dict, list, str, int, float, bool, and None."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str), f"{where} key {key!r}"
            assert_plain_json(item, f"{where}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            assert_plain_json(item, f"{where}[{i}]")
    else:
        assert value is None or isinstance(value, (str, int, float, bool)), f"{where}: {type(value)}"


# ---------------------------------------------------------------------------
# Fixture builders


def text_ref(text: str) -> record.TextRef:
    """Return the TextRef the store contract assigns to text, computed independently with hashlib."""
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return record.TextRef(sha256=sha, path=f"texts/{sha[:2]}/{sha}.txt")


def example_bench() -> record.BenchItem:
    """Return the bench item that matches EXAMPLE_ID."""
    return record.BenchItem(suite="lassi-hecbench-10", item="entropy", split="eval", direction="omp-cuda")


def example_model() -> record.ModelInfo:
    """Return a fixed mock model with fixed sampling."""
    return record.ModelInfo(
        backend="mock", id="mock-fixture", sampling=interfaces.Sampling(temperature=0.2, top_p=0.95, max_tokens=4096)
    )


def model_with(**sampling_changes: Any) -> record.ModelInfo:
    """Return example_model with the given sampling values changed (interfaces.Sampling checks nothing)."""
    sampling = dataclasses.replace(example_model().sampling, **sampling_changes)
    return record.ModelInfo(backend="mock", id="mock-fixture", sampling=sampling)


def example_provenance(**changes: Any) -> record.Provenance:
    """Return the provenance of a compile-only run with git available, with the given fields changed.

    The sdk is None, as the runner writes the manifest's driver until an
    executor reports one.
    """
    fields: dict[str, Any] = {
        "commit": FIXTURE_COMMIT,
        "dirty": False,
        "device": "none (compile only)",
        "sdk": None,
        "date": FIXTURE_DATE,
    }
    fields.update(changes)
    return record.Provenance(**fields)


def full_provenance() -> record.Provenance:
    """Return a Provenance with every field set to a synthetic value."""
    return record.Provenance(
        commit=FIXTURE_COMMIT, dirty=True, device="fixture-device", sdk="fixture-sdk", date=FIXTURE_DATE
    )


def unknown_provenance() -> record.Provenance:
    """Return the provenance of a run where git was unavailable and the executor named no device or SDK."""
    return record.Provenance(commit=None, dirty=None, device=None, sdk=None, date=FIXTURE_DATE)


def required_trial_fields() -> dict[str, Any]:
    """Return the required Trial fields for EXAMPLE_ID other than provenance."""
    return {
        "trial_id": EXAMPLE_ID,
        "recipe_hash": RECIPE_HASH,
        "bench_item": example_bench(),
        "model": example_model(),
    }


def trial_with(**changes: Any) -> record.Trial:
    """Construct a Trial from the required fields for EXAMPLE_ID, with the given fields changed."""
    fields: dict[str, Any] = {**required_trial_fields(), "provenance": example_provenance()}
    fields.update(changes)
    return record.Trial(**fields)


def minimal_trial(attempts: list[record.Attempt] | None = None) -> record.Trial:
    """Return a Trial with only the required fields set, plus the given attempts."""
    return trial_with(attempts=list(attempts or []))


def sample_diagnostic() -> record.Diagnostic:
    """Return a Diagnostic with every field set."""
    return record.Diagnostic(
        stage="compile",
        severity="error",
        code="20",
        file="main.cu",
        line=3,
        column=24,
        message='identifier "blockDimx" is undefined',
    )


def full_requests(ref: Callable[[str], record.TextRef]) -> list[record.Request]:
    """Return a context request with a warning and one request per attempt, each text referenced through `ref`."""

    def messages(system: str, user: str) -> list[record.RequestMessage]:
        return [
            record.RequestMessage(role="system", ref=ref(system)),
            record.RequestMessage(role="user", ref=ref(user)),
        ]

    warning = record.Diagnostic(stage="parse", severity="warning", code="invalid-text", message="synthetic warning")
    return [
        record.Request(
            index=0,
            stage="summarize_context",
            attempt_index=None,
            messages=messages(SYSTEM_GENERAL, SUMMARY_REQUEST),
            reply_ref=ref(KNOWLEDGE),
            diagnostics=[warning],
        ),
        record.Request(
            index=1,
            stage="generate",
            attempt_index=0,
            messages=messages(SYSTEM_DIRECTION, PROMPTS[0]),
            reply_ref=ref(RESPONSES[0]),
        ),
        record.Request(
            index=2,
            stage="compile_loop",
            attempt_index=1,
            messages=messages(SYSTEM_DIRECTION, PROMPTS[1]),
            reply_ref=ref(RESPONSES[1]),
        ),
    ]


def full_trial(ref: Callable[[str], record.TextRef]) -> record.Trial:
    """Return a two-attempt Trial that sets every kind of field, with CRLF and non-ASCII text.

    Every text reference comes from `ref`: text_ref for tests that need no
    store, a store's put for tests that render trial.md.
    """
    first = record.Attempt(
        index=0,
        prompt_ref=ref(PROMPTS[0]),
        response_text=RESPONSES[0],
        files={"main.cu": CODE_0},
        stage_reached="S1",
        diagnostics=[
            sample_diagnostic(),
            record.Diagnostic(stage="compile", severity="note", message=f"1 error | caf{E_ACUTE}"),
        ],
    )
    second = record.Attempt(
        index=1,
        prompt_ref=ref(PROMPTS[1]),
        response_text=RESPONSES[1],
        files={"main.cu": CODE_1, "kernels/k.cuh": "#pragma once"},
        diff_from_previous=record.unified_diff(
            {"main.cu": CODE_0}, {"main.cu": CODE_1, "kernels/k.cuh": "#pragma once"}
        ),
        stage_reached="S5",
        run=record.RunInfo(exit_code=0, hang=False, sim_ub=None, wall_s=None, stdout_ref=ref(STDOUT), outputs_ref=None),
        alignment=record.Alignment(per_input=[1.0, 0.5], mean=0.75),
        profile=record.Profile(),
        guards=record.Guards(host_compute=False, harness_tamper=False, oracle_access=None),
        score=record.ScoreBreakdown(components={"alignment": 0.75, "energy": None}, scalar=None),
    )
    return record.Trial(
        trial_id=EXAMPLE_ID,
        recipe_hash=RECIPE_HASH,
        toolchain_pins=record.ToolchainPins(cuda="fixture-cuda", nvhpc="fixture-nvhpc"),
        provenance=full_provenance(),
        bench_item=example_bench(),
        model=example_model(),
        context=record.Context(knowledge_summary=KNOWLEDGE),
        requests=full_requests(ref),
        attempts=[first, second],
        final=record.Final(stage_reached="S5", alignment=0.75, score=None, corrections=1, wall_s=None),
    )


def json_trial() -> record.Trial:
    """Return full_trial with references computed by hashlib, for tests that need no store."""
    return full_trial(text_ref)


def stored_trial(text_store: store.TextStore) -> record.Trial:
    """Return full_trial with every referenced text put in the store, so trial.md can resolve them."""
    return full_trial(text_store.put)


def public_defs(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    """Return public top-level classes and functions, plus the public methods of public classes."""
    found: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, *FUNCTION_NODES)) or node.name.startswith("_"):
            continue
        found.append((node.name, node))
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, FUNCTION_NODES) and not item.name.startswith("_"):
                    found.append((f"{node.name}.{item.name}", item))
    return found


def is_typed(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True when every parameter except self or cls, and the return value, are annotated."""
    args = node.args
    params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    params += [a for a in (args.vararg, args.kwarg) if a is not None]
    params = [p for p in params if p.arg not in ("self", "cls")]
    return node.returns is not None and all(p.annotation is not None for p in params)


def mentions(message: str, value: Any) -> bool:
    """Return True when an error message names a value in str or repr form."""
    return str(value) in message or repr(value) in message


@pytest.fixture
def text_store(tmp_path: Path) -> store.TextStore:
    """Return a text store rooted in a fresh temporary directory."""
    root = tmp_path / "store"
    root.mkdir()
    return store.TextStore(root)


# ---------------------------------------------------------------------------
# Field names and constants from the bible


def test_bible_block_names_every_record_block() -> None:
    fields = bible_record_fields()
    assert tuple(fields) == BIBLE_BLOCKS
    assert list(fields["Diagnostic"]) == ["stage", "severity", "code", "file", "line", "column", "message"]
    trial = list(fields["Trial"])
    assert trial[trial.index("context") :][:3] == ["context", "requests", "attempts"]
    assert fields["Trial"]["requests"] == "[Request]"
    assert fields["Request"]["messages"] == "[RequestMessage]"


@pytest.mark.parametrize("name", BIBLE_BLOCKS)
def test_record_fields_match_bible(name: str) -> None:
    visited = check_record_fields(getattr(record, name), bible_record_fields()[name], name)
    assert visited == NESTED_RECORDS[name]


def test_json_keys_match_bible() -> None:
    fields = bible_record_fields()
    # An end reason is set so that its nested keys are checked too (the fixture's own final block has none).
    reason = record.EndReason(code="correction-cap", message="synthetic: the cap stopped the loop")
    trial = json_trial()
    trial = dataclasses.replace(trial, final=dataclasses.replace(trial.final, end_reason=reason))
    data = record.to_dict(trial)
    assert check_dict_keys(data, fields["Trial"], "Trial") == NESTED_RECORDS["Trial"]
    assert len(data["attempts"]) == 2
    for attempt in data["attempts"]:
        assert check_dict_keys(attempt, fields["Attempt"], "Attempt") == NESTED_RECORDS["Attempt"]
    diagnostics = data["attempts"][0]["diagnostics"]
    assert len(diagnostics) == 2
    for diagnostic in diagnostics:
        check_dict_keys(diagnostic, fields["Diagnostic"], "Diagnostic")
    assert len(data["requests"]) == 3
    for request in data["requests"]:
        assert check_dict_keys(request, fields["Request"], "Request") == NESTED_RECORDS["Request"]
        for message in request["messages"]:
            visited = check_dict_keys(message, fields["RequestMessage"], "RequestMessage")
            assert visited == NESTED_RECORDS["RequestMessage"]
    (diagnostic,) = data["requests"][0]["diagnostics"]
    check_dict_keys(diagnostic, fields["Diagnostic"], "Diagnostic")


def test_stage_constants_match_bible() -> None:
    assert record.STAGES == bible_choices("Attempt", "stage_reached")
    assert record.DIAGNOSTIC_STAGES == bible_choices("Diagnostic", "stage")
    assert record.SEVERITIES == bible_choices("Diagnostic", "severity")


def test_toolchain_pin_names_match_bible() -> None:
    pins = inline_keys(bible_record_fields()["Trial"]["toolchain_pins"])
    assert pins is not None
    assert record.TOOLCHAIN_PIN_NAMES == tuple(pins)
    assert [f.name for f in dataclasses.fields(record.ToolchainPins)] == list(record.TOOLCHAIN_PIN_NAMES)


# ---------------------------------------------------------------------------
# Provenance (P0.18)


def test_provenance_fields_are_the_bible_fields_in_order() -> None:
    bible = inline_keys(bible_record_fields()["Trial"]["provenance"])
    assert bible is not None and tuple(bible) == PROVENANCE_FIELDS
    assert [f.name for f in dataclasses.fields(record.Provenance)] == list(PROVENANCE_FIELDS)


def test_provenance_is_frozen_keyword_only_and_every_field_is_required() -> None:
    cls = record.Provenance
    assert dataclasses.is_dataclass(cls)
    assert cls.__dataclass_params__.frozen, "Provenance is not frozen"
    assert all(f.kw_only for f in dataclasses.fields(cls)), "Provenance has positional fields"
    for spec in dataclasses.fields(cls):
        assert spec.default is dataclasses.MISSING, f"Provenance.{spec.name} has a default"
        assert spec.default_factory is dataclasses.MISSING, f"Provenance.{spec.name} has a default factory"


@pytest.mark.parametrize("name", PROVENANCE_FIELDS)
def test_provenance_without_a_field_is_refused_naming_it(name: str) -> None:
    fields = record.to_dict(example_provenance())
    del fields[name]
    with pytest.raises((TypeError, ValueError)) as info:
        record.Provenance(**fields)
    assert name in str(info.value)


def test_provenance_holds_the_values_of_a_run_manifest() -> None:
    known = example_provenance()
    assert (known.commit, known.dirty, known.device, known.sdk, known.date) == (
        FIXTURE_COMMIT,
        False,
        "none (compile only)",
        None,
        FIXTURE_DATE,
    )
    # provenance.json holds null for the commit and the dirty flag when git is unavailable, for the device when
    # the executor names none, and for the driver (the sdk here) until an executor reports one.
    unknown = unknown_provenance()
    assert (unknown.commit, unknown.dirty, unknown.device, unknown.sdk, unknown.date) == (
        None,
        None,
        None,
        None,
        FIXTURE_DATE,
    )


def test_provenance_takes_a_sha1_or_sha256_commit_and_the_runner_date_format() -> None:
    # git object ids are 40 hex characters, or 64 in a SHA-256 repository; the runner writes started_utc with
    # datetime.isoformat(timespec="seconds") on a UTC time.
    assert example_provenance(commit="ab" * 32).commit == "ab" * 32
    date = datetime(2026, 9, 23, 12, 34, 56, tzinfo=timezone.utc).isoformat(timespec="seconds")
    assert example_provenance(date=date).date == date == FIXTURE_DATE


def test_trial_provenance_is_a_required_provenance_record() -> None:
    names = [f.name for f in dataclasses.fields(record.Trial)]
    assert names.index("provenance") == names.index("toolchain_pins") + 1
    spec = {f.name: f for f in dataclasses.fields(record.Trial)}["provenance"]
    assert spec.default is dataclasses.MISSING, "Trial.provenance has a default"
    assert spec.default_factory is dataclasses.MISSING, "Trial.provenance has a default factory"
    assert typing.get_type_hints(record.Trial)["provenance"] is record.Provenance, "Trial.provenance is optional"


def test_trial_without_provenance_is_refused_naming_the_field() -> None:
    with pytest.raises((TypeError, ValueError)) as info:
        record.Trial(**required_trial_fields())
    assert "provenance" in str(info.value)


def test_trial_keeps_the_provenance_it_was_given() -> None:
    assert trial_with().provenance == example_provenance()
    assert trial_with(provenance=unknown_provenance()).provenance == unknown_provenance()
    trial = trial_with(provenance=full_provenance())
    assert trial.with_attempt(record.Attempt(index=0, stage_reached="S0")).provenance == full_provenance()


# ---------------------------------------------------------------------------
# Dataclass shape and defaults


@pytest.mark.parametrize("cls", RECORD_CLASSES, ids=lambda c: c.__name__)
def test_records_are_frozen_and_keyword_only(cls: type) -> None:
    assert dataclasses.is_dataclass(cls)
    assert cls.__dataclass_params__.frozen, f"{cls.__name__} is not frozen"
    assert all(f.kw_only for f in dataclasses.fields(cls)), f"{cls.__name__} has positional fields"


@pytest.mark.parametrize("name", CORE_MODULES)
def test_core_modules_documented_typed_ascii(name: str) -> None:
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin, f"missing module {name}"
    source = Path(spec.origin).read_bytes()
    assert source.isascii(), f"{name} has non-ASCII source text"
    tree = ast.parse(source.decode("ascii"), filename=spec.origin)
    undocumented = [qual for qual, node in public_defs(tree) if not ast.get_docstring(node)]
    assert not undocumented, f"{name}: no docstring on {undocumented}"
    untyped = [qual for qual, node in public_defs(tree) if isinstance(node, FUNCTION_NODES) and not is_typed(node)]
    assert not untyped, f"{name}: missing type hints on {untyped}"


def test_positional_construction_is_rejected() -> None:
    with pytest.raises(TypeError):
        record.Diagnostic("compile", "error", message="m")  # type: ignore[misc]


def test_records_cannot_be_mutated() -> None:
    trial = minimal_trial()
    with pytest.raises(dataclasses.FrozenInstanceError):
        trial.recipe_hash = VALID_SHA  # type: ignore[misc]


def test_model_info_reuses_interfaces_sampling() -> None:
    assert unwrap_optional(typing.get_type_hints(record.ModelInfo)["sampling"]) is interfaces.Sampling
    model = example_model()
    assert isinstance(model.sampling, interfaces.Sampling)


def test_attempt_defaults() -> None:
    attempt = record.Attempt(index=0, stage_reached="S0")
    assert attempt.prompt_ref is None
    assert attempt.response_text == ""
    assert attempt.files == {}
    assert attempt.diff_from_previous == ""
    assert attempt.diagnostics == []
    assert attempt.run == record.RunInfo()
    assert attempt.alignment == record.Alignment()
    assert attempt.profile == record.Profile()
    assert attempt.guards == record.Guards()
    assert attempt.score == record.ScoreBreakdown()


def test_mutable_defaults_are_not_shared() -> None:
    first = record.Attempt(index=0, stage_reached="S0")
    second = record.Attempt(index=0, stage_reached="S0")
    assert first.files is not second.files
    assert first.diagnostics is not second.diagnostics
    assert record.Alignment().per_input is not record.Alignment().per_input
    assert record.ScoreBreakdown().components is not record.ScoreBreakdown().components
    assert minimal_trial().attempts is not minimal_trial().attempts


def test_trial_and_part_defaults() -> None:
    trial = minimal_trial()
    assert trial.toolchain_pins == record.ToolchainPins()
    assert all(getattr(trial.toolchain_pins, name) is None for name in record.TOOLCHAIN_PIN_NAMES)
    assert trial.context == record.Context(knowledge_summary="", source_description="")
    assert trial.requests is None, "requests default to not recorded"
    assert trial.attempts == []
    assert trial.final == record.Final(stage_reached=None, alignment=None, score=None, corrections=0, wall_s=None)
    assert record.RunInfo() == record.RunInfo(
        exit_code=None,
        hang=None,
        sim_ub=None,
        wall_s=None,
        stdout_ref=None,
        outputs_ref=None,
        stdout_truncated=None,
        stderr_truncated=None,
        workdir_incomplete=None,
    )
    assert record.Alignment() == record.Alignment(per_input=[], mean=None)
    assert record.Profile() == record.Profile(runtime_s=None, avg_power_w=None, energy_j=None)
    assert record.Guards() == record.Guards(host_compute=None, harness_tamper=None, oracle_access=None)
    assert record.ScoreBreakdown() == record.ScoreBreakdown(components={}, scalar=None)
    diagnostic = record.Diagnostic(stage="run", severity="warning", message="m")
    assert (diagnostic.code, diagnostic.file, diagnostic.line, diagnostic.column) == (None, None, None, None)


def test_boundary_values_are_accepted() -> None:
    record.Alignment(per_input=[0.0, 1.0, 0.5], mean=0.0)
    record.Alignment(per_input=[], mean=1.0)
    record.Attempt(index=0, stage_reached="S5")
    record.Final(stage_reached="S0", corrections=0)
    record.TextRef(sha256=VALID_SHA, path="texts/aa/x.txt")
    for stage in record.DIAGNOSTIC_STAGES:
        for severity in record.SEVERITIES:
            record.Diagnostic(stage=stage, severity=severity, message="m")


# ---------------------------------------------------------------------------
# Validation

NAN = float("nan")
INF = float("inf")

FIELD_ERRORS = [
    pytest.param(lambda: record.TextRef(sha256="A" * 64, path="texts/aa/x.txt"), "sha256", "A" * 64, id="sha-upper"),
    pytest.param(lambda: record.TextRef(sha256="a" * 63, path="texts/aa/x.txt"), "sha256", "a" * 63, id="sha-short"),
    pytest.param(lambda: record.TextRef(sha256="g" * 64, path="texts/gg/x.txt"), "sha256", "g" * 64, id="sha-not-hex"),
    pytest.param(lambda: record.TextRef(sha256=VALID_SHA, path=""), "path", "", id="path-empty"),
    pytest.param(
        lambda: record.TextRef(sha256=VALID_SHA, path="/texts/aa/x.txt"), "path", "/texts/aa/x.txt", id="path-abs"
    ),
    pytest.param(
        lambda: record.TextRef(sha256=VALID_SHA, path="texts\\aa\\x.txt"), "path", "texts\\aa\\x.txt", id="path-bs"
    ),
    pytest.param(
        lambda: record.TextRef(sha256=VALID_SHA, path="texts/../x.txt"), "path", "texts/../x.txt", id="path-dotdot"
    ),
    pytest.param(lambda: record.TextRef(sha256=VALID_SHA, path="../x.txt"), "path", "../x.txt", id="path-lead-dotdot"),
    pytest.param(
        lambda: record.TextRef(sha256=VALID_SHA, path="C:/texts/x.txt"), "path", "C:/texts/x.txt", id="path-drive"
    ),
    pytest.param(
        lambda: record.TextRef(sha256=VALID_SHA, path="texts/aa/x\n.txt"), "path", "texts/aa/x\n.txt", id="path-newline"
    ),
    pytest.param(
        lambda: record.TextRef(sha256=VALID_SHA, path="texts//x.txt"), "path", "texts//x.txt", id="path-empty-segment"
    ),
    pytest.param(lambda: record.TextRef(sha256=VALID_SHA, path="texts/aa/"), "path", "texts/aa/", id="path-trailing"),
    pytest.param(lambda: record.TextRef(sha256=VALID_SHA, path=5), "path", 5, id="path-int"),
    pytest.param(
        lambda: record.Diagnostic(stage="link", severity="error", message="m"), "stage", "link", id="diag-stage"
    ),
    pytest.param(
        lambda: record.Diagnostic(stage="run", severity="fatal", message="m"), "severity", "fatal", id="diag-sev"
    ),
    pytest.param(lambda: record.Attempt(index=-1, stage_reached="S0"), "index", -1, id="attempt-index"),
    pytest.param(lambda: record.Attempt(index=0, stage_reached="S6"), "stage_reached", "S6", id="attempt-stage"),
    pytest.param(lambda: record.Attempt(index=0, stage_reached="s1"), "stage_reached", "s1", id="attempt-stage-case"),
    pytest.param(lambda: record.Alignment(per_input=[0.5, 1.5]), "per_input", 1.5, id="per-input-high"),
    pytest.param(lambda: record.Alignment(per_input=[-0.25]), "per_input", -0.25, id="per-input-low"),
    pytest.param(lambda: record.Alignment(per_input=[INF]), "per_input", INF, id="per-input-inf"),
    pytest.param(lambda: record.Alignment(mean=1.25), "mean", 1.25, id="mean-high"),
    pytest.param(lambda: record.Alignment(mean=NAN), "mean", NAN, id="mean-nan"),
    pytest.param(lambda: record.RunInfo(wall_s=NAN), "wall_s", NAN, id="run-wall-nan"),
    pytest.param(lambda: record.RunInfo(wall_s=-INF), "wall_s", -INF, id="run-wall-neg-inf"),
    pytest.param(lambda: record.Profile(runtime_s=INF), "runtime_s", INF, id="runtime-inf"),
    pytest.param(lambda: record.Profile(avg_power_w=NAN), "avg_power_w", NAN, id="power-nan"),
    pytest.param(lambda: record.Profile(energy_j=-INF), "energy_j", -INF, id="energy-neg-inf"),
    pytest.param(lambda: record.ScoreBreakdown(scalar=NAN), "scalar", NAN, id="scalar-nan"),
    pytest.param(lambda: record.ScoreBreakdown(components={"energy": NAN}), "components", NAN, id="component-nan"),
    pytest.param(lambda: record.ScoreBreakdown(components={"energy": INF}), "components", INF, id="component-inf"),
    pytest.param(lambda: record.ScoreBreakdown(components={"e": -INF}), "components", -INF, id="component-neg-inf"),
    pytest.param(lambda: model_with(temperature=NAN), "sampling.temperature", NAN, id="sampling-temperature-nan"),
    pytest.param(lambda: model_with(temperature=-INF), "sampling.temperature", -INF, id="sampling-temperature-inf"),
    pytest.param(lambda: model_with(top_p=INF), "sampling.top_p", INF, id="sampling-top-p-inf"),
    pytest.param(lambda: model_with(top_p=NAN), "sampling.top_p", NAN, id="sampling-top-p-nan"),
    pytest.param(lambda: model_with(max_tokens="x"), "sampling.max_tokens", "x", id="sampling-max-tokens-str"),
    pytest.param(lambda: model_with(max_tokens=True), "sampling.max_tokens", True, id="sampling-max-tokens-bool"),
    pytest.param(lambda: record.Final(stage_reached="S7"), "stage_reached", "S7", id="final-stage"),
    pytest.param(lambda: record.Final(corrections=-1), "corrections", -1, id="final-corrections"),
    pytest.param(lambda: record.Final(alignment=INF), "alignment", INF, id="final-alignment-inf"),
    pytest.param(lambda: record.Final(alignment=1.5), "alignment", 1.5, id="final-alignment-high"),
    pytest.param(lambda: record.Final(alignment=-0.25), "alignment", -0.25, id="final-alignment-low"),
    pytest.param(lambda: record.Final(score=NAN), "score", NAN, id="final-score-nan"),
    pytest.param(lambda: record.Final(score=10**400), "score", 10**400, id="final-score-huge-int"),
    pytest.param(lambda: record.Final(wall_s=INF), "wall_s", INF, id="final-wall-inf"),
    pytest.param(
        lambda: record.BenchItem(suite="", item="i", split="eval", direction="d"), "suite", "", id="bench-suite"
    ),
    pytest.param(
        lambda: record.BenchItem(suite="s", item="", split="eval", direction="d"), "item", "", id="bench-item"
    ),
    pytest.param(lambda: record.BenchItem(suite="s", item="i", split="", direction="d"), "split", "", id="bench-split"),
    pytest.param(
        lambda: record.BenchItem(suite="s", item="i", split="eval", direction=""), "direction", "", id="bench-dir"
    ),
    pytest.param(lambda: trial_with(recipe_hash="abc"), "recipe_hash", "abc", id="recipe-hash-short"),
    pytest.param(lambda: trial_with(recipe_hash="AB" * 32), "recipe_hash", "AB" * 32, id="recipe-hash-upper"),
    pytest.param(lambda: trial_with(trial_id="not/a/trial"), "trial_id", "not/a/trial", id="trial-id"),
    pytest.param(lambda: example_provenance(commit=""), "commit", "", id="provenance-commit-empty"),
    pytest.param(lambda: example_provenance(commit="0123abc"), "commit", "0123abc", id="provenance-commit-short"),
    pytest.param(
        lambda: example_provenance(commit=FIXTURE_COMMIT.upper()),
        "commit",
        FIXTURE_COMMIT.upper(),
        id="provenance-commit-upper",
    ),
    pytest.param(lambda: example_provenance(commit="g" * 40), "commit", "g" * 40, id="provenance-commit-not-hex"),
    pytest.param(lambda: example_provenance(device=""), "device", "", id="provenance-device-empty"),
    pytest.param(lambda: example_provenance(sdk=""), "sdk", "", id="provenance-sdk-empty"),
    pytest.param(lambda: example_provenance(date=""), "date", "", id="provenance-date-empty"),
    pytest.param(lambda: example_provenance(date="yesterday"), "date", "yesterday", id="provenance-date-text"),
    pytest.param(
        lambda: example_provenance(date="2026-09-23T12:34:56"),
        "date",
        "2026-09-23T12:34:56",
        id="provenance-date-no-offset",
    ),
    pytest.param(
        lambda: example_provenance(date="2026-09-23T14:34:56+02:00"),
        "date",
        "2026-09-23T14:34:56+02:00",
        id="provenance-date-not-utc",
    ),
]


def request_with(**changes: Any) -> record.Request:
    """Return a context request with one user message, with the given fields changed."""
    fields: dict[str, Any] = {
        "index": 0,
        "stage": "summarize_context",
        "attempt_index": None,
        "messages": [record.RequestMessage(role="user", ref=text_ref(SUMMARY_REQUEST))],
        "reply_ref": text_ref(KNOWLEDGE),
    }
    fields.update(changes)
    return record.Request(**fields)


FIELD_ERRORS += [
    pytest.param(lambda: request_with(index=-1), "index", -1, id="request-index"),
    pytest.param(lambda: request_with(stage=""), "stage", "", id="request-stage-empty"),
    pytest.param(lambda: request_with(attempt_index=-1), "attempt_index", -1, id="request-attempt-index"),
    pytest.param(lambda: request_with(messages=[]), "messages", [], id="request-no-message"),
    pytest.param(
        lambda: record.RequestMessage(role="", ref=text_ref(SUMMARY_REQUEST)), "role", "", id="request-message-role"
    ),
]

# Wrong types: the constructors take exactly what from_dict gives back, so every
# record that can be built also survives the JSON round trip.
TYPE_ERRORS = [
    pytest.param(lambda: record.Attempt(index=False, stage_reached="S0"), "index", False, id="attempt-index-bool"),
    pytest.param(lambda: record.Attempt(index="0", stage_reached="S0"), "index", "0", id="attempt-index-str"),
    pytest.param(
        lambda: record.Attempt(index=0, stage_reached="S0", files={"a.c": 1}), "files", 1, id="attempt-files-value"
    ),
    pytest.param(
        lambda: record.Attempt(index=0, stage_reached="S0", diagnostics=["m"]), "diagnostics", "m", id="attempt-diag"
    ),
    pytest.param(lambda: record.Final(corrections=True), "corrections", True, id="final-corrections-bool"),
    pytest.param(lambda: record.Final(corrections="3"), "corrections", "3", id="final-corrections-str"),
    pytest.param(lambda: record.Final(score="1.0"), "score", "1.0", id="final-score-str"),
    pytest.param(lambda: record.Final(score=False), "score", False, id="final-score-bool"),
    pytest.param(lambda: record.Alignment(per_input=[True]), "per_input", True, id="per-input-bool"),
    pytest.param(lambda: record.Alignment(per_input=["0.5"]), "per_input", "0.5", id="per-input-str"),
    pytest.param(lambda: record.Alignment(per_input=(0.5,)), "per_input", (0.5,), id="per-input-tuple"),
    pytest.param(lambda: record.RunInfo(hang=1), "hang", 1, id="run-hang-int"),
    pytest.param(lambda: record.RunInfo(exit_code=True), "exit_code", True, id="run-exit-code-bool"),
    pytest.param(lambda: record.RunInfo(stdout_ref="texts/x.txt"), "stdout_ref", "texts/x.txt", id="run-ref-str"),
    pytest.param(
        lambda: record.Diagnostic(stage="run", severity="error", line=True, message="m"), "line", True, id="diag-line"
    ),
    pytest.param(lambda: record.Diagnostic(stage="run", severity="error", message=5), "message", 5, id="diag-message"),
    pytest.param(lambda: record.Guards(host_compute="yes"), "host_compute", "yes", id="guards-str"),
    pytest.param(lambda: record.ToolchainPins(cuda=12.5), "cuda", 12.5, id="pin-float"),
    pytest.param(lambda: record.Context(knowledge_summary=None), "knowledge_summary", None, id="context-none"),
    pytest.param(lambda: record.ScoreBreakdown(components={1: 0.5}), "components", {1: 0.5}, id="component-key"),
    pytest.param(
        lambda: record.ModelInfo(backend=None, id="m", sampling=example_model().sampling), "backend", None, id="backend"
    ),
    pytest.param(
        lambda: record.ModelInfo(backend="m", id="m", sampling={"top_p": 1.0}),
        "sampling",
        {"top_p": 1.0},
        id="sampling",
    ),
    pytest.param(
        lambda: trial_with(attempts=(record.Attempt(index=0, stage_reached="S0"),)),
        "attempts",
        (record.Attempt(index=0, stage_reached="S0"),),
        id="trial-attempts-tuple",
    ),
    pytest.param(lambda: request_with(index=True), "index", True, id="request-index-bool"),
    pytest.param(lambda: request_with(attempt_index="0"), "attempt_index", "0", id="request-attempt-index-str"),
    pytest.param(lambda: request_with(messages=["user"]), "messages", "user", id="request-messages-str"),
    pytest.param(
        lambda: request_with(reply_ref=f"texts/{VALID_SHA[:2]}/{VALID_SHA}.txt"),
        "reply_ref",
        f"texts/{VALID_SHA[:2]}/{VALID_SHA}.txt",
        id="request-reply-ref-str",
    ),
    pytest.param(lambda: request_with(diagnostics=["m"]), "diagnostics", "m", id="request-diagnostics-str"),
    pytest.param(lambda: record.RequestMessage(role=None, ref=text_ref("x")), "role", None, id="message-role-none"),
    pytest.param(lambda: record.RequestMessage(role="user", ref="x"), "ref", "x", id="message-ref-str"),
    pytest.param(
        lambda: trial_with(requests=(request_with(),)), "requests", (request_with(),), id="trial-requests-tuple"
    ),
    pytest.param(lambda: example_provenance(commit=5), "commit", 5, id="provenance-commit-int"),
    pytest.param(lambda: example_provenance(dirty="false"), "dirty", "false", id="provenance-dirty-str"),
    pytest.param(lambda: example_provenance(dirty=0), "dirty", 0, id="provenance-dirty-int"),
    pytest.param(lambda: example_provenance(device=3), "device", 3, id="provenance-device-int"),
    pytest.param(lambda: example_provenance(sdk=12.6), "sdk", 12.6, id="provenance-sdk-float"),
    pytest.param(lambda: example_provenance(date=None), "date", None, id="provenance-date-none"),
    pytest.param(lambda: example_provenance(date=20260923), "date", 20260923, id="provenance-date-int"),
    pytest.param(lambda: trial_with(provenance=None), "provenance", None, id="trial-provenance-none"),
    pytest.param(
        lambda: trial_with(provenance={"commit": FIXTURE_COMMIT}),
        "provenance",
        {"commit": FIXTURE_COMMIT},
        id="trial-provenance-dict",
    ),
]


@pytest.mark.parametrize(("build", "field", "value"), FIELD_ERRORS + TYPE_ERRORS)
def test_invalid_field_raises_value_error_naming_field_and_value(
    build: Callable[[], object], field: str, value: Any
) -> None:
    with pytest.raises(ValueError) as info:
        build()
    message = str(info.value)
    assert field in message, message
    assert mentions(message, value), message


CROSS_FIELD_ERRORS = [
    pytest.param({"suite": "other-suite"}, [], "bench_item.suite", "other-suite", id="suite-vs-bench-segment"),
    pytest.param({"direction": "cuda-omp"}, [], "bench_item.direction", "cuda-omp", id="direction-vs-segment"),
    pytest.param({"item": "layout"}, [], "bench_item.item", "layout", id="item-vs-item-segment"),
    pytest.param({}, [1], "attempts[0].index", 1, id="first-attempt-index-1"),
    pytest.param({}, [0, 0], "attempts[1].index", 0, id="repeated-index"),
    pytest.param({}, [0, 2], "attempts[1].index", 2, id="skipped-index"),
]


@pytest.mark.parametrize(("bench_changes", "indexes", "field", "value"), CROSS_FIELD_ERRORS)
def test_trial_cross_field_rules(bench_changes: dict[str, str], indexes: list[int], field: str, value: Any) -> None:
    bench = dataclasses.replace(example_bench(), **bench_changes)
    attempts = [record.Attempt(index=i, stage_reached="S0") for i in indexes]
    with pytest.raises(ValueError) as info:
        trial_with(bench_item=bench, attempts=attempts)
    message = str(info.value)
    assert field in message, message
    assert mentions(message, value), message


REQUEST_ORDER_ERRORS = [
    pytest.param([1], 0, "requests[0].index", 1, id="first-request-index-1"),
    pytest.param([0, 0], 0, "requests[1].index", 0, id="repeated-request-index"),
    pytest.param([0, 2], 0, "requests[1].index", 2, id="skipped-request-index"),
]


@pytest.mark.parametrize(("indexes", "attempts", "field", "value"), REQUEST_ORDER_ERRORS)
def test_trial_requests_are_in_index_order(indexes: list[int], attempts: int, field: str, value: Any) -> None:
    requests = [request_with(index=index) for index in indexes]
    with pytest.raises(ValueError) as info:
        trial_with(requests=requests)
    message = str(info.value)
    assert field in message, message
    assert mentions(message, value), message


def test_a_request_attempt_index_must_name_an_attempt_of_the_trial() -> None:
    attempt = record.Attempt(index=0, stage_reached="S1")
    trial = trial_with(requests=[request_with(attempt_index=0)], attempts=[attempt])
    assert trial.requests is not None and trial.requests[0].attempt_index == 0
    with pytest.raises(ValueError) as info:
        trial_with(requests=[request_with(attempt_index=1)], attempts=[attempt])
    assert "requests[0].attempt_index" in str(info.value) and mentions(str(info.value), 1)


def test_with_request_appends_and_starts_a_list_on_a_trial_not_recorded() -> None:
    empty = minimal_trial()
    first = request_with()
    second = request_with(index=1, stage="describe_source")
    one = empty.with_request(first)
    assert empty.requests is None
    assert one.requests == [first]
    assert one.with_request(second).requests == [first, second]
    assert trial_with(requests=[]).with_request(first).requests == [first]
    for wrong in (request_with(index=1), request_with(index=0)):
        with pytest.raises(ValueError) as info:
            (empty if wrong.index else one).with_request(wrong)
        assert "request.index" in str(info.value)


def test_int_values_in_float_fields_become_floats() -> None:
    model = model_with(temperature=0, top_p=1)
    assert model.sampling == interfaces.Sampling(temperature=0.0, top_p=1.0, max_tokens=4096)
    assert [type(model.sampling.temperature), type(model.sampling.top_p)] == [float, float]
    final = record.Final(alignment=1, score=0, wall_s=2)
    assert [type(final.alignment), type(final.score), type(final.wall_s)] == [float, float, float]
    alignment = record.Alignment(per_input=[1, 0], mean=1)
    assert [type(value) for value in [*alignment.per_input, alignment.mean]] == [float, float, float]
    score = record.ScoreBreakdown(components={"a": 1, "b": None}, scalar=0)
    assert type(score.components["a"]) is float and type(score.scalar) is float
    assert type(record.RunInfo(wall_s=3).wall_s) is float
    assert type(record.Profile(energy_j=5).energy_j) is float


def test_records_copy_caller_containers() -> None:
    files = {"a.c": "x\n"}
    per_input = [0.5]
    attempt = record.Attempt(index=0, stage_reached="S0", files=files, alignment=record.Alignment(per_input=per_input))
    files["b.c"] = "y\n"
    per_input.append(1.0)
    assert attempt.files == {"a.c": "x\n"}
    assert attempt.alignment.per_input == [0.5]


def test_int_valued_record_json_is_stable_across_a_round_trip() -> None:
    trial = trial_with(model=model_with(temperature=0, top_p=1), final=record.Final(alignment=1, wall_s=2))
    text = record.to_json(trial)
    back = record.from_json(record.Trial, text)
    assert back == trial
    assert record.to_json(back) == text
    assert '"temperature": 0.0,' in text


def test_trial_split_is_not_part_of_the_id() -> None:
    trial = trial_with(bench_item=dataclasses.replace(example_bench(), split="train"))
    assert trial.bench_item.split == "train"


# ---------------------------------------------------------------------------
# Naming

VALID_IDS = [
    (
        EXAMPLE_ID,
        ("lassi-repro", "qwen3-coder-30b-a3b-fp8", "lassi-hecbench-10", "omp-cuda", "entropy", 1),
    ),
    (
        "lassi-ee/qwen3-coder-30b-gspo-lora-r8/hecbench/cuda-omp/layout/run12",
        ("lassi-ee", "qwen3-coder-30b-gspo-lora-r8", "hecbench", "cuda-omp", "layout", 12),
    ),
    (
        "lassi-df/Model_v1.2+q4/2024/c-ttkernel/item_3/run100",
        ("lassi-df", "Model_v1.2+q4", "2024", "c-ttkernel", "item_3", 100),
    ),
    ("p/a/b/d/i/run00", ("p", "a", "b", "d", "i", 0)),
    ("p/a/b/d/i/run007", ("p", "a", "b", "d", "i", 7)),
]

INVALID_IDS = [
    pytest.param("lassi-repro/arm/bench/omp-cuda/run01", id="five-segments"),
    pytest.param("lassi-repro/arm/bench/omp-cuda/entropy/extra/run01", id="seven-segments"),
    pytest.param("lassi-repro/arm/bench/omp-cuda/entropy/run1", id="run1"),
    pytest.param("lassi-repro/arm/bench/omp-cuda/entropy/Run01", id="Run01"),
    pytest.param("lassi-repro/arm/bench/omp-cuda/entropy/runXY", id="run-no-digits"),
    pytest.param("lassi-repro/arm/bench/omp-cuda/entropy/run01a", id="run-suffix"),
    pytest.param("lassi-repro/arm/bench/omp-cuda/entropy/run01\n", id="run-trailing-newline"),
    pytest.param("lassi-repro//bench/omp-cuda/entropy/run01", id="empty-segment"),
    pytest.param("lassi-repro/../bench/omp-cuda/entropy/run01", id="dotdot-segment"),
    pytest.param("lassi-repro/./bench/omp-cuda/entropy/run01", id="dot-segment"),
    pytest.param("lassi-repro/arm/bench/omp-cuda/my item/run01", id="space-in-segment"),
    pytest.param(" lassi-repro/arm/bench/omp-cuda/entropy/run01", id="leading-space"),
    pytest.param("-lassi/arm/bench/omp-cuda/entropy/run01", id="leading-dash-project"),
    pytest.param("lassi-repro/-arm/bench/omp-cuda/entropy/run01", id="leading-dash-arm"),
    pytest.param("/lassi-repro/arm/bench/omp-cuda/entropy/run01", id="leading-slash"),
    pytest.param("lassi-repro/arm/bench/omp-cuda/entropy/run01/", id="trailing-slash"),
    pytest.param("lassi-repro/arm/bench/omp-cuda/entropy\\x/run01", id="backslash"),
    pytest.param("lassi-repro/arm/bench/omp:cuda/entropy/run01", id="colon"),
    pytest.param(f"lassi-repro/arm/bench/omp-cuda/entrop{E_ACUTE}/run01", id="non-ascii"),
    pytest.param("", id="empty"),
]


@pytest.mark.parametrize(("trial_id", "parts"), VALID_IDS)
def test_parse_trial_id_valid(trial_id: str, parts: tuple[Any, ...]) -> None:
    parsed = record.parse_trial_id(trial_id)
    assert isinstance(parsed, record.TrialId)
    assert parsed == record.TrialId(*parts)


def test_trial_id_fields() -> None:
    assert record.TrialId._fields == ("project", "arm", "bench", "direction", "item", "run")
    parsed = record.parse_trial_id(EXAMPLE_ID)
    assert parsed.bench == "lassi-hecbench-10"
    assert parsed.run == 1
    assert isinstance(parsed.run, int)


@pytest.mark.parametrize("trial_id", INVALID_IDS)
def test_parse_trial_id_rejects_invalid(trial_id: str) -> None:
    with pytest.raises(ValueError) as info:
        record.parse_trial_id(trial_id)
    assert mentions(str(info.value), trial_id), str(info.value)


@pytest.mark.parametrize("trial_id", INVALID_IDS)
def test_trial_rejects_invalid_id(trial_id: str) -> None:
    with pytest.raises(ValueError):
        trial_with(trial_id=trial_id)


def test_make_trial_id_formats_example() -> None:
    made = record.make_trial_id("lassi-repro", "qwen3-coder-30b-a3b-fp8", "lassi-hecbench-10", "omp-cuda", "entropy", 1)
    assert made == EXAMPLE_ID


def test_make_trial_id_pads_and_extends_run() -> None:
    assert record.make_trial_id("p", "a", "b", "d", "i", 0) == "p/a/b/d/i/run00"
    assert record.make_trial_id("p", "a", "b", "d", "i", 9) == "p/a/b/d/i/run09"
    assert record.make_trial_id("p", "a", "b", "d", "i", 123) == "p/a/b/d/i/run123"


@pytest.mark.parametrize(("trial_id", "parts"), VALID_IDS[:4])
def test_make_and_parse_round_trip(trial_id: str, parts: tuple[Any, ...]) -> None:
    assert record.make_trial_id(*parts) == trial_id
    assert record.parse_trial_id(record.make_trial_id(*parts)) == record.TrialId(*parts)


@pytest.mark.parametrize(
    "parts",
    [
        pytest.param(("p", "a", "b", "d", "bad item", 1), id="space"),
        pytest.param(("", "a", "b", "d", "i", 1), id="empty-project"),
        pytest.param(("p", "-a", "b", "d", "i", 1), id="leading-dash"),
        pytest.param(("p", "a", "b/c", "d", "i", 1), id="slash-in-segment"),
        pytest.param(("p", "a", "b", "d", "..", 1), id="dotdot"),
        pytest.param(("p", "a", "b", "d", "i", -1), id="negative-run"),
        pytest.param(("p", "a", "b", "d", "i", True), id="bool-run"),
        pytest.param(("p", "a", "b", "d", "i", "1"), id="str-run"),
        pytest.param((5, "a", "b", "d", "i", 1), id="int-segment"),
    ],
)
def test_make_trial_id_rejects_invalid(parts: tuple[Any, ...]) -> None:
    with pytest.raises(ValueError):
        record.make_trial_id(*parts)


# ---------------------------------------------------------------------------
# with_attempt


def test_with_attempt_appends_without_mutating() -> None:
    empty = minimal_trial()
    first = record.Attempt(index=0, stage_reached="S1")
    second = record.Attempt(index=1, stage_reached="S5")
    one = empty.with_attempt(first)
    two = one.with_attempt(second)
    assert empty.attempts == []
    assert one.attempts == [first]
    assert two.attempts == [first, second]
    assert one is not empty
    assert dataclasses.replace(two, attempts=[]) == empty


@pytest.mark.parametrize("index", [1, 2])
def test_with_attempt_rejects_wrong_index_on_empty(index: int) -> None:
    with pytest.raises(ValueError):
        minimal_trial().with_attempt(record.Attempt(index=index, stage_reached="S0"))


def test_with_attempt_rejects_repeated_index() -> None:
    trial = minimal_trial([record.Attempt(index=0, stage_reached="S0")])
    with pytest.raises(ValueError):
        trial.with_attempt(record.Attempt(index=0, stage_reached="S0"))
    assert len(trial.attempts) == 1


# ---------------------------------------------------------------------------
# unified_diff

TEN_LINES = "".join(f"l{i}\n" for i in range(1, 11))

DIFF_CASES = [
    pytest.param(
        {"src/k.cu": "int a;\nint b;\nint c;\n"},
        {"src/k.cu": "int a;\nint B;\nint c;\n"},
        "--- a/src/k.cu\n+++ b/src/k.cu\n@@ -1,3 +1,3 @@\n int a;\n-int b;\n+int B;\n int c;\n",
        id="changed",
    ),
    pytest.param(
        {},
        {"new.h": "#pragma once\n"},
        "--- /dev/null\n+++ b/new.h\n@@ -0,0 +1 @@\n+#pragma once\n",
        id="added",
    ),
    pytest.param(
        {"old.c": "int x;\nint y;\n"},
        {},
        "--- a/old.c\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-int x;\n-int y;\n",
        id="removed",
    ),
    pytest.param(
        {"m.c": "a\nb"},
        {"m.c": "a\nc"},
        "--- a/m.c\n+++ b/m.c\n@@ -1,2 +1,2 @@\n a\n-b\n\\ No newline at end of file\n"
        "+c\n\\ No newline at end of file\n",
        id="no-final-newline-both",
    ),
    pytest.param(
        {"m.c": "a\nb"},
        {"m.c": "a\nb\n"},
        "--- a/m.c\n+++ b/m.c\n@@ -1,2 +1,2 @@\n a\n-b\n\\ No newline at end of file\n+b\n",
        id="final-newline-added",
    ),
    pytest.param(
        {"m.c": "a\nb"},
        {"m.c": "A\nb"},
        "--- a/m.c\n+++ b/m.c\n@@ -1,2 +1,2 @@\n-a\n+A\n b\n\\ No newline at end of file\n",
        id="no-final-newline-context",
    ),
    pytest.param(
        {"t.c": TEN_LINES},
        {"t.c": TEN_LINES.replace("l1\n", "L1\n", 1).replace("l10\n", "L10\n")},
        "--- a/t.c\n+++ b/t.c\n@@ -1,4 +1,4 @@\n-l1\n+L1\n l2\n l3\n l4\n@@ -7,4 +7,4 @@\n l7\n l8\n l9\n-l10\n+L10\n",
        id="three-lines-of-context",
    ),
    pytest.param(
        {"b.c": "1\n", "a.c": "x\n"},
        {"c.c": "z\n", "b.c": "1\n", "a.c": "y\n"},
        "--- a/a.c\n+++ b/a.c\n@@ -1 +1 @@\n-x\n+y\n--- /dev/null\n+++ b/c.c\n@@ -0,0 +1 @@\n+z\n",
        id="sorted-paths-unchanged-skipped",
    ),
    pytest.param({}, {"b/__init__.py": ""}, "--- /dev/null\n+++ b/b/__init__.py\n", id="added-empty"),
    pytest.param({"e.c": ""}, {}, "--- a/e.c\n+++ /dev/null\n", id="removed-empty"),
    pytest.param(
        {"x.py": "a\n"},
        {"x.py": "a\n", "e.py": "", "f.py": "b\n"},
        "--- /dev/null\n+++ b/e.py\n--- /dev/null\n+++ b/f.py\n@@ -0,0 +1 @@\n+b\n",
        id="added-empty-beside-others",
    ),
    pytest.param({"e.c": ""}, {"e.c": "x\n"}, "--- a/e.c\n+++ b/e.c\n@@ -0,0 +1 @@\n+x\n", id="empty-filled"),
    pytest.param({"a.c": "x\n", "b.c": "y"}, {"b.c": "y", "a.c": "x\n"}, "", id="no-change"),
    pytest.param({"e.c": ""}, {"e.c": ""}, "", id="empty-unchanged"),
    pytest.param({}, {}, "", id="both-empty"),
]


@pytest.mark.parametrize(("previous", "current", "expected"), DIFF_CASES)
def test_unified_diff(previous: dict[str, str], current: dict[str, str], expected: str) -> None:
    assert record.unified_diff(previous, current) == expected


def test_unified_diff_lines_all_end_with_newline() -> None:
    diff = record.unified_diff({"m.c": "a\nb"}, {"m.c": "a\nc", "n.c": "q"})
    assert diff.endswith("\n")
    assert all(line.endswith("\n") for line in diff.splitlines(keepends=True))


# ---------------------------------------------------------------------------
# JSON

DIAGNOSTIC_JSON = """{
  "stage": "compile",
  "severity": "error",
  "code": "20",
  "file": "main.cu",
  "line": 3,
  "column": 24,
  "message": "identifier \\"blockDimx\\" is undefined"
}
"""


def test_diagnostic_to_json_exact() -> None:
    assert record.to_json(sample_diagnostic()) == DIAGNOSTIC_JSON


def test_minimal_attempt_to_dict_exact() -> None:
    assert record.to_dict(record.Attempt(index=0, stage_reached="S0")) == {
        "index": 0,
        "prompt_ref": None,
        "response_text": "",
        "files": {},
        "diff_from_previous": "",
        "stage_reached": "S0",
        "diagnostics": [],
        "run": {
            "exit_code": None,
            "hang": None,
            "sim_ub": None,
            "wall_s": None,
            "stdout_ref": None,
            "outputs_ref": None,
            "stdout_truncated": None,
            "stderr_truncated": None,
            "workdir_incomplete": None,
        },
        "alignment": {"per_input": [], "mean": None},
        "profile": {"runtime_s": None, "avg_power_w": None, "energy_j": None},
        "guards": {"host_compute": None, "harness_tamper": None, "oracle_access": None},
        "score": {"components": {}, "scalar": None},
    }


def test_to_dict_nested_values() -> None:
    data = record.to_dict(json_trial())
    assert data["model"] == {
        "backend": "mock",
        "id": "mock-fixture",
        "sampling": {"temperature": 0.2, "top_p": 0.95, "max_tokens": 4096},
    }
    prompt, stdout = text_ref(PROMPTS[0]), text_ref(STDOUT)
    assert data["attempts"][0]["prompt_ref"] == {"sha256": prompt.sha256, "path": prompt.path}
    assert data["attempts"][1]["run"]["stdout_ref"] == {"sha256": stdout.sha256, "path": stdout.path}
    assert data["attempts"][1]["score"] == {"components": {"alignment": 0.75, "energy": None}, "scalar": None}
    assert_plain_json(data)


def test_to_json_format() -> None:
    trial = json_trial()
    text = record.to_json(trial)
    assert text == json.dumps(record.to_dict(trial), indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    assert text.isascii()
    assert "\r" not in text


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(sample_diagnostic, id="diagnostic"),
        pytest.param(
            lambda: record.Diagnostic(stage="jit", severity="note", message=f"caf{E_ACUTE}\r\nx"), id="diagnostic-min"
        ),
        pytest.param(lambda: record.Attempt(index=0, stage_reached="S0"), id="attempt-min"),
        pytest.param(lambda: json_trial().attempts[0], id="attempt-diagnostics"),
        pytest.param(lambda: json_trial().attempts[1], id="attempt-run"),
        pytest.param(minimal_trial, id="trial-min"),
        pytest.param(json_trial, id="trial-full"),
        pytest.param(example_provenance, id="provenance"),
        pytest.param(unknown_provenance, id="provenance-unknown"),
        pytest.param(lambda: trial_with(provenance=unknown_provenance()), id="trial-unknown-provenance"),
        pytest.param(lambda: json_trial().requests[0], id="request-context"),
        pytest.param(lambda: json_trial().requests[1], id="request-attempt"),
        pytest.param(lambda: json_trial().requests[0].messages[0], id="request-message"),
        pytest.param(lambda: trial_with(requests=[]), id="trial-no-request"),
    ],
)
def test_json_round_trip(build: Callable[[], Any]) -> None:
    value = build()
    assert record.from_json(type(value), record.to_json(value)) == value
    assert record.from_dict(type(value), record.to_dict(value)) == value


def test_round_trip_rebuilds_nested_types() -> None:
    trial = record.from_json(record.Trial, record.to_json(json_trial()))
    assert isinstance(trial.model.sampling, interfaces.Sampling)
    assert isinstance(trial.bench_item, record.BenchItem)
    assert isinstance(trial.toolchain_pins, record.ToolchainPins)
    assert isinstance(trial.provenance, record.Provenance)
    assert trial.provenance == full_provenance()
    assert isinstance(trial.final, record.Final)
    assert all(isinstance(a, record.Attempt) for a in trial.attempts)
    assert all(isinstance(d, record.Diagnostic) for d in trial.attempts[0].diagnostics)
    assert isinstance(trial.attempts[0].prompt_ref, record.TextRef)
    assert isinstance(trial.attempts[1].run.stdout_ref, record.TextRef)
    assert trial.attempts[1].run.outputs_ref is None
    assert trial.requests is not None and all(isinstance(r, record.Request) for r in trial.requests)
    assert all(isinstance(m, record.RequestMessage) for r in trial.requests for m in r.messages)
    assert isinstance(trial.requests[0].reply_ref, record.TextRef)
    assert isinstance(trial.requests[0].diagnostics[0], record.Diagnostic)
    assert (trial.requests[0].attempt_index, trial.requests[2].attempt_index) == (None, 1)


def test_a_trial_without_requests_reads_as_not_recorded() -> None:
    data = record.to_dict(json_trial())
    del data["requests"]
    assert record.from_dict(record.Trial, data).requests is None
    data["requests"] = None
    assert record.from_dict(record.Trial, data).requests is None
    data["requests"] = []
    assert record.from_dict(record.Trial, data).requests == []


def test_trial_json_carries_the_provenance_as_the_bible_names_it() -> None:
    data = record.to_dict(minimal_trial())
    assert list(data)[:5] == ["trial_id", "recipe_hash", "toolchain_pins", "provenance", "bench_item"]
    assert data["provenance"] == {
        "commit": FIXTURE_COMMIT,
        "dirty": False,
        "device": "none (compile only)",
        "sdk": None,
        "date": FIXTURE_DATE,
    }
    assert list(data["provenance"]) == list(PROVENANCE_FIELDS)
    text = record.to_json(trial_with(provenance=unknown_provenance()))
    assert '"provenance": {\n    "commit": null,\n    "dirty": null,\n    "device": null,\n    "sdk": null,\n' in text


def test_from_dict_refuses_a_null_provenance() -> None:
    data = record.to_dict(minimal_trial())
    data["provenance"] = None
    with pytest.raises(ValueError) as info:
        record.from_dict(record.Trial, data)
    assert "provenance" in str(info.value)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        pytest.param("commit", 5, id="commit-int"),
        pytest.param("dirty", "false", id="dirty-str"),
        pytest.param("dirty", 1, id="dirty-int"),
        pytest.param("device", ["none"], id="device-list"),
        pytest.param("sdk", 12.6, id="sdk-float"),
        pytest.param("date", None, id="date-null"),
    ],
)
def test_from_dict_checks_provenance_types(key: str, value: Any) -> None:
    data = record.to_dict(minimal_trial())
    data["provenance"][key] = value
    with pytest.raises(ValueError) as info:
        record.from_dict(record.Trial, data)
    assert f"provenance.{key}" in str(info.value)


def diagnostic_data(**changes: Any) -> dict[str, Any]:
    """Return to_dict of sample_diagnostic with the given keys set."""
    data = dict(record.to_dict(sample_diagnostic()))
    data.update(changes)
    return data


def test_from_dict_rejects_unknown_key() -> None:
    with pytest.raises(ValueError) as info:
        record.from_dict(record.Diagnostic, diagnostic_data(hint="x"))
    assert "hint" in str(info.value)
    assert "Diagnostic" in str(info.value)


@pytest.mark.parametrize(
    ("path", "key", "owner"),
    [
        pytest.param(("attempts", 1, "run"), "signal", "RunInfo", id="run"),
        pytest.param(("model", "sampling"), "seed", "Sampling", id="sampling"),
        pytest.param(("attempts", 0, "diagnostics", 0), "hint", "Diagnostic", id="diagnostic"),
        pytest.param((), "notes", "Trial", id="trial"),
        pytest.param(("provenance",), "host", "Provenance", id="provenance"),
        pytest.param(("requests", 0), "notes", "Request", id="request"),
        pytest.param(("requests", 0, "messages", 1), "text", "RequestMessage", id="request-message"),
    ],
)
def test_from_dict_rejects_unknown_nested_key(path: tuple[Any, ...], key: str, owner: str) -> None:
    data = record.to_dict(json_trial())
    target = data
    for step in path:
        target = target[step]
    target[key] = 1
    with pytest.raises(ValueError) as info:
        record.from_dict(record.Trial, data)
    assert key in str(info.value)
    assert owner in str(info.value)


@pytest.mark.parametrize(
    ("path", "key"),
    [
        pytest.param((), "bench_item", id="trial-bench-item"),
        pytest.param((), "trial_id", id="trial-id"),
        pytest.param(("model", "sampling"), "top_p", id="sampling-top-p"),
        pytest.param(("attempts", 0), "stage_reached", id="attempt-stage"),
        pytest.param(("attempts", 0, "diagnostics", 0), "severity", id="diagnostic-severity"),
        pytest.param((), "provenance", id="trial-provenance"),
        pytest.param(("provenance",), "commit", id="provenance-commit"),
        pytest.param(("provenance",), "dirty", id="provenance-dirty"),
        pytest.param(("provenance",), "device", id="provenance-device"),
        pytest.param(("provenance",), "sdk", id="provenance-sdk"),
        pytest.param(("provenance",), "date", id="provenance-date"),
        pytest.param(("requests", 0), "stage", id="request-stage"),
        pytest.param(("requests", 0), "attempt_index", id="request-attempt-index"),
        pytest.param(("requests", 1), "messages", id="request-messages"),
        pytest.param(("requests", 1), "reply_ref", id="request-reply-ref"),
        pytest.param(("requests", 1, "messages", 0), "role", id="request-message-role"),
    ],
)
def test_from_dict_rejects_missing_required_key(path: tuple[Any, ...], key: str) -> None:
    data = record.to_dict(json_trial())
    target = data
    for step in path:
        target = target[step]
    del target[key]
    with pytest.raises(ValueError) as info:
        record.from_dict(record.Trial, data)
    assert key in str(info.value)


def test_from_dict_fills_missing_optional_keys() -> None:
    data = {"stage": "run", "severity": "warning", "message": "m"}
    assert record.from_dict(record.Diagnostic, data) == record.Diagnostic(stage="run", severity="warning", message="m")
    assert record.from_dict(record.Attempt, {"index": 0, "stage_reached": "S0"}) == record.Attempt(
        index=0, stage_reached="S0"
    )


@pytest.mark.parametrize(
    ("cls_name", "data"),
    [
        pytest.param("Diagnostic", lambda: diagnostic_data(line=True), id="int-line"),
        pytest.param("Diagnostic", lambda: diagnostic_data(column=False), id="int-column"),
        pytest.param("Attempt", lambda: {"index": False, "stage_reached": "S0"}, id="int-index"),
        pytest.param("Final", lambda: {"corrections": True}, id="int-corrections"),
        pytest.param("RunInfo", lambda: {"exit_code": True}, id="int-exit-code"),
        pytest.param("Profile", lambda: {"runtime_s": True}, id="float-runtime"),
        pytest.param("Alignment", lambda: {"per_input": [True], "mean": None}, id="float-list"),
        pytest.param("ScoreBreakdown", lambda: {"components": {"a": True}, "scalar": None}, id="float-dict"),
        pytest.param("Final", lambda: {"score": False}, id="float-score"),
    ],
)
def test_from_dict_rejects_bool_for_number(cls_name: str, data: Callable[[], dict[str, Any]]) -> None:
    with pytest.raises(ValueError):
        record.from_dict(getattr(record, cls_name), data())


def test_from_dict_converts_int_to_float() -> None:
    profile = record.from_dict(record.Profile, {"runtime_s": 2, "avg_power_w": None, "energy_j": 0})
    assert profile == record.Profile(runtime_s=2.0, energy_j=0.0)
    assert type(profile.runtime_s) is float
    assert type(profile.energy_j) is float
    alignment = record.from_dict(record.Alignment, {"per_input": [1, 0], "mean": 1})
    assert [type(v) for v in alignment.per_input] == [float, float]
    assert type(alignment.mean) is float
    score = record.from_dict(record.ScoreBreakdown, {"components": {"a": 1, "b": None}, "scalar": 0})
    assert score == record.ScoreBreakdown(components={"a": 1.0, "b": None}, scalar=0.0)
    assert type(score.components["a"]) is float
    assert type(score.scalar) is float


def test_from_dict_converts_sampling_ints_to_float() -> None:
    data = record.to_dict(minimal_trial())
    data["model"]["sampling"] = {"temperature": 0, "top_p": 1, "max_tokens": 16}
    trial = record.from_dict(record.Trial, data)
    assert trial.model.sampling == interfaces.Sampling(temperature=0.0, top_p=1.0, max_tokens=16)
    assert type(trial.model.sampling.temperature) is float
    assert type(trial.model.sampling.top_p) is float


NON_FINITE_LITERALS = [
    pytest.param("NaN", id="nan"),
    pytest.param("Infinity", id="inf"),
    pytest.param("-Infinity", id="neg-inf"),
    pytest.param("9" * 400, id="int-too-large-for-float"),
]


@pytest.mark.parametrize("literal", NON_FINITE_LITERALS)
@pytest.mark.parametrize("key", ["temperature", "top_p"])
def test_from_json_rejects_non_finite_sampling(key: str, literal: str) -> None:
    data = record.to_dict(minimal_trial())
    data["model"]["sampling"][key] = "@VALUE@"
    text = json.dumps(data).replace('"@VALUE@"', literal)
    with pytest.raises(ValueError) as info:
        record.from_json(record.Trial, text)
    assert f"sampling.{key}" in str(info.value)


@pytest.mark.parametrize("literal", NON_FINITE_LITERALS)
def test_from_json_rejects_non_finite_score(literal: str) -> None:
    with pytest.raises(ValueError) as info:
        record.from_json(record.Final, '{"score": ' + literal + "}")
    assert "Final.score" in str(info.value)


@pytest.mark.parametrize("value", [NAN, INF, -INF])
def test_json_text_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValueError):
        record.json_text({"x": value})


def test_from_dict_runs_validation() -> None:
    with pytest.raises(ValueError):
        record.from_dict(record.Diagnostic, diagnostic_data(stage="link"))
    data = record.to_dict(minimal_trial())
    data["bench_item"]["suite"] = "other-suite"
    with pytest.raises(ValueError):
        record.from_dict(record.Trial, data)


# ---------------------------------------------------------------------------
# Text store


def test_sha256_text_known_values() -> None:
    assert store.sha256_text("") == SHA_EMPTY
    assert store.sha256_text("abc") == SHA_ABC
    assert store.sha256_text(f"caf{E_ACUTE}") == hashlib.sha256(f"caf{E_ACUTE}".encode("utf-8")).hexdigest()


def test_text_not_found_error_is_key_error() -> None:
    assert issubclass(store.TextNotFoundError, KeyError)


def test_put_writes_one_file_at_hash_path(text_store: store.TextStore) -> None:
    assert isinstance(text_store.root, Path)
    ref = text_store.put("abc")
    assert ref == record.TextRef(sha256=SHA_ABC, path=f"texts/ba/{SHA_ABC}.txt")
    files = sorted(p.relative_to(text_store.root).as_posix() for p in text_store.root.rglob("*") if p.is_file())
    assert files == [f"texts/ba/{SHA_ABC}.txt"]


def test_put_stores_each_text_once(text_store: store.TextStore) -> None:
    first = text_store.put("same text\n")
    path = text_store.root / first.path
    os.utime(path, (1_000_000_000, 1_000_000_000))
    second = text_store.put("same text\n")
    assert second == first
    assert path.stat().st_mtime == 1_000_000_000
    text_store.put("other text\n")
    files = [p for p in text_store.root.rglob("*") if p.is_file()]
    assert len(files) == 2


def test_put_keeps_exact_bytes(text_store: store.TextStore) -> None:
    text = f"line one\r\nline two\n caf{E_ACUTE} {APOSTROPHE} {EMOJI}\rend"
    ref = text_store.put(text)
    assert (text_store.root / ref.path).read_bytes() == text.encode("utf-8")
    assert text_store.get(ref) == text
    assert text_store.get(ref.sha256) == text


def test_put_empty_text(text_store: store.TextStore) -> None:
    ref = text_store.put("")
    assert ref == record.TextRef(sha256=SHA_EMPTY, path=f"texts/e3/{SHA_EMPTY}.txt")
    assert (text_store.root / ref.path).read_bytes() == b""
    assert text_store.get(ref) == ""


def test_contains(text_store: store.TextStore) -> None:
    ref = text_store.put("abc")
    assert ref in text_store
    assert SHA_ABC in text_store
    assert SHA_EMPTY not in text_store
    assert text_ref("absent") not in text_store


def test_get_missing_text_raises(text_store: store.TextStore) -> None:
    with pytest.raises(store.TextNotFoundError):
        text_store.get(SHA_EMPTY)
    with pytest.raises(store.TextNotFoundError):
        text_store.get(text_ref("absent"))


def test_get_detects_tampered_text(text_store: store.TextStore) -> None:
    ref = text_store.put("original\n")
    (text_store.root / ref.path).write_bytes(b"tampered\n")
    with pytest.raises(ValueError):
        text_store.get(ref)
    with pytest.raises(ValueError):
        text_store.get(ref.sha256)


def test_get_treats_a_directory_at_the_text_path_as_missing(text_store: store.TextStore) -> None:
    (text_store.root / "texts" / SHA_ABC[:2] / f"{SHA_ABC}.txt").mkdir(parents=True)
    with pytest.raises(store.TextNotFoundError):
        text_store.get(SHA_ABC)
    assert SHA_ABC not in text_store


def test_put_writes_through_a_temporary_file(text_store: store.TextStore, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(src: Any, dst: Any) -> None:
        calls.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(store.os, "replace", recording_replace)
    ref = text_store.put("atomic\n")
    target = text_store.root / ref.path
    assert len(calls) == 1
    src, dst = calls[0]
    assert dst == target
    assert src.parent == target.parent and src != target
    assert [p.name for p in target.parent.iterdir()] == [target.name]
    text_store.put("atomic\n")
    assert len(calls) == 1


def put_at_once(text_store: store.TextStore, text: str, workers: int) -> list[record.TextRef]:
    """Return the references from `workers` threads that each put `text` at the same moment."""
    barrier = threading.Barrier(workers)

    def put_once(_: int) -> record.TextRef:
        barrier.wait()
        return text_store.put(text)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(put_once, range(workers)))


@pytest.mark.parametrize("text", ["", "same prompt\n", "same prompt\n" * 20000], ids=["empty", "short", "long"])
def test_put_same_text_concurrently(tmp_path: Path, text: str) -> None:
    ref = text_ref(text)
    for round_index in range(10):
        text_store = store.TextStore(tmp_path / f"store{round_index}")
        assert put_at_once(text_store, text, workers=8) == [ref] * 8
        assert text_store.get(ref) == text
        stored = [p.relative_to(text_store.root).as_posix() for p in text_store.root.rglob("*") if p.is_file()]
        assert stored == [ref.path]


def test_put_still_raises_when_nothing_was_stored(text_store: store.TextStore, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_replace(src: Any, dst: Any) -> None:
        raise PermissionError(13, "denied", str(dst))

    monkeypatch.setattr(store.os, "replace", failing_replace)
    with pytest.raises(PermissionError):
        text_store.put("abc")
    assert SHA_ABC not in text_store
    assert [p for p in text_store.root.rglob("*") if p.is_file()] == []


def test_get_rejects_a_reference_outside_the_store_layout(text_store: store.TextStore) -> None:
    ref = text_store.put("abc")
    elsewhere = record.TextRef(sha256=ref.sha256, path="texts/elsewhere.txt")
    with pytest.raises(ValueError) as info:
        text_store.get(elsewhere)
    assert "texts/elsewhere.txt" in str(info.value)
    assert elsewhere not in text_store
    assert ref in text_store
    attempt = record.Attempt(index=0, stage_reached="S0", prompt_ref=elsewhere)
    with pytest.raises(ValueError):
        trial_md.render_trial_md(minimal_trial([attempt]), text_store)


# ---------------------------------------------------------------------------
# trial_dir, write_trial, read_trial


def test_trial_dir(tmp_path: Path) -> None:
    expected = tmp_path / "lassi-repro" / "qwen3-coder-30b-a3b-fp8" / "lassi-hecbench-10" / "omp-cuda"
    assert store.trial_dir(tmp_path, EXAMPLE_ID) == expected / "entropy" / "run01"


def test_trial_dir_rejects_invalid_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        store.trial_dir(tmp_path, "lassi-repro/../x/omp-cuda/entropy/run01")


def expected_trial_json(trial: record.Trial) -> str:
    """Return the trial.json text the contract specifies: to_dict with each response_text replaced by its ref."""
    data = record.to_dict(trial)
    for attempt in data["attempts"]:
        ref = text_ref(attempt["response_text"])
        attempt["response_text"] = {"sha256": ref.sha256, "path": ref.path}
    return json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def test_write_trial_files(tmp_path: Path, text_store: store.TextStore) -> None:
    trial = stored_trial(text_store)
    run_root = tmp_path / "runs"
    out = store.write_trial(trial, run_root, text_store)
    assert out == store.trial_dir(run_root, EXAMPLE_ID)
    assert sorted(p.name for p in out.iterdir()) == ["trial.json", "trial.md"]
    assert (out / "trial.json").read_bytes() == expected_trial_json(trial).encode("ascii")
    assert (out / "trial.md").read_bytes() == trial_md.render_trial_md(trial, text_store).encode("ascii")


def test_write_trial_externalizes_responses(tmp_path: Path, text_store: store.TextStore) -> None:
    trial = stored_trial(text_store)
    out = store.write_trial(trial, tmp_path / "runs", text_store)
    raw = (out / "trial.json").read_text(encoding="ascii")
    assert "distinctive_marker_0" not in raw
    data = json.loads(raw)
    for attempt, text in zip(data["attempts"], RESPONSES, strict=True):
        ref = text_ref(text)
        assert attempt["response_text"] == {"sha256": ref.sha256, "path": ref.path}
        assert text_store.get(ref) == text
    assert data["attempts"][1]["response_text"]["sha256"] == SHA_EMPTY


def test_write_trial_output_is_ascii_lf(tmp_path: Path, text_store: store.TextStore) -> None:
    out = store.write_trial(stored_trial(text_store), tmp_path / "runs", text_store)
    for name in ("trial.json", "trial.md"):
        data = (out / name).read_bytes()
        assert data.isascii(), name
        assert b"\r" not in data, name
        assert data.endswith(b"\n") and not data.endswith(b"\n\n"), name


@pytest.mark.parametrize("target", ["dir", "json"])
def test_read_trial_round_trip(tmp_path: Path, text_store: store.TextStore, target: str) -> None:
    trial = stored_trial(text_store)
    out = store.write_trial(trial, tmp_path / "runs", text_store)
    path = out if target == "dir" else out / "trial.json"
    back = store.read_trial(path, text_store)
    assert back == trial
    assert back.attempts[0].response_text == RESPONSES[0]


def test_trial_json_round_trips_the_provenance(tmp_path: Path, text_store: store.TextStore) -> None:
    trial = stored_trial(text_store)
    out = store.write_trial(trial, tmp_path / "runs", text_store)
    data = json.loads((out / "trial.json").read_text(encoding="ascii"))
    assert data["provenance"] == {
        "commit": FIXTURE_COMMIT,
        "dirty": True,
        "device": "fixture-device",
        "sdk": "fixture-sdk",
        "date": FIXTURE_DATE,
    }
    assert store.read_trial(out, text_store).provenance == full_provenance()
    unknown = trial_with(provenance=unknown_provenance())
    other = store.write_trial(unknown, tmp_path / "other-runs", text_store)
    assert store.read_trial(other, text_store).provenance == unknown_provenance()


@pytest.mark.parametrize(
    ("path", "key"),
    [
        pytest.param((), "provenance", id="provenance"),
        pytest.param(("provenance",), "commit", id="provenance-commit"),
        pytest.param(("provenance",), "date", id="provenance-date"),
    ],
)
def test_read_trial_refuses_a_trial_json_without_provenance(
    tmp_path: Path, text_store: store.TextStore, path: tuple[str, ...], key: str
) -> None:
    out = store.write_trial(stored_trial(text_store), tmp_path / "runs", text_store)
    data = json.loads((out / "trial.json").read_text(encoding="ascii"))
    target = data
    for step in path:
        target = target[step]
    del target[key]
    (out / "trial.json").write_bytes(json.dumps(data).encode("ascii"))
    with pytest.raises(ValueError) as info:
        store.read_trial(out, text_store)
    assert key in str(info.value)
    assert str(out / "trial.json") in str(info.value), "the error names the trial.json that lacks the field"


def test_read_trial_missing_response_text(tmp_path: Path, text_store: store.TextStore) -> None:
    out = store.write_trial(stored_trial(text_store), tmp_path / "runs", text_store)
    empty_root = tmp_path / "empty-store"
    empty_root.mkdir()
    with pytest.raises(store.TextNotFoundError):
        store.read_trial(out, store.TextStore(empty_root))


def test_write_trial_without_attempts(tmp_path: Path, text_store: store.TextStore) -> None:
    trial = minimal_trial()
    out = store.write_trial(trial, tmp_path / "runs", text_store)
    assert store.read_trial(out, text_store) == trial


def test_write_trial_rewrites_the_same_trial(tmp_path: Path, text_store: store.TextStore) -> None:
    trial = stored_trial(text_store)
    out = store.write_trial(trial, tmp_path / "runs", text_store)
    assert store.write_trial(trial, tmp_path / "runs", text_store) == out
    assert store.read_trial(out, text_store) == trial


def test_write_trial_refuses_to_overwrite_another_trial(tmp_path: Path, text_store: store.TextStore) -> None:
    run_root = tmp_path / "runs"
    other_id = EXAMPLE_ID.replace("/run01", "/run02")
    first_dir = store.write_trial(minimal_trial(), run_root, text_store)
    # Stand in for a case-insensitive file system, where two ids can share one directory.
    other_dir = store.trial_dir(run_root, other_id)
    other_dir.mkdir(parents=True)
    shutil.copy(first_dir / "trial.json", other_dir / "trial.json")
    before = (other_dir / "trial.json").read_bytes()
    with pytest.raises(ValueError) as info:
        store.write_trial(trial_with(trial_id=other_id), run_root, text_store)
    assert EXAMPLE_ID in str(info.value) and other_id in str(info.value)
    assert (other_dir / "trial.json").read_bytes() == before
    assert not (other_dir / "trial.md").exists()


def missing_prompt_trial() -> record.Trial:
    """Return a one-attempt Trial for EXAMPLE_ID whose prompt text is in no store."""
    attempt = record.Attempt(index=0, prompt_ref=text_ref("never stored\n"), response_text="r\n", stage_reached="S2")
    return minimal_trial([attempt])


def test_write_trial_writes_nothing_when_a_prompt_is_missing(tmp_path: Path, text_store: store.TextStore) -> None:
    run_root = tmp_path / "runs"
    with pytest.raises(store.TextNotFoundError):
        store.write_trial(missing_prompt_trial(), run_root, text_store)
    out = store.trial_dir(run_root, EXAMPLE_ID)
    assert not (out / "trial.json").exists()
    assert not (out / "trial.md").exists()


def test_failed_rewrite_keeps_the_previous_files(tmp_path: Path, text_store: store.TextStore) -> None:
    out = store.write_trial(stored_trial(text_store), tmp_path / "runs", text_store)
    before = {name: (out / name).read_bytes() for name in ("trial.json", "trial.md")}
    with pytest.raises(store.TextNotFoundError):
        store.write_trial(missing_prompt_trial(), tmp_path / "runs", text_store)
    assert sorted(p.name for p in out.iterdir()) == ["trial.json", "trial.md"]
    assert {name: (out / name).read_bytes() for name in ("trial.json", "trial.md")} == before


def test_write_trial_does_not_replace_a_record_it_cannot_read(
    tmp_path: Path, text_store: store.TextStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = store.write_trial(minimal_trial(), tmp_path / "runs", text_store)
    before = {name: (out / name).read_bytes() for name in ("trial.json", "trial.md")}
    trial = stored_trial(text_store)

    def refuse(self: Path, *args: Any, **kwargs: Any) -> str:
        raise PermissionError(13, "in use by another process", str(self))

    monkeypatch.setattr(Path, "read_text", refuse)
    with pytest.raises(PermissionError):
        store.write_trial(trial, tmp_path / "runs", text_store)
    monkeypatch.undo()
    assert {name: (out / name).read_bytes() for name in ("trial.json", "trial.md")} == before


def test_write_trial_ids_differing_by_case_do_not_overwrite(tmp_path: Path, text_store: store.TextStore) -> None:
    (tmp_path / "Probe").mkdir()
    if not (tmp_path / "probe").exists():
        pytest.skip("the file system is case-sensitive, so these ids get separate directories")
    upper_id = EXAMPLE_ID.replace("/entropy/", "/Entropy/")
    upper = trial_with(trial_id=upper_id, bench_item=dataclasses.replace(example_bench(), item="Entropy"))
    out = store.write_trial(minimal_trial(), tmp_path / "runs", text_store)
    with pytest.raises(ValueError):
        store.write_trial(upper, tmp_path / "runs", text_store)
    assert store.read_trial(out, text_store) == minimal_trial()


def test_read_trial_rejects_a_response_path_outside_the_store_layout(
    tmp_path: Path, text_store: store.TextStore
) -> None:
    out = store.write_trial(stored_trial(text_store), tmp_path / "runs", text_store)
    data = json.loads((out / "trial.json").read_text(encoding="ascii"))
    data["attempts"][0]["response_text"]["path"] = "texts/elsewhere.txt"
    (out / "trial.json").write_bytes(json.dumps(data).encode("ascii"))
    with pytest.raises(ValueError) as info:
        store.read_trial(out, text_store)
    assert "texts/elsewhere.txt" in str(info.value)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("[]", id="list"),
        pytest.param("5", id="number"),
        pytest.param('{"attempts": {}}', id="attempts-object"),
        pytest.param('{"attempts": [1]}', id="attempt-number"),
        pytest.param('{"attempts": [{}]}', id="attempt-without-response"),
    ],
)
def test_read_trial_rejects_malformed_json(tmp_path: Path, text_store: store.TextStore, text: str) -> None:
    path = tmp_path / "trial.json"
    path.write_bytes(text.encode("ascii"))
    with pytest.raises(ValueError):
        store.read_trial(path, text_store)
