"""Tests for the adversary interface: what an adversary declares, sees, and proposes (task P16.2).

Bible: Component Interfaces (the Agent row; contract rules; capabilities),
Agents And Judges (the Adversary role; the Agent interface), Adversarial
Setups (items 1 and 3; the adversary rule), Harness Contract (held-out
inputs never appear in a prompt), Agent Rules 3, 6, and 14. Plan:
plans/p16-adversarial.md, P16.2, and its Constraints (a proposal is data,
never code; an adversary never sees a held-out input, an item's seed, or a
reference output).

The contract these tests fix:

- lassi.core.capabilities.PROPOSES_INPUTS is "proposes_inputs", the
  capability of an Agent that is an adversary. lassi.core.capabilities
  Adversary is the Protocol of what such an Agent provides, as
  HostComputeGuard and OutputFileOracle are for their capabilities: one
  method, propose(view), documented and type hinted, that takes an
  AdversaryView and returns its Proposals. The Agent Protocol's docstring
  in lassi.core.interfaces states both, generically. No Protocol is added
  to lassi.core.interfaces, which keeps the bible's thirteen
  (tests/core/test_interfaces.py).
- lassi.core.interfaces.AdversaryView is a frozen dataclass whose fields
  are exactly contract (the item's InputContract, lassi.harness), source
  (the source program's files, path -> text), candidate (the candidate's
  files, path -> text), budget_left (the input sets the adversary may still
  propose, an int of at least 0), and outcomes (earlier proposals'
  outcomes, ProposalOutcome items). It refuses anything else in those
  fields, so a held-out input (a LassiArray or file bytes), an item's seed,
  or a reference output has no place in it. The contract is exactly an
  InputContract whose inputs are a non-empty tuple of exactly ContractInput,
  each field of each of the type input_contract gives it; paths and texts
  are exactly str.
- lassi.core.interfaces.Proposal is a frozen dataclass of data only: its
  `arrays`, a tuple of LassiArray, each a named array of element bytes
  whose name and dtype are exactly str and whose shape holds exactly ints.
  It defines no public method, refuses anything else (a program text,
  bytes, a callable, a code object, a callable as an array's name), and
  accepts arrays outside any contract, since check_inputs refuses those
  before any run and the stage counts the refusal (P16.1, P16.5).
- lassi.core.interfaces.ProposalOutcome is a frozen dataclass holding the
  proposal and `code`, a short token (lower-case letters, digits, and
  hyphens, starting with a letter, at most 64 characters), never free
  text: a CONTRACT_REASONS reason for a proposal check_inputs refused, or
  the stage's own code for one that ran (P16.5 fixes those). No field of it
  but the proposal carries arrays, bytes, paths, runs, or output
  statistics.
- AdversaryView, Proposal, and ProposalOutcome take no subclass, so none
  can add a field (a seed, a held-out array) that their checks never see.

Every array and program text here is SYNTHETIC; no value in this module is
a measurement.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
import struct
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lassi.core import capabilities, interfaces
from lassi.core.interfaces import RunResult
from lassi.harness import InputContract, InputContractError, LassiArray, check_inputs, input_contract, write_array
from lassi.harness.contract import CONTRACT_REASONS, ContractInput

CAPABILITY = "proposes_inputs"
VIEW_FIELDS = {"contract", "source", "candidate", "budget_left", "outcomes"}

# A SYNTHETIC held-out input spec (generate_inputs entries) and the contract read from it.
SPEC = (
    {"name": "x", "dtype": "f32", "shape": [4], "dist": "uniform", "lo": -1.0, "hi": 1.0},
    {"name": "n", "dtype": "i32", "shape": [2], "dist": "integers", "lo": 0, "hi": 16},
)
CONTRACT = input_contract(SPEC)
SOURCE = {"main.cpp": "// SYNTHETIC source program\nint main() { return 0; }\n"}
CANDIDATE = {"main.cpp": "// SYNTHETIC candidate\nint main() { return 0; }\n"}

# Words no field of the view, or of anything it holds, may be named after: a held-out input, the item's seed, and a
# reference output have no place in it.
FORBIDDEN_WORDS = ("seed", "held", "reference", "expected", "golden")
# Types no field of the view or of an outcome may carry, apart from the proposal's own arrays: run results, output
# statistics, files on disk, raw bytes, or the record's and the registry's types.
FORBIDDEN_TYPES = ("RunResult", "OutputStats", "Alignment", "Path", "bytes", "Trial", "Attempt", "SuiteItem")


def need(module: ModuleType, name: str) -> Any:
    """Return `module`'s attribute `name`, failing with the task's name when it does not exist yet."""
    assert hasattr(module, name), f"{module.__name__} has no {name} (task P16.2)"
    return getattr(module, name)


def floats(values: list[float]) -> bytes:
    """Return f32 element bytes, little-endian."""
    return struct.pack(f"<{len(values)}f", *values)


def ints(values: list[int]) -> bytes:
    """Return i32 element bytes, little-endian."""
    return struct.pack(f"<{len(values)}i", *values)


X_INSIDE = LassiArray("x", "f32", (4,), floats([-1.0, -0.5, 0.0, 0.5]))
X_AT_HI = LassiArray("x", "f32", (4,), floats([-1.0, -0.5, 0.0, 1.0]))  # 1.0 is the contract's hi, outside it
N_INSIDE = LassiArray("n", "i32", (2,), ints([0, 15]))
X_SPEC = CONTRACT.inputs[0]
SEEDED_ENTRY = {"seed": 1234, **SPEC[0]}  # a held-out spec entry with the item's seed, offered as a contract input


@dataclasses.dataclass(frozen=True)
class SeededContract(InputContract):
    """A SYNTHETIC InputContract subclass that would carry the item's seed and a held-out array to an adversary."""

    seed: int = 1234
    held: tuple[LassiArray, ...] = (X_INSIDE,)


@dataclasses.dataclass(frozen=True)
class SeededInput(ContractInput):
    """A SYNTHETIC ContractInput subclass that would carry the item's seed to an adversary."""

    seed: int = 1234


class TaggedText(str):
    """A SYNTHETIC str subclass whose instances could carry more than their text."""


def contract_with(**changes: Any) -> InputContract:
    """Return CONTRACT with its first input's fields changed by `changes` (dataclasses.replace)."""
    return InputContract((dataclasses.replace(X_SPEC, **changes), *CONTRACT.inputs[1:]))


def proposal(*arrays: LassiArray) -> Any:
    """Return a Proposal of `arrays`."""
    return need(interfaces, "Proposal")(arrays=tuple(arrays))


def view_kwargs(**changes: Any) -> dict[str, Any]:
    """Return a valid AdversaryView's keyword arguments with `changes` applied."""
    kwargs: dict[str, Any] = {
        "contract": CONTRACT,
        "source": dict(SOURCE),
        "candidate": dict(CANDIDATE),
        "budget_left": 3,
        "outcomes": (),
    }
    kwargs.update(changes)
    return kwargs


def type_text(spec: dataclasses.Field[Any]) -> str:
    """Return a dataclass field's annotation as text, whether it is a string or an evaluated type."""
    return spec.type if isinstance(spec.type, str) else str(spec.type)


def mentions(text: str, name: str) -> bool:
    """Return True when `text` names `name` as a whole word (Proposal does not match ProposalOutcome)."""
    return re.search(rf"\b{re.escape(name)}\b", text) is not None


def carried_types(specs: Mapping[str, str]) -> dict[str, list[str]]:
    """Return, for each field annotation in `specs` that names one, the forbidden types or LassiArray it names."""
    found = {name: [kind for kind in (*FORBIDDEN_TYPES, "LassiArray") if mentions(text, kind)] for name, text in
             specs.items()}
    return {name: kinds for name, kinds in found.items() if kinds}


def field_names_under(cls: type, known: Mapping[str, type], seen: set[str] | None = None) -> list[str]:
    """Return every field name of `cls` and of each known dataclass its annotations name, recursively."""
    seen = set() if seen is None else seen
    if cls.__name__ in seen:
        return []
    seen.add(cls.__name__)
    names: list[str] = []
    for spec in dataclasses.fields(cls):
        names.append(f"{cls.__name__}.{spec.name}")
        for label, inner in known.items():
            if mentions(type_text(spec), label):
                names += field_names_under(inner, known, seen)
    return names


# ---------------------------------------------------------------------------
# What an adversary declares and provides


def test_the_capability_is_proposes_inputs() -> None:
    assert need(capabilities, "PROPOSES_INPUTS") == CAPABILITY


def test_the_adversary_protocol_provides_one_documented_propose_method() -> None:
    adversary = need(capabilities, "Adversary")
    assert inspect.isclass(adversary) and getattr(adversary, "_is_protocol", False), "Adversary is not a Protocol"
    assert adversary.__doc__ and CAPABILITY in adversary.__doc__, "Adversary's docstring names no proposes_inputs"
    propose = vars(adversary).get("propose")
    assert inspect.isfunction(propose), "Adversary defines no propose()"
    assert propose.__doc__ and propose.__doc__.strip(), "Adversary.propose has no docstring"
    parameters = list(inspect.signature(propose).parameters.values())
    assert [parameter.name for parameter in parameters] == ["self", "view"]
    assert mentions(str(parameters[1].annotation), "AdversaryView"), parameters[1].annotation
    assert mentions(str(propose.__annotations__.get("return", "")), "Proposal"), propose.__annotations__


def test_the_adversary_protocol_is_not_a_fourteenth_interface() -> None:
    adversary = need(capabilities, "Adversary")
    assert adversary.__module__ == "lassi.core.capabilities"
    assert not hasattr(interfaces, "Adversary"), "lassi.core.interfaces keeps the bible's thirteen Protocols"


def test_the_agent_interface_states_what_an_adversary_declares_and_provides() -> None:
    doc = interfaces.Agent.__doc__ or ""
    assert CAPABILITY in doc, "the Agent docstring names no proposes_inputs"
    assert "propose(" in doc, "the Agent docstring names no propose method"


# ---------------------------------------------------------------------------
# The view: exactly what an adversary sees


def test_the_view_is_a_frozen_dataclass_of_exactly_the_five_fields() -> None:
    view = need(interfaces, "AdversaryView")
    assert dataclasses.is_dataclass(view)
    assert view.__dataclass_params__.frozen, "AdversaryView is not frozen"
    assert {spec.name for spec in dataclasses.fields(view)} == VIEW_FIELDS


def test_the_view_fields_hold_a_contract_text_files_a_count_and_outcomes() -> None:
    specs = {spec.name: type_text(spec) for spec in dataclasses.fields(need(interfaces, "AdversaryView"))}
    assert mentions(specs["contract"], "InputContract"), specs["contract"]
    for name in ("source", "candidate"):
        assert mentions(specs[name], "str"), specs[name]
    assert mentions(specs["budget_left"], "int"), specs["budget_left"]
    assert mentions(specs["outcomes"], "ProposalOutcome"), specs["outcomes"]
    carried = carried_types(specs)
    assert not carried, f"view fields carry {carried}"


def test_a_view_holds_what_it_was_given() -> None:
    outcome = need(interfaces, "ProposalOutcome")(proposal=proposal(X_AT_HI, N_INSIDE), code="out-of-range")
    view = need(interfaces, "AdversaryView")(**view_kwargs(outcomes=(outcome,)))
    assert view.contract == CONTRACT
    assert dict(view.source) == SOURCE
    assert dict(view.candidate) == CANDIDATE
    assert view.budget_left == 3
    assert tuple(view.outcomes) == (outcome,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.budget_left = 4  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("contract", list(SPEC), id="contract-the-held-out-spec"),
        pytest.param("contract", X_INSIDE, id="contract-an-array"),
        pytest.param("contract", None, id="contract-none"),
        pytest.param("source", {"main.cpp": b"int main() { return 0; }\n"}, id="source-bytes"),
        pytest.param("source", {"x.bin": X_INSIDE}, id="source-an-array"),
        pytest.param("source", {1: "int main() { return 0; }\n"}, id="source-a-non-str-path"),
        pytest.param("source", "int main() { return 0; }\n", id="source-bare-text"),
        pytest.param("candidate", {"main.cpp": b"\x7fELF"}, id="candidate-bytes"),
        pytest.param("candidate", {"out.bin": X_INSIDE}, id="candidate-an-array"),
        pytest.param("budget_left", -1, id="budget-negative"),
        pytest.param("budget_left", True, id="budget-a-bool"),
        pytest.param("budget_left", 2.0, id="budget-a-float"),
        pytest.param("outcomes", (RunResult(exit_code=0, hang=False, stdout="", stderr=""),), id="outcome-a-run"),
        pytest.param("outcomes", ({"code": "out-of-range"},), id="outcome-a-mapping"),
        pytest.param("outcomes", (X_INSIDE,), id="outcome-an-array"),
    ],
)
def test_a_view_refuses_anything_else_in_its_fields(field: str, value: Any) -> None:
    view = need(interfaces, "AdversaryView")
    with pytest.raises((TypeError, ValueError)) as info:
        view(**view_kwargs(**{field: value}))
    assert field in str(info.value), f"the refusal does not name {field}: {info.value}"


@pytest.mark.parametrize(
    "contract",
    [
        pytest.param(SeededContract(CONTRACT.inputs), id="a-subclass-with-a-seed-and-held-arrays"),
        pytest.param(InputContract((X_INSIDE,)), id="inputs-a-held-out-array"),
        pytest.param(InputContract((SEEDED_ENTRY,)), id="inputs-a-seeded-spec-entry"),  # type: ignore[arg-type]
        pytest.param(InputContract(list(CONTRACT.inputs)), id="inputs-a-list"),  # type: ignore[arg-type]
        pytest.param(InputContract(()), id="inputs-empty"),
        pytest.param(InputContract((SeededInput(**dataclasses.asdict(X_SPEC)),)), id="input-a-subclass-with-a-seed"),
        pytest.param(contract_with(name=X_INSIDE), id="input-name-an-array"),
        pytest.param(contract_with(name=TaggedText("x")), id="input-name-a-str-subclass"),
        pytest.param(contract_with(dtype=None), id="input-dtype-none"),
        pytest.param(contract_with(shape=[4]), id="input-shape-a-list"),
        pytest.param(contract_with(shape=(4.0,)), id="input-shape-a-float-dim"),
        pytest.param(contract_with(shape=(True,)), id="input-shape-a-bool-dim"),
        pytest.param(contract_with(lo=X_INSIDE), id="input-lo-an-array"),
        pytest.param(contract_with(hi="1.0"), id="input-hi-text"),
        pytest.param(contract_with(lo=False), id="input-lo-a-bool"),
        pytest.param(contract_with(allow_nan=1), id="input-allow-nan-an-int"),
        pytest.param(contract_with(allow_inf=None), id="input-allow-inf-none"),
    ],
)
def test_a_view_refuses_a_contract_that_is_not_exactly_an_input_contract_of_contract_inputs(contract: Any) -> None:
    view = need(interfaces, "AdversaryView")
    with pytest.raises((TypeError, ValueError)) as info:
        view(**view_kwargs(contract=contract))
    assert "contract" in str(info.value), f"the refusal does not name contract: {info.value}"


def test_a_view_takes_a_contract_with_a_declared_range_and_flags() -> None:
    # input_contract(spec, declared=...) gives float bounds and true flags; the view's checks take all of them.
    declared = [
        {"name": "x", "dtype": "f32", "lo": -2.0, "hi": 2.0, "allow_nan": True, "allow_inf": True},
        {"name": "n", "dtype": "i32", "lo": -4, "hi": 32},
    ]
    contract = input_contract(SPEC, declared)
    assert need(interfaces, "AdversaryView")(**view_kwargs(contract=contract)).contract == contract


@pytest.mark.parametrize(
    ("field", "files"),
    [
        pytest.param("source", {TaggedText("main.cpp"): "int main() { return 0; }\n"}, id="source-a-str-subclass-path"),
        pytest.param("candidate", {"main.cpp": TaggedText("int main() { return 0; }\n")}, id="candidate-str-subclass"),
    ],
)
def test_a_view_takes_paths_and_texts_only_as_exactly_str(field: str, files: dict[Any, Any]) -> None:
    view = need(interfaces, "AdversaryView")
    with pytest.raises(TypeError) as info:
        view(**view_kwargs(**{field: files}))
    assert field in str(info.value), f"the refusal does not name {field}: {info.value}"


def test_nothing_the_view_holds_is_named_after_a_held_out_input_a_seed_or_a_reference() -> None:
    known = {
        "AdversaryView": need(interfaces, "AdversaryView"),
        "ProposalOutcome": need(interfaces, "ProposalOutcome"),
        "Proposal": need(interfaces, "Proposal"),
        "InputContract": InputContract,
        "ContractInput": ContractInput,
        "LassiArray": LassiArray,
    }
    names = field_names_under(known["AdversaryView"], known)
    reached = {name.split(".")[0] for name in names}
    assert reached == set(known), f"the walk reached {sorted(reached)}"
    found = [name for name in names if any(word in name.split(".")[1].lower() for word in FORBIDDEN_WORDS)]
    assert not found, f"fields named after what an adversary never sees: {found}"


# ---------------------------------------------------------------------------
# A proposal is data, never code


def test_a_proposal_is_a_frozen_tuple_of_named_arrays() -> None:
    cls = need(interfaces, "Proposal")
    assert dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen, "Proposal is not a frozen dataclass"
    specs = {spec.name: type_text(spec) for spec in dataclasses.fields(cls)}
    assert "arrays" in specs and mentions(specs["arrays"], "LassiArray"), specs
    made = proposal(X_INSIDE, N_INSIDE)
    assert made.arrays == (X_INSIDE, N_INSIDE)
    assert isinstance(made.arrays, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        made.arrays = ()  # type: ignore[misc]


def test_a_proposal_defines_no_public_method() -> None:
    cls = need(interfaces, "Proposal")
    methods = [name for name, value in vars(cls).items() if not name.startswith("_") and callable(value)]
    assert methods == [], f"Proposal defines methods {methods}; a proposal is data"


def _function() -> None:
    """A SYNTHETIC function, offered where an array belongs."""


@pytest.mark.parametrize(
    "item",
    [
        pytest.param("int main() { return 0; }\n", id="program-text"),
        pytest.param(b"\x00\x00\x80\x3f", id="raw-bytes"),
        pytest.param(lambda: X_INSIDE, id="a-lambda"),
        pytest.param(_function, id="a-function"),
        pytest.param(compile("1 + 1", "<synthetic>", "eval"), id="a-code-object"),
        pytest.param({"name": "x", "dtype": "f32", "shape": [4], "data": b""}, id="a-mapping"),
        pytest.param(None, id="none"),
    ],
)
def test_a_proposal_refuses_anything_but_arrays(item: object) -> None:
    cls = need(interfaces, "Proposal")
    with pytest.raises((TypeError, ValueError)):
        cls(arrays=(X_INSIDE, item))


@pytest.mark.parametrize("arrays", [lambda: (X_INSIDE,), "x", X_INSIDE], ids=["a-callable", "text", "one-array"])
def test_a_proposal_refuses_arrays_that_are_not_a_sequence_of_arrays(arrays: object) -> None:
    cls = need(interfaces, "Proposal")
    with pytest.raises((TypeError, ValueError)):
        cls(arrays=arrays)


FOUR_ZEROS = b"\0" * 16


@pytest.mark.parametrize(
    "array",
    [
        pytest.param(LassiArray(lambda: 1, "f32", (4,), FOUR_ZEROS), id="name-a-callable"),  # type: ignore[arg-type]
        pytest.param(LassiArray(TaggedText("x"), "f32", (4,), FOUR_ZEROS), id="name-a-str-subclass"),
        pytest.param(LassiArray("x", None, (4,), FOUR_ZEROS), id="dtype-none"),  # type: ignore[arg-type]
        pytest.param(LassiArray("x", _function, (4,), FOUR_ZEROS), id="dtype-a-function"),  # type: ignore[arg-type]
        pytest.param(LassiArray("x", "f32", (4.0,), FOUR_ZEROS), id="shape-a-float-dim"),  # type: ignore[arg-type]
        pytest.param(LassiArray("x", "f32", (True,), FOUR_ZEROS), id="shape-a-bool-dim"),
        pytest.param(LassiArray("x", "f32", ("4",), FOUR_ZEROS), id="shape-a-text-dim"),  # type: ignore[arg-type]
    ],
)
def test_a_proposal_refuses_an_array_whose_name_dtype_or_shape_is_not_data(array: LassiArray) -> None:
    cls = need(interfaces, "Proposal")
    with pytest.raises(TypeError) as info:
        cls(arrays=(N_INSIDE, array))
    assert "Proposal.arrays[1]" in str(info.value), f"the refusal does not name the array: {info.value}"


@pytest.mark.parametrize(
    "arrays",
    [
        pytest.param((), id="no-array"),
        pytest.param((X_INSIDE, X_INSIDE, N_INSIDE), id="a-repeated-name"),
        pytest.param((X_AT_HI, N_INSIDE), id="an-element-at-hi"),
        pytest.param((LassiArray("x", "f32", (2, 2), X_INSIDE.data), N_INSIDE), id="another-shape"),
        pytest.param((X_INSIDE, LassiArray("extra", "u8", (1,), b"\x00"), N_INSIDE), id="an-extra-input"),
    ],
)
def test_a_proposal_outside_the_contract_is_still_a_proposal(arrays: tuple[LassiArray, ...]) -> None:
    # check_inputs refuses these before any run and the stage counts the refusal (P16.1, P16.5); the proposal
    # itself only holds data, so it hides no refusal.
    assert proposal(*arrays).arrays == arrays


# ---------------------------------------------------------------------------
# Earlier proposals' outcomes


def test_an_outcome_holds_the_proposal_and_a_code() -> None:
    cls = need(interfaces, "ProposalOutcome")
    assert dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen, "ProposalOutcome is not frozen"
    made = cls(proposal=proposal(X_AT_HI, N_INSIDE), code="out-of-range")
    assert made.proposal == proposal(X_AT_HI, N_INSIDE)
    assert made.code == "out-of-range"
    with pytest.raises(dataclasses.FrozenInstanceError):
        made.code = "nan"  # type: ignore[misc]


def test_an_outcome_carries_no_run_and_no_output_beside_its_proposal() -> None:
    specs = {spec.name: type_text(spec) for spec in dataclasses.fields(need(interfaces, "ProposalOutcome"))}
    assert mentions(specs.pop("proposal", ""), "Proposal"), "ProposalOutcome has no proposal field"
    assert "code" in specs, "ProposalOutcome has no code field"
    carried = carried_types(specs)
    assert not carried, f"outcome fields carry {carried}"


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        pytest.param({"proposal": (X_INSIDE, N_INSIDE)}, "proposal", id="arrays-not-a-proposal"),
        pytest.param({"proposal": X_INSIDE}, "proposal", id="an-array-not-a-proposal"),
        pytest.param({"code": ""}, "code", id="an-empty-code"),
        pytest.param({"code": None}, "code", id="no-code"),
        pytest.param({"code": 3}, "code", id="a-number-code"),
    ],
)
def test_an_outcome_refuses_anything_else(changes: dict[str, Any], field: str) -> None:
    cls = need(interfaces, "ProposalOutcome")
    kwargs: dict[str, Any] = {"proposal": proposal(X_INSIDE, N_INSIDE), "code": "out-of-range", **changes}
    with pytest.raises((TypeError, ValueError)) as info:
        cls(**kwargs)
    assert field in str(info.value), f"the refusal does not name {field}: {info.value}"


@pytest.mark.parametrize(
    "code",
    [
        pytest.param("diverged; reference stdout: 0.125 0.25 seed=1234", id="free-text"),
        pytest.param("out-of-range\n0.125", id="a-newline"),
        pytest.param("out-of-range\n", id="a-trailing-newline"),
        pytest.param("diverged 0.125", id="a-space"),
        pytest.param("Out-Of-Range", id="upper-case"),
        pytest.param("out_of_range", id="an-underscore"),
        pytest.param("-diverged", id="a-leading-hyphen"),
        pytest.param("1diverged", id="a-leading-digit"),
        pytest.param("a" * 65, id="65-characters"),
        pytest.param("seed=1234", id="an-equals-sign"),
    ],
)
def test_an_outcome_code_is_a_short_token_never_free_text(code: str) -> None:
    # Free text could carry a run's output values or the item's seed to an adversary; a token cannot hold them.
    cls = need(interfaces, "ProposalOutcome")
    with pytest.raises(ValueError) as info:
        cls(proposal=proposal(X_INSIDE, N_INSIDE), code=code)
    assert "code" in str(info.value), f"the refusal does not name code: {info.value}"


def test_an_outcome_code_is_exactly_a_str() -> None:
    cls = need(interfaces, "ProposalOutcome")
    with pytest.raises(TypeError) as info:
        cls(proposal=proposal(X_INSIDE, N_INSIDE), code=TaggedText("out-of-range"))
    assert "code" in str(info.value), f"the refusal does not name code: {info.value}"


@pytest.mark.parametrize("code", [*CONTRACT_REASONS, "a", "a" * 64, "diverged", "contract-finding", "run-2"])
def test_an_outcome_takes_every_contract_reason_and_other_tokens(code: str) -> None:
    assert need(interfaces, "ProposalOutcome")(proposal=proposal(X_INSIDE, N_INSIDE), code=code).code == code


# ---------------------------------------------------------------------------
# No subclass can add a field the checks never see


@pytest.mark.parametrize("name", ["AdversaryView", "Proposal", "ProposalOutcome"])
def test_the_view_proposal_and_outcome_take_no_subclass(name: str) -> None:
    base = need(interfaces, name)
    with pytest.raises(TypeError) as info:
        dataclasses.dataclass(frozen=True)(type(f"Seeded{name}", (base,), {"__annotations__": {"seed": int}}))
    assert name in str(info.value), f"the refusal does not name {name}: {info.value}"


# ---------------------------------------------------------------------------
# Round trip: a proposal is written as lassi_io files and checked, and nothing runs


class EdgeAdversary:
    """A SYNTHETIC adversary: proposes the contract's inside set, then the same with one element at hi."""

    name = "synthetic-edge"
    capabilities = frozenset({CAPABILITY})

    def __init__(self) -> None:
        """Start with no view seen."""
        self.views: list[Any] = []

    def propose(self, view: Any) -> tuple[Any, ...]:
        """Return two proposals for `view`; read only its contract."""
        self.views.append(view)
        assert view.contract.names == ("x", "n")
        return (proposal(X_INSIDE, N_INSIDE), proposal(X_AT_HI, N_INSIDE))


def written(directory: Path, made: Any) -> list[Path]:
    """Write each array of `made` as a lassi_io file under `directory` and return the paths."""
    directory.mkdir()
    paths = []
    for array in made.arrays:
        path = directory / f"{array.name}.bin"
        write_array(path, array.name, array.dtype, array.shape, array.data)
        paths.append(path)
    return paths


def test_proposals_are_written_as_files_and_checked_against_the_view_contract(tmp_path: Path) -> None:
    adversary = EdgeAdversary()
    view = need(interfaces, "AdversaryView")(**view_kwargs())
    inside, outside = adversary.propose(view)
    assert adversary.views == [view]
    assert check_inputs(view.contract, written(tmp_path / "inside", inside)) is None
    with pytest.raises(InputContractError) as info:
        check_inputs(view.contract, written(tmp_path / "outside", outside))
    assert (info.value.reason, info.value.name) == ("out-of-range", "x")


def test_the_edge_adversary_fits_the_protocol_shape() -> None:
    adversary = need(capabilities, "Adversary")
    expected: Callable[..., Any] = vars(adversary)["propose"]
    assert list(inspect.signature(EdgeAdversary.propose).parameters) == list(inspect.signature(expected).parameters)
    assert capabilities.declares(EdgeAdversary(), need(capabilities, "PROPOSES_INPUTS"))
