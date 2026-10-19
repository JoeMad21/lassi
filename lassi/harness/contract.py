"""Input contracts: what an item's inputs may be, and the check that refuses a proposed input set outside one.

Bible: Adversarial Setups (the adversary rule: inputs that violate the input
contract, by type, range, or size, are rejected before execution, because a
divergence on invalid input is not a bug), Harness Contract (lassi_io files;
held-out input files), Benchmark Suites (held-out inputs; the element cap),
Risks And Questions (out-of-contract inputs from an adversary, answered by
an input-contract check before execution).

Rules [DESIGN, task P16.1]:

- An item's contract names each of its inputs with a dtype and a shape and
  bounds every element to [lo, hi). input_contract(spec) reads an item's
  held-out input spec (generate_inputs entries) as one: each entry's name,
  dtype, shape, lo, and hi, with no NaN and no infinity allowed. A spec
  names at least one input.
- input_contract(spec, declared=...) takes the value of the optional
  manifest key `contract` (lassi.bench.registry): a list of exactly one
  entry per spec input, each a mapping with the keys name (a str), dtype,
  lo, and hi and the optional allow_nan and allow_inf (true or false, false
  when left out). Its dtype must be the input's. Its range may widen the
  input's, never narrow it: it must hold every dtype value of the spec's
  [lo, hi), the values generate_inputs draws from, so an item's own
  held-out inputs always pass its contract. Its lo and hi then replace the
  input's range, and a flag set true lets a NaN, or an infinity of either
  sign, through whatever the range; a flag set true on an integer dtype is
  refused. The names, dtypes, and shapes stay the spec's, and neither
  argument is changed.
- Bounds follow generate_inputs' bound rules, checked by the same code
  (lassi_io _float_bounds and _int_bounds): for a float dtype, finite
  numbers (an int only when it is exactly a double) inside the dtype's
  finite range, its largest finite value included, with lo below hi, hi - lo
  finite in double, and some dtype value in [lo, hi); for an integer dtype,
  ints (not bools) with lo below hi and [lo, hi) inside the dtype's range. A
  spec input holds at most MAX_INPUT_ELEMENTS elements, counted as lassi_io
  element_count counts them (a zero dim gives 0). input_contract raises
  ValueError, its message starting with `where`, for anything else.
- check_inputs(contract, files) reads a proposed input set, lassi_io files
  in any order, and matches arrays by the name in each file's header (as
  binary_io matches outputs), never by file name. It returns None when the
  set is inside the contract and otherwise raises InputContractError for
  the first refusal it meets, its `reason` one of CONTRACT_REASONS. It
  checks in three stages, in this order:
  1. File by file in the order given, each in full before the next is
     opened: unreadable when the file cannot be opened or read, or its
     header breaks read_array's rules (checked in file order);
     too-many-elements when its header's dims hold more than
     MAX_INPUT_ELEMENTS elements, whatever the contract's shape; unreadable
     when its data is not the size its dims give. The data is read only
     after these checks, and of the header no more than the longest header
     of its rank, so no file is read past MAX_INPUT_ELEMENTS elements.
  2. The arrays' names, file by file in the order given: repeated-input for
     a name an earlier file holds, extra-input for one the contract lacks;
     then missing-input for the first contract input, in the contract's
     order, that no file holds.
  3. Input by input in the contract's order: dtype, then shape (a scalar,
     shape (), is not a one-element array, shape (1,)), then over the whole
     array: nan when any element is a NaN the contract does not allow, else
     infinity when any is an infinity, either sign, it does not allow, else
     out-of-range when a finite element lies outside [lo, hi); the message
     names the first such element. Values compare as numbers: lo is inside
     and hi is not, -0 equals +0, and each smallest subnormal lies on its
     sign's side of 0. A NaN or infinity never reads as out-of-range.
  CONTRACT_REASONS lists the reasons in this order (stage 1 gives its two
  reasons file by file). The error's `name` is the input's name (None for
  an unreadable file) and its `path` the file (None for a missing input);
  the message starts with the reason and names each one present. The check
  writes, changes, and removes nothing.

Known cases, not complete: a contract bounds each element alone; relations
between elements or between inputs (a sorted array, an index below another
input's length) are not expressible.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lassi.harness.lassi_io import (
    _BITS_CODE,
    _FLOATS,
    _INTS,
    DTYPES,
    LassiArray,
    LassiIOError,
    _bits_value,
    _float_bounds,
    _int_bounds,
    _key_range,
    _read_data,
    _read_header,
    _shown,
    check_input_spec,
    element_count,
)

# The most elements one input may hold (PHASE-NOTES P4, the P4.3 review item): 2**20 = 1048576, about 2.5 times the
# largest Tier A input (a 640 x 640 matrix, 409600 elements); the pure-Python generator draws each one. load_suite caps
# every held-out input at it (lassi.bench.registry), and check_inputs refuses a proposed file past it.
MAX_INPUT_ELEMENTS = 1 << 20
# The refusal reasons of check_inputs, in the order of its stages (the module docstring gives the exact order).
CONTRACT_REASONS = (
    "unreadable",
    "too-many-elements",
    "repeated-input",
    "extra-input",
    "missing-input",
    "dtype",
    "shape",
    "nan",
    "infinity",
    "out-of-range",
)
_DECLARED_KEYS = frozenset({"name", "dtype", "lo", "hi"})
_FLAGS = ("allow_nan", "allow_inf")


class InputContractError(ValueError):
    """A proposed input set outside its contract: `reason` (CONTRACT_REASONS), the input's `name`, the file's `path`.

    `name` is None for an unreadable file, and `path` for a missing input.
    The message starts with the reason; `detail` is the rest of it. The
    error pickles with its attributes, so it can cross a process boundary.
    """

    def __init__(self, reason: str, detail: str, *, name: str | None = None, path: str | None = None) -> None:
        """Build the refusal; its message is '<reason>: <detail>'."""
        if reason not in CONTRACT_REASONS:
            raise ValueError(f"unknown contract refusal reason {reason!r}")
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail
        self.name = name
        self.path = path

    def __reduce__(self) -> tuple[Any, ...]:
        """Rebuild from the reason and detail, then restore name and path (both keyword-only)."""
        return type(self), (self.reason, self.detail), {"name": self.name, "path": self.path}


@dataclass(frozen=True)
class ContractInput:
    """One input of a contract: its name, dtype, shape, and range [lo, hi), and whether a NaN or an infinity may pass.

    lo and hi are floats for a float dtype and ints for an integer dtype.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    lo: float | int
    hi: float | int
    allow_nan: bool = False
    allow_inf: bool = False


@dataclass(frozen=True)
class InputContract:
    """An item's input contract: one ContractInput per input, in the spec's order. Contracts compare by value."""

    inputs: tuple[ContractInput, ...]

    @property
    def names(self) -> tuple[str, ...]:
        """Return the input names in the spec's order."""
        return tuple(spec.name for spec in self.inputs)


# ---------------------------------------------------------------------------
# Building a contract


def input_contract(
    spec: Sequence[Mapping[str, Any]], declared: Sequence[Mapping[str, Any]] | None = None, *, where: str = "contract"
) -> InputContract:
    """Return the input contract of a held-out input spec, read as one or with a declared `contract` key.

    `spec` holds generate_inputs entries; `declared`, when given, is the
    manifest key's value (rules in the module docstring). Raises ValueError,
    its message starting with `where`, for a spec generate_inputs would
    refuse, an empty spec, an input past MAX_INPUT_ELEMENTS, or a malformed
    declaration, one that narrows an input's range among them. Neither
    argument is changed.
    """
    try:
        check_input_spec(spec, where)
    except LassiIOError as error:
        raise ValueError(str(error)) from None
    if not spec:
        raise ValueError(f"{where}: a contract names at least one input; an item without inputs has none")
    dtypes = {entry["name"]: entry["dtype"] for entry in spec}
    entries = {} if declared is None else _declared_entries(declared, dtypes, where)
    return InputContract(tuple(_contract_input(entry, entries.get(entry["name"]), where) for entry in spec))


def _contract_input(entry: Mapping[str, Any], declared: Mapping[str, Any] | None, where: str) -> ContractInput:
    """Return a checked spec entry's ContractInput: its own range, or its declared entry's range and flags."""
    name, dtype, shape = entry["name"], entry["dtype"], tuple(entry["shape"])
    at = f"{where}: input {name!r}"
    if element_count(shape, MAX_INPUT_ELEMENTS) is None:
        raise ValueError(f"{at} holds more than {MAX_INPUT_ELEMENTS} elements, the input cap")
    lo, hi, first, last = _bounds(at, dtype, entry["lo"], entry["hi"])
    if declared is None:
        return ContractInput(name, dtype, shape, lo, hi)
    wide_lo, wide_hi, wide_first, wide_last = _bounds(at, dtype, declared["lo"], declared["hi"])
    if wide_first > first or wide_last < last:
        raise ValueError(
            f"{at}: the declared range [{_shown(wide_lo)}, {_shown(wide_hi)}) does not hold every {dtype} value of "
            f"the spec's range [{_shown(lo)}, {_shown(hi)}); a contract may widen an input's range, never narrow it"
        )
    flags = [declared.get(flag, False) for flag in _FLAGS]
    if any(flags) and dtype not in _FLOATS:
        raise ValueError(f"{at}: allow_nan and allow_inf apply to a float dtype, not {dtype}")
    return ContractInput(name, dtype, shape, wide_lo, wide_hi, *flags)


def _declared_entries(declared: object, dtypes: Mapping[str, str], where: str) -> dict[str, Mapping[str, Any]]:
    """Check a declared contract's entries against the spec's names and dtypes; return them by input name."""
    if isinstance(declared, (str, bytes, Mapping)) or not isinstance(declared, Sequence):
        kind = type(declared).__name__
        raise ValueError(f"{where}: a declared contract is a list of one entry per input, not a {kind}")
    found: dict[str, Mapping[str, Any]] = {}
    for index, entry in enumerate(declared):
        at = f"{where}: entry {index}"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{at} is not a mapping")
        keys = set(entry)
        missing = sorted(_DECLARED_KEYS - keys)
        extra = sorted(_shown(key) for key in keys - _DECLARED_KEYS - set(_FLAGS))
        if missing or extra:
            raise ValueError(
                f"{at}: the keys are name, dtype, lo, and hi, and optionally allow_nan and allow_inf; "
                f"missing {missing}, extra {extra}"
            )
        name = entry["name"]
        if not isinstance(name, str) or name not in dtypes:
            raise ValueError(f"{at}: {_shown(name)} names no input; the inputs are {', '.join(dtypes)}")
        if name in found:
            raise ValueError(f"{at}: input {name!r} has a second entry; each input has exactly one")
        if not isinstance(entry["dtype"], str) or entry["dtype"] != dtypes[name]:
            raise ValueError(f"{at} ({name!r}): dtype {_shown(entry['dtype'])} is not the input's dtype {dtypes[name]}")
        for flag in _FLAGS:
            if not isinstance(entry.get(flag, False), bool):
                raise ValueError(f"{at} ({name!r}): {flag} must be true or false, got {_shown(entry[flag])}")
        found[name] = entry
    absent = [name for name in dtypes if name not in found]
    if absent:
        raise ValueError(f"{where}: input {absent[0]!r} has no entry; a declared contract has one entry per input")
    return found


def _bounds(where: str, dtype: str, lo: Any, hi: Any) -> tuple[Any, Any, int, int]:
    """Check [lo, hi) under generate_inputs' bound rules; return lo, hi, and the closed range of dtype values in it.

    The closed range is given as ints that order the values: the values
    themselves, lo to hi - 1, for an integer dtype, and their ordering keys
    (lassi_io _key_range) for a float dtype, whose lo and hi come back as
    doubles. A refusal is a ValueError naming `where`.
    """
    try:
        if dtype in _INTS:
            low, high = _int_bounds(where, dtype, lo, hi)
            return low, high, low, high - 1
        return _float_bounds(where, dtype, lo, hi)
    except LassiIOError as error:
        raise ValueError(str(error)) from None


# ---------------------------------------------------------------------------
# The check


def check_inputs(contract: InputContract, files: Sequence[str | os.PathLike[str]]) -> None:
    """Check a proposed input set, lassi_io files in any order, against `contract`; the files are only read.

    Returns None when the set is inside the contract; otherwise raises
    InputContractError for the first refusal in the module docstring's
    order: each file in the order given (unreadable, too-many-elements),
    then the names in file order (repeated-input, extra-input) and the
    contract's inputs (missing-input), then each input in the contract's
    order (dtype, shape, nan, infinity, out-of-range). Raises TypeError when
    `contract` is not an InputContract (an item with no held-out inputs has
    none) or `files` is not a sequence of paths, naming the type it got.
    """
    if not isinstance(contract, InputContract):
        raise TypeError(f"check_inputs needs an InputContract, got {type(contract).__name__}")
    if isinstance(files, (str, bytes, os.PathLike)) or not isinstance(files, Sequence):
        raise TypeError(f"check_inputs takes a sequence of file paths, got a {type(files).__name__}")
    read = [_read(path) for path in files]
    matched = _matched(contract, read)
    for spec in contract.inputs:
        where, array = matched[spec.name]
        _check_array(spec, where, array)


def _read(path: str | os.PathLike[str]) -> tuple[str, LassiArray]:
    """Read one proposed file, header first; refuse it as unreadable or past the cap. Return its path and array.

    The data is read only once the header is checked, its dims hold at most
    MAX_INPUT_ELEMENTS elements, and the data is their size.
    """
    where = os.fspath(path)
    try:
        with open(where, "rb") as handle:
            header = _read_header(where, handle)
            if element_count(header.dims, MAX_INPUT_ELEMENTS) is None:
                raise InputContractError(
                    "too-many-elements",
                    f"input {header.name!r} ({where}) has dims {_shown(header.dims)}, more than "
                    f"{MAX_INPUT_ELEMENTS} elements; an input holds at most {MAX_INPUT_ELEMENTS}",
                    name=header.name,
                    path=where,
                )
            return where, _read_data(where, handle, header)
    except LassiIOError as error:
        raise InputContractError("unreadable", str(error), path=where) from None
    except OSError as error:
        raise InputContractError(
            "unreadable", f"{where}: the file cannot be read ({error.strerror or type(error).__name__})", path=where
        ) from None


def _matched(contract: InputContract, read: Sequence[tuple[str, LassiArray]]) -> dict[str, tuple[str, LassiArray]]:
    """Match the files' arrays to the contract's inputs by name; refuse a repeated, extra, or missing input."""
    names = set(contract.names)
    matched: dict[str, tuple[str, LassiArray]] = {}
    for where, array in read:
        if array.name in matched:
            raise InputContractError(
                "repeated-input",
                f"input {array.name!r} is held by two files, {matched[array.name][0]} and {where}",
                name=array.name,
                path=where,
            )
        if array.name not in names:
            raise InputContractError(
                "extra-input",
                f"input {array.name!r} ({where}) is not in the contract, whose inputs are {', '.join(contract.names)}",
                name=array.name,
                path=where,
            )
        matched[array.name] = (where, array)
    for name in contract.names:
        if name not in matched:
            raise InputContractError("missing-input", f"input {name!r} is held by no file", name=name)
    return matched


def _check_array(spec: ContractInput, where: str, array: LassiArray) -> None:
    """Refuse an array whose dtype, shape, or elements break its contract input."""
    at = f"input {spec.name!r} ({where})"
    if array.dtype != spec.dtype:
        raise InputContractError(
            "dtype", f"{at} is {array.dtype}; the contract's dtype is {spec.dtype}", name=spec.name, path=where
        )
    if array.shape != spec.shape:
        raise InputContractError(
            "shape",
            f"{at} has shape {_shown(array.shape)}; the contract's shape is {_shown(spec.shape)}",
            name=spec.name,
            path=where,
        )
    if spec.dtype in _FLOATS:
        _check_floats(spec, at, where, array.data)
    else:
        _check_ints(spec, at, where, array.data)


def _first(flags: Iterable[bool]) -> int:
    """Return the index of the first true flag (the caller knows one exists)."""
    return next(index for index, flag in enumerate(flags) if flag)


def _check_floats(spec: ContractInput, at: str, where: str, data: bytes) -> None:
    """Refuse a float array holding a NaN or infinity the contract does not allow, or a finite element out of range.

    Elements are read as bit patterns: a NaN's magnitude bits lie above an
    infinity's, and a finite element's ordering key (lassi_io _key, both
    zeros 0) lies in _key_range(lo, hi) exactly when its value is in [lo, hi).
    """
    fmt = _FLOATS[spec.dtype]
    sign = 1 << (fmt.width - 1)
    infinity = ((1 << fmt.exp_bits) - 1) << fmt.frac_bits
    bits = struct.unpack(f"<{len(data) * 8 // fmt.width}{_BITS_CODE[fmt.width]}", data)
    magnitudes = [element & (sign - 1) for element in bits]
    index, reason, what = -1, "", ""
    if not spec.allow_nan and magnitudes and max(magnitudes) > infinity:
        index, reason = _first(m > infinity for m in magnitudes), "nan"
        what = "a NaN, which the contract does not allow"
    elif not spec.allow_inf and infinity in magnitudes:
        index, reason = magnitudes.index(infinity), "infinity"
        what = f"{'-' if bits[index] & sign else '+'}infinity, which the contract does not allow"
    else:
        lo_key, hi_key = _key_range(float(spec.lo), float(spec.hi), fmt)
        keys = [-m if element & sign else m for element, m in zip(bits, magnitudes, strict=True)]
        outside = [m < infinity and not lo_key <= key <= hi_key for key, m in zip(keys, magnitudes, strict=True)]
        if any(outside):
            index, reason = _first(outside), "out-of-range"
            what = f"{_bits_value(bits[index], fmt)!r}, outside [{spec.lo!r}, {spec.hi!r})"
    if reason:
        shown = f"0x{bits[index]:0{fmt.width // 4}x}"
        raise InputContractError(reason, f"{at}: element {index} is {what} (bits {shown})", name=spec.name, path=where)


def _check_ints(spec: ContractInput, at: str, where: str, data: bytes) -> None:
    """Refuse an integer array holding an element outside [lo, hi)."""
    code = _INTS[spec.dtype][0]
    values = struct.unpack(f"<{len(data) // DTYPES[spec.dtype].itemsize}{code}", data)
    if values and (min(values) < spec.lo or max(values) >= spec.hi):
        index = _first(not spec.lo <= value < spec.hi for value in values)
        raise InputContractError(
            "out-of-range",
            f"{at}: element {index} is {values[index]}, outside [{spec.lo}, {spec.hi})",
            name=spec.name,
            path=where,
        )
