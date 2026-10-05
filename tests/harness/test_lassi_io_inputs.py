"""Tests for the seeded generator of held-out input files (task P4.3).

Bible: Harness Contract (the harness owns inputs; reward and evaluation use
held-out inputs that never appear in a prompt), Agent Rule 10 (a run is
reproducible from its recorded recipe).

The contract these tests fix, in lassi/harness/lassi_io.py (the file format
is stated in test_lassi_io_format.py):

- generate_inputs(spec, seed, out_dir) -> dict[str, Path] writes one
  lassi_io file per named input into the existing directory `out_dir`,
  named <name>.lassiio, and returns name -> path. It writes nothing else.
  `spec` is a sequence of mappings, as a suite manifest holds them:

      {"name": "a", "dtype": "f32", "shape": [64, 64], "dist": "uniform", "lo": -1.0, "hi": 1.0}
      {"name": "k", "dtype": "i32", "shape": [16], "dist": "integers", "lo": 0, "hi": 10}

  "uniform" draws floats in [lo, hi) and "integers" draws integers in
  [lo, hi); both bounds are half-open, and every value written, after
  rounding to the dtype, lies in [lo, hi). `seed` is an int in [0, 2**64).
- Refused with ValueError (LassiIOError is one), before any file is
  written: a seed outside [0, 2**64) or a bool; an entry with lo >= hi, a
  non-finite bound, an unknown dist or dtype, a bad name, a negative dim, or
  an integer range the dtype cannot hold; two entries with the same name.
- The generator is platform-independent and needs no array library, so the
  same seed gives the same bytes on every host. Stated algorithm:
  - Pcg32(initstate, initseq): the PCG-XSH-RR 64/32 generator with the
    reference seeding (state = 0; inc = (initseq << 1) | 1 mod 2**64; step;
    state += initstate; step), where a step is
    state = state * 6364136223846793005 + inc mod 2**64, and next_u32()
    returns the XSH-RR output of the state before the step. The reference
    implementation's demo (seed 42, sequence 54) prints the six outputs
    pinned below.
  - Each input has its own stream: initstate = seed and initseq = the first
    8 bytes of sha256(name as ASCII), read as a little-endian u64. So an
    input's bytes depend on the seed and its own entry alone; adding or
    reordering other inputs changes nothing.
  - Elements are drawn in C order. For "uniform" on f32, each element takes
    one draw r and is lo + (hi - lo) * ((r >> 8) * 2**-24), computed in
    double and rounded to f32 to nearest, ties to even. The bytes of one
    such file are pinned by sha256. How other dtypes turn draws into values,
    and what happens to a value that rounds up to hi, is the
    implementation's to state in the Harness Contract; these tests check
    determinism, ranges, and spread for them.

The pinned digest was computed at authoring time from the stated algorithm
by the reference functions in this module; it is a known-answer value, not
a measurement. The range and spread checks hold with overwhelming
probability for any generator that draws each element independently and
uniformly (the weakest fails with probability below 10**-90).
"""

from __future__ import annotations

import hashlib
import importlib
import math
import struct
from pathlib import Path
from types import ModuleType

import pytest

MAGIC = b"LASSIIO\x00"
M64 = (1 << 64) - 1
M32 = (1 << 32) - 1
PCG_MULTIPLIER = 6364136223846793005
# The reference implementation's demo, seeded with initstate 42 and initseq 54: its first six 32-bit outputs.
PCG_DEMO = (42, 54, [0xA15C02B7, 0x7B47F409, 0xBA1D3330, 0x83D2F293, 0xBFA4784B, 0xCBED606E])

PINNED_ENTRY = {"name": "x", "dtype": "f32", "shape": [4, 4], "dist": "uniform", "lo": -1.0, "hi": 1.0}
PINNED_SEED = 12345
# sha256 of x.lassiio generated from PINNED_ENTRY and PINNED_SEED; see the module docstring.
PINNED_SHA256 = "a1d5b3df12d43685d60722b9e3a815695d6d6b9a8018675fa03476da82f5de7f"

SPEC = [
    {"name": "a", "dtype": "f32", "shape": [8, 8], "dist": "uniform", "lo": -1.0, "hi": 1.0},
    {"name": "b_bf16", "dtype": "bf16", "shape": [16], "dist": "uniform", "lo": -2.0, "hi": 2.0},
    {"name": "idx", "dtype": "i32", "shape": [32], "dist": "integers", "lo": 0, "hi": 10},
    {"name": "s", "dtype": "f64", "shape": [], "dist": "uniform", "lo": 0.0, "hi": 1.0},
]
FORMATS = {"f32": "f", "f16": "e", "f64": "d", "i8": "b", "u8": "B", "i32": "i", "u32": "I", "i64": "q"}


def lassi_io() -> ModuleType:
    """Import and return lassi.harness.lassi_io."""
    return importlib.import_module("lassi.harness.lassi_io")


# ---------------------------------------------------------------------------
# The reference: the stated algorithm, written out independently of the implementation


class ReferencePcg32:
    """PCG-XSH-RR 64/32 with the reference seeding, as the module docstring states."""

    def __init__(self, initstate: int, initseq: int) -> None:
        """Seed the generator the way the reference implementation's srandom does."""
        self.state = 0
        self.inc = ((initseq << 1) | 1) & M64
        self.next_u32()
        self.state = (self.state + initstate) & M64
        self.next_u32()

    def next_u32(self) -> int:
        """Step the state and return the XSH-RR output of the old state."""
        old = self.state
        self.state = (old * PCG_MULTIPLIER + self.inc) & M64
        shifted = (((old >> 18) ^ old) >> 27) & M32
        rotation = old >> 59
        return ((shifted >> rotation) | (shifted << ((-rotation) & 31))) & M32


def stream_of(name: str) -> int:
    """Return the input's stream: the first 8 bytes of sha256(name), little-endian."""
    return int.from_bytes(hashlib.sha256(name.encode("ascii")).digest()[:8], "little")


def reference_f32_file(name: str, shape: list[int], seed: int, lo: float, hi: float) -> bytes:
    """Return the file the stated algorithm writes for a uniform f32 input."""
    rng = ReferencePcg32(seed, stream_of(name))
    count = math.prod(shape)
    values = [lo + (hi - lo) * ((rng.next_u32() >> 8) * 2.0**-24) for _ in range(count)]
    raw = name.encode("ascii")
    head = MAGIC + struct.pack("<III", 1, 1, len(shape)) + struct.pack(f"<{len(shape)}Q", *shape)
    head += struct.pack("<I", len(raw)) + raw
    return head + bytes(-len(head) % 8) + struct.pack(f"<{count}f", *values)


def decode(dtype: str, data: bytes) -> list[float] | list[int]:
    """Decode little-endian element bytes; bf16 widens to f32 by appending 16 zero bits."""
    if dtype == "bf16":
        return [struct.unpack("<f", b"\x00\x00" + data[i : i + 2])[0] for i in range(0, len(data), 2)]
    fmt = FORMATS[dtype]
    return list(struct.unpack(f"<{len(data) // struct.calcsize(fmt)}{fmt}", data))


def generated(spec: list[dict], seed: int, out_dir: Path) -> dict[str, bytes]:
    """Run generate_inputs into a fresh `out_dir`; return name -> file bytes."""
    out_dir.mkdir()
    paths = lassi_io().generate_inputs(spec, seed, out_dir)
    return {name: Path(path).read_bytes() for name, path in paths.items()}


# ---------------------------------------------------------------------------
# The generator core


def test_the_reference_matches_the_published_pcg32_demo() -> None:
    initstate, initseq, outputs = PCG_DEMO
    rng = ReferencePcg32(initstate, initseq)
    assert [rng.next_u32() for _ in outputs] == outputs


def test_pcg32_reproduces_the_published_demo_sequence() -> None:
    initstate, initseq, outputs = PCG_DEMO
    rng = lassi_io().Pcg32(initstate, initseq)
    assert [rng.next_u32() for _ in outputs] == outputs


def test_the_generated_f32_file_follows_the_stated_algorithm(tmp_path: Path) -> None:
    files = generated([PINNED_ENTRY], PINNED_SEED, tmp_path / "out")
    expected = reference_f32_file("x", [4, 4], PINNED_SEED, -1.0, 1.0)
    assert files == {"x": expected}


def test_the_generated_f32_file_has_the_pinned_sha256(tmp_path: Path) -> None:
    files = generated([PINNED_ENTRY], PINNED_SEED, tmp_path / "out")
    assert hashlib.sha256(files["x"]).hexdigest() == PINNED_SHA256


# ---------------------------------------------------------------------------
# Files, names, and determinism


def test_one_file_per_named_input_named_after_it(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    module = lassi_io()
    paths = module.generate_inputs(SPEC, 7, out)
    assert {name: Path(path) for name, path in paths.items()} == {
        entry["name"]: out / f"{entry['name']}.lassiio" for entry in SPEC
    }
    assert sorted(child.name for child in out.iterdir()) == sorted(f"{entry['name']}.lassiio" for entry in SPEC)
    for entry in SPEC:
        array = module.read_array(out / f"{entry['name']}.lassiio")
        assert (array.name, array.dtype, array.shape) == (entry["name"], entry["dtype"], tuple(entry["shape"]))


def test_the_same_seed_gives_the_same_bytes(tmp_path: Path) -> None:
    first = generated(SPEC, 20260925, tmp_path / "first")
    second = generated(SPEC, 20260925, tmp_path / "second")
    assert first == second


def test_a_different_seed_gives_different_bytes(tmp_path: Path) -> None:
    first = generated(SPEC, 1, tmp_path / "first")
    second = generated(SPEC, 2, tmp_path / "second")
    same = sorted(name for name in first if first[name] == second[name])
    assert not same, f"inputs whose bytes do not depend on the seed: {same}"


def test_inputs_with_the_same_entry_but_different_names_differ(tmp_path: Path) -> None:
    entry = {"dtype": "f32", "shape": [64], "dist": "uniform", "lo": -1.0, "hi": 1.0}
    files = generated([{"name": "p", **entry}, {"name": "q", **entry}], 3, tmp_path / "out")
    data = {name: lassi_io().read_array(tmp_path / "out" / f"{name}.lassiio").data for name in files}
    assert data["p"] != data["q"]


def test_adding_or_reordering_inputs_leaves_an_input_unchanged(tmp_path: Path) -> None:
    alone = generated([SPEC[0]], 11, tmp_path / "alone")
    among = generated(list(reversed(SPEC)), 11, tmp_path / "among")
    assert among["a"] == alone["a"]


@pytest.mark.parametrize("seed", [0, 2**64 - 1])
def test_the_ends_of_the_seed_range_are_accepted(seed: int, tmp_path: Path) -> None:
    files = generated([PINNED_ENTRY], seed, tmp_path / "out")
    assert set(files) == {"x"}


# ---------------------------------------------------------------------------
# Ranges and spread


@pytest.mark.parametrize(
    ("dtype", "lo", "hi"), [("f32", -1.0, 1.0), ("f16", 0.0, 1.0), ("bf16", -2.0, 2.0), ("f64", 10.0, 20.0)]
)
def test_uniform_values_lie_in_lo_hi_and_spread_over_it(dtype: str, lo: float, hi: float, tmp_path: Path) -> None:
    entry = {"name": "u", "dtype": dtype, "shape": [64, 64], "dist": "uniform", "lo": lo, "hi": hi}
    generated([entry], 99, tmp_path / "out")
    values = decode(dtype, lassi_io().read_array(tmp_path / "out" / "u.lassiio").data)
    width = hi - lo
    assert len(values) == 4096
    outside = [value for value in values if not (math.isfinite(value) and lo <= value < hi)]
    assert not outside, f"values outside [{lo}, {hi}): {outside[:10]}"
    assert min(values) < lo + 0.05 * width and max(values) >= hi - 0.05 * width
    assert abs(sum(values) / len(values) - (lo + hi) / 2) < 0.05 * width
    assert len(set(values)) > 100


@pytest.mark.parametrize(
    ("dtype", "lo", "hi", "count", "check"),
    [
        ("i32", -3, 4, 4096, "cover"),
        ("u8", 0, 256, 8192, "cover"),
        ("i8", -128, 128, 8192, "cover"),
        ("u32", 0, 2**32, 4096, "wide"),
        ("i64", -(2**40), 2**40, 4096, "wide"),
    ],
)
def test_integer_values_lie_in_lo_hi(dtype: str, lo: int, hi: int, count: int, check: str, tmp_path: Path) -> None:
    entry = {"name": "k", "dtype": dtype, "shape": [count], "dist": "integers", "lo": lo, "hi": hi}
    generated([entry], 5, tmp_path / "out")
    values = decode(dtype, lassi_io().read_array(tmp_path / "out" / "k.lassiio").data)
    assert len(values) == count
    outside = [value for value in values if not lo <= value < hi]
    assert not outside, f"values outside [{lo}, {hi}): {outside[:10]}"
    if check == "cover":
        assert set(values) == set(range(lo, hi))
    else:
        # Wider than one 32-bit draw can reach for i64: the lowest and the highest quarter both occur.
        assert min(values) < lo + (hi - lo) // 4 and max(values) >= hi - (hi - lo) // 4


# ---------------------------------------------------------------------------
# Refusals


def entry_for(dtype: str, dist: str, lo: float, hi: float, name: str = "b", shape: list[int] | None = None) -> dict:
    """Return one spec entry with the given fields; four elements unless `shape` says otherwise."""
    return {"name": name, "dtype": dtype, "shape": [4] if shape is None else shape, "dist": dist, "lo": lo, "hi": hi}


# Each entry follows SPEC[0], a valid entry named "a", in the refused spec.
REFUSED_ENTRIES = {
    "lo equals hi": entry_for("f32", "uniform", 1.0, 1.0),
    "lo above hi": entry_for("i32", "integers", 5, 2),
    "infinite bound": entry_for("f32", "uniform", -math.inf, 1.0),
    "NaN bound": entry_for("f64", "uniform", 0.0, math.nan),
    "unknown dist": entry_for("f32", "normal", 0.0, 1.0),
    "unknown dtype": entry_for("f128", "uniform", 0.0, 1.0),
    "u8 range too wide": entry_for("u8", "integers", 0, 257),
    "i8 range too low": entry_for("i8", "integers", -129, 0),
    "u32 below zero": entry_for("u32", "integers", -1, 5),
    "bad name": entry_for("f32", "uniform", 0.0, 1.0, name="a b"),
    "negative dim": entry_for("f32", "uniform", 0.0, 1.0, shape=[-1]),
    "duplicate name": entry_for("f32", "uniform", 0.0, 1.0, name="a"),
}


@pytest.mark.parametrize("case", sorted(REFUSED_ENTRIES))
def test_a_refused_spec_writes_nothing(case: str, tmp_path: Path) -> None:
    entry = REFUSED_ENTRIES[case]
    out = tmp_path / "out"
    out.mkdir()
    spec = [SPEC[0], entry]
    with pytest.raises(ValueError):
        lassi_io().generate_inputs(spec, 1, out)
    assert list(out.iterdir()) == [], "a refused spec left files behind"


@pytest.mark.parametrize("seed", [-1, 2**64, True])
def test_a_seed_outside_u64_is_refused(seed: int, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError):
        lassi_io().generate_inputs([PINNED_ENTRY], seed, out)
    assert list(out.iterdir()) == []
