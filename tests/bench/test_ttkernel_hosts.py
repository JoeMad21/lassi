"""Tests for task P17.37, built under OQ-103 (a) by task P17.41: ttkernel host programs that pass CB indices.

Bible: Level Rules (a ttkernel kernel names a circular buffer only through
an argument: tt-mlir 5f396fd6, TTKernelOps.td:2756-2800), Benchmark Suites
(tt-pairs-v0-ttkernel), Harness Contract, Agent Rule 5 (no upstream kernel
text in a tracked file or a prompt). The owner's direction of 2026-10-26
that the MLIR fixes get done is taken as OQ-103, option (a).

What they fix:

- eltwise_binary, eltwise_sfpu, matmul_single_core, and matmul_multi_core
  each have a tracked ttkernel host program,
  assets/bench/tt-pairs-v0-ttkernel/<item>/<item>.cpp, named with its
  sha256 as the ttkernel language's host support in
  tt-pairs-v0-ttkernel.yaml; loopback, whose kernel uses no circular buffer,
  keeps its tt host program.
- Each is the item's tt host program with one change: each kernel's circular
  buffer indices are appended to its compile-time arguments, after the
  arguments it already had, and its header comment documents the positions.
  The diff test strips the appended lines and the CB entries of the
  compile-argument lists and finds the tt program's code, byte for byte
  outside whitespace and the header comment.
- The documented positions are the code's: one TensorAccessorArgs argument
  per interleaved DRAM buffer (tt-metal 5280a9cf,
  tt_metal/impl/buffers/tensor_accessor_args.cpp:155-165), then the CB
  indices in the order pushed.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from lassi.bench import load_suite

REPO = Path(__file__).resolve().parents[2]
SUITE = load_suite(REPO / "assets" / "bench" / "tt-pairs-v0-ttkernel.yaml")
CHANGED = ("eltwise_binary", "eltwise_sfpu", "matmul_multi_core", "matmul_single_core")
# The positions each kernel's circular buffer indices take, by the kernel's file stem: (position, CB name).
POSITIONS = {
    "eltwise_binary": {"read_tiles": [(2, "c_0"), (3, "c_1")], "write_tile": [(1, "c_16")],
                       "tiles_add": [(0, "c_0"), (1, "c_1"), (2, "c_16")]},
    "eltwise_sfpu": {"read_tile": [(1, "c_0")], "write_tile": [(1, "c_16")], "eltwise_sfpu": [(0, "c_0"), (1, "c_16")]},
    "matmul_single_core": {"reader_single_core_mm": [(2, "c_0"), (3, "c_1")], "writer_single_core_mm": [(1, "c_16")],
                           "mm": [(3, "c_0"), (4, "c_1"), (5, "c_16")]},
    "matmul_multi_core": {"reader_mm_output_tiles_partitioned": [(2, "c_0"), (3, "c_1")],
                          "writer_unary_interleaved_start_id": [(1, "c_16")],
                          "mm": [(0, "c_0"), (1, "c_1"), (2, "c_16")]},
}
# An appended line: a push of one CB index onto a compile-argument vector, with an optional comment.
PUSHED = re.compile(r"^\s*\w+\.push_back\((?:tt::)?(?:CBIndex::c_\d+|kInputCb|kOutputCb)\);(?:\s*//.*)?$")
# A CB entry of a compile-argument list, with the comma before or after it.
CB_ENTRY = re.compile(r",(?:tt::)?CBIndex::c_\d+|(?:tt::)?CBIndex::c_\d+,?|,kInputCb|,kOutputCb|kInputCb,?|kOutputCb,?")


def host_paths(item: str) -> tuple[Path, Path]:
    """Return the item's tt host program and its ttkernel host program."""
    tt = SUITE.items[item].languages["tt"]
    return REPO / tt.dir / tt.files[0], REPO / f"assets/bench/tt-pairs-v0-ttkernel/{item}/{item}.cpp"


def split(text: str) -> tuple[list[str], list[str]]:
    """Return a host program's header comment lines (before the first #include) and its other lines."""
    lines = text.split("\n")
    first = next(index for index, line in enumerate(lines) if line.startswith("#include"))
    return lines[:first], lines[first:]


def compile_args_lists(text: str) -> str:
    """Return `text` with every CB entry removed from its `.compile_args = {...}` lists."""
    def strip(match: re.Match[str]) -> str:
        """Return one list without its CB entries."""
        return ".compile_args={" + CB_ENTRY.sub("", match.group(1)) + "}"

    return re.sub(r"\.compile_args=\{([^{}]*)\}", strip, text)


def normalized(lines: list[str], ttkernel: bool) -> str:
    """Return the code lines as the diff test compares them: whitespace removed; for ttkernel, the CB changes undone."""
    if ttkernel:
        lines = [line for line in lines if not PUSHED.match(line)]
        lines = [line for line in lines if not (line.strip().startswith("//") and "ttkernel" in line)]
    text = "".join("".join(line.split()) for line in lines)
    if ttkernel:
        text = compile_args_lists(text).replace(",.compile_args={}", "")
    return text.replace(",.compile_args={}", "")


@pytest.mark.parametrize("item", CHANGED)
def test_the_manifest_names_the_ttkernel_host_program_with_its_sha256(item: str) -> None:
    spec = SUITE.items[item].languages["ttkernel"]
    tt_path, path = host_paths(item)
    entry = spec.support[tt_path.name]
    assert entry.tracked and entry.path == path.relative_to(REPO).as_posix()
    assert entry.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert set(spec.support) == {tt_path.name, "lassi_io.h"}


def test_loopback_keeps_its_tt_host_program() -> None:
    tt_path, _ = host_paths("loopback")
    entry = SUITE.items["loopback"].languages["ttkernel"].support[tt_path.name]
    assert entry.path == tt_path.relative_to(REPO).as_posix()
    assert not (REPO / "assets/bench/tt-pairs-v0-ttkernel/loopback/loopback.cpp").exists()


@pytest.mark.parametrize("item", CHANGED)
def test_the_ttkernel_host_program_is_the_tt_one_with_cb_indices_appended(item: str) -> None:
    tt_path, path = host_paths(item)
    tt_head, tt_code = split(tt_path.read_text(encoding="utf-8"))
    head, code = split(path.read_text(encoding="utf-8"))
    assert all(line.startswith("//") or not line for line in head)
    assert normalized(code, ttkernel=True) == normalized(tt_code, ttkernel=False), "only the CB indices are added"
    pushed = [line for line in code if PUSHED.match(line)]
    assert pushed and not [line for line in tt_code if PUSHED.match(line)]
    assert path.read_bytes().isascii() and b"\r" not in path.read_bytes()


@pytest.mark.parametrize("item", CHANGED)
def test_the_header_documents_each_kernels_cb_positions(item: str) -> None:
    _, path = host_paths(item)
    head, _ = split(path.read_text(encoding="utf-8"))
    text = " ".join(" ".join(line.removeprefix("//").split()) for line in head)
    assert "OQ-103" in text and "compile-time argument" in text
    for stem, positions in POSITIONS[item].items():
        assert stem in text, f"{item}: the header names the kernel {stem}"
        for position, name in positions:
            assert f"{position} {name}" in text, f"{item}: {stem} takes {name} at {position}"


@pytest.mark.parametrize("item", CHANGED)
def test_each_kernel_appends_its_cbs_in_the_documented_order(item: str) -> None:
    _, path = host_paths(item)
    _, code = split(path.read_text(encoding="utf-8"))
    joined = "\n".join(code)
    pushed = [re.search(r"(c_\d+|kInputCb|kOutputCb)", line).group(1)  # type: ignore[union-attr]
              for line in code if PUSHED.match(line)]
    expected = [name for stem, positions in POSITIONS[item].items() for _, name in positions
                if not stem.startswith(("tiles_add", "eltwise_sfpu", "mm"))]
    names = [{"kInputCb": "c_0", "kOutputCb": "c_16"}.get(name, name) for name in pushed]
    assert names == expected, (names, expected)
    assert re.search(r"\.compile_args\s*=\s*\{[^{}]*CBIndex::c_0[^{}]*\}|\.compile_args\s*=\s*\{kInputCb", joined)
