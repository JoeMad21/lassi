"""Tests for tt-par-v0-kh, tt-par-v0's items with the model writing the host program and its kernels (task P17.69).

Bible: Benchmark Suites (tt-par-v0; the support and `reference:` keys), Harness Contract (a kernel is written
uncompiled at its path for the kernel JIT), Agent Rule 5 (the development items dev and the held-out items eval,
OQ-129; no reference kernel and no tracked host program in a prompt).

What they fix, as tests/bench/test_kh_suite.py fixes tt-pairs-v0-kh against tt-pairs-v0:

- assets/bench/tt-par-v0-kh.yaml holds tt-par-v0's items unchanged (the same splits, inputs, seeds, outputs,
  tolerances, work, notes, and cpp and tt languages), plus a ttkh language, with tt-par-v0's repo, commit, and pin.
- The ttkh files are the item's tracked TT host program's name, then each kernel at the path that host program
  launches it, once each, in the order the host program first names them.
- The ttkh support is lassi_io.h alone, with its sha256: no kernel and no host program, since either would be the
  answer.
- `reference: tt`: the reference of cpp -> ttkh is the item's tt program.
- A cpp -> ttkh prompt shows no kernel and no host program for any item.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import pytest

from lassi.bench import Direction, load_suite
from lassi.core.stages import host_files, kernel_files

REPO = Path(__file__).resolve().parents[2]
SIBLING = REPO / "assets" / "bench" / "tt-par-v0-kh.yaml"
ORIGINAL = REPO / "assets" / "bench" / "tt-par-v0.yaml"
DEV = ("matmul_multicore_reuse", "matmul_multicore_reuse_mcast", "pad_multi_core", "vecadd_multi_core",
       "vecadd_sharding")
HELD_OUT = ("col_max", "col_scale", "row_softmax", "row_sum", "transpose_2d")
ITEMS = tuple(sorted(DEV + HELD_OUT))
DIRECTION = Direction("cpp", "ttkh")
KERNEL_LITERAL = re.compile(r'"([^"\s]*kernels/[^"\s]*\.cpp)"')
HARNESS = REPO / "assets" / "harness" / "c" / "lassi_io.h"
ITEM_KEYS = ("split", "inputs", "outputs", "seed", "tolerance", "unpack_to_dest", "contract", "work")


def sibling() -> Any:
    """Return the loaded sibling suite, failing with the task's words when its manifest is missing."""
    if not SIBLING.is_file():
        pytest.fail(f"{SIBLING.relative_to(REPO).as_posix()} does not exist (task P17.69)")
    return load_suite(SIBLING)


def test_the_sibling_holds_tt_par_v0s_items_unchanged() -> None:
    """Every item, its keys, and its cpp and tt languages are tt-par-v0's; each item adds ttkh."""
    suite, original = sibling(), load_suite(ORIGINAL)
    assert suite.name == "tt-par-v0-kh"
    assert sorted(suite.items) == list(ITEMS) == sorted(original.items)
    assert (suite.repo, suite.commit, suite.pin) == (original.repo, original.commit, original.pin)
    for name in ITEMS:
        ours, theirs = suite.items[name], original.items[name]
        for key in ITEM_KEYS:
            assert getattr(ours, key) == getattr(theirs, key), f"{name}.{key} drifted from tt-par-v0"
        for language in ("cpp", "tt"):
            assert ours.languages[language] == theirs.languages[language], f"{name}.{language} drifted"
        assert sorted(ours.languages) == ["cpp", "tt", "ttkh"]
        assert ours.split == ("dev" if name in DEV else "eval"), f"{name}: the split OQ-129 gave it"


def test_the_manifest_is_plain_ascii_and_names_its_task() -> None:
    """The manifest is plain ASCII with LF line ends and names task P17.69 and OQ-129."""
    sibling()
    data = SIBLING.read_bytes()
    assert data.isascii() and b"\r" not in data
    text = data.decode("ascii")
    assert "P17.69" in text and "OQ-129" in text


@pytest.mark.parametrize("name", ITEMS)
def test_the_ttkh_files_are_the_host_program_then_each_kernel_it_launches(name: str) -> None:
    """ttkh files: the tt host program's name, then each launched kernel once, in first-named order."""
    item = sibling().items[name]
    spec, tt = item.languages["ttkh"], item.languages["tt"]
    host_text = (REPO / tt.dir / tt.files[0]).read_text(encoding="utf-8")
    launched = list(dict.fromkeys(KERNEL_LITERAL.findall(host_text)))
    assert launched, f"{name}: the host program launches no kernel by path"
    assert list(spec.files) == [tt.files[0], *launched]
    kernels = [path for path in spec.files if "kernels" in PurePosixPath(path).parts[:-1]]
    assert kernels == launched and sorted(kernels) == sorted(path for path in tt.support if path in launched)
    assert spec.reference == "tt" and spec.tracked
    assert not (REPO / spec.dir).exists(), "the ttkh directory is never read, so none is tracked"


@pytest.mark.parametrize("name", ITEMS)
def test_the_ttkh_support_is_the_harness_header_alone_with_its_sha256(name: str) -> None:
    """The ttkh support is lassi_io.h alone, tracked, with the harness header's sha256."""
    spec = sibling().items[name].languages["ttkh"]
    assert set(spec.support) == {"lassi_io.h"}, "no kernel and no host program: either would be the answer"
    entry = spec.support["lassi_io.h"]
    assert entry.tracked and entry.path == "assets/harness/c/lassi_io.h"
    assert entry.sha256 == hashlib.sha256(HARNESS.read_bytes()).hexdigest()


def test_the_reference_of_cpp_to_ttkh_is_the_items_tt_program(tmp_path: Path) -> None:
    """cpp -> ttkh's reference is the item's tt program, and its support is lassi_io.h alone."""
    suite = sibling()
    files = suite.reference_target("row_sum", DIRECTION, tmp_path, purpose="eval")
    assert "row_sum.cpp" in files
    assert suite.reference_language("row_sum", "ttkh") == "tt"
    support = suite.support_files("row_sum", tmp_path, purpose="eval", language="ttkh")
    assert set(support) == {"lassi_io.h"}


@pytest.mark.parametrize("name", ITEMS)
def test_a_cpp_to_ttkh_prompt_shows_no_kernel_and_no_host_program(name: str, tmp_path: Path) -> None:
    """The kernel_files and host_files template fields are empty for every item's cpp -> ttkh prompt."""
    context: Any = SimpleNamespace(suite=sibling(), item=name, direction=DIRECTION, sources_root=tmp_path)
    assert kernel_files(context) == "", "the reference's kernels would be the answer (Agent Rule 5)"
    assert host_files(context) == "", "the tracked host program would be the answer (Agent Rule 5)"
