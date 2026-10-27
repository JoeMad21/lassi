"""Tests for the C++ -> TT suite tt-pairs-v0-kh, in which the model writes the host program and its kernels (P17.39).

Bible: Benchmark Suites (tt-pairs-v0; manifest keys, the support and
`reference:` keys), Harness Contract (a kernel is written uncompiled at its
path for the kernel JIT), Agent Rule 5 (tt-pairs-v0 is a development set,
OQ-083 (b); no upstream kernel and no tracked host program in a prompt).

What they fix:

- assets/bench/tt-pairs-v0-kh.yaml is a sibling of tt-pairs-v0.yaml: the
  same five items with the same splits, inputs, seeds, outputs, tolerances,
  notes, and cpp and tt languages, plus a ttkh language.
- The ttkh files are the item's tracked TT host program's name, then each
  kernel at the path that host program launches it, in launch order: the
  whole program the model writes.
- The ttkh support is lassi_io.h alone, with its sha256: no kernel and no
  host program, since either would be the answer.
- `reference: tt`: the reference of cpp -> ttkh is the item's tt program
  (Suite.reference_target, Suite.reference_language), built and run as the
  tt reference is.
- A cpp -> ttkh prompt shows no kernel and no host program: the template
  fields kernel_files and host_files are empty for every item.
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
SIBLING = REPO / "assets" / "bench" / "tt-pairs-v0-kh.yaml"
ORIGINAL = REPO / "assets" / "bench" / "tt-pairs-v0.yaml"
ITEMS = ("eltwise_binary", "eltwise_sfpu", "loopback", "matmul_multi_core", "matmul_single_core")
DIRECTION = Direction("cpp", "ttkh")
KERNEL_LITERAL = re.compile(r'"([^"\s]*kernels/[^"\s]*\.cpp)"')
HARNESS = REPO / "assets" / "harness" / "c" / "lassi_io.h"


def sibling() -> Any:
    """Return the loaded sibling suite."""
    return load_suite(SIBLING)


def test_the_sibling_holds_the_development_items_unchanged() -> None:
    suite, original = sibling(), load_suite(ORIGINAL)
    assert suite.name == "tt-pairs-v0-kh"
    assert sorted(suite.items) == list(ITEMS)
    assert (suite.repo, suite.commit, suite.pin) == (original.repo, original.commit, original.pin)
    for name in ITEMS:
        ours, theirs = suite.items[name], original.items[name]
        for key in ("split", "inputs", "outputs", "seed", "tolerance", "unpack_to_dest", "contract"):
            assert getattr(ours, key) == getattr(theirs, key), f"{name}.{key} drifted from tt-pairs-v0"
        for language in ("cpp", "tt"):
            assert ours.languages[language] == theirs.languages[language], f"{name}.{language} drifted"
        assert sorted(ours.languages) == ["cpp", "tt", "ttkh"]
        assert ours.split == "unassigned"


@pytest.mark.parametrize("name", ITEMS)
def test_the_ttkh_files_are_the_host_program_then_each_kernel_it_launches(name: str) -> None:
    item = sibling().items[name]
    spec, tt = item.languages["ttkh"], item.languages["tt"]
    host_text = (REPO / tt.dir / tt.files[0]).read_text(encoding="utf-8")
    launched = KERNEL_LITERAL.findall(host_text)
    assert launched, f"{name}: the host program launches no kernel by path"
    assert list(spec.files) == [tt.files[0], *launched]
    kernels = [path for path in spec.files if "kernels" in PurePosixPath(path).parts[:-1]]
    assert kernels == launched and sorted(kernels) == sorted(path for path in tt.support if path in launched)
    assert spec.reference == "tt" and spec.tracked
    assert not (REPO / spec.dir).exists(), "the ttkh directory is never read, so none is tracked"


@pytest.mark.parametrize("name", ITEMS)
def test_the_ttkh_support_is_the_harness_header_alone_with_its_sha256(name: str) -> None:
    spec = sibling().items[name].languages["ttkh"]
    assert set(spec.support) == {"lassi_io.h"}, "no kernel and no host program: either would be the answer"
    entry = spec.support["lassi_io.h"]
    assert entry.tracked and entry.path == "assets/harness/c/lassi_io.h"
    assert entry.sha256 == hashlib.sha256(HARNESS.read_bytes()).hexdigest()


def test_the_reference_of_cpp_to_ttkh_is_the_items_tt_program(tmp_path: Path) -> None:
    suite = sibling()
    files = suite.reference_target("loopback", DIRECTION, tmp_path, purpose="eval")
    assert set(files) == {"loopback.cpp"}
    assert suite.reference_language("loopback", "ttkh") == "tt"
    support = suite.support_files("loopback", tmp_path, purpose="eval", language="ttkh")
    assert set(support) == {"lassi_io.h"}


@pytest.mark.parametrize("name", ITEMS)
def test_a_cpp_to_ttkh_prompt_shows_no_kernel_and_no_host_program(name: str, tmp_path: Path) -> None:
    context: Any = SimpleNamespace(suite=sibling(), item=name, direction=DIRECTION, sources_root=tmp_path)
    assert kernel_files(context) == "", "the upstream kernels would be the answer (Agent Rule 5)"
    assert host_files(context) == "", "the tracked host program would be the answer (Agent Rule 5)"
