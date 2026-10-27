"""Tests for the cpp -> ttkernel suite tt-pairs-v0-ttkernel and the registry keys it needs (task P17.36).

Bible: Benchmark Suites (tt-pairs-v0; manifest keys), Level Rules (TT host
programs are never model output), Harness Contract, Agent Rule 5 (tt-pairs-v0
is a development set, OQ-083 (b); no upstream kernel text in a tracked file
or a prompt).

What they fix:

- assets/bench/tt-pairs-v0-ttkernel.yaml is a sibling of tt-pairs-v0.yaml:
  the same five items with the same splits, inputs, seeds, outputs,
  tolerances, notes, and cpp and tt languages, plus a ttkernel language. Its
  files are the kernel modules the model writes, one per kernel the item's
  host program launches, at that kernel's path with .mlir in place of .cpp;
  its support is the tracked host program and lassi_io.h, each with its
  sha256, and no kernel (the upstream kernel C++ would be the answer). The
  host program is the item's tt one for loopback and, for the other four
  items, the ttkernel host program of task P17.37 (built by task P17.41,
  tests/bench/test_ttkernel_hosts.py), which passes the CB indices.
- A language may name `reference: <language>`, another language of the
  item: its reference program is that language's (Suite.reference_target
  and Suite.reference_language), and it must be tracked: true, since its
  own files are never read. A reference to an unknown language, to itself,
  or to a language with a reference of its own is refused at load.
- A tracked support entry may carry sha256; Suite.support_files then
  refuses a file whose bytes differ, so the trusted host program a build
  gets is the manifest's.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

from lassi.bench import Direction, load_suite, registry

REPO = Path(__file__).resolve().parents[2]
SIBLING = REPO / "assets" / "bench" / "tt-pairs-v0-ttkernel.yaml"
ORIGINAL = REPO / "assets" / "bench" / "tt-pairs-v0.yaml"
ITEMS = ("eltwise_binary", "eltwise_sfpu", "loopback", "matmul_multi_core", "matmul_single_core")
DIRECTION = Direction("cpp", "ttkernel")
KERNEL_LITERAL = re.compile(r'"([^"\s]*kernels/[^"\s]*\.cpp)"')


def sibling() -> registry.Suite:
    """Return the loaded sibling suite."""
    return load_suite(SIBLING)


def test_the_sibling_holds_the_development_items_unchanged() -> None:
    suite, original = sibling(), load_suite(ORIGINAL)
    assert suite.name == "tt-pairs-v0-ttkernel"
    assert sorted(suite.items) == list(ITEMS)
    assert (suite.repo, suite.commit, suite.pin) == (original.repo, original.commit, original.pin)
    for name in ITEMS:
        ours, theirs = suite.items[name], original.items[name]
        for key in ("split", "inputs", "outputs", "seed", "tolerance", "unpack_to_dest", "contract"):
            assert getattr(ours, key) == getattr(theirs, key), f"{name}.{key} drifted from tt-pairs-v0"
        for language in ("cpp", "tt"):
            assert ours.languages[language] == theirs.languages[language], f"{name}.{language} drifted"
        assert ours.split == "unassigned"


@pytest.mark.parametrize("name", ITEMS)
def test_the_ttkernel_files_are_the_kernels_the_host_program_launches(name: str) -> None:
    spec = sibling().items[name].languages["ttkernel"]
    tt = sibling().items[name].languages["tt"]
    host_text = (REPO / tt.dir / tt.files[0]).read_text(encoding="utf-8")
    launched = KERNEL_LITERAL.findall(host_text)
    assert launched, f"{name}: the host program launches no kernel by path"
    assert list(spec.files) == [str(PurePosixPath(path).with_suffix(".mlir")) for path in launched]
    assert spec.reference == "tt" and spec.tracked


@pytest.mark.parametrize("name", ITEMS)
def test_the_ttkernel_support_is_the_tracked_host_program_and_harness_with_sha256(name: str) -> None:
    spec = sibling().items[name].languages["ttkernel"]
    tt = sibling().items[name].languages["tt"]
    assert set(spec.support) == {tt.files[0], "lassi_io.h"}, "the host program and lassi_io.h, never a kernel"
    host = spec.support[tt.files[0]]
    own = tt.dir if name == "loopback" else f"assets/bench/tt-pairs-v0-ttkernel/{name}"
    assert host.tracked and host.path == f"{own}/{tt.files[0]}"
    for entry in spec.support.values():
        assert entry.tracked and entry.sha256 == hashlib.sha256((REPO / entry.path).read_bytes()).hexdigest()
        assert "kernels" not in PurePosixPath(entry.path).parts


def test_the_reference_of_cpp_to_ttkernel_is_the_items_tt_program(tmp_path: Path) -> None:
    suite = sibling()
    files = suite.reference_target("loopback", DIRECTION, tmp_path, purpose="eval")
    assert set(files) == {"loopback.cpp"}
    assert suite.reference_language("loopback", "ttkernel") == "tt"
    assert suite.reference_language("loopback", "cpp") == "cpp"
    support = suite.support_files("loopback", tmp_path, purpose="eval", language="ttkernel")
    assert set(support) == {"loopback.cpp", "lassi_io.h"}


def test_support_files_refuse_a_host_program_whose_bytes_differ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = tmp_path / "repo"
    for path in ("assets/bench/tt-pairs-v0/loopback/tt/loopback.cpp", "assets/harness/c/lassi_io.h"):
        (copy / path).parent.mkdir(parents=True, exist_ok=True)
        (copy / path).write_bytes((REPO / path).read_bytes())
    (copy / "assets/bench/tt-pairs-v0/loopback/tt/loopback.cpp").write_bytes(b"int main() { return 0; }\n")
    monkeypatch.setattr(registry, "TRACKED_ROOT", copy)
    with pytest.raises(ValueError, match="sha256"):
        sibling().support_files("loopback", tmp_path, purpose="eval", language="ttkernel")


def _write(tmp_path: Path, change: Any) -> Path:
    """Write a copy of the sibling manifest with `change` applied to loopback's languages; return its path."""
    data = yaml.safe_load(SIBLING.read_bytes().decode("utf-8"))
    change(data["items"]["loopback"]["languages"])
    path = tmp_path / SIBLING.name
    path.write_bytes(yaml.safe_dump(data).encode("utf-8"))
    return path


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda langs: langs["ttkernel"].update(reference="rust"), "reference"),
        (lambda langs: langs["ttkernel"].update(reference="ttkernel"), "reference"),
        (lambda langs: langs["tt"].update(reference="cpp"), "reference"),
        (lambda langs: langs["ttkernel"].update(reference=3), "reference"),
        (lambda langs: langs["ttkernel"].pop("tracked"), "tracked"),
        (lambda langs: langs["ttkernel"]["support"]["lassi_io.h"].update(sha256="xyz"), "sha256"),
    ],
)
def test_bad_reference_and_support_keys_are_refused(tmp_path: Path, change: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_suite(_write(tmp_path, change))
