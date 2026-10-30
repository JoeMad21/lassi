"""Tests for the cpp -> ttir and cpp -> d2m suites tt-pairs-v0-ttir and tt-pairs-v0-d2m (task P17.48).

Bible: Benchmark Suites (tt-pairs-v0; manifest keys), IR Strategy (TTIR and
D2M above ttkernel), Agent Rule 5 (tt-pairs-v0 is a development set,
OQ-083 (b); no item's answer in a tracked file or a prompt).

What they fix:

- Each suite is a sibling of tt-pairs-v0.yaml, as tt-pairs-v0-ttkernel is:
  the same five items with the same splits, held-out inputs, seeds,
  outputs, tolerances, notes, analytic work, and cpp and tt languages, the
  same repo, commit, and pin, plus one language (ttir or d2m).
- That language's files are one module per item, <item>.mlir; it is
  tracked, has no support file (so no answer and no harness file), and
  names `reference: tt`, so its reference program is the item's tt program
  and no reference module exists in the repository.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lassi.bench import Direction, load_suite

REPO = Path(__file__).resolve().parents[2]
ORIGINAL = REPO / "assets" / "bench" / "tt-pairs-v0.yaml"
ITEMS = ("eltwise_binary", "eltwise_sfpu", "loopback", "matmul_multi_core", "matmul_single_core")
LANGUAGES = ("ttir", "d2m")
KEYS = ("split", "inputs", "outputs", "seed", "tolerance", "unpack_to_dest", "contract", "work", "support")


@pytest.mark.parametrize("language", LANGUAGES)
def test_the_sibling_holds_tt_pairs_v0_unchanged(language: str) -> None:
    suite, original = load_suite(REPO / "assets" / "bench" / f"tt-pairs-v0-{language}.yaml"), load_suite(ORIGINAL)
    kernel = load_suite(REPO / "assets" / "bench" / "tt-pairs-v0-ttkernel.yaml")
    assert suite.name == f"tt-pairs-v0-{language}"
    assert sorted(suite.items) == list(ITEMS)
    assert (suite.repo, suite.commit, suite.pin) == (original.repo, original.commit, original.pin)
    for name in ITEMS:
        ours, theirs = suite.items[name], original.items[name]
        for key in KEYS:
            assert getattr(ours, key) == getattr(theirs, key), f"{name}.{key} drifted from tt-pairs-v0"
        assert ours.compute is not None and ours.compute == kernel.items[name].compute, "the cgeist compute units"
        assert set(ours.languages) == {"cpp", "tt", language}
        for kept in ("cpp", "tt"):
            assert ours.languages[kept] == theirs.languages[kept], f"{name}.{kept} drifted"
        assert ours.split == "unassigned"


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("name", ITEMS)
def test_the_new_language_is_one_module_with_the_tt_reference_and_no_support(language: str, name: str) -> None:
    suite = load_suite(REPO / "assets" / "bench" / f"tt-pairs-v0-{language}.yaml")
    spec = suite.items[name].languages[language]
    assert spec.files == (f"{name}.mlir",)
    assert spec.tracked and spec.reference == "tt" and spec.support == {}
    assert suite.reference_language(name, language) == "tt"
    assert not (REPO / spec.dir).exists(), "no reference module is tracked: it would be the item's answer"
    assert suite.bench_item(name, Direction("cpp", language)).direction == f"cpp-{language}"


@pytest.mark.parametrize("language", LANGUAGES)
def test_the_manifest_is_plain_ascii_and_names_its_task(language: str) -> None:
    data = (REPO / "assets" / "bench" / f"tt-pairs-v0-{language}.yaml").read_bytes()
    assert data.isascii() and b"\r" not in data
    text = data.decode("ascii")
    assert "P17.48" in text and "plans/spikes/ttir-d2m.md" in text and "reference: tt" in text
