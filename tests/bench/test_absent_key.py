"""Tests for the manifest key `absent`: languages an item has no version of at the pin (task P9.7).

Bible: Benchmark Suites (lassi-ee-22; manifest keys), Agent Rule 5. OQ-089
(a) lists lassi-ee-22's four apps without an OpenMP version at 692cba3 and
names them as not ported; the manifest says so with this key.

The contract (lassi.bench.registry):

- per item, `absent: {<language>: <note>}`, optional: each language the item
  has no version of at the pinned commit, with a non-empty note saying why.
  SuiteItem.absent maps language -> note, and is empty when the key is left
  out.
- A language may not be both absent and under `languages`; `languages` may
  be empty only on an item that declares `absent` (an app with no sources at
  the pin). passfail may not name an absent language, which is not a
  language of the item.
- Reading an absent language's sources (reference_target, source_files,
  support_files with that language) raises ValueError quoting its note.
- The key changes no split rule: an eval item that declares it is refused to
  training as before.

Manifests and sources here are synthetic, written under tmp_path.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lassi import bench

NOTE = "SYNTHETIC: no OpenMP version at the pin"


def write_manifest(tmp_path: Path, item: str) -> Path:
    """Write a one-item manifest whose item entry is `item` (YAML, indented for the item) and return its path."""
    path = tmp_path / "absent-suite.yaml"
    path.write_bytes(
        (
            "suite: absent-suite\n"
            "repo: https://example.invalid/demo.git\n"
            f"commit: {'b' * 40}\n"
            "items:\n"
            "  saxpy:\n"
            "    split: eval\n" + item
        ).encode("ascii")
    )
    return path


CUDA_ONLY = "    languages:\n      cuda: {dir: src/saxpy-cuda, files: [main.cu]}\n"


def test_an_item_declares_its_absent_languages_each_with_a_note(tmp_path: Path) -> None:
    suite = bench.load_suite(write_manifest(tmp_path, f"    absent: {{omp: '{NOTE}'}}\n" + CUDA_ONLY))
    item = suite.items["saxpy"]
    assert dict(item.absent) == {"omp": NOTE}
    assert set(item.languages) == {"cuda"}


def test_an_item_without_the_key_has_no_absent_language(tmp_path: Path) -> None:
    item = bench.load_suite(write_manifest(tmp_path, CUDA_ONLY)).items["saxpy"]
    assert dict(item.absent) == {}


def test_an_item_with_no_sources_at_the_pin_declares_every_language_absent(tmp_path: Path) -> None:
    entry = f"    absent: {{omp: '{NOTE}', cuda: '{NOTE}'}}\n    languages: {{}}\n"
    item = bench.load_suite(write_manifest(tmp_path, entry)).items["saxpy"]
    assert dict(item.languages) == {}
    assert set(item.absent) == {"omp", "cuda"}


def test_empty_languages_without_absent_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"saxpy\.languages"):
        bench.load_suite(write_manifest(tmp_path, "    languages: {}\n"))


@pytest.mark.parametrize(
    "absent",
    [
        "absent: {cuda: 'SYNTHETIC: both'}",  # also under languages
        "absent: {}",  # empty
        "absent: [omp]",  # not a mapping
        "absent: {omp: ''}",  # an empty note
        "absent: {omp: '   '}",  # a blank note
        "absent: {omp: 3}",  # a note that is not text
        "absent: {omp: null}",  # no note
    ],
)
def test_a_malformed_absent_key_is_refused_naming_it(tmp_path: Path, absent: str) -> None:
    with pytest.raises(ValueError, match=r"saxpy\.absent"):
        bench.load_suite(write_manifest(tmp_path, f"    {absent}\n" + CUDA_ONLY))


def test_passfail_may_not_name_an_absent_language(tmp_path: Path) -> None:
    entry = f"    absent: {{omp: '{NOTE}'}}\n    passfail: [omp]\n" + CUDA_ONLY
    with pytest.raises(ValueError, match=r"passfail names 'omp'"):
        bench.load_suite(write_manifest(tmp_path, entry))


def test_reading_an_absent_language_raises_quoting_its_note(tmp_path: Path) -> None:
    suite = bench.load_suite(write_manifest(tmp_path, f"    absent: {{omp: '{NOTE}'}}\n" + CUDA_ONLY))
    to_omp, from_omp = bench.Direction("cuda", "omp"), bench.Direction("omp", "cuda")
    with pytest.raises(ValueError, match=re.escape(NOTE)):
        suite.reference_target("saxpy", to_omp, tmp_path, purpose="eval")
    with pytest.raises(ValueError, match=re.escape(NOTE)):
        suite.source_files("saxpy", from_omp, tmp_path, purpose="eval")
    with pytest.raises(ValueError, match=re.escape(NOTE)):
        suite.support_files("saxpy", tmp_path, purpose="eval", language="omp")


def test_a_language_neither_present_nor_absent_keeps_the_plain_refusal(tmp_path: Path) -> None:
    suite = bench.load_suite(write_manifest(tmp_path, f"    absent: {{omp: '{NOTE}'}}\n" + CUDA_ONLY))
    with pytest.raises(ValueError, match="has no 'hip' sources; languages: cuda"):
        suite.reference_target("saxpy", bench.Direction("cuda", "hip"), tmp_path, purpose="eval")


def test_an_eval_item_with_absent_languages_is_still_refused_to_training(tmp_path: Path) -> None:
    entry = f"    absent: {{omp: '{NOTE}', cuda: '{NOTE}'}}\n    languages: {{}}\n"
    suite = bench.load_suite(write_manifest(tmp_path, entry))
    with pytest.raises(bench.EvalSplitError, match="saxpy"):
        suite.item("saxpy", purpose="train")
    assert suite.item("saxpy", purpose="eval").name == "saxpy"
