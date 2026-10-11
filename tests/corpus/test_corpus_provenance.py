"""Tests for the corpus record's provenance hash chain, moved from v0's manifest (task P5.11).

Bible: Review Of v0 (the Keep row: manifest with parent-child hashes, moved
into lassi/corpus/), Corpus Pipeline (Record: provenance hashes), Agent
Rules 1 and 10. Plan: plans/p5-ir.md, task P5.11; plans/spikes/p5-v0-corpus.md,
Results 5, the migration map's manifest row: the source sha256 over the
file's bytes, then each step's output hash with its parent's, source to
raised IR to normalized IR to lowered program, so a record names the chain
that produced it. v0 hashed text decoded with replacement characters
(tools/parse_snapshots.py:92 at 3ccd280), which differs from the file's
sha256 for a non-UTF-8 source; the move hashes bytes. Test: a record's
hashes equal the sha256 of the fixture files' bytes, parent links form one
chain, and a record with a broken link is refused.

The contract these tests fix (lassi.corpus):

- HashLink(step=<str>, sha256=<64 lowercase hex>, parent=<64 hex or None>)
  is frozen; a sha256 or parent that is not 64 lowercase hex characters, or
  an empty step, raises ValueError.
- hash_chain([(step, bytes), ...]) returns one HashLink per output, in
  order: sha256 is hashlib's digest of that output's bytes, parent is the
  previous link's sha256, and the first link (the source) has parent None.
  An output that is not bytes (a str among them, so no caller hashes
  decoded text) raises TypeError or ValueError.
- check_chain(links) returns when the links form one chain and raises
  ValueError otherwise: no link, a first link with a parent, or a later
  link whose parent is not the previous link's sha256 (None included),
  the message naming that link's step.
- A CorpusRecord checks its provenance with the same rule, so a record
  with a broken chain is refused.

Every byte string is SYNTHETIC. No value here is a measurement.
"""

from __future__ import annotations

import dataclasses
import hashlib

import pytest
from corpus_fixtures import SOURCE_TEXT, corpus, cpu_ir, make_record, tt_ir

# SYNTHETIC source bytes that are not UTF-8: v0's replace-decoding would change their hash.
NON_UTF8_SOURCE = b"/* SYNTHETIC */ int x = 1; /* \xff\xfe not UTF-8 */\n"


def sha(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def outputs() -> list[tuple[str, bytes]]:
    """Return SYNTHETIC outputs of a source, raise, normalize, and lower, each as bytes."""
    return [
        ("source", NON_UTF8_SOURCE),
        ("raise", ("// SYNTHETIC raised form\n" + cpu_ir()).encode("utf-8")),
        ("normalize", cpu_ir().encode("utf-8")),
        ("lower", tt_ir().encode("utf-8")),
    ]


def test_the_link_is_a_frozen_record_of_step_hash_and_parent() -> None:
    module = corpus()
    assert dataclasses.is_dataclass(module.HashLink) and module.HashLink.__dataclass_params__.frozen
    assert [item.name for item in dataclasses.fields(module.HashLink)] == ["step", "sha256", "parent"]


def test_each_hash_is_the_sha256_of_the_outputs_bytes_and_parents_form_one_chain() -> None:
    module = corpus()
    given = outputs()
    links = list(module.hash_chain(given))
    digests = [sha(data) for _, data in given]
    assert links == [
        module.HashLink(step=step, sha256=digest, parent=digests[index - 1] if index else None)
        for index, ((step, _), digest) in enumerate(zip(given, digests, strict=True))
    ]
    module.check_chain(links)


def test_the_source_hash_is_over_the_files_bytes_not_replace_decoded_text() -> None:
    (source, *_) = corpus().hash_chain([("source", NON_UTF8_SOURCE)])
    assert source.sha256 == sha(NON_UTF8_SOURCE)
    assert source.sha256 != sha(NON_UTF8_SOURCE.decode("utf-8", errors="replace").encode("utf-8"))


def test_hashing_decoded_text_is_refused() -> None:
    with pytest.raises((TypeError, ValueError)):
        corpus().hash_chain([("source", SOURCE_TEXT)])


@pytest.mark.parametrize(
    "changes",
    [{"sha256": "abc"}, {"sha256": "A" * 64}, {"parent": "xyz"}, {"step": ""}],
    ids=["short-hash", "upper-case-hash", "bad-parent", "empty-step"],
)
def test_a_link_with_a_bad_hash_or_step_is_refused(changes: dict[str, str]) -> None:
    fields = {"step": "raise", "sha256": "0" * 64, "parent": "1" * 64, **changes}
    with pytest.raises(ValueError):
        corpus().HashLink(**fields)


def broken(kind: str) -> list[object]:
    """Return a SYNTHETIC chain broken as `kind` says; the broken link's step is `normalize` where one is."""
    module = corpus()
    links = list(module.hash_chain(outputs()))
    if kind == "empty":
        return []
    if kind == "first-has-a-parent":
        return [dataclasses.replace(links[0], parent="2" * 64), *links[1:]]
    if kind == "wrong-parent":
        links[2] = dataclasses.replace(links[2], parent="3" * 64)
    elif kind == "missing-parent":
        links[2] = dataclasses.replace(links[2], parent=None)
    elif kind == "link-left-out":
        del links[1]
    return links


@pytest.mark.parametrize("kind", ["empty", "first-has-a-parent", "wrong-parent", "missing-parent", "link-left-out"])
def test_a_broken_chain_is_refused(kind: str) -> None:
    links = broken(kind)
    with pytest.raises(ValueError) as refused:
        corpus().check_chain(links)
    named = {"first-has-a-parent": "source", "wrong-parent": "normalize", "missing-parent": "normalize"}
    if kind in named:
        assert named[kind] in str(refused.value), str(refused.value)


@pytest.mark.parametrize("kind", ["empty", "first-has-a-parent", "wrong-parent", "link-left-out"])
def test_a_record_with_a_broken_chain_is_refused(kind: str) -> None:
    links = broken(kind)
    with pytest.raises(ValueError):
        make_record(provenance=links)


def test_a_record_keeps_its_chain() -> None:
    module = corpus()
    links = list(module.hash_chain(outputs()))
    assert list(make_record(provenance=links).provenance) == links
