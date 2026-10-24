"""Tests for lassi.prompts.retrieval, the lexical documentation retriever (owner direction of 2026-10-23).

The contract these tests fix: tokenize keeps a C++ name whole and adds its
parts; Index.search ranks chunks by BM25 over their path, heading, and text,
returns at most k Passages (path, line range, heading, text, rounded score),
best first, ties in (path, first line) order, and is the same for the same
index and query however the index was built or loaded; index_bytes is
canonical and load_index refuses a file with another sha256, another format
or tokenizer, a malformed chunk, or a non-canonical form, naming
tools/build_tt_rag.py; query_from_diagnostic gives a query of the
diagnostic's identifiers, its message, and its file's name.

Every corpus here is SYNTHETIC text written for these tests; no upstream
documentation text is copied. No value here is a measurement.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from lassi.core.record import Diagnostic
from lassi.prompts.retrieval import (
    GENERATOR,
    INDEX_FORMAT,
    TOKENIZER,
    Chunk,
    Index,
    Passage,
    index_bytes,
    load_index,
    query_from_diagnostic,
    tokenize,
)

META = {"corpus": "synthetic", "bm25": {"k1": 1.2, "b": 0.75}}

# SYNTHETIC chunks, one topic each, so each known-answer query has one right first passage.
CORPUS = (
    Chunk("docs/circular_buffers.md", 1, 3, "Circular buffers",
          "A circular buffer holds tiles between two kernels on one core.\n"
          "A consumer calls cb_wait_front before it reads a tile,\n"
          "then cb_pop_front to free the slot for the producer."),
    Chunk("docs/noc.md", 4, 6, "Reads over the NoC",
          "noc_async_read copies bytes from a NoC address into local memory.\n"
          "Call noc_async_read_barrier before the kernel uses the data.\n"
          "Data movement processors issue every read."),
    Chunk("docs/semaphores.rst", 10, 11, "Semaphores",
          "A semaphore lets two cores signal each other.\n"
          "noc_semaphore_set writes a value and noc_semaphore_wait spins until it changes."),
    Chunk("docs/dram.rst", 1, 2, "DRAM",
          "A DRAM buffer is interleaved across the banks one page at a time.\n"
          "The page size is usually one tile."),
    Chunk("api/tile_layout.hpp", 20, 22, "",
          "// Rearranges a row-major matrix into tile order.\n"
          "template <typename T>\n"
          "std::vector<T> tilize_rows(const std::vector<T>& input, uint32_t m, uint32_t n);"),
)


def corpus_index(chunks: Sequence[Chunk] = CORPUS) -> Index:
    """Return an Index of `chunks` with META."""
    return Index(list(chunks), META)


def write_index(tmp_path: Path, chunks: Sequence[Chunk] = CORPUS,
                meta: Mapping[str, Any] = META) -> tuple[Path, str]:
    """Write the canonical index of `chunks` under tmp_path; return its path and sha256."""
    data = index_bytes(list(chunks), meta)
    path = tmp_path / "index.json"
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def keys(passages: list[Passage]) -> list[tuple[str, int]]:
    """Return (path, first line) per passage."""
    return [(p.path, p.start_line) for p in passages]


def test_tokenize_keeps_a_name_whole_and_adds_its_parts() -> None:
    tokens = tokenize("tilize_nfaces CreateCircularBuffer")
    assert tokens[:3] == ["tilize_nfaces", "tilize", "nfaces"]
    assert tokens[3:] == ["createcircularbuffer", "create", "circular", "buffer"]


def test_tokenize_lowers_case_and_drops_one_letter_parts() -> None:
    assert tokenize("Noc_Async_READ") == ["noc_async_read", "noc", "async", "read"]
    assert tokenize("c_0 uint32_t") == ["c_0", "uint32_t", "uint", "32"]
    assert tokenize("a plain sentence.") == ["a", "plain", "sentence"]


@pytest.mark.parametrize(("query", "expected"), [
    ("how do I wait for a tile in a circular buffer", "docs/circular_buffers.md"),
    ("noc_async_read_barrier", "docs/noc.md"),
    ("semaphore wait between cores", "docs/semaphores.rst"),
    ("interleaved DRAM banks page size", "docs/dram.rst"),
    ("tilize_rows", "api/tile_layout.hpp"),
])
def test_known_answer_queries_rank_their_chunk_first(query: str, expected: str) -> None:
    passages = corpus_index().search(query, 3)
    assert passages, query
    assert passages[0].path == expected


def test_a_passage_carries_the_chunk_range_text_and_a_rounded_positive_score() -> None:
    passage = corpus_index().search("noc_async_read", 1)[0]
    chunk = CORPUS[1]
    assert (passage.path, passage.start_line, passage.end_line) == (chunk.path, chunk.start_line, chunk.end_line)
    assert passage.text == chunk.text and passage.heading == chunk.heading
    assert passage.score > 0 and passage.score == round(passage.score, 6)
    assert passage.as_dict() == {"path": chunk.path, "lines": [4, 6], "heading": chunk.heading,
                                 "text": chunk.text, "score": passage.score}


def test_scores_never_increase_down_the_ranking() -> None:
    passages = corpus_index().search("noc read tile buffer semaphore page", 5)
    scores = [p.score for p in passages]
    assert scores == sorted(scores, reverse=True) and len(passages) == 5


def test_search_is_the_same_for_the_same_index_and_query_however_built(tmp_path: Path) -> None:
    query = "read a tile from the circular buffer over the noc"
    forward = corpus_index()
    backward = corpus_index(tuple(reversed(CORPUS)))
    path, digest = write_index(tmp_path)
    loaded = load_index(path, sha256=digest)
    results = [index.search(query, 5) for index in (forward, forward, backward, loaded)]
    assert all(result == results[0] for result in results)
    assert forward.sha256 == backward.sha256 == loaded.sha256 == digest
    assert [c.path for c in forward.chunks] == sorted(c.path for c in CORPUS)


def test_equal_scores_keep_path_then_first_line_order() -> None:
    text = "The fabric router forwards packets between chips."
    twins = [Chunk("b/x.md", 1, 1, "", text), Chunk("a/x.md", 5, 5, "", text), Chunk("a/x.md", 1, 1, "", text)]
    passages = corpus_index(twins).search("fabric router packets", 3)
    assert keys(passages) == [("a/x.md", 1), ("a/x.md", 5), ("b/x.md", 1)]
    assert len({p.score for p in passages}) == 1


def test_k_bounds_the_results_and_unknown_terms_give_none() -> None:
    index = corpus_index()
    assert len(index.search("tile", 1)) == 1
    assert index.search("zyzzyva quokka", 5) == []
    assert index.search("", 5) == []


@pytest.mark.parametrize("k", [0, -1, True, 1.5, "3"])
def test_k_must_be_an_int_of_at_least_one(k: object) -> None:
    with pytest.raises(ValueError, match="k must be an int"):
        corpus_index().search("tile", k)


def test_a_query_must_be_text() -> None:
    with pytest.raises(TypeError, match="a query must be a str"):
        corpus_index().search(b"tile", 1)


def test_index_bytes_are_canonical_ascii_json(tmp_path: Path) -> None:
    unicode_chunk = Chunk("docs/u.md", 1, 1, "", "Copyright " + chr(0xA9) + " the authors")
    data = index_bytes([*CORPUS, unicode_chunk], META)
    assert data == index_bytes([unicode_chunk, *reversed(CORPUS)], META)
    assert data.endswith(b"\n") and data.isascii()
    document = json.loads(data)
    assert document["format"] == INDEX_FORMAT and document["tokenizer"] == TOKENIZER
    assert [row["path"] for row in document["chunks"]] == sorted(c.path for c in [*CORPUS, unicode_chunk])


def test_load_index_refuses_another_sha256(tmp_path: Path) -> None:
    path, _ = write_index(tmp_path)
    with pytest.raises(ValueError, match=f"not the manifest's {'0' * 64}.*{GENERATOR}"):
        load_index(path, sha256="0" * 64)


def test_load_index_refuses_a_tampered_text(tmp_path: Path) -> None:
    path, digest = write_index(tmp_path)
    path.write_bytes(path.read_bytes().replace(b"cb_pop_front", b"cb_pop_back!"))
    with pytest.raises(ValueError, match="sha256"):
        load_index(path, sha256=digest)


def test_load_index_refuses_a_non_canonical_file(tmp_path: Path) -> None:
    path, _ = write_index(tmp_path)
    path.write_text(json.dumps(json.loads(path.read_bytes()), indent=2), encoding="ascii")
    with pytest.raises(ValueError, match=f"canonical form.*{GENERATOR}"):
        load_index(path)


@pytest.mark.parametrize(("key", "value"), [("format", INDEX_FORMAT + 1), ("tokenizer", "another-tokenizer")])
def test_load_index_refuses_another_format_or_tokenizer(tmp_path: Path, key: str, value: int | str) -> None:
    path, _ = write_index(tmp_path)
    document = json.loads(path.read_bytes())
    document[key] = value
    path.write_bytes(json.dumps(document, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n")
    with pytest.raises(ValueError, match=f"not {INDEX_FORMAT} and '{TOKENIZER}'"):
        load_index(path)


@pytest.mark.parametrize("change", [
    {"end": 2},
    {"start": 0, "end": 0},
    {"path": "/abs/x.md"},
    {"path": "docs/../x.md"},
    {"text": "   "},
    {"extra": 1},
])
def test_load_index_refuses_a_malformed_chunk(tmp_path: Path, change: dict) -> None:
    path, _ = write_index(tmp_path, chunks=[Chunk("docs/x.md", 1, 1, "", "one line")])
    document = json.loads(path.read_bytes())
    document["chunks"][0].update(change)
    path.write_bytes(json.dumps(document, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n")
    with pytest.raises(ValueError, match=GENERATOR):
        load_index(path)


def test_an_index_refuses_bad_bm25_parameters_and_repeated_spans() -> None:
    with pytest.raises(ValueError, match="bm25"):
        Index(list(CORPUS), {"bm25": {"k1": 0, "b": 0.75}})
    with pytest.raises(ValueError, match="bm25"):
        Index(list(CORPUS), {"bm25": {"k1": 1.2, "b": 1.5}})
    with pytest.raises(ValueError, match="share one path and first line"):
        Index([CORPUS[0], CORPUS[0]], META)


def diagnostic(message: str, file: str | None = "/work/build/kernels/compute/mm.cpp") -> Diagnostic:
    """Return a compile-stage error Diagnostic with `message` and `file`."""
    return Diagnostic(stage="compile", severity="error", file=file, line=12, column=5, message=message)


def test_a_diagnostic_query_holds_its_identifiers_message_and_file_name() -> None:
    message = "use of undeclared identifier 'tilize_nfaces'"
    query = query_from_diagnostic(diagnostic(message))
    assert query == f"tilize_nfaces {message} mm.cpp"


def test_a_diagnostic_query_reads_typographic_quotes_and_scopes() -> None:
    left, right = chr(0x2018), chr(0x2019)
    message = f"{left}CreateCircularBufer{right} is not a member of {left}tt::tt_metal{right}"
    query = query_from_diagnostic(diagnostic(message, file=None))
    assert query == f"CreateCircularBufer tt::tt_metal {message}"


def test_a_diagnostic_query_keeps_unquoted_code_names_once() -> None:
    message = "no matching call to CreateKernel; noc_async_read takes 3 arguments, noc_async_read was given 2"
    query = query_from_diagnostic(diagnostic(message, file="C:\\work\\reader.cpp"))
    assert query == f"CreateKernel noc_async_read {message} reader.cpp"


def test_a_diagnostic_query_finds_the_declaration_it_names() -> None:
    declaration = Chunk("api/tilize_utils.hpp", 30, 32, "",
                        "// Converts a row-major matrix of m x n elements to tiled order.\n"
                        "template <typename T>\n"
                        "std::vector<T> tilize_nfaces(const std::vector<T>& input, uint32_t m, uint32_t n);")
    distractor = Chunk("docs/tiles.md", 1, 1, "Tiles", "Tilize a matrix before the device reads it as tiles.")
    index = corpus_index([*CORPUS, declaration, distractor])
    query = query_from_diagnostic(diagnostic("no member named 'tilize_nfaces' in namespace 'tt::tt_metal'"))
    assert keys(index.search(query, 2))[0] == ("api/tilize_utils.hpp", 30)
