"""A lexical retriever over a documentation index: BM25 over chunks (owner direction of 2026-10-23).

The Tenstorrent documentation retriever is the fallback beside the fixed
API-reference context pack: later work may let a recipe opt in to it in the
generation and correction prompts. This module is only the retriever. No
stage calls it yet.

An index is one generated JSON file, built by GENERATOR from a pinned
documentation tree (lassi.prompts.rag_corpus cuts the files into chunks and
applies the Agent Rule 5 exclusions) and never tracked; the tracked manifest
beside its corpus spec records the index's sha256. It holds `format`,
`tokenizer`, `meta` (the corpus name, its source, the chunking parameters,
and `bm25`: {k1, b}), and `chunks`, each {path, start, end, heading, text}:
a file's relative POSIX path, the 1-based first and last line the chunk
covers, the heading path it sits under ("" when none), and the exact text of
those lines joined by newlines.

Retrieval is deterministic and auditable, with no model: the same index and
query always give the same ranked passages. A chunk's terms are the tokens
(tokenize) of its path, heading, and text. A query's distinct tokens score
each chunk by BM25, with idf = ln(1 + (N - df + 0.5) / (df + 0.5)); each score
is rounded to SCORE_DIGITS decimals, and ties are broken by the chunk's path,
then its first line, the order the chunks are stored in.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lassi.core.record import Diagnostic

# The tool that builds an index; every refusal of an index names it.
GENERATOR = "tools/build_tt_rag.py"
# The index file layout and the tokenizer below; an index written for another of either is refused.
INDEX_FORMAT = 1
TOKENIZER = "words-and-identifier-parts-v1"
# Scores are rounded to this many decimals before ranking, so a last-bit difference in math.log is unlikely to
# reorder them; it still can when two scores lie within that last bit of a rounding boundary, so it is not impossible.
SCORE_DIGITS = 6
# The keys of one chunk in an index file.
CHUNK_KEYS = ("end", "heading", "path", "start", "text")

# A word: letters, digits, and underscores. Its parts: underscore pieces split at case changes and digit runs.
_WORD = re.compile(r"[A-Za-z0-9_]+")
_PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
# A part shorter than this is not a token of its own (the whole word still is).
_MIN_PART = 2
# A quoted span in a compiler message ('x', the typographic quotes gcc prints, "x", or `x`), and a C++ name.
_QUOTED = re.compile("'([^']+)'|\u2018([^\u2019]+)\u2019|\"([^\"]+)\"|`([^`]+)`")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*")
_CAMEL = re.compile(r"[a-z0-9][A-Z]")


@dataclass(frozen=True)
class Chunk:
    """One indexed span of a source file: its path, first and last line (1-based), heading path, and exact text."""

    path: str
    start_line: int
    end_line: int
    heading: str
    text: str


@dataclass(frozen=True)
class Passage:
    """One search result: the chunk's path, line range, heading path, and text, with its rounded BM25 score."""

    path: str
    start_line: int
    end_line: int
    heading: str
    text: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        """Return {path, lines: [first, last], heading, text, score}, for a prompt builder or a record."""
        return {
            "path": self.path,
            "lines": [self.start_line, self.end_line],
            "heading": self.heading,
            "text": self.text,
            "score": self.score,
        }


def tokenize(text: str) -> list[str]:
    """Return the lower-case tokens of `text`, in order: each word, then its parts when it has several.

    A word is a run of letters, digits, and underscores, so a C++ name such
    as tilize_nfaces or CreateCircularBuffer stays one token, which a query
    for that name matches exactly. Its parts (split at underscores, case
    changes, and digit runs; at least two characters) follow it, so a query
    in words (circular buffer) also matches the name.
    """
    tokens: list[str] = []
    for word in _WORD.findall(text):
        lower = word.lower()
        tokens.append(lower)
        parts = [part.lower() for piece in word.split("_") for part in _PART.findall(piece)]
        if len(parts) > 1:
            tokens.extend(part for part in parts if len(part) >= _MIN_PART)
    return tokens


class Index:
    """A BM25 index over chunks, stored in (path, first line, last line) order; search() ranks them."""

    def __init__(self, chunks: Sequence[Chunk], meta: Mapping[str, Any]) -> None:
        """Check the chunks and meta (whose `bm25` gives k1 > 0 and b in [0, 1]) and build the postings.

        Raises ValueError naming the first chunk or meta field that is not
        what an index holds, and for two chunks with one path and first line.
        """
        self.meta = _checked_meta(meta)
        self.chunks = tuple(sorted((_checked_chunk(chunk) for chunk in chunks), key=_chunk_key))
        keys = [(chunk.path, chunk.start_line) for chunk in self.chunks]
        if len(set(keys)) != len(keys):
            raise ValueError("two chunks share one path and first line; an index holds each span once")
        self.sha256 = hashlib.sha256(index_bytes(self.chunks, self.meta)).hexdigest()
        self._k1 = float(self.meta["bm25"]["k1"])
        self._b = float(self.meta["bm25"]["b"])
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._lengths: list[int] = []
        for doc, chunk in enumerate(self.chunks):
            counts = Counter(tokenize(chunk.path) + tokenize(chunk.heading) + tokenize(chunk.text))
            self._lengths.append(sum(counts.values()))
            for term in sorted(counts):
                self._postings.setdefault(term, []).append((doc, counts[term]))
        total = len(self.chunks)
        self._average = (sum(self._lengths) / total) if total else 0.0
        self._idf = {
            term: math.log(1.0 + (total - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }

    def __len__(self) -> int:
        """Return the number of chunks."""
        return len(self.chunks)

    def search(self, query: str, k: int) -> list[Passage]:
        """Return the `k` best passages for `query`, best first; fewer when fewer chunks share a term with it.

        Each distinct query token adds its BM25 term score to every chunk
        that holds it. Scores are rounded to SCORE_DIGITS decimals; equal
        scores keep the stored order (path, then first line). A chunk with
        no query token is never returned. Raises TypeError for a query that
        is not a str and ValueError for a k that is not an int of at least 1.
        """
        if not isinstance(query, str):
            raise TypeError(f"a query must be a str, got {type(query).__name__}")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be an int of at least 1, got {k!r}")
        scores: dict[int, float] = {}
        for term in sorted(set(tokenize(query))):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for doc, count in postings:
                norm = count + self._k1 * (1.0 - self._b + self._b * self._lengths[doc] / self._average)
                scores[doc] = scores.get(doc, 0.0) + idf * count * (self._k1 + 1.0) / norm
        ranked = sorted(((round(score, SCORE_DIGITS), doc) for doc, score in scores.items()),
                        key=lambda pair: (-pair[0], pair[1]))
        return [_passage(self.chunks[doc], score) for score, doc in ranked[:k] if score > 0.0]


def index_bytes(chunks: Sequence[Chunk], meta: Mapping[str, Any]) -> bytes:
    """Return the canonical bytes of an index file: sorted-key ASCII JSON, chunks in stored order, one newline.

    The same chunks and meta always give the same bytes, so the manifest's
    sha256 of the index pins it exactly.
    """
    ordered = sorted(chunks, key=_chunk_key)
    document = {
        "format": INDEX_FORMAT,
        "tokenizer": TOKENIZER,
        "meta": dict(meta),
        "chunks": [
            {"path": c.path, "start": c.start_line, "end": c.end_line, "heading": c.heading, "text": c.text}
            for c in ordered
        ],
    }
    return (json.dumps(document, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n").encode("ascii")


def load_index(path: Path | str, *, sha256: str | None = None) -> Index:
    """Load the index file at `path`; with `sha256` (the manifest's), refuse a file with another digest.

    Raises ValueError naming GENERATOR when the file cannot be read, has
    another sha256, is not an index of INDEX_FORMAT and TOKENIZER, holds a
    malformed chunk, or is not in the canonical form index_bytes writes.
    """
    path = Path(path)
    rebuild = f"rebuild it with uv run {GENERATOR} build"
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read the index {path} ({error}); {rebuild}") from error
    digest = hashlib.sha256(data).hexdigest()
    if sha256 is not None and digest != sha256:
        raise ValueError(f"{path} has sha256 {digest}, not the manifest's {sha256}; {rebuild}")
    try:
        document = json.loads(data.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is not an index file ({error}); {rebuild}") from error
    if not isinstance(document, dict) or sorted(document) != ["chunks", "format", "meta", "tokenizer"]:
        raise ValueError(f"{path} is not an index file: its keys must be chunks, format, meta, tokenizer; {rebuild}")
    if document["format"] != INDEX_FORMAT or document["tokenizer"] != TOKENIZER:
        raise ValueError(f"{path} is an index of format {document['format']!r} and tokenizer "
                         f"{document['tokenizer']!r}, not {INDEX_FORMAT} and {TOKENIZER!r}; {rebuild}")
    try:
        index = Index([_chunk_from(row) for row in _rows(document["chunks"])], document["meta"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {error}; {rebuild}") from error
    if index.sha256 != digest:
        raise ValueError(f"{path} is not in the canonical form the generator writes; {rebuild}")
    return index


def query_from_diagnostic(diagnostic: Diagnostic) -> str:
    """Return a search query for one compile Diagnostic: its identifiers, its message, and its file's name.

    The identifiers are the C++ names quoted in the message ('x', the
    typographic quotes gcc prints, "x", or `x`) and every other name in it
    that holds an underscore, a scope (::), or an inner capital, each once,
    in order. The file is the diagnostic's file name without its directory,
    which is the build directory's and says nothing about the API.
    """
    message = diagnostic.message
    if not isinstance(message, str):
        raise TypeError(f"a diagnostic's message must be a str, got {type(message).__name__}")
    words = [*_identifiers(message), message]
    if diagnostic.file:
        words.append(PurePosixPath(diagnostic.file.replace("\\", "/")).name)
    return " ".join(words)


def _identifiers(message: str) -> list[str]:
    """Return the quoted names of `message`, then its other code-like names, each once."""
    names: list[str] = []
    for match in _QUOTED.finditer(message):
        quoted = next(group for group in match.groups() if group is not None)
        names.extend(_IDENT.findall(quoted))
    names.extend(word for word in _IDENT.findall(message)
                 if "_" in word or "::" in word or _CAMEL.search(word))
    return list(dict.fromkeys(names))


def _passage(chunk: Chunk, score: float) -> Passage:
    """Return the Passage of `chunk` at `score`."""
    return Passage(chunk.path, chunk.start_line, chunk.end_line, chunk.heading, chunk.text, score)


def _chunk_key(chunk: Chunk) -> tuple[str, int, int]:
    """Return the stored order and tie-break key of a chunk: path, first line, last line."""
    return (chunk.path, chunk.start_line, chunk.end_line)


def _rows(value: Any) -> list[Mapping[str, Any]]:
    """Return the chunk rows of an index document; raise ValueError unless it is a list of objects."""
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("chunks must be a list of objects")
    return value


def _chunk_from(row: Mapping[str, Any]) -> Chunk:
    """Return the Chunk of one index row; raise ValueError unless its keys are exactly CHUNK_KEYS."""
    if tuple(sorted(row)) != CHUNK_KEYS:
        raise ValueError(f"a chunk's keys must be {', '.join(CHUNK_KEYS)}, got {', '.join(sorted(row))}")
    return Chunk(row["path"], row["start"], row["end"], row["heading"], row["text"])


def _checked_chunk(chunk: Chunk) -> Chunk:
    """Return `chunk` when its fields are what an index holds; else raise ValueError naming the field."""
    if not isinstance(chunk, Chunk):
        raise ValueError(f"an index holds Chunk values, got {type(chunk).__name__}")
    where = f"chunk {chunk.path!r}"
    path = chunk.path
    if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path \
            or any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError(f"{where}: the path must be a relative POSIX path")
    for name in ("start_line", "end_line"):
        value = getattr(chunk, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{where}: {name} must be an int of at least 1, got {value!r}")
    if chunk.end_line < chunk.start_line:
        raise ValueError(f"{where}: end_line {chunk.end_line} is before start_line {chunk.start_line}")
    if not isinstance(chunk.heading, str) or not isinstance(chunk.text, str) or not chunk.text.strip():
        raise ValueError(f"{where}: heading must be a str and text a non-blank str")
    if chunk.text.count("\n") != chunk.end_line - chunk.start_line:
        raise ValueError(f"{where}: its text must hold one line per line of its range")
    return chunk


def _checked_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of `meta` when its `bm25` gives k1 > 0 and b in [0, 1]; else raise ValueError."""
    if not isinstance(meta, Mapping):
        raise ValueError(f"meta must be a mapping, got {type(meta).__name__}")
    bm25 = meta.get("bm25")
    k1 = bm25.get("k1") if isinstance(bm25, Mapping) else None
    b = bm25.get("b") if isinstance(bm25, Mapping) else None
    numbers = all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (k1, b))
    if not numbers or not (k1 > 0 and 0 <= b <= 1):
        raise ValueError(f"meta.bm25 must give k1 > 0 and b in [0, 1], got {bm25!r}")
    return json.loads(json.dumps(dict(meta), sort_keys=True))
