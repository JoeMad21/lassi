"""Tests for the notebook's tiktoken similarity, sim_t_tiktoken, and its offline loader (task P4.15).

Bible: Evaluation Protocol (LASSI Paper Metrics, [OPEN] on the Sim-T
tokenizer; LASSI Score Profile), Source Papers (quirk table, Sim-T row);
OQ-022 (owner: keep both token similarities); Agent Rule 7 (the cache lives
under the scratch root).

The contract these tests fix (lassi.scoring.similarity):

- `sim_t_tiktoken(reference, candidate, encoding=None) -> float | None`:
  each text, the reference first, is encoded with the encoding's defaults;
  the value is difflib.SequenceMatcher(None, ids_ref, ids_cand).ratio(),
  autojunk on, as the notebook's "tiktoken" method computes it. A
  ValueError from encode (a special-token string in a text) or ids that do
  not decode back to the text, where the notebook raises, give None.
- `cl100k_base()` reads the encoding only from the directory
  TIKTOKEN_CACHE_DIR names. An unset or empty variable, a path that is not a
  directory, a missing cached file, or a file whose sha256 is not the one
  tiktoken expects raises TiktokenCacheError; nothing is downloaded (no
  socket opens), and no cached file is deleted. While tiktoken loads, its
  cached-file reader is replaced by `cached_reader(cache_dir)` and restored
  afterwards.
- `tiktoken_version()` is the installed tiktoken's version, the one
  pyproject.toml pins exactly.

No test reads the real cl100k_base file or opens a socket: the encodings here
are SYNTHETIC (a character encoding, and a small byte-level tiktoken
Encoding built in memory), and the offline path is driven through tiktoken's
own loader with a SYNTHETIC constructor and source address. The remote test
in test_similarity_tiktoken_remote.py checks the real file on the build
host. No value here is a measurement.
"""

from __future__ import annotations

import base64
import difflib
import hashlib
import re
import socket
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
import tiktoken
import tiktoken.load
import tiktoken.registry

from lassi.scoring import similarity
from lassi.scoring.similarity import (
    TIKTOKEN_CACHE_ENV,
    TiktokenCacheError,
    cached_reader,
    cl100k_base,
    sim_t_tiktoken,
    tiktoken_version,
)

pytestmark = pytest.mark.real_tiktoken  # the real offline loader, not tests/conftest.py's fake

REPO = Path(__file__).resolve().parents[2]
SPECIAL = "<|endoftext|>"
# A SYNTHETIC source address; the real one is read at run time from tiktoken and never written in this repository.
SOURCE = "https://example.invalid/synthetic-small.tiktoken"


class CharEncoding:
    """A SYNTHETIC encoding: one id per character; a special-token string is refused as tiktoken refuses it."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, text: str) -> list[int]:
        """Return one id per character; raise ValueError for a special-token string."""
        self.calls.append(text)
        if SPECIAL in text:
            raise ValueError("SYNTHETIC: disallowed special token")
        return [ord(char) for char in text]

    def decode(self, tokens: Sequence[int]) -> str:
        """Return the characters of the ids."""
        return "".join(chr(token) for token in tokens)


class LossyEncoding(CharEncoding):
    """A SYNTHETIC encoding whose decode turns a tab into a space, so a text with a tab does not round-trip."""

    def decode(self, tokens: Sequence[int]) -> str:
        """Return the characters of the ids with every tab written as a space."""
        return super().decode(tokens).replace("\t", " ")


def small_bpe() -> bytes:
    """Return a SYNTHETIC tiktoken BPE file: every single byte, then the merges 'in' and 'int'."""
    ranks = [bytes([value]) for value in range(256)] + [b"in", b"int"]
    return b"".join(base64.b64encode(token) + b" " + str(rank).encode() + b"\n" for rank, token in enumerate(ranks))


def small_constructor(expected_hash: str) -> Callable[[], dict[str, Any]]:
    """Return a SYNTHETIC encoding constructor that loads its ranks through tiktoken's own loader."""

    def construct() -> dict[str, Any]:
        ranks = tiktoken.load.load_tiktoken_bpe(SOURCE, expected_hash=expected_hash)
        return {"name": "cl100k_base", "pat_str": r"\w+|\s+|[^\w\s]+", "mergeable_ranks": ranks,
                "special_tokens": {SPECIAL: len(ranks)}}

    return construct


@pytest.fixture
def no_socket(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Refuse and record every socket open while the test runs."""
    attempts: list[str] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        attempts.append("socket")
        raise OSError("SYNTHETIC: sockets are refused in this test")

    for name in ("socket", "create_connection", "getaddrinfo"):
        monkeypatch.setattr(socket, name, refuse)
    return attempts


@pytest.fixture
def fresh_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give tiktoken an empty encoding cache and clear cl100k_base's, so every load goes through the loader."""
    monkeypatch.setattr(tiktoken.registry, "ENCODINGS", {})
    cl100k_base.cache_clear()
    yield
    cl100k_base.cache_clear()


def small_cache(tmp_path: Path, data: bytes) -> Path:
    """Return a cache directory holding `data` under the name tiktoken's cache rule gives SOURCE."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / hashlib.sha1(SOURCE.encode()).hexdigest()).write_bytes(data)
    return cache


# ---------------------------------------------------------------------------
# The value


def test_the_value_is_the_ratio_of_the_id_lists_reference_first() -> None:
    encoding = CharEncoding()
    reference, candidate = "int main() { return 0; }\n", "int main(void) { return 1; }\n"
    value = sim_t_tiktoken(reference, candidate, encoding)
    assert encoding.calls == [reference, candidate], "the reference is encoded first"
    ids = [[ord(char) for char in text] for text in (reference, candidate)]
    assert value == difflib.SequenceMatcher(None, ids[0], ids[1]).ratio()


def test_autojunk_is_on_as_in_the_notebook() -> None:
    # SYNTHETIC: 300 copies of one id in the candidate make it popular (more than 301 // 100 + 1 times in a
    # sequence of 200 or more), so autojunk drops it as a match seed; the first ids differ, so no match grows
    # from the start either, and nothing matches.
    reference, candidate = "y" + "x" * 50, "z" + "x" * 300
    ids = [[ord(char) for char in text] for text in (reference, candidate)]
    value = sim_t_tiktoken(reference, candidate, CharEncoding())
    assert value == difflib.SequenceMatcher(None, ids[0], ids[1]).ratio() == 0.0
    assert difflib.SequenceMatcher(None, ids[0], ids[1], autojunk=False).ratio() > 0.0


def test_two_empty_texts_score_one() -> None:
    assert sim_t_tiktoken("", "", CharEncoding()) == 1.0


@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_a_special_token_string_gives_none(side: str) -> None:
    texts = {"reference": "int a;\n", "candidate": "int b;\n"}
    texts[side] = f"// {SPECIAL}\n" + texts[side]
    assert sim_t_tiktoken(texts["reference"], texts["candidate"], CharEncoding()) is None


@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_a_text_that_does_not_decode_back_gives_none(side: str) -> None:
    texts = {"reference": "int a;\n", "candidate": "int b;\n"}
    texts[side] = "\t" + texts[side]
    assert sim_t_tiktoken(texts["reference"], texts["candidate"], LossyEncoding()) is None
    assert sim_t_tiktoken("int a;\n", "int b;\n", LossyEncoding()) is not None


def test_a_real_tiktoken_encoding_refuses_special_tokens_by_default() -> None:
    """A small byte-level tiktoken Encoding built in memory: tiktoken's own default refusal gives None."""
    mergeable = {base64.b64decode(line.split()[0]): int(line.split()[1]) for line in small_bpe().splitlines()}
    encoding = tiktoken.Encoding(name="synthetic-small", pat_str=r"\w+|\s+|[^\w\s]+", mergeable_ranks=mergeable,
                                 special_tokens={SPECIAL: len(mergeable)})
    reference, candidate = "int main() { return 0; }\n", "int main(void) { return 0; }\n"
    ids = [encoding.encode(text) for text in (reference, candidate)]
    assert sim_t_tiktoken(reference, candidate, encoding) == difflib.SequenceMatcher(None, *ids).ratio()
    assert sim_t_tiktoken(reference + SPECIAL, candidate, encoding) is None


# ---------------------------------------------------------------------------
# The offline loader


@pytest.mark.parametrize("value", [None, "", "   "])
def test_an_unset_or_empty_cache_variable_is_refused(
    value: str | None, monkeypatch: pytest.MonkeyPatch, no_socket: list[str], fresh_registry: None
) -> None:
    if value is None:
        monkeypatch.delenv(TIKTOKEN_CACHE_ENV, raising=False)
    else:
        monkeypatch.setenv(TIKTOKEN_CACHE_ENV, value)
    with pytest.raises(TiktokenCacheError, match=TIKTOKEN_CACHE_ENV):
        cl100k_base()
    assert no_socket == []


def test_a_cache_path_that_is_not_a_directory_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_socket: list[str], fresh_registry: None
) -> None:
    monkeypatch.setenv(TIKTOKEN_CACHE_ENV, str(tmp_path / "absent"))
    with pytest.raises(TiktokenCacheError, match="not a directory"):
        cl100k_base()
    assert no_socket == []


def test_the_offline_loader_reads_the_cached_file_through_tiktoken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_socket: list[str], fresh_registry: None
) -> None:
    data = small_bpe()
    cache = small_cache(tmp_path, data)
    monkeypatch.setenv(TIKTOKEN_CACHE_ENV, str(cache))
    monkeypatch.setattr(tiktoken.registry, "ENCODING_CONSTRUCTORS",
                        {"cl100k_base": small_constructor(hashlib.sha256(data).hexdigest())})
    original = tiktoken.load.read_file_cached
    encoding = cl100k_base()
    assert tiktoken.load.read_file_cached is original, "tiktoken's reader is restored after the load"
    assert encoding.decode(encoding.encode("int x = 1;\n")) == "int x = 1;\n"
    assert sim_t_tiktoken("int a;\n", "int a;\n") == 1.0, "the default encoding is the one cl100k_base() loaded"
    assert no_socket == []
    assert sorted(path.name for path in cache.iterdir()) == [hashlib.sha1(SOURCE.encode()).hexdigest()]


def test_a_missing_cached_file_is_refused_without_a_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_socket: list[str], fresh_registry: None
) -> None:
    cache = tmp_path / "empty-cache"
    cache.mkdir()
    monkeypatch.setenv(TIKTOKEN_CACHE_ENV, str(cache))
    monkeypatch.setattr(tiktoken.registry, "ENCODING_CONSTRUCTORS", {"cl100k_base": small_constructor("0" * 64)})
    original = tiktoken.load.read_file_cached
    with pytest.raises(TiktokenCacheError, match="never downloads") as refused:
        cl100k_base()
    assert SOURCE not in str(refused.value), "a message never names the source address"
    assert tiktoken.load.read_file_cached is original
    assert no_socket == [] and list(cache.iterdir()) == [], "nothing is downloaded or written"


def test_a_cached_file_with_another_hash_is_refused_and_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_socket: list[str], fresh_registry: None
) -> None:
    cache = small_cache(tmp_path, small_bpe())
    monkeypatch.setenv(TIKTOKEN_CACHE_ENV, str(cache))
    monkeypatch.setattr(tiktoken.registry, "ENCODING_CONSTRUCTORS", {"cl100k_base": small_constructor("0" * 64)})
    with pytest.raises(TiktokenCacheError, match="left in place"):
        cl100k_base()
    kept = cache / hashlib.sha1(SOURCE.encode()).hexdigest()
    assert kept.read_bytes() == small_bpe(), "the cached file is never deleted"
    assert no_socket == []


def test_the_cached_reader_uses_tiktokens_cache_rule(tmp_path: Path) -> None:
    data = small_bpe()
    cache = small_cache(tmp_path, data)
    read = cached_reader(cache)
    assert read(SOURCE, hashlib.sha256(data).hexdigest()) == data
    assert read(SOURCE, None) == data
    with pytest.raises(TiktokenCacheError) as refused:
        read(SOURCE + "-other", None)
    assert hashlib.sha1((SOURCE + "-other").encode()).hexdigest() in str(refused.value)
    assert SOURCE not in str(refused.value)


# ---------------------------------------------------------------------------
# The pin


def test_the_installed_tiktoken_is_the_exact_version_pyproject_pins() -> None:
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    pins = re.findall(r'"tiktoken==([0-9][0-9A-Za-z.]*)"', text)
    assert len(pins) == 1, f"pyproject.toml pins tiktoken exactly once, got {pins}"
    assert tiktoken_version() == pins[0]


def test_the_module_names_no_source_address() -> None:
    source = Path(similarity.__file__).read_text(encoding="utf-8")
    assert "://" not in source, "the encoding's source address is read at run time, never written here"
