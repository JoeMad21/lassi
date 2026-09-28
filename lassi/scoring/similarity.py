"""Sim-T and Sim-L: code similarity between a reference target and a candidate.

Bible: Source Papers (LASSI results table; quirk table, Sim-T row), Evaluation
Protocol (LASSI reproduction row), Repository Layout (`scoring/`).

Every measure takes the reference target first and the candidate second, the
order the LASSI notebook uses. The Sim-T ratios need not be symmetric, since
difflib's block matching depends on the order. Sim-L is symmetric: its match
count is the multiset overlap of stripped lines.

Faithful measures (Design Principle 4: upstream behavior, quirks included):

- `sim_t`: Python's own `tokenize` module runs over the C, C++, or CUDA text
  (the Sim-T quirk). Each side becomes the list of `TokenInfo` tuples, whole
  (type, string, start, end, line), so positions and line text count, and the
  ENCODING token is included. The ratio is `difflib.SequenceMatcher` with its
  defaults (autojunk on). Tokenize errors keep the tokens read before them: an
  IndentationError, a TokenError, or any other exception ends the list there,
  and an encoding error before the first token leaves it empty. Upstream prints
  a message in those branches; here they are silent (a print is not a value).
- `sim_l`: lines match after `str.strip()`; each reference line consumes the
  first equal candidate line. The ratio is the match count over the larger of
  the two `str.splitlines()` counts, and 0.0 when both texts have no lines.

Python's tokenize changes between interpreter versions, so the faithful Sim-T
depends on the interpreter. The project pins Python 3.10, and every written
value carries `platform.python_version()` (see `measure`).

C-aware measure (a separate name, never a faithful value; quirk table, Sim-T
row: "keep; add a C-aware Sim-T"):

- `sim_t_c`: the matched-token ratio 2M/T of `difflib.SequenceMatcher` over the
  token strings of `c_tokens`, with autojunk off.

Design choices of task P1.8, recorded in the bible's Evaluation Protocol
(LASSI Score Profile) with a Decision Log entry by task P2.7; the tests pin
the lexer rules:

- The C lexer drops layout (whitespace, backslash-newline splices) and both
  comment styles. String and char literals are one token each, escapes
  included; numbers follow the C preprocessing-number rule, so suffixes and
  exponent signs stay in the token. Multi-character operators are one token
  each by longest match, with the CUDA launch brackets `<<<` and `>>>` among
  them. `#` in a directive is its own token. The lexer never raises and never
  yields an empty or whitespace token.
- `sim_t_c` turns autojunk off. With autojunk on, difflib ignores as match
  seeds every token that occurs more than len/100 + 1 times (integer
  division) in a candidate of 200 or more tokens; in C that is the common
  punctuation and keywords, so matched runs of them would go uncounted. Upstream's positional tuples are nearly all
  distinct, which is why autojunk hardly acts on the faithful Sim-T.
- Two texts with no C tokens score 1.0 under `sim_t_c` (difflib's value for two
  empty sequences).

The ScoreProfile `lassi` (lassi.scoring.lassi_profile) gives these values as
its sim_t, sim_t_c, and sim_l components.

The notebook's second token similarity (task P4.15; OQ-022, owner: keep both
token similarities; a separate name, never reported as `sim_t`):

- `sim_t_tiktoken`: the token ids of tiktoken's cl100k_base encoding, with
  tiktoken's default special-token handling, compared by
  `difflib.SequenceMatcher` with its defaults (autojunk on), reference first,
  as the notebook's "tiktoken" method does. The notebook raises when a text
  holds a special-token string (encode refuses it) or does not decode back
  from its ids (its round-trip assert); there is no value to reproduce then,
  so the function returns None and the caller notes why [DESIGN].
- `cl100k_base()` reads the encoding offline, only from the directory the
  environment variable TIKTOKEN_CACHE_DIR names, where tiktoken's own cache
  keeps it (an `rx run` caches it under the scratch root). It never downloads
  and never deletes a cached file: while tiktoken loads, its cached-file
  reader is replaced by `cached_reader`, which reads the file tiktoken's
  cache rule names and checks tiktoken's expected sha256, and refuses
  otherwise. An unset or empty variable, a missing directory, a missing
  file, or a hash mismatch raises TiktokenCacheError, an environment error,
  never a null value. No source address appears in any message.
"""

from __future__ import annotations

import difflib
import hashlib
import io
import os
import platform
import re
import tokenize
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Protocol

# One alternative per token class, tried in order at each position. The last
# alternative takes any single character, so every position matches something.
_C_PATTERNS = (
    ("skip", r"\s+|\\\r?\n|//[^\n]*|/\*[\s\S]*?(?:\*/|\Z)"),
    ("string", r'"(?:\\[\s\S]|[^"\\\n])*"?'),
    ("char", r"'(?:\\[\s\S]|[^'\\\n])*'?"),
    ("number", r"\.?[0-9](?:[eEpP][+-]|[0-9A-Za-z_.])*"),
    ("name", r"[^\W\d]\w*"),
    ("operator", r"<<<|>>>|<<=|>>=|\.\.\.|->|\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^]=|::|##"),
    ("other", r"[\s\S]"),
)
_C_TOKEN = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in _C_PATTERNS))


def _python_tokens(text: str) -> list[tokenize.TokenInfo]:
    """Return the tokens Python's tokenize yields for `text` before its first error (faithful Sim-T)."""
    # Upstream retries after an IndentationError, but a generator that raised
    # is finished, so the retry ends the list too. Every Exception therefore
    # keeps the tokens read so far, and an encoding error keeps none.
    tokens: list[tokenize.TokenInfo] = []
    try:
        for token in tokenize.tokenize(io.BytesIO(text.encode("utf-8")).readline):
            tokens.append(token)
    except Exception:
        pass
    return tokens


def sim_t(reference: str, candidate: str) -> float:
    """Return the faithful Sim-T of `candidate` against `reference` (Python tokenize over C text).

    Equals upstream's token similarity with the "tokenize" method; see the
    module docstring for the token form and the error handling.
    """
    return difflib.SequenceMatcher(None, _python_tokens(reference), _python_tokens(candidate)).ratio()


def sim_l(reference: str, candidate: str) -> float:
    """Return the faithful Sim-L: stripped lines of `reference` matched in `candidate`, in any order.

    Each reference line consumes the first equal candidate line. The count is
    divided by the larger of the two line counts; two texts without lines give 0.0.
    """
    reference_lines = reference.splitlines()
    candidate_lines = candidate.splitlines()
    unmatched = [line.strip() for line in candidate_lines]
    matched = 0
    for line in reference_lines:
        key = line.strip()
        if key in unmatched:
            unmatched.remove(key)  # list.remove drops the first equal entry
            matched += 1
    total = max(len(reference_lines), len(candidate_lines))
    return matched / total if total else 0.0


def c_tokens(text: str) -> list[str]:
    """Split C, C++, or CUDA `text` into token strings for the C-aware Sim-T.

    Rules, applied by first match at each position:
    - Layout is dropped: whitespace, form feeds, and backslash-newline splices.
    - Comments are dropped: `//` to the end of the line, and `/* ... */`; an
      unclosed block comment runs to the end of the text.
    - A string literal or char literal is one token, quotes and escapes
      included; an unclosed one ends at the end of its line. An encoding
      prefix such as `L` or `u8` is a separate name token.
    - A number follows the C preprocessing-number rule: an optional `.`, a
      digit, then digits, letters, `_`, `.`, and `e+ e- E+ E- p+ p- P+ P-`,
      so `1.5e-3f`, `0x1Fu`, and `42UL` are one token each.
    - A name is a letter or `_` followed by letters, digits, and `_`.
    - An operator is one token by longest match among `<<< >>> <<= >>= ...
      -> ++ -- << >> <= >= == != && || += -= *= /= %= &= |= ^= :: ##`
      (`<<<` and `>>>` are the CUDA launch brackets). `#` is its own token.
    - Any other character is a one-character token.
    The lexer never raises and never yields an empty or whitespace token.
    """
    return [match.group() for match in _C_TOKEN.finditer(text) if match.lastgroup != "skip"]


def sim_t_c(reference: str, candidate: str) -> float:
    """Return the C-aware Sim-T: 2M/T over the `c_tokens` of both texts (autojunk off).

    A separate measure from the faithful `sim_t`, never reported under its name.
    """
    return difflib.SequenceMatcher(None, c_tokens(reference), c_tokens(candidate), autojunk=False).ratio()


@dataclass(frozen=True)
class Similarity:
    """Sim-T, Sim-L, and the C-aware Sim-T of one candidate, with the interpreter version that computed them."""

    sim_t: float
    sim_l: float
    sim_t_c: float
    python: str

    def to_dict(self) -> dict[str, float | str]:
        """Return the record as a JSON-safe dict, the form in which values are written."""
        return asdict(self)


def measure(reference: str, candidate: str) -> Similarity:
    """Return every similarity value of `candidate` against `reference`, tagged with the Python version."""
    return Similarity(
        sim_t=sim_t(reference, candidate),
        sim_l=sim_l(reference, candidate),
        sim_t_c=sim_t_c(reference, candidate),
        python=platform.python_version(),
    )


# The notebook's tiktoken similarity (task P4.15; see the module docstring).
TIKTOKEN_ENCODING = "cl100k_base"
TIKTOKEN_CACHE_ENV = "TIKTOKEN_CACHE_DIR"


class TiktokenCacheError(OSError):
    """The encoding cannot be read from the offline cache: an environment error, never a null value.

    An OSError, so a caller that refuses a profile it cannot build (lassi score, lassi run) refuses this too.
    """


class TokenEncoding(Protocol):
    """What sim_t_tiktoken needs of an encoding: tiktoken's encode and decode with their defaults."""

    def encode(self, text: str) -> list[int]:
        """Return the token ids of `text`; raise ValueError for a special-token string in it."""
        ...

    def decode(self, tokens: Sequence[int]) -> str:
        """Return the text of `tokens`."""
        ...


def sim_t_tiktoken(reference: str, candidate: str, encoding: TokenEncoding | None = None) -> float | None:
    """Return the notebook's tiktoken similarity of `candidate` against `reference`, or None where it raises.

    `encoding` defaults to cl100k_base(). Each text, the reference first, is
    encoded with the encoding's defaults. A ValueError from encode (the text
    holds a special-token string) or ids that do not decode back to the text
    (the notebook's round-trip assert) give None; otherwise the value is the
    SequenceMatcher ratio of the two id lists, autojunk on.
    """
    enc = cl100k_base() if encoding is None else encoding
    ids: list[list[int]] = []
    for text in (reference, candidate):
        try:
            tokens = list(enc.encode(text))
        except ValueError:
            return None
        if enc.decode(tokens) != text:
            return None
        ids.append(tokens)
    return difflib.SequenceMatcher(None, ids[0], ids[1]).ratio()


def cached_reader(cache_dir: Path) -> Callable[[str, str | None], bytes]:
    """Return a stand-in for tiktoken's cached-file reader that only reads the cache in `cache_dir`.

    It reads the file tiktoken's cache rule names, the sha1 hex digest of the
    source address (tiktoken 0.14.0, load.read_file_cached), and checks the
    sha256 tiktoken expects. It never downloads, writes, or deletes, and its
    messages name the cache file, never the source address.
    """

    def read(source: str, expected_hash: str | None = None) -> bytes:
        name = hashlib.sha1(source.encode()).hexdigest()
        path = cache_dir / name
        if not path.is_file():
            raise TiktokenCacheError(
                f"{TIKTOKEN_CACHE_ENV} names {cache_dir.as_posix()}, which holds no cached file {name} for "
                f"{TIKTOKEN_ENCODING}; cache it there first (an rx run with {TIKTOKEN_CACHE_ENV} set), since "
                "scoring never downloads it"
            )
        data = path.read_bytes()
        if expected_hash is not None and hashlib.sha256(data).hexdigest() != expected_hash:
            raise TiktokenCacheError(
                f"{TIKTOKEN_CACHE_ENV}: the cached file {path.as_posix()} does not have the sha256 tiktoken expects "
                f"({expected_hash}); "
                "it is left in place, and scoring never downloads a replacement"
            )
        return data

    return read


def tiktoken_cache_dir() -> Path:
    """Return the directory TIKTOKEN_CACHE_DIR names; raise TiktokenCacheError when it is unset, empty, or absent."""
    value = os.environ.get(TIKTOKEN_CACHE_ENV, "")
    if not value.strip():
        raise TiktokenCacheError(
            f"sim_t_tiktoken reads {TIKTOKEN_ENCODING} offline from the directory {TIKTOKEN_CACHE_ENV} names, "
            "and it is unset or empty; set it to the cache under the scratch root"
        )
    path = Path(value)
    if not path.is_dir():
        raise TiktokenCacheError(f"{TIKTOKEN_CACHE_ENV} names {path.as_posix()}, which is not a directory")
    return path


@contextmanager
def _offline(cache_dir: Path) -> Iterator[None]:
    """Replace tiktoken's cached-file reader with cached_reader(cache_dir) while the block runs."""
    import tiktoken.load

    original = tiktoken.load.read_file_cached
    tiktoken.load.read_file_cached = cached_reader(cache_dir)
    try:
        yield
    finally:
        tiktoken.load.read_file_cached = original


@lru_cache(maxsize=1)
def cl100k_base() -> TokenEncoding:
    """Return tiktoken's cl100k_base encoding, read offline from TIKTOKEN_CACHE_DIR; raise TiktokenCacheError."""
    cache_dir = tiktoken_cache_dir()
    import tiktoken

    with _offline(cache_dir):
        return tiktoken.get_encoding(TIKTOKEN_ENCODING)


def tiktoken_version() -> str:
    """Return the installed tiktoken version, which a sim_t_tiktoken note names beside the interpreter."""
    return metadata.version("tiktoken")
