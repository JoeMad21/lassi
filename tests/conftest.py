"""Repository-wide test fixtures (task P4.15).

The lassi score profile reads tiktoken's cl100k_base encoding when it is
built (lassi.scoring.similarity.cl100k_base), and so does `lassi run` when a
recipe offers the profile's metrics; that file is cached only on the build
host, under TIKTOKEN_CACHE_DIR. So that no test needs the network or the
cached file:

- `fake_cl100k_base` (session, autouse) replaces
  lassi.scoring.similarity.cl100k_base, before any module or class fixture
  builds a profile, with a loader that returns FakeEncoding, a SYNTHETIC
  character-level encoding that refuses a special-token string as tiktoken
  does by default.
- `real_tiktoken_when_marked` (autouse) makes that loader call the real one
  for a test marked `real_tiktoken`: the tests of the loader itself and the
  remote test of the real file.

No value computed with the fake encoding is a measurement.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest

from lassi.scoring import similarity

# A special-token string tiktoken refuses by default; the fake refuses it the same way.
SPECIAL_TOKEN = "<|endoftext|>"


class FakeEncoding:
    """A SYNTHETIC stand-in for cl100k_base: one id per character, special-token strings refused."""

    name = "synthetic-characters"

    def encode(self, text: str) -> list[int]:
        """Return one id per character; raise ValueError, as tiktoken does, for a special-token string."""
        if SPECIAL_TOKEN in text:
            raise ValueError("SYNTHETIC: a disallowed special token")
        return [ord(char) for char in text]

    def decode(self, tokens: Sequence[int]) -> str:
        """Return the characters of the ids."""
        return "".join(chr(token) for token in tokens)


FAKE_ENCODING = FakeEncoding()
_REAL_LOADER = similarity.cl100k_base
_USE_REAL = {"now": False}


def _loader() -> similarity.TokenEncoding:
    """Return the real cl100k_base for a test marked real_tiktoken, else FAKE_ENCODING."""
    return _REAL_LOADER() if _USE_REAL["now"] else FAKE_ENCODING


@pytest.fixture(scope="session", autouse=True)
def fake_cl100k_base() -> Iterator[None]:
    """Replace the offline cl100k_base loader for the whole session (see the module docstring)."""
    patch = pytest.MonkeyPatch()
    patch.setattr(similarity, "cl100k_base", _loader)
    yield
    patch.undo()


@pytest.fixture(autouse=True)
def real_tiktoken_when_marked(request: pytest.FixtureRequest) -> Iterator[None]:
    """Let the loader call the real cl100k_base while a test marked real_tiktoken runs."""
    _USE_REAL["now"] = request.node.get_closest_marker("real_tiktoken") is not None
    yield
    _USE_REAL["now"] = False
