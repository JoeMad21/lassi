"""Remote test of sim_t_tiktoken with the real cl100k_base file (task P4.15).

Bible: Evaluation Protocol (LASSI Score Profile; LASSI Paper Metrics, [OPEN]
on the Sim-T tokenizer), Agent Rules 6 and 7; OQ-022.

On the build host, with the encoding cached under the scratch root and the
pinned upstream checkout fetched, these tests check that
lassi.scoring.similarity.sim_t_tiktoken equals the notebook's own
token_similarity(reference, candidate, "tiktoken"), run from the pinned
notebook cell in memory under the conftest's process and socket guard
(fixture upstream_with_tiktoken), exactly, on:

- every ordered pair of the 20 upstream *_main sources, and
- each source as a perfect copy sent in a fenced block and read the way the
  notebook reads it (lassi.core.fragments.first_fence, the faithful port),
  for the fence tags cpp, c++, c, cuda, and none.

The encoding loads first through lassi's offline loader (cl100k_base), so
the notebook's own get_encoding call finds it in tiktoken's in-memory cache
and reads nothing. The tests are marked `remote` and skip unless
TIKTOKEN_CACHE_DIR is set and the upstream checkout is fetched; with
LASSI_REQUIRE_TIKTOKEN=1 they fail instead, so a silent skip never passes
for evidence. A set variable whose directory lacks the cached file fails
(TiktokenCacheError), never skips. Run them with `uv run tools/rx.py run --
'<command>'`, the command being `uv run tools/fetch_upstream.py &&
TIKTOKEN_CACHE_DIR="$LASSI_SCRATCH/.cache/tiktoken" LASSI_REQUIRE_TIKTOKEN=1
uv run pytest -q -m remote tests/scoring/test_similarity_tiktoken_remote.py`.
They write nothing. No value here is a measurement.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from lassi.core.fragments import first_fence
from lassi.scoring.similarity import TIKTOKEN_CACHE_ENV, TIKTOKEN_ENCODING, cl100k_base, sim_t_tiktoken

pytestmark = [pytest.mark.remote, pytest.mark.real_tiktoken]

REPO = Path(__file__).resolve().parents[2]
REQUIRE = os.environ.get("LASSI_REQUIRE_TIKTOKEN") == "1"
FENCE_TAGS = ("cpp", "c++", "c", "cuda", "")


def need(condition: bool, why: str) -> None:
    """Skip the test when `condition` is false, or fail it under LASSI_REQUIRE_TIKTOKEN=1."""
    if not condition:
        if REQUIRE:
            pytest.fail(why)
        pytest.skip(why)


@pytest.fixture(scope="module")
def encoding() -> Any:
    """Return cl100k_base read offline; skip (or fail under the require flag) when the cache variable is unset."""
    need(bool(os.environ.get(TIKTOKEN_CACHE_ENV, "").strip()),
         f"{TIKTOKEN_CACHE_ENV} is not set; cache {TIKTOKEN_ENCODING} under the scratch root first")
    return cl100k_base()


@pytest.fixture(scope="module")
def notebook(request: pytest.FixtureRequest, upstream_pin: Any) -> Any:
    """Return the notebook's similarity functions with the real tiktoken; skip or fail when upstream is absent."""
    need((REPO / upstream_pin.path / ".git").exists(), f"{upstream_pin.path} is not fetched; {upstream_pin.hint}")
    return request.getfixturevalue("upstream_with_tiktoken")


def test_the_cached_encoding_loads_offline(encoding: Any) -> None:
    assert encoding.name == TIKTOKEN_ENCODING
    text = "int main() { return 0; }\n"
    assert encoding.decode(encoding.encode(text)) == text


def test_sim_t_tiktoken_equals_the_notebook_on_every_ordered_pair(
    encoding: Any, notebook: Any, upstream_mains: dict[tuple[str, str], str]
) -> None:
    mismatches = []
    for reference_key, reference in upstream_mains.items():
        for candidate_key, candidate in upstream_mains.items():
            want = notebook.sim_t_tiktoken(reference, candidate)
            got = sim_t_tiktoken(reference, candidate, encoding)
            if got != want:
                mismatches.append((reference_key, candidate_key, want, got))
    assert mismatches == [], f"{len(mismatches)} of 400 ordered pairs differ: {mismatches[:5]}"


@pytest.mark.parametrize("tag", FENCE_TAGS)
def test_sim_t_tiktoken_equals_the_notebook_on_fenced_perfect_copies(
    tag: str, encoding: Any, notebook: Any, upstream_mains: dict[tuple[str, str], str]
) -> None:
    for key, reference in upstream_mains.items():
        candidate = first_fence(f"```{tag}\n{reference}```\n").text
        assert sim_t_tiktoken(reference, candidate, encoding) == notebook.sim_t_tiktoken(reference, candidate), key
