"""The P2.5 spike recipe loads and binds what the stability check needs (plans/p2-scoring.md, P2.5)."""

from __future__ import annotations

from pathlib import Path

from lassi.core import runner  # noqa: F401  (importing the runner registers every component)
from lassi.core.recipe import load_recipe

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / "tests" / "fixtures" / "recipes" / "p2-proxy-mock.yaml"


def test_the_proxy_mock_recipe_loads_with_the_mock_the_proxy_and_the_oracle() -> None:
    data = load_recipe(RECIPE).data
    assert data["model"] == {"backend": "mock", "id": "mock-reference"}
    assert data["directions"] == [{"source": "cuda", "target": "omp"}]
    assert data["toolchain"]["omp"] == "nvcpp-multicore"
    assert data["executor"]["kind"] == "native"
    assert data["oracle"] == {"kind": "stdout_mask", "passfail": True}
    assert data["trials"]["n"] == 3
    assert "items" not in data["bench"], "every item of the suite runs"
    assert data["faithful"] is False, "the proxy is never faithful"
