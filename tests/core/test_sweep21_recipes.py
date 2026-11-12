"""Tests for sweep21: bale-v0-tt, all eight bale apps, C++ -> TT-Metalium on ttsim for the 30B-class models (P17.64).

Bible: Project Recipes (template sets, the examples, context, lookup, and
retrieval sections, the loop opt-ins), Execution Backends (ttsim), Oracles
(binary_io), Agent Rules 2 and 5.

What they fix for projects/lassi-demo/sweep21/<slug>/, one directory per row
of plans/runs/suite/models-30b.tsv (the owner's window of 2026-11-12):

- Eight recipes per model, two arms of four: bale-ex-1 to -4 (sweep10's
  worked examples alone) and bale-all-1 to -4 (sweep11's bale-all: the
  kernels pack and its lookup, retrieval over the tt-metal docs, the template
  set bale-tt-v1, and restarts); each arm's four recipes cover the eight
  bale-v0-tt items once.
- cpp -> ttkh on ttsim with the exact output check, every item unassigned.
- Each recipe has sweep20's model id and sampling for its model and no
  gpt-oss reasoning_effort; every other setting is sweep10's bale-cpp-a
  (ex) or sweep11's bale-all-cpp-a (all) for gpt-oss-120b.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.recipe import load_recipe

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "projects" / "lassi-demo"
SWEEP = DEMO / "sweep21"
TABLE = REPO / "plans" / "runs" / "suite" / "models-30b.tsv"
ITEMS = {"histo", "ig", "permute_matrix", "randperm", "sssp", "toposort", "transpose_matrix", "triangle"}
ARMS = {"ex": DEMO / "sweep10" / "gpt-oss-120b" / "bale-cpp-a.yaml",
        "all": DEMO / "sweep11" / "gpt-oss-120b" / "bale-all-cpp-a.yaml"}
# Keys that differ from the gpt-oss-120b recipes on purpose: the model and its sampling, the items, and the file's
# own project name. Every other key, fixes included, is compared and must be equal.
OWN = ("project", "model", "llm", "bench")


def load(path: Path) -> dict[str, Any]:
    """Return the resolved data of one recipe."""
    return load_recipe(path).data


def slugs() -> list[str]:
    """Return the model slugs of plans/runs/suite/models-30b.tsv, in its order."""
    lines = TABLE.read_text(encoding="ascii").splitlines()
    return [line.split("\t")[0] for line in lines if line.strip() and not line.startswith("#")]


def sweep20(slug: str) -> dict[str, Any]:
    """Return sweep20's ttir-l0 recipe of the model, whose id and sampling sweep21 reuses."""
    return load(DEMO / "sweep20" / slug / "ttir-l0.yaml")


def test_each_model_has_the_eight_recipes() -> None:
    """Every models-30b.tsv model has a directory with bale-ex-1 to -4 and bale-all-1 to -4, and nothing else."""
    assert sorted(path.name for path in SWEEP.iterdir()) == sorted(slugs())
    names = sorted(f"bale-{arm}-{group}.yaml" for arm in ARMS for group in "1234")
    for slug in slugs():
        assert sorted(path.name for path in (SWEEP / slug).iterdir()) == names, slug


@pytest.mark.parametrize("slug", ["qwen3-coder-30b", "gemma4-31b", "qwen3-32b", "qwen3-30b-instruct", "exaone4-32b",
                                  "exaone45-33b"])
def test_recipes_take_sweep20s_model_and_sampling_and_cover_every_item(slug: str) -> None:
    """Each recipe has sweep20's model and sampling, no reasoning effort, and each arm covers the eight items once."""
    want = sweep20(slug)
    for arm in ARMS:
        seen: list[str] = []
        for group in "1234":
            data = load(SWEEP / slug / f"bale-{arm}-{group}.yaml")
            assert data["model"]["id"] == want["model"]["id"]
            assert data["llm"]["sampling"] == want["llm"]["sampling"]
            assert "reasoning_effort" not in data["llm"]["sampling"]
            assert data["bench"]["suite"] == "bale-v0-tt" and data["bench"]["split"] == "unassigned"
            assert data["directions"] == [{"source": "cpp", "target": "ttkh"}]
            assert data["executor"]["ttkh"]["kind"] == "ttsim"
            assert data["oracle"] == {"kind": "binary_io", "metric": "max_abs", "threshold": 0}
            seen += data["bench"]["items"]
        assert sorted(seen) == sorted(ITEMS), (slug, arm)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_each_arm_matches_its_gpt_oss_recipe_but_for_the_model(arm: str) -> None:
    """Every setting but the model, its sampling, and the items is the gpt-oss-120b recipe's of the same arm."""
    reference = load(ARMS[arm])
    for slug in slugs():
        data = load(SWEEP / slug / f"bale-{arm}-1.yaml")
        differ = {key for key in set(data) | set(reference) if key not in OWN and data.get(key) != reference.get(key)}
        assert differ == set(), (slug, arm, sorted(differ))
        assert data["model"]["backend"] == reference["model"]["backend"]
        assert data["model"]["timeout_s"] == reference["model"]["timeout_s"]


def test_the_all_arm_adds_the_pack_lookup_retrieval_template_set_and_restarts() -> None:
    """bale-all-1 differs from bale-ex-1 exactly in sweep11's bale-all features."""
    for slug in slugs():
        ex = load(SWEEP / slug / "bale-ex-1.yaml")
        full = load(SWEEP / slug / "bale-all-1.yaml")
        differ = {key for key in set(ex) | set(full) if key != "project" and ex.get(key) != full.get(key)}
        assert differ == {"prompts", "loop", "context", "lookup", "retrieval"}, (slug, sorted(differ))
        assert full["prompts"] == "bale-tt-v1" and full["retrieval"]["index"] == "tt-metal-docs"
        assert full["loop"]["restart_after"] == 3 and full["loop"]["restart_on_repeat"] is True


def test_headers_say_ttsim_and_lines_fit() -> None:
    """Every sweep21 file and its base say ttsim, a simulator, and keep lines within 120 characters, plain ASCII."""
    files = sorted(SWEEP.rglob("*.yaml")) + [DEMO / "bale-tt-30b-base.yaml"]
    for path in files:
        text = path.read_text(encoding="ascii")
        header = " ".join(line[2:] for line in text.splitlines() if line.startswith("# "))
        assert "ttsim, a simulator" in header, path
        assert max(len(line) for line in text.splitlines()) <= 120, path
