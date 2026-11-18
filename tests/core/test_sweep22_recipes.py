"""Tests for sweep22: the first five rungs of the tt-par development ladder for the 30B-class models (task P17.74).

Bible: Project Recipes (template sets, the skeleton bullet), Benchmark Suites
(tt-par-v0, tt-par-v0-kh), Execution Backends (ttsim), Oracles (binary_io),
Agent Rules 2 and 5.

What they fix for projects/lassi-demo/sweep22/<slug>/, one directory per row
of plans/runs/suite/models-30b.tsv (the owner's window of 2026-11-17):

- Ten recipes per model: five rungs (bare, sk0, p2sk0, sk1, p2sk1) in two
  variants (host: cpp -> tt on tt-par-v0; kh: cpp -> ttkh on tt-par-v0-kh).
- Each rung differs from the bare loop only in its template set and skeleton,
  as P17.74 lists them; the correction rungs wait for P17.73.
- Every recipe reads the development split only, with retrieval, worked
  examples, and API packs off, and runs every TT program on ttsim.
- Each recipe has sweep20's model id and sampling for its model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.recipe import load_recipe

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "projects" / "lassi-demo"
SWEEP = DEMO / "sweep22"
TABLE = REPO / "plans" / "runs" / "suite" / "models-30b.tsv"
ITEMS = ["vecadd_multi_core", "matmul_multicore_reuse", "pad_multi_core"]
# variant: (suite, target, old template set, core-grid template set, skeleton prefix)
VARIANTS = {
    "host": ("tt-par-v0", "tt", "tt-host-v1", "tt-host-v2", "tt-host-skeleton-"),
    "kh": ("tt-par-v0-kh", "ttkh", "tt-kh-v0", "tt-kh-v1", "tt-kh-skeleton-"),
}
# rung: (core-grid template set?, skeleton version or None), as P17.74 lists the rungs
RUNGS = {"bare": (False, None), "sk0": (False, "v0"), "p2sk0": (True, "v0"), "sk1": (False, "v1"),
         "p2sk1": (True, "v1")}
UNUSED = ("retrieval", "examples", "context", "lookup")


def load(path: Path) -> dict[str, Any]:
    """Return the resolved data of one recipe."""
    return load_recipe(path).data


def slugs() -> list[str]:
    """Return the model slugs of plans/runs/suite/models-30b.tsv, in its order."""
    lines = TABLE.read_text(encoding="ascii").splitlines()
    return [line.split("\t")[0] for line in lines if line.strip() and not line.startswith("#")]


def test_each_model_has_the_ten_recipes() -> None:
    """Every models-30b.tsv model has a directory with the five rungs in both variants, and nothing else."""
    assert sorted(path.name for path in SWEEP.iterdir()) == sorted(slugs())
    names = sorted(f"{rung}-{variant}.yaml" for rung in RUNGS for variant in VARIANTS)
    for slug in slugs():
        assert sorted(path.name for path in (SWEEP / slug).iterdir()) == names, slug


@pytest.mark.parametrize("slug", ["qwen3-coder-30b", "gemma4-31b", "qwen3-32b", "qwen3-30b-instruct", "exaone4-32b",
                                  "exaone45-33b"])
def test_recipes_take_sweep20s_model_and_sampling(slug: str) -> None:
    """Each recipe has sweep20's model id and sampling for its model and no reasoning effort."""
    want = load(DEMO / "sweep20" / slug / "ttir-l0.yaml")
    for path in sorted((SWEEP / slug).glob("*.yaml")):
        data = load(path)
        assert data["model"]["id"] == want["model"]["id"], path.name
        assert data["model"]["backend"] == "openai_compat"
        assert data["llm"]["sampling"] == want["llm"]["sampling"], path.name
        assert "reasoning_effort" not in data["llm"]["sampling"]


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_each_rung_sets_its_template_set_and_skeleton(rung: str, variant: str) -> None:
    """A rung's template set and skeleton are P17.74's; suite, direction, split, items, and ttsim are the variant's."""
    suite, target, old, new, prefix = VARIANTS[variant]
    grid, skeleton = RUNGS[rung]
    for slug in slugs():
        data = load(SWEEP / slug / f"{rung}-{variant}.yaml")
        assert data["prompts"] == (new if grid else old), (slug, rung, variant)
        if skeleton is None:
            assert "skeleton" not in data, (slug, rung, variant)
        else:
            assert data["skeleton"]["name"] == prefix + skeleton, (slug, rung, variant)
            assert data["skeleton"]["prompts"] == ["generate", "correct"]
        assert data["bench"] == {"suite": suite, "split": "dev", "items": ITEMS}
        assert data["directions"] == [{"source": "cpp", "target": target}]
        assert data["executor"][target]["kind"] == "ttsim" and data["executor"]["tt"]["kind"] == "ttsim"
        assert data["fixes"]["baseline_both"] is True
        assert data["oracle"]["metric"] == "from_item" and data["oracle"]["threshold"] == "from_item"
        assert data["trials"]["n"] == 1 and data["loop"]["max_corrections"] == 8
        assert not any(key in data for key in UNUSED), (slug, rung, variant)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_rungs_differ_from_the_bare_loop_only_in_template_set_and_skeleton(variant: str) -> None:
    """Every key but the file's own project name, the template set, and the skeleton is the bare rung's."""
    for slug in slugs():
        bare = load(SWEEP / slug / f"bare-{variant}.yaml")
        for rung in RUNGS:
            data = load(SWEEP / slug / f"{rung}-{variant}.yaml")
            differ = {key for key in set(bare) | set(data) if bare.get(key) != data.get(key)}
            assert differ <= {"project", "prompts", "skeleton"}, (slug, rung, sorted(differ))


def test_headers_say_ttsim_and_lines_fit() -> None:
    """Every sweep22 file and its bases say ttsim, a simulator, and keep lines within 120 characters, plain ASCII."""
    files = sorted(SWEEP.rglob("*.yaml")) + [DEMO / "tt-par-30b-base.yaml", DEMO / "tt-par-30b-kh-base.yaml"]
    for path in files:
        text = path.read_text(encoding="ascii")
        header = " ".join(line[2:] for line in text.splitlines() if line.startswith("# "))
        assert "ttsim, a simulator" in header, path
        assert max(len(line) for line in text.splitlines()) <= 120, path
