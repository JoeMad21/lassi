"""Tests for the development split, dev, and tt-pairs-v0's items in it (task P17.31; OQ-083, option (b)).

Bible: Benchmark Suites (split rules: tt-pairs-v0 is a development set,
2026-10-25), the Decision Log row of 2026-10-25 that makes tt-pairs-v0 a
development set, and Agent Rule 5. Plan: plans/p17-portable.md, P17.31.

The contract these tests fix (lassi.bench.registry):

- SPLITS gains a fourth value, dev, appended: ("train", "eval",
  "unassigned", "dev"). PURPOSES are unchanged.
- HELD_SPLITS, the splits kept from training and corpus harvest, is
  ("eval", "unassigned", "dev").
- REFUSED_PURPOSES maps each split to the purposes Suite.item refuses it:
  train to none; eval and unassigned to train, prompt-tuning, and harvest,
  as before; dev to train and harvest. A dev item is given for eval and
  prompt-tuning (prompt and feature work steered by its outcomes is
  allowed), and refused to train and harvest with EvalSplitError whose text
  names the suite, the item, the split, and the purpose's words, as for an
  eval or unassigned item: "<suite>/<item> is in the dev split and may
  never be used for <words> (Agent Rule 5)".
- load_suite accepts split dev; any other value is still refused, naming
  every split.
- Suite.train_view().held holds each dev name with its split, and
  TrainView.item refuses a dev name with Suite.item's training words; the
  view holds no dev SuiteItem.
- Manifests: every item of tt-pairs-v0 and its four siblings (kh, ttkernel,
  ttir, d2m) has split dev; every other suite keeps its splits, and no other
  suite holds a dev item.

The suite written under tmp_path is SYNTHETIC. No value here is a
measurement.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from lassi import bench

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "assets" / "bench"
DEV_SUITES = ("tt-pairs-v0", "tt-pairs-v0-kh", "tt-pairs-v0-ttkernel", "tt-pairs-v0-ttir", "tt-pairs-v0-d2m")
TT_PAIRS_ITEMS = ("eltwise_binary", "eltwise_sfpu", "loopback", "matmul_multi_core", "matmul_single_core")
# Other committed suites that hold dev items beside items of other splits (tt-par-v0, task P17.67; OQ-129).
MIXED_DEV_SUITES = ("tt-par-v0",)
# The splits every other committed suite keeps (task P17.31 changes none of them).
OTHER_SUITES = {
    "bale-v0": "unassigned",
    "bale-v0-tt": "unassigned",
    "bale-v0-ttnn": "unassigned",
    "adv-seeded-v0": "unassigned",
    "lassi-hecbench-10": "eval",
}
HELD_PURPOSES = ("train", "prompt-tuning", "harvest")
DEV_REFUSED = ("train", "harvest")
DEV_GIVEN = ("eval", "prompt-tuning")
# The words each refused purpose is named with (lassi.bench.registry FORBIDDEN_PURPOSES).
WORDS = {"train": "training", "prompt-tuning": "prompt tuning", "harvest": "corpus harvest"}

SUITE = "dev-suite"
# SYNTHETIC: one item of each split, each with an omp and a cuda version.
MANIFEST = (
    f"suite: {SUITE}\n"
    "repo: https://example.invalid/dev-suite.git\n"
    f"commit: {'d' * 40}\n"
    "items:\n"
    + "".join(
        f"  {name}:\n"
        f"    split: {split}\n"
        "    languages:\n"
        f"      omp: {{dir: src/{name}-omp, files: [main.cpp]}}\n"
        f"      cuda: {{dir: src/{name}-cuda, files: [main.cu]}}\n"
        for name, split in (("alpha", "train"), ("held", "eval"), ("open", "unassigned"), ("tuned", "dev"))
    )
)
BY_SPLIT = {"train": "alpha", "eval": "held", "unassigned": "open", "dev": "tuned"}
TO_CUDA = bench.Direction("omp", "cuda")


def registry() -> ModuleType:
    """Import and return lassi.bench.registry."""
    return importlib.import_module("lassi.bench.registry")


def registry_name(name: str) -> Any:
    """Return `name` from lassi.bench.registry, failing the test clearly while it does not exist (task P17.31)."""
    found = registry()
    if not hasattr(found, name):
        pytest.fail(f"lassi.bench.registry has no {name} yet (task P17.31)")
    return getattr(found, name)


def dev_suite(tmp_path: Path) -> bench.Suite:
    """Load the SYNTHETIC four-split suite; fail clearly while load_suite refuses split dev (task P17.31)."""
    path = tmp_path / f"{SUITE}.yaml"
    path.write_bytes(MANIFEST.encode("ascii"))
    try:
        return bench.load_suite(path)
    except ValueError as error:
        if "split" in str(error) and "'dev'" in str(error):
            pytest.fail(f"load_suite refuses split dev, which is not a split yet (task P17.31): {error}")
        raise


def write_sources(root: Path) -> Path:
    """Write every SYNTHETIC source file of the four-split suite under `root`; return `root`."""
    for name in BY_SPLIT.values():
        for language, file in (("omp", "main.cpp"), ("cuda", "main.cu")):
            path = root / "src" / f"{name}-{language}" / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"// SYNTHETIC {name} {language}\n".encode("ascii"))
    return root


def held_message(suite: str, item: str, split: str, purpose: str) -> str:
    """Return the EvalSplitError text the registry gives for a held item refused to `purpose`."""
    return f"{suite}/{item} is in the {split} split and may never be used for {WORDS[purpose]} (Agent Rule 5)"


def raw(suite: str) -> dict[str, Any]:
    """Return a committed suite manifest as YAML data, without the registry's checks."""
    return yaml.safe_load((BENCH / f"{suite}.yaml").read_bytes().decode("utf-8"))


# ---------------------------------------------------------------------------
# The split values and what each refuses


def test_dev_is_the_fourth_split_appended() -> None:
    """SPLITS keeps its three values in order and appends dev; PURPOSES are unchanged."""
    assert registry().SPLITS == ("train", "eval", "unassigned", "dev")
    assert registry().PURPOSES == ("train", "eval", "prompt-tuning", "harvest")


def test_held_splits_keep_dev_from_training_and_harvest() -> None:
    """HELD_SPLITS, the splits kept from training and corpus harvest, adds dev after eval and unassigned."""
    assert registry().HELD_SPLITS == ("eval", "unassigned", "dev")


def test_refused_purposes_map_each_split_to_what_it_refuses() -> None:
    """REFUSED_PURPOSES: train none; eval and unassigned train, prompt-tuning, harvest; dev train and harvest."""
    refused: Mapping[str, Any] = registry_name("REFUSED_PURPOSES")
    assert set(refused) == set(registry().SPLITS)
    assert {split: set(purposes) for split, purposes in refused.items()} == {
        "train": set(),
        "eval": set(HELD_PURPOSES),
        "unassigned": set(HELD_PURPOSES),
        "dev": set(DEV_REFUSED),
    }


def test_every_refused_purpose_is_a_purpose_with_its_words() -> None:
    """Each purpose REFUSED_PURPOSES names is one of PURPOSES and has its words in FORBIDDEN_PURPOSES."""
    refused: Mapping[str, Any] = registry_name("REFUSED_PURPOSES")
    named = {purpose for purposes in refused.values() for purpose in purposes}
    assert named <= set(registry().PURPOSES)
    assert all(registry().FORBIDDEN_PURPOSES[purpose] == WORDS[purpose] for purpose in named)


def test_the_held_splits_are_those_that_refuse_training() -> None:
    """A split is held exactly when REFUSED_PURPOSES says it refuses training and corpus harvest."""
    refused: Mapping[str, Any] = registry_name("REFUSED_PURPOSES")
    held = tuple(split for split in registry().SPLITS if {"train", "harvest"} <= set(refused[split]))
    assert held == registry().HELD_SPLITS


# ---------------------------------------------------------------------------
# load_suite and Suite.item


def test_load_suite_accepts_split_dev(tmp_path: Path) -> None:
    """A manifest item with split dev loads and keeps its split."""
    suite = dev_suite(tmp_path)
    assert {name: item.split for name, item in suite.items.items()} == {
        "alpha": "train", "held": "eval", "open": "unassigned", "tuned": "dev",
    }


def test_another_split_is_still_refused_naming_every_split(tmp_path: Path) -> None:
    """A split that is none of the four is refused at load, and the message lists all four in order."""
    path = tmp_path / "bad" / f"{SUITE}.yaml"
    path.parent.mkdir()
    path.write_bytes(MANIFEST.replace("split: dev", "split: devel").encode("ascii"))
    with pytest.raises(ValueError) as refused:
        bench.load_suite(path)
    message = str(refused.value)
    assert "tuned" in message and "devel" in message, message
    assert "train, eval, unassigned, dev" in message, message


@pytest.mark.parametrize("purpose", DEV_GIVEN)
def test_a_dev_item_is_given_for_evaluation_and_prompt_tuning(tmp_path: Path, purpose: str) -> None:
    """Suite.item gives a dev item for eval and for prompt-tuning (OQ-083 (b))."""
    suite = dev_suite(tmp_path)
    found = suite.item("tuned", purpose=purpose)
    assert found is suite.items["tuned"] and found.split == "dev"


@pytest.mark.parametrize("purpose", DEV_REFUSED)
def test_a_dev_item_is_refused_to_training_and_harvest_naming_the_split(tmp_path: Path, purpose: str) -> None:
    """Suite.item refuses a dev item to train and harvest with EvalSplitError in the held items' words."""
    suite = dev_suite(tmp_path)
    with pytest.raises(bench.EvalSplitError) as refused:
        suite.item("tuned", purpose=purpose)
    assert str(refused.value) == held_message(SUITE, "tuned", "dev", purpose)


@pytest.mark.parametrize("split", ["eval", "unassigned"])
@pytest.mark.parametrize("purpose", HELD_PURPOSES)
def test_eval_and_unassigned_items_keep_refusing_every_forbidden_purpose(
    tmp_path: Path, split: str, purpose: str
) -> None:
    """Beside a dev item, eval and unassigned items still refuse training, prompt tuning, and harvest."""
    suite = dev_suite(tmp_path)
    name = BY_SPLIT[split]
    with pytest.raises(bench.EvalSplitError) as refused:
        suite.item(name, purpose=purpose)
    assert str(refused.value) == held_message(SUITE, name, split, purpose)
    assert suite.item(name, purpose="eval").split == split


def test_a_train_item_still_serves_every_purpose(tmp_path: Path) -> None:
    """A train item beside a dev item serves every purpose."""
    suite = dev_suite(tmp_path)
    for purpose in registry().PURPOSES:
        assert suite.item("alpha", purpose=purpose).split == "train"


@pytest.mark.parametrize("split", ["train", "eval", "unassigned", "dev"])
def test_suite_item_refuses_exactly_the_purposes_refused_purposes_names(tmp_path: Path, split: str) -> None:
    """For each split, Suite.item raises EvalSplitError for a purpose if and only if REFUSED_PURPOSES names it."""
    refused: Mapping[str, Any] = registry_name("REFUSED_PURPOSES")
    suite = dev_suite(tmp_path)
    name = BY_SPLIT[split]
    for purpose in registry().PURPOSES:
        if purpose in refused[split]:
            with pytest.raises(bench.EvalSplitError):
                suite.item(name, purpose=purpose)
        else:
            assert suite.item(name, purpose=purpose).split == split


def test_a_dev_items_files_follow_its_purposes(tmp_path: Path) -> None:
    """reference_target and source_files give a dev item's files for prompt-tuning and refuse them to training."""
    suite = dev_suite(tmp_path)
    root = write_sources(tmp_path / "sources")
    assert suite.reference_target("tuned", TO_CUDA, root, purpose="prompt-tuning") == {
        "main.cu": "// SYNTHETIC tuned cuda\n",
    }
    assert suite.source_files("tuned", TO_CUDA, root, purpose="eval") == {"main.cpp": "// SYNTHETIC tuned omp\n"}
    for purpose in DEV_REFUSED:
        with pytest.raises(bench.EvalSplitError):
            suite.reference_target("tuned", TO_CUDA, root, purpose=purpose)
        with pytest.raises(bench.EvalSplitError):
            suite.source_files("tuned", TO_CUDA, root, purpose=purpose)


# ---------------------------------------------------------------------------
# The train-only view (task P17.9)


def test_the_train_view_holds_dev_names_as_held(tmp_path: Path) -> None:
    """Suite.train_view() keeps only the train item and lists the eval, unassigned, and dev names as held."""
    view = dev_suite(tmp_path).train_view()
    assert sorted(view.train_items) == ["alpha"]
    assert dict(view.held) == {"held": "eval", "open": "unassigned", "tuned": "dev"}


def test_the_train_view_refuses_a_dev_name_with_suite_items_training_words(tmp_path: Path) -> None:
    """TrainView.item refuses a dev name with the words Suite.item uses for training."""
    suite = dev_suite(tmp_path)
    with pytest.raises(bench.EvalSplitError) as by_suite:
        suite.item("tuned", purpose="train")
    with pytest.raises(bench.EvalSplitError) as by_view:
        suite.train_view().item("tuned")
    assert str(by_view.value) == str(by_suite.value) == held_message(SUITE, "tuned", "dev", "train")


def test_the_train_view_holds_no_dev_item(tmp_path: Path) -> None:
    """No SuiteItem other than the train item is reachable from the train view."""
    view = dev_suite(tmp_path).train_view()
    found = reachable(view)
    assert not [obj for obj in found if isinstance(obj, bench.Suite)], "the view holds the whole suite"
    assert sorted(obj.split for obj in found if isinstance(obj, bench.SuiteItem)) == ["train"]


def reachable(root: object) -> list[object]:
    """Return every object reachable from `root` through attributes, mapping keys and values, and sequences."""
    seen: dict[int, object] = {}
    stack = [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen or isinstance(obj, (str, bytes, int, float, bool, type(None), type, ModuleType)):
            continue
        seen[id(obj)] = obj
        if isinstance(obj, Mapping):
            stack.extend(obj.keys())
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        if hasattr(obj, "__dict__"):
            stack.extend(vars(obj).values())
        for slot in getattr(type(obj), "__slots__", ()):
            if hasattr(obj, slot):
                stack.append(getattr(obj, slot))
    return list(seen.values())


# ---------------------------------------------------------------------------
# The committed manifests


@pytest.mark.parametrize("suite", DEV_SUITES)
def test_every_item_of_the_development_suites_is_dev(suite: str) -> None:
    """tt-pairs-v0 and its four siblings list the five Tier A items, each with split dev (OQ-083 (b))."""
    items = raw(suite)["items"]
    assert sorted(items) == list(TT_PAIRS_ITEMS)
    splits = {name: entry["split"] for name, entry in items.items()}
    assert splits == dict.fromkeys(TT_PAIRS_ITEMS, "dev"), f"{suite}: {splits}"


@pytest.mark.parametrize("suite", DEV_SUITES)
def test_the_development_suites_give_their_items_for_prompt_tuning_and_refuse_them_to_training(suite: str) -> None:
    """Each loaded development suite gives every item for eval and prompt-tuning and refuses train and harvest."""
    loaded = bench.load_suite(BENCH / f"{suite}.yaml")
    for name in TT_PAIRS_ITEMS:
        for purpose in DEV_GIVEN:
            assert loaded.item(name, purpose=purpose).split == "dev", f"{suite}/{name} for {purpose}"
        for purpose in DEV_REFUSED:
            with pytest.raises(bench.EvalSplitError) as refused:
                loaded.item(name, purpose=purpose)
            assert str(refused.value) == held_message(suite, name, "dev", purpose)


@pytest.mark.parametrize(("suite", "split"), list(OTHER_SUITES.items()))
def test_every_other_suite_keeps_its_splits(suite: str, split: str) -> None:
    """bale-v0, bale-v0-tt, bale-v0-ttnn, adv-seeded-v0, and lassi-hecbench-10 keep their splits."""
    items = raw(suite)["items"]
    assert items and {entry["split"] for entry in items.values()} == {split}, suite


def test_no_suite_outside_the_development_set_holds_a_dev_item() -> None:
    """Only the five tt-pairs-v0 suites and MIXED_DEV_SUITES hold dev items among the committed manifests."""
    found = {path.stem: raw(path.stem) for path in sorted(BENCH.glob("*.yaml"))}
    assert set(DEV_SUITES) | set(OTHER_SUITES) <= set(found)
    with_dev = sorted(name for name, data in found.items()
                      if any(entry.get("split") == "dev" for entry in data["items"].values()))
    assert not set(with_dev) - set(DEV_SUITES) - set(MIXED_DEV_SUITES), with_dev
