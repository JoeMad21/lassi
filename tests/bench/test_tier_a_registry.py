"""Tests for the bench registry's Tier A additions (task P4.13; bible Benchmark Suites, Harness Contract).

Plan: plans/p4-ttsim.md, P4.13 and the planning decision "Tier A (P4.13)";
PHASE-NOTES P4, "For P4.13 (OQ-025 audit)" and the P4.3 review item. The
contract these tests fix:

- A third split, `unassigned` (OQ-025, option (d)). Suite.item refuses an
  eval or unassigned item to every purpose Agent Rule 5 forbids: training
  ("train"), prompt tuning ("prompt-tuning"), and corpus harvest
  ("harvest"), with EvalSplitError naming the split; evaluation ("eval")
  takes it. A train item serves every purpose; another purpose is a
  ValueError naming `purpose`.
- A language may be `tracked: true`: its `dir` and files are paths in this
  repository (under assets/), read from it rather than from the fetched
  sources, and carry no sha256 (git holds their bytes).
- A language may list its own support files, `support: {<build-dir name>:
  <entry>}`, each entry either `{tracked: <path under assets/>}` or
  `{upstream: <path in the pinned sources>, sha256: <64 hex>}`.
  Suite.support_files(..., language=<language>) gives the item-level
  support files plus that language's; without a language, the item-level
  ones only, as before.
- An item may declare held-out inputs: `inputs` (generate_inputs entries,
  each of at most MAX_INPUT_ELEMENTS = 2**20 elements), `outputs` (the
  output array names), and `seed`, all three together and never with
  run_args. program_args gives `@inputs/<input>.lassiio` per input, in
  order, then `<output>.lassiio` per output; stage_inputs writes the
  inputs under <workdir>/@inputs, a fresh directory each call, and returns
  those arguments. An item may also carry an `unpack_to_dest` note.
- A suite may name its source's toolchain pin, `pin: <name>`: the pin's
  COMMIT and URL must be the manifest's commit and repo.
- Any value that starts with PLACEHOLDER is refused, naming its key, so an
  unfinished manifest never loads.

The manifest, sources, and kernel text are SYNTHETIC; tracked files live
under a temporary TRACKED_ROOT. No value here is a measurement.
"""

from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi import bench
from lassi.bench import registry
from lassi.harness.lassi_io import generate_inputs, read_array
from lassi.toolchains import pins

SUITE = "tier-a-fixture"
COMMIT = "b" * 40
ITEM = "vadd"
HEADER = "// SYNTHETIC lassi_io.h stand-in\n"
KERNEL = "// SYNTHETIC compute kernel\n"
TT_PROGRAM = "// SYNTHETIC tt host program\n"
CPP_PROGRAM = "// SYNTHETIC C++ counterpart\n"
KERNEL_NAME = "vadd/kernels/compute/add.cpp"
KERNEL_UPSTREAM = "examples/vadd/kernels/compute/add.cpp"
FORBIDDEN = ("train", "prompt-tuning", "harvest")
INPUTS = [
    {"name": "a", "dtype": "f32", "shape": [2, 4], "dist": "uniform", "lo": -1.0, "hi": 1.0},
    {"name": "b", "dtype": "i32", "shape": [8], "dist": "integers", "lo": 0, "hi": 10},
]


def digest(text: str) -> str:
    """Return the sha256 hex digest of `text` as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manifest_data() -> dict[str, Any]:
    """Return the SYNTHETIC one-item manifest with every Tier A key."""
    header = {"tracked": "assets/fixture/lassi_io.h"}
    return {
        "suite": SUITE,
        "repo": "https://example.invalid/kernels.git",
        "commit": COMMIT,
        "items": {
            ITEM: {
                "split": "unassigned",
                "seed": 7,
                "inputs": copy.deepcopy(INPUTS),
                "outputs": ["c"],
                "tolerance": {"metric": "max_abs", "threshold": 0.01},
                "unpack_to_dest": "SYNTHETIC note: no fp32_dest_acc_en, no unpack_to_dest_mode",
                "languages": {
                    "tt": {
                        "dir": "assets/fixture/vadd/tt",
                        "tracked": True,
                        "files": ["vadd.cpp"],
                        "support": {
                            "lassi_io.h": dict(header),
                            KERNEL_NAME: {"upstream": KERNEL_UPSTREAM, "sha256": digest(KERNEL)},
                        },
                    },
                    "cpp": {
                        "dir": "assets/fixture/vadd/cpp",
                        "tracked": True,
                        "files": ["vadd.cpp"],
                        "support": {"lassi_io.h": dict(header)},
                    },
                },
            }
        },
    }


def write_manifest(directory: Path, data: dict[str, Any]) -> Path:
    """Write `data` as <directory>/<suite>.yaml with LF newlines and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{data['suite']}.yaml"
    path.write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    return path


@pytest.fixture
def tracked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the registry's TRACKED_ROOT at a temporary repository holding the SYNTHETIC tracked files."""
    root = tmp_path / "repo"
    for path, text in {
        "assets/fixture/lassi_io.h": HEADER,
        "assets/fixture/vadd/tt/vadd.cpp": TT_PROGRAM,
        "assets/fixture/vadd/cpp/vadd.cpp": CPP_PROGRAM,
    }.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(text.encode("ascii"))
    monkeypatch.setattr(registry, "TRACKED_ROOT", root)
    return root


@pytest.fixture
def sources(tmp_path: Path) -> Path:
    """Return a fetched-sources root holding only the upstream kernel."""
    root = tmp_path / "sources"
    (root / KERNEL_UPSTREAM).parent.mkdir(parents=True)
    (root / KERNEL_UPSTREAM).write_bytes(KERNEL.encode("ascii"))
    return root


def load(tmp_path: Path, data: dict[str, Any]) -> bench.Suite:
    """Write and load a manifest."""
    return bench.load_suite(write_manifest(tmp_path / "manifests", data))


def item_of(data: dict[str, Any]) -> dict[str, Any]:
    """Return the one item entry of `data`, for editing."""
    return data["items"][ITEM]


# ---------------------------------------------------------------------------
# Splits and purposes (Agent Rule 5; OQ-025)


def test_unassigned_is_a_split_and_rule_5_names_three_forbidden_purposes() -> None:
    assert set(registry.SPLITS) == {"train", "eval", "unassigned"}
    assert set(registry.PURPOSES) == {"train", "eval", "prompt-tuning", "harvest"}


@pytest.mark.parametrize("split", ["unassigned", "eval"])
@pytest.mark.parametrize("purpose", FORBIDDEN)
def test_eval_and_unassigned_items_are_refused_to_every_forbidden_purpose(
    tmp_path: Path, tracked: Path, sources: Path, split: str, purpose: str
) -> None:
    data = manifest_data()
    item_of(data)["split"] = split
    suite = load(tmp_path, data)
    with pytest.raises(bench.EvalSplitError, match=split):
        suite.item(ITEM, purpose=purpose)
    with pytest.raises(bench.EvalSplitError):
        suite.reference_target(ITEM, bench.Direction("cpp", "tt"), sources, purpose=purpose)
    with pytest.raises(bench.EvalSplitError):
        suite.support_files(ITEM, sources, purpose=purpose, language="tt")
    assert suite.item(ITEM, purpose="eval").split == split


def test_a_train_item_serves_every_purpose(tmp_path: Path, tracked: Path) -> None:
    data = manifest_data()
    item_of(data)["split"] = "train"
    suite = load(tmp_path, data)
    for purpose in registry.PURPOSES:
        assert suite.item(ITEM, purpose=purpose).split == "train"


def test_another_purpose_is_refused_naming_purpose(tmp_path: Path, tracked: Path) -> None:
    suite = load(tmp_path, manifest_data())
    with pytest.raises(ValueError, match="purpose"):
        suite.item(ITEM, purpose="tune")


# ---------------------------------------------------------------------------
# Tracked languages and per-language support files


def test_a_tracked_language_is_read_from_the_repository_not_the_sources(
    tmp_path: Path, tracked: Path, sources: Path
) -> None:
    suite = load(tmp_path, manifest_data())
    to_tt, to_cpp = bench.Direction("cpp", "tt"), bench.Direction("tt", "cpp")
    assert suite.reference_target(ITEM, to_tt, sources, purpose="eval") == {"vadd.cpp": TT_PROGRAM}
    assert suite.source_files(ITEM, to_tt, sources, purpose="eval") == {"vadd.cpp": CPP_PROGRAM}
    assert suite.reference_target(ITEM, to_cpp, sources, purpose="eval") == {"vadd.cpp": CPP_PROGRAM}
    spec = suite.items[ITEM].languages["tt"]
    assert spec.tracked is True and spec.sha256 == {}
    assert spec.base(sources) == tracked / "assets/fixture/vadd/tt"


def test_support_files_per_language_join_the_item_level_ones(tmp_path: Path, tracked: Path, sources: Path) -> None:
    suite = load(tmp_path, manifest_data())
    assert suite.support_files(ITEM, sources, purpose="eval", language="tt") == {
        "lassi_io.h": HEADER,
        KERNEL_NAME: KERNEL,
    }
    assert suite.support_files(ITEM, sources, purpose="eval", language="cpp") == {"lassi_io.h": HEADER}
    assert suite.support_files(ITEM, sources, purpose="eval") == {}, "without a language: item-level only"
    entry = suite.items[ITEM].languages["tt"].support[KERNEL_NAME]
    assert (entry.path, entry.tracked, entry.sha256) == (KERNEL_UPSTREAM, False, digest(KERNEL))
    with pytest.raises(ValueError, match="hip"):
        suite.support_files(ITEM, sources, purpose="eval", language="hip")


def test_item_level_support_still_reaches_every_language(tmp_path: Path, tracked: Path, sources: Path) -> None:
    data = manifest_data()
    item_of(data)["support"] = {"common.h": "examples/common.h"}
    (sources / "examples/common.h").write_bytes(b"// SYNTHETIC common header\n")
    suite = load(tmp_path, data)
    assert suite.support_files(ITEM, sources, purpose="eval")["common.h"] == "// SYNTHETIC common header\n"
    assert set(suite.support_files(ITEM, sources, purpose="eval", language="cpp")) == {"common.h", "lassi_io.h"}


def edit(path: str, value: Any) -> Any:
    """Return a function that sets the key `path` of the item entry to `value` (None deletes it).

    `path` names one key per level, joined by " > "; a level that is a list
    takes its index as digits.
    """

    def apply(item: dict[str, Any]) -> None:
        *parents, last = path.split(" > ")
        node: Any = item
        for part in parents:
            node = node[int(part)] if isinstance(node, list) else node[part]
        if value is None:
            del node[last]
        else:
            node[last] = value

    return apply


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (edit("languages > tt > dir", "lassi/core"), "assets/"),
        (edit("languages > tt > tracked", "yes"), "tracked"),
        (edit("languages > tt > sha256", {"vadd.cpp": "0" * 64}), "sha256"),
        (edit("languages > tt > support > lassi_io.h", {"tracked": "lassi/core/files.py"}), "assets/"),
        (edit("languages > tt > support > lassi_io.h", {"tracked": "assets/../lassi/x.h"}), "support"),
        (edit(f"languages > tt > support > {KERNEL_NAME}", {"upstream": KERNEL_UPSTREAM}), "sha256"),
        (edit(f"languages > tt > support > {KERNEL_NAME}", {"upstream": KERNEL_UPSTREAM, "sha256": "g" * 64}),
         "sha256"),
        (edit("languages > tt > support > lassi_io.h", {"tracked": "assets/x.h", "upstream": "x.h"}), "support"),
        (edit("languages > tt > support > lassi_io.h", {"tracked": "assets/x.h", "mode": 1}), "support"),
        (edit("languages > tt > support > lassi_io.h", "assets/fixture/lassi_io.h"), "support"),
        (edit("languages > tt > support > vadd.cpp", {"tracked": "assets/x.h"}), "vadd.cpp"),
    ],
    ids=[
        "tracked-dir-outside-assets", "tracked-not-bool", "tracked-with-sha256", "tracked-support-outside-assets",
        "tracked-support-dotdot", "upstream-without-sha256", "upstream-bad-sha256", "two-origins",
        "unknown-entry-key", "bare-string-entry", "support-shadows-own-file",
    ],
)
def test_bad_tracked_or_support_entries_are_refused(tmp_path: Path, change: Any, message: str) -> None:
    load(tmp_path / "good", manifest_data())
    data = manifest_data()
    change(item_of(data))
    with pytest.raises(ValueError, match=message):
        load(tmp_path / "bad", data)


def test_a_language_support_name_may_not_repeat_an_item_level_one(tmp_path: Path) -> None:
    data = manifest_data()
    item_of(data)["support"] = {"lassi_io.h": "examples/lassi_io.h"}
    with pytest.raises(ValueError, match="lassi_io.h"):
        load(tmp_path, data)


# ---------------------------------------------------------------------------
# Held-out inputs, outputs, seed, and the element cap


def test_inputs_outputs_seed_and_note_are_kept(tmp_path: Path) -> None:
    item = load(tmp_path, manifest_data()).items[ITEM]
    assert [dict(entry) for entry in item.inputs] == INPUTS
    assert item.outputs == ("c",)
    assert item.seed == 7
    assert item.unpack_to_dest == "SYNTHETIC note: no fp32_dest_acc_en, no unpack_to_dest_mode"
    assert item.run_args == ()


def test_program_args_name_the_inputs_then_the_outputs(tmp_path: Path) -> None:
    item = load(tmp_path, manifest_data()).items[ITEM]
    assert registry.INPUTS_DIR == "@inputs"
    assert registry.program_args(item) == ["@inputs/a.lassiio", "@inputs/b.lassiio", "c.lassiio"]


def test_an_item_without_inputs_keeps_its_run_args(tmp_path: Path) -> None:
    data = manifest_data()
    item = item_of(data)
    for key in ("inputs", "outputs", "seed"):
        del item[key]
    item["run_args"] = ["10", "1"]
    loaded = load(tmp_path, data).items[ITEM]
    assert registry.program_args(loaded) == ["10", "1"]
    workdir = tmp_path / "work"
    workdir.mkdir()
    assert registry.stage_inputs(loaded, workdir) == ["10", "1"]
    assert list(workdir.iterdir()) == [], "an item without inputs writes nothing"


def test_stage_inputs_writes_the_seeded_inputs_under_the_reserved_directory(tmp_path: Path) -> None:
    item = load(tmp_path, manifest_data()).items[ITEM]
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "main").write_bytes(b"PLACEHOLDER artifact\n")
    args = registry.stage_inputs(item, workdir)
    assert args == ["@inputs/a.lassiio", "@inputs/b.lassiio", "c.lassiio"]
    assert sorted(path.name for path in workdir.iterdir()) == ["@inputs", "main"], "outputs are the program's"
    expected = tmp_path / "expected"
    expected.mkdir()
    generate_inputs(INPUTS, 7, expected)
    for name in ("a", "b"):
        written = workdir / "@inputs" / f"{name}.lassiio"
        assert written.read_bytes() == (expected / f"{name}.lassiio").read_bytes()
    array = read_array(workdir / "@inputs" / "a.lassiio")
    assert (array.name, array.dtype, array.shape) == ("a", "f32", (2, 4))


def test_stage_inputs_starts_from_a_fresh_directory_each_time(tmp_path: Path) -> None:
    item = load(tmp_path, manifest_data()).items[ITEM]
    workdir = tmp_path / "work"
    workdir.mkdir()
    registry.stage_inputs(item, workdir)
    first = (workdir / "@inputs" / "a.lassiio").read_bytes()
    (workdir / "@inputs" / "a.lassiio").write_bytes(b"changed by a run\n")
    (workdir / "@inputs" / "stray.txt").write_bytes(b"left by a run\n")
    registry.stage_inputs(item, workdir)
    assert sorted(path.name for path in (workdir / "@inputs").iterdir()) == ["a.lassiio", "b.lassiio"]
    assert (workdir / "@inputs" / "a.lassiio").read_bytes() == first


def test_stage_inputs_never_follows_a_link_left_at_the_reserved_name(tmp_path: Path) -> None:
    item = load(tmp_path, manifest_data()).items[ITEM]
    workdir, outside = tmp_path / "work", tmp_path / "outside"
    workdir.mkdir()
    outside.mkdir()
    try:
        os.symlink(outside, workdir / "@inputs", target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"this host cannot create a directory link ({error})")
    registry.stage_inputs(item, workdir)
    assert list(outside.iterdir()) == [], "nothing was written through the link"
    assert (workdir / "@inputs").is_dir() and not (workdir / "@inputs").is_symlink()


def test_the_element_cap_is_two_to_the_twentieth(tmp_path: Path) -> None:
    assert registry.MAX_INPUT_ELEMENTS == 1 << 20
    data = manifest_data()
    item_of(data)["inputs"][1]["shape"] = [1024, 1024]
    assert load(tmp_path / "at", data).items[ITEM].inputs[1]["shape"] == [1024, 1024]
    for shape in ([1024, 1025], [1 << 40], [1 << 63, 1 << 63, 2]):
        item_of(data)["inputs"][1]["shape"] = shape
        with pytest.raises(ValueError, match="element"):
            load(tmp_path / "past", data)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (edit("inputs", "a.lassiio"), "inputs"),
        (edit("inputs", []), "inputs"),
        (edit("inputs > 0 > dist", "normal"), "dist"),
        (edit("inputs > 0 > lo", 2.0), "lo"),
        (edit("inputs > 1 > name", "a"), "twice"),
        (edit("inputs > 0 > extra", 1), "keys"),
        (edit("inputs > 0 > shape", [-1]), "shape|dim"),
        (edit("seed", None), "seed"),
        (edit("seed", True), "seed"),
        (edit("seed", -1), "seed"),
        (edit("seed", 1 << 64), "seed"),
        (edit("outputs", None), "outputs"),
        (edit("outputs", []), "outputs"),
        (edit("outputs", ["c", "c"]), "outputs"),
        (edit("outputs", ["bad name"]), "outputs"),
        (edit("inputs", None), "inputs"),
        (edit("run_args", ["1"]), "run_args"),
        (edit("unpack_to_dest", ""), "unpack_to_dest"),
        (edit("unpack_to_dest", 1), "unpack_to_dest"),
    ],
    ids=[
        "inputs-not-a-list", "inputs-empty", "unknown-dist", "lo-above-hi", "repeated-name", "extra-key",
        "negative-dim", "no-seed", "bool-seed", "negative-seed", "seed-too-big", "no-outputs", "outputs-empty",
        "outputs-repeated", "outputs-bad-name", "outputs-without-inputs", "run-args-with-inputs",
        "empty-note", "note-not-text",
    ],
)
def test_bad_inputs_outputs_seed_or_note_are_refused(tmp_path: Path, change: Any, message: str) -> None:
    load(tmp_path / "good", manifest_data())
    data = manifest_data()
    change(item_of(data))
    with pytest.raises(ValueError, match=message):
        load(tmp_path / "bad", data)


# ---------------------------------------------------------------------------
# The pin and the placeholder


def pinned_data() -> dict[str, Any]:
    """Return the SYNTHETIC manifest pinned to toolchains/tt-metal.pin's commit and URL."""
    pin = pins.read_pin("tt-metal")
    data = manifest_data()
    data.update({"repo": pin["URL"], "commit": pin["COMMIT"], "pin": "tt-metal"})
    return data


def test_a_suite_may_name_the_pin_of_its_source(tmp_path: Path) -> None:
    assert load(tmp_path, pinned_data()).pin == "tt-metal"
    assert load(tmp_path / "none", manifest_data()).pin is None


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [("commit", "c" * 40, "COMMIT"), ("repo", "https://example.invalid/other.git", "URL"), ("pin", "no-such", "pin")],
)
def test_a_pin_that_disagrees_with_the_manifest_is_refused(tmp_path: Path, key: str, value: str, message: str) -> None:
    data = pinned_data()
    data[key] = value
    with pytest.raises(ValueError, match=message):
        load(tmp_path, data)


@pytest.mark.parametrize(
    "path",
    ["inputs", "tolerance", f"languages > tt > support > {KERNEL_NAME} > sha256", "unpack_to_dest"],
)
def test_a_placeholder_anywhere_is_refused_naming_its_key(tmp_path: Path, path: str) -> None:
    data = manifest_data()
    edit(path, "PLACEHOLDER: not filled yet")(item_of(data))
    with pytest.raises(ValueError, match=r"PLACEHOLDER") as raised:
        load(tmp_path, data)
    assert path.split(" > ")[-1] in str(raised.value)
