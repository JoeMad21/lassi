"""Tests for the prompt sets that ask for the core grid: tt-host-v2 and tt-kh-v1 (task P17.72).

tt-host-v2 is tt-host-v1, and tt-kh-v1 is tt-kh-v0, with exactly the edits below and nothing else, so a run on either
set of a pair is comparable with the other, and the v1 and v0 sets keep their results (Agent Rule 4: an opt-in set,
named by a recipe). The edits are generic: they state the goal, that the work should be spread over the device's
compute grid, and list the multi-core calls at the pin (tt-metal 5280a9cf: compute_with_storage_grid_size,
split_work_to_cores in tt_metal/api/tt-metalium/work_split.hpp, corerange_to_cores in core_coord.hpp), and they name
no item (Agent Rule 5; plans/p17-portable.md, P17.72). Only development items may steer them; the held-out items of
tt-par-v0 were committed before this task (P17.68), and the leak test there checks that no held-out file shares three
lines with any prompt set.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lassi.prompts import render

REPO = Path(__file__).resolve().parents[2]
PROMPTS = REPO / "assets" / "prompts"
BENCH = REPO / "assets" / "bench"

GOAL = (
    "  Spread the work over the device's whole compute grid: a program whose kernels run on one core is many\n"
    "  times slower than one that uses every core.\n"
)
CORES_OLD = (
    "  kernel, core, args), whose args are buffer addresses (buffer->address()) and counts; a single-core\n"
    "  program uses CoreCoord core = {0, 0};\n"
)
CORES_NEW = (
    "  kernel, core, args), whose args are buffer addresses (buffer->address()) and counts;\n"
    "- the cores: CoreCoord grid = device->compute_with_storage_grid_size(); then\n"
    "  auto [n_cores, all_cores, cores_1, cores_2, units_1, units_2] = split_work_to_cores(grid, n_units)\n"
    "  (<tt-metalium/work_split.hpp>) shares n_units units of work (tiles, or rows of tiles) over the grid,\n"
    "  units_1 to each core of cores_1 and units_2 to each core of cores_2; pass all_cores (a CoreRangeSet)\n"
    "  to CreateCircularBuffer and CreateKernel, and call SetRuntimeArgs once for each core of\n"
    "  corerange_to_cores(cores_1), then of cores_2, with that core's first unit and its unit count;\n"
)
HOST_NEVER = "  device kernels, never by the host.\n"
VALUE_NEVER = "  value must be computed by the device kernels, never by the host.\n"
KH_NEVER_GENERATE = "  kernels run on the device and compute every output value; the host program never does.\n"
KH_NEVER = "  computed by the device kernels, never by the host.\n"
KH_ACCESSOR = "first, and args.next_compile_time_args_offset() for the one after args.\n"
KH_SHARE = (
    "On several cores, each kernel reads its core's first tile and tile count from its runtime args and works\n"
    "on those tiles only.\n"
)

# (new set, old set) -> template name -> [(old text, new text)], each old text occurring exactly once.
PAIRS: dict[tuple[str, str], dict[str, list[tuple[str, str]]]] = {
    ("tt-host-v2", "tt-host-v1"): {
        "generate": [(HOST_NEVER, HOST_NEVER + GOAL), (CORES_OLD, CORES_NEW)],
        "correct": [(VALUE_NEVER, VALUE_NEVER + GOAL)],
        "correct_mismatch": [(VALUE_NEVER, VALUE_NEVER + GOAL)],
    },
    ("tt-kh-v1", "tt-kh-v0"): {
        "generate": [(KH_NEVER_GENERATE, KH_NEVER_GENERATE + GOAL), (CORES_OLD, CORES_NEW),
                     (KH_ACCESSOR, KH_ACCESSOR + KH_SHARE)],
        "correct": [(KH_NEVER, KH_NEVER + GOAL)],
        "correct_mismatch": [(KH_NEVER, KH_NEVER + GOAL)],
    },
}
NEW_LINES_MAX = 110


def edited(edits: list[tuple[str, str]], text: str, where: str) -> str:
    """Return text with each (old, new) edit applied; each old text must occur exactly once."""
    for old, new in edits:
        assert text.count(old) == 1, f"{where}: {old!r} occurs {text.count(old)} times"
        text = text.replace(old, new)
    return text


@pytest.mark.parametrize("pair", sorted(PAIRS), ids=lambda pair: pair[0])
def test_the_new_set_is_the_old_with_exactly_the_edits(pair: tuple[str, str]) -> None:
    """Every template of the new set is the old set's with exactly the listed edits; no file added or lost."""
    new_set, old_set = pair
    new_dir, old_dir = PROMPTS / new_set, PROMPTS / old_set
    assert new_dir.is_dir(), f"assets/prompts/{new_set} does not exist (task P17.72)"
    names = sorted(path.name for path in old_dir.iterdir())
    assert sorted(path.name for path in new_dir.iterdir()) == names
    for name in names:
        old = (old_dir / name).read_bytes().decode("ascii")
        new = (new_dir / name).read_bytes()
        assert new.isascii() and b"\r" not in new, f"{new_set}/{name}"
        expected = edited(PAIRS[pair].get(name.removesuffix(".txt"), []), old, f"{old_set}/{name}")
        assert new.decode("ascii") == expected, f"{new_set}/{name}"


@pytest.mark.parametrize("pair", sorted(PAIRS), ids=lambda pair: pair[0])
def test_the_new_set_states_the_goal_and_the_multi_core_calls(pair: tuple[str, str]) -> None:
    """generate states the grid goal and the multi-core calls; the corrections state the goal; one core is gone."""
    new_set = pair[0]
    generate = " ".join((PROMPTS / new_set / "generate.txt").read_text(encoding="ascii").split())
    for phrase in ("whole compute grid", "compute_with_storage_grid_size()", "split_work_to_cores(grid, n_units)",
                   "<tt-metalium/work_split.hpp>", "corerange_to_cores(cores_1)", "first unit and its unit count"):
        assert phrase in generate, f"{new_set}/generate.txt lacks {phrase!r}"
    assert "a single-core program uses CoreCoord core = {0, 0}" not in generate
    for name in ("correct", "correct_mismatch"):
        text = " ".join((PROMPTS / new_set / f"{name}.txt").read_text(encoding="ascii").split())
        assert "whole compute grid" in text, f"{new_set}/{name}.txt lacks the goal"


@pytest.mark.parametrize("pair", sorted(PAIRS), ids=lambda pair: pair[0])
def test_the_new_lines_are_short(pair: tuple[str, str]) -> None:
    """Every line an edit adds is at most NEW_LINES_MAX characters, as the v1 sets keep theirs."""
    for edits in PAIRS[pair].values():
        for _, new in edits:
            assert max(len(line) for line in new.splitlines()) <= NEW_LINES_MAX


@pytest.mark.parametrize("pair", sorted(PAIRS), ids=lambda pair: pair[0])
def test_each_edited_template_renders_with_its_old_fields(pair: tuple[str, str]) -> None:
    """The edits add no template field: each edited template renders with the fields its old template uses."""
    new_set, old_set = pair
    for name in PAIRS[pair]:
        old = (PROMPTS / old_set / f"{name}.txt").read_text(encoding="ascii")
        new = (PROMPTS / new_set / f"{name}.txt").read_text(encoding="ascii")
        fields = sorted(set(re.findall(r"\$(\w+)", old)))
        assert sorted(set(re.findall(r"\$(\w+)", new))) == fields, f"{new_set}/{name}.txt"
        rendered = render(new_set, name, {field: f"<{field}>" for field in fields})
        assert all(f"<{field}>" in rendered for field in fields), f"{new_set}/{name}.txt"


def item_names() -> set[str]:
    """Return every item name of every committed suite manifest."""
    import yaml

    names: set[str] = set()
    for manifest in BENCH.glob("*.yaml"):
        names |= set(yaml.safe_load(manifest.read_text(encoding="utf-8"))["items"])
    return names


@pytest.mark.parametrize("pair", sorted(PAIRS), ids=lambda pair: pair[0])
def test_the_edits_name_no_item(pair: tuple[str, str]) -> None:
    """No line an edit adds names an item of any suite (Agent Rule 5)."""
    added = "\n".join(new for edits in PAIRS[pair].values() for _, new in edits)
    words = set(re.findall(r"[A-Za-z0-9_]+", added))
    assert not words & item_names(), sorted(words & item_names())
