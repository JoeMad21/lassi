"""Tests for the held-out items of the multi-core suite tt-par-v0 (task P17.68; OQ-129).

Bible: Benchmark Suites (a suite's splits), Harness Contract (lassi_io, the
2**20-element input cap, the CPU -> TT guard), Oracles (binary_io), ttsim
Facts (unpack_to_dest), Agent Rule 5. Plan: plans/p17-portable.md, "The
multi-core C++ -> TT test", P17.67 (each C++ program says nothing of the
device, its cores, or how the reference splits the work; the same holds for
P17.68's), and P17.68. What they fix:

- assets/bench/tt-par-v0.yaml's eval items are exactly the five
  computations the owner chose (row_sum, col_max, transpose_2d,
  row_softmax, col_scale), each given for evaluation and refused to
  training, prompt tuning, and corpus harvest (Agent Rule 5), each with the
  seed chosen for it, distinct within the manifest and from every
  tt-pairs-v0 seed.
- Each eval item's inputs have the plan's shapes, each within
  MAX_INPUT_ELEMENTS; its one output has the name, dtype, and rank its
  programs write; its units of work (tile rows, tile columns, or output
  tiles) number more than an 8x9 grid's 72 cores; its tolerance and
  analytic work are the ones set for it, the work following from the
  shapes.
- Each eval item has languages cpp and tt, tracked under
  assets/bench/tt-par-v0/<item>/; the tt support holds lassi_io.h and
  every file under tt/kernels/, each a tracked entry at its kernels/...
  tail, and nothing else; the TT program names each support kernel as a
  string literal and no other kernel path; every program and kernel opens
  with the P17.68 provenance line and is plain ASCII with LF line ends.
- Each TT program reads host_compute False under the CPU -> TT guard, and
  its unpack_to_dest note holds for its code (no fp32_dest_acc_en, no
  unpack_to_dest_mode, Float16_b the only DataFormat).
- The leak test: no held-out file shares a run of three consecutive
  non-blank lines (each stripped, inner whitespace collapsed, and each
  holding a letter or digit) with any tracked file under assets/prompts/,
  assets/skeletons/, assets/examples/, assets/train/, or assets/context/,
  the places whose text can reach a prompt or training (P17.68); another
  item's program is not among them, so shared boilerplate with one (an
  include block, the start of main) is no leak. Its built half, marked remote,
  makes the same check over the built context packs (assets/context/<pack>/,
  beside each MANIFEST.yaml) and the built retrieval indexes
  ($LASSI_SCRATCH/rag/<corpus>@<VERSION>/index.json), whose text is not
  tracked, and skips where none is built.
- Every tt-par-v0 C++ program, development and held-out, is neutral: no
  OpenMP pragma, no omp.h or <thread> include, and outside its first line
  no word that names the device, its cores, its grid, sharding,
  multicast, kernels, Tensix, or math fidelity.

Items are read through Suite.item with purpose eval only; nothing here
trains, tunes a prompt, or harvests (Agent Rule 5). Every count here is
arithmetic over the items' shapes; none is a measurement. Until the
manifest holds the eval items, every eval test fails naming the item.
"""

from __future__ import annotations

import functools
import math
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

from lassi.bench import registry
from lassi.core.tolerance import Tolerance
from lassi.harness.contract import MAX_INPUT_ELEMENTS
from lassi.prompts import assets as prompt_assets
from lassi.prompts import rag_corpus
from lassi.prompts.retrieval import load_index
from lassi.toolchains._cxx_scan import IDENT, LITERAL, lex
from lassi.toolchains.pins import read_pin
from lassi.toolchains.ttmetal_build import TtMetalHost

REPO = Path(__file__).resolve().parents[2]
SUITE = "tt-par-v0"
MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
MANIFEST_NAME = f"assets/bench/{SUITE}.yaml"
SIBLING = REPO / "assets" / "bench" / "tt-pairs-v0.yaml"
PROGRAMS = REPO / "assets" / "bench" / SUITE
HARNESS = "assets/harness/c/lassi_io.h"
# The five held-out computations the owner chose on 2026-11-17 (OQ-129), in the plan's order.
HELD_OUT = ("row_sum", "col_max", "transpose_2d", "row_softmax", "col_scale")
# The five development items of task P17.67, whose C++ programs the neutrality test checks too.
DEV_ITEMS = (
    "matmul_multicore_reuse",
    "matmul_multicore_reuse_mcast",
    "pad_multi_core",
    "vecadd_multi_core",
    "vecadd_sharding",
)
LANGUAGES = ("cpp", "tt")
FORBIDDEN = ("train", "prompt-tuning", "harvest")
# Each eval item's seed, chosen for task P17.68 (21 to 25), clear of the dev items' 11 to 15 and tt-pairs-v0's 1 to 5.
SEEDS = {"row_sum": 21, "col_max": 22, "transpose_2d": 23, "row_softmax": 25, "col_scale": 24}
# Each eval item's inputs, in order, as (name, dtype, shape, dist, lo, hi): the plan's shapes (P17.68) and the ranges
# set in task P17.68 with the items.
INPUTS = {
    "row_sum": [("a", "bf16", [4096, 256], "uniform", -1.0, 1.0)],
    "col_max": [("a", "bf16", [256, 4096], "uniform", -1.0, 0.0625)],
    "transpose_2d": [("a", "bf16", [1024, 1024], "uniform", -4.0, 4.0)],
    "row_softmax": [("x", "bf16", [4096, 256], "uniform", 64.0, 128.0)],
    "col_scale": [
        ("a", "bf16", [1024, 1024], "uniform", -1.0, 1.0),
        ("s", "bf16", [1024], "uniform", -2.0, 2.0),
    ],
}
# Each eval item's one output as (name, dtype, shape): row_sum's sums are widened to f32 on both sides; every other
# output is bf16, as the device writes it.
OUTPUTS = {
    "row_sum": ("s", "f32", [4096]),
    "col_max": ("m", "bf16", [4096]),
    "transpose_2d": ("b", "bf16", [1024, 1024]),
    "row_softmax": ("y", "bf16", [4096, 256]),
    "col_scale": ("c", "bf16", [1024, 1024]),
}
# The tile edge: 32 x 32 elements per tile.
TILE_EDGE = 32
# The largest compute grid the spike names: 8 x 9 (ttsim's), over card 0's 8 x 7.
GRID_CORES = 8 * 9
# Each eval item's units of work at these shapes: tile rows (row_sum, row_softmax), tile columns (col_max), or output
# tiles (transpose_2d, col_scale). Arithmetic from the shapes.
WORK_UNITS = {"row_sum": 128, "col_max": 128, "transpose_2d": 1024, "row_softmax": 128, "col_scale": 1024}
# Each eval item's tolerance, set in task P17.68 from the device's arithmetic (exact equality where it is exact).
TOLERANCES = {
    "row_sum": Tolerance(metric="max_abs", threshold=4.0),
    "col_max": Tolerance(metric="max_abs", threshold=0),
    "transpose_2d": Tolerance(metric="max_abs", threshold=0),
    "row_softmax": Tolerance(metric="max_abs", threshold=0.03),
    "col_scale": Tolerance(metric="max_abs", threshold=0.015625),
}
# bf16 on the device: two bytes per element.
DEVICE_BYTES = 2
# Each eval item's analytic work as (flop, ops, bytes), set with the items in task P17.68 (arithmetic, never a
# measurement).
WORK = {
    "row_sum": (1_044_480, {"add": 1_044_480}, 2_105_344),
    "col_max": (None, {"max": 1_044_480}, 2_105_344),
    "transpose_2d": (None, {}, 4_194_304),
    "row_softmax": (
        4_186_112,
        {"max": 1_044_480, "sub": 1_048_576, "exp": 1_048_576, "add": 1_044_480, "recip": 4096, "mul": 1_048_576},
        4_194_304,
    ),
    "col_scale": (1_048_576, {"mul": 1_048_576}, 4_196_352),
}
FIRST_LINE_START = "// Written for LASSI-DF tt-par-v0 (task P17.68); a held-out item:"
SOURCE_SUFFIXES = (".cpp", ".cc", ".c", ".hpp", ".h")
KERNELS_DIR = "kernels"
# The identifiers that would take the path ttsim issue #18 breaks (P4.9), never named by a held-out TT program.
DEST_NAMES = ("fp32_dest_acc_en", "unpack_to_dest_mode", "UnpackToDestFp32")
# The leak test: a run is this many consecutive non-blank lines; the tracked trees it reads (P17.68's list).
RUN_LINES = 3
LEAK_ROOTS = ("assets/prompts", "assets/skeletons", "assets/examples", "assets/train", "assets/context")
_TEXT = re.compile(r"[A-Za-z0-9]")
# The neutrality test's words, matched case-insensitively against the words of a C++ program's text after its first
# line, a word being a run of letters split at every other character and at a lower-to-upper case change.
NEUTRAL_WORDS = frozenset({
    "device", "devices", "core", "cores", "grid", "grids", "shard", "shards", "sharded", "sharding",
    "multicast", "multicasts", "multicasting", "kernel", "kernels", "tensix", "fidelity",
})
_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+")
# The built half of the leak test: where the runner reads each built text (lassi.core.runner _retrieval_index).
CONTEXT_ROOT = REPO / "assets" / "context"
RAG_ROOT = REPO / "assets" / "rag"
INDEX_FILE = "index.json"


def raw() -> dict[str, Any]:
    """Return the manifest as YAML reads it; fail naming it when it does not exist."""
    if not MANIFEST.is_file():
        pytest.fail(f"{MANIFEST_NAME} does not exist (task P17.67)")
    return yaml.safe_load(MANIFEST.read_bytes().decode("utf-8"))


@functools.lru_cache(maxsize=1)
def _loaded() -> registry.Suite:
    """Load the manifest once with load_suite."""
    return registry.load_suite(MANIFEST)


def suite() -> registry.Suite:
    """Return the loaded manifest; fail naming it when it does not exist."""
    if not MANIFEST.is_file():
        pytest.fail(f"{MANIFEST_NAME} does not exist (task P17.67)")
    return _loaded()


def item_for_eval(item: str) -> registry.SuiteItem:
    """Return `item` as Suite.item gives it for evaluation; fail naming it when the manifest lacks it."""
    loaded = suite()
    if item not in loaded.items:
        pytest.fail(f"{MANIFEST_NAME} has no item {item} (task P17.68 adds the held-out items)")
    return loaded.item(item, purpose="eval")


def program(item: str, language: str) -> Path:
    """Return the path of the item's program in `language`, as the loaded manifest names it."""
    spec = item_for_eval(item).languages[language]
    return REPO / spec.dir / spec.files[0]


def read_text(path: Path, what: str) -> str:
    """Return the UTF-8 text of `path`; fail naming `what` when it does not exist."""
    if not path.is_file():
        pytest.fail(f"{what}: {path} does not exist")
    return path.read_bytes().decode("utf-8")


def program_text(item: str, language: str) -> str:
    """Return the text of the item's program in `language`; fail when it does not exist."""
    return read_text(program(item, language), f"{item} ({language})")


def support_kernels(item: str) -> dict[str, str]:
    """Return {support name: tracked path} of the item's tt support kernels (every entry but lassi_io.h)."""
    support = item_for_eval(item).languages["tt"].support
    return {name: entry.path for name, entry in support.items() if name != "lassi_io.h"}


def held_out_files(item: str) -> dict[str, str]:
    """Return {repository path: text} of the item's C++ program, TT host program, and tt support kernels."""
    paths = [program(item, language) for language in LANGUAGES]
    paths += [REPO / path for path in support_kernels(item).values()]
    return {path.relative_to(REPO).as_posix(): read_text(path, item) for path in paths}


def kernel_literals(text: str, name: str) -> set[str]:
    """Return the string literals of a program's code (comments aside) that read as a kernel or source path."""
    found = set()
    for token in lex(text, name).tokens:
        if token.kind != LITERAL or not token.text.startswith('"'):
            continue
        value = token.text[1:-1]
        if not value or any(char.isspace() for char in value):
            continue
        if value.endswith(SOURCE_SUFFIXES) or KERNELS_DIR in PurePosixPath(value).parts[:-1]:
            found.add(value)
    return found


def output_writes(text: str, name: str) -> list[tuple[str, str, str]]:
    """Return (array name, dtype code, rank) as written in each lassi_io_write call of a program's code."""
    lexed = lex(text, name)
    tokens = lexed.tokens
    calls = []
    for index, token in enumerate(tokens[:-1]):
        if token.kind != IDENT or token.text != "lassi_io_write" or tokens[index + 1].text != "(":
            continue
        args: list[str] = [""]
        position, closer = index + 2, lexed.match[index + 1]
        while position < closer:
            if tokens[position].text == ",":
                args.append("")
                position += 1
                continue
            end = lexed.match[position] + 1 if tokens[position].text in "([{" else position + 1
            args[-1] += "".join(part.text for part in tokens[position:end])
            position = end
        calls.append((args[1].strip('"'), args[2], args[3]) if len(args) >= 4 else ("", "", ""))
    return calls


def output_shape(item: str, inputs: tuple[Any, ...]) -> list[int]:
    """Return the shape of the item's one output, from its first input's shape and its computation."""
    rows, cols = list(inputs[0]["shape"])
    if item == "row_sum":
        return [rows]
    if item == "col_max":
        return [cols]
    if item == "transpose_2d":
        return [cols, rows]
    return [rows, cols]


def work_units(item: str, inputs: tuple[Any, ...]) -> int:
    """Return the item's units of work: tile rows, tile columns, or output tiles of its output."""
    rows, cols = list(inputs[0]["shape"])
    assert rows % TILE_EDGE == 0 and cols % TILE_EDGE == 0, f"{item}: [{rows}, {cols}] is not whole tiles"
    if item in ("row_sum", "row_softmax"):
        return rows // TILE_EDGE
    if item == "col_max":
        return cols // TILE_EDGE
    out_rows, out_cols = output_shape(item, inputs)
    return (out_rows // TILE_EDGE) * (out_cols // TILE_EDGE)


def expected_work(item: str, inputs: tuple[Any, ...]) -> tuple[int | None, dict[str, int], int]:
    """Return (flop, ops, bytes) the item's shapes give: each input read and the output written once, in bf16."""
    rows, cols = list(inputs[0]["shape"])
    read = sum(math.prod(entry["shape"]) for entry in inputs)
    nbytes = DEVICE_BYTES * (read + math.prod(output_shape(item, inputs)))
    if item == "row_sum":
        return rows * (cols - 1), {"add": rows * (cols - 1)}, nbytes
    if item == "col_max":
        return None, {"max": cols * (rows - 1)}, nbytes
    if item == "transpose_2d":
        return None, {}, nbytes
    if item == "row_softmax":
        ops = {
            "max": rows * (cols - 1), "sub": rows * cols, "exp": rows * cols,
            "add": rows * (cols - 1), "recip": rows, "mul": rows * cols,
        }
        return ops["max"] + ops["sub"] + ops["add"] + ops["mul"], ops, nbytes
    return rows * cols, {"mul": rows * cols}, nbytes


def runs(text: str) -> dict[tuple[str, ...], int]:
    """Return each run of RUN_LINES consecutive non-blank lines of `text`, each holding a letter or digit.

    Each line is stripped with its inner whitespace collapsed; blank lines
    are skipped. A run maps to the 1-based line number of its first line.
    """
    lines = text.replace("\r\n", "\n").split("\n")
    kept = [(number, " ".join(line.split())) for number, line in enumerate(lines, start=1) if line.strip()]
    found: dict[tuple[str, ...], int] = {}
    for index in range(len(kept) - RUN_LINES + 1):
        window = kept[index:index + RUN_LINES]
        if all(_TEXT.search(line) for _, line in window):
            found.setdefault(tuple(line for _, line in window), window[0][0])
    return found


def git_tracked(*roots: str) -> list[str]:
    """Return the repository paths git tracks under `roots`, sorted."""
    done = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "-z", "--", *roots], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, f"git ls-files failed: {done.stderr}"
    return sorted(name for name in done.stdout.split("\0") if name)


@functools.lru_cache(maxsize=1)
def tracked_corpus() -> dict[str, str]:
    """Return {path: text} of every tracked file under LEAK_ROOTS."""
    names = git_tracked(*LEAK_ROOTS)
    return {name: (REPO / name).read_bytes().decode("utf-8", errors="replace") for name in names}


def shared_runs(files: dict[str, str], corpus: dict[str, str]) -> list[str]:
    """Return one line per run a held-out file shares with a corpus text: the two places and the run's lines."""
    index: dict[tuple[str, ...], tuple[str, int]] = {}
    for name, text in corpus.items():
        for run, line in runs(text).items():
            index.setdefault(run, (name, line))
    hits = []
    for name, text in files.items():
        for run, line in runs(text).items():
            if run in index:
                other, other_line = index[run]
                hits.append(f"{name}:{line} and {other}:{other_line}: {' | '.join(run)!r}")
    return hits


def built_texts() -> tuple[dict[str, str], list[str]]:
    """Return ({label: text} of every built context pack entry and index chunk found, [what is not built]).

    A context pack is built when every entry its MANIFEST.yaml lists has
    its <key>.txt beside it (lassi.prompts.assets load_tree reads it,
    checking each sha256); a retrieval index is built when
    $LASSI_SCRATCH/rag/<corpus>@<VERSION of its source pin>/index.json
    exists (lassi.prompts.retrieval load_index reads it with the
    manifest's sha256). A pack with only some entries present fails in
    load_tree, naming its generator.
    """
    texts: dict[str, str] = {}
    absent: list[str] = []
    for manifest in sorted(CONTEXT_ROOT.glob(f"*/{prompt_assets.MANIFEST}")):
        pack = manifest.parent
        keys = [entry["key"] for entry in yaml.safe_load(manifest.read_text(encoding="utf-8"))["entries"]]
        if not any((pack / f"{key}.txt").is_file() for key in keys):
            absent.append(f"assets/context/{pack.name}")
            continue
        for key, text in prompt_assets.load_tree(pack).items():
            texts[f"assets/context/{pack.name}/{key}.txt"] = text
    scratch = os.environ.get("LASSI_SCRATCH", "")
    for manifest in sorted(RAG_ROOT.glob(f"*/{prompt_assets.MANIFEST}")):
        document = rag_corpus.read_manifest(manifest)
        version = read_pin(document["source"]["pin"])["VERSION"]
        path = Path(scratch) / "rag" / f"{document['corpus']}@{version}" / INDEX_FILE if scratch else None
        if path is None or not path.is_file():
            absent.append(f"the {document['corpus']} index ($LASSI_SCRATCH/rag/{document['corpus']}@{version})")
            continue
        for chunk in load_index(path, sha256=document["index"]["sha256"]).chunks:
            texts[f"{path}:{chunk.path}:{chunk.start_line}"] = chunk.text
    return texts, absent


def words_after_first_line(text: str) -> list[tuple[int, str]]:
    """Return (line number, word) for each word of `text` after its first line, every tt-par-v0 item name removed."""
    names = sorted(set(HELD_OUT) | set(DEV_ITEMS) | set(raw()["items"]), key=len, reverse=True)
    found = []
    for number, line in enumerate(text.split("\n")[1:], start=2):
        for name in names:
            line = line.replace(name, " ")
        found += [(number, word) for word in _WORD.findall(line)]
    return found


# ---------------------------------------------------------------------------
# The eval split and the seeds


def test_the_eval_items_are_exactly_the_five_the_owner_chose() -> None:
    """The items in split eval are row_sum, col_max, transpose_2d, row_softmax, and col_scale (OQ-129)."""
    data = raw()
    assert sorted(name for name, entry in data["items"].items() if entry.get("split") == "eval") == sorted(HELD_OUT)
    loaded = suite()
    assert sorted(name for name, item in loaded.items.items() if item.split == "eval") == sorted(HELD_OUT)


@pytest.mark.parametrize("item", HELD_OUT)
def test_an_eval_item_is_given_for_evaluation_only(item: str) -> None:
    """An eval item serves evaluation and is refused to training, prompt tuning, and corpus harvest (Agent Rule 5)."""
    assert item_for_eval(item).split == "eval"
    for purpose in FORBIDDEN:
        with pytest.raises(registry.EvalSplitError, match="eval split"):
            suite().item(item, purpose=purpose)


def test_each_eval_seed_is_its_own_and_clear_of_tt_pairs_v0() -> None:
    """Each eval item has its chosen seed, no other item of the manifest shares it, and no tt-pairs-v0 item has it."""
    for name in HELD_OUT:
        item_for_eval(name)
    items = suite().items
    assert {name: items[name].seed for name in HELD_OUT} == SEEDS
    seeds = [item.seed for item in items.values() if item.seed is not None]
    assert len(seeds) == len(set(seeds)), f"seeds repeat within {SUITE}: {sorted(seeds)}"
    sibling = {item.seed for item in registry.load_suite(SIBLING).items.values()}
    assert not set(SEEDS.values()) & sibling, f"an eval seed is a tt-pairs-v0 seed: {sorted(sibling)}"


# ---------------------------------------------------------------------------
# Shapes, outputs, units of work, tolerances, and work


@pytest.mark.parametrize("item", HELD_OUT)
def test_an_eval_items_inputs_have_the_plans_shapes_within_the_cap(item: str) -> None:
    """Each input has its name, dtype, shape, and range, and at most MAX_INPUT_ELEMENTS elements; no run_args."""
    found = item_for_eval(item)
    assert found.run_args == ()
    declared = [(e["name"], e["dtype"], list(e["shape"]), e["dist"], e["lo"], e["hi"]) for e in found.inputs]
    assert declared == INPUTS[item], f"{item}: inputs {declared}"
    for entry in found.inputs:
        count = math.prod(entry["shape"])
        assert 0 < count <= MAX_INPUT_ELEMENTS, f"{item}: {entry['name']} holds {count} elements"


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("item", HELD_OUT)
def test_an_eval_items_output_is_the_one_its_programs_write(item: str, language: str) -> None:
    """One output, named in the manifest; each program writes it once with its dtype and its inputs' output rank."""
    found = item_for_eval(item)
    name, dtype, shape = OUTPUTS[item]
    assert found.outputs == (name,), f"{item}: outputs {found.outputs}"
    assert output_shape(item, found.inputs) == shape
    writes = output_writes(program_text(item, language), f"{item}.cpp")
    assert writes == [(name, f"LASSI_IO_{dtype.upper()}", f"{len(shape)}u")], f"{item} ({language}): {writes}"


@pytest.mark.parametrize("item", HELD_OUT)
def test_an_eval_items_units_of_work_outnumber_an_8x9_grids_cores(item: str) -> None:
    """Tile rows (row_sum, row_softmax), tile columns (col_max), or output tiles number more than 72 cores."""
    units = work_units(item, item_for_eval(item).inputs)
    assert units == WORK_UNITS[item]
    assert units > GRID_CORES


@pytest.mark.parametrize("item", HELD_OUT)
def test_an_eval_items_tolerance_is_the_one_set_for_it(item: str) -> None:
    """The tolerance set from the device's arithmetic: exact for col_max and transpose_2d, max_abs bounds otherwise."""
    assert item_for_eval(item).tolerance == TOLERANCES[item]


@pytest.mark.parametrize("item", HELD_OUT)
def test_an_eval_item_declares_the_work_its_shapes_give(item: str) -> None:
    """work is present, its flop, ops, and bytes follow from the shapes, and they equal the figures set for it."""
    found = item_for_eval(item)
    work = found.work
    assert isinstance(work, registry.Work), f"{item} declares no work"
    declared = (work.flop, dict(work.ops), work.bytes)
    assert declared == expected_work(item, found.inputs)
    assert declared == WORK[item]
    assert work.basis.strip() and work.basis.isascii()


# ---------------------------------------------------------------------------
# Languages, support files, and the tracked programs


@pytest.mark.parametrize("item", HELD_OUT)
def test_an_eval_items_languages_are_tracked_cpp_and_tt(item: str) -> None:
    """cpp and tt, each tracked at assets/bench/tt-par-v0/<item>/<language>/<item>.cpp; cpp gets lassi_io.h alone."""
    found = item_for_eval(item)
    assert sorted(found.languages) == list(LANGUAGES)
    for language in LANGUAGES:
        spec = found.languages[language]
        assert spec.tracked and spec.dir == f"assets/bench/{SUITE}/{item}/{language}"
        assert spec.files == (f"{item}.cpp",)
        assert spec.reference is None
    assert found.support == {}, "every support file belongs to one language"
    cpp, tt = found.languages["cpp"].support, found.languages["tt"].support
    assert set(cpp) == {"lassi_io.h"} and (cpp["lassi_io.h"].tracked, cpp["lassi_io.h"].path) == (True, HARNESS)
    assert (tt["lassi_io.h"].tracked, tt["lassi_io.h"].path) == (True, HARNESS)


@pytest.mark.parametrize("item", HELD_OUT)
def test_an_eval_items_tt_support_is_lassi_io_and_its_tracked_kernels(item: str) -> None:
    """The tt support is lassi_io.h and each file under tt/kernels/, tracked at its kernels/... tail, nothing else."""
    tt = item_for_eval(item).languages["tt"].support
    root = PROGRAMS / item / "tt"
    on_disk = sorted(path.relative_to(root).as_posix() for path in (root / KERNELS_DIR).rglob("*") if path.is_file())
    assert on_disk, f"{item}: no kernel under {root / KERNELS_DIR}"
    assert sorted(tt) == sorted(["lassi_io.h", *on_disk]), f"{item}: tt support {sorted(tt)}"
    for name in on_disk:
        entry = tt[name]
        assert entry.tracked, f"{item}: {name} is a tracked file"
        assert entry.path == f"assets/bench/{SUITE}/{item}/tt/{name}", f"{item}: {name} at {entry.path}"
    files = suite().support_files(item, REPO, purpose="eval", language="tt")
    assert sorted(files) == sorted(tt), f"{item}: support_files reads {sorted(files)}"


@pytest.mark.parametrize("item", HELD_OUT)
def test_an_eval_items_directory_holds_only_its_programs_and_kernels(item: str) -> None:
    """assets/bench/tt-par-v0/<item>/ holds its two programs and its tt kernels, and no other file."""
    item_for_eval(item)
    root = PROGRAMS / item
    on_disk = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    expected = sorted([f"cpp/{item}.cpp", f"tt/{item}.cpp", *(f"tt/{name}" for name in support_kernels(item))])
    assert on_disk == expected, f"{item}: files {on_disk}"


@pytest.mark.parametrize("item", HELD_OUT)
def test_every_held_out_file_opens_with_its_provenance_line_and_is_plain_ascii(item: str) -> None:
    """Each program and kernel is plain ASCII with LF line ends and opens with the P17.68 provenance line."""
    for name, text in held_out_files(item).items():
        assert text.isascii() and "\r" not in text, f"{name}: plain ASCII with LF line ends"
        first = text.split("\n", 1)[0]
        assert first.startswith(FIRST_LINE_START), f"{name}: {first}"


@pytest.mark.parametrize("item", HELD_OUT)
def test_an_eval_items_cpp_program_is_host_only(item: str) -> None:
    """The C++ program includes lassi_io.h and nothing from tt-metal, and names no kernel path."""
    text = program_text(item, "cpp")
    includes = [name for name, _ in lex(text, f"{item}.cpp").includes]
    assert "lassi_io.h" in includes
    assert not [name for name in includes if name.startswith(("tt-metalium/", "tt_metal/", "tt-", "api/"))], includes
    assert kernel_literals(text, f"{item}.cpp") == set(), f"{item}: the C++ program names a kernel"


@pytest.mark.parametrize("item", HELD_OUT)
def test_an_eval_items_tt_program_names_exactly_its_support_kernels(item: str) -> None:
    """The TT program names each support kernel as a string literal, and no other kernel or source path."""
    named = kernel_literals(program_text(item, "tt"), f"{item}.cpp")
    kernels = set(support_kernels(item))
    assert named == kernels, f"{item}: names {sorted(named)}, support {sorted(kernels)}"


@pytest.mark.parametrize("item", HELD_OUT)
def test_an_eval_items_tt_program_passes_the_cpu_to_tt_guard(item: str) -> None:
    """The guard reads host_compute False with no diagnostic, with lassi_io.h alone and with the full tt support."""
    text = program_text(item, "tt")
    harness = {"lassi_io.h": (REPO / HARNESS).read_bytes().decode("utf-8")}
    support = suite().support_files(item, REPO, purpose="eval", language="tt")
    for given in (harness, support):
        reading = TtMetalHost.host_compute_guard({f"{item}.cpp": text}, given)
        assert reading.host_compute is False, f"{item}: host_compute {reading.host_compute}; {reading.diagnostics}"
        assert [diagnostic.code for diagnostic in reading.diagnostics] == [], f"{item}: {reading.diagnostics}"


@pytest.mark.parametrize("item", HELD_OUT)
def test_an_eval_items_unpack_to_dest_note_holds_for_its_code(item: str) -> None:
    """The note says not taken; no program or kernel names fp32_dest_acc_en or unpack_to_dest_mode; only Float16_b."""
    note = item_for_eval(item).unpack_to_dest
    assert note and "unpack_to_dest" in note and note.startswith("not taken"), f"{item}: {note!r}"
    host_formats: set[str] = set()
    for name, text in held_out_files(item).items():
        if name.endswith(f"/cpp/{item}.cpp"):
            continue
        tokens = lex(text, PurePosixPath(name).name).tokens
        names = {token.text for token in tokens if token.kind == IDENT}
        for dest in DEST_NAMES:
            assert dest not in names, f"{name}: names {dest} (ttsim issue #18's path; P4.9)"
        formats = {
            tokens[index + 2].text
            for index in range(len(tokens) - 2)
            if tokens[index].text == "DataFormat" and tokens[index + 1].text == "::"
        }
        assert formats <= {"Float16_b"}, f"{name}: every circular buffer is Float16_b, got {sorted(formats)}"
        if name.endswith(f"/tt/{item}.cpp"):
            host_formats = formats
    assert host_formats == {"Float16_b"}, f"{item}: the TT host program names its circular buffers' format"


# ---------------------------------------------------------------------------
# The leak test


@pytest.mark.parametrize("item", HELD_OUT)
def test_no_held_out_file_shares_three_lines_with_tracked_prompt_material(item: str) -> None:
    """No run of three non-blank lines of the item's files is in a tracked prompt, skeleton, example, or pack."""
    files = held_out_files(item)
    corpus = tracked_corpus()
    assert any(name.startswith("assets/prompts/") for name in corpus), "git lists the tracked prompt sets"
    hits = shared_runs(files, corpus)
    assert hits == [], f"{item} shares {len(hits)} runs of {RUN_LINES} lines: " + "; ".join(hits[:20])


@pytest.mark.remote
def test_no_held_out_file_shares_three_lines_with_a_built_context_pack_or_index() -> None:
    """The same check over the built context packs and retrieval indexes; skipped where none is built."""
    texts, absent = built_texts()
    if not texts:
        pytest.skip(f"no built context pack or retrieval index here: {', '.join(absent)}")
    files = {name: text for item in HELD_OUT for name, text in held_out_files(item).items()}
    hits = shared_runs(files, texts)
    assert hits == [], f"held-out files share {len(hits)} runs of {RUN_LINES} lines: " + "; ".join(hits[:20])


# ---------------------------------------------------------------------------
# Neutrality of every C++ program


@pytest.mark.parametrize("item", DEV_ITEMS + HELD_OUT)
def test_a_cpp_program_says_nothing_of_the_device_or_the_split(item: str) -> None:
    """No OpenMP pragma, omp.h, or <thread>; after line 1 no word naming the device, cores, grid, or kernels."""
    path = program(item, "cpp")
    text = read_text(path, f"{item} (cpp)")
    lexed = lex(text, path.name)
    includes = [name for name, _ in lexed.includes]
    assert not {"omp.h", "thread"} & set(includes), f"{item}: includes {includes}"
    pragmas = [d.text for d in lexed.directives if [t.text for t in d.tokens[1:3]] == ["pragma", "omp"]]
    assert pragmas == [], f"{item}: {pragmas}"
    named = [f"line {number}: {word}" for number, word in words_after_first_line(text) if word.lower() in NEUTRAL_WORDS]
    assert named == [], f"{item}: the C++ program names " + "; ".join(named)
