"""Tests for the suite bale-v0: three apps of bale as serial C++ and host OpenMP C++ pairs (task P17.40).

Bible: Benchmark Suites (a bale-v0 row; split rules; manifest keys; held-out
inputs; Item tolerance), Harness Contract (lassi_io.h; held-out input files),
Oracles (binary_io: integer outputs match exactly), Agent Rules 4 and 5.
Plan: plans/p17-portable.md, P17.40. Evidence: plans/spikes/bale.md (the
owner's answers of 2026-10-26: scope (a), every item unassigned, fetch only).

The contract these tests fix:

- assets/bench/bale-v0.yaml loads with load_suite as a fetched suite: repo
  https://github.com/jdevinney/bale.git at commit 1b8f673b, no pin. Its
  three items, histo, ig, and triangle, are all unassigned, with the seeds
  4101, 4102, and 4103, held-out inputs, outputs, and an exact tolerance
  (max_abs 0) as the spike's manifest sketch gives them, and two languages,
  cpp (serial C++) and omp (OpenMP C++ on the host CPU), each tracked at
  assets/bench/bale-v0/<item>/<language>/<item>.cpp with two support files:
  lassi_io.h, tracked, and upstream/<item>.c, bale's serial C file named by
  its path at the pin and its sha256 (the spike's section 3), which
  tools/fetch_bench.py fetches and checks and gcc never compiles.
- No bale text is tracked: the manifest names bale's files by path and
  sha256 only, and no program holds an identifier of bale's serial code.
- Each program reads its inputs and writes its outputs through lassi_io.h,
  takes exactly the item's input files and then its output files, and
  writes each output under its declared array name. Each names bale's file
  it follows and IDA's copyright of that file. A cpp program holds no
  OpenMP construct and no thread; an omp program holds at least one OpenMP
  construct, by the OpenMP-construct tag of the toolchain gcc-openmp
  (lassi.toolchains.openmp_tag), which tags a cpp program and passes an
  omp one.
- tools/fetch_bench.py lists exactly the three upstream files, with their
  sha256, and nothing else.

Items are read through Suite.item with purpose eval only; nothing here
trains, tunes a prompt, or harvests (Agent Rule 5). No value here is a
measurement: the seeds, sizes, and bounds are the spike's choices, and the
sha256 values were read from bale's blobs at the pin (plans/spikes/bale.md).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from lassi.bench import load_suite
from lassi.toolchains.openmp_tag import NO_OPENMP_CODE, read_openmp_tags

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import fetch_bench  # noqa: E402

MANIFEST = REPO / "assets" / "bench" / "bale-v0.yaml"
SUITE = load_suite(MANIFEST)
REPO_URL = "https://github.com/jdevinney/bale.git"
COMMIT = "1b8f673b56645b2bd74a8af6213462bbc9d559fe"
ITEMS = ("histo", "ig", "triangle")
LANGUAGES = ("cpp", "omp")
# bale's serial C files at the pin, by path and sha256 (plans/spikes/bale.md, section 3).
UPSTREAM = {
    "histo": "8391f5a020ff67d6d3bcb55558bec37b3a97ee4f6176d8cbc7d605dd53c3ccf2",
    "ig": "2cb08618a08eddb9979ca27c5cb5b259325ca34d766eae34f0a9ef961c2144d0",
    "triangle": "07b3d5ba0858fb0667a75b6044e5e7c14741b0c45600582881d7815886a4237f",
}
SEEDS = {"histo": 4101, "ig": 4102, "triangle": 4103}
TABLE = 1 << 20
INPUTS = {
    "histo": [
        {"name": "table_size", "dtype": "i64", "shape": [], "dist": "integers", "lo": TABLE, "hi": TABLE + 1},
        {"name": "index", "dtype": "i64", "shape": [100000], "dist": "integers", "lo": 0, "hi": TABLE},
    ],
    "ig": [
        {"name": "table", "dtype": "i64", "shape": [TABLE], "dist": "integers", "lo": -(1 << 40), "hi": 1 << 40},
        {"name": "index", "dtype": "i64", "shape": [100000], "dist": "integers", "lo": 0, "hi": TABLE},
    ],
    "triangle": [
        {"name": "num_vertices", "dtype": "i64", "shape": [], "dist": "integers", "lo": 50000, "hi": 50001},
        {"name": "edge_u", "dtype": "i64", "shape": [500000], "dist": "integers", "lo": 0, "hi": 50000},
        {"name": "edge_v", "dtype": "i64", "shape": [500000], "dist": "integers", "lo": 0, "hi": 50000},
    ],
}
OUTPUTS = {"histo": ("counts",), "ig": ("tgt",), "triangle": ("triangles",)}
# Identifiers of bale's serial code that a program written here never needs (none is a C or C++ library name).
BALE_IDENTIFIERS = (
    "histo_generic", "histo_buffered", "histo_sorted", "ig_generic", "ig_buffered", "triangles_matrix",
    "sparsemat_t", "wall_seconds", "rand_int64", "rand_seed", "std_args_t", "bale_app", "numtriangles",
    "num_ups", "tbl_size", "num_req",
)
NOTICE = "Copyright (c) 2020, Institute for Defense Analyses"
THREADS = (r"<thread>", r"\bstd::thread\b", r"\bpthread_", r"<omp\.h>", r"\bomp_")


def program(item: str, language: str) -> str:
    """Return the text of one tracked program."""
    spec = SUITE.items[item].languages[language]
    return (REPO / spec.dir / spec.files[0]).read_bytes().decode("ascii")


def test_the_manifest_pins_bale_at_the_spikes_commit_as_a_fetched_suite() -> None:
    assert (SUITE.name, SUITE.repo, SUITE.commit, SUITE.pin) == ("bale-v0", REPO_URL, COMMIT, None)
    assert not SUITE.tracked
    assert tuple(sorted(SUITE.items)) == ITEMS


@pytest.mark.parametrize("item", ITEMS)
def test_each_item_is_unassigned_with_the_spikes_held_out_inputs(item: str) -> None:
    found = SUITE.item(item, purpose="eval")
    assert found.split == "unassigned"
    assert found.seed == SEEDS[item]
    assert [dict(entry) for entry in found.inputs] == INPUTS[item]
    assert found.outputs == OUTPUTS[item]
    assert (found.tolerance.metric, found.tolerance.threshold) == ("max_abs", 0)
    assert not found.support and not found.run_args and not found.passfail
    assert found.unpack_to_dest is None


def test_the_seeds_are_distinct_across_every_manifest() -> None:
    seeds = []
    for path in sorted((REPO / "assets" / "bench").glob("*.yaml")):
        suite = load_suite(path)
        if suite.name.startswith("tt-pairs-v0-") or suite.name == "bale-v0-ttnn":
            continue  # a sibling of tt-pairs-v0 or bale-v0-tt that reuses its items and seeds (tasks P17.36, P17.57)
        seeds += [item.seed for item in suite.items.values() if item.seed is not None]
    assert len(seeds) == len(set(seeds))


@pytest.mark.parametrize("item", ITEMS)
def test_each_item_is_refused_to_every_purpose_rule_5_forbids(item: str) -> None:
    from lassi.bench import EvalSplitError

    for purpose in ("train", "prompt-tuning", "harvest"):
        with pytest.raises(EvalSplitError):
            SUITE.item(item, purpose=purpose)


@pytest.mark.parametrize("item", ITEMS)
@pytest.mark.parametrize("language", LANGUAGES)
def test_each_language_is_tracked_with_lassi_io_and_bales_file_by_sha256(item: str, language: str) -> None:
    spec = SUITE.items[item].languages[language]
    assert (spec.dir, spec.files, spec.tracked) == (f"assets/bench/bale-v0/{item}/{language}", (f"{item}.cpp",), True)
    assert spec.reference is None and not spec.sha256
    assert set(spec.support) == {"lassi_io.h", f"upstream/{item}.c"}
    header = spec.support["lassi_io.h"]
    assert (header.path, header.tracked, header.sha256) == ("assets/harness/c/lassi_io.h", True, None)
    upstream = spec.support[f"upstream/{item}.c"]
    assert (upstream.path, upstream.tracked) == (f"src/other_serial/C/{item}.c", False)
    assert upstream.sha256 == UPSTREAM[item]


def test_fetch_bench_lists_exactly_the_three_upstream_files_with_their_sha256() -> None:
    paths = [f"src/other_serial/C/{item}.c" for item in ITEMS]
    assert fetch_bench.listed_files(SUITE) == paths
    assert fetch_bench.sparse_dirs(SUITE) == paths
    assert fetch_bench._digests(SUITE) == {f"src/other_serial/C/{item}.c": UPSTREAM[item] for item in ITEMS}


def test_no_bale_text_is_tracked_under_the_suites_directory() -> None:
    # The directory also holds the programs of bale-v0-tt's other five apps (task P17.43), checked in its own tests.
    root = REPO / "assets" / "bench" / "bale-v0"
    items = sorted(load_suite(REPO / "assets" / "bench" / "bale-v0-tt.yaml").items)
    assert set(ITEMS) <= set(items)
    found = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    assert found == [f"{item}/{language}/{item}.cpp" for item in items for language in LANGUAGES]


@pytest.mark.parametrize("item", ITEMS)
@pytest.mark.parametrize("language", LANGUAGES)
def test_each_program_is_ascii_on_the_harness_and_writes_its_outputs(item: str, language: str) -> None:
    spec = SUITE.items[item].languages[language]
    data = (REPO / spec.dir / spec.files[0]).read_bytes()
    assert data.isascii() and b"\r" not in data and data.endswith(b"\n")
    text = data.decode("ascii")
    assert '#include "lassi_io.h"' in text
    arguments = len(INPUTS[item]) + len(OUTPUTS[item])
    assert f"argc != {arguments + 1}" in text, "the input files, then the output files"
    for name in OUTPUTS[item]:
        assert f'"{name}", LASSI_IO_I64' in text, f"writes {name} as an i64 array under its declared name"
    assert text.count("lassi_io_read(") == 1 and text.count("lassi_io_write(") == 1


@pytest.mark.parametrize("item", ITEMS)
@pytest.mark.parametrize("language", LANGUAGES)
def test_each_program_names_the_bale_file_it_follows_and_copies_no_bale_code(item: str, language: str) -> None:
    text = program(item, language)
    assert f"src/other_serial/C/{item}.c" in text and "github.com/jdevinney/bale" in text and NOTICE in text
    assert "no bale text is copied" in text
    found = [name for name in BALE_IDENTIFIERS if re.search(rf"\b{name}\b", text)]
    assert found == [], f"bale identifiers in {item}/{language}: {found}"


@pytest.mark.parametrize("item", ITEMS)
def test_a_cpp_program_holds_no_openmp_construct_and_no_thread(item: str) -> None:
    text = program(item, "cpp")
    tags = read_openmp_tags({f"{item}.cpp": text})
    assert [tag.code for tag in tags] == [NO_OPENMP_CODE], "the tag fires on the serial program"
    assert not any(re.search(pattern, text) for pattern in THREADS)
    assert "pragma" not in text


@pytest.mark.parametrize("item", ITEMS)
def test_an_omp_program_holds_an_openmp_construct(item: str) -> None:
    text = program(item, "omp")
    assert read_openmp_tags({f"{item}.cpp": text}) == ()
    assert re.search(r"^#pragma omp parallel for\b", text, re.MULTILINE)
    assert not any(re.search(pattern, text) for pattern in THREADS), "OpenMP pragmas only, no runtime calls"


@pytest.mark.parametrize("item", ITEMS)
def test_the_two_programs_of_an_item_take_the_same_arguments(item: str) -> None:
    usage = {language: re.search(r'"usage: [^"]*"', program(item, language)) for language in LANGUAGES}
    assert usage["cpp"] is not None and usage["omp"] is not None
    assert usage["cpp"].group() == usage["omp"].group()


def test_histo_adds_atomically_and_triangle_reduces() -> None:
    assert "#pragma omp atomic" in program("histo", "omp")
    assert "reduction(+ : total)" in program("triangle", "omp")
