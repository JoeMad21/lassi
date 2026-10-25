"""Tests for the lassi-ee-22 masks under assets/harness/masks/ (task P9.7).

Bible: Oracles (stdout_mask and passfail rules: masks derived from the print
statements of the item's pinned sources in every language), Harness
Contract, Benchmark Suites (lassi-ee-22).

The pattern is lassi-hecbench-10's (tests/oracles/test_harness_masks.py),
whose literal and printf helpers this module loads by path. Every item with
sources has masks and no other item does (miniMDock has none). The render
test reads each language's build files from the pinned sources (its files
and its support files, as its build directory holds them), finds each print
to stdout (printf, fprintf(stdout, ...), and std::cout or cout), and renders
it into sample lines with several number styles. A print is run-varying when
its literal text holds "time" or a rate per second ("/s", as in G/s and
Lookups/s), so XSBench's and segment-reduce's rates are masked with the
timings, and lid-driven-cavity's "Time = " (simulated time) is not. Each
item's masks must match every rendered run-varying line of every language
and no other rendered line. Sources are only read, never built or run; the
render tests skip without the fetched sources.

The expected run-varying print counts per item were read from the pinned
sources (their print statements), not from any run.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from lassi.bench import load_suite, sources_dir

REPO = Path(__file__).resolve().parents[2]
SUITE = "lassi-ee-22"
MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
MASK_FILE = REPO / "assets" / "harness" / "masks" / f"{SUITE}.yaml"
HECBENCH10_RENDERER = REPO / "tests" / "oracles" / "test_harness_masks.py"

# Run-varying prints per item, the same in every language of the item, counted in the pinned sources' build files.
VARYING_PRINTS = {
    "all-pairs-distance": 3,
    "bsearch": 4,
    "chacha20": 1,
    "colorwheel": 1,
    "dense-embedding": 2,
    "entropy": 2,
    "extrema": 3,
    "floydwarshall": 1,
    "jaccard": 1,
    "jacobi": 2,
    "keogh": 1,
    "layout": 2,
    "lid-driven-cavity": 1,
    "marchingCubes": 1,
    "matrix-rotate": 1,
    "murmurhash3": 1,
    "pathfinder": 2,
    "randomAccess": 1,
    "segment-reduce": 1,
    "threadfence": 1,
    "XSBench": 6,
}

# Lines no mask may match, whatever the item.
NEVER_MASKED = ("PASS", "FAIL", "SUCCESS", "", "Verification checksum: 7 (Valid)", "Lookups:               7")

# A print that is run-varying: its literal names a time or a rate per second.
VARYING = re.compile(r"time|/s\b")
# printf, fprintf, and std::cout or cout (some sources say `using namespace std`).
_PRINT = re.compile(r"\b(?:printf|fprintf)\s*\(|(?:std::)?\bcout\s*<<")
_ENDL = ("std::endl", "endl")
_MANIPULATORS = ("std::setprecision", "std::fixed", "std::scientific", "setprecision", "fixed", "scientific")


def renderer() -> ModuleType:
    """Import lassi-hecbench-10's mask test module by path, for its literal and printf helpers."""
    spec = importlib.util.spec_from_file_location("hecbench10_masks", HECBENCH10_RENDERER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPER = renderer()
NUMBER_STYLES = HELPER.NUMBER_STYLES


# ---------------------------------------------------------------------------
# The mask file


def items_with_sources() -> set[str]:
    """Return the manifest's items that have at least one language at the pin."""
    return {name for name, item in load_suite(MANIFEST).items.items() if item.languages}


def test_mask_file_is_under_assets_harness_in_plain_ascii() -> None:
    assert MASK_FILE.is_file(), f"missing {MASK_FILE.relative_to(REPO).as_posix()}"
    data = MASK_FILE.read_bytes()
    assert data.isascii(), "the mask file must be plain ASCII"
    assert b"\r" not in data, "the mask file must use LF newlines"
    assert yaml.safe_load(data)["suite"] == SUITE


def test_every_item_with_sources_has_masks_and_no_other_item_does() -> None:
    from lassi.oracles import load_masks

    masks = load_masks(SUITE)
    assert set(masks) == items_with_sources()
    assert "miniMDock" not in masks
    for item, patterns in masks.items():
        assert patterns, f"{item} has no mask"
        for pattern in patterns:
            re.compile(pattern)


def test_items_shared_with_lassi_hecbench_10_keep_its_masks() -> None:
    from lassi.oracles import load_masks

    ours, theirs = load_masks(SUITE), load_masks("lassi-hecbench-10")
    for item in sorted(set(ours) & set(theirs)):
        assert ours[item] == theirs[item], item


# ---------------------------------------------------------------------------
# Rendering the pinned sources' prints


@pytest.fixture(scope="module")
def build_sources() -> dict[tuple[str, str], list[str]]:
    """Return each (item, language)'s build files from the pinned sources, LF newlines; skip without them."""
    scratch = os.environ.get("LASSI_SCRATCH", "")
    if not scratch:
        pytest.skip("LASSI_SCRATCH is not set; run `uv run tools/fetch_bench.py assets/bench/lassi-ee-22.yaml` there")
    suite = load_suite(MANIFEST)
    root = sources_dir(Path(scratch), suite)
    if not root.is_dir():
        pytest.skip(f"{root} is missing; run `uv run tools/fetch_bench.py assets/bench/lassi-ee-22.yaml`")
    found: dict[tuple[str, str], list[str]] = {}
    for name, item in suite.items.items():
        for language, spec in item.languages.items():
            paths = [root / spec.dir / file for file in spec.files]
            paths += [root / entry.path for entry in spec.support.values()]
            found[(name, language)] = [path.read_bytes().decode("utf-8").replace("\r\n", "\n") for path in paths]
    return found


def _print_statements(source: str) -> list[tuple[str, str]]:
    """Return (opening, rest up to the semicolon) for every printf, fprintf, std::cout, and cout statement."""
    source = HELPER._strip_comments(source)
    found = []
    for match in _PRINT.finditer(source):
        index = match.end()
        while index < len(source) and source[index] != ";":
            index = HELPER._literal_end(source, index) if source[index] in "\"'" else index + 1
        found.append((match.group(0), source[match.end() : index]))
    return found


def _render_cout(rest: str, real: str) -> tuple[str, str]:
    """Return (literal text, rendered text) of a cout statement; endl ends a line, manipulators print nothing."""
    literals: list[str] = []
    pieces: list[str] = []
    for operand in HELPER._split_top(rest, "<<"):
        text, stripped = HELPER._literal_text(operand), operand.strip()
        if text is not None:
            literals.append(text)
            pieces.append(text)
        elif stripped in _ENDL:
            pieces.append("\n")
        elif not stripped.startswith(_MANIPULATORS):
            pieces.append(real)
    return "".join(literals), "".join(pieces)


def rendered_lines(sources: list[str], integer: str, real: str) -> tuple[list[str], list[str], int]:
    """Return (run-varying lines, other lines, run-varying print count) that the sources' stdout prints render to."""
    varying: list[str] = []
    other: list[str] = []
    count = 0
    for source in sources:
        for opening, rest in _print_statements(source):
            if "cout" in opening:
                rendered: tuple[str, str] | None = _render_cout(rest, real)
            else:
                rendered = HELPER._render_printf(opening, rest, integer, real)
            if rendered is None:
                continue
            literal, text = rendered
            lines = [line for line in text.split("\n") if line]
            if VARYING.search(literal):
                count += 1
                varying += [line for line in lines if VARYING.search(line)]
            else:
                other += lines
    return varying, other, count


def _matched(line: str, patterns: list[str]) -> bool:
    """Return True when any pattern matches the line."""
    return any(re.search(pattern, line) for pattern in patterns)


def test_the_renderer_finds_every_run_varying_print(build_sources: dict[tuple[str, str], list[str]]) -> None:
    """Guards the helper: it finds the run-varying prints the pinned sources hold, one line each."""
    for (item, language), sources in sorted(build_sources.items()):
        varying, _, count = rendered_lines(sources, *NUMBER_STYLES[0])
        assert count == VARYING_PRINTS[item], (item, language, count)
        assert len(varying) == count, (item, language, varying)


def test_each_item_masks_every_run_varying_line(build_sources: dict[tuple[str, str], list[str]]) -> None:
    from lassi.oracles import load_masks

    masks = load_masks(SUITE)
    missed = []
    for (item, language), sources in sorted(build_sources.items()):
        for integer, real in NUMBER_STYLES:
            varying, _, _ = rendered_lines(sources, integer, real)
            missed += [(item, language, line) for line in varying if not _matched(line, list(masks[item]))]
    assert missed == [], f"run-varying lines no mask of their item matches: {missed[:5]}"


def test_no_mask_matches_a_line_that_does_not_vary(build_sources: dict[tuple[str, str], list[str]]) -> None:
    from lassi.oracles import load_masks

    masks = load_masks(SUITE)
    wrongly = []
    for (item, language), sources in sorted(build_sources.items()):
        for integer, real in NUMBER_STYLES:
            _, other, _ = rendered_lines(sources, integer, real)
            wrongly += [(item, language, line) for line in [*other, *NEVER_MASKED] if _matched(line, list(masks[item]))]
    assert wrongly == [], f"lines a mask matches that do not vary: {wrongly[:5]}"


def test_segment_reduce_masks_its_rate_inside_the_line_two_earlier_prints_start() -> None:
    # Its stdout line is "num_segments = <n> segment_size = <m> Throughput = <x> (G/s)": three prints, one line.
    from lassi.oracles import load_masks

    line = "num_segments = 4096 segment_size = 4 Throughput = 12.5 (G/s)"
    assert _matched(line, list(load_masks(SUITE)["segment-reduce"]))
