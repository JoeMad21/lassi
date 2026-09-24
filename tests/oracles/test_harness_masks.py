"""Tests for the timing masks under assets/harness/ (task P1.7).

Bible: Oracles (stdout_mask row), Repository Layout (`assets/harness/`:
masking rules), Harness Contract, Benchmark Suites (lassi-hecbench-10).

The mask file of a suite is `assets/harness/masks/<suite>.yaml`, plain ASCII
with LF newlines:

    suite: <suite name>
    items:
      <item>: [<regex>, ...]

`lassi.oracles.load_masks(suite, root=None)` reads `<root>/<suite>.yaml`
(root defaults to assets/harness/masks) and returns item -> regexes. It
fails loudly: a missing file raises an error naming the suite, and a regex
that does not compile raises ValueError naming its item.

Every item has masks derived from its sources' print statements. The
derivation test reads the 20 model-facing sources at the upstream pin
(byte-identical to the pinned HeCBench files, whose sha256 the suite
manifest records), finds each print to stdout (printf, fprintf(stdout, ...),
std::cout), and renders it into sample lines with several number styles. A
print whose literal text holds "time" is a timing print. Each item's masks
must match every rendered timing line of both languages and no other
rendered line. The renderer here is a small test helper for these 20 files,
not a C parser. Upstream sources are only read, never run.

The expected timing print counts per item and language below were read from
the pinned sources (grep of their print statements), not from any run.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
MASK_FILE = REPO / "assets" / "harness" / "masks" / f"{SUITE}.yaml"
PIN_MANIFEST = REPO / "assets" / "upstream" / "lassi.yaml"
MAINS = "translated_code/input_codes/HeCBench"
EXTENSIONS = {"omp": "cpp", "cuda": "cu"}

# Timing prints per item, the same in both languages, counted in the pinned sources.
TIMING_PRINTS = {
    "atomicCost": 2,
    "bsearch": 4,
    "colorwheel": 1,
    "dense-embedding": 2,
    "entropy": 2,
    "jacobi": 2,
    "layout": 2,
    "matrix-rotate": 1,
    "pathfinder": 2,
    "randomAccess": 1,
}

# (integer, real) samples for printed numbers; a mask must not depend on how a number is formatted.
NUMBER_STYLES = (("7", "1.500000"), ("0", "0.000012"), ("123456789", "12345.678900"), ("42", "1.5e-05"), ("1", "3"))

# Lines no mask may match, whatever the item.
NEVER_MASKED = ("PASS", "FAIL", "", "Found 0 errors in 1024 locations (PASS).")

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\", '"': '"', "'": "'"}
_ESCAPE = re.compile(r"\\(.)")
_LITERAL = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
_PRINT = re.compile(r"\b(?:printf|fprintf)\s*\(|std::cout\s*<<")
_SPEC = re.compile(r"%(?:(%)|[-+ #0]*(?:\d+|\*)?(?:\.(?:\d+|\*))?(?:hh|h|ll|l|L|z|j|t)?([diouxXeEfFgGaAcsp]))")
_STREAM_MANIPULATORS = ("std::setprecision", "std::fixed", "std::scientific")


# ---------------------------------------------------------------------------
# The mask file


def test_mask_file_is_under_assets_harness_in_plain_ascii() -> None:
    assert MASK_FILE.is_file(), f"missing {MASK_FILE.relative_to(REPO).as_posix()}"
    data = MASK_FILE.read_bytes()
    assert data.isascii(), "the mask file must be plain ASCII"
    assert b"\r" not in data, "the mask file must use LF newlines"
    assert yaml.safe_load(data)["suite"] == SUITE


def test_every_item_of_the_suite_has_masks_and_no_other_item_does() -> None:
    from lassi.oracles import load_masks

    items = set(yaml.safe_load(SUITE_MANIFEST.read_text(encoding="utf-8"))["items"])
    masks = load_masks(SUITE)
    assert set(masks) == items
    for item, patterns in masks.items():
        assert len(patterns) >= 1, f"{item} has no mask"
        for pattern in patterns:
            assert isinstance(pattern, str) and pattern, f"{item}: {pattern!r} is not a non-empty string"
            re.compile(pattern)


def test_load_masks_fails_loudly_on_an_unknown_suite(tmp_path: Path) -> None:
    from lassi.oracles import load_masks

    with pytest.raises((ValueError, OSError), match="no-such-suite"):
        load_masks("no-such-suite", root=tmp_path)


def test_load_masks_refuses_a_regex_that_does_not_compile_naming_its_item(tmp_path: Path) -> None:
    from lassi.oracles import load_masks

    (tmp_path / "broken.yaml").write_bytes(b"suite: broken\nitems:\n  someapp: ['(unclosed']\n")
    with pytest.raises(ValueError, match="someapp"):
        load_masks("broken", root=tmp_path)


# ---------------------------------------------------------------------------
# Derivation from the pinned sources' print statements


def _git(root: Path, *args: str) -> bytes:
    """Return the stdout of git with `args` in `root` (it only reads objects here)."""
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True).stdout


@pytest.fixture(scope="module")
def pinned_sources() -> dict[tuple[str, str], str]:
    """Return the 20 model-facing sources at the upstream pin, keyed by (item, language), LF newlines.

    Skips when the upstream checkout is not fetched; fails when a source's
    sha256 differs from the suite manifest.
    """
    pin = yaml.safe_load(PIN_MANIFEST.read_text(encoding="utf-8"))
    root = REPO / pin["path"]
    if not (root / ".git").exists():
        pytest.skip(f"{pin['path']} is not fetched; run `uv run tools/fetch_upstream.py`")
    manifest = yaml.safe_load(SUITE_MANIFEST.read_text(encoding="utf-8"))["items"]
    sources: dict[tuple[str, str], str] = {}
    for item, spec in manifest.items():
        for language, extension in EXTENSIONS.items():
            data = _git(root, "show", f"{pin['commit']}:{MAINS}/{item}/{item}-{language}_main.{extension}")
            (name,) = spec["languages"][language]["files"]
            assert hashlib.sha256(data).hexdigest() == spec["languages"][language]["sha256"][name], (item, language)
            sources[(item, language)] = data.decode("utf-8").replace("\r\n", "\n")
    return sources


def _literal_end(text: str, start: int) -> int:
    """Return the index just past the string or character literal that opens at `start`."""
    quote, index = text[start], start + 1
    while index < len(text) and text[index] != quote:
        index += 2 if text[index] == "\\" else 1
    return index + 1


def _strip_comments(source: str) -> str:
    """Return `source` with // and /* */ comments removed; string and character literals are kept."""
    out: list[str] = []
    index = 0
    while index < len(source):
        if source[index] in "\"'":
            end = _literal_end(source, index)
            out.append(source[index:end])
            index = end
        elif source.startswith("//", index):
            end = source.find("\n", index)
            index = len(source) if end < 0 else end
        elif source.startswith("/*", index):
            end = source.find("*/", index + 2)
            out.append(" ")
            index = len(source) if end < 0 else end + 2
        else:
            out.append(source[index])
            index += 1
    return "".join(out)


def _split_top(text: str, separator: str) -> list[str]:
    """Split `text` on `separator` outside literals and brackets."""
    parts: list[str] = []
    current: list[str] = []
    depth = index = 0
    while index < len(text):
        char = text[index]
        if char in "\"'":
            end = _literal_end(text, index)
            current.append(text[index:end])
            index = end
            continue
        depth += (char in "([{") - (char in ")]}")
        if depth == 0 and text.startswith(separator, index):
            parts.append("".join(current))
            current = []
            index += len(separator)
            continue
        current.append(char)
        index += 1
    parts.append("".join(current))
    return parts


def _literal_text(expression: str) -> str | None:
    """Return the decoded text of an expression made only of adjacent string literals, else None."""
    stripped = expression.strip()
    if not stripped or _LITERAL.sub("", stripped).strip():
        return None
    bodies = _LITERAL.findall(stripped)
    return "".join(_ESCAPE.sub(lambda m: _ESCAPES.get(m.group(1), m.group(1)), body) for body in bodies)


def _print_statements(source: str) -> list[tuple[str, str]]:
    """Return (opening, rest up to the semicolon) for every printf, fprintf, and std::cout statement."""
    source = _strip_comments(source)
    found = []
    for match in _PRINT.finditer(source):
        index = match.end()
        while index < len(source) and source[index] != ";":
            index = _literal_end(source, index) if source[index] in "\"'" else index + 1
        found.append((match.group(0), source[match.end() : index]))
    return found


def _render_cout(rest: str, real: str) -> tuple[str, str]:
    """Return (literal text, rendered text) of a std::cout statement."""
    literals: list[str] = []
    pieces: list[str] = []
    for operand in _split_top(rest, "<<"):
        text, stripped = _literal_text(operand), operand.strip()
        if text is not None:
            literals.append(text)
            pieces.append(text)
        elif stripped == "std::endl":
            pieces.append("\n")
        elif not stripped.startswith(_STREAM_MANIPULATORS):
            pieces.append(real)
    return "".join(literals), "".join(pieces)


def _render_printf(opening: str, rest: str, integer: str, real: str) -> tuple[str, str] | None:
    """Return (format, rendered text) of a printf or fprintf(stdout, ...) statement; None for another stream."""
    arguments = _split_top(rest[: rest.rstrip().rfind(")")], ",")
    if opening.startswith("fprintf") and arguments.pop(0).strip() != "stdout":
        return None
    fmt = _literal_text(arguments[0])
    assert fmt is not None, f"a print whose format is not a literal: {rest[:60]!r}"

    def sample(match: re.Match[str]) -> str:
        if match.group(1):
            return "%"
        conversion = match.group(2)
        if conversion in "diouxX":
            return integer
        return real if conversion in "eEfFgGaA" else "text"

    return fmt, _SPEC.sub(sample, fmt)


def rendered_lines(source: str, integer: str, real: str) -> tuple[list[str], list[str], int]:
    """Return (timing lines, other lines, timing print count) that the source's stdout prints render to."""
    timing: list[str] = []
    other: list[str] = []
    count = 0
    for opening, rest in _print_statements(source):
        if opening.startswith("std::cout"):
            rendered: tuple[str, str] | None = _render_cout(rest, real)
        else:
            rendered = _render_printf(opening, rest, integer, real)
        if rendered is None:
            continue
        literal, text = rendered
        lines = [line for line in text.split("\n") if line]
        if "time" in literal:
            count += 1
            timing += [line for line in lines if "time" in line]
        else:
            other += lines
    return timing, other, count


def _matched(line: str, patterns: list[str]) -> bool:
    """Return True when any pattern matches the line."""
    return any(re.search(pattern, line) for pattern in patterns)


def test_the_source_renderer_finds_every_timing_print(pinned_sources: dict[tuple[str, str], str]) -> None:
    """Guards the helper: it finds the timing prints the pinned sources hold, one line each."""
    for (item, language), source in sorted(pinned_sources.items()):
        timing, _, count = rendered_lines(source, *NUMBER_STYLES[0])
        assert count == TIMING_PRINTS[item], (item, language)
        assert len(timing) == count, (item, language)


def test_each_item_masks_every_timing_line_its_sources_print(pinned_sources: dict[tuple[str, str], str]) -> None:
    from lassi.oracles import load_masks

    masks = load_masks(SUITE)
    missed = []
    for (item, language), source in sorted(pinned_sources.items()):
        for integer, real in NUMBER_STYLES:
            timing, _, _ = rendered_lines(source, integer, real)
            missed += [(item, language, line) for line in timing if not _matched(line, list(masks[item]))]
    assert missed == [], f"timing lines no mask of their item matches: {missed[:5]}"


def test_no_mask_matches_a_line_that_is_not_a_timing_line(pinned_sources: dict[tuple[str, str], str]) -> None:
    from lassi.oracles import load_masks

    masks = load_masks(SUITE)
    wrongly = []
    for (item, language), source in sorted(pinned_sources.items()):
        for integer, real in NUMBER_STYLES:
            _, other, _ = rendered_lines(source, integer, real)
            wrongly += [(item, language, line) for line in [*other, *NEVER_MASKED] if _matched(line, list(masks[item]))]
    assert wrongly == [], f"non-timing lines a mask matches: {wrongly[:5]}"
