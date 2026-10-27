"""Tests for the OpenMP-construct tag and the Toolchain gcc-openmp (task P17.40).

Bible: Harness Contract (the OpenMP-construct tag: tagged, never failed),
Component Interfaces (Toolchain row; capability rule), Execution Backends
(native row), Toolchain Pins (gcc), Agent Rules 3 and 4. Plan:
plans/p17-portable.md, P17.40. Evidence: plans/spikes/bale.md, section 7.2
(a cpp -> omp candidate that returns the serial program unchanged would pass
binary_io, since both builds use -fopenmp).

The contract these tests fix:

- lassi.core.capabilities gains STATIC_TAGS ("static_tags") and
  static_tags(toolchain, files, harness): () for a toolchain that does not
  declare the capability, whatever methods it has, and otherwise the tuple
  its static_tags(files, harness) returns. A ValueError naming the
  toolchain and the capability refuses a declared capability without a
  callable method, and tags outside the guard contract: not a tuple, or an
  item that is not a Diagnostic, is not parse-stage, is not a warning or a
  note, or whose code does not start with "guard-".
- lassi.toolchains.openmp_tag read_openmp_tags(files) reads the model's C
  and C++ files (.c, .cc, .cpp, .cxx, .h, .hh, .hpp, .hxx, .inc, .inl,
  .ipp) as lassi.toolchains._cxx_scan lexes them: an OpenMP construct is a
  `#pragma omp` directive line outside an `#if 0` or `#if false` region
  (spaces allowed around `#` and between the words) or a `_Pragma` whose
  string literal starts with "omp"; a comment, a string literal elsewhere,
  a dropped region, `#pragma ompx`, and an omp_ runtime call are none. With
  no construct in any file it gives one parse-stage warning, code
  guard-openmp-none, naming the first file read; with one it gives ().
  A file the lexer cannot read gives one note, guard-openmp-not-checked,
  naming the reason, and no warning. Files of other suffixes and the
  harness are never read.
- The Toolchain "gcc-openmp" is GccNative's build with the same pin, flags,
  and command line, plus the capability static_tags; it is not upstream's,
  so a faithful recipe may not bind it.

No compiler runs here: the fake runner returns SYNTHETIC results. The
program texts are SYNTHETIC. No value here is a measurement.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from lassi.core import capabilities
from lassi.core.recipe import NOT_UPSTREAM_TOOLCHAINS
from lassi.core.record import Diagnostic
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.toolchains import GccNative, GccOpenMP
from lassi.toolchains._base import CommandResult
from lassi.toolchains.openmp_tag import NO_OPENMP_CODE, NOT_CHECKED_CODE, read_openmp_tags

SERIAL = (
    "// SYNTHETIC serial program\nint main() {\n    int s = 0;\n    for (int i = 0; i < 4; ++i) s += i;\n"
    "    return s;\n}\n"
)
PARALLEL = (
    "// SYNTHETIC parallel program\nint main() {\n    int s = 0;\n#pragma omp parallel for reduction(+ : s)\n"
    "    for (int i = 0; i < 4; ++i) s += i;\n    return s;\n}\n"
)


def codes(tags: tuple[Diagnostic, ...]) -> list[str | None]:
    """Return the codes of `tags`, in order."""
    return [tag.code for tag in tags]


def test_the_capability_name_is_static_tags() -> None:
    assert capabilities.STATIC_TAGS == "static_tags"


def test_a_serial_program_is_tagged_with_one_parse_stage_warning() -> None:
    (tag,) = read_openmp_tags({"main.cpp": SERIAL})
    assert (tag.stage, tag.severity, tag.code, tag.file) == ("parse", "warning", NO_OPENMP_CODE, "main.cpp")
    assert NO_OPENMP_CODE == "guard-openmp-none"
    assert "no OpenMP construct" in tag.message and "tagged, not failed" in tag.message
    assert tag.message.isascii()


def test_a_parallel_program_is_not_tagged() -> None:
    assert read_openmp_tags({"main.cpp": PARALLEL}) == ()


@pytest.mark.parametrize(
    "line",
    [
        "#pragma omp parallel for",
        "  #  pragma   omp   parallel",
        "#pragma omp atomic",
        "#pragma omp simd",
        "_Pragma(\"omp parallel for\")",
        "_Pragma ( \"omp parallel\" )",
    ],
)
def test_each_construct_form_counts(line: str) -> None:
    text = f"// SYNTHETIC\nint main() {{\n{line}\n    for (int i = 0; i < 2; ++i) {{}}\n    return 0;\n}}\n"
    assert read_openmp_tags({"main.cpp": text}) == (), line


@pytest.mark.parametrize(
    "text",
    [
        "// #pragma omp parallel for\nint main() { return 0; }\n",
        "/* #pragma omp parallel for */\nint main() { return 0; }\n",
        "#if 0\n#pragma omp parallel for\n#endif\nint main() { return 0; }\n",
        "const char* s = \"#pragma omp parallel for\";\nint main() { return 0; }\n",
        "#pragma ompx parallel\nint main() { return 0; }\n",
        "#pragma once\nint omp_get_max_threads();\nint main() { return omp_get_max_threads(); }\n",
        "int main() { const char* p = \"omp parallel\"; return p != 0; }\n",
    ],
)
def test_text_that_is_no_construct_is_tagged(text: str) -> None:
    assert codes(read_openmp_tags({"main.cpp": text})) == [NO_OPENMP_CODE]


def test_a_construct_in_any_model_header_counts_and_other_suffixes_are_not_read() -> None:
    assert read_openmp_tags({"main.cpp": SERIAL, "kernel.hpp": PARALLEL}) == ()
    assert codes(read_openmp_tags({"main.cpp": SERIAL, "notes.txt": PARALLEL})) == [NO_OPENMP_CODE]


def test_the_warning_names_the_first_file_in_sorted_order() -> None:
    (tag,) = read_openmp_tags({"z.cpp": SERIAL, "a.cpp": SERIAL})
    assert tag.file == "a.cpp"


def test_no_c_or_cpp_file_names_no_file() -> None:
    (tag,) = read_openmp_tags({})
    assert (tag.code, tag.file) == (NO_OPENMP_CODE, None)


def test_an_unreadable_file_gives_a_not_checked_note_and_no_warning() -> None:
    (note,) = read_openmp_tags({"main.cpp": "int main() { return 0; }\n/* unterminated\n"})
    assert (note.stage, note.severity, note.code) == ("parse", "note", NOT_CHECKED_CODE)
    assert NOT_CHECKED_CODE == "guard-openmp-not-checked"
    assert "main.cpp" in note.message


def test_a_construct_in_another_file_wins_over_an_unreadable_one() -> None:
    assert read_openmp_tags({"a.cpp": "/* unterminated\n", "b.cpp": PARALLEL}) == ()


class Declaring:
    """A SYNTHETIC toolchain that declares static_tags and returns a scripted answer."""

    name = "fake-tagger"
    capabilities = frozenset({"diagnostics", "static_tags"})

    def __init__(self, answer: Any) -> None:
        """Keep the answer."""
        self.answer = answer
        self.calls: list[tuple[dict[str, str], dict[str, str]]] = []

    def static_tags(self, files: Mapping[str, str], harness: Mapping[str, str]) -> Any:
        """Record the call and return the scripted answer."""
        self.calls.append((dict(files), dict(harness)))
        return self.answer


TAG = Diagnostic(stage="parse", severity="warning", code="guard-openmp-none", file="main.cpp", message="SYNTHETIC")


def test_a_toolchain_without_the_capability_gives_no_tags() -> None:
    class Plain(Declaring):
        capabilities = frozenset({"diagnostics"})

    plain = Plain((TAG,))
    assert capabilities.static_tags(plain, {"main.cpp": SERIAL}, {}) == ()
    assert plain.calls == [], "the method of an undeclaring toolchain is never called"


def test_a_declaring_toolchain_gives_its_tags_with_files_and_harness() -> None:
    tagger = Declaring((TAG,))
    assert capabilities.static_tags(tagger, {"main.cpp": SERIAL}, {"lassi_io.h": "x"}) == (TAG,)
    assert tagger.calls == [({"main.cpp": SERIAL}, {"lassi_io.h": "x"})]


@pytest.mark.parametrize(
    "answer",
    [
        [TAG],
        (object(),),
        (Diagnostic(stage="compile", severity="warning", code="guard-x", message="SYNTHETIC"),),
        (Diagnostic(stage="parse", severity="error", code="guard-x", message="SYNTHETIC"),),
        (Diagnostic(stage="parse", severity="warning", code="openmp-none", message="SYNTHETIC"),),
        (Diagnostic(stage="parse", severity="warning", code=None, message="SYNTHETIC"),),
    ],
)
def test_tags_outside_the_guard_contract_are_refused_naming_the_toolchain(answer: Any) -> None:
    with pytest.raises(ValueError, match="fake-tagger.*static_tags"):
        capabilities.static_tags(Declaring(answer), {"main.cpp": SERIAL}, {})


def test_a_declared_capability_without_a_method_is_refused() -> None:
    class NoMethod:
        name = "fake-no-method"
        capabilities = frozenset({"static_tags"})

    with pytest.raises(ValueError, match="fake-no-method.*static_tags"):
        capabilities.static_tags(NoMethod(), {}, {})


def test_gcc_openmp_is_registered_as_gcc_natives_build_with_the_tag() -> None:
    entry = DEFAULT_REGISTRY.get("Toolchain", "gcc-openmp")
    assert entry.factory is GccOpenMP and entry.capabilities == GccOpenMP.capabilities
    assert issubclass(GccOpenMP, GccNative)
    assert GccOpenMP.name == "gcc-openmp" and GccOpenMP.PIN == GccNative.PIN == "gcc"
    assert GccOpenMP.capabilities == GccNative.capabilities | {"static_tags"}
    assert "static_tags" not in GccNative.capabilities, "gcc-native keeps its capabilities"
    assert GccOpenMP().command(["a.cpp"]) == GccNative().command(["a.cpp"])


def test_gcc_openmp_tags_through_the_seam() -> None:
    toolchain = GccOpenMP()
    assert codes(capabilities.static_tags(toolchain, {"main.cpp": SERIAL}, {"lassi_io.h": PARALLEL})) == [
        NO_OPENMP_CODE
    ], "the harness is never read"
    assert capabilities.static_tags(toolchain, {"main.cpp": PARALLEL}, {}) == ()


def test_gcc_openmp_builds_as_gcc_native(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    def runner(argv: list[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Record the command and write a SYNTHETIC artifact."""
        seen.append(list(argv))
        (Path(cwd) / "main").write_bytes(b"PLACEHOLDER artifact\n")
        return CommandResult(returncode=0, stdout="", stderr="")

    result = GccOpenMP(executable="/usr/bin/g++-12", runner=runner).build(
        {"main.cpp": SERIAL}, tmp_path, {"upstream/histo.c": "/* SYNTHETIC */\n"}
    )
    assert result.artifact is not None and result.diagnostics == []
    assert seen == [["/usr/bin/g++-12", "-O3", "-fopenmp", "-o", "main", "main.cpp"]], "a .c harness file is no source"


def test_a_faithful_recipe_may_not_bind_gcc_openmp() -> None:
    assert "gcc-openmp" in NOT_UPSTREAM_TOOLCHAINS and "gcc-native" in NOT_UPSTREAM_TOOLCHAINS
