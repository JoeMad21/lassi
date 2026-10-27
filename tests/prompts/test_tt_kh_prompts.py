"""Tests for the tt-kh-v0 template set: C++ -> TT where the model writes the host program and its kernels (P17.39).

Bible: Project Recipes (template sets; the context, examples, lookup, and
skeleton sections), Benchmark Suites (tt-pairs-v0-kh; kernels in a prompt),
Harness Contract (a kernel is written uncompiled at its path and compiled
by the kernel JIT at first launch), Agent Rule 5 (tt-pairs-v0 is a
development set, OQ-083 (b); no item's upstream kernel or tracked host
program in a prompt).

What they fix:

- assets/prompts/tt-kh-v0 holds every template tt-host-v1 holds, in
  tt-host-v1's style; each is plain ASCII with LF line ends, uses only its
  stage's fields, and renders.
- generate.txt, correct.txt, and correct_mismatch.txt never use the fields
  kernel_files or host_files, so the set can show no support kernel or host
  program of any suite; for every tt-pairs-v0-kh item generate shows the
  C++ source and names the files to write, the host program and each
  kernel, and shows no upstream kernel.
- generate.txt explains the ttkh language id, the kernel kinds (reader,
  compute, writer), circular buffers, the kernel entry point and include
  paths at the pin, the argument calls, and the NoC and tile calls.
- correct.txt says a kernel's compile errors come from the kernel compile
  at the program's first launch, listed with the kernel's path.
- No template names a tt-pairs-v0 item, kernel, or file (named_tier_a).
"""

from __future__ import annotations

import sys
from pathlib import Path
from string import Template
from types import SimpleNamespace
from typing import Any

import pytest

from lassi.bench import Direction, load_suite
from lassi.core.stages import (
    CORRECT_FIELDS,
    GENERATE_FIELDS,
    MISMATCH_FIELDS,
    MISMATCH_PROMPT,
    PURPOSE,
    GenerateStage,
    _correction_messages,
)
from lassi.prompts import render

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "core"))

from test_template_context import named_tier_a  # noqa: E402

SET = "tt-kh-v0"
SET_DIR = REPO / "assets" / "prompts" / SET
V1_DIR = REPO / "assets" / "prompts" / "tt-host-v1"
SUITE = load_suite(REPO / "assets" / "bench" / "tt-pairs-v0-kh.yaml")
DIRECTION = Direction("cpp", "ttkh")
FIELDS = {"generate": GENERATE_FIELDS, "correct": CORRECT_FIELDS, MISMATCH_PROMPT: MISMATCH_FIELDS}
CONTEXT_FIELDS = {
    "summarize_context": ("target_language", "context_pack"), "knowledge": ("knowledge_summary",),
    "examples": ("examples",), "example": ("name", "files"), "retrieved": ("passages",),
    "passage": ("path", "heading", "start", "end", "text"), "looked_up": ("declarations",),
    "declaration": ("name", "header", "scope", "text"), "undeclared": ("name", "candidates"),
    "core_api": ("declarations",),
}
SUPPORT_FIELDS = ("kernel_files", "host_files")
# What generate.txt must say about the kernels at the pin (tt_metal/hw/inc/api at 5280a9cf; the kernel API pack).
KERNEL_WORDS = (
    "ttkh", "5280a9cf", "void kernel_main()", "\"api/dataflow/dataflow_api.h\"", "\"api/compute/common.h\"",
    "reader", "writer", "compute kernel", "circular buffer", "DataMovementConfig", "ComputeConfig",
    "cb_reserve_back", "cb_push_back", "cb_wait_front", "cb_pop_front", "get_write_ptr", "get_read_ptr",
    "noc_async_read_page", "noc_async_read_barrier", "noc_async_write_page", "noc_async_write_barrier",
    "TensorAccessorArgs<", "TensorAccessor(", "next_compile_time_args_offset()", "get_arg_val<uint32_t>",
    "get_compile_time_arg_val", "tile_regs_acquire()", "tile_regs_commit()", "tile_regs_wait()",
    "tile_regs_release()", "copy_tile", "pack_tile", "CreateKernel",
)


def stand_in(path: str) -> str:
    """Return the SYNTHETIC text written in place of a fetched upstream file."""
    return f"// SYNTHETIC stand-in for the fetched kernel {path}\n"


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a sources root with a SYNTHETIC stand-in at every upstream support path of the suite."""
    root = tmp_path_factory.mktemp("kh-sources")
    for found in SUITE.items.values():
        for spec in found.languages.values():
            for entry in spec.support.values():
                if not entry.tracked:
                    (root / entry.path).parent.mkdir(parents=True, exist_ok=True)
                    (root / entry.path).write_bytes(stand_in(entry.path).encode("ascii"))
    return root


def text_of(name: str) -> str:
    """Return template `name` of the set as ASCII text; fail when it is missing or not plain ASCII with LF ends."""
    path = SET_DIR / f"{name}.txt"
    assert path.is_file(), f"{SET} has no {name}.txt"
    data = path.read_bytes()
    assert data.isascii() and b"\r" not in data, f"{name}.txt is not plain ASCII with LF line ends"
    return data.decode("ascii")


def placeholders(text: str) -> set[str]:
    """Return the `$name` placeholders of a template; fail on a `$` that is no placeholder."""
    names = set()
    for match in Template.pattern.finditer(text):
        assert match.group("invalid") is None
        name = match.group("named") or match.group("braced")
        if name:
            names.add(name)
    return names


def context(item: str, sources: Path) -> Any:
    """Return a stand-in RunContext for one item of cpp -> ttkh under tt-kh-v0."""
    return SimpleNamespace(fragments={}, suite=SUITE, item=item, direction=DIRECTION, sources_root=sources,
                           prompts=SET)


def test_the_set_holds_every_template_tt_host_v1_holds() -> None:
    assert sorted(path.name for path in SET_DIR.iterdir()) == sorted(path.name for path in V1_DIR.iterdir())


@pytest.mark.parametrize("name", sorted({**FIELDS, **CONTEXT_FIELDS}))
def test_each_template_is_ascii_and_uses_only_its_fields(name: str) -> None:
    text = text_of(name)
    allowed = {**FIELDS, **CONTEXT_FIELDS}[name]
    assert placeholders(text) <= set(allowed), name
    assert render(SET, name, {field: f"<SYNTHETIC {field}>" for field in allowed})


@pytest.mark.parametrize("name", sorted(FIELDS))
def test_no_prompt_shows_a_support_kernel_or_host_program(name: str) -> None:
    assert not placeholders(text_of(name)) & set(SUPPORT_FIELDS), (
        f"{name}.txt uses {SUPPORT_FIELDS}: an item's upstream kernels or tracked host program would be the answer")


@pytest.mark.parametrize("name", sorted({**FIELDS, **CONTEXT_FIELDS}))
def test_each_template_is_generic(name: str) -> None:
    assert named_tier_a(text_of(name)) == [], f"{name}.txt names a tt-pairs-v0 item or file (Agent Rule 5)"


@pytest.mark.parametrize("words", KERNEL_WORDS)
def test_generate_explains_the_kernels_at_the_pin(words: str) -> None:
    assert words in text_of("generate"), words


@pytest.mark.parametrize("name", ["correct", MISMATCH_PROMPT])
def test_the_corrections_define_ttkh_and_ask_for_every_file(name: str) -> None:
    text = text_of(name)
    assert "- ttkh:" in text and "$target_files" in text and "kernel" in text


def test_correct_says_a_kernel_compile_error_comes_at_the_first_launch() -> None:
    text = " ".join(text_of("correct").split())
    for words in ("first launches", "kernel's path"):
        assert words in text, words


@pytest.mark.parametrize("item", sorted(SUITE.items))
def test_generate_shows_the_source_and_the_files_to_write_and_no_kernel(item: str, sources: Path) -> None:
    expected = list(SUITE.items[item].languages["ttkh"].files)
    stage: Any = SimpleNamespace(context=context(item, sources))
    system, prompt = GenerateStage._prompt(stage, None, expected)
    assert system is None
    source = SUITE.source_files(item, DIRECTION, sources, purpose=PURPOSE)
    assert all(text in prompt for text in source.values())
    assert ", ".join(expected) in prompt
    assert "SYNTHETIC stand-in for the fetched kernel" not in prompt, "never an upstream kernel"
    tt = SUITE.items[item].languages["tt"]
    host = (REPO / tt.dir / tt.files[0]).read_text(encoding="utf-8")
    assert host not in prompt, "never the tracked host program"
    assert "$" not in prompt.replace("".join(source.values()), "")


@pytest.mark.parametrize("item", sorted(SUITE.items))
def test_the_corrections_show_the_files_and_the_errors_and_no_kernel(item: str, sources: Path) -> None:
    expected = list(SUITE.items[item].languages["ttkh"].files)
    previous: Any = SimpleNamespace(files={path: f"// SYNTHETIC previous {path}\n" for path in expected})
    for mismatch in (False, True):
        _, prompt = _correction_messages(context(item, sources), previous, expected,
                                         "SYNTHETIC error text", run_error=False, mismatch=mismatch)
        assert "SYNTHETIC error text" in prompt
        assert all(f"// SYNTHETIC previous {path}" in prompt for path in expected)
        assert "SYNTHETIC stand-in for the fetched kernel" not in prompt
