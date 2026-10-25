"""Tests for task P17.26: the context pack's declarations for the identifiers a compile error names (opt-in).

Bible: Component Interfaces (the context blocks under a template set),
Project Recipes (a new optional section, beside the Retrieval bullet),
Harness Contract (a prompt shows only what the trial may see), Agent Rules 4
(a departure from upstream is opt-in and refused under faithful: true) and 5
(no Tier A text in a template). plans/p17-portable.md, task P17.26: the API
reference reaches the model as its own summary of the pack, which can lose
exact signatures, so a compile correction can carry the exact declarations.

The contract these tests fix (the P17.26 design, points 1 to 7):

1. An optional run recipe section `lookup`, as in
   `lookup: {max_chars: 8000, prompts: [correct]}`, with two required keys:
   max_chars, the declarations' total text per prompt, an integer from 500
   to 32000 (a bool or a float refused); and prompts, a non-empty list of
   distinct names among {correct} (generate is refused: a generation prompt
   shows no error to read). A section that is not a mapping, a null, an
   unknown or missing key, and a value of the wrong type or range are
   refused at load naming `lookup.<key>`; the section is refused at load
   under faithful: true, inherited or not, naming `lookup`. Absent, nothing
   is materialized, so a recipe keeps its resolved recipe and recipe hash; a
   section that is set enters both as written.
2. The runner refuses the section, before any directory exists: with a
   fragment prompt set (upstream's prompts carry no such block; the message
   names lookup and the fragment set, whatever else the recipe lacks);
   without a context pack serving the target language (the recipe names
   none, or only a pack of another language); and with a template set that
   lacks looked_up.txt or declaration.txt (the message names the file).
3. The identifiers: of each error-severity Diagnostic a compile correction's
   prompt lists (listed_diagnostics of prompt_diagnostics: within the count
   and byte caps, guard diagnostics left out; warnings and notes never), in
   listed order, the names its message gives as query_from_diagnostic
   (lassi.prompts.retrieval) reads them (the quoted names, then the other
   names holding an underscore, a scope, or an inner capital), each once, in
   order of first mention. A name the pack declares no function or type of
   is not a tt-metal identifier and gives nothing.
4. The declarations: the target language's pack (RunContext.packs, its
   entries joined as the loader joins them) holds header parts, each from a
   line `// ==== #include <H> ====` (the include line of
   assets/context/tt-metalium-5280a9cf.head.txt, which tools/build_tt_context.py
   writes before each header's declarations) to the next such line or the
   pack's end; H is the part's header. A declaration is a namespace-scope
   function declaration or type definition (class, struct, or enum) in a
   part: its text is verbatim from the pack, from its first character
   through its closing `;`, a class's members and inner blank lines
   included, never cut; its name is the function's or the type's; its scope
   is its enclosing namespaces joined by `::`. A member (a constructor, say)
   is part of its class's declaration, not one of its own; a declaration
   that only mentions an identifier (as a parameter type) is not that
   identifier's; text outside the parts (the guide, the preamble) holds no
   declaration. For each identifier in order, every declaration whose name
   it is, in pack order (overloads included); one whose text would take the
   carried declarations' total text (their $text, as retrieval counts a
   passage's text) past max_chars is skipped whole, and a later smaller one
   may still fit.
5. The block: each carried declaration rendered with declaration.txt
   ($name, $header, $scope, $text), the entries one blank line apart (with
   or without a last line feed), as $declarations of looked_up.txt. It goes
   before a compile correction's prompt, after the knowledge block and the
   retrieved block when there are any, one blank line apart, each block
   without its trailing line feeds. With no declaration carried there is no
   block, and the prompt is byte for byte what it is without the section.
   The generation prompt, a run-error correction, and an oracle-mismatch
   correction get none, even when their text names a declared identifier.
6. A request that carried declarations gets exactly one Diagnostic with code
   `looked-up`, severity note, on its Request (never on an attempt, whose
   diagnostics a correction lists), whose message names each carried
   identifier and the header part of each carried declaration, and no name
   that carried nothing (one only a warning gives, or one whose declarations
   were all skipped). It never enters a prompt.
7. tt-host-v0 gains looked_up.txt and declaration.txt, plain ASCII, each
   rendering with its fields and naming no Tier A item, kernel, or file.

Where the design leaves a detail to the implementer, the tests read only
behavior: the template wording comes from the template files
(lassi.prompts.render), and a block is compared as
tests/core/test_template_context.py compares one. Without the section every
prompt matches the goldens under tests/core/golden/template-prompts/ (that
module gives their provenance), with a pack named whose declarations the
goldens' errors name, and the committed recipes keep the recipe hashes it
records.

Every pack, declaration, program, reply, corpus, and array here is SYNTHETIC,
written in a temporary directory; the generated tt-metalium pack is never
read. No value here is a measurement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from test_retrieval_stage import (
    CORPUS,
    FRAGMENT_SET,
    FRAGMENTS,
    MAIN_CHUNKS,
    fragment_recipe,
    index_path,
    kept,
    retrieved_blocks,
    write_corpus,
)
from test_template_context import (
    BARE_ERROR,
    BROKEN,
    CONTEXT_STAGES,
    ERROR,
    PROMPT_KINDS,
    PROMPTS,
    RECIPE_HASHES,
    REPO,
    REQUEST_STAGES,
    SUMMARY,
    WARNING,
    Script,
    crashed,
    doubled,
    full_script,
    golden,
    joined,
    knowledge_block,
    load,
    make_assets,
    named_tier_a,
    program,
    recipe_data,
    refused,
    run,
    template_set,
    template_text,
    user_prompts,
    write_suite,
    write_tree,
    wrong,
)

from lassi.core.recipe import RecipeError, load_recipe
from lassi.core.record import Diagnostic, Request, Trial
from lassi.core.stages import DIAGNOSTIC_COUNT_CAP, prompt_diagnostics
from lassi.prompts import render
from lassi.prompts.retrieval import load_index, query_from_diagnostic

# The section, its two templates with their fields, and the code of its note.
LOOKUP = "lookup"
SECTION = {"max_chars": 8000, "prompts": ["correct"]}
LOOKED_UP = "looked_up"
DECLARATION = "declaration"
LOOKUP_TEMPLATES = {LOOKED_UP: ("declarations",), DECLARATION: ("name", "header", "scope", "text")}
LOOKED_UP_CODE = "looked-up"
# SYNTHETIC stand-ins for the two templates, for sets built here to lack one of them.
SYN_LOOKUP_TEMPLATES = {
    LOOKED_UP: "SYNTHETIC: declarations from the reference follow.\n\n$declarations\n",
    DECLARATION: "// $header ($scope): $name\n$text\n",
}
# The retrieval section the ordering test turns on beside the lookup: passages for corrections only.
RETRIEVAL_SECTION = {"index": CORPUS, "k": 2, "max_chars": 8000, "prompts": ["correct"]}


@dataclass(frozen=True)
class Declaration:
    """One declaration of the SYNTHETIC pack: its name, its header part, its namespace scope, and its exact text."""

    name: str
    header: str
    scope: str
    text: str


# The SYNTHETIC pack's header parts and scopes.
HOST_PART = "tt-metalium/syn_host.hpp"
TYPES_PART = "tt-metalium/syn_types.hpp"
MESH_PART = "tt-metalium/syn_mesh.hpp"
SIZED_PART = "tt-metalium/syn_sized.hpp"
SCOPE = "tt::tt_metal"
MESH_SCOPE = "tt::tt_metal::distributed"

# SYNTHETIC declarations in the shape of the generated reference: no comments, no bodies.
KERNEL_AT_CORE = Declaration(
    "CreateKernel", HOST_PART, SCOPE,
    "KernelHandle CreateKernel(\n"
    "    Program& program,\n"
    "    const std::string& file_name,\n"
    "    const CoreCoord& core,\n"
    "    const KernelSettings& settings);",
)
KERNEL_ON_RANGE = Declaration(
    "CreateKernel", HOST_PART, SCOPE,
    "KernelHandle CreateKernel(Program& program, const std::string& file_name, const CoreRange& cores);",
)
CIRCULAR_BUFFER = Declaration(
    "CreateCircularBuffer", HOST_PART, SCOPE,
    "CBHandle CreateCircularBuffer(Program& program, const CoreCoord& core, const CircularBufferConfig& config);",
)
RUNTIME_ARGS = Declaration(
    "SetRuntimeArgs", HOST_PART, SCOPE,
    "void SetRuntimeArgs(\n"
    "    Program& program, KernelHandle kernel, const CoreCoord& core, const std::vector<uint32_t>& args);",
)
BUFFER_CONFIG = Declaration(
    "CircularBufferConfig", TYPES_PART, SCOPE,
    "class CircularBufferConfig {\n"
    "public:\n"
    "    CircularBufferConfig(uint32_t total_size, const std::map<uint8_t, DataFormat>& formats);\n"
    "\n"
    "    CircularBufferConfig& set_page_size(uint8_t index, uint32_t page_size);\n"
    "};",
)
BUFFER_TYPE = Declaration("BufferType", TYPES_PART, SCOPE, "enum class BufferType {\n    DRAM,\n    L1,\n};")
MESH_WRITE = Declaration(
    "EnqueueWriteMeshBuffer", MESH_PART, MESH_SCOPE,
    "void EnqueueWriteMeshBuffer(MeshCommandQueue& queue, MeshBuffer& buffer, const void* data, bool blocking);",
)


def sized(name: str, size: int) -> Declaration:
    """Return a one-line SYNTHETIC function declaration `name` in SIZED_PART whose text is exactly `size` characters."""
    head, tail = f"void {name}(uint32_t ", ");"
    return Declaration(name, SIZED_PART, SCOPE, head + "x" * (size - len(head) - len(tail)) + tail)


# Declarations sized for SIZED_MAX_CHARS: named in this order, 300 and 150 fit together and 400 does not fit after
# 300; 600 never fits.
SIZED_MAX_CHARS = 500
ALPHA = sized("AlphaTileOp", 300)
BETA = sized("BetaTileOp", 400)
GAMMA = sized("GammaTileOp", 150)
DELTA = sized("DeltaTileOp", 600)


def namespace(scope: str, declarations: Sequence[Declaration]) -> str:
    """Return a namespace block holding `declarations`, one blank line apart, as the generated reference writes it."""
    return f"namespace {scope} {{\n\n" + "\n\n".join(item.text for item in declarations) + "\n\n}\n"


def header_part(header: str, body: str) -> str:
    """Return one header part: its include line, a blank line, and its declarations."""
    return f"// ==== #include <{header}> ====\n\n{body}"


# The SYNTHETIC pack: a guide that mentions CreateKernel in prose, then the reference, whose preamble precedes four
# header parts. Its entries are joined in manifest order, guide first, so the last part runs to the pack's end.
DECL_PACK = "syn-decl-pack"
GUIDE_TEXT = "SYNTHETIC guide: call CreateKernel once per kernel; a host program opens the device first.\n"
PREAMBLE = (
    "# SYNTHETIC host API reference\n\n"
    "SYNTHETIC preamble: declarations only, each part after the include line that declares it.\n"
)
HOST_API_TEXT = "\n".join([
    PREAMBLE,
    header_part(HOST_PART, namespace(SCOPE, [KERNEL_AT_CORE, KERNEL_ON_RANGE, CIRCULAR_BUFFER, RUNTIME_ARGS])),
    header_part(TYPES_PART, namespace(SCOPE, [BUFFER_CONFIG, BUFFER_TYPE])),
    header_part(MESH_PART, f"namespace {SCOPE} {{\n\n" + namespace("distributed", [MESH_WRITE]) + "\n}\n"),
    header_part(SIZED_PART, namespace(SCOPE, [ALPHA, BETA, GAMMA, DELTA])),
])
DECL_PACK_ENTRIES = {"guide.tt": GUIDE_TEXT, "host_api.tt": HOST_API_TEXT}
# A pack that serves only the source language.
CPP_PACK = "syn-cpp-pack"
CPP_PACK_ENTRIES = {"guide.cpp": "SYNTHETIC guide for cpp programs.\n"}

# A reply that does not build: three errors naming declared functions and types (CreateKernel twice, the second time
# after CircularBufferConfig), a member name the pack declares nowhere, a namespace, and a warning naming a declared
# function, which carries nothing.
NAMING = program(
    0,
    f"{ERROR}no matching function for call to 'CreateKernel'",
    f"{WARNING}SYNTHETIC result of 'SetRuntimeArgs' is unused",
    f"{ERROR}'CircularBufferConfig' has no member named 'set_tile_dims'; see 'CreateKernel'",
    f"{ERROR}cannot convert 'BufferType' in a call to 'EnqueueWriteMeshBuffer' in namespace 'tt::tt_metal'",
)
NAMED = [KERNEL_AT_CORE, KERNEL_ON_RANGE, BUFFER_CONFIG, BUFFER_TYPE, MESH_WRITE]
# BROKEN's two errors name CreateCircularBuffer and EnqueueWriteMeshBuffer, which the pack declares.
BROKEN_NAMED = [CIRCULAR_BUFFER, MESH_WRITE]


# ---------------------------------------------------------------------------
# Fixtures and helpers


@pytest.fixture(autouse=True)
def scratch_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Set LASSI_SCRATCH to the test's scratch directory (a retrieval index lies under it), TMPDIR inside it."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_TOOLCHAINS"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "scratch" / "tmp").mkdir(parents=True)
    monkeypatch.setenv("LASSI_SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch" / "tmp"))


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """Return the scratch directory every run of a test writes under."""
    return tmp_path / "scratch"


@pytest.fixture
def assets(base: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return the temporary assets root: tt-host-v0, the SYNTHETIC packs, and the SYNTHETIC documentation corpus."""
    root = make_assets(base, monkeypatch)
    write_tree(root / "context" / DECL_PACK, DECL_PACK_ENTRIES)
    write_tree(root / "context" / CPP_PACK, CPP_PACK_ENTRIES)
    write_corpus(root, base, CORPUS, MAIN_CHUNKS)
    return root


@pytest.fixture
def bench(base: Path, monkeypatch: pytest.MonkeyPatch, assets: Path) -> Path:
    """Return the bench root of the SYNTHETIC suite, with the assets root in place."""
    return write_suite(base, monkeypatch)


def section(**changes: Any) -> dict[str, Any]:
    """Return SECTION with `changes`."""
    return {**SECTION, **changes}


def entry_text(declaration: Declaration) -> str:
    """Return the prompt set's declaration.txt rendered with one declaration's fields."""
    fields = {"name": declaration.name, "header": declaration.header, "scope": declaration.scope,
              "text": declaration.text}
    return render(PROMPTS, DECLARATION, fields)


def lookup_blocks(declarations: Sequence[Declaration]) -> set[str]:
    """Return the lookup block for `declarations`: entries one blank line apart, with or without a last line feed."""
    text = "\n\n".join(entry_text(item).rstrip("\n") for item in declarations)
    return {render(PROMPTS, LOOKED_UP, {"declarations": value}) for value in (text, text + "\n")}


def carrying(prompt: str, declarations: Sequence[Declaration], before: Sequence[str] = ()) -> set[str]:
    """Return the forms `prompt` may take with the lookup block for `declarations`, after the blocks `before`."""
    return {joined([*before, block], prompt) for block in lookup_blocks(declarations)}


def looked_up_notes(request: Request) -> list[Diagnostic]:
    """Return the looked-up diagnostics on a request."""
    return [item for item in request.diagnostics if item.code == LOOKED_UP_CODE]


def attempt_notes(trial: Trial) -> list[Diagnostic]:
    """Return every looked-up diagnostic on any attempt of `trial` (there must be none)."""
    return [item for attempt in trial.attempts for item in attempt.diagnostics if item.code == LOOKED_UP_CODE]


def four_prompts(first: str, *prefix: str) -> Script:
    """Return the four-prompt script with `first` as attempt 0: a compile error, a failed run, a mismatch, a pass."""
    return Script(replies=[*prefix, first, program(1), program(2), program(3)], attempts=[crashed, wrong, doubled])


def lookup_pair(base: Path, bench: Path, first: str, data: Mapping[str, Any]) -> tuple[list[str], list[str], Trial]:
    """Run the four-prompt script on the pack without the section, then with `data`; return both prompts and a trial."""
    plain = four_prompts(first)
    run(base, bench, plain, recipe_data(context=[DECL_PACK]), "plain")
    script = four_prompts(first)
    _, trial = run(base, bench, script, data, "looking")
    return user_prompts(plain), user_prompts(script), trial


def correction_pair(base: Path, bench: Path, first: str, lookup: Mapping[str, Any]) -> tuple[str, str, Trial]:
    """Run [generate, one compile correction] on the pack without the section and with `lookup`.

    Return the plain correction prompt, the one sent with the section, and
    the trial run with it.
    """
    plain = Script(replies=[first, program(1)], attempts=[doubled])
    run(base, bench, plain, recipe_data(context=[DECL_PACK], loop={}), "plain")
    script = Script(replies=[first, program(1)], attempts=[doubled])
    _, trial = run(base, bench, script, recipe_data(context=[DECL_PACK], loop={}, lookup=dict(lookup)), "looking")
    assert [request.stage for request in trial.requests or []] == ["generate", "compile_loop"]
    plain_prompts, prompts = user_prompts(plain), user_prompts(script)
    assert len(plain_prompts) == len(prompts) == 2
    return plain_prompts[1], prompts[1], trial


# ---------------------------------------------------------------------------
# The recipe section (design point 1)


def load_refusal(base: Path, data: Mapping[str, Any]) -> str:
    """Load `data`, expect a RecipeError from a loader that knows the section, and return its message."""
    with pytest.raises(RecipeError) as caught:
        load(base, data)
    message = str(caught.value)
    assert f"unknown key {LOOKUP};" not in message, f"the loader does not know the section: {message}"
    return message


def test_the_section_loads_as_written_and_enters_the_hash(base: Path) -> None:
    (base / "plain").mkdir()
    (base / "with").mkdir()
    plain = load(base / "plain", recipe_data(), "recipe")
    assert LOOKUP not in plain.data and LOOKUP not in plain.canonical_yaml, "absent, nothing materializes"
    with_section = load(base / "with", recipe_data(lookup=SECTION), "recipe")
    assert with_section.data[LOOKUP] == SECTION
    assert with_section.recipe_hash != plain.recipe_hash, "a section that is set enters the hash"


@pytest.mark.parametrize("key", sorted(SECTION))
def test_a_missing_key_is_refused_naming_it(base: Path, key: str) -> None:
    partial = {name: value for name, value in SECTION.items() if name != key}
    assert f"{LOOKUP}.{key}" in load_refusal(base, recipe_data(lookup=partial))


def test_an_unknown_key_is_refused_naming_it(base: Path) -> None:
    assert f"{LOOKUP}.extra" in load_refusal(base, recipe_data(lookup=section(extra=1)))


@pytest.mark.parametrize("value", [["correct"], "correct", 8000, None], ids=["list", "string", "int", "null"])
def test_a_section_that_is_not_a_mapping_is_refused(base: Path, value: Any) -> None:
    assert LOOKUP in load_refusal(base, recipe_data(lookup=value))


BAD_VALUES = [
    ("max_chars", 499), ("max_chars", 32001), ("max_chars", True), ("max_chars", "8000"), ("max_chars", 8000.5),
    ("prompts", []), ("prompts", ["correct", "correct"]), ("prompts", ["generate"]), ("prompts", ["fix"]),
    ("prompts", "correct"), ("prompts", [1]), ("prompts", {"correct": True}),
]


@pytest.mark.parametrize(("key", "value"), BAD_VALUES, ids=[f"{key}-{value!r}" for key, value in BAD_VALUES])
def test_a_value_of_the_wrong_type_or_range_is_refused_naming_its_key(base: Path, key: str, value: Any) -> None:
    assert f"{LOOKUP}.{key}" in load_refusal(base, recipe_data(lookup=section(**{key: value})))


@pytest.mark.parametrize("value", [500, 32000])
def test_the_bounds_of_max_chars_load(base: Path, value: int) -> None:
    assert load(base, recipe_data(lookup=section(max_chars=value))).data[LOOKUP]["max_chars"] == value


def test_the_section_is_refused_under_faithful(base: Path) -> None:
    message = load_refusal(base, recipe_data(lookup=SECTION, faithful=True, loop={}))
    assert LOOKUP in message and "faithful: true" in message and "loaded.yaml" in message, message


def test_an_inherited_section_is_refused_when_the_child_turns_faithful_on(base: Path) -> None:
    load(base, recipe_data(lookup=SECTION, loop={}), "lookup-parent")
    message = load_refusal(base, {"extends": "lookup-parent.yaml", "faithful": True})
    assert LOOKUP in message and "faithful: true" in message, message


# ---------------------------------------------------------------------------
# Off when absent: the goldens and the committed hashes (design point 1)


def test_without_the_section_every_prompt_matches_the_goldens(base: Path, bench: Path) -> None:
    assert all(f"'{item.name}'" in BROKEN for item in BROKEN_NAMED), "the goldens' errors name declared functions"
    script = full_script()
    _, trial = run(base, bench, script, recipe_data(context=[DECL_PACK]), "plain")
    assert [request.stage for request in trial.requests or []] == REQUEST_STAGES
    for kind, prompt in zip(PROMPT_KINDS, user_prompts(script), strict=True):
        assert prompt == golden(kind), f"the {kind} prompt changed without the section"
    assert all(looked_up_notes(request) == [] for request in trial.requests or [])


@pytest.mark.parametrize("path", sorted(RECIPE_HASHES))
def test_a_committed_recipe_keeps_its_recipe_hash(path: str) -> None:
    recipe = load_recipe(REPO / path)
    assert LOOKUP not in recipe.data
    assert recipe.recipe_hash == RECIPE_HASHES[path]


# ---------------------------------------------------------------------------
# The runner's refusals (design point 2)


def run_refusal(base: Path, bench: Path, data: Mapping[str, Any]) -> str:
    """Return the runner's refusal of `data`, after checking that the loader knew the section."""
    message = refused(base, bench, data)
    assert f"unknown key {LOOKUP};" not in message, f"the loader does not know the section: {message}"
    return message


def test_the_runner_refuses_the_section_with_a_fragment_prompt_set(base: Path, bench: Path, assets: Path) -> None:
    write_tree(assets / "prompts" / FRAGMENT_SET, FRAGMENTS)
    message = run_refusal(base, bench, fragment_recipe(lookup=SECTION))
    assert LOOKUP in message and "fragment" in message, message
    script = Script(replies=[program(0)], attempts=[])
    _, trial = run(base, bench, script, fragment_recipe(), "upstream")
    assert [request.stage for request in trial.requests or []] == ["generate"], "without it the same recipe runs"


@pytest.mark.parametrize("packs", [[], [CPP_PACK]], ids=["no-pack", "source-language-pack"])
def test_the_runner_refuses_the_section_without_a_pack_for_the_target(
    base: Path, bench: Path, packs: list[str]
) -> None:
    data = recipe_data(lookup=SECTION, **({"context": packs} if packs else {}))
    message = run_refusal(base, bench, data)
    assert LOOKUP in message and "pack" in message, message


@pytest.mark.parametrize("missing", [LOOKED_UP, DECLARATION])
def test_the_runner_refuses_a_template_set_without_a_lookup_template(
    base: Path, bench: Path, assets: Path, missing: str
) -> None:
    other = DECLARATION if missing == LOOKED_UP else LOOKED_UP
    name = template_set(assets, f"lacks-{missing}", drop=[missing], texts={other: SYN_LOOKUP_TEMPLATES[other]})
    message = run_refusal(base, bench, recipe_data(prompts=name, context=[DECL_PACK], lookup=SECTION))
    assert f"{missing}.txt" in message, message


# ---------------------------------------------------------------------------
# The identifiers, the declarations, and the block (design points 3, 4, and 5)


def test_a_compile_correction_carries_the_declarations_its_errors_name(base: Path, bench: Path) -> None:
    plain, prompts, trial = lookup_pair(base, bench, NAMING, recipe_data(context=[DECL_PACK], lookup=SECTION))
    assert [request.stage for request in trial.requests or []] == REQUEST_STAGES
    listed = prompt_diagnostics(trial.attempts[0].diagnostics)
    assert [item.severity for item in listed] == ["error", "warning", "error", "error"], "the fixture's diagnostics"
    # CreateKernel's two overloads in pack order, once though named twice; the class whole, its constructor inside
    # it; the enum; the function of the nested namespace. Not CreateCircularBuffer, whose parameter only mentions
    # CircularBufferConfig, not SetRuntimeArgs, which only the warning names, and nothing from the guide's prose.
    assert prompts[1] in carrying(plain[1], NAMED), prompts[1]


def test_the_generation_run_error_and_mismatch_prompts_carry_none(base: Path, bench: Path) -> None:
    plain, prompts, trial = lookup_pair(base, bench, NAMING, recipe_data(context=[DECL_PACK], lookup=SECTION))
    assert "CreateKernel" in plain[2], "the run error the run-error correction shows names a declared function"
    for number, kind in enumerate(PROMPT_KINDS):
        if number == 1:
            continue
        assert prompts[number] == plain[number], f"the {kind} prompt is unchanged"
        assert looked_up_notes((trial.requests or [])[number]) == [], f"the {kind} request carried no declaration"


NO_MATCH = {
    "no-declared-name": [f"{ERROR}use of undeclared identifier 'zorbquant_tile'"],
    "empty-message": [BARE_ERROR],
    "warning-only": [
        f"{WARNING}SYNTHETIC result of 'CreateKernel' is unused",
        f"{ERROR}SYNTHETIC expected a semicolon",
    ],
    "unlisted-error": [
        *(f"{ERROR}SYNTHETIC filler error {number}" for number in range(DIAGNOSTIC_COUNT_CAP)),
        f"{ERROR}use of undeclared identifier 'CreateKernel'",
    ],
}


@pytest.mark.parametrize("case", sorted(NO_MATCH))
def test_no_block_when_no_listed_error_names_a_declared_identifier(base: Path, bench: Path, case: str) -> None:
    plain, prompt, trial = correction_pair(base, bench, program(0, *NO_MATCH[case]), SECTION)
    if case == "unlisted-error":
        assert "[1 more diagnostic(s) not shown]" in plain, "the error naming CreateKernel is past the count cap"
    assert prompt == plain, "no declaration, no block"
    assert all(looked_up_notes(request) == [] for request in trial.requests or [])


def test_max_chars_skips_a_declaration_whole_and_keeps_a_later_smaller_one(base: Path, bench: Path) -> None:
    reply = program(0, f"{ERROR}SYNTHETIC no matching overload among 'AlphaTileOp', 'BetaTileOp', and 'GammaTileOp'")
    plain, prompt, trial = correction_pair(base, bench, reply, section(max_chars=SIZED_MAX_CHARS))
    assert [len(item.text) for item in (ALPHA, BETA, GAMMA)] == [300, 400, 150], "the fixture's sizes"
    assert prompt in carrying(plain, [ALPHA, GAMMA]), prompt
    assert BETA.text not in prompt, "a declaration past max_chars is skipped whole, never cut"
    (note,) = looked_up_notes((trial.requests or [])[1])
    assert ALPHA.name in note.message and GAMMA.name in note.message, note.message
    assert BETA.name not in note.message, "a skipped declaration's name is not listed"


def test_a_declaration_past_max_chars_alone_gives_no_block(base: Path, bench: Path) -> None:
    reply = program(0, f"{ERROR}use of undeclared identifier 'DeltaTileOp'")
    plain, prompt, trial = correction_pair(base, bench, reply, section(max_chars=SIZED_MAX_CHARS))
    assert len(DELTA.text) > SIZED_MAX_CHARS
    assert prompt == plain, "nothing fits, so there is no block"
    assert all(looked_up_notes(request) == [] for request in trial.requests or [])


def test_the_block_goes_after_the_knowledge_and_retrieved_blocks(base: Path, bench: Path) -> None:
    plain = Script(replies=[BROKEN, program(1)], attempts=[doubled])
    run(base, bench, plain, recipe_data(loop={}), "plain")
    script = Script(replies=[SUMMARY, BROKEN, program(1)], attempts=[doubled])
    data = recipe_data(context=[DECL_PACK], stages=CONTEXT_STAGES, retrieval=RETRIEVAL_SECTION, lookup=SECTION,
                       loop={})
    _, trial = run(base, bench, script, data, "all-blocks")
    requests = trial.requests or []
    assert [request.stage for request in requests] == ["summarize_context", "generate", "compile_loop"]
    errors = [item for item in prompt_diagnostics(trial.attempts[0].diagnostics) if item.severity == "error"]
    query = "\n".join(query_from_diagnostic(item) for item in errors)
    passages = kept(load_index(index_path(base, CORPUS)).search(query, RETRIEVAL_SECTION["k"]),
                    RETRIEVAL_SECTION["max_chars"])
    assert passages, "the fixture's errors match the SYNTHETIC documentation"
    expected = {
        form
        for passage_block in retrieved_blocks(passages)
        for form in carrying(user_prompts(plain)[1], BROKEN_NAMED, [knowledge_block(), passage_block])
    }
    prompt = user_prompts(script)[2]
    assert prompt in expected, prompt
    assert len(looked_up_notes(requests[2])) == 1
    assert len([item for item in requests[2].diagnostics if item.code == "retrieved"]) == 1


# ---------------------------------------------------------------------------
# The record (design point 6)


def test_a_request_that_carried_declarations_records_one_looked_up_note(base: Path, bench: Path) -> None:
    _, prompts, trial = lookup_pair(base, bench, NAMING, recipe_data(context=[DECL_PACK], lookup=SECTION))
    requests = trial.requests or []
    assert len(requests) == 4
    (note,) = looked_up_notes(requests[1])
    assert note.severity == "note", "the record's informational severity"
    for item in NAMED:
        assert item.name in note.message and item.header in note.message, f"{item.name}, {item.header}: {note.message}"
    assert RUNTIME_ARGS.name not in note.message, "only a warning names SetRuntimeArgs, so it carried nothing"
    assert attempt_notes(trial) == [], "the note is on the Request, never on an attempt"
    assert all(looked_up_notes(request) == [] for number, request in enumerate(requests) if number != 1)
    assert not any(f"[{LOOKED_UP_CODE}]" in prompt for prompt in prompts), "the note never enters a prompt"


# ---------------------------------------------------------------------------
# The two templates of tt-host-v0 (design point 7)


@pytest.mark.parametrize("name", sorted(LOOKUP_TEMPLATES))
def test_tt_host_v0_holds_the_lookup_template_and_it_renders_with_its_fields(name: str) -> None:
    template_text(name)
    fields = {field_name: f"<SYNTHETIC {field_name}>" for field_name in LOOKUP_TEMPLATES[name]}
    rendered = render(PROMPTS, name, fields)
    for field_name, value in fields.items():
        assert value in rendered, f"{name}.txt does not show ${field_name}"


@pytest.mark.parametrize("name", sorted(LOOKUP_TEMPLATES))
def test_tt_host_v0_lookup_template_is_generic(name: str) -> None:
    assert named_tier_a(template_text(name)) == [], f"{name}.txt names a Tier A item or file (Agent Rule 5)"
