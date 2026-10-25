"""Tests for the worked examples of the cpp -> tt prompts (opt-in; OQ-083 (b), the owner's answer of 2026-10-25).

Bible: Component Interfaces (the context blocks under a template set),
Project Recipes (a new optional section, beside the Lookup bullet), Harness
Contract (a prompt shows only what the trial may see), Agent Rules 4 (a
departure from upstream is opt-in and refused under faithful: true) and 5
(no Tier A text in a template or a pack). The evidence: the build-error
report of 2026-10-25 (results/build-errors-20261025/summary.md) shows the
models writing the TT-Metalium host API from memory, an older API, so a
prompt may carry complete host programs at the pin.

The contract these tests fix:

1. An optional run recipe section `examples`, as in
   `examples: {pack: tt-examples-v0, max_chars: 24000, prompts: [generate]}`,
   with three required keys: pack, a plain name (a directory under
   assets/examples/ holding MANIFEST.yaml); max_chars, the carried examples'
   total text per prompt, an integer from 1000 to 64000 (a bool or a float
   refused); and prompts, a list of distinct names among {generate,
   correct} that holds generate. A section that is not a mapping, a null,
   an unknown or missing key, and a value of the wrong type or range are
   refused at load naming `examples.<key>`; the section is refused at load
   under faithful: true, inherited or not, naming `examples`. Absent,
   nothing is materialized, so a recipe keeps its resolved recipe and hash.
2. The runner refuses the section before any directory exists: with a
   fragment prompt set; with a template set that lacks examples.txt or
   example.txt, or whose template does not render with its fields (the
   message names the file); and when the pack's MANIFEST.yaml is missing or
   a fragment is missing, of another size or sha256, or the manifest's pack
   sha256 is not that of its entries (the message names examples.pack).
3. The block: each carried example rendered with example.txt ($name, and
   $files, its files as FILE blocks in the pack's order, one blank line
   apart), the entries one blank line apart, as $examples of examples.txt.
   Examples go whole in pack order; one whose $files text would take the
   carried examples' total past max_chars is skipped whole, never cut, and a
   later smaller one may still fit. The block goes before the generation
   prompt, and before each compile correction when prompts lists correct,
   after the knowledge block and before the retrieved and looked-up blocks,
   one blank line apart. A run-error correction and an oracle-mismatch
   correction get none; with prompts [generate] no correction gets one; with
   no example carried there is no block.
4. Each request that carried examples gets exactly one Diagnostic with code
   `examples`, severity note, stage parse, on its Request (never on an
   attempt), naming each carried example and no skipped one; it never enters
   a prompt.
5. generate's and compile_loop's describe() name the pack when the section
   lists their prompt, and say nothing of it otherwise.
6. tt-host-v0 gains examples.txt and example.txt, plain ASCII, rendering
   with their fields and naming no Tier A item, kernel, or file. Without the
   section every prompt matches the goldens under
   tests/core/golden/template-prompts/ (test_template_context gives their
   provenance), and the committed recipes keep their recipe hashes.

Every pack, program, reply, and array here is SYNTHETIC, written in a
temporary directory; the generated tt-examples-v0 pack is never read. No
value here is a measurement.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_declaration_lookup import (
    BROKEN_NAMED,
    CORE_API,
    DECL_PACK,
    DECL_PACK_ENTRIES,
    HOST_DECLARATIONS,
    HOST_PART,
    TYPES_DECLARATIONS,
    TYPES_PART,
    lookup_blocks,
)
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
    BROKEN,
    CONTEXT_STAGES,
    ITEM,
    KERNEL,
    KERNEL_TEXT,
    PROMPT_KINDS,
    PROMPTS,
    RECIPE_HASHES,
    REPO,
    REQUEST_STAGES,
    SOURCE_TEXT,
    SUMMARY,
    UNFILLED,
    Script,
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
    stage_context,
    template_set,
    template_text,
    user_prompts,
    write_suite,
    write_tree,
)

from lassi.core.files import render_file_blocks
from lassi.core.recipe import RecipeError, load_recipe
from lassi.core.record import Diagnostic, Request, Trial
from lassi.core.stages import CompileLoopStage, GenerateStage, prompt_diagnostics
from lassi.prompts import render
from lassi.prompts.retrieval import load_index

# The section, its two templates with their fields, and the code of its note.
EXAMPLES = "examples"
PACK = "syn-examples"
SECTION = {"pack": PACK, "max_chars": 64000, "prompts": ["generate"]}
BOTH = {**SECTION, "prompts": ["generate", "correct"]}
EXAMPLES_TEMPLATE = "examples"
EXAMPLE_TEMPLATE = "example"
TEMPLATES = {EXAMPLES_TEMPLATE: ("examples",), EXAMPLE_TEMPLATE: ("name", "files")}
EXAMPLES_CODE = "examples"
GENERATOR = "tools/build_tt_examples.py"
SYN_TEMPLATES = {
    EXAMPLES_TEMPLATE: "SYNTHETIC: worked examples follow.\n\n$examples\n",
    EXAMPLE_TEMPLATE: "SYNTHETIC example $name\n$files\n",
}

Files = Sequence[tuple[str, str]]


def host(name: str, size: int = 0) -> str:
    """Return a SYNTHETIC host program for example `name`, padded with comment lines to at least `size` characters."""
    text = f"// SYNTHETIC host program of {name}\n#include <tt-metalium/host_api.hpp>\nint main() {{ return 0; }}\n"
    while len(text) < size:
        text += f"// SYNTHETIC padding line of {name}, written to size the example\n"
    return text


def kernel(name: str) -> str:
    """Return a SYNTHETIC kernel of example `name`."""
    return f"// SYNTHETIC kernel of {name}\nvoid kernel_main() {{}}\n"


# The SYNTHETIC examples, in pack order: zeta before alpha, so pack order is not name order, and each host program
# before its kernels, which are not in name order with it.
ZETA = ("syn_zeta", [("syn_zeta.cpp", host("syn_zeta")), ("kernels/a_reader.cpp", kernel("zeta reader")),
                     ("kernels/b_writer.cpp", kernel("zeta writer"))])
ALPHA = ("syn_alpha", [("syn_alpha.cpp", host("syn_alpha")), ("kernels/reader.cpp", kernel("alpha"))])
# For max_chars 1000: BIG's text passes it alone, SMALL's fits after it is skipped.
BIG = ("syn_big", [("syn_big.cpp", host("syn_big", 1500))])
SMALL = ("syn_small", [("syn_small.cpp", host("syn_small"))])
SIZED = "syn-sized-examples"
SIZED_MAX_CHARS = 1000


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
    """Return the temporary assets root: tt-host-v0, the SYNTHETIC packs, corpus, and examples packs."""
    root = make_assets(base, monkeypatch)
    write_tree(root / "context" / DECL_PACK, DECL_PACK_ENTRIES)
    write_corpus(root, base, CORPUS, MAIN_CHUNKS)
    write_pack(root, PACK, [ZETA, ALPHA])
    write_pack(root, SIZED, [BIG, SMALL])
    return root


@pytest.fixture
def bench(base: Path, monkeypatch: pytest.MonkeyPatch, assets: Path) -> Path:
    """Return the bench root of the SYNTHETIC suite, with the assets root in place."""
    return write_suite(base, monkeypatch)


def sha256(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def write_pack(
    assets: Path, name: str, examples: Sequence[tuple[str, Files]], *, pack_sha256: str | None = None
) -> Path:
    """Write a SYNTHETIC examples pack under `assets`/examples/`name`: each file, and the manifest; return the tree.

    The manifest lists each example with its files' pack paths
    (<example>/<path>), sha256, and size, and the pack's sha256: that of the
    lines "<sha256>  <path>\\n" of every file in order, unless `pack_sha256`
    replaces it.
    """
    tree = assets / "examples" / name
    rows: list[dict[str, Any]] = []
    for example, files in examples:
        entries = []
        for relative, text in files:
            data = text.encode("ascii")
            path = f"{example}/{relative}"
            (tree / path).parent.mkdir(parents=True, exist_ok=True)
            (tree / path).write_bytes(data)
            entries.append({"path": path, "sha256": sha256(data), "bytes": len(data)})
        rows.append({"name": example, "files": entries})
    lines = "".join(f"{entry['sha256']}  {entry['path']}\n" for row in rows for entry in row["files"])
    document = {"generator": GENERATOR, "pack": name, "repo": "https://example.invalid/SYNTHETIC.git",
                "commit": "0" * 40, "examples": rows, "sha256": pack_sha256 or sha256(lines.encode("ascii"))}
    (tree / "MANIFEST.yaml").write_bytes(yaml.safe_dump(document, sort_keys=False).encode("ascii"))
    return tree


def section(**changes: Any) -> dict[str, Any]:
    """Return SECTION with `changes`."""
    return {**SECTION, **changes}


def file_blocks(files: Files) -> str:
    """Return `files` as FILE blocks in the order given, one blank line apart (each block as render_file_blocks)."""
    return "\n".join(render_file_blocks({path: text}) for path, text in files)


def entry_text(example: tuple[str, Files]) -> str:
    """Return the prompt set's example.txt rendered with one example."""
    name, files = example
    return render(PROMPTS, EXAMPLE_TEMPLATE, {"name": name, "files": file_blocks(files)})


def example_blocks(examples: Sequence[tuple[str, Files]]) -> set[str]:
    """Return the examples block: entries one blank line apart, with or without a last line feed."""
    text = "\n\n".join(entry_text(item).rstrip("\n") for item in examples)
    return {render(PROMPTS, EXAMPLES_TEMPLATE, {"examples": value}) for value in (text, text + "\n")}


def carrying(prompt: str, examples: Sequence[tuple[str, Files]], before: Sequence[str] = (),
             after: Sequence[str] = ()) -> set[str]:
    """Return the forms `prompt` may take with the examples block, after the blocks `before`, before `after`."""
    return {joined([*before, block, *after], prompt) for block in example_blocks(examples)}


def example_notes(request: Request) -> list[Diagnostic]:
    """Return the examples diagnostics on a request."""
    return [item for item in request.diagnostics if item.code == EXAMPLES_CODE]


def attempt_notes(trial: Trial) -> list[Diagnostic]:
    """Return every examples diagnostic on any attempt of `trial` (there must be none)."""
    return [item for attempt in trial.attempts for item in attempt.diagnostics if item.code == EXAMPLES_CODE]


def full_pair(base: Path, bench: Path, data: Mapping[str, Any]) -> tuple[list[str], list[str], Trial]:
    """Run the four-prompt script without the section, then with `data`; return both prompt lists and the trial."""
    plain = full_script()
    run(base, bench, plain, recipe_data(), "plain")
    script = full_script()
    _, trial = run(base, bench, script, data, "examples")
    assert [request.stage for request in trial.requests or []] == REQUEST_STAGES
    return user_prompts(plain), user_prompts(script), trial


def load_refusal(base: Path, data: Mapping[str, Any]) -> str:
    """Load `data`, expect a RecipeError from a loader that knows the section, and return its message."""
    with pytest.raises(RecipeError) as caught:
        load(base, data)
    message = str(caught.value)
    assert f"unknown key {EXAMPLES};" not in message, f"the loader does not know the section: {message}"
    return message


def run_refusal(base: Path, bench: Path, data: Mapping[str, Any]) -> str:
    """Return the runner's refusal of `data`, after checking that the loader knew the section."""
    message = refused(base, bench, data)
    assert f"unknown key {EXAMPLES};" not in message, f"the loader does not know the section: {message}"
    return message


# ---------------------------------------------------------------------------
# The recipe section (contract point 1)


def test_the_section_loads_as_written_and_enters_the_hash(base: Path) -> None:
    (base / "plain").mkdir()
    (base / "with").mkdir()
    plain = load(base / "plain", recipe_data(), "recipe")
    assert EXAMPLES not in plain.data and "max_chars" not in plain.canonical_yaml, "absent, nothing materializes"
    with_section = load(base / "with", recipe_data(examples=SECTION), "recipe")
    assert with_section.data[EXAMPLES] == SECTION
    assert with_section.recipe_hash != plain.recipe_hash, "a section that is set enters the hash"


@pytest.mark.parametrize("key", sorted(SECTION))
def test_a_missing_key_is_refused_naming_it(base: Path, key: str) -> None:
    partial = {name: value for name, value in SECTION.items() if name != key}
    assert f"{EXAMPLES}.{key}" in load_refusal(base, recipe_data(examples=partial))


def test_an_unknown_key_is_refused_naming_it(base: Path) -> None:
    assert f"{EXAMPLES}.extra" in load_refusal(base, recipe_data(examples=section(extra=1)))


@pytest.mark.parametrize("value", [["generate"], "generate", 8000, None], ids=["list", "string", "int", "null"])
def test_a_section_that_is_not_a_mapping_is_refused(base: Path, value: Any) -> None:
    assert EXAMPLES in load_refusal(base, recipe_data(examples=value))


BAD_VALUES = [
    ("pack", ""), ("pack", "../tt-examples-v0"), ("pack", "a/b"), ("pack", 5), ("pack", None),
    ("max_chars", 999), ("max_chars", 64001), ("max_chars", True), ("max_chars", "8000"), ("max_chars", 8000.5),
    ("prompts", []), ("prompts", ["correct"]), ("prompts", ["generate", "generate"]), ("prompts", ["fix"]),
    ("prompts", "generate"), ("prompts", [1]), ("prompts", {"generate": True}),
]


@pytest.mark.parametrize(("key", "value"), BAD_VALUES, ids=[f"{key}-{value!r}" for key, value in BAD_VALUES])
def test_a_value_of_the_wrong_type_or_range_is_refused_naming_its_key(base: Path, key: str, value: Any) -> None:
    assert f"{EXAMPLES}.{key}" in load_refusal(base, recipe_data(examples=section(**{key: value})))


GOOD_VALUES = [("max_chars", 1000), ("max_chars", 64000), ("prompts", ["generate"]),
               ("prompts", ["generate", "correct"]), ("pack", "tt-examples-v0")]


@pytest.mark.parametrize(("key", "value"), GOOD_VALUES, ids=[f"{key}-{value!r}" for key, value in GOOD_VALUES])
def test_the_bounds_and_both_prompt_lists_load(base: Path, key: str, value: Any) -> None:
    assert load(base, recipe_data(examples=section(**{key: value}))).data[EXAMPLES][key] == value


def test_the_section_is_refused_under_faithful(base: Path) -> None:
    message = load_refusal(base, recipe_data(examples=SECTION, faithful=True, loop={}))
    assert EXAMPLES in message and "faithful: true" in message and "loaded.yaml" in message, message


def test_an_inherited_section_is_refused_when_the_child_turns_faithful_on(base: Path) -> None:
    load(base, recipe_data(examples=SECTION, loop={}), "examples-parent")
    message = load_refusal(base, {"extends": "examples-parent.yaml", "faithful": True})
    assert EXAMPLES in message and "faithful: true" in message, message


# ---------------------------------------------------------------------------
# Off when absent: the goldens and the committed hashes (contract point 6)


def test_without_the_section_every_prompt_matches_the_goldens(base: Path, bench: Path) -> None:
    script = full_script()
    _, trial = run(base, bench, script, recipe_data(), "plain")
    assert [request.stage for request in trial.requests or []] == REQUEST_STAGES
    for kind, prompt in zip(PROMPT_KINDS, user_prompts(script), strict=True):
        assert prompt == golden(kind), f"the {kind} prompt changed without the section"
    assert all(example_notes(request) == [] for request in trial.requests or [])


@pytest.mark.parametrize("path", sorted(RECIPE_HASHES))
def test_a_committed_recipe_keeps_its_recipe_hash(path: str) -> None:
    recipe = load_recipe(REPO / path)
    assert EXAMPLES not in recipe.data
    assert recipe.recipe_hash == RECIPE_HASHES[path]


# ---------------------------------------------------------------------------
# The runner's refusals (contract point 2)


def test_the_runner_refuses_the_section_with_a_fragment_prompt_set(base: Path, bench: Path, assets: Path) -> None:
    write_tree(assets / "prompts" / FRAGMENT_SET, FRAGMENTS)
    message = run_refusal(base, bench, fragment_recipe(examples=SECTION))
    assert EXAMPLES in message and "fragment" in message, message
    script = Script(replies=[program(0)], attempts=[])
    _, trial = run(base, bench, script, fragment_recipe(), "upstream")
    assert [request.stage for request in trial.requests or []] == ["generate"], "without it the same recipe runs"


@pytest.mark.parametrize("missing", [EXAMPLES_TEMPLATE, EXAMPLE_TEMPLATE])
def test_the_runner_refuses_a_template_set_without_an_examples_template(
    base: Path, bench: Path, assets: Path, missing: str
) -> None:
    other = EXAMPLE_TEMPLATE if missing == EXAMPLES_TEMPLATE else EXAMPLES_TEMPLATE
    name = template_set(assets, f"lacks-{missing}", drop=[missing], texts={other: SYN_TEMPLATES[other]})
    message = run_refusal(base, bench, recipe_data(prompts=name, examples=SECTION))
    assert f"{missing}.txt" in message, message


@pytest.mark.parametrize("broken", [EXAMPLES_TEMPLATE, EXAMPLE_TEMPLATE])
def test_the_runner_refuses_an_examples_template_that_does_not_render(
    base: Path, bench: Path, assets: Path, broken: str
) -> None:
    texts = dict(SYN_TEMPLATES)
    texts[broken] = f"{texts[broken]}{UNFILLED}\n"
    name = template_set(assets, f"broken-{broken}", texts=texts)
    message = run_refusal(base, bench, recipe_data(prompts=name, examples=SECTION))
    assert f"{broken}.txt" in message, message


@pytest.mark.parametrize("made", [False, True], ids=["no-directory", "no-manifest"])
def test_the_runner_refuses_a_pack_with_no_manifest(base: Path, bench: Path, assets: Path, made: bool) -> None:
    if made:
        (assets / "examples" / "syn-empty").mkdir(parents=True)
    message = run_refusal(base, bench, recipe_data(examples=section(pack="syn-empty")))
    assert f"{EXAMPLES}.pack" in message and "MANIFEST.yaml" in message, message


def test_the_runner_refuses_a_missing_fragment(base: Path, bench: Path, assets: Path) -> None:
    (assets / "examples" / PACK / "syn_alpha" / "kernels" / "reader.cpp").unlink()
    message = run_refusal(base, bench, recipe_data(examples=SECTION))
    assert f"{EXAMPLES}.pack" in message and "reader.cpp" in message and GENERATOR in message, message


@pytest.mark.parametrize("change", ["same-size", "longer"])
def test_the_runner_refuses_a_fragment_that_fails_the_manifest(
    base: Path, bench: Path, assets: Path, change: str
) -> None:
    path = assets / "examples" / PACK / "syn_zeta" / "syn_zeta.cpp"
    data = path.read_bytes()
    path.write_bytes(data.replace(b"SYNTHETIC", b"SYNTHETIc") if change == "same-size" else data + b"// more\n")
    message = run_refusal(base, bench, recipe_data(examples=SECTION))
    assert f"{EXAMPLES}.pack" in message and "syn_zeta.cpp" in message, message


def test_the_runner_refuses_a_manifest_whose_pack_sha256_is_not_its_entries(
    base: Path, bench: Path, assets: Path
) -> None:
    write_pack(assets, "syn-stale", [ZETA], pack_sha256="f" * 64)
    message = run_refusal(base, bench, recipe_data(examples=section(pack="syn-stale")))
    assert f"{EXAMPLES}.pack" in message and "sha256" in message, message


# ---------------------------------------------------------------------------
# The block (contract point 3)


def test_the_generation_prompt_carries_every_example_in_pack_order(base: Path, bench: Path) -> None:
    plain, prompts, trial = full_pair(base, bench, recipe_data(examples=SECTION))
    assert prompts[0] in carrying(plain[0], [ZETA, ALPHA]), prompts[0]
    zeta, alpha = prompts[0].index("syn_zeta.cpp"), prompts[0].index("syn_alpha.cpp")
    assert zeta < prompts[0].index("kernels/a_reader.cpp") < prompts[0].index("kernels/b_writer.cpp") < alpha
    for number, kind in enumerate(PROMPT_KINDS[1:], start=1):
        assert prompts[number] == plain[number], f"with prompts [generate] the {kind} prompt is unchanged"


def test_with_correct_listed_a_compile_correction_carries_them_and_no_other_correction(
    base: Path, bench: Path
) -> None:
    plain, prompts, trial = full_pair(base, bench, recipe_data(examples=BOTH))
    assert prompts[0] in carrying(plain[0], [ZETA, ALPHA])
    assert prompts[1] in carrying(plain[1], [ZETA, ALPHA]), prompts[1]
    assert prompts[2] == plain[2], "a run-error correction carries no examples"
    assert prompts[3] == plain[3], "an oracle-mismatch correction carries no examples"


def test_max_chars_skips_an_example_whole_and_keeps_a_later_smaller_one(base: Path, bench: Path) -> None:
    assert len(file_blocks(BIG[1])) > SIZED_MAX_CHARS >= len(file_blocks(SMALL[1])), "the fixture's sizes"
    data = recipe_data(examples=section(pack=SIZED, max_chars=SIZED_MAX_CHARS))
    plain, prompts, trial = full_pair(base, bench, data)
    assert prompts[0] in carrying(plain[0], [SMALL]), prompts[0]
    assert "syn_big" not in prompts[0], "an example past max_chars is skipped whole, never cut"
    (note,) = example_notes((trial.requests or [])[0])
    assert "syn_small" in note.message and "syn_big" not in note.message, note.message


def test_no_block_and_no_note_when_no_example_fits(base: Path, bench: Path, assets: Path) -> None:
    write_pack(assets, "syn-big-only", [BIG])
    both = ["generate", "correct"]
    data = recipe_data(examples=section(pack="syn-big-only", max_chars=SIZED_MAX_CHARS, prompts=both))
    plain, prompts, trial = full_pair(base, bench, data)
    assert prompts == plain, "nothing fits, so there is no block"
    assert all(example_notes(request) == [] for request in trial.requests or [])


def test_the_block_goes_after_the_knowledge_block_and_before_the_retrieved_block(base: Path, bench: Path) -> None:
    plain = full_script()
    run(base, bench, plain, recipe_data(), "plain")
    script = full_script(SUMMARY)
    retrieval = {"index": CORPUS, "k": 2, "max_chars": 8000, "prompts": ["generate"]}
    data = recipe_data(context=[DECL_PACK], stages=CONTEXT_STAGES, retrieval=retrieval, examples=SECTION)
    _, trial = run(base, bench, script, data, "all-blocks")
    requests = trial.requests or []
    assert [request.stage for request in requests] == ["summarize_context", *REQUEST_STAGES]
    # generate's query: the item's kernels, then its source files (test_retrieval_stage fixes it).
    query = f"{render_file_blocks({KERNEL: KERNEL_TEXT})}\n{render_file_blocks({f'{ITEM}.cpp': SOURCE_TEXT})}"
    passages = kept(load_index(index_path(base, CORPUS)).search(query, 2), 8000)
    assert passages, "the fixture's kernels match the SYNTHETIC documentation"
    expected = {
        form
        for passage_block in retrieved_blocks(passages)
        for form in carrying(user_prompts(plain)[0], [ZETA, ALPHA], [knowledge_block()], [passage_block])
    }
    assert user_prompts(script)[1] in expected, user_prompts(script)[1]
    codes = [item.code for item in requests[1].diagnostics]
    assert codes.count(EXAMPLES_CODE) == 1 and codes.index(EXAMPLES_CODE) < codes.index("retrieved"), codes


def test_the_block_goes_before_the_looked_up_block_in_a_compile_correction(base: Path, bench: Path) -> None:
    plain = Script(replies=[BROKEN, program(1)], attempts=[doubled])
    run(base, bench, plain, recipe_data(context=[DECL_PACK], loop={}), "plain")
    script = Script(replies=[BROKEN, program(1)], attempts=[doubled])
    data = recipe_data(context=[DECL_PACK], loop={}, examples=BOTH, lookup={"max_chars": 8000, "prompts": ["correct"]})
    _, trial = run(base, bench, script, data, "examples-lookup")
    assert [request.stage for request in trial.requests or []] == ["generate", "compile_loop"]
    expected = {
        form
        for looked in lookup_blocks(BROKEN_NAMED)
        for form in carrying(user_prompts(plain)[1], [ZETA, ALPHA], after=[looked])
    }
    assert user_prompts(script)[1] in expected, user_prompts(script)[1]


# ---------------------------------------------------------------------------
# The record (contract point 4)


def test_a_request_that_carried_examples_records_one_examples_note(base: Path, bench: Path) -> None:
    _, prompts, trial = full_pair(base, bench, recipe_data(examples=BOTH))
    requests = trial.requests or []
    for number in (0, 1):
        (note,) = example_notes(requests[number])
        assert (note.stage, note.severity) == ("parse", "note"), note
        assert "syn_zeta" in note.message and "syn_alpha" in note.message and PACK in note.message, note.message
    assert example_notes(requests[2]) == [] and example_notes(requests[3]) == []
    assert attempt_notes(trial) == [], "the note is on the Request, never on an attempt"
    assert not any(f"[{EXAMPLES_CODE}]" in prompt for prompt in prompts), "the note never enters a prompt"
    listed = prompt_diagnostics(trial.attempts[0].diagnostics)
    assert all(item.code != EXAMPLES_CODE for item in listed)


# ---------------------------------------------------------------------------
# describe() (contract point 5)


@pytest.mark.parametrize("prompts", [["generate"], ["generate", "correct"]])
def test_describe_names_the_pack_where_the_section_lists_the_prompt(base: Path, prompts: list[str]) -> None:
    recipe = load(base, recipe_data(examples=section(prompts=prompts)))
    plain = load(base, recipe_data(), "plain")
    generate = GenerateStage(context=stage_context(recipe, PROMPTS)).describe()
    assert PACK in generate and EXAMPLES in generate, generate
    assert PACK not in GenerateStage(context=stage_context(plain, PROMPTS)).describe()
    toolchains: dict[str, Any] = {}
    compile_loop = CompileLoopStage(context=stage_context(recipe, PROMPTS, toolchains=toolchains)).describe()
    assert (PACK in compile_loop) == ("correct" in prompts), compile_loop
    assert PACK not in CompileLoopStage(context=stage_context(plain, PROMPTS, toolchains=toolchains)).describe()


# ---------------------------------------------------------------------------
# The two templates of tt-host-v0 (contract point 6)


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_tt_host_v0_holds_the_examples_template_and_it_renders_with_its_fields(name: str) -> None:
    template_text(name)
    fields = {field_name: f"<SYNTHETIC {field_name}>" for field_name in TEMPLATES[name]}
    rendered = render(PROMPTS, name, fields)
    for field_name, value in fields.items():
        assert value in rendered, f"{name}.txt does not show ${field_name}"


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_tt_host_v0_examples_template_is_generic(name: str) -> None:
    assert named_tier_a(template_text(name)) == [], f"{name}.txt names a Tier A item or file (Agent Rule 5)"


# ---------------------------------------------------------------------------
# Examples beside lookup.core: the generation prompt's blocks in order (knowledge, examples, retrieved, looked-up,
# core) and generate's description naming both

CORE_GENERATE = {"max_chars": 8000, "prompts": ["generate"], "core": [TYPES_PART, HOST_PART]}


def test_the_generation_prompt_carries_the_examples_then_the_core_block(base: Path, bench: Path) -> None:
    data = recipe_data(context=[DECL_PACK], examples=SECTION, lookup=dict(CORE_GENERATE))
    plain, prompts, trial = full_pair(base, bench, data)
    cores = lookup_blocks([*HOST_DECLARATIONS, *TYPES_DECLARATIONS], CORE_API)
    expected = {joined([block, core], plain[0]) for block in example_blocks([ZETA, ALPHA]) for core in cores}
    assert prompts[0] in expected, prompts[0]
    request = (trial.requests or [])[0]
    assert len(example_notes(request)) == 1
    assert [item for item in request.diagnostics if item.code == "looked-up"], request.diagnostics
    assert prompts[1:] == plain[1:], "neither section lists correct"


def test_generate_describes_the_examples_and_the_core_declarations(base: Path) -> None:
    recipe = load(base, recipe_data(context=[DECL_PACK], examples=SECTION, lookup=dict(CORE_GENERATE)))
    text = GenerateStage(context=stage_context(recipe, PROMPTS)).describe()
    assert PACK in text and "header part(s)" in text, text
    assert text.index(PACK) < text.index("header part(s)"), text
