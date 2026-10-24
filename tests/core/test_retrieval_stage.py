"""Tests for task P17.22: documentation passages in generation and corrections (compiler-guided lookup).

Bible: Component Interfaces (the Stage row: generate, compile_loop,
run_loop), Project Recipes (a new optional section), Harness Contract (what a
prompt may show), Agent Rules 4 (a departure from upstream is opt-in and
refused under faithful: true) and 5 (no Tier A text in a template).
plans/p17-portable.md, task P17.22; the owner's direction of 2026-10-23
("Ready a RAG of Tenstorrent documentation as well in case this is not
sufficient.").

The contract these tests fix (the P17.22 design, points 1 to 8):

1. An optional recipe section `retrieval` with four required keys: index (a
   directory under assets/rag/ holding MANIFEST.yaml), k (an int from 1 to
   16), max_chars (an int from 500 to 32000), and prompts (a non-empty subset
   of {generate, correct}, no repeats). A section that is not a mapping, an
   unknown or missing key, or a value of the wrong type or range is refused
   at load naming the key; the section is refused at load under faithful:
   true, naming `retrieval`. Absent, nothing is materialized and the resolved
   recipe and hash are unchanged.
2. The runner refuses the section, before any directory exists, with a
   fragment prompt set, with a template set lacking retrieved.txt or
   passage.txt, when assets/rag/<index>/MANIFEST.yaml is missing, and when the
   index file is missing or fails the manifest's sha256. The index loads once
   per run and every trial searches that one object.
3. Queries: generate's is the item's kernels (kernel_files) and then the
   source files' text, joined by one newline; a correction's is
   query_from_diagnostic of each error-severity Diagnostic the prompt lists,
   joined by one newline, or, for a run-error correction, the run error text
   the prompt shows. An oracle-mismatch correction gets no passages; an empty
   query gives none.
4. Passages are Index.search(query, k) in rank order; one whose text would
   take the running total of passage text past max_chars is skipped whole,
   and a later smaller one may still fit. Each is rendered with passage.txt
   ($path, $heading, $start, $end, $text), the passages joined by one blank
   line, and the block rendered with retrieved.txt ($passages).
5. The block goes before the prompt, after the knowledge block when there
   is one, separated by one blank line. Without the section, or with no
   passage, the prompt is byte for byte as without it.
6. Each request that carried passages gets one Diagnostic, code `retrieved`,
   whose message lists each passage carried as <path>:<start>-<end> in rank
   order with its score. The record's severities are error, warning, and
   note, so the design's informational severity is `note`; the tests accept
   the note on the attempt the request produced or on the request itself,
   both of which the record already holds. The note never enters a prompt.
7. tt-host-v0 gains retrieved.txt (naming the tt-metal commit 5280a9cf) and
   passage.txt, plain ASCII and naming no Tier A item, kernel, or file.
8. describe() of generate, compile_loop, and run_loop mentions the passages
   exactly when retrieval is on for that stage's prompt.

Where the design leaves a detail open, the tests read behavior only: the
template wording comes from the template files (lassi.prompts.render); the
queries and the passages each prompt got are read from Index.search, which a
spy wraps; a block is compared as described in tests/core/test_template_context.py
(a block without its trailing line feeds, one blank line, the prompt), and
$passages may or may not end in a line feed. Two choices the design leaves to
the tooling are fixed here as the generator fixes them: the corpus directory
is <assets root>/rag/<index>, under lassi.prompts.assets.default_root(), the
root prompt sets and packs come from; and the index file is where
tools/build_tt_rag.py writes it by default,
$LASSI_SCRATCH/rag/<corpus>@<VERSION of the manifest's source pin>/index.json.

Every corpus, index, manifest, program, reply, and array is SYNTHETIC, built
in a temporary directory with lassi.prompts.retrieval.index_bytes; the
generated tt-metal-docs index is never read. No value here is a measurement.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_template_context import (
    BARE_ERROR,
    BROKEN,
    CONTEXT_STAGES,
    CPP,
    ERROR,
    ITEM,
    KERNEL,
    KERNEL_TEXT,
    PACK,
    PROMPTS,
    SOURCE_TEXT,
    SUMMARY,
    SYN_TEMPLATES,
    TT,
    FakeToolchain,
    Script,
    crashed,
    directive,
    doubled,
    full_script,
    joined,
    knowledge_block,
    load,
    make_assets,
    named_tier_a,
    program,
    recipe_data,
    refused,
    run,
    scripted_executor,
    stage_context,
    template_set,
    template_text,
    user_prompts,
    write_suite,
    write_tree,
)

from lassi.bench import Direction
from lassi.core import fragments
from lassi.core.files import parse_file_blocks, render_file_blocks
from lassi.core.interfaces import Limits
from lassi.core.recipe import Recipe, RecipeError
from lassi.core.record import Diagnostic, Request, Trial
from lassi.core.stages import CompileLoopStage, GenerateStage, RunLoopStage, prompt_diagnostics, run_error_text
from lassi.prompts import rag_corpus, render, retrieval
from lassi.prompts.retrieval import INDEX_FORMAT, TOKENIZER, Chunk, Passage, index_bytes, load_index
from lassi.toolchains.pins import read_pin

GENERATOR = "tools/build_tt_rag.py"
SOURCE_PIN = "tt-metal"
PIN = read_pin(SOURCE_PIN)
BM25 = {"k1": 1.2, "b": 0.75}
CHUNKING = {"max_lines": 40, "max_chars": 3000}
RETRIEVED = "retrieved"
# The section the tests turn on, and what the search returns before max_chars when it does not bound.
CORPUS = "syn-docs"
SECTION = {"index": CORPUS, "k": 2, "max_chars": 8000, "prompts": ["generate", "correct"]}
# The search's own function, kept before any test wraps it.
SEARCH = retrieval.Index.search

# The SYNTHETIC documentation: no chunk holds the tokens scale, cpp, qqxv, or wvzz, so a query of only those
# finds nothing. Each text holds one line per line of its range.
MAIN_CHUNKS = (
    Chunk("docs/circular_buffers.md", 1, 2, "Circular buffers",
          "cb_wait_front waits until a circular buffer holds tiles.\n"
          "CreateCircularBuffer(program, core, config) makes a circular buffer."),
    Chunk("docs/circular_buffers.md", 3, 4, "Circular buffers > Config",
          "CircularBufferConfig sets the page size of a circular buffer.\n"
          "Call CreateCircularBuffer with it before the kernels run."),
    Chunk("docs/mesh_buffers.md", 1, 2, "Mesh buffers",
          "EnqueueWriteMeshBuffer(queue, buffer, data, false) writes a mesh buffer.\n"
          "It lives in the namespace tt::tt_metal::distributed."),
    Chunk("docs/mesh_buffers.md", 3, 3, "Mesh buffers > Reading",
          "EnqueueReadMeshBuffer(queue, data, buffer, true) reads a mesh buffer back."),
    Chunk("docs/kernels.md", 1, 2, "Kernels",
          "CreateKernel(program, path, core, config) adds a kernel to a program.\n"
          "kernel_main is the entry point of every kernel."),
    Chunk("docs/kernels.md", 3, 4, "Kernels > Errors",
          "A TT_FATAL from CreateKernel means the kernel path was not found.\n"
          "Check the path given to CreateKernel."),
    Chunk("docs/tiles.md", 1, 1, "Tiles", "zorbquant_tile multiplies each element of a tile by a constant."),
    Chunk("docs/host.md", 1, 1, "Host programs", "A host program opens the device, then creates the buffers."),
)

# A corpus of three passages that the item's kernel alone matches, ranked largest-first by BM25 but not by size:
# 300, 400, then 150 characters of text, for a max_chars of 500 (the search's order is checked in the test).
SIZED = "syn-sized"
SIZED_MAX_CHARS = 500
_FILLER = "lorem ipsum dolor sit amet consectetur adipiscing elit sed eiusmod tempor incididunt ut labore et dolore "


def sized_text(size: int, count: int) -> str:
    """Return a one-line SYNTHETIC text of exactly `size` characters holding zorbquant_tile `count` times."""
    return (("zorbquant_tile " * count) + _FILLER * 20)[:size]


SIZED_CHUNKS = (
    Chunk("notes/alpha.md", 1, 1, "Alpha", sized_text(300, 6)),
    Chunk("notes/beta.md", 1, 1, "Beta", sized_text(400, 4)),
    Chunk("notes/gamma.md", 1, 1, "Gamma", sized_text(150, 1)),
)
SIZED_SECTION = {"index": SIZED, "k": 3, "max_chars": SIZED_MAX_CHARS, "prompts": ["generate"]}

# A synthetic fragment set holding every key generate reads for cpp -> tt.
FRAGMENT_SET = "syn-fragments"
FRAGMENTS = {fragments.fragment_key(key, Direction(CPP, TT)): f"<{key}>" for key in fragments.GENERATE_KEYS}
# Each retrieval template of tt-host-v0 and the fields it renders with.
RETRIEVAL_TEMPLATES = {"retrieved": ("passages",), "passage": ("path", "heading", "start", "end", "text")}
# The limits a run-error text is read with; the scripted runs never hang, so only the exit status shows.
LIMITS = Limits(wall_s=30.0, memory_mb=16384, cpus=1)


@dataclass(frozen=True)
class Search:
    """One Index.search call a run made: the index object's id, the query, k, and the passages returned."""

    index: int
    query: str
    k: int
    found: tuple[Passage, ...]


def index_path(scratch: Path, corpus: str) -> Path:
    """Return where tools/build_tt_rag.py writes a corpus's index by default, under `scratch` as LASSI_SCRATCH."""
    return scratch / "rag" / f"{corpus}@{PIN['VERSION']}" / "index.json"


def write_corpus(
    assets: Path, scratch: Path, corpus: str, chunks: Sequence[Chunk], *, sha256: str | None = None,
    write_index: bool = True,
) -> Path:
    """Write a SYNTHETIC corpus: its index (unless not `write_index`) and its manifest; return the index path.

    The manifest holds the keys tools/build_tt_rag.py writes, with the
    index's own sha256 unless `sha256` replaces it, and an empty list for
    any other top-level key lassi.prompts.rag_corpus declares (its
    MANIFEST_KEYS), for which this corpus has no rows.
    """
    meta = {"corpus": corpus, "corpus_sha256": hashlib.sha256(b"SYNTHETIC corpus spec\n").hexdigest(),
            "source": {"repo": PIN["URL"], "commit": PIN["COMMIT"]}, "chunking": dict(CHUNKING), "bm25": dict(BM25)}
    data = index_bytes(chunks, meta)
    path = index_path(scratch, corpus)
    if write_index:
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
    paths = sorted({chunk.path for chunk in chunks})
    document = {
        "corpus": corpus, "generator": GENERATOR, "corpus_sha256": meta["corpus_sha256"],
        "source": {"pin": SOURCE_PIN, "repo": PIN["URL"], "commit": PIN["COMMIT"]},
        "index": {"format": INDEX_FORMAT, "tokenizer": TOKENIZER, "sha256": sha256 or hashlib.sha256(data).hexdigest(),
                  "bytes": len(data), "files": len(paths), "chunks": len(chunks)},
        "chunking": dict(CHUNKING), "bm25": dict(BM25),
        "tier_a": {"suite": "assets/bench/tt-pairs-v0.yaml", "examples": "tt_metal/programming_examples",
                   "run_lines": 3, "max_file_overlap": 0.1, "dirs": [], "references": []},
        "files": [{"path": name, "sha256": hashlib.sha256(name.encode("ascii")).hexdigest(),
                   "chunks": sum(chunk.path == name for chunk in chunks)} for name in paths],
        "excluded": [], "excised": [],
    }
    for key in sorted(set(getattr(rag_corpus, "MANIFEST_KEYS", ())) - set(document)):
        document[key] = []
    manifest = assets / "rag" / corpus / "MANIFEST.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(yaml.safe_dump(document, sort_keys=False).encode("ascii"))
    return path


def kept(found: Sequence[Passage], max_chars: int) -> list[Passage]:
    """Return the passages a prompt carries: in rank order, each skipped whole when it would pass `max_chars`."""
    carried: list[Passage] = []
    total = 0
    for passage in found:
        if total + len(passage.text) > max_chars:
            continue
        carried.append(passage)
        total += len(passage.text)
    return carried


def passage_text(passage: Passage) -> str:
    """Return the prompt set's passage.txt rendered with one passage's fields."""
    fields = {"path": passage.path, "heading": passage.heading, "start": str(passage.start_line),
              "end": str(passage.end_line), "text": passage.text}
    return render(PROMPTS, "passage", fields)


def retrieved_blocks(passages: Sequence[Passage]) -> set[str]:
    """Return the retrieved block for `passages`: passages one blank line apart, with or without a last line feed."""
    text = "\n\n".join(passage_text(passage).rstrip("\n") for passage in passages)
    return {render(PROMPTS, RETRIEVED, {"passages": value}) for value in (text, text + "\n")}


def carried(prompt: str, passages: Sequence[Passage], knowledge: str | None = None) -> set[str]:
    """Return the forms `prompt` may take with `passages` (none: no block), after the knowledge block when given."""
    before = [] if knowledge is None else [knowledge]
    if not passages:
        return {joined(before, prompt)}
    return {joined([*before, block], prompt) for block in retrieved_blocks(passages)}


def retrieved_notes(trial: Trial, request: Request) -> list[Diagnostic]:
    """Return the `retrieved` diagnostics of a request: on the request itself and on the attempt it produced."""
    found = [*request.diagnostics]
    if request.attempt_index is not None:
        found += trial.attempts[request.attempt_index].diagnostics
    return [item for item in found if item.code == RETRIEVED]


def shows_score(segment: str, score: float) -> bool:
    """Return True when `segment` holds a number that is `score` to the places it is written with."""
    for number in re.findall(r"-?\d+(?:\.\d+)?", segment):
        places = len(number.partition(".")[2])
        if abs(float(number) - score) <= 0.5 * 10.0**-places + 1e-12:
            return True
    return False


def check_listing(message: str, passages: Sequence[Passage]) -> None:
    """Check that `message` names each passage as <path>:<start>-<end> in order, each followed by its score."""
    locators = [f"{passage.path}:{passage.start_line}-{passage.end_line}" for passage in passages]
    position = 0
    for number, (locator, passage) in enumerate(zip(locators, passages, strict=True)):
        found = message.find(locator, position)
        assert found >= 0, f"{locator} is not listed in rank order: {message!r}"
        position = found + len(locator)
        end = message.find(locators[number + 1], position) if number + 1 < len(locators) else len(message)
        assert shows_score(message[position:end], passage.score), f"{locator} has no score {passage.score}: {message!r}"


def main_index(base: Path) -> retrieval.Index:
    """Return the main SYNTHETIC index, loaded in the test."""
    return load_index(index_path(base, CORPUS))


def section(**changes: Any) -> dict[str, Any]:
    """Return SECTION with `changes`."""
    return {**SECTION, **changes}


def broken_errors() -> list[Diagnostic]:
    """Return the error-severity diagnostics the fake toolchain gives for BROKEN, in the order it gives them."""
    text = parse_file_blocks(BROKEN, [f"{ITEM}.cpp"]).files[f"{ITEM}.cpp"]
    found = [directive(f"{ITEM}.cpp", number, line) for number, line in enumerate(text.splitlines(), start=1)]
    return [item for item in found if item is not None and item.severity == "error"]


@pytest.fixture(autouse=True)
def scratch_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Set LASSI_SCRATCH to the test's scratch directory (the index lies under it), TMPDIR inside it."""
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
    """Return the temporary assets root with tt-host-v0, the pack, and the two SYNTHETIC corpora."""
    root = make_assets(base, monkeypatch)
    write_corpus(root, base, CORPUS, MAIN_CHUNKS)
    write_corpus(root, base, SIZED, SIZED_CHUNKS)
    return root


@pytest.fixture
def bench(base: Path, monkeypatch: pytest.MonkeyPatch, assets: Path) -> Path:
    """Return the bench root of the SYNTHETIC suite, with the assets root in place."""
    return write_suite(base, monkeypatch)


@pytest.fixture
def searches(monkeypatch: pytest.MonkeyPatch) -> list[Search]:
    """Wrap Index.search so every call a run makes is kept, in order, with what it returned."""
    calls: list[Search] = []

    def spy(self: retrieval.Index, query: str, k: int) -> list[Passage]:
        found = SEARCH(self, query, k)
        calls.append(Search(id(self), query, k, tuple(found)))
        return found

    monkeypatch.setattr(retrieval.Index, "search", spy)
    return calls


def full_pair(base: Path, bench: Path, data: Mapping[str, Any], *prefix: str) -> tuple[list[str], list[str], Trial]:
    """Run the four-prompt script without the section and with `data`; return both prompt lists and the trial."""
    plain = full_script()
    run(base, bench, plain, recipe_data(), "plain")
    script = full_script(*prefix)
    _, trial = run(base, bench, script, data, "retrieving")
    return user_prompts(plain), user_prompts(script), trial


# ---------------------------------------------------------------------------
# The recipe section (design point 1)


def load_refusal(base: Path, data: Mapping[str, Any]) -> str:
    """Load `data`, expect a RecipeError from a loader that knows the section, and return its message."""
    with pytest.raises(RecipeError) as caught:
        load(base, data)
    message = str(caught.value)
    assert "unknown key retrieval;" not in message, f"the loader does not know the section: {message}"
    return message


def test_the_section_loads_as_written_and_enters_the_hash(base: Path) -> None:
    (base / "plain").mkdir()
    (base / "with").mkdir()
    plain = load(base / "plain", recipe_data(), "recipe")
    with_section = load(base / "with", recipe_data(retrieval=SECTION), "recipe")
    assert "retrieval" not in plain.data and "retrieval" not in plain.canonical_yaml, "absent, nothing materializes"
    assert with_section.data["retrieval"] == SECTION
    assert with_section.recipe_hash != plain.recipe_hash, "a section that is set enters the hash"


@pytest.mark.parametrize("key", sorted(SECTION))
def test_a_missing_key_is_refused_naming_it(base: Path, key: str) -> None:
    partial = {name: value for name, value in SECTION.items() if name != key}
    assert f"retrieval.{key}" in load_refusal(base, recipe_data(retrieval=partial))


def test_an_unknown_key_is_refused_naming_it(base: Path) -> None:
    assert "retrieval.extra" in load_refusal(base, recipe_data(retrieval=section(extra=1)))


@pytest.mark.parametrize("value", [["generate"], CORPUS, 4, None], ids=["list", "string", "int", "null"])
def test_a_section_that_is_not_a_mapping_is_refused(base: Path, value: Any) -> None:
    assert "retrieval" in load_refusal(base, recipe_data(retrieval=value))


BAD_VALUES = [
    ("k", 0), ("k", 17), ("k", True), ("k", "4"), ("k", 4.0),
    ("max_chars", 499), ("max_chars", 32001), ("max_chars", True), ("max_chars", "8000"), ("max_chars", 8000.5),
    ("prompts", []), ("prompts", ["generate", "generate"]), ("prompts", ["fix"]), ("prompts", "generate"),
    ("prompts", [1]), ("prompts", {"generate": True}),
    ("index", 5), ("index", True), ("index", [CORPUS]),
]


@pytest.mark.parametrize(("key", "value"), BAD_VALUES, ids=[f"{key}-{value!r}" for key, value in BAD_VALUES])
def test_a_value_of_the_wrong_type_or_range_is_refused_naming_its_key(base: Path, key: str, value: Any) -> None:
    assert f"retrieval.{key}" in load_refusal(base, recipe_data(retrieval=section(**{key: value})))


GOOD_VALUES = [("k", 1), ("k", 16), ("max_chars", 500), ("max_chars", 32000), ("prompts", ["correct"]),
               ("prompts", ["generate"]), ("prompts", ["correct", "generate"])]


@pytest.mark.parametrize(("key", "value"), GOOD_VALUES, ids=[f"{key}-{value!r}" for key, value in GOOD_VALUES])
def test_the_bounds_and_subsets_load(base: Path, key: str, value: Any) -> None:
    assert load(base, recipe_data(retrieval=section(**{key: value}))).data["retrieval"][key] == value


def test_the_section_is_refused_under_faithful(base: Path) -> None:
    message = load_refusal(base, recipe_data(retrieval=SECTION, faithful=True, loop={}))
    assert "retrieval" in message and "faithful: true" in message and "loaded.yaml" in message, message


def test_an_inherited_section_is_refused_when_the_child_turns_faithful_on(base: Path) -> None:
    load(base, recipe_data(retrieval=SECTION, loop={}), "retrieval-parent")
    message = load_refusal(base, {"extends": "retrieval-parent.yaml", "faithful": True})
    assert "retrieval" in message and "faithful: true" in message, message


# ---------------------------------------------------------------------------
# The runner's refusals and the one index (design point 2)


def fragment_recipe(**changes: Any) -> dict[str, Any]:
    """Return a generate-only recipe on the SYNTHETIC fragment set; `changes` replace keys."""
    data = recipe_data(prompts=FRAGMENT_SET, stages=["generate"], loop={}, **changes)
    data.pop("oracle")
    return data


def test_the_runner_refuses_the_section_with_a_fragment_prompt_set(base: Path, bench: Path, assets: Path) -> None:
    write_tree(assets / "prompts" / FRAGMENT_SET, FRAGMENTS)
    message = refused(base, bench, fragment_recipe(retrieval=SECTION))
    assert "retrieval" in message and "fragment" in message, message
    script = Script(replies=[program(0)], attempts=[])
    _, trial = run(base, bench, script, fragment_recipe(), "upstream")
    assert [request.stage for request in trial.requests or []] == ["generate"], "without it the same recipe runs"


@pytest.mark.parametrize("missing", ["retrieved", "passage"])
def test_the_runner_refuses_a_template_set_without_a_retrieval_template(
    base: Path, bench: Path, assets: Path, missing: str
) -> None:
    other = "passage" if missing == "retrieved" else "retrieved"
    name = template_set(assets, f"lacks-{missing}", drop=[missing], texts={other: SYN_TEMPLATES[other]})
    message = refused(base, bench, recipe_data(prompts=name, retrieval=SECTION))
    assert f"{missing}.txt" in message, message


@pytest.mark.parametrize("made", [False, True], ids=["no-directory", "no-manifest"])
def test_the_runner_refuses_an_index_with_no_manifest(base: Path, bench: Path, assets: Path, made: bool) -> None:
    if made:
        (assets / "rag" / "empty-docs").mkdir(parents=True)
    message = refused(base, bench, recipe_data(retrieval=section(index="empty-docs")))
    assert "MANIFEST.yaml" in message and "empty-docs" in message, message


def test_the_runner_refuses_a_missing_index_file(base: Path, bench: Path, assets: Path) -> None:
    write_corpus(assets, base, "lost-docs", MAIN_CHUNKS, write_index=False)
    message = refused(base, bench, recipe_data(retrieval=section(index="lost-docs")))
    assert GENERATOR in message and "index" in message, message


def test_the_runner_refuses_an_index_that_fails_its_sha256(base: Path, bench: Path, assets: Path) -> None:
    write_corpus(assets, base, "changed-docs", MAIN_CHUNKS, sha256="0" * 64)
    message = refused(base, bench, recipe_data(retrieval=section(index="changed-docs")))
    assert GENERATOR in message and "sha256" in message, message


def test_the_index_loads_once_per_run_and_every_trial_searches_it(
    base: Path, bench: Path, searches: list[Search], monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[int] = []
    construct = retrieval.Index.__init__

    def counting(self: retrieval.Index, chunks: Sequence[Chunk], meta: Mapping[str, Any]) -> None:
        built.append(id(self))
        construct(self, chunks, meta)

    monkeypatch.setattr(retrieval.Index, "__init__", counting)
    script = Script(replies=[program(0), program(0)], attempts=[doubled, doubled])
    run(base, bench, script, recipe_data(retrieval=SECTION, loop={}, trials={"n": 2}), "twice")
    assert len(built) == 1, "the index loads once per run, not once per trial"
    assert len(searches) == 2 and {call.index for call in searches} == {built[0]}, "both trials search that index"


# ---------------------------------------------------------------------------
# The queries (design point 3)


def test_the_generate_query_is_the_kernels_then_the_sources(base: Path, bench: Path, searches: list[Search]) -> None:
    script = Script(replies=[program(0)], attempts=[doubled])
    run(base, bench, script, recipe_data(retrieval=SECTION, loop={}), "query")
    (call,) = searches
    kernels = render_file_blocks({KERNEL: KERNEL_TEXT})
    sources = {f"{ITEM}.cpp": SOURCE_TEXT}
    assert call.query in {f"{kernels}\n{render_file_blocks(sources)}", f"{kernels}\n{SOURCE_TEXT}"}, call.query
    assert call.k == SECTION["k"]
    prompt = user_prompts(script)[0]
    assert all(line in prompt for line in call.query.splitlines()), "the query holds only what the prompt shows"


def test_a_compile_correction_queries_the_errors_it_lists(base: Path, bench: Path, searches: list[Search]) -> None:
    script = Script(replies=[BROKEN, program(1)], attempts=[doubled])
    _, trial = run(base, bench, script, recipe_data(retrieval=SECTION, loop={}), "compiling")
    errors = broken_errors()
    listed = [item for item in prompt_diagnostics(trial.attempts[0].diagnostics) if item.severity == "error"]
    assert listed == errors and len(errors) == 2, "the fixture's two errors, around one warning"
    assert [request.stage for request in trial.requests or []] == ["generate", "compile_loop"]
    assert len(searches) == 2
    assert searches[1].query == "\n".join(retrieval.query_from_diagnostic(item) for item in errors)


def test_a_run_error_correction_queries_the_run_error_text(base: Path, bench: Path, searches: list[Search]) -> None:
    script = Script(replies=[program(0), program(1)], attempts=[crashed, doubled])
    _, trial = run(base, bench, script, recipe_data(retrieval=SECTION, loop={}), "running")
    assert [request.stage for request in trial.requests or []] == ["generate", "run_loop"]
    shown = run_error_text(script.runs[0], LIMITS, {})
    assert shown in user_prompts(script)[1], "the correction shows the run error text"
    assert len(searches) == 2 and searches[1].query == shown


@pytest.mark.parametrize("line", [f"{ERROR}qqxv wvzz", BARE_ERROR], ids=["no-match", "empty-query"])
def test_a_correction_with_no_passage_is_unchanged(
    base: Path, bench: Path, searches: list[Search], line: str
) -> None:
    replies = [program(0, line), program(1)]
    plain = Script(replies=list(replies), attempts=[doubled])
    run(base, bench, plain, recipe_data(loop={}), "plain")
    script = Script(replies=list(replies), attempts=[doubled])
    _, trial = run(base, bench, script, recipe_data(retrieval=section(prompts=["correct"]), loop={}), "nothing")
    (error,) = [item for item in trial.attempts[0].diagnostics if item.severity == "error"]
    query = retrieval.query_from_diagnostic(error)
    assert SEARCH(main_index(base), query, 16) == [], f"the fixture's query {query!r} matches no chunk"
    assert user_prompts(script) == user_prompts(plain), "no passage, no block"
    assert all(retrieved_notes(trial, request) == [] for request in trial.requests or [])


def test_a_mismatch_correction_gets_no_passages(base: Path, bench: Path, searches: list[Search]) -> None:
    plain, prompts, trial = full_pair(base, bench, recipe_data(retrieval=SECTION))
    assert [request.stage for request in trial.requests or []] == ["generate", "compile_loop", "run_loop", "run_loop"]
    assert prompts[3] == plain[3], "the oracle-mismatch correction is unchanged"
    assert retrieved_notes(trial, (trial.requests or [])[3]) == []


# ---------------------------------------------------------------------------
# The passages and where they go (design points 4 and 5)


def test_k_passages_in_rank_order_go_before_each_prompt(base: Path, bench: Path, searches: list[Search]) -> None:
    plain, prompts, _ = full_pair(base, bench, recipe_data(retrieval=SECTION))
    index = main_index(base)
    assert len(searches) >= 3
    for number, call in enumerate(searches[:3]):
        assert call.k == SECTION["k"]
        assert len(SEARCH(index, call.query, 16)) > SECTION["k"], "the fixture has more hits than k"
        assert list(call.found) == SEARCH(index, call.query, SECTION["k"])
        assert prompts[number] in carried(plain[number], kept(call.found, SECTION["max_chars"])), prompts[number]


def test_max_chars_skips_a_passage_whole_and_keeps_a_later_smaller_one(
    base: Path, bench: Path, searches: list[Search]
) -> None:
    plain = Script(replies=[program(0)], attempts=[doubled])
    run(base, bench, plain, recipe_data(loop={}), "plain")
    script = Script(replies=[program(0)], attempts=[doubled])
    _, trial = run(base, bench, script, recipe_data(retrieval=SIZED_SECTION, loop={}), "sized")
    (call,) = searches
    assert [len(passage.text) for passage in call.found] == [300, 400, 150], "the fixture's rank order"
    first, second, third = call.found
    prompt = user_prompts(script)[0]
    assert prompt in carried(user_prompts(plain)[0], [first, third]), prompt
    assert second.text not in prompt
    (note,) = retrieved_notes(trial, (trial.requests or [])[0])
    check_listing(note.message, [first, third])
    assert f"{second.path}:" not in note.message, "a skipped passage is not listed"


def test_the_passages_go_after_the_knowledge_block(base: Path, bench: Path, searches: list[Search]) -> None:
    data = recipe_data(context=[PACK], stages=CONTEXT_STAGES, retrieval=SECTION)
    plain, prompts, _ = full_pair(base, bench, data, SUMMARY)
    knowledge = knowledge_block()
    assert len(prompts) == 5 and len(searches) >= 3
    for number, call in enumerate(searches[:3]):
        expected = carried(plain[number], kept(call.found, SECTION["max_chars"]), knowledge)
        assert prompts[number + 1] in expected, prompts[number + 1]
    assert prompts[4] == joined([knowledge], plain[3]), "the mismatch correction carries the summary alone"


@pytest.mark.parametrize("listed", [["generate"], ["correct"]])
def test_only_the_listed_prompts_carry_passages(
    base: Path, bench: Path, searches: list[Search], listed: list[str]
) -> None:
    plain, prompts, _ = full_pair(base, bench, recipe_data(retrieval=section(prompts=listed)))
    carrying = [0] if listed == ["generate"] else [1, 2]
    calls = iter(searches)
    for number in range(4):
        if number not in carrying:
            assert prompts[number] == plain[number], f"prompt {number} is unchanged"
            continue
        call = next(calls)
        assert prompts[number] in carried(plain[number], kept(call.found, SECTION["max_chars"])), prompts[number]


# ---------------------------------------------------------------------------
# The record (design point 6)


def test_each_request_with_passages_records_one_retrieved_note(
    base: Path, bench: Path, searches: list[Search]
) -> None:
    _, prompts, trial = full_pair(base, bench, recipe_data(retrieval=SECTION))
    requests = trial.requests or []
    assert len(requests) == 4 and len(searches) >= 3
    for request, call in zip(requests[:3], searches[:3], strict=True):
        (note,) = retrieved_notes(trial, request)
        assert note.severity == "note", "the record's informational severity"
        check_listing(note.message, kept(call.found, SECTION["max_chars"]))
    assert retrieved_notes(trial, requests[3]) == [], "the mismatch correction carried none"
    assert not any(f"[{RETRIEVED}]" in prompt for prompt in prompts), "the note never enters a prompt"


# ---------------------------------------------------------------------------
# The two templates of tt-host-v0 (design point 7)


@pytest.mark.parametrize("name", sorted(RETRIEVAL_TEMPLATES))
def test_tt_host_v0_holds_the_retrieval_template_and_it_renders_with_its_fields(name: str) -> None:
    template_text(name)
    fields = {field_name: f"<SYNTHETIC {field_name}>" for field_name in RETRIEVAL_TEMPLATES[name]}
    rendered = render(PROMPTS, name, fields)
    for field_name, value in fields.items():
        assert value in rendered, f"{name}.txt does not show ${field_name}"


@pytest.mark.parametrize("name", sorted(RETRIEVAL_TEMPLATES))
def test_tt_host_v0_retrieval_template_is_generic(name: str) -> None:
    text = template_text(name)
    assert named_tier_a(text) == [], f"{name}.txt names a Tier A item or file (Agent Rule 5)"
    if name == RETRIEVED:
        assert PIN["VERSION"] in text, "the passages are introduced as tt-metal documentation at its commit"


# ---------------------------------------------------------------------------
# describe() (design point 8)


def describes(recipe: Recipe) -> dict[str, str]:
    """Return describe() of generate, compile_loop, and run_loop for `recipe` on tt-host-v0."""
    executor = scripted_executor(Script(replies=[], attempts=[]))()
    context = stage_context(recipe, PROMPTS, toolchains={CPP: FakeToolchain(), TT: FakeToolchain()}, executor=executor)
    return {stage.name: stage(context=context).describe() for stage in (GenerateStage, CompileLoopStage, RunLoopStage)}


@pytest.mark.parametrize("listed", [["generate"], ["correct"], ["generate", "correct"]])
def test_describe_mentions_the_passages_only_where_they_go(base: Path, listed: list[str]) -> None:
    (base / "plain").mkdir()
    (base / "on").mkdir()
    plain = describes(load(base / "plain", recipe_data(loop={}), "recipe"))
    on = describes(load(base / "on", recipe_data(loop={}, retrieval=section(prompts=listed)), "recipe"))
    for stage, prompt in (("generate", "generate"), ("compile_loop", "correct"), ("run_loop", "correct")):
        if prompt in listed:
            assert "passage" in on[stage].lower(), on[stage]
        else:
            assert on[stage] == plain[stage], f"{stage} has no passages, so its description is unchanged"
