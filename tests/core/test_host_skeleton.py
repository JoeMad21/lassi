"""Tests for the host program skeleton of the cpp -> tt prompts (opt-in; OQ-096, option (a), 2026-10-25).

Bible: Component Interfaces (the context blocks under a template set), Project Recipes (Notes, the Skeleton
bullet, beside the Examples bullet), Result Record (Trial.requests), Harness Contract (a prompt shows only what
the trial may see), Agent Rules 1, 4 (a departure from upstream is opt-in and refused under faithful: true), 5 (no
tt-pairs-v0 program in a skeleton), and 10 (the bytes a run reads are the bytes its recipe pins). The evidence:
in the C++ -> TT trials of 2026-10-25, 79 of 82 ended at the correction cap at S1, a build error
(results/rngd-matrix-20261025/summary.md). In the 134 cpp -> tt trials read before that window (the runs of
2026-10-20 to 2026-10-24, and 11 earlier mock-reference trials with no compile error), some build-error bucket
recurs across the trial's attempts in 121 (results/build-errors-20261025/summary.md); that the buckets are
API-shape mistakes against tt-metal 5280a9cf is a reading, not measured, since the report's masks hide every
identifier. tt-pairs-v0 is a development set (OQ-083, option (b)).

The contract these tests fix:

1. An optional run recipe section `skeleton`, as in
   `skeleton: {name: tt-host-skeleton-v0, prompts: [generate, correct]}`, with two required keys: name, a plain
   name (a directory under assets/skeletons/); and prompts, a list of distinct names among {generate, correct}
   that holds generate; and one optional key, sha256, the skeleton's sha256. A section that is not a mapping, a
   null, an unknown or missing key, and a value of the wrong type are refused at load naming `skeleton.<key>`;
   the section is refused at load under faithful: true, inherited or not, naming `skeleton`.
2. The loader validates the skeleton by name (lassi.prompts.skeletons): the directory must exist and hold
   MANIFEST.yaml, host.cpp, and block.txt, every file matching the manifest's sha256 and size and the manifest's
   sha256 that of its entries, and block.txt must render with its one field; otherwise it refuses naming
   `skeleton.name`. Its sha256 enters the resolved recipe as skeleton.sha256 and so the recipe hash; a recipe
   that states another is refused naming skeleton.sha256. A direction whose target is not the skeleton's
   language is refused naming the target. Absent, nothing is materialized, so a recipe keeps its resolved recipe
   and hash.
3. The runner refuses the section with a fragment prompt set, and a skeleton that changed after the recipe loaded
   (it no longer loads, or its sha256 is not the recipe's skeleton.sha256) naming `skeleton.name`, before any
   directory exists.
4. The block: block.txt with $skeleton the skeleton's host.cpp, the last block before the generation prompt and,
   when prompts lists correct, before each compile correction: after the knowledge, examples, retrieved,
   looked-up, and core blocks, one blank line apart. A run-error correction and an oracle-mismatch correction get
   none; with prompts [generate] no correction gets one.
5. Each request that carried the skeleton gets exactly one Diagnostic with code `skeleton`, severity note, stage
   parse, on its Request (never on an attempt), naming the skeleton and its sha256; it never enters a prompt.
6. generate's and compile_loop's describe() name the skeleton when the section lists their prompt.

Every skeleton, program, reply, and array here is SYNTHETIC, written in a temporary directory, except where a test
copies the repository's tt-host-skeleton-v0 to run its real block. No value here is a measurement.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
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
from test_retrieval_stage import FRAGMENT_SET, FRAGMENTS, fragment_recipe
from test_template_context import (
    BROKEN,
    PROMPT_KINDS,
    PROMPTS,
    RECIPE_HASHES,
    REPO,
    REQUEST_STAGES,
    Script,
    doubled,
    full_script,
    joined,
    load,
    make_assets,
    program,
    recipe_data,
    refused,
    run,
    stage_context,
    user_prompts,
    write_suite,
    write_tree,
)
from test_worked_examples import ALPHA, PACK, ZETA, example_blocks, write_pack

from lassi.core import runner as runner_module
from lassi.core.recipe import Recipe, RecipeError, load_recipe
from lassi.core.record import Diagnostic, Request, Trial
from lassi.core.stages import CompileLoopStage, GenerateStage, prompt_diagnostics

SKELETON = "skeleton"
NAME = "syn-skeleton"
SECTION = {"name": NAME, "prompts": ["generate"]}
BOTH = {**SECTION, "prompts": ["generate", "correct"]}
SKELETON_CODE = "skeleton"
SYN_HOST = (
    "// SYNTHETIC skeleton host program\n"
    "int main() {\n"
    "    // BLANK body: SYNTHETIC placeholder\n"
    "    return 0;\n"
    "    // END BLANK body\n"
    "}\n"
)
SYN_BLOCK = "SYNTHETIC: start from this skeleton and fill only its BLANK blocks.\n\n```cpp\n$skeleton```\n"
REAL = "tt-host-skeleton-v0"
REAL_TREE = REPO / "assets" / "skeletons" / REAL


# ---------------------------------------------------------------------------
# Fixtures and helpers


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate variables a test could inherit, and point TMPDIR at a test directory."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "scratch" / "tmp").mkdir(parents=True)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch" / "tmp"))


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """Return the scratch directory every run and recipe of a test writes under."""
    return tmp_path / "scratch"


@pytest.fixture
def assets(base: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return the temporary assets root: tt-host-v0, the SYNTHETIC packs, and the SYNTHETIC skeleton."""
    root = make_assets(base, monkeypatch)
    write_tree(root / "context" / DECL_PACK, DECL_PACK_ENTRIES)
    write_pack(root, PACK, [ZETA, ALPHA])
    write_skeleton(root, NAME)
    return root


@pytest.fixture
def bench(base: Path, monkeypatch: pytest.MonkeyPatch, assets: Path) -> Path:
    """Return the bench root of the SYNTHETIC suite, with the assets root in place."""
    return write_suite(base, monkeypatch)


def sha256(data: bytes) -> str:
    """Return the sha256 hex digest of `data`."""
    return hashlib.sha256(data).hexdigest()


def write_skeleton(
    assets: Path, name: str, *, host: str = SYN_HOST, block: str = SYN_BLOCK, language: str = "tt",
    digest: str | None = None,
) -> Path:
    """Write a SYNTHETIC skeleton under `assets`/skeletons/`name`: host.cpp, block.txt, MANIFEST.yaml; return it.

    The manifest lists both files with their sha256 and size, and its sha256 is that of the lines
    "<sha256>  <path>\\n" of the files in order, unless `digest` replaces it.
    """
    tree = assets / "skeletons" / name
    tree.mkdir(parents=True)
    entries = []
    for path, text in (("host.cpp", host), ("block.txt", block)):
        data = text.encode("ascii")
        (tree / path).write_bytes(data)
        entries.append({"path": path, "sha256": sha256(data), "bytes": len(data)})
    lines = "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries)
    document = {"skeleton": name, "language": language, "pin": "tt-metal", "commit": "0" * 40, "files": entries,
                "sha256": digest or sha256(lines.encode("ascii"))}
    (tree / "MANIFEST.yaml").write_bytes(yaml.safe_dump(document, sort_keys=False).encode("ascii"))
    return tree


def manifest_sha256(tree: Path) -> str:
    """Return the sha256 a skeleton's MANIFEST.yaml records."""
    return yaml.safe_load((tree / "MANIFEST.yaml").read_bytes().decode("ascii"))["sha256"]


def section(**changes: Any) -> dict[str, Any]:
    """Return SECTION with `changes`."""
    return {**SECTION, **changes}


def skeleton_block(host: str = SYN_HOST, block: str = SYN_BLOCK) -> str:
    """Return the block a prompt carries: `block` with $skeleton replaced by `host`."""
    return block.replace("$skeleton", host)


def skeleton_notes(request: Request) -> list[Diagnostic]:
    """Return the skeleton diagnostics on a request."""
    return [item for item in request.diagnostics if item.code == SKELETON_CODE]


def attempt_notes(trial: Trial) -> list[Diagnostic]:
    """Return every skeleton diagnostic on any attempt of `trial` (there must be none)."""
    return [item for attempt in trial.attempts for item in attempt.diagnostics if item.code == SKELETON_CODE]


def full_pair(base: Path, bench: Path, data: Mapping[str, Any]) -> tuple[list[str], list[str], Trial]:
    """Run the four-prompt script without the section, then with `data`; return both prompt lists and the trial."""
    plain = full_script()
    run(base, bench, plain, recipe_data(), "plain")
    script = full_script()
    _, trial = run(base, bench, script, data, "skeleton")
    assert [request.stage for request in trial.requests or []] == REQUEST_STAGES
    return user_prompts(plain), user_prompts(script), trial


def load_refusal(base: Path, data: Mapping[str, Any]) -> str:
    """Load `data`, expect a RecipeError from a loader that knows the section, and return its message."""
    with pytest.raises(RecipeError) as caught:
        load(base, data)
    message = str(caught.value)
    assert f"unknown key {SKELETON};" not in message, f"the loader does not know the section: {message}"
    return message


# ---------------------------------------------------------------------------
# The recipe section (contract points 1 and 2)


def test_the_section_loads_with_its_sha256_and_enters_the_hash(base: Path, assets: Path) -> None:
    (base / "plain").mkdir()
    (base / "with").mkdir()
    plain = load(base / "plain", recipe_data(), "recipe")
    assert SKELETON not in plain.data and "sha256" not in plain.canonical_yaml, "absent, nothing materializes"
    with_section = load(base / "with", recipe_data(skeleton=SECTION), "recipe")
    digest = manifest_sha256(assets / "skeletons" / NAME)
    assert with_section.data[SKELETON] == {**SECTION, "sha256": digest}, "the skeleton's sha256 enters the recipe"
    assert with_section.recipe_hash != plain.recipe_hash, "a section that is set enters the hash"


def test_a_changed_skeleton_changes_the_recipe_hash(base: Path, assets: Path) -> None:
    write_skeleton(assets, "syn-other", host=SYN_HOST.replace("SYNTHETIC skeleton", "SYNTHETIC other"))
    (base / "one").mkdir()
    (base / "two").mkdir()
    one = load(base / "one", recipe_data(skeleton=SECTION), "recipe")
    two = load(base / "two", recipe_data(skeleton=section(name="syn-other")), "recipe")
    assert one.data[SKELETON]["sha256"] != two.data[SKELETON]["sha256"]
    assert one.recipe_hash != two.recipe_hash


def test_a_stated_sha256_that_matches_loads_and_one_that_differs_is_refused(base: Path, assets: Path) -> None:
    digest = manifest_sha256(assets / "skeletons" / NAME)
    assert load(base, recipe_data(skeleton=section(sha256=digest))).data[SKELETON]["sha256"] == digest
    message = load_refusal(base, recipe_data(skeleton=section(sha256="f" * 64)))
    assert f"{SKELETON}.sha256" in message and digest in message, message


@pytest.mark.parametrize("key", sorted(SECTION))
def test_a_missing_key_is_refused_naming_it(base: Path, assets: Path, key: str) -> None:
    partial = {name: value for name, value in SECTION.items() if name != key}
    assert f"{SKELETON}.{key}" in load_refusal(base, recipe_data(skeleton=partial))


def test_an_unknown_key_is_refused_naming_it(base: Path, assets: Path) -> None:
    assert f"{SKELETON}.extra" in load_refusal(base, recipe_data(skeleton=section(extra=1)))


@pytest.mark.parametrize("value", [["generate"], NAME, 5, None], ids=["list", "string", "int", "null"])
def test_a_section_that_is_not_a_mapping_is_refused(base: Path, assets: Path, value: Any) -> None:
    assert SKELETON in load_refusal(base, recipe_data(skeleton=value))


BAD_VALUES = [
    ("name", ""), ("name", "../syn-skeleton"), ("name", "a/b"), ("name", 5), ("name", None),
    ("prompts", []), ("prompts", ["correct"]), ("prompts", ["generate", "generate"]), ("prompts", ["fix"]),
    ("prompts", "generate"), ("prompts", [1]), ("prompts", {"generate": True}),
    ("sha256", "abc"), ("sha256", 5), ("sha256", "F" * 64),
]


@pytest.mark.parametrize(("key", "value"), BAD_VALUES, ids=[f"{key}-{value!r}" for key, value in BAD_VALUES])
def test_a_value_of_the_wrong_type_is_refused_naming_its_key(base: Path, assets: Path, key: str, value: Any) -> None:
    assert f"{SKELETON}.{key}" in load_refusal(base, recipe_data(skeleton=section(**{key: value})))


@pytest.mark.parametrize("prompts", [["generate"], ["generate", "correct"], ["correct", "generate"]])
def test_both_prompt_lists_load(base: Path, assets: Path, prompts: list[str]) -> None:
    assert load(base, recipe_data(skeleton=section(prompts=prompts))).data[SKELETON]["prompts"] == prompts


def test_the_section_is_refused_under_faithful(base: Path, assets: Path) -> None:
    message = load_refusal(base, recipe_data(skeleton=SECTION, faithful=True, loop={}))
    assert SKELETON in message and "faithful: true" in message and "loaded.yaml" in message, message


def test_an_inherited_section_is_refused_when_the_child_turns_faithful_on(base: Path, assets: Path) -> None:
    load(base, recipe_data(skeleton=SECTION, loop={}), "skeleton-parent")
    message = load_refusal(base, {"extends": "skeleton-parent.yaml", "faithful": True})
    assert SKELETON in message and "faithful: true" in message, message


@pytest.mark.parametrize("made", [False, True], ids=["no-directory", "no-manifest"])
def test_a_name_with_no_skeleton_is_refused(base: Path, assets: Path, made: bool) -> None:
    if made:
        (assets / "skeletons" / "syn-empty").mkdir(parents=True)
    message = load_refusal(base, recipe_data(skeleton=section(name="syn-empty")))
    assert f"{SKELETON}.name" in message and "MANIFEST.yaml" in message, message


@pytest.mark.parametrize("missing", ["host.cpp", "block.txt"])
def test_a_skeleton_missing_a_file_is_refused(base: Path, assets: Path, missing: str) -> None:
    (assets / "skeletons" / NAME / missing).unlink()
    message = load_refusal(base, recipe_data(skeleton=SECTION))
    assert f"{SKELETON}.name" in message and missing in message, message


@pytest.mark.parametrize("changed", ["host.cpp", "block.txt"])
@pytest.mark.parametrize("change", ["same-size", "longer"])
def test_a_file_that_fails_the_manifest_is_refused(base: Path, assets: Path, changed: str, change: str) -> None:
    path = assets / "skeletons" / NAME / changed
    data = path.read_bytes()
    path.write_bytes(data.replace(b"SYNTHETIC", b"SYNTHETIc") if change == "same-size" else data + b"// more\n")
    message = load_refusal(base, recipe_data(skeleton=SECTION))
    assert f"{SKELETON}.name" in message and changed in message, message


def test_a_manifest_whose_sha256_is_not_its_entries_is_refused(base: Path, assets: Path) -> None:
    write_skeleton(assets, "syn-stale", digest="f" * 64)
    message = load_refusal(base, recipe_data(skeleton=section(name="syn-stale")))
    assert f"{SKELETON}.name" in message and "sha256" in message, message


@pytest.mark.parametrize("block", ["no field\n", "$skeleton and $other\n", "a lone $ sign\n"],
                         ids=["no-field", "unknown-field", "lone-dollar"])
def test_a_block_that_does_not_render_with_its_field_is_refused(base: Path, assets: Path, block: str) -> None:
    write_skeleton(assets, "syn-bad-block", block=block)
    message = load_refusal(base, recipe_data(skeleton=section(name="syn-bad-block")))
    assert f"{SKELETON}.name" in message and "block.txt" in message, message


def test_a_direction_to_another_language_is_refused(base: Path, assets: Path) -> None:
    directions = [{"source": "cpp", "target": "tt"}, {"source": "tt", "target": "cpp"}]
    message = load_refusal(base, recipe_data(skeleton=SECTION, directions=directions))
    assert SKELETON in message and "'cpp'" in message and "'tt'" in message, message


@pytest.mark.parametrize("path", sorted(RECIPE_HASHES))
def test_a_committed_recipe_keeps_its_recipe_hash(path: str) -> None:
    recipe = load_recipe(REPO / path)
    assert SKELETON not in recipe.data
    assert recipe.recipe_hash == RECIPE_HASHES[path]


def test_the_repository_skeleton_loads_by_name(base: Path) -> None:
    recipe = load(base, recipe_data(skeleton={"name": REAL, "prompts": ["generate", "correct"]}))
    assert recipe.data[SKELETON]["sha256"] == manifest_sha256(REAL_TREE)


# ---------------------------------------------------------------------------
# The runner's refusal (contract point 3)


def test_the_runner_refuses_the_section_with_a_fragment_prompt_set(base: Path, bench: Path, assets: Path) -> None:
    write_tree(assets / "prompts" / FRAGMENT_SET, FRAGMENTS)
    message = refused(base, bench, fragment_recipe(skeleton=SECTION))
    assert SKELETON in message and "fragment" in message, message
    script = Script(replies=[program(0)], attempts=[])
    _, trial = run(base, bench, script, fragment_recipe(), "upstream")
    assert [request.stage for request in trial.requests or []] == ["generate"], "without it the same recipe runs"


@pytest.mark.parametrize("with_manifest", [True, False], ids=["manifest-updated", "file-only"])
def test_the_runner_refuses_a_skeleton_changed_after_the_recipe_loaded(
    base: Path, bench: Path, assets: Path, monkeypatch: pytest.MonkeyPatch, with_manifest: bool
) -> None:
    # The SYNTHETIC skeleton changes between the loader's read and the runner's: with its manifest written again to
    # match, it still loads but its sha256 is not the recipe's; with host.cpp alone changed, it no longer loads.
    tree = assets / "skeletons" / NAME
    loaded = manifest_sha256(tree)
    changed = SYN_HOST.replace("SYNTHETIC skeleton", "SYNTHETIC changed skeleton")
    real_load = runner_module.load_recipe

    def load_then_change(*args: Any, **kwargs: Any) -> Recipe:
        recipe = real_load(*args, **kwargs)
        if with_manifest:
            shutil.rmtree(tree)
            write_skeleton(assets, NAME, host=changed)
        else:
            (tree / "host.cpp").write_bytes(changed.encode("ascii"))
        return recipe

    monkeypatch.setattr(runner_module, "load_recipe", load_then_change)
    message = refused(base, bench, recipe_data(skeleton=SECTION))
    assert f"{SKELETON}.name" in message and NAME in message, message
    if with_manifest:
        now = manifest_sha256(tree)
        assert now != loaded and now in message and loaded in message, message
    else:
        assert "host.cpp" in message, message


# ---------------------------------------------------------------------------
# The block (contract point 4)


def test_the_generation_prompt_carries_the_block_and_no_correction_does(base: Path, bench: Path) -> None:
    plain, prompts, _ = full_pair(base, bench, recipe_data(skeleton=SECTION))
    assert prompts[0] == joined([skeleton_block()], plain[0]), prompts[0]
    for number, kind in enumerate(PROMPT_KINDS[1:], start=1):
        assert prompts[number] == plain[number], f"with prompts [generate] the {kind} prompt is unchanged"


def test_with_correct_listed_a_compile_correction_carries_it_and_no_other_correction(
    base: Path, bench: Path
) -> None:
    plain, prompts, _ = full_pair(base, bench, recipe_data(skeleton=BOTH))
    assert prompts[0] == joined([skeleton_block()], plain[0])
    assert prompts[1] == joined([skeleton_block()], plain[1]), prompts[1]
    assert prompts[2] == plain[2], "a run-error correction carries no skeleton"
    assert prompts[3] == plain[3], "an oracle-mismatch correction carries no skeleton"


def test_the_block_goes_after_the_examples_and_core_blocks(base: Path, bench: Path) -> None:
    core = {"max_chars": 8000, "prompts": ["generate"], "core": [TYPES_PART, HOST_PART]}
    examples = {"pack": PACK, "max_chars": 64000, "prompts": ["generate"]}
    data = recipe_data(context=[DECL_PACK], examples=examples, lookup=core, skeleton=SECTION)
    plain, prompts, trial = full_pair(base, bench, data)
    cores = lookup_blocks([*HOST_DECLARATIONS, *TYPES_DECLARATIONS], CORE_API)
    expected = {joined([block, core_block, skeleton_block()], plain[0])
                for block in example_blocks([ZETA, ALPHA]) for core_block in cores}
    assert prompts[0] in expected, prompts[0]
    codes = [item.code for item in (trial.requests or [])[0].diagnostics]
    assert codes.index("examples") < codes.index("looked-up") < codes.index(SKELETON_CODE), codes


def test_the_block_goes_after_the_looked_up_block_in_a_compile_correction(base: Path, bench: Path) -> None:
    plain = Script(replies=[BROKEN, program(1)], attempts=[doubled])
    run(base, bench, plain, recipe_data(context=[DECL_PACK], loop={}), "plain")
    script = Script(replies=[BROKEN, program(1)], attempts=[doubled])
    lookup = {"max_chars": 8000, "prompts": ["correct"]}
    data = recipe_data(context=[DECL_PACK], loop={}, skeleton=BOTH, lookup=lookup)
    _, trial = run(base, bench, script, data, "skeleton-lookup")
    assert [request.stage for request in trial.requests or []] == ["generate", "compile_loop"]
    expected = {joined([looked, skeleton_block()], user_prompts(plain)[1]) for looked in lookup_blocks(BROKEN_NAMED)}
    assert user_prompts(script)[1] in expected, user_prompts(script)[1]


def test_the_repository_skeleton_is_carried_with_its_own_block(base: Path, bench: Path, assets: Path) -> None:
    shutil.copytree(REAL_TREE, assets / "skeletons" / REAL)
    host = (REAL_TREE / "host.cpp").read_bytes().decode("ascii")
    block = (REAL_TREE / "block.txt").read_bytes().decode("ascii")
    plain, prompts, _ = full_pair(base, bench, recipe_data(skeleton={"name": REAL, "prompts": ["generate", "correct"]}))
    assert prompts[0] == joined([skeleton_block(host, block)], plain[0])
    assert prompts[1] == joined([skeleton_block(host, block)], plain[1])
    assert host in prompts[0] and "$skeleton" not in prompts[0]


# ---------------------------------------------------------------------------
# The record and describe() (contract points 5 and 6)


def test_a_request_that_carried_the_skeleton_records_one_note(base: Path, bench: Path, assets: Path) -> None:
    _, prompts, trial = full_pair(base, bench, recipe_data(skeleton=BOTH))
    requests = trial.requests or []
    digest = manifest_sha256(assets / "skeletons" / NAME)
    for number in (0, 1):
        (note,) = skeleton_notes(requests[number])
        assert (note.stage, note.severity) == ("parse", "note"), note
        assert NAME in note.message and digest in note.message, note.message
    assert skeleton_notes(requests[2]) == [] and skeleton_notes(requests[3]) == []
    assert attempt_notes(trial) == [], "the note is on the Request, never on an attempt"
    assert not any(f"[{SKELETON_CODE}]" in prompt for prompt in prompts), "the note never enters a prompt"
    assert all(item.code != SKELETON_CODE for item in prompt_diagnostics(trial.attempts[0].diagnostics))


@pytest.mark.parametrize("prompts", [["generate"], ["generate", "correct"]])
def test_describe_names_the_skeleton_where_the_section_lists_the_prompt(
    base: Path, assets: Path, prompts: list[str]
) -> None:
    recipe = load(base, recipe_data(skeleton=section(prompts=prompts)))
    plain = load(base, recipe_data(), "plain")
    generate = GenerateStage(context=stage_context(recipe, PROMPTS)).describe()
    assert NAME in generate and SKELETON in generate, generate
    assert NAME not in GenerateStage(context=stage_context(plain, PROMPTS)).describe()
    toolchains: dict[str, Any] = {}
    compile_loop = CompileLoopStage(context=stage_context(recipe, PROMPTS, toolchains=toolchains)).describe()
    assert (NAME in compile_loop) == ("correct" in prompts), compile_loop
    assert NAME not in CompileLoopStage(context=stage_context(plain, PROMPTS, toolchains=toolchains)).describe()
