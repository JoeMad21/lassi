"""The template field kernel_files: the kernels a prompt shows, read-only (the owner's decision of 2026-10-06).

A template prompt set may show the TT kernels a bench item provides, for
every tier of TT items. lassi.core.stages gains the field kernel_files in
GENERATE_FIELDS and CORRECT_FIELDS, and GenerateStage._prompt and
_correction_messages fill it, for a template set only, as follows:

- A support file is a kernel when its build path has a directory named
  kernels (where the kernel JIT finds it in the run's working directory).
  That rule decides, not whether the file is tracked or upstream, so a
  tracked kernel counts as an upstream one does; an item-level support file
  under kernels counts too, and lassi_io.h, a header named kernels.h, a
  directory named my_kernels, and a file named kernels do not.
- The field holds the kernel support files of the direction's source and
  target languages (Suite.support_files with each language), as FILE blocks
  (render_file_blocks), in path order, each once; it is "" when there are
  none.
- p0-smoke does not use the field, so its prompts stay byte for byte what
  the four earlier fields gave, and a fragment set (lassi-2024) never shows
  a kernel. The runner's check, which renders every template a stage
  declares with every field empty, still accepts tt-host-v0 and p0-smoke.
- A kernel stays unwritable: a reply that returns one is refused at the TT
  build with bad-path (lassi/toolchains/_base.py, _unwritable), as before.
  The generic rule is also covered by
  tests/toolchains/test_harness_files.py
  (test_a_model_file_cannot_replace_a_harness_file); the test here is the
  TT form, with the kernel paths of every tt-pairs-v0 item.

tests/prompts/test_tt_host_prompts.py checks what tt-host-v0 shows for every
tt-pairs-v0 item. The fixture suite, its programs and kernels, the
fragments, and the kernels standing in for tt-pairs-v0's fetched ones are
SYNTHETIC (no tt-metal text); the p0-smoke sha256 values were computed with
sha256sum from the files at commit affa596 on 2026-10-06, and a hash is a
function of its input, not a measurement.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from string import Template
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from lassi.bench import Direction
from lassi.bench import registry as bench_registry
from lassi.bench.registry import Suite, load_suite
from lassi.core import fragments as fragment_text
from lassi.core import runner as runner_module
from lassi.core import stages
from lassi.core.files import parse_file_blocks, render_file_blocks
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.core.stages import CORRECT_FIELDS, GENERATE_FIELDS, PURPOSE, GenerateStage, _correction_messages
from lassi.toolchains import CommandResult
from lassi.toolchains.ttmetal_build import TtMetalHost

REPO = Path(__file__).resolve().parents[2]
FIELD = "kernel_files"
KERNEL_DIR = "kernels"
HEADER = "lassi_io.h"
ERRORS = "SYNTHETIC error text"
LAST_ANSWER = "// SYNTHETIC last answer\n"

# tt-pairs-v0, whose kernels are fetched files this machine does not hold: SYNTHETIC stand-ins take their place.
TT_SUITE = load_suite(REPO / "assets" / "bench" / "tt-pairs-v0.yaml")
TT_CASES = [(item, d) for item in sorted(TT_SUITE.items) for d in (Direction("cpp", "tt"), Direction("tt", "cpp"))]
TT_IDS = [f"{item}-{d.source}-to-{d.target}" for item, d in TT_CASES]
# The suite the p0-smoke runner tests use, which has no kernel; entropy has an item-level support file, layout none.
HECBENCH = load_suite(REPO / "assets" / "bench" / "lassi-hecbench-10.yaml")

SMOKE = REPO / "assets" / "prompts" / "p0-smoke"
SMOKE_SHA256 = {
    "generate": "3dd71562539fe867946b4c5ba986073f0367426bec1995d503c851e5ae262f96",
    "correct": "a1dad3aff0ece3f5003b7265aba6375bbc389601acf6a92b59085b4470964ea7",
}
# The stages a Tier A recipe lists, whose declared templates the runner renders before any model call.
TIER_A_STAGES = ("baseline", "generate", "compile_loop", "run_loop")

# The SYNTHETIC fixture suite: tracked files (under a stand-in repository) and upstream files (under its sources).
FIXTURE = "kernel-fixture"
TRACKED = {
    "assets/fixture/lassi_io.h": "// SYNTHETIC harness header\n",
    "assets/fixture/kernels.h": "// SYNTHETIC header named kernels.h, not under a kernels directory\n",
    "assets/fixture/my_kernels/helper.cpp": "// SYNTHETIC file under a directory named my_kernels\n",
    "assets/fixture/include/kernels": "// SYNTHETIC file named kernels\n",
    "assets/fixture/dev/kernels/reader.cpp": "// SYNTHETIC tracked reader kernel\n",
    "assets/fixture/dev/kernels/common.h": "// SYNTHETIC tracked kernel header\n",
}
UPSTREAM = {
    "upstream/dev/kernels/compute/add.cpp": "// SYNTHETIC upstream compute kernel\n",
    "upstream/dev2/kernels/writer.cpp": "// SYNTHETIC upstream writer kernel\n",
    "upstream/item/kernels/item_kernel.cpp": "// SYNTHETIC item-level kernel\n",
    "upstream/item/item_data.h": "// SYNTHETIC item-level header\n",
}
# Each kernel build path of the fixture suite and the file it reads.
KERNEL_SOURCES = {
    "dev/kernels/common.h": TRACKED["assets/fixture/dev/kernels/common.h"],
    "kernels/compute/add.cpp": UPSTREAM["upstream/dev/kernels/compute/add.cpp"],
    "kernels/dataflow/reader.cpp": TRACKED["assets/fixture/dev/kernels/reader.cpp"],
    "kernels/dataflow/writer.cpp": UPSTREAM["upstream/dev2/kernels/writer.cpp"],
    "kernels/shared/item_kernel.cpp": UPSTREAM["upstream/item/kernels/item_kernel.cpp"],
}
DEV_KERNELS = ("dev/kernels/common.h", "kernels/compute/add.cpp", "kernels/dataflow/reader.cpp")
FIXTURE_CASES = [
    pytest.param("mixed", "host", "host2", (), id="decoys-in-the-source"),
    pytest.param("mixed", "host2", "host", (), id="decoys-in-the-target"),
    pytest.param("mixed", "host", "dev", DEV_KERNELS, id="target-kernels-tracked-and-upstream"),
    pytest.param("mixed", "dev", "host", DEV_KERNELS, id="source-kernels"),
    pytest.param("mixed", "dev", "dev2", (*DEV_KERNELS, "kernels/dataflow/writer.cpp"), id="both-sides-once"),
    pytest.param("itemlevel", "host", "host2", ("kernels/shared/item_kernel.cpp",), id="item-level-once"),
]


def tracked(path: str) -> dict[str, str]:
    """Return a tracked support entry."""
    return {"tracked": path}


def upstream(path: str) -> dict[str, str]:
    """Return an upstream support entry with its sha256."""
    return {"upstream": path, "sha256": hashlib.sha256(UPSTREAM[path].encode("ascii")).hexdigest()}


def fixture_languages(item: str, support: dict[str, dict[str, dict[str, str]]]) -> dict[str, Any]:
    """Return the tracked languages of a fixture item, one program each, with their support entries."""
    return {
        language: {"dir": f"assets/fixture/{item}/{language}", "tracked": True, "files": ["prog.cpp"],
                   "support": entries}
        for language, entries in support.items()
    }


def fixture_manifest() -> dict[str, Any]:
    """Return the SYNTHETIC suite: `mixed` with kernels per language, `itemlevel` with an item-level kernel."""
    header = {HEADER: tracked("assets/fixture/lassi_io.h")}
    add = upstream("upstream/dev/kernels/compute/add.cpp")
    mixed = {
        "host": {**header, "kernels.h": tracked("assets/fixture/kernels.h"),
                 "my_kernels/helper.cpp": tracked("assets/fixture/my_kernels/helper.cpp"),
                 "include/kernels": tracked("assets/fixture/include/kernels")},
        "host2": dict(header),
        "dev": {**header, "kernels/compute/add.cpp": add,
                "kernels/dataflow/reader.cpp": tracked("assets/fixture/dev/kernels/reader.cpp"),
                "dev/kernels/common.h": tracked("assets/fixture/dev/kernels/common.h")},
        "dev2": {**header, "kernels/compute/add.cpp": add,
                 "kernels/dataflow/writer.cpp": upstream("upstream/dev2/kernels/writer.cpp")},
    }
    itemlevel = {
        "split": "unassigned",
        "support": {"kernels/shared/item_kernel.cpp": "upstream/item/kernels/item_kernel.cpp",
                    "item_data.h": "upstream/item/item_data.h"},
        "languages": fixture_languages("itemlevel", {"host": dict(header), "host2": dict(header)}),
    }
    return {
        "suite": FIXTURE,
        "repo": "https://example.invalid/kernel-fixture.git",
        "commit": "e" * 40,
        "items": {"mixed": {"split": "unassigned", "languages": fixture_languages("mixed", mixed)},
                  "itemlevel": itemlevel},
    }


def write_files(root: Path, files: dict[str, str]) -> None:
    """Write each relative path's text under `root` as ASCII bytes."""
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(text.encode("ascii"))


@pytest.fixture
def fixture_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Suite, Path]:
    """Return the fixture suite and its sources root; its tracked files lie under a stand-in repository."""
    repo = tmp_path / "repo"
    programs = {f"assets/fixture/{item}/{language}/prog.cpp": f"// SYNTHETIC {item} program in {language}\n"
                for item, languages in (("mixed", ("host", "host2", "dev", "dev2")), ("itemlevel", ("host", "host2")))
                for language in languages}
    write_files(repo, {**TRACKED, **programs})
    monkeypatch.setattr(bench_registry, "TRACKED_ROOT", repo)
    sources = tmp_path / "sources"
    write_files(sources, UPSTREAM)
    manifest = tmp_path / f"{FIXTURE}.yaml"
    manifest.write_bytes(yaml.safe_dump(fixture_manifest(), sort_keys=False).encode("ascii"))
    return load_suite(manifest), sources


def stand_in_text(path: str) -> str:
    """Return the SYNTHETIC text written in place of the fetched file at upstream path `path`."""
    return f"// SYNTHETIC stand-in for the fetched file {path}\n"


@pytest.fixture(scope="module")
def tt_sources(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a sources root with a SYNTHETIC stand-in at every upstream support path of tt-pairs-v0."""
    root = tmp_path_factory.mktemp("tt-pairs-sources")
    for found in TT_SUITE.items.values():
        paths = list(found.support.values())
        for spec in found.languages.values():
            paths += [entry.path for entry in spec.support.values() if not entry.tracked]
        write_files(root, {path: stand_in_text(path) for path in paths})
    return root


@pytest.fixture
def rendered_fields(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, str]]]:
    """Record the prompt name and fields of each template the stages render, and render it as before."""
    calls: list[tuple[str, dict[str, str]]] = []
    real_render = stages.render

    def recording_render(set_name: str, prompt_name: str, fields: Any) -> str:
        calls.append((prompt_name, dict(fields)))
        return real_render(set_name, prompt_name, fields)

    monkeypatch.setattr(stages, "render", recording_render)
    return calls


def is_kernel(path: str) -> bool:
    """Return True for a build path with a directory named kernels in it."""
    return KERNEL_DIR in PurePosixPath(path).parts[:-1]


def tt_kernels(item: str, direction: Direction, sources: Path) -> dict[str, str]:
    """Return the kernel support files of both languages of a tt-pairs-v0 direction (build path -> text)."""
    found: dict[str, str] = {}
    for language in (direction.source, direction.target):
        files = TT_SUITE.support_files(item, sources, purpose=PURPOSE, language=language)
        found.update({path: text for path, text in files.items() if is_kernel(path)})
    return found


def stand_in(
    suite: Suite, item: str, direction: Direction, sources: Path, prompts: str, fragments: dict[str, str] | None = None
) -> Any:
    """Return a stand-in RunContext for the prompt builders: a template set unless `fragments` is given."""
    return SimpleNamespace(suite=suite, item=item, direction=direction, sources_root=sources, prompts=prompts,
                           fragments=fragments or {}, packs={})


def target_names(context: Any) -> list[str]:
    """Return the target language's file names of the context's item."""
    return list(context.suite.item(context.item, purpose=PURPOSE).languages[context.direction.target].files)


def generate(context: Any) -> tuple[str | None, str]:
    """Return what GenerateStage._prompt gives for the context, with an empty trial context."""
    trial = SimpleNamespace(context=SimpleNamespace(knowledge_summary="", source_description=""))
    return GenerateStage._prompt(SimpleNamespace(context=context), trial, target_names(context))


def correct(context: Any, files: dict[str, str], *, run_error: bool = False) -> tuple[str | None, str]:
    """Return what _correction_messages gives for a last answer of `files` and the error text ERRORS."""
    previous = SimpleNamespace(files=files)
    return _correction_messages(context, previous, target_names(context), ERRORS, run_error=run_error)


def field_values(context: Any, calls: Sequence[tuple[str, dict[str, str]]]) -> dict[str, dict[str, str]]:
    """Build the generate and correction prompts of `context` and return the fields each rendered with."""
    generate(context)
    correct(context, {name: LAST_ANSWER for name in target_names(context)})
    assert [name for name, _ in calls] == ["generate", "correct"]
    return dict(calls)


def smoke_text(name: str) -> str:
    """Return the text of a p0-smoke template after checking its bytes are the pinned ones."""
    data = (SMOKE / f"{name}.txt").read_bytes()
    assert hashlib.sha256(data).hexdigest() == SMOKE_SHA256[name], f"p0-smoke/{name}.txt changed"
    return data.decode("ascii")


def synthetic_fragments(direction: Direction) -> dict[str, str]:
    """Return a SYNTHETIC fragment set for `direction`: each key the generate and correction prompts read."""
    keys = {fragment_text.fragment_key(key, direction) for key in (*fragment_text.GENERATE_KEYS, *stages.RUN_LOOP_KEYS)}
    return {key: f"<{key}>" for key in sorted(keys)}


def test_kernel_files_is_a_field_of_both_templates() -> None:
    assert FIELD in GENERATE_FIELDS, "generate.txt may not use $kernel_files"
    assert FIELD in CORRECT_FIELDS, "correct.txt may not use $kernel_files"


@pytest.mark.parametrize(("item", "source", "target", "names"), FIXTURE_CASES)
def test_kernel_files_holds_each_kernel_of_both_languages_once_in_path_order(
    fixture_suite: tuple[Suite, Path],
    rendered_fields: list[tuple[str, dict[str, str]]],
    item: str,
    source: str,
    target: str,
    names: tuple[str, ...],
) -> None:
    suite, sources = fixture_suite
    context = stand_in(suite, item, Direction(source, target), sources, "p0-smoke")
    want = render_file_blocks({name: KERNEL_SOURCES[name] for name in names}) if names else ""
    for prompt, fields in field_values(context, rendered_fields).items():
        assert FIELD in fields, f"the {prompt} prompt is rendered with no {FIELD} field"
        assert fields[FIELD] == want, prompt


HECBENCH_DIRECTIONS = [Direction("omp", "cuda"), Direction("cuda", "omp")]


@pytest.mark.parametrize("direction", HECBENCH_DIRECTIONS, ids=["omp-cuda", "cuda-omp"])
@pytest.mark.parametrize("item", ["entropy", "layout"])
def test_kernel_files_is_empty_for_a_suite_without_kernels(
    tmp_path: Path, rendered_fields: list[tuple[str, dict[str, str]]], item: str, direction: Direction
) -> None:
    found = HECBENCH.items[item]
    files = {f"{spec.dir}/{name}": f"// SYNTHETIC {item} source\n" for spec in found.languages.values()
             for name in spec.files}
    write_files(tmp_path, {**files, **{path: "// SYNTHETIC support file\n" for path in found.support.values()}})
    context = stand_in(HECBENCH, item, direction, tmp_path, "p0-smoke")
    for prompt, fields in field_values(context, rendered_fields).items():
        assert FIELD in fields, f"the {prompt} prompt is rendered with no {FIELD} field"
        assert fields[FIELD] == "", prompt


@pytest.mark.parametrize("name", sorted(SMOKE_SHA256))
def test_p0_smoke_templates_are_unchanged(name: str) -> None:
    smoke_text(name)


@pytest.mark.parametrize(("item", "direction"), TT_CASES, ids=TT_IDS)
def test_p0_smoke_prompts_stay_as_the_four_fields_gave_them(tt_sources: Path, item: str, direction: Direction) -> None:
    # The p0-smoke prompts of the hecbench suite are compared with the rendered templates in tests/core/test_runner.py
    # (test_mock_trial_prompt_is_the_rendered_generate_template, test_correction_prompt_carries_the_files_and_the_
    # diagnostics); these items have kernels, which p0-smoke never shows.
    context = stand_in(TT_SUITE, item, direction, tt_sources, "p0-smoke")
    names = ", ".join(target_names(context))
    source = TT_SUITE.source_files(item, direction, tt_sources, purpose=PURPOSE)
    before = {"source_language": direction.source, "target_language": direction.target,
              "source_files": render_file_blocks(source), "target_files": names}
    assert generate(context) == (None, Template(smoke_text("generate")).substitute(before))
    last = {name: LAST_ANSWER for name in target_names(context)}
    before = {"target_language": direction.target, "files": render_file_blocks(last), "diagnostics": ERRORS,
              "target_files": names}
    assert correct(context, last) == (None, Template(smoke_text("correct")).substitute(before))


@pytest.mark.parametrize(("item", "direction"), TT_CASES, ids=TT_IDS)
def test_a_fragment_set_never_shows_a_kernel(tt_sources: Path, item: str, direction: Direction) -> None:
    # lassi-2024's own prompts are compared exactly in tests/core/test_faithful_generation.py; here a fragment set
    # meets items with kernels, and its prompts are what the fragment helpers build from the source alone.
    fragments = synthetic_fragments(direction)
    context = stand_in(TT_SUITE, item, direction, tt_sources, "synthetic-fragments", fragments=fragments)
    (source,) = TT_SUITE.source_files(item, direction, tt_sources, purpose=PURPOSE).values()
    system = fragments[fragment_text.fragment_key(fragment_text.DIRECTION_SYSTEM, direction)]
    want = fragment_text.generation_prompt(fragments, direction, fragment_text.as_text_mode(source), None, "", "")
    assert generate(context) == (system, want)
    last = {name: LAST_ANSWER for name in target_names(context)}
    want = fragment_text.correction_prompt(fragments, direction, LAST_ANSWER, ERRORS)
    assert correct(context, last) == (system, want)
    _, run_prompt = correct(context, last, run_error=True)
    for path, text in tt_kernels(item, direction, tt_sources).items():
        assert f"// FILE: {path}" not in run_prompt and text not in run_prompt, path


@pytest.mark.parametrize("prompt_set", ["tt-host-v0", "p0-smoke"])
def test_the_runner_accepts_the_template_sets(prompt_set: str) -> None:
    # _check_prompts renders each template a listed stage declares with every declared field set to "".
    entries = [DEFAULT_REGISTRY.get("Stage", name) for name in TIER_A_STAGES]
    recipe = SimpleNamespace(path=Path(f"{prompt_set}-stand-in.yaml"))
    settings = SimpleNamespace(fragments={}, prompts=prompt_set)
    runner_module._check_prompts(recipe, entries, settings)


@pytest.mark.parametrize("item", sorted(TT_SUITE.items))
def test_a_reply_that_returns_a_shown_kernel_is_refused_at_the_tt_build(
    tmp_path: Path, tt_sources: Path, item: str
) -> None:
    harness = TT_SUITE.support_files(item, tt_sources, purpose=PURPOSE, language="tt")
    kernels = sorted(path for path in harness if is_kernel(path))
    assert kernels, f"{item} declares no kernel"
    host = TT_SUITE.source_files(item, Direction("tt", "cpp"), tt_sources, purpose=PURPOSE)
    reply = render_file_blocks({**host, **{path: "// SYNTHETIC kernel a model returned\n" for path in kernels}})
    parsed = parse_file_blocks(reply, list(host))
    assert parsed.diagnostics == [] and set(parsed.files) == {*host, *kernels}
    calls: list[Sequence[str]] = []

    def runner(argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        calls.append(argv)
        return CommandResult(returncode=0, stdout="", stderr="")

    workdir = tmp_path / "build"
    workdir.mkdir()
    result = TtMetalHost(tree=tmp_path / "tree", runner=runner).build(parsed.files, workdir, harness=harness)
    assert (calls, result.artifact) == ([], None), "nothing is compiled"
    assert [(found.code, found.file) for found in result.diagnostics] == [("bad-path", path) for path in kernels]
    assert all("the harness provides that file" in found.message for found in result.diagnostics)
    assert list(workdir.iterdir()) == [], "nothing is written"
