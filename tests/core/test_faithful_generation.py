"""Tests for faithful generation: summarize_context, describe_source, and generate (task P1.4).

Bible: Source Papers (LASSI pipeline steps 2 to 4; quirk table, fence row),
Component Interfaces (Stage row and contract rules), Result Record
(Trial.context, Diagnostic), Project Recipes (Notes), Design Principle 4.

The contract these tests fix, from the P1.4 acceptance criteria:

- The stages `summarize_context`, `describe_source`, and `generate` are
  registered in DEFAULT_REGISTRY once lassi.core.runner is imported. The
  runner no longer refuses a recipe's `context`; it loads the recipe's
  prompt fragments and context packs with lassi.prompts.load_recipe_assets
  (default assets root) and passes them to the stages. An unknown pack is a
  RunError before any directory is created or any model is asked.
- summarize_context sends [system, user]: upstream's general system prompt,
  then the summary request (intro, the OpenMP note only when the target is
  OpenMP, outro) followed by the pack upstream picks for the target language
  (openmp-4.0-card for omp, cuda-12.5-ch5 for cuda). describe_source sends
  [system, user]: the general system prompt, then the description intro
  followed by the source text. Neither prompt is changed further. The
  replies fill Trial.context (knowledge_summary, source_description) as
  returned, and neither stage appends an attempt.
- Faithful generate sends [system, user]: the direction's system prompt,
  then upstream's assembled prompt (context block, summary block, the two
  leads, the description, the request lead, the direction's translation
  prompt, one space, the source) with every run of spaces collapsed to one
  space, source indentation included, as the pinned notebook does before its
  first generation call. Tabs and newlines stay.
- The source reaches both prompts as upstream reads it: the notebook opens
  the source file in text mode, so a CRLF line end arrives as LF (upstream's
  pathfinder sources are CRLF).
- Under faithful extraction the attempt's one target file (the manifest's
  target file name) is the text of the first fenced block after upstream's
  tag stripping: a cpp or c++ tag is cut, then one more leading 'c' is cut.
  A cuda tag therefore leaves text starting "uda"; that attempt carries a
  parse-stage warning Diagnostic with code `fence-quirk`, which the Parquet
  diagnostics table lets a metric count. A fence whose tag strips exactly
  (cpp, c++, c, none) carries no such Diagnostic.
- Each reproduced quirk is a named fix in lassi.core.recipe.FIXES, on by
  default and off under `faithful: true`: `fence_tag` (P0) and
  `prompt_spaces` (this task; on keeps the generation prompt's runs of
  spaces). The runner refuses a fix turned off only when no bound stage
  reproduces its quirk.
- With every fix on, generate keeps the P0 path: one user message rendered
  from the prompt set's generate.txt and FILE blocks parsed from the reply,
  with no `fence-quirk` Diagnostic.

Since P1.5, `faithful: true` also turns off baseline_both, prompt_newlines,
and parsed_diagnostics, which the baseline and compile_loop stages
reproduce, so the faithful recipes here list them too, with a fake toolchain
for the target language that compiles nothing and reports a PLACEHOLDER
artifact. Each attempt 0 therefore ends at S4, and the model is asked no
more than before.

The expected prompts are computed here from the fragments and packs loaded
out of a fresh extraction of the pinned upstream checkout into a temporary
directory (tools/extract_lassi_assets.py); tests that need it skip, naming
that tool, when third_party/LASSI is absent or not at the pin. No upstream
text is copied into this file (OQ-018): identifiers such as fragment keys
and dictionary entry names are not upstream prose. Bench sources and model
replies are short synthetic texts written here. No model, compiler, or
upstream code runs, and no value in this module is a measurement.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

import lassi.prompts as prompts_module
import lassi.prompts.assets as prompt_assets
from lassi.bench import Direction, load_suite
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Message, Sampling
from lassi.core.parquet import read_run_parquet
from lassi.core.recipe import FIXES, load_recipe
from lassi.core.record import Trial, make_trial_id
from lassi.core.registry import DEFAULT_REGISTRY, Registry, RegistryError
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.executors import NoneExecutor
from lassi.prompts import RecipeAssets, load_recipe_assets, render

REPO = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = REPO / "third_party" / "LASSI"
UPSTREAM_PIN = "74b46812523f2ff79b53b6880a4521690d7478b0"
TOOL = REPO / "tools" / "extract_lassi_assets.py"
SUITE_MANIFEST = REPO / "assets" / "bench" / "lassi-hecbench-10.yaml"
SKIP_REASON = (
    "needs the upstream checkout at third_party/LASSI on commit 74b4681 so that tools/extract_lassi_assets.py "
    "can generate the lassi-2024 fragments; run uv run tools/fetch_upstream.py first"
)

SUITE = "lassi-hecbench-10"
ITEM = "layout"
MODEL_ID = "scripted-fixture"
PROMPT_SET = "lassi-2024"
P0_PROMPT_SET = "p0-smoke"
FAITHFUL_STAGES = ("baseline", "summarize_context", "describe_source", "generate", "compile_loop")
# The fake toolchain bound for each target language, registered under the preset's name (compiles nothing).
TOOLCHAIN_OF = {"cuda": "nvcc-sm80", "omp": "nvcpp-cc80"}
# The pack upstream's experimental setup selects for each target language (its context knowledge entry).
PACK_OF = {"omp": "openmp-4.0-card", "cuda": "cuda-12.5-ch5"}
# Upstream's dictionary entry name for each direction: `<SOURCE>_to_<TARGET>` in upstream's language spelling.
DIRECTION_KEY = {("omp", "cuda"): "OMP_to_CUDA", ("cuda", "omp"): "CUDA_to_OMP"}
DIRECTIONS = [Direction("omp", "cuda"), Direction("cuda", "omp")]
DIRECTION_IDS = ["omp-cuda", "cuda-omp"]
# The stage ladder rungs of an attempt whose target file was extracted, and of one that was built (bible Training
# Module, Reward Function).
PARSED = "S1"
COMPILED = "S4"
FENCE_QUIRK = "fence-quirk"
# The fixes this task's stages reproduce: P0's fence tag quirk and the generation prompt's space collapse.
REPRODUCED_FIXES = ("fence_tag", "prompt_spaces")

# Synthetic bench sources: runs of spaces in indentation and code, and one tab, so the collapse shows.
OMP_SOURCE = (
    "#include <cstdio>\n"
    "int main() {\n"
    "    int  n  =  4;\n"
    "\t#pragma omp target teams distribute parallel for\n"
    "    for (int i = 0; i < n; i++) { }\n"
    '    std::printf("n   = %d\\n", n);\n'
    "    return 0;\n"
    "}\n"
)
CUDA_SOURCE = (
    "#include <cstdio>\n"
    "__global__ void kernel(int *out) {\n"
    "    out[threadIdx.x]  =  1;\n"
    "}\n"
    "int main() {\n"
    "    int *out  = nullptr;\n"
    "    kernel<<<1,   32>>>(out);\n"
    "    return 0;\n"
    "}\n"
)
SOURCES = {"omp": OMP_SOURCE, "cuda": CUDA_SOURCE}

# Synthetic model replies. The summary and description hold runs of spaces that only the generation prompt collapses.
SUMMARY_REPLY = "Synthetic summary:  the pack    covers offload.\n  Indented second line.  "
DESCRIPTION_REPLY = "Synthetic description:   the program  fills an array\tand prints it."
CODE = '#include <cstdio>\nint main() {\n    std::printf("ok\\n");\n    return 0;\n}\n'
GENERATION_REPLY = "Synthetic reply.\n```\n" + CODE + "```\nEnd of reply.\n"


# ---------------------------------------------------------------------------
# Environment, upstream extraction, and bench sources


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate variables a test could inherit and point TMPDIR at a test directory."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))


def checkout_at_pin() -> bool:
    """Return True when third_party/LASSI is its own git checkout whose HEAD is the upstream pin."""
    if not (UPSTREAM_DIR / ".git").exists():
        return False
    done = subprocess.run(
        ["git", "-C", str(UPSTREAM_DIR), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return done.returncode == 0 and done.stdout.strip() == UPSTREAM_PIN


def load_extractor() -> ModuleType:
    """Load tools/extract_lassi_assets.py as a module."""
    spec = importlib.util.spec_from_file_location("extract_lassi_assets_for_p14", TOOL)
    assert spec and spec.loader, f"cannot load {TOOL}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def assets_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a temporary assets root holding a fresh extraction of the pinned checkout, or skip naming the tool."""
    if not checkout_at_pin():
        pytest.skip(SKIP_REASON)
    out = tmp_path_factory.mktemp("lassi-assets")
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        status = load_extractor().main(["--upstream", str(UPSTREAM_DIR), "--out", str(out)])
    assert status == 0, f"tools/extract_lassi_assets.py exited {status}: {sink.getvalue()[-2000:]}"
    return out


@pytest.fixture
def lassi_assets(assets_root: Path, monkeypatch: pytest.MonkeyPatch) -> RecipeAssets:
    """Point the asset loader and the prompt sets at the fresh extraction; return its fragments and both packs."""
    monkeypatch.setattr(prompt_assets, "default_root", lambda: assets_root)
    monkeypatch.setattr(prompts_module, "default_roots", lambda: (assets_root / "prompts", REPO / "assets" / "prompts"))
    return load_recipe_assets({"prompts": PROMPT_SET, "context": list(PACK_OF.values())}, root=assets_root)


def write_bench(root: Path, sources: Mapping[str, str]) -> Path:
    """Write the item's synthetic source per language where the suite manifest lays it out; return `root`."""
    item = load_suite(SUITE_MANIFEST).items[ITEM]
    for language, text in sources.items():
        spec = item.languages[language]
        path = root / spec.dir / spec.files[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    """Return a bench root holding the synthetic sources of the item in both languages (LF line ends)."""
    return write_bench(tmp_path / "bench", SOURCES)


def item_file(language: str) -> str:
    """Return the item's one file name in `language`, from the suite manifest."""
    files = load_suite(SUITE_MANIFEST).items[ITEM].languages[language].files
    assert len(files) == 1, f"{ITEM} has {len(files)} {language} files; these tests expect one"
    return files[0]


def target_file(direction: Direction) -> str:
    """Return the item's one target file name for `direction`."""
    return item_file(direction.target)


# ---------------------------------------------------------------------------
# Components and runs


@dataclass
class Script:
    """The replies a scripted backend gives, in order, and every request it received."""

    replies: list[str]
    requests: list[tuple[list[Message], Sampling]] = field(default_factory=list)


def scripted_backend(script: Script) -> type:
    """Return an LLMBackend class, registered here as "scripted", that answers from `script` in order."""

    class ScriptedBackend:
        """Answers each request with the next scripted reply; a request past the script's end fails the test."""

        name = "scripted"
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id, as every backend does."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Record the request and return the next scripted reply."""
            script.requests.append((list(messages), sampling))
            assert script.replies, "the backend was asked for more replies than the script holds"
            return Completion(text=script.replies.pop(0), prompt_tokens=0, completion_tokens=0)

    return ScriptedBackend


class NoopStage:
    """A stage that reproduces no upstream quirk and returns the trial unchanged."""

    name = "noop"
    capabilities: frozenset[str] = frozenset()
    requires: dict[str, frozenset[str]] = {}
    prompt_fields: dict[str, tuple[str, ...]] = {}

    def __init__(self, *, context: Any) -> None:
        """Keep the trial's run context."""
        self.context = context

    def __call__(self, trial: Trial) -> Trial:
        """Return `trial` unchanged."""
        return trial

    def describe(self) -> str:
        """Return a one-line description of the stage."""
        return "noop: returns the trial unchanged"


class FakeToolchain:
    """A toolchain that writes nothing, compiles nothing, and reports a PLACEHOLDER artifact for every build."""

    capabilities = frozenset({"diagnostics"})

    def build(self, files: Any, workdir: Path, harness: Any = None) -> BuildResult:
        """Return a build with a PLACEHOLDER artifact path and no diagnostics; nothing is written or run."""
        return BuildResult(artifact=Path(workdir) / "PLACEHOLDER-artifact", diagnostics=[])


def registered_stage(name: str) -> type:
    """Return the Stage class registered as `name` in DEFAULT_REGISTRY; fail clearly while it is missing."""
    try:
        return DEFAULT_REGISTRY.get("Stage", name).factory
    except RegistryError as error:
        pytest.fail(f"no stage is registered as {name!r} ({error}); task P1.4 adds it")


def make_registry(script: Script, stages: Sequence[str]) -> Registry:
    """Return a test Registry: the scripted backend, the none executor, fake toolchains, NoopStage, and real stages."""
    registry = Registry()
    registry.register("LLMBackend", "scripted", scripted_backend(script))
    registry.register("Executor", "none", NoneExecutor)
    for name in TOOLCHAIN_OF.values():
        registry.register("Toolchain", name, FakeToolchain)
    registry.register("Stage", "noop", NoopStage)
    for name in stages:
        if name != "noop":
            registry.register("Stage", name, registered_stage(name))
    return registry


def recipe_data(direction: Direction, **changes: Any) -> dict[str, Any]:
    """Return a faithful recipe for one item and `direction`; top-level `changes` replace keys, None drops one."""
    data: dict[str, Any] = {
        "extends": "base",
        "faithful": True,
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": direction.source, "target": direction.target}],
        "prompts": PROMPT_SET,
        "context": list(PACK_OF.values()),
        "toolchain": {direction.target: TOOLCHAIN_OF[direction.target]},
        "stages": list(FAITHFUL_STAGES),
        "executor": {"kind": "none"},
        "trials": {"n": 1},
    }
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = copy.deepcopy(value)
    return data


def write_recipe(directory: Path, name: str, data: Mapping[str, Any]) -> Path:
    """Write `data` as the recipe `<directory>/<name>.yaml` and return its path."""
    path = directory / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


@dataclass
class Outcome:
    """One finished run: its directory, its one trial read back from the run tree, and the model requests."""

    run_dir: Path
    trial: Trial
    requests: list[tuple[list[Message], Sampling]]

    def messages(self, index: int) -> list[tuple[str, str]]:
        """Return request `index` as (role, content) pairs."""
        return [(message.role, message.content) for message in self.requests[index][0]]


def run_one(tmp_path: Path, bench_root: Path, name: str, data: Mapping[str, Any], replies: Sequence[str]) -> Outcome:
    """Run a one-trial recipe with the scripted backend answering `replies`; return the outcome."""
    script = Script(list(replies))
    registry = make_registry(script, data["stages"])
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=bench_root, registry=registry)
    run_dir = run_recipe(write_recipe(tmp_path, name, data), options)
    direction = data["directions"][0]
    trial_id = make_trial_id(name, MODEL_ID, SUITE, f"{direction['source']}-{direction['target']}", ITEM, 1)
    trial = read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir))
    return Outcome(run_dir, trial, script.requests)


def assert_refused_before_anything_runs(
    tmp_path: Path, bench_root: Path, name: str, data: Mapping[str, Any], match: str
) -> None:
    """Run the recipe `data` and assert a RunError matching `match`, with no run tree and no model call."""
    script = Script([])
    registry = make_registry(script, data["stages"])
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=bench_root, registry=registry)
    with pytest.raises(RunError, match=match):
        run_recipe(write_recipe(tmp_path, name, data), options)
    assert not (tmp_path / "runs-root").exists(), "a refused run creates no directory"
    assert script.requests == [], "a refused run asks no model"


# ---------------------------------------------------------------------------
# The prompts upstream sends, computed from the loaded fragments


def as_upstream_reads(text: str) -> str:
    """Return a source text as the notebook reads it: a file opened in text mode, CRLF and CR arriving as LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def summary_prompt(assets: RecipeAssets, target: str) -> str:
    """Return the summary request: intro, the OpenMP note for an OpenMP target, outro, then the target's pack."""
    fragments = assets.fragments
    text = fragments["summarize_context.intro"]
    if target == "omp":
        text += fragments["summarize_context.omp_note"]
    return text + fragments["summarize_context.outro"] + assets.packs[PACK_OF[target]]


def description_prompt(assets: RecipeAssets, source: str) -> str:
    """Return the description request: the intro, then the source as upstream reads it."""
    return assets.fragments["describe_source.intro"] + as_upstream_reads(source)


def assembled_prompt(assets: RecipeAssets, direction: Direction, summary: str, description: str, source: str) -> str:
    """Return upstream's generation prompt before its space collapse, in the notebook's order of joins."""
    fragments = assets.fragments
    key = DIRECTION_KEY[(direction.source, direction.target)]
    pack = assets.packs[PACK_OF[direction.target]]
    return (
        fragments["generate.context_open"] + pack + fragments["generate.context_close"]
        + fragments["generate.summary_open"] + summary + fragments["generate.summary_close"]
        + fragments["generate.context_lead"]
        + fragments["generate.description_lead"] + description
        + fragments["generate.request_lead"] + fragments[f"codetranslate_prompt_dict.{key}"]
        + " " + as_upstream_reads(source)
    )


def collapse_spaces(text: str) -> str:
    """Return `text` with every run of spaces cut to one space; tabs and newlines stay."""
    return re.sub(" +", " ", text)


def fence_quirks(trial: Trial) -> list[Any]:
    """Return every Diagnostic with code fence-quirk across the trial's attempts."""
    return [diagnostic for attempt in trial.attempts for diagnostic in attempt.diagnostics
            if diagnostic.code == FENCE_QUIRK]


# ---------------------------------------------------------------------------
# summarize_context and describe_source


@pytest.mark.parametrize("direction", DIRECTIONS, ids=DIRECTION_IDS)
def test_summarize_context_sends_the_general_system_prompt_and_the_target_pack(
    tmp_path: Path, bench: Path, lassi_assets: RecipeAssets, direction: Direction
) -> None:
    replies = [SUMMARY_REPLY, DESCRIPTION_REPLY, GENERATION_REPLY]
    outcome = run_one(tmp_path, bench, "faithful-summary", recipe_data(direction), replies)
    assert len(outcome.requests) == 3, "summarize_context, describe_source, and generate each ask the model once"
    expected = [
        ("system", lassi_assets.fragments["system_prompt_dict.general_system"]),
        ("user", summary_prompt(lassi_assets, direction.target)),
    ]
    assert outcome.messages(0) == expected


@pytest.mark.parametrize("direction", DIRECTIONS, ids=DIRECTION_IDS)
def test_describe_source_sends_the_general_system_prompt_and_the_source(
    tmp_path: Path, bench: Path, lassi_assets: RecipeAssets, direction: Direction
) -> None:
    replies = [SUMMARY_REPLY, DESCRIPTION_REPLY, GENERATION_REPLY]
    outcome = run_one(tmp_path, bench, "faithful-description", recipe_data(direction), replies)
    expected = [
        ("system", lassi_assets.fragments["system_prompt_dict.general_system"]),
        ("user", description_prompt(lassi_assets, SOURCES[direction.source])),
    ]
    assert outcome.messages(1) == expected
    assert "  " in outcome.messages(1)[1][1], "the description request keeps the source's runs of spaces"


@pytest.mark.parametrize("direction", DIRECTIONS, ids=DIRECTION_IDS)
def test_summary_and_description_fill_trial_context_and_append_no_attempt(
    tmp_path: Path, bench: Path, lassi_assets: RecipeAssets, direction: Direction
) -> None:
    replies = [SUMMARY_REPLY, DESCRIPTION_REPLY, GENERATION_REPLY]
    outcome = run_one(tmp_path, bench, "faithful-context", recipe_data(direction), replies)
    assert outcome.trial.context.knowledge_summary == SUMMARY_REPLY
    assert outcome.trial.context.source_description == DESCRIPTION_REPLY
    assert [attempt.index for attempt in outcome.trial.attempts] == [0], "only generate appends an attempt"


def test_an_unknown_context_pack_is_refused_before_any_model_call(
    tmp_path: Path, bench: Path, lassi_assets: RecipeAssets
) -> None:
    data = recipe_data(DIRECTIONS[0], context=[PACK_OF["cuda"], "no-such-pack"])
    assert_refused_before_anything_runs(tmp_path, bench, "unknown-pack", data, "no-such-pack")


# ---------------------------------------------------------------------------
# Faithful generate: the prompt


@pytest.mark.parametrize("direction", DIRECTIONS, ids=DIRECTION_IDS)
def test_faithful_generate_sends_the_direction_system_prompt_and_the_space_collapsed_prompt(
    tmp_path: Path, bench: Path, lassi_assets: RecipeAssets, direction: Direction
) -> None:
    replies = [SUMMARY_REPLY, DESCRIPTION_REPLY, GENERATION_REPLY]
    outcome = run_one(tmp_path, bench, "faithful-generate", recipe_data(direction), replies)
    key = DIRECTION_KEY[(direction.source, direction.target)]
    before = assembled_prompt(lassi_assets, direction, SUMMARY_REPLY, DESCRIPTION_REPLY, SOURCES[direction.source])
    assert collapse_spaces(before) != before, "the synthetic inputs must hold runs of spaces for this test to bite"
    expected = [
        ("system", lassi_assets.fragments[f"system_prompt_dict.{key}"]),
        ("user", collapse_spaces(before)),
    ]
    assert outcome.messages(2) == expected
    sent = outcome.messages(2)[1][1]
    assert "  " not in sent, "every run of spaces is collapsed, source indentation included"


def test_faithful_prompts_read_the_source_as_upstream_reads_it(
    tmp_path: Path, lassi_assets: RecipeAssets
) -> None:
    direction = DIRECTIONS[0]
    crlf = {"omp": OMP_SOURCE.replace("\n", "\r\n"), "cuda": CUDA_SOURCE}
    bench_root = write_bench(tmp_path / "bench-crlf", crlf)
    replies = [SUMMARY_REPLY, DESCRIPTION_REPLY, GENERATION_REPLY]
    outcome = run_one(tmp_path, bench_root, "faithful-crlf", recipe_data(direction), replies)
    assert outcome.messages(1)[1] == ("user", description_prompt(lassi_assets, crlf["omp"]))
    before = assembled_prompt(lassi_assets, direction, SUMMARY_REPLY, DESCRIPTION_REPLY, crlf["omp"])
    assert outcome.messages(2)[1] == ("user", collapse_spaces(before))
    for index in (1, 2):
        assert "\r" not in outcome.messages(index)[1][1], "the notebook reads the source in text mode: CRLF is LF"


# ---------------------------------------------------------------------------
# Faithful generate: extraction and the fence quirk


FENCE_CASES = [
    pytest.param("Synthetic reply.\n```cuda\n" + CODE + "```\nEnd of reply.\n", "uda\n" + CODE, True, id="cuda-tag"),
    pytest.param("Synthetic reply.\n```cpp\n" + CODE + "```\nEnd of reply.\n", "\n" + CODE, False, id="cpp-tag"),
    pytest.param("Synthetic reply.\n```c++\n" + CODE + "```\nEnd of reply.\n", "\n" + CODE, False, id="cxx-tag"),
    pytest.param("Synthetic reply.\n```c\n" + CODE + "```\nEnd of reply.\n", "\n" + CODE, False, id="c-tag"),
    pytest.param("Synthetic reply.\n```\n" + CODE + "```\nEnd of reply.\n", "\n" + CODE, False, id="untagged"),
    pytest.param(
        "First.\n```cpp\n" + CODE + "```\nSecond.\n```cuda\nint other;\n```\n", "\n" + CODE, False,
        id="first-of-several",
    ),
]


@pytest.mark.parametrize(("reply", "text", "quirk"), FENCE_CASES)
def test_faithful_extraction_takes_the_first_fence_after_upstream_tag_stripping(
    tmp_path: Path, bench: Path, lassi_assets: RecipeAssets, reply: str, text: str, quirk: bool
) -> None:
    direction = DIRECTIONS[0]
    replies = [SUMMARY_REPLY, DESCRIPTION_REPLY, reply]
    outcome = run_one(tmp_path, bench, "faithful-fence", recipe_data(direction), replies)
    attempt = outcome.trial.attempts[0]
    assert attempt.response_text == reply
    assert attempt.files == {target_file(direction): text}
    assert attempt.stage_reached == COMPILED, "extracted (S1), then built by compile_loop with the fake toolchain"
    assert not [d for d in attempt.diagnostics if d.code == "no-fence"], "a fence was found, so no S0 was built"
    hits = fence_quirks(outcome.trial)
    if quirk:
        assert len(hits) == 1, f"one fence-quirk Diagnostic, got {hits}"
        assert (hits[0].stage, hits[0].severity) == ("parse", "warning")
        assert hits[0].message, "the Diagnostic says what happened"
    else:
        assert hits == [], "a tag upstream strips exactly is no fence-quirk hit"
    rows = read_run_parquet(outcome.run_dir / "parquet")["diagnostics"]
    counted = [row for row in rows if row["code"] == FENCE_QUIRK]
    assert len(counted) == (1 if quirk else 0), "a metric counts fence-quirk hits from the Parquet diagnostics"
    for row in counted:
        assert (row["stage"], row["severity"], row["attempt_index"]) == ("parse", "warning", 0)


# ---------------------------------------------------------------------------
# Fixes: named toggles, faithful overrides, and the runner's refusal rule


def test_each_reproduced_quirk_is_a_named_fix_off_under_faithful(tmp_path: Path) -> None:
    for name in REPRODUCED_FIXES:
        assert name in FIXES, f"{name} is not a named fix in lassi.core.recipe.FIXES"
        assert isinstance(FIXES[name], str) and FIXES[name].strip(), f"fix {name} needs a description"
    common = {
        "extends": "base",
        "model": {"backend": "mock", "id": "mock-reference"},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": "omp", "target": "cuda"}],
        "prompts": P0_PROMPT_SET,
        "stages": ["generate"],
        "executor": {"kind": "none"},
    }
    faithful = load_recipe(write_recipe(tmp_path, "faithful-fixes", {**common, "faithful": True})).data
    plain = load_recipe(write_recipe(tmp_path, "plain-fixes", common)).data
    for name in REPRODUCED_FIXES:
        assert faithful["fixes"][name] is False, f"faithful: true turns fixes.{name} off"
        assert plain["fixes"][name] is True, f"fixes.{name} is on by default"


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"faithful": True}, r"fixes\.baseline_both is off \(faithful: true\)"),
        ({"faithful": False, "fixes": {"prompt_spaces": False}}, r"fixes\.prompt_spaces is off \(faithful: false\)"),
    ],
    ids=["faithful", "prompt-spaces-off"],
)
def test_a_fix_turned_off_is_refused_when_no_bound_stage_reproduces_its_quirk(
    tmp_path: Path, bench: Path, changes: dict[str, Any], match: str
) -> None:
    data = recipe_data(DIRECTIONS[0], stages=["noop"], prompts=P0_PROMPT_SET, context=None, **changes)
    assert_refused_before_anything_runs(tmp_path, bench, "unreproduced-fix", data, match)


def test_fixes_on_keep_the_file_block_path_with_no_fence_quirk(tmp_path: Path, bench: Path) -> None:
    direction = DIRECTIONS[0]
    data = recipe_data(direction, faithful=False, prompts=P0_PROMPT_SET, context=None, stages=["generate"])
    reply = render_file_blocks({target_file(direction): CODE})
    assert reply.startswith("```cuda\n"), "the P0 FILE block for a .cu file is cuda-tagged"
    outcome = run_one(tmp_path, bench, "fixes-on", data, [reply])
    fields = {
        "source_language": direction.source,
        "target_language": direction.target,
        "source_files": render_file_blocks({item_file(direction.source): OMP_SOURCE}),
        "target_files": target_file(direction),
    }
    assert outcome.messages(0) == [("user", render(P0_PROMPT_SET, "generate", fields))]
    attempt = outcome.trial.attempts[0]
    assert attempt.files == {target_file(direction): CODE}
    assert attempt.stage_reached == PARSED
    assert fence_quirks(outcome.trial) == []
