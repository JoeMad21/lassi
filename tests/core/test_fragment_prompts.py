"""Tests for fragment prompt sets beyond the faithful path of test_faithful_generation.py (task P1.4).

The fragment set and packs here are synthetic: short marker texts written
under a temporary assets root with their manifests, so these tests need no
upstream checkout and copy no upstream text (OQ-018). They cover the pure
helpers of lassi.core.fragments, the runner's checks of a fragment set
(a missing fragment, a stage of the other prompt kind, a missing or doubled
pack, context with a template set, a context field no earlier stage fills,
an item with two source files), the generation prompt without context, a
reply with no fenced block, and a lone surrogate in a context reply. No
model, compiler, or upstream code runs, and no value here is a measurement.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

import lassi.prompts.assets as prompt_assets
from lassi.bench import Direction, load_suite
from lassi.core import fragments
from lassi.core import runner as runner_module
from lassi.core.interfaces import Completion, Message, Sampling
from lassi.core.record import make_trial_id
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.executors import NoneExecutor

REPO = Path(__file__).resolve().parents[2]
SUITE_MANIFEST = REPO / "assets" / "bench" / "lassi-hecbench-10.yaml"
SUITE = "lassi-hecbench-10"
ITEM = "layout"
MODEL_ID = "scripted-fixture"
SET = "synthetic-fragments"
OMP_TO_CUDA = Direction("omp", "cuda")
SOURCE = "int  main() {\r\n    return 0;\r\n}\r\n"

# A synthetic fragment set: every key the stages read, each a short marker text.
FRAGMENTS = {
    **{key: f"<{key}>" for key in fragments.GENERATE_KEYS},
    **{key: f"<{key}>" for key in fragments.SUMMARY_KEYS + fragments.DESCRIPTION_KEYS},
}
FRAGMENTS = {fragments.fragment_key(key, OMP_TO_CUDA): text for key, text in FRAGMENTS.items()}


def write_tree(tree: Path, texts: Mapping[str, str]) -> None:
    """Write a manifest tree: one `<key>.txt` per entry and MANIFEST.yaml with each file's sha256."""
    tree.mkdir(parents=True)
    entries = []
    for key, text in texts.items():
        data = text.encode("ascii")
        (tree / f"{key}.txt").write_bytes(data)
        entries.append({"key": key, "source": "synthetic", "sha256": hashlib.sha256(data).hexdigest()})
    (tree / "MANIFEST.yaml").write_bytes(yaml.safe_dump({"entries": entries}).encode("ascii"))


@pytest.fixture
def assets_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a temporary assets root with the synthetic set and three packs; the loader reads it by default."""
    root = tmp_path / "assets"
    write_tree(root / "prompts" / SET, FRAGMENTS)
    write_tree(root / "context" / "cuda-pack", {"packdict.cuda": "<cuda pack>"})
    write_tree(root / "context" / "other-cuda-pack", {"otherdict.cuda": "<other cuda pack>"})
    write_tree(root / "context" / "omp-pack", {"packdict.omp": "<omp pack>"})
    monkeypatch.setattr(prompt_assets, "default_root", lambda: root)
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS"):
        monkeypatch.delenv(name, raising=False)
    return root


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    """Return a bench root holding a synthetic CRLF source for each language of the item."""
    item = load_suite(SUITE_MANIFEST).items[ITEM]
    for spec in item.languages.values():
        path = tmp_path / "bench" / spec.dir / spec.files[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(SOURCE.encode("ascii"))
    return tmp_path / "bench"


class FakeToolchain:
    """A toolchain the recipe can bind; these tests refuse every plan that would build, so it never builds."""

    name = "fake-cc"
    capabilities = frozenset({"diagnostics"})


class TemplateOnlyStage:
    """A stage that renders a template and reads no fragment, so a fragment prompt set cannot serve it; never run."""

    name = "template_only"
    capabilities: frozenset[str] = frozenset()
    requires: dict[str, frozenset[str]] = {}
    prompt_fields = {"correct": ("target_language",)}

    def __init__(self, *, context: Any) -> None:
        """Keep the trial's run context."""
        self.context = context

    def __call__(self, trial: Any) -> Any:
        """Return `trial` unchanged."""
        return trial

    def describe(self) -> str:
        """Return a one-line description of the stage."""
        return "template_only: renders correct.txt"


def run(
    tmp_path: Path, bench_root: Path, replies: Sequence[str], **changes: Any
) -> tuple[Path, list[list[Message]]]:
    """Run a one-trial omp to cuda recipe on the synthetic set; return the run directory and the requests."""
    requests: list[list[Message]] = []

    class Scripted:
        """Answers each request with the next reply."""

        name = "scripted"
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Record the request and return the next reply."""
            requests.append(list(messages))
            return Completion(text=replies[len(requests) - 1], prompt_tokens=0, completion_tokens=0)

    # generate reproduces fence_tag and prompt_spaces; the other fixes stay on, since no baseline stage is listed.
    data: dict[str, Any] = {
        "extends": "base",
        "faithful": False,
        "fixes": {"fence_tag": False, "prompt_spaces": False},
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 64}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": "omp", "target": "cuda"}],
        "prompts": SET,
        "stages": ["generate"],
        "executor": {"kind": "none"},
        "trials": {"n": 1},
        **changes,
    }
    registry = Registry()
    registry.register("LLMBackend", "scripted", Scripted)
    registry.register("Executor", "none", NoneExecutor)
    registry.register("Toolchain", "fake-cc", FakeToolchain)
    registry.register("Stage", "template_only", TemplateOnlyStage)
    for name in {*data["stages"], "compile_loop"} - {"template_only"}:
        registry.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    path = tmp_path / "fragment-run.yaml"
    path.write_bytes(yaml.safe_dump(data).encode("ascii"))
    options = RunOptions(runs_root=tmp_path / "runs-root", run_id="r", bench_root=bench_root, registry=registry)
    return run_recipe(path, options), requests


# ---------------------------------------------------------------------------
# Pure helpers


def test_first_fence_cases() -> None:
    assert fragments.first_fence("no block here") == fragments.Fence(found=False, text="")
    cuda = fragments.first_fence("a\n```cuda\nint x;\n```\n")
    assert (cuda.text, cuda.removed, cuda.quirk) == ("uda\nint x;\n", "c", True)
    joined = fragments.first_fence("```cppcode\n```")
    assert (joined.text, joined.removed, joined.quirk) == ("ode\n", "cppc", True)
    spaced = fragments.first_fence("```c \nint x;\n```")
    assert (spaced.text, spaced.quirk) == (" \nint x;\n", False)
    other = fragments.first_fence("```python\nx = 1\n```")
    assert (other.text, other.removed, other.quirk) == ("python\nx = 1\n", "", False)
    assert "uda" in fragments.fence_quirk_message(cuda) and fragments.fence_quirk_message(cuda).isascii()


def test_text_helpers() -> None:
    assert fragments.collapse_spaces("a  b\t\t c   \n  d") == "a b\t\t c \n d"
    assert fragments.as_text_mode("a\r\nb\rc\n") == "a\nb\nc\n"
    assert fragments.fragment_key(fragments.DIRECTION_SYSTEM, OMP_TO_CUDA) == "system_prompt_dict.OMP_to_CUDA"
    assert fragments.fragment_key(fragments.SUMMARY_NOTE, OMP_TO_CUDA) == "summarize_context.cuda_note"
    assert fragments.pack_language("codeknowledge_dict.omp") == "omp"


# ---------------------------------------------------------------------------
# The runner's checks of a fragment set


def test_generate_without_context_uses_the_no_context_lead(tmp_path: Path, bench: Path, assets_root: Path) -> None:
    run_dir, requests = run(tmp_path, bench, ["```\nint main() {}\n```"])
    f = FRAGMENTS
    source = "int main() {\n return 0;\n}\n"
    user = (
        f["system_prompt_dict.general_system"] + f["generate.no_context_lead"] + f["generate.request_lead"]
        + f["codetranslate_prompt_dict.OMP_to_CUDA"] + " " + source
    )
    expected = [("system", f["system_prompt_dict.OMP_to_CUDA"]), ("user", user)]
    assert [(m.role, m.content) for m in requests[0]] == expected


def test_a_reply_with_no_fence_is_s0_with_an_empty_file(tmp_path: Path, bench: Path, assets_root: Path) -> None:
    run_dir, _ = run(tmp_path, bench, ["no code at all"])
    trial_id = make_trial_id("fragment-run", MODEL_ID, SUITE, "omp-cuda", ITEM, 1)
    attempt = read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir)).attempts[0]
    assert attempt.stage_reached == "S0"
    assert attempt.files == {"main.cu": ""}
    assert [(d.stage, d.severity, d.code) for d in attempt.diagnostics] == [("parse", "warning", "no-fence")]


def test_a_lone_surrogate_in_a_context_reply_becomes_a_replacement_character_and_a_warning(
    tmp_path: Path, bench: Path, assets_root: Path
) -> None:
    # json.loads turns the escape "\\ud83d" in an HTTP reply body into a lone surrogate, which is not Unicode text.
    summary = "summary " + chr(0xD83D) + " end"
    description = "description " + chr(0xDC00) + chr(0xDC01) + " end"
    stage_list = ["summarize_context", "describe_source", "generate"]
    replies = [summary, description, "```\nint main() {}\n```"]
    run_dir, requests = run(tmp_path, bench, replies, stages=stage_list, context=["cuda-pack"])
    assert (run_dir / "run.md").is_file(), "the run finishes"
    trial_id = make_trial_id("fragment-run", MODEL_ID, SUITE, "omp-cuda", ITEM, 1)
    trial = read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir))
    replaced = chr(0xFFFD)
    assert trial.context.knowledge_summary == "summary " + replaced + " end"
    assert trial.context.source_description == "description " + replaced + replaced + " end"
    sent = requests[2][1].content
    assert trial.context.knowledge_summary in sent and trial.context.source_description in sent
    (attempt,) = trial.attempts
    assert attempt.stage_reached == "S1"
    assert attempt.diagnostics == [], "a context reply's warning rides on its request (P2.1), not on attempt 0"
    assert trial.requests is not None
    notes = [[(d.stage, d.severity, d.code) for d in request.diagnostics] for request in trial.requests]
    invalid = ("parse", "warning", "invalid-text")
    assert notes == [[invalid], [invalid], []], "one warning on each context request whose reply held a surrogate"
    messages = [request.diagnostics[0].message for request in trial.requests[:2]]
    assert "held 1 lone surrogate" in messages[0] and "Trial.context.knowledge_summary" in messages[0]
    assert "held 2 lone surrogate" in messages[1] and "Trial.context.source_description" in messages[1]


def test_context_without_a_replacement_character_adds_no_warning(
    tmp_path: Path, bench: Path, assets_root: Path
) -> None:
    stage_list = ["summarize_context", "describe_source", "generate"]
    replies = ["summary", "description", "```\nint main() {}\n```"]
    run_dir, _ = run(tmp_path, bench, replies, stages=stage_list, context=["cuda-pack"])
    trial_id = make_trial_id("fragment-run", MODEL_ID, SUITE, "omp-cuda", ITEM, 1)
    trial = read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir))
    (attempt,) = trial.attempts
    assert attempt.diagnostics == []
    assert trial.requests is not None and all(request.diagnostics == [] for request in trial.requests)


# A synthetic suite whose one item has two OpenMP source files and one CUDA file.
TWO_SOURCES_MANIFEST = """\
suite: synth-two-sources
repo: https://example.invalid/synth-two-sources
commit: 0123456789abcdef0123456789abcdef01234567
items:
  alpha:
    split: eval
    languages:
      omp: {dir: src/alpha-omp, files: [a.cpp, b.cpp]}
      cuda: {dir: src/alpha-cuda, files: [main.cu]}
"""


def test_a_fragment_set_refuses_an_item_with_more_than_one_source_file(
    tmp_path: Path, assets_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "synth-two-sources.yaml").write_bytes(TWO_SOURCES_MANIFEST.encode("ascii"))
    monkeypatch.setattr(runner_module, "BENCH_DIR", manifests)
    sources = tmp_path / "synth-bench" / "src"
    for relative in ("alpha-omp/a.cpp", "alpha-omp/b.cpp", "alpha-cuda/main.cu"):
        (sources / relative).parent.mkdir(parents=True, exist_ok=True)
        (sources / relative).write_bytes(SOURCE.encode("ascii"))
    changes = {"bench": {"suite": "synth-two-sources", "split": "eval", "items": ["alpha"]}}
    with pytest.raises(RunError, match=r"synth-two-sources/alpha \(omp-cuda\) has 2 'omp' source files"):
        run(tmp_path, sources.parent, [], **changes)
    assert not (tmp_path / "runs-root").exists()


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"context": ["cuda-pack", "other-cuda-pack"]}, r"'cuda-pack' and 'other-cuda-pack' both serve 'cuda'"),
        ({"stages": ["summarize_context", "generate"], "context": ["omp-pack"]}, r"context pack for .*'cuda'"),
        ({"stages": ["generate", "template_only"]}, r"'template_only' renders templates"),
        (
            {"stages": ["generate", "compile_loop"], "toolchain": {"cuda": "fake-cc"}},
            r"'compile_loop': the prompt set 'synthetic-fragments' has no fragment correct\.compile_error_head",
        ),
        ({"stages": ["summarize_context", "generate"], "prompts": "p0-smoke"}, r"is a template prompt set"),
        ({"directions": [{"source": "cuda", "target": "omp"}]}, r"no fragment .*CUDA_to_OMP"),
        ({"prompts": "p0-smoke", "context": ["cuda-pack"]}, r"only a fragment prompt set uses context packs"),
        (
            {"context": ["cuda-pack"]},
            r"'generate' joins the Trial\.context field\(s\) knowledge_summary, source_description into its prompt "
            r"when a context pack serves 'cuda', but no earlier listed stage fills them",
        ),
        (
            {"stages": ["summarize_context", "generate", "describe_source"], "context": ["cuda-pack"]},
            r"'generate' joins the Trial\.context field\(s\) source_description into",
        ),
    ],
    ids=["two-packs-one-language", "no-pack-for-target", "templates-in-fragment-set",
         "correction-fragments-missing", "fragments-in-template-set", "missing-fragment",
         "context-with-template-set", "pack-without-context-stages", "context-stage-too-late"],
)
def test_the_runner_refuses_a_fragment_plan_that_cannot_work(
    tmp_path: Path, bench: Path, assets_root: Path, changes: dict[str, Any], match: str
) -> None:
    if changes.get("prompts") == "p0-smoke":
        changes = {**changes, "faithful": False}
    with pytest.raises(RunError, match=match):
        run(tmp_path, bench, [], **changes)
    assert not (tmp_path / "runs-root").exists()
