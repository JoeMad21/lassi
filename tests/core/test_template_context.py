"""Tests for task P17.21: the API reference (the context stage) under a template prompt set.

Bible: Component Interfaces (the Stage row: summarize_context, generate,
compile_loop, run_loop), Harness Contract (what a prompt may show), Design
Principle 3 (prompts are data files), Agent Rule 5 (no Tier A item, kernel, or
file enters a prompt template). plans/p17-portable.md, task P17.21; the owner's
direction of 2026-10-23 ("Okay, get that built and added.").

The contract these tests fix (the P17.21 design, points 1 to 5):

1. A template prompt set may serve context packs when it holds the templates
   summarize_context.txt and knowledge.txt. The runner still refuses context
   packs with a template set that lacks either one, before any directory
   exists, and its message names the missing template. A fragment set
   behaves as before.
2. Under a template set, summarize_context sends one user message and no
   system prompt: summarize_context.txt rendered with $target_language and
   $context_pack (the target language's pack, its entries joined as the
   loader joins them). The reply fills Trial.context.knowledge_summary
   through the existing context-reply path (the request is recorded under
   summarize_context with no attempt; a request past the model's context ends
   the trial at context-exceeded). describe() names the template.
3. When the summary was filled, every later model prompt (generate.txt,
   compile_loop's and run_loop's correct.txt, run_loop's
   correct_mismatch.txt) gets knowledge.txt rendered with $knowledge_summary
   before it, separated by one blank line. Without a summary each prompt is
   byte for byte what it was before the task (the golden prompts below),
   also when the recipe names a pack but lists no summarize_context.
4. assets/prompts/tt-host-v0 gains the two templates: plain ASCII, generic,
   naming no Tier A item, kernel, or file; the set stays a template set (no
   MANIFEST.yaml).
5. The runner renders both templates with their fields, before any
   directory exists, when summarize_context is listed or the recipe names
   packs under a template set, and refuses one that does not render.

No new recipe key: committed recipes keep their recipe hashes.

How the expectations are built. The two new templates' wording is the
implementer's: every expected prompt that holds one is rendered from the
template file itself (lassi.prompts.render), never written out here. A block
"before the prompt, separated by one blank line" is compared as
`<block without its trailing line feeds>` + "\\n\\n" + `<prompt>`; the
SYNTHETIC summary ends without a line feed, so a template that ends in one
line feed reads the same either way. The runs read the assets root through
lassi.prompts.assets.default_root() and prompt sets through
lassi.prompts.default_roots(), the two functions that name them, which the
fixtures point at a temporary root holding a copy of the repository's
tt-host-v0 and the SYNTHETIC pack.

Provenance of the fixed values. The golden prompts under
tests/core/golden/template-prompts/ were written by running this module's
off flow (test_without_either_feature_the_prompts_match_the_goldens) through
lassi.core.stages at commit 6bb3232 on 2026-10-23, before either feature
existed; the recipe hashes were computed by lassi.core.recipe at the same
commit and date (the working tree's recipe.py then held another change's
model.checkpoint keys, which these recipes do not set). A prompt and a hash
are functions of their inputs, not measurements.

The suite, programs, kernel, pack, summary, replies, and arrays are
SYNTHETIC; fake toolchains compile nothing and the scripted executor starts
no process. No value here is a measurement.
"""

from __future__ import annotations

import copy
import hashlib
import re
import shutil
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

import lassi.prompts as prompts_module
from lassi.bench import Direction
from lassi.bench.registry import load_suite
from lassi.core import runner as runner_module
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, ContextExceeded, Limits, Message, RunResult, Sampling
from lassi.core.recipe import Recipe, load_recipe
from lassi.core.record import Diagnostic, Trial, make_trial_id
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.stages import RunContext, SummarizeContextStage
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.harness.lassi_io import read_array, write_array
from lassi.prompts import assets as prompt_assets
from lassi.prompts import render

REPO = Path(__file__).resolve().parents[2]
TEMPLATE_SET = REPO / "assets" / "prompts" / "tt-host-v0"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "template-prompts"
PROMPTS = "tt-host-v0"
SUITE = "template-features"
ITEM = "scale"
CPP, TT = "cpp", "tt"
DIRECTION = Direction(CPP, TT)
TOOLCHAINS = {CPP: "fake-cpp", TT: "fake-tt"}
STAGES = ["baseline", "generate", "compile_loop", "run_loop", "oracle"]
CONTEXT_STAGES = ["baseline", "summarize_context", "generate", "compile_loop", "run_loop", "oracle"]
MODEL_ID = "scripted-fixture"
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
# A commit id for the synthetic manifest; not a commit of this repository or of any source.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
# The held-out input spec and tolerance of tests/core/test_correct_on_mismatch.py, whose oracle-mismatch report
# these runs reach too.
SEED = 9876543210123
INPUTS = [{"name": "src0", "dtype": "f32", "shape": [16], "dist": "uniform", "lo": 4096.0, "hi": 4097.0}]
OUTPUT = "result"
TOLERANCE = {"metric": "max_abs", "threshold": 0.05}
AGREEMENT = 2.0**-8
OFFSET = 1.5
ORACLE = {"kind": "binary_io", "metric": "max_abs", "threshold": "from_baseline"}

# The item's tt kernel, a support file under a kernels directory, so kernel_files shows it; and its programs.
KERNEL = f"{ITEM}/kernels/compute/scale_tiles.cpp"
KERNEL_UPSTREAM = "upstream/scale_tiles.cpp"
KERNEL_TEXT = (
    "// SYNTHETIC compute kernel\n"
    "void kernel_main() {\n"
    "    cb_wait_front(tt::CBIndex::c_0, 1);\n"
    "    zorbquant_tile(0);\n"
    "}\n"
)
SOURCE_TEXT = '// SYNTHETIC source\n#include "lassi_io.h"\nint main() { return lassi_io_read(nullptr, nullptr); }\n'
REFERENCE_TEXT = "// SYNTHETIC reference\nint main() { return 0; }\n"

# The SYNTHETIC context pack: two entries serving tt, joined by the loader in manifest order.
PACK = "syn-tt-pack"
PACK_ENTRIES = {
    "guide.tt": "SYNTHETIC guide: a host program opens the device first.\n",
    "host_api.tt": "SYNTHETIC header: CBHandle CreateCircularBuffer(Program&, CoreCoord, const Config&);\n",
}
PACK_TEXT = prompt_assets.PACK_JOINER.join(PACK_ENTRIES.values())
# The model's SYNTHETIC summary of the pack; it ends without a line feed (see the module docstring).
SUMMARY = "SYNTHETIC summary: call CreateCircularBuffer(program, core, config) before CreateKernel."

# The run error a crashed attempt reports.
RUN_STATUS = 3
RUN_STDERR = "SYNTHETIC TT_FATAL: CreateKernel could not find the kernel file\n"

# The fake toolchain's directives: a line `#error <message>` or `#warning <message>` in a file becomes a compile
# diagnostic at that line; a line BARE_ERROR becomes an error with no location and an empty message.
ERROR, WARNING, BARE_ERROR = "#error ", "#warning ", "#error-bare"

# The four model prompts of a template-set run without either feature, in request order, by golden file name.
PROMPT_KINDS = ("generate", "compile-correction", "run-correction", "mismatch-correction")
REQUEST_STAGES = ["generate", "compile_loop", "run_loop", "run_loop"]

# Committed recipes that use tt-host-v0 and neither feature, with their recipe hashes (see the module docstring).
RECIPE_HASHES = {
    "projects/lassi-demo/tier-a-rngd-maxabs.yaml":
        "8bdaeac1d6b46ef4293c7187a831b5f3c6b054ef2bf02e1161fd9e8f8ca4e992",
    "projects/lassi-demo/sweep3/llama31-8b-maxabs.yaml":
        "4ebf6f98594a84e82e13a40f71b1ad32c3da46f300e630afe97000f23ca348b0",
    "projects/lassi-demo/sweep3/qwen3-coder-30b-pcc.yaml":
        "797d4ac06582344590697d65f03a9b46e612627fd2f12c69327ba3650e1402c6",
}

# SYNTHETIC stand-ins for the new templates, for sets built here to lack or break one of them.
SYN_TEMPLATES = {
    "summarize_context": "SYNTHETIC: summarize the $target_language reference below.\n\n$context_pack\n",
    "knowledge": "SYNTHETIC: your earlier summary of the reference follows.\n$knowledge_summary\n",
    "retrieved": "SYNTHETIC: documentation excerpts follow.\n\n$passages\n",
    "passage": "$path:$start-$end ($heading)\n$text\n",
}
# A placeholder no stage fills.
UNFILLED = "$unfilled_field"

Values = list[float]
# What one scripted run does: its exit status, its stderr, and its output files (file name -> f32 values).
Ran = tuple[int, str, dict[str, Values]]
Behavior = Callable[[Values], Ran]


def doubled(values: Values) -> Ran:
    """Return the target reference's run, and a correct candidate's: result = 2 * src0."""
    return 0, "", {f"{OUTPUT}.lassiio": [2.0 * value for value in values]}


def agreeing(values: Values) -> Ran:
    """Return the C++ reference's run: 2 * src0, with AGREEMENT added to the first element."""
    result = [2.0 * value for value in values]
    result[0] += AGREEMENT
    return 0, "", {f"{OUTPUT}.lassiio": result}


def wrong(values: Values) -> Ran:
    """Return a clean run whose output misses the tolerance: 2 * src0 + OFFSET."""
    return 0, "", {f"{OUTPUT}.lassiio": [2.0 * value + OFFSET for value in values]}


def crashed(values: Values) -> Ran:
    """Return a failed run: exit status RUN_STATUS, RUN_STDERR, and no output file."""
    return RUN_STATUS, RUN_STDERR, {}


def program(number: int, *lines: str) -> str:
    """Return a SYNTHETIC reply: one FILE block of the target's one file, numbered, holding `lines` first."""
    body = "".join(f"{line}\n" for line in lines)
    return render_file_blocks({f"{ITEM}.cpp": f"// SYNTHETIC candidate {number}\n{body}int main() {{ return 0; }}\n"})


# A reply that does not build: two errors and a warning between them, each naming a tt-metal call.
BROKEN = program(
    0,
    f"{ERROR}use of undeclared identifier 'CreateCircularBuffer'",
    f"{WARNING}SYNTHETIC unused variable 'cb'",
    f"{ERROR}no member named 'EnqueueWriteMeshBuffer' in namespace 'tt::tt_metal'",
)


def directive(path: str, number: int, line: str) -> Diagnostic | None:
    """Return the compile diagnostic the fake toolchain reads from line `number` of file `path`, or None."""
    if line == BARE_ERROR:
        return Diagnostic(stage="compile", severity="error", code="synthetic", message="")
    for prefix, severity in ((ERROR, "error"), (WARNING, "warning")):
        if line.startswith(prefix):
            return Diagnostic(stage="compile", severity=severity, code="synthetic", file=path, line=number,
                              message=line[len(prefix):])
    return None


# ---------------------------------------------------------------------------
# The SYNTHETIC suite, assets, and fakes


def write_suite(base: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the one-item manifest (a tt kernel among the tt support files) and its sources; return the bench root."""
    kernel = {"upstream": KERNEL_UPSTREAM, "sha256": hashlib.sha256(KERNEL_TEXT.encode("ascii")).hexdigest()}
    languages = {
        CPP: {"dir": "src/cpp", "files": [f"{ITEM}.cpp"]},
        TT: {"dir": "src/tt", "files": [f"{ITEM}.cpp"], "support": {KERNEL: kernel}},
    }
    item = {"split": "eval", "seed": SEED, "inputs": [dict(spec) for spec in INPUTS], "outputs": [OUTPUT],
            "tolerance": TOLERANCE, "languages": languages}
    data = {"suite": SUITE, "repo": "https://example.invalid/template-features", "commit": FAKE_COMMIT,
            "items": {ITEM: item}}
    manifests = base / "manifests"
    manifests.mkdir(parents=True)
    (manifests / f"{SUITE}.yaml").write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    monkeypatch.setattr(runner_module, "BENCH_DIR", manifests)
    root = base / "bench"
    for path, text in ((f"src/cpp/{ITEM}.cpp", SOURCE_TEXT), (f"src/tt/{ITEM}.cpp", REFERENCE_TEXT),
                       (KERNEL_UPSTREAM, KERNEL_TEXT)):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(text.encode("ascii"))
    return root


def write_tree(tree: Path, texts: Mapping[str, str]) -> None:
    """Write a manifest tree: one `<key>.txt` per entry and a MANIFEST.yaml with each file's sha256."""
    tree.mkdir(parents=True)
    entries = []
    for key, text in texts.items():
        data = text.encode("ascii")
        (tree / f"{key}.txt").write_bytes(data)
        entries.append({"key": key, "source": "synthetic", "sha256": hashlib.sha256(data).hexdigest()})
    (tree / prompt_assets.MANIFEST).write_bytes(yaml.safe_dump({"entries": entries}, sort_keys=False).encode("ascii"))


def make_assets(base: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a temporary assets root with a copy of tt-host-v0 and the SYNTHETIC pack, which the runner then reads.

    The copy holds whatever templates the repository's tt-host-v0 holds when
    the test runs, so the real templates are the ones the runs render.
    """
    root = base / "assets"
    shutil.copytree(TEMPLATE_SET, root / "prompts" / PROMPTS, ignore=shutil.ignore_patterns("__pycache__"))
    write_tree(root / "context" / PACK, PACK_ENTRIES)
    monkeypatch.setattr(prompt_assets, "default_root", lambda: root)
    monkeypatch.setattr(prompts_module, "default_roots", lambda: (root / "prompts", REPO / "assets" / "prompts"))
    return root


def template_set(assets: Path, name: str, *, drop: Sequence[str] = (), texts: Mapping[str, str] | None = None) -> str:
    """Write the template set `name` under `assets`: tt-host-v0's templates, less `drop`, plus `texts`; return name."""
    target = assets / "prompts" / name
    shutil.copytree(TEMPLATE_SET, target, ignore=shutil.ignore_patterns("__pycache__"))
    for prompt, text in (texts or {}).items():
        (target / f"{prompt}.txt").write_bytes(text.encode("ascii"))
    for prompt in drop:
        (target / f"{prompt}.txt").unlink(missing_ok=True)
    return name


@dataclass
class Script:
    """What the fakes answer and run: the replies in order, each attempt's run, and what they saw and returned.

    A reply that is an exception is raised by the backend instead of answered.
    """

    replies: list[str | BaseException]
    attempts: list[Behavior]
    references: dict[str, Behavior] = field(default_factory=lambda: {TT: doubled, CPP: agreeing})
    requests: list[list[Message]] = field(default_factory=list)
    runs: list[RunResult] = field(default_factory=list)


def f32_values(path: Path) -> Values:
    """Return the elements of the f32 lassi_io file at `path`."""
    array = read_array(path)
    assert array.dtype == "f32"
    return list(struct.unpack(f"<{len(array.data) // 4}f", array.data))


class FakeToolchain:
    """Writes the files; each directive line is a diagnostic, and an error fails the build. Compiles nothing."""

    name = "fake"
    capabilities = frozenset({"diagnostics"})

    def build(
        self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
    ) -> BuildResult:
        """Write `files`; return their directives' diagnostics, and a PLACEHOLDER artifact when none is an error."""
        diagnostics: list[Diagnostic] = []
        for path, text in sorted(files.items()):
            (Path(workdir) / path).parent.mkdir(parents=True, exist_ok=True)
            (Path(workdir) / path).write_bytes(text.encode("utf-8"))
            for number, line in enumerate(text.splitlines(), start=1):
                found = directive(path, number, line)
                if found is not None:
                    diagnostics.append(found)
        if any(item.severity == "error" for item in diagnostics):
            return BuildResult(artifact=None, diagnostics=diagnostics)
        artifact = Path(workdir) / "main"
        artifact.write_bytes(b"PLACEHOLDER artifact\n")
        return BuildResult(artifact=artifact, diagnostics=diagnostics)


def scripted_executor(script: Script) -> type:
    """Return an Executor class that reads the held-out input and runs the scripted behavior; it runs nothing."""

    class ScriptedExecutor:
        """Computes SYNTHETIC outputs from the input file in its arguments; each attempt run is kept in the script."""

        name = "scripted"
        capabilities = frozenset({"runs_code", "sandboxed"})

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Read @inputs/src0.lassiio, write the behavior's outputs beside the artifact, and return its run."""
            workdir = Path(artifact).parent
            kind = workdir.parent.name
            values = f32_values(workdir / inputs[0])
            reference = kind.startswith("baseline-")
            behavior = script.references[kind.removeprefix("baseline-")] if reference else script.attempts.pop(0)
            status, stderr, outputs = behavior(values)
            files = {}
            for file, result in outputs.items():
                write_array(workdir / file, OUTPUT, "f32", (len(result),), struct.pack(f"<{len(result)}f", *result))
                files[file] = workdir / file
            run = RunResult(exit_code=status, hang=False, stdout="SYNTHETIC\n", stderr=stderr, output_files=files,
                            wall_s=1.0)
            if not reference:
                script.runs.append(run)
            return run

    return ScriptedExecutor


def scripted_backend(script: Script) -> type:
    """Return an LLMBackend class that answers from the script in order and records every request."""

    class ScriptedBackend:
        """Answers each request with the next scripted reply, or raises it; a request past the script fails."""

        name = "scripted"
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Record the request and return the next scripted reply (raising it when it is an exception)."""
            script.requests.append(list(messages))
            assert script.replies, "the backend was asked for more replies than the script holds"
            reply = script.replies.pop(0)
            if isinstance(reply, BaseException):
                raise reply
            return Completion(text=reply, prompt_tokens=0, completion_tokens=0)

    return ScriptedBackend


def make_registry(script: Script) -> Registry:
    """Return a test Registry: the scripted backend and executor, fake toolchains, the real stages and Oracles."""
    registry = Registry()
    registry.register("LLMBackend", "scripted", scripted_backend(script))
    registry.register("Executor", "scripted", scripted_executor(script))
    for name in TOOLCHAINS.values():
        registry.register("Toolchain", name, FakeToolchain)
    for name in ("binary_io", "stdout_mask"):
        registry.register("Oracle", name, DEFAULT_REGISTRY.get("Oracle", name).factory)
    for name in CONTEXT_STAGES:
        registry.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    return registry


def recipe_data(**changes: Any) -> dict[str, Any]:
    """Return a one-trial cpp -> tt recipe on tt-host-v0 with loop.correct_on_mismatch on; `changes` replace keys."""
    data: dict[str, Any] = {
        "extends": "base",
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": SAMPLING.max_tokens}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": CPP, "target": TT}],
        "prompts": PROMPTS,
        "toolchain": dict(TOOLCHAINS),
        "fixes": {"baseline_both": True},
        "stages": list(STAGES),
        "executor": {"kind": "scripted"},
        "oracle": dict(ORACLE),
        "loop": {"correct_on_mismatch": True},
        "trials": {"n": 1},
    }
    data.update(copy.deepcopy(changes))
    return data


def write_recipe(base: Path, data: Mapping[str, Any], name: str) -> Path:
    """Write `data` as <base>/<name>.yaml and return its path."""
    path = base / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


def runs_root(base: Path) -> Path:
    """Return the runs root every run of a test uses."""
    return base / "runs-root"


def run(base: Path, bench: Path, script: Script, data: Mapping[str, Any], name: str) -> tuple[Path, Trial]:
    """Run recipe `data` as <base>/<name>.yaml (project and run id `name`); return the run directory and trial 1."""
    options = RunOptions(runs_root=runs_root(base), run_id=name, bench_root=bench, registry=make_registry(script))
    run_dir = run_recipe(write_recipe(base, data, name), options)
    trial_id = make_trial_id(name, MODEL_ID, SUITE, f"{CPP}-{TT}", ITEM, 1)
    return run_dir, read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir))


def refused(base: Path, bench: Path, data: Mapping[str, Any], name: str = "refused") -> str:
    """Run `data`, expect a RunError before any directory exists, and return its message."""
    options = RunOptions(runs_root=runs_root(base), run_id=name, bench_root=bench,
                         registry=make_registry(Script(replies=[], attempts=[])))
    with pytest.raises(RunError) as caught:
        run_recipe(write_recipe(base, data, name), options)
    assert not (runs_root(base) / "runs" / name).exists(), "a refused run creates no directory"
    return str(caught.value)


def user_prompts(script: Script) -> list[str]:
    """Return each request's one message, which under a template set is the user message with no system prompt."""
    prompts = []
    for request in script.requests:
        assert [message.role for message in request] == ["user"], "a template set sends one user message"
        prompts.append(request[0].content)
    return prompts


def full_script(*prefix: str | BaseException) -> Script:
    """Return the four-prompt script: a compile error, a failed run, an oracle mismatch, then a pass.

    `prefix` replies come first (a summary, say). Attempt 0 does not build
    (compile_loop asks), attempt 1 crashes (run_loop asks with the run
    error), attempt 2 runs clean but misses the tolerance (run_loop asks with
    the mismatch report), and attempt 3 passes.
    """
    return Script(replies=[*prefix, BROKEN, program(1), program(2), program(3)], attempts=[crashed, wrong, doubled])


def joined(blocks: Sequence[str], prompt: str) -> str:
    """Return `prompt` with each block before it, in order, each separated from the next by one blank line."""
    return "\n\n".join([*(block.rstrip("\n") for block in blocks), prompt])


def knowledge_block(summary: str = SUMMARY) -> str:
    """Return the prompt set's knowledge.txt rendered with `summary`."""
    return render(PROMPTS, "knowledge", {"knowledge_summary": summary})


def golden(kind: str) -> str:
    """Return the golden prompt of `kind` (PROMPT_KINDS), as recorded before either feature existed."""
    return (GOLDEN_DIR / f"{kind}.txt").read_bytes().decode("ascii")


def load(base: Path, data: Mapping[str, Any], name: str = "loaded") -> Recipe:
    """Load `data` written as <base>/<name>.yaml with a test registry."""
    return load_recipe(write_recipe(base, data, name), registry=make_registry(Script(replies=[], attempts=[])))


def stage_context(recipe: Recipe, prompts: str, **fields: Any) -> RunContext:
    """Return a RunContext for describe(): the recipe, the prompt set, and `fields`; nothing in it runs."""
    values: dict[str, Any] = {
        "recipe": recipe, "backend": None, "sampling": SAMPLING, "toolchains": {}, "executor": None, "store": None,
        "suite": None, "sources_root": Path("."), "item": ITEM, "direction": DIRECTION, "build_root": Path("."),
        "prompts": prompts, "max_corrections": 10,
    }
    values.update(fields)
    return RunContext(**values)


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate variables a test could inherit, and point TMPDIR at a test directory."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "scratch" / "tmp").mkdir(parents=True)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch" / "tmp"))


@pytest.fixture
def base(tmp_path: Path) -> Path:
    """Return the directory every run of a test writes under."""
    return tmp_path / "scratch"


@pytest.fixture
def assets(base: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return the temporary assets root (make_assets)."""
    return make_assets(base, monkeypatch)


@pytest.fixture
def bench(base: Path, monkeypatch: pytest.MonkeyPatch, assets: Path) -> Path:
    """Return the bench root of the SYNTHETIC suite, with the assets root in place."""
    return write_suite(base, monkeypatch)


# ---------------------------------------------------------------------------
# Neither feature: every prompt and hash as before (design point 3; no new key)


def test_without_either_feature_the_prompts_match_the_goldens(base: Path, bench: Path) -> None:
    script = full_script()
    _, trial = run(base, bench, script, recipe_data(), "plain")
    assert [request.stage for request in trial.requests or []] == REQUEST_STAGES
    assert trial.final.end_reason is None and trial.final.corrections == 3
    prompts = user_prompts(script)
    assert len(prompts) == len(PROMPT_KINDS)
    for kind, prompt in zip(PROMPT_KINDS, prompts, strict=True):
        assert prompt == golden(kind), f"the {kind} prompt changed with neither feature on"
    assert trial.context.knowledge_summary == ""


def test_a_pack_named_without_summarize_context_changes_no_prompt(base: Path, bench: Path) -> None:
    # The recipe names a pack, so the set must hold both templates, but no stage fills a summary to place.
    script = full_script()
    _, trial = run(base, bench, script, recipe_data(context=[PACK]), "packonly")
    assert [request.stage for request in trial.requests or []] == REQUEST_STAGES
    for kind, prompt in zip(PROMPT_KINDS, user_prompts(script), strict=True):
        assert prompt == golden(kind), f"the {kind} prompt changed with no summary to place"


@pytest.mark.parametrize("path", sorted(RECIPE_HASHES))
def test_a_committed_recipe_keeps_its_recipe_hash(path: str) -> None:
    recipe = load_recipe(REPO / path)
    assert recipe.data["prompts"] == PROMPTS and "context" not in recipe.data
    assert recipe.recipe_hash == RECIPE_HASHES[path]


# ---------------------------------------------------------------------------
# summarize_context under a template set (design points 1 and 2)


def test_summarize_context_sends_the_pack_as_one_user_message(base: Path, bench: Path) -> None:
    script = full_script(SUMMARY)
    run_dir, trial = run(base, bench, script, recipe_data(context=[PACK], stages=CONTEXT_STAGES), "summary")
    expected = render(PROMPTS, "summarize_context", {"target_language": TT, "context_pack": PACK_TEXT})
    assert script.requests[0] == [Message("user", expected)], "one user message, no system prompt"
    for text in PACK_ENTRIES.values():
        assert text in expected, "the request holds the whole pack"
    request = (trial.requests or [])[0]
    assert (request.stage, request.attempt_index) == ("summarize_context", None)
    store = TextStore(run_dir)
    assert [store.get(message.ref) for message in request.messages] == [expected]
    assert store.get(request.reply_ref) == SUMMARY
    assert trial.context.knowledge_summary == SUMMARY
    assert [item.stage for item in trial.requests or []] == ["summarize_context", *REQUEST_STAGES]


def test_a_summary_request_past_the_context_ends_the_trial(base: Path, bench: Path) -> None:
    script = Script(replies=[ContextExceeded(9000, 4096, 8192)], attempts=[])
    _, trial = run(base, bench, script, recipe_data(context=[PACK], stages=CONTEXT_STAGES), "exceeded")
    reason = trial.final.end_reason
    assert reason is not None and reason.code == "context-exceeded"
    assert reason.message.startswith("summarize_context"), reason.message
    assert trial.requests == [] and trial.attempts == [] and trial.context.knowledge_summary == ""
    assert len(script.requests) == 1, "no later stage asks the model"


def test_summarize_context_describes_its_template(base: Path) -> None:
    recipe = load(base, recipe_data(context=[PACK], stages=CONTEXT_STAGES))
    text = SummarizeContextStage(context=stage_context(recipe, PROMPTS, packs={TT: PACK_TEXT})).describe()
    assert "summarize_context.txt" in text, text
    fragments = {"system_prompt_dict.general_system": "<system>"}
    upstream = SummarizeContextStage(context=stage_context(recipe, "syn-fragments", fragments=fragments)).describe()
    assert upstream == "summarize_context: summarize the tt context pack (syn-fragments)", "a fragment set as before"


@pytest.mark.parametrize("listed", [True, False], ids=["stage-listed", "pack-only"])
@pytest.mark.parametrize("missing", ["summarize_context", "knowledge"])
def test_packs_are_refused_with_a_template_set_that_lacks_a_template(
    base: Path, bench: Path, assets: Path, missing: str, listed: bool
) -> None:
    other = "knowledge" if missing == "summarize_context" else "summarize_context"
    name = template_set(assets, f"lacks-{missing}", drop=[missing], texts={other: SYN_TEMPLATES[other]})
    stages = CONTEXT_STAGES if listed else STAGES
    message = refused(base, bench, recipe_data(prompts=name, context=[PACK], stages=stages))
    assert f"{missing}.txt" in message, message


# ---------------------------------------------------------------------------
# The summary before every later prompt (design point 3)


def test_the_summary_goes_before_every_later_prompt(base: Path, bench: Path) -> None:
    plain = full_script()
    run(base, bench, plain, recipe_data(), "plain")
    script = full_script(SUMMARY)
    _, trial = run(base, bench, script, recipe_data(context=[PACK], stages=CONTEXT_STAGES), "knowing")
    assert trial.final.end_reason is None and trial.final.corrections == 3
    knowledge = knowledge_block()
    assert SUMMARY in knowledge
    prompts = user_prompts(script)
    assert len(prompts) == 1 + len(PROMPT_KINDS)
    for kind, before, after in zip(PROMPT_KINDS, user_prompts(plain), prompts[1:], strict=True):
        assert after == joined([knowledge], before), f"the {kind} prompt does not carry the summary first"


# ---------------------------------------------------------------------------
# The runner renders both templates (design point 5)


@pytest.mark.parametrize("listed", [True, False], ids=["stage-listed", "pack-only"])
@pytest.mark.parametrize("broken", ["summarize_context", "knowledge"])
def test_a_template_that_does_not_render_with_its_fields_is_refused(
    base: Path, bench: Path, assets: Path, broken: str, listed: bool
) -> None:
    texts = {name: SYN_TEMPLATES[name] for name in ("summarize_context", "knowledge")}
    texts[broken] = f"{texts[broken]}{UNFILLED}\n"
    name = template_set(assets, f"broken-{broken}", texts=texts)
    stages = CONTEXT_STAGES if listed else STAGES
    message = refused(base, bench, recipe_data(prompts=name, context=[PACK], stages=stages))
    assert f"{broken}.txt" in message, message


# ---------------------------------------------------------------------------
# The two templates of tt-host-v0 (design point 4)

# Each new template of tt-host-v0 and the fields it renders with.
CONTEXT_TEMPLATES = {"summarize_context": ("target_language", "context_pack"), "knowledge": ("knowledge_summary",)}


def tier_a_names() -> set[str]:
    """Return the names a generic template may not hold: tt-pairs-v0's items, kernel files, and their stems."""
    suite = load_suite(REPO / "assets" / "bench" / "tt-pairs-v0.yaml")
    names = set(suite.items) | {"programming_examples"}
    for item in suite.items.values():
        files = set(item.support) | {file for spec in item.languages.values() for file in spec.support}
        for file in files:
            path = PurePosixPath(file)
            if path.name == "lassi_io.h":
                continue
            names.add(path.name)
            if len(path.stem) > 3:
                names.add(path.stem)
    return names


def named_tier_a(text: str) -> list[str]:
    """Return each Tier A name (tier_a_names) that `text` holds as a whole word, in any letter case, sorted."""
    return sorted(name for name in tier_a_names()
                  if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text, re.IGNORECASE))


def template_text(name: str) -> str:
    """Return tt-host-v0's template `name` as ASCII text; fail when the set lacks it."""
    path = TEMPLATE_SET / f"{name}.txt"
    assert path.is_file(), f"tt-host-v0 has no {name}.txt"
    data = path.read_bytes()
    assert data.isascii() and b"\r" not in data, f"{name}.txt is plain ASCII with LF line ends"
    return data.decode("ascii")


@pytest.mark.parametrize("name", sorted(CONTEXT_TEMPLATES))
def test_tt_host_v0_holds_the_template_and_it_renders_with_its_fields(name: str) -> None:
    template_text(name)
    fields = {field_name: f"<SYNTHETIC {field_name}>" for field_name in CONTEXT_TEMPLATES[name]}
    rendered = render(PROMPTS, name, fields)
    for field_name, value in fields.items():
        assert value in rendered, f"{name}.txt does not show ${field_name}"


@pytest.mark.parametrize("name", sorted(CONTEXT_TEMPLATES))
def test_tt_host_v0_template_is_generic(name: str) -> None:
    assert named_tier_a(template_text(name)) == [], f"{name}.txt names a Tier A item or file (Agent Rule 5)"


def test_tt_host_v0_stays_a_template_set() -> None:
    assert not (TEMPLATE_SET / prompt_assets.MANIFEST).exists(), "tt-host-v0 is rendered, not loaded as fragments"
