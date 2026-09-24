"""Tests for the faithful correction loop and the correction fixes (task P1.5, second half).

Bible: Source Papers (LASSI pipeline steps 1 and 4; quirk table, loop and
fence rows), Component Interfaces (Toolchain contract rules), Harness
Contract, Result Record (Final), Design Principle 4, Agent Rule 4.

The contract these tests fix, from the P1.5 acceptance criteria
(plans/p1-faithful.md) and the P1.5 Decision Log entries (baseline stage,
Trial.reference_run, final.end_reason; faithful correction prompt):

- Two new named fixes in lassi.core.recipe.FIXES, on by default and off under
  `faithful: true`, both named in compile_loop's `reproduces` (with
  `fence_tag`, since compile_loop reads correction replies too):
  `prompt_newlines` (off: every LF is removed from a correction prompt, as
  upstream does) and `parsed_diagnostics` (on: a correction prompt carries
  the parsed diagnostics, capped; off: it carries the whole raw compiler
  stderr attachment, read as a file opened in text mode reads it).
- Each faithful behavior is keyed on a named fix or on loop.max_corrections,
  never on the `faithful` flag: a recipe with `faithful: false`, every fix
  off, and a numeric loop.max_corrections sends the same requests and
  records the same trial as `faithful: true`, except that the loop stops at
  the cap. A cap hit (an error remains and the cap stops the loop) sets
  final.end_reason to code `correction-cap` with a message, under any prompt
  set; a trial whose last attempt compiles has no end reason.
- With a template prompt set (p0-smoke) the correction prompt is
  correct.txt, as in P0. With `parsed_diagnostics` on, $diagnostics holds
  one diagnostic_line per diagnostic, in order, whole lines only, at most
  DIAGNOSTIC_COUNT_CAP of them totalling at most DIAGNOSTIC_BYTES_CAP bytes
  (two int constants in the module that defines compile_loop). When any is
  left out, the prompt states how many as a decimal number; when none is,
  the prompt is exactly P0's. With it off, $diagnostics is the raw stderr.
  With `prompt_newlines` off, the rendered prompt loses every LF.
- With a fragment prompt set (lassi-2024), compile_loop declares its
  fragment keys (correct.* and setup.<target>.*), and a correction for a
  compile error sends [system, user]: the direction's system prompt, then
  the previous attempt's target file text, correct.compile_error_head, the
  setup.<target>.compiler fragment, one space, the setup.<target>.flags
  fragment, correct.compile_error_tail, the error text, and correct.outro.
  The error text is the raw stderr attachment when `parsed_diagnostics` is
  off. With `prompt_newlines` off every LF of the joined prompt is removed
  (a CR in the model's code stays, as upstream's re.sub on LF leaves it; the
  stderr has none left, since upstream reads it in text mode). Runs of
  spaces stay: correction prompts are never collapsed. Upstream's compiler
  text is used, never the pinned executable's path. The prompt is stored as
  the attempt's prompt_ref.
- With `fence_tag` off a correction reply is read as generate reads attempt
  0 (lassi.core.fragments.first_fence, `fence-quirk` and `no-fence`
  warnings), and an attempt with no fenced block (S0, an empty target file)
  is still built, as upstream compiles the empty file it writes.
- The pinned toolchain presets' flags equal upstream's flag text (the
  setup.<target>.flags fragments), and their compiler names match.

The expected prompts are computed from the fragments of a fresh extraction
of the pinned upstream checkout into a temporary directory
(tools/extract_lassi_assets.py); those tests skip, naming the tool, when
third_party/LASSI is absent or not at the pin. No upstream text and no
upstream compiler path is copied into this file (OQ-018). Model replies,
compiler stderr, and bench sources are SYNTHETIC texts written here; fake
toolchains compile nothing and executors run nothing. No value in this
module is a measurement.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import pytest
import yaml

import lassi.prompts as prompts_module
import lassi.prompts.assets as prompt_assets
from lassi.bench import Direction, load_suite
from lassi.core import runner as runner_module  # noqa: F401  (importing the runner registers every component)
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Limits, Message, RunResult, Sampling
from lassi.core.recipe import FIXES, load_recipe
from lassi.core.record import Diagnostic, Trial, make_trial_id, unified_diff
from lassi.core.registry import DEFAULT_REGISTRY, Registry, RegistryError
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.stages import diagnostic_line
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.executors.workdir import build_dir
from lassi.prompts import RecipeAssets, load_recipe_assets, render
from lassi.toolchains import NvccSm80, NvcppCc80

REPO = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = REPO / "third_party" / "LASSI"
UPSTREAM_PIN = "74b46812523f2ff79b53b6880a4521690d7478b0"
TOOL = REPO / "tools" / "extract_lassi_assets.py"
SKIP_REASON = (
    "needs the upstream checkout at third_party/LASSI on commit 74b4681 so that tools/extract_lassi_assets.py "
    "can generate the lassi-2024 fragments; run uv run tools/fetch_upstream.py first"
)

SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
ITEM = "layout"
MODEL_ID = "scripted-fixture"
FRAGMENT_SET = "lassi-2024"
TEMPLATE_SET = "p0-smoke"
OMP_TO_CUDA = Direction("omp", "cuda")
CUDA_TO_OMP = Direction("cuda", "omp")
DIRECTIONS = [OMP_TO_CUDA, CUDA_TO_OMP]
DIRECTION_IDS = ["omp-cuda", "cuda-omp"]
TOOLCHAINS = {"cuda": "nvcc-sm80", "omp": "nvcpp-cc80"}
PACK_OF = {"omp": "openmp-4.0-card", "cuda": "cuda-12.5-ch5"}
# Upstream's dictionary entry name for each direction: `<SOURCE>_to_<TARGET>` in upstream's language spelling.
DIRECTION_KEY = {("omp", "cuda"): "OMP_to_CUDA", ("cuda", "omp"): "CUDA_to_OMP"}
# run_loop reproduces fixes.execution_gate (P1.6), which faithful: true turns off, so every faithful list names it.
FAITHFUL_STAGES = ["baseline", "summarize_context", "describe_source", "generate", "compile_loop", "run_loop"]
TEMPLATE_STAGES = ["generate", "compile_loop"]
CORRECTION_FIXES = ("prompt_newlines", "parsed_diagnostics")
COUNT_CAP = "DIAGNOSTIC_COUNT_CAP"
BYTES_CAP = "DIAGNOSTIC_BYTES_CAP"
COMPILED = "S4"
# A stand-in for a pinned compiler path under the toolchains root; the fake toolchains carry it as `executable`.
PINNED_PLACEHOLDER = "/PLACEHOLDER-toolchains-root/pinned-compiler@0/bin/compiler"

# SYNTHETIC bench sources, with runs of spaces, as in the P1.4 tests.
OMP_SOURCE = "#include <cstdio>\nint main() {\n    int  n  =  4;\n    std::printf(\"n   = %d\\n\", n);\n}\n"
CUDA_SOURCE = "#include <cstdio>\n__global__ void kernel(int *out) {\n    out[threadIdx.x]  =  1;\n}\n"
SOURCES = {"omp": OMP_SOURCE, "cuda": CUDA_SOURCE}
BROKEN_SOURCE = "#error SYNTHETIC reference that does not build\n"

# SYNTHETIC model replies. BAD_CODE holds an #error line, which the fake toolchains refuse.
SUMMARY_REPLY = "Synthetic summary of the pack."
DESCRIPTION_REPLY = "Synthetic description of the source."
BAD_CODE = "#error SYNTHETIC not translated yet\nint main() {\n    return 1;\n}\n"
GOOD_CODE = "int main() {\n    return 0;\n}\n"
NO_FENCE_REPLY = "Synthetic reply that holds no code block.\n"

# SYNTHETIC compiler output: the raw stderr attachment and the parsed diagnostics of a failed build differ, so a
# test can tell which one a prompt carries. The raw text keeps runs of spaces and several lines.
RAW_STDERR = (
    '"main.x", line 1: error: SYNTHETIC-RAW-STDERR compiler   text\n'
    "  #error SYNTHETIC not translated yet\n"
    "  ^\n"
    "SYNTHETIC-RAW-STDERR remark:   a report line the parser keeps out\n"
)
EMPTY_STDERR = "SYNTHETIC-EMPTY-STDERR: the empty file defines no program entry\n"
PARSED = Diagnostic(stage="compile", severity="error", code="synthetic", message="SYNTHETIC parsed diagnostic")
EMPTY_PARSED = Diagnostic(stage="compile", severity="error", code="synthetic", message="SYNTHETIC empty build")

# Template-set replies and a source with no digit, so the only number in a correction prompt is the left-out count.
TEMPLATE_BAD = "#error SYNTHETIC not translated\nint main() { return EXIT_FAILURE; }\n"
TEMPLATE_BAD_REPLY = render_file_blocks({"main.cu": TEMPLATE_BAD})
TEMPLATE_GOOD_REPLY = render_file_blocks({"main.cu": "int main() { return EXIT_SUCCESS; }\n"})


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate and compile variables a test could inherit, and point TMPDIR at a test directory."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CPATH", raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))


# ---------------------------------------------------------------------------
# Upstream extraction (fragment-set tests only)


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
    spec = importlib.util.spec_from_file_location("extract_lassi_assets_for_p15", TOOL)
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
    return load_recipe_assets({"prompts": FRAGMENT_SET, "context": list(PACK_OF.values())}, root=assets_root)


# ---------------------------------------------------------------------------
# Names this task adds, looked up so a missing one fails its test with a clear message


def require_fixes(*names: str) -> None:
    """Fail the test clearly when a named fix is not in lassi.core.recipe.FIXES yet."""
    missing = [name for name in names if name not in FIXES]
    if missing:
        pytest.fail(f"lassi.core.recipe.FIXES has no {', '.join(missing)}; task P1.5 adds it")


def registered_stage(name: str) -> type:
    """Return the Stage class registered as `name` in DEFAULT_REGISTRY; fail clearly while it is missing."""
    try:
        return DEFAULT_REGISTRY.get("Stage", name).factory
    except RegistryError as error:
        pytest.fail(f"no stage is registered as {name!r} ({error}); task P1.5 adds the baseline stage")


def cap_constant(name: str) -> int:
    """Return the int constant `name` of the module that defines compile_loop; fail clearly while it is missing."""
    module = sys.modules[registered_stage("compile_loop").__module__]
    value = getattr(module, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        pytest.fail(f"{module.__name__}.{name} must be an int of at least 1, got {value!r}; task P1.5 adds it")
    return value


def end_reason_of(trial: Trial) -> Any:
    """Return trial.final.end_reason; fail the test clearly while Final has no such field."""
    if not hasattr(trial.final, "end_reason"):
        pytest.fail("Final has no end_reason field; task P1.5 adds final.end_reason")
    return trial.final.end_reason


def reference_run_of(trial: Trial) -> Any:
    """Return trial.reference_run; fail the test clearly while Trial has no such field."""
    if not hasattr(trial, "reference_run"):
        pytest.fail("Trial has no reference_run field; task P1.5 adds the reference run to the Result Record")
    return trial.reference_run


# ---------------------------------------------------------------------------
# Fake components


@dataclass
class Build:
    """One build a fake toolchain was asked for."""

    toolchain: str
    workdir: Path
    files: dict[str, str]
    harness: dict[str, str]


@dataclass
class Log:
    """The script the fakes follow and what they saw.

    `replies` are the model's replies in order; `failed_stderr` and
    `failed_diagnostics` are what a build of a file holding `#error` gives;
    `run_results` maps a language to what the scripted executor returns.
    """

    replies: list[str] = field(default_factory=list)
    failed_stderr: bytes = RAW_STDERR.encode("ascii")
    failed_diagnostics: list[Diagnostic] = field(default_factory=lambda: [PARSED])
    run_results: dict[str, RunResult] = field(default_factory=dict)
    builds: list[Build] = field(default_factory=list)
    runs: list[list[str]] = field(default_factory=list)
    requests: list[tuple[list[Message], Sampling]] = field(default_factory=list)


def fake_toolchain(registered_as: str, log: Log) -> type:
    """Return a Toolchain class without PIN that records builds and keeps a raw stderr attachment, as build() does.

    A file holding `#error` fails with the log's raw stderr and diagnostics;
    files that are all empty fail with EMPTY_STDERR; anything else writes a
    PLACEHOLDER artifact. The raw stderr goes to `compile.stderr` in the
    workdir, named by BuildResult.stderr_ref. It runs nothing.
    """

    class FakeToolchain:
        """Writes the files, keeps a raw stderr attachment, and reports a build; compiles nothing."""

        name = registered_as
        capabilities = frozenset({"diagnostics"})
        executable = PINNED_PLACEHOLDER

        def build(
            self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
        ) -> BuildResult:
            """Record the build, write every file and the stderr attachment, and return the result."""
            workdir = Path(workdir)
            assert workdir.is_dir() and not any(workdir.iterdir()), f"{workdir} is not a fresh build directory"
            log.builds.append(Build(registered_as, workdir, dict(files), dict(harness or {})))
            for path, text in [*files.items(), *(harness or {}).items()]:
                target = workdir / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            if any("#error" in text for text in files.values()):
                stderr, diagnostics = log.failed_stderr, list(log.failed_diagnostics)
            elif not any(text.strip() for text in files.values()):
                stderr, diagnostics = EMPTY_STDERR.encode("ascii"), [EMPTY_PARSED]
            else:
                stderr, diagnostics = b"", []
            (workdir / "compile.stderr").write_bytes(stderr)
            artifact = None
            if not diagnostics:
                artifact = workdir / "main"
                artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=diagnostics, stderr_ref="compile.stderr")

    return FakeToolchain


def scripted_executor(log: Log) -> type:
    """Return an Executor class that runs programs and answers from `log.run_results` by the artifact's language."""

    class ScriptedExecutor:
        """Records each run's inputs and returns the SYNTHETIC RunResult of the artifact's language."""

        name = "scripted"
        capabilities = frozenset({"runs_code", "sandboxed"})

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Record the inputs and return the scripted result."""
            log.runs.append(list(inputs))
            language = "cuda" if (Path(artifact).parent / "main.cu").is_file() else "omp"
            return log.run_results[language]

    return ScriptedExecutor


class CompileOnlyExecutor:
    """A compile-only Executor registered as "none"; being asked to run anything fails the test."""

    name = "none"
    capabilities = frozenset({"compile_only"})

    def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
        """Fail the test: a compile-only executor is never asked to run a program."""
        raise AssertionError(f"a compile-only executor was asked to run {artifact}")


def scripted_backend(log: Log) -> type:
    """Return an LLMBackend class, registered as "scripted", that answers from `log.replies` in order."""

    class ScriptedBackend:
        """Records each request and answers with the next scripted reply."""

        name = "scripted"
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id, as every backend does."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Record the request and return the next scripted reply."""
            log.requests.append((list(messages), sampling))
            assert log.replies, "the backend was asked for more replies than the script holds"
            return Completion(text=log.replies.pop(0), prompt_tokens=0, completion_tokens=0)

    return ScriptedBackend


def make_registry(log: Log, stages: Sequence[str]) -> Registry:
    """Return a test Registry: the scripted backend, both executors, fake toolchains, and the named real stages."""
    registry = Registry()
    registry.register("LLMBackend", "scripted", scripted_backend(log))
    registry.register("Executor", "none", CompileOnlyExecutor)
    registry.register("Executor", "scripted", scripted_executor(log))
    for name in TOOLCHAINS.values():
        registry.register("Toolchain", name, fake_toolchain(name, log))
    for name in stages:
        registry.register("Stage", name, registered_stage(name))
    return registry


# ---------------------------------------------------------------------------
# Recipes, bench sources, and runs


def write_bench(root: Path, sources: Mapping[str, str] | None = None) -> Path:
    """Write the item's SYNTHETIC source per language where the suite manifest lays it out; return `root`."""
    spec = load_suite(SUITE_MANIFEST).items[ITEM]
    for language, text in (sources or SOURCES).items():
        layout = spec.languages[language]
        path = root / layout.dir / layout.files[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def target_file(direction: Direction) -> str:
    """Return the item's one target file name for `direction`."""
    return load_suite(SUITE_MANIFEST).items[ITEM].languages[direction.target].files[0]


def template_recipe(
    *, loop: int | str | None = None, fixes: Mapping[str, bool] | None = None, stages: Sequence[str] = TEMPLATE_STAGES
) -> dict[str, Any]:
    """Return a p0-smoke recipe for layout, omp to cuda; `loop` sets loop.max_corrections and `fixes` the fixes."""
    data: dict[str, Any] = {
        "extends": "base",
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": "omp", "target": "cuda"}],
        "prompts": TEMPLATE_SET,
        "toolchain": {"cuda": TOOLCHAINS["cuda"]},
        "stages": list(stages),
        "executor": {"kind": "none"},
        "trials": {"n": 1},
    }
    if loop is not None:
        data["loop"] = {"max_corrections": loop}
    if fixes is not None:
        data["fixes"] = dict(fixes)
    return data


def fragment_recipe(direction: Direction = OMP_TO_CUDA, *, faithful: bool = True, **changes: Any) -> dict[str, Any]:
    """Return a lassi-2024 recipe for layout and `direction`, only the target toolchain bound; `changes` add keys."""
    data: dict[str, Any] = {
        "extends": "base",
        "faithful": faithful,
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": direction.source, "target": direction.target}],
        "prompts": FRAGMENT_SET,
        "context": list(PACK_OF.values()),
        "toolchain": {direction.target: TOOLCHAINS[direction.target]},
        "stages": list(FAITHFUL_STAGES),
        "executor": {"kind": "none"},
        "trials": {"n": 1},
    }
    data.update(copy.deepcopy(changes))
    return data


def all_fixes_off(**overrides: bool) -> dict[str, bool]:
    """Return every named fix turned off, except those `overrides` set."""
    return {**{name: False for name in FIXES}, **overrides}


def write_recipe(directory: Path, name: str, data: Mapping[str, Any]) -> Path:
    """Write `data` as the recipe `<directory>/<name>.yaml` and return its path."""
    path = directory / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


@dataclass
class Outcome:
    """One finished run: its directory, its one trial read back from the run tree, its id, and the log."""

    run_dir: Path
    trial_id: str
    trial: Trial
    log: Log

    def messages(self, index: int) -> list[tuple[str, str]]:
        """Return model request `index` as (role, content) pairs."""
        return [(message.role, message.content) for message in self.log.requests[index][0]]

    def prompt(self, index: int) -> str:
        """Return the stored prompt of attempt `index`."""
        ref = self.trial.attempts[index].prompt_ref
        assert ref is not None, f"attempt {index} has no prompt"
        return TextStore(self.run_dir).get(ref)

    def attempt_build(self, index: int) -> Build:
        """Return the build made in attempt `index`'s build directory."""
        workdir = build_dir(self.run_dir, self.trial_id, index).resolve()
        (build,) = [build for build in self.log.builds if build.workdir.resolve() == workdir]
        return build


def run_one(tmp_path: Path, bench_root: Path, name: str, data: Mapping[str, Any], log: Log) -> Outcome:
    """Run a one-trial recipe with the fakes following `log`; return the outcome."""
    if "fixes" in data:
        require_fixes(*data["fixes"])
    registry = make_registry(log, data["stages"])
    options = RunOptions(runs_root=tmp_path / name / "runs-root", run_id="test-run", bench_root=bench_root,
                         registry=registry)
    run_dir = run_recipe(write_recipe(tmp_path, name, data), options)
    direction = data["directions"][0]
    trial_id = make_trial_id(name, MODEL_ID, SUITE, f"{direction['source']}-{direction['target']}", ITEM, 1)
    return Outcome(run_dir, trial_id, read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir)), log)


# ---------------------------------------------------------------------------
# Expected prompts, computed from the loaded fragments


def as_upstream_reads(text: str) -> str:
    """Return a text as a file or pipe read in text mode gives it: each CRLF and each lone CR becomes LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def fenced(code: str, tag: str = "") -> str:
    """Return a SYNTHETIC reply holding `code` in one fenced block tagged `tag`."""
    return f"Synthetic reply.\n```{tag}\n{code}```\nEnd of reply.\n"


def faithful_correction(fragments: Mapping[str, str], direction: Direction, code: str, stderr: str) -> str:
    """Return upstream's correction prompt for a compile error, joined in the notebook's order, LF removed."""
    target = direction.target
    text = (
        code
        + fragments["correct.compile_error_head"]
        + fragments[f"setup.{target}.compiler"] + " " + fragments[f"setup.{target}.flags"]
        + fragments["correct.compile_error_tail"]
        + as_upstream_reads(stderr)
        + fragments["correct.outro"]
    )
    return text.replace("\n", "")


def direction_system(fragments: Mapping[str, str], direction: Direction) -> str:
    """Return the direction's system prompt fragment."""
    return fragments[f"system_prompt_dict.{DIRECTION_KEY[(direction.source, direction.target)]}"]


def correct_fields(diagnostics: str, code: str = TEMPLATE_BAD) -> dict[str, str]:
    """Return the correct.txt fields of a template-set correction after `code` failed to build."""
    return {
        "target_language": "cuda",
        "files": render_file_blocks({"main.cu": code}),
        "diagnostics": diagnostics,
        "target_files": "main.cu",
    }


def letters(index: int) -> str:
    """Return a distinct lowercase name for `index` with no digit: a, b, ..., z, ba, bb, and so on."""
    text = ""
    while True:
        text = chr(ord("a") + index % 26) + text
        index //= 26
        if index == 0:
            return text


def many_diagnostics(count: int, width: int = 0) -> list[Diagnostic]:
    """Return `count` distinct SYNTHETIC compile errors with no location and no digit; `width` pads each message."""
    pad = " " + "x" * width if width else ""
    return [
        Diagnostic(stage="compile", severity="error", code="s", message=f"problem {letters(index)}{pad}")
        for index in range(count)
    ]


def numbers_in(text: str) -> list[str]:
    """Return every decimal number written in `text`."""
    return re.findall(r"[0-9]+", text)


# ---------------------------------------------------------------------------
# The fixes


def test_the_correction_fixes_are_named_fixes_on_by_default_off_under_faithful_and_reproduced_by_compile_loop(
    tmp_path: Path,
) -> None:
    require_fixes(*CORRECTION_FIXES)
    for name in CORRECTION_FIXES:
        assert isinstance(FIXES[name], str) and FIXES[name].strip(), f"fix {name} needs a description"
    reproduces = set(getattr(registered_stage("compile_loop"), "reproduces", ()))
    assert {*CORRECTION_FIXES, "fence_tag"} <= reproduces, f"compile_loop reproduces {sorted(reproduces)}"
    registry = make_registry(Log(), TEMPLATE_STAGES)
    plain = load_recipe(write_recipe(tmp_path, "plain", template_recipe()), registry=registry).data["fixes"]
    faithful_data = {**template_recipe(), "faithful": True}
    faithful = load_recipe(write_recipe(tmp_path, "faithful", faithful_data), registry=registry).data["fixes"]
    for name in CORRECTION_FIXES:
        assert (plain[name], faithful[name]) == (True, False), f"fixes.{name}: on by default, off under faithful"


# ---------------------------------------------------------------------------
# Template prompt set: parsed diagnostics, capped


def test_fixes_on_keep_every_diagnostic_up_to_the_count_cap_with_no_note(tmp_path: Path) -> None:
    count_cap, bytes_cap = cap_constant(COUNT_CAP), cap_constant(BYTES_CAP)
    diagnostics = many_diagnostics(count_cap)
    lines = [diagnostic_line(diagnostic) for diagnostic in diagnostics]
    if sum(len(line) + 1 for line in lines) > bytes_cap:
        pytest.fail(f"{COUNT_CAP} short lines must fit in {BYTES_CAP} for this test; they take more")
    log = Log(replies=[TEMPLATE_BAD_REPLY, TEMPLATE_GOOD_REPLY], failed_diagnostics=diagnostics)
    outcome = run_one(tmp_path, write_bench(tmp_path / "bench"), "at-count-cap", template_recipe(), log)
    assert outcome.prompt(1) == render(TEMPLATE_SET, "correct", correct_fields("\n".join(lines))), (
        "at the cap nothing is left out, so the prompt is P0's, with no note"
    )


def test_fixes_on_cap_the_diagnostic_count_and_say_how_many_were_left_out(tmp_path: Path) -> None:
    count_cap, bytes_cap = cap_constant(COUNT_CAP), cap_constant(BYTES_CAP)
    diagnostics = many_diagnostics(count_cap + 7)
    lines = [diagnostic_line(diagnostic) for diagnostic in diagnostics]
    if sum(len(line) + 1 for line in lines[:count_cap]) > bytes_cap:
        pytest.fail(f"{COUNT_CAP} short lines must fit in {BYTES_CAP} for this test; they take more")
    log = Log(replies=[TEMPLATE_BAD_REPLY, TEMPLATE_GOOD_REPLY], failed_diagnostics=diagnostics)
    outcome = run_one(tmp_path, write_bench(tmp_path / "bench"), "past-count-cap", template_recipe(), log)
    prompt = outcome.prompt(1)
    kept = [line for line in lines if line in prompt.split("\n")]
    assert kept == lines[:count_cap], "the first diagnostics, in order, up to the count cap"
    assert "7" in numbers_in(prompt), "the prompt says how many diagnostics it left out (7)"


def test_fixes_on_cap_the_diagnostic_bytes_and_say_how_many_were_left_out(tmp_path: Path) -> None:
    count_cap, bytes_cap = cap_constant(COUNT_CAP), cap_constant(BYTES_CAP)
    if count_cap < 2 or bytes_cap < 64:
        pytest.fail(f"this test needs {COUNT_CAP} >= 2 and {BYTES_CAP} >= 64")
    diagnostics = many_diagnostics(min(count_cap, 4), width=bytes_cap // 2)
    lines = [diagnostic_line(diagnostic) for diagnostic in diagnostics]
    log = Log(replies=[TEMPLATE_BAD_REPLY, TEMPLATE_GOOD_REPLY], failed_diagnostics=diagnostics)
    outcome = run_one(tmp_path, write_bench(tmp_path / "bench"), "past-bytes-cap", template_recipe(), log)
    prompt = outcome.prompt(1)
    kept = [line for line in lines if line in prompt.split("\n")]
    assert kept and kept == lines[: len(kept)], "the first diagnostics, in order, whole"
    assert sum(len(line.encode("utf-8")) for line in kept) <= bytes_cap
    left_out = len(lines) - len(kept)
    assert left_out >= 1, "each line takes over half the byte cap, so not all of them fit"
    for index in range(len(kept), len(lines)):
        assert f"problem {letters(index)} " not in prompt, "a diagnostic left out leaves no partial line"
    assert str(left_out) in numbers_in(prompt), f"the prompt says how many diagnostics it left out ({left_out})"


def test_parsed_diagnostics_off_sends_the_raw_stderr_attachment_in_place_of_the_diagnostics(tmp_path: Path) -> None:
    log = Log(replies=[TEMPLATE_BAD_REPLY, TEMPLATE_GOOD_REPLY])
    data = template_recipe(fixes={"parsed_diagnostics": False})
    outcome = run_one(tmp_path, write_bench(tmp_path / "bench"), "raw-stderr", data, log)
    assert outcome.prompt(1) == render(TEMPLATE_SET, "correct", correct_fields(RAW_STDERR))
    assert outcome.messages(1) == [("user", outcome.prompt(1))]


def test_parsed_diagnostics_off_falls_back_to_the_diagnostics_when_the_failed_build_left_no_stderr(
    tmp_path: Path,
) -> None:
    # Upstream reads an empty compiler stderr as a success, so it has no counterpart here (a sandbox timeout,
    # or a build with no program and exit status 0); the prompt carries the parsed errors rather than nothing.
    log = Log(replies=[TEMPLATE_BAD_REPLY, TEMPLATE_GOOD_REPLY], failed_stderr=b"")
    data = template_recipe(fixes={"parsed_diagnostics": False})
    outcome = run_one(tmp_path, write_bench(tmp_path / "bench"), "empty-stderr", data, log)
    assert outcome.prompt(1) == render(TEMPLATE_SET, "correct", correct_fields(diagnostic_line(PARSED)))


def test_parsed_diagnostics_off_keeps_a_stderr_of_only_whitespace_as_upstream_does(tmp_path: Path) -> None:
    # Upstream reads any stderr other than "" as a compile error and sends it as it is.
    log = Log(replies=[TEMPLATE_BAD_REPLY, TEMPLATE_GOOD_REPLY], failed_stderr=b" \n")
    data = template_recipe(fixes={"parsed_diagnostics": False})
    outcome = run_one(tmp_path, write_bench(tmp_path / "bench"), "blank-stderr", data, log)
    assert outcome.prompt(1) == render(TEMPLATE_SET, "correct", correct_fields(" \n"))


def test_faithful_true_runs_the_baseline_and_the_loop_over_the_template_set(tmp_path: Path) -> None:
    # The fragment-set runs skip without third_party/LASSI; this one keeps `faithful: true` covered end to end.
    log = Log(replies=[fenced(TEMPLATE_BAD), fenced(GOOD_CODE)])
    data = {**template_recipe(stages=["baseline", "generate", "compile_loop", "run_loop"]), "faithful": True}
    outcome = run_one(tmp_path, write_bench(tmp_path / "bench"), "faithful-template", data, log)
    resolved = yaml.safe_load((outcome.run_dir / "recipe.resolved.yaml").read_text(encoding="utf-8"))
    assert resolved["fixes"] == {name: False for name in FIXES}, "faithful: true turns every fix off"
    assert [build.toolchain for build in log.builds] == [TOOLCHAINS["cuda"]] * 3, (
        "the baseline builds the target reference only, then attempts 0 and 1 are built"
    )
    attempt_dirs = {build_dir(outcome.run_dir, outcome.trial_id, index).resolve() for index in (0, 1)}
    assert log.builds[0].workdir.resolve() not in attempt_dirs, "the first build is the baseline's"
    first = outcome.trial.attempts[0]
    assert first.files == {"main.cu": "\n" + TEMPLATE_BAD}, "the first fence, read as upstream strips it"
    expected = render(TEMPLATE_SET, "correct", correct_fields(RAW_STDERR, code="\n" + TEMPLATE_BAD))
    assert outcome.prompt(1) == expected.replace("\n", ""), "the raw stderr, with every line feed removed"
    assert (outcome.trial.final.stage_reached, outcome.trial.final.corrections) == (COMPILED, 1)
    assert end_reason_of(outcome.trial) is None


def test_prompt_newlines_off_removes_every_line_feed_from_the_correction_prompt(tmp_path: Path) -> None:
    log = Log(replies=[TEMPLATE_BAD_REPLY, TEMPLATE_GOOD_REPLY])
    data = template_recipe(fixes={"prompt_newlines": False})
    outcome = run_one(tmp_path, write_bench(tmp_path / "bench"), "no-newlines", data, log)
    rendered = render(TEMPLATE_SET, "correct", correct_fields(diagnostic_line(PARSED)))
    assert outcome.prompt(1) == rendered.replace("\n", "")
    assert outcome.messages(1) == [("user", rendered.replace("\n", ""))]
    assert outcome.prompt(0) == outcome.messages(0)[0][1] and "\n" in outcome.prompt(0), (
        "only correction prompts lose their line feeds"
    )


# ---------------------------------------------------------------------------
# The cap: final.end_reason correction-cap


def test_a_cap_hit_ends_the_trial_with_correction_cap(tmp_path: Path) -> None:
    log = Log(replies=[TEMPLATE_BAD_REPLY] * 3)
    outcome = run_one(tmp_path, write_bench(tmp_path / "bench"), "cap-hit", template_recipe(loop=2), log)
    assert [attempt.stage_reached for attempt in outcome.trial.attempts] == ["S1"] * 3
    reason = end_reason_of(outcome.trial)
    assert reason is not None and reason.code == "correction-cap"
    assert isinstance(reason.message, str) and reason.message.strip()
    assert (outcome.trial.final.stage_reached, outcome.trial.final.corrections) == ("S1", 2)
    page = (trial_dir(outcome.run_dir, outcome.trial_id) / "trial.md").read_text(encoding="ascii")
    assert "correction-cap" in page and reason.message in page


def test_a_trial_that_compiles_within_the_cap_has_no_end_reason(tmp_path: Path) -> None:
    log = Log(replies=[TEMPLATE_BAD_REPLY, TEMPLATE_GOOD_REPLY])
    outcome = run_one(tmp_path, write_bench(tmp_path / "bench"), "within-cap", template_recipe(loop=2), log)
    assert outcome.trial.final.stage_reached == COMPILED
    assert end_reason_of(outcome.trial) is None


# ---------------------------------------------------------------------------
# Fragment prompt set: the pinned flags are upstream's


@pytest.mark.parametrize(
    ("language", "toolchain", "source"),
    [("cuda", NvccSm80, "main.cu"), ("omp", NvcppCc80, "main.cpp")],
    ids=["nvcc-sm80", "nvcpp-cc80"],
)
def test_the_pinned_toolchain_flags_equal_upstreams_flag_text(
    lassi_assets: RecipeAssets, language: str, toolchain: type, source: str
) -> None:
    fragments = lassi_assets.fragments
    command = toolchain().command([source])
    flags = shlex.split(fragments[f"setup.{language}.flags"])
    assert command[1 : 1 + len(flags)] == flags, "the preset's flags are upstream's flag text, in order"
    assert command[1 + len(flags) :] == ["-o", "main", source]
    assert PurePosixPath(fragments[f"setup.{language}.compiler"]).name == PurePosixPath(command[0]).name


# ---------------------------------------------------------------------------
# Fragment prompt set: the faithful correction prompt


@pytest.mark.parametrize("direction", DIRECTIONS, ids=DIRECTION_IDS)
def test_the_faithful_correction_prompt_for_a_compile_error(
    tmp_path: Path, lassi_assets: RecipeAssets, direction: Direction
) -> None:
    fragments = lassi_assets.fragments
    log = Log(replies=[SUMMARY_REPLY, DESCRIPTION_REPLY, fenced(BAD_CODE), fenced(GOOD_CODE)])
    outcome = run_one(tmp_path, write_bench(tmp_path / "bench"), "faithful-correct", fragment_recipe(direction), log)
    expected = faithful_correction(fragments, direction, "\n" + BAD_CODE, RAW_STDERR)
    assert outcome.messages(3) == [("system", direction_system(fragments, direction)), ("user", expected)]
    assert outcome.prompt(1) == expected, "the attempt keeps the correction prompt it was asked with"
    assert "\n" not in expected
    assert RAW_STDERR.replace("\n", "") in expected, "the whole raw stderr, runs of spaces kept"
    assert "  " in expected, "a correction prompt is never space-collapsed"
    assert PINNED_PLACEHOLDER not in expected, "upstream's compiler text, never the pinned executable path"
    first, second = outcome.trial.attempts
    assert (first.stage_reached, second.stage_reached) == ("S1", COMPILED)
    assert second.files == {target_file(direction): "\n" + GOOD_CODE}
    assert second.diff_from_previous == unified_diff(first.files, second.files)
    assert outcome.trial.final.corrections == 1
    assert end_reason_of(outcome.trial) is None


def test_the_faithful_correction_prompt_reads_the_stderr_attachment_in_text_mode(
    tmp_path: Path, lassi_assets: RecipeAssets
) -> None:
    crlf = RAW_STDERR.replace("\n", "\r\n")
    log = Log(replies=[SUMMARY_REPLY, DESCRIPTION_REPLY, fenced(BAD_CODE), fenced(GOOD_CODE)],
              failed_stderr=crlf.encode("ascii"))
    outcome = run_one(tmp_path, write_bench(tmp_path / "bench"), "faithful-crlf", fragment_recipe(), log)
    expected = faithful_correction(lassi_assets.fragments, OMP_TO_CUDA, "\n" + BAD_CODE, crlf)
    assert outcome.messages(3)[1] == ("user", expected)
    assert "\r" not in expected, "upstream reads compiler output in text mode, so no CR reaches the prompt"


def test_the_faithful_correction_prompt_removes_line_feeds_only(tmp_path: Path, lassi_assets: RecipeAssets) -> None:
    code = BAD_CODE.replace("\n", "\r\n")
    log = Log(replies=[SUMMARY_REPLY, DESCRIPTION_REPLY, fenced(code), fenced(GOOD_CODE)])
    outcome = run_one(tmp_path, write_bench(tmp_path / "bench"), "faithful-cr", fragment_recipe(), log)
    expected = faithful_correction(lassi_assets.fragments, OMP_TO_CUDA, "\n" + code, RAW_STDERR)
    assert "\r" in expected, "upstream removes LF only, so a CR in the model's code stays"
    assert outcome.messages(3)[1] == ("user", expected)


def test_the_faithful_loop_has_no_correction_cap(tmp_path: Path, lassi_assets: RecipeAssets) -> None:
    log = Log(replies=[SUMMARY_REPLY, DESCRIPTION_REPLY, *[fenced(BAD_CODE)] * 12, fenced(GOOD_CODE)])
    outcome = run_one(tmp_path, write_bench(tmp_path / "bench"), "faithful-uncapped", fragment_recipe(), log)
    assert [attempt.stage_reached for attempt in outcome.trial.attempts] == ["S1"] * 12 + [COMPILED]
    assert outcome.trial.final.corrections == 12, "past projects/base.yaml's cap of 10"
    assert end_reason_of(outcome.trial) is None
    assert log.replies == []


@pytest.mark.parametrize(
    ("tag", "text", "quirk"),
    [("cuda", "uda\n" + GOOD_CODE, True), ("cpp", "\n" + GOOD_CODE, False)],
    ids=["cuda-tag", "cpp-tag"],
)
def test_faithful_correction_replies_are_read_with_upstreams_fence_stripping(
    tmp_path: Path, lassi_assets: RecipeAssets, tag: str, text: str, quirk: bool
) -> None:
    reply = fenced(GOOD_CODE, tag)
    log = Log(replies=[SUMMARY_REPLY, DESCRIPTION_REPLY, fenced(BAD_CODE), reply])
    outcome = run_one(tmp_path, write_bench(tmp_path / "bench"), "faithful-fence", fragment_recipe(), log)
    second = outcome.trial.attempts[1]
    assert second.response_text == reply
    assert second.files == {target_file(OMP_TO_CUDA): text}
    hits = [item for item in second.diagnostics if item.code == "fence-quirk"]
    assert [(item.stage, item.severity) for item in hits] == ([("parse", "warning")] if quirk else [])


def test_a_no_fence_attempt_is_built_as_an_empty_target_file(tmp_path: Path, lassi_assets: RecipeAssets) -> None:
    log = Log(replies=[SUMMARY_REPLY, DESCRIPTION_REPLY, NO_FENCE_REPLY, fenced(GOOD_CODE)])
    outcome = run_one(tmp_path, write_bench(tmp_path / "bench"), "faithful-no-fence", fragment_recipe(), log)
    first, second = outcome.trial.attempts
    empty = {target_file(OMP_TO_CUDA): ""}
    assert (first.stage_reached, first.files) == ("S0", empty)
    assert "no-fence" in [item.code for item in first.diagnostics]
    assert outcome.attempt_build(0).files == empty, "upstream compiles the empty file it writes"
    expected = faithful_correction(lassi_assets.fragments, OMP_TO_CUDA, "", EMPTY_STDERR)
    assert outcome.messages(3)[1] == ("user", expected)
    assert second.stage_reached == COMPILED


def test_parsed_diagnostics_on_puts_the_parsed_diagnostics_where_upstream_puts_the_raw_stderr(
    tmp_path: Path, lassi_assets: RecipeAssets
) -> None:
    fragments = lassi_assets.fragments
    log = Log(replies=[SUMMARY_REPLY, DESCRIPTION_REPLY, fenced(BAD_CODE), fenced(GOOD_CODE)])
    data = fragment_recipe(faithful=False, fixes=all_fixes_off(parsed_diagnostics=True))
    outcome = run_one(tmp_path, write_bench(tmp_path / "bench"), "parsed-fragments", data, log)
    system, (role, sent) = outcome.messages(3)
    assert (system, role) == (("system", direction_system(fragments, OMP_TO_CUDA)), "user")
    head = ("\n" + BAD_CODE + fragments["correct.compile_error_head"] + fragments["setup.cuda.compiler"] + " "
            + fragments["setup.cuda.flags"] + fragments["correct.compile_error_tail"]).replace("\n", "")
    assert sent.startswith(head)
    assert sent.endswith(fragments["correct.outro"].replace("\n", ""))
    assert diagnostic_line(PARSED) in sent
    assert "SYNTHETIC-RAW-STDERR" not in sent, "the parsed diagnostics replace the raw stderr"


def test_a_faithful_baseline_failure_asks_no_model(tmp_path: Path, lassi_assets: RecipeAssets) -> None:
    log = Log(replies=[SUMMARY_REPLY, DESCRIPTION_REPLY, fenced(GOOD_CODE)])
    bench = write_bench(tmp_path / "bench", {**SOURCES, "cuda": BROKEN_SOURCE})
    outcome = run_one(tmp_path, bench, "faithful-baseline", fragment_recipe(), log)
    reason = end_reason_of(outcome.trial)
    assert reason is not None and reason.code == "baseline-compile"
    assert log.requests == [], "the context stages never ask the model after a baseline failure"
    assert [build.toolchain for build in log.builds] == [TOOLCHAINS["cuda"]], "faithful builds the target only"


# ---------------------------------------------------------------------------
# Every faithful behavior is keyed on a fix or the cap, never on the faithful flag


def equivalence_replies() -> list[str]:
    """Return replies that exercise the loop: no fence, then a cuda-tagged failing correction, then a fix."""
    return [SUMMARY_REPLY, DESCRIPTION_REPLY, NO_FENCE_REPLY, fenced(BAD_CODE, "cuda"), fenced(GOOD_CODE)]


def equivalence_log(replies: Sequence[str]) -> Log:
    """Return a log whose executor runs programs, so the baseline records a reference run."""
    stdout = "SYNTHETIC reference stdout\nPASS\n"
    result = RunResult(exit_code=0, hang=False, stdout=stdout, stderr="", wall_s=1.25)
    return Log(replies=list(replies), run_results={"cuda": result, "omp": result})


def test_all_fixes_off_with_a_numeric_cap_behaves_exactly_as_faithful(
    tmp_path: Path, lassi_assets: RecipeAssets
) -> None:
    bench = write_bench(tmp_path / "bench")
    faithful = run_one(tmp_path, bench, "faithful-true", fragment_recipe(executor={"kind": "scripted"}),
                       equivalence_log(equivalence_replies()))
    data = fragment_recipe(faithful=False, fixes=all_fixes_off(), loop={"max_corrections": 20},
                           executor={"kind": "scripted"})
    unfaithful = run_one(tmp_path, bench, "fixes-off", data, equivalence_log(equivalence_replies()))
    assert unfaithful.log.requests == faithful.log.requests, "the same messages and sampling, in order"
    assert unfaithful.trial.attempts == faithful.trial.attempts
    assert unfaithful.trial.context == faithful.trial.context
    assert reference_run_of(unfaithful.trial) == reference_run_of(faithful.trial)
    assert replace(unfaithful.trial.final, wall_s=None) == replace(faithful.trial.final, wall_s=None)
    built = [[(build.files, build.harness) for build in outcome.log.builds] for outcome in (faithful, unfaithful)]
    assert built[0] == built[1]
    assert unfaithful.log.runs == faithful.log.runs
    # run_loop runs the compiling attempt on this executor (P1.6), and its SYNTHETIC run exits 0: S5.
    assert [attempt.stage_reached for attempt in faithful.trial.attempts] == ["S0", "S1", "S5"]


def test_all_fixes_off_stops_at_the_numeric_cap_with_correction_cap(
    tmp_path: Path, lassi_assets: RecipeAssets
) -> None:
    bench = write_bench(tmp_path / "bench")
    faithful = run_one(tmp_path, bench, "faithful-true", fragment_recipe(), equivalence_log(equivalence_replies()))
    data = fragment_recipe(faithful=False, fixes=all_fixes_off(), loop={"max_corrections": 1})
    capped = run_one(tmp_path, bench, "fixes-off-capped", data, equivalence_log(equivalence_replies()[:4]))
    assert capped.log.requests == faithful.log.requests[:4], "upstream's requests up to the cap"
    assert capped.trial.attempts == faithful.trial.attempts[:2]
    reason = end_reason_of(capped.trial)
    assert reason is not None and reason.code == "correction-cap"
    assert isinstance(reason.message, str) and reason.message.strip()
    assert capped.trial.final.corrections == 1
    assert end_reason_of(faithful.trial) is None
