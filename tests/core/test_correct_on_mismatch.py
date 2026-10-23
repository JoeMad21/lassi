"""Tests for the opt-in recipe key loop.correct_on_mismatch: an oracle mismatch fed back to the model.

Bible: Harness Contract (held-out inputs never appear in a prompt), Component
Interfaces (the Stage row; the Oracle capabilities), Oracles (binary_io
rules, whose failure kinds the report names), Result Record
(Attempt.diagnostics, Alignment.outputs, Trial.requests, the end reasons),
Project Recipes (the key), Design Principle 4 and Agent Rule 4 (a departure
from upstream is a named toggle, never upstream behavior under faithful:
true), Agent Rule 5 (the prompt is generic and never tuned on item
results). The owner's decisions of 2026-10-23: a wrong number or an output
out of tolerance is fed back, and the feedback is a coarse report (OQ-077,
option (a)).

Upstream LASSI's loop corrects only compile errors and failed runs, and so
does run_loop with the key off or absent: an attempt that runs cleanly but
whose output files fail the binary_io oracle on the held-out inputs ends the
loop with no correction. The contract these tests fix:

- loop.correct_on_mismatch is a bool. Absent, it is false, and the loader
  materializes nothing, so every existing recipe keeps its resolved recipe
  and recipe hash. Set true, it is refused at load under faithful: true,
  naming the key. The runner refuses it, before any directory exists, when
  no listed stage honors it (run_loop does), when the recipe binds no Oracle
  that declares aligns_output_files, and when the prompt set has no
  correct_mismatch template.
- With it true, a clean run (S5) whose output files fail the oracle (an
  output outside the threshold, or a missing or extra one) gains a
  run-stage error Diagnostic, code oracle-mismatch, and the oracle's
  Alignment; while the correction cap leaves room, run_loop asks for a
  correction with the prompt set's correct_mismatch template, records the
  request under run_loop, and the new attempt is built, run, and judged
  again. The correction counts against loop.max_corrections with every
  other correction; a mismatch left at the cap ends the trial at
  correction-cap.
- The correction prompt carries a coarse report only: per output its name,
  passed or failed, its pcc to 3 decimals when it was computed, and fixed
  words for how far it is from the tolerance as an order of magnitude (or,
  for a pcc metric, a pattern match below the required level), or the kind
  of a failure that is not one of the metric's value (a missing, extra,
  repeated, or unreadable output among them). It never carries the exact
  max_abs, max_ulp, or threshold, the oracle's note text, a held-out input
  value, the item's seed, a reference output value, its magnitude, or its
  float bits, nor any byte of a candidate file that is not a lassi_io file,
  even for an all-zero candidate or a one-element output corrected twice.

The suite, programs, replies, and arrays are SYNTHETIC; fake toolchains
compile nothing and the scripted executor starts no process: it reads the
held-out inputs the stages wrote and writes SYNTHETIC outputs computed from
them. No value here is a measurement.
"""

from __future__ import annotations

import copy
import json
import math
import re
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from lassi.core import runner as runner_module
from lassi.core.files import render_file_blocks
from lassi.core.interfaces import BuildResult, Completion, Limits, Message, RunResult, Sampling
from lassi.core.recipe import RecipeError, load_recipe
from lassi.core.record import Diagnostic, OutputStats, Trial, make_trial_id
from lassi.core.registry import DEFAULT_REGISTRY, Registry
from lassi.core.runner import RunError, RunOptions, run_recipe
from lassi.core.stages import (
    ERROR_BANDS,
    ERROR_OVER,
    NO_ERROR_ALLOWED,
    PCC_BELOW,
    WITHIN_TOLERANCE,
    output_line,
    tolerance_band,
)
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.harness.lassi_io import generate_inputs, read_array, write_array
from lassi.oracles.binary_io import BinaryIOOracle, pearson

KEY = "loop.correct_on_mismatch"
MISMATCH = "oracle-mismatch"
SUITE = "mismatch-runs"
ITEM = "scale"
CPP, TT = "cpp", "tt"
TOOLCHAINS = {CPP: "fake-cpp", TT: "fake-tt"}
STAGES = ["baseline", "generate", "compile_loop", "run_loop", "oracle"]
PROMPTS = "tt-host-v0"
MODEL_ID = "scripted-fixture"
# A commit id for the synthetic manifest; not a commit of this repository or of any source.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
# The held-out input spec, shaped like a Tier A item's (one input, one output, a seed, a max_abs tolerance), with
# bounds and a seed chosen so that each value's text is distinctive and a check for it in a prompt bites.
SEED = 9876543210123
INPUTS = [{"name": "src0", "dtype": "f32", "shape": [16], "dist": "uniform", "lo": 4096.0, "hi": 4097.0}]
OUTPUT = "result"
TOLERANCE = {"metric": "max_abs", "threshold": 0.05}
# The C++ reference adds this to result[0], inside the tolerance, so the references agree without being equal and
# threshold from_baseline gives result the threshold 2**-8.
AGREEMENT = 2.0**-8
# The wrong candidate's offset: inside the f32 binade of 2 * src0, so every result is exact and max_abs is exactly it.
OFFSET = 1.5
ORACLE = {"kind": "binary_io", "metric": "max_abs", "threshold": "from_baseline"}

Values = list[float]
# A run's output files: file name -> (array name, values) for an f32 lassi_io file, or raw bytes for any other file.
Outputs = dict[str, tuple[str, Values] | bytes]
# A run's behavior: the input values -> its output files.
Behavior = Callable[[Values], Outputs]
# The bytes of a SYNTHETIC candidate file that is not a lassi_io file: letters no template or report holds, then
# bytes that are not ASCII, so a check for any part of them in a prompt bites.
GARBAGE = b"QZJXWVKYqzjxwvky\xfe\xed\xfa\xce"
GARBAGE_FILE = "garbage.bin"


def doubled(values: Values) -> Outputs:
    """Return the target reference's output, and a correct candidate's: result = 2 * src0."""
    return {f"{OUTPUT}.lassiio": (OUTPUT, [2.0 * value for value in values])}


def agreeing(values: Values) -> Outputs:
    """Return the C++ reference's output: 2 * src0, with AGREEMENT added to the first element."""
    result = [2.0 * value for value in values]
    result[0] += AGREEMENT
    return {f"{OUTPUT}.lassiio": (OUTPUT, result)}


def wrong(values: Values) -> Outputs:
    """Return a wrong candidate's output: 2 * src0 + OFFSET, past the threshold, with every element finite."""
    return {f"{OUTPUT}.lassiio": (OUTPUT, [2.0 * value + OFFSET for value in values])}


def zeros(values: Values) -> Outputs:
    """Return an all-zero candidate output: its max_abs is the reference's largest magnitude."""
    return {f"{OUTPUT}.lassiio": (OUTPUT, [0.0] * len(values))}


def missing(values: Values) -> Outputs:
    """Return no output at all: the run is clean, but it writes no file."""
    return {}


def extra(values: Values) -> Outputs:
    """Return the right result and an extra output the reference never writes."""
    return {**doubled(values), "debug.lassiio": ("debug", [1.0])}


def unreadable(values: Values) -> Outputs:
    """Return the right result and a file of GARBAGE, which is not a lassi_io file."""
    return {**doubled(values), GARBAGE_FILE: GARBAGE}


def candidate(number: int) -> str:
    """Return a SYNTHETIC reply: one FILE block of the target's one file, numbered so each attempt differs."""
    return render_file_blocks({f"{ITEM}.cpp": f"// SYNTHETIC candidate {number}\nint main() {{ return 0; }}\n"})


# A reply whose file the fake toolchain refuses to build.
BROKEN = render_file_blocks({f"{ITEM}.cpp": "#error SYNTHETIC\n"})


@pytest.fixture(autouse=True)
def clean_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate variables a test could inherit, and point TMPDIR at a test directory."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "compile-tmp").mkdir()
    monkeypatch.setenv("TMPDIR", str(tmp_path / "compile-tmp"))


# ---------------------------------------------------------------------------
# The SYNTHETIC suite and fakes


def write_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inputs: Sequence[Mapping[str, Any]] = INPUTS
) -> Path:
    """Write the one-item manifest with held-out `inputs`, and its sources; return the bench root."""
    languages = {
        CPP: {"dir": "src/cpp", "files": [f"{ITEM}.cpp"]},
        TT: {"dir": "src/tt", "files": [f"{ITEM}.cpp"]},
    }
    item = {
        "split": "eval", "seed": SEED, "inputs": [dict(spec) for spec in inputs], "outputs": [OUTPUT],
        "tolerance": TOLERANCE,
        "languages": languages,
    }
    data = {"suite": SUITE, "repo": "https://example.invalid/mismatch-runs", "commit": FAKE_COMMIT,
            "items": {ITEM: item}}
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / f"{SUITE}.yaml").write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    monkeypatch.setattr(runner_module, "BENCH_DIR", manifests)
    root = tmp_path / "bench"
    for language in languages.values():
        path = root / language["dir"] / language["files"][0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"// SYNTHETIC source\nint main() { return 0; }\n")
    return root


@dataclass
class Script:
    """What the fakes answer and run: the replies in order, each attempt's behavior, and what they saw."""

    replies: list[str]
    attempts: list[Behavior]
    references: dict[str, Behavior] = field(default_factory=lambda: {TT: doubled, CPP: agreeing})
    requests: list[list[Message]] = field(default_factory=list)
    inputs: list[Values] = field(default_factory=list)


def f32_values(path: Path) -> Values:
    """Return the elements of the f32 lassi_io file at `path`."""
    array = read_array(path)
    assert array.dtype == "f32"
    return list(struct.unpack(f"<{len(array.data) // 4}f", array.data))


class FakeToolchain:
    """Writes the files and a PLACEHOLDER artifact; a file holding `#error` fails the build. Compiles nothing."""

    name = "fake"
    capabilities = frozenset({"diagnostics"})

    def build(
        self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None
    ) -> BuildResult:
        """Write `files`; return a compile error for an `#error` file, else a placeholder artifact."""
        for path, text in files.items():
            (Path(workdir) / path).parent.mkdir(parents=True, exist_ok=True)
            (Path(workdir) / path).write_bytes(text.encode("utf-8"))
        if any("#error" in text for text in files.values()):
            error = Diagnostic(stage="compile", severity="error", code="synthetic", message="SYNTHETIC error")
            return BuildResult(artifact=None, diagnostics=[error])
        artifact = Path(workdir) / "main"
        artifact.write_bytes(b"PLACEHOLDER artifact\n")
        return BuildResult(artifact=artifact, diagnostics=[])


def scripted_executor(script: Script) -> type:
    """Return an Executor class that reads the held-out input and writes the scripted outputs; it runs nothing."""

    class ScriptedExecutor:
        """Computes SYNTHETIC outputs from the input file in its arguments and returns a clean RunResult."""

        name = "scripted"
        capabilities = frozenset({"runs_code", "sandboxed"})

        def run(self, artifact: Path, inputs: Sequence[str], limits: Limits) -> RunResult:
            """Read @inputs/src0.lassiio, write this run's outputs beside the artifact, and report exit status 0."""
            workdir = Path(artifact).parent
            kind = workdir.parent.name
            values = f32_values(workdir / inputs[0])
            script.inputs.append(values)
            behavior = script.references[kind.removeprefix("baseline-")] if kind.startswith("baseline-") else (
                script.attempts.pop(0)
            )
            files = {}
            for file, written in behavior(values).items():
                if isinstance(written, bytes):
                    (workdir / file).write_bytes(written)
                else:
                    name, result = written
                    write_array(workdir / file, name, "f32", (len(result),), struct.pack(f"<{len(result)}f", *result))
                files[file] = workdir / file
            return RunResult(exit_code=0, hang=False, stdout="SYNTHETIC\n", stderr="", output_files=files, wall_s=1.0)

    return ScriptedExecutor


def scripted_backend(script: Script) -> type:
    """Return an LLMBackend class that answers from the script in order and records every request."""

    class ScriptedBackend:
        """Answers each request with the next scripted reply; a request past the script fails the test."""

        name = "scripted"
        capabilities = frozenset({"chat"})

        def __init__(self, model_id: str) -> None:
            """Keep the model id."""
            self.model_id = model_id

        def complete(self, messages: Sequence[Message], sampling: Sampling) -> Completion:
            """Record the request and return the next scripted reply."""
            script.requests.append(list(messages))
            assert script.replies, "the backend was asked for more replies than the script holds"
            return Completion(text=script.replies.pop(0), prompt_tokens=0, completion_tokens=0)

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
    for name in STAGES:
        registry.register("Stage", name, DEFAULT_REGISTRY.get("Stage", name).factory)
    return registry


def recipe_data(loop: Mapping[str, Any] | None = None, **changes: Any) -> dict[str, Any]:
    """Return a one-trial recipe for the item from cpp to tt; `loop` merges over base.yaml's loop section."""
    data: dict[str, Any] = {
        "extends": "base",
        "model": {"backend": "scripted", "id": MODEL_ID},
        "llm": {"sampling": {"max_tokens": 4096}},
        "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
        "directions": [{"source": CPP, "target": TT}],
        "prompts": PROMPTS,
        "toolchain": dict(TOOLCHAINS),
        "fixes": {"baseline_both": True},
        "stages": list(STAGES),
        "executor": {"kind": "scripted"},
        "oracle": dict(ORACLE),
        "trials": {"n": 1},
    }
    if loop is not None:
        data["loop"] = dict(loop)
    data.update(copy.deepcopy(changes))
    return data


def write_recipe(tmp_path: Path, data: Mapping[str, Any], name: str = "mismatch") -> Path:
    """Write `data` as <tmp_path>/<name>.yaml and return its path."""
    path = tmp_path / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(dict(data), sort_keys=False).encode("ascii"))
    return path


def options(tmp_path: Path, bench: Path, script: Script) -> RunOptions:
    """Return the run options: a runs root under tmp_path, a fixed run id, the bench root, the test registry."""
    return RunOptions(runs_root=tmp_path / "runs-root", run_id="test-run", bench_root=bench,
                      registry=make_registry(script))


def run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: Script,
    loop: Mapping[str, Any] | None,
    *,
    inputs: Sequence[Mapping[str, Any]] = INPUTS,
    **changes: Any,
) -> tuple[Path, Trial]:
    """Run the recipe with `loop` and `changes` on the held-out `inputs`; return the run directory and its trial."""
    bench = write_suite(tmp_path, monkeypatch, inputs)
    run_dir = run_recipe(write_recipe(tmp_path, recipe_data(loop, **changes)), options(tmp_path, bench, script))
    trial_id = make_trial_id("mismatch", MODEL_ID, SUITE, f"{CPP}-{TT}", ITEM, 1)
    return run_dir, read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir))


def refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: Mapping[str, Any]) -> str:
    """Run `data`, expect a RunError before any directory exists, and return its message."""
    bench = write_suite(tmp_path, monkeypatch)
    with pytest.raises(RunError) as caught:
        run_recipe(write_recipe(tmp_path, data), options(tmp_path, bench, Script(replies=[], attempts=[])))
    assert not (tmp_path / "runs-root").exists(), "a refused run creates no directory"
    return str(caught.value)


def mismatches(trial: Trial, index: int) -> list[Diagnostic]:
    """Return the oracle-mismatch diagnostics of attempt `index`."""
    return [item for item in trial.attempts[index].diagnostics if item.code == MISMATCH]


ON = {"correct_on_mismatch": True}


# ---------------------------------------------------------------------------
# Off: today's behavior


@pytest.mark.parametrize("loop", [None, {"correct_on_mismatch": False}], ids=["absent", "false"])
def test_off_a_failing_oracle_attempt_ends_the_trial_uncorrected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loop: Mapping[str, Any] | None
) -> None:
    script = Script(replies=[candidate(0)], attempts=[wrong])
    _, trial = run(tmp_path, monkeypatch, script, loop)
    (attempt,) = trial.attempts
    assert attempt.stage_reached == "S5" and attempt.alignment.mean == 0.0, "the oracle stage fails the clean run"
    assert trial.final.end_reason is None and trial.final.corrections == 0
    assert [request.stage for request in trial.requests or []] == ["generate"]
    assert len(script.requests) == 1, "no correction is asked for an oracle mismatch with the key off"
    assert mismatches(trial, 0) == []


# ---------------------------------------------------------------------------
# On: a mismatch is fed back


def test_on_a_failing_attempt_is_corrected_and_a_passing_second_attempt_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Script(replies=[candidate(0), candidate(1)], attempts=[wrong, doubled])
    run_dir, trial = run(tmp_path, monkeypatch, script, ON)
    first, second = trial.attempts
    assert (first.stage_reached, second.stage_reached) == ("S5", "S5"), "an oracle mismatch is never a rung"
    (note,) = mismatches(trial, 0)
    assert (note.stage, note.severity) == ("run", "error") and OUTPUT in note.message
    assert first.alignment.mean == 0.0 and first.alignment.outputs is not None
    assert second.alignment.mean == 1.0 and mismatches(trial, 1) == []
    assert [request.stage for request in trial.requests or []] == ["generate", "run_loop"]
    assert trial.final.end_reason is None and trial.final.corrections == 1 and trial.final.alignment == 1.0
    page = trial_dir(run_dir, trial.trial_id)
    assert MISMATCH in (page / "trial.md").read_text(encoding="ascii")
    record = json.loads((page / "trial.json").read_text(encoding="ascii"))
    assert MISMATCH in [item["code"] for item in record["attempts"][0]["diagnostics"]]
    assert "correct_on_mismatch: true" in (run_dir / "run.md").read_text(encoding="ascii"), "the resolved recipe"


def test_on_the_cap_is_respected_and_shared_with_the_other_corrections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Attempt 0 does not build (a compile correction), attempt 1 runs wrong (an oracle-mismatch correction), and
    # attempt 2 runs wrong with the cap of 2 reached, so the trial ends there with three requests and no fourth.
    script = Script(replies=[BROKEN, candidate(1), candidate(2)], attempts=[wrong, wrong])
    _, trial = run(tmp_path, monkeypatch, script, {"max_corrections": 2, **ON})
    assert [attempt.stage_reached for attempt in trial.attempts] == ["S1", "S5", "S5"]
    assert [request.stage for request in trial.requests or []] == ["generate", "compile_loop", "run_loop"]
    assert len(script.requests) == 3 and script.attempts == []
    reason = trial.final.end_reason
    assert reason is not None and reason.code == "correction-cap" and "loop.max_corrections" in reason.message
    assert "oracle" in reason.message
    assert mismatches(trial, 1) and mismatches(trial, 2)
    assert trial.final.corrections == 2 and trial.final.alignment == 0.0, "the judged standing attempt keeps 0.0"


@pytest.mark.parametrize(("behavior", "named"), [(missing, OUTPUT), (extra, "debug")], ids=["missing", "extra"])
def test_on_a_missing_or_extra_output_also_triggers_a_correction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, behavior: Behavior, named: str
) -> None:
    script = Script(replies=[candidate(0), candidate(1)], attempts=[behavior, doubled])
    _, trial = run(tmp_path, monkeypatch, script, ON)
    first, second = trial.attempts
    assert first.stage_reached == "S5" and first.alignment.mean == 0.0
    (note,) = mismatches(trial, 0)
    assert named in note.message
    assert second.alignment.mean == 1.0 and trial.final.end_reason is None
    prompt = script.requests[1][-1].content
    word = "missing output" if behavior is missing else "extra output"
    assert f"{named}: failed" in prompt and word in prompt, prompt


# ---------------------------------------------------------------------------
# On: the prompt is a coarse report (OQ-077, option (a))

# Every phrase the report may give a compared output after its pcc, and the form of a report line: an output's name,
# its verdict, then its pcc and one such phrase, or one kind word.
BAND_WORDS = (WITHIN_TOLERANCE, PCC_BELOW, NO_ERROR_ALLOWED, *(words for _, words in ERROR_BANDS), ERROR_OVER)
REPORT_LINE = re.compile(r"[\w.]+: (?:passed|failed); .*")
BANDED_LINE = re.compile(rf"{OUTPUT}: failed; pcc -?[01]\.\d{{3}}; (?:{'|'.join(map(re.escape, BAND_WORDS))})")
# A threshold for the all-zero candidate: the largest reference element, about 8193, is about 400 times it.
ZERO_THRESHOLD = 20.0
# A one-element version of the item's held-out input.
ONE_ELEMENT = [{**INPUTS[0], "shape": [1]}]


def held_out_values(tmp_path: Path, inputs: Sequence[Mapping[str, Any]]) -> Values:
    """Return the held-out input values the stages write for `inputs` and SEED, drawn again in a test directory."""
    (tmp_path / "expected-inputs").mkdir()
    return f32_values(generate_inputs([dict(spec) for spec in inputs], SEED, tmp_path / "expected-inputs")["src0"])


def value_texts(value: float) -> set[str]:
    """Return the texts a message could show an f32 `value` as: decimal forms, repr, its integer part, and its bits."""
    packed = struct.pack("<f", value)
    bits = struct.unpack("<I", packed)[0]
    texts = {repr(value), str(value), f"{value:g}", f"{value:e}", *(f"{value:.{places}f}" for places in range(1, 7))}
    if abs(value) >= 100:
        texts.add(str(int(value)))
    return texts | {str(bits), f"{bits:x}", f"{bits:X}", f"{bits:#x}", packed.hex(), repr(packed)}


def report_lines(prompt: str) -> list[str]:
    """Return the lines of a mismatch prompt's field `outputs`: those that name an output and its verdict."""
    return [line for line in prompt.splitlines() if REPORT_LINE.fullmatch(line)]


def sent_texts(script: Script, trial: Trial) -> list[str]:
    """Return every message the backend was sent and every oracle-mismatch message the trial recorded."""
    notes = [note.message for index in range(len(trial.attempts)) for note in mismatches(trial, index)]
    return [*(message.content for request in script.requests for message in request), *notes]


def leaks(texts: Sequence[str], forbidden: set[str]) -> list[str]:
    """Return each forbidden text that some text holds, sorted."""
    return sorted(text for text in forbidden if any(text in found for found in texts))


def test_on_the_prompt_holds_the_coarse_report_and_no_held_out_or_reference_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Script(replies=[candidate(0), candidate(1)], attempts=[wrong, doubled])
    _, trial = run(tmp_path, monkeypatch, script, ON)
    (stats,) = trial.attempts[0].alignment.outputs or []
    assert (stats.name, stats.max_abs, stats.passed) == (OUTPUT, OFFSET, False), "the record keeps the exact value"
    prompt = script.requests[1][-1].content
    assert [message.role for message in script.requests[1]] == ["user"], "a template set sends one message"
    assert trial.requests is not None and trial.requests[1].stage == "run_loop"
    # max_abs is OFFSET, 384 times the threshold AGREEMENT that from_baseline gives.
    assert report_lines(prompt) == [f"{OUTPUT}: failed; pcc 1.000; about 100 to 1000 times the allowed error"]
    for text in (f"// FILE: {ITEM}.cpp\n// SYNTHETIC candidate 0\n", f"{TT} program as these files: {ITEM}.cpp."):
        assert text in prompt, f"the mismatch prompt lacks {text!r}:\n{prompt}"
    # The held-out input values the stages wrote (the executor read the same values every run), the seed, both
    # references' output values, the exact statistics, the threshold, and the oracle's note never appear.
    held_out = held_out_values(tmp_path, INPUTS)
    assert all(values == held_out for values in script.inputs) and len(script.inputs) == 4
    references = [*doubled(held_out).values(), *agreeing(held_out).values()]
    forbidden = {str(SEED), *(text for value in held_out for text in value_texts(value))}
    forbidden |= {text for _, values in references for value in values for text in value_texts(value)}
    forbidden |= {repr(OFFSET), repr(AGREEMENT), f"{AGREEMENT:g}", str(stats.max_ulp), "384", stats.note or ""}
    assert leaks(sent_texts(script, trial), forbidden) == []


def test_on_an_all_zero_candidate_shows_a_band_and_no_reference_value_magnitude_or_bits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With exact statistics, an all-zero output makes max_abs the reference's largest element and max_ulp its bits.
    oracle = {"kind": "binary_io", "metric": "max_abs", "threshold": ZERO_THRESHOLD}
    script = Script(replies=[candidate(0), candidate(1)], attempts=[zeros, doubled])
    _, trial = run(tmp_path, monkeypatch, script, ON, oracle=oracle)
    held_out = held_out_values(tmp_path, INPUTS)
    reference = [2.0 * value for value in held_out]
    largest = max(reference)
    (stats,) = trial.attempts[0].alignment.outputs or []
    assert stats.max_abs == largest and stats.max_ulp == struct.unpack("<I", struct.pack("<f", largest))[0]
    prompt = script.requests[1][-1].content
    assert report_lines(prompt) == [f"{OUTPUT}: failed; pcc 0.000; about 100 to 1000 times the allowed error"]
    assert trial.attempts[1].alignment.mean == 1.0 and trial.final.corrections == 1
    (note,) = mismatches(trial, 0)
    assert "about 100 to 1000 times the allowed error" in note.message
    ratio = largest / ZERO_THRESHOLD
    forbidden = {text for value in [*reference, *held_out, reference[0] + AGREEMENT] for text in value_texts(value)}
    forbidden |= {str(SEED), repr(ZERO_THRESHOLD), repr(ratio), f"{ratio:g}", str(int(ratio)), stats.note or ""}
    assert leaks(sent_texts(script, trial), forbidden) == []


def test_on_a_one_element_output_corrected_twice_shows_only_band_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Exact statistics recover a one-element reference in two corrections; the bands give an order of magnitude.
    script = Script(replies=[candidate(0), candidate(1), candidate(2)], attempts=[zeros, wrong, doubled])
    _, trial = run(tmp_path, monkeypatch, script, ON, inputs=ONE_ELEMENT)
    assert [request.stage for request in trial.requests or []] == ["generate", "run_loop", "run_loop"]
    assert trial.final.corrections == 2 and trial.attempts[-1].alignment.mean == 1.0
    (source,) = held_out_values(tmp_path, ONE_ELEMENT)
    reference = 2.0 * source
    prompts = [request[-1].content for request in script.requests[1:]]
    assert [report_lines(prompt) for prompt in prompts] == [
        [f"{OUTPUT}: failed; pcc 0.000; over 1000 times the allowed error"],
        [f"{OUTPUT}: failed; pcc 0.000; about 100 to 1000 times the allowed error"],
    ]
    for prompt in prompts:
        assert all(BANDED_LINE.fullmatch(line) for line in report_lines(prompt)), prompt
    forbidden = {text for value in (reference, reference + AGREEMENT, source) for text in value_texts(value)}
    forbidden |= {str(SEED), repr(OFFSET), repr(AGREEMENT), f"{AGREEMENT:g}", "384"}
    forbidden |= {entry.note or "" for attempt in trial.attempts[:2] for entry in attempt.alignment.outputs or []}
    assert leaks(sent_texts(script, trial), forbidden - {""}) == []


def test_on_an_unreadable_output_is_named_by_its_kind_with_no_candidate_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = Script(replies=[candidate(0), candidate(1)], attempts=[unreadable, doubled])
    _, trial = run(tmp_path, monkeypatch, script, ON)
    stats = {entry.name: entry for entry in trial.attempts[0].alignment.outputs or []}
    note = stats[GARBAGE_FILE].note
    assert note is not None and GARBAGE[:8].decode("ascii") in note, "the oracle's note quotes the file's first bytes"
    prompt = script.requests[1][-1].content
    assert report_lines(prompt) == [
        f"{OUTPUT}: passed; pcc 1.000; within tolerance",
        f"{GARBAGE_FILE}: failed; unreadable output",
    ]
    printable = GARBAGE[:16].decode("ascii")
    forbidden = {printable[start:start + 3] for start in range(len(printable) - 2)}
    forbidden |= {f"\\x{byte:02x}" for byte in GARBAGE[16:]} | {GARBAGE[16:].hex(), "magic", note}
    assert leaks(sent_texts(script, trial), forbidden) == []


# ---------------------------------------------------------------------------
# The coarse words of each kind of output, from the real binary_io oracle

# SYNTHETIC values exact in f32, whose texts no report line holds.
A, B, C = 7351.625, 9182.375, 4410.125
# A file's content: (array name, dtype, values), or raw bytes for a file that is not a lassi_io file.
Content = tuple[str, str, Sequence[float]] | bytes
MAX_ABS_ONE = {"metric": "max_abs", "threshold": 1.0}
# Each case: the oracle's config, the reference's files, the candidate's files, the output shown, and its line.
KIND_CASES: dict[str, tuple[dict[str, Any], dict[str, Content], dict[str, Content], str, str]] = {
    "passed": (MAX_ABS_ONE, {"x": ("x", "f32", [A, B])}, {"x": ("x", "f32", [A, B])}, "x",
               "x: passed; pcc 1.000; within tolerance"),
    "up-to-10": (MAX_ABS_ONE, {"x": ("x", "f32", [A, B])}, {"x": ("x", "f32", [A + 5, B])}, "x",
                 "x: failed; pcc 1.000; up to 10 times the allowed error"),
    "10-to-100": (MAX_ABS_ONE, {"x": ("x", "f32", [A, B])}, {"x": ("x", "f32", [A + 50, B])}, "x",
                  "x: failed; pcc 1.000; about 10 to 100 times the allowed error"),
    "100-to-1000": (MAX_ABS_ONE, {"x": ("x", "f32", [A, B])}, {"x": ("x", "f32", [A + 500, B])}, "x",
                    "x: failed; pcc 1.000; about 100 to 1000 times the allowed error"),
    "over-1000": (MAX_ABS_ONE, {"x": ("x", "f32", [A, B])}, {"x": ("x", "f32", [A + 5000, B])}, "x",
                  "x: failed; pcc -1.000; over 1000 times the allowed error"),
    "no-error-allowed": ({"metric": "max_abs", "threshold": 0}, {"x": ("x", "f32", [A])},
                         {"x": ("x", "f32", [A + 0.25])}, "x",
                         "x: failed; pcc 0.000; not an exact match, where the tolerance allows no error"),
    "ulp": ({"metric": "ulp", "threshold": 4}, {"x": ("x", "f32", [C])}, {"x": ("x", "f32", [C + 50 * 2.0**-11])},
            "x", "x: failed; pcc 0.000; about 10 to 100 times the allowed error"),
    "pcc-below": ({"metric": "pcc", "threshold": 0.99}, {"x": ("x", "f32", [A, B, C])},
                  {"x": ("x", "f32", [B, A, C])}, "x", "x: failed; pcc PCC; pattern match below the required level"),
    "integer": (MAX_ABS_ONE, {"x": ("x", "i32", [70001, 123456789])}, {"x": ("x", "i32", [70001, 987654321])}, "x",
                "x: failed; pcc 1.000; not an exact match, which an integer output needs"),
    "nan": (MAX_ABS_ONE, {"x": ("x", "f32", [float("nan"), C])}, {"x": ("x", "f32", [1234.5, C])}, "x",
            "x: failed; pcc 1.000; unmatched NaN or infinity"),
    "overflow": (MAX_ABS_ONE, {"x": ("x", "f64", [1.5e308, C])}, {"x": ("x", "f64", [-1.5e308, C])}, "x",
                 "x: failed; pcc -1.000; a difference too large for a double"),
    "missing": (MAX_ABS_ONE, {"x": ("x", "f32", [A])}, {}, "x", "x: failed; missing output"),
    "extra": (MAX_ABS_ONE, {"x": ("x", "f32", [A])}, {"x": ("x", "f32", [A]), "y": ("y", "f32", [B])}, "y",
              "y: failed; extra output"),
    "repeated": (MAX_ABS_ONE, {"x": ("x", "f32", [A])}, {"x1": ("x", "f32", [A]), "x2": ("x", "f32", [B])}, "x",
                 "x: failed; repeated output"),
    "unreadable": (MAX_ABS_ONE, {"x": ("x", "f32", [A])}, {"x": ("x", "f32", [A]), GARBAGE_FILE: GARBAGE},
                   GARBAGE_FILE, f"{GARBAGE_FILE}: failed; unreadable output"),
    "dtype": (MAX_ABS_ONE, {"x": ("x", "f32", [A])}, {"x": ("x", "i32", [7351])}, "x", "x: failed; dtype mismatch"),
    "shape": (MAX_ABS_ONE, {"x": ("x", "f32", [A, B])}, {"x": ("x", "f32", [A, B, C])}, "x",
              "x: failed; shape mismatch"),
}


def written_files(directory: Path, contents: Mapping[str, Content]) -> dict[str, Path]:
    """Write each content as <directory>/<key>.lassiio (or raw bytes under its key) and return file -> path."""
    directory.mkdir()
    files = {}
    for key, content in contents.items():
        if isinstance(content, bytes):
            files[key] = directory / key
            files[key].write_bytes(content)
            continue
        name, dtype, values = content
        code = {"f32": "f", "f64": "d", "i32": "i"}[dtype]
        files[f"{key}.lassiio"] = directory / f"{key}.lassiio"
        data = struct.pack(f"<{len(values)}{code}", *values)
        write_array(files[f"{key}.lassiio"], name, dtype, (len(values),), data)
    return files


def number_texts(value: float) -> set[str]:
    """Return the decimal texts of a SYNTHETIC element value; empty for a NaN."""
    return set() if math.isnan(value) else {repr(value), str(value), f"{value:g}"}


def check_coarse(line: str, entry: OutputStats, contents: Sequence[Content]) -> None:
    """Check that `line` holds no element value of `contents`, no note, and no number but its pcc and band bounds."""
    values = [value for content in contents if not isinstance(content, bytes) for value in content[2]]
    forbidden = {text for value in values for text in number_texts(value)}
    assert leaks([line], forbidden | ({entry.note} if entry.note else set())) == [], line
    rest = line.removeprefix(f"{entry.name}: ")
    for words in (*BAND_WORDS, "" if entry.pcc is None else f"pcc {entry.pcc:.3f}"):
        rest = rest.replace(words, "") if words else rest
    assert not any(character.isdigit() for character in rest), f"a number other than pcc and a band: {line}"


@pytest.mark.parametrize("case", sorted(KIND_CASES))
def test_each_kind_of_output_has_fixed_words_and_no_value_threshold_or_note(tmp_path: Path, case: str) -> None:
    config, ours, theirs, shown, expected = KIND_CASES[case]
    oracle = BinaryIOOracle(**config)
    stats = oracle.compare(written_files(tmp_path / "ours", ours), written_files(tmp_path / "theirs", theirs))
    (entry,) = [item for item in stats if item.name == shown]
    if "PCC" in expected:
        assert entry.pcc is not None and entry.pcc < 0.99
        expected = expected.replace("PCC", f"{entry.pcc:.3f}")
    line = output_line(oracle, entry)
    assert line == expected
    check_coarse(line, entry, [*ours.values(), *theirs.values()])


def test_an_output_with_no_threshold_in_the_agreement_says_so(tmp_path: Path) -> None:
    # from_baseline, bound to an agreement that gives the output no value, fails it; the report shows no number.
    agreement = [OutputStats(name="other", pcc=1.0, max_abs=0.0, max_ulp=0, passed=True, note=None)]
    oracle = BinaryIOOracle(metric="max_abs", threshold="from_baseline").with_baseline(agreement)
    ours: dict[str, Content] = {"x": ("x", "f32", [A, B])}
    theirs: dict[str, Content] = {"x": ("x", "f32", [A + 5, B])}
    (entry,) = oracle.compare(written_files(tmp_path / "ours", ours), written_files(tmp_path / "theirs", theirs))
    assert entry.note is not None and "threshold" in entry.note and not entry.passed
    line = output_line(oracle, entry)
    assert line == "x: failed; pcc 1.000; no threshold recorded for this output"
    check_coarse(line, entry, [*ours.values(), *theirs.values()])


@pytest.mark.parametrize(
    ("metric", "value", "threshold", "words"),
    [
        ("max_abs", 1.0, 1.0, WITHIN_TOLERANCE),
        ("max_abs", 10.0, 1.0, "up to 10 times the allowed error"),
        ("max_abs", 10.5, 1.0, "about 10 to 100 times the allowed error"),
        ("ulp", 1000, 1, "about 100 to 1000 times the allowed error"),
        ("ulp", 1001, 1, ERROR_OVER),
        ("max_abs", 1e300, 1e-300, ERROR_OVER),
        ("max_abs", 0.5, 0, NO_ERROR_ALLOWED),
        ("pcc", 0.95, 0.9, WITHIN_TOLERANCE),
        ("pcc", 0.2, 0.9, PCC_BELOW),
    ],
)
def test_the_band_is_an_order_of_magnitude_only(metric: str, value: float, threshold: float, words: str) -> None:
    assert tolerance_band(metric, value, threshold) == words


def test_pcc_rounded_recovers_no_element_of_a_one_or_two_element_output() -> None:
    # pcc is free of scale and shift: one element gives 0 or 1, two give -1, 0, or 1, whatever the values.
    values = [-8193.25, -1.5, 0.0, 2.0**-8, 3.25, 4096.125, 8193.75]
    singles = {f"{pearson([x], [y]):.3f}" for x in values for y in values}
    pairs = {f"{pearson([x0, x1], [y0, y1]):.3f}" for x0 in values for x1 in values for y0 in values[:3]
             for y1 in values[3:]}
    assert singles == {"0.000", "1.000"}
    assert pairs <= {"-1.000", "0.000", "1.000"}, pairs


# ---------------------------------------------------------------------------
# The recipe key


def load(tmp_path: Path, data: Mapping[str, Any], name: str = "mismatch") -> Any:
    """Load `data` written as <tmp_path>/<name>.yaml with the test registry."""
    return load_recipe(write_recipe(tmp_path, data, name), registry=make_registry(Script(replies=[], attempts=[])))


def test_true_is_refused_under_faithful(tmp_path: Path) -> None:
    with pytest.raises(RecipeError) as caught:
        load(tmp_path, recipe_data(ON, faithful=True))
    message = str(caught.value)
    assert KEY in message and "faithful: true" in message and "mismatch.yaml" in message, message


def test_true_inherited_is_refused_when_the_child_turns_faithful_on(tmp_path: Path) -> None:
    load(tmp_path, recipe_data(ON), name="mismatch-parent")
    path = write_recipe(tmp_path, {"extends": "mismatch-parent.yaml", "faithful": True})
    with pytest.raises(RecipeError, match=r"loop\.correct_on_mismatch"):
        load_recipe(path, registry=make_registry(Script(replies=[], attempts=[])))


def test_false_loads_under_faithful_and_a_bad_value_is_refused_naming_the_key(tmp_path: Path) -> None:
    assert load(tmp_path, recipe_data({"correct_on_mismatch": False}, faithful=True)).data["loop"] == {
        "max_corrections": "uncapped", "correct_on_mismatch": False
    }
    for bad in (1, "yes", None):
        with pytest.raises(RecipeError, match=r"loop\.correct_on_mismatch"):
            load(tmp_path, recipe_data({"correct_on_mismatch": bad}))


def test_absent_the_key_adds_nothing_and_set_it_enters_the_hash(tmp_path: Path) -> None:
    loaded = {}
    for label, loop in (("plain", None), ("on", ON), ("off", {"correct_on_mismatch": False})):
        (tmp_path / label).mkdir()
        loaded[label] = load(tmp_path / label, recipe_data(loop))
    assert loaded["plain"].data["loop"] == {"max_corrections": 10}
    assert "correct_on_mismatch" not in loaded["plain"].canonical_yaml
    assert len({recipe.recipe_hash for recipe in loaded.values()}) == 3, "a key that is set enters the hash"


# The resolved hash of the Tier A max_abs run recipe at commit 8342709, before the key existed.
TIER_A_MAXABS_HASH = "8bdaeac1d6b46ef4293c7187a831b5f3c6b054ef2bf02e1161fd9e8f8ca4e992"


def test_an_existing_recipe_keeps_its_resolved_form_and_hash() -> None:
    recipe = load_recipe(Path(__file__).resolve().parents[2] / "projects" / "lassi-demo" / "tier-a-rngd-maxabs.yaml")
    assert recipe.data["loop"] == {"max_corrections": 10}
    assert recipe.recipe_hash == TIER_A_MAXABS_HASH


# ---------------------------------------------------------------------------
# The runner's refusals, before any directory exists


def test_the_runner_refuses_the_key_with_a_prompt_set_that_has_no_mismatch_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    message = refused(tmp_path, monkeypatch, recipe_data(ON, prompts="p0-smoke"))
    assert KEY in message and "correct_mismatch" in message, message


def test_the_runner_refuses_the_key_without_an_output_file_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    message = refused(tmp_path, monkeypatch, recipe_data(ON, oracle={"kind": "stdout_mask", "passfail": False}))
    assert KEY in message and "aligns_output_files" in message, message


def test_the_runner_refuses_the_key_when_no_listed_stage_honors_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stages = ["baseline", "generate", "compile_loop", "oracle"]
    message = refused(tmp_path, monkeypatch, recipe_data(ON, stages=stages))
    assert KEY in message and "run_loop" in message, message
