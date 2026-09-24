"""Tests for the lassi score profile on hand-built trials (task P2.7).

Bible: Evaluation Protocol (LASSI reproduction row; LASSI Paper Metrics;
Acceptance Criteria, the compile-only label), Project Recipes (the
lassi-repro `metrics` line), Source Papers (LASSI quirk table, Sim-T and
fence rows), Oracles (stdout_mask rules), Harness Contract (the proxy checks
outputs, never runtime), Component Interfaces (ScoreProfile).

The contract these tests fix, from the P2.7 acceptance criteria
(plans/p2-scoring.md), its planning decision on `correct`, and the P2.4
definitions (Evaluation Protocol, LASSI Paper Metrics):

- Importing lassi.scoring registers the ScoreProfile `lassi` in
  DEFAULT_REGISTRY. Its class lives in lassi.scoring.lassi_profile and is
  built as `factory(bench_root=<root of the suite's fetched sources>)`, the
  root the runner's RunOptions.bench_root and `lassi score --bench-root`
  name. The reference target is the item's one file in the direction's
  target language, found through the suite manifest
  assets/bench/<bench_item.suite>.yaml under that root and read in text mode
  (each CRLF and each lone CR becomes LF), as the P1.9 replay reads it.
- `score(trial)` returns a lassi.core.interfaces.Score whose `components`
  maps exactly the names in COMPONENTS, in that order, to a float or None;
  whose `scalar` is components["correct"]; and whose `notes` maps a
  component name to a plain ASCII text: why the value is None, or its
  label, and for each similarity value the interpreter that computed it
  ("python <platform.python_version()>").
- correct: None when the trial is compile-only (the target reference was
  not run and no attempt ran), its note labeling it a compile-stage
  reproduction. Otherwise 1.0 when the output that stands (the last attempt
  that ran, stale output included) came from a clean run (exit 0, no hang)
  and the oracle gave it 1.0, else 0.0; 0.0 when no output stands.
- correct_paper: the paper criterion (manual inspection). Never computed:
  None, its note naming manual inspection.
- within_10pct: None while no timing profiler exists, its note naming the
  missing profiler, even when an attempt carries a runtime.
- first_try: 1.0 when correct is 1.0 and final.corrections is 0, 0.0 when
  either fails, None when correct is None.
- sim_t, sim_t_c, sim_l: lassi.scoring.similarity's sim_t, sim_t_c, and
  sim_l of the text-mode reference target against the last attempt's
  target file ("" when the last attempt has none).
- self_corr: final.corrections. cap_hit: 1.0 when final.end_reason is
  correction-cap, else 0.0. fence_quirk: the number of `fence-quirk`
  diagnostics over all attempts.
- compiled: 1.0 when the last attempt reached S4 or S5, else 0.0.
  compiled_first_try: 1.0 when compiled is 1.0 and final.corrections is 0,
  else 0.0.
- assets/scoring/lassi.yaml holds `components` (the component names in
  order), `scalar: correct`, and `notes` with the fixed texts
  `within_10pct`, `correct_paper`, and `compile_only`, which the profile
  gives as those notes.
- Cases the acceptance criteria leave open, decided with the task and
  tested at the end of this module: Score.notes is empty unless a profile
  writes one. A clean standing run the oracle never aligned gives correct
  None with the file's `not_aligned` note. A trial with no attempt that
  ended at the baseline gives correct, first_try, compiled,
  compiled_first_try, and the similarity values None, each noted with the
  file's `baseline` text and the end code; any other trial with no attempt
  gives compiled 0.0 and the similarity values None with the `no_attempt`
  note. A missing reference file raises FileNotFoundError naming its path.
  The profile file sets the component order and is checked when the
  profile is built, and the bench root must be a directory.

Every program text, reply, and run here is SYNTHETIC and written for these
tests; wall times and runtimes are PLACEHOLDER fixture values. The expected
similarity values are the P1.8 functions' outputs on these synthetic texts.
No value in this module is a measurement, and no upstream text is copied
into it (OQ-018).
"""

from __future__ import annotations

import dataclasses
import importlib
import platform
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from lassi.bench import Direction, load_suite
from lassi.core.interfaces import Sampling, Score
from lassi.core.record import (
    Alignment,
    Attempt,
    Diagnostic,
    EndReason,
    Final,
    ModelInfo,
    Profile,
    Provenance,
    RunInfo,
    ScoreBreakdown,
    Trial,
    make_trial_id,
)
from lassi.core.registry import DEFAULT_REGISTRY, RegistryError
from lassi.core.store import TextStore
from lassi.scoring.similarity import sim_l, sim_t, sim_t_c

REPO = Path(__file__).resolve().parents[2]
PROFILE_NAME = "lassi"
PROFILE_MODULE = "lassi.scoring.lassi_profile"
PROFILE_FILE = REPO / "assets" / "scoring" / "lassi.yaml"
SUITE = "lassi-hecbench-10"
SUITE_MANIFEST = REPO / "assets" / "bench" / f"{SUITE}.yaml"
ITEM = "layout"
OMP_TO_CUDA = Direction("omp", "cuda")
CUDA_TO_OMP = Direction("cuda", "omp")
MODEL_ID = "scoring-fixture"
SAMPLING = Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)
# A commit id for synthetic provenance; not a commit of this repository.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"

COMPONENTS = (
    "correct", "correct_paper", "within_10pct", "first_try", "sim_t", "sim_t_c", "sim_l",
    "self_corr", "cap_hit", "fence_quirk", "compiled", "compiled_first_try",
)
SIMILARITY = ("sim_t", "sim_t_c", "sim_l")
FENCE_QUIRK = "fence-quirk"
STALE_OUTPUT = "stale-output"
CORRECTION_CAP = "correction-cap"
UPSTREAM_CRASH = "upstream-crash"

# SYNTHETIC programs: a made-up fill program in both languages, and made-up candidates.
REFERENCE = {
    "cuda": (
        "// SYNTHETIC scoring fixture: CUDA reference of a made-up fill program.\n"
        "#include <cstdio>\n"
        "\n"
        "__global__ void fixture_fill(int *out, int count) {\n"
        "    int k = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (k < count) out[k] = 2 * k;\n"
        "}\n"
        "\n"
        "int main() {\n"
        "    printf(\"fixture fill done\\n\");\n"
        "    return 0;\n"
        "}\n"
    ),
    "omp": (
        "// SYNTHETIC scoring fixture: OpenMP reference of a made-up fill program.\n"
        "#include <cstdio>\n"
        "\n"
        "int main() {\n"
        "    int out[8];\n"
        "    #pragma omp target teams loop map(from: out)\n"
        "    for (int k = 0; k < 8; k++) out[k] = 2 * k;\n"
        "    printf(\"fixture fill done\\n\");\n"
        "    return 0;\n"
        "}\n"
    ),
}
CANDIDATE = (
    "#include <cstdio>\n"
    "__global__ void made_up_kernel(int *values, int n) {\n"
    "  int i = threadIdx.x;\n"
    "  if (i < n) values[i] = i + i;\n"
    "}\n"
    "int main() { printf(\"fixture fill done\\n\"); return 0; }\n"
)
BAD_CODE = "#error SYNTHETIC not translated yet\nint main() { return 1; }\n"
# SYNTHETIC program output, and PLACEHOLDER wall times and runtime in seconds.
REFERENCE_STDOUT = "fixture fill done\n"
PLACEHOLDER_REFERENCE_WALL_S = 1.25
PLACEHOLDER_ATTEMPT_WALL_S = 0.5
PLACEHOLDER_RUNTIME_S = 0.75
SIGNAL_EXIT = 139  # a death by signal 11 in the shell's form (128 + 11)
RUN_ERROR_EXIT = 3


# ---------------------------------------------------------------------------
# The profile, looked up so that a missing one fails each test with a clear message


def profile_class() -> type:
    """Return the class registered as ScoreProfile `lassi`; fail the test clearly while none is registered."""
    importlib.import_module("lassi.scoring")
    try:
        return DEFAULT_REGISTRY.get("ScoreProfile", PROFILE_NAME).factory
    except RegistryError as error:
        pytest.fail(f"task P2.7 registers the ScoreProfile {PROFILE_NAME!r} when lassi.scoring is imported: {error}")


def score_of(trial: Trial, bench_root: Path) -> Score:
    """Build the lassi profile over `bench_root` and return its score of `trial`."""
    return profile_class()(bench_root=bench_root).score(trial)


# ---------------------------------------------------------------------------
# SYNTHETIC bench sources


def text_mode(text: str) -> str:
    """Return `text` as a file opened in text mode reads it: each CRLF and each lone CR becomes LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def target_file(direction: Direction) -> str:
    """Return the item's one file name in the direction's target language."""
    return load_suite(SUITE_MANIFEST).items[ITEM].languages[direction.target].files[0]


# The target file of the default direction (OpenMP to CUDA), which most attempts write.
TARGET = target_file(OMP_TO_CUDA)


def write_bench(root: Path, direction: Direction, target: bytes, source: bytes) -> Path:
    """Write SYNTHETIC target and source programs where the suite manifest lays out the item; return `root`."""
    spec = load_suite(SUITE_MANIFEST).items[ITEM]
    for language, data in ((direction.target, target), (direction.source, source)):
        layout = spec.languages[language]
        path = root / layout.dir / layout.files[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


def default_bench(root: Path, direction: Direction = OMP_TO_CUDA) -> Path:
    """Write the SYNTHETIC reference programs of `direction` (LF line ends) under `root`; return `root`."""
    return write_bench(root, direction, REFERENCE[direction.target].encode("ascii"),
                       REFERENCE[direction.source].encode("ascii"))


# ---------------------------------------------------------------------------
# SYNTHETIC attempts and trials


def run_error() -> Diagnostic:
    """Return a SYNTHETIC run-stage run-error Diagnostic."""
    return Diagnostic(stage="run", severity="error", code="run-error", message="SYNTHETIC run error")


def ran(index: int, store: TextStore, code: str, *, exit_code: int | None = 0, hang: bool = False,
        value: float | None = None, runtime_s: float | None = None, name: str = TARGET) -> Attempt:
    """Return a SYNTHETIC attempt that compiled and ran: S5 after a clean run, else S4 with a run error.

    `code` is its target file `name`; `value` is the oracle's alignment of its stdout (unset for None).
    """
    clean = exit_code == 0 and not hang
    run = RunInfo(exit_code=exit_code, hang=hang, wall_s=PLACEHOLDER_ATTEMPT_WALL_S,
                  stdout_ref=store.put(f"SYNTHETIC stdout of attempt {index}\n"),
                  stdout_truncated=False, stderr_truncated=False, workdir_incomplete=False)
    return Attempt(
        index=index, response_text="SYNTHETIC reply", files={name: code},
        stage_reached="S5" if clean else "S4", diagnostics=[] if clean else [run_error()], run=run,
        alignment=Alignment() if value is None else Alignment(per_input=[value], mean=value),
        profile=Profile(runtime_s=runtime_s),
    )


def compiled_unrun(index: int, code: str, diagnostics: Sequence[Diagnostic] = ()) -> Attempt:
    """Return a SYNTHETIC attempt that compiled and never ran (S4)."""
    return Attempt(index=index, response_text="SYNTHETIC reply", files={TARGET: code},
                   stage_reached="S4", diagnostics=list(diagnostics))


def compile_failed(index: int, code: str = BAD_CODE, *, quirk: bool = False) -> Attempt:
    """Return a SYNTHETIC attempt that parsed (S1) and did not compile, with a fence-quirk warning when asked."""
    diagnostics = [Diagnostic(stage="compile", severity="error", code="synthetic", message="SYNTHETIC error")]
    if quirk:
        diagnostics.insert(0, Diagnostic(stage="parse", severity="warning", code=FENCE_QUIRK,
                                         message="SYNTHETIC fence-quirk hit"))
    return Attempt(index=index, response_text="SYNTHETIC reply", files={TARGET: code},
                   stage_reached="S1", diagnostics=diagnostics)


def no_output(index: int) -> Attempt:
    """Return a SYNTHETIC attempt with no extractable output (S0, no files) and a parse-stage warning."""
    return Attempt(index=index, response_text="SYNTHETIC reply", files={}, stage_reached="S0",
                   diagnostics=[Diagnostic(stage="parse", severity="warning", code="no-fence",
                                           message="SYNTHETIC no fenced block")])


def stale_warning(index: int) -> Diagnostic:
    """Return a SYNTHETIC stale-output warning naming attempt `index`."""
    return Diagnostic(stage="run", severity="warning", code=STALE_OUTPUT,
                      message=f"SYNTHETIC: the stdout of attempt {index} stands as the trial's output")


def past_the_gate(standing: Attempt, last_code: str) -> list[Attempt]:
    """Return the faithful loop's stale-output shape: `standing` ran, 1 to 7 failed, 8 compiled past the gate."""
    return [
        standing,
        *(compile_failed(index) for index in range(1, 8)),
        compiled_unrun(8, last_code, [stale_warning(standing.index)]),
    ]


def make_trial(attempts: Sequence[Attempt], store: TextStore, *, end: str | None = None,
               reference_ran: bool = True, direction: Direction = OMP_TO_CUDA) -> Trial:
    """Return a SYNTHETIC trial of the item holding `attempts`, with the final block the runner would write.

    `reference_ran` False leaves Trial.reference_run unset, as a compile-only
    executor does. final.alignment is the alignment mean of the last attempt
    that ran.
    """
    reference = RunInfo()
    if reference_ran:
        reference = RunInfo(exit_code=0, hang=False, wall_s=PLACEHOLDER_REFERENCE_WALL_S,
                            stdout_ref=store.put(REFERENCE_STDOUT), stdout_truncated=False,
                            stderr_truncated=False, workdir_incomplete=False)
    standing = [attempt for attempt in attempts if attempt.run.stdout_ref is not None]
    return Trial(
        trial_id=make_trial_id("lassi-repro", MODEL_ID, SUITE, direction.name, ITEM, 1),
        recipe_hash="0123456789abcdef" * 4,
        provenance=Provenance(commit=FAKE_COMMIT, dirty=False, device="hand-built (SYNTHETIC)", sdk=None,
                              date="2026-09-24T00:00:00+00:00"),
        bench_item=load_suite(SUITE_MANIFEST).bench_item(ITEM, direction),
        model=ModelInfo(backend="mock", id=MODEL_ID, sampling=SAMPLING),
        reference_run=reference,
        requests=[],
        attempts=list(attempts),
        final=Final(
            stage_reached=attempts[-1].stage_reached if attempts else None,
            alignment=standing[-1].alignment.mean if standing else None,
            corrections=max(len(attempts) - 1, 0),
            wall_s=PLACEHOLDER_REFERENCE_WALL_S,
            end_reason=None if end is None else EndReason(code=end, message=f"SYNTHETIC end: {end}"),
        ),
    )


# ---------------------------------------------------------------------------
# The cases: each builds its attempts and states every component but the similarity values


# The default direction is OpenMP to CUDA: its target reference and a SYNTHETIC CUDA candidate.
REF, CAND = REFERENCE["cuda"], CANDIDATE


@dataclass(frozen=True)
class Case:
    """One hand-built trial: its attempts, how it ended, and the components it must score."""

    name: str
    attempts: Callable[[TextStore], list[Attempt]]
    last_code: str  # the last attempt's target file ("" when it has none)
    expected: Mapping[str, float | None]  # every component but sim_t, sim_t_c, and sim_l
    end: str | None = None
    reference_ran: bool = True


def expect(correct: float | None, first_try: float | None, self_corr: float, cap_hit: float, fence_quirk: float,
           compiled: float, compiled_first_try: float) -> dict[str, float | None]:
    """Return the expected non-similarity components; correct_paper and within_10pct are always None."""
    return {
        "correct": correct, "correct_paper": None, "within_10pct": None, "first_try": first_try,
        "self_corr": self_corr, "cap_hit": cap_hit, "fence_quirk": fence_quirk, "compiled": compiled,
        "compiled_first_try": compiled_first_try,
    }


CASES = (
    Case("clean-pass-first-try",
         lambda s: [ran(0, s, REF, value=1.0, runtime_s=PLACEHOLDER_RUNTIME_S)],
         REF, expect(1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0)),
    Case("clean-pass-after-two-corrections",
         lambda s: [compile_failed(0), ran(1, s, CAND, exit_code=RUN_ERROR_EXIT, value=0.0),
                    ran(2, s, CAND, value=1.0)],
         CAND, expect(1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0)),
    Case("clean-run-output-differs",
         lambda s: [ran(0, s, CAND, value=0.0)],
         CAND, expect(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)),
    Case("oracle-match-from-a-signal-death",
         lambda s: past_the_gate(ran(0, s, CAND, exit_code=SIGNAL_EXIT, value=1.0), CAND),
         CAND, expect(0.0, 0.0, 8.0, 0.0, 0.0, 1.0, 0.0)),
    Case("oracle-match-from-a-nonzero-exit",
         lambda s: past_the_gate(ran(0, s, CAND, exit_code=RUN_ERROR_EXIT, value=1.0), CAND),
         CAND, expect(0.0, 0.0, 8.0, 0.0, 0.0, 1.0, 0.0)),
    Case("oracle-match-from-a-hang",
         lambda s: past_the_gate(ran(0, s, CAND, exit_code=None, hang=True, value=1.0), CAND),
         CAND, expect(0.0, 0.0, 8.0, 0.0, 0.0, 1.0, 0.0)),
    Case("stale-output",
         lambda s: past_the_gate(ran(0, s, REF, exit_code=RUN_ERROR_EXIT, value=0.0), CAND),
         CAND, expect(0.0, 0.0, 8.0, 0.0, 0.0, 1.0, 0.0)),
    Case("compile-only-first-try",
         lambda s: [compiled_unrun(0, CAND)],
         CAND, expect(None, None, 0.0, 0.0, 0.0, 1.0, 1.0), reference_ran=False),
    Case("compile-only-after-a-correction",
         lambda s: [compile_failed(0), compiled_unrun(1, CAND)],
         CAND, expect(None, None, 1.0, 0.0, 0.0, 1.0, 0.0), reference_ran=False),
    Case("upstream-crash",
         lambda s: [*(compile_failed(index) for index in range(9)), compiled_unrun(9, CAND)],
         CAND, expect(0.0, 0.0, 9.0, 0.0, 0.0, 1.0, 0.0), end=UPSTREAM_CRASH),
    Case("cap-hit-at-a-run-error",
         lambda s: [compile_failed(0), ran(1, s, CAND, exit_code=RUN_ERROR_EXIT),
                    ran(2, s, CAND, exit_code=RUN_ERROR_EXIT)],
         CAND, expect(0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0), end=CORRECTION_CAP),
    Case("cap-hit-at-a-compile-error",
         lambda s: [ran(0, s, CAND, exit_code=RUN_ERROR_EXIT), compile_failed(1)],
         BAD_CODE, expect(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0), end=CORRECTION_CAP),
    Case("cap-hit-with-no-output",
         lambda s: [compile_failed(0), no_output(1)],
         "", expect(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0), end=CORRECTION_CAP),
    Case("fence-quirk-hits",
         lambda s: [no_output(0), compile_failed(1, quirk=True), compile_failed(2, quirk=True),
                    ran(3, s, REF, value=1.0)],
         REF, expect(1.0, 0.0, 3.0, 0.0, 2.0, 1.0, 0.0)),
)
CASE_IDS = [case.name for case in CASES]
COMPILE_ONLY = [case for case in CASES if not case.reference_ran]


def build(case: Case, tmp_path: Path) -> tuple[Trial, Score]:
    """Return the case's trial and the lassi profile's score of it over a SYNTHETIC bench under `tmp_path`."""
    store = TextStore(tmp_path / "store")
    trial = make_trial(case.attempts(store), store, end=case.end, reference_ran=case.reference_ran)
    return trial, score_of(trial, default_bench(tmp_path / "bench"))


def expected_similarity(reference: str, candidate: str) -> dict[str, float]:
    """Return the three similarity values of `candidate` against the text-mode `reference` (P1.8 functions)."""
    reference = text_mode(reference)
    return {"sim_t": sim_t(reference, candidate), "sim_t_c": sim_t_c(reference, candidate),
            "sim_l": sim_l(reference, candidate)}


# ---------------------------------------------------------------------------
# Registration


def test_lassi_is_a_registered_score_profile_in_its_module() -> None:
    factory = profile_class()
    assert factory.__module__ == PROFILE_MODULE, f"the lassi profile lives in {PROFILE_MODULE}"
    assert factory.name == PROFILE_NAME
    assert isinstance(factory.capabilities, frozenset)


# ---------------------------------------------------------------------------
# Components per case


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_each_trial_gets_its_components(case: Case, tmp_path: Path) -> None:
    trial, score = build(case, tmp_path)
    assert isinstance(score, Score), f"ScoreProfile.score returns a Score, got {type(score).__name__}"
    assert tuple(score.components) == COMPONENTS, "the lassi profile gives exactly these components, in order"
    got = {name: score.components[name] for name in case.expected}
    assert got == dict(case.expected)
    sims = {name: score.components[name] for name in SIMILARITY}
    assert sims == expected_similarity(REF, case.last_code), "Sim values compare the last attempt's target file"
    assert score.components["self_corr"] == float(trial.final.corrections)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_scalar_is_correct(case: Case, tmp_path: Path) -> None:
    _, score = build(case, tmp_path)
    assert score.scalar == score.components["correct"]
    assert (score.scalar is None) == (case.expected["correct"] is None)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_every_value_is_a_float_or_none_and_fits_the_result_record(case: Case, tmp_path: Path) -> None:
    _, score = build(case, tmp_path)
    wrong = {name: value for name, value in score.components.items() if value is not None and type(value) is not float}
    assert not wrong, f"components are floats or None, never bool or int: {wrong}"
    assert score.scalar is None or type(score.scalar) is float
    breakdown = ScoreBreakdown(components=dict(score.components), scalar=score.scalar)
    assert breakdown.components == dict(score.components)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_every_none_component_carries_its_reason(case: Case, tmp_path: Path) -> None:
    _, score = build(case, tmp_path)
    missing = [name for name, value in score.components.items() if value is None and not score.notes.get(name)]
    assert not missing, f"a None component needs a note saying why: {missing}"
    for name, note in score.notes.items():
        assert name in COMPONENTS, f"notes are keyed by component name, got {name!r}"
        assert isinstance(note, str) and note.isascii(), f"note {name!r} must be plain ASCII text, got {note!r}"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_within_10pct_is_none_with_its_reason_while_no_timing_profiler_exists(case: Case, tmp_path: Path) -> None:
    _, score = build(case, tmp_path)
    assert score.components["within_10pct"] is None
    assert "profiler" in score.notes.get("within_10pct", "").lower()


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_paper_criterion_is_never_computed_and_is_labeled(case: Case, tmp_path: Path) -> None:
    _, score = build(case, tmp_path)
    assert score.components["correct_paper"] is None
    assert "manual inspection" in score.notes.get("correct_paper", "").lower()


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_each_similarity_value_carries_the_interpreter_version(case: Case, tmp_path: Path) -> None:
    _, score = build(case, tmp_path)
    version = f"python {platform.python_version()}"
    for name in SIMILARITY:
        assert score.components[name] is not None
        assert version in score.notes.get(name, "").lower(), f"{name} carries the interpreter version ({version})"


# ---------------------------------------------------------------------------
# Single cases that need more than the table


@pytest.mark.parametrize("case", COMPILE_ONLY, ids=[case.name for case in COMPILE_ONLY])
def test_compile_only_correct_is_labeled_a_compile_stage_reproduction(case: Case, tmp_path: Path) -> None:
    _, score = build(case, tmp_path)
    assert score.components["correct"] is None and score.scalar is None
    assert "compile-stage reproduction" in score.notes.get("correct", "").lower()


def test_an_oracle_match_from_a_crashed_run_is_not_correct(tmp_path: Path) -> None:
    case = next(case for case in CASES if case.name == "oracle-match-from-a-signal-death")
    trial, score = build(case, tmp_path)
    assert trial.final.alignment == 1.0, "the oracle matched the output that stands"
    assert score.components["correct"] == 0.0 and score.scalar == 0.0


def test_stale_output_similarity_reads_the_last_attempt_not_the_output_that_stands(tmp_path: Path) -> None:
    case = next(case for case in CASES if case.name == "stale-output")
    trial, score = build(case, tmp_path)
    assert trial.attempts[0].files[TARGET] == REF, "the standing attempt holds the reference"
    assert score.components["sim_t"] != 1.0
    assert {name: score.components[name] for name in SIMILARITY} == expected_similarity(REF, CAND)


@pytest.mark.parametrize("direction", [OMP_TO_CUDA, CUDA_TO_OMP], ids=lambda d: d.name)
def test_the_reference_target_is_read_in_text_mode(direction: Direction, tmp_path: Path) -> None:
    reference = REFERENCE[direction.target]
    crlf = reference.replace("\n", "\r\n").encode("ascii")
    bench = write_bench(tmp_path / "bench", direction, crlf, REFERENCE[direction.source].encode("ascii"))
    store = TextStore(tmp_path / "store")
    trial = make_trial([ran(0, store, reference, value=1.0, name=target_file(direction))], store,
                       direction=direction)
    assert trial.attempts[0].files == {target_file(direction): reference}
    score = score_of(trial, bench)
    assert sim_t(crlf.decode("ascii"), reference) != 1.0, "the fixture tells text mode from a binary read"
    assert {name: score.components[name] for name in SIMILARITY} == {"sim_t": 1.0, "sim_t_c": 1.0, "sim_l": 1.0}


@pytest.mark.parametrize("direction", [OMP_TO_CUDA, CUDA_TO_OMP], ids=lambda d: d.name)
def test_similarity_compares_with_the_target_language_reference(direction: Direction, tmp_path: Path) -> None:
    source = REFERENCE[direction.source]
    bench = default_bench(tmp_path / "bench", direction)
    store = TextStore(tmp_path / "store")
    trial = make_trial([ran(0, store, source, value=0.0, name=target_file(direction))], store,
                       direction=direction)
    score = score_of(trial, bench)
    want = expected_similarity(REFERENCE[direction.target], source)
    assert want["sim_t"] != 1.0
    assert {name: score.components[name] for name in SIMILARITY} == want


# ---------------------------------------------------------------------------
# The profile file


def profile_file() -> dict:
    """Return assets/scoring/lassi.yaml as a mapping; fail clearly while it does not exist."""
    if not PROFILE_FILE.is_file():
        pytest.fail(f"task P2.7 adds the lassi profile file {PROFILE_FILE.relative_to(REPO).as_posix()}")
    raw = PROFILE_FILE.read_bytes()
    assert raw.isascii(), "the profile file is plain ASCII"
    data = yaml.safe_load(raw.decode("ascii"))
    assert isinstance(data, dict), "the profile file is a mapping"
    return data


def test_the_profile_file_lists_the_components_in_order_and_names_the_scalar() -> None:
    data = profile_file()
    assert data.get("components") == list(COMPONENTS)
    assert data.get("scalar") == "correct"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_profile_gives_the_fixed_notes_of_its_file(case: Case, tmp_path: Path) -> None:
    notes = profile_file().get("notes")
    assert isinstance(notes, dict) and {"within_10pct", "correct_paper", "compile_only"} <= set(notes)
    _, score = build(case, tmp_path)
    assert score.notes["within_10pct"] == notes["within_10pct"]
    assert score.notes["correct_paper"] == notes["correct_paper"]
    if not case.reference_ran:
        assert score.notes["correct"] == notes["compile_only"]


# ---------------------------------------------------------------------------
# Cases the acceptance criteria leave open (decided with the task)


def test_score_notes_are_empty_unless_a_profile_writes_one(tmp_path: Path) -> None:
    score = Score(components={"not-measured": None}, scalar=None)
    assert score.notes == {} and score.components == {"not-measured": None}
    store = TextStore(tmp_path / "store")
    trial = make_trial([ran(0, store, REF, value=1.0)], store)
    df_v0 = DEFAULT_REGISTRY.get("ScoreProfile", "df-v0").factory()
    assert df_v0.score(trial).notes == {}, "the df-v0 profile writes no notes"


def test_a_clean_run_the_oracle_never_aligned_is_not_scored(tmp_path: Path) -> None:
    store = TextStore(tmp_path / "store")
    trial = make_trial([ran(0, store, CAND, value=None)], store)
    assert trial.final.alignment is None
    score = score_of(trial, default_bench(tmp_path / "bench"))
    assert score.components["correct"] is None and score.scalar is None
    assert score.components["first_try"] is None
    not_aligned = profile_file()["notes"]["not_aligned"]
    assert score.notes["correct"] == score.notes["first_try"] == not_aligned
    assert (score.components["compiled"], score.components["compiled_first_try"]) == (1.0, 1.0)


def test_an_unclean_run_the_oracle_never_aligned_is_not_correct(tmp_path: Path) -> None:
    store = TextStore(tmp_path / "store")
    trial = make_trial([ran(0, store, CAND, exit_code=RUN_ERROR_EXIT, value=None)], store)
    score = score_of(trial, default_bench(tmp_path / "bench"))
    assert score.components["correct"] == 0.0 and "correct" not in score.notes


BASELINE_ENDS = (("baseline-compile", None), ("baseline-run", RUN_ERROR_EXIT))


@pytest.mark.parametrize(("end", "reference_exit"), BASELINE_ENDS, ids=[end for end, _ in BASELINE_ENDS])
def test_a_trial_that_ended_at_the_baseline_is_not_scored(
    end: str, reference_exit: int | None, tmp_path: Path
) -> None:
    store = TextStore(tmp_path / "store")
    trial = make_trial([], store, end=end, reference_ran=False)
    if reference_exit is not None:
        failed = RunInfo(exit_code=reference_exit, hang=False, wall_s=PLACEHOLDER_REFERENCE_WALL_S,
                         stdout_ref=store.put(REFERENCE_STDOUT), stdout_truncated=False, stderr_truncated=False,
                         workdir_incomplete=False)
        trial = dataclasses.replace(trial, reference_run=failed)
    score = score_of(trial, default_bench(tmp_path / "bench"))
    unscored = ("correct", "first_try", "compiled", "compiled_first_try", *SIMILARITY)
    assert all(score.components[name] is None for name in unscored)
    baseline = profile_file()["notes"]["baseline"]
    for name in unscored:
        assert score.notes[name].startswith(baseline) and end in score.notes[name], name
    assert score.scalar is None
    assert {name: score.components[name] for name in ("self_corr", "cap_hit", "fence_quirk")} == {
        "self_corr": 0.0, "cap_hit": 0.0, "fence_quirk": 0.0}


def test_a_trial_with_no_attempt_past_a_clean_baseline_is_not_correct(tmp_path: Path) -> None:
    store = TextStore(tmp_path / "store")
    trial = make_trial([], store)
    score = score_of(trial, default_bench(tmp_path / "bench"))
    assert score.components["correct"] == 0.0 and score.components["first_try"] == 0.0
    assert (score.components["compiled"], score.components["compiled_first_try"]) == (0.0, 0.0)
    no_attempt = profile_file()["notes"]["no_attempt"]
    assert all(score.components[name] is None and score.notes[name] == no_attempt for name in SIMILARITY)


def test_a_missing_reference_file_raises_naming_its_path(tmp_path: Path) -> None:
    spec = load_suite(SUITE_MANIFEST).items[ITEM]
    bench = tmp_path / "bench"
    layout = spec.languages[OMP_TO_CUDA.source]
    (bench / layout.dir).mkdir(parents=True)
    (bench / layout.dir / layout.files[0]).write_bytes(REFERENCE["omp"].encode("ascii"))
    target = spec.languages[OMP_TO_CUDA.target]
    missing = (bench / target.dir / target.files[0]).as_posix()
    store = TextStore(tmp_path / "store")
    trial = make_trial([ran(0, store, CAND, value=1.0)], store)
    with pytest.raises(FileNotFoundError, match=re.escape(missing)):
        score_of(trial, bench)


def test_the_bench_root_must_be_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bench root"):
        profile_class()(bench_root=tmp_path / "not-fetched")


def write_profile(path: Path, **changes: object) -> Path:
    """Write a copy of assets/scoring/lassi.yaml with `changes` applied to its top level; return `path`."""
    data = {**profile_file(), **changes}
    path.write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    return path


def test_the_profile_file_sets_the_component_order(tmp_path: Path) -> None:
    reordered = write_profile(tmp_path / "reordered.yaml", components=list(reversed(COMPONENTS)))
    store = TextStore(tmp_path / "store")
    trial = make_trial([ran(0, store, REF, value=1.0)], store)
    score = profile_class()(bench_root=default_bench(tmp_path / "bench"), profile_path=reordered).score(trial)
    assert tuple(score.components) == tuple(reversed(COMPONENTS))
    assert score.scalar == score.components["correct"] == 1.0


def _notes_with(key: str, text: str | None) -> dict[str, str]:
    """Return the profile file's notes with note `key` set to `text`, or left out when `text` is None."""
    notes = {name: note for name, note in profile_file()["notes"].items() if name != key}
    return notes if text is None else {**notes, key: text}


# Each bad file: its id, its changes to the profile file (built when the test runs), and the name the error gives.
BAD_FILES = (
    ("missing-component", lambda: {"components": list(COMPONENTS[1:])}, "correct"),
    ("unknown-component", lambda: {"components": [*COMPONENTS, "speedup"]}, "speedup"),
    ("repeated-component", lambda: {"components": [*COMPONENTS, "sim_l"]}, "sim_l"),
    ("scalar-not-a-component", lambda: {"scalar": "pass_at_1"}, "scalar"),
    ("missing-note", lambda: {"notes": _notes_with("not_aligned", None)}, "not_aligned"),
    ("empty-note", lambda: {"notes": _notes_with("baseline", " ")}, "baseline"),
    ("non-ascii-note", lambda: {"notes": _notes_with("sim_l", "SYNTHETIC caf" + chr(0xE9))}, "sim_l"),
)


@pytest.mark.parametrize(("changes", "named"), [bad[1:] for bad in BAD_FILES], ids=[bad[0] for bad in BAD_FILES])
def test_a_bad_profile_file_is_refused_when_the_profile_is_built(
    changes: Callable[[], dict], named: str, tmp_path: Path
) -> None:
    path = write_profile(tmp_path / "bad.yaml", **changes())
    with pytest.raises(ValueError, match=re.escape(named)):
        profile_class()(bench_root=default_bench(tmp_path / "bench"), profile_path=path)
