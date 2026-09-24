"""Tests for the notebook replay harness and its fixtures (task P1.9).

Bible: Build Roadmap (P1 row, Gate: replaying recorded responses reproduces
upstream notebook decisions), Source Papers (LASSI pipeline; quirk table),
Design Principle 4, Agent Rules 1, 4, and 6.

The contract these tests fix, from the P1.9 acceptance criteria
(plans/p1-faithful.md) and the P1 phase notes ("Upstream guard", "Run report
fragments", "Bench facts", "Prompt fragments"):

- tests/replay runs the pinned notebook's pipeline function (its cells from
  third_party/LASSI at the pin, in a temporary directory) and our faithful
  stages (faithful: true; baseline, summarize_context, describe_source,
  generate, compile_loop, run_loop) on the same recorded responses and
  outcomes. The notebook's LLM, compile, execute, unload, and
  tokenizer-import calls are stubbed; a test fails if either side starts a
  subprocess or opens a socket.
- Per scenario it compares every message sent, each extracted block, which
  attempts compiled and ran, the final correction count, Sim-T, and Sim-L
  (and the reference's build and run, the end, the output that stands, and
  the fence-quirk hits). Each side's decisions must also equal the
  scenario's decisions derived by hand from its script, so two sides that
  agree on doing nothing cannot pass.
- Scenarios, in both directions: first-try success; fences tagged cpp, c++,
  c, or cuda, and untagged; no fence, several fences; compile error then fix
  (stderr from tests/toolchains/fixtures/); run error then fix; a run, then
  compile errors past 8 corrections, ending unexecuted with stale output; 9
  or more compile errors with no earlier run (the notebook raises
  UnboundLocalError; ours ends `upstream-crash`); baseline compile failure;
  baseline run failure; runs of spaces in the source.
- Fixtures are scripted, plain ASCII, and labeled synthetic in a README; they
  copy no upstream source or text. The Decision Log records that the replay
  gate reads recorded responses as scripted synthetic fixtures.
- The output shows a decision table per scenario and the fence-quirk hit
  count, labeled a check of decision logic on synthetic fixtures, not a
  measurement.

Every reply, program, and run outcome here is SYNTHETIC; the compiler stderr
files are the captured fixtures of tests/toolchains/fixtures, reused as
scripted outcomes. No value in this module is a measurement, and no upstream
text is copied into it (OQ-018).
"""

from __future__ import annotations

import ast
import json
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from notebook_replay import (
    COMPILER_FIXTURES,
    DIRECTIONS,
    FIXTURES,
    GUARD_FAILURE,
    GUARDED,
    REPO,
    REPORT_TITLE,
    RESULTS,
    Case,
    CaseResult,
    differences,
    expectation_problems,
    load_cases,
    load_manifest,
    render_report,
    run_notebook,
    run_ours,
)

from lassi.llm.replay import ReplayBackend

CASES = load_cases()
BIBLE = REPO / "docs" / "BIBLE.md"
SCORING_CONFTEST = REPO / "tests" / "scoring" / "conftest.py"

# The scenarios the plan lists (P1.9 acceptance), by the names scenarios.json gives them.
PLAN_SCENARIOS = {
    "first-try-success",
    "fence-cpp",
    "fence-c-plus-plus",
    "fence-c",
    "fence-cuda",
    "fence-untagged",
    "no-fence",
    "several-fences",
    "compile-error-then-fix",
    "run-error-then-fix",
    "stale-output-past-gate",
    "crash-past-gate-no-run",
    "baseline-compile-failure",
    "baseline-run-failure",
    "spaces-in-source",
}
# Upstream source lines at least this long must never appear in a fixture (short lines like braces are common C).
SOURCE_LINE_MIN = 40
# Upstream string constants at least this long must never appear in a fixture (the tests/prompts leak guard's bound).
TEXT_MIN = 40


def case_by_id(case_id: str) -> Case:
    """Return the case with `case_id`."""
    return next(case for case in CASES if case.id == case_id)


# ---------------------------------------------------------------------------
# The replay


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_notebook_and_faithful_stages_make_the_same_decisions(case: Case, replay_env, tmp_path: Path) -> None:
    notebook = run_notebook(case, replay_env.notebook, replay_env.fragments, tmp_path / "notebook")
    ours = run_ours(case, tmp_path / "ours")
    problems = differences(notebook, ours)
    RESULTS.append(CaseResult(
        scenario=case.scenario,
        direction=case.direction.name,
        codes=ours.codes(),
        corrections=ours.corrections,
        end=ours.end,
        sims_equal=(notebook.sim_t, notebook.sim_l) == (ours.sim_t, ours.sim_l),
        fence_quirk=ours.fence_quirk,
        same=not problems,
    ))
    assert not problems, f"{case.id}: the notebook and our faithful stages decide differently:\n" + "\n".join(problems)
    for side, decisions in (("notebook", notebook), ("ours", ours)):
        wrong = expectation_problems(case.expected, decisions)
        assert not wrong, f"{case.id}: {side} does not follow the scenario's script ({case.what}):\n" + "\n".join(wrong)


def test_every_scenario_the_plan_lists_runs_in_both_directions() -> None:
    names = {case.scenario for case in CASES}
    missing = sorted(PLAN_SCENARIOS - names)
    assert not missing, f"scenarios.json lacks the plan's scenarios {missing}"
    for name in names:
        directions = sorted(case.direction.name for case in CASES if case.scenario == name)
        assert directions == sorted(direction.name for direction in DIRECTIONS), f"{name} runs in {directions}"


def test_the_crash_scenario_has_nine_or_more_compile_errors_and_no_earlier_run() -> None:
    case = case_by_id("crash-past-gate-no-run-cuda-omp")
    attempt_compiles = case.compiles[1:]
    assert sum(not outcome.ok for outcome in attempt_compiles) >= 9
    assert [outcome.ok for outcome in attempt_compiles][-1] is True
    assert len(case.runs) == 1, "only the reference runs"


def test_the_stale_output_scenario_runs_once_then_compiles_past_eight_corrections() -> None:
    case = case_by_id("stale-output-past-gate-omp-cuda")
    assert case.expected.ran == (0,) and case.expected.corrections is not None and case.expected.corrections >= 8
    assert case.expected.stale_output and case.expected.output_from == 0


# ---------------------------------------------------------------------------
# The guard


def test_guard_fails_when_the_notebook_side_starts_a_process(replay_env, tmp_path: Path) -> None:
    case = case_by_id("first-try-success-cuda-omp")
    with pytest.raises(pytest.fail.Exception, match=GUARD_FAILURE):
        run_notebook(case, replay_env.notebook, replay_env.fragments, tmp_path / "notebook", processes=subprocess)


@pytest.mark.parametrize("attempt", ["process", "socket"])
def test_guard_fails_when_our_side_starts_a_process_or_opens_a_socket(attempt: str, replay_env, tmp_path: Path) -> None:
    case = case_by_id("first-try-success-omp-cuda")

    def start_something() -> None:
        try:
            if attempt == "process":
                subprocess.run([sys.executable, "-c", "pass"], check=False)
            else:
                socket.create_connection(("127.0.0.1", 9), timeout=1)
        except Exception:  # a swallowed refusal must still fail the test
            pass

    with pytest.raises(pytest.fail.Exception, match=GUARD_FAILURE):
        run_ours(case, tmp_path / "ours", on_build=start_something)


def test_guard_table_matches_the_scoring_guard() -> None:
    tree = ast.parse(SCORING_CONFTEST.read_bytes().decode("utf-8"))
    tables = [
        node.value for node in tree.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "GUARDED"
    ]
    assert len(tables) == 1, "tests/scoring/conftest.py must define one GUARDED table"
    assert ast.literal_eval(tables[0]) == GUARDED


# ---------------------------------------------------------------------------
# The fixtures


def fixture_files() -> list[Path]:
    """Return every file under tests/fixtures/replay."""
    return sorted(path for path in FIXTURES.rglob("*") if path.is_file())


def test_fixtures_are_plain_ascii() -> None:
    files = fixture_files()
    assert files, f"no fixture under {FIXTURES}"
    offending = [path.relative_to(REPO).as_posix() for path in files if not path.read_bytes().isascii()]
    assert not offending, f"fixtures that are not plain ASCII: {offending}"


def test_readme_labels_the_fixtures_synthetic_and_not_a_measurement() -> None:
    readme = FIXTURES / "README.md"
    assert readme.is_file(), f"missing {readme}"
    text = " ".join(readme.read_bytes().decode("ascii").split())
    for phrase in ("SYNTHETIC", "not model output", "not a measurement", "scripted"):
        assert phrase.lower() in text.lower(), f"{readme} must say {phrase!r}"


def test_scenarios_and_every_recording_are_labeled_synthetic() -> None:
    manifest = load_manifest()
    recordings = {FIXTURES / scenario["recording"] for scenario in manifest["scenarios"]}
    on_disk = set((FIXTURES / "recordings").glob("*.json"))
    assert recordings == on_disk, f"recordings without a scenario or missing: {sorted(recordings ^ on_disk)}"
    for path in sorted(recordings):
        assert json.loads(path.read_bytes())["synthetic"] is True, f'{path} must be labeled "synthetic": true'


def test_every_recording_loads_in_the_replay_backend_and_is_used_up_exactly() -> None:
    manifest = load_manifest()
    for scenario in manifest["scenarios"]:
        backend = ReplayBackend("fixture-check", recording=FIXTURES / scenario["recording"])
        count = len(json.loads((FIXTURES / scenario["recording"]).read_bytes())["completions"])
        expected_calls = 0 if scenario["expect"]["attempts"] == 0 else 2 + scenario["expect"]["attempts"]
        assert count == expected_calls, f"{scenario['name']}: {count} replies for {expected_calls} model calls"
        assert backend.model_id == "fixture-check"


def test_compile_outcomes_use_the_captured_compiler_stderr() -> None:
    manifest = load_manifest()
    for name, kind in manifest["compile_kinds"].items():
        for language, file_name in kind["stderr"].items():
            path = COMPILER_FIXTURES / file_name
            assert path.is_file(), f"compile kind {name} ({language}) names {file_name}, which is not a fixture"
    failing = {name for name, kind in manifest["compile_kinds"].items() if not kind["ok"]}
    for name in failing:
        for file_name in manifest["compile_kinds"][name]["stderr"].values():
            assert (COMPILER_FIXTURES / file_name).read_bytes().strip(), f"{name}: {file_name} holds no stderr"


def upstream_texts(checkout: Path) -> tuple[set[str], set[str]]:
    """Return upstream's long source lines (the 20 mains) and its long string constants (dictionary and notebook)."""
    lines: set[str] = set()
    for path in (checkout / "translated_code").rglob("*_main.*"):
        for line in path.read_bytes().decode("utf-8", "replace").splitlines():
            if len(line.strip()) >= SOURCE_LINE_MIN:
                lines.add(line.strip())
    texts: set[str] = set()
    tree = ast.parse((checkout / "prompt_dictionary.py").read_bytes().decode("utf-8"))
    texts |= {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    notebook = json.loads((checkout / "LASSI_pipeline_v0.ipynb").read_bytes().decode("utf-8"))
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        try:
            cell_tree = ast.parse(source)
        except SyntaxError:
            continue
        texts |= {
            node.value for node in ast.walk(cell_tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
    return lines, {text.strip() for text in texts if len(text.strip()) >= TEXT_MIN}


def fixture_texts() -> list[tuple[str, str]]:
    """Return (label, text) for every fixture file and every string inside the JSON fixtures, decoded."""
    found = []
    for path in fixture_files():
        label = path.relative_to(REPO).as_posix()
        raw = path.read_bytes().decode("ascii")
        found.append((label, raw))
        if path.suffix == ".json":
            stack = [json.loads(raw)]
            while stack:
                value = stack.pop()
                if isinstance(value, str):
                    found.append((label, value))
                elif isinstance(value, dict):
                    stack.extend(value.values())
                elif isinstance(value, list):
                    stack.extend(value)
    return found


def test_fixtures_copy_no_upstream_source_or_text(upstream_checkout: Path) -> None:
    lines, texts = upstream_texts(upstream_checkout)
    assert lines and texts, "the upstream checkout yielded nothing to compare"
    hits = []
    for label, text in fixture_texts():
        normalized = text.replace("\r\n", "\n")
        stripped_lines = {line.strip() for line in normalized.splitlines()}
        hits += [f"{label}: upstream source line {line[:50]!r}" for line in lines if line in stripped_lines]
        hits += [f"{label}: upstream text {value[:50]!r}" for value in texts if value in normalized]
    assert not hits, "fixtures hold upstream source or text (OQ-018):\n" + "\n".join(sorted(set(hits)))


# ---------------------------------------------------------------------------
# The report and the Decision Log


def test_report_shows_a_decision_table_and_the_fence_quirk_count_labeled_synthetic() -> None:
    rows = [
        CaseResult("fence-cuda", "cuda-omp", "x R", 1, "complete", True, 1, True),
        CaseResult("crash-past-gate-no-run", "omp-cuda", "x x c", 2, "upstream-crash", True, 0, False),
    ]
    report = render_report(rows)
    assert report.splitlines()[0] == REPORT_TITLE
    flat = " ".join(report.split())
    assert "check of decision logic on synthetic fixtures" in flat and "not a measurement" in flat
    assert "| fence-cuda | cuda-omp | x R | 1 | complete |" in report
    assert "| crash-past-gate-no-run | omp-cuda | x x c | 2 | upstream-crash |" in report
    assert re.search(r"Fence-quirk hits on the synthetic fixtures: 1\b", report)
    assert "not the Evaluation Protocol's fence-quirk replay count" in flat


def decision_log_rows() -> list[str]:
    """Return the table rows of the bible's Decision Log section."""
    text = BIBLE.read_bytes().decode("utf-8").replace("\r\n", "\n")
    section = text.split("\n## Decision Log\n", 1)[1].split("\n## ", 1)[0]
    return [line for line in section.splitlines() if re.match(r"\| \d{4}-\d{2}-\d{2} \|", line)]


def test_decision_log_records_that_the_replay_reads_scripted_synthetic_fixtures() -> None:
    rows = [row for row in decision_log_rows() if "P1.9" in row]
    matching = [row for row in rows if "replay" in row.lower() and "synthetic" in row.lower()]
    assert matching, (
        "docs/BIBLE.md's Decision Log has no P1.9 entry recording that the replay gate reads recorded responses "
        "as scripted synthetic fixtures"
    )
