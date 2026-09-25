"""The lassi score profile: the LASSI reproduction's metrics of one trial.

Bible: Evaluation Protocol (LASSI reproduction row; LASSI Paper Metrics;
Acceptance Criteria, the compile-only label), Project Recipes (the
lassi-repro `metrics` line), Source Papers (LASSI quirk table, Sim-T and
fence rows), Oracles (stdout_mask rules), Harness Contract (the proxy checks
outputs, never runtime), Component Interfaces (ScoreProfile).

A profile file, by default assets/scoring/lassi.yaml, names the components
in the order a Score lists them, the component that is the scalar, and every
note text; this module holds the definitions. The profile is built as
`LassiProfile(bench_root=<root of the suite's fetched sources>)`. The
reference target is the item's one file in the direction's target language,
found through the suite manifest assets/bench/<suite>.yaml under that root
and read in text mode (lassi.core.fragments.as_text_mode), as the notebook
reads it. A missing reference file raises FileNotFoundError naming its path;
nothing stands in for it.

Components of a trial (task P2.7), each a float, or None with a note:

- correct: 1.0 when the output that stands (lassi.core.record
  standing_attempt) came from a clean run, exit status 0 and hang False,
  that the oracle aligned at 1.0. 0.0 when that run was not clean, when the
  oracle gave less than 1.0, or when no output stands. None in three cases,
  each with its note: a trial with no attempt that ended at the baseline
  (the model was never asked); a compile-only trial (the target reference
  was not run and no attempt ran), labeled a compile-stage reproduction;
  and a clean standing run the oracle never aligned (no reference stdout,
  or a truncated one). The scalar is correct.
- correct_paper: the paper's criterion, a manual inspection of stdout.
  Never computed: always None.
- within_10pct: None while no timing profiler exists, even when an attempt
  carries a runtime.
- first_try: 1.0 when correct is 1.0 and final.corrections is 0, None when
  correct is None (with correct's note), else 0.0.
- sim_t, sim_t_c, sim_l: lassi.scoring.similarity over the text-mode
  reference target and the last attempt's target file ("" when that attempt
  has none). Each note names the interpreter that computed the value. None
  when the trial holds no attempt.
- self_corr: final.corrections. cap_hit: 1.0 when final.end_reason is
  correction-cap, else 0.0. fence_quirk: the number of fence-quirk
  diagnostics over all attempts.
- compiled: 1.0 when the last attempt reached S4 or S5. compiled_first_try:
  1.0 when compiled is 1.0 and final.corrections is 0. Both are None for a
  trial that ended at the baseline and 0.0 for any other trial with no
  attempt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from lassi.bench import Direction, Suite, load_suite
from lassi.core.capabilities import READS_BENCH_SOURCES
from lassi.core.fragments import as_text_mode
from lassi.core.interfaces import Score
from lassi.core.record import RunInfo, Trial, standing_attempt
from lassi.core.registry import register
from lassi.scoring.similarity import measure

REPO = Path(__file__).resolve().parents[2]
# The profile file a recipe's `score: lassi` reads.
PROFILE_FILE = REPO / "assets" / "scoring" / "lassi.yaml"
# Where suite manifests live: bench_item.suite names <BENCH_DIR>/<suite>.yaml.
BENCH_DIR = REPO / "assets" / "bench"
# Scoring reads an item's reference target; it never trains on it (Agent Rule 5).
PURPOSE = "eval"

CORRECT, CORRECT_PAPER, WITHIN_10PCT, FIRST_TRY = "correct", "correct_paper", "within_10pct", "first_try"
SIMILARITY = ("sim_t", "sim_t_c", "sim_l")
SELF_CORR, CAP_HIT, FENCE_QUIRK = "self_corr", "cap_hit", "fence_quirk"
COMPILED, COMPILED_FIRST_TRY = "compiled", "compiled_first_try"
# Every component the profile computes; the profile file lists each one once, in the order a Score gives them.
COMPONENTS = frozenset(
    {CORRECT, CORRECT_PAPER, WITHIN_10PCT, FIRST_TRY, *SIMILARITY, SELF_CORR, CAP_HIT, FENCE_QUIRK, COMPILED,
     COMPILED_FIRST_TRY}
)
# The note texts the profile file holds, in file order.
COMPILE_ONLY, NOT_ALIGNED, BASELINE, NO_ATTEMPT = "compile_only", "not_aligned", "baseline", "no_attempt"
NOTE_KEYS = (WITHIN_10PCT, CORRECT_PAPER, COMPILE_ONLY, NOT_ALIGNED, BASELINE, NO_ATTEMPT, *SIMILARITY)
FILE_KEYS = ("components", "scalar", "notes")

# Record codes the components read.
BASELINE_ENDS = frozenset({"baseline-compile", "baseline-run"})
CAP_END = "correction-cap"
FENCE_QUIRK_CODE = "fence-quirk"
COMPILED_STAGES = frozenset({"S4", "S5"})


@dataclass(frozen=True)
class ProfileFile:
    """The lassi profile file as load_profile reads it: component order, the scalar's component, the notes."""

    components: tuple[str, ...]
    scalar: str
    notes: Mapping[str, str]


def _check_names(names: Sequence[Any], expected: Sequence[str] | frozenset[str], what: str, where: str) -> None:
    """Raise ValueError naming each name of `expected` missing from `names`, each unknown name, and each repeat."""
    missing = sorted(name for name in expected if name not in names)
    unknown = sorted(str(name) for name in names if name not in expected)
    repeated = sorted({str(name) for name in names if list(names).count(name) > 1})
    for problem, found in (("missing", missing), ("unknown", unknown), ("repeated", repeated)):
        if found:
            raise ValueError(f"{where}: {problem} {what} {', '.join(found)}; the lassi profile has no default")


def load_profile(path: str | Path) -> ProfileFile:
    """Return the lassi profile file at `path`.

    Raises ValueError naming the file and the key for a missing file, a
    missing or unknown key, a components list that does not name every
    component the profile computes exactly once, a scalar that is not one of
    them, or a note that is not a non-empty plain ASCII string.
    """
    where = Path(path).as_posix()
    if not Path(path).is_file():
        raise ValueError(f"the lassi profile file {where} does not exist")
    data = yaml.safe_load(Path(path).read_bytes().decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a mapping with the keys {', '.join(FILE_KEYS)}, got {data!r}")
    _check_names(list(data), FILE_KEYS, "key", where)
    components = data["components"]
    if not isinstance(components, list) or not all(isinstance(name, str) for name in components):
        raise ValueError(f"{where} components: expected a list of component names, got {components!r}")
    _check_names(components, COMPONENTS, "component", f"{where} components")
    scalar = data["scalar"]
    if scalar not in components:
        raise ValueError(f"{where} scalar: expected one of the components, got {scalar!r}")
    notes = data["notes"]
    if not isinstance(notes, dict):
        raise ValueError(f"{where} notes: expected a mapping of note texts, got {notes!r}")
    _check_names(list(notes), NOTE_KEYS, "note", f"{where} notes")
    for key in NOTE_KEYS:
        text = notes[key]
        if not isinstance(text, str) or not text.strip() or not text.isascii():
            raise ValueError(f"{where} notes.{key}: expected a non-empty plain ASCII text, got {text!r}")
    ordered = {key: notes[key] for key in NOTE_KEYS}
    return ProfileFile(components=tuple(components), scalar=scalar, notes=MappingProxyType(ordered))


def _ran(run: RunInfo) -> bool:
    """Return True when the record shows that the run happened: an exit status, a hang, or kept stdout."""
    return run.exit_code is not None or run.hang is True or run.stdout_ref is not None


def baseline_end(trial: Trial) -> str | None:
    """Return the end reason code of a trial that holds no attempt and ended at the baseline, else None."""
    end = trial.final.end_reason
    if not trial.attempts and end is not None and end.code in BASELINE_ENDS:
        return end.code
    return None


def correct_value(trial: Trial, notes: Mapping[str, str]) -> tuple[float | None, str | None]:
    """Return the component correct and, when it is None, the note saying why (see the module docstring)."""
    end = baseline_end(trial)
    if end is not None:
        return None, f"{notes[BASELINE]} ({end})"
    standing = standing_attempt(trial)
    if standing is None:
        return (0.0, None) if _ran(trial.reference_run) else (None, notes[COMPILE_ONLY])
    run = standing.run
    if run.exit_code != 0 or run.hang is not False:
        return 0.0, None
    if standing.alignment.mean is None:
        return None, notes[NOT_ALIGNED]
    return (1.0 if standing.alignment.mean == 1.0 else 0.0), None


def outcome_components(trial: Trial, notes: Mapping[str, str]) -> tuple[dict[str, float | None], dict[str, str]]:
    """Return correct, first_try, compiled, and compiled_first_try, with a note for each one that is None."""
    first = trial.final.corrections == 0
    correct, why = correct_value(trial, notes)
    values: dict[str, float | None] = {
        CORRECT: correct,
        FIRST_TRY: None if correct is None else float(correct == 1.0 and first),
    }
    written: dict[str, str] = {} if why is None else {CORRECT: why, FIRST_TRY: why}
    end = baseline_end(trial)
    if end is not None:
        values[COMPILED] = values[COMPILED_FIRST_TRY] = None
        written[COMPILED] = written[COMPILED_FIRST_TRY] = f"{notes[BASELINE]} ({end})"
    else:
        compiled = bool(trial.attempts) and trial.attempts[-1].stage_reached in COMPILED_STAGES
        values[COMPILED] = float(compiled)
        values[COMPILED_FIRST_TRY] = float(compiled and first)
    return values, written


def count_components(trial: Trial) -> dict[str, float]:
    """Return self_corr, cap_hit, and fence_quirk, each as a float."""
    end = trial.final.end_reason
    hits = sum(item.code == FENCE_QUIRK_CODE for attempt in trial.attempts for item in attempt.diagnostics)
    return {
        SELF_CORR: float(trial.final.corrections),
        CAP_HIT: float(end is not None and end.code == CAP_END),
        FENCE_QUIRK: float(hits),
    }


def _direction(suite: Suite, item: str, name: str) -> Direction:
    """Return the direction of `item` whose name is `name`; raise ValueError when no pair of its languages has it."""
    languages = suite.item(item, purpose=PURPOSE).languages
    found = [Direction(a, b) for a in languages for b in languages if a != b and Direction(a, b).name == name]
    if len(found) != 1:
        raise ValueError(
            f"{suite.name}/{item}: the direction {name!r} names no pair of its languages ({', '.join(languages)})"
        )
    return found[0]


@register("ScoreProfile", "lassi")
class LassiProfile:
    """The lassi ScoreProfile: the LASSI reproduction's components of one trial, with correct as the scalar.

    Built as `LassiProfile(bench_root=<root of the suite's fetched sources>)`
    it reads PROFILE_FILE; `profile_path` names another profile file. The
    file is read and checked, and the bench root must be a directory, when
    the profile is built. It declares READS_BENCH_SOURCES, so
    lassi.scoring.profiles.build_profile passes it the bench root.
    """

    name = "lassi"
    capabilities = frozenset({"scores_trials", READS_BENCH_SOURCES})

    def __init__(self, *, bench_root: str | Path, profile_path: str | Path | None = None) -> None:
        """Check the bench root and read the profile file: `profile_path`, or PROFILE_FILE when it is None."""
        root = Path(bench_root)
        if not root.is_dir():
            raise ValueError(f"the bench root {root.as_posix()} is not a directory; name the suite's fetched sources")
        self.bench_root = root
        self.profile_path = PROFILE_FILE if profile_path is None else Path(profile_path)
        self.profile = load_profile(self.profile_path)
        self._suites: dict[str, Suite] = {}

    def score(self, trial: Trial) -> Score:
        """Return the trial's Score: every component in the profile file's order, the scalar, and the notes."""
        notes = self.profile.notes
        values: dict[str, float | None] = {CORRECT_PAPER: None, WITHIN_10PCT: None}
        written = {CORRECT_PAPER: notes[CORRECT_PAPER], WITHIN_10PCT: notes[WITHIN_10PCT]}
        outcome, outcome_notes = outcome_components(trial, notes)
        similarity, similarity_notes = self.similarity_components(trial)
        for found, noted in ((outcome, outcome_notes), (similarity, similarity_notes), (count_components(trial), {})):
            values.update(found)
            written.update(noted)
        components = {name: values[name] for name in self.profile.components}
        ordered = {name: written[name] for name in self.profile.components if name in written}
        return Score(components=components, scalar=components[self.profile.scalar], notes=ordered)

    def similarity_components(self, trial: Trial) -> tuple[dict[str, float | None], dict[str, str]]:
        """Return sim_t, sim_t_c, and sim_l of the last attempt's target file, each noted with its interpreter."""
        if not trial.attempts:
            end = baseline_end(trial)
            why = self.profile.notes[NO_ATTEMPT] if end is None else f"{self.profile.notes[BASELINE]} ({end})"
            return dict.fromkeys(SIMILARITY), dict.fromkeys(SIMILARITY, why)
        name, reference = self.reference_target(trial)
        values = measure(reference, trial.attempts[-1].files.get(name, ""))
        found = {"sim_t": values.sim_t, "sim_t_c": values.sim_t_c, "sim_l": values.sim_l}
        notes = {key: f"{self.profile.notes[key]}; computed by python {values.python}" for key in SIMILARITY}
        return found, notes

    def reference_target(self, trial: Trial) -> tuple[str, str]:
        """Return the name of the trial item's one target-language file and its text as text mode reads it.

        Raises ValueError when the suite, item, or direction is unknown or the
        target language has more or fewer than one file, and FileNotFoundError
        naming the path when the file is missing under the bench root.
        """
        bench = trial.bench_item
        suite = self._suite(bench.suite)
        direction = _direction(suite, bench.item, bench.direction)
        spec = suite.item(bench.item, purpose=PURPOSE).languages[direction.target]
        where = f"{suite.name}/{bench.item} ({direction.name})"
        if len(spec.files) != 1:
            raise ValueError(f"{where}: the lassi profile compares one target file; the manifest has {len(spec.files)}")
        path = self.bench_root / spec.dir / spec.files[0]
        if not path.is_file():
            raise FileNotFoundError(f"the reference target of {where} is missing: {path.as_posix()}")
        files = suite.reference_target(bench.item, direction, self.bench_root, purpose=PURPOSE)
        return spec.files[0], as_text_mode(files[spec.files[0]])

    def _suite(self, name: str) -> Suite:
        """Return the suite manifest assets/bench/<name>.yaml, loaded once per profile."""
        if name not in self._suites:
            manifest = BENCH_DIR / f"{name}.yaml"
            if not manifest.is_file():
                raise ValueError(f"the suite {name!r} has no manifest; expected {manifest.as_posix()}")
            self._suites[name] = load_suite(manifest)
        return self._suites[name]
