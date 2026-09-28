"""P2 review questions 4 and 5 (task P4.15): the run metric readings on SYNTHETIC groups, before and after.

Usage, from the root of the tree under test (PYTHONPATH=<tree> pins the imports to it):

    <python> plans/spikes/p4-p2-review/q5_populations.py <tree>

It imports the hand-built SYNTHETIC trials and scores of <tree>/tests/analysis/test_run_metrics.py (Spec, MAIN,
COMPILE_ONLY, pairs_of) and runs lassi.analysis.metrics.metric_tables of <tree> over five groups:

- (a) MAIN with layout run 4 read as the lassi profile reads a trial that ended at the baseline (compiled,
  correct, first_try, and sim_t None), for the five trial rates;
- (b) MAIN plus two entropy trials whose correct is None (clean runs the oracle never aligned), for pass@k;
- (c) six correct trials whose sim_t sits at the 0.6 boundary, for Sim-T >= 0.6;
- (d) the compile-only group plus one trial that ended at the baseline, for the compile-stage label;
- (e) MAIN in each direction, for the B0 criterion.

Every value is SYNTHETIC, not a measurement. It prints counts, values, and notes only.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path


def rows(table: object) -> dict[str, object]:
    """Return the table's rows by name."""
    return {row.name: row for row in table.rows}  # type: ignore[attr-defined]


def show(label: str, row: object) -> None:
    """Print one row: its count, value, and note."""
    count = "-" if row.numerator is None else f"{row.numerator:g}/{row.denominator}"  # type: ignore[attr-defined]
    value = "None" if row.value is None else f"{row.value:.4f}"  # type: ignore[attr-defined]
    print(f"{label} {row.name}: {count} value {value} | {row.note}")  # type: ignore[attr-defined]


def main() -> None:
    """Print the rows each group turns on."""
    tree = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(tree / "tests" / "analysis"))
    import test_run_metrics as t

    from lassi.analysis.metrics import metric_tables

    def table(specs: tuple, direction: str = t.OMP_TO_CUDA, model: str = t.MODEL_A, **kwargs: bool) -> object:
        return metric_tables(t.pairs_of(specs, model, direction, **kwargs))[0]

    baseline = dict(stage=None, correct=None, first_try=None, sim_t=None, compiled=None,
                    correct_note="SYNTHETIC note: not computed: the trial ended at the baseline (baseline-run)")
    group_a = tuple(replace(spec, **baseline) if (spec.item, spec.run) == ("layout", 4) else spec for spec in t.MAIN)
    for name in ("compile_rate", "run_rate", "correct_rate", "cap_hit_rate", "fence_quirk_rate"):
        show("(a)", rows(table(group_a))[name])
    never = tuple(t.Spec("entropy", run, "S5", 0, None, None, 0.5, 1.0, 0.0, 0.0,
                         correct_note="SYNTHETIC note: not computed: a clean run the oracle did not align")
                  for run in (1, 2))
    for name in ("pass@1", "pass@3"):
        show("(b)", rows(table(t.MAIN + never))[name])
    boundary = tuple(t.Spec("bsearch", run, "S5", 0, 1.0, 1.0, value, 1.0, 0.0, 0.0)
                     for run, value in enumerate((0.5996, 0.5951, 0.595, 0.6, 0.59, 0.5949), 1))
    show("(c) sim_t 0.5996 0.5951 0.595 0.6 0.59 0.5949:", rows(table(boundary))["sim_t_ge_0.6_rate"])
    ended = replace(t.MAIN[3], item="bsearch", run=4, **baseline)
    labeled = table((*t.COMPILE_ONLY, ended), t.CUDA_TO_OMP, t.MODEL_C, compile_only=True)
    print(f"(d) compile_only {labeled.compile_only}; correct_rate paper values: "
          f"{rows(labeled)['correct_rate'].paper_recount}")
    for direction in (t.OMP_TO_CUDA, t.CUDA_TO_OMP):
        print(f"(e) {direction} b0_criterion: {getattr(table(t.MAIN, direction), 'b0_criterion', 'no such field')}")


if __name__ == "__main__":
    main()
