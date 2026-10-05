"""P2 review question 8 (task P4.15): what the df-v0 weights imply, by exact arithmetic.

Usage: <python> plans/spikes/p4-p2-review/q8_weights.py < assets/scoring/df-v0.yaml

It reads the weights file on stdin (for example `git show 1ccf1fb:assets/scoring/df-v0.yaml`), imports no
lassi code, and applies the bible's formula (Reward Function) in exact fractions: R = b(s) - w min(W, cap)
+ a A 1[s = S5], a guard violation sets R = g, and the multi-turn value is R_final - p x corrections, where W
counts only at S4 or S5 and A only at S5. It prints the implied ranges and orderings. Arithmetic, not
measurements.
"""

from __future__ import annotations

import sys
from fractions import Fraction

import yaml


def main() -> None:
    """Print the ranges and orderings the weights imply."""
    data = yaml.safe_load(sys.stdin.read())
    base = {stage: Fraction(str(value)) for stage, value in data["stage_base"].items()}
    w, cap = Fraction(str(data["warning_weight"])), int(data["warning_cap"])
    a, g, p = (Fraction(str(data[key])) for key in ("alignment_weight", "guard_violation", "correction_penalty"))

    def r(stage: str, warnings: int = 0, alignment: Fraction = Fraction(0)) -> Fraction:
        counted = min(warnings, cap) if stage in ("S4", "S5") else 0
        return base[stage] - w * counted + (a * alignment if stage == "S5" else 0)

    def multi(stage: str, corrections: int, warnings: int = 0, alignment: Fraction = Fraction(0)) -> Fraction:
        return r(stage, warnings, alignment) - p * corrections

    show = float
    ranges = {stage: (show(r(stage, cap, Fraction(0))), show(r(stage, 0, Fraction(1)))) for stage in base}
    print("unguarded attempt R range:", ranges)
    print("min S5 - max S4 =", show(r("S5", cap) - r("S4")))
    print("warning_weight x warning_cap =", show(w * cap), "; S5 base - S4 base =", show(base["S5"] - base["S4"]))
    print("guard_violation =", show(g), "; S0 base =", show(base["S0"]), "; S1 base =", show(base["S1"]))
    print("corrections per stage step (0.2 / correction_penalty) =", show(Fraction("0.2") / p))
    print("S5 clean, A = 1, 5 corrections:", show(multi("S5", 5, 0, Fraction(1))),
          "; S5 clean, A = 0, 5 corrections:", show(multi("S5", 5)), "; S4 clean, 0 corrections:", show(multi("S4", 0)))
    print("S2, 5 corrections:", show(multi("S2", 5)), "; S1, 0 corrections:", show(multi("S1", 0)))
    print("S5 at W >= 10, A = 0, 1 correction:", show(multi("S5", 1, cap)))
    for limit in (5, 10):
        low = min(min(g, min(r(stage, cap) for stage in base)) - p * limit, g - p * limit)
        high = max(r(stage, 0, Fraction(1)) for stage in base)
        print(f"cap {limit}: multi_turn range [{show(low)}, {show(high)}]")


if __name__ == "__main__":
    main()
