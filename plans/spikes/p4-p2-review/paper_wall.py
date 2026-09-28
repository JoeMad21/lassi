"""P2 review question 2 (task P4.15): would the attempt wall limit have cut any of the paper's correct trials?

Usage: <python> plans/spikes/p4-p2-review/paper_wall.py <pypdf text of arXiv:2407.01638v2>

It reads Table IV (each app's reference runtime per target language) and Tables VI and VII (each correct
trial's Runtime and Ratio) the way the P2.4 spike's recount.py does (plans/spikes/p2-lassi-metrics.md,
Recount), and compares each correct trial's Runtime with the limit run_loop sets under sandbox.wall_s
baseline_x10: ten times the reference, never less than 30 s (bible Sandbox). These are counts read from
published tables, not measurements; the harness times whole processes, while the paper's timings are its own
(an unstated method). It prints numbers only.
"""

from __future__ import annotations

import re
import sys

APPS = ["matrix-rotate", "jacobi", "layout", "atomicCost", "dense-embedding",
        "pathfinder", "bsearch", "entropy", "colorwheel", "randomAccess"]
MODELS = ["GPT-4", "Codestral", "WizardCoder", "DeepSeek"]
FLOOR_S, TIMES = 30.0, 10


def table_iv(text: str) -> dict[str, dict[str, float]]:
    """Return each app's reference runtime by target language (CUDA, OMP) from Table IV."""
    found = {}
    for app in APPS:
        match = re.search(re.escape(app) + r" (?:\[[^\]]*\]|None) ([0-9.]+) ([0-9.]+)", text)
        found[app] = {"CUDA": float(match.group(1)), "OMP": float(match.group(2))}
    return found


def panel(text: str, start: str, end: str) -> list[tuple[str, str, float, float]]:
    """Return (app, model, Runtime, Ratio) of every correct trial between the markers."""
    segment = text[text.index(start):text.index(end)]
    found = []
    for app in APPS:
        tokens = re.search(r"^" + re.escape(app) + r" (.*)$", segment, re.M).group(1).split()
        assert len(tokens) == 10, (app, len(tokens))
        for model, half in zip(MODELS[:2] if "(a)" in start else MODELS[2:], (tokens[:5], tokens[5:]), strict=True):
            if half[0] != "N/A":
                found.append((app, model, float(half[0]), float(half[1])))
    return found


def main() -> None:
    """Print, per direction, the slowest correct trials against their limits and how many exceed the limit."""
    text = open(sys.argv[1], encoding="utf-8").read()
    reference = table_iv(text)
    directions = {
        "OMP -> CUDA": ("CUDA", panel(text, "(a) Panel A", "(b) Panel B")
                        + panel(text, "(b) Panel B", "represents the number")),
        "CUDA -> OMP": ("OMP", panel(text, "(a) *Panel A", "(b) *Panel B")
                        + panel(text, "(b) *Panel B", "D. Discussion")),
    }
    for name, (target, trials) in directions.items():
        over = []
        for app, model, runtime, _ratio in trials:
            limit = max(TIMES * reference[app][target], FLOOR_S)
            if runtime > limit:
                over.append((app, model, runtime, limit))
        slowest = sorted(trials, key=lambda trial: trial[3])[:3]
        shown = ", ".join(f"{model} {app} Ratio {ratio} Runtime {runtime} (reference {reference[app][target]}, "
                          f"limit {max(TIMES * reference[app][target], FLOOR_S):g})"
                          for app, model, runtime, ratio in slowest)
        print(f"{name}: correct trials {len(trials)}; over the limit {len(over)} {over}; lowest Ratios: {shown}")


if __name__ == "__main__":
    main()
