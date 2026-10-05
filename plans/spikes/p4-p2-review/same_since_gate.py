"""P2 review questions 2 and 3 (task P4.15): which functions the reviewed readings rest on changed since the gate.

Usage, from the repository root: <python> plans/spikes/p4-p2-review/same_since_gate.py <old commit> <new commit>

For each named function it reads the file at both commits with `git show`, finds the function (or method) with
ast, and prints SAME or CHANGED with the sha256 of its source segment at each commit, or ABSENT. Then it prints
the sha256 of each assets/scoring file at the new commit. It prints names and digests only.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
import sys

FUNCTIONS = {
    "lassi/core/record.py": ["standing_attempt"],
    "lassi/core/stages.py": ["RunLoopStage.__call__", "_past_gate", "_parsed_attempt", "CompileLoopStage._buildable",
                             "CompileLoopStage._needs_correction", "GenerateStage.__call__"],
    "lassi/core/oracle_stage.py": ["OracleStage.__call__", "OracleStage.align_runs"],
    "lassi/scoring/lassi_profile.py": ["correct_value", "outcome_components", "count_components"],
    "lassi/scoring/df_v0.py": ["warning_count", "guard_state", "DfV0Profile.score_attempt", "DfV0Profile._trial_score"],
}
ASSETS = ("assets/scoring/df-v0.yaml", "assets/scoring/lassi.yaml", "assets/scoring/lassi-paper.yaml")


def show(commit: str, path: str) -> str:
    """Return the file at `commit`."""
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, text=True, check=True).stdout


def segment(source: str, dotted: str) -> str | None:
    """Return the source of the function or method `dotted` (Class.method or name), or None when absent."""
    tree = ast.parse(source)
    parts = dotted.split(".")
    nodes = tree.body
    for index, name in enumerate(parts):
        found = [node for node in nodes if getattr(node, "name", None) == name]
        if not found:
            return None
        if index == len(parts) - 1:
            return ast.get_source_segment(source, found[0])
        nodes = found[0].body
    return None


def digest(text: str | None) -> str:
    """Return the first 12 hex digits of the sha256 of `text`, or ABSENT."""
    return "ABSENT" if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def main() -> None:
    """Print one line per function, then the asset digests."""
    old, new = sys.argv[1], sys.argv[2]
    for path, names in FUNCTIONS.items():
        before, after = show(old, path), show(new, path)
        for name in names:
            a, b = segment(before, name), segment(after, name)
            state = "SAME" if a == b and a is not None else "CHANGED"
            print(f"{state} {path}:{name} {old} {digest(a)} {new} {digest(b)}")
    for path in ASSETS:
        data = subprocess.run(["git", "show", f"{new}:{path}"], capture_output=True, check=True).stdout
        print(f"sha256 {new}:{path} {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
