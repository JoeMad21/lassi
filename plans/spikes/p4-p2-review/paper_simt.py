"""P2 review question 6 (task P4.15): the paper's Sim-T and Sim-L columns over its correct trials.

Usage: <python> plans/spikes/p4-p2-review/paper_simt.py <pypdf text of arXiv:2407.01638v2>

It parses Tables VI and VII the way the P2.4 spike's recount.py does
(plans/spikes/p2-lassi-metrics.md, Recount) and prints, per direction and
pooled, the number of correct trials (rows with values), the Sim-T range,
the count at Sim-T >= 0.6, the count where Sim-T exceeds Sim-L, and
Pearson's r between Sim-T and Sim-L. These are counts read from published
tables, not measurements. It prints numbers only.
"""

from __future__ import annotations

import re
import statistics
import sys

APPS = ["matrix-rotate", "jacobi", "layout", "atomicCost", "dense-embedding",
        "pathfinder", "bsearch", "entropy", "colorwheel", "randomAccess"]


def panel(text: str, start: str, end: str) -> list[tuple[float, float]]:
    """Return (Sim-T, Sim-L) of every row half with values between the markers, as recount.py reads them."""
    segment = text[text.index(start):text.index(end)]
    found = []
    for app in APPS:
        tokens = re.search(r"^" + re.escape(app) + r" (.*)$", segment, re.M).group(1).split()
        assert len(tokens) == 10, (app, len(tokens))
        for half in (tokens[:5], tokens[5:]):
            if half[0] != "N/A":
                found.append((float(half[2]), float(half[3])))
    return found


def line(label: str, rows: list[tuple[float, float]]) -> str:
    """Return the printed summary of one set of rows."""
    sim_t = [row[0] for row in rows]
    sim_l = [row[1] for row in rows]
    r = statistics.correlation(sim_t, sim_l)
    above = sum(t > sl for t, sl in rows)
    return (f"{label}: correct trials {len(rows)}; Sim-T {min(sim_t):.2f}..{max(sim_t):.2f}; "
            f"Sim-T >= 0.6 {sum(value >= 0.6 for value in sim_t)}; Sim-T > Sim-L {above}; r(Sim-T, Sim-L) {r:.3f}")


def main() -> None:
    """Print the three summary lines."""
    text = open(sys.argv[1], encoding="utf-8").read()
    omp_cuda = panel(text, "(a) Panel A", "(b) Panel B") + panel(text, "(b) Panel B", "represents the number")
    cuda_omp = panel(text, "(a) *Panel A", "(b) *Panel B") + panel(text, "(b) *Panel B", "D. Discussion")
    print(line("OMP -> CUDA", omp_cuda))
    print(line("CUDA -> OMP", cuda_omp))
    print(line("pooled", omp_cuda + cuda_omp))


if __name__ == "__main__":
    main()
