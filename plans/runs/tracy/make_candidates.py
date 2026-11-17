"""Write plans/runs/tracy/candidates.tsv from the read-only inventory's output (task P17.65).

Usage, on the workstation (it reads a file and writes one; it contacts nothing):

    uv run python plans/runs/tracy/make_candidates.py plans/runs/tracy/inventory-rx-20261116-161608-exec-0f2e.txt \
        plans/runs/tracy/candidates.tsv

The inventory (inventory.py, run through `rx exec` as rx 20261116-161608-exec-0f2e) lists every exact match on ttsim
whose target is a TT program, one row per distinct final file set. A row is kept when its target is a TT-Metalium
program (cpp -> tt, cpp -> ttkh), a served model wrote it (not mock-reference, the mock that answers with the tracked
reference), and its host-compute guard is False (the program launches kernels on the device). The header records how
many rows each rule dropped, counted here. The order: bale-v0-tt, then tt-pairs-v0-kh, then tt-pairs-v0, each group
round-robin over its items (ITEM_ORDER), so a short window still reaches every item.
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Sequence

GROUPS = ("bale-v0-tt", "tt-pairs-v0-kh", "tt-pairs-v0")
ITEM_ORDER = ("transpose_matrix", "randperm", "loopback", "eltwise_sfpu", "eltwise_binary", "matmul_single_core",
              "matmul_multi_core")
TT_DIRECTIONS = ("cpp-tt", "cpp-ttkh")
MOCK = "mock-reference"


def inventory_rows(path: str) -> list[dict[str, str]]:
    """Return the inventory's table rows (after its header line) as dicts."""
    rows, started = [], False
    for line in open(path, encoding="ascii").read().splitlines():
        if line.startswith("item\tbench\t"):
            started = True
            continue
        if not started or line.startswith(("#####", "[rx]")) or not line.strip():
            continue
        item, bench, direction, model, sweep, _attempts, host, _kernels, sha, _size, trial = line.split("\t")
        rows.append({"item": item, "suite": bench, "direction": direction, "model": model, "sweep": sweep,
                     "host": host, "sha": sha, "trial": trial})
    return rows


def why_dropped(row: dict[str, str]) -> str | None:
    """Return the rule that drops `row`, or None when it is kept."""
    if row["direction"] not in TT_DIRECTIONS:
        return "a target other than tt or ttkh"
    if row["model"] == MOCK:
        return "the mock that answers with the tracked reference"
    if row["host"] != "False":
        return "the host-compute guard True (computes on the host CPU)"
    return None


def ordered(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """Return the kept rows by group, then round-robin over ITEM_ORDER, each item's rows by sweep and digest."""
    result = []
    for group in GROUPS:
        queues = {item: sorted((row for row in rows if row["suite"] == group and row["item"] == item),
                               key=lambda row: (row["sweep"], row["sha"])) for item in ITEM_ORDER}
        while any(queues.values()):
            for item in ITEM_ORDER:
                if queues[item]:
                    result.append(queues[item].pop(0))
    if len(result) != len(rows):
        raise SystemExit(f"{len(rows) - len(result)} kept row(s) fall outside GROUPS and ITEM_ORDER")
    return result


def main(argv: Sequence[str]) -> int:
    """Read the inventory at argv[0] and write candidates.tsv at argv[1]; print the counts."""
    rows = inventory_rows(argv[0])
    dropped = Counter(reason for reason in map(why_dropped, rows) if reason is not None)
    kept = ordered([row for row in rows if why_dropped(row) is None])
    lines = [
        "# The saved TT translations task P17.65 profiles on the Wormhole cards, written by make_candidates.py from",
        "# the read-only inventory rx 20261116-161608-exec-0f2e (inventory-rx-20261116-161608-exec-0f2e.txt).",
        f"# Of its {len(rows)} distinct exact matches on ttsim with a TT target, {len(kept)} are kept; dropped:",
        *(f"#   {count} for {reason}" for reason, count in sorted(dropped.items())),
        "# files_sha1 is the first 10 hex digits of sha1(json.dumps(files, sort_keys=True)) of the final attempt's",
        "# files; trial is relative to $LASSI_RUNS_ROOT/runs. Order: bale-v0-tt, tt-pairs-v0-kh, tt-pairs-v0, each",
        "# round-robin over its items.",
        "order\tsuite\titem\tsource\ttarget\tmodel\tsweep\tfiles_sha1\ttrial",
    ]
    for number, row in enumerate(kept, 1):
        source, target = row["direction"].split("-")
        lines.append("\t".join([str(number), row["suite"], row["item"], source, target, row["model"], row["sweep"],
                                row["sha"], row["trial"]]))
    with open(argv[1], "w", encoding="ascii", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"{len(kept)} kept of {len(rows)}; dropped {dict(dropped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
