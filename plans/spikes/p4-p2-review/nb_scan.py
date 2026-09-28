"""P2 review questions 5 and 6 (task P4.15): how the upstream notebooks store and compare similarities.

Usage, with the interpreter of the repository's environment:

    <python> plans/spikes/p4-p2-review/nb_scan.py <upstream checkout> <commit>:<notebook> ...

For each notebook, read with `git -C <checkout> show <commit>:<notebook>`,
the script prints the number of code cells, the kernel's Python version from
the notebook metadata, how many code lines format a value to two decimals
(`:.2f`), and how many code lines compare a value with a threshold (a
comparison operator next to a decimal number, or the word threshold). It
prints counts, cell ids, and line numbers only, never upstream text (OQ-018).
It starts git once per notebook and nothing else.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

TWO_DECIMALS = re.compile(r":\.2f")
THRESHOLD = re.compile(r"(?:[<>]=?|==)\s*0?\.\d|0?\.\d+\s*(?:[<>]=?|==)|threshold", re.IGNORECASE)


def scan(checkout: str, spec: str) -> None:
    """Print the counts for one `<commit>:<notebook>`."""
    text = subprocess.run(["git", "-C", checkout, "show", spec], capture_output=True, text=True, check=True).stdout
    notebook = json.loads(text)
    version = notebook.get("metadata", {}).get("language_info", {}).get("version")
    cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
    formats, thresholds = [], []
    for cell in cells:
        for number, line in enumerate("".join(cell.get("source", [])).splitlines(), 1):
            where = f"{str(cell.get('id', '?'))[:8]}:{number}"
            if TWO_DECIMALS.search(line):
                formats.append(where)
            if THRESHOLD.search(line):
                thresholds.append(where)
    print(f"{spec}: kernel python {version}; code cells {len(cells)}; two-decimal formats {len(formats)} "
          f"({', '.join(formats)}); threshold-like lines {len(thresholds)} ({', '.join(thresholds)})")


def main() -> None:
    """Scan every notebook named on the command line."""
    checkout, specs = sys.argv[1], sys.argv[2:]
    for spec in specs:
        scan(checkout, spec)


if __name__ == "__main__":
    main()
