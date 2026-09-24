# P1.1 upstream LASSI pin

[MEASURED] provenance.json:
- run: rx 20260924-002439-desktop-8r113ei-p1-faithful-8e73;
- commit: a815c45 (clean tree);
- host: alpha01, 2026-09-24;
- toolchain pins: recorded in provenance.json;
- device: none.

Commands:
- `uv run python tools/fetch_upstream.py` printed that the checkout already holds https://github.com/SPEAR-UIC/LASSI at 74b46812523f2ff79b53b6880a4521690d7478b0 (fetch_rc=0). The slot kept an earlier checkout at the pin; the tool left it as it was.
- `uv run pytest -q -m remote -p no:cacheprovider tests/bench/test_upstream_pin.py`: 3 passed, pytest_rc=0.
  - One test fetches the pin from GitHub into an empty directory under the scratch root.
  - One runs the default fetch twice in the slot.
  - One checks the spike's HeCBench table (blob ids, the commits holding all 20 files, the support-file paths) against a blobless HeCBench clone.

Limits: this checks the pin, not performance. The blob ids are git content hashes.
