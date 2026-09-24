# P1 gate: Faithful LASSI

Gate text (bible Build Roadmap, P1 row, verbatim): "Replaying recorded responses reproduces upstream notebook decisions; mock dry run compiles 20/20".

- Date: 2026-09-24.
- Branch and commit: p1-faithful at fe01ab1 (clean tree for both runs).
- Host: alpha01.
- Toolchain pins: cuda 12.6.3 and nvhpc 24.11, recorded in each provenance.json.
- Upstream: SPEAR-UIC/LASSI at 74b4681, fetched by tools/fetch_upstream.py. HeCBench: 692cba3.

## Check 1: the notebook replay

- Command: `uv run tools/fetch_upstream.py && uv run tools/extract_lassi_assets.py && uv run pytest -q -rs -p no:cacheprovider tests/replay`
- Run: rx 20260924-104048-desktop-8r113ei-p1-faithful-5fd3. Provenance is in replay/provenance.json and the log in the run's output.
- Device: none. The test runs the pinned notebook's pipeline function and the faithful stages in one process, under a guard that fails the test if either side starts a process or opens a socket.
- Outcome: 47 passed and none skipped; pytest's own exit status was 0.
  - All 32 cases matched the notebook's decisions: 16 scenarios, each in both directions.
  - Each case compares every message sent, each extracted block, which attempts compiled and ran, the reference build and run, the final correction count, the end, the output that stands, Sim-T, and Sim-L.
- The recorded responses are scripted synthetic fixtures (tests/fixtures/replay/README.md; Decision Log, 2026-09-24). They are not model output.
- The report counts 4 fence-quirk hits. That count checks decision logic on synthetic fixtures. It is not the Evaluation Protocol's fence-quirk replay count, and it is not a measurement or a reproduction metric.

## Check 2: the mock dry run

- Command: `uv run tools/fetch_upstream.py && uv run tools/extract_lassi_assets.py && uv run tools/fetch_bench.py assets/bench/lassi-hecbench-10.yaml && uv run lassi run tests/fixtures/recipes/p1-dry-run.yaml --run-id p1-gate-dry-run`
- Run: rx 20260924-104105-desktop-8r113ei-p1-faithful-c002. Provenance is in dry-run/provenance.json.
  - The run tree is p1-gate-dry-run under the runs root.
  - The recipe is p1-dry-run, hash ad12fb6dd08e. It is projects/lassi-repro/recipe.yaml with the mock backend and executor none.
- Device: none (compile only). The compiles ran in the compile sandbox with the pinned nvcc and nvc++ and upstream's flags.
- Outcome: `lassi run` exited 0 (lassi_rc=0 in the log).
  - All 20 trials (10 apps, 2 directions) reached S4 on attempt 0 with no corrections and no end reason.
  - All 20 baseline target references compiled: every trial's baseline build directory holds the built program.
- This is a compile-stage run with the mock backend, which answers with the item's reference in one untagged fence. It is never compared with the paper.

## Verdict

PASS. Both parts of the gate hold from one clean commit: the replay reproduces the notebook's decisions, and the mock dry run compiles 20 of 20.

Open item, not part of the gate: P1.12 applies OQ-018 (upstream text in the repository) and waits for the owner. It does not change any generated text.
