# P0 phase gate

The gate is the P0 row of the bible's Build Roadmap: the mock model runs one
HeCBench app end to end, compile-only, and the text-policy check blocks a
seeded violating commit locally and in CI. All three parts passed on
2026-09-23 (UTC-07:00 on alpha01; the run's own clock reads 2026-09-24 UTC).

## 1. Mock compile-only run on alpha01

Source: `provenance.json` in this directory, the rx run record.

- rx id: `20260923-221344-desktop-8r113ei-p0-core-7cfa` (state done, rc 0),
  host alpha01 (Ubuntu 22.04.5 LTS, Python 3.10.12), 22:13:44 to 22:13:46 -07:00.
- Commit `bdcf5d2b3dfd6aad60239d949dc9b73f3c4b02cd` on branch p0-core, clean:
  `dirty` is false, and `git status --short | wc -l` printed 0 in the slot.
- Command: `uv sync --quiet && git status --short | wc -l && git rev-parse HEAD && uv run python tools/fetch_bench.py assets/bench/lassi-hecbench-10.yaml && uv run lassi run tests/fixtures/recipes/p0-smoke.yaml`.
  fetch_bench found HeCBench 7d2d3c567be522a2104065165de0a4a233a6ea1a
  already in place.
- Result:
  - The run wrote `$LASSI_RUNS_ROOT/runs/20260924-051344`, and `lassi run`
    exited 0.
  - The one trial, `p0-smoke/mock-reference/lassi-hecbench-10/omp-cuda/layout/run01`,
    reached stage S4 (compiles) with 0 corrections and 0 diagnostics.
  - Its build directory holds `main.cu`, the reply the mock backend returned,
    and `main`, the artifact the pinned nvcc-sm80 built inside the compile
    sandbox.
  - The trial record carries its provenance: commit bdcf5d2, dirty false,
    device "none (compile only)", sdk null.
  - run.md shows the same values and the recipe hash
    3233a7af11a749a8d06b4b847daf2f71a318b206b42af9007e8485ea9c50f0e2.
- Pins: cuda 12.6.3 (the redistributable install, toolchains/cuda.pin) and
  nvhpc 24.11. provenance.json holds both pin files. Device: none; nothing
  ran, since the executor is `none`.

## 2. Text-policy canary, local

`uv run tools/policy_canary.py local` ran on the workstation on 2026-09-24,
where the owner's pattern list is installed. It printed
`{"content_commit_blocked": true, "message_commit_blocked": true}`. Both
blocks came from real findings: `canary.txt:1` and `commit message:1` each
matched the built-in canary pattern, and there was no configuration error.
The same check also passes on the build host, which has no pattern list:
there the hooks fail closed (plans/OWNER-QUEUE.md, OQ-015).

## 3. Text-policy canary, CI

The owner pushed the canary, since its push and cleanup steps skip the git
hooks and force the canary branch, which agents may not do (P0.13).
`uv run tools/policy_canary.py status` shows run 35922699182 on p0-canary,
head 42ea4b6cd5c15ed2e4c265581805e55f47c4bcff, concluded `failure`. The run
is at https://github.com/JoeMad21/lassi/actions/runs/35922699182. Its log
shows the canary found in the commit message and in canary.txt, and the
pattern list masked as `***`, since CI reads it from an Actions secret
(OQ-013). The canary branch was deleted afterwards.
