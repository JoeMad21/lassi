# P0.20 compile hardening: acceptance from the clean commit

Sources: `suite/provenance.json` and `capture/provenance.json` in this
directory, the rx run records. Every value below comes from them, from the
runs' output, or from the capture's manifest on alpha01, compared against
`tests/toolchains/fixtures/captures.json`.

Both runs used commit `1de7db6cebe32b35578a1d525f0d0abd35591f38` (1de7db6) on
branch p0-core with a clean tree: `dirty` is false, and
`git status --short | wc -l` printed 0 in the slot. Host alpha01, Ubuntu
22.04.5 LTS, Python 3.10.12. Pins: cuda 12.6.3 and nvhpc 24.11 (both pin
files are in each provenance.json). Device: none. The compiles ran on the
host CPU, and no built program ran on an accelerator.

## Test suites

rx `20260923-195751-desktop-8r113ei-p0-core-7f98`, 2026-09-23 19:57:51 to
20:02:28 -07:00, rc 0. Command:
`uv sync --quiet && git status --short | wc -l && uv run pytest -q 2>&1 | tail -3 && LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote -rfEs 2>&1 | tail -3`.

- Remote suite: 63 passed and 2392 deselected, in 113.02 s. That covers
  every remote test: the P0.16 sandbox, the P0.20 compile sandbox with the
  real pinned nvcc and nvc++, and the fixture recapture test.
- Fast suite on Linux: 2441 passed, 13 skipped, and 1 failed:
  `tests/tools/test_text_policy_modes.py::test_canary_local_blocks_both_seeded_commits`.
  It fails because the repository's git hooks are stored without the
  executable bit, so git on Linux does not run them and the seeded canary
  commit went through. P0.20 did not cause this. It is plans/OWNER-QUEUE.md
  OQ-015, and the test is kept as it is, since it caught a real gap.

## Fixture recapture (acceptance A6)

rx `20260923-200343-desktop-8r113ei-p0-core-bcf2`, 2026-09-23 20:03:43 to
20:03:54 -07:00, rc 0. Command:
`uv sync --quiet && git status --short | wc -l && uv run python tools/capture_toolchain_fixtures.py`.

- All 12 byte-stable scenarios are byte-identical to
  `tests/toolchains/fixtures/<scenario>.stderr` and match their `stderr_sha256`
  in captures.json. Every compile now runs inside the sandbox.
- All 14 scenarios kept their recorded exit status and diagnostic count.
- The two per-run captures differ, as expected: nvcc_linker_error (353
  bytes, was 250) and nvcpp_linker_error (897 bytes, was 796). Their
  temporary paths now lie in the private TMPDIR under each build directory.
- Recorded compile environments: nvcc-sm80 gets LANG, LC_ALL and PATH;
  nvcpp-cc80 gets LANG, LC_ALL, NVHPC_CUDA_HOME and PATH. Neither gets HOME.

The exploratory runs during development came from dirty-tree snapshots and
are not reportable. They are in plans/spikes/p0-compile-hardening.md.
