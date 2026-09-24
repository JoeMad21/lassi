# P0.16 sandbox hardening: remote acceptance

Source: `provenance.json` in this directory, the rx run record. Every value
under Run and Result comes from it or from the run's output; the values under
After the run come from the separate rx exec named there.

## Run

- rx id: `20260923-173420-desktop-8r113ei-p0-core-192f` (kind run, state done,
  rc 0), host alpha01 (Ubuntu 22.04.5 LTS, kernel
  6.6.29+main+3.0.0r1-amd64-gio-epilmore-dev+, Python 3.10.12).
- Commit `59b57999cca70873750a985e6d4cf394a91ad663` (59b5799) on branch
  p0-core, clean tree: `dirty` is false, and `git status --short | wc -l`
  printed 0 in the slot.
- Date: 2026-09-23, 17:34:20-07:00 to 17:35:25-07:00.
- Command: `uv sync --quiet && git status --short | wc -l && LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -m remote -rfEs`.
- Pins: cuda 12.6.3 and nvhpc 24.11 (provenance.json holds both pin files).
- Device: none. The sandbox tests ran their own test programs in the sandbox
  on the host CPU; the capture test ran the pinned nvcc and nvc++ as
  compile-only jobs outside it. Nothing ran on an accelerator.

## Result

46 passed and 2292 deselected, with no failures, errors, or skips, in
65.36 s. The 46 are every remote test in the repository. That covers the 23
P0.10 sandbox tests, the P0.16 tests for R1 to R7 (private /dev, read-only
mounts that fail closed, the default-deny view, output caps, the workdir
disk cap, core dumps, and the runner kill), the tests that followed the
three review rounds, and the pinned-compiler capture test.

## After the run

rx 20260923-173541-exec-8a6d at 2026-09-23T17:35:41-07:00:
- `coredumpctl list --since "2026-09-23 17:30:00"` for this user found no
  coredumps.
- 0 leftover `lassi-sandbox-test*` directories under `$LASSI_SCRATCH/tmp` and
  `$LASSI_RUNS_ROOT`.
- `du -sh /mnt/nvme10/joseph_ufl` was 100G, under the owner's 120G cap.

## Earlier runs

The exploratory remote runs during development came from dirty-tree
snapshots and are not reportable. The spike's probes A to K (inline
`rx exec` scripts) are in plans/spikes/p0-sandbox-hardening.md.
