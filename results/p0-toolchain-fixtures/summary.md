# P0.15 toolchain stderr fixture capture

Sources: `provenance.json` in this directory (the rx run record) and
`tests/toolchains/fixtures/captures.json` (the capture tool's manifest, copied
byte for byte from the run's `manifest.json`). Every value below is copied
from one of the two.

## Run

- rx id: `20260923-112105-desktop-8r113ei-p0-core-ba1a` (kind run, state done,
  rc 0), host alpha01 (Ubuntu 22.04.5 LTS, kernel
  6.6.29+main+3.0.0r1-amd64-gio-epilmore-dev+).
- Commit `ebe6b075c84d44a83d628b5227f1ecb3a5c2a1bf` (ebe6b07) on branch
  p0-core, clean tree: `dirty` false and `snapshot_of` null in both files.
- Date: 2026-09-23T11:21:06-07:00 (manifest); the run went from
  11:21:05-07:00 to 11:21:10-07:00.
- Command: `uv sync --quiet && git status --short | wc -l && uv run python tools/capture_toolchain_fixtures.py`.
- Pins: cuda 12.6.3 (nvcc banner "Cuda compilation tools, release 12.6,
  V12.6.85") for nvcc-sm80; nvhpc 24.11 (nvc++ banner "nvc++ 24.11-0 64-bit
  target on x86-64 Linux -tp znver4") with cuda 12.6.3 as NVHPC_CUDA_HOME for
  nvcpp-cc80. provenance.json holds both pin files verbatim.
- Environment: PATH, HOME, TMPDIR, LANG=C, LC_ALL=C (plus NVHPC_CUDA_HOME for
  nvc++); the manifest records the names, and the values of LANG and LC_ALL
  only.
- Device: none. Every scenario only compiled; nothing ran on a device and no
  built program ran. The exit statuses are compiler exit statuses, not
  performance.

## Scenarios

Exit status and diagnostic count per scenario, from captures.json. The count
is what the parser at ebe6b07 gave on alpha01 at capture time. For all 14
scenarios it equals the length of the expected Diagnostic list derived by
hand in tests/toolchains/test_diagnostics.py, which the parser after P0.15
gives on these files.

| Scenario | Toolchain | Exit status | Diagnostics | stderr bytes |
| --- | --- | --- | --- | --- |
| nvcc_clean | nvcc-sm80 | 0 | 0 | 0 |
| nvcc_fatal (ARCH override sm_35) | nvcc-sm80 | 1 | 1 | 74 |
| nvcc_host_gcc_warning | nvcc-sm80 | 0 | 1 | 295 |
| nvcc_linker_error | nvcc-sm80 | 1 | 2 | 250 |
| nvcc_ptxas_error | nvcc-sm80 | 255 | 1 | 104 |
| nvcc_undefined_identifier | nvcc-sm80 | 2 | 1 | 183 |
| nvcc_warning_177 | nvcc-sm80 | 0 | 1 | 211 |
| nvcpp_backend_error | nvcpp-cc80 | 2 | 1 | 710 |
| nvcpp_edg_error | nvcpp-cc80 | 2 | 1 | 192 |
| nvcpp_edg_warning | nvcpp-cc80 | 0 | 1 | 717 |
| nvcpp_fatal_abort | nvcpp-cc80 | 2 | 1 | 177 |
| nvcpp_linker_error | nvcpp-cc80 | 2 | 1 | 796 |
| nvcpp_minfo_clean | nvcpp-cc80 | 0 | 0 | 485 |
| nvcpp_missing_include | nvcpp-cc80 | 2 | 1 | 230 |

The 14 `.stderr` files in tests/toolchains/fixtures/ match the sha256 and
size recorded for each scenario; all are plain ASCII.

## Per-run text

Two captures hold text that belongs to this run and changes on a recapture:

- nvcc_linker_error: the `tmpxft_0013778e_00000000-*` temporary names and
  the absolute TMPDIR `/mnt/nvme10/joseph_ufl/tmp` in the ld line.
- nvcpp_linker_error: the capture's absolute workdir, which holds the rx id
  (`/mnt/nvme10/joseph_ufl/lassi-runs/fixture-captures/20260923-112105-desktop-8r113ei-p0-core-ba1a/work/nvcpp_linker_error/main.cpp`),
  and the temporary object `nvc++xwe4ef5MXtbZI.o` under the same TMPDIR.

## Not recorded

The manifest does not record which host GCC nvcc used, so this capture does
not establish the GCC version.
