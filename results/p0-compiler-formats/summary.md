# P0.17 compiler error format capture

Sources: `provenance.json` in this directory (the rx run record) and
`tests/toolchains/fixtures/captures.json` (the capture tool's manifest, copied
byte for byte from the run's `manifest.json`). Every value below is copied
from one of the two, or from comparing the captured files with the P0.15
fixtures they replaced.

## Run

- rx id: `20260923-211958-desktop-8r113ei-p0-core-d221` (kind run, state done,
  rc 0), host alpha01 (Ubuntu 22.04.5 LTS, kernel
  6.6.29+main+3.0.0r1-amd64-gio-epilmore-dev+).
- Commit `674bdd3cec5d2a3b293e71d520b3efaabbde0930` (674bdd3) on branch
  p0-core, clean tree: `dirty` false and `snapshot_of` null in both files.
- Date: 2026-09-23T21:19:58-07:00 (manifest); the run went from
  21:19:58-07:00 to 21:20:11-07:00.
- Command: `uv sync --quiet && git status --short | wc -l && uv run python tools/capture_toolchain_fixtures.py`.
- Pins: cuda 12.6.3 (the runfile installer, MD5
  29d297908c72b810c9ceaa5177142abd; nvcc banner "Cuda compilation tools,
  release 12.6, V12.6.85") for nvcc-sm80; nvhpc 24.11 (nvc++ banner "nvc++
  24.11-0 64-bit target on x86-64 Linux -tp znver4") with cuda 12.6.3 as
  NVHPC_CUDA_HOME for nvcpp-cc80. provenance.json holds both pin files
  verbatim, and both toolchains record `version_exit_status` 0.
- Environment: PATH, LANG=C, LC_ALL=C (plus NVHPC_CUDA_HOME for nvc++); the
  manifest records the names, and the values of LANG and LC_ALL only. Each
  compile ran in the P0.20 compile sandbox, with no HOME and a private TMPDIR
  (`@lassi-tmp`) under its workdir.
- Device: none. Every scenario only compiled; nothing ran on a device and no
  built program ran. The exit statuses are compiler exit statuses, not
  performance.

## Scenarios

Exit status and diagnostic count per scenario, from captures.json. The count
is what the parser at 674bdd3 gave on alpha01 at capture time. For all 16
scenarios it equals the length of the expected Diagnostic list derived by
hand in tests/toolchains/test_diagnostics.py.

| Scenario | Toolchain | Exit status | Diagnostics | stderr bytes | Per-run text |
| --- | --- | --- | --- | --- | --- |
| nvcc_clean | nvcc-sm80 | 0 | 0 | 0 | no |
| nvcc_fatal (ARCH override sm_35) | nvcc-sm80 | 1 | 1 | 74 | no |
| nvcc_host_gcc_warning | nvcc-sm80 | 0 | 1 | 295 | no |
| nvcc_linker_error | nvcc-sm80 | 1 | 2 | 353 | yes |
| nvcc_ptxas_error | nvcc-sm80 | 255 | 1 | 104 | no |
| nvcc_ptxas_inline_asm (new in P0.17) | nvcc-sm80 | 255 | 2 | 273 | yes |
| nvcc_undefined_identifier | nvcc-sm80 | 2 | 1 | 183 | no |
| nvcc_warning_177 | nvcc-sm80 | 0 | 1 | 211 | no |
| nvcpp_backend_error | nvcpp-cc80 | 2 | 1 | 710 | no |
| nvcpp_edg_error | nvcpp-cc80 | 2 | 1 | 192 | no |
| nvcpp_edg_warning | nvcpp-cc80 | 0 | 1 | 717 | no |
| nvcpp_fatal_abort | nvcpp-cc80 | 2 | 1 | 177 | no |
| nvcpp_linker_error | nvcpp-cc80 | 2 | 1 | 897 | yes |
| nvcpp_minfo_clean | nvcpp-cc80 | 0 | 0 | 485 | no |
| nvcpp_missing_include | nvcpp-cc80 | 2 | 1 | 230 | no |
| nvcpp_nvlink_error (new in P0.17) | nvcpp-cc80 | 2 | 1 | 773 | yes |

The 16 `.stderr` files in tests/toolchains/fixtures/ match the sha256 and
size recorded for each scenario; all are plain ASCII with LF line endings.

## Against the P0.15 capture

The 14 scenarios that were already fixtures came from rx
`20260923-112105-desktop-8r113ei-p0-core-ba1a` (commit ebe6b07; its record is
`results/p0-toolchain-fixtures/`). In this capture:

- the 12 byte-stable scenarios (all but the two linker errors) are
  byte-identical to their P0.15 fixtures;
- all 14 kept their argv, exit status, and diagnostic count;
- the two linker errors changed only in per-run text, so their size changed
  (nvcc_linker_error 250 to 353 bytes, nvcpp_linker_error 796 to 897 bytes);
- the toolchain environment names lost HOME and TMPDIR, as P0.20 intended.

## Per-run text

Four captures hold text that belongs to this run and changes on a recapture.
Each names the capture's absolute workdir, which holds the rx id
(`/mnt/nvme10/joseph_ufl/lassi-runs/fixture-captures/20260923-211958-desktop-8r113ei-p0-core-d221/work/<scenario>/`):

- nvcc_linker_error: the ld line names nvcc's temporary object
  `@lassi-tmp/tmpxft_00000002_00000000-11_main.o` under the workdir, and the
  next line names `tmpxft_00000002_00000000-6_main.cudafe1.cpp`.
- nvcpp_linker_error: the workdir in the ld line, with the temporary object
  `@lassi-tmp/nvc++vc-2bVcpD9.o`, and in the undefined reference line that
  names `main.cpp:21`.
- nvcc_ptxas_inline_asm: the workdir in the path of the temporary PTX,
  `@lassi-tmp/tmpxft_00000002_00000000-6_main.ptx`, which ptxas names with
  its line 28.
- nvcpp_nvlink_error: the workdir in the path of the temporary object
  `@lassi-tmp/nvc++bcdFQVDqP8.o`, whose name is random on each run.

The other 12 captures hold no per-run text.

## Not recorded

No capture shows an nvc++ driver line (`nvc++-Error-` or `nvc++-Fatal-`);
tests/toolchains/fixtures/README.md lists the sources P0.17 tried and their
exploratory rx ids. The manifest does not record which host GCC nvcc used,
so this capture does not establish the GCC version.
