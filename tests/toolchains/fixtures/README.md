# Toolchain stderr fixtures

Each `.stderr` file here is the raw stderr of one compile, captured on alpha01
with the pinned compilers and copied byte for byte. `captures.json` is the
capture's provenance manifest, copied byte for byte from its `manifest.json`.
`sources/<scenario>/` holds the files each scenario compiled, which the EDG
echo and caret lines point into, and `scenarios.json` names each scenario's
toolchain preset, any ARCH or GPU override, and what its source does. The
exact Diagnostic list each file must parse into is in
tests/toolchains/test_diagnostics.py, derived by hand from the raw stderr.

No value in these files is a measurement: they record compiler output, not
performance. Nothing ran on a device, and no built program ran.

## Provenance

All 14 files come from one capture:

- rx run `20260923-112105-desktop-8r113ei-p0-core-ba1a` on alpha01, dated
  2026-09-23T11:21:06-07:00.
- Commit ebe6b07 (`ebe6b075c84d44a83d628b5227f1ecb3a5c2a1bf`) with a clean
  tree: the manifest has `dirty` false and `snapshot_of` null.
- Pins: toolchains/cuda.pin 12.6.3 (nvcc V12.6.85) for the nvcc-sm80
  scenarios; toolchains/nvhpc.pin 24.11 (nvc++ 24.11-0) for the nvcpp-cc80
  scenarios, with NVHPC_CUDA_HOME set to the pinned CUDA 12.6.3.
- Environment: the stage runner's clean compile environment, PATH, HOME,
  TMPDIR, `LANG=C` and `LC_ALL=C`, so GCC quotes are plain ASCII.
- Device: none.

`tools/capture_toolchain_fixtures.py` built each preset the way the stage
runner does (pinned executable, clean environment, linked prefixes) and called
its `build(files, workdir)` on the scenario's source tree in a fresh workdir.
Each `.stderr` file is that build's `compile.stderr`. `captures.json` records
per scenario the argv, exit status, stderr sha256 and size, and the parsed
diagnostic count, and per toolchain the pin values, the executable, the
environment variable names, and the `--version` banner. The rx run record is
`results/p0-toolchain-fixtures/provenance.json`, and
`results/p0-toolchain-fixtures/summary.md` summarizes the capture.

## Fixtures

| File | Scenario | Exit status | rx id |
| --- | --- | --- | --- |
| `nvcc_undefined_identifier.stderr` | EDG error: undefined identifier in a kernel, source echo and caret, error summary | 2 | `20260923-112105-desktop-8r113ei-p0-core-ba1a` |
| `nvcc_warning_177.stderr` | EDG warning #177-D in an included header, echo and caret, Remark line | 0 | `20260923-112105-desktop-8r113ei-p0-core-ba1a` |
| `nvcc_host_gcc_warning.stderr` | Host GCC -Wsign-compare warning, "In function" line, GCC echo and caret | 0 | `20260923-112105-desktop-8r113ei-p0-core-ba1a` |
| `nvcc_fatal.stderr` | nvcc driver fatal for the ARCH override sm_35 | 1 | `20260923-112105-desktop-8r113ei-p0-core-ba1a` |
| `nvcc_ptxas_error.stderr` | ptxas error: too much static shared data in a kernel | 255 | `20260923-112105-desktop-8r113ei-p0-core-ba1a` |
| `nvcc_linker_error.stderr` | Linker undefined reference with its "in function" line, then the collect2 line | 1 | `20260923-112105-desktop-8r113ei-p0-core-ba1a` |
| `nvcc_clean.stderr` | Clean build (empty stderr) | 0 | `20260923-112105-desktop-8r113ei-p0-core-ba1a` |
| `nvcpp_edg_error.stderr` | EDG error: undefined identifier in a target loop, echo and caret, error summary | 2 | `20260923-112105-desktop-8r113ei-p0-core-ba1a` |
| `nvcpp_edg_warning.stderr` | EDG warning with a `[declared_but_not_referenced]` tag, echo and caret, Remark line, then the `-Minfo` report | 0 | `20260923-112105-desktop-8r113ei-p0-core-ba1a` |
| `nvcpp_backend_error.stderr` | NVC++-S-1101 backend error (GPU stack limit) with file and line, `-Minfo` report, severe-errors summary | 2 | `20260923-112105-desktop-8r113ei-p0-core-ba1a` |
| `nvcpp_fatal_abort.stderr` | NVC++-F-0000 internal compiler error with file and line, compilation aborted summary | 2 | `20260923-112105-desktop-8r113ei-p0-core-ba1a` |
| `nvcpp_minfo_clean.stderr` | Clean build: the `-Minfo` report only | 0 | `20260923-112105-desktop-8r113ei-p0-core-ba1a` |
| `nvcpp_linker_error.stderr` | `-Minfo` report, then a linker undefined reference with its "in function" line and the pgacclnk status line | 2 | `20260923-112105-desktop-8r113ei-p0-core-ba1a` |
| `nvcpp_missing_include.stderr` | EDG catastrophic error for a missing header, echo and caret, summary and "Compilation terminated." | 2 | `20260923-112105-desktop-8r113ei-p0-core-ba1a` |

## Stored text

The files hold stderr as build() keeps it in its `compile.stderr`
attachment: the compiler's bytes decoded as UTF-8 with invalid bytes
replaced, then written as UTF-8. All 14 files are plain ASCII with LF line
endings, so the bytes here are the compiler's bytes; `nvcc_clean.stderr` is
empty.

Two files carry text that belongs to their run and changes on a recapture:

- `nvcc_linker_error.stderr`: the `tmpxft_...` temporary file names and the
  absolute TMPDIR, `/mnt/nvme10/joseph_ufl/tmp`, in the ld line.
- `nvcpp_linker_error.stderr`: the capture's absolute workdir, which holds the
  rx id (`/mnt/nvme10/joseph_ufl/lassi-runs/fixture-captures/<rx id>/work/nvcpp_linker_error/main.cpp`),
  and a temporary object name, `nvc++xwe4ef5MXtbZI.o`, under the same TMPDIR.

## Recapturing

1. Commit every change to `sources/` and `scenarios.json` first: the capture
   must run from a clean commit, since a dirty-tree rx run is exploratory only.
2. Capture every scenario in one run, since `captures.json` must cover every
   fixture:
   `uv run tools/rx.py run -- 'uv sync --quiet && git status --short | wc -l && uv run python tools/capture_toolchain_fixtures.py'`
   (the count must be 0: the slot holds the clean commit).
   The tool writes to `$LASSI_RUNS_ROOT/fixture-captures/<rx id>/` on alpha01
   and prints one line per scenario.
3. Fetch each `<scenario>.stderr` and `manifest.json` from
   `lassi-runs/fixture-captures/<rx id>/` into a local scratch directory with
   `uv run tools/rx.py pull --path <file> --into <dir>`, and check that the
   manifest has `dirty` false.
4. Copy each `.stderr` byte for byte over the file here and `manifest.json` to
   `captures.json`, then fetch the run record with
   `uv run tools/rx.py pull <rx id> --into results/p0-toolchain-fixtures`.
5. Derive the expected Diagnostic lists in test_diagnostics.py again by hand
   from the new stderr, and update the table and provenance above.
