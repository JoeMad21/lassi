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

All 16 files come from one capture:

- rx run `20260923-211958-desktop-8r113ei-p0-core-d221` on alpha01, dated
  2026-09-23T21:19:58-07:00.
- Commit 674bdd3 (`674bdd3cec5d2a3b293e71d520b3efaabbde0930`) with a clean
  tree: the manifest has `dirty` false and `snapshot_of` null, and the rx run
  record has `dirty` false.
- Pins, as `pin_files` in captures.json records them: toolchains/cuda.pin
  12.6.3 (nvcc V12.6.85, from the runfile installer with MD5
  `29d297908c72b810c9ceaa5177142abd`) for the nvcc-sm80 scenarios;
  toolchains/nvhpc.pin 24.11 (nvc++ 24.11-0) for the nvcpp-cc80 scenarios,
  with NVHPC_CUDA_HOME set to the pinned CUDA 12.6.3.
- Environment: the stage runner's clean compile environment, PATH,
  `LANG=C` and `LC_ALL=C` (so GCC quotes are plain ASCII), plus
  NVHPC_CUDA_HOME for nvc++. Since P0.20 each compile runs in the compile
  sandbox under `prlimit --core=1`, with no HOME and a private TMPDIR,
  `@lassi-tmp`, inside its own workdir.
- Device: none.

`tools/capture_toolchain_fixtures.py` built each preset the way the stage
runner does (pinned executable, clean environment, linked prefixes, the
compile sandbox, and the `--version` check against the pin's EXPECT_VERSION)
and called its `build(files, workdir)` on the scenario's source tree in a
fresh workdir. Each `.stderr` file is that build's `compile.stderr`.
`captures.json` records per scenario the argv, exit status, stderr sha256 and
size, and the parsed diagnostic count, and per toolchain the pin values, the
executable, the environment variable names, and the `--version` banner. The
rx run record is `results/p0-compiler-formats/provenance.json`, and
`results/p0-compiler-formats/summary.md` summarizes the capture.

This capture replaced the P0.15 capture, rx
`20260923-112105-desktop-8r113ei-p0-core-ba1a` from commit ebe6b07, whose
record is in `results/p0-toolchain-fixtures/`. Of the 14 scenarios both
captures hold, the 12 byte-stable ones came back byte for byte, and all 14
kept their argv, exit status, and diagnostic count; only the two linker
errors changed, in their per-run text (see Stored text). The environment
names lost HOME and TMPDIR, as P0.20 intended. P0.17 added the last two
rows of the table below.

## Fixtures

| File | Scenario | Exit status | rx id |
| --- | --- | --- | --- |
| `nvcc_undefined_identifier.stderr` | EDG error: undefined identifier in a kernel, source echo and caret, error summary | 2 | `20260923-211958-desktop-8r113ei-p0-core-d221` |
| `nvcc_warning_177.stderr` | EDG warning #177-D in an included header, echo and caret, Remark line | 0 | `20260923-211958-desktop-8r113ei-p0-core-d221` |
| `nvcc_host_gcc_warning.stderr` | Host GCC -Wsign-compare warning, "In function" line, GCC echo and caret | 0 | `20260923-211958-desktop-8r113ei-p0-core-d221` |
| `nvcc_fatal.stderr` | nvcc driver fatal for the ARCH override sm_35 | 1 | `20260923-211958-desktop-8r113ei-p0-core-d221` |
| `nvcc_ptxas_error.stderr` | ptxas error: too much static shared data in a kernel | 255 | `20260923-211958-desktop-8r113ei-p0-core-d221` |
| `nvcc_linker_error.stderr` | Linker undefined reference with its "in function" line, then the collect2 line | 1 | `20260923-211958-desktop-8r113ei-p0-core-d221` |
| `nvcc_clean.stderr` | Clean build (empty stderr) | 0 | `20260923-211958-desktop-8r113ei-p0-core-d221` |
| `nvcpp_edg_error.stderr` | EDG error: undefined identifier in a target loop, echo and caret, error summary | 2 | `20260923-211958-desktop-8r113ei-p0-core-d221` |
| `nvcpp_edg_warning.stderr` | EDG warning with a `[declared_but_not_referenced]` tag, echo and caret, Remark line, then the `-Minfo` report | 0 | `20260923-211958-desktop-8r113ei-p0-core-d221` |
| `nvcpp_backend_error.stderr` | NVC++-S-1101 backend error (GPU stack limit) with file and line, `-Minfo` report, severe-errors summary | 2 | `20260923-211958-desktop-8r113ei-p0-core-d221` |
| `nvcpp_fatal_abort.stderr` | NVC++-F-0000 internal compiler error with file and line, compilation aborted summary | 2 | `20260923-211958-desktop-8r113ei-p0-core-d221` |
| `nvcpp_minfo_clean.stderr` | Clean build: the `-Minfo` report only | 0 | `20260923-211958-desktop-8r113ei-p0-core-d221` |
| `nvcpp_linker_error.stderr` | `-Minfo` report, then a linker undefined reference with its "in function" line and the pgacclnk status line | 2 | `20260923-211958-desktop-8r113ei-p0-core-d221` |
| `nvcpp_missing_include.stderr` | EDG catastrophic error for a missing header, echo and caret, summary and "Compilation terminated." | 2 | `20260923-211958-desktop-8r113ei-p0-core-d221` |
| `nvcc_ptxas_inline_asm.stderr` | ptxas error naming line 28 of the temporary .ptx nvcc wrote (inline PTX asm with the unknown modifier .bogus), then the ptxas fatal line | 255 | `20260923-211958-desktop-8r113ei-p0-core-d221` |
| `nvcpp_nvlink_error.stderr` | File name headers for two sources, `-Minfo` report, nvlink undefined reference naming a temporary object, then the pgacclnk status line naming nvdd | 2 | `20260923-211958-desktop-8r113ei-p0-core-d221` |

## Stored text

The files hold stderr as build() keeps it in its `compile.stderr`
attachment: the compiler's bytes decoded as UTF-8 with invalid bytes
replaced, then written as UTF-8. All 16 files are plain ASCII with LF line
endings, so the bytes here are the compiler's bytes; `nvcc_clean.stderr` is
empty.

Four files carry text that belongs to their run and changes on a recapture.
Each names the capture's absolute workdir, which holds the rx id
(`/mnt/nvme10/joseph_ufl/lassi-runs/fixture-captures/<rx id>/work/<scenario>/`):

- `nvcc_linker_error.stderr`: the ld line names nvcc's temporary object
  `@lassi-tmp/tmpxft_00000002_00000000-11_main.o` under the workdir, and the
  next line names `tmpxft_00000002_00000000-6_main.cudafe1.cpp`. In the P0.15
  capture, before the compile sandbox, these were `tmpxft_0013778e_...` files
  under the shared TMPDIR `/mnt/nvme10/joseph_ufl/tmp`.
- `nvcpp_linker_error.stderr`: the workdir twice, in the ld line with the
  temporary object `@lassi-tmp/nvc++vc-2bVcpD9.o`, and in the undefined
  reference line that names `main.cpp` and its line 21.
- `nvcc_ptxas_inline_asm.stderr`: the workdir in the path of the temporary
  PTX, `@lassi-tmp/tmpxft_00000002_00000000-6_main.ptx`. The P0.17 stage A
  exploratory dirty-tree run, rx
  `20260923-202909-desktop-8r113ei-p0-core-09c6`, printed the same PTX name
  under its own workdir.
- `nvcpp_nvlink_error.stderr`: the workdir in the path of the temporary
  object `@lassi-tmp/nvc++bcdFQVDqP8.o`, whose name is random on each run
  (the same exploratory run printed `nvc++8c0N0-iy_9.o`).

These four are RUN_DEPENDENT in tests/tools/test_capture_toolchain_fixtures.py.
The other 12 are byte-stable: a recapture must give them byte for byte (P0.20
A6), and this capture did.

## The nvc++ driver format

No fixture shows an nvc++ driver line, `nvc++-Error-...` or
`nvc++-Fatal-...`: none could be produced from sources with the preset flags
without crashing a compiler, and no compiler was crashed on purpose. P0.17
stage A tried these sources under nvcpp-cc80 with the preset flags:

- two sources with the same file name in two directories (`a/util.cpp` and
  `b/util.cpp`) with `main.cpp`;
- two sources with the same stem and different suffixes (`util.cc` and
  `util.cpp`) with `main.cpp`;
- a C source (`first.c`) with `main.cpp` and `second.cpp`;
- host inline asm with an unknown mnemonic;
- inline PTX asm with an unknown modifier inside an OpenMP target region;
- `#pragma omp requires unified_shared_memory` before a target loop;
- `#pragma omp requires reverse_offload` before a target loop;
- a 256 KiB team-local array in team memory (`omp_pteam_mem_alloc`) inside
  `omp target teams`.

The attempts ran as exploratory dirty-tree captures, never [MEASURED]: rx
`20260923-202434-desktop-8r113ei-p0-core-0586`,
`20260923-202616-desktop-8r113ei-p0-core-aea5`, and
`20260923-202719-desktop-8r113ei-p0-core-c211`. The driver itself was
inspected on the host with read-only runs rx `20260923-202732-exec-6aa7`,
`20260923-202740-exec-ba76`, `20260923-202746-exec-b407`,
`20260923-202753-exec-32b7`, `20260923-202802-exec-e6ca`, and
`20260923-202823-exec-119c`.

The parser reads the format anyway (lassi/toolchains/nvcpp.py). Its line
tests in test_diagnostics.py use the two lines on record, copied verbatim:

- `nvc++-Error-A CUDA toolkit matching the current driver version (0) or a
  supported older version (11.8) was not installed with this HPC SDK.`, from
  plans/spikes/p0-toolchains-verify.md run 2a, clean-commit rx
  `20260923-045538-desktop-8r113ei-p0-core-bde2`, before NVHPC_CUDA_HOME was
  set;
- `nvc++-Fatal-<toolchain path>/bin/tools/nvcpfe TERMINATED by signal 11`,
  from the P0.15 exploratory dirty-tree probe rx
  `20260923-104313-desktop-8r113ei-p0-core-2b22`, whose source of that time
  crashed the front end by accident; that probe is exploratory and never
  [MEASURED].

## Parser order

P0.17 changed the order in which the adapters try their line patterns:

- nvcc: the ptxas line (`ptxas <severity> : ...`), the ptxas line with a PTX
  place (`ptxas <PTX file>, line <n>; <severity> : ...`), and the nvcc driver
  fatal line (`nvcc fatal : ...`) are now tried before the EDG and GCC
  patterns; before, they came after them. The collect2 line and the linker's
  undefined reference stay last. A ptxas or driver message, or the PTX name,
  which nvcc derives from the source stem the model chooses, may hold text
  shaped like an EDG or GCC place (`x(3): error: ` or `a:1:2: error: `);
  tried first, the tool's own pattern keeps that text in the message instead
  of reading it as a file and line that index no built file.
- nvc++: the new driver (`nvc++-<Severity>-...`) and nvlink
  (`nvlink <severity> : ...`) patterns come after the EDG and backend ones and
  before the GCC style one, for the same reason.

The 14 fixtures that predate P0.17 parse into the same hand-derived
Diagnostic lists under the new order.

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
4. Check that the byte-stable files are byte-identical to the ones here and
   that every scenario kept its exit status and diagnostic count; record any
   difference. Then copy each `.stderr` byte for byte over the file here and
   `manifest.json` to `captures.json`, and fetch the run record with
   `uv run tools/rx.py pull <rx id> --into results/<name>`, a results
   directory for that capture (this one's is `results/p0-compiler-formats`).
5. Derive the expected Diagnostic lists in test_diagnostics.py again by hand
   from the new stderr, and update the table and provenance above. The
   sample lines there that name a workdir (PTX_FILE, NVLINK_OBJECT, and the
   linker lines) follow the new capture, and a new scenario whose stderr
   names per-run text joins RUN_DEPENDENT.
