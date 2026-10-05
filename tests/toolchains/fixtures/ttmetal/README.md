# ttmetal-host stderr fixtures

Each `.stderr` file here is the raw stderr of one compile, captured on alpha01
with the pinned host clang++-20 against the pinned tt-metal tree and copied
byte for byte. `captures.json` is the capture's provenance manifest, copied
byte for byte from its `manifest.json`. `sources/<case>/` holds the files each
case compiled, and `scenarios.json` names each case's toolchain and what its
source does. Every source is a small host program written for these fixtures;
none is upstream text. The exact Diagnostic list each file must parse into is
in `<case>.json`, derived by hand from the raw stderr (its `derivation` field
says how), never copied from the parser's output;
tests/toolchains/test_ttmetal_diagnostics.py checks the parse against it.

No value in these files is a measurement: they record compiler output, not
performance. Nothing ran on a device, no built program ran, and neither the
kernel JIT nor ttsim started.

## Provenance

All eight files come from one capture:

- rx run `20260926-142744-desktop-8r113ei-detached-cf5f1274-326e` on
  alpha01, dated 2026-09-26T14:27:45-07:00.
- Commit cf5f127 (`cf5f1274ef483e4bbd5d9f04f605b7a848407fe8`) with a clean
  tree: the manifest has `dirty` false and `snapshot_of` null, and the rx run
  record has `dirty` false.
- Pin, as `pin_files` in captures.json records it: toolchains/tt-metal.pin,
  VERSION 5280a9cf (tt-metal 5280a9cfb00998fd49667a29523d03aee905c129),
  PREFIX_NAME `tt-metal@5280a9cf`, EXECUTABLE `/usr/bin/clang++-20`, the
  HOST_* flags, and EXPECT_VERSION
  `Ubuntu clang version 20.1.8 (++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132)`.
  The version check printed that line, then `Target: x86_64-pc-linux-gnu`,
  `Thread model: posix`, and `InstalledDir: /usr/lib/llvm-20/bin`, and
  exited 0. Every argv builds against the installed tree
  `/mnt/nvme10/joseph_ufl/toolchains/tt-metal@5280a9cf`, which the
  toolchain checked first (its lassi-install.txt names the pinned commit and
  its lassi-cpm-sources.txt equals toolchains/tt-metal-cpm-sources.txt).
- Environment: the stage runner's clean compile environment, PATH,
  `LANG=C` and `LC_ALL=C` (so clang's quotes are plain ASCII), no HOME, no
  loader variable, and no tt-metal variable. Each compile runs in the
  compile sandbox under `prlimit --core=1`, with a private TMPDIR,
  `@lassi-tmp`, inside its own workdir.
- Device: none.

The capture ran `uv run python tools/capture_toolchain_fixtures.py
--fixtures tests/toolchains/fixtures/ttmetal`, which built ttmetal-host the
way the stage runner does (the tree checks, the pinned EXECUTABLE, the clean
environment, the compile sandbox, and the `--version` check against the pin's
EXPECT_VERSION) and called its `build(files, workdir)` on each case's source
tree in a fresh workdir. Each `.stderr` file is that build's `compile.stderr`,
and the source trees the capture compiled are byte-identical to `sources/`.
`captures.json` records per case the argv, exit status, stderr sha256 and
size, and the parsed diagnostic count, and for the toolchain the pin values,
the executable, the environment variable names, and the `--version` banner.
The rx run record is `results/p4-ttmetal-fixtures/provenance.json`, and
`results/p4-ttmetal-fixtures/summary.md` summarizes the capture.

## Fixtures

| File | Case | Exit status | rx id |
| --- | --- | --- | --- |
| `ttm_clean.stderr` | Clean build of a host program that includes the pinned host API, beside a kernel under kernels/ that the host compiler did not build; empty stderr | 0 | `20260926-142744-desktop-8r113ei-detached-cf5f1274-326e` |
| `ttm_header_error.stderr` | Error in an included header: the include chain line, then the error in ./host/scale.h with echo and caret | 1 | `20260926-142744-desktop-8r113ei-detached-cf5f1274-326e` |
| `ttm_linker_error.stderr` | ld.lld undefined symbol with its two `>>>` context lines, then the clang driver's linker command failed line | 1 | `20260926-142744-desktop-8r113ei-detached-cf5f1274-326e` |
| `ttm_missing_header.stderr` | Fatal error for a header the pinned tree does not hold, with echo and caret at the include line | 1 | `20260926-142744-desktop-8r113ei-detached-cf5f1274-326e` |
| `ttm_note_chain.stderr` | No matching overload: the error, then one candidate note per overload, each with echo and caret | 1 | `20260926-142744-desktop-8r113ei-detached-cf5f1274-326e` |
| `ttm_syntax_error.stderr` | Missing semicolon: the error past the end of the declaration, with echo, caret, and a fix-it line | 1 | `20260926-142744-desktop-8r113ei-detached-cf5f1274-326e` |
| `ttm_undeclared_identifier.stderr` | Undeclared identifier in a loop body, with echo and caret | 1 | `20260926-142744-desktop-8r113ei-detached-cf5f1274-326e` |
| `ttm_unused_parameter.stderr` | Unused parameter: a warning with code -Wunused-parameter; without -Werror (OQ-027, option (b)) the build succeeds | 0 | `20260926-142744-desktop-8r113ei-detached-cf5f1274-326e` |

## What the capture showed

- clang names a header it found beside the including file with a leading
  `./` (`./host/scale.h:5:20: error: ...`); the parser drops the `./`, so the
  diagnostic names the built file `host/scale.h` and keeps its column.
- The fatal error for a missing header points at the `<` that starts the
  header name, column 10 of `#include <tt-metalium/p410_no_such_header.hpp>`.
- ttm_clean built with exit status 0 and an empty stderr, so its kernel
  `kernels/dataflow/p410_noop.cpp`, which includes the device header
  `dataflow_api.h`, was written beside the program and never given to the
  host compiler.

## Stored text

The files hold stderr as build() keeps it in its `compile.stderr`
attachment: the compiler's bytes decoded as UTF-8 with invalid bytes
replaced, then written as UTF-8. All eight files are plain ASCII with LF line
endings, so the bytes here are the compiler's bytes; `ttm_clean.stderr` is
empty.

One file carries text that belongs to its run and changes on a recapture:
`ttm_linker_error.stderr` names the capture's workdir, which holds the rx id
(`/mnt/nvme10/joseph_ufl/lassi-runs/fixture-captures/<rx id>/work/ttm_linker_error/`),
and clang's temporary object in it, `@lassi-tmp/main-eaa0c1.o`, whose name
is random on each run. The other seven name no workdir, temporary file, or
other per-run text.

## Recapturing

1. Commit every change to `sources/` and `scenarios.json` first: the capture
   must run from a clean commit, since a dirty-tree rx run is exploratory only.
2. Capture every case in one run, since `captures.json` must cover every
   fixture:
   `uv run tools/rx.py run -- 'uv sync --quiet && git status --short | wc -l && uv run python tools/capture_toolchain_fixtures.py --fixtures tests/toolchains/fixtures/ttmetal'`
   (the count must be 0: the slot holds the clean commit). The tool writes
   to `$LASSI_RUNS_ROOT/fixture-captures/<rx id>/` on alpha01.
3. Pull that directory into its own local folder with
   `uv run tools/rx.py pull --path lassi-runs/fixture-captures/<rx id> --into <dir>`,
   check that the manifest has `dirty` false and `snapshot_of` null, then
   copy each `.stderr` byte for byte here and `manifest.json` to
   `captures.json`, and fetch the run record with
   `uv run tools/rx.py pull <rx id> --into results/<name>`.
4. Derive each `<case>.json` again by hand from the new stderr, and update
   the table and provenance above.
