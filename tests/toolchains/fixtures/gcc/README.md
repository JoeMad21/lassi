# gcc-native stderr fixtures

Each `.stderr` file here is the raw stderr of one compile, captured on alpha01
with the pinned host g++ and copied byte for byte. `captures.json` is the
capture's provenance manifest, copied byte for byte from its `manifest.json`.
`sources/<case>/` holds the files each case compiled, and `scenarios.json`
names each case's toolchain and what its source does. The exact Diagnostic
list each file must parse into is in `<case>.json`, derived by hand from the
raw stderr (its `derivation` field says how), never copied from the parser's
output; tests/toolchains/test_gcc_diagnostics.py checks the parse against it.

No value in these files is a measurement: they record compiler output, not
performance. Nothing ran on a device, and no built program ran.

## Provenance

All six files come from one capture:

- rx run `20260925-195420-desktop-8r113ei-detached-94368a41-e825` on
  alpha01, dated 2026-09-25T19:54:21-07:00.
- Commit 94368a4 (`94368a41ba63c71a68773daecda83d184d543f0e`) with a clean
  tree: the manifest has `dirty` false and `snapshot_of` null, and the rx run
  record has `dirty` false.
- Pin, as `pin_files` in captures.json records it: toolchains/gcc.pin,
  VERSION 12.3.0, EXECUTABLE `/usr/bin/g++-12`, FLAGS `-O3 -fopenmp`, and
  EXPECT_VERSION `(Ubuntu 12.3.0-1ubuntu1~22.04.3) 12.3.0`. The version
  check printed `g++-12 (Ubuntu 12.3.0-1ubuntu1~22.04.3) 12.3.0` and exited 0.
- Environment: the stage runner's clean compile environment, PATH,
  `LANG=C` and `LC_ALL=C` (so GCC quotes are plain ASCII), no HOME. Each
  compile runs in the compile sandbox under `prlimit --core=1`, with a
  private TMPDIR, `@lassi-tmp`, inside its own workdir.
- Device: none.

The capture ran `uv run python tools/capture_toolchain_fixtures.py
--fixtures tests/toolchains/fixtures/gcc`, which built gcc-native the way the
stage runner does (the pinned EXECUTABLE, the clean environment, the compile
sandbox, and the `--version` check against the pin's EXPECT_VERSION) and
called its `build(files, workdir)` on each case's source tree in a fresh
workdir. Each `.stderr` file is that build's `compile.stderr`, and the
source trees the capture compiled are byte-identical to `sources/`.
`captures.json` records per case the argv, exit status, stderr sha256 and
size, and the parsed diagnostic count, and for the toolchain the pin values,
the executable, the environment variable names, and the `--version` banner.
The rx run record is `results/p4-gcc-fixtures/provenance.json`, and
`results/p4-gcc-fixtures/summary.md` summarizes the capture.

## Fixtures

| File | Case | Exit status | rx id |
| --- | --- | --- | --- |
| `gcc_clean.stderr` | Clean build: an OpenMP parallel loop with a reduction; empty stderr | 0 | `20260925-195420-desktop-8r113ei-detached-94368a41-e825` |
| `gcc_header_error.stderr` | Error in an included header: the include chain and function context lines, then the error in kernels/scale.h with echo and caret | 1 | `20260925-195420-desktop-8r113ei-detached-94368a41-e825` |
| `gcc_note_chain.stderr` | No matching overload: the error, then a candidate note and a nested conversion note for each overload, each with echo and caret | 1 | `20260925-195420-desktop-8r113ei-detached-94368a41-e825` |
| `gcc_syntax_error.stderr` | Missing semicolon: the error at the next token, then the recovery error for the loop variable | 1 | `20260925-195420-desktop-8r113ei-detached-94368a41-e825` |
| `gcc_undeclared_identifier.stderr` | Undeclared identifier in a loop body, with echo and caret | 1 | `20260925-195420-desktop-8r113ei-detached-94368a41-e825` |
| `gcc_wall_warning.stderr` | Unused variable, turned on by a diagnostic pragma: a warning with code -Wunused-variable; the build succeeds | 0 | `20260925-195420-desktop-8r113ei-detached-94368a41-e825` |

## Stored text

The files hold stderr as build() keeps it in its `compile.stderr`
attachment: the compiler's bytes decoded as UTF-8 with invalid bytes
replaced, then written as UTF-8. All six files are plain ASCII with LF line
endings, so the bytes here are the compiler's bytes; `gcc_clean.stderr` is
empty. None names a workdir, a temporary file, or other per-run text.

## Recapturing

1. Commit every change to `sources/` and `scenarios.json` first: the capture
   must run from a clean commit, since a dirty-tree rx run is exploratory only.
2. Capture every case in one run, since `captures.json` must cover every
   fixture:
   `uv run tools/rx.py run -- 'uv sync --quiet && git status --short | wc -l && uv run python tools/capture_toolchain_fixtures.py --fixtures tests/toolchains/fixtures/gcc'`
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
