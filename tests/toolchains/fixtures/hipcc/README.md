# hipcc-gfx942 stderr fixtures

Each `.stderr` file here is the raw stderr of one build, captured on alpha01
with the pinned host hipcc and copied byte for byte. `captures.json` is the
capture's provenance manifest, copied byte for byte from its
`manifest.json`. `sources/<case>/main.hip` is the file each case built, and
`scenarios.json` names each case's toolchain and what its source does.
Every source is a small HIP program written for these fixtures; none is
upstream text. The exact Diagnostic list each file must parse into is in
`<case>.json`, derived by hand from the raw stderr (its `derivation` field
says how), never copied from the parser's output;
tests/toolchains/test_hipcc.py checks the parse against it.

No value in these files is a measurement: they record compiler output, not
performance. Each build compiled host code and gfx942 device code and
linked the host program on alpha01's CPU; no built program ran, and no GPU
device was opened.

## Provenance

All eight files come from one capture:

- rx run `20261006-121638-desktop-8r113ei-detached-f03fa62a-961a` on
  alpha01, dated 2026-10-06T12:16:45-07:00.
- Commit f03fa62 (`f03fa62a5207436bbd8131911576a9382a0e0d42`) with a clean
  tree: the manifest has `dirty` false and `snapshot_of` null, the rx run
  record has `dirty` false, and the slot's `git status --short | wc -l`
  printed 0.
- Pin, as `pin_files` in captures.json records it: toolchains/hipcc.pin,
  VERSION `7.12.60610-2bd1678d3d`, EXECUTABLE
  `/opt/rocm/core-7.12/bin/hipcc`, FLAGS `--offload-arch=gfx942 -Wall -O3`,
  and EXPECT_VERSION `HIP version: 7.12.60610-2bd1678d3d`. The version
  check, `/opt/rocm/core-7.12/bin/hipcc --offload-arch=gfx942 --version`,
  printed that line, then
  `AMD clang version 22.0.0git (https://github.com/ROCm/llvm-project.git c849bc16b0e49951d313756f20b73c2b28d321d7+PATCHED:9a6ac45c97a1e511db838c5b46257324d2de1780)`,
  `Target: x86_64-unknown-linux-gnu`, `Thread model: posix`, and
  `InstalledDir: /opt/rocm/core-7.12/lib/llvm/bin`, and exited 0.
- Environment: the stage runner's clean compile environment, PATH,
  `LANG=C` and `LC_ALL=C` (so clang's quotes are plain ASCII), no HOME, no
  loader variable, and no HIP or ROCm variable; no ROCm environment script
  was sourced. Each build runs in the compile sandbox under
  `prlimit --core=1`, with a private TMPDIR, `@lassi-tmp`, inside its own
  workdir.
- Device: none. The run used alpha01's CPU only.

The capture ran `uv run python tools/capture_toolchain_fixtures.py
--fixtures tests/toolchains/fixtures/hipcc` in the same rx run, after the
remote tests (tests/toolchains/test_hipcc_remote.py) printed `6 passed`.
It built hipcc-gfx942 the way the stage runner does (the pinned
EXECUTABLE, the clean environment, the compile sandbox, and the
`--version` check against the pin's EXPECT_VERSION) and called its
`build(files, workdir)` on each case's source in a fresh workdir. Each
`.stderr` file is that build's `compile.stderr`, and the sources the
capture built are byte-identical to `sources/`. `captures.json` records per
case the argv, exit status, stderr sha256 and size, and the parsed
diagnostic count, and for the toolchain the pin values, the executable, the
environment variable names, and the `--version` banner. The rx run record
is `results/p17-hipcc-fixtures/provenance.json`, and
`results/p17-hipcc-fixtures/summary.md` summarizes the run.

## Fixtures

| File | Case | Exit status | rx id |
| --- | --- | --- | --- |
| `hip_clean.stderr` | Clean build of the saxpy program the remote test keeps; empty stderr | 0 | `20261006-121638-desktop-8r113ei-detached-f03fa62a-961a` |
| `hip_device_undefined.stderr` | gfx942 device link: `lld:` undefined hidden symbol with two `>>>` context lines, then the clang++ driver's amdgcn-link command failed line and hipcc's own line | 1 | `20261006-121638-desktop-8r113ei-detached-f03fa62a-961a` |
| `hip_host_calls_device.stderr` | Host function calls a `__device__` function: the error, then a candidate note, each with echo and caret; one summary for the gfx942 pass; hipcc's own line | 1 | `20261006-121638-desktop-8r113ei-detached-f03fa62a-961a` |
| `hip_kernel_calls_host.stderr` | Kernel calls a host function: the error, then a candidate note, each with echo and caret; one summary for the gfx942 pass; hipcc's own line | 1 | `20261006-121638-desktop-8r113ei-detached-f03fa62a-961a` |
| `hip_linker_error.stderr` | Host link: ld.lld undefined symbol with its two `>>>` context lines, then the clang++ driver's linker command failed line and hipcc's own line | 1 | `20261006-121638-desktop-8r113ei-detached-f03fa62a-961a` |
| `hip_missing_header.stderr` | Fatal error for a missing quoted header, with echo and caret at the include line; hipcc's own line | 1 | `20261006-121638-desktop-8r113ei-detached-f03fa62a-961a` |
| `hip_undeclared_identifier.stderr` | Undeclared identifier in a kernel body, with echo and caret, printed once; hipcc's own line | 1 | `20261006-121638-desktop-8r113ei-detached-f03fa62a-961a` |
| `hip_unused_variable.stderr` | Unused variable in a kernel: a warning with code -Wunused-variable, printed by the gfx942 pass and again by the host pass; without -Werror the build succeeds | 0 | `20261006-121638-desktop-8r113ei-detached-f03fa62a-961a` |

## What the capture showed

- hip_clean built with exit status 0 and an empty stderr: no driver
  warning, no summary line, and no line from hipcc.
- A warning in a kernel is printed twice, once by each pass: the warning,
  its echo and caret, and `1 warning generated when compiling for gfx942.`,
  then the same three lines and `1 warning generated when compiling for
  host.` The parse keeps both warnings, as stderr shows them; dropping the
  repeat is left to a later task.
- An error stops the build after the gfx942 pass, so each compile error is
  printed once, followed by `1 error generated when compiling for gfx942.`
  and no host summary. That holds for the host function calling a
  `__device__` function too: its error is reported under the gfx942 pass.
- The host and device target errors are both `no matching function for
  call to '<name>'` at the call, followed by a note at the declaration,
  `candidate function not viable: call to __device__ function from
  __host__ function` or `... call to __host__ function from __global__
  function`; file `main.hip`, with line and column of the name.
- The host link's linker is ld.lld (`ld.lld: error: undefined symbol:
  helper(int)`), not GNU ld, and the driver line is `clang++: error: linker
  command failed with exit code 1 (use -v to see invocation)`.
- The gfx942 device link fails in lld, which names itself `lld:` without
  the `ld.` (`lld: error: undefined hidden symbol: device_helper(int)`),
  and the driver line calls the step `amdgcn-link`. The part 1 parser read
  no such line, so the manifest counts 1 diagnostic for
  hip_device_undefined; lassi.toolchains.hipcc now reads it with its own
  DEVICE_LLD pattern, and the case parses into 2. ttmetal-host's patterns
  are unchanged.
- Every failed build ends with one line from hipcc itself,
  `failed to execute:` and the clang++ command it ran
  (`/opt/rocm/core-7.12/lib/llvm/bin/clang++  --offload-arch=gfx942
  --driver-mode=g++ --hip-link  -Wall -O3 -o "main" -x hip main.hip`, two
  blanks where hipcc printed them). It is not a compiler diagnostic and
  parses to nothing. No line starts with `hipcc`, and none comes from
  clang-offload-bundler.
- The summary lines (`N warning(s)` or `N error(s) generated when compiling
  for gfx942.` or `... for host.`) parse to nothing.

## Stored text

The files hold stderr as build() keeps it in its `compile.stderr`
attachment: the compiler's bytes decoded as UTF-8 with invalid bytes
replaced, then written as UTF-8. All eight files are plain ASCII with LF line
endings, so the bytes here are the compiler's bytes; `hip_clean.stderr` is
empty.

Two files carry text that belongs to their run and changes on a recapture,
in lld's `>>> ` context lines: each names the capture's workdir, which holds
the rx id
(`/mnt/nvme10/joseph_ufl/lassi-runs/fixture-captures/<rx id>/work/<case>/`),
and clang's temporary object in it, whose name is random on each run:
`hip_linker_error.stderr` names `@lassi-tmp/main-8098dc.o`, and
`hip_device_undefined.stderr` names `@lassi-tmp/main-gfx942-72a9c2.o` (in
two lines). The other six name no workdir, temporary file, or other per-run
text. hipcc's `failed to execute:` line holds no per-run text either: it
names the pinned clang++, the flags, and the relative `main` and `main.hip`.

## Recapturing

1. Commit every change to `sources/` and `scenarios.json` first: the capture
   must run from a clean commit, since a dirty-tree rx run is exploratory only.
2. Run the remote tests and, only if all pass, capture every case in one run,
   since `captures.json` must cover every fixture:
   `uv run tools/rx.py run -- 'uv sync --quiet && git status --short | wc -l && { LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -x -p no:cacheprovider -m remote tests/toolchains/test_hipcc_remote.py; t=$?; c=skipped; if [ "$t" -eq 0 ]; then uv run python tools/capture_toolchain_fixtures.py --fixtures tests/toolchains/fixtures/hipcc; c=$?; fi; echo "remote_rc=$t capture_rc=$c"; [ "$t" -eq 0 ] && [ "$c" = 0 ]; }'`
   (the count must be 0: the slot holds the clean commit). The tool writes
   to `$LASSI_RUNS_ROOT/fixture-captures/<rx id>/` on alpha01.
3. Pull that directory into its own local folder with
   `uv run tools/rx.py pull --path lassi-runs/fixture-captures/<rx id> --into <dir>`,
   check that the manifest has `dirty` false and `snapshot_of` null, then
   copy each `.stderr` byte for byte here and `manifest.json` to
   `captures.json`, and fetch the run record with
   `uv run tools/rx.py pull <rx id> --into results/<name>`.
4. Derive each `<case>.json` again by hand from the new stderr, and update
   the table, provenance, and per-run text above. The capture tool counts
   diagnostics with the parser of the commit it runs, so after a recapture
   every count in `captures.json` is the current parser's, and the
   PART_1_COUNTS exception in tests/toolchains/test_hipcc.py goes.
