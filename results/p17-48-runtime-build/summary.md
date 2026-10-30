# P17.48: The tt-mlir Runtime Install On alpha01

Task P17.48 (plans/p17-portable.md), owner decision OQ-111 (b): the pinned tt-mlir runtime, built on alpha01 with
toolchains/tt-mlir-runtime.sh. This is a toolchain install, not an evaluation: it ran no model, no generated program,
and no device. Every value below comes from provenance.json and output-tail.txt in this folder, except the fetched-sources list's
line count and sha256, which come from the job's output.log lines 6706 to 6732 (the list as the script printed it)
and can be recomputed from toolchains/tt-mlir-runtime-fetched-sources.txt.

## Provenance

| Field | Value (provenance.json) |
| --- | --- |
| rx job | 20261029-193649-p1748-tt-mlir-runtime-f0ad, state done, rc 0 |
| Commit | 3c6a69c0f3c2504ba8b991bdf4d34c6fbc97e223, clean (dirty: false) |
| Host | alpha01, Ubuntu 22.04.5 LTS, kernel 6.6.29+main+3.0.0r1-amd64-gio-epilmore-dev+, 256 CPUs |
| Device | none (a CPU build) |
| Job kind | a regular rx job (big: false): the night's suite job held the gate's one big-job slot |
| Command | `LASSI_ALLOWED_ROOT=/mnt/nvme10/joseph_ufl LASSI_STOP_GIB=345 LASSI_FREE_FLOOR_GIB=30 bash toolchains/tt-mlir-runtime.sh` (the spike's proposed settings under OQ-111) |
| Start, end | 2026-10-29T19:36:49-07:00 to 2026-10-29T19:58:24-07:00 (21 minutes 35 seconds) |
| Scratch free | 78.3 GB before, 59.1 GB after |
| Pins | the first 4000 characters of each toolchains/*.pin (the gate's limit cuts the longer pins); the full pins are those of clean commit 3c6a69c |

## Result [MEASURED, alpha01, 2026-10-29]

From output-tail.txt (the job's own log):

- Installed `tt-mlir@0.9.0.dev20260221-runtime` at tt-mlir 5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c; LLVM at
  4efe170d858eb54432f520abb4e7f0086236748b with only the pinned affine patch; flatbuffers at fb9afbaf; tt-metal and its
  submodules at their pinned commits, unchanged.
- ttrt 0.0.2995 built as a wheel and installed in the install's venv (log lines 16662 and 16673).
- ldd resolves every library of libTTMLIRRuntime.so, the runtime's Python module, and ttrt's runtime module; tt-metal's
  build holds libtt_metal.so, libdevice.so, libtt_stl.so, _ttnncpp.so, and _ttnn.so.
- ttmlir-opt and ttmlir-translate print 'LLVM version 22.0.0git' and were built by clang 17.0.6 and LLD 17.0.6.
- Nothing under the compiler install changed, and the root-filesystem check passed over the paths it could read (nothing
  changed under /tmp, /var/tmp, or /opt; find rc=1, 101 unreadable paths skipped).
- The install measures 18643512 KiB (du), about 17.8 GiB; the scratch root measured 290002864 KiB after it.

The install recorded its fetched sources (26 lines, sha256 c3308a86...0504133), copied into
toolchains/tt-mlir-runtime-fetched-sources.txt so that later runs of the script compare against them.

## Not Shown Here

Nothing here runs a flatbuffer: P17.48's driver and executor are still to be written, and until then the TTIR and D2M
arms (sweep13) are compile-only. The PROJECTED size and time in plans/spikes/ttmlir-runtime.md were estimates; this
install's measured values are the ones above.
