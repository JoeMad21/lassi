# The tt-mlir runtime build at the joint pin, and the LASSI driver on it (task P17.48, 2026-10-29)

## Question

The owner answered OQ-111 on 2026-10-29 with option (b): build tt-mlir's runtime at the joint pin (tt-mlir
0.9.0.dev20260221, commit 5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c) with MLIR's Python bindings and ttrt, in a new
prefix beside the compiler install, so TTIR and D2M flatbuffers can run on ttsim and later on a card; the build runs on
alpha01 on 2026-10-30, daytime, as one big rx job, CPU only, niced, no device. This note gives what the build needs
(read at the pins, with file:line), the pin and script that prepare it, the job's command line, its PROJECTED space and
time, the checks after it, and a design for the LASSI C++ driver of P17.48. Nothing here has been built or run; every
size and time is PROJECTED.

## Evidence and provenance

- Sources, read on 2026-10-29: tt-mlir at 5f396fd6 (the unpacked tarball of plans/spikes/ttir-d2m.md, sha256
  67afd5515290f7c98dc44170bba1a12d489a98d14044c5b7af2b3843850f03dc); tt-metal at 5280a9cfb00998fd49667a29523d03aee905c129,
  llvm-project at 4efe170d858eb54432f520abb4e7f0086236748b, nanobind at c5a3a378aa61d104c82ca053cb1e367782cd3618 (tt-metal's
  CPM pin of v2.10.2), CMake at v3.31.6, pip at 22.0.2, and pypa/wheel's docs/news.rst on main, each through `gh api`; PyPI's
  JSON API (release dates) and the PyTorch CPU index (torch 2.7.0+cpu: the cp310 manylinux_2_28_x86_64 wheel's size from a
  HEAD request, 175,951,851 bytes, and its sha256 from the index page; its metadata file). Every file:line below is at those
  commits.
- One read-only probe on alpha01 under the owner's standing permission for short CPU-only runs (plans/LESSONS.md, 2026-10-26):
  rx 20261029-175313-exec-9dd3 (alpha01's clock, UTC-7; 2026-10-30T00:53:13Z), CPU only, under one minute of reads plus a
  du of the scratch root, nothing written (the probe script was sent in the command; exploratory, not [MEASURED]). It read:

| What | Value |
| --- | --- |
| Host | 256 CPUs, 3022 GiB memory; LASSI_JOBS 32 |
| Scratch disk (df -Pk /mnt/nvme10) | 77,427,612 KiB available (73.8 GiB; 79.3 GB), 98% used |
| Scratch root (du -skx /mnt/nvme10/joseph_ufl) | 270,324,260 KiB (257.8 GiB), under the owner's lift of the 150G cap (OQ-074) |
| Compiler install | lassi-install.txt names tt-mlir 5f396fd6 and LLVM 4efe170d; lassi-fetched-sources.txt reads `none` |
| Its parts (KiB) | src/tt-mlir 65,748; src/llvm-project 2,638,040; src/flatbuffers 21,480; build/flatbuffers 16,632; build/llvm-project 4,374,396; build/tt-mlir 557,420; toolchain 103,884 (venv 97,548); all 7,777,620 |
| Its venv | pip 22.0.2, setuptools 59.6.0 |
| Its Ninja logs (steps, span, summed step time) | LLVM 5,812 steps, 422 s, 13,059 s; tt-mlir 776, 78 s, 1,996 s; flatbuffers 45, 8 s, 100 s |
| Its file times | 2026-10-25 23:33:42 (record created) to 23:43:41 -0700 (ttmlir-opt): about 10 minutes for the whole install |
| Python 3.10 development files | /usr/include/python3.10/Python.h; /usr/lib/x86_64-linux-gnu/libpython3.10.so |
| MPI | /opt/openmpi-v5.0.7-ulfm/lib/libmpi.so.40 present; /opt/tenstorrent/sfpi present |

## What the runtime build needs, at the pins

- tt-mlir's options (CMakeLists.txt): TTMLIR_ENABLE_RUNTIME (:33) turns on the TTNN, TTMetal, and distributed runtimes by
  default (:50-58). libTTMLIRRuntime links TT-NN's _ttnncpp.so whatever they are set to (runtime/CMakeLists.txt:29,
  runtime/lib/CMakeLists.txt:30-53). The pin keeps TTMetal and TTNN on (upstream's tested setting:
  .github/workflows/call-build-release.yml:109-124) and turns the distributed runtime off (a multi-host controller and
  worker, runtime/lib/distributed/CMakeLists.txt:5-8, runtime/bin/distributed/CMakeLists.txt:1-3).
- ttrt and the bindings: ttrt builds only under TTMLIR_ENABLE_BINDINGS_PYTHON (:84) and MLIR_ENABLE_BINDINGS_PYTHON
  (tools/CMakeLists.txt:6, :23-25), which tt-mlir reads from LLVM's MLIRConfig (cmake/modules/FindMLIR.cmake:17-20), so
  LLVM is built again with the bindings (env/CMakeLists.txt:68 has them on). The compiler install's LLVM build is not
  reused: changing its configuration would change that install. LLVM's MLIR bindings need Python's Development.Module,
  pybind11 2.10 or newer, and nanobind 2.9 or newer (mlir/cmake/modules/MLIRDetectPythonEnv.cmake:22-23, :50, :57), with
  mlir/python/requirements.txt:1-7's bounds, which env/CMakeLists.txt:55 installs. tt-mlir's Python extension links
  MLIRTestToLLVMIRTranslation (python/CMakeLists.txt:144), defined only in MLIR's test library
  (mlir/test/lib/Dialect/Test/CMakeLists.txt:116 at LLVM 4efe170d), which exists only under MLIR_INCLUDE_TESTS, whose
  default is LLVM_INCLUDE_TESTS (mlir/CMakeLists.txt:160-162, :247-257); so LLVM_INCLUDE_TESTS goes back to on, LLVM's
  and upstream's default, where the compiler pin had it off.
- ttrt's build (tools/ttrt/CMakeLists.txt:11-37) runs `python -m pip install -r requirements.txt` (torch==2.7.0 from the
  PyTorch CPU index, tools/ttrt/requirements.txt:1), then `python -m pip wheel . --wheel-dir build` on a setup.py project
  with no pyproject.toml, then `python -m pip install build/*.whl --force-reinstall`. At the venv's pip 22.0.2 a project
  without pyproject.toml takes the legacy `setup.py bdist_wheel` path (src/pip/_internal/pyproject.py:93-94;
  wheel_builder.py:63-65), which needs the wheel package (0.46.2 restored its bdist_wheel for setuptools older than 70.1,
  and the venv keeps setuptools 59.6.0) and no build environment from an index; `pip wheel` also collects ttrt's
  install_requires, nanobind==2.10.2 (setup.py:45-46), as a wheel file, which the wheelhouse provides offline. setup.py
  copies libTTMLIRRuntime, the runtime's Python module, tt-metal's libraries, and tt-metal's tt_metal, runtime (sfpi
  included), and ttnn trees into the wheel (setup.py:63-195).
- tt-metal: with the runtime on, tt-mlir's ExternalProject clones tt-metal at TT_METAL_VERSION 5280a9cf, the joint pin's
  (third_party/CMakeLists.txt:3), whole and with its submodules (:103-131 set no GIT_SHALLOW; CMake v3.31.6
  ExternalProject/shared_internal_commands.cmake:417-425, gitclone.cmake.in:24-75), into
  src/tt-mlir/third_party/tt-metal/src/tt-metal, keeps its CPM cache there unless CPM_SOURCE_CACHE is set (:9-13),
  configures it with WITH_PYTHON_BINDINGS=ON, ENABLE_DISTRIBUTED=ON on x86_64, ENABLE_TRACY from the perf-trace option,
  ENABLE_LIBCXX=OFF, and tt-mlir's TT_USE_SYSTEM_SFPI (:95-126), builds its default target (TT-NN and its nanobind module
  included), and installs it into its own build_Release (:102, :114). No option takes an installed tt-metal, and LASSI's
  tt-metal installs have no TT-NN, so nothing is reused. tt-mlir's TT_USE_SYSTEM_SFPI defaults on (:48); on, tt-metal
  takes /opt/tenstorrent/sfpi (tt_metal/hw/CMakeLists.txt:62-64), which on alpha01 is not the pinned sfpi (bible, Toolchain
  Pins, the tt-xla venv: 7.48.0) and fails tt-metal's version check (:125-130), so the pin sets it off and tt-metal
  fetches sfpi 7.25.0 checked by URL_HASH (:65-76), the txz toolchains/tt-metal.pin records. ENABLE_DISTRIBUTED links the
  ULFM MPI under /opt/openmpi-v5.0.7-ulfm when its libmpi.so.40 exists, else a system MPI, else the configure fails
  (tt_metal/distributed/CMakeLists.txt:37-75). tt-metal's CMake has no clang version check (cmake/compilers.cmake:1-22),
  and tt-mlir's CI builds this tt-metal with clang-17 (.github/Dockerfile.base:53-60). ttnn's nanobind library is named
  -abi3 (ttnn/CMakeLists.txt:797-798), but at nanobind 2.10.2 nanobind_build_library defines Py_LIMITED_API only through
  nanobind_add_module on Python 3.12 or newer (cmake/nanobind-config.cmake:356-361, :400-402), so on Python 3.10 it builds
  a regular extension (read, not built).
- Python 3.10: tt-mlir's environment uses Python 3.12 (env/init_venv.sh:10) and its CI 3.12 (.github/Dockerfile.base:44);
  LASSI pins the host's 3.10.12 (toolchains/tt-mlir.pin). Every .py file under tools/ttrt, python/ttmlir, and
  python/common compiles under CPython 3.10 (38 files, the workstation's 3.10.21, compiled, not run), and the bounds
  above admit 3.10. Upstream does not test 3.10; that is a risk below.
- Parallelism: tt-metal's inner build is ExternalProject's `cmake --build .` (ExternalProject.cmake:1576-1589 at CMake
  v3.31.6), which takes its job count from CMAKE_BUILD_PARALLEL_LEVEL; with no value Ninja would run 258 jobs on this
  host. The script sets it to LASSI_JOBS and builds tt-metal's step targets alone (third_party/CMakeLists.txt:133;
  custom steps join the main target, ExternalProject.cmake:2064-2069), so no more than LASSI_JOBS compile at once.

## What was prepared

- toolchains/tt-mlir-runtime.pin: BUILD runtime, PREFIX_NAME tt-mlir@0.9.0.dev20260221-runtime, COMPILER_PREFIX_NAME
  tt-mlir@0.9.0.dev20260221 (never touched); the compiler pin's sources, host tools, and layout; LLVM_CMAKE_FLAGS with the
  bindings and tests on; tt-mlir's CMAKE_FLAGS for the runtime; tt-metal's commit, submodules, sfpi, and the settings read
  back from its cache; the build targets in order; INSTALL_KIB (30 GiB), GUARD_SECONDS, NICE_MIN. Each value carries its
  file:line.
- toolchains/tt-mlir-runtime-requirements.txt: the compiler venv's 25 pins unchanged, plus numpy 2.1.2, pybind11 2.13.6,
  PyYAML 6.0.1, ml_dtypes 0.5.4, typing_extensions 4.15.0 (newest inside mlir/python/requirements.txt's bounds on
  2026-02-21, the release date), torch 2.7.0+cpu with filelock, fsspec, mpmath, networkx, and sympy, and wheel 0.46.3; 37
  pins, each with its sha256 list, from two uv compiles (the header gives the commands).
- toolchains/tt-mlir-runtime-fetched-sources.txt: PLACEHOLDER until the first install records its list.
- toolchains/tt-mlir-runtime.sh, a sibling of toolchains/tt-mlir.sh (which is unchanged): the same value checks,
  allowed-root checks, 1-byte core limit, cleared environment, fetches by commit, patch checks, cache checks,
  root-filesystem check, and in-place install record; and in addition a required LASSI_FREE_FLOOR_GIB, a refusal of any
  layout in which the two prefixes hold one another, a niceness check (NICE_MIN, the gate's 10), a host check of the MPI
  library, a venv installed from a hash-checked wheelhouse and pip offline after it (PIP_NO_INDEX, PIP_FIND_LINKS), every
  Ninja run at LASSI_JOBS under a free-space guard that stops it (SIGTERM) when the disk's free space falls under the
  floor, tt-metal's clone checked (commit, gitlinks, sfpi-version) before its configure and its cache and sfpi checked
  before its build, the fetched sources (tt-metal's clone and CPM cache) recorded before tt-metal is built, checks of the
  runtime library and module (ldd, compiler and linker), tt-mlir's Python package (an import and an MLIR context),
  ttrt's installed version (its metadata, not an import), and a refusal of the record when anything under the compiler
  install changed during the run. It never deletes anything but its start marker.
- tests/toolchains/test_tt_mlir_runtime_pins.py: 121 tests of the pin, the lists, the script's text, and its functions
  run in bash with stand-ins (no network, no build, no device).

## Space

PROJECTED, not measured, in GiB:

| Part | PROJECTED | Basis |
| --- | --- | --- |
| src/tt-mlir, flatbuffers, its build and install | 0.1 | the compiler install's parts (rx 9dd3) |
| src/llvm-project | 2.5 | the same commit at depth 1 (rx 9dd3: 2,638,040 KiB) |
| build/llvm-project | 5.0 to 6.5 | 4.2 measured without the bindings and MLIR's test libraries (rx 9dd3), plus those |
| build/tt-mlir | 2.0 to 4.0 | 0.53 for the two tools (rx 9dd3), plus the runtime, the TTNN runtime's operations, the Python CAPI library and extension, and the Python packages |
| tt-metal (clone, CPM cache, sfpi, build_Release with TT-NN, its install) | 6.3 to 8.8 | the TurboQuant checkout, a whole clone with TT-NN built: .git 1.39, CPM cache 1.55, build_Release 1.34, sfpi 0.43, sources 0.42 (plans/spikes/p4-tt-pins.md); GitHub gives tt-metal's repository today as 2,444,047 KB (`gh api repos/tenstorrent/tt-metal`, 2026-10-29), so the clone may be larger than 1.39 |
| ttrt (the package's copies, setuptools' build tree, the wheel, the venv's copy) | 3.4 to 5.2 | setup.py:145-168 copies tt_metal, runtime (sfpi), and ttnn three to four times |
| venv and wheelhouse | 1.1 to 1.3 | torch's cp310 wheel 0.16 (175,951,851 bytes) installs to about 0.8 (inferred) |
| All | about 20 to 29 | INSTALL_KIB plans for 30 |

- The disk: 73.8 GiB free on 2026-10-29 (rx 9dd3). The gate refuses a big job under 60 GB (1e9 bytes) free
  (tools/server/config.default.json, min_free_gb_big). The script refuses to start unless the planned 30 GiB plus the
  floor fit in the free space, so with LASSI_FREE_FLOOR_GIB=30 it needs 60 GiB (64.4 GB) free at its start, and it stops
  any build whose disk falls under 30 GiB free, checked every 30 seconds.
- The stop line: the scratch root holds 257.8 GiB under the owner's lift of the 150G cap (OQ-074, plans/LESSONS.md); the
  command line below gives LASSI_STOP_GIB=345, the value plans/runs/train-tt/run.sh:7 used under the lift, and du plus
  the planned 30 GiB is about 288 GiB.
- Space the owner must free first: none at today's reading. If the disk holds under 60 GiB free when the job starts
  (tonight's run writes to it), the script refuses before any download, and the owner frees the difference.

## Time

PROJECTED, not measured: 40 to 100 minutes at 32 jobs. The compiler install on this host took about 10 minutes from its
record's creation to its last tool (file times read in rx 9dd3, not a logged run), LLVM's build 422 s of Ninja span; the
P4.2 tt-metal install took 3 min 44 s with a 131 s configure [MEASURED, results/p4-tt-install]. By part: the venv 1 to 3
minutes; LLVM 9 to 15 (more targets: MLIR's test libraries and bindings); tt-metal's clone 2 to 10 (network; its whole
history); its configure 2 to 4 (P4.2's 131 s, CPM downloads); its build with TT-NN and its Python module 12 to 35 (about
1,088 TT-NN host files at the pin, plans/spikes/ttir-d2m.md section 4, in unity batches); tt-mlir's targets 4 to 10;
ttrt's wheel 3 to 8; the checks, du of the 258 GiB scratch root twice among them, 2 to 6. The job timeout below is
4 hours.

## The job

Before: the pin, script, lists, and tests committed and pushed on the branch the job runs from (rx syncs the current
commit), `uv run tools/rx.py doctor` (free space, running jobs: none big), and no other big job. Then:

```
uv run tools/rx.py job start --big --name p1748-tt-mlir-runtime --timeout 14400 -- \
  'LASSI_ALLOWED_ROOT=/mnt/nvme10/joseph_ufl LASSI_STOP_GIB=345 LASSI_FREE_FLOOR_GIB=30 bash toolchains/tt-mlir-runtime.sh'
uv run tools/rx.py job wait <id> --timeout 600
```

The command names no device and none of the words the gate's tt_guard reads (tools/server/gate.py, check_command), and
the gate runs it under nice 10 and ionice (gate.py, wrap). The script exits 0 with lassi-install.txt written, or non-zero
with its unfinished tree left for a rerun at the same pin to resume (each fetch and finished ExternalProject step is
kept). The first install compares no fetched sources: copy $LASSI_TOOLCHAINS/tt-mlir@0.9.0.dev20260221-runtime/
lassi-fetched-sources.txt into toolchains/tt-mlir-runtime-fetched-sources.txt and commit it, as for the compiler pin;
until then a rerun refuses the install.

## Checks after the job

One read-only `rx exec`, CPU only, a minute or two, nothing written. No command loads the runtime: ldd lists the
libraries a file links in the loader's trace mode, which runs none of their code, and the import below is the
compiler's MLIR bindings alone. If the gate's device guard refuses a command, the command is not reworded to pass
it: the check waits for the owner's approval or is left out.

```
R=$LASSI_TOOLCHAINS/tt-mlir@0.9.0.dev20260221-runtime
head -3 $R/lassi-install.txt
wc -l $R/lassi-fetched-sources.txt
ldd $R/build/tt-mlir/runtime/lib/libTTMLIRRuntime.so
ldd $R/build/tt-mlir/runtime/python/_ttmlir_runtime.cpython-310-x86_64-linux-gnu.so | grep -c 'not found'
$R/toolchain/venv/bin/python -I -c "import sys; sys.path.insert(0, '$R/build/tt-mlir/python_packages'); import ttmlir.ir as ir; ir.Context(); print('ttmlir ok')"
du -sk $R; df -Pk /mnt/nvme10/joseph_ufl
```

ttrt is neither imported nor run on alpha01 as a check: its import loads the runtime libraries, and the cards are
present there (the gate's silicon guard, tools/server/gate.py:309-312). The install script verifies ttrt's version
from its installed metadata (expected `ttrt 0.0.2995`, TTMLIRVersion.cmake:44-63, setup.py:11-15, PROJECTED) and
runs the other checks above before it writes the record. No ttrt subcommand runs on alpha01.

## Risks, unverified

- tt-mlir's Python package is configured against LLVM's build tree, as the compiler install's tools are; upstream
  configures against an installed LLVM (env/CMakeLists.txt:61). MLIR exports its declared Python sources for the build
  tree too (inferred from mlir/cmake/modules and the standalone example; not built). If tt-mlir's configure refuses
  them, the fallback is an LLVM install into toolchain/, upstream's layout, about 3 to 5 GiB more (PROJECTED), a pin
  change.
- Python 3.10, which upstream does not test (above); TT-NN's Python module and the MLIR bindings compile against it by
  reading only.
- MLIRTestToLLVMIRTranslation from the build tree: linked by name from LLVM's library directory
  (CMakeLists.txt:166, link_directories) or as an exported target (inferred).
- The clone of tt-metal is whole: its size and the network time are the least known parts above.
- The free-space guard stops a build with SIGTERM between its checks: up to 30 seconds of writes can pass the floor.
- With the TTNN runtime on, ttrt's wheel carries TT-NN's library and kernels; ttrt's Python module and runtime are built
  as upstream builds them, but nothing here runs them.

## The LASSI driver (design note for P17.48)

P17.48's executor runs a TTIR or D2M flatbuffer through the pinned runtime on ttsim (later a card, which a human starts),
with the held-out inputs as tensors and the outputs as lassi_io files. ttrt cannot load given inputs
(plans/spikes/ttir-d2m.md, section 4), so a small C++20 program, the driver, does it on libTTMLIRRuntime. Not written yet.

Build: clang++-17, C++20, against the runtime prefix: include runtime/include and build/tt-mlir/runtime/include (the
generated flatbuffer headers), build/tt-mlir/include/ttmlir/Target/Common, and toolchain/include (flatbuffers); link
build/tt-mlir/runtime/lib/libTTMLIRRuntime.so, whose RUNPATH ($ORIGIN and tt-metal's library directory,
runtime/lib/CMakeLists.txt:64-67) finds tt-metal's libraries; read and write arrays with assets/harness/c/lassi_io.h
(lassi_io_read, lassi_io_write).

Run, one flatbuffer per process, one device open and one close (Agent Rule 9: no looped device opens), inside the sandbox
executor with the ttsim row's environment (TT_METAL_SIMULATOR naming ttsim v1.3.4's library, TT_METAL_SLOW_DISPATCH_MODE=1,
TT_METAL_RUNTIME_ROOT at the runtime prefix's tt-metal tree, TT_METAL_CACHE a LASSI-owned directory), with
/opt/openmpi-v5.0.7-ulfm readable (tt-metal's RUNPATH names it, tt_metal/CMakeLists.txt:285-290). The calls, at
5f396fd6 (runtime/include/tt/runtime/runtime.h and types.h; definitions in runtime/lib/runtime.cpp and
runtime/lib/ttmetal/runtime.cpp):

1. Load: `Binary::loadFromPath(path)` (types.h:320). Refuse unless `getFileIdentifier()` is "TTM0" (types.h:284;
   include/ttmlir/Target/TTMetal/binary.fbs:61) and `checkSchemaHash()` holds (types.h:287).
2. Runtime: `setCompatibleDeviceRuntime(binary)` (runtime.h:38; runtime.cpp:186-203), which selects TTMetal for TTM0.
3. Descriptor: refuse unless `binary.getSystemDescAsJson()` (types.h:326) equals the recorded ttsim descriptor's
   `asJson()` (`SystemDesc::loadFromPath`, types.h:296; `asJson`, :289): a flatbuffer built against the mock descriptor
   places L1 buffers below tt-metal's reserved L1 and the runtime does not check addresses (plans/spikes/ttir-d2m.md,
   section 1). The descriptor is written once, by the driver's second mode, `getCurrentSystemDesc()` (runtime.h:44-47;
   runtime.cpp:229-247; runtime/lib/common/system_desc.cpp:265-285) then `store(path)` (types.h:279), on ttsim, and
   recorded with the pins; the toolchains pass it to ttmlir-opt as system-desc-path.
4. Program and its tensors: `getNumPrograms()` (types.h:325), program 0 (or by `getProgramName`, :327);
   `getProgramInputs(0)` and `getProgramOutputs(0)` (types.h:335-336) give each tensor's TensorDesc: logical shape, data
   type, item size, stride, and physical volume (types.h:169-189).
5. Inputs: read each lassi_io input, in the program's input order, named by the item's inputs list. Refuse a dtype or
   element count that differs from the TensorDesc; map LASSI_IO_F32, BF16, I32, U32, and U8 to Float32, BFloat16, Int32,
   UInt32, and UInt8 (include/ttmlir/Target/Common/types.fbs:18-31), and refuse F16, I8, and I64, which the TTMetal
   runtime does not support (runtime/lib/common/utils.cpp:172-193). Lay each array out at the TensorDesc's stride in a
   zeroed buffer of `sizeBytes()` (types.h:186), row by row when `isPadded()` (:188), since the programs take host
   tensors of the logical shape with padded strides (plans/spikes/ttir-d2m.md, section 5).
6. Device: `openMeshDevice(options)` with meshShape {1, 1} (runtime.h:145; types.h:202-212;
   ttmetal/runtime.cpp:249-280).
7. Tensors: `createBorrowedHostTensor(data, shape, stride, itemsize, dataType)` (runtime.h:55-59; runtime.cpp:284-306,
   which makes the TTMetal runtime's TensorDesc-backed host tensor, ttmetal/runtime.cpp:76-85). Not
   createOwnedHostTensor: its HostBuffer tensor is one the TTMetal toLayout refuses (ttmetal/runtime.cpp:44-62), and the
   TTMetal path needs no layout change (getLayout returns an empty layout, :38-42).
8. Run: `submit(device, binary, 0, inputs)` (runtime.h:268-270; runtime.cpp:1166-1184; ttmetal/runtime.cpp:943-1004),
   then `wait(outputs)` (runtime.h:206-207; ttmetal/runtime.cpp:508-512).
9. Outputs: for each, `toHost(output, true)` (runtime.h:211-212; a no-op for host tensors, ttmetal/runtime.cpp:514-529)
   and `memcpy(buffer, host)` (runtime.h:227-228; ttmetal/runtime.cpp:580-604) into a buffer of the output TensorDesc's
   `sizeBytes()`; drop the stride padding back to the logical shape; write the lassi_io file named by the item's
   outputs list, in the program's output order, with the output's dtype.
10. Close: `deallocateTensor(output, true)` (runtime.h:234) for each, then `closeMeshDevice(device)` (runtime.h:147).

Failures: the runtime's LOG_FATAL and LOG_ASSERT throw (runtime/include/tt/runtime/detail/common/logger.h:388-402); the
driver catches std::exception, prints one diagnostic line, and exits non-zero, and the executor reads a signal (a
simulator abort) as a run failure with the signal recorded, as the ttsim row's programs are read now. A Toolchain per
language (ttir, d2m) runs `ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=<ttsim descriptor>
enable-multicast-inference=false"` and `ttmlir-translate --ttmetal-to-flatbuffer` from the runtime prefix's build tree
(the same commit's tools), as plans/spikes/ttir-d2m.md's consequences give.

## Proposed Decision Log entry (for the main session; a pin change, Agent Rule 10)

> | 2026-10-29 | The tt-mlir runtime build variant (task P17.48; the owner's answer to OQ-111, option (b); Toolchain
> Pins): toolchains/tt-mlir-runtime.pin builds the joint pin's tt-mlir 5f396fd6 with its runtime (TTMetal and TTNN on,
> distributed off), MLIR's and tt-mlir's Python bindings, and ttrt at $LASSI_TOOLCHAINS/tt-mlir@0.9.0.dev20260221-runtime,
> beside the compiler install, which it never touches; LLVM 4efe170d is built again with the bindings and MLIR's test
> library; tt-mlir's own tt-metal 5280a9cf is cloned and built with TT-NN and its Python module; the venv adds numpy
> 2.1.2, pybind11 2.13.6, PyYAML 6.0.1, ml_dtypes 0.5.4, typing_extensions 4.15.0, torch 2.7.0+cpu, and wheel 0.46.3,
> each by sha256 | Running TTIR and D2M flatbuffers needs tt-mlir's runtime at the same commit, and ttrt needs MLIR's
> Python bindings (tools/CMakeLists.txt:6, :23-25); tt-mlir's Python extension links MLIR's test translation
> (python/CMakeLists.txt:144); /opt/tenstorrent/sfpi on alpha01 is not the pinned sfpi, so TT_USE_SYSTEM_SFPI is off.
> PROJECTED: about 20 to 29 GiB and 40 to 100 minutes at 32 jobs. plans/spikes/ttmlir-runtime.md records the sources,
> the probe (rx 20261029-175313-exec-9dd3), and the design |

## Result of the first install (2026-10-29)

The job ran on alpha01 from clean commit 3c6a69c as a regular rx job, not --big, because the night's suite job held the
gate's one big-job slot (rx 20261029-193649-p1748-tt-mlir-runtime-f0ad, rc 0; results/p17-48-runtime-build):
21 minutes 35 seconds, an install of about 17.8 GiB (du), scratch free 78.3 GB before and 59.1 GB after, ttrt
0.0.2995 installed, and the job's own checks (ldd, the tools' versions, the pinned sources unchanged, the compiler
install unchanged, the root filesystem) printed as passing in its log (output-tail.txt there). These replace the
PROJECTED space, time, and ttrt version above [MEASURED, alpha01]. The install's fetched-sources list (26 lines) is
copied into toolchains/tt-mlir-runtime-fetched-sources.txt.

## The driver, the executor, and the first ttsim runs (2026-10-30, exploratory)

Exploratory, never [MEASURED]: every run below came from a dirty worktree (detached at bd5e51f; rx sent each tree as
a snapshot commit), on alpha01 under the owner's standing permission for short CPU-only runs: `rx run` or `rx exec`,
each under `--timeout 600` or less, CPU only, no `--big`, no model. Every process that loaded the runtime ran inside
the sandbox (lassi.executors.sandbox) with the ttsim row's environment, TT_METAL_SIMULATOR on ttsim v1.3.4's library,
where no /dev/tenstorrent node is visible, and opened at most one device, on ttsim; no card was opened. rx ids use
alpha01's clock (UTC-7, 2026-10-29 evening there); an id written `<stamp>-...-<tag>` below stands for
`<stamp>-desktop-8r113ei-detached-bd5e51fc-<tag>`, the worktree's slot. The scratch disk read 68G available (`df -BG /mnt/nvme10`) before
the first build and after the last run; the window3 job ran beside them.

What was written (the repository side): toolchains/tt-mlir-driver/lassi_ttmlir_run.cpp (the driver, both modes),
toolchains/tt-mlir-driver.pin and toolchains/tt-mlir-driver.sh (its build), lassi/executors/ttmlir_ttsim.py (Executor
"ttmlir-ttsim", the ttsim executor with the driver's argv, the runtime and driver install checks, and the driver's
diagnostics; lassi/executors/ttsim.py gained the `_argv` and `_toolchains_root` hooks it uses),
tools/ttmlir_system_desc.py (the descriptor's write and install), tests (tests/executors/test_ttmlir_ttsim.py,
tests/toolchains/test_tt_mlir_driver_pin.py), and the probe files of this section in plans/spikes/ttmlir-runtime/
(add.mlir, add_bf16.mlir, histo.mlir, probe.py, compare.py, diag.py).

| rx id | What | Result |
| --- | --- | --- |
| 20261029-205005-exec-2cd4 | Read-only: the runtime prefix's layout, readelf of libTTMLIRRuntime.so, clang++-17, df | include and library paths as the pin names them; RUNPATH $ORIGIN and tt-metal's build_Release/lib; 68G available |
| 20261029-210313-...-e1e8 | Build driver fb17bf849a92 | built and installed on the first try; ldd finds every library |
| 20261029-210329-...-a45a | describe mode in the sandbox | exit 139 (SIGSEGV) after 0.4 s, no output |
| 20261029-210412-...-c10e | diag.py usage, refuse, describe | all three exit 139 in 0.4 s: the fault comes before main |
| 20261029-210430-...-91dc | diag.py ldd, gdb | ldd in the sandbox finds every library (MPI from /opt/openmpi-v5.0.7-ulfm); gdb: `vfork: Resource temporarily unavailable` |
| 20261029-210448-exec-e421 | Read-only: the user's task counts and limits | 29,863 threads of the user (io_uring workers of window3), ulimit -u 12,347,776, the user slice's pids.current 24,582 of 1,384,119: not the cause |
| 20261029-210525-...-1e06 | diag.py usage, refuse with TasksMax 4096 | still exit 139: not the sandbox's task cap |
| 20261029-210538-...-e2a1 | diag.py gdb with TasksMax 4096 | SIGSEGV at 0x7ffff75b217c; gdb cannot read the process (/proc/PID/mem opens read-only in the sandbox) |
| 20261029-210603-...-0555 | LD_DEBUG=init | not an option of this glibc |
| 20261029-210614-...-7d39 | LD_DEBUG=files | the last `calling init:` before the fault is _ttnncpp.so |
| 20261029-210814-...-bde4 | Build driver 5e91d6a405ff (setenv HOME in .preinit_array); usage, refuse | built; still exit 139 |
| 20261029-210833-...-759c | readelf of 5e91d6a405ff; withhome (HOME=/tmp through env, a diagnosis) | PREINIT_ARRAY present; with HOME set the driver runs: usage refusal, exit 2 |
| two runs, ids not kept | LD_DEBUG=files on 5e91d6a405ff | `calling preinit` comes first, then `calling init` of ld.so and libc, then the others |
| 20261029-211006-...-d412 | Build of dc422de5d697 | nothing ran: the local shell rewrote the command's leading `VAR=/path`; rerun with MSYS_NO_PATHCONV=1 |
| 20261029-211022-...-7302 | Build driver dc422de5d697 (re-exec with HOME in .preinit_array); usage, refuse | built; usage exit 2, refuse exit 2 (`cannot open the flatbuffer missing.ttm`) |
| 20261029-211036-...-ddac | tools/ttmlir_system_desc.py --install | exit 0, 2.0 s simulator wall time; 552 bytes, sha256 54679eed160e559a75758b0bcf1f577f890194a6b1f83dc99f5b755199f5e302, installed in the driver prefix |
| 20261029-211100-...-0497 | probe.py add | compiled; the probe stopped before the run (its @inputs directory was not made) |
| 20261029-211111-...-d0a3 | probe.py add (TT-Metal) | exit 0, outputs compared (below) |
| 20261029-211133-...-e509 | probe.py histo (TT-NN) | exit 1, ttsim UndefinedBehavior (below) |
| 20261029-211152-exec-cb60 | Read-only: histo's lowered module and its JIT cache | 64 ttnn.scatter; tilize kernels only (below) |
| 20261029-211322-...-b94e | probe.py add-ttnn (f32 through TT-NN) | exit 1, ttsim UndefinedBehavior (below) |
| 20261029-211351-...-a3c1 | probe.py add-bf16-ttnn | exit 0, outputs compared (below) |
| 20261029-212050-...-655c | Build driver 52325796c083 (a comment of the source changed); tools/ttmlir_system_desc.py --install | the program's sha256 is dc422de5d697's, 1dd776affd27f02d181a4ea5a682184b4a99ed53ab5d47779dc0baf0caeea287 (the same bytes); describe exit 0 after 1.9 s; the same 552 bytes, sha256 54679eed...e302, installed in the new prefix |
| 20261029-211420-exec-a9de | Read-only: the descriptor as JSON (the runtime toolchain's flatc 24.3.25), du, df | below; this session's files on alpha01: three driver prefixes of 84, 84, and 88 KiB, lassi-runs/ttmlir-runtime-spike 45,488 KiB, lassi-runs/ttmlir-system-desc 2,516 KiB |

Findings:

- The driver builds against the runtime prefix with the runtime's clang++-17 and ld.lld-17 (toolchains/tt-mlir-driver.pin)
  and its libraries resolve inside the sandbox, MPI from /opt/openmpi-v5.0.7-ulfm included.
- HOME. With no HOME in its environment (the ttsim row's, bible Execution Backends), the driver died by SIGSEGV while
  its libraries loaded: TT-NN's configuration, a static object of _ttnncpp.so (ttnn/core/config.cpp:9 at tt-metal
  5280a9cf), builds its cache paths from `std::filesystem::path{std::getenv("HOME")}` with no check
  (ttnn/api/ttnn/config.hpp:23-24), and libTTMLIRRuntime links _ttnncpp.so whatever runtime runs
  (runtime/lib/CMakeLists.txt:30-53), so TT-Metal flatbuffers need HOME too. With HOME set the driver runs (rx 759c). A
  setenv in the executable's .preinit_array is lost, since libc's initializer runs after it and reinstalls the initial
  environment array (LD_DEBUG=files: preinit first, then libc). The driver now reads the initial environment in its
  preinit function and, when it has no HOME, runs itself again once with HOME set to its working directory, the run's
  workdir (lassi_ttmlir_run.cpp, lassiDefaultHome; SUPERSEDED on 2026-11-01 by the owner's answer to OQ-115: the
  executor sets HOME and the re-exec is gone, see the part dated 2026-11-01 at the end of the section on the TTIR and D2M arms). The sandbox still passes no HOME and nothing from the caller's
  environment reaches the program; the ttsim row's "HOME unset" stays true of the environment the executor sets, but
  the driver's process holds HOME=<workdir>. That is a design point for the owner's review (Execution Backends; Agent
  Rule 12's intent is kept: no host HOME and no secret).
- Diagnosis in the sandbox: gdb starts about one worker thread per host CPU (256 here), past the sandbox's TasksMax of
  256 (`vfork: Resource temporarily unavailable`), and even with a larger cap cannot read the process's memory, since
  /proc is read-only there; `env LD_DEBUG=files <program>` worked.
- The ttsim descriptor (flatc of ttsim.ttsys): version 0.0.2995, ttmlir_git_hash 5f396fd6, product_identifier unknown;
  one chip: grid_size y 9, x 8; coord_translation_offsets 18, 18; l1_size 1499136; 12 DRAM channels of 1073741824
  bytes; l1_unreserved_base 103712 (the mock descriptor's is 1024, plans/spikes/ttir-d2m.md section 1);
  erisc_l1_unreserved_base 69632; dram_unreserved_base 32; dram_unreserved_end 1073050400; alignments: NoC L1 16, PCIe
  32, NoC DRAM 32; the 13 supported data types the runtime lists (runtime/lib/common/system_desc.cpp:179-186); 6 tile
  sizes; dst_physical_size_tiles 16, num_cbs 64; chip_capabilities HostMMIO; chip_channels empty. The run's log names
  UMD's simulation device ("Creating Simulation device", EmulationDriver PCI vendor_id 0x1e52 device_id 0x401e).
- The descriptor check compares what ttrt compares (tools/ttrt/common/util.py:868-907): the flatbuffer's
  system_desc table as JSON (Binary::getSystemDescAsJson) against the recorded root's system_desc member, both
  minified; the design note's "equals asJson()" would have compared a table with its root. A flatbuffer compiled for
  this descriptor passed it on both runtimes.
- TT-Metal (add.mlir, f32 64x128, `--ttir-to-ttmetal-pipeline="system-desc-path=<descriptor>
  enable-multicast-inference=false"`, then `--ttmetal-to-flatbuffer`): compile 0.1 s and 0.02 s; head
  5cc20000 20000000 54544d30 ("TTM0"); the run exited 0 after 4.5 s simulator wall time with no finding and wrote c,
  f32 [64,128]. Against numpy 2.1.2's f32 sum of the same inputs (seed 7, uniform [-1, 1)): max |difference|
  1.4484e-3, 8190 of 8192 elements differ, sums 65.80548 and 65.85814; the outputs carry about 10 mantissa bits
  (0.278564453125 = 1141 x 2^-12 where numpy gives 0.27873552), consistent with f32 operands reduced to TF32 in the
  source registers (inferred, not traced).
- TT-NN histo (histo.mlir: one ttir.scatter, SUM, si32 4096 zeros, 16384 indices in [0, 4096), 16384 ones;
  `--ttir-to-ttnn-backend-pipeline="system-desc-path=<descriptor>"`, then `--ttnn-to-flatbuffer`): compiled with one
  warning, `Empty argument type map provided. Skipping argument type population.`; the lowering splits the scatter into
  64 ttnn.scatter of 256 indices each (128 ttnn.slice_static, 130 ttnn.to_layout, 324 ttnn.deallocate); head
  ... 54544e4e ("TTNN"). The run exited 1 after 1.9 s with one ttsim finding, `[10918] ERROR: UndefinedBehavior:
  tensix_execute_unpacr: unpack_to_dst=0 in_data_format=8`, and wrote no output, so nothing was compared. The JIT
  cache holds only tilize_wh, reader_unary_pad_multicore_both_dims, and writer_unary_interleaved_start_id: the run
  stopped in the input conversion, before any scatter. The runtime tilizes an INT32 or FLOAT32 input on the device
  (runtime/lib/ttnn/utils/utils.cpp:76-84, :98-103; runtime/lib/ttnn/types/layout_converter.cpp:108-114), and
  ttsim calls an unpack of a 32-bit format to the source registers undefined (data format 8 is Int32 in tt-metal's
  DataFormat, tt_metal/api/tt-metalium/tt_backend_api_types.hpp:21-43; that ttsim prints the same codes is inferred).
- TT-NN f32 add (add.mlir through the TT-NN pipeline): the same stop, `UndefinedBehavior: tensix_execute_unpacr:
  unpack_to_dst=0 in_data_format=0 out_data_format=0` (0 is Float32), exit 1 after 2.8 s.
- TT-NN bf16 add (add_bf16.mlir): exit 0 after 3.3 s with no finding; c, bf16 [64,128]. Against numpy's f32 sum of the
  bf16 inputs widened exactly: max |difference| 0.00390625, half a bf16 step in [1, 2), 4036 of 8192 elements differ,
  consistent with the sum rounded to bf16 (inferred; not compared element by element with round-to-nearest-even).

Finding: the driver and the ttmlir-ttsim executor run TT-Metal ("TTM0") and TT-NN ("TTNN") flatbuffers on ttsim, a
program compiled for the recorded ttsim descriptor, with lassi_io inputs and outputs (exploratory). Route A's first
measurement did not reach ttnn.scatter: on ttsim v1.3.4 every TT-NN program with a 32-bit input stops at the
runtime's on-device tilize of that input with ttsim's UndefinedBehavior finding, before the program's own ops run
(inferred: two programs were run, histo with si32 inputs and add with f32 inputs, and the runtime code cited above
tilizes every INT32 or FLOAT32 input on the device; no other 32-bit program was run).
Whether ttnn.scatter, sort, moreh_cumsum, and embedding run on ttsim stays unmeasured. What could move it, each a
choice for the owner (OQ-112's fallback names route C when TT-NN does not run on ttsim): a ttsim release that takes
the 32-bit unpack (a pin change); a host-side tilize of 32-bit inputs (the pinned runtime offers none through its API;
a runtime patch, or the driver calling TT-NN's host conversion itself, linking _ttnncpp.so); 16-bit index and value
types where exact (lassi_io has no u16, and counts past 256 are not exact in bf16); or route C, whose TT-Metal path ran
here.

Not shown here: a run from a clean commit (the descriptor's sha256 in toolchains/tt-mlir-driver.pin comes from rx ddac
and rx 655c and is to be confirmed so; the runs above used driver dc422de5d697, whose program has the bytes of the
pinned 52325796c083's), the TTIR and D2M toolchains compiling with the descriptor (SUPERSEDED by the next section:
when this was written, lassi.toolchains.ttir_compile still used the compiler install and the mock descriptor and
declared the compiler pin under tt-mlir; the ttsim presets ttir-ttmetal-ttsim and d2m-ttmetal-ttsim now compile with
the runtime prefix's tools for the recorded descriptor under the runtime pin), and the
older driver prefixes fb17bf849a92, 5e91d6a405ff, and dc422de5d697, left on alpha01 (never deleted; 84 to 88 KiB
each).

## The TTIR and D2M arms on ttsim, end to end (2026-10-30, exploratory)

Exploratory, never [MEASURED]: every run below came from a dirty worktree (detached at 40bd4a8, the tip of
p16-adversarial, with the driver and executor work above uncommitted; rx sent each tree as a snapshot commit), on
alpha01 under the owner's standing permission for short CPU-only runs: `rx run` or `rx exec`, each under
`--timeout 600` or less, CPU only, no `--big`, no model (the replay backend gives hand-written replies), no RNGD card
touched. Every flatbuffer ran through the ttmlir-ttsim executor in the sandbox with TT_METAL_SIMULATOR on ttsim v1.3.4,
one device open per run process; no card was opened. rx ids use alpha01's clock (UTC-7, 2026-10-29 evening there);
`<stamp>-...-<tag>` stands for `<stamp>-desktop-8r113ei-detached-40bd4a88-<tag>`. The scratch disk read 68G available
(`df -BG /mnt/nvme10`) before the first run and before each run after it.

What changed in the repository (uncommitted, in the worktree):

- Pins. The executor declares the runtime pin under its own name, tt-mlir-runtime (it declared it as tt-mlir, the
  compiler pin's name, so the runner refused the pair). lassi.core.runner pin_field records a pin under the field of
  its name, and a build variant kept in its own pin file, <NAME>-<BUILD>.pin with those NAME and BUILD keys, under
  NAME's field: tt-mlir-runtime as tt_mlir, 0.9.0.dev20260221-runtime, as the runtime pin's header and the executor
  intended. Two pin names that would put two versions in one field are refused before any directory exists, so the
  runtime executor beside a compile-only toolchain (compiler pin, mock descriptor) is refused; the Trial record's
  fields are unchanged.
- Toolchains. ttir-ttmetal-ttsim and d2m-ttmetal-ttsim (lassi.toolchains.ttir_compile) are ttir-ttmetal-compile and
  d2m-ttmetal-compile with the runtime build's ttmlir-opt and ttmlir-translate (PIN and TOOL_PINS tt-mlir-runtime,
  the executor's pin) and `--ttir-to-ttmetal-pipeline=system-desc-path=<descriptor> enable-multicast-inference=false`,
  the descriptor at <toolchains root>/<driver PREFIX_NAME>/ttsim.ttsys; check_tool_pin reads the runtime's install
  record and the descriptor's sha256 against toolchains/tt-mlir-driver.pin before any process starts. The compile-only
  presets and sweep13, sweep15, and sweep16 are unchanged.
- Outputs. binary_io takes an optional recipe key `shape`: `exact` (the default, the rule before) or `elements`, which
  compares two arrays of one dtype whose shapes differ but hold the same element count in row-major order, as lassi_io
  stores them; another count is still a shape mismatch, its note giving both counts. The driver writes each output in
  its tensor's logical shape, so a module that writes a 65536-element array as 256x256 writes c as [256, 256] where the
  tt program writes [65536] (read below).
- Reference. As the ttkernel recipes have it (mlir1, sweep10): baseline builds and runs the item's tt program, the ttir
  and d2m languages' reference (`reference: tt`), on ttsim, and the cpp program natively (fixes.baseline_both); the
  oracle judges each candidate's outputs against the tt program's within the references' agreement (threshold
  from_baseline). The tt program, not the cpp program, is the target reference, so a candidate that computes what the
  device computes (a bf16 add) matches it exactly.
- Prompts. tt-ttir-bare-v1 is tt-ttir-bare-v0 with correct.txt worded for run failures too (how the program ended and
  its standard error, the driver's refusals among them) and a bare correct_mismatch.txt (the per-output report, no
  guidance); tt-d2m-bare-v1 is the same for D2M, with tt-d2m-v0's reference, retrieval, and cgeist templates.
- Recipes. projects/lassi-demo/sweep17/gpt-oss-120b: ttir-min (sweep16's ttir-min plus the run), ttir-full (plus rung
  l2's retrieval and rung l3's cgeist source), and d2m-min (rung l1's features for D2M), each in a max_abs half
  (eltwise_binary, eltwise_sfpu; d2m adds loopback) and a pcc half (matmul_multi_core), every one with run_loop and
  loop.correct_on_mismatch within the cap of 8. projects/lassi-demo/sweep17/check: four replay recipes on SYNTHETIC
  recordings (tests/fixtures/ttmlir-ttsim/) of the hand-written modules in this directory (eltwise_binary_ok,
  _wrong, _refused, eltwise_sfpu_ok, matmul_multi_core_ok, d2m_eltwise_sfpu_ok); summarize.py prints a run's trials.

| rx id | What | Result |
| --- | --- | --- |
| 20261029-221047-exec-6e66 | Read-only: df, the toolchains root, the runtime's record, the driver prefix, the descriptor | 68G available; record `tt-mlir 5f396fd6... 0.9.0.dev20260221 runtime`; ttsim.ttsys 552 bytes, sha256 54679eed...e302, the pin's; bench sources of tt-pairs-v0-ttir and -d2m fetched |
| 20261029-221124-...-f7cd | The runtime build's ttmlir-opt and ttmlir-translate on eltwise_binary_ok and _wrong for the descriptor | both TTM0 (size prefixes 74788 and 75404 bytes), 5 ttmetal.enqueue_program each; ttmlir-opt --version `LLVM version 22.0.0git`, `Optimized build with assertions` |
| 20261029-221408-...-2d69 | `uv run lassi --graphics off run projects/lassi-demo/sweep17/check/ttir-eltwise-replay.yaml --run-id ttir-ttsim-check-1` (eltwise-binary-wrong-then-ok.json, loop.max_corrections 1) | rc 0; `stage S5  corrections 1  wall_s 26.42` |
| 20261029-221446-exec-779c, 221501-exec-0a0c, 221533-...-7d88 | Reads of ttir-ttsim-check-1 (221516-...-ee40 named a wrong blob path) | below |
| 20261029-221627-...-8be5 | the same recipe on eltwise-binary-check.json, loop.max_corrections 2, `--run-id ttir-ttsim-check-2` | rc 0; `stage S5  corrections 2  wall_s 28.80` |
| 20261029-221707-exec-90de | Read of ttir-ttsim-check-2 | below |
| 20261029-221908-...-dc61 | The runtime build's tools on eltwise_sfpu_ok, matmul_multi_core_ok, d2m_eltwise_sfpu_ok | TTM0 each; 3, 4, and 3 programs |
| id not returned (the ssh session dropped; the runs' provenance names snapshot 718f18cb) | ttir-sfpu-replay, ttir-matmul-replay, d2m-sfpu-replay, run ids ttsim-check-<name>-1 | all three finished (rx 20261029-222117-exec-3982, -222206-exec-9ca3) |
| 20261029-222151-...-e1da | summarize.py on the three | below |

Results (simulator wall time of each run, never performance; Agent Rule 2):

- ttir-ttsim-check-2, eltwise_binary, one trial of three replies. Trial pins tt_mlir 0.9.0.dev20260221-runtime, ttsim
  v1.3.4; device `ttsim v1.3.4 (libttsim_wh.so, a virtual Wormhole) on the host CPU, with the tt-mlir runtime
  0.9.0.dev20260221-runtime (tt-mlir 5f396fd6, tt-metal 5280a9cf) and lassi-ttmlir-run 52325796c083: a simulator, not
  silicon`. The tt reference ran clean in 2.11 s; the references' agreement on c: pcc 0.99999885, max_abs 0.001953125
  (the threshold). Attempt 0 (128x256 inputs): exit status 2 in 0.42 s, S4, diagnostics `driver-refused: the driver
  refused the run: input 0 (@inputs/a.lassiio) holds 65536 elements; the program's input 0 [128x256] holds 32768` and
  `run-error: the program exited with status 2`; the correction prompt (correct.txt) carried `the program exited with
  status 2; its standard error follows:` and the driver's line. Attempt 1 (a - b): clean in 5.46 s, S5, alignment 0.0:
  `c: pcc -0.0015, max_abs 2.00390625, passed False, note max_abs 2.00390625 is above the max_abs threshold
  0.001953125`, diagnostic `oracle-mismatch`; the mismatch prompt read `c: failed; pcc -0.001; over 1000 times the
  allowed error`. Attempt 2 (a + b widened to f32): clean in 5.45 s, S5, alignment 1.0: `c: pcc 1.0, max_abs 0.0,
  max_ulp 0, passed True`. Final: S5, alignment 1.0, corrections 2, end_reason none. ttir-ttsim-check-1 read the same
  for its two replies (attempt 0 a - b failed with the same statistics; attempt 1 passed, max_abs 0.0; corrections 1).
- The output shapes (lassi.harness.lassi_io read_array on the blob store): the tt reference's c f32 (65536,), each
  attempt's c f32 (256, 256); with `shape: exact` every attempt would have failed as a shape mismatch.
- ttsim-check-ttir-sfpu-1, eltwise_sfpu (one ttir.exp): tt reference clean in 2.14 s, result bf16 (65536,); agreement
  pcc 0.99995954, max_abs 0.015625; attempt 0 clean in 3.84 s, result bf16 (256, 256), pcc 1.0, max_abs 0.0, passed.
- ttsim-check-ttir-matmul-1, matmul_multi_core (one ttir.matmul, metric pcc): tt reference clean in 3.19 s, c bf16
  (640, 640); agreement pcc 0.99994025, max_abs 0.25; attempt 0 clean in 6.71 s, c bf16 (640, 640), pcc 1.0, max_abs
  0.0, passed.
- ttsim-check-d2m-sfpu-1, eltwise_sfpu through d2m-ttmetal-ttsim (one generic of d2m.tile_exp on an 8x8 grid): tt
  reference clean in 2.08 s; attempt 0 clean in 3.80 s, result bf16 (256, 256), pcc 1.0, max_abs 0.0, passed.
- No run printed a ttsim finding (sim_ub false, no gap), and no run wrote a file into its workdir but its outputs.

Finding (exploratory): the TTIR and D2M TT-Metal path runs end to end through the runner on ttsim: the ttsim preset
compiles for the recorded descriptor with the runtime's tools, the driver runs the flatbuffer, the oracle reads the
outputs against the tt program's, and run_loop feeds a driver refusal and an oracle mismatch back as corrections and
stops at a pass. Hand-written modules of eltwise_binary, eltwise_sfpu, and matmul_multi_core in TTIR, and eltwise_sfpu
in D2M, equal the tt program's outputs bit for bit on ttsim.

Not shown here: any sweep17 arm with the model (no model was served); verbatim_context, retrieval, and source_mlir on
these recipes (unchanged features of sweep15 and sweep16; the check recipes leave them out); loopback and
matmul_multi_core in D2M; a run from a clean commit; and the TT-NN route (the 32-bit unpack finding above holds).

For the owner:

- The driver's HOME workaround (answered: OQ-115 chose (b); see the part dated 2026-11-01 at the end of the section on the TTIR and D2M arms). TT-NN's static configuration in _ttnncpp.so builds its cache paths from
  getenv("HOME") with no check, and libTTMLIRRuntime links _ttnncpp.so whatever runtime runs, so a process with no HOME
  dies by SIGSEGV while its libraries load, before main (above). The ttsim row's environment passes no HOME (bible,
  Execution Backends), and the sandbox passes nothing from the caller, so the driver re-executes itself once, from its
  preinit function, with HOME set to its working directory, the run's workdir inside the sandbox
  (lassi_ttmlir_run.cpp, lassiDefaultHome); it is kept, and every run above went through it. No host HOME and no
  secret reaches the program (Agent Rule 12's intent), and in the runs above nothing was written under that HOME (each
  run's only new workdir files were its outputs), but the program's process holds a HOME that the recorded
  environment does not show. The choice: (a) keep it in the driver and name it in the ttsim row as the ttmlir-ttsim
  executor's exception; (b) have the executor set HOME=<workdir> for this row in the environment it records, and drop
  the re-exec from the driver (a change to the ttsim row and a driver rebuild, a new driver VERSION); or (c) patch the
  runtime build so TT-NN's configuration tolerates no HOME (a pin change and a rebuild). Recommendation: (b), so the
  recorded environment says what the program sees and the driver holds no environment logic; (a) stands until then.
- The record field (answered: OQ-115 chose to keep tt_mlir). The runtime build is recorded as tt_mlir, 0.9.0.dev20260221-runtime (pin_field), with no change
  to the Trial record; a field of its own (tt_mlir_runtime) would let one run record both builds, at the cost of a
  record schema change (bible, the Trial record; Parquet and corpus columns).
- The threshold (answered: OQ-115 chose the item's tolerance; see the part dated 2026-11-01 at the end of the section on the TTIR and D2M arms). The arms judged
  against the references' agreement (from_baseline), as the ttkernel recipes do; for
  eltwise_binary that is 0.001953125 (half a bf16 step in [0.5, 1)), far inside the item's 1e-2, so a module that
  adds in f32 on the device (about 10 mantissa bits, above) may fail where the item's tolerance would pass it.
- Output dtypes in the prompt (answered: OQ-115 chose to leave them out). The bare prompts show each output's name alone ("- output c"); eltwise_binary's c is f32 in both references (the
  cpp program's comment and write call say so), so a module that returns bf16 reads "dtype mismatch" until it widens.
- The bible (done: two Decision Log rows dated 2026-10-30, with the driver bullet, the ttmlir-ttsim row, binary_io's
  shape and from_item, and the sandbox's HOME rule).

### 2026-11-01: the owner's answers to OQ-115 applied, and the replays again (exploratory)

Exploratory, never [MEASURED]: every run below came from a dirty worktree (detached at 40bd4a8, the work above still
uncommitted; rx sent each tree as a snapshot commit), on alpha01 under the owner's standing permission for short
CPU-only runs: `rx run` or `rx exec`, each under `--timeout 600` or less, CPU only, no `--big`, no model (the replay
backend gives the hand-written SYNTHETIC replies of tests/fixtures/ttmlir-ttsim/), no RNGD or Tenstorrent card
touched. Every flatbuffer ran through the ttmlir-ttsim executor in the sandbox with TT_METAL_SIMULATOR on ttsim
v1.3.4, one device open per run process. rx ids use alpha01's clock (UTC-7); `<stamp>-...-<tag>` stands for
`<stamp>-desktop-8r113ei-detached-40bd4a88-<tag>`. The scratch disk read 66G available (`df -BG /mnt/nvme10`) before
the first run and after the last.

What changed (uncommitted, in the worktree):

- Threshold (the owner's "Item tolerance"). No recipe value bound the item's declared tolerance in the oracle stage
  (the baseline judged the references' agreement by it, and the adversary stage bound it under from_baseline), so
  binary_io gained one generic choice: `metric: from_item, threshold: from_item` (the two go together), which judges
  each output against the item's declared tolerance in the suite manifest. The Oracle protocol gained
  tolerance_setting beside agreement_setting; lassi.core.stages agreement_bound binds such an oracle with
  with_tolerance to the item's tolerance in the oracle stage and in run_loop's correct_on_mismatch, the adversary
  stage binds it as under from_baseline, and the runner refuses it, before any directory exists, for an item that
  declares no tolerance. Since each item now carries its own metric and threshold, the max_abs and pcc halves were
  merged: sweep17/gpt-oss-120b holds ttir-min, ttir-full, and d2m-min, each over its items (ttir: eltwise_binary
  max_abs 0.01, eltwise_sfpu max_abs 0.05, matmul_multi_core pcc 0.97; d2m adds loopback max_abs 0), and the check
  recipes use the same oracle. Tests: tests/oracles/test_binary_io_from_item.py, tests/core/test_binary_io_baseline.py
  (a candidate past the references' agreement but within the item's tolerance passes under from_item and fails under
  from_baseline; the run refusal), tests/core/test_sweep17_recipes.py.
- HOME (the owner's "Executor sets it"). The ttmlir-ttsim executor passes HOME=<workdir>/@ttsim/home in the sandbox
  environment it records, a reserved directory made before the run and never collected as an output (files under
  @ttsim are not outputs); the sandbox admits HOME only for a program's run, as an absolute path at or under that
  run's workdir with no `..` part, and refuses it in a compile environment. The driver's .preinit_array re-exec
  (lassiDefaultHome) is removed; toolchains/tt-mlir-driver.pin: SOURCE_SHA256
  f9375024160a7ad20b3e4fbd641b7e4b68806db917d7bc9b446eacc3eacea795, VERSION a90585ca29fa, PREFIX_NAME
  lassi-ttmlir-driver@a90585ca29fa, a new prefix; the older prefixes stay on alpha01, untouched.

| rx id | What | Result |
| --- | --- | --- |
| 20261101-140651-exec-ea43 | Read-only: `df -BG /mnt/nvme10`, the new prefix | 66G available; no prefix lassi-ttmlir-driver@a90585ca29fa yet |
| 20261101-140709-...-0514 | `LASSI_ALLOWED_ROOT=/mnt/nvme10/joseph_ufl bash toolchains/tt-mlir-driver.sh` | rc 0; bin/lassi-ttmlir-run sha256 e04bcf102f856741e1aa37e1913a7ac40a9de45189dcecd668bdd9adb94d64cb |
| 20261101-141227-...-f60a | `uv run tools/ttmlir_system_desc.py --install`; diag.py usage and refuse without and with the executor's HOME | describe exit 0 after 2.0 s simulator wall time; 552 bytes, sha256 54679eed...e302 (the pin's), installed in the new prefix; without HOME both exit 139 (SIGSEGV) in 0.4 s, the driver no longer working around it; with HOME both exit 2 with their refusal lines |
| 20261101-141243-exec-f133 | Read-only: df, the describe run's @ttsim/home, the install record | 66G; @ttsim/home empty; record `lassi-ttmlir-driver a90585ca29fa f9375024... de51ccc4... tt-mlir@0.9.0.dev20260221-runtime` |
| 20261101-141959-exec-89ab | Read-only: `df -BG /mnt/nvme10` | 66G available |
| 20261101-142021-...-f37f | For each n of ttir-eltwise, ttir-sfpu, ttir-matmul, d2m-sfpu: `uv run lassi --graphics off run projects/lassi-demo/sweep17/check/<n>-replay.yaml --run-id item-check-<n>-1`; then summarize.py on each run; then df | rc 0 for all four; 66G after |
| 20261101-142217-exec-1c16 | Read-only: the driver each trial's device names, the resolved recipe | lassi-ttmlir-run a90585ca29fa in all four; recipe.resolved.yaml holds from_item |

Results (simulator wall time of each run, never performance; Agent Rule 2):

- item-check-ttir-eltwise-1, eltwise_binary (item tolerance max_abs 0.01), the three replies of
  eltwise-binary-check.json: the tt reference clean in 2.03 s, c f32 (65536,); the references' agreement on c: pcc
  0.99999885, max_abs 0.001953125, passed (by the item's tolerance). Attempt 0 (128x256 inputs): exit 2 in 0.42 s,
  S4, `driver-refused: ... input 0 (@inputs/a.lassiio) holds 65536 elements; the program's input 0 [128x256] holds
  32768` and `run-error: the program exited with status 2`. Attempt 1 (a - b): clean in 5.40 s, S5, alignment 0.0:
  `c: pcc -0.0015, max_abs 2.00390625, passed False, note max_abs 2.00390625 is above the max_abs threshold 0.01`
  (the item's threshold, where the run of 2026-10-30 read 0.001953125, the agreement), diagnostic `oracle-mismatch
  ... c (about 100 to 1000 times the allowed error)`. Attempt 2 (a + b widened to f32): clean in 5.35 s, S5,
  alignment 1.0: `c: pcc 1.0, max_abs 0.0, max_ulp 0, passed True`. Final: S5, alignment 1.0, corrections 2,
  end_reason none.
- item-check-ttir-sfpu-1, eltwise_sfpu (max_abs 0.05): tt reference clean in 1.98 s; agreement pcc 0.99995954,
  max_abs 0.015625, passed; attempt 0 clean in 3.89 s, result bf16 (256, 256), pcc 1.0, max_abs 0.0, passed; final S5,
  alignment 1.0.
- item-check-ttir-matmul-1, matmul_multi_core (pcc 0.97): tt reference clean in 3.10 s, c bf16 (640, 640); agreement
  pcc 0.99994025, max_abs 0.25, passed; attempt 0 clean in 6.66 s, pcc 1.0, max_abs 0.0, passed; final S5, alignment
  1.0.
- item-check-d2m-sfpu-1, eltwise_sfpu through d2m-ttmetal-ttsim (max_abs 0.05): tt reference clean in 2.08 s;
  agreement pcc 0.99995954, max_abs 0.015625, passed; attempt 0 clean in 3.79 s, result bf16 (256, 256), pcc 1.0,
  max_abs 0.0, passed; final S5, alignment 1.0.
- No run printed a ttsim finding (sim_ub false, no gap).

Finding (exploratory): with driver a90585ca29fa, HOME set by the executor, and the item-tolerance oracle, the four
replays read as before: the refused run and the wrong module fail, and each correct module passes; the wrong
module's note now names the item's threshold, 0.01.

Not shown here: a run from a clean commit (the descriptor's hash stays from exploratory runs until one confirms it);
any sweep17 arm with the model; loopback and matmul_multi_core in D2M; a module that passes only under the item's
tolerance on ttsim (the f32 add of the first section, max_abs 1.4484e-3, would by its numbers; not rerun here).

## Open questions

- The free-space floor, LASSI_FREE_FLOOR_GIB=30 on the proposed command line, is a choice for the owner: the disk is
  shared, and the value sets both the start condition (60 GiB free with the 30 GiB plan) and the stop line of every build.
- The stop line LASSI_STOP_GIB=345 follows the lift of OQ-074; the owner may name another.
- TT_RUNTIME_DEBUG (CMakeLists.txt:60) would compile in the runtime's address checks (runtime/include/tt/runtime/debug.h:24-39)
  at some cost; the pin keeps upstream's default, off, and relies on the driver's descriptor check.
- The fetched-sources list of the first install is copied into the tracked list and committed by the main session, as for
  the compiler install.
