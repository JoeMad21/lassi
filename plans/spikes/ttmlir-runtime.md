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

## Open questions

- The free-space floor, LASSI_FREE_FLOOR_GIB=30 on the proposed command line, is a choice for the owner: the disk is
  shared, and the value sets both the start condition (60 GiB free with the 30 GiB plan) and the stop line of every build.
- The stop line LASSI_STOP_GIB=345 follows the lift of OQ-074; the owner may name another.
- TT_RUNTIME_DEBUG (CMakeLists.txt:60) would compile in the runtime's address checks (runtime/include/tt/runtime/debug.h:24-39)
  at some cost; the pin keeps upstream's default, off, and relies on the driver's descriptor check.
- The fetched-sources list of the first install is copied into the tracked list and committed by the main session, as for
  the compiler install.
