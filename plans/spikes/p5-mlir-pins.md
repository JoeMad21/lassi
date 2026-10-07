# Spike P5.1: Polygeist and tt-mlir pins, builds, and budget

- Task: P5.1 (plans/p5-ir.md, planning decision "One LLVM or two (P5.1)"). Bible: Toolchain Pins, IR Strategy (cpu-mlir, ttkernel, and llvm-ir rows), Level Rules, Language Frontends (C and C++ rows), Host Facts (scratch cap, existing assets), Risks And Questions (scratch cap; toolchain churn), Agent Rules 7 and 10.
- Date: 2026-10-07 on the workstation. Read-only probes of alpha01 ran from 21:48 to 22:03 on 2026-10-06 by the alpha01 clock (UTC-07:00), that is 04:48 to 05:03 UTC on 2026-10-07; each rx id carries its start time. Upstream facts were read the same session through the GitHub API (`gh api`, authenticated read-only) and from source tarballs unpacked on the workstation.
- Base: branch p5-ir at 51bf083, bible mirror at master revision 266. Every remote probe was `uv run tools/rx.py doctor` or a read-only `uv run tools/rx.py exec`, which runs from the scratch root without a checkout. Nothing was configured, built, installed, written, or deleted on alpha01, and no job ran.
- Device: none. No probe opened a device or named a device tool.
- Labels: (host) marks a value read on alpha01 by a read-only rx exec, cited by its rx id; (source) a value read from upstream source at a pinned commit, cited as path:line; (metadata) a value from the GitHub API; (CI) an upstream CI record, which is not a measurement of ours. Every size and time of a build that has not run is PROJECTED, with its basis. Nothing here is [MEASURED], since no run came from a clean commit.
- Output conventions: outputs are verbatim except where `[... trimmed ...]` says otherwise. No upstream file content beyond names, versions, hashes, and short quoted lines enters this file (OQ-018 practice).

## Polygeist pin rule (fixed before any candidate was read)

This section was written to this file before any Polygeist commit, its submodule pointer, or its CI was read, as the P4 plan fixed the joint-pin rule before P4.1 looked. The rule is the one P5.1's acceptance states:

> The Polygeist main commit whose llvm-project submodule commit is nearest tt-mlir's LLVM 4efe170d in llvm-project's history (fewest commits apart, the newer on a tie), unless upstream CI, issues, or source show a commit building against 4efe170d itself, which is then preferred.

Reading fixed here, before any candidate:

1. Candidates are the commits on Polygeist's default branch (llvm/Polygeist, branch main) as GitHub serves it on the spike's date. Each candidate's LLVM is the gitlink of its llvm-project submodule at that commit.
2. Distance is the count of commits between that gitlink and 4efe170d858eb54432f520abb4e7f0086236748b in llvm-project's history, read from GitHub's compare API: ahead_by plus behind_by of `compare/<gitlink>...4efe170d`. When one is an ancestor of the other this is the plain commit count between them.
3. Ties go to the newer Polygeist commit (by committer date on main).
4. The preference clause needs positive evidence that a Polygeist commit builds against 4efe170d itself: a CI run, an issue or pull request reporting a build, or a gitlink equal to 4efe170d. Reasoning that a build "should work" is not that evidence.
5. If no evidence of the clause exists, the rule's nearest commit is the pin, and the choice of one LLVM or two follows from whether that commit's gitlink is 4efe170d (one LLVM possible) or not (two stacks, each with its own LLVM, per Toolchain Pins).

## Question

What P5.4, P5.5, and P5.8 build on, as P5.1's acceptance asks:

1. Polygeist: the commit the rule above selects, its llvm-project commit, the LLVM projects and targets cgeist needs, whether it builds against tt-mlir's LLVM 4efe170d, and so one LLVM or two stacks.
2. tt-mlir at the joint pin (5f396fd6): what env/CMakeLists.txt builds beside LLVM and what can be off; the configuration that gives ttmlir-opt and ttmlir-translate with the runtime off; whether that needs Python 3.11; the packages its configure fetches; TTMLIR_TOOLCHAIN_DIR and any other default path outside the scratch root; and what its ttkernel-to-EmitC conversion handles of what the raiser emits (scf, arith, memref with globals, func.call).
3. The clang the TT raiser links, and what a Metalium kernel at tt-metal 5280a9cf needs to parse on the host.
4. Host prerequisites present or missing, with a user-space route for each missing one.
5. The budget for both stacks together against the scratch root's du and the 115G stop line.

Why it matters: P5.4 and P5.5 install what this spike pins, and they start OWNER on OQ-049 unless the projection for both stacks together fits under the stop line; P5.6, P5.7, and P5.10 run the pinned tools; P5.8 builds the raiser against the clang chosen here.

Classification: factual. Every part is answered from upstream source at a pinned commit, GitHub metadata, upstream CI records, and read-only probes of alpha01. Sizes and wall times of builds that have not run are PROJECTED.

## Host state

- `uv run tools/rx.py doctor` (no rx id), 2026-10-06 about 21:48 on the alpha01 clock: stop false, scratch_free_gb 394.8, root_free_gb 224.9, nproc 256, mem_avail_gb 2895.1, load 22.9 / 29.2 / 37.3, config max_big_jobs 1 and max_build_jobs 32, devices_enabled tt_silicon false, rocm_gpu false, nvidia_gpu false, rngd true, running_jobs empty, seven slots.
- rx 20261006-214855-exec-9935 (host):

```
2026-10-06T21:48:55-07:00
102646204	/mnt/nvme10/joseph_ufl
98G	/mnt/nvme10/joseph_ufl
/dev/nvme25n1p1  3.5T  3.1T  368G  90% /mnt/nvme10
--- top entries: 83
--- toolchains
233596	/mnt/nvme10/joseph_ufl/toolchains/cuda@12.6.3
12844208	/mnt/nvme10/joseph_ufl/toolchains/nvhpc@24.11
3157352	/mnt/nvme10/joseph_ufl/toolchains/tt-metal@5280a9cf
172	/mnt/nvme10/joseph_ufl/toolchains/ttsim@v1.3.4
--- env
TMPDIR=/mnt/nvme10/joseph_ufl/tmp LASSI_TOOLCHAINS=/mnt/nvme10/joseph_ufl/toolchains LASSI_JOBS=32 LASSI_SCRATCH=/mnt/nvme10/joseph_ufl HOME=/mnt/nvme10/joseph_ufl XDG_CACHE_HOME=/mnt/nvme10/joseph_ufl/.cache PIP_CACHE_DIR=/mnt/nvme10/joseph_ufl/.cache/pip UV_CACHE_DIR=/mnt/nvme10/joseph_ufl/.cache/uv
```

- The scratch root holds 102,646,204 KiB (97.9 GiB). The stop line, 115G, is 120,586,240 KiB, so 17,940,036 KiB (17.1 GiB) remain under it, and 23,182,916 KiB (22.1 GiB) under the 120G cap (125,829,120 KiB).
- rx 20261006-215953-exec-0b88 (host), the folders OQ-049 names and the caches:

```
2026-10-06T21:59:53-07:00
18870648	amd
7300128	cuda-12.6.3
28588884	.cache
262892	tmp
544712	lassi-runs
79624	lassi-wt
--- .cache
23205380	.cache/huggingface
4109000	.cache/furiosa
1231588	.cache/uv
41244	.cache/tt-metal-cache
1648	.cache/tiktoken
12	.cache/Microsoft
8	.cache/git
```

  amd/ and cuda-12.6.3/ together hold 26,170,776 KiB (25.0 GiB). No pip cache directory exists yet under .cache.
- One probe was refused before it ran: a `du` whose argument list named the gate's own directory ("gate refused: command refused by policy pattern /lassi-gate/"). It was rerun without that directory (rx 20261006-215953-exec-0b88). It was not a device need, so no owner-queue item follows.

## Results

### 1. Polygeist: the pin, its LLVM, and one LLVM or two

Candidates (metadata, 2026-10-07). llvm/Polygeist's default branch is main; its head is 77c04bb2a7a2406ca9480bcc9e729b07d2c8d077 (2024-07-31, "Implement builtin functions __builtin_ctz{s,,l,ll} (#424)"). `gh api "repos/llvm/Polygeist/commits?path=llvm-project&sha=main" --paginate` lists 68 main commits that moved the llvm-project submodule, the newest 50d4f21d6db736a5aa5528fa4c1cf2fef61d6408 (2023-10-06, "Bump LLVM Version"). Reading the gitlink at each of them (`contents/llvm-project?ref=<commit>`) gives 65 distinct llvm-project commits. Each was compared with 4efe170d by `gh api repos/llvm/llvm-project/compare/<gitlink>...4efe170d858eb54432f520abb4e7f0086236748b`:

```
26eb4285b56edd8c897642078d91f16ff0fd3472 ahead ahead=78791 behind=0
cbc378ecb87e3f31dd5aff91f2a621d500640412 ahead ahead=115231 behind=0
4e43a14bdbe1d3ae57701aa6d280fef46a6ea14b ahead ahead=117373 behind=0
3262794804ad23ed4a511669ffc97d128512bc37 ahead ahead=119710 behind=0
00a12585933ef63ff1204bf5cd265f0071d04642 ahead ahead=130834 behind=0
[... trimmed: 46 more, each farther ...]
```

- 51 of the 65 are in llvm-project and are ancestors of 4efe170d (behind 0). The other 14 answer 404 there (commits outside llvm-project's history); every Polygeist commit that set one of them is dated 2022-01-12 or earlier, and the rule measures distance in llvm-project's history, so they have none.
- The nearest is 26eb4285b56edd8c897642078d91f16ff0fd3472 (2023-09-28, "[MLIR][LLVM] Add vararg support in LLVM::CallOp and InvokeOp (#67274)"), 78,791 commits behind 4efe170d (dated 2025-10-06). Its LLVM version is 18.0.0git (llvm/CMakeLists.txt at 26eb4285: LLVM_VERSION_MAJOR 18, MINOR 0, PATCH 0, SUFFIX git); 4efe170d's is 22.0.0git (cmake/Modules/LLVMVersion.cmake).
- Ties: 50d4f21 set 26eb4285, no later main commit moved the submodule, and `compare/50d4f21...77c04bb` reads ahead 263, behind 0, so 264 main commits carry 26eb4285 at equal distance, and the rule takes the newest, the head 77c04bb. The gitlink at main reads 26eb4285 (`contents/llvm-project?ref=main`).
- The preference clause: there is no evidence of any Polygeist commit building against 4efe170d. Of Polygeist's 78 branches the newest tip is `sql` (2025-06-19), before 4efe170d's date, and it carries 26eb4285. The six pull requests updated since 2025-10-06 (412, 441, 443, 444, 445, 446) carry 26eb4285 at their heads; none is merged and none changes the submodule. Polygeist's build workflow (.github/workflows/build.yml) builds the submodule's own commit (it reads src/.git/modules/llvm-project/HEAD), and every one of its retained runs (128, all pull-request runs, 2025-10-15 to 2026-09-11) ran on a head that carries 26eb4285 (`contents/llvm-project?ref=<head>` for each). None of the 14 issues and pull requests opened since 2025-01-01 reports a build at another LLVM.

Pin by the rule: Polygeist 77c04bb2a7a2406ca9480bcc9e729b07d2c8d077, llvm-project 26eb4285b56edd8c897642078d91f16ff0fd3472.

Does it build against 4efe170d? No, as it stands (source; issues):

- Polygeist at 77c04bb includes two headers that llvm-project removed before 4efe170d. Every `#include "mlir/..."`, `"clang/..."`, and `"llvm/..."` in its lib/, include/, tools/cgeist/, and tools/polygeist-opt/ (205 headers; generated .inc files and llvm/Config left out) was looked up at 4efe170d; two are missing there and present at 26eb4285:

```
MISSING clang/Parse/ParseDiagnostic.h   included at tools/cgeist/Lib/pragmaHandler.cc:17
  removed by llvm-project 834dfd231553 (2024-11-18) "[Parse] Remove ParseDiagnostic.h (#116496)"
MISSING mlir/Support/MathExtras.h       included at lib/polygeist/Passes/CollectKernelStatistics.cpp:12,
                                        ConvertParallelToGPU.cpp:26, ParallelLoopUnroll.cpp:23
  removed by llvm-project 0fb216fb2fbb (2024-06-11) "mlir/MathExtras: consolidate with llvm/MathExtras (#95087)"
```

  Header presence is a lower bound on the breakage; API changes over 78,791 commits were not surveyed.
- Issue 433 (2025-01-06, "CollectKernelStatistics.cpp:39:22: error: 'ceilDiv' is not a member of 'mlir'"): a maintainer answered "Polygeist does not support any other version" than the submodule's commit (2025-01-07). Issue 440 (2025-10-27, "IS this project not being actively maintained?"): a Polygeist author answered on 2025-11-11 that development moved to EnzymeAD/Enzyme-JaX and that this repository's passes are to be updated from there; nothing has been merged to main since 2024-07-31.
- Upstream CI does build main's code at 26eb4285: pull request 446's run 33356614241 (2026-08-31; its head changes only tools/cgeist/Lib/clang-mlir.cc) passed all its jobs, `ninja check-polygeist-opt` and `ninja check-cgeist` included (job 99379904188: Passed 86, Expectedly Failed 3; Passed 164, Expectedly Failed 10). These are upstream CI results, not measurements of ours.

So one LLVM cannot serve both stacks: two stacks, each with its own LLVM, CPU modules checked with Polygeist's LLVM tools and TT modules with tt-mlir's (Toolchain Pins; the P5 plan's Pins constraint). Making Polygeist build at 4efe170d would be a port of Polygeist, which is outside P5's scope.

What cgeist needs (source, Polygeist at 77c04bb):

- LLVM projects mlir and clang, host target: README.md:24-28 (`-DLLVM_ENABLE_PROJECTS="mlir;clang" -DLLVM_TARGETS_TO_BUILD="host" -DLLVM_ENABLE_ASSERTIONS=ON`); CMakeLists.txt:54 and :58 (`find_package(MLIR REQUIRED CONFIG)`, `find_package(Clang REQUIRED CONFIG)`). README.md:89-95 also gives a single in-tree build (`-DLLVM_EXTERNAL_PROJECTS="polygeist"`), which puts LLVM, clang, MLIR, and Polygeist in one build tree. Polly is needed only for Polymer with ISL (README.md:37), and CUDA, ROCm, and Polymer are off by default (CMakeLists.txt:5-10).
- Upstream CI builds that LLVM with `-DLLVM_ENABLE_PROJECTS="llvm;clang;mlir;polly"`, all targets, Release, and /bin/clang (build.yml:91), then Polygeist, and runs `ninja check-polygeist-opt` and `ninja check-cgeist` (build.yml:112-114).
- P5's tools from that stack: cgeist, and that LLVM's mlir-opt, mlir-translate, and clang (P5.5).
- cgeist finds clang's builtin headers from its own executable path when no `-resource-dir` is given (tools/cgeist/Lib/clang-mlir.cc:6027-6030), so cgeist must sit where that LLVM's lib/clang/<version>/include is found, as in the in-tree build, or be passed `-resource-dir`.
- Two default paths outside the scratch root: POLYGEIST_PGO_DEFAULT_DATA_DIR is /var/tmp/polygeist/pgo/ (CMakeLists.txt:44), read by the PGO passes (lib/polygeist/Passes/ConvertPolygeistToLLVM.cpp:1853, LowerAlternatives.cpp:54); and when cgeist emits a binary it writes LLVM IR to a temporary file named /tmp/intermediate%%%%%%%.ll (tools/cgeist/driver.cc:1201-1202). P5's route stops cgeist at MLIR and lowers with mlir-opt, mlir-translate, and clang (IR Strategy, cpu-mlir row), so it never takes that path; P5.5 sets the PGO directory under the scratch root anyway. /var/tmp/polygeist does not exist on alpha01 (rx 20261006-220320-exec-8e14; its output is not quoted in this file). /tmp was not probed; cgeist's file there is a temporary name that P5's route never creates.

### 2. tt-mlir at the joint pin (5f396fd6)

Source: tt-mlir at 5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c (its tarball through `gh api repos/tenstorrent/tt-mlir/tarball/<commit>`, read on the workstation). The LLVM part of env/CMakeLists.txt there matches main's today except that main added `-DLLVM_TARGETS_TO_BUILD=host` and a pip constraints file and dropped `pip install --force-reinstall 'nanobind==2.10.2'` from the PATCH_COMMAND (:55) (a diff of the two files).

What env/CMakeLists.txt builds, and what can be off:

| Part | env/CMakeLists.txt | Switch |
| --- | --- | --- |
| Toolchain directory | read from env/activate (:12-25); FATAL if it does not exist (:27-29) | TTMLIR_TOOLCHAIN_DIR, default /opt/ttmlir-toolchain (env/activate:1-4) |
| Python venv | target python-venv, ALL (:31): env/init_venv.sh makes a venv with python3.12 by default (init_venv.sh:10), pip-installs env/build-requirements.txt, env/ttnn-requirements.txt, and test/python/requirements.txt (torch 2.7.0 among them), then runs env/install-tt-triage.sh, which downloads tt-metal's scripts/install_debugger.sh at the pin and runs it (init_venv.sh:28-34) | none |
| flatbuffers fb9afbafc7dfe226b9db54d4923bfb8839635274 | git, Release, installed into the toolchain directory (:4, :33-46) | none |
| llvm-project 4efe170d | git (ExternalProject with a commit as GIT_TAG, so a full clone); its PATCH_COMMAND pip-installs mlir/python/requirements.txt and nanobind 2.10.2 and applies env/patches/affine-allow-symbol-vars.patch (:55); MinSizeRel by default (:8); projects mlir and lld (:64); LLVM_INSTALL_UTILS, LLVM_INSTALL_GTEST, assertions, and MLIR Python bindings on (:65-68); all targets (no LLVM_TARGETS_TO_BUILD) | TTMLIR_BUILD_LLVM (:48) |
| stablehlo 0a4440a5, shardy edfd6730 | git sources only, under the toolchain directory, shardy with two patches (:81-100) | none |

So only the LLVM build can be switched off; the venv (with its tt-metal debugger install), flatbuffers, stablehlo, and shardy always run. A minimal route therefore does not run env/CMakeLists.txt: it scripts the two parts the main build needs, flatbuffers at fb9afbaf and LLVM at 4efe170d with the affine patch (git blob 2ffdae14d2bcb8244189d8eb044dd093642fa436 at the pin and on main; sha256 61e8ae545e956551b87dc7fb63067d570035becbcf0636ce81ca0971e30f3aac of the raw file, workstation), and points tt-mlir at them.

What the main build needs (tt-mlir's CMakeLists.txt and modules):

- An activated environment: TTMLIR_ENV_ACTIVATED (CMakeLists.txt:12-14) and TTMLIR_TOOLCHAIN_DIR (:98-100) set in the environment; Python3_EXECUTABLE is forced to $TTMLIR_VENV_DIR/bin/python3 (:134), and TTMLIR_VENV_DIR defaults to $TTMLIR_TOOLCHAIN_DIR/venv (env/activate:11-14). env/activate also sets TT_METAL_RUNTIME_ROOT and TT_METAL_HOME to tt-mlir's third_party checkout (env/activate:26-27), so P5.4 sets the variables itself rather than sourcing it.
- MLIR and LLD (cmake/modules/FindMLIR.cmake:2-3); lib/Target/LLVM links lldELF and the host's X86 libraries (lib/Target/LLVM/CMakeLists.txt:14, :37). No other LLVM target is linked, and tt-mlir main has built its LLVM with the host target alone since 2026-04-04 (pull request 7130).
- flatc and the flatbuffers library (cmake/modules/BuildFlatbuffers.cmake:1; lib/CMakeLists.txt:111).
- Defaults: the compilers `clang` and `clang++` when none is given (CMakeLists.txt:3-8); -Werror with -Wall -Wextra -Wpedantic (:107); ld.lld-<clang major> when found (:110-125). Upstream CI builds with clang-17 (.github/Dockerfile.base installs it with llvm.sh 17 and links clang to clang-17), and docs/src/getting-started.md:18 asks for Clang 14 to 18.
- Options, and the configuration for ttmlir-opt and ttmlir-translate with the runtime off: TTMLIR_ENABLE_RUNTIME and TTMLIR_ENABLE_OPMODEL default OFF (CMakeLists.txt:33, :39), and with both off third_party/CMakeLists.txt declares only empty interface libraries for tt-metal (:87, :323-334), so no tt-metal is fetched or built (this settles the "For P5" concern of plans/spikes/p4-tt-pins.md for runtime-off builds). With TTMLIR_ENABLE_BINDINGS_PYTHON OFF (default ON, :84), python/ is skipped (:200-204) and tools/ skips builder, golden, profiler, explorer, and ttrt (tools/CMakeLists.txt:6-27); TTMLIR_ENABLE_TESTS OFF skips test/ (:205-207); tt-alchemist needs the runtime (tools/CMakeLists.txt:35-37). ttmlir-opt and ttmlir-translate build under TTMLIR_ENABLE_TOOLS (default ON; tools/CMakeLists.txt:1-4, :29-32) and link TTMLIRCompilerStatic. Building only the targets ttmlir-opt and ttmlir-translate leaves out the shared TTMLIRCompiler (lib/CMakeLists.txt:93), ttmlir-lsp-server, and ttnn-standalone.
- Two configure-time steps that no option turns off:
  - docs/ is added unconditionally (CMakeLists.txt:210). docs/CMakeLists.txt runs `${Python_EXECUTABLE} -m pip install sphinx sphinx-markdown-builder` at configure time (:8-14), unpinned, and stops the configure with FATAL_ERROR unless sphinx-build and sphinx-apidoc are found (:17-33). So the venv needs those two packages, installed at pinned versions by P5.4's script, or the configure fails.
  - cmake/modules/TTMLIRVersion.cmake lists the tags matching v[0-9]*.[0-9]* (:10-14) and, if there are none, adds an `upstream` remote and runs `git fetch --tags upstream` (:21-40); it then takes `git describe --tags --match v[0-9]*.[0-9]* --abbrev=0` (:44-49) and passes the result to `string(REGEX MATCH ...)` (:60). tt-mlir's only such tag is v0.0 (commit a637748e, 2995 commits before 5f396fd6 by the compare API). In a depth-1 clone, describe finds no tag in the history, the result is empty, and line 60's command loses its input argument, which CMake rejects (read from source, not run). So P5.4 clones tt-mlir with its history (GitHub repository size 96,514 KB, metadata), the tag is local, and no fetch happens at configure.
- Python: with the bindings off nothing in the main build needs Python 3.11 or 3.12. Python runs tools/scripts/sha256-include-gen.py for the flatbuffers schemas (BuildFlatbuffers.cmake:62; standard library only) and `python -c "from distutils import sysconfig; ..."` (cmake/modules/TTMLIRPythonSitePackages.cmake:1-3), which Python 3.10 has (3.12 removed distutils); LLVM 4efe170d asks for Python 3.8 or newer (llvm/CMakeLists.txt:1019). So a venv of the host's Python 3.10.12 serves, with sphinx and sphinx-markdown-builder (read from source; P5.4's configure is the check). The cp311 ttmlir wheel is not used (the P5 plan's constraint). If a later step needs 3.11, uv can install a managed CPython under the scratch root, where HOME and UV_CACHE_DIR point; none is installed now (rx 20261006-214920-exec-4a93).

Packages fetched, by route:

| Route | What it downloads |
| --- | --- |
| env/CMakeLists.txt as upstream runs it | flatbuffers, llvm-project (full clone), stablehlo, and shardy (git); pip: build-requirements, ttnn-requirements, test/python/requirements (torch 2.7.0 from the PyTorch CPU index), mlir/python/requirements.txt, and nanobind 2.10.2; tt-metal's install_debugger.sh, ttexalens_ref.txt, and tools/triage/requirements.txt at 5280a9cf, then the debugger it installs |
| The minimal route (P5.4) | flatbuffers at fb9afbaf and llvm-project at 4efe170d (git, by commit); tt-mlir at 5f396fd6 with its history; pip: sphinx and sphinx-markdown-builder at pinned versions |
| tt-mlir's main configure with the runtime, OpModel, bindings, explorer, and runtime tests off | nothing by CMake: its only other downloads are tt-metal (third_party/CMakeLists.txt:103-131, with the runtime or OpModel), model-explorer (tools/explorer/CMakeLists.txt:8-11, with the bindings and explorer), and googletest (runtime/test/CMakeLists.txt:7-9, with the runtime tests); plus the docs pip install and, without a local v-tag, the tag fetch above |

Default paths outside the scratch root: /opt/ttmlir-toolchain (env/activate:1-4), with the venv inside it (env/activate:11-14) and stablehlo and shardy under it (env/CMakeLists.txt:82); the docs ask for `sudo mkdir -p` there (docs/src/getting-started.md:82-87), which P5.4 never does. /opt/ttmlir-toolchain does not exist on alpha01 (rx 20261006-220320-exec-8e14), so a configure that left TTMLIR_TOOLCHAIN_DIR unset would stop at env/CMakeLists.txt:27-29 rather than use another user's toolchain. LLVM's CMAKE_INSTALL_PREFIX defaults to /usr/local, so P5.4 builds in place, or sets the prefix, and never runs a bare install. The CPM cache default sits in tt-mlir's source tree (third_party/CMakeLists.txt:9-13) and is used only with the runtime.

ttmlir-opt's version: tools/ttmlir-opt/ttmlir-opt.cpp:21-22 calls MlirOptMain and registers no version printer of its own, so `--version` prints MLIR's default banner, which names the LLVM, not tt-mlir's commit. P5.4 records what it prints as EXPECT_VERSION; tt-mlir's commit lives in the pin file.

What the ttkernel-to-EmitC conversion handles (source; lib/Conversion/TTKernelToEmitC/TTKernelToEmitC.cpp at the pin, and the upstream conversions at 4efe170d):

- It converts only functions that carry the ttkernel thread attribute (:949-952), one function at a time (`applyFullConversion(funcOp, ...)`, :1244), with arith, scf, memref, and ttkernel illegal and emitc and func legal (:955-960), so a func.call stays as it is.
- scf: upstream SCFToEmitC at 4efe170d converts scf.for, scf.if, and scf.index_switch (mlir/lib/Conversion/SCFToEmitC/SCFToEmitC.cpp:341-343); scf.while has no pattern there, so a raised while loop fails the full conversion.
- arith: upstream ArithToEmitC at 4efe170d (ArithToEmitC.cpp:806-850: constant; add, sub, mul, div, and rem; the bitwise and shift ops; cmpi, cmpf, negf, and select; the integer and float casts) plus tt-mlir's floordivsi, bitcast, maxui, and minui patterns (TTKernelToEmitC.cpp:791-930). Not in either list (known cases, not complete): arith.maxsi, arith.minsi, arith.ceildivsi, and arith.remf.
- memref: upstream MemRefToEmitC at 4efe170d converts alloca, alloc, copy, global, get_global, load, and store (MemRefToEmitC.cpp:63-410). Inside a kernel function memref.get_global converts, but memref.global sits at module level, outside the function the pass converts, so it stays a memref op.
- Translation: tt-mlir's own translation is `ttmlir-translate --ttkernel-to-cpp` (lib/Target/TTKernel/TTKernelToCppRegistration.cpp:22-23), which its tests use (test/ttmlir/Translate/TTKernel/*.mlir). For each thread-attributed function it clones only that function's body into a new module, as kernel_main, with the thread's includes (lib/Target/TTKernel/TTKernelToCpp.cpp:361-386, :46-70), and prints that module. Module-level globals and other functions are not cloned, so a constant table held as memref.global and a func.call to a helper in the same module do not reach the emitted C++. Upstream `--mlir-to-cpp` is also registered (ttmlir-translate.cpp:46, registerAllTranslations) and prints a whole module, but adds none of the kernel includes and needs the globals and helpers already in emitc form; the upstream passes convert-memref-to-emitc, convert-func-to-emitc, convert-arith-to-emitc, and convert-scf-to-emitc exist at 4efe170d (mlir/include/mlir/Conversion/Passes.td:857, :437, :167, :1107), and ttmlir-opt registers all upstream passes, `inline` among them (tools/ttmlir-opt/ttmlir-opt.cpp:14).
- Tests at the pin: none of test/ttmlir/Conversion/TTKernelToEmitC/*.mlir or test/ttmlir/Translate/TTKernel/*.mlir holds an scf op, a memref op, or a call (0 of each by grep); ttkernel.mlir holds 215 lines with arith ops. So scf, memref, and calls inside kernels are untested there. Elsewhere, the ttir-to-ttmetal pipeline runs the conversion after lower-affine (lib/Dialect/TTMetal/Pipelines/TTMetalPipelines.cpp:198-203, :251), and test/ttmlir/Conversion/D2MToTTKernel/use_tile_matmul_*.mlir run that pipeline, so kernels with scf loops do pass through the conversion in tests.

So, read from source and not run: the raiser's scf.for and scf.if, the arith ops above, and memref loads and stores in the kernel function convert; a while loop, a module-level constant table, and a helper function reached by func.call do not reach the C++ through `--ttkernel-to-cpp` as they stand. Two forms would: helpers inlined into the kernel function (`--inline`) with constant tables kept inside it, or upstream's module-level conversions followed by `--mlir-to-cpp` with the includes added by P5's target. P5.4's remote test, which the plan already gives a global and a func.call, decides which.

### 3. The clang the TT raiser links, and what a kernel needs to parse

Host clang development libraries (rx 20261006-214936-exec-a225, `dpkg-query -W`; lines reordered, some trimmed as marked, and the two cmake directory listings annotated):

```
dpkg-query: no packages found matching libclang-20-dev
dpkg-query: no packages found matching libclang-cpp20-dev
dpkg-query: no packages found matching libclang-17-dev
dpkg-query: no packages found matching libclang-cpp17-dev
libclang-common-17-dev 1:17.0.6~++20231209124227+6009708b4367-1~exp1~20231209124336.77 ii
libclang-common-20-dev 1:20.1.8~++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132 ii
libclang-cpp20 1:20.1.8~++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132 ii
llvm-17-dev 1:17.0.6~++20231209124227+6009708b4367-1~exp1~20231209124336.77 ii
llvm-20-dev  un
--- cmake dirs
clang          (ls /usr/lib/llvm-20/lib/cmake)
clang llvm     (ls /usr/lib/llvm-17/lib/cmake)
ls: cannot access '/usr/lib/llvm-20/include/clang': No such file or directory
ls: cannot access '/usr/lib/llvm-20/lib/libclangTooling.a': No such file or directory
/usr/lib/llvm-20/lib/libclang-cpp.so.20.1
[... trimmed: the libmlir-20-dev, mlir-20-tools, and flatbuffers package lines, the llvm-20 include/clang-c and llvm-17 include/clang listings, and the libLLVM.so line ...]
```

The clang-20 runtime library is there, but no clang headers, no clangTooling, and no LLVM 20 development package: the host's clang-20 development libraries are missing, and so are clang-17's. A user-space route exists in principle (download the matching apt.llvm.org packages and unpack them with `dpkg-deb -x` under $LASSI_TOOLCHAINS); whether that repository still serves build 20.1.8~++20250708082409 was not checked.

Choice: the clang of Polygeist's LLVM, llvm-project 26eb4285 (clang 18.0.0git). P5.5 builds it in any case, its static clang libraries (clangTooling among them) sit in that build tree (and in P5.5's install, which keeps them if OQ-049's answer lets the job remove its build tree; Results 5), and the raiser links no MLIR library (planning decision), so it needs no LLVM-scale build of its own and pins to an LLVM commit P5 already pins. tt-mlir's LLVM builds no clang (env/CMakeLists.txt:64), and adding clang to it would change upstream's configuration of the TT stack. The raiser's build therefore follows P5.5's.

What a Metalium kernel at tt-metal 5280a9cf needs to parse (source, tt-metal at 5280a9cfb00998fd49667a29523d03aee905c129):

- Language: the JIT compiles every kernel with `-std=c++17 -flto=auto -ffast-math -fno-exceptions` (tt_metal/jit_build/build.cpp:158), using sfpi's riscv-tt-elf-g++ (build.cpp:140-154).
- Include directories, relative to the tree: `.`, `..`, the root, ttnn, ttnn/cpp, tt_metal, tt_metal/hw/inc, tt_metal/hostdevcommon/api, and tt_metal/api/ (build.cpp:280-296); on Wormhole also tt_metal/hw/ckernels/wormhole_b0/metal/common and metal/llk_io, tt_metal/hw/inc/internal/tt-1xx, tt-1xx/wormhole, wormhole/wormhole_b0_defines, and wormhole/noc, and tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc and llk_lib; for compute kernels tt_metal/hw/ckernels/wormhole_b0/metal/llk_api and llk_api/llk_sfpu; for Tensix kernels tt_metal/hw/firmware/src/tt-1xx (tt_metal/llrt/hal/tt-1xx/wormhole/wh_hal.cpp:100-133). Compute kernels also get sfpi's include directory (build.cpp:147, :336), and every kernel its own source directory (build.cpp:466-469).
- Defines: the device's NUM_DRAM_BANKS, NUM_L1_BANKS, LOG_BASE_2_OF_NUM_DRAM_BANKS and LOG_BASE_2_OF_NUM_L1_BANKS (or IS_NOT_POW2_NUM_DRAM_BANKS and IS_NOT_POW2_NUM_L1_BANKS), PCIE_NOC_X, and PCIE_NOC_Y (tt_metal/jit_build/build_env_manager.cpp:69-104); TENSIX_FIRMWARE and LOCAL_MEM_EN=0 (build.cpp:183); PROCESSOR_INDEX, then COMPILE_FOR_BRISC or COMPILE_FOR_NCRISC for data movement, or for compute UCK_CHLKC_UNPACK, UCK_CHLKC_MATH, or UCK_CHLKC_PACK with NAMESPACE set to chlkc_unpack, chlkc_math, or chlkc_pack and COMPILE_FOR_TRISC set to 0, 1, or 2 (tt_metal/llrt/hal/tt-1xx/hal_1xx_common.cpp:18-49); ARCH_WORMHOLE (wh_hal.cpp:146); DISPATCH_MESSAGE_ADDR and KERNEL_BUILD (build.cpp:359-369); and the kernel's own defines, KERNEL_COMPILE_TIME_ARGS, and KERNEL_COMPILE_TIME_ARG_MAP (build.cpp:420-450). Debug features add more (build.cpp:185-276), none in a plain ttsim run.
- Generated headers: a compute kernel is compiled three times through generated files (chlkc_unpack.cpp, chlkc_math.cpp, chlkc_pack.cpp, defines_generated.h, and the chlkc_* data-format and tile descriptors; tt_metal/jit_build/genfiles.cpp:132-236, :373-480), which a host parse must supply.
- Capturing the exact lines: TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1 (tt_metal/llrt/rtoptions.cpp:615) makes the JIT log each kernel's full compile command (build.cpp:473-475), so one ttsim run of a kernel under the ttsim row's settings gives its includes, its defines, and the generated directory.
- Parsing for the device's target: clang's RISC-V target information is compiled whatever LLVM_TARGETS_TO_BUILD holds (at 26eb4285, clang/lib/Basic/CMakeLists.txt:111 lists Targets/RISCV.cpp unconditionally and clang/lib/Basic/Targets.cpp:431 handles riscv32), so a host-target build's clang can run `-fsyntax-only` for riscv32. The JIT passes `-mcpu=tt-wh-tensix` to compute Tensix kernels and `-mcpu=tt-wh` to the others (wh_hal.cpp:164-167); both are GCC-only, and the raiser drops both.
- A known gap: sfpi 7.25.0's include/sfpi_builtins.h maps SFPU calls onto GCC builtins named `__builtin_rvtt_*` (47 lines), which only sfpi's GCC provides (GitHub code search finds no rvtt builtin in llvm/llvm-project; its three hits are SPIR-V target files). A compute kernel whose includes reach sfpi.h therefore needs declarations of those builtins from the raiser to parse. Known cases, not complete; P5.8 counts the rest per kernel.

### 4. Host prerequisites

rx 20261006-214920-exec-4a93 (tools) and rx 20261006-214936-exec-a225 (packages):

```
cmake: /usr/local/bin/cmake /usr/bin/cmake /bin/cmake
ninja: /usr/bin/ninja /bin/ninja
clang:
clang-17: /usr/bin/clang-17 /bin/clang-17
clang-20: /usr/bin/clang-20 /bin/clang-20
ld.lld:
ld.lld-17: /usr/bin/ld.lld-17 /bin/ld.lld-17
ld.lld-20: /usr/bin/ld.lld-20 /bin/ld.lld-20
ld.mold:
python3.10: /usr/bin/python3.10 /bin/python3.10
python3.11:
python3.12:
flatc:
ccache:
--- versions
cmake version 3.31.6
cmake version 4.2.1
1.10.1
Ubuntu clang version 17.0.6 (++20231209124227+6009708b4367-1~exp1~20231209124336.77)
Ubuntu clang version 20.1.8 (++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132)
Ubuntu LLD 17.0.6 (compatible with GNU linkers)
Ubuntu LLD 20.1.8 (compatible with GNU linkers)
Python 3.10.12
--- uv pythons
ls: cannot access '/mnt/nvme10/joseph_ufl/.local/share/uv/python/*': No such file or directory
[... trimmed: the clang++-17, clang++-20, gcc-12, g++-12, python3, and git lines, all present ...]
```

| Need | alpha01 | Route if missing |
| --- | --- | --- |
| CMake: 3.24 for tt-mlir (CMakeLists.txt:1); 3.20 for both LLVMs and the env | /usr/local/bin/cmake 3.31.6, first on PATH; /usr/bin/cmake 4.2.1 | Present. P5.4 and P5.5 name CMake by path rather than rely on PATH order inside a job |
| Ninja | /usr/bin/ninja 1.10.1 | Present. tt-mlir's docs tie a stale-file error with ninja 1.10 to the Python C API target (getting-started.md:193-204), which the bindings-off build does not have |
| C and C++ compilers | clang-17 17.0.6, clang-20 20.1.8, gcc-12 (12.3.0 is the libstdc++-12-dev package version, rx 20261006-214936-exec-a225); no unversioned clang or clang++ | Present. tt-mlir's CI compiler is clang-17; pass it by name, since tt-mlir defaults to `clang` (CMakeLists.txt:3-8) |
| Linker | ld.lld-17 17.0.6, ld.lld-20 20.1.8, GNU ld; no ld.mold and no unversioned ld.lld | Present. tt-mlir finds ld.lld-17 for clang-17 by name (CMakeLists.txt:110-125); an LLVM build that asks for lld names the versioned linker |
| Python | 3.10.12 with python3-venv, python3.10-venv, and python3-dev; no 3.11 or 3.12; uv at /mnt/nvme10/joseph_ufl/bin/uv | Present for the bindings-off route (Results 2). If 3.11 is ever needed: a uv-managed CPython under the scratch root (not done) |
| flatc and the flatbuffers library | Missing (no flatc; flatbuffers-compiler and libflatbuffers-dev not installed) | Build flatbuffers fb9afbaf from source with CMake into $LASSI_TOOLCHAINS (the env's own step, env/CMakeLists.txt:33-46), in user space |
| sphinx and sphinx-markdown-builder (tt-mlir's configure) | Not on the host | pip into P5.4's venv at pinned versions; PIP_CACHE_DIR already points to the scratch root |
| clang development libraries (the raiser) | Missing for 17 and 20 (Results 3) | Not needed: the raiser links Polygeist's LLVM build |
| zlib, zstd, libxml2, libffi, and ncurses development packages | zlib1g-dev, libzstd-dev 1.4.8, libxml2-dev, libffi-dev, and libncurses-dev installed; libedit-dev not installed | Present; libedit is optional for LLVM (left off) |
| git | 2.34.1 | Present (tt-mlir's configure runs git) |
| ccache | Missing | Not needed |

### 5. The budget

Basis (metadata and CI; upstream figures, not measurements of ours):

- tt-mlir pull request 7130 (commit 8026be2f8e, 2026-04-04), on tt-mlir's LLVM at 4efe170d with mlir and lld, MinSizeRel, assertions, and Python bindings, moving from all targets to the host target: "lowers build time of the whole environment from ~55 min to ~42 min locally on my machine. Also it lowers the size of the LLVM build directory from 14 GB to 11 GB". The machine, and whether "build directory" includes the source clone, are not stated; reading it as the build tree alone is the larger reading.
- tt-mlir CI, run 35540812010, job 106158084635 (2026-09-20; tt-mlir main, the same LLVM commit and patch, host target, MinSizeRel, bindings on): the environment step took 1811.6 s at CMAKE_BUILD_PARALLEL_LEVEL=12, of which the LLVM build ran 5585 Ninja steps from 661.1 s to 1804.0 s; the installed toolchain directory then read `5.1G /opt/ttmlir-toolchain` (bin 2.4G, lib 822M, venv 1.4G, python_packages 225M, include 139M, src 139M; `du -h --max-depth=2`). Job 108717221223 of run 36353714567 (2026-09-27) built the same environment step in 1121.2 s on a 64-CPU builder.
- tt-mlir CI, run 36353714567, job 108722691617 ("release-build / Build speedy Release", 2026-09-27; 64 CPUs, CMAKE_BUILD_PARALLEL_LEVEL=32, clang-20, Release): tt-mlir main with the runtime, OpModel, StableHLO, the bindings, and the runtime tests on built 1582 Ninja steps in its Build step, 22:33:10 to 22:42:56 UTC (9 min 46 s), with a ccache restored from an earlier run (1253 hits of 2007 cacheable calls over the whole job). It built far more than P5.4 will, tt-metal among it, but reused cached objects, so it gives a scale for tt-mlir's own build, not a bound. It is the only tt-mlir build timing read.
- The pinned release's wheel, ttmlir-0.9.0.dev20260221-cp311-cp311-manylinux_2_34_x86_64.whl (64,837,089 bytes, the GitHub release asset; its member list read on the workstation): its two largest members, ttmlir/_mlir_libs/_ttmlir.cpython-311-x86_64-linux-gnu.so (164,187,752 bytes) and ttmlir/_mlir_libs/libTTMLIRPythonCAPI.so (67,870,592 bytes), together carry tt-mlir's compiler and the upstream MLIR it links.
- Polygeist CI, run 33356614241 (2026-08-31): the LLVM build at 26eb4285 (llvm, clang, mlir, and polly; all targets; Release) ran 6915 Ninja steps in 1 h 26 min (job 99379904188) to 2 h 43 min (job 99379904185) on GitHub-hosted ubuntu-22.04 runners, and its zstd cache archive was 1,490,835,178 bytes; Polygeist itself, built with `ninja -j1` and then tested, took 5 to 11 min.
- Source trees (GitHub tree API, apparent bytes of all blobs): llvm-project at 4efe170d is 1,933,027,917 bytes in 164,148 files, or 1,251,064,094 bytes in 85,568 files for llvm, mlir, lld, cmake, third-party, and utils; at 26eb4285 it is 1,513,866,872 bytes in 134,659 files, or 1,224,884,370 bytes in 93,842 files for llvm, clang, mlir, cmake, third-party, and utils. tt-mlir at 5f396fd6 unpacks to 33,035 KiB and Polygeist at 77c04bb to 9,821 KiB (workstation). GitHub gives repository sizes of 4,499,679 KB for llvm-project, 96,514 KB for tt-mlir, and 1,113,804 KB for Polygeist, so both llvm-project checkouts and Polygeist's are fetched at depth 1.
- File counts: tt-mlir has 272 .cpp files under lib/ (and 134 under runtime/, not built); Polygeist has 39 .cpp and .cc files in lib/, tools/cgeist, and tools/polygeist-opt, leaving out the tests under tools/cgeist/Test.

PROJECTED budget in GiB, built in place under $LASSI_TOOLCHAINS with the build trees kept (agents never delete) and no separate install: P5's tools run from the build trees, and tt-mlir and the raiser build against them. Configuration A: static libraries (the layout both upstreams build), Release (the P5 plan's choice; tt-mlir's environment defaults to MinSizeRel), assertions, the host target, Python bindings and tests off, `all` built. Configuration B is the shared-library, minimal-target candidate: BUILD_SHARED_LIBS on, only the targets P5 calls built, and sparse llvm-project checkouts. Neither upstream CI builds B, so it is a candidate only. Item rows are ranges; each stack row and the total are the exact sums of the item rows above them, and the prose rounds a sum outward to whole GiB (the low end down, the upper end up).

| Item | A | B | Basis |
| --- | --- | --- | --- |
| llvm-project 4efe170d checkout, depth 1 | 2.0 to 2.6 | 1.4 to 1.8 | tree sizes above, plus block overhead and a depth-1 pack |
| tt-mlir's LLVM build tree (mlir, lld; X86) | 9 to 14 | 5 to 9 | pull request 7130's 11 GB (host target) and 14 GB (all targets), both MinSizeRel with the bindings on. A's low end is 11 less the outputs of the bindings and of tests not built; its upper end allows about a quarter more than 11 for Release code, which is larger than MinSizeRel's (an allowance, not a measured ratio). B drops the static archives and the statically linked tools |
| flatbuffers fb9afbaf: source, build, install | 0.1 to 0.3 | 0.1 to 0.3 | repository size 22,261 KB |
| tt-mlir 5f396fd6 clone with its history | 0.1 to 0.2 | 0.1 to 0.2 | repository and tree sizes |
| tt-mlir build tree (ttmlir-opt, ttmlir-translate) | 2 to 6 | 1 to 4 | 272 library sources and the MLIR libraries they link |
| Python 3.10 venv with sphinx; pip cache | 0.05 to 0.2 | 0.05 to 0.2 | no torch or test requirements (the env route's venv read 1.4G in CI) |
| tt-mlir stack (sum of the six rows above) | 13.25 to 23.30 | 7.65 to 15.50 | |
| Polygeist 77c04bb, depth 1 | 0.01 to 0.05 | 0.01 to 0.05 | 9,821 KiB unpacked |
| llvm-project 26eb4285 checkout, depth 1 | 1.6 to 2.1 | 1.3 to 1.7 | tree sizes above |
| In-tree build of LLVM, clang, MLIR, and Polygeist (X86) | 10 to 17 | 5 to 10 | tt-mlir's LLVM row, plus clang, less an older and smaller MLIR |
| Polygeist stack (sum of the three rows above) | 11.61 to 19.15 | 6.31 to 11.75 | |
| Both stacks together (sum of the two stack rows) | 24.86 to 42.45 | 13.96 to 27.25 | about 24 to 43 (A) and 13 to 28 (B), rounded outward |
| Held for P17.11's batch if the owner approves it (OQ-047) | 2 to 3 | 2 to 3 | the P5 plan; its job, if approved, also waits its turn among P5's big jobs, one at a time |
| The TT raiser's build tree (P5.8), outside the two stacks' total | 0.1 to 0.5 | 0.05 to 0.2 | a few source files and one executable, linked statically against clang's libraries in A (about the size of clang's own binary) and dynamically in B; a placeholder until P5.8's du |

Installs, PROJECTED in GiB. In place, as above, no LLVM-scale build is installed (flatbuffers' small install is in its row). Under OQ-049's (b) or (d) each job installs, checks the install, and then removes its own build tree; its sources, the venv, and the install stay, since (b) names the build tree only.

| Build | What its install keeps | A | B | Basis |
| --- | --- | --- | --- | --- |
| tt-mlir's LLVM 4efe170d | LLVM's, MLIR's, and LLD's libraries, headers, CMake package files, and tools | 3 to 4.5 | 1 to 2 | job 106158084635's toolchain directory: 5.1G less the venv (1.4G), python_packages (225M), and src (139M, the stablehlo and shardy sources) is about 3.3G (bin 2.4G, lib 822M, include 139M) at MinSizeRel, flatbuffers' small install included. Turning the bindings off removes python_packages, already left out; A allows for Release code. B installs each library once as a shared object, so the statically linked tools in bin shrink to small executables |
| tt-mlir 5f396fd6 | ttmlir-opt and ttmlir-translate, and in B tt-mlir's own shared libraries | 0.3 to 0.6 | 0.1 to 0.3 | the pinned wheel's two shared objects above, 232 MB together for tt-mlir's compiler and the MLIR it links; in A each binary links about as much |
| LLVM 26eb4285 with clang, MLIR, and Polygeist | cgeist, polygeist-opt, mlir-opt, mlir-translate, and clang with its resource headers; for P5.8's raiser and P5.14's rebuilds of it, clang's and LLVM's libraries (static in A, shared in B), their headers, and their CMake package files | 3 to 6 | 1.5 to 3.5 | the install above (3.3G for LLVM with mlir and lld), plus clang's libraries, headers, and binary, less the tools P5 does not call and an older, smaller MLIR |
| The TT raiser (P5.8) | one executable | 0.1 to 0.3 | 0.01 to 0.1 | its build tree row; a placeholder until P5.8's du |

Under (b) or (d) the order is fixed: P5.4's jobs, then P5.5's (P5.5 depends on P5.4, and big jobs run one at a time), then P5.8's raiser build. By then P5.5's build tree is gone, so the raiser links P5.5's install, which is why that install keeps clang's and LLVM's libraries, headers, and CMake package files; P5.14 rebuilds the raiser against the same install. The space in use at each point, from the two tables (GiB, exact sums):

| Point | A | B |
| --- | --- | --- |
| P5.4's LLVM job, before it removes its build tree: the 4efe170d checkout, flatbuffers, the venv, the tt-mlir clone, the LLVM build tree, and its install | 14.25 to 21.80 | 7.65 to 13.50 |
| P5.4's tt-mlir job, before it removes its build tree: the same without the LLVM build tree, plus tt-mlir's build tree and install | 7.55 to 14.40 | 3.75 to 8.80 |
| After P5.4: the sources, the venv, and the two installs | 5.55 to 8.40 | 2.75 to 4.80 |
| P5.5's job, before it removes its build tree: the above, plus Polygeist, the 26eb4285 checkout, the in-tree build tree, and its install | 20.16 to 33.55 | 10.56 to 20.05 |
| After P5.5 | 10.16 to 16.55 | 5.56 to 10.05 |

If the owner's answer also lets a job remove its own source checkout, the figures after each job fall by that checkout's row.

Caches the builds fill: CPM none (tt-mlir with the runtime off fetches no tt-metal, and LLVM and Polygeist with Polymer off fetch nothing at configure); pip only sphinx and sphinx-markdown-builder (inside the venv row); no uv Python. Taking the env route instead would add its venv with the test requirements and python_packages (1.4G and 225M in CI) and the stablehlo and shardy sources.

Against the stop line: 17.1 GiB remain under 115G at du 102,646,204 KiB. The projected range for both stacks together does not fit. Configuration A needs 24.86 to 42.45 GiB (about 24 to 43), so both of its ends pass the stop line, the low end by about 8 GiB. Configuration B needs 13.96 to 27.25 GiB (about 13 to 28): its upper end passes, and only its low end fits, with or without the P17.11 reserve (15.96 to 16.96 GiB at that end with it, against 17.1). Verdict, under the plan's test that the projection for both stacks together fits under the stop line: it does not, so P5.4 and P5.5 stay OWNER on OQ-049.

What OQ-049's options give, by the same projection, with 2 to 3 GiB held for P17.11's batch:

- (a) frees 26,170,776 KiB (25.0 GiB) and leaves 44,110,812 KiB (42.1 GiB) under the stop line. That holds B in place (15.96 to 30.25 GiB with the reserve) but not A at its upper end (42.45 GiB without the reserve, 45.45 with it).
- (b) alone leaves 17.1 GiB, and the peak, in P5.5's job, is 20.16 to 33.55 GiB in A, which passes the stop line even at its low end, and 10.56 to 20.05 GiB in B, which fits only toward its low end. So (b) alone holds neither range.
- (d), both, holds A at its upper end: the same peak of 33.55 GiB against 39.1 GiB left after a 3 GiB reserve, and 10.16 to 16.55 GiB stay in use after P5.5 (B: 5.56 to 10.05).
- (c) would need a cap of about 150G for A's upper end in place with the reserve: du 97.9 GiB plus 45.45 GiB is about 143.3 GiB, which the stop line, 5G under the cap, must clear.

Wall time at LASSI_JOBS=32, PROJECTED, one big job at a time on a shared host (alpha01's load average read 23 to 37):

- tt-mlir's LLVM: 15 to 45 min. Basis: job 106158084635's 5585 Ninja steps, 661.1 s to 1804.0 s (about 19 min) at 12 jobs, and job 108717221223's whole environment step, 1121.2 s on 64 CPUs.
- tt-mlir (ttmlir-opt and ttmlir-translate): 10 to 30 min. Basis: job 108722691617's Build step, 9 min 46 s for 1582 steps at 32 jobs on 64 CPUs, which built the runtime and tt-metal too but with cached objects; and the 272 library sources.
- The in-tree build of LLVM, clang, MLIR, and Polygeist: 20 to 60 min. Basis: Polygeist CI's 6915 steps in 1.4 to 2.7 h on GitHub-hosted runners, for all targets and polly, and Polygeist's own `ninja -j1` build and tests in 5 to 11 min.
- The raiser: a few minutes, a placeholder (a few source files and one link against clang's libraries) until P5.8 measures it.

These are shorter than the plan's 1 to 4 hours per build; P5.4's and P5.5's jobs measure them.

## Finding

- Polygeist: 77c04bb2a7a2406ca9480bcc9e729b07d2c8d077 (main's head, 2024-07-31) with llvm-project 26eb4285b56edd8c897642078d91f16ff0fd3472 (LLVM 18.0.0git), by the rule: the nearest of main's submodule commits to 4efe170d (78,791 commits behind it), the newest of the 264 main commits that carry it, and no commit shown building against 4efe170d. cgeist needs an LLVM with clang and mlir for the host target.
- One LLVM or two: two. Polygeist includes two headers llvm-project removed before 4efe170d, and its maintainers support only the submodule's commit. CPU modules go through Polygeist's LLVM tools, TT modules through tt-mlir's.
- tt-mlir: the runtime-off build fetches and builds no tt-metal. The minimal route skips env/CMakeLists.txt, builds flatbuffers fb9afbaf and LLVM 4efe170d with tt-mlir's affine patch, then tt-mlir with the bindings, tests, and runtime off. Its configure still needs sphinx and sphinx-markdown-builder, and the v0.0 tag in the clone. Python 3.10 serves. The default toolchain directory is /opt/ttmlir-toolchain; P5 sets it under $LASSI_TOOLCHAINS.
- The conversion handles scf.for, scf.if, the listed arith ops, and memref loads and stores in the kernel function; `--ttkernel-to-cpp` drops module-level globals and helper functions, and scf.while has no pattern at 4efe170d.
- The raiser links the clang of Polygeist's LLVM; the host's clang development libraries are missing.
- Host: everything is present but flatc and the sphinx packages, both with user-space routes.
- Budget: both stacks together, built in place, take about 24 to 43 GiB static, or about 13 to 28 GiB in the shared-library candidate, PROJECTED, against 17.1 GiB under the stop line: neither range fits (the shared-library one only at its low end). Under OQ-049's (d) the static configuration peaks at 33.55 GiB, in P5.5's job, against 39.1 GiB after the reserve.
- Confidence: high for the pins, the distances, the removed headers, and the configure steps (upstream source at the pinned commits and GitHub's compare API); medium for the conversion's reach (read from source; the tests in test/ttmlir/Conversion/TTKernelToEmitC and test/ttmlir/Translate/TTKernel hold no scf, memref, or call, while the ttir-to-ttmetal pipeline tests take kernels with scf loops through the conversion); low for the sizes and times, which rest on one upstream author's local figure, upstream CI, and file counts, and stay PROJECTED until P5.4's and P5.5's du before and after.

## Consequences for the plan

- P5.4 and P5.5 stay OWNER on OQ-049; this spike's figures go to it as a Response. OQ-049's recommendation (d) is the one option that holds the static configuration at the projection's upper end.
- P5.4: script the LLVM and flatbuffers steps rather than run env/CMakeLists.txt, which would make a Python 3.12 venv with torch and run tt-metal's debugger installer; record the affine patch by its blob and sha256; clone tt-mlir with its history; give the venv pinned sphinx and sphinx-markdown-builder, with sphinx at 8.1.x or older, since PyPI lists 8.2.0 and later as needing Python 3.11 or newer (9.1.0 needs 3.12) and 8.1.3 as needing 3.10; put the venv's bin directory first on PATH, since docs/CMakeLists.txt:9 runs `${Python_EXECUTABLE}`, not the Python3_EXECUTABLE that CMakeLists.txt:134 sets, and TTMLIRPythonSitePackages.cmake:2 runs a bare `python`, or pass Python_EXECUTABLE, SPHINX_EXECUTABLE, and SPHINX_APIDOC_EXECUTABLE; install flatbuffers into TTMLIR_TOOLCHAIN_DIR (CMakeLists.txt:165 adds its include directory, and BuildFlatbuffers.cmake:1 finds flatc with find_program) or put its flatc on PATH; name CMake by path (/usr/local/bin/cmake 3.31.6 and /usr/bin/cmake 4.2.1, Results 4); set TTMLIR_ENV_ACTIVATED, TTMLIR_TOOLCHAIN_DIR, and TTMLIR_VENV_DIR under $LASSI_TOOLCHAINS and the compilers by name (clang-17 and clang++-17, as upstream CI); turn the runtime, OpModel, bindings, tests, explorer, ttrt, and alchemist off; build the targets ttmlir-opt and ttmlir-translate. Whether LLVM builds as Release (the plan's planning decision) or MinSizeRel (the env's default) is P5.4's recorded choice. Its remote conversion test should take the module with a global and a func.call both through `--ttkernel-to-cpp` and through the upstream module-level conversions with `--mlir-to-cpp`, since by source the first drops both.
- P5.5: build LLVM 26eb4285 with clang and mlir and Polygeist in one in-tree build (README.md:89-95), host target, with Polymer, CUDA, and ROCm off and POLYGEIST_PGO_DEFAULT_DATA_DIR under the scratch root, and CMake named by path; cgeist stops at MLIR, so its /tmp temporary file is never written. If OQ-049's answer lets the job remove its build tree, its install keeps clang's and LLVM's libraries, headers, and CMake package files for the raiser (Results 5). Trial.toolchain_pins needs the two LLVMs told apart (the record edit P5.5's acceptance anticipates).
- P5.8: the raiser builds against P5.5's build tree, or against P5.5's install if OQ-049's answer lets P5.5's job remove its build tree (Results 5; P5.14's rebuilds likewise), so it runs after P5.5's job (the plan's dependencies reach P5.5 through P5.6), and its build is short; the Build time constraint's "the TT raiser unless it links the host's clang-20 libraries" no longer adds an LLVM-scale build. It parses kernels for riscv32 with the flags of Results 3, takes each kernel's compile line from one ttsim run with TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1, declares sfpi's rvtt builtins itself, emits no scf.while, and keeps constant tables and helpers in a form the chosen translation route carries (P5.4's test).
- P5.10 and the bible: IR Strategy's ttkernel row names `ttmlir-translate --mlir-to-cpp`, while tt-mlir's kernel translation is `--ttkernel-to-cpp`. The row is corrected from the code when P5.10 builds the target (bible text after the code), not here.
- Toolchain Pins gains the pins with a Decision Log entry (Proposed bible edit).

## Proposed bible edit

Toolchain Pins, a new bullet before "- Pin files live in `toolchains/<name>.pin` and record commit, build flags, and install path.":

```
- MLIR stacks [DESIGN] (task P5.1, plans/spikes/p5-mlir-pins.md): two LLVMs, so CPU and TT modules stay apart, each verified by its own toolchain. Polygeist is pinned by the P5 plan's rule, the Polygeist main commit whose llvm-project submodule commit is nearest tt-mlir's LLVM 4efe170d in llvm-project's history (fewest commits apart, the newer on a tie), unless upstream CI, issues, or source show a commit building against 4efe170d itself: 77c04bb2a7a2406ca9480bcc9e729b07d2c8d077 (2024-07-31, the head of llvm/Polygeist main on 2026-10-07), whose llvm-project submodule commit is 26eb4285b56edd8c897642078d91f16ff0fd3472 (2023-09-28, LLVM 18.0.0git), an ancestor of 4efe170d and 78,791 commits behind it. Every main commit since 2023-10-06 carries that submodule commit, so the tie goes to the head, and no Polygeist branch or pull request updated since 4efe170d's date carries 4efe170d. Polygeist does not build against 4efe170d as it stands: it includes clang/Parse/ParseDiagnostic.h and mlir/Support/MathExtras.h, which llvm-project removed before 4efe170d, and its maintainers support only the submodule's commit (issue 433). Its LLVM builds clang and mlir for the host target. tt-mlir stays at the joint pin with its LLVM 4efe170d858eb54432f520abb4e7f0086236748b, to which its env/CMakeLists.txt applies env/patches/affine-allow-symbol-vars.patch (git blob 2ffdae14d2bcb8244189d8eb044dd093642fa436), and with flatbuffers fb9afbafc7dfe226b9db54d4923bfb8839635274, whose flatc tt-mlir's build needs; that LLVM builds mlir and lld, no clang. The TT raiser links the clang of Polygeist's LLVM, since the build host lacks the clang development libraries (libclang-17-dev, libclang-20-dev, libclang-cpp17-dev, and libclang-cpp20-dev are not installed; rx 20261006-214936-exec-a225). [OPEN] until tasks P5.4 and P5.5 install them: the build configurations, and the space, which P5.1 projects for both stacks together, built in place, at about 24 to 43 GiB with static libraries, or about 13 to 28 GiB with shared libraries and only the needed targets (PROJECTED), against 17.1 GiB left under the 115G stop line, 5G under the 120G scratch cap (Host Facts), at du 102,646,204 KiB on 2026-10-06 (rx 20261006-214855-exec-9935); neither range fits, so both installs wait for the owner (OQ-049). A pin change is a Decision Log entry.
```

Decision Log, a new top row; the count sentence moves from "One hundred and thirty-one ... and one on 2026-10-07" to "One hundred and thirty-two ... and two on 2026-10-07":

```
| 2026-10-07 | The pins of P5's two MLIR stacks (task P5.1; Toolchain Pins): Polygeist 77c04bb2a7a2406ca9480bcc9e729b07d2c8d077 with its llvm-project submodule commit 26eb4285b56edd8c897642078d91f16ff0fd3472, chosen by the P5 plan's rule (the Polygeist main commit whose submodule commit is nearest tt-mlir's LLVM 4efe170d in llvm-project's history, the newer on a tie, unless one is shown building against 4efe170d itself), so two LLVMs, with CPU and TT modules kept apart; tt-mlir at the joint pin with LLVM 4efe170d, tt-mlir's affine-allow-symbol-vars patch, and flatbuffers fb9afbaf; and the TT raiser linking the clang of Polygeist's LLVM. The build configurations and the space for them stay [OPEN], and tasks P5.4 and P5.5 stay held for the owner (OQ-049) | Polygeist's main has not moved its submodule since 2023-10-06; of the 51 submodule commits its main has carried that llvm-project holds, its head's 26eb4285 is the nearest, an ancestor 78,791 commits behind 4efe170d. No Polygeist branch or pull request updated since 4efe170d's date carries 4efe170d, Polygeist includes two headers llvm-project removed before it, and its maintainers support only the submodule's commit, so one LLVM cannot serve both stacks. tt-mlir's LLVM builds mlir and lld only and the host has no clang development libraries, while Polygeist's LLVM builds clang anyway, so the raiser needs no LLVM of its own. PROJECTED from upstream build figures and source sizes, both stacks together, built in place, take about 24 to 43 GiB with static libraries, or about 13 to 28 GiB with shared libraries and only the needed targets, against 17.1 GiB left under the 115G stop line at du 102,646,204 KiB (rx 20261006-214855-exec-9935); the shared-library range fits only at its low end, so neither projected range fits under the stop line before the owner's answer. plans/spikes/p5-mlir-pins.md records the commands, outputs, and sources |
```

## Owner queue

No new item. OQ-049 gains this Response:

```
Response (task P5.1, 2026-10-07): Read on alpha01: `du -sk /mnt/nvme10/joseph_ufl` read 102,646,204 KiB (98G by `du -sh`) at 2026-10-06T21:48:55-07:00 (rx 20261006-214855-exec-9935), so 17,940,036 KiB (17.1 GiB) remain under the 115G stop line (120,586,240 KiB) and 22.1 GiB under the 120G cap; amd/ holds 18,870,648 KiB (18.0 GiB) and cuda-12.6.3/ 7,300,128 KiB (7.0 GiB) (rx 20261006-215953-exec-0b88). One LLVM cannot serve both stacks: Polygeist's pin carries llvm-project 26eb4285, 78,791 commits before tt-mlir's 4efe170d, and does not build against 4efe170d, so P5 needs two LLVM-scale builds. PROJECTED budget (plans/spikes/p5-mlir-pins.md, Results 5; from tt-mlir's upstream figures of 11 GB for its LLVM build directory with the host target and 14 GB with all targets, tt-mlir pull request 7130, its CI toolchain install of 5.1G, job 106158084635, and the pinned tt-mlir release wheel; Polygeist's CI build logs; GitHub's source tree sizes; and file counts; ranges are exact sums of the spike's rows, and an 'about' range rounds one outward to whole GiB): both stacks together, built in place with their build trees kept, take 24.86 to 42.45 GiB, about 24 to 43, in the static configuration (tt-mlir stack 13.25 to 23.30, Polygeist stack 11.61 to 19.15), or 13.96 to 27.25 GiB, about 13 to 28, with shared libraries and only the needed targets, a configuration neither upstream CI builds; plus 2 to 3 GiB held for P17.11's batch if you approve it (OQ-047). Verdict: neither range fits. Both ends of the static range pass the stop line, and the shared-library range fits only at its low end, with or without the P17.11 reserve, so P5.4 and P5.5 stay OWNER on this item. By the same projection, with the reserve held: (a) alone leaves 42.1 GiB, which holds the shared-library configuration in place but not the static one at its upper end (42.45 GiB, 45.45 with the reserve). Under (b), each job installs, checks the install, and removes its own build tree, keeping its sources and install; P5.5's install then keeps clang's and LLVM's libraries, headers, and CMake files, since the TT raiser (P5.8) links it. The peak comes in P5.5's job, at 20.16 to 33.55 GiB in the static configuration and 10.56 to 20.05 GiB in the shared-library one, so (b) alone, with 17.1 GiB, holds neither range: the static one passes even at its low end, and the shared-library one fits only toward its low end. (d), the recommendation, holds the static configuration at its upper end (the 33.55 GiB peak against 39.1 GiB after a 3 GiB reserve), and 10.16 to 16.55 GiB stay in use after both jobs. (c) needs a cap of about 150G for the static upper end in place with the reserve. Nothing was built, written, or deleted on alpha01.
```

## Sources

All read on 2026-10-07 on the workstation, or from 2026-10-06 21:48 to 22:03 on the alpha01 clock (UTC-07:00).

- Polygeist: https://github.com/llvm/Polygeist (main, branches, pull requests 412, 441, 443, 444, 445, and 446, issues 433, 436, and 440, and the Actions runs and job logs of run 33356614241); at 77c04bb2a7a2406ca9480bcc9e729b07d2c8d077: CMakeLists.txt, README.md, .gitmodules, .github/workflows/build.yml, tools/cgeist/driver.cc, tools/cgeist/Lib/clang-mlir.cc, tools/cgeist/Lib/pragmaHandler.cc, lib/polygeist/Passes/*.cpp, and lib/polygeist/ExecutionEngine/PGORuntime.h; API: commits?path=llvm-project&sha=main, contents/llvm-project?ref=<commit>, and compare/50d4f21...77c04bb.
- llvm-project: compare/<gitlink>...4efe170d858eb54432f520abb4e7f0086236748b for each of the 65 gitlinks; commits 26eb4285b56edd8c897642078d91f16ff0fd3472, 4efe170d858eb54432f520abb4e7f0086236748b, 834dfd231553, and 0fb216fb2fbb; at 26eb4285: llvm/CMakeLists.txt, clang/lib/Basic/CMakeLists.txt, and clang/lib/Basic/Targets.cpp; at 4efe170d: cmake/Modules/LLVMVersion.cmake, llvm/CMakeLists.txt, mlir/lib/Conversion/MemRefToEmitC/MemRefToEmitC.cpp, mlir/lib/Conversion/SCFToEmitC/SCFToEmitC.cpp, mlir/lib/Conversion/ArithToEmitC/ArithToEmitC.cpp, and mlir/include/mlir/Conversion/Passes.td; the git trees at both commits (blob sizes); code search for rvtt.
- tt-mlir: https://github.com/tenstorrent/tt-mlir at 5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c (tarball): env/CMakeLists.txt, env/activate, env/init_venv.sh, env/build-requirements.txt, env/ttnn-requirements.txt, env/install-tt-triage.sh, env/patches/affine-allow-symbol-vars.patch, CMakeLists.txt, third_party/CMakeLists.txt, cmake/modules/FindMLIR.cmake, BuildFlatbuffers.cmake, TTMLIRVersion.cmake, TTMLIRPythonSitePackages.cmake, and ConfigureSphinx.cmake, docs/CMakeLists.txt, docs/src/getting-started.md, tools/CMakeLists.txt, tools/ttmlir-opt, tools/ttmlir-translate, tools/explorer/CMakeLists.txt, runtime/test/CMakeLists.txt, lib/CMakeLists.txt, lib/Target/LLVM/CMakeLists.txt, lib/Conversion/TTKernelToEmitC/TTKernelToEmitC.cpp, lib/Target/TTKernel/TTKernelToCpp.cpp and its registration, test/ttmlir/Conversion/TTKernelToEmitC, test/ttmlir/Translate/TTKernel, test/python/requirements.txt, .github/Dockerfile.base, and .github/Dockerfile.ci; env/CMakeLists.txt on main and its history; pull request 7130 (commit 8026be2f8e); the tags matching v (v0.0, a637748e); CI run 35540812010 (job 106158084635) and run 36353714567 (jobs 108717221223 and 108722691617, and the job metadata of 108722691617); the release asset ttmlir-0.9.0.dev20260221-cp311-cp311-manylinux_2_34_x86_64.whl (its member list); lib/Dialect/TTMetal/Pipelines/TTMetalPipelines.cpp and test/ttmlir/Conversion/D2MToTTKernel/use_tile_matmul_*.mlir.
- PyPI: the requires_python of sphinx 8.1.3, 8.2.0, 9.0.0, and 9.1.0.
- tt-metal at 5280a9cfb00998fd49667a29523d03aee905c129: tt_metal/jit_build/build.cpp, build.hpp, genfiles.cpp, and build_env_manager.cpp; tt_metal/llrt/hal/tt-1xx/wormhole/wh_hal.cpp; tt_metal/llrt/hal/tt-1xx/hal_1xx_common.cpp; tt_metal/llrt/rtoptions.cpp.
- sfpi 7.25.0: https://github.com/tenstorrent/sfpi include/sfpi_builtins.h.
- alpha01: `rx doctor` (2026-10-06 about 21:48, no rx id) and the read-only rx exec ids 20261006-214855-exec-9935, 20261006-214920-exec-4a93, 20261006-214936-exec-a225, 20261006-215953-exec-0b88, and 20261006-220320-exec-8e14 (its output is not quoted in this file); one refused probe (no id).
