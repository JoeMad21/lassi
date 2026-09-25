# Spike P4.1: joint pin of tt-mlir, tt-metal, and ttsim; build budget

- Task: P4.1 (plans/p4-ttsim.md, planning decision "Joint pin (P4.1)"). Bible: Toolchain Pins, ttsim Facts, Execution Backends (ttsim row), Host Facts (existing assets, scratch cap), Risks And Questions.
- Date: 2026-09-24. Remote probes ran from 18:51 to 19:03 on the alpha01 clock (UTC-07:00); each rx id carries its start time. Upstream facts were read the same evening from GitHub through the GitHub API (`gh api`, authenticated read-only).
- Base: local commit 2c5f491 on p4-ttsim. Every remote probe used `uv run tools/rx.py doctor` or read-only `uv run tools/rx.py exec`, which runs from the scratch root without a checkout, so no repository code took part. Nothing was installed, downloaded, or built on alpha01, and no job ran.
- Device: none. No probe opened a device. The gate refused two probes whose text matched its tt_silicon patterns (see "Gate refusals"); neither was a device need.
- Local downloads: two ttsim libraries (162016 and 219352 bytes) and one soc descriptor went to the workstation's session scratchpad only, to re-hash them and read their ELF version needs. Nothing from them enters the repository.
- Output conventions: outputs are verbatim except where `[... trimmed ...]` says otherwise. Upstream file content is quoted only as names, versions, and hashes (plans/p4-ttsim.md: no tt-metal file enters a tracked file).

## Question

Which tt-mlir, tt-metal, and ttsim does the P4 plan's rule pin: "the newest tt-mlir release (or main commit, if it has no releases) whose tt-metal pin a published ttsim release supports on Wormhole"? Does the TurboQuant checkout at /mnt/nvme10/joseph_ufl/tt-metal serve as that tt-metal? What does the host lack for the build, and how much disk will the pinned build take?

Why it matters: P4.2 installs what this spike pins. P4.9, P4.10, P4.11, and P4.13 build and run against it, and P5 builds tt-mlir at the same pin (Toolchain Pins: tt-mlir, tt-metal, and ttsim are pinned together).

Classification: factual. The rule is stated, and every input it needs is on GitHub or on alpha01. The spike reads "release" literally, as a GitHub release, prerelease or not; see "Owner queue" for the one reading the owner may want to revisit.

## Host state (rx doctor)

`uv run tools/rx.py doctor`, 2026-09-24 about 18:51:

```
"stop": false,
"scratch_free_gb": 427.4,
"root_free_gb": 299.2,
"host": {"hostname": "alpha01", "python": "3.10.12", "nproc": 256, "os": "Ubuntu 22.04.5 LTS", [... trimmed ...]},
"config": {"min_free_gb": 5, "min_free_gb_big": 60, "max_jobs": 3, "max_big_jobs": 1, "max_build_jobs": 32},
"devices_enabled": {"rngd": true, "tt_silicon": false, "rocm_gpu": false, "nvidia_gpu": false},
"running_jobs": [],
```

rx 20260924-185954-exec-1e9c:

```
2026-09-24T18:59:54-07:00
94G	/mnt/nvme10/joseph_ufl
98022860	/mnt/nvme10/joseph_ufl
/dev/nvme23n1p1  3.5T  3.1T  399G  89% /mnt/nvme10
--- toolchains
229M	/mnt/nvme10/joseph_ufl/toolchains/cuda@12.6.3
13G	/mnt/nvme10/joseph_ufl/toolchains/nvhpc@24.11
--- env
TMPDIR=/mnt/nvme10/joseph_ufl/tmp LASSI_TOOLCHAINS=/mnt/nvme10/joseph_ufl/toolchains LASSI_JOBS=32
--- system sfpi
/opt/tenstorrent
/opt/tenstorrent/sfpi
--- glibc
ldd (Ubuntu GLIBC 2.35-0ubuntu3.15) 2.35
```

The scratch root holds 98022860 KiB (94G by `du -sh`) against the owner's 120G cap (125829120 KiB). The plan's stop line of 115G (120586240 KiB) leaves 22563380 KiB, about 21.5 GiB.

## The TurboQuant checkout

Read only. rx 20260924-185152-exec-3dce:

```
2026-09-24T18:51:52-07:00
drwxrwxr-x 30 joseph_ufl joseph_ufl 4096 Aug 27 15:00 /mnt/nvme10/joseph_ufl/tt-metal
91f5a51fb64cbf5f7ab58f1dd6b6b674c2672ebd
91f5a51fb64cbf5f7ab58f1dd6b6b674c2672ebd 2026-03-05T15:20:22+00:00 fix graph tracing on ttnn.topk operation (#39204)
v0.68.0-dev20260305-12-g91f5a51fb6
main
origin	https://github.com/tenstorrent/tt-metal.git (fetch)
origin	https://github.com/tenstorrent/tt-metal.git (push)
--- status
 M tt_metal/fabric/control_plane.cpp
 M tt_metal/impl/program/program.cpp
 M tt_metal/llrt/rtoptions.cpp
?? kernels/
?? <an untracked directory named after the Tenstorrent SMI tool>/
?? tt_metal/fabric/mesh_graph_descriptors/n300_single_chip_h13dsh.textproto
?? tt_metal/programming_examples/rref_tenstorrent/
--- status count
7
```

(The one untracked directory name is paraphrased: the gate refuses any command text that names that tool, see "Gate refusals".)

- Commit: 91f5a51fb64cbf5f7ab58f1dd6b6b674c2672ebd (2026-03-05, `v0.68.0-dev20260305-12-g91f5a51fb6`), branch main, origin tenstorrent/tt-metal.
- Tree state: dirty. Three tracked files are modified and four paths are untracked. rx 20260924-190145-exec-0104 (`git --no-optional-locks diff --stat`): the edits add 23 lines in total (control_plane.cpp 6, program.cpp 9, rtoptions.cpp 8). The rtoptions.cpp edit adds a variable TT_METAL_CLUSTER_DESC_PATH for a real cluster descriptor in silicon mode. None of the edits touches the simulator path. The untracked `kernels/` is 20K (one entry, `rref`), and `rref_tenstorrent/` is 268K.
- Size: 5390248 KiB (5.2G), measured part by part in "Disk estimate".
- Existing build: build_Release (symlink `build`), rx 20260924-185220-exec-c1f2, CMakeCache: CMAKE_BUILD_TYPE Release, compilers clang-20 and clang++-20, generator Ninja, BUILD_PROGRAMMING_EXAMPLES ON, TT_METAL_BUILD_TESTS OFF, TTNN_BUILD_TESTS OFF, WITH_PYTHON_BINDINGS ON, ENABLE_TRACY ON, ENABLE_DISTRIBUTED ON, ENABLE_LIBCXX FALSE, ENABLE_CCACHE OFF, CPM_SOURCE_CACHE `<checkout>/.cpmcache`, CMAKE_INSTALL_PREFIX `<checkout>/build_Release`. Its sfpi is 7.29.0 (tt_metal/sfpi-version, runtime/sfpi).
- Submodules (rx 20260924-185200-exec-93ca): llama reference 29125b7a, tracy 0aaefbb6, tt_llk d090e3bf, umd a0c988a5 (v0.9.3).
- The first status read used plain `git status --porcelain`, as the task allowed; that command may rewrite the stat data in the checkout's .git/index, which was not checked (no tracked content can change). Later reads used `git --no-optional-locks`, so git takes no index lock and writes nothing.

## tt-mlir candidates and their tt-metal pins

tt-mlir records its tt-metal pin as `set(TT_METAL_VERSION "<sha>")` in third_party/CMakeLists.txt, and its LLVM pin as LLVM_PROJECT_VERSION in env/CMakeLists.txt.

Releases (GitHub API, 2026-09-24):

- 232 releases and 239 tags. Every release is flagged prerelease; `releases/latest` answers 404, since GitHub's latest skips prereleases. Every release is a nightly named `0.<n>.0.dev<YYYYMMDD>`, the oldest 0.1.0.dev20250620. The newest is 0.9.0.dev20260221, published 2026-02-21T04:31:50Z; none has been published since. Main is active: its head on 2026-09-24 is 70b7117e56ff9cd7bd2cf97224fd9e57756d2770 (16:05:46Z). The six tags outside the nightly scheme (v0.0, tt-mlir-0.1.0.dev20250620, two nightly-20250616-*, d2m-mm-v0, advchal-v3/advisor-pin) are older or not releases. No 0.10 or 1.x tag exists. Why the GitHub nightlies stopped is not established: `.github/workflows/schedule-nightly.yml` is unchanged between 2026-02-15 and 2026-03-15 (unverified cause).
- 0.9.0.dev20260221 ("Nightly 0.9.0.dev20260221"): the annotated tag f305713d0997b64f41420d3b2a948db0566e20a9 points at commit 5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c (tagged 2026-02-21T04:31:39Z). The release object's target_commitish is c7b8bd7784307cce46577bca1c2d33fdab31fccc, the 0.9.0.dev20260220 tag's commit; both commits pin the same tt-metal (below). Release notes: `pip install ttmlir==0.9.0.dev20260221` from Tenstorrent's extra index; tests workflow run 22211454385 with its status marked failed; "no changes". Asset: ttmlir-0.9.0.dev20260221-cp311-cp311-manylinux_2_34_x86_64.whl, 64837089 bytes, GitHub digest sha256 37b8fd1e7ee1d12b1ba531df9e63038db3cbea97acab2f21547083b4a5bb962e (not re-hashed here).

Pins read at each candidate (`gh api repos/tenstorrent/tt-mlir/contents/third_party/CMakeLists.txt?ref=<sha>` and env/CMakeLists.txt):

```
== tt-mlir 5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c   (tag 0.9.0.dev20260221)
set(TT_METAL_VERSION "5280a9cfb00998fd49667a29523d03aee905c129")
set(LLVM_PROJECT_VERSION "4efe170d858eb54432f520abb4e7f0086236748b")
== tt-mlir c7b8bd7784307cce46577bca1c2d33fdab31fccc   (release target_commitish)
set(TT_METAL_VERSION "5280a9cfb00998fd49667a29523d03aee905c129")
set(LLVM_PROJECT_VERSION "4efe170d858eb54432f520abb4e7f0086236748b")
== tt-mlir 70b7117e56ff9cd7bd2cf97224fd9e57756d2770   (main head, 2026-09-24)
set(TT_METAL_VERSION "d04395ed862b4c65eb6877000c40200f456cb74e")
set(LLVM_PROJECT_VERSION "4efe170d858eb54432f520abb4e7f0086236748b")
```

| Candidate | tt-mlir | tt-metal pin | tt-metal date | ttsim tt-metal's CI used at that commit |
| --- | --- | --- | --- | --- |
| Newest release (the rule's) | 0.9.0.dev20260221, commit 5f396fd6 | 5280a9cfb00998fd49667a29523d03aee905c129, "Improve model tracer infra (#37390)" | 2026-02-17T10:38:27Z | v1.3.4 |
| Main head (not a release) | 70b7117e | d04395ed862b4c65eb6877000c40200f456cb74e, "[skip ci] Fix hal.cpp CODEOWNERS: move off infra fallback (#54856)" | 2026-08-29T22:28:27Z | v1.10.3 (tt_metal/ttsim-version) |
| TurboQuant checkout | none: no tt-mlir commit pins it | 91f5a51fb64cbf5f7ab58f1dd6b6b674c2672ebd | 2026-03-05T15:20:22Z | v1.4.0 (ttsim.yaml default) |

- tt-metal compare API: 91f5a51 is 557 commits ahead of 5280a9cf and 0 behind, so the pin is an ancestor of the checkout's commit. d04395ed is 7385 commits ahead of 91f5a51.
- tt-mlir's history of third_party/CMakeLists.txt (2026-02-10 to 2026-03-20) shows the tt-metal uplifts: 7dda4187 (02-10), d640ed5c (02-17), 5280a9cf (02-19, commit b7d90c81), 8a085d07 (02-28), 883c7142 (03-02), beccdb23 (03-03), and 30419c8e (03-18). tt-mlir's uplift commits name the tt-metal commit in their subject, and GitHub's commit search (`gh search commits --repo tenstorrent/tt-mlir 91f5a51`) finds none naming 91f5a51. So no tt-mlir commit pins the TurboQuant checkout's commit, and it cannot be the joint pin.

## ttsim release for the pinned tt-metal on Wormhole

tt-metal names the ttsim release its own CI runs. Until 2026-07-01 this was the `ttsim-version` input of .github/workflows/ttsim.yaml. Since then it has been the file tt_metal/ttsim-version, added by e3e324b5 "Pipeline reorg: ttsim sku integration (#47870)" on 2026-07-01; it reads `v1.10.9` on main today. A second file, tt_metal/tt-llk/tests/ttsim-version, pins the LLK test harness's ttsim with per-library sha256 (v1.10.3 today).

At 5280a9cf (`gh api .../git/trees/5280a9cf...?recursive=1`, then the files):

- The only ttsim workflow is .github/workflows/ttsim.yaml. Its input `ttsim-version` has default "v1.3.4". Its integration job runs a matrix of wormhole_b0 (soc_desc wormhole_b0_80_arch.yaml) and blackhole with env CC gcc-12, CXX g++-12, TT_METAL_SIMULATOR, TT_METAL_SIMULATOR_HOME, and TT_METAL_SLOW_DISPATCH_MODE 1. It installs the tt-metalium Debian packages, downloads libttsim_<arch>.so of that tag, copies the soc descriptor beside it as soc_descriptor.yaml, then builds with `cmake -G Ninja` and runs each of eight programming examples whose directory exists (it warns and skips a missing one): add_2_integers_in_compute, add_2_integers_in_riscv, eltwise_binary, eltwise_sfpu, hello_world_compute_kernel, hello_world_datamovement_kernel, matmul_multi_core, and matmul_multicore_reuse.
- Callers: merge-gate.yaml and pr-gate.yaml call ./.github/workflows/ttsim.yaml, and neither passes ttsim-version (0 matches in each), so both used v1.3.4.
- CI results at 5280a9cf (push to main, 2026-02-17). PR Gate run 22095847892 succeeded, including job 63853437574 "ttsim-integration-tests / TTSim Integration - wormhole_b0" and the TTNN ttsim groups on wormhole_b0. Merge Gate run 22095848013 succeeded, including "ttsim-unit-tests / TTSim C++ tests - wormhole_b0". The job log has expired (`GET .../actions/jobs/63853437574/logs` answers HTTP 410), so the version rests on the workflow file at that commit, which a same-repository reusable workflow runs as is. These are upstream CI results, not measurements of ours.
- ttsim releases around that date: v1.3.4 on 2026-02-06T22:30:45Z, v1.3.5 on 2026-02-18T16:58:13Z (the day after the pin), and v1.3.6 on 2026-02-20. v1.3.4 was the newest release on the pin's date. ttsim has 63 releases in all, the newest v1.10.9 (2026-09-18).
- The pin reads the bible ttsim row's variables. tt_metal/llrt/rtoptions.cpp at 5280a9cf has EnvVarID entries TT_METAL_SIMULATOR, TT_METAL_SLOW_DISPATCH_MODE, and TT_METAL_DISABLE_SFPLOADMACRO. The CI job at the pin did not set TT_METAL_DISABLE_SFPLOADMACRO; tt-metal's current setup-ttsim action does. Also at the pin: tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml (3125 bytes, sha256 24fd3dfae80435a7d9113e255d6d6af9cdf83f6ae3b1c385e1d48a24795ccf49, git blob ff4e5576b1b5eab137c4ae25996f6bedd2c9ad24), and tt_metal/programming_examples/add_2_integers_in_riscv, whose target is `metal_example_add_2_integers_in_riscv`.

ttsim v1.3.4 (`gh api repos/tenstorrent/ttsim/releases/tags/v1.3.4`):

```
v1.3.4	2026-02-06T22:30:45Z	prerelease false	target main
- Various Tensix bug fixes/features, including BH packer/unpacker fixes for tilize/untilize
- Add unlatched high 32 bits of tile WALL_CLOCK register
libttsim_bh.so	170208	sha256:a09f406264ea729910af1c1571381fc29da3678b488498e0fa17df345fa1642e
libttsim_wh.so	162016	sha256:7b10aa05a5297c4a28f274e39526bfe6c69373661d1b4d4e47c4257e44b79507
tag v1.3.4 -> commit 2174bdb9dce5acbd775fad3fbfb96710ffe1bd87
```

Local re-hash on the workstation (`gh release download v1.3.4 --repo tenstorrent/ttsim --pattern libttsim_wh.so`, then `sha256sum`), and the same for v1.10.3 for comparison:

```
07c357534aa66163c36f36c8113d988028db4813dbf039198b009bcc5c0287c4  ttsim-v1.10.3/libttsim_wh.so   (219352 bytes)
7b10aa05a5297c4a28f274e39526bfe6c69373661d1b4d4e47c4257e44b79507  ttsim-v1.3.4/libttsim_wh.so    (162016 bytes)
```

v1.3.4 matches GitHub's digest. v1.10.3 matches the ttsim_wh_so_hash in tt-metal's tt_metal/tt-llk/tests/ttsim-version.

ELF version needs, read by a stdlib Python parser of .dynamic and .gnu.version_r (session scratchpad, not tracked):

```
== ttsim-v1.3.4/libttsim_wh.so
NEEDED: libc.so.6
  libc.so.6: GLIBC_2.14
  libc.so.6: GLIBC_2.2.5
  libc.so.6: GLIBC_2.3.4
  libc.so.6: GLIBC_2.4
max GLIBC needed: GLIBC_2.14
== ttsim-v1.10.3/libttsim_wh.so
[... same five lines ...]
max GLIBC needed: GLIBC_2.14
```

alpha01 has glibc 2.35, so by symbol versions the library loads there; loading it for real is P4.2's check. ttsim's docs/libttsim_api.md says the library links only against the C library and the loader, and that release binaries are built on ubuntu-24.04 but expected to run on 22.04. The README gives the Wormhole setup: TT_METAL_SIMULATOR points at libttsim_wh.so, the directory holding it also holds soc_descriptor.yaml copied from wormhole_b0_80_arch.yaml, and TT_METAL_SLOW_DISPATCH_MODE=1. Neither document names a tt-metal version, which is why tt-metal's own CI pin is the evidence of support.

Tier A coverage: of the bible's Tier A examples, eltwise_binary and eltwise_sfpu are in that CI list and ran, and the gate's add_2_integers_in_riscv did too. matmul_multi_core is in the list, but at the pin tt_metal/programming_examples/CMakeLists.txt installs the matmul examples as one directory, examples/matmul/, so the job's per-example directory check fails and it skipped both matmul entries (Sources: tt-metal example install rules at the pin). So loopback, matmul_single_core, and matmul_multi_core are unverified on ttsim v1.3.4 until P4.9 runs them.

## ttsim issue #18

`gh api repos/tenstorrent/ttsim/issues/18`, its comments, and its timeline:

- Title, in short: on ttsim Wormhole, every unpack_to_dest=true LLK path fails with `UndefinedBehavior: tensix_unpacr: unpack_to_dst=0 in_data_format=0 out_data_format=0`, while Blackhole passes.
- Opened 2026-09-14T13:49:31Z. The reporter used the tt-llk pytest harness, tt-metal main at 70f840f, tt-exalens 0.3.31, and aarch64 builds. Float32 datacopy with dest_acc Yes (unpack_to_dest) and the sqrt infinity regression test report UB on ttsim wh 1.10.3, 1.10.6, and 1.10.7, and pass on bh 1.10.3; datacopy with dest_acc No passes everywhere. The reporter reads the cause as a difference between which config context the LLK writes Unpack_if_sel_cntx in and which one the simulator reads. The issue body also says the same kernels run correctly on Wormhole silicon and on the Blackhole simulator. The closing comment quotes the Wormhole ISA text that silicon behavior in this case has not been validated and must not be relied upon; it does not address the config-context point.
- Closed 2026-09-14T21:01:29Z, state_reason "completed", by a ttsim maintainer (mcraigheadTT). The comment cites tt-isa-documentation WormholeB0/TensixTile/TensixCoprocessor/UNPACR_Regular.md: FP32 or INT32 unpack without UnpackToDst is intentionally UndefinedBehavior, so the case "is therefore not legal SW behavior". The timeline holds two cross-references (tt-metal issues 56490, open, and 56532, closed; neither is an LLK fix for this), the comment, and the close. No ttsim commit is linked.

Does issue #18 still stand? As an open simulator bug, no: it is closed, and ttsim's position is that the UB report is correct. As a behavior, yes: nothing was changed, so on the reporter's evidence ttsim Wormhole 1.10.3, 1.10.6, and 1.10.7 (the versions the reporter ran) still stop those Float32 unpack_to_dest paths with UndefinedBehavior. The comment does not answer the reporter's context-register point. Whether ttsim v1.3.4 with tt-metal 5280a9cf behaves the same is not verified. The bible's rule to check every reference kernel before it enters a suite stays, and P4.9 does the check.

## Host build prerequisites

Requirements at 5280a9cf, from GitHub: CMakeLists.txt `cmake_minimum_required(VERSION 3.24...4.2)` and CMAKE_CXX_STANDARD 20. build_metal.sh defaults to cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake, which sets clang-20 and clang++-20 and ENABLE_LIBCXX FALSE, and uses ld.mold if found, else ld.lld-20 if found. gcc-12 and gcc-14 toolchain files also exist. install_dependencies.sh lists the Ubuntu packages build-essential, cmake, curl, g++-12, git, libc++-20-dev, libc++abi-20-dev, libcapstone-dev, libhwloc-dev, libnuma-dev, libopenmpi-dev, libssl-dev, libtbb-dev, ninja-build, openmpi-bin, pandoc, pkg-config, python3-dev, python3-pip, python3-pkg-resources, python3-venv, wget, and xz-utils. tt_metal/distributed/CMakeLists.txt defaults ENABLE_DISTRIBUTED to OFF and uses ULFM MPI at /opt/openmpi-v5.0.7-ulfm only when it is on. tt_metal/hw/CMakeLists.txt fetches sfpi with CMake FetchContent into `<source>/runtime/sfpi`, checked by URL_HASH against tt_metal/sfpi-version, unless TT_USE_SYSTEM_SFPI selects /opt/tenstorrent.

On alpha01, from rx 20260924-185239-exec-2d3a (tools), 20260924-185249-exec-1540 (dpkg), 20260924-190008-exec-486d (sfpi), and 20260924-190250-exec-f426 (linkers, cmake):

| Need at the pin | alpha01 | Route if missing |
| --- | --- | --- |
| CMake 3.24 to 4.2 | /usr/local/bin/cmake 3.31.6, first on PATH; /usr/bin/cmake 4.2.1 (package cmake 4.2.1-0kitware1ubuntu22.04.1) | Present. P4.2 names the path it uses in the pin |
| Ninja | /usr/bin/ninja 1.10.1 (ninja-build) | Present |
| clang-20, the default toolchain | clang-20 and clang++-20, Ubuntu clang 20.1.8; clang-17 also present | Present |
| Linker | ld.lld-20 (lld-20 20.1.8) and GNU ld 2.38; no ld.mold | Present (mold is optional) |
| g++-12 (gcc-12 toolchain; CI's example builds) | gcc-12 and g++-12 12.3.0, libstdc++-12-dev | Present |
| libc++-20-dev, libc++abi-20-dev | Missing (libc++-17-dev and libc++abi-17-dev present) | Not needed: the default toolchain sets ENABLE_LIBCXX FALSE, as the TurboQuant build did. If a libc++ build were ever wanted, an unverified user-space route is extracting the LLVM apt packages with `dpkg-deb -x` under $LASSI_TOOLCHAINS |
| sfpi 7.25.0 (the pin's tt_metal/sfpi-version, build 252) | Missing. /opt/tenstorrent/sfpi is the root-owned sfpi 7.48.0 Debian package (riscv-tt-elf-g++ 15.1.0); the checkout's runtime/sfpi is 7.29.0 | tt-metal's own CMake FetchContent downloads sfpi_7.25.0_x86_64_debian.txz (88805716 bytes, sha256 6a8883c448df537d9661e239e67721b1ba2a4d0e8ea7a573070d2f6c6d015093, equal to both the pin's sfpi-version and GitHub's asset digest; sfpi release 7.25.0 of 2026-02-10) into the pinned tree under $LASSI_TOOLCHAINS. User space, no root. Keep TT_USE_SYSTEM_SFPI off |
| Python 3 with headers, venv, pip | Python 3.10.12, /usr/include/python3.10/Python.h, python3-venv, python3-pip 22.0.2, python3-pkg-resources | Present |
| hwloc, numa, tbb, capstone, ssl, pkg-config, xz, pandoc | libhwloc-dev 2.7.0, libnuma-dev 2.0.14, libtbb-dev 2021.5.0, libcapstone-dev 4.0.2, libssl-dev 3.0.2, pkg-config 0.29.2, xz-utils 5.2.5, pandoc 2.9.2.1 | Present |
| git, git-lfs, curl, wget, build-essential | git 2.34.1, git-lfs 3.0.2, curl, wget, build-essential 12.9 | Present |
| MPI (optional, ENABLE_DISTRIBUTED) | openmpi-ulfm 5.0.7 at /opt/openmpi-v5.0.7-ulfm (mpirun 5.0.7rc2); openmpi-bin and libopenmpi-dev 4.1.2 | Present. A single-chip ttsim build does not need it; P4.2 records its choice |
| ccache (optional, ENABLE_CCACHE) | Missing | Not needed |

Also for P4.2: CPM downloads the build's C++ dependencies at configure time into CPM_SOURCE_CACHE, which the TurboQuant build kept inside its tree. The pinned build keeps it inside its own tree or under $LASSI_SCRATCH, and TMPDIR stays on scratch (the gate sets it).

## Disk estimate

The checkout, measured part by part (rx 20260924-190108-exec-2770; du counts each block once across its arguments, and the parts add up exactly to the total):

| Part | KiB | Note |
| --- | --- | --- |
| .git | 1452808 | full history, with .git/modules (the four submodules) at 168288 |
| .cpmcache | 1620848 | CPM sources of the build's dependencies |
| build_Release | 1408144 | libexec 505444 (the installed tt-metalium tree), ttnn 301188, _deps 268176, lib 87436, programming_examples 25300 |
| runtime/sfpi | 445764 | sfpi 7.29.0 |
| python_env | 19400 | |
| sources and the rest | 443284 | |
| Total | 5390248 | 5.2G by `du -sh` (rx 20260924-185200-exec-93ca) |

PROJECTED, not measured: a pinned build at 5280a9cf with the checkout's configuration (Release, programming examples on, Python bindings on, tracy on, tests off, full clones) takes about 5390248 KiB, 5.1 GiB. The pin is 557 commits older, has the same four submodules, and its sfpi 7.25.0 unpacks to about the same size, so the checkout is a close proxy. Range 4 to 7 GiB, depending on shallow clones and on whether Python bindings and tracy stay on; planning bound 8 GiB. ttsim adds 162016 bytes and the 3125-byte descriptor. With the scratch root at 98022860 KiB, the root would reach about 99G after the build (PROJECTED), or about 102G at the bound. Both stay under the 115G stop line and the 120G cap. This narrows the plan's earlier PROJECTED 10 to 20G; P4.2's du before and after replaces it with a measurement.

For P5, PROJECTED and unverified: tt-mlir's third_party/CMakeLists.txt at the release builds its own tt-metal under third_party/tt-metal/src/tt-metal, with its own CPM cache. That would be a second copy of about 5G, plus the LLVM build at 4efe170d, which is not estimated here, unless P5 points tt-mlir at the pinned tree. Whether tt-mlir supports that is not checked.

## Gate refusals

Two read-only probes were refused before running, and both were rerun without the matching text:

- `ls` of the Tenstorrent device node directory: "gate refused: device class 'tt_silicon' is disabled by the owner; command matches //dev/tenstorrent/". The bible's Environment State already records that node as absent, so nothing is lost.
- `du` of the checkout's untracked directory named after the Tenstorrent SMI tool: "command matches /\btt-smi\b/". The directory was left out of the size breakdown (it is inside the 443284 KiB of "sources and the rest").

Neither is a device need, so no owner-queue item follows. A later task that must read that directory by name needs another route (the gate pattern is owner-set).

## Finding

- The rule selects tt-mlir 0.9.0.dev20260221 (commit 5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c), its tt-metal pin 5280a9cfb00998fd49667a29523d03aee905c129, and ttsim v1.3.4 (libttsim_wh.so, sha256 7b10aa05a5297c4a28f274e39526bfe6c69373661d1b4d4e47c4257e44b79507). The newest tt-mlir release already meets the ttsim condition: at 5280a9cf tt-metal's own CI ran its Wormhole ttsim integration job with v1.3.4 and passed. So the rule stops at the newest release, and the main-commit branch of the rule does not apply.
- Scratch cap: no conflict. The build is PROJECTED at about 5.1 GiB (bound 8), taking scratch from 94G to about 99G (102G at the bound).
- TurboQuant: no conflict with its use of the checkout. The checkout (91f5a51, dirty) is not the pin; it is 557 commits past it, and no tt-mlir commit pins 91f5a51. So, as the planning decision and PHASE-NOTES P4 say, P4.2 builds a separate pinned tree under $LASSI_TOOLCHAINS, and the checkout is neither built in nor written to. What stays open for P5 is whether the Tier B kernels, written against 91f5a51 plus 23 lines of local silicon-mode edits, build and run on 5280a9cf. That is unverified and is P5's check.
- Confidence: high for the pins and hashes (tags, tt-mlir's pin file, tt-metal's CI workflow at the commit, and local re-hashes). Medium for "supports on Wormhole": it rests on an upstream CI job that passed and that lists eight examples, of which the two matmul entries were skipped at the pin (their install directory is examples/matmul/); its log has expired, so which examples actually ran cannot be re-read. P4.9 confirms it on alpha01.

## Consequences for the plan

- P4.2 can proceed; it does not go OWNER. Its pin files record: tt-metal 5280a9cfb00998fd49667a29523d03aee905c129 with the submodule commits at that tree (tt_llk f990966829c2831bf2a58d8ce31779acbb476139, umd 0450f1be3a21cbf17adb97517038586a6a3af4c0, tracy 0aaefbb689b4c60694edc905545fc4709fd13f6a, llama reference 29125b7ad8b5513eeaa4417ed92892bf39c8bd74); sfpi 7.25.0 with the txz sha256 above; the toolchain file, compiler, and CMake paths; the build flags; ttsim v1.3.4 libttsim_wh.so with its sha256; and the soc descriptor copied from the pinned tree (sha256 24fd3dfa...ccf49). The planning bound is 8 GiB.
- P4.9 runs loopback, matmul_single_core, and matmul_multi_core, which the upstream CI did not run at the pin, checks issue #18's Float32 unpack_to_dest case at the pin, and checks whether TT_METAL_DISABLE_SFPLOADMACRO changes any result at the pin (the CI job at the pin did not set it).
- P4.13's tt-pairs-v0 manifest pins tt-metal 5280a9cf.
- P5 builds tt-mlir 0.9.0.dev20260221 at LLVM 4efe170d858eb54432f520abb4e7f0086236748b. Its wheel is cp311 only, while alpha01's python3 is 3.10, so the wheel does not serve the host interpreter. P5 also checks Tier B against 5280a9cf, and plans how tt-mlir finds the pinned tt-metal (disk estimate above).

## Proposed bible edit

Toolchain Pins. Old:

```
- tt-mlir, tt-metal, and ttsim are pinned together; emitted kernel C++ must match the tt-metal API version.
```

New:

```
- tt-mlir, tt-metal, and ttsim are pinned together; emitted kernel C++ must match the tt-metal API version.
- Joint pin [DESIGN] (P4.1, plans/spikes/p4-tt-pins.md), by the P4 plan's rule, the newest tt-mlir release (or main commit, if it has no releases) whose tt-metal pin a published ttsim release supports on Wormhole: tt-mlir 0.9.0.dev20260221, the newest of its GitHub releases, which are all nightly prereleases and stop on 2026-02-21 (tag commit 5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c; LLVM 4efe170d858eb54432f520abb4e7f0086236748b; its release notes mark that nightly's own test run failed, run 22211454385); tt-metal 5280a9cfb00998fd49667a29523d03aee905c129 (2026-02-17), the TT_METAL_VERSION of that release's third_party/CMakeLists.txt, with its sfpi 7.25.0 (sfpi_7.25.0_x86_64_debian.txz, sha256 6a8883c448df537d9661e239e67721b1ba2a4d0e8ea7a573070d2f6c6d015093), which its CMake fetches; and ttsim v1.3.4, the release tt-metal's own ttsim CI ran on Wormhole at that commit (libttsim_wh.so, 162016 bytes, sha256 7b10aa05a5297c4a28f274e39526bfe6c69373661d1b4d4e47c4257e44b79507). The TurboQuant checkout at /mnt/nvme10/joseph_ufl/tt-metal (91f5a51, 557 commits past the pin, with local edits) is not the pin and is never built in or written to; the pinned tree installs under $LASSI_TOOLCHAINS (P4.2). A pin change is a Decision Log entry.
```

ttsim Facts, change 1. Old:

```
Source: [tenstorrent/ttsim](https://github.com/tenstorrent/ttsim).
```

New:

```
Source: [tenstorrent/ttsim](https://github.com/tenstorrent/ttsim). Pinned release v1.3.4 with tt-metal 5280a9cf (Toolchain Pins). tt-metal names the ttsim release its own CI runs: the ttsim-version input of .github/workflows/ttsim.yaml until 2026-07-01, and the file tt_metal/ttsim-version since. At the pinned commit that CI's Wormhole job passed (PR Gate run 22095847892); the job lists eight programming examples for a virtual Wormhole and skips any whose install directory is missing; at the pin, add_2_integers_in_riscv, eltwise_binary, and eltwise_sfpu are among those it ran, while the two matmul entries install under examples/matmul/, so the job skipped them. Its log has expired. This is an upstream CI result, not a LASSI measurement. libttsim_wh.so v1.3.4 needs only the C library, at GLIBC_2.14 at most (alpha01 has 2.35).
```

ttsim Facts, change 2. Old:

```
- On Wormhole, every LLK path with `unpack_to_dest=true` fails today ([issue #18](https://github.com/tenstorrent/ttsim/issues/18)); Blackhole passes. Check every reference kernel before it enters a suite.
```

New:

```
- On Wormhole, Float32 LLK paths with `unpack_to_dest=true` (dest_acc) stop with `UndefinedBehavior: tensix_unpacr: unpack_to_dst=0` on ttsim Wormhole 1.10.3, 1.10.6, and 1.10.7 (the reporter's runs), while Blackhole 1.10.3 passes ([issue #18](https://github.com/tenstorrent/ttsim/issues/18), reported against tt-metal 70f840f). The issue was closed on 2026-09-14 with no simulator change: ttsim's maintainer ruled that FP32 or INT32 unpack without UnpackToDst is UndefinedBehavior under the Wormhole ISA documentation, so ttsim keeps reporting it; the reporter holds that the LLK sets UnpackToDst in its other config context and that the same kernels pass on Wormhole silicon, which the closing comment did not answer. Whether it occurs at the pinned ttsim v1.3.4 with tt-metal 5280a9cf is not verified. Check every reference kernel before it enters a suite (P4.9).
```

Decision Log, new top row (and the count sentence goes from "Seventy-eight ... and thirty-two on 2026-09-24" to "Seventy-nine ... and thirty-three on 2026-09-24", or one more than it reads when the row is applied):

```
| 2026-09-24 | The joint pin of tt-mlir, tt-metal, and ttsim (task P4.1): tt-mlir 0.9.0.dev20260221 (tag commit 5f396fd6), tt-metal 5280a9cfb00998fd49667a29523d03aee905c129, which that release's third_party/CMakeLists.txt names, with its sfpi 7.25.0, and ttsim v1.3.4 (libttsim_wh.so sha256 7b10aa05a5297c4a28f274e39526bfe6c69373661d1b4d4e47c4257e44b79507), chosen by the P4 plan's rule: the newest tt-mlir release whose tt-metal pin a published ttsim release supports on Wormhole. The pinned tt-metal installs under $LASSI_TOOLCHAINS, never in the TurboQuant checkout (91f5a51, dirty, 557 commits past the pin). ttsim Facts records that issue #18 was closed as intended behavior, not fixed | tt-mlir's GitHub releases are nightly prereleases ending at 0.9.0.dev20260221, which pins tt-metal 5280a9cf. At that commit tt-metal's own CI ran its Wormhole ttsim job with v1.3.4, its workflow default and the newest ttsim release that day, and passed (PR Gate run 22095847892), so the rule stops at the newest release. No tt-mlir commit pins the checkout's commit, so the checkout cannot be the pin; building apart leaves TurboQuant's tree unchanged. The build is PROJECTED at about 5 GiB from the checkout's measured 5390248 KiB, so the scratch root stays near 99G of the 120G cap. plans/spikes/p4-tt-pins.md records the commands, outputs, and sources |
```

## Owner queue

No item is required by the planning decision's triggers. The scratch cap holds (PROJECTED about 99G, under the 115G stop line), and the pinned build leaves the TurboQuant checkout untouched (apart from the index stat data the first status read may have rewritten).

One reading is the owner's to revisit, and the caller may file it if it judges it a choice. The spike applied the rule as written. The draft below was filed as OQ-026, and the owner answered it in the working session on 2026-09-24 with option (a), the rule's result.

```
## OQ-<nnn> Joint pin: tt-mlir's releases are stale nightlies
State: OPEN
Kind: decision
Blocks: none (P4.2 proceeds on the rule's pin unless the owner picks (b))
Evidence: plans/spikes/p4-tt-pins.md
Question: The P4 plan's joint-pin rule takes the newest tt-mlir release. tt-mlir's 232 GitHub releases are all nightly prereleases; the last, 0.9.0.dev20260221, marks its own test run failed in its release notes, and main has moved on for seven months since. The rule therefore pins tt-metal 5280a9cf (2026-02-17) and ttsim v1.3.4 (2026-02-06). tt-mlir main instead pins tt-metal d04395ed (2026-08-29), whose own tt_metal/ttsim-version names ttsim v1.10.3. Keep the rule's result, or read "release" as excluding nightly prereleases, so the rule falls to a main commit?
Options:
(a) Keep: tt-mlir 0.9.0.dev20260221, tt-metal 5280a9cf, ttsim v1.3.4. Tagged and reproducible; tt-metal's own CI passed its Wormhole ttsim job at that commit with v1.3.4; 557 commits before the TurboQuant checkout's tt-metal, the closest any tt-mlir pin comes to Tier B's source. Seven months of ttsim fixes are absent, so more sim-gap endings are possible (P4.9 measures), and P5 gets February's tt-mlir dialects.
(b) Main: tt-mlir main at a stated commit (70b7117e on 2026-09-24), tt-metal d04395ed, ttsim v1.10.3. A current simulator and current dialects, and tt-metal names its ttsim release in-tree. Not a release, so the rule changes (Decision Log) and needs its own rule for which main commit; 7385 commits past the TurboQuant checkout, so the Tier B port risk grows. Same build budget (PROJECTED about 5 GiB).
Recommendation: (a). Tagged artifacts with upstream CI evidence on Wormhole meet the rule; the gate's example and two of the five Tier A examples ran in that CI job; and it stays closest to Tier B's tt-metal. Revisit before P5 if P4.9 finds simulator gaps that a newer ttsim closes.
Answer:
```

## Sources

All read on 2026-09-24.

- tt-metal example install rules at the pin: https://github.com/tenstorrent/tt-metal/blob/5280a9cfb00998fd49667a29523d03aee905c129/tt_metal/programming_examples/CMakeLists.txt (installs add_2_integers_in_riscv, eltwise_binary, eltwise_sfpu, and one matmul directory into tt-metalium/examples; the CI job runs an example only if /usr/share/tt-metalium/examples/<name> exists).
- tt-mlir releases and tags: https://github.com/tenstorrent/tt-mlir/releases, https://github.com/tenstorrent/tt-mlir/releases/tag/0.9.0.dev20260221 (API: repos/tenstorrent/tt-mlir/releases, /tags, /git/ref/tags/0.9.0.dev20260221, /git/tags/f305713d0997b64f41420d3b2a948db0566e20a9).
- tt-mlir pins: https://github.com/tenstorrent/tt-mlir/blob/5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c/third_party/CMakeLists.txt, https://github.com/tenstorrent/tt-mlir/blob/5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c/env/CMakeLists.txt, the same files at c7b8bd7784307cce46577bca1c2d33fdab31fccc and 70b7117e56ff9cd7bd2cf97224fd9e57756d2770, and the history of third_party/CMakeLists.txt (API commits?path=third_party/CMakeLists.txt, 2026-02-10 to 2026-03-20).
- tt-metal at the pin: https://github.com/tenstorrent/tt-metal/commit/5280a9cfb00998fd49667a29523d03aee905c129; https://github.com/tenstorrent/tt-metal/blob/5280a9cfb00998fd49667a29523d03aee905c129/.github/workflows/ttsim.yaml; the same commit's merge-gate.yaml, pr-gate.yaml, tt_metal/sfpi-version, tt_metal/sfpi-info.sh, tt_metal/hw/CMakeLists.txt, tt_metal/distributed/CMakeLists.txt, CMakeLists.txt, build_metal.sh, cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake, install_dependencies.sh, tt_metal/llrt/rtoptions.cpp, .gitmodules, and tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml.
- tt-metal CI at the pin: https://github.com/tenstorrent/tt-metal/actions/runs/22095847892 (PR Gate), https://github.com/tenstorrent/tt-metal/actions/runs/22095847892/job/63853437574 (TTSim Integration - wormhole_b0), https://github.com/tenstorrent/tt-metal/actions/runs/22095848013 (Merge Gate).
- tt-metal ttsim pins on main: https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/ttsim-version and its history; https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/tt-llk/tests/ttsim-version; https://github.com/tenstorrent/tt-metal/blob/main/.github/actions/fetch-ttsim/action.yml; https://github.com/tenstorrent/tt-metal/blob/main/.github/actions/setup-ttsim/action.yml; tt_metal/ttsim-version at d04395ed862b4c65eb6877000c40200f456cb74e; ttsim.yaml at 91f5a51fb64cbf5f7ab58f1dd6b6b674c2672ebd.
- tt-metal compare: repos/tenstorrent/tt-metal/compare/5280a9cf...91f5a51 (ahead 557, behind 0) and 91f5a51...d04395ed (ahead 7385, behind 0).
- ttsim: https://github.com/tenstorrent/ttsim (README), https://github.com/tenstorrent/ttsim/blob/main/docs/libttsim_api.md, https://github.com/tenstorrent/ttsim/releases/tag/v1.3.4, https://github.com/tenstorrent/ttsim/releases/tag/v1.10.9, https://github.com/tenstorrent/ttsim/releases (63 releases).
- ttsim issue: https://github.com/tenstorrent/ttsim/issues/18 (comments and timeline); cross-references https://github.com/tenstorrent/tt-metal/issues/56490 and https://github.com/tenstorrent/tt-metal/issues/56532; the cited ISA page https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/UNPACR_Regular.md (not opened here).
- sfpi: https://github.com/tenstorrent/sfpi/releases/tag/7.25.0.
- alpha01: rx doctor (2026-09-24 about 18:51-07:00) and rx exec ids 20260924-185152-exec-3dce, 20260924-185200-exec-93ca, 20260924-185220-exec-c1f2, 20260924-185230-exec-5682 (its output is not quoted in this file), 20260924-185239-exec-2d3a, 20260924-185249-exec-1540, 20260924-185954-exec-1e9c, 20260924-190008-exec-486d, 20260924-190108-exec-2770, 20260924-190145-exec-0104, 20260924-190250-exec-f426.
