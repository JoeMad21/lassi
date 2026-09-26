# P4.2: install of the pinned tt-metal and ttsim

[MEASURED] provenance.json (rx job 20260925-173117-p4-tt-metal-9c48):

- commit: f9bb560 (clean tree, run from a detached clean worktree; the job printed "changed paths 0");
- host: alpha01, Ubuntu 22.04.5 LTS; start 2026-09-25 17:31:17, end 17:35:01 (UTC-07:00), 3 minutes 44 seconds for the whole job, of which the configure took 131.2 s (job log); job rc 0, and the command's last line was `lassi_rc=0`;
- build parallelism: $LASSI_JOBS, which the gate sets on alpha01 to 32 (rx doctor's max_build_jobs and the gate's environment, plans/spikes/p4-tt-pins.md; the job log does not print it);
- toolchain pins: toolchains/tt-metal.pin and toolchains/ttsim.pin (their text is in provenance.json under pins);
- device: none. The job built and installed files only; no tt-metal program ran, so neither the kernel JIT nor ttsim started.

Command: `bash toolchains/tt-metal.sh && bash toolchains/ttsim.sh`, wrapped with du, free, and the root-filesystem check (provenance.json, cmd). The full job log is kept locally under .rx/pulls/, not in the repository.

## Results

- tt-metal 5280a9cfb00998fd49667a29523d03aee905c129 with its four submodules at their pinned commits (Install record, below) is installed at $LASSI_TOOLCHAINS/tt-metal@5280a9cf, 3157352 KiB. The one Ninja run took 676 steps for the targets tt_metal and hw_toolchain and ten programming examples: the gate's add_2_integers_in_riscv; the Tier A examples loopback, eltwise_binary, eltwise_sfpu, matmul_single_core, and matmul_multi_core; and add_2_integers_in_compute, hello_world_compute_kernel, hello_world_datamovement_kernel, and matmul_multicore_reuse, the four others the pin's ttsim CI job lists (it skipped matmul_multicore_reuse, as it did matmul_multi_core; bible, ttsim Facts). None was run here.
- The gate's example, metal_example_add_2_integers_in_riscv, records that clang 20.1.8 compiled it and LLD 20.1.8 linked it (read from its .comment section; the log's check_linked line).
- The tree's tracked files and its submodules were at their pinned commits, unchanged, before the configure and after the build (the log's verify_tree lines; untracked files are not part of that check). tt-metal's configure reported its version as `0.0-alpha0+1.5280a9cfb0+m` (job log); why it carries that suffix is not established here.
- sfpi 7.25.0: tt-metal's own lookup named sfpi_7.25.0_x86_64_debian.txz and its pinned sha256 for this host, and the fetched txz has that sha256 (6a8883c4...5093), so no sfpi source build ran.
- ttsim v1.3.4 is installed at $LASSI_TOOLCHAINS/ttsim@v1.3.4, 172 KiB: libttsim_wh.so, sha256 7b10aa05a5297c4a28f274e39526bfe6c69373661d1b4d4e47c4257e44b79507, with soc_descriptor.yaml, sha256 24fd3dfae80435a7d9113e255d6d6af9cdf83f6ae3b1c385e1d48a24795ccf49, beside it.
- Host tools, from the job log: cmake 3.31.6 (/usr/local/bin/cmake), ninja 1.10.1, clang-20 and clang++-20 20.1.8, ld.lld-20 20.1.8, python3 3.10.12 (/usr/bin/python3); clang-format was not on PATH.
- Scratch root: 98040196 KiB before and 101197976 KiB after, a change of 3157780 KiB (about 3.0 GiB). toolchains/tt-metal.pin's PROJECTED figure was about 3.5 GiB; the space check planned for 8 GiB against the 115 GiB stop line. `free -g` showed 3022 GiB of memory in total, 2827 GiB free before and 2824 GiB after.
- Root filesystem: both scripts' checks passed, nothing changed under /tmp or /var/tmp since the start, and the job's own find listed no file. The checks cover files the job's user owns and skipped 44 unreadable paths (job log).

## The CMake-fetched packages (a gap in the acceptance)

The configure fetched 25 packages through CPM, recorded after the fetch (CMake-fetched packages, below) by git commit (24) or by a sha256 of the unpacked files (boost); changed=N counts the tracked files that differ from the fetched commit (toolchains/tt-metal.sh, record_cpm_sources), and six show some. They are not pinned in advance, and nothing compares a later fetch with this record. Many are fetched by upstream tag, so a fresh rerun after a move-aside could take a moved tag unrefused, and boost's hash is of its unpacked files, not its download. P4.2's "the sha256 of every download ... a mismatch is refused" therefore holds for the tree, its submodules, sfpi, CPM.cmake, the ttsim library, and the descriptor, and only as an after-the-fact record for the CPM packages. plans/PHASE-NOTES.md, P4, carries the follow-up: compare the tree's lassi-cpm-sources.txt with the list below before any later rebuild is used.

## Install record

$LASSI_TOOLCHAINS/tt-metal@5280a9cf/lassi-install.txt, written by toolchains/tt-metal.sh after every check passed, as read by rx exec 20260925-173747-exec-fd38 after the job, with trailing whitespace and blank lines trimmed (the last line is the exec's rx status):

```
tt-metal 5280a9cfb00998fd49667a29523d03aee905c129 https://github.com/tenstorrent/tt-metal.git
submodule tt_metal/third_party/tt_llk f990966829c2831bf2a58d8ce31779acbb476139
submodule tt_metal/third_party/umd 0450f1be3a21cbf17adb97517038586a6a3af4c0
submodule tt_metal/third_party/tracy 0aaefbb689b4c60694edc905545fc4709fd13f6a
submodule models/demos/t3000/llama2_70b/reference/llama 29125b7ad8b5513eeaa4417ed92892bf39c8bd74
sfpi 7.25.0 https://github.com/tenstorrent/sfpi/releases/download/7.25.0/sfpi_7.25.0_x86_64_debian.txz 6a8883c448df537d9661e239e67721b1ba2a4d0e8ea7a573070d2f6c6d015093
cpm https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.2/CPM.cmake c8cdc32c03816538ce22781ed72964dc864b2a34a310d3b7104812a5ca2d835d
host /usr/local/bin/cmake (cmake version 3.31.6); /usr/bin/ninja (1.10.1); clang-20, clang++-20 (clang version 20.1.8); ld.lld-20 (LLD 20.1.8)
toolchain cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake
flags -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_TRACY=OFF -DENABLE_DISTRIBUTED=OFF -DWITH_PYTHON_BINDINGS=OFF -DBUILD_PROGRAMMING_EXAMPLES=ON -DTT_METAL_BUILD_TESTS=OFF -DTTNN_BUILD_TESTS=OFF -DTT_UNITY_BUILDS=ON -DTT_ENABLE_LIGHT_METAL_TRACE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF -DENABLE_FAKE_KERNELS_TARGET=OFF -DENABLE_CCACHE=FALSE -DTT_USE_SYSTEM_SFPI=OFF
local -DCMAKE_TOOLCHAIN_FILE=/mnt/nvme10/joseph_ufl/toolchains/tt-metal@5280a9cf/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja -DCPM_SOURCE_CACHE=/mnt/nvme10/joseph_ufl/toolchains/tt-metal@5280a9cf/.cpmcache -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF
build build_Release; targets tt_metal hw_toolchain; examples add_2_integers_in_riscv loopback eltwise_binary eltwise_sfpu matmul_single_core matmul_multi_core add_2_integers_in_compute hello_world_compute_kernel hello_world_datamovement_kernel matmul_multicore_reuse
[rx] id=20260925-173747-exec-fd38 rc=0 state=done
```

## CMake-fetched packages

$LASSI_TOOLCHAINS/tt-metal@5280a9cf/lassi-cpm-sources.txt, read by the same exec: `<name>/<cache key> git <commit> changed=<tracked files that differ from the commit>`, or `files-sha256 <sha256 of the unpacked files>`. Recorded after the fetch, not pinned in advance.

```
benchmark/17a125dc196d0eea81711f65879e8410dd54303f git c58e6d0710581e3a08d65c349664128a8d9a2461 changed=0
boost/1359e136761ab2d10afa1c4e21086c8d824735cd files-sha256 68d9f1beda653ac43fbf929aaca110d21d0c1ba7e2bec4ea6fe4de41ba4759de
capnproto/8fee591276e6b6c91fac50b2687e44f3a44ea857 git d135c9ca5e15219eaf131dfce1a41afdbaea9aab changed=2
cxxopts/dda70ad5c1afdb5368541192bace55dbcd7872e3 git dbf4c6a66816f6c3872b46cc6af119ad227e04e1 changed=0
enchantum/2fb7ab238e36c101b9848892ddb6382276b65837 git 8ca5b0eb7e7ebe0252e5bc6915083f1dd1b8294e changed=0
flatbuffers/2c4062bffa52fa4157b1b4deeae73395df475fda git 595bf0007ab1929570c7671f091313c8fc20644e changed=0
fmt/69912fb6b71fcb1f7e5deca191a2bb4748c4e7b6 git 123913715afeb8a437e6388b4473fcc4753e1c9a changed=0
googletest/96129d89f45386492ae46d6bb8c027bc3df5f949 git b796f7d44681514f58a683a3a71ff17c94edb0c1 changed=0
libuv/2b05484ccff8fdf26b363fb50e4830b774d7eb53 git 5152db2cbfeb5582e9c27c5ea1dba2cd9e10759b changed=0
nanobind/8be24e8fbbe8c9cee0b8c553c21d090a5ede3d79 git c5a3a378aa61d104c82ca053cb1e367782cd3618 changed=0
nanomsg/28cc32d5bdb6a858fe53b3ccf7e923957e53eada git 29b73962b939a6fbbf6ea8d5d7680bb06d0eeb99 changed=0
nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308 git 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03 changed=0
picosha2/c5737937b9b264d5f413bff750447403d1603c55 git 161cb3fc4170fa7a3eca9e582cebd27cc4d1fe29 changed=0
protobuf/169ca95ddbe8ed0eebbba297015fe568604901e6 git f0dc78d7e6e331b8c6bb2d5283e06aa26883ca7c changed=1
range-v3/0_12_0_patched git a81477931a8aa2ad025c6bda0609f38e09e4d7ec changed=1
reflect/f93e77475670eaeacf332927dfe8b50e3f3812e0 git 1dbce7ae71fc1d32ad786642d9f1e1994a3efafe changed=0
simd-everywhere/b3b426f78574ef837b17f42e86bab88314c5e4db git 71fd833d9666141edcd1d3c109a80e228303d8d7 changed=0
spdlog/b1c2586bb5c35a7929362e87f62433eb68206873 git 48bcf39a661a13be22666ac64db8a7f886f2637e changed=0
taskflow/52063f60902bfeb362fa4616b1394ab5efe30994 git 7d9e85b6b2e9bf501021f857f2f3cbe43bc37c85 changed=0
tt-logger/87c1a5f2e9d2dd011200eb49c86426c26dec719e git e3aa7fd553c97310d6c58b4211f702fff18a3c60 changed=0
umd_asio/83a029aeca9a00050695989a093d0cc1fcd67ba8 git 12e0ce9e0500bf0f247dbd1ae894272656456079 changed=0
xtensor/0_26_0_patched git f31d415a507b84d0097436a38293df3f56906ad1 changed=2
xtensor-blas/0_22_0_patched git 894ff33d05aeb5297d5d3fde8e7591613b824de0 changed=4
xtl/0_8_0_patched git ef84e9f27020ad54961c99acd293d80d4b775dd3 changed=2
yaml-cpp/0_8_0_upstream_patched git 2f86d13775d119edbb69af52e5f566fd65c6953b changed=0
```
