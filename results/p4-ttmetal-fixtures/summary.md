# P4.10: ttmetal-host stderr fixtures and remote tests

[MEASURED] provenance.json (rx run 20260926-142744-desktop-8r113ei-detached-cf5f1274-326e):

- commit: cf5f127 (cf5f1274ef483e4bbd5d9f04f605b7a848407fe8), clean tree: `dirty` false and `snapshot_of` null in provenance.json and in the capture manifest, and the slot's `git status --short | wc -l` printed 0 (output.log, first line);
- host: alpha01, Ubuntu 22.04.5 LTS, Python 3.10.12; start 2026-09-26T14:27:44-07:00, end 2026-09-26T14:29:45-07:00; rc 0, and the command's last line was `capture_rc=0 remote_rc=0`;
- toolchain pin: toolchains/tt-metal.pin, VERSION 5280a9cf (tt-metal 5280a9cfb00998fd49667a29523d03aee905c129), PREFIX_NAME tt-metal@5280a9cf, EXECUTABLE /usr/bin/clang++-20. The capture manifest (tests/toolchains/fixtures/ttmetal/captures.json, pin_files) holds every pair of the pin; provenance.json's pins holds the first 4000 characters of each pin file, as the gate records it, and tt-metal.pin is 7668 bytes, so the HOST_* keys appear in full only in the manifest;
- compiler: /usr/bin/clang++-20; its --version, run through the compile sandbox before the first build, printed `Ubuntu clang version 20.1.8 (++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132)`, `Target: x86_64-pc-linux-gnu`, `Thread model: posix`, `InstalledDir: /usr/lib/llvm-20/bin` and exited 0; the pin's EXPECT_VERSION is the first of those lines;
- tree: /mnt/nvme10/joseph_ufl/toolchains/tt-metal@5280a9cf, named in every argv (results/p4-tt-install holds its install);
- compile environment: LANG, LC_ALL, PATH (LANG=C, LC_ALL=C); no HOME, no loader variable, and no tt-metal variable;
- scratch root free: 582.0 GB before and 582.0 GB after; root filesystem free 302.6 GB (provenance.json);
- device: none. The run compiled programs and read ELF headers only; no built program ran, and neither the kernel JIT nor ttsim started.

Command (provenance.json, cmd):

```
uv sync --quiet && git status --short | wc -l && { uv run python tools/capture_toolchain_fixtures.py --fixtures tests/toolchains/fixtures/ttmetal; c=$?; LASSI_REQUIRE_SANDBOX=1 uv run pytest -q -p no:cacheprovider -m remote tests/toolchains/test_ttmetal_host_remote.py; t=$?; echo "capture_rc=$c remote_rc=$t"; [ "$c" -eq 0 ] && [ "$t" -eq 0 ]; }
```

## Results

The capture (output.log; the manifest dated 2026-09-26T14:27:45-07:00) built each case once with the ttmetal-host toolchain, as the stage runner builds it, in a fresh workdir:

| Case | Exit status | Stderr bytes | Diagnostics parsed | Stderr sha256 |
| --- | --- | --- | --- | --- |
| ttm_clean | 0 | 0 | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| ttm_header_error | 1 | 183 | 1 | `0479bc19ffc370f5e5110f65c2c10eb2138e9a3816f78dd10f0ebc2cbd242b89` |
| ttm_linker_error | 1 | 335 | 2 | `b2cecfe2dad9004f1b9200d22df5ae518405370bbb0e53487655f292b507f30c` |
| ttm_missing_header | 1 | 210 | 1 | `7afd302d299da0189e7cb0c494df3c43eac933649daba8dc79e261fff2c286db` |
| ttm_note_chain | 1 | 496 | 3 | `e3a5cbeae5118ed7011ea1cb735586ae30ee034de98325389118604476ed2115` |
| ttm_syntax_error | 1 | 156 | 1 | `f2f5907cca6f613c2f7ff72a7a62c5f6f4fab698974907c21d4ea8c9bf27e4a2` |
| ttm_undeclared_identifier | 1 | 153 | 1 | `c244ae0babd891c4c34194fb23182eb2ae5582766e4a54cc9efe6873f15e8e8f` |
| ttm_unused_parameter | 0 | 196 | 1 | `a530fd411394483b2f9d833a8b9cb915350c62f12387767cf8a5429c6f1d6a63` |

- tests/toolchains/fixtures/ttmetal/<case>.stderr and captures.json are byte-identical to the capture's `<case>.stderr` files and manifest.json (compared with cmp after the pull), and each stderr's sha256 is the manifest's. All eight are plain ASCII with LF line endings; ttm_clean.stderr is empty.
- The source trees the capture compiled (work/<case>/ in the pull) are byte-identical to tests/toolchains/fixtures/ttmetal/sources/.
- Each <case>.json was derived by hand from its stderr, and the parser gives exactly those lists (tests/toolchains/test_ttmetal_diagnostics.py).
- What the capture showed: clang names a header it found beside the including file with a leading `./` (`./host/scale.h:5:20: error: ...`), which the parser drops, so the diagnostic names the built file host/scale.h and keeps its column; the fatal error for a missing header points at column 10, the `<` that starts the header name; ttm_clean built with exit status 0 and an empty stderr, with its kernel kernels/dataflow/p410_noop.cpp (which includes the device header dataflow_api.h) written beside it and not compiled; ttm_unused_parameter is a warning, code -Wunused-parameter, and the build exited 0, since the host flags leave out -Werror (OQ-027, option (b)).
- ttm_linker_error.stderr names the capture's workdir and clang's temporary object in it, `@lassi-tmp/main-eaa0c1.o`; that text changes on a recapture. The other seven name no per-run text.
- The remote tests, run with LASSI_REQUIRE_SANDBOX=1 so that none could skip, printed `13 passed in 74.37s (0:01:14)` (output.log): the --version check in the compile sandbox; the tree checks (lassi-install.txt names the pinned commit; lassi-cpm-sources.txt equals toolchains/tt-metal-cpm-sources.txt); the gate's example and the five Tier A examples, each built from the pinned sources with the kernels it names placed at their paths and each clean, with the NEEDED list and RUNPATH directories of the pinned build's own binary (readelf -d), and each defining OVERRIDE_KERNEL_PREFIX as "" when it is unset; the fixture program with its kernel placed and not compiled; a compile error on the built file; and the refusal of a pin with another EXPECT_VERSION and of a tracked CPM list with a changed line.

## Where the host flags come from

toolchains/tt-metal.pin's HOST_* keys record the pinned build's own compile and link line for the gate's example, metal_example_add_2_integers_in_riscv, as two read-only rx execs read it from the installed tree on 2026-09-25. In the output below, T stands for the tree, $LASSI_TOOLCHAINS/tt-metal@5280a9cf, and C for T/.cpmcache (the probes replaced both paths with sed); the last line of each is rx's status line.

rx exec 20260925-223402-exec-32ce: the files it read, then the line count of lassi-cpm-sources.txt; the gate example's one object edge and its link edge in build_Release/build.ninja (DEFINES, FLAGS, and INCLUDES one word per line, `-isystem` and `-Xclang` joined to the next word; link FLAGS already among the compile FLAGS are not repeated); readelf -d of the pinned binary; and the compiler:

```
== files
T/build_Release/build.ninja
T/build_Release/CMakeFiles/rules.ninja
T/lassi-cpm-sources.txt
T/lassi-install.txt
25
== object edges 1  | tt_metal/programming_examples/add_2_integers_in_riscv/CMakeFiles/metal_example_add_2_integers_in_riscv.dir/add_2_integers_in_riscv.cpp.o: T/tt_metal/programming_examples/add_2_integers_in_riscv/add_2_integers_in_riscv.cpp
== compile DEFINES 8
DEFINES -DENCHANTUM_ENABLE_MSVC_SPEEDUP=1
DEFINES -DFMT_HEADER_ONLY=1
DEFINES -DOVERRIDE_KERNEL_PREFIX=\"tt_metal/programming_examples/\"
DEFINES -DSPDLOG_FMT_EXTERNAL
DEFINES -DSPDLOG_FWRITE_UNLOCKED
DEFINES -DTRACY_IMPORTS
DEFINES -DTRACY_TIMER_FALLBACK
DEFINES -DTT_ENABLE_LIGHT_METAL_TRACE=1
== compile FLAGS 18
FLAGS -O3
FLAGS -DNDEBUG
FLAGS -std=c++20
FLAGS -fPIE
FLAGS -fsized-deallocation
FLAGS -gz
FLAGS -march=x86-64-v3
FLAGS -fPIC
FLAGS -pipe
FLAGS -fvisibility-inlines-hidden
FLAGS -Wall
FLAGS -Werror
FLAGS -Wconditional-uninitialized
FLAGS -Wno-deprecated-declarations
FLAGS -Xclang -fno-pch-timestamp
FLAGS -Wunused-parameter
FLAGS -Wno-c++11-narrowing
FLAGS -Wno-int-to-pointer-cast
== compile INCLUDES 13
INCLUDES -IT/tt_stl/.
INCLUDES -IT/tt_metal/api
INCLUDES -IT/build_Release/tt_metal/api
INCLUDES -IT/tt_metal/hostdevcommon/api
INCLUDES -IT
INCLUDES -isystem C/reflect/f93e77475670eaeacf332927dfe8b50e3f3812e0
INCLUDES -isystem C/enchantum/2fb7ab238e36c101b9848892ddb6382276b65837/enchantum/include
INCLUDES -isystem C/nlohmann_json/798e0374658476027d9723eeb67a262d0f3c8308/include
INCLUDES -isystem C/fmt/69912fb6b71fcb1f7e5deca191a2bb4748c4e7b6/include
INCLUDES -isystem C/tt-logger/87c1a5f2e9d2dd011200eb49c86426c26dec719e/include
INCLUDES -isystem C/spdlog/b1c2586bb5c35a7929362e87f62433eb68206873/include
INCLUDES -isystem T/tt_metal/third_party/umd/device/api
INCLUDES -isystem T/tt_metal/third_party/tracy/public
== link edge programming_examples/metal_example_add_2_integers_in_riscv:
== link FLAGS 2
== link LINK_FLAGS 4
LLINK_FLAGS -Wl,--compress-debug-sections=zlib
LLINK_FLAGS -fuse-ld=lld
LLINK_FLAGS -Xlinker
LLINK_FLAGS --dependency-file=tt_metal/programming_examples/add_2_integers_in_riscv/CMakeFiles/metal_example_add_2_integers_in_riscv.dir/link.d
== link LINK_LIBRARIES 6
LLINK_LIBRARIES -Wl,-rpath,T/build_Release/tt_metal:T/build_Release/tt_stl:T/build_Release/lib:T/build_Release/tt_metal/third_party/umd/device
LLINK_LIBRARIES tt_metal/libtt_metal.so
LLINK_LIBRARIES tt_stl/libtt_stl.so
LLINK_LIBRARIES lib/libtracy.so.0.10.0
LLINK_LIBRARIES -ldl
LLINK_LIBRARIES tt_metal/third_party/umd/device/libdevice.so
== link LINK_PATH 0
== readelf programming_examples/metal_example_add_2_integers_in_riscv
(RUNPATH) [T/build_Release/tt_metal:T/build_Release/tt_stl:T/build_Release/lib:T/build_Release/tt_metal/third_party/umd/device]
(NEEDED) [libtt_metal.so]
(NEEDED) [libtt_stl.so]
(NEEDED) [libtracy.so.0.10.0]
(NEEDED) [libdevice.so]
(NEEDED) [libstdc++.so.6]
(NEEDED) [libm.so.6]
(NEEDED) [libgcc_s.so.1]
(NEEDED) [libc.so.6]
== compiler
/usr/bin/clang++-20
Ubuntu clang version 20.1.8 (++20250708082409+6fb913d3e2ec-1~exp1~20250708202428.132)
[rx] id=20260925-223402-exec-32ce rc=0 state=done
```

rx exec 20260925-223734-exec-4611: the compile and link rules; each Tier A example's build words that differ from the gate example's (`+` the example has it, `-` the example lacks it; the `gate link FLAGS` lines are the gate's link FLAGS); and each target's source and binary. The exec also printed host source lines that name OVERRIDE_KERNEL_PREFIX or include a quoted header, and a kernel file listing; both are left out here, since no upstream source text enters a tracked file:

```
== rules
  command = ${LAUNCHER}${CODE_CHECK}/usr/bin/clang++-20 $DEFINES $INCLUDES $FLAGS -MD -MT $out -MF $DEP_FILE -o $out -c $in
  command = $PRE_LINK && /usr/bin/clang++-20 $FLAGS $LINK_FLAGS $in -o $TARGET_FILE $LINK_PATH $LINK_LIBRARIES && $POST_BUILD
- eltwise_binary 2:LINK_FLAGS -Xlinker --dependency-file=tt_metal/programming_examples/add_2_integers_in_riscv/CMakeFiles/metal_example_add_2_integers_in_riscv.dir/link.d
+ eltwise_binary 2:LINK_FLAGS -Xlinker --dependency-file=tt_metal/programming_examples/eltwise_binary/CMakeFiles/metal_example_eltwise_binary.dir/link.d
- eltwise_sfpu 2:LINK_FLAGS -Xlinker --dependency-file=tt_metal/programming_examples/add_2_integers_in_riscv/CMakeFiles/metal_example_add_2_integers_in_riscv.dir/link.d
+ eltwise_sfpu 2:LINK_FLAGS -Xlinker --dependency-file=tt_metal/programming_examples/eltwise_sfpu/CMakeFiles/metal_example_eltwise_sfpu.dir/link.d
gate link FLAGS -DNDEBUG
gate link FLAGS -O3
- loopback 2:LINK_FLAGS -Xlinker --dependency-file=tt_metal/programming_examples/add_2_integers_in_riscv/CMakeFiles/metal_example_add_2_integers_in_riscv.dir/link.d
+ loopback 2:LINK_FLAGS -Xlinker --dependency-file=tt_metal/programming_examples/loopback/CMakeFiles/metal_example_loopback.dir/link.d
+ matmul_multi_core 1:INCLUDES -IT/tt_metal/programming_examples/matmul/matmul_common
- matmul_multi_core 2:LINK_FLAGS -Xlinker --dependency-file=tt_metal/programming_examples/add_2_integers_in_riscv/CMakeFiles/metal_example_add_2_integers_in_riscv.dir/link.d
+ matmul_multi_core 2:LINK_FLAGS -Xlinker --dependency-file=tt_metal/programming_examples/matmul/matmul_multi_core/CMakeFiles/metal_example_matmul_multi_core.dir/link.d
+ matmul_single_core 1:INCLUDES -IT/tt_metal/programming_examples/matmul/matmul_common
- matmul_single_core 2:LINK_FLAGS -Xlinker --dependency-file=tt_metal/programming_examples/add_2_integers_in_riscv/CMakeFiles/metal_example_add_2_integers_in_riscv.dir/link.d
+ matmul_single_core 2:LINK_FLAGS -Xlinker --dependency-file=tt_metal/programming_examples/matmul/matmul_single_core/CMakeFiles/metal_example_matmul_single_core.dir/link.d
== target add_2_integers_in_riscv objects 1 src T/tt_metal/programming_examples/add_2_integers_in_riscv/add_2_integers_in_riscv.cpp out programming_examples/metal_example_add_2_integers_in_riscv:
== target eltwise_binary objects 1 src T/tt_metal/programming_examples/eltwise_binary/eltwise_binary.cpp out programming_examples/metal_example_eltwise_binary:
== target eltwise_sfpu objects 1 src T/tt_metal/programming_examples/eltwise_sfpu/eltwise_sfpu.cpp out programming_examples/metal_example_eltwise_sfpu:
== target loopback objects 1 src T/tt_metal/programming_examples/loopback/loopback.cpp out programming_examples/metal_example_loopback:
== target matmul_multi_core objects 1 src T/tt_metal/programming_examples/matmul/matmul_multi_core/matmul_multi_core.cpp out programming_examples/metal_example_matmul_multi_core:
== target matmul_single_core objects 1 src T/tt_metal/programming_examples/matmul/matmul_single_core/matmul_single_core.cpp out programming_examples/metal_example_matmul_single_core:
[rx] id=20260925-223734-exec-4611 rc=0 state=done
```

Reading, with the pin's values in captures.json (pin_files):

- HOST_DEFINES: the DEFINES without `-DOVERRIDE_KERNEL_PREFIX=\"tt_metal/programming_examples/\"`, which only the programming examples' CMakeLists adds; each example defines it as "" when it is unset (checked for all six by the remote tests).
- HOST_INCLUDES: the INCLUDES as printed, T written {TREE} and C written {TREE}/.cpmcache.
- HOST_FLAGS: the FLAGS without -Werror (OQ-027, option (b)).
- HOST_LINK_FLAGS: the LINK_FLAGS without `-Xlinker --dependency-file=...`, CMake's dependency file. The link FLAGS, -O3 and -DNDEBUG, are the first two FLAGS.
- HOST_LINK_LIBRARIES: the LINK_LIBRARIES, each library path written under {TREE}/build_Release, since build.ninja names them relative to the build directory; the rpath names the four directories the pinned binary's RUNPATH names.
- The other five examples build with the same words; the two matmul ones add only `-IT/tt_metal/programming_examples/matmul/matmul_common` (Matmul::Common). The toolchain has no such word; the remote tests place Matmul::Common's bmm_op.hpp in the build directory at the path an example's source includes it by, if it includes it, where the one word the toolchain adds, `-idirafter .`, finds it, and both matmul examples built clean. Which include form they use is not recorded here.
- EXECUTABLE is the compiler the rules name and `command -v clang++-20` printed, /usr/bin/clang++-20; EXPECT_VERSION is the first line its --version printed in exec 32ce, and the capture's version check printed the same line.

